#include "ffs_viewer/geometry/final_cloud_processor.hpp"

#include <cub/cub.cuh>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ffs_viewer::geometry {
namespace {

struct FinalCloudConstants {
    int width;
    int height;
    float fx;
    float fy;
    float cx;
    float cy;
    float baseline_m;
    float z_min_m;
    float z_max_m;
    float radius_m;
    int min_neighbors;
};

__constant__ FinalCloudConstants c_final_cloud;
constexpr int kThreads = 256;
constexpr int kCellBias = 1 << 20;
constexpr std::uint64_t kCellMask = (1ULL << 21) - 1ULL;

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__device__ std::uint64_t cellKey(int x, int y, int z) {
    const std::uint64_t ux = static_cast<std::uint64_t>(x + kCellBias) & kCellMask;
    const std::uint64_t uy = static_cast<std::uint64_t>(y + kCellBias) & kCellMask;
    const std::uint64_t uz = static_cast<std::uint64_t>(z + kCellBias) & kCellMask;
    return (ux << 42) | (uy << 21) | uz;
}

__global__ void projectKernel(float* disparity, float* xyz, std::uint8_t* valid, int* indices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = c_final_cloud.width * c_final_cloud.height;
    if (i >= count) return;
    const int x = i % c_final_cloud.width;
    float d = disparity[i];
    if (static_cast<float>(x) - d < 0.0F) d = -1.0F;
    disparity[i] = d;
    indices[i] = i;
    const float z = d > 0.0F ? c_final_cloud.fx * c_final_cloud.baseline_m / d : 0.0F;
    const bool keep = isfinite(z) && z >= c_final_cloud.z_min_m && z <= c_final_cloud.z_max_m;
    valid[i] = keep ? 1 : 0;
    xyz[3 * i] = keep ? (static_cast<float>(x) - c_final_cloud.cx) * z / c_final_cloud.fx : 0.0F;
    xyz[3 * i + 1] = keep ? -((static_cast<float>(i / c_final_cloud.width) - c_final_cloud.cy) * z / c_final_cloud.fy) : 0.0F;
    xyz[3 * i + 2] = keep ? z : 0.0F;
}

__global__ void keyKernel(const int* active, int n, const float* xyz, std::uint64_t* keys, int* ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int id = active[i];
    const float r = c_final_cloud.radius_m;
    keys[i] = cellKey(__float2int_rd(xyz[3 * id] / r), __float2int_rd(xyz[3 * id + 1] / r),
                      __float2int_rd(xyz[3 * id + 2] / r));
    ids[i] = id;
}

__device__ int lowerBound(const std::uint64_t* keys, int n, std::uint64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { const int mid = lo + (hi - lo) / 2; if (keys[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

__global__ void radiusDenoiseKernel(const std::uint64_t* keys, const int* ids, int n,
                                    const float* xyz, std::uint8_t* valid) {
    const int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i >= n) return;
    const int id = ids[sorted_i];
    const float* p = xyz + 3 * id;
    const float r = c_final_cloud.radius_m;
    const float r2 = r * r;
    const int vx = __float2int_rd(p[0] / r), vy = __float2int_rd(p[1] / r), vz = __float2int_rd(p[2] / r);
    int neighbors = 0;
    for (int dz = -1; dz <= 1 && neighbors < c_final_cloud.min_neighbors; ++dz)
        for (int dy = -1; dy <= 1 && neighbors < c_final_cloud.min_neighbors; ++dy)
            for (int dx = -1; dx <= 1 && neighbors < c_final_cloud.min_neighbors; ++dx) {
                const int begin = lowerBound(keys, n, cellKey(vx + dx, vy + dy, vz + dz));
                for (int j = begin; j < n && keys[j] == cellKey(vx + dx, vy + dy, vz + dz) && neighbors < c_final_cloud.min_neighbors; ++j) {
                    const float* q = xyz + 3 * ids[j];
                    const float ddx = p[0] - q[0], ddy = p[1] - q[1], ddz = p[2] - q[2];
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) ++neighbors;
                }
            }
    if (neighbors < c_final_cloud.min_neighbors) valid[id] = 0;
}

}  // namespace
__global__ void writeVerticesKernel(const float* xyz, const std::uint8_t* valid, const float* left_gray,
                                    int left_row_offset, int display_step, int image_width, int image_height, const std::uint8_t* mask,
                                    int mask_width, int mask_height, GpuPointVertex* vertices, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int x = i % image_width;
    const int y = i / image_width;
    const bool selected = display_step <= 1 || (x % display_step == 0 && y % display_step == 0);
    const bool keep = valid[i] != 0 && selected;
    const bool masked = mask != nullptr && mask[
        min(mask_height - 1, y * mask_height / image_height) * mask_width +
        min(mask_width - 1, x * mask_width / image_width)] != 0;
    const auto gray = static_cast<std::uint8_t>(fminf(255.0F, fmaxf(0.0F, left_gray[i + left_row_offset])));
    vertices[i] = {xyz[3 * i], xyz[3 * i + 1], keep ? xyz[3 * i + 2] : 100.0F,
                   masked ? std::uint8_t(0) : gray, masked ? std::uint8_t(255) : gray,
                   masked ? std::uint8_t(0) : gray, 255};
}


__device__ bool meshMasked(const std::uint8_t* mask, int mask_width, int mask_height,
                           int x, int y, int image_width, int image_height) {
    return mask != nullptr && mask[min(mask_height - 1, y * mask_height / image_height) * mask_width +
                                   min(mask_width - 1, x * mask_width / image_width)] != 0;
}

__device__ float triangleArea(const float* p, int a, int b, int c) { const float abx=p[3*b]-p[3*a], aby=p[3*b+1]-p[3*a+1], abz=p[3*b+2]-p[3*a+2], acx=p[3*c]-p[3*a], acy=p[3*c+1]-p[3*a+1], acz=p[3*c+2]-p[3*a+2]; const float x=aby*acz-abz*acy, y=abz*acx-abx*acz, z=abx*acy-aby*acx; return .5F*sqrtf(x*x+y*y+z*z); }
__global__ void initializeMeshKernel(const float* xyz, const std::uint8_t* valid, const std::uint8_t* mask, int mw, int mh, int w, int h, int step, float depth_threshold_m, int* parent, float* areas, int cells_x, int cells) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x; if(cell>=cells) return; const int x=(cell%cells_x)*step, y=(cell/cells_x)*step, a=y*w+x, b=a+step, c=a+step*w, d=c+step;
    const bool selected=meshMasked(mask,mw,mh,x,y,w,h)&&meshMasked(mask,mw,mh,x+step,y,w,h)&&meshMasked(mask,mw,mh,x,y+step,w,h)&&meshMasked(mask,mw,mh,x+step,y+step,w,h);
    const float za=xyz[3*a+2], zb=xyz[3*b+2], zc=xyz[3*c+2], zd=xyz[3*d+2]; const bool keep=selected&&valid[a]&&valid[b]&&valid[c]&&valid[d]&&fmaxf(fmaxf(za,zb),fmaxf(zc,zd))-fminf(fminf(za,zb),fminf(zc,zd))<=depth_threshold_m;
    parent[cell]=keep?cell:-1; areas[cell]=keep?triangleArea(xyz,a,b,c)+triangleArea(xyz,b,d,c):0.F;
}
__global__ void propagateMeshLabelsKernel(int* parent, int cells_x, int cells_y, int cells) { const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=cells||parent[i]<0)return; int p=parent[i],x=i%cells_x,y=i/cells_x; if(x&&parent[i-1]>=0)p=min(p,parent[i-1]); if(x+1<cells_x&&parent[i+1]>=0)p=min(p,parent[i+1]); if(y&&parent[i-cells_x]>=0)p=min(p,parent[i-cells_x]); if(y+1<cells_y&&parent[i+cells_x]>=0)p=min(p,parent[i+cells_x]); parent[i]=p; }
__global__ void accumulateMeshAreasKernel(const int* parent,const float* cells_area,float* component_area,int cells) { const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<cells&&parent[i]>=0)atomicAdd(component_area+parent[i],cells_area[i]); }
__global__ void findLargestMeshKernel(const int* parent,const float* area,int* best,int cells) { const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<cells&&parent[i]==i) atomicMax(best,__float_as_int(area[i])); }
__global__ void selectLargestMeshRootKernel(const int* parent,const float* area,const int* best,int* root,int cells) { const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<cells&&parent[i]==i&&__float_as_int(area[i])==*best) atomicMin(root,i); }
__global__ void writeMeshIndicesKernel(const int* parent,const int* root,int w,int step,std::uint32_t* indices,int cells_x,int cells) { const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=cells)return; const int x=(i%cells_x)*step,y=(i/cells_x)*step,a=y*w+x,b=a+step,c=a+step*w,d=c+step,o=6*i; if(parent[i]>=0&&parent[i]==*root){indices[o]=a;indices[o+1]=b;indices[o+2]=c;indices[o+3]=b;indices[o+4]=d;indices[o+5]=c;}else{indices[o]=indices[o+1]=indices[o+2]=indices[o+3]=indices[o+4]=indices[o+5]=0;} }

struct FinalCloudProcessor::Impl {
    int width, height, count;
    float* d_xyz = nullptr;
    std::uint8_t* d_valid = nullptr;
    int* d_indices = nullptr;
    int* d_active = nullptr;
    int* d_ids_in = nullptr;
    int* d_ids_out = nullptr;
    std::uint64_t* d_keys_in = nullptr;
    std::uint64_t* d_keys_out = nullptr;
    void* d_select_temp = nullptr;
    void* d_sort_temp = nullptr;
    std::size_t select_bytes = 0, sort_bytes = 0;
    int* d_active_count = nullptr;
    int* d_mesh_parent = nullptr;
    float* d_mesh_area = nullptr;
    float* d_mesh_cell_area = nullptr;
    int* d_mesh_best_root = nullptr;
    int* d_mesh_best_area_bits = nullptr;

    Impl(int w, int h) : width(w), height(h), count(w * h) {
        check(cudaMalloc(&d_xyz, 3ULL * count * sizeof(float)), "allocate final XYZ");
        check(cudaMalloc(&d_valid, count), "allocate final validity");
        check(cudaMalloc(&d_indices, count * sizeof(int)), "allocate final indices");
        check(cudaMalloc(&d_active, count * sizeof(int)), "allocate active indices");
        check(cudaMalloc(&d_ids_in, count * sizeof(int)), "allocate hash ids");
        check(cudaMalloc(&d_ids_out, count * sizeof(int)), "allocate sorted hash ids");
        check(cudaMalloc(&d_keys_in, count * sizeof(std::uint64_t)), "allocate hash keys");
        check(cudaMalloc(&d_keys_out, count * sizeof(std::uint64_t)), "allocate sorted hash keys");
        check(cudaMalloc(&d_active_count, sizeof(int)), "allocate active count");
        check(cudaMalloc(&d_mesh_parent, (width - 1) * (height - 1) * sizeof(int)), "allocate mesh parents");
        check(cudaMalloc(&d_mesh_area, (width - 1) * (height - 1) * sizeof(float)), "allocate mesh areas");
        check(cudaMalloc(&d_mesh_cell_area, (width - 1) * (height - 1) * sizeof(float)), "allocate mesh cell areas");
        check(cudaMalloc(&d_mesh_best_root, sizeof(int)), "allocate mesh best root");
        check(cudaMalloc(&d_mesh_best_area_bits, sizeof(int)), "allocate mesh best area");
        check(cub::DeviceSelect::Flagged(nullptr, select_bytes, d_indices, d_valid, d_active, d_active_count, count), "query CUB select");
        check(cudaMalloc(&d_select_temp, select_bytes), "allocate CUB select workspace");
        check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, d_keys_in, d_keys_out, d_ids_in, d_ids_out, count), "query CUB sort");
        check(cudaMalloc(&d_sort_temp, sort_bytes), "allocate CUB sort workspace");
    }
    ~Impl() {
        cudaFree(d_sort_temp); cudaFree(d_select_temp); cudaFree(d_mesh_best_area_bits); cudaFree(d_mesh_best_root); cudaFree(d_mesh_area); cudaFree(d_mesh_cell_area); cudaFree(d_mesh_parent); cudaFree(d_active_count); cudaFree(d_keys_out); cudaFree(d_keys_in);
        cudaFree(d_ids_out); cudaFree(d_ids_in); cudaFree(d_active); cudaFree(d_indices); cudaFree(d_valid); cudaFree(d_xyz);
    }
};

FinalCloudProcessor::FinalCloudProcessor(int width, int height) : impl_(std::make_unique<Impl>(width, height)) {}
FinalCloudProcessor::~FinalCloudProcessor() = default;

FinalCloudFrame FinalCloudProcessor::process(float* d_disparity, const float* d_left_input, cudaStream_t stream,
                                             const io::StereoCalibration& calibration,
                                             int source_width, int source_height, int left_row_offset,
                                             float z_max_m, const std::function<void()>& on_denoise) {
    if (source_width <= 0 || source_height <= 0 || z_max_m <= 0.0F) throw std::invalid_argument("invalid final-cloud parameters");
    const float sx = static_cast<float>(impl_->width) / source_width;
    const float sy = static_cast<float>(impl_->height) / source_height;
    const FinalCloudConstants constants{impl_->width, impl_->height, calibration.left.fx * sx, calibration.left.fy * sy,
                                        calibration.left.cx * sx, calibration.left.cy * sy, calibration.baseline_m,
                                        0.1F, z_max_m, 0.03F, 30};
    check(cudaMemcpyToSymbolAsync(c_final_cloud, &constants, sizeof(constants), 0, cudaMemcpyHostToDevice, stream), "upload final-cloud constants");
    const int blocks = (impl_->count + kThreads - 1) / kThreads;
    projectKernel<<<blocks, kThreads, 0, stream>>>(d_disparity, impl_->d_xyz, impl_->d_valid, impl_->d_indices);
    check(cudaGetLastError(), "launch final-cloud projection");
    check(cub::DeviceSelect::Flagged(impl_->d_select_temp, impl_->select_bytes, impl_->d_indices, impl_->d_valid,
                                     impl_->d_active, impl_->d_active_count, impl_->count, stream), "compact valid final points");
    int active = 0;
    check(cudaMemcpyAsync(&active, impl_->d_active_count, sizeof(int), cudaMemcpyDeviceToHost, stream), "copy active count");
    check(cudaStreamSynchronize(stream), "synchronize active count");
    if (on_denoise) on_denoise();
    if (active > 0) {
        const int active_blocks = (active + kThreads - 1) / kThreads;
        keyKernel<<<active_blocks, kThreads, 0, stream>>>(impl_->d_active, active, impl_->d_xyz, impl_->d_keys_in, impl_->d_ids_in);
        check(cudaGetLastError(), "launch voxel-key generation");
        check(cub::DeviceRadixSort::SortPairs(impl_->d_sort_temp, impl_->sort_bytes, impl_->d_keys_in, impl_->d_keys_out,
                                              impl_->d_ids_in, impl_->d_ids_out, active, 0, 64, stream), "sort voxel keys");
        radiusDenoiseKernel<<<active_blocks, kThreads, 0, stream>>>(impl_->d_keys_out, impl_->d_ids_out, active, impl_->d_xyz, impl_->d_valid);
        check(cudaGetLastError(), "launch radius denoise");
    }
    FinalCloudFrame result;
    result.disparity.width = impl_->width; result.disparity.height = impl_->height;
    result.disparity.values.resize(impl_->count);
    check(cudaMemcpyAsync(result.disparity.values.data(), d_disparity, impl_->count * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy processed disparity");

    check(cudaStreamSynchronize(stream), "synchronize final cloud");
    result.d_xyz = impl_->d_xyz;
    result.d_valid = impl_->d_valid;
    result.d_left_gray = d_left_input;
    result.left_row_offset = left_row_offset;
    result.point_count = impl_->count;
    result.d_mesh_cell_area = impl_->d_mesh_cell_area;
    result.d_mesh_parent = impl_->d_mesh_parent;
    result.d_mesh_area = impl_->d_mesh_area;
    result.d_mesh_best_root = impl_->d_mesh_best_root;
    result.d_mesh_best_area_bits = impl_->d_mesh_best_area_bits;
    return result;
}

void writeFinalCloudVertices(const FinalCloudFrame& cloud, GpuPointVertex* d_vertices,
                             cudaStream_t stream) {
    if (cloud.d_xyz == nullptr || cloud.d_valid == nullptr || cloud.d_left_gray == nullptr || d_vertices == nullptr || cloud.point_count <= 0 || cloud.disparity.width <= 0) {
        throw std::invalid_argument("invalid GPU final-cloud vertex target");
    }
    const int blocks = (cloud.point_count + 255) / 256;
    writeVerticesKernel<<<blocks, 256, 0, stream>>>(cloud.d_xyz, cloud.d_valid, cloud.d_left_gray,
                                                     cloud.left_row_offset, cloud.display_step, cloud.disparity.width, cloud.disparity.height,
                                                     cloud.d_mask, cloud.mask_width, cloud.mask_height, d_vertices, cloud.point_count);
    check(cudaGetLastError(), "launch OpenGL vertex write");
}

int finalCloudMeshIndexCount(const FinalCloudFrame& cloud) {
    if (cloud.display_step <= 0 || cloud.disparity.width <= 0 || cloud.disparity.height <= 0) return 0;
    return 6 * ((cloud.disparity.width - 1) / cloud.display_step) *
               ((cloud.disparity.height - 1) / cloud.display_step);
}

void writeFinalCloudMeshIndices(const FinalCloudFrame& cloud, std::uint32_t* d_indices,
                                cudaStream_t stream) {
    const int count = finalCloudMeshIndexCount(cloud);
    if (count <= 0 || d_indices == nullptr || cloud.d_mesh_parent == nullptr || cloud.d_mesh_best_root == nullptr) return;
    const int cells_x = (cloud.disparity.width - 1) / cloud.display_step;
    const int cells = count / 6;
    writeMeshIndicesKernel<<<(cells + 255) / 256, 256, 0, stream>>>(cloud.d_mesh_parent, cloud.d_mesh_best_root,
        cloud.disparity.width, cloud.display_step, d_indices, cells_x, cells);
    check(cudaGetLastError(), "launch final-cloud mesh index write");
}


float finalCloudLargestMeshAreaM2(const FinalCloudFrame& cloud, cudaStream_t stream) {
    const int count = finalCloudMeshIndexCount(cloud);
    if (count <= 0 || !cloud.d_xyz || !cloud.d_valid || !cloud.d_mask || !cloud.d_mesh_parent || !cloud.d_mesh_cell_area || !cloud.d_mesh_area || !cloud.d_mesh_best_root || !cloud.d_mesh_best_area_bits) return 0.F;
    const int cells_x = (cloud.disparity.width - 1) / cloud.display_step, cells_y = (cloud.disparity.height - 1) / cloud.display_step, cells = count / 6, blocks = (cells + 255) / 256;
    initializeMeshKernel<<<blocks, 256, 0, stream>>>(cloud.d_xyz, cloud.d_valid, cloud.d_mask, cloud.mask_width, cloud.mask_height, cloud.disparity.width, cloud.disparity.height, cloud.display_step, cloud.mesh_depth_threshold_m, cloud.d_mesh_parent, cloud.d_mesh_cell_area, cells_x, cells);
    check(cudaGetLastError(), "initialize final-cloud mesh");
    for (int i = 0; i < cells_x + cells_y - 2; ++i) propagateMeshLabelsKernel<<<blocks, 256, 0, stream>>>(cloud.d_mesh_parent, cells_x, cells_y, cells);
    check(cudaGetLastError(), "connect final-cloud mesh");
    check(cudaMemsetAsync(cloud.d_mesh_area, 0, std::size_t(cells) * sizeof(float), stream), "clear mesh areas");
    accumulateMeshAreasKernel<<<blocks, 256, 0, stream>>>(cloud.d_mesh_parent, cloud.d_mesh_cell_area, cloud.d_mesh_area, cells);
    check(cudaGetLastError(), "accumulate final-cloud mesh areas");
    check(cudaMemsetAsync(cloud.d_mesh_best_root, 0x7f, sizeof(int), stream), "clear mesh best root");
    check(cudaMemsetAsync(cloud.d_mesh_best_area_bits, 0, sizeof(int), stream), "clear mesh best area");
    findLargestMeshKernel<<<blocks, 256, 0, stream>>>(cloud.d_mesh_parent, cloud.d_mesh_area, cloud.d_mesh_best_area_bits, cells);
    selectLargestMeshRootKernel<<<blocks, 256, 0, stream>>>(cloud.d_mesh_parent, cloud.d_mesh_area, cloud.d_mesh_best_area_bits, cloud.d_mesh_best_root, cells);
    check(cudaGetLastError(), "find largest final-cloud mesh");
    float area_m2 = 0.F;
    check(cudaMemcpyAsync(&area_m2, cloud.d_mesh_best_area_bits, sizeof(area_m2), cudaMemcpyDeviceToHost, stream), "copy largest mesh area");
    check(cudaStreamSynchronize(stream), "synchronize largest mesh area");
    return area_m2;
}
}  // namespace ffs_viewer::geometry

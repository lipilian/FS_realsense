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
        check(cub::DeviceSelect::Flagged(nullptr, select_bytes, d_indices, d_valid, d_active, d_active_count, count), "query CUB select");
        check(cudaMalloc(&d_select_temp, select_bytes), "allocate CUB select workspace");
        check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, d_keys_in, d_keys_out, d_ids_in, d_ids_out, count), "query CUB sort");
        check(cudaMalloc(&d_sort_temp, sort_bytes), "allocate CUB sort workspace");
    }
    ~Impl() {
        cudaFree(d_sort_temp); cudaFree(d_select_temp); cudaFree(d_active_count); cudaFree(d_keys_out); cudaFree(d_keys_in);
        cudaFree(d_ids_out); cudaFree(d_ids_in); cudaFree(d_active); cudaFree(d_indices); cudaFree(d_valid); cudaFree(d_xyz);
    }
};

FinalCloudProcessor::FinalCloudProcessor(int width, int height) : impl_(std::make_unique<Impl>(width, height)) {}
FinalCloudProcessor::~FinalCloudProcessor() = default;

FinalCloudFrame FinalCloudProcessor::process(float* d_disparity, const float* d_left_input, cudaStream_t stream,
                                             const io::StereoCalibration& calibration,
                                             int source_width, int source_height, float z_max_m,
                                             const std::function<void()>& on_denoise) {
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
    result.left_row_offset = 4 * impl_->width;
    result.point_count = impl_->count;
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

}  // namespace ffs_viewer::geometry

#include "ffs_viewer/geometry/live_cloud_processor.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ffs_viewer::geometry {
namespace {

constexpr int kThreads = 256;
constexpr float kMinDepthM = 0.1F;
constexpr float kMaxDepthM = 10.0F;
constexpr float kHiddenDepthM = 100.0F;

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

struct LiveCloudAllocation {
    GpuPointVertex* d_vertices = nullptr;
    cudaEvent_t ready_event = nullptr;

    ~LiveCloudAllocation() {
        if (ready_event != nullptr)
            cudaEventDestroy(ready_event);
        if (d_vertices != nullptr)
            cudaFree(d_vertices);
    }
};

__global__ void projectLiveCloudKernel(const float* disparity, const std::uint8_t* left_bgr,
                                       int width, int height, float fx, float fy, float cx, float cy,
                                       float baseline_m, GpuPointVertex* vertices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = width * height;
    if (index >= count)
        return;

    const int x = index % width;
    const int y = index / width;
    const float d = disparity[index];
    const float z = d > 0.0F ? fx * baseline_m / d : 0.0F;
    const bool valid = isfinite(d) && isfinite(z) && d > 0.0F && static_cast<float>(x) - d >= 0.0F &&
                       z >= kMinDepthM && z <= kMaxDepthM;
    const int bgr = 3 * index;
    vertices[index] = {
        valid ? (static_cast<float>(x) - cx) * z / fx : 0.0F,
        valid ? -((static_cast<float>(y) - cy) * z / fy) : 0.0F,
        valid ? z : kHiddenDepthM,
        left_bgr[bgr + 2], left_bgr[bgr + 1], left_bgr[bgr], 255,
    };
}

} // namespace

LiveCloudFrame makeLiveCloudFrame(const float* d_disparity, const std::uint8_t* d_left_bgr,
                                  int width, int height,
                                  const io::StereoCalibration& calibration,
                                  cudaStream_t stream) {
    if (d_disparity == nullptr || d_left_bgr == nullptr || width <= 0 || height <= 0 ||
        calibration.left.fx <= 0.0F || calibration.left.fy <= 0.0F || calibration.baseline_m <= 0.0F) {
        throw std::invalid_argument("invalid live GPU point-cloud input");
    }

    const int count = width * height;
    auto allocation = std::make_shared<LiveCloudAllocation>();
    check(cudaMalloc(reinterpret_cast<void**>(&allocation->d_vertices),
                     static_cast<std::size_t>(count) * sizeof(GpuPointVertex)),
          "allocate live GPU point cloud");
    check(cudaEventCreateWithFlags(&allocation->ready_event, cudaEventDisableTiming),
          "create live point-cloud completion event");
    projectLiveCloudKernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        d_disparity, d_left_bgr, width, height, calibration.left.fx, calibration.left.fy,
        calibration.left.cx, calibration.left.cy, calibration.baseline_m, allocation->d_vertices);
    check(cudaGetLastError(), "launch live point-cloud projection");
    check(cudaEventRecord(allocation->ready_event, stream), "record live point-cloud completion event");

    LiveCloudFrame result;
    result.d_vertices = allocation->d_vertices;
    result.point_count = count;
    result.ready_event = allocation->ready_event;
    result.owner = std::move(allocation);
    return result;
}

} // namespace ffs_viewer::geometry

#pragma once

#include "ffs_viewer/geometry/gpu_point_vertex.hpp"
#include "ffs_viewer/io/stereo_source.hpp"

#include <cuda_runtime_api.h>

#include <memory>

namespace ffs_viewer::geometry {

// Dense GPU-resident vertices for the live FFS point cloud. The opaque owner
// keeps both the CUDA allocation and completion event alive for OpenGL upload.
struct LiveCloudFrame {
    const GpuPointVertex* d_vertices = nullptr;
    int point_count = 0;
    cudaEvent_t ready_event = nullptr;
    std::shared_ptr<void> owner;

    bool valid() const noexcept { return d_vertices != nullptr && point_count > 0; }
};

// Projects a FFS disparity map directly on CUDA. Invalid pixels are placed
// outside the viewer frustum, so the OpenGL VBO remains dense and needs no
// CPU-side compaction.
LiveCloudFrame makeLiveCloudFrame(const float* d_disparity, const std::uint8_t* d_left_bgr,
                                  int width, int height,
                                  const io::StereoCalibration& calibration,
                                  cudaStream_t stream);

} // namespace ffs_viewer::geometry

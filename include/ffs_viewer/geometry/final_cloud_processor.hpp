#pragma once

#include "ffs_viewer/geometry/gpu_point_vertex.hpp"
#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/stereo_source.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ffs_viewer::geometry {

struct FinalCloudCpuMesh {
    int width = 0;
    int height = 0;
    int display_step = 1;
    std::vector<float> xyz;
    std::vector<int> cells_valid;
};

struct FinalCloudFrame {
    inference::DisparityFrame disparity;
    const float* d_xyz = nullptr;
    const std::uint8_t* d_valid = nullptr;
    const float* d_left_bgr_chw = nullptr;
    int left_row_offset = 0;
    int left_plane_stride = 0;
    int point_count = 0;
    int display_step = 1;
    float mesh_depth_threshold_m = .01F;
    const std::uint8_t* d_mask = nullptr;
    int mask_width = 0;
    int mask_height = 0;
    int* d_mesh_parent = nullptr;
};

void writeFinalCloudVertices(const FinalCloudFrame& cloud, GpuPointVertex* d_vertices,
                             cudaStream_t stream);
int finalCloudMeshIndexCount(const FinalCloudFrame& cloud);
void writeFinalCloudMeshIndices(const FinalCloudFrame& cloud, std::uint32_t* d_indices,
                                cudaStream_t stream);
// Marks every mask-selected, depth-continuous grid cell, then copies the
// source XYZ data and cell-validity flags for CPU mesh processing.
// Call this before writeFinalCloudMeshIndices().
FinalCloudCpuMesh prepareFinalCloudMeshForCpu(const FinalCloudFrame& cloud, cudaStream_t stream);

// Owns reusable GPU buffers for final-capture geometry. The TensorRT disparity
// buffer remains on GPU until invisible filtering, projection, and denoising
// have all completed.
class FinalCloudProcessor final {
  public:
    FinalCloudProcessor(int width, int height);
    ~FinalCloudProcessor();

    FinalCloudProcessor(const FinalCloudProcessor&) = delete;
    FinalCloudProcessor& operator=(const FinalCloudProcessor&) = delete;

    FinalCloudFrame process(float* d_disparity, const float* d_left_input, cudaStream_t stream,
                            const io::StereoCalibration& calibration,
                            int source_width, int source_height, int left_row_offset,
                            int left_plane_stride, float z_max_m,
                            const std::function<void()>& on_denoise);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ffs_viewer::geometry

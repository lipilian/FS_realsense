#pragma once

#include "ffs_viewer/geometry/live_cloud_processor.hpp"
#include "ffs_viewer/io/stereo_source.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ffs_viewer::inference {

struct InferenceTiming {
    float h2d_ms = 0.0F;
    float inference_ms = 0.0F;
    float d2h_ms = 0.0F;
    float gpu_total_ms = 0.0F;
    float host_total_ms = 0.0F;
};

struct DisparityFrame {
    int width = 0;
    int height = 0;
    std::vector<float> values;
    InferenceTiming timing;
};

class FfsRunner final {
public:
    explicit FfsRunner(std::string engine_dir);
    ~FfsRunner();

    FfsRunner(const FfsRunner&) = delete;
    FfsRunner& operator=(const FfsRunner&) = delete;

    // Runs the fixed-resolution FFS TensorRT engine on a rectified monochrome pair.
    DisparityFrame infer(const io::StereoFrame& stereo);

    // Runs the fixed-resolution FFS TensorRT engine on BGR8 HWC images. The
    // runtime's RGB preprocessing kernel performs the BGR-to-RGB channel reorder.
    DisparityFrame inferBgr(int width, int height, const std::vector<std::uint8_t>& left_bgr,
                            const std::vector<std::uint8_t>& right_bgr);

    // Must be called immediately after inferBgr(). Projects the current GPU
    // disparity and left BGR input directly into CUDA-resident GL vertices.
    geometry::LiveCloudFrame makeLiveCloud(const io::StereoCalibration& calibration);

    int modelWidth() const;
    int modelHeight() const;
    int maxDisparity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ffs_viewer::inference

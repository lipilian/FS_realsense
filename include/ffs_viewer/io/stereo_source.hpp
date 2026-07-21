#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace ffs_viewer::io {

struct CameraIntrinsics {
    int width = 0;
    int height = 0;
    float fx = 0.0F;
    float fy = 0.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    int distortion_model = 0; // distortion model type in realsense
    std::array<float, 5> distortion_coeffs{};
};

struct StereoCalibration {
    CameraIntrinsics left;
    CameraIntrinsics right;
    int fps = 0;
    std::array<float, 9> right_to_left_rotation{};
    std::array<float, 3> right_to_left_translation{};
    float baseline_m = 0.0F;
};

struct StereoFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> left_y8;
    std::vector<std::uint8_t> right_y8;
    std::uint64_t left_frame_number = 0;
    std::uint64_t right_frame_number = 0;
    double left_timestamp_ms = 0.0;
    double right_timestamp_ms = 0.0;
};

class StereoSource {
public:
    virtual ~StereoSource() = default;

    virtual bool next(StereoFrame& frame) = 0;
    virtual StereoCalibration calibration() const = 0;
};

}  // namespace ffs_viewer::io

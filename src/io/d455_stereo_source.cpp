#include "ffs_viewer/io/d455_stereo_source.hpp"

#include <librealsense2/rs.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ffs_viewer::io {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 800;
constexpr int kFps = 15;

CameraIntrinsics toCameraIntrinsics(const rs2_intrinsics& intrinsics) {
    CameraIntrinsics result;
    result.width = intrinsics.width;
    result.height = intrinsics.height;
    result.fx = intrinsics.fx;
    result.fy = intrinsics.fy;
    result.cx = intrinsics.ppx;
    result.cy = intrinsics.ppy;
    result.distortion_model = static_cast<int>(intrinsics.model);
    std::copy(std::begin(intrinsics.coeffs), std::end(intrinsics.coeffs),
              result.distortion_coeffs.begin());
    return result;
}

void validateLiveY8Profile(const rs2::video_stream_profile& profile, int stream_index) {
    if (!profile || profile.stream_type() != RS2_STREAM_INFRARED ||
        profile.stream_index() != stream_index || profile.format() != RS2_FORMAT_Y8 ||
        profile.width() != kWidth || profile.height() != kHeight || profile.fps() != kFps) {
        throw std::runtime_error(
            "D455 must provide Infrared " + std::to_string(stream_index) +
            " as 1280x800 Y8 @ 15 Hz");
    }
}

void copyY8(const rs2::video_frame& source, std::vector<std::uint8_t>& destination) {
    if (!source || source.get_profile().format() != RS2_FORMAT_Y8 ||
        source.get_bytes_per_pixel() != 1) {
        throw std::runtime_error("Expected a valid D455 Y8 infrared frame");
    }

    const int width = source.get_width();
    const int height = source.get_height();
    const int stride = source.get_stride_in_bytes();
    if (width != kWidth || height != kHeight || stride < width) {
        throw std::runtime_error("D455 frame layout differs from the configured 1280x800 profile");
    }

    destination.resize(static_cast<std::size_t>(width) * height);
    const auto* source_bytes = static_cast<const std::uint8_t*>(source.get_data());
    for (int row = 0; row < height; ++row) {
        std::memcpy(destination.data() + static_cast<std::size_t>(row) * width,
                    source_bytes + static_cast<std::size_t>(row) * stride,
                    static_cast<std::size_t>(width));
    }
}

}  // namespace

struct D455StereoSource::Impl {
    rs2::pipeline pipeline;
    bool started = false;
    std::uint64_t dropped_pairs = 0;
};

D455StereoSource::D455StereoSource() : impl_(std::make_unique<Impl>()) {}

D455StereoSource::~D455StereoSource() {
    if (impl_->started) {
        try {
            impl_->pipeline.stop();
        } catch (const rs2::error&) {
            // Destructors must not throw if the camera was disconnected.
        }
    }
}

void D455StereoSource::open() {
    if (impl_->started) {
        throw std::logic_error("D455 stereo source is already open");
    }

    try {
        rs2::config config;
        config.enable_stream(RS2_STREAM_INFRARED, 1, kWidth, kHeight, RS2_FORMAT_Y8, kFps);
        config.enable_stream(RS2_STREAM_INFRARED, 2, kWidth, kHeight, RS2_FORMAT_Y8, kFps);
        const rs2::pipeline_profile pipeline_profile = impl_->pipeline.start(config);

        const auto left_profile =
            pipeline_profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
        const auto right_profile =
            pipeline_profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
        validateLiveY8Profile(left_profile, 1);
        validateLiveY8Profile(right_profile, 2);

        calibration_.left = toCameraIntrinsics(left_profile.get_intrinsics());
        calibration_.right = toCameraIntrinsics(right_profile.get_intrinsics());
        calibration_.fps = kFps;
        const rs2_extrinsics right_to_left = right_profile.get_extrinsics_to(left_profile);
        std::copy(std::begin(right_to_left.rotation), std::end(right_to_left.rotation),
                  calibration_.right_to_left_rotation.begin());
        std::copy(std::begin(right_to_left.translation), std::end(right_to_left.translation),
                  calibration_.right_to_left_translation.begin());
        calibration_.baseline_m = std::sqrt(
            right_to_left.translation[0] * right_to_left.translation[0] +
            right_to_left.translation[1] * right_to_left.translation[1] +
            right_to_left.translation[2] * right_to_left.translation[2]);
        impl_->started = true;
    } catch (...) {
        try {
            impl_->pipeline.stop();
        } catch (const rs2::error&) {
        }
        throw;
    }
}

bool D455StereoSource::isOpen() const noexcept {
    return impl_->started;
}

bool D455StereoSource::next(StereoFrame& frame) {
    if (!impl_->started) {
        throw std::logic_error("D455 stereo source must be opened before reading frames");
    }

    rs2::frameset newest = impl_->pipeline.wait_for_frames();
    rs2::frameset candidate;
    while (impl_->pipeline.poll_for_frames(&candidate)) {
        newest = candidate;
        ++impl_->dropped_pairs;
    }

    const rs2::video_frame left = newest.get_infrared_frame(1);
    const rs2::video_frame right = newest.get_infrared_frame(2);
    if (!left || !right) {
        throw std::runtime_error("D455 frameset does not contain both Infrared 1 and Infrared 2");
    }

    frame.width = left.get_width();
    frame.height = left.get_height();
    frame.left_frame_number = left.get_frame_number();
    frame.right_frame_number = right.get_frame_number();
    frame.left_timestamp_ms = left.get_timestamp();
    frame.right_timestamp_ms = right.get_timestamp();
    copyY8(left, frame.left_y8);
    copyY8(right, frame.right_y8);
    return true;
}

StereoCalibration D455StereoSource::calibration() const {
    if (!impl_->started) {
        throw std::logic_error("D455 stereo source must be opened before reading calibration");
    }
    return calibration_;
}

std::uint64_t D455StereoSource::droppedPairs() const noexcept {
    return impl_->dropped_pairs;
}

}  // namespace ffs_viewer::io

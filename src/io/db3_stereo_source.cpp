#include "ffs_viewer/io/db3_stereo_source.hpp"

#include <librealsense2/rs.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace ffs_viewer::io {
namespace {

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
} // convert rs2_intrinsics to CameraIntrinsics

void validateY8Profile(const rs2::video_stream_profile& profile, int stream_index) {
    if (!profile || profile.stream_type() != RS2_STREAM_INFRARED ||
        profile.stream_index() != stream_index || profile.format() != RS2_FORMAT_Y8) {
        throw std::runtime_error(
            "DB3 must contain Infrared " + std::to_string(stream_index) + " in Y8 format");
    }

    if (profile.width() != 1280 || profile.height() != 800 || profile.fps() <= 0) {
        throw std::runtime_error(
            "DB3 Infrared " + std::to_string(stream_index) +
            " must use the planned 1280x800 Y8 profile with a positive frame rate");
    }
} // validate that the video stream profile is Infrared 1 or Infrared 2 in Y8 format with expected dimensions and positive frame rate

void copyY8(const rs2::video_frame& source, std::vector<std::uint8_t>& destination) {
    if (!source || source.get_profile().format() != RS2_FORMAT_Y8 ||
        source.get_bytes_per_pixel() != 1) {
        throw std::runtime_error("Expected a valid Y8 infrared frame");
    }

    const int width = source.get_width();
    const int height = source.get_height();
    const int stride = source.get_stride_in_bytes();
    if (width <= 0 || height <= 0 || stride < width) {
        throw std::runtime_error("Invalid Y8 infrared frame layout");
    }

    destination.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    const auto* source_bytes = static_cast<const std::uint8_t*>(source.get_data());
    for (int row = 0; row < height; ++row) {
        std::memcpy(destination.data() + static_cast<std::size_t>(row) * width,
                    source_bytes + static_cast<std::size_t>(row) * stride,
                    static_cast<std::size_t>(width));
    }
}

}  // namespace

struct Db3StereoSource::Impl {
    rs2::pipeline pipeline;
    bool started = false;
};

Db3StereoSource::Db3StereoSource(std::string recording_path)
    : recording_path_(std::move(recording_path)), impl_(std::make_unique<Impl>()) {}

Db3StereoSource::~Db3StereoSource() {
    if (impl_->started) {
        try {
            impl_->pipeline.stop();
        } catch (const rs2::error&) {
            // Destructors must not throw. Playback may already have stopped at EOF.
        }
    }
}

void Db3StereoSource::open() {
    if (impl_->started) {
        throw std::logic_error("DB3 stereo source is already open");
    }
    if (!std::filesystem::is_regular_file(recording_path_)) {
        throw std::runtime_error("DB3 recording does not exist: " + recording_path_);
    }

    try {
        rs2::config config;
        config.enable_device_from_file(recording_path_, false);
        const rs2::pipeline_profile pipeline_profile = impl_->pipeline.start(config);
        const rs2::playback playback = pipeline_profile.get_device().as<rs2::playback>();
        if (!playback) {
            throw std::runtime_error("Configured recording is not a playback device");
        }
        playback.set_real_time(false);

        const auto left_profile =
            pipeline_profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
        const auto right_profile =
            pipeline_profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
        validateY8Profile(left_profile, 1);
        validateY8Profile(right_profile, 2);

        if (left_profile.fps() != right_profile.fps()) {
            throw std::runtime_error("Infrared 1 and Infrared 2 have different frame rates");
        }

        calibration_.left = toCameraIntrinsics(left_profile.get_intrinsics());
        calibration_.right = toCameraIntrinsics(right_profile.get_intrinsics());
        calibration_.fps = left_profile.fps();

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

bool Db3StereoSource::isOpen() const noexcept {
    return impl_->started;
}

bool Db3StereoSource::next(StereoFrame& frame) {
    if (!impl_->started) {
        throw std::logic_error("DB3 stereo source must be opened before reading frames");
    }

    const rs2::frameset frames = impl_->pipeline.wait_for_frames();
    const rs2::video_frame left = frames.get_infrared_frame(1);
    const rs2::video_frame right = frames.get_infrared_frame(2);
    if (!left || !right) {
        throw std::runtime_error("Playback frameset does not contain both Infrared 1 and Infrared 2");
    }

    if (left.get_width() != calibration_.left.width ||
        left.get_height() != calibration_.left.height ||
        right.get_width() != calibration_.right.width ||
        right.get_height() != calibration_.right.height) {
        throw std::runtime_error("Playback frame dimensions differ from the recorded calibration profile");
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

StereoCalibration Db3StereoSource::calibration() const {
    if (!impl_->started) {
        throw std::logic_error("DB3 stereo source must be opened before reading calibration");
    }
    return calibration_;
}

}  // namespace ffs_viewer::io

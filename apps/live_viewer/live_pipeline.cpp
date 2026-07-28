#include "live_pipeline.hpp"

#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/d455_stereo_source.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

namespace ffs_viewer::live {
namespace {

cv::Mat bgr(const std::vector<std::uint8_t> &pixels, int width, int height) {
    cv::Mat gray(height, width, CV_8UC1, const_cast<std::uint8_t *>(pixels.data()));
    cv::Mat result;
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    return result;
}

cv::Mat disparityVisualization(const inference::DisparityFrame &disparity) {
    float maximum = 1.F;
    for (const float value : disparity.values) {
        if (std::isfinite(value))
            maximum = std::max(maximum, value);
    }
    cv::Mat source(disparity.height, disparity.width, CV_32F, const_cast<float *>(disparity.values.data()));
    cv::Mat normalized;
    cv::Mat color;
    source.convertTo(normalized, CV_8U, 255.F / maximum);
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    return color;
}

void buildCloud(RenderFrame &output, const inference::DisparityFrame &disparity,
                const io::StereoFrame &frame, const io::StereoCalibration &calibration,
                int point_step, float max_depth_m) {
    output.xyz.reserve(size_t(disparity.width / point_step) * size_t(disparity.height / point_step) * 3);
    output.rgb.reserve(output.xyz.capacity());

    const float focal_baseline = calibration.left.fx * calibration.baseline_m;
    for (int y = 0; y < disparity.height; y += point_step) {
        for (int x = 0; x < disparity.width; x += point_step) {
            const size_t index = size_t(y) * disparity.width + x;
            const float value = disparity.values[index];
            if (!std::isfinite(value) || value <= 0.F)
                continue;
            const float z = focal_baseline / value;
            if (z < .1F || z > max_depth_m)
                continue;
            output.xyz.insert(output.xyz.end(), {(x - calibration.left.cx) * z / calibration.left.fx,
                                                  -(y - calibration.left.cy) * z / calibration.left.fy, z});
            const auto gray = frame.left_y8[index];
            output.rgb.insert(output.rgb.end(), {gray, gray, gray});
        }
    }
}

} // namespace

LivePipeline::LivePipeline(LivePipelineOptions options) : options_(std::move(options)) {
    if (options_.engine_dir.empty() || options_.point_step <= 0 || options_.max_depth_m <= 0.F) {
        throw std::invalid_argument("Invalid live-pipeline options");
    }
    runner_ = std::make_unique<inference::FfsRunner>(options_.engine_dir);
}

LivePipeline::~LivePipeline() {
    stop();
}

void LivePipeline::start() {
    stop();
    {
        std::scoped_lock lock(frame_mutex_);
        latest_frame_.reset();
    }
    {
        std::scoped_lock lock(status_mutex_);
        running_ = true;
        status_ = "Starting...";
    }
    worker_ = std::jthread([this](std::stop_token stop_token) { run(stop_token); });
}

void LivePipeline::stop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
    {
        std::scoped_lock lock(frame_mutex_);
        latest_frame_.reset();
    }
    std::scoped_lock lock(status_mutex_);
    running_ = false;
    status_ = "Stopped";
}

bool LivePipeline::running() const {
    std::scoped_lock lock(status_mutex_);
    return running_;
}

std::string LivePipeline::status() const {
    std::scoped_lock lock(status_mutex_);
    return status_;
}

std::shared_ptr<const RenderFrame> LivePipeline::latestFrame() const {
    std::scoped_lock lock(frame_mutex_);
    return latest_frame_;
}

void LivePipeline::run(std::stop_token stop_token) {
    try {
        io::D455StereoSource source;
        source.open();
        const auto calibration = source.calibration();
        setStatus("Running");

        while (!stop_token.stop_requested()) {
            io::StereoFrame frame;
            source.next(frame);
            if (stop_token.stop_requested())
                break;
            const auto disparity = runner_->infer(frame);
            auto result = std::make_shared<RenderFrame>();
            result->left = bgr(frame.left_y8, frame.width, frame.height);
            result->right = bgr(frame.right_y8, frame.width, frame.height);
            result->disparity = disparityVisualization(disparity);
            buildCloud(*result, disparity, frame, calibration, options_.point_step, options_.max_depth_m);
            std::scoped_lock lock(frame_mutex_);
            latest_frame_ = std::move(result);
        }
    } catch (const std::exception &error) {
        setStatus(std::string("Stopped: ") + error.what());
    }
    std::scoped_lock lock(status_mutex_);
    running_ = false;
}

void LivePipeline::setStatus(std::string status) {
    std::scoped_lock lock(status_mutex_);
    status_ = std::move(status);
}

} // namespace ffs_viewer::live

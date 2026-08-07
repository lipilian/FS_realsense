#include "ffs_viewer/inference/sentech_final_capture_pipeline.hpp"

#include "ffs_viewer/inference/fs_runner.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ffs_viewer::inference {
namespace {

constexpr int kCropPixelsPerSide = 8;
constexpr int kWidth = 608;
constexpr int kHeight = 512;
constexpr float kMaxDepthM = 10.0F;

void requireBgr(const io::BgrFrame &frame, const char *name) {
    const std::size_t expected = static_cast<std::size_t>(frame.width) * frame.height * 3U;
    if (!frame.valid() || frame.pixels.size() != expected)
        throw std::invalid_argument(std::string(name) + " must be a valid BGR8 frame");
}

io::BgrFrame resizeBgr(const io::BgrFrame &source, const cv::Rect &crop) {
    const cv::Mat source_image(source.height, source.width, CV_8UC3,
                               const_cast<std::uint8_t *>(source.pixels.data()));
    cv::Mat resized;
    cv::resize(source_image(crop), resized, cv::Size(kWidth, kHeight), 0.0, 0.0, cv::INTER_AREA);
    io::BgrFrame result;
    result.width = resized.cols;
    result.height = resized.rows;
    result.frame_id = source.frame_id;
    result.timestamp_ns = source.timestamp_ns;
    result.pixels.resize(resized.total() * resized.elemSize());
    std::memcpy(result.pixels.data(), resized.data, result.pixels.size());
    return result;
}

io::StereoCalibration makeCalibration(const calibration::RectifiedStereoCamera &camera) {
    const int cropped_width = camera.width - 2 * kCropPixelsPerSide;
    if (cropped_width <= 0 || camera.height <= 0 || camera.fx <= 0.0 || camera.fy <= 0.0 ||
        camera.baseline_m <= 0.0) {
        throw std::invalid_argument("Invalid rectified camera for final Sentech capture");
    }
    const float sx = static_cast<float>(kWidth) / cropped_width;
    const float sy = static_cast<float>(kHeight) / camera.height;
    io::StereoCalibration result;
    result.left.width = kWidth;
    result.left.height = kHeight;
    result.left.fx = static_cast<float>(camera.fx) * sx;
    result.left.fy = static_cast<float>(camera.fy) * sy;
    result.left.cx = (static_cast<float>(camera.cx) - kCropPixelsPerSide) * sx;
    result.left.cy = static_cast<float>(camera.cy) * sy;
    result.right = result.left;
    result.baseline_m = static_cast<float>(camera.baseline_m);
    return result;
}

io::BgrFrame visualizeDisparity(const inference::DisparityFrame &disparity,
                                const io::BgrFrame &source) {
    const cv::Mat disparity_image(disparity.height, disparity.width, CV_32F,
                                  const_cast<float *>(disparity.values.data()));
    float maximum = 1.0F;
    for (const float value : disparity.values) {
        if (std::isfinite(value) && value > 0.0F)
            maximum = std::max(maximum, value);
    }
    cv::Mat normalized, color;
    disparity_image.convertTo(normalized, CV_8U, 255.0F / maximum);
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    for (int y = 0; y < disparity.height; ++y) {
        const float *input = disparity_image.ptr<float>(y);
        cv::Vec3b *output = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < disparity.width; ++x) {
            if (!std::isfinite(input[x]) || input[x] <= 0.0F)
                output[x] = cv::Vec3b{};
        }
    }
    io::BgrFrame result;
    result.width = color.cols;
    result.height = color.rows;
    result.frame_id = source.frame_id;
    result.timestamp_ns = source.timestamp_ns;
    result.pixels.resize(color.total() * color.elemSize());
    std::memcpy(result.pixels.data(), color.data, result.pixels.size());
    return result;
}

} // namespace

struct SentechFinalCapturePipeline::Impl {
    explicit Impl(std::filesystem::path directory) : engine_directory(std::move(directory)) {}

    std::filesystem::path engine_directory;
    mutable std::mutex mutex;
    std::jthread worker;
    std::unique_ptr<FsRunner> runner;
    bool running = false;
    std::string status = "Final capture idle";
    std::shared_ptr<const SentechFinalCaptureResult> latest_result;
};

SentechFinalCapturePipeline::SentechFinalCapturePipeline(std::filesystem::path engine_directory)
    : impl_(std::make_unique<Impl>(std::move(engine_directory))) {}

SentechFinalCapturePipeline::~SentechFinalCapturePipeline() {
    if (impl_->worker.joinable()) {
        impl_->worker.request_stop();
        impl_->worker.join();
    }
}

void SentechFinalCapturePipeline::capture(
    const io::BgrFrame &rectified_left, const io::BgrFrame &rectified_right,
    const calibration::RectifiedStereoCamera &rectified_camera) {
    requireBgr(rectified_left, "Rectified left frame");
    requireBgr(rectified_right, "Rectified right frame");
    if (rectified_left.width != rectified_right.width || rectified_left.height != rectified_right.height)
        throw std::invalid_argument("Final capture requires equal rectified image dimensions");
    if (rectified_left.width <= 2 * kCropPixelsPerSide)
        throw std::invalid_argument("Final capture images are too narrow for the shared crop");
    if (!std::filesystem::is_regular_file(impl_->engine_directory / "fs.engine"))
        throw std::runtime_error("Sentech FS engine is missing from " + impl_->engine_directory.string());

    const cv::Rect crop(kCropPixelsPerSide, 0,
                        rectified_left.width - 2 * kCropPixelsPerSide, rectified_left.height);
    const io::BgrFrame left = resizeBgr(rectified_left, crop);
    const io::BgrFrame right = resizeBgr(rectified_right, crop);
    const io::StereoCalibration camera = makeCalibration(rectified_camera);

    {
        std::scoped_lock lock(impl_->mutex);
        if (impl_->running)
            throw std::runtime_error("A final Sentech capture is already running");
        impl_->running = true;
        impl_->status = "Preparing FoundationStereo final capture...";
    }
    if (impl_->worker.joinable())
        impl_->worker.join();
    impl_->worker = std::jthread([this, left, right, camera](std::stop_token stop_token) {
        try {
            if (!impl_->runner) {
                {
                    std::scoped_lock lock(impl_->mutex);
                    impl_->status = "Loading FoundationStereo 608 x 512 engine...";
                }
                impl_->runner = std::make_unique<FsRunner>(impl_->engine_directory.string());
                if (impl_->runner->modelWidth() != kWidth || impl_->runner->modelHeight() != kHeight ||
                    impl_->runner->padTop() != 0 || impl_->runner->padBottom() != 0) {
                    throw std::runtime_error("Sentech FS engine must be an unpadded 608 x 512 engine");
                }
            }
            if (stop_token.stop_requested())
                return;
            {
                std::scoped_lock lock(impl_->mutex);
                impl_->status = "Running FoundationStereo and cloud post-processing...";
            }
            auto cloud = impl_->runner->inferFinalBgr(
                kWidth, kHeight, left.pixels, right.pixels, camera, kMaxDepthM,
                [this] {
                    std::scoped_lock lock(impl_->mutex);
                    impl_->status = "FoundationStereo complete; denoising final cloud...";
                });
            if (stop_token.stop_requested())
                return;
            auto result = std::make_shared<SentechFinalCaptureResult>();
            result->left = left;
            result->right = right;
            result->disparity_visualization = visualizeDisparity(cloud.disparity, left);
            result->cloud = std::move(cloud);
            std::scoped_lock lock(impl_->mutex);
            impl_->latest_result = std::move(result);
            impl_->status = "Final capture complete; press Draw to select a surface";
        } catch (const std::exception &error) {
            std::scoped_lock lock(impl_->mutex);
            impl_->status = "Final capture failed: " + std::string(error.what());
        }
        std::scoped_lock lock(impl_->mutex);
        impl_->running = false;
    });
}

bool SentechFinalCapturePipeline::running() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->running;
}

std::string SentechFinalCapturePipeline::status() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->status;
}

std::shared_ptr<const SentechFinalCaptureResult> SentechFinalCapturePipeline::latestResult() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->latest_result;
}

} // namespace ffs_viewer::inference

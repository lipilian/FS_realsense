#include "ffs_viewer/inference/sentech_ffs_pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ffs_viewer::inference {
namespace {

constexpr int kCropPixelsPerSide = 8;
constexpr int kFfsWidth = 608;
constexpr int kFfsHeight = 512;

void requireBgr(const io::BgrFrame &frame, const char *name) {
    const std::size_t expected = static_cast<std::size_t>(frame.width) *
                                 static_cast<std::size_t>(frame.height) * 3U;
    if (!frame.valid() || frame.pixels.size() != expected)
        throw std::invalid_argument(std::string(name) + " must be a valid BGR8 frame");
}

struct FfsBgrInput {
    std::vector<std::uint8_t> left_bgr;
    std::vector<std::uint8_t> right_bgr;
};

FfsBgrInput makeFfsInput(const io::BgrFrame &left, const io::BgrFrame &right) {
    requireBgr(left, "Rectified left frame");
    requireBgr(right, "Rectified right frame");
    if (left.width != right.width || left.height != right.height)
        throw std::invalid_argument("FFS requires equal left/right rectified image dimensions");
    if (left.width <= 2 * kCropPixelsPerSide || left.height <= 0)
        throw std::invalid_argument("Rectified images are too small for the shared 8-pixel FFS crop");

    const cv::Mat left_bgr(left.height, left.width, CV_8UC3,
                           const_cast<std::uint8_t *>(left.pixels.data()));
    const cv::Mat right_bgr(right.height, right.width, CV_8UC3,
                            const_cast<std::uint8_t *>(right.pixels.data()));
    const int cropped_width = left.width - 2 * kCropPixelsPerSide;
    const cv::Rect shared_crop(kCropPixelsPerSide, 0, cropped_width, left.height);
    const cv::Mat left_crop = left_bgr(shared_crop);
    const cv::Mat right_crop = right_bgr(shared_crop);

    cv::Mat left_small_bgr, right_small_bgr;
    cv::resize(left_crop, left_small_bgr, cv::Size(kFfsWidth, kFfsHeight), 0.0, 0.0,
               cv::INTER_AREA);
    cv::resize(right_crop, right_small_bgr, cv::Size(kFfsWidth, kFfsHeight), 0.0, 0.0,
               cv::INTER_AREA);

    FfsBgrInput input;
    input.left_bgr.assign(left_small_bgr.datastart, left_small_bgr.dataend);
    input.right_bgr.assign(right_small_bgr.datastart, right_small_bgr.dataend);
    return input;
}

void buildPointCloud(const DisparityFrame &disparity, const std::vector<std::uint8_t> &left_bgr,
                     const SentechFfsCameraModel &camera, SentechFfsResult &result) {
    constexpr int kPointStep = 1;
    constexpr float kMinDepthM = 0.1F;
    constexpr float kMaxDepthM = 10.0F;
    const std::size_t image_bytes = static_cast<std::size_t>(disparity.width) * disparity.height * 3U;
    if (camera.fx <= 0.0F || camera.fy <= 0.0F || camera.baseline_m <= 0.0F ||
        left_bgr.size() != image_bytes)
        return;

    result.xyz.reserve(static_cast<std::size_t>(disparity.width / kPointStep) *
                       static_cast<std::size_t>(disparity.height / kPointStep) * 3U);
    result.rgb.reserve(result.xyz.capacity());
    for (int y = 0; y < disparity.height; y += kPointStep) {
        for (int x = 0; x < disparity.width; x += kPointStep) {
            const std::size_t pixel = static_cast<std::size_t>(y) * disparity.width + x;
            const float disparity_px = disparity.values.at(pixel);
            if (!std::isfinite(disparity_px) || disparity_px <= 0.0F ||
                static_cast<float>(x) - disparity_px < 0.0F)
                continue;
            const float z = camera.fx * camera.baseline_m / disparity_px;
            if (!std::isfinite(z) || z < kMinDepthM || z > kMaxDepthM)
                continue;
            result.xyz.insert(result.xyz.end(), {
                (static_cast<float>(x) - camera.cx) * z / camera.fx,
                -(static_cast<float>(y) - camera.cy) * z / camera.fy,
                z,
            });
            const std::size_t bgr = 3U * pixel;
            result.rgb.insert(result.rgb.end(), {left_bgr[bgr + 2], left_bgr[bgr + 1],
                                                  left_bgr[bgr]});
        }
    }
}

io::BgrFrame visualizeDisparity(const DisparityFrame &disparity, const io::BgrFrame &source) {
    if (disparity.width <= 0 || disparity.height <= 0 ||
        disparity.values.size() != static_cast<std::size_t>(disparity.width) * disparity.height) {
        throw std::runtime_error("FFS returned an invalid disparity frame");
    }

    float maximum = 1.0F;
    for (const float value : disparity.values) {
        if (std::isfinite(value) && value > 0.0F)
            maximum = std::max(maximum, value);
    }

    const cv::Mat disparity_image(disparity.height, disparity.width, CV_32F,
                                  const_cast<float *>(disparity.values.data()));
    cv::Mat normalized, color;
    disparity_image.convertTo(normalized, CV_8U, 255.0F / maximum);
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);

    for (int y = 0; y < disparity.height; ++y) {
        const float *disparity_row = disparity_image.ptr<float>(y);
        cv::Vec3b *color_row = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < disparity.width; ++x) {
            if (!std::isfinite(disparity_row[x]) || disparity_row[x] <= 0.0F)
                color_row[x] = cv::Vec3b{};
        }
    }

    io::BgrFrame visualization;
    visualization.width = color.cols;
    visualization.height = color.rows;
    visualization.frame_id = source.frame_id;
    visualization.timestamp_ns = source.timestamp_ns;
    visualization.pixels.resize(color.total() * color.elemSize());
    std::memcpy(visualization.pixels.data(), color.data, visualization.pixels.size());
    return visualization;
}

} // namespace

struct SentechFfsPipeline::Impl {
    explicit Impl(std::filesystem::path directory) : engine_directory(std::move(directory)) {}

    std::filesystem::path engine_directory;
    mutable std::mutex mutex;
    std::condition_variable input_ready;
    std::jthread worker;
    bool running = false;
    bool has_pending_input = false;
    std::uint64_t last_submitted_left_frame_id = 0;
    std::uint64_t last_submitted_right_frame_id = 0;
    io::BgrFrame pending_left;
    io::BgrFrame pending_right;
    std::shared_ptr<const SentechFfsResult> latest_result;
    SentechFfsCameraModel camera_model;
    bool has_camera_model = false;
    std::string status = "FFS stopped";
};

SentechFfsPipeline::SentechFfsPipeline(std::filesystem::path engine_directory)
    : impl_(std::make_unique<Impl>(std::move(engine_directory))) {}

SentechFfsPipeline::~SentechFfsPipeline() {
    stop();
}

void SentechFfsPipeline::start() {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->running)
        return;
    if (!std::filesystem::is_regular_file(impl_->engine_directory / "fast_foundationstereo.engine") ||
        !std::filesystem::is_regular_file(impl_->engine_directory / "onnx.yaml")) {
        throw std::runtime_error("Sentech FFS engine files are missing from " +
                                 impl_->engine_directory.string());
    }

    impl_->running = true;
    impl_->has_pending_input = false;
    impl_->last_submitted_left_frame_id = 0;
    impl_->last_submitted_right_frame_id = 0;
    impl_->latest_result.reset();
    impl_->status = "Starting FFS...";
    impl_->worker = std::jthread([this](std::stop_token stop_token) {
        try {
            FfsRunner runner(impl_->engine_directory.string());
            if (runner.modelWidth() != kFfsWidth || runner.modelHeight() != kFfsHeight) {
                throw std::runtime_error("Sentech FFS engine must use 608 x 512 input");
            }
            {
                std::scoped_lock worker_lock(impl_->mutex);
                impl_->status = "FFS ready; waiting for rectified frames";
            }

            while (!stop_token.stop_requested()) {
                io::BgrFrame left;
                io::BgrFrame right;
                {
                    std::unique_lock input_lock(impl_->mutex);
                    impl_->input_ready.wait(input_lock, [&] {
                        return stop_token.stop_requested() || impl_->has_pending_input;
                    });
                    if (stop_token.stop_requested())
                        break;
                    left = std::move(impl_->pending_left);
                    right = std::move(impl_->pending_right);
                    impl_->has_pending_input = false;
                }

                const FfsBgrInput input = makeFfsInput(left, right);
                SentechFfsCameraModel camera_model;
                bool has_camera_model = false;
                {
                    std::scoped_lock camera_lock(impl_->mutex);
                    camera_model = impl_->camera_model;
                    has_camera_model = impl_->has_camera_model;
                }
                const DisparityFrame disparity =
                    runner.inferBgr(kFfsWidth, kFfsHeight, input.left_bgr, input.right_bgr);
                auto result = std::make_shared<SentechFfsResult>();
                result->visualization = visualizeDisparity(disparity, left);
                if (has_camera_model)
                    buildPointCloud(disparity, input.left_bgr, camera_model, *result);
                result->left_frame_id = left.frame_id;
                result->right_frame_id = right.frame_id;
                result->timing = disparity.timing;

                std::scoped_lock result_lock(impl_->mutex);
                impl_->latest_result = std::move(result);
                impl_->status = "FFS live: GPU " +
                                std::to_string(disparity.timing.gpu_total_ms) + " ms";
            }
        } catch (const std::exception &error) {
            std::scoped_lock error_lock(impl_->mutex);
            impl_->status = "FFS stopped: " + std::string(error.what());
        }
        std::scoped_lock stopped_lock(impl_->mutex);
        impl_->running = false;
    });
}

void SentechFfsPipeline::stop() {
    if (impl_->worker.joinable()) {
        impl_->worker.request_stop();
        impl_->input_ready.notify_all();
        impl_->worker.join();
    }
    std::scoped_lock lock(impl_->mutex);
    impl_->running = false;
    impl_->has_pending_input = false;
    impl_->pending_left = {};
    impl_->pending_right = {};
    impl_->latest_result.reset();
    impl_->status = "FFS stopped";
}

bool SentechFfsPipeline::running() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->running;
}

std::string SentechFfsPipeline::status() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->status;
}

void SentechFfsPipeline::setCameraModel(const SentechFfsCameraModel &camera_model) {
    const int cropped_width = camera_model.width - 2 * kCropPixelsPerSide;
    if (camera_model.width <= 2 * kCropPixelsPerSide || camera_model.height <= 0 ||
        camera_model.fx <= 0.0F || camera_model.fy <= 0.0F || camera_model.baseline_m <= 0.0F) {
        throw std::invalid_argument("Sentech FFS camera model is invalid");
    }

    const float scale_x = static_cast<float>(kFfsWidth) / cropped_width;
    const float scale_y = static_cast<float>(kFfsHeight) / camera_model.height;
    SentechFfsCameraModel scaled;
    scaled.width = kFfsWidth;
    scaled.height = kFfsHeight;
    scaled.fx = camera_model.fx * scale_x;
    scaled.fy = camera_model.fy * scale_y;
    scaled.cx = (camera_model.cx - kCropPixelsPerSide) * scale_x;
    scaled.cy = camera_model.cy * scale_y;
    scaled.baseline_m = camera_model.baseline_m;

    std::scoped_lock lock(impl_->mutex);
    impl_->camera_model = scaled;
    impl_->has_camera_model = true;
}

void SentechFfsPipeline::submit(const io::BgrFrame &rectified_left,
                                const io::BgrFrame &rectified_right) {
    if (!rectified_left.valid() || !rectified_right.valid())
        return;
    std::scoped_lock lock(impl_->mutex);
    if (!impl_->running ||
        (rectified_left.frame_id == impl_->last_submitted_left_frame_id &&
         rectified_right.frame_id == impl_->last_submitted_right_frame_id)) {
        return;
    }
    impl_->pending_left = rectified_left;
    impl_->pending_right = rectified_right;
    impl_->last_submitted_left_frame_id = rectified_left.frame_id;
    impl_->last_submitted_right_frame_id = rectified_right.frame_id;
    impl_->has_pending_input = true;
    impl_->input_ready.notify_one();
}

std::shared_ptr<const SentechFfsResult> SentechFfsPipeline::latestResult() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->latest_result;
}

} // namespace ffs_viewer::inference

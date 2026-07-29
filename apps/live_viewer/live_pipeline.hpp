#pragma once

#include "ffs_viewer/io/stereo_source.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <opencv2/core/mat.hpp>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace ffs_viewer::inference {
class FfsRunner;
class FsRunner;
}

namespace ffs_viewer::live {

struct RenderFrame {
    cv::Mat left;
    cv::Mat right;
    cv::Mat disparity;
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
};

struct LivePipelineOptions {
    std::string engine_dir;
    std::string fs_engine_dir;
    int point_step = 4;
    float max_depth_m = 10.F;
};

class LivePipeline {
  public:
    explicit LivePipeline(LivePipelineOptions options);
    ~LivePipeline();

    LivePipeline(const LivePipeline &) = delete;
    LivePipeline &operator=(const LivePipeline &) = delete;

    void start();
    void capture();
    void stop();
    bool running() const;
    std::string status() const;
    std::shared_ptr<const RenderFrame> latestFrame() const;

  private:
    void run(std::stop_token stop_token);
    void setStatus(std::string status);

    LivePipelineOptions options_;
    std::unique_ptr<inference::FfsRunner> runner_;
    // Loaded for the complete viewer lifetime. It is intentionally not called
    // by the live FFS preview loop; final capture will use it in a later step.
    std::unique_ptr<inference::FsRunner> fs_runner_;
    std::jthread worker_;
    mutable std::mutex capture_mutex_;
    std::optional<io::StereoFrame> latest_stereo_;
    std::optional<io::StereoFrame> pending_capture_;
    mutable std::mutex frame_mutex_;
    std::shared_ptr<const RenderFrame> latest_frame_;
    mutable std::mutex status_mutex_;
    std::string status_ = "Stopped";
    bool running_ = false;
};

} // namespace ffs_viewer::live

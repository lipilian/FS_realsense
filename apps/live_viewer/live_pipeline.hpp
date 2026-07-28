#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

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
    void stop();
    bool running() const;
    std::string status() const;
    std::shared_ptr<const RenderFrame> latestFrame() const;

  private:
    void run(std::stop_token stop_token);
    void setStatus(std::string status);

    LivePipelineOptions options_;
    std::jthread worker_;
    mutable std::mutex frame_mutex_;
    std::shared_ptr<const RenderFrame> latest_frame_;
    mutable std::mutex status_mutex_;
    std::string status_ = "Stopped";
    bool running_ = false;
};

} // namespace ffs_viewer::live

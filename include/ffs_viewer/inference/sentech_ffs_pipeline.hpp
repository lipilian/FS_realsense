#pragma once

#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/sentech_stereo_source.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ffs_viewer::inference {

struct SentechFfsCameraModel {
    int width = 0;
    int height = 0;
    float fx = 0.0F;
    float fy = 0.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    float baseline_m = 0.0F;
};

struct SentechFfsResult {
    io::BgrFrame visualization;
    std::uint64_t left_frame_id = 0;
    std::uint64_t right_frame_id = 0;
    InferenceTiming timing;
    geometry::LiveCloudFrame gpu_cloud;
};

// Runs FFS on the latest rectified Sentech pair. Inputs are deliberately not
// queued: when inference is slower than acquisition, old pairs are discarded.
class SentechFfsPipeline final {
  public:
    explicit SentechFfsPipeline(std::filesystem::path engine_directory);
    ~SentechFfsPipeline();

    SentechFfsPipeline(const SentechFfsPipeline &) = delete;
    SentechFfsPipeline &operator=(const SentechFfsPipeline &) = delete;

    void start();
    void stop();
    bool running() const;
    std::string status() const;

    // The frames must already be stereo-rectified BGR8 images. The pipeline
    // removes 8 px from both left and right edges of both images, then
    // downsamples the common 2432 x 2048 crop to 608 x 512 BGR8.
    void setCameraModel(const SentechFfsCameraModel &camera_model);
    void submit(const io::BgrFrame &rectified_left, const io::BgrFrame &rectified_right);
    std::shared_ptr<const SentechFfsResult> latestResult() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ffs_viewer::inference

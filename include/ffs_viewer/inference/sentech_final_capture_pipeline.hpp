#pragma once

#include "ffs_viewer/calibration/stereo_rectifier.hpp"
#include "ffs_viewer/geometry/final_cloud_processor.hpp"
#include "ffs_viewer/io/sentech_stereo_source.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace ffs_viewer::inference {

struct SentechFinalCaptureResult {
    io::BgrFrame left;
    io::BgrFrame right;
    io::BgrFrame disparity_visualization;
    geometry::FinalCloudFrame cloud;
};

// Performs a frozen, high-quality FoundationStereo capture and preserves the
// CUDA cloud buffers for the full-resolution Draw/mesh/area workflow.
class SentechFinalCapturePipeline final {
  public:
    explicit SentechFinalCapturePipeline(std::filesystem::path engine_directory);
    ~SentechFinalCapturePipeline();

    SentechFinalCapturePipeline(const SentechFinalCapturePipeline &) = delete;
    SentechFinalCapturePipeline &operator=(const SentechFinalCapturePipeline &) = delete;

    void capture(const io::BgrFrame &rectified_left, const io::BgrFrame &rectified_right,
                 const calibration::RectifiedStereoCamera &rectified_camera);
    bool running() const;
    std::string status() const;
    std::shared_ptr<const SentechFinalCaptureResult> latestResult() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ffs_viewer::inference

#pragma once

#include "ffs_viewer/geometry/final_cloud_processor.hpp"
#include "ffs_viewer/inference/ffs_runner.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ffs_viewer::inference {

// Adapter for fixed-resolution FoundationStereo TensorRT engines. The legacy
// 960x608 engine retains its 4-row top/bottom padding; the Sentech 608x512
// engine uses its full image height without padding.
class FsRunner final {
public:
    // Accepts either an fs.engine path or the directory containing fs.engine.
    explicit FsRunner(std::string engine_path);
    ~FsRunner();

    FsRunner(const FsRunner&) = delete;
    FsRunner& operator=(const FsRunner&) = delete;


    // Runs FS and processes its final disparity and organized cloud on CUDA.
    geometry::FinalCloudFrame inferFinal(const io::StereoFrame& stereo,
                                         const io::StereoCalibration& calibration,
                                         float z_max_m,
                                         const std::function<void()>& on_denoise);
    geometry::FinalCloudFrame inferFinalBgr(int width, int height,
                                            const std::vector<std::uint8_t>& left_bgr,
                                            const std::vector<std::uint8_t>& right_bgr,
                                            const io::StereoCalibration& calibration,
                                            float z_max_m,
                                            const std::function<void()>& on_denoise);
    // Returns an unpadded 960x600 disparity map in model-input pixel units.
    DisparityFrame infer(const io::StereoFrame& stereo);

    int modelWidth() const;
    int modelHeight() const;
    int contentWidth() const;
    int contentHeight() const;
    int padTop() const;
    int padBottom() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ffs_viewer::inference

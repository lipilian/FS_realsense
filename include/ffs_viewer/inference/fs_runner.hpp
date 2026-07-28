#pragma once

#include "ffs_viewer/inference/ffs_runner.hpp"

#include <memory>
#include <string>

namespace ffs_viewer::inference {

// Adapter for the fixed 960x608, 32-iteration FoundationStereo TensorRT engine.
// It accepts D455 Y8 images, replicates the monochrome samples into RGB, resizes
// content to 960x600, and applies FoundationStereo's 4-row top/bottom padding.
class FsRunner final {
public:
    // Accepts either an fs.engine path or the directory containing fs.engine.
    explicit FsRunner(std::string engine_path);
    ~FsRunner();

    FsRunner(const FsRunner&) = delete;
    FsRunner& operator=(const FsRunner&) = delete;

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

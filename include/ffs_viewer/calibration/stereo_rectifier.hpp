#pragma once

#include "ffs_viewer/calibration/stereo_charuco_calibrator.hpp"
#include "ffs_viewer/io/sentech_stereo_source.hpp"

#include <memory>

namespace ffs_viewer::calibration {

struct RectifiedStereoCamera {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline_m = 0.0;
};

class StereoRectifier final {
  public:
    StereoRectifier();
    ~StereoRectifier();

    StereoRectifier(const StereoRectifier &) = delete;
    StereoRectifier &operator=(const StereoRectifier &) = delete;

    void setCalibration(const StereoCharucoCalibrationResult &calibration);
    bool hasCalibration() const noexcept;
    RectifiedStereoCamera rectifiedCamera(int width, int height);
    void rectify(const io::BgrFrame &left, const io::BgrFrame &right,
                 io::BgrFrame &rectified_left, io::BgrFrame &rectified_right);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ffs_viewer::calibration

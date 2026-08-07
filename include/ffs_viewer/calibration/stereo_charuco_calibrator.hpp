#pragma once

#include "ffs_viewer/calibration/live_charuco_detector.hpp"

#include <array>
#include <filesystem>
#include <vector>

namespace ffs_viewer::calibration {

struct StereoCharucoCalibrationResult {
    double left_rms = 0.0;
    double right_rms = 0.0;
    double stereo_rms = 0.0;
    int single_camera_pair_count = 0;
    int stereo_pair_count = 0;
    std::array<double, 9> left_camera_matrix{};
    std::array<double, 9> right_camera_matrix{};
    std::vector<double> left_distortion;
    std::vector<double> right_distortion;
    std::array<double, 9> right_to_left_rotation{};
    std::array<double, 3> right_to_left_translation{};
};

StereoCharucoCalibrationResult calibrateStereoCharuco(
    const CharucoBoardConfig &config, const std::vector<CharucoDetection> &left_detections,
    const std::vector<CharucoDetection> &right_detections);

void saveStereoCharucoCalibration(const std::filesystem::path &path,
                                  const CharucoBoardConfig &config,
                                  const StereoCharucoCalibrationResult &result);
bool loadStereoCharucoCalibration(const std::filesystem::path &path,
                                  CharucoBoardConfig &config,
                                  StereoCharucoCalibrationResult &result);

} // namespace ffs_viewer::calibration

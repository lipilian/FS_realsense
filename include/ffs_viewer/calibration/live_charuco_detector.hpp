#pragma once

#include "ffs_viewer/io/sentech_stereo_source.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ffs_viewer::calibration {

struct CharucoBoardConfig {
    int squares_x = 8;
    int squares_y = 5;
    float square_length_m = 0.054F;
    float marker_length_m = 0.040F;
    std::string dictionary_name = "DICT_4X4_250";
};

struct CharucoCorner {
    int id = -1;
    float x = 0.0F;
    float y = 0.0F;
};

struct CharucoDetection {
    io::BgrFrame annotated_frame;
    int marker_count = 0;
    int corner_count = 0;
    std::vector<CharucoCorner> corners;

    bool valid() const {
        return corner_count >= 4;
    }
};

class LiveCharucoDetector final {
  public:
    LiveCharucoDetector();
    ~LiveCharucoDetector();

    LiveCharucoDetector(const LiveCharucoDetector &) = delete;
    LiveCharucoDetector &operator=(const LiveCharucoDetector &) = delete;

    // Validates and applies a new board without changing camera acquisition.
    void setBoardConfig(const CharucoBoardConfig &config);
    CharucoBoardConfig boardConfig() const;

    // Stores and restores board parameters as JSON. loadBoardConfig() returns false if absent.
    void saveBoardConfig(const std::filesystem::path &path) const;
    bool loadBoardConfig(const std::filesystem::path &path);

    // Detects the configured ChArUco board in one BGR frame.
    void detect(const io::BgrFrame &frame, CharucoDetection &result) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ffs_viewer::calibration

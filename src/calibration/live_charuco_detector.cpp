#include "ffs_viewer/calibration/live_charuco_detector.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ffs_viewer::calibration {
namespace {

constexpr int kMinimumCharucoCorners = 4;

int dictionaryIdFromName(const std::string &name) {
    static constexpr std::array<std::pair<const char *, int>, 16> kDictionaries{{
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    }};
    for (const auto &[dictionary_name, dictionary_id] : kDictionaries) {
        if (name == dictionary_name)
            return dictionary_id;
    }
    throw std::invalid_argument("Unsupported ArUco dictionary: " + name);
}

void validateConfig(const CharucoBoardConfig &config) {
    if (config.squares_x < 2 || config.squares_y < 2)
        throw std::invalid_argument("ChArUco board needs at least 2 squares in each direction");
    if (config.square_length_m <= 0.0F || config.marker_length_m <= 0.0F)
        throw std::invalid_argument("ChArUco lengths must be positive");
    if (config.marker_length_m >= config.square_length_m)
        throw std::invalid_argument("Marker length must be smaller than square length");
    (void)dictionaryIdFromName(config.dictionary_name);
}

} // namespace

struct LiveCharucoDetector::Impl {
    CharucoBoardConfig config;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::CharucoBoard> board;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    Impl() {
        configure(config);
    }

    void configure(const CharucoBoardConfig &new_config) {
        validateConfig(new_config);
        const cv::Ptr<cv::aruco::Dictionary> new_dictionary =
            cv::aruco::getPredefinedDictionary(dictionaryIdFromName(new_config.dictionary_name));
        const cv::Ptr<cv::aruco::CharucoBoard> new_board = cv::aruco::CharucoBoard::create(
            new_config.squares_x, new_config.squares_y, new_config.square_length_m,
            new_config.marker_length_m, new_dictionary);
        dictionary = new_dictionary;
        board = new_board;
        config = new_config;
    }
};

LiveCharucoDetector::LiveCharucoDetector() : impl_(std::make_unique<Impl>()) {}

LiveCharucoDetector::~LiveCharucoDetector() = default;

void LiveCharucoDetector::setBoardConfig(const CharucoBoardConfig &config) {
    impl_->configure(config);
}

CharucoBoardConfig LiveCharucoDetector::boardConfig() const {
    return impl_->config;
}

void LiveCharucoDetector::detect(const io::BgrFrame &frame, CharucoDetection &result) const {
    result = {};
    if (!frame.valid())
        return;

    const std::size_t expected_bytes =
        static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) * 3U;
    if (frame.width <= 0 || frame.height <= 0 || frame.pixels.size() != expected_bytes)
        throw std::runtime_error("Invalid BGR frame layout for ChArUco detection");

    result.annotated_frame = frame;
    cv::Mat overlay(result.annotated_frame.height, result.annotated_frame.width, CV_8UC3,
                    result.annotated_frame.pixels.data());
    cv::Mat gray;
    cv::cvtColor(overlay, gray, cv::COLOR_BGR2GRAY);

    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> marker_ids;
    cv::aruco::detectMarkers(gray, impl_->dictionary, marker_corners, marker_ids, impl_->parameters);
    result.marker_count = static_cast<int>(marker_ids.size());

    if (!marker_ids.empty()) {
        cv::aruco::drawDetectedMarkers(overlay, marker_corners, marker_ids);

        cv::Mat charuco_corners;
        cv::Mat charuco_ids;
        cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, gray, impl_->board,
                                              charuco_corners, charuco_ids);
        result.corner_count = charuco_ids.rows;
        if (result.corner_count >= kMinimumCharucoCorners)
            cv::aruco::drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids,
                                                  cv::Scalar(0, 255, 0));
    }

    const std::string status = "Markers: " + std::to_string(result.marker_count) +
                               "  ChArUco corners: " + std::to_string(result.corner_count);
    const cv::Scalar color = result.valid() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(overlay, status, cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                cv::LINE_AA);
}

} // namespace ffs_viewer::calibration

#include "ffs_viewer/calibration/live_charuco_detector.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
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

void LiveCharucoDetector::saveBoardConfig(const std::filesystem::path &path) const {
    cv::FileStorage storage(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened())
        throw std::runtime_error("Unable to write ChArUco board config: " + path.string());

    const CharucoBoardConfig config = boardConfig();
    storage << "squares_x" << config.squares_x;
    storage << "squares_y" << config.squares_y;
    storage << "square_length_m" << config.square_length_m;
    storage << "marker_length_m" << config.marker_length_m;
    storage << "dictionary_name" << config.dictionary_name;
}

bool LiveCharucoDetector::loadBoardConfig(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path))
        return false;

    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened())
        throw std::runtime_error("Unable to read ChArUco board config: " + path.string());

    const cv::FileNode squares_x = storage["squares_x"];
    const cv::FileNode squares_y = storage["squares_y"];
    const cv::FileNode square_length_m = storage["square_length_m"];
    const cv::FileNode marker_length_m = storage["marker_length_m"];
    const cv::FileNode dictionary_name = storage["dictionary_name"];
    if (squares_x.empty() || squares_y.empty() || square_length_m.empty() ||
        marker_length_m.empty() || dictionary_name.empty()) {
        throw std::runtime_error("ChArUco board config is missing required fields: " + path.string());
    }

    CharucoBoardConfig config;
    config.squares_x = static_cast<int>(squares_x);
    config.squares_y = static_cast<int>(squares_y);
    config.square_length_m = static_cast<float>(square_length_m.real());
    config.marker_length_m = static_cast<float>(marker_length_m.real());
    config.dictionary_name = static_cast<std::string>(dictionary_name);
    setBoardConfig(config);
    return true;
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
        result.corners.reserve(static_cast<std::size_t>(result.corner_count));
        for (int index = 0; index < result.corner_count; ++index) {
            const cv::Point2f point = charuco_corners.at<cv::Point2f>(index);
            result.corners.push_back({charuco_ids.at<int>(index), point.x, point.y});
        }
        if (result.corner_count >= kMinimumCharucoCorners)
            cv::aruco::drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids,
                                                  cv::Scalar(0, 255, 0));
    }

}

} // namespace ffs_viewer::calibration

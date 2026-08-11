#include "ffs_viewer/calibration/stereo_charuco_calibrator.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ffs_viewer::calibration {
namespace {

int dictionaryIdFromName(const std::string &name) {
    static const std::unordered_map<std::string, int> dictionaries{
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50}, {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250}, {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50}, {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250}, {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50}, {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250}, {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50}, {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250}, {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    };
    const auto it = dictionaries.find(name);
    if (it == dictionaries.end())
        throw std::invalid_argument("Unsupported ArUco dictionary: " + name);
    return it->second;
}

void validateConfig(const CharucoBoardConfig &config) {
    if (config.squares_x < 2 || config.squares_y < 2 || config.square_length_m <= 0.0F ||
        config.marker_length_m <= 0.0F || config.marker_length_m >= config.square_length_m) {
        throw std::invalid_argument("Invalid ChArUco board configuration");
    }
}

void appendForSingleCamera(const CharucoDetection &detection, std::vector<cv::Mat> &corners,
                           std::vector<cv::Mat> &ids) {
    if (detection.corners.size() < 4)
        return;
    cv::Mat image_corners(static_cast<int>(detection.corners.size()), 1, CV_32FC2);
    cv::Mat image_ids(static_cast<int>(detection.corners.size()), 1, CV_32SC1);
    for (int index = 0; index < image_corners.rows; ++index) {
        const CharucoCorner &corner = detection.corners.at(static_cast<std::size_t>(index));
        image_corners.at<cv::Point2f>(index) = {corner.x, corner.y};
        image_ids.at<int>(index) = corner.id;
    }
    corners.push_back(image_corners);
    ids.push_back(image_ids);
}

void copyMatrix3x3(const cv::Mat &matrix, std::array<double, 9> &output) {
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            output.at(static_cast<std::size_t>(row * 3 + col)) = matrix.at<double>(row, col);
}

std::vector<double> copyVector(const cv::Mat &matrix) {
    std::vector<double> output;
    output.reserve(matrix.total());
    for (std::size_t index = 0; index < matrix.total(); ++index)
        output.push_back(matrix.at<double>(static_cast<int>(index)));
    return output;
}

cv::Mat matrix3x3(const std::array<double, 9> &values) {
    cv::Mat matrix(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            matrix.at<double>(row, col) = values.at(static_cast<std::size_t>(row * 3 + col));
    return matrix;
}

cv::Mat distortionMatrix(const std::vector<double> &values, const char *camera_name) {
    if (values.empty())
        throw std::invalid_argument(std::string(camera_name) + " distortion vector is empty");
    return cv::Mat(values, true).reshape(1, 1);
}

double reprojectionRms(const std::vector<cv::Point2f> &observed,
                       const std::vector<cv::Point2f> &projected) {
    if (observed.size() != projected.size() || observed.empty())
        throw std::invalid_argument("Reprojection points must have equal non-zero size");
    double squared_error_sum = 0.0;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        const cv::Point2f error = observed[index] - projected[index];
        squared_error_sum += static_cast<double>(error.dot(error));
    }
    return std::sqrt(squared_error_sum / static_cast<double>(observed.size()));
}

} // namespace

StereoCharucoCalibrationResult calibrateStereoCharuco(
    const CharucoBoardConfig &config, const std::vector<CharucoDetection> &left_detections,
    const std::vector<CharucoDetection> &right_detections) {
    validateConfig(config);
    if (left_detections.size() != right_detections.size() || left_detections.empty())
        throw std::invalid_argument("Left and right ChArUco detection lists must have equal non-zero size");

    const auto dictionary = cv::aruco::getPredefinedDictionary(dictionaryIdFromName(config.dictionary_name));
    const auto board = cv::aruco::CharucoBoard::create(config.squares_x, config.squares_y,
                                                         config.square_length_m, config.marker_length_m,
                                                         dictionary);
    const cv::Size image_size(left_detections.front().annotated_frame.width,
                              left_detections.front().annotated_frame.height);
    if (image_size.width <= 0 || image_size.height <= 0)
        throw std::runtime_error("Calibration pair has no valid image size");

    std::vector<cv::Mat> left_corners, left_ids, right_corners, right_ids;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> stereo_left_points, stereo_right_points;
    for (std::size_t pair_index = 0; pair_index < left_detections.size(); ++pair_index) {
        const auto &left = left_detections[pair_index];
        const auto &right = right_detections[pair_index];
        if (left.annotated_frame.width != image_size.width || left.annotated_frame.height != image_size.height ||
            right.annotated_frame.width != image_size.width || right.annotated_frame.height != image_size.height) {
            throw std::runtime_error("Calibration pairs have inconsistent image dimensions");
        }
        appendForSingleCamera(left, left_corners, left_ids);
        appendForSingleCamera(right, right_corners, right_ids);

        std::unordered_map<int, CharucoCorner> right_by_id;
        for (const CharucoCorner &corner : right.corners)
            right_by_id.emplace(corner.id, corner);
        std::vector<cv::Point3f> pair_object_points;
        std::vector<cv::Point2f> pair_left_points;
        std::vector<cv::Point2f> pair_right_points;
        for (const CharucoCorner &left_corner : left.corners) {
            const auto right_it = right_by_id.find(left_corner.id);
            if (right_it == right_by_id.end() || left_corner.id < 0 ||
                left_corner.id >= static_cast<int>(board->chessboardCorners.size()))
                continue;
            pair_object_points.push_back(board->chessboardCorners.at(left_corner.id));
            pair_left_points.push_back({left_corner.x, left_corner.y});
            pair_right_points.push_back({right_it->second.x, right_it->second.y});
        }
        if (pair_object_points.size() >= 4) {
            object_points.push_back(std::move(pair_object_points));
            stereo_left_points.push_back(std::move(pair_left_points));
            stereo_right_points.push_back(std::move(pair_right_points));
        }
    }

    if (left_corners.size() < 3 || right_corners.size() < 3)
        throw std::runtime_error("Need at least 3 valid ChArUco detections for each camera");
    if (object_points.size() < 3)
        throw std::runtime_error("Need at least 3 pairs with 4 matched ChArUco corners");

    cv::Mat left_camera_matrix, left_distortion, right_camera_matrix, right_distortion;
    std::vector<cv::Mat> ignored_rvecs, ignored_tvecs;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
    StereoCharucoCalibrationResult result;
    result.left_rms = cv::aruco::calibrateCameraCharuco(left_corners, left_ids, board, image_size,
                                                         left_camera_matrix, left_distortion,
                                                         ignored_rvecs, ignored_tvecs, 0, criteria);
    result.right_rms = cv::aruco::calibrateCameraCharuco(right_corners, right_ids, board, image_size,
                                                          right_camera_matrix, right_distortion,
                                                          ignored_rvecs, ignored_tvecs, 0, criteria);

    cv::Mat rotation, translation, essential, fundamental;
    result.stereo_rms = cv::stereoCalibrate(object_points, stereo_left_points, stereo_right_points,
                                             left_camera_matrix, left_distortion, right_camera_matrix,
                                             right_distortion, image_size, rotation, translation, essential,
                                             fundamental, cv::CALIB_FIX_INTRINSIC, criteria);
    result.single_camera_pair_count = static_cast<int>(std::min(left_corners.size(), right_corners.size()));
    result.stereo_pair_count = static_cast<int>(object_points.size());
    copyMatrix3x3(left_camera_matrix, result.left_camera_matrix);
    copyMatrix3x3(right_camera_matrix, result.right_camera_matrix);
    result.left_distortion = copyVector(left_distortion);
    result.right_distortion = copyVector(right_distortion);
    copyMatrix3x3(rotation, result.right_to_left_rotation);
    for (int index = 0; index < 3; ++index)
        result.right_to_left_translation.at(static_cast<std::size_t>(index)) = translation.at<double>(index);
    return result;
}

StereoCharucoCalibrationCheckResult checkStereoCharucoCalibration(
    const CharucoBoardConfig &config, const StereoCharucoCalibrationResult &calibration,
    const CharucoDetection &left_detection, const CharucoDetection &right_detection) {
    validateConfig(config);
    const cv::Size image_size(left_detection.annotated_frame.width,
                              left_detection.annotated_frame.height);
    if (image_size.width <= 0 || image_size.height <= 0 ||
        right_detection.annotated_frame.width != image_size.width ||
        right_detection.annotated_frame.height != image_size.height) {
        throw std::invalid_argument("Calibration check requires equal-size valid stereo frames");
    }

    const auto dictionary = cv::aruco::getPredefinedDictionary(dictionaryIdFromName(config.dictionary_name));
    const auto board = cv::aruco::CharucoBoard::create(config.squares_x, config.squares_y,
                                                         config.square_length_m, config.marker_length_m,
                                                         dictionary);
    std::unordered_map<int, CharucoCorner> right_by_id;
    for (const CharucoCorner &corner : right_detection.corners)
        right_by_id.emplace(corner.id, corner);

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> left_points;
    std::vector<cv::Point2f> right_points;
    for (const CharucoCorner &left_corner : left_detection.corners) {
        const auto right_it = right_by_id.find(left_corner.id);
        if (right_it == right_by_id.end() || left_corner.id < 0 ||
            left_corner.id >= static_cast<int>(board->chessboardCorners.size())) {
            continue;
        }
        object_points.push_back(board->chessboardCorners.at(left_corner.id));
        left_points.emplace_back(left_corner.x, left_corner.y);
        right_points.emplace_back(right_it->second.x, right_it->second.y);
    }
    if (object_points.size() < 4)
        throw std::runtime_error("Need at least 4 matched ChArUco corners to check calibration");

    const cv::Mat left_camera_matrix = matrix3x3(calibration.left_camera_matrix);
    const cv::Mat right_camera_matrix = matrix3x3(calibration.right_camera_matrix);
    const cv::Mat left_distortion = distortionMatrix(calibration.left_distortion, "Left camera");
    const cv::Mat right_distortion = distortionMatrix(calibration.right_distortion, "Right camera");
    const cv::Mat stereo_rotation = matrix3x3(calibration.right_to_left_rotation);
    cv::Mat stereo_translation(3, 1, CV_64F);
    for (int index = 0; index < 3; ++index)
        stereo_translation.at<double>(index) =
            calibration.right_to_left_translation.at(static_cast<std::size_t>(index));

    cv::Mat left_rvec, left_translation;
    if (!cv::solvePnP(object_points, left_points, left_camera_matrix, left_distortion, left_rvec,
                      left_translation, false, cv::SOLVEPNP_ITERATIVE)) {
        throw std::runtime_error("Unable to estimate ChArUco board pose for calibration check");
    }

    std::vector<cv::Point2f> projected_left;
    cv::projectPoints(object_points, left_rvec, left_translation, left_camera_matrix, left_distortion,
                      projected_left);
    cv::Mat left_rotation;
    cv::Rodrigues(left_rvec, left_rotation);
    const cv::Mat right_rotation = stereo_rotation * left_rotation;
    const cv::Mat right_translation = stereo_rotation * left_translation + stereo_translation;
    cv::Mat right_rvec;
    cv::Rodrigues(right_rotation, right_rvec);
    std::vector<cv::Point2f> projected_right;
    cv::projectPoints(object_points, right_rvec, right_translation, right_camera_matrix,
                      right_distortion, projected_right);

    StereoCharucoCalibrationCheckResult result;
    result.matched_corner_count = static_cast<int>(object_points.size());
    result.left_reprojection_rms = reprojectionRms(left_points, projected_left);
    result.right_reprojection_rms = reprojectionRms(right_points, projected_right);
    result.stereo_reprojection_rms = std::sqrt(
        (result.left_reprojection_rms * result.left_reprojection_rms +
         result.right_reprojection_rms * result.right_reprojection_rms) /
        2.0);
    return result;
}

void saveStereoCharucoCalibration(const std::filesystem::path &path,
                                  const CharucoBoardConfig &config,
                                  const StereoCharucoCalibrationResult &result) {
    cv::FileStorage storage(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened())
        throw std::runtime_error("Unable to write calibration JSON: " + path.string());

    storage << "squares_x" << config.squares_x;
    storage << "squares_y" << config.squares_y;
    storage << "square_length_m" << config.square_length_m;
    storage << "marker_length_m" << config.marker_length_m;
    storage << "dictionary_name" << config.dictionary_name;
    storage << "left_rms" << result.left_rms;
    storage << "right_rms" << result.right_rms;
    storage << "stereo_rms" << result.stereo_rms;
    storage << "single_camera_pair_count" << result.single_camera_pair_count;
    storage << "stereo_pair_count" << result.stereo_pair_count;

    cv::Mat left_camera_matrix(3, 3, CV_64F);
    cv::Mat right_camera_matrix(3, 3, CV_64F);
    cv::Mat rotation(3, 3, CV_64F);
    cv::Mat translation(3, 1, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const std::size_t index = static_cast<std::size_t>(row * 3 + col);
            left_camera_matrix.at<double>(row, col) = result.left_camera_matrix.at(index);
            right_camera_matrix.at<double>(row, col) = result.right_camera_matrix.at(index);
            rotation.at<double>(row, col) = result.right_to_left_rotation.at(index);
        }
        translation.at<double>(row) = result.right_to_left_translation.at(static_cast<std::size_t>(row));
    }
    const cv::Mat left_distortion(result.left_distortion, true);
    const cv::Mat right_distortion(result.right_distortion, true);
    storage << "left_camera_matrix" << left_camera_matrix;
    storage << "right_camera_matrix" << right_camera_matrix;
    storage << "left_distortion" << left_distortion;
    storage << "right_distortion" << right_distortion;
    storage << "right_to_left_rotation" << rotation;
    storage << "right_to_left_translation" << translation;
}

bool loadStereoCharucoCalibration(const std::filesystem::path &path,
                                  CharucoBoardConfig &config,
                                  StereoCharucoCalibrationResult &result) {
    if (!std::filesystem::exists(path))
        return false;
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened())
        throw std::runtime_error("Unable to read calibration JSON: " + path.string());

    const auto required = [&](const char *name) -> cv::FileNode {
        const cv::FileNode node = storage[name];
        if (node.empty())
            throw std::runtime_error("Calibration JSON is missing field: " + std::string(name));
        return node;
    };
    config.squares_x = static_cast<int>(required("squares_x"));
    config.squares_y = static_cast<int>(required("squares_y"));
    config.square_length_m = static_cast<float>(required("square_length_m").real());
    config.marker_length_m = static_cast<float>(required("marker_length_m").real());
    config.dictionary_name = static_cast<std::string>(required("dictionary_name"));
    validateConfig(config);
    result.left_rms = required("left_rms").real();
    result.right_rms = required("right_rms").real();
    result.stereo_rms = required("stereo_rms").real();
    result.single_camera_pair_count = static_cast<int>(required("single_camera_pair_count"));
    result.stereo_pair_count = static_cast<int>(required("stereo_pair_count"));

    cv::Mat left_camera_matrix, right_camera_matrix, left_distortion, right_distortion, rotation, translation;
    required("left_camera_matrix") >> left_camera_matrix;
    required("right_camera_matrix") >> right_camera_matrix;
    required("left_distortion") >> left_distortion;
    required("right_distortion") >> right_distortion;
    required("right_to_left_rotation") >> rotation;
    required("right_to_left_translation") >> translation;
    if (left_camera_matrix.rows != 3 || left_camera_matrix.cols != 3 ||
        right_camera_matrix.rows != 3 || right_camera_matrix.cols != 3 ||
        rotation.rows != 3 || rotation.cols != 3 || translation.total() != 3) {
        throw std::runtime_error("Calibration JSON has invalid matrix dimensions");
    }
    left_camera_matrix.convertTo(left_camera_matrix, CV_64F);
    right_camera_matrix.convertTo(right_camera_matrix, CV_64F);
    rotation.convertTo(rotation, CV_64F);
    translation.convertTo(translation, CV_64F);
    copyMatrix3x3(left_camera_matrix, result.left_camera_matrix);
    copyMatrix3x3(right_camera_matrix, result.right_camera_matrix);
    copyMatrix3x3(rotation, result.right_to_left_rotation);
    result.left_distortion = copyVector(left_distortion.reshape(1, 1));
    result.right_distortion = copyVector(right_distortion.reshape(1, 1));
    for (int index = 0; index < 3; ++index)
        result.right_to_left_translation.at(static_cast<std::size_t>(index)) = translation.at<double>(index);
    return true;
}

} // namespace ffs_viewer::calibration

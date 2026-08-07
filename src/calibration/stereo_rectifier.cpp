#include "ffs_viewer/calibration/stereo_rectifier.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ffs_viewer::calibration {
namespace {

cv::Mat matrix3x3(const std::array<double, 9> &values) {
    cv::Mat matrix(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            matrix.at<double>(row, col) = values.at(static_cast<std::size_t>(row * 3 + col));
    return matrix;
}

cv::Mat vectorMatrix(const std::vector<double> &values) {
    if (values.empty())
        throw std::invalid_argument("Calibration distortion vector is empty");
    return cv::Mat(values, true).reshape(1, 1);
}

void validateFrame(const io::BgrFrame &frame) {
    const std::size_t expected = static_cast<std::size_t>(frame.width) *
                                 static_cast<std::size_t>(frame.height) * 3U;
    if (!frame.valid() || frame.pixels.size() != expected)
        throw std::invalid_argument("Stereo rectification requires valid BGR8 frames");
}

void copyMatToFrame(const cv::Mat &image, const io::BgrFrame &source, io::BgrFrame &destination) {
    destination.width = image.cols;
    destination.height = image.rows;
    destination.frame_id = source.frame_id;
    destination.timestamp_ns = source.timestamp_ns;
    destination.pixels.resize(image.total() * image.elemSize());
    std::memcpy(destination.pixels.data(), image.data, destination.pixels.size());
}

} // namespace

struct StereoRectifier::Impl {
    bool has_calibration = false;
    cv::Mat left_camera_matrix;
    cv::Mat right_camera_matrix;
    cv::Mat left_distortion;
    cv::Mat right_distortion;
    cv::Mat rotation;
    cv::Mat translation;
    cv::Size map_size;
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;
    cv::Mat left_projection;
    cv::Mat right_projection;

    void buildMaps(const cv::Size &image_size) {
        cv::Mat left_rectification, right_rectification, disparity_to_depth;
        cv::stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion,
                          image_size, rotation, translation, left_rectification, right_rectification,
                          left_projection, right_projection, disparity_to_depth, cv::CALIB_ZERO_DISPARITY);
        cv::initUndistortRectifyMap(left_camera_matrix, left_distortion, left_rectification,
                                    left_projection, image_size, CV_32FC1, left_map_x, left_map_y);
        cv::initUndistortRectifyMap(right_camera_matrix, right_distortion, right_rectification,
                                    right_projection, image_size, CV_32FC1, right_map_x, right_map_y);
        map_size = image_size;
    }
};

StereoRectifier::StereoRectifier() : impl_(std::make_unique<Impl>()) {}
StereoRectifier::~StereoRectifier() = default;

void StereoRectifier::setCalibration(const StereoCharucoCalibrationResult &calibration) {
    impl_->left_camera_matrix = matrix3x3(calibration.left_camera_matrix);
    impl_->right_camera_matrix = matrix3x3(calibration.right_camera_matrix);
    impl_->left_distortion = vectorMatrix(calibration.left_distortion);
    impl_->right_distortion = vectorMatrix(calibration.right_distortion);
    impl_->rotation = matrix3x3(calibration.right_to_left_rotation);
    impl_->translation = cv::Mat(3, 1, CV_64F);
    for (int index = 0; index < 3; ++index)
        impl_->translation.at<double>(index) = calibration.right_to_left_translation.at(index);
    impl_->map_size = {};
    impl_->left_map_x.release();
    impl_->left_map_y.release();
    impl_->right_map_x.release();
    impl_->right_map_y.release();
    impl_->has_calibration = true;
}

bool StereoRectifier::hasCalibration() const noexcept {
    return impl_->has_calibration;
}

RectifiedStereoCamera StereoRectifier::rectifiedCamera(int width, int height) {
    if (!impl_->has_calibration)
        throw std::logic_error("Rectified camera requested without calibration");
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("Rectified camera dimensions must be positive");
    const cv::Size image_size(width, height);
    if (impl_->map_size != image_size)
        impl_->buildMaps(image_size);

    const double focal_x = impl_->left_projection.at<double>(0, 0);
    const double focal_y = impl_->left_projection.at<double>(1, 1);
    if (focal_x <= 0.0 || focal_y <= 0.0)
        throw std::runtime_error("Stereo rectification produced invalid focal length");
    const double baseline_m = std::abs(impl_->right_projection.at<double>(0, 3) / focal_x);
    if (baseline_m <= 0.0)
        throw std::runtime_error("Stereo rectification produced invalid baseline");

    return {width, height, focal_x, focal_y, impl_->left_projection.at<double>(0, 2),
            impl_->left_projection.at<double>(1, 2), baseline_m};
}

void StereoRectifier::rectify(const io::BgrFrame &left, const io::BgrFrame &right,
                               io::BgrFrame &rectified_left, io::BgrFrame &rectified_right) {
    if (!impl_->has_calibration)
        throw std::logic_error("Stereo rectification requested without calibration");
    validateFrame(left);
    validateFrame(right);
    if (left.width != right.width || left.height != right.height)
        throw std::invalid_argument("Stereo rectification requires equal left/right image dimensions");

    const cv::Size image_size(left.width, left.height);
    if (impl_->map_size != image_size)
        impl_->buildMaps(image_size);

    const cv::Mat left_image(left.height, left.width, CV_8UC3,
                             const_cast<std::uint8_t *>(left.pixels.data()));
    const cv::Mat right_image(right.height, right.width, CV_8UC3,
                              const_cast<std::uint8_t *>(right.pixels.data()));
    cv::Mat rectified_left_image, rectified_right_image;
    cv::remap(left_image, rectified_left_image, impl_->left_map_x, impl_->left_map_y, cv::INTER_LINEAR);
    cv::remap(right_image, rectified_right_image, impl_->right_map_x, impl_->right_map_y, cv::INTER_LINEAR);
    copyMatToFrame(rectified_left_image, left, rectified_left);
    copyMatToFrame(rectified_right_image, right, rectified_right);
}

} // namespace ffs_viewer::calibration

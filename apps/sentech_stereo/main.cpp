#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ffs_viewer/calibration/live_charuco_detector.hpp"
#include "ffs_viewer/calibration/stereo_charuco_calibrator.hpp"
#include "ffs_viewer/calibration/stereo_rectifier.hpp"
#include "ffs_viewer/inference/sentech_ffs_pipeline.hpp"
#include "ffs_viewer/inference/sentech_final_capture_pipeline.hpp"
#include "ffs_viewer/io/sentech_stereo_source.hpp"
#include "ffs_viewer/ui/sentech_point_cloud_viewer.hpp"
#include "opengl_point_cloud_viewer.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <spawn.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <sys/wait.h>
#include <vector>
#include <unistd.h>
extern char **environ;


namespace {


constexpr std::array<const char *, 16> kCharucoDictionaryNames{
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", "DICT_5X5_50",
    "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000", "DICT_6X6_50", "DICT_6X6_100",
    "DICT_6X6_250", "DICT_6X6_1000", "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250",
    "DICT_7X7_1000",
};

int charucoDictionaryIndex(const std::string &name) {
    for (std::size_t index = 0; index < kCharucoDictionaryNames.size(); ++index) {
        if (name == kCharucoDictionaryNames[index])
            return static_cast<int>(index);
    }
    return 0;
}
struct AnyLabelingMask {
    cv::Mat mask;
    int polygon_count = 0;
};

class AnyLabelingBridge final {
  public:
    AnyLabelingBridge(std::filesystem::path executable, std::filesystem::path work_directory,
                      std::string auto_labeling_model)
        : executable_(std::move(executable)), work_directory_(std::move(work_directory)),
          auto_labeling_model_(std::move(auto_labeling_model)),
          image_path_(work_directory_ / "final_left.png"),
          annotation_path_(work_directory_ / "final_left.json") {}

    bool running() const noexcept {
        return process_id_ > 0;
    }

    const std::string &status() const noexcept {
        return status_;
    }

    void launch(const ffs_viewer::io::BgrFrame &frame) {
        if (running())
            throw std::logic_error("AnyLabeling is already running");
        if (!frame.valid())
            throw std::invalid_argument("AnyLabeling requires a valid final-capture image");
        image_width_ = frame.width;
        image_height_ = frame.height;
        if (!std::filesystem::is_regular_file(executable_)) {
            throw std::runtime_error("AnyLabeling executable was not found: " + executable_.string());
        }

        std::filesystem::create_directories(work_directory_);
        std::error_code error;
        std::filesystem::remove(annotation_path_, error);
        if (error) {
            throw std::runtime_error("Unable to replace AnyLabeling annotation JSON: " +
                                     error.message());
        }

        const cv::Mat image(frame.height, frame.width, CV_8UC3,
                            const_cast<std::uint8_t *>(frame.pixels.data()));
        if (!cv::imwrite(image_path_.string(), image))
            throw std::runtime_error("Unable to write AnyLabeling input image: " + image_path_.string());

        std::vector<std::string> arguments{executable_.string(), image_path_.string(), "--output",
                                           annotation_path_.string(), "--autosave", "--nodata"};
        if (!auto_labeling_model_.empty()) {
            arguments.push_back("--auto-labeling-model");
            arguments.push_back(auto_labeling_model_);
        }
        std::vector<char *> argv;
        argv.reserve(arguments.size() + 1);
        for (std::string &argument : arguments)
            argv.push_back(argument.data());
        argv.push_back(nullptr);

        pid_t child_process = -1;
        const int spawn_error = posix_spawn(&child_process, executable_.c_str(), nullptr, nullptr,
                                            argv.data(), environ);
        if (spawn_error != 0) {
            throw std::runtime_error("Unable to launch AnyLabeling: " +
                                     std::string(std::strerror(spawn_error)));
        }
        process_id_ = child_process;
        status_ = "AnyLabeling is open. Save the annotation, then close AnyLabeling to import the mask.";
    }

    void discardSavedAnnotation() {
        std::error_code error;
        std::filesystem::remove(annotation_path_, error);
        if (error) {
            throw std::runtime_error("Unable to remove stale AnyLabeling annotation: " +
                                     error.message());
        }
    }

    std::optional<AnyLabelingMask> loadSavedMask() const {
        if (!std::filesystem::is_regular_file(annotation_path_))
            return std::nullopt;
        return loadMask();
    }

    std::optional<AnyLabelingMask> pollImport() {
        if (!running())
            return std::nullopt;

        int child_status = 0;
        const pid_t wait_result = waitpid(process_id_, &child_status, WNOHANG);
        if (wait_result == 0)
            return std::nullopt;
        if (wait_result < 0) {
            process_id_ = -1;
            throw std::runtime_error("Unable to monitor AnyLabeling: " +
                                     std::string(std::strerror(errno)));
        }
        process_id_ = -1;
        const bool successful_exit = WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
        // A window-manager close can report a non-zero status even after AnyLabeling
        // has autosaved the annotation. A valid JSON from this launch is authoritative.
        if (!successful_exit && !std::filesystem::is_regular_file(annotation_path_)) {
            status_ = "AnyLabeling closed without a saved annotation.";
            throw std::runtime_error(status_);
        }

        AnyLabelingMask imported_mask = loadMask();
        status_ = "Imported " + std::to_string(imported_mask.polygon_count) +
                  " AnyLabeling polygon(s) from " + annotation_path_.string();
        if (!successful_exit)
            status_ += " after the AnyLabeling window was closed.";
        return imported_mask;
    }

  private:
    AnyLabelingMask loadMask() const {
        std::ifstream input(annotation_path_);
        if (!input)
            throw std::runtime_error("AnyLabeling did not save an annotation JSON");
        const std::string source((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
        std::string normalized;
        normalized.reserve(source.size());
        bool in_string = false;
        bool escaped = false;
        for (std::size_t index = 0; index < source.size(); ++index) {
            const char character = source[index];
            if (in_string) {
                normalized.push_back(character);
                if (escaped) {
                    escaped = false;
                } else if (character == static_cast<char>(92)) {
                    escaped = true;
                } else if (character == static_cast<char>(34)) {
                    in_string = false;
                }
                continue;
            }
            if (character == static_cast<char>(34)) {
                in_string = true;
                normalized.push_back(character);
                continue;
            }
            const bool has_null = source.compare(index, 4, "null") == 0;
            const bool token_start = index == 0 ||
                !std::isalnum(static_cast<unsigned char>(source[index - 1]));
            const bool token_end = index + 4 == source.size() ||
                !std::isalnum(static_cast<unsigned char>(source[index + 4]));
            if (has_null && token_start && token_end) {
                normalized.push_back(static_cast<char>(48));
                index += 3;
            } else {
                normalized.push_back(character);
            }
        }

        cv::FileStorage storage(normalized, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                              cv::FileStorage::FORMAT_JSON);
        if (!storage.isOpened())
            throw std::runtime_error("AnyLabeling did not save an annotation JSON");

        const cv::FileNode shapes = storage["shapes"];
        if (shapes.empty() || !shapes.isSeq())
            throw std::runtime_error("AnyLabeling JSON has no shapes array");

        cv::Mat mask(image_height_, image_width_, CV_8UC1, cv::Scalar(0));
        int polygon_count = 0;
        for (auto shape_it = shapes.begin(); shape_it != shapes.end(); ++shape_it) {
            const cv::FileNode shape = *shape_it;
            const cv::FileNode points_node = shape["points"];
            if (points_node.empty() || !points_node.isSeq())
                continue;

            std::vector<cv::Point> polygon;
            for (auto point_it = points_node.begin(); point_it != points_node.end(); ++point_it) {
                const cv::FileNode point = *point_it;
                if (!point.isSeq() || point.size() != 2)
                    continue;
                polygon.emplace_back(cvRound(static_cast<double>(point[0])),
                                     cvRound(static_cast<double>(point[1])));
            }

            const cv::FileNode type_node = shape["shape_type"];
            const std::string shape_type =
                type_node.empty() ? "polygon" : static_cast<std::string>(type_node);
            if (shape_type == "rectangle" && polygon.size() == 2) {
                const cv::Point top_left(std::min(polygon[0].x, polygon[1].x),
                                         std::min(polygon[0].y, polygon[1].y));
                const cv::Point bottom_right(std::max(polygon[0].x, polygon[1].x),
                                             std::max(polygon[0].y, polygon[1].y));
                polygon = {top_left, {bottom_right.x, top_left.y}, bottom_right,
                           {top_left.x, bottom_right.y}};
            }
            if (shape_type != "polygon" && shape_type != "rectangle")
                continue;
            if (polygon.size() < 3)
                continue;

            cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255),
                         cv::LINE_8);
            ++polygon_count;
        }
        if (polygon_count == 0)
            throw std::runtime_error("AnyLabeling JSON contains no supported polygon or rectangle");
        return {std::move(mask), polygon_count};
    }

    std::filesystem::path executable_;
    std::filesystem::path work_directory_;
    std::string auto_labeling_model_;
    std::filesystem::path image_path_;
    std::filesystem::path annotation_path_;
    pid_t process_id_ = -1;
    int image_width_ = 0;
    int image_height_ = 0;
    std::string status_ = "AnyLabeling is idle";
};


class ImageTexture {
  public:
    ~ImageTexture() {
        if (texture_ != 0)
            glDeleteTextures(1, &texture_);
    }

    void upload(const ffs_viewer::io::BgrFrame &frame) {
        if (!frame.valid())
            return;
        if (texture_ == 0)
            glGenTextures(1, &texture_);

        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, frame.width, frame.height, 0, GL_BGR,
                     GL_UNSIGNED_BYTE, frame.pixels.data());
        width_ = frame.width;
        height_ = frame.height;
    }

    void upload(const cv::Mat &image) {
        if (image.empty() || image.type() != CV_8UC3)
            throw std::invalid_argument("Image texture upload requires a BGR8 image");
        const cv::Mat continuous = image.isContinuous() ? image : image.clone();
        if (texture_ == 0)
            glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, continuous.cols, continuous.rows, 0, GL_BGR,
                     GL_UNSIGNED_BYTE, continuous.data);
        width_ = continuous.cols;
        height_ = continuous.rows;
    }

    bool valid() const {
        return texture_ != 0;
    }

    ImTextureRef id() const {
        return ImTextureRef(ImTextureID(texture_));
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

  private:
    unsigned int texture_ = 0;
    int width_ = 0;
    int height_ = 0;
};

struct CalibrationCandidate {
    ffs_viewer::io::BgrFrame left;
    ffs_viewer::io::BgrFrame right;
    std::uint64_t timestamp_delta_ns = 0;
};

constexpr std::size_t kMaxCalibrationPairHistory = 20;

struct CalibrationPair {
    ffs_viewer::calibration::CharucoDetection left;
    ffs_viewer::calibration::CharucoDetection right;
    std::uint64_t left_timestamp_ns = 0;
    std::uint64_t right_timestamp_ns = 0;
    std::uint64_t timestamp_delta_ns = 0;
    ffs_viewer::calibration::CharucoBoardConfig board_config;
};

bool sameBoardConfig(const ffs_viewer::calibration::CharucoBoardConfig &a,
                     const ffs_viewer::calibration::CharucoBoardConfig &b) {
    return a.squares_x == b.squares_x && a.squares_y == b.squares_y &&
           a.square_length_m == b.square_length_m && a.marker_length_m == b.marker_length_m &&
           a.dictionary_name == b.dictionary_name;
}

std::uint64_t timestampDifference(std::uint64_t left, std::uint64_t right) {
    return left >= right ? left - right : right - left;
}

ffs_viewer::io::BgrFrame makeFoundationStereoInputLeft(
    const ffs_viewer::io::BgrFrame &rectified_left) {
    constexpr int kCropPixelsPerSide = 8;
    constexpr int kFsInputWidth = 608;
    constexpr int kFsInputHeight = 512;
    const std::size_t expected_size =
        static_cast<std::size_t>(rectified_left.width) * rectified_left.height * 3U;
    if (!rectified_left.valid() || rectified_left.pixels.size() != expected_size ||
        rectified_left.width <= 2 * kCropPixelsPerSide) {
        throw std::invalid_argument(
            "Saving AnnotateMe requires a valid rectified left stereo frame");
    }

    const cv::Mat rectified_image(rectified_left.height, rectified_left.width, CV_8UC3,
                                  const_cast<std::uint8_t *>(rectified_left.pixels.data()));
    cv::Mat resized;
    cv::resize(rectified_image(cv::Rect(kCropPixelsPerSide, 0,
                                        rectified_left.width - 2 * kCropPixelsPerSide,
                                        rectified_left.height)),
               resized, cv::Size(kFsInputWidth, kFsInputHeight), 0.0, 0.0, cv::INTER_AREA);

    ffs_viewer::io::BgrFrame result;
    result.width = resized.cols;
    result.height = resized.rows;
    result.frame_id = rectified_left.frame_id;
    result.timestamp_ns = rectified_left.timestamp_ns;
    result.pixels.assign(resized.datastart, resized.dataend);
    return result;
}

std::filesystem::path saveRawStereoPair(const ffs_viewer::io::BgrFrame &left,
                                         const ffs_viewer::io::BgrFrame &right,
                                         const ffs_viewer::io::BgrFrame &foundation_stereo_left) {
    if (!left.valid() || !right.valid() || !foundation_stereo_left.valid())
        throw std::invalid_argument(
            "Saving a raw stereo pair requires raw left/right and FoundationStereo input frames");

    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_time{};
    if (localtime_r(&now, &local_time) == nullptr)
        throw std::runtime_error("Unable to determine the local time for the raw-pair directory");

    std::ostringstream directory_name_stream;
    directory_name_stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    const std::string directory_name = directory_name_stream.str();
    const std::filesystem::path data_directory = FFS_WORKSPACE_DATA_DIRECTORY;
    std::filesystem::path output_directory = data_directory / directory_name;
    for (std::size_t suffix = 1; std::filesystem::exists(output_directory); ++suffix)
        output_directory = data_directory / (directory_name + "_" + std::to_string(suffix));
    std::filesystem::create_directories(output_directory);

    const cv::Mat left_image(left.height, left.width, CV_8UC3,
                             const_cast<std::uint8_t *>(left.pixels.data()));
    const cv::Mat right_image(right.height, right.width, CV_8UC3,
                              const_cast<std::uint8_t *>(right.pixels.data()));
    const cv::Mat annotate_me_image(foundation_stereo_left.height, foundation_stereo_left.width,
                                    CV_8UC3,
                                    const_cast<std::uint8_t *>(foundation_stereo_left.pixels.data()));
    if (!cv::imwrite((output_directory / "left.png").string(), left_image) ||
        !cv::imwrite((output_directory / "right.png").string(), right_image) ||
        !cv::imwrite((output_directory / "AnnotateMe.png").string(), annotate_me_image)) {
        throw std::runtime_error("Unable to save the raw stereo pair to " + output_directory.string());
    }
    return output_directory;
}

std::vector<std::filesystem::path> savedRawPairDirectories() {
    const std::filesystem::path data_directory = FFS_WORKSPACE_DATA_DIRECTORY;
    std::vector<std::filesystem::path> directories;
    std::error_code error;
    if (!std::filesystem::is_directory(data_directory, error))
        return directories;

    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(data_directory, error)) {
        if (error)
            throw std::runtime_error("Unable to browse the data directory: " + error.message());
        if (!entry.is_directory())
            continue;

        const std::filesystem::path left_path = entry.path() / "left.png";
        const std::filesystem::path right_path = entry.path() / "right.png";
        if (std::filesystem::is_regular_file(left_path) &&
            std::filesystem::is_regular_file(right_path)) {
            directories.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end(), std::greater<>());
    return directories;
}

ffs_viewer::io::BgrFrame loadRawStereoImage(const std::filesystem::path &image_path) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty())
        throw std::runtime_error("Unable to read image: " + image_path.string());
    if (image.type() != CV_8UC3)
        throw std::runtime_error("Loaded image is not BGR8: " + image_path.string());
    if (!image.isContinuous())
        image = image.clone();

    ffs_viewer::io::BgrFrame frame;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.pixels.assign(image.datastart, image.dataend);
    if (!frame.valid())
        throw std::runtime_error("Loaded image is empty: " + image_path.string());
    return frame;
}

std::pair<ffs_viewer::io::BgrFrame, ffs_viewer::io::BgrFrame>
loadRawStereoPair(const std::filesystem::path &directory) {
    const std::filesystem::path left_path = directory / "left.png";
    const std::filesystem::path right_path = directory / "right.png";
    if (!std::filesystem::is_regular_file(left_path) ||
        !std::filesystem::is_regular_file(right_path)) {
        throw std::runtime_error("Selected folder must contain left.png and right.png");
    }

    auto left = loadRawStereoImage(left_path);
    auto right = loadRawStereoImage(right_path);
    if (left.width != right.width || left.height != right.height) {
        throw std::runtime_error("Loaded left.png and right.png have different dimensions");
    }
    return {std::move(left), std::move(right)};
}

void drawLiveDisparity(const ImageTexture &disparity, const std::string &status) {
    ImGui::Begin("Live Disparity");
    ImGui::TextWrapped("%s", status.c_str());
    if (!disparity.valid()) {
        ImGui::TextUnformatted("Waiting for the first FFS disparity frame.");
        ImGui::End();
        return;
    }

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float scale = std::min(available.x / static_cast<float>(disparity.width()),
                                 available.y / static_cast<float>(disparity.height()));
    ImGui::Image(disparity.id(), ImVec2(static_cast<float>(disparity.width()) * scale,
                                        static_cast<float>(disparity.height()) * scale));
    ImGui::End();
}

void drawLivePointCloud(ffs_viewer::ui::SentechPointCloudViewer &viewer,
                        const std::string &status) {
    ImGui::Begin("Live Point Cloud");
    ImGui::TextWrapped("%s", status.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Reset View"))
        viewer.resetToLeftCameraView();

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 canvas_size(std::max(1.0F, available.x), std::max(1.0F, available.y - 22.0F));
    const ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##sentech-point-cloud-canvas", canvas_size);
    const ImGuiIO &io = ImGui::GetIO();
    viewer.interact(ImGui::IsItemHovered(), ImGui::IsMouseDragging(ImGuiMouseButton_Left),
                    ImGui::IsMouseDragging(ImGuiMouseButton_Right), io.MouseDelta.x, io.MouseDelta.y,
                    io.MouseWheel);
    viewer.draw(ImGui::GetWindowDrawList(), canvas_origin, canvas_size, io.DisplayFramebufferScale.x,
                io.DisplayFramebufferScale.y, io.DisplaySize.y * io.DisplayFramebufferScale.y);
    ImGui::Text("Points: %d | left-drag orbit, right-drag pan, wheel zoom", viewer.pointCount());
    ImGui::End();
}

void drawFinalPointCloud(ffs_viewer::ui::OpenGLPointCloudViewer &viewer) {
    ImGui::Begin("Live Point Cloud");
    ImGui::Text("FoundationStereo final: %d GPU vertices", viewer.pointCount());
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 canvas_size(std::max(1.0F, available.x), std::max(1.0F, available.y));
    const ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##sentech-final-cloud-canvas", canvas_size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    const ImGuiIO &io = ImGui::GetIO();
    viewer.interact(hovered, ImGui::IsMouseDragging(ImGuiMouseButton_Left),
                    ImGui::IsMouseDragging(ImGuiMouseButton_Right), io.MouseDelta.x, io.MouseDelta.y,
                    hovered ? io.MouseWheel : 0.0F);
    viewer.draw(ImGui::GetWindowDrawList(), canvas_origin, canvas_size, io.DisplayFramebufferScale.x,
                io.DisplayFramebufferScale.y, io.DisplaySize.y * io.DisplayFramebufferScale.y);
    ImGui::End();
}

void drawFinalMeshAreas(const ffs_viewer::ui::OpenGLPointCloudViewer &viewer) {
    constexpr float kMinimumComponentAreaCm2 = 10.F;
    const auto &components = viewer.meshComponentAreas();
    float retained_total_cm2 = 0.F;
    std::size_t retained_count = 0;
    for (const auto &component : components) {
        const float area_cm2 = component.area_m2 * 10000.F;
        if (area_cm2 > kMinimumComponentAreaCm2) {
            retained_total_cm2 += area_cm2;
            ++retained_count;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(520.F, 0.F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Mesh Areas");
    ImGui::Text("Retained total (> %.0f cm^2)", kMinimumComponentAreaCm2);
    ImGui::SetWindowFontScale(4.F);
    ImGui::Text("%.1f cm^2", retained_total_cm2);
    ImGui::SetWindowFontScale(1.F);
    ImGui::Text("Components above threshold: %zu", retained_count);
    ImGui::Separator();
    if (retained_count == 0U) {
        ImGui::TextUnformatted("Draw or import a mask with an area above 10 cm^2.");
    } else {
        std::size_t displayed_index = 0;
        for (const auto &component : components) {
            const float area_cm2 = component.area_m2 * 10000.F;
            if (area_cm2 <= kMinimumComponentAreaCm2)
                continue;
            ++displayed_index;
            ImGui::Text("Component %zu", displayed_index);
            ImGui::SetWindowFontScale(4.F);
            ImGui::Text("%.1f cm^2", area_cm2);
            ImGui::SetWindowFontScale(1.F);
            ImGui::Text("%d triangles", component.triangle_count);
        }
    }
    ImGui::End();
}

void drawStereoView(const ImageTexture &left, const ImageTexture &right) {
    ImGui::Begin("Stereo View");
    if (!left.valid() || !right.valid()) {
        ImGui::TextUnformatted("Press Start to visualize both cameras.");
        ImGui::End();
        return;
    }

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float total_width = static_cast<float>(left.width() + right.width());
    const float maximum_height = static_cast<float>(std::max(left.height(), right.height()));
    const float scale = std::min((available.x - spacing) / total_width, available.y / maximum_height);
    const ImVec2 left_size(static_cast<float>(left.width()) * scale,
                           static_cast<float>(left.height()) * scale);
    const ImVec2 right_size(static_cast<float>(right.width()) * scale,
                            static_cast<float>(right.height()) * scale);

    ImGui::Image(left.id(), left_size);
    ImGui::SameLine(0.F, spacing);
    ImGui::Image(right.id(), right_size);
    ImGui::End();
}

bool drawCalibrationPair(const ImageTexture &left, const ImageTexture &right,
                         const std::vector<CalibrationPair> &pairs,
                         std::size_t &selected_pair_index, bool *open,
                         const char *window_name = "Calibration Pair",
                         bool allow_delete = true) {
    ImGui::Begin(window_name, open);
    if (pairs.empty()) {
        ImGui::TextUnformatted("No calibration pair has been captured.");
        ImGui::End();
        return false;
    }

    bool changed_pair = false;
    if (ImGui::Button("Previous") && selected_pair_index > 0) {
        --selected_pair_index;
        changed_pair = true;
    }
    ImGui::SameLine();
    ImGui::Text("Pair %zu / %zu", selected_pair_index + 1, pairs.size());
    ImGui::SameLine();
    if (ImGui::Button("Next") && selected_pair_index + 1 < pairs.size()) {
        ++selected_pair_index;
        changed_pair = true;
    }
    if (changed_pair) {
        ImGui::End();
        return false;
    }
    ImGui::SameLine();
    const bool delete_this_pair = allow_delete && ImGui::Button("Delete This Pair");
    if (delete_this_pair) {
        ImGui::End();
        return true;
    }

    const CalibrationPair &pair = pairs.at(selected_pair_index);
    ImGui::Text("Left timestamp:  %llu ns", static_cast<unsigned long long>(pair.left_timestamp_ns));
    ImGui::Text("Right timestamp: %llu ns", static_cast<unsigned long long>(pair.right_timestamp_ns));
    ImGui::Text("Timestamp difference: %llu ns (%.3f ms)",
                static_cast<unsigned long long>(pair.timestamp_delta_ns),
                static_cast<double>(pair.timestamp_delta_ns) / 1'000'000.0);
    ImGui::Text("Left:  %d markers, %d ChArUco corners", pair.left.marker_count,
                pair.left.corner_count);
    ImGui::Text("Right: %d markers, %d ChArUco corners", pair.right.marker_count,
                pair.right.corner_count);
    const auto matched_corner_count = std::count_if(
        pair.left.corners.begin(), pair.left.corners.end(), [&](const auto &left_corner) {
            return std::any_of(pair.right.corners.begin(), pair.right.corners.end(),
                               [&](const auto &right_corner) { return right_corner.id == left_corner.id; });
        });
    ImGui::Text("Matched ChArUco corners: %zu", matched_corner_count);
    ImGui::Separator();

    if (!left.valid() || !right.valid()) {
        ImGui::TextUnformatted("Selected pair is not available.");
        ImGui::End();
        return false;
    }

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float total_width = static_cast<float>(left.width() + right.width());
    const float maximum_height = static_cast<float>(std::max(left.height(), right.height()));
    const float scale = std::min((available.x - spacing) / total_width, available.y / maximum_height);
    const ImVec2 left_size(static_cast<float>(left.width()) * scale,
                           static_cast<float>(left.height()) * scale);
    const ImVec2 right_size(static_cast<float>(right.width()) * scale,
                            static_cast<float>(right.height()) * scale);
    ImGui::Image(left.id(), left_size);
    const ImVec2 left_image_origin = ImGui::GetItemRectMin();
    ImGui::SameLine(0.F, spacing);
    ImGui::Image(right.id(), right_size);
    const ImVec2 right_image_origin = ImGui::GetItemRectMin();

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    for (const auto &left_corner : pair.left.corners) {
        const auto right_it = std::find_if(
            pair.right.corners.begin(), pair.right.corners.end(), [&](const auto &right_corner) {
                return right_corner.id == left_corner.id;
            });
        if (right_it == pair.right.corners.end())
            continue;

        const ImVec2 left_point(left_image_origin.x + left_corner.x * scale,
                                left_image_origin.y + left_corner.y * scale);
        const ImVec2 right_point(right_image_origin.x + right_it->x * scale,
                                 right_image_origin.y + right_it->y * scale);
        draw_list->AddLine(left_point, right_point, IM_COL32(255, 220, 0, 210), 1.5F);
        draw_list->AddCircleFilled(left_point, 3.0F, IM_COL32(255, 220, 0, 255));
        draw_list->AddCircleFilled(right_point, 3.0F, IM_COL32(255, 220, 0, 255));
    }
    ImGui::End();
    return false;
}

} // namespace

int main() {
    try {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
        GLFWwindow *window = glfwCreateWindow(1600, 900, "Sentech Stereo Viewer", nullptr, nullptr);
        if (window == nullptr) {
            glfwTerminate();
            throw std::runtime_error("OpenGL window creation failed");
        }

        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("GLEW initialization failed");
        }
        glGetError();

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        {
            ffs_viewer::io::SentechStereoSource capture;
            ffs_viewer::calibration::LiveCharucoDetector charuco_detector;
            const std::filesystem::path charuco_config_path = "sentech_stereo_charuco.json";
            std::string charuco_config_status = "Using default board parameters";
            try {
                if (charuco_detector.loadBoardConfig(charuco_config_path))
                    charuco_config_status = "Loaded " + charuco_config_path.string();
            } catch (const std::exception &error) {
                charuco_config_status = "Config load error: " + std::string(error.what());
            }
            ffs_viewer::calibration::CharucoDetection left_charuco;
            ffs_viewer::calibration::CharucoDetection right_charuco;
            ffs_viewer::calibration::CharucoBoardConfig charuco_config =
                charuco_detector.boardConfig();
            int charuco_dictionary_index = charucoDictionaryIndex(charuco_config.dictionary_name);
            bool live_charuco_detection = false;
            bool collecting_calibration_pair = false;
            bool checking_calibration = false;
            bool show_calibration_pair = false;
            bool show_calibration_panel = false;
            std::uint64_t last_calibration_left_frame_id = 0;
            std::uint64_t last_calibration_right_frame_id = 0;
            std::vector<CalibrationCandidate> calibration_candidates;
            std::vector<CalibrationPair> calibration_pair_history;
            std::size_t selected_calibration_pair_index = 0;
            std::optional<std::size_t> uploaded_calibration_pair_index;
            std::vector<CalibrationPair> calibration_check_pair_visualization;
            std::size_t selected_calibration_check_pair_index = 0;
            bool show_calibration_check_pair = false;
            bool calibration_check_pair_texture_dirty = false;
            std::optional<ffs_viewer::calibration::StereoCharucoCalibrationResult> calibration_result;
            std::optional<ffs_viewer::calibration::CharucoBoardConfig> calibration_board_config;
            std::optional<ffs_viewer::calibration::StereoCharucoCalibrationCheckResult>
                calibration_check_result;
            ffs_viewer::calibration::StereoRectifier stereo_rectifier;
            ffs_viewer::inference::SentechFfsPipeline ffs_pipeline(FFS_SENTECH_FFS_ENGINE_DIR);
            ffs_viewer::inference::SentechFinalCapturePipeline final_capture_pipeline(
                FFS_SENTECH_FS_ENGINE_DIR);
            bool show_rectified_images = false;
            ffs_viewer::io::BgrFrame rectified_left_frame;
            ffs_viewer::io::BgrFrame rectified_right_frame;
            ffs_viewer::io::BgrFrame captured_raw_left_frame;
            ffs_viewer::io::BgrFrame captured_raw_right_frame;
            ffs_viewer::io::BgrFrame captured_foundation_stereo_left_frame;
            bool raw_pair_ready_to_save = false;
            bool show_load_pair_dialog = false;
            std::vector<std::filesystem::path> saved_raw_pair_directories;
            std::string load_pair_status;
            std::optional<ffs_viewer::calibration::RectifiedStereoCamera> active_rectified_camera;
            const std::filesystem::path calibration_result_path = "sentech_stereo_calibration.json";
            std::string calibration_capture_status = "No calibration pair captured";
            std::string image_display_status = "Showing raw camera images";
            try {
                ffs_viewer::calibration::CharucoBoardConfig loaded_board_config;
                ffs_viewer::calibration::StereoCharucoCalibrationResult loaded_result;
                if (ffs_viewer::calibration::loadStereoCharucoCalibration(
                        calibration_result_path, loaded_board_config, loaded_result)) {
                    charuco_detector.setBoardConfig(loaded_board_config);
                    charuco_config = loaded_board_config;
                    charuco_dictionary_index = charucoDictionaryIndex(charuco_config.dictionary_name);
                    calibration_result = std::move(loaded_result);
                    calibration_board_config = loaded_board_config;
                    stereo_rectifier.setCalibration(*calibration_result);
                    calibration_capture_status = "Loaded calibration from " + calibration_result_path.string();
                }
            } catch (const std::exception &error) {
                calibration_capture_status = "Calibration JSON load error: " + std::string(error.what());
            }
            ImageTexture left_texture;
            ImageTexture right_texture;
            ImageTexture calibration_left_texture;
            ImageTexture calibration_right_texture;
            ImageTexture calibration_check_left_texture;
            ImageTexture calibration_check_right_texture;
            ImageTexture disparity_texture;
            ffs_viewer::ui::SentechPointCloudViewer point_cloud_viewer;
            ImageTexture final_left_texture;
            ImageTexture final_right_texture;
            ImageTexture final_mask_texture;
            ffs_viewer::ui::OpenGLPointCloudViewer final_point_cloud_viewer;
            final_point_cloud_viewer.setMaxDepth(1.0F);
            std::shared_ptr<const ffs_viewer::inference::SentechFinalCaptureResult> displayed_final_capture;
            bool show_final_mask_editor = false;
            bool show_final_capture_view = false;
            cv::Mat final_mask_source;
            cv::Mat final_mask;
            std::optional<cv::Mat> pending_anylabeling_mask;
            int final_mask_brush_radius = 4;
            bool final_mask_stroke_active = false;
            float final_mesh_depth_threshold_cm = 1.0F;
            int final_mesh_downsample = 4;
            cv::Point previous_final_mask_pixel;
            std::uint64_t uploaded_disparity_left_frame_id = 0;
            std::uint64_t uploaded_disparity_right_frame_id = 0;
            AnyLabelingBridge anylabeling_bridge(FFS_ANYLABELING_EXECUTABLE,
                                                  FFS_ANYLABELING_WORK_DIRECTORY,
                                                  FFS_ANYLABELING_AUTO_LABELING_MODEL);

            auto startFinalCapture = [&](const ffs_viewer::io::BgrFrame &raw_left,
                                         const ffs_viewer::io::BgrFrame &raw_right,
                                         const ffs_viewer::io::BgrFrame &rectified_left,
                                         const ffs_viewer::io::BgrFrame &rectified_right,
                                         const ffs_viewer::calibration::RectifiedStereoCamera &rectified_camera,
                                         const std::string &source) {
                if (show_final_mask_editor) {
                    throw std::logic_error(
                        "Close the Draw window before taking another final capture");
                }
                if (!raw_left.valid() || !raw_right.valid() || !rectified_left.valid() ||
                    !rectified_right.valid()) {
                    throw std::invalid_argument("Final capture requires a valid stereo pair");
                }

                if (anylabeling_bridge.running()) {
                    throw std::logic_error(
                        "Close AnyLabeling before taking another final capture");
                }
                anylabeling_bridge.discardSavedAnnotation();
                pending_anylabeling_mask.reset();
                captured_raw_left_frame = raw_left;
                captured_raw_right_frame = raw_right;
                captured_foundation_stereo_left_frame =
                    makeFoundationStereoInputLeft(rectified_left);
                raw_pair_ready_to_save = true;
                rectified_left_frame = rectified_left;
                rectified_right_frame = rectified_right;
                active_rectified_camera = rectified_camera;
                show_rectified_images = true;

                // The final FS run owns the GPU. Its input pair is copied by capture(),
                // so live inference and acquisition can stop before it begins.
                ffs_pipeline.stop();
                capture.stop();
                final_capture_pipeline.capture(rectified_left_frame, rectified_right_frame,
                                               *active_rectified_camera);
                image_display_status = "FoundationStereo final capture started from " + source +
                                       "; live FFS and cameras stopped";
            };

            auto startFinalCaptureFromRawPair = [&](const ffs_viewer::io::BgrFrame &raw_left,
                                                    const ffs_viewer::io::BgrFrame &raw_right,
                                                    const std::string &source) {
                if (!stereo_rectifier.hasCalibration()) {
                    throw std::logic_error(
                        "Load or complete stereo calibration before loading a raw pair");
                }

                ffs_viewer::io::BgrFrame rectified_left;
                ffs_viewer::io::BgrFrame rectified_right;
                stereo_rectifier.rectify(raw_left, raw_right, rectified_left, rectified_right);
                const auto rectified_camera =
                    stereo_rectifier.rectifiedCamera(rectified_left.width, rectified_left.height);
                startFinalCapture(raw_left, raw_right, rectified_left, rectified_right,
                                  rectified_camera, source);
            };

            auto refreshFinalMask = [&](bool rebuild_mesh = true) {
                if (!displayed_final_capture || final_mask_source.empty() || final_mask.empty())
                    return;
                cv::Mat overlay = final_mask_source.clone();
                cv::Mat green(final_mask_source.size(), final_mask_source.type(), cv::Scalar(0, 255, 0));
                green.copyTo(overlay, final_mask);
                cv::Mat blended;
                cv::addWeighted(final_mask_source, 0.8, overlay, 0.2, 0.0, blended);
                final_mask_texture.upload(blended);

                if (rebuild_mesh) {
                    auto cloud = displayed_final_capture->cloud;
                    // Preserve every mask-highlighted point; only the mesh grid is reduced.
                    cloud.display_step = 1;
                    cloud.mesh_step = final_mesh_downsample;
                    cloud.mesh_depth_threshold_m = 0.01F * final_mesh_depth_threshold_cm;
                    final_point_cloud_viewer.updateCudaFinal(cloud, final_mask.data, final_mask.cols,
                                                             final_mask.rows, true);
                }
            };

            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                try {
                    if (auto imported_mask = anylabeling_bridge.pollImport()) {
                        if (!displayed_final_capture) {
                            throw std::logic_error(
                                "AnyLabeling mask returned without an active final capture");
                        }
                        const auto &frozen_left = displayed_final_capture->left;
                        if (imported_mask->mask.cols != frozen_left.width ||
                            imported_mask->mask.rows != frozen_left.height) {
                            throw std::runtime_error(
                                "AnyLabeling mask dimensions differ from the final-capture image");
                        }
                        pending_anylabeling_mask = std::move(imported_mask->mask);
                        show_final_mask_editor = false;
                        image_display_status = "Imported AnyLabeling mask with " +
                                               std::to_string(imported_mask->polygon_count) +
                                               " polygon(s); press Draw Manually to refine it";
                    }
                } catch (const std::exception &error) {
                    image_display_status =
                        "AnyLabeling import failed: " + std::string(error.what());
                }

                try {
                    capture.poll();
                    if (collecting_calibration_pair && capture.running()) {
                        const auto &left_frame = capture.leftFrame();
                        const auto &right_frame = capture.rightFrame();
                        if (left_frame.valid() && right_frame.valid() &&
                            left_frame.frame_id != last_calibration_left_frame_id &&
                            right_frame.frame_id != last_calibration_right_frame_id) {
                            last_calibration_left_frame_id = left_frame.frame_id;
                            last_calibration_right_frame_id = right_frame.frame_id;
                            calibration_candidates.push_back(
                                {left_frame, right_frame,
                                 timestampDifference(left_frame.timestamp_ns, right_frame.timestamp_ns)});
                            calibration_capture_status = "Collecting candidate " +
                                                         std::to_string(calibration_candidates.size()) + " / 5";

                            if (calibration_candidates.size() == 5) {
                                const auto best = std::min_element(
                                    calibration_candidates.begin(), calibration_candidates.end(),
                                    [](const CalibrationCandidate &a, const CalibrationCandidate &b) {
                                        return a.timestamp_delta_ns < b.timestamp_delta_ns;
                                    });
                                CalibrationPair pair;
                                pair.left_timestamp_ns = best->left.timestamp_ns;
                                pair.right_timestamp_ns = best->right.timestamp_ns;
                                pair.timestamp_delta_ns = best->timestamp_delta_ns;
                                pair.board_config = charuco_detector.boardConfig();
                                charuco_detector.detect(best->left, pair.left);
                                charuco_detector.detect(best->right, pair.right);
                                calibration_candidates.clear();
                                collecting_calibration_pair = false;
                                if (checking_calibration) {
                                    calibration_check_pair_visualization.clear();
                                    calibration_check_pair_visualization.push_back(std::move(pair));
                                    selected_calibration_check_pair_index = 0;
                                    calibration_check_pair_texture_dirty = true;
                                    show_calibration_check_pair = true;
                                    const CalibrationPair &captured_pair =
                                        calibration_check_pair_visualization.back();
                                    try {
                                        if (!calibration_result.has_value() ||
                                            !calibration_board_config.has_value()) {
                                            throw std::logic_error(
                                                "No saved or completed stereo calibration is available");
                                        }
                                        if (!sameBoardConfig(captured_pair.board_config,
                                                             *calibration_board_config)) {
                                            throw std::logic_error(
                                                "The active ChArUco board differs from the calibration board");
                                        }
                                        calibration_check_result =
                                            ffs_viewer::calibration::checkStereoCharucoCalibration(
                                                captured_pair.board_config, *calibration_result,
                                                captured_pair.left, captured_pair.right);
                                        calibration_capture_status =
                                            "Calibration check complete: stereo reprojection RMS " +
                                            std::to_string(
                                                calibration_check_result->stereo_reprojection_rms) +
                                            " px";
                                    } catch (const std::exception &error) {
                                        calibration_check_result.reset();
                                        calibration_capture_status =
                                            "Calibration check failed: " + std::string(error.what());
                                    }
                                    checking_calibration = false;
                                } else {
                                    if (calibration_pair_history.size() == kMaxCalibrationPairHistory)
                                        calibration_pair_history.erase(calibration_pair_history.begin());
                                    calibration_pair_history.push_back(std::move(pair));
                                    selected_calibration_pair_index = calibration_pair_history.size() - 1;
                                    uploaded_calibration_pair_index.reset();
                                    show_calibration_pair = true;
                                    calibration_capture_status =
                                        "Selected the smallest timestamp difference from 5 pairs; saved in history";
                                }
                            }
                        }
                    }
                    const ffs_viewer::io::BgrFrame *display_left = &capture.leftFrame();
                    const ffs_viewer::io::BgrFrame *display_right = &capture.rightFrame();
                    if (show_rectified_images) {
                        try {
                            stereo_rectifier.rectify(*display_left, *display_right, rectified_left_frame,
                                                     rectified_right_frame);
                            display_left = &rectified_left_frame;
                            display_right = &rectified_right_frame;
                            const auto rectified_camera =
                                stereo_rectifier.rectifiedCamera(display_left->width, display_left->height);
                            active_rectified_camera = rectified_camera;
                            if (ffs_pipeline.running()) {
                                ffs_pipeline.setCameraModel({
                                    rectified_camera.width,
                                    rectified_camera.height,
                                    static_cast<float>(rectified_camera.fx),
                                    static_cast<float>(rectified_camera.fy),
                                    static_cast<float>(rectified_camera.cx),
                                    static_cast<float>(rectified_camera.cy),
                                    static_cast<float>(rectified_camera.baseline_m),
                                });
                                ffs_pipeline.submit(*display_left, *display_right);
                            }
                        } catch (const std::exception &error) {
                            show_rectified_images = false;
                            ffs_pipeline.stop();
                            image_display_status = "Rectification error; showing raw images: " +
                                                   std::string(error.what());
                        }
                    }
                    if (live_charuco_detection) {
                        charuco_detector.detect(*display_left, left_charuco);
                        charuco_detector.detect(*display_right, right_charuco);
                        left_texture.upload(left_charuco.annotated_frame);
                        right_texture.upload(right_charuco.annotated_frame);
                    } else {
                        left_texture.upload(*display_left);
                        right_texture.upload(*display_right);
                    }
                } catch (const std::exception &error) {
                    capture.stop();
                    ffs_pipeline.stop();
                    std::cerr << "Streaming error: " << error.what() << '\n';
                }

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                if (!capture.running())
                    show_calibration_panel = false;
                // Keep acquisition controls and calibration controls in separate blocks.
                ImGui::Begin("Control Panel");
                if (ImGui::Button(capture.running() ? "Stop" : "Start")) {
                    if (capture.running()) {
                        capture.stop();
                        ffs_pipeline.stop();
                    } else {
                        try {
                            capture.start();
                            show_final_capture_view = false;
                            show_final_mask_editor = false;
                            image_display_status = "Camera acquisition restarted";
                        } catch (const std::exception &error) {
                            std::cerr << "Start failed: " << error.what() << '\n';
                        }
                    }
                }
                ImGui::SameLine();
                ImGui::BeginDisabled(!capture.running());
                if (ImGui::Button(show_calibration_panel ? "End Calibration"
                                                         : "Start Calibration")) {
                    show_calibration_panel = !show_calibration_panel;
                    if (!show_calibration_panel)
                        live_charuco_detection = false;
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Quit"))
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                ImGui::TextUnformatted(capture.status().c_str());
                ImGui::BeginDisabled(!capture.running());
                if (ImGui::Button(show_rectified_images ? "Show Raw Images" : "Show Rectified Images")) {
                    if (show_rectified_images) {
                        show_rectified_images = false;
                        ffs_pipeline.stop();
                        image_display_status = "Showing raw camera images";
                    } else if (!stereo_rectifier.hasCalibration()) {
                        image_display_status =
                            "Rectified images require a saved or completed stereo calibration";
                    } else {
                        show_rectified_images = true;
                        image_display_status =
                            "Showing stereo-rectified images; FFS is disabled";
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(!capture.running() || !show_rectified_images);
                if (ImGui::Button(ffs_pipeline.running() ? "Disable FFS" : "Enable FFS")) {
                    if (ffs_pipeline.running()) {
                        ffs_pipeline.stop();
                        image_display_status = "Live FFS disabled; showing rectified images";
                    } else {
                        try {
                            ffs_pipeline.start();
                            image_display_status = "Live FFS enabled for rectified images";
                        } catch (const std::exception &error) {
                            image_display_status = "Cannot start Sentech FFS: " +
                                                   std::string(error.what());
                        }
                    }
                }
                ImGui::EndDisabled();
                ImGui::TextWrapped("%s", image_display_status.c_str());
                // A final capture always starts from the frozen raw pair. This keeps
                // raw and rectified preview modes equivalent: the raw pair is
                // rectified here, then the final-capture pipeline applies its shared
                // crop and 608 x 512 preprocessing before FoundationStereo runs.
                ImGui::BeginDisabled(!capture.running() || !stereo_rectifier.hasCalibration() ||
                                     final_capture_pipeline.running());
                if (ImGui::Button("Capture")) {
                    if (show_final_mask_editor) {
                        image_display_status =
                            "Close the Draw window before taking another final capture";
                    } else {
                        try {
                            const auto raw_left = capture.leftFrame();
                            const auto raw_right = capture.rightFrame();
                            startFinalCaptureFromRawPair(raw_left, raw_right,
                                                         "the live camera pair");
                        } catch (const std::exception &error) {
                            image_display_status = "Final capture error: " +
                                                   std::string(error.what());
                        }
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(!raw_pair_ready_to_save ||
                                     !captured_foundation_stereo_left_frame.valid());
                if (ImGui::Button("Save Pair")) {
                    try {
                        const std::filesystem::path output_directory =
                            saveRawStereoPair(captured_raw_left_frame, captured_raw_right_frame,
                                              captured_foundation_stereo_left_frame);
                        if (final_point_cloud_viewer.hasVcgMesh()) {
                            try {
                                final_point_cloud_viewer.saveVcgMeshObj(output_directory / "mesh.obj");
                                image_display_status =
                                    "Saved pair, AnnotateMe.png, and mesh.obj to " +
                                    output_directory.string();
                            } catch (const std::exception &error) {
                                image_display_status =
                                    "Saved pair and AnnotateMe.png to " + output_directory.string() +
                                    "; mesh.obj export failed: " + error.what();
                            }
                        } else {
                            image_display_status =
                                "Saved pair and AnnotateMe.png to " + output_directory.string() +
                                "; no VCG mesh was generated";
                        }
                    } catch (const std::exception &error) {
                        image_display_status =
                            "Pair save failed: " + std::string(error.what());
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Load Pair")) {
                    try {
                        saved_raw_pair_directories = savedRawPairDirectories();
                        load_pair_status = "Select a folder from " +
                                           std::filesystem::path(FFS_WORKSPACE_DATA_DIRECTORY).string();
                        show_load_pair_dialog = true;
                    } catch (const std::exception &error) {
                        image_display_status = "Unable to list saved raw pairs: " +
                                               std::string(error.what());
                    }
                }
                if (show_load_pair_dialog) {
                    ImGui::OpenPopup("Load Pair");
                    show_load_pair_dialog = false;
                }
                if (ImGui::BeginPopupModal("Load Pair", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::TextUnformatted("Choose a saved raw stereo pair folder:");
                    ImGui::TextWrapped("%s",
                                       std::filesystem::path(FFS_WORKSPACE_DATA_DIRECTORY).string().c_str());
                    if (ImGui::Button("Refresh")) {
                        try {
                            saved_raw_pair_directories = savedRawPairDirectories();
                            load_pair_status = "Folder list refreshed";
                        } catch (const std::exception &error) {
                            load_pair_status = "Refresh failed: " + std::string(error.what());
                        }
                    }
                    ImGui::Separator();
                    if (saved_raw_pair_directories.empty()) {
                        ImGui::TextUnformatted("No folders with left.png and right.png were found.");
                    }
                    for (const std::filesystem::path &directory : saved_raw_pair_directories) {
                        const std::string name = directory.filename().string();
                        if (!ImGui::Selectable(name.c_str()))
                            continue;
                        try {
                            auto [left, right] = loadRawStereoPair(directory);
                            startFinalCaptureFromRawPair(left, right, "saved pair " + name);
                            load_pair_status = "Loaded " + directory.string();
                            ImGui::CloseCurrentPopup();
                        } catch (const std::exception &error) {
                            load_pair_status = "Load failed: " + std::string(error.what());
                        }
                    }
                    ImGui::Separator();
                    ImGui::TextWrapped("%s", load_pair_status.c_str());
                    if (ImGui::Button("Cancel"))
                        ImGui::CloseCurrentPopup();
                    ImGui::EndPopup();
                }
                ImGui::BeginDisabled(!show_final_capture_view || !displayed_final_capture ||
                                     displayed_final_capture->left.pixels.empty() || anylabeling_bridge.running());
                if (ImGui::Button("Draw Manually") && show_final_capture_view &&
                    displayed_final_capture && !displayed_final_capture->left.pixels.empty()) {
                    const auto &frozen_left = displayed_final_capture->left;
                    final_mask_source = cv::Mat(frozen_left.height, frozen_left.width, CV_8UC3,
                                                 const_cast<std::uint8_t *>(frozen_left.pixels.data()))
                                            .clone();
                    if (auto saved_mask = anylabeling_bridge.loadSavedMask()) {
                        if (saved_mask->mask.size() != final_mask_source.size()) {
                            throw std::runtime_error(
                                "Saved AnyLabeling mask dimensions differ from the final-capture image");
                        }
                        final_mask = std::move(saved_mask->mask);
                        pending_anylabeling_mask.reset();
                        image_display_status =
                            "Loaded the AnyLabeling mask for manual refinement";
                    } else if (pending_anylabeling_mask.has_value()) {
                        final_mask = std::move(*pending_anylabeling_mask);
                        pending_anylabeling_mask.reset();
                        image_display_status =
                            "Opened the imported AnyLabeling mask for manual refinement";
                    } else {
                        final_mask = cv::Mat::zeros(final_mask_source.size(), CV_8UC1);
                    }
                    final_mask_stroke_active = false;
                    show_final_mask_editor = true;
                    refreshFinalMask();
                }
                ImGui::SameLine();
                if (ImGui::Button("AnyLabeling Draw")) {
                    try {
                        anylabeling_bridge.launch(displayed_final_capture->left);
                        image_display_status = anylabeling_bridge.status();
                    } catch (const std::exception &error) {
                        image_display_status =
                            "Unable to start AnyLabeling: " + std::string(error.what());
                    }
                }
                ImGui::EndDisabled();
                ImGui::TextWrapped("%s", final_capture_pipeline.status().c_str());
                ImGui::TextWrapped("%s", anylabeling_bridge.status().c_str());
                ImGui::End();

                if (show_calibration_panel) {
                  ImGui::Begin("Calibration");
                  ImGui::InputInt("Squares X", &charuco_config.squares_x);
                  ImGui::InputInt("Squares Y", &charuco_config.squares_y);
                  ImGui::InputFloat("Square length (m)",
                                    &charuco_config.square_length_m, 0.001F,
                                    0.010F, "%.4f");
                  ImGui::InputFloat("Marker length (m)",
                                    &charuco_config.marker_length_m, 0.001F,
                                    0.010F, "%.4f");
                  ImGui::Combo(
                      "Dictionary", &charuco_dictionary_index,
                      kCharucoDictionaryNames.data(),
                      static_cast<int>(kCharucoDictionaryNames.size()));
                  if (ImGui::Button("Apply ChArUco Board")) {
                    charuco_config.dictionary_name = kCharucoDictionaryNames.at(
                        static_cast<std::size_t>(charuco_dictionary_index));
                    try {
                      charuco_detector.setBoardConfig(charuco_config);
                      charuco_detector.saveBoardConfig(charuco_config_path);
                      charuco_config_status = "Applied and saved to " +
                                              charuco_config_path.string();
                    } catch (const std::exception &error) {
                      charuco_config_status =
                          "Board parameter error: " + std::string(error.what());
                    }
                  }
                  ImGui::TextWrapped("%s", charuco_config_status.c_str());
                  ImGui::Separator();
                  if (ImGui::Button(live_charuco_detection
                                        ? "Stop Live ChArUco Detection"
                                        : "Live ChArUco Detection")) {
                    live_charuco_detection = !live_charuco_detection;
                  }
                  ImGui::TextUnformatted(live_charuco_detection
                                             ? "Live detection enabled"
                                             : "Live detection disabled");
                  if (live_charuco_detection) {
                    ImGui::Text("Left:  %d markers, %d ChArUco corners",
                                left_charuco.marker_count,
                                left_charuco.corner_count);
                    ImGui::Text("Right: %d markers, %d ChArUco corners",
                                right_charuco.marker_count,
                                right_charuco.corner_count);
                  }
                  ImGui::Separator();
                  if (ImGui::Button("Capture One Calibration Pair")) {
                    if (!capture.running()) {
                      calibration_capture_status =
                          "Start both cameras before capturing a calibration "
                          "pair";
                    } else {
                      checking_calibration = false;
                      collecting_calibration_pair = true;
                      calibration_candidates.clear();
                      last_calibration_left_frame_id = 0;
                      last_calibration_right_frame_id = 0;
                      calibration_capture_status =
                          "Collecting 5 timestamp candidates";
                    }
                  }
                  ImGui::SameLine();
                  if (ImGui::Button("Check Calibration")) {
                    if (!capture.running()) {
                      calibration_capture_status =
                          "Start both cameras before checking the calibration";
                    } else if (collecting_calibration_pair) {
                      calibration_capture_status =
                          "Wait for the current calibration-pair capture to "
                          "finish";
                    } else if (!calibration_result.has_value() ||
                               !calibration_board_config.has_value()) {
                      calibration_capture_status =
                          "Complete or load a stereo calibration before "
                          "checking it";
                    } else if (!sameBoardConfig(charuco_detector.boardConfig(),
                                                *calibration_board_config)) {
                      calibration_capture_status =
                          "Apply the ChArUco board used for the saved "
                          "calibration before checking it";
                    } else {
                      checking_calibration = true;
                      collecting_calibration_pair = true;
                      calibration_candidates.clear();
                      last_calibration_left_frame_id = 0;
                      last_calibration_right_frame_id = 0;
                      calibration_check_result.reset();
                      calibration_capture_status =
                          "Checking calibration: collecting 5 timestamp "
                          "candidates";
                    }
                  }
                  if (calibration_check_result.has_value()) {
                    constexpr double kMaxCalibrationCheckRmsPx = 1.0;
                    const bool calibration_check_passed =
                        calibration_check_result->stereo_reprojection_rms <=
                        kMaxCalibrationCheckRmsPx;
                    const ImVec4 check_color =
                        calibration_check_passed
                            ? ImVec4(0.20F, 0.90F, 0.30F, 1.0F)
                            : ImVec4(1.0F, 0.25F, 0.25F, 1.0F);
                    ImGui::TextColored(check_color, "%s", calibration_capture_status.c_str());
                  } else {
                    ImGui::TextWrapped("%s", calibration_capture_status.c_str());
                  }
                  if (ImGui::Button("Finish Calibration")) {
                    live_charuco_detection = false;
                    const auto active_board_config =
                        charuco_detector.boardConfig();
                    std::vector<ffs_viewer::calibration::CharucoDetection>
                        left_detections;
                    std::vector<ffs_viewer::calibration::CharucoDetection>
                        right_detections;
                    for (const CalibrationPair &pair :
                         calibration_pair_history) {
                      if (sameBoardConfig(pair.board_config,
                                          active_board_config)) {
                        left_detections.push_back(pair.left);
                        right_detections.push_back(pair.right);
                      }
                    }
                    try {
                      auto result =
                          ffs_viewer::calibration::calibrateStereoCharuco(
                              active_board_config, left_detections,
                              right_detections);
                      ffs_viewer::calibration::saveStereoCharucoCalibration(
                          calibration_result_path, active_board_config, result);
                      calibration_result = std::move(result);
                      calibration_board_config = active_board_config;
                      calibration_check_result.reset();
                      stereo_rectifier.setCalibration(*calibration_result);
                      calibration_capture_status =
                          "Calibration complete and saved: left RMS " +
                          std::to_string(calibration_result->left_rms) +
                          ", right RMS " +
                          std::to_string(calibration_result->right_rms) +
                          ", stereo RMS " +
                          std::to_string(calibration_result->stereo_rms);
                    } catch (const std::exception &error) {
                      calibration_capture_status =
                          "Calibration failed: " + std::string(error.what());
                    }
                  }
                  if (calibration_result.has_value()) {
                    ImGui::Text("Single-camera pairs: %d, stereo pairs: %d",
                                calibration_result->single_camera_pair_count,
                                calibration_result->stereo_pair_count);
                    constexpr double kMaxCalibrationRmsPx = 1.0;
                    const bool calibration_passed =
                        calibration_result->stereo_rms <= kMaxCalibrationRmsPx;
                    const ImVec4 calibration_color =
                        calibration_passed ? ImVec4(0.20F, 0.90F, 0.30F, 1.0F)
                                           : ImVec4(1.0F, 0.25F, 0.25F, 1.0F);
                    ImGui::Text("RMS: left %.4f, right %.4f",
                                calibration_result->left_rms,
                                calibration_result->right_rms);
                    ImGui::TextColored(calibration_color, "Stereo RMS: %.4f px",
                                       calibration_result->stereo_rms);
                    if (calibration_passed) {
                      ImGui::TextColored(
                          calibration_color,
                          "Calibration passed (threshold: 1.0 px).");
                    } else {
                      ImGui::TextColored(calibration_color,
                                         "Calibration error exceeds 1.0 px. "
                                         "Please recalibrate.");
                    }
                  }
                  if (calibration_check_result.has_value()) {
                    ImGui::Text("Check: %d matched ChArUco corners",
                                calibration_check_result->matched_corner_count);
                    ImGui::Text(
                        "Reprojection RMS (px): left %.4f, right %.4f",
                        calibration_check_result->left_reprojection_rms,
                        calibration_check_result->right_reprojection_rms);
                  }
                  ImGui::Separator();
                  ImGui::End();
                }

                const auto final_capture = final_capture_pipeline.latestResult();
                if (!show_final_mask_editor && final_capture && final_capture != displayed_final_capture) {
                    final_point_cloud_viewer.resetToLeftCameraView();
                    final_point_cloud_viewer.updateCudaFinal(final_capture->cloud);
                    final_left_texture.upload(final_capture->left);
                    final_right_texture.upload(final_capture->right);
                    disparity_texture.upload(final_capture->disparity_visualization);
                    displayed_final_capture = final_capture;
                    show_final_capture_view = true;
                }

                const bool showing_final_capture =
                    show_final_capture_view && displayed_final_capture != nullptr;
                drawStereoView(showing_final_capture ? final_left_texture : left_texture,
                               showing_final_capture ? final_right_texture : right_texture);
                if (show_rectified_images && ffs_pipeline.running() && !showing_final_capture) {
                    const auto disparity = ffs_pipeline.latestResult();
                    if (disparity &&
                        (disparity->left_frame_id != uploaded_disparity_left_frame_id ||
                         disparity->right_frame_id != uploaded_disparity_right_frame_id)) {
                        disparity_texture.upload(disparity->visualization);
                        if (disparity->gpu_cloud.valid())
                            point_cloud_viewer.updateCuda(disparity->gpu_cloud);
                        uploaded_disparity_left_frame_id = disparity->left_frame_id;
                        uploaded_disparity_right_frame_id = disparity->right_frame_id;
                    }
                    drawLivePointCloud(point_cloud_viewer, ffs_pipeline.status());
                }
                if ((show_rectified_images && ffs_pipeline.running()) || showing_final_capture) {
                    drawLiveDisparity(disparity_texture,
                                      showing_final_capture ? "FoundationStereo final disparity"
                                                            : ffs_pipeline.status());
                }
                if (showing_final_capture) {
                    drawFinalPointCloud(final_point_cloud_viewer);
                    drawFinalMeshAreas(final_point_cloud_viewer);
                }
                if (show_final_mask_editor) {
                    ImGui::Begin("Final Surface Draw", &show_final_mask_editor);
                    ImGui::TextUnformatted("Highlighted points use the original resolution; mesh uses its own grid.");
                    ImGui::SameLine();
                    if (ImGui::Button("Clear mask")) {
                        final_mask.setTo(0);
                        refreshFinalMask();
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(180.0F);
                    ImGui::SliderInt("Brush radius", &final_mask_brush_radius, 1, 20, "%d px");
                    ImGui::SetNextItemWidth(180.0F);
                    if (ImGui::SliderInt("Mesh downsample", &final_mesh_downsample, 1, 16,
                                         "%dx")) {
                        refreshFinalMask();
                    }
                    ImGui::TextDisabled("The mesh follows a simplified mask contour; its interior is sampled at this rate.");
                    if (ImGui::SliderFloat("Mesh depth threshold", &final_mesh_depth_threshold_cm, 0.1F,
                                           10.0F, "%.1f cm")) {
                        refreshFinalMask();
                    }
                    if (final_mask_texture.valid()) {
                        const ImVec2 available = ImGui::GetContentRegionAvail();
                        const float scale = std::min(available.x / static_cast<float>(final_mask_texture.width()),
                                                     available.y / static_cast<float>(final_mask_texture.height()));
                        const ImVec2 image_size(static_cast<float>(final_mask_texture.width()) * scale,
                                                 static_cast<float>(final_mask_texture.height()) * scale);
                        const ImVec2 image_pos = ImGui::GetCursorScreenPos();
                        ImGui::Image(final_mask_texture.id(), image_size);
                        ImGui::SetCursorScreenPos(image_pos);
                        ImGui::InvisibleButton("##sentech-final-mask-canvas", image_size,
                                               ImGuiButtonFlags_MouseButtonLeft |
                                                   ImGuiButtonFlags_MouseButtonRight);
                        const bool hovered = ImGui::IsItemHovered();
                        const bool paint = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left);
                        const bool erase = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Right);
                        if (paint || erase) {
                            const ImVec2 mouse = ImGui::GetMousePos();
                            const cv::Point pixel(
                                std::clamp(static_cast<int>((mouse.x - image_pos.x) / image_size.x *
                                                            final_mask.cols),
                                           0, final_mask.cols - 1),
                                std::clamp(static_cast<int>((mouse.y - image_pos.y) / image_size.y *
                                                            final_mask.rows),
                                           0, final_mask.rows - 1));
                            const cv::Scalar value = erase ? cv::Scalar(0) : cv::Scalar(255);
                            if (final_mask_stroke_active) {
                                cv::line(final_mask, previous_final_mask_pixel, pixel, value,
                                         2 * final_mask_brush_radius + 1, cv::LINE_8);
                            } else {
                                cv::circle(final_mask, pixel, final_mask_brush_radius, value, cv::FILLED,
                                           cv::LINE_8);
                            }
                            previous_final_mask_pixel = pixel;
                            final_mask_stroke_active = true;
                            // Keep the mask preview responsive while deferring CUDA/VCG work.
                            refreshFinalMask(false);
                        } else if (final_mask_stroke_active &&
                                   !ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                                   !ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                            final_mask_stroke_active = false;
                            refreshFinalMask();
                        }
                    }
                    ImGui::End();
                }
                if (show_calibration_pair && !calibration_pair_history.empty()) {
                    if (!uploaded_calibration_pair_index.has_value() ||
                        *uploaded_calibration_pair_index != selected_calibration_pair_index) {
                        const CalibrationPair &pair =
                            calibration_pair_history.at(selected_calibration_pair_index);
                        calibration_left_texture.upload(pair.left.annotated_frame);
                        calibration_right_texture.upload(pair.right.annotated_frame);
                        uploaded_calibration_pair_index = selected_calibration_pair_index;
                    }
                    const bool delete_selected_pair =
                        drawCalibrationPair(calibration_left_texture, calibration_right_texture,
                                            calibration_pair_history, selected_calibration_pair_index,
                                            &show_calibration_pair);
                    if (delete_selected_pair) {
                        calibration_pair_history.erase(
                            calibration_pair_history.begin() +
                            static_cast<std::ptrdiff_t>(selected_calibration_pair_index));
                        uploaded_calibration_pair_index.reset();
                        if (calibration_pair_history.empty()) {
                            show_calibration_pair = false;
                        } else if (selected_calibration_pair_index >= calibration_pair_history.size()) {
                            selected_calibration_pair_index = calibration_pair_history.size() - 1;
                        }
                    }
                }
                if (show_calibration_check_pair &&
                    !calibration_check_pair_visualization.empty()) {
                    if (calibration_check_pair_texture_dirty) {
                        const CalibrationPair &pair = calibration_check_pair_visualization.front();
                        calibration_check_left_texture.upload(pair.left.annotated_frame);
                        calibration_check_right_texture.upload(pair.right.annotated_frame);
                        calibration_check_pair_texture_dirty = false;
                    }
                    drawCalibrationPair(
                        calibration_check_left_texture, calibration_check_right_texture,
                        calibration_check_pair_visualization, selected_calibration_check_pair_index,
                        &show_calibration_check_pair, "Calibration Check Pair", false);
                }

                ImGui::Render();
                int framebuffer_width = 0;
                int framebuffer_height = 0;
                glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
                glViewport(0, 0, framebuffer_width, framebuffer_height);
                glClearColor(0.08F, 0.08F, 0.10F, 1.F);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                glfwSwapBuffers(window);
            }

            capture.stop();
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }

    return 1;
}

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ffs_viewer/calibration/live_charuco_detector.hpp"
#include "ffs_viewer/io/sentech_stereo_source.hpp"
#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
};

std::uint64_t timestampDifference(std::uint64_t left, std::uint64_t right) {
    return left >= right ? left - right : right - left;
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
                         std::size_t &selected_pair_index, bool *open) {
    ImGui::Begin("Calibration Pair", open);
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
    const bool delete_this_pair = ImGui::Button("Delete This Pair");
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
    ImGui::SameLine(0.F, spacing);
    ImGui::Image(right.id(), right_size);
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
            bool show_calibration_pair = false;
            std::uint64_t last_calibration_left_frame_id = 0;
            std::uint64_t last_calibration_right_frame_id = 0;
            std::vector<CalibrationCandidate> calibration_candidates;
            std::vector<CalibrationPair> calibration_pair_history;
            std::size_t selected_calibration_pair_index = 0;
            std::optional<std::size_t> uploaded_calibration_pair_index;
            std::string calibration_capture_status = "No calibration pair captured";
            ImageTexture left_texture;
            ImageTexture right_texture;
            ImageTexture calibration_left_texture;
            ImageTexture calibration_right_texture;

            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
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
                                charuco_detector.detect(best->left, pair.left);
                                charuco_detector.detect(best->right, pair.right);
                                if (calibration_pair_history.size() == kMaxCalibrationPairHistory)
                                    calibration_pair_history.erase(calibration_pair_history.begin());
                                calibration_pair_history.push_back(std::move(pair));
                                selected_calibration_pair_index = calibration_pair_history.size() - 1;
                                uploaded_calibration_pair_index.reset();
                                calibration_candidates.clear();
                                collecting_calibration_pair = false;
                                show_calibration_pair = true;
                                calibration_capture_status =
                                    "Selected the smallest timestamp difference from 5 pairs; saved in history";
                            }
                        }
                    }
                    if (live_charuco_detection) {
                        charuco_detector.detect(capture.leftFrame(), left_charuco);
                        charuco_detector.detect(capture.rightFrame(), right_charuco);
                        left_texture.upload(left_charuco.annotated_frame);
                        right_texture.upload(right_charuco.annotated_frame);
                    } else {
                        left_texture.upload(capture.leftFrame());
                        right_texture.upload(capture.rightFrame());
                    }
                } catch (const std::exception &error) {
                    capture.stop();
                    std::cerr << "Streaming error: " << error.what() << '\n';
                }

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Keep acquisition controls and calibration controls in separate blocks.
                ImGui::Begin("Acquisition");
                if (ImGui::Button("Start") && !capture.running()) {
                    try {
                        capture.start();
                    } catch (const std::exception &error) {
                        std::cerr << "Start failed: " << error.what() << '\n';
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Stop") && capture.running())
                    capture.stop();
                ImGui::SameLine();
                if (ImGui::Button("Quit"))
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                ImGui::TextUnformatted(capture.status().c_str());
                ImGui::End();

                ImGui::Begin("Calibration");
                ImGui::InputInt("Squares X", &charuco_config.squares_x);
                ImGui::InputInt("Squares Y", &charuco_config.squares_y);
                ImGui::InputFloat("Square length (m)", &charuco_config.square_length_m, 0.001F,
                                  0.010F, "%.4f");
                ImGui::InputFloat("Marker length (m)", &charuco_config.marker_length_m, 0.001F,
                                  0.010F, "%.4f");
                ImGui::Combo("Dictionary", &charuco_dictionary_index,
                             kCharucoDictionaryNames.data(),
                             static_cast<int>(kCharucoDictionaryNames.size()));
                if (ImGui::Button("Apply ChArUco Board")) {
                    charuco_config.dictionary_name =
                        kCharucoDictionaryNames.at(static_cast<std::size_t>(charuco_dictionary_index));
                    try {
                        charuco_detector.setBoardConfig(charuco_config);
                        charuco_detector.saveBoardConfig(charuco_config_path);
                        charuco_config_status = "Applied and saved to " + charuco_config_path.string();
                    } catch (const std::exception &error) {
                        charuco_config_status = "Board parameter error: " + std::string(error.what());
                    }
                }
                ImGui::TextWrapped("%s", charuco_config_status.c_str());
                ImGui::Separator();
                if (ImGui::Button(live_charuco_detection ? "Stop Live ChArUco Detection"
                                                         : "Live ChArUco Detection")) {
                    live_charuco_detection = !live_charuco_detection;
                }
                ImGui::TextUnformatted(live_charuco_detection ? "Live detection enabled"
                                                               : "Live detection disabled");
                if (live_charuco_detection) {
                    ImGui::Text("Left:  %d markers, %d ChArUco corners", left_charuco.marker_count,
                                left_charuco.corner_count);
                    ImGui::Text("Right: %d markers, %d ChArUco corners", right_charuco.marker_count,
                                right_charuco.corner_count);
                }
                ImGui::Separator();
                if (ImGui::Button("Capture One Calibration Pair")) {
                    if (!capture.running()) {
                        calibration_capture_status = "Start both cameras before capturing a calibration pair";
                    } else {
                        collecting_calibration_pair = true;
                        calibration_candidates.clear();
                        last_calibration_left_frame_id = 0;
                        last_calibration_right_frame_id = 0;
                        calibration_capture_status = "Collecting 5 timestamp candidates";
                    }
                }
                ImGui::TextWrapped("%s", calibration_capture_status.c_str());
                ImGui::Separator();
                ImGui::End();

                drawStereoView(left_texture, right_texture);
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

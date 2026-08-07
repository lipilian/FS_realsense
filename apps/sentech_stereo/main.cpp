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
#include <iostream>
#include <memory>
#include <mutex>
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
            ffs_viewer::calibration::CharucoDetection left_charuco;
            ffs_viewer::calibration::CharucoDetection right_charuco;
            ffs_viewer::calibration::CharucoBoardConfig charuco_config =
                charuco_detector.boardConfig();
            int charuco_dictionary_index = charucoDictionaryIndex(charuco_config.dictionary_name);
            std::string charuco_config_status = "Board parameters applied";
            bool live_charuco_detection = false;
            ImageTexture left_texture;
            ImageTexture right_texture;

            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                try {
                    capture.poll();
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
                        charuco_config_status = "Board parameters applied";
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
                ImGui::End();

                drawStereoView(left_texture, right_texture);

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

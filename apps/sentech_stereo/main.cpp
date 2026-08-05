#include <StApi_IP.h>
#include <StApi_TL.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kCameraCount = 2;
constexpr std::size_t kLeftCameraIndex = 0;
constexpr std::size_t kRightCameraIndex = 1;
constexpr std::array<const char *, kCameraCount> kCameraLabels{"Left camera", "Right camera"};
constexpr const char *kLeftCameraName = "STC-MCS500U3V(21LJ530)";
constexpr const char *kRightCameraName = "STC-MCS500U3V(21LJ548)";

enum class FrameRotation {
    Clockwise90,
    CounterClockwise90,
};

struct BgrFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    bool valid() const {
        return width > 0 && height > 0 && !pixels.empty();
    }
};

class StereoCapture {
  public:
    void start() {
        if (running_)
            return;

        try {
            system_.Reset(StApi::CreateIStSystem());
            for (std::size_t discovered = 0; discovered < kCameraCount; ++discovered) {
                StApi::CIStDevicePtr device(system_->CreateFirstIStDevice());
                const std::size_t index = cameraIndexForName(*device->GetIStDeviceInfo());
                if (devices_[index].IsValid())
                    throw std::runtime_error("Both discovered cameras match " +
                                             std::string(kCameraLabels[index]));
                devices_[index].Reset(device.Move());
            }

            for (std::size_t index = 0; index < kCameraCount; ++index) {
                streams_[index].Reset(devices_[index]->CreateIStDataStream(0));
                converters_[index].Reset(
                    StApi::CreateIStConverter(StApi::StConverterType_PixelFormat));
                converters_[index]->SetDestinationPixelFormat(StApi::StPFNC_BGR8);
                converted_images_[index].Reset(StApi::CreateIStImageBuffer());

                std::cout << kCameraLabels[index] << " connected: "
                          << devices_[index]->GetIStDeviceInfo()->GetDisplayName() << '\n';
                streams_[index]->StartAcquisition();
                devices_[index]->AcquisitionStart();
            }
            running_ = true;
            status_ = "Streaming two Sentech cameras";
        } catch (...) {
            stopNoThrow();
            throw;
        }
    }

    void stop() {
        stopNoThrow();
        status_ = "Stopped";
    }

    void poll() {
        if (!running_)
            return;

        for (std::size_t index = 0; index < kCameraCount; ++index) {
            StApi::IStStreamBufferReleasable *raw_buffer =
                streams_[index]->RetrieveBuffer(1, StApi::StTimeoutHandling_Return);
            if (raw_buffer == nullptr)
                continue;

            StApi::CIStStreamBufferPtr buffer(raw_buffer);
            if (!buffer->GetIStStreamBufferInfo()->IsImagePresent())
                continue;

            converters_[index]->Convert(buffer->GetIStImage(), converted_images_[index]);
            StApi::IStImage *image = converted_images_[index]->GetIStImage();
            // Normalize orientation at the capture boundary so every
            // downstream consumer receives the correctly oriented image.
            const FrameRotation rotation = index == kLeftCameraIndex
                                               ? FrameRotation::Clockwise90
                                               : FrameRotation::CounterClockwise90;
            copyBgrImage(*image, frames_[index], rotation);
        }
    }

    bool running() const {
        return running_;
    }

    const std::string &status() const {
        return status_;
    }

    const BgrFrame &leftFrame() const {
        return frames_.at(kLeftCameraIndex);
    }

    const BgrFrame &rightFrame() const {
        return frames_.at(kRightCameraIndex);
    }

    ~StereoCapture() {
        stopNoThrow();
    }

  private:
    static std::size_t cameraIndexForName(const StApi::IStDeviceInfo &info) {
        const std::string display_name(info.GetDisplayName().c_str());
        const std::string user_defined_name(info.GetUserDefinedName().c_str());
        const auto matches = [&](const char *configured_name) {
            return display_name == configured_name || user_defined_name == configured_name;
        };

        if (matches(kLeftCameraName))
            return kLeftCameraIndex;
        if (matches(kRightCameraName))
            return kRightCameraIndex;

        throw std::runtime_error("Unknown Sentech camera. DisplayName=\"" + display_name +
                                 "\", UserDefinedName=\"" + user_defined_name + "\"");
    }

    static void copyBgrImage(const StApi::IStImage &image, BgrFrame &frame,
                             FrameRotation rotation) {
        const int source_width = static_cast<int>(image.GetImageWidth());
        const int source_height = static_cast<int>(image.GetImageHeight());
        const int destination_width = source_height;
        const int destination_height = source_width;
        const std::size_t destination_row_bytes = static_cast<std::size_t>(destination_width) * 3U;

        frame.width = destination_width;
        frame.height = destination_height;
        frame.pixels.resize(destination_row_bytes * static_cast<std::size_t>(destination_height));

        // Wrap the SDK buffer directly, preserving its line pitch. One OpenCV
        // rotation replaces all per-pixel CPU loops.
        const cv::Mat source(source_height, source_width, CV_8UC3, image.GetImageBuffer(),
                             image.GetImageLinePitch());
        cv::Mat destination(destination_height, destination_width, CV_8UC3, frame.pixels.data(),
                            destination_row_bytes);
        const int operation = rotation == FrameRotation::Clockwise90
                                  ? cv::ROTATE_90_CLOCKWISE
                                  : cv::ROTATE_90_COUNTERCLOCKWISE;
        cv::rotate(source, destination, operation);
    }

    void stopNoThrow() noexcept {
        for (auto &device : devices_) {
            if (device.IsValid()) {
                try {
                    device->AcquisitionStop();
                } catch (...) {
                }
            }
        }
        for (auto &stream : streams_) {
            if (stream.IsValid()) {
                try {
                    stream->StopAcquisition();
                } catch (...) {
                }
            }
        }
        for (auto &stream : streams_)
            stream.Reset();
        for (auto &converter : converters_)
            converter.Reset();
        for (auto &image : converted_images_)
            image.Reset();
        for (auto &device : devices_)
            device.Reset();
        system_.Reset();
        running_ = false;
    }

    StApi::CIStSystemPtr system_;
    std::array<StApi::CIStDevicePtr, kCameraCount> devices_;
    std::array<StApi::CIStDataStreamPtr, kCameraCount> streams_;
    std::array<StApi::CIStPixelFormatConverterPtr, kCameraCount> converters_;
    std::array<StApi::CIStImageBufferPtr, kCameraCount> converted_images_;
    std::array<BgrFrame, kCameraCount> frames_;
    bool running_ = false;
    std::string status_ = "Stopped";
};

class ImageTexture {
  public:
    ~ImageTexture() {
        if (texture_ != 0)
            glDeleteTextures(1, &texture_);
    }

    void upload(const BgrFrame &frame) {
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

void configureGenTlPath() {
    if (std::getenv("GENICAM_GENTL64_PATH") == nullptr &&
        setenv("GENICAM_GENTL64_PATH", FFS_SENTECH_GENTL_DIRECTORY, 0) != 0) {
        throw std::runtime_error("Unable to configure GENICAM_GENTL64_PATH");
    }
}

} // namespace

int main() {
    try {
        configureGenTlPath();
        StApi::CStApiAutoInit stapi;

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
            StereoCapture capture;
            ImageTexture left_texture;
            ImageTexture right_texture;

            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                try {
                    capture.poll();
                    left_texture.upload(capture.leftFrame());
                    right_texture.upload(capture.rightFrame());
                } catch (const GenICam::GenericException &error) {
                    capture.stop();
                    std::cerr << "Streaming error: " << error.GetDescription() << '\n';
                }

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Keep acquisition controls in their own block. Calibration
                // controls can be added as a separate ImGui block later.
                ImGui::Begin("Acquisition");
                if (ImGui::Button("Start") && !capture.running()) {
                    try {
                        capture.start();
                    } catch (const GenICam::GenericException &error) {
                        std::cerr << "Start failed: " << error.GetDescription() << '\n';
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
    } catch (const GenICam::GenericException &error) {
        std::cerr << "Sentech StApi error: " << error.GetDescription() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }

    return 1;
}

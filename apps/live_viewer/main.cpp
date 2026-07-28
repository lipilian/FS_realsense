#include "live_pipeline.hpp"
#include "opengl_point_cloud_viewer.hpp"
#include <GL/glew.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string engine_dir = FFS_VIEWER_DEFAULT_ENGINE_DIR;
    int point_step = 4;
    float max_depth_m = 10.F;
};
Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--engine-dir <dir>] [--point-step <n>] [--max-depth-m <m>]\n";
            std::exit(0);
        }
        if (a == "--engine-dir" || a == "--point-step" || a == "--max-depth-m") {
            if (++i >= argc)
                throw std::runtime_error("Missing value for " + a);
            if (a == "--engine-dir")
                o.engine_dir = argv[i];
            else if (a == "--point-step")
                o.point_step = std::stoi(argv[i]);
            else
                o.max_depth_m = std::stof(argv[i]);
        } else
            throw std::runtime_error("Unknown option: " + a);
    }
    if (o.point_step <= 0 || o.max_depth_m <= 0)
        throw std::runtime_error("Invalid option");
    return o;
}

class ImageTexture {
  public:
    ~ImageTexture() {
        if (texture_ != 0)
            glDeleteTextures(1, &texture_);
    }

    void upload(const cv::Mat &image) {
        if (image.empty())
            return;
        if (texture_ == 0)
            glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE,
                     image.data);
        width_ = image.cols;
        height_ = image.rows;
    }

    bool valid() const { return texture_ != 0; }
    ImTextureRef texture() const { return ImTextureRef(ImTextureID(texture_)); }
    int width() const { return width_; }
    int height() const { return height_; }

  private:
    unsigned int texture_ = 0;
    int width_ = 0;
    int height_ = 0;
};
void showImagePanel(const char *title, const ImageTexture &texture) {
    ImGui::Begin(title);
    if (!texture.valid()) {
        ImGui::TextUnformatted("Waiting for a frame");
        ImGui::End();
        return;
    }
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float scale = std::min(available.x / float(texture.width()), available.y / float(texture.height()));
    ImGui::Image(texture.texture(), ImVec2(float(texture.width()) * scale, float(texture.height()) * scale),
                 ImVec2(0.F, 0.F), ImVec2(1.F, 1.F));
    ImGui::End();
}
void showStereoPanel(const ImageTexture &left, const ImageTexture &right) {
    ImGui::Begin("Stereo Pair");
    if (!left.valid() || !right.valid()) {
        ImGui::TextUnformatted("Waiting for a stereo frame");
        ImGui::End();
        return;
    }
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float image_width = float(left.width() + right.width());
    const float image_height = float(std::max(left.height(), right.height()));
    const float scale = std::min((available.x - spacing) / image_width, available.y / image_height);
    const ImVec2 left_size(float(left.width()) * scale, float(left.height()) * scale);
    const ImVec2 right_size(float(right.width()) * scale, float(right.height()) * scale);
    ImGui::Image(left.texture(), left_size);
    ImGui::SameLine(0.F, spacing);
    ImGui::Image(right.texture(), right_size);
    ImGui::End();
}

} // namespace
int main(int argc, char **argv) {
    try {
        const auto options = parse(argc, argv);
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
        GLFWwindow *window = glfwCreateWindow(1600, 1000, "FFS Live Viewer", nullptr, nullptr);
        if (!window) {
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
        ImGuiIO &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        {
        ffs_viewer::ui::OpenGLPointCloudViewer viewer;
        ImageTexture left_texture;
        ImageTexture right_texture;
        ImageTexture disparity_texture;
        ffs_viewer::live::LivePipeline pipeline({options.engine_dir, options.point_step, options.max_depth_m});
        std::shared_ptr<const ffs_viewer::live::RenderFrame> displayed_frame;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::DockSpaceOverViewport();
            ImGui::Begin("Controls");
            const bool running = pipeline.running();
            ImGui::Text("Status: %s", pipeline.status().c_str());
            if (ImGui::Button("Start") && !running)
                pipeline.start();
            ImGui::SameLine();
            if (ImGui::Button("Stop") && running) {
                pipeline.stop();
                displayed_frame.reset();
            }
            ImGui::SameLine();
            if (ImGui::Button("Quit"))
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            ImGui::Separator();
            ImGui::Text("Point step: %d   Max depth: %.2f m", options.point_step, options.max_depth_m);
            ImGui::End();

            const auto latest = pipeline.latestFrame();
            if (latest && latest != displayed_frame) {
                viewer.update(latest->xyz, latest->rgb);
                left_texture.upload(latest->left);
                right_texture.upload(latest->right);
                disparity_texture.upload(latest->disparity);
                displayed_frame = latest;
            }
            showStereoPanel(left_texture, right_texture);
            showImagePanel("Disparity", disparity_texture);
            ImGui::Begin("Point Cloud");
            ImGui::Text("%d valid points", viewer.pointCount());
            const ImVec2 panel_pos = ImGui::GetCursorScreenPos();
            const ImVec2 panel_size = ImGui::GetContentRegionAvail();
            ImGui::InvisibleButton("cloud_canvas", panel_size,
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            const bool hovered = ImGui::IsItemHovered();
            viewer.interact(hovered, ImGui::IsMouseDragging(ImGuiMouseButton_Left),
                            ImGui::IsMouseDragging(ImGuiMouseButton_Right), io.MouseDelta.x, io.MouseDelta.y,
                            hovered ? io.MouseWheel : 0.F);
            viewer.draw(ImGui::GetWindowDrawList(), panel_pos, panel_size, io.DisplayFramebufferScale.x,
                        io.DisplayFramebufferScale.y, io.DisplaySize.y * io.DisplayFramebufferScale.y);
            ImGui::End();
            ImGui::Render();
            int framebuffer_width = 0;
            int framebuffer_height = 0;
            glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
            glViewport(0, 0, framebuffer_width, framebuffer_height);
            glClearColor(.08F, .08F, .08F, 1.F);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }
        pipeline.stop();
        }
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ffs_live_viewer: " << error.what() << '\n';
        return 1;
}
    }

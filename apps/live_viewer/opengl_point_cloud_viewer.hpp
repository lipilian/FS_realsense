#pragma once

#include <cstdint>
#include <vector>

struct GLFWwindow;

namespace ffs_viewer::ui {

class OpenGLPointCloudViewer {
  public:
    OpenGLPointCloudViewer();
    ~OpenGLPointCloudViewer() noexcept;

    OpenGLPointCloudViewer(const OpenGLPointCloudViewer &) = delete;
    OpenGLPointCloudViewer &operator=(const OpenGLPointCloudViewer &) = delete;

    bool shouldClose() const;
    void render(const std::vector<float> &xyz, const std::vector<std::uint8_t> &rgb);
    void pollEvents() const;

  private:
    struct Camera {
        float yaw = 20.F;
        float pitch = -15.F;
        float distance = 4.F;
        float pan_x = 0.F;
        float pan_y = 0.F;
        double last_x = 0.0;
        double last_y = 0.0;
        bool left = false;
        bool right = false;
    };

    static Camera *cameraFor(GLFWwindow *window);
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int modifiers);
    static void cursorCallback(GLFWwindow *window, double x, double y);
    static void scrollCallback(GLFWwindow *window, double x_offset, double y_offset);

    GLFWwindow *window_ = nullptr;
    unsigned int xyz_vbo_ = 0;
    unsigned int rgb_vbo_ = 0;
    Camera camera_;
};

} // namespace ffs_viewer::ui

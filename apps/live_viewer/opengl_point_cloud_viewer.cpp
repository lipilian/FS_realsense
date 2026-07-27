#include "opengl_point_cloud_viewer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ffs_viewer::ui {

OpenGLPointCloudViewer::OpenGLPointCloudViewer() {
    if (!glfwInit())
        throw std::runtime_error("GLFW initialization failed");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    window_ = glfwCreateWindow(1000, 800, "FFS OpenGL Point Cloud", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("OpenGL window creation failed");
    }

    glfwMakeContextCurrent(window_);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        throw std::runtime_error("GLEW initialization failed");
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.F, 0.F, 0.F, 1.F);
    glGenBuffers(1, &xyz_vbo_);
    glGenBuffers(1, &rgb_vbo_);

    glfwSetWindowUserPointer(window_, &camera_);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorCallback);
    glfwSetScrollCallback(window_, scrollCallback);
}

OpenGLPointCloudViewer::~OpenGLPointCloudViewer() noexcept {
    if (window_) {
        glfwMakeContextCurrent(window_);
        glDeleteBuffers(1, &xyz_vbo_);
        glDeleteBuffers(1, &rgb_vbo_);
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool OpenGLPointCloudViewer::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void OpenGLPointCloudViewer::render(const std::vector<float> &xyz,
                                    const std::vector<std::uint8_t> &rgb) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (height == 0)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(xyz.size() * sizeof(float)), xyz.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(rgb.size()), rgb.data(), GL_DYNAMIC_DRAW);

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    constexpr float near_plane = .05F;
    constexpr float top = near_plane * .55F;
    glFrustum(-top * float(width) / float(height), top * float(width) / float(height), -top, top,
              near_plane, 40.F);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(camera_.pan_x, camera_.pan_y, -camera_.distance);
    glRotatef(camera_.pitch, 1.F, 0.F, 0.F);
    glRotatef(camera_.yaw, 0.F, 1.F, 0.F);
    glTranslatef(0.F, 0.F, -2.F);
    glPointSize(2.F);
    glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);
    glDrawArrays(GL_POINTS, 0, GLsizei(xyz.size() / 3));
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glfwSwapBuffers(window_);
}

void OpenGLPointCloudViewer::pollEvents() const {
    glfwPollEvents();
}

OpenGLPointCloudViewer::Camera *OpenGLPointCloudViewer::cameraFor(GLFWwindow *window) {
    return static_cast<Camera *>(glfwGetWindowUserPointer(window));
}

void OpenGLPointCloudViewer::mouseButtonCallback(GLFWwindow *window, int button, int action, int) {
    auto *camera = cameraFor(window);
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    camera->last_x = x;
    camera->last_y = y;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        camera->left = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        camera->right = action == GLFW_PRESS;
}

void OpenGLPointCloudViewer::cursorCallback(GLFWwindow *window, double x, double y) {
    auto *camera = cameraFor(window);
    float dx = float(x - camera->last_x);
    float dy = float(y - camera->last_y);
    if (camera->left) {
        camera->yaw += dx * .35F;
        camera->pitch = std::clamp(camera->pitch + dy * .35F, -89.F, 89.F);
    }
    if (camera->right) {
        camera->pan_x += dx * .002F * camera->distance;
        camera->pan_y -= dy * .002F * camera->distance;
    }
    camera->last_x = x;
    camera->last_y = y;
}

void OpenGLPointCloudViewer::scrollCallback(GLFWwindow *window, double, double y_offset) {
    auto *camera = cameraFor(window);
    camera->distance = std::clamp(camera->distance * std::pow(.85F, float(y_offset)), .2F, 30.F);
}

} // namespace ffs_viewer::ui

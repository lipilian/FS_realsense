#include "opengl_point_cloud_viewer.hpp"

#include <GL/glew.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ffs_viewer::ui {

OpenGLPointCloudViewer::OpenGLPointCloudViewer() {
    glGenBuffers(1, &xyz_vbo_);
    glGenBuffers(1, &rgb_vbo_);
}

OpenGLPointCloudViewer::~OpenGLPointCloudViewer() noexcept {
    glDeleteBuffers(1, &xyz_vbo_);
    glDeleteBuffers(1, &rgb_vbo_);
}


void OpenGLPointCloudViewer::update(const std::vector<float> &xyz,
                                    const std::vector<std::uint8_t> &rgb) {

    glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(xyz.size() * sizeof(float)), xyz.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(rgb.size()), rgb.data(), GL_DYNAMIC_DRAW);
    point_count_ = int(xyz.size() / 3);

}

void OpenGLPointCloudViewer::interact(bool hovered, bool orbiting, bool panning, float delta_x,
                                      float delta_y, float wheel) {
    if (!hovered)
        return;
    if (orbiting) {
        camera_.yaw += delta_x * .35F;
        camera_.pitch = std::clamp(camera_.pitch + delta_y * .35F, -89.F, 89.F);
    }
    if (panning) {
        camera_.pan_x += delta_x * .002F * camera_.distance;
        camera_.pan_y -= delta_y * .002F * camera_.distance;
    }
    if (wheel != 0.F)
        camera_.distance = std::clamp(camera_.distance * std::pow(.85F, wheel), .2F, 30.F);
}

void OpenGLPointCloudViewer::draw(ImDrawList *draw_list, const ImVec2 &screen_pos, const ImVec2 &size,
                                  float scale_x, float scale_y, float framebuffer_height) {
    if (size.x <= 0.F || size.y <= 0.F)
        return;
    draw_request_.renderer = this;
    draw_request_.x = screen_pos.x;
    draw_request_.y = screen_pos.y;
    draw_request_.width = size.x;
    draw_request_.height = size.y;
    draw_request_.scale_x = scale_x;
    draw_request_.scale_y = scale_y;
    draw_request_.framebuffer_height = framebuffer_height;
    draw_list->AddRectFilled(screen_pos, ImVec2(screen_pos.x + size.x, screen_pos.y + size.y),
                             IM_COL32(12, 15, 20, 255));
    draw_list->AddCallback(drawCallback, &draw_request_);
    draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

}
void OpenGLPointCloudViewer::drawCallback(const ImDrawList *, const ImDrawCmd *command) {
    const auto *request = static_cast<const DrawRequest *>(command->UserCallbackData);
    request->renderer->render(*request);
}
void OpenGLPointCloudViewer::render(const DrawRequest &request) const {
    const int width = std::max(1, int(request.width * request.scale_x));
    const int height = std::max(1, int(request.height * request.scale_y));
    const int x = int(request.x * request.scale_x);
    const int y = int(request.framebuffer_height - (request.y + request.height) * request.scale_y);

    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, width, height);
    glViewport(x, y, width, height);
    glClearColor(0.F, 0.F, 0.F, 1.F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
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
    glDrawArrays(GL_POINTS, 0, point_count_);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_SCISSOR_TEST);
}



} // namespace ffs_viewer::ui

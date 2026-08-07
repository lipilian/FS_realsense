#include "ffs_viewer/ui/sentech_point_cloud_viewer.hpp"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <imgui.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace ffs_viewer::ui {

SentechPointCloudViewer::SentechPointCloudViewer() {
    glGenBuffers(1, &xyz_vbo_);
    glGenBuffers(1, &rgb_vbo_);
    glGenBuffers(1, &cuda_vbo_);
}

SentechPointCloudViewer::~SentechPointCloudViewer() noexcept {
    if (cuda_resource_ != nullptr)
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource *>(cuda_resource_));
    glDeleteBuffers(1, &xyz_vbo_);
    glDeleteBuffers(1, &rgb_vbo_);
    glDeleteBuffers(1, &cuda_vbo_);
}

void SentechPointCloudViewer::update(const std::vector<float> &xyz,
                                     const std::vector<std::uint8_t> &rgb) {
    if (cuda_resource_ != nullptr) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource *>(cuda_resource_));
        cuda_resource_ = nullptr;
    }
    use_cuda_vbo_ = false;
    point_count_ = static_cast<int>(xyz.size() / 3U);
    glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(xyz.size() * sizeof(float)),
                 xyz.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(rgb.size()), rgb.data(), GL_DYNAMIC_DRAW);
}

void SentechPointCloudViewer::updateCuda(const ffs_viewer::geometry::LiveCloudFrame &cloud) {
    if (!cloud.valid())
        return;
    if (cuda_resource_ != nullptr) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource *>(cuda_resource_));
        cuda_resource_ = nullptr;
    }
    glBindBuffer(GL_ARRAY_BUFFER, cuda_vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(cloud.point_count) *
                     sizeof(ffs_viewer::geometry::GpuPointVertex),
                 nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsResource *resource = nullptr;
    if (cudaGraphicsGLRegisterBuffer(&resource, cuda_vbo_, cudaGraphicsRegisterFlagsWriteDiscard) !=
        cudaSuccess)
        throw std::runtime_error("CUDA failed to register live point-cloud VBO");
    cuda_resource_ = resource;
    if (cudaGraphicsMapResources(1, &resource, 0) != cudaSuccess)
        throw std::runtime_error("CUDA failed to map live point-cloud VBO");
    void *device_vertices = nullptr;
    std::size_t bytes = 0;
    if (cudaGraphicsResourceGetMappedPointer(&device_vertices, &bytes, resource) != cudaSuccess ||
        bytes < static_cast<std::size_t>(cloud.point_count) *
                    sizeof(ffs_viewer::geometry::GpuPointVertex))
        throw std::runtime_error("CUDA live point-cloud VBO has unexpected size");
    if (cloud.ready_event != nullptr && cudaStreamWaitEvent(0, cloud.ready_event, 0) != cudaSuccess)
        throw std::runtime_error("CUDA failed to wait for live point-cloud projection");
    if (cudaMemcpyAsync(device_vertices, cloud.d_vertices,
                        static_cast<std::size_t>(cloud.point_count) *
                            sizeof(ffs_viewer::geometry::GpuPointVertex),
                        cudaMemcpyDeviceToDevice, 0) != cudaSuccess ||
        cudaStreamSynchronize(0) != cudaSuccess || cudaGraphicsUnmapResources(1, &resource, 0) != cudaSuccess)
        throw std::runtime_error("CUDA failed to upload live point-cloud VBO");
    point_count_ = cloud.point_count;
    use_cuda_vbo_ = true;
}

void SentechPointCloudViewer::resetToLeftCameraView() {
    camera_ = {};
    camera_.distance = -1.5F;
}

void SentechPointCloudViewer::interact(bool hovered, bool orbiting, bool panning, float delta_x,
                                       float delta_y, float wheel) {
    if (!hovered)
        return;
    if (orbiting) {
        camera_.yaw += delta_x * 0.35F;
        camera_.pitch = std::clamp(camera_.pitch + delta_y * 0.35F, -89.0F, 89.0F);
    }
    if (panning) {
        camera_.pan_x += delta_x * 0.002F * camera_.distance;
        camera_.pan_y -= delta_y * 0.002F * camera_.distance;
    }
    if (wheel != 0.0F)
        camera_.distance = std::clamp(camera_.distance - wheel * 0.15F, -12.0F, 30.0F);
}

void SentechPointCloudViewer::draw(ImDrawList *draw_list, const ImVec2 &screen_pos, const ImVec2 &size,
                                   float framebuffer_scale_x, float framebuffer_scale_y,
                                   float framebuffer_height) {
    if (size.x <= 0.0F || size.y <= 0.0F)
        return;
    draw_request_ = {this, screen_pos.x, screen_pos.y, size.x, size.y, framebuffer_scale_x,
                     framebuffer_scale_y, framebuffer_height};
    draw_list->AddRectFilled(screen_pos, ImVec2(screen_pos.x + size.x, screen_pos.y + size.y),
                             IM_COL32(12, 15, 20, 255));
    draw_list->AddCallback(drawCallback, &draw_request_);
    draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

int SentechPointCloudViewer::pointCount() const noexcept {
    return point_count_;
}

void SentechPointCloudViewer::drawCallback(const ImDrawList *, const ImDrawCmd *command) {
    static_cast<const DrawRequest *>(command->UserCallbackData)->viewer->render(
        *static_cast<const DrawRequest *>(command->UserCallbackData));
}

void SentechPointCloudViewer::render(const DrawRequest &request) const {
    const int width = std::max(1, static_cast<int>(request.width * request.scale_x));
    const int height = std::max(1, static_cast<int>(request.height * request.scale_y));
    const int x = static_cast<int>(request.x * request.scale_x);
    const int y = static_cast<int>(request.framebuffer_height -
                                   (request.y + request.height) * request.scale_y);

    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, width, height);
    glViewport(x, y, width, height);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(0);
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    constexpr float near_plane = 0.05F;
    constexpr float top = near_plane * 0.55F;
    glFrustum(-top * static_cast<float>(width) / height, top * static_cast<float>(width) / height,
              -top, top, near_plane, 40.0F);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(camera_.pan_x, camera_.pan_y, -camera_.distance);
    glRotatef(camera_.pitch, 1.0F, 0.0F, 0.0F);
    glRotatef(camera_.yaw, 0.0F, 1.0F, 0.0F);
    glTranslatef(0.0F, 0.0F, -2.0F);
    glScalef(1.0F, 1.0F, -1.0F);
    glPointSize(2.0F);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    if (use_cuda_vbo_) {
        glBindBuffer(GL_ARRAY_BUFFER, cuda_vbo_);
        glVertexPointer(3, GL_FLOAT, sizeof(ffs_viewer::geometry::GpuPointVertex), nullptr);
        glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(ffs_viewer::geometry::GpuPointVertex),
                       reinterpret_cast<void *>(3 * sizeof(float)));
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);
    }
    glDrawArrays(GL_POINTS, 0, point_count_);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_SCISSOR_TEST);
}

} // namespace ffs_viewer::ui

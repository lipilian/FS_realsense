#pragma once

#include <cstdint>
#include <vector>

namespace ffs_viewer::geometry { struct FinalCloudFrame; }

struct ImDrawCmd;
struct ImDrawList;
struct ImVec2;

namespace ffs_viewer::ui {

class OpenGLPointCloudViewer {
  public:
    OpenGLPointCloudViewer();
    ~OpenGLPointCloudViewer() noexcept;

    OpenGLPointCloudViewer(const OpenGLPointCloudViewer &) = delete;
    OpenGLPointCloudViewer &operator=(const OpenGLPointCloudViewer &) = delete;

    void update(const std::vector<float> &xyz, const std::vector<std::uint8_t> &rgb);
    void updateCudaFinal(const ffs_viewer::geometry::FinalCloudFrame &cloud);
    void setMaxDepth(float max_depth_m);
    void draw(ImDrawList *draw_list, const ImVec2 &screen_pos, const ImVec2 &size, float scale_x,
              float scale_y, float framebuffer_height);
    void interact(bool hovered, bool orbiting, bool panning, float delta_x, float delta_y, float wheel);
    int pointCount() const { return point_count_; }

  private:
    struct Camera {
        float yaw = 20.F;
        float pitch = -15.F;
        float distance = .5F;
        float pan_x = 0.F;
        float pan_y = 0.F;
    };

    struct DrawRequest {
        OpenGLPointCloudViewer *renderer = nullptr;
        float x = 0.F;
        float y = 0.F;
        float width = 0.F;
        float height = 0.F;
        float scale_x = 1.F;
        float scale_y = 1.F;
        float framebuffer_height = 0.F;
    };

    static void drawCallback(const ImDrawList *parent_list, const ImDrawCmd *command);
    void render(const DrawRequest &request) const;

    unsigned int xyz_vbo_ = 0;
    unsigned int rgb_vbo_ = 0;
    unsigned int cuda_vbo_ = 0;
    void* cuda_resource_ = nullptr;
    bool use_cuda_vbo_ = false;
    int point_count_ = 0;
    float max_depth_m_ = 10.F;
    static constexpr float kCloudDepthOffsetM = 2.F;
    Camera camera_;
    DrawRequest draw_request_;
};

} // namespace ffs_viewer::ui

#pragma once

#include <cstdint>
#include <vector>

struct ImDrawCmd;
struct ImDrawList;
struct ImVec2;

namespace ffs_viewer::ui {

class SentechPointCloudViewer final {
  public:
    SentechPointCloudViewer();
    ~SentechPointCloudViewer() noexcept;

    SentechPointCloudViewer(const SentechPointCloudViewer &) = delete;
    SentechPointCloudViewer &operator=(const SentechPointCloudViewer &) = delete;

    void update(const std::vector<float> &xyz, const std::vector<std::uint8_t> &rgb);
    void resetToLeftCameraView();
    void interact(bool hovered, bool orbiting, bool panning, float delta_x, float delta_y,
                  float wheel);
    void draw(ImDrawList *draw_list, const ImVec2 &screen_pos, const ImVec2 &size,
              float framebuffer_scale_x, float framebuffer_scale_y, float framebuffer_height);
    int pointCount() const noexcept;

  private:
    struct Camera {
        float yaw = 0.0F;
        float pitch = 0.0F;
        float pan_x = 0.0F;
        float pan_y = 0.0F;
        float distance = -1.5F;
    };
    struct DrawRequest {
        SentechPointCloudViewer *viewer = nullptr;
        float x = 0.0F;
        float y = 0.0F;
        float width = 0.0F;
        float height = 0.0F;
        float scale_x = 1.0F;
        float scale_y = 1.0F;
        float framebuffer_height = 0.0F;
    };

    static void drawCallback(const ImDrawList *, const ImDrawCmd *command);
    void render(const DrawRequest &request) const;

    unsigned int xyz_vbo_ = 0;
    unsigned int rgb_vbo_ = 0;
    int point_count_ = 0;
    Camera camera_;
    DrawRequest draw_request_;
};

} // namespace ffs_viewer::ui

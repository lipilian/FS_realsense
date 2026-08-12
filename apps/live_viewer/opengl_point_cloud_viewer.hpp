#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace ffs_viewer::geometry { struct FinalCloudFrame; }

struct ImDrawCmd;
struct ImDrawList;
struct ImVec2;

namespace ffs_viewer::ui {

struct MeshComponentArea {
    int triangle_count = 0;
    float area_m2 = 0.F;
};

class OpenGLPointCloudViewer {
  public:
    OpenGLPointCloudViewer();
    ~OpenGLPointCloudViewer() noexcept;

    OpenGLPointCloudViewer(const OpenGLPointCloudViewer &) = delete;
    OpenGLPointCloudViewer &operator=(const OpenGLPointCloudViewer &) = delete;

    void update(const std::vector<float> &xyz, const std::vector<std::uint8_t> &rgb);
    void updateCudaFinal(const ffs_viewer::geometry::FinalCloudFrame &cloud,
                         const std::uint8_t* host_mask = nullptr, int mask_width = 0, int mask_height = 0,
                         bool show_mesh = false);
    void resetToLeftCameraView();
    void setMaxDepth(float max_depth_m);
    void draw(ImDrawList *draw_list, const ImVec2 &screen_pos, const ImVec2 &size, float scale_x,
              float scale_y, float framebuffer_height);
    void interact(bool hovered, bool orbiting, bool panning, float delta_x, float delta_y, float wheel);
    int pointCount() const { return point_count_; }
    float meshAreaM2() const noexcept { return mesh_area_m2_; }
    const std::vector<MeshComponentArea> &meshComponentAreas() const noexcept {
        return mesh_component_areas_;
    }
    bool hasVcgMesh() const noexcept;
    void saveVcgMeshObj(const std::filesystem::path &path) const;

  private:
    struct Camera {
        float yaw = 0.F;
        float pitch = 0.F;
        float distance = -1.5F;
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
    unsigned int cuda_ebo_ = 0;
    void* cuda_resource_ = nullptr;
    void* cuda_index_resource_ = nullptr;
    std::uint8_t* d_mask_ = nullptr;
    std::size_t mask_capacity_ = 0;
    bool use_cuda_vbo_ = false;
    bool show_mesh_ = false;
    int mesh_index_count_ = 0;
    float mesh_area_m2_ = 0.F;
    std::vector<MeshComponentArea> mesh_component_areas_;
    struct VcgMeshStorage;
    std::unique_ptr<VcgMeshStorage> vcg_mesh_;
    int point_count_ = 0;
    float max_depth_m_ = 10.F;
    static constexpr float kCloudDepthOffsetM = 2.F;
    Camera camera_;
    DrawRequest draw_request_;
};

} // namespace ffs_viewer::ui

#include "opengl_point_cloud_viewer.hpp"
#include "ffs_viewer/geometry/final_cloud_processor.hpp"

#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/complex.h>

#include <opencv2/imgproc.hpp>

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace ffs_viewer::ui {
namespace {

class VcgAreaVertex;
class VcgAreaFace;

struct VcgAreaUsedTypes : public vcg::UsedTypes<
    vcg::Use<VcgAreaVertex>::AsVertexType, vcg::Use<VcgAreaFace>::AsFaceType> {};

class VcgAreaVertex final
    : public vcg::Vertex<VcgAreaUsedTypes, vcg::vertex::Coord3f> {};

class VcgAreaFace final
    : public vcg::Face<VcgAreaUsedTypes, vcg::face::VertexRef, vcg::face::FFAdj,
                       vcg::face::Mark, vcg::face::BitFlags> {};

class VcgAreaMesh final
    : public vcg::tri::TriMesh<std::vector<VcgAreaVertex>, std::vector<VcgAreaFace>> {};

struct AdaptiveMesh {
    std::vector<std::uint32_t> indices;
    std::vector<MeshComponentArea> components;
};

bool triangleIsFullyInsideMask(const cv::Mat &mask, const cv::Point &a,
                               const cv::Point &b, const cv::Point &c) {
    const int min_x = std::max(0, std::min({a.x, b.x, c.x}));
    const int max_x = std::min(mask.cols - 1, std::max({a.x, b.x, c.x}));
    const int min_y = std::max(0, std::min({a.y, b.y, c.y}));
    const int max_y = std::min(mask.rows - 1, std::max({a.y, b.y, c.y}));
    const auto cross = [](const cv::Point &p, const cv::Point &q, int x, int y) {
        return static_cast<long long>(q.x - p.x) * (y - p.y) -
               static_cast<long long>(q.y - p.y) * (x - p.x);
    };
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const long long ab = cross(a, b, x, y);
            const long long bc = cross(b, c, x, y);
            const long long ca = cross(c, a, x, y);
            if ((ab < 0 || bc < 0 || ca < 0) && (ab > 0 || bc > 0 || ca > 0))
                continue;
            if (mask.at<std::uint8_t>(y, x) == 0)
                return false;
        }
    }
    return true;
}

AdaptiveMesh buildContourMesh(const ffs_viewer::geometry::FinalCloudCpuMesh &cpu_mesh,
                              const std::uint8_t *host_mask, int mask_width, int mask_height,
                              int mesh_step, float depth_threshold_m, VcgAreaMesh &mesh) {
    AdaptiveMesh result;
    if (cpu_mesh.width <= 1 || cpu_mesh.height <= 1 || host_mask == nullptr ||
        mask_width <= 0 || mask_height <= 0 || mesh_step <= 0)
        return result;

    const std::size_t pixel_count =
        static_cast<std::size_t>(cpu_mesh.width) * cpu_mesh.height;
    if (cpu_mesh.xyz.size() != pixel_count * 3U || cpu_mesh.valid.size() != pixel_count)
        throw std::runtime_error("CPU final mesh has invalid dimensions");

    cv::Mat source_mask(mask_height, mask_width, CV_8UC1,
                        const_cast<std::uint8_t *>(host_mask));
    cv::Mat mask;
    cv::resize(source_mask, mask, cv::Size(cpu_mesh.width, cpu_mesh.height), 0.0, 0.0,
               cv::INTER_NEAREST);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<int> candidate_at(pixel_count, -1);
    std::vector<cv::Point> candidates;
    const auto add_candidate = [&](const cv::Point &point) {
        if (point.x < 0 || point.x >= mask.cols || point.y < 0 || point.y >= mask.rows ||
            mask.at<std::uint8_t>(point) == 0)
            return;
        const int pixel = point.y * mask.cols + point.x;
        if (cpu_mesh.valid[pixel] == 0 || candidate_at[pixel] >= 0)
            return;
        candidate_at[pixel] = static_cast<int>(candidates.size());
        candidates.push_back(point);
    };

    const double contour_epsilon = std::max(0.5, 0.5 * static_cast<double>(mesh_step));
    for (const auto &contour : contours) {
        std::vector<cv::Point> simplified;
        cv::approxPolyDP(contour, simplified, contour_epsilon, true);
        for (const cv::Point &point : simplified)
            add_candidate(point);
    }
    const int interior_offset = mesh_step / 2;
    for (int y = interior_offset; y < mask.rows; y += mesh_step)
        for (int x = interior_offset; x < mask.cols; x += mesh_step)
            add_candidate({x, y});

    if (candidates.size() < 3U)
        return result;

    cv::Subdiv2D triangulation(cv::Rect(0, 0, mask.cols, mask.rows));
    for (const cv::Point &point : candidates)
        triangulation.insert(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));

    std::vector<cv::Vec6f> triangles;
    triangulation.getTriangleList(triangles);
    std::vector<std::array<int, 3>> accepted;
    std::unordered_set<std::uint64_t> seen_triangles;
    for (const cv::Vec6f &triangle : triangles) {
        const cv::Point points[] = {
            {cvRound(triangle[0]), cvRound(triangle[1])},
            {cvRound(triangle[2]), cvRound(triangle[3])},
            {cvRound(triangle[4]), cvRound(triangle[5])},
        };
        int ids[3];
        bool valid_triangle = true;
        for (int i = 0; i < 3; ++i) {
            if (points[i].x < 0 || points[i].x >= mask.cols || points[i].y < 0 ||
                points[i].y >= mask.rows) {
                valid_triangle = false;
                break;
            }
            ids[i] = candidate_at[points[i].y * mask.cols + points[i].x];
            if (ids[i] < 0)
                valid_triangle = false;
        }
        if (!valid_triangle || ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] ||
            !triangleIsFullyInsideMask(mask, points[0], points[1], points[2]))
            continue;

        const float z0 = cpu_mesh.xyz[3U * (candidates[ids[0]].y * mask.cols + candidates[ids[0]].x) + 2U];
        const float z1 = cpu_mesh.xyz[3U * (candidates[ids[1]].y * mask.cols + candidates[ids[1]].x) + 2U];
        const float z2 = cpu_mesh.xyz[3U * (candidates[ids[2]].y * mask.cols + candidates[ids[2]].x) + 2U];
        if (!std::isfinite(z0) || !std::isfinite(z1) || !std::isfinite(z2) ||
            std::max({z0, z1, z2}) - std::min({z0, z1, z2}) > depth_threshold_m)
            continue;

        std::array<int, 3> ordered{ids[0], ids[1], ids[2]};
        std::sort(ordered.begin(), ordered.end());
        const std::uint64_t key = static_cast<std::uint64_t>(ordered[0]) |
                                  (static_cast<std::uint64_t>(ordered[1]) << 21U) |
                                  (static_cast<std::uint64_t>(ordered[2]) << 42U);
        if (seen_triangles.insert(key).second)
            accepted.push_back({ids[0], ids[1], ids[2]});
    }

    std::vector<int> mesh_vertex(candidates.size(), -1);
    for (const auto &triangle : accepted)
        for (const int id : triangle)
            mesh_vertex[id] = 0;
    int mesh_vertex_count = 0;
    for (int &vertex : mesh_vertex)
        if (vertex == 0)
            vertex = mesh_vertex_count++;
    mesh.Clear();
    auto vertex = vcg::tri::Allocator<VcgAreaMesh>::AddVertices(mesh, mesh_vertex_count);
    for (std::size_t id = 0; id < candidates.size(); ++id) {
        if (mesh_vertex[id] < 0)
            continue;
        const int pixel = candidates[id].y * mask.cols + candidates[id].x;
        vertex[mesh_vertex[id]].P() = vcg::Point3f(cpu_mesh.xyz[3U * pixel],
                                                    cpu_mesh.xyz[3U * pixel + 1U],
                                                    cpu_mesh.xyz[3U * pixel + 2U]);
    }
    auto face = vcg::tri::Allocator<VcgAreaMesh>::AddFaces(mesh, accepted.size());
    for (const auto &triangle : accepted) {
        for (int corner = 0; corner < 3; ++corner) {
            const int id = triangle[corner];
            face->V(corner) = &mesh.vert[mesh_vertex[id]];
            const cv::Point &point = candidates[id];
            result.indices.push_back(static_cast<std::uint32_t>(point.y * mask.cols + point.x));
        }
        ++face;
    }

    vcg::tri::UpdateTopology<VcgAreaMesh>::FaceFace(mesh);
    std::vector<std::pair<int, VcgAreaMesh::FacePointer>> components;
    vcg::tri::Clean<VcgAreaMesh>::ConnectedComponents(mesh, components);
    result.components.reserve(components.size());
    for (const auto &[triangle_count, seed_face] : components) {
        float area_m2 = 0.F;
        vcg::tri::ConnectedComponentIterator<VcgAreaMesh> face_iterator;
        for (face_iterator.start(mesh, seed_face); !face_iterator.completed(); ++face_iterator)
            area_m2 += 0.5F * vcg::DoubleArea(**face_iterator);
        result.components.push_back({triangle_count, area_m2});
    }
    std::sort(result.components.begin(), result.components.end(),
              [](const MeshComponentArea &a, const MeshComponentArea &b) {
                  return a.area_m2 > b.area_m2;
              });
    return result;
}

} // namespace

struct OpenGLPointCloudViewer::VcgMeshStorage {
    VcgAreaMesh mesh;
};

OpenGLPointCloudViewer::OpenGLPointCloudViewer()
    : vcg_mesh_(std::make_unique<VcgMeshStorage>()) {
    glGenBuffers(1, &xyz_vbo_);
    glGenBuffers(1, &rgb_vbo_);
    glGenBuffers(1, &cuda_vbo_);
    glGenBuffers(1, &cuda_ebo_);
}

OpenGLPointCloudViewer::~OpenGLPointCloudViewer() noexcept {
    if (cuda_resource_ != nullptr) cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_resource_));
    if (d_mask_ != nullptr) cudaFree(d_mask_);
    if (cuda_index_resource_ != nullptr) cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_index_resource_));
    glDeleteBuffers(1, &xyz_vbo_);
    glDeleteBuffers(1, &rgb_vbo_);
    glDeleteBuffers(1, &cuda_vbo_);
    glDeleteBuffers(1, &cuda_ebo_);
}


void OpenGLPointCloudViewer::update(const std::vector<float> &xyz,
                                    const std::vector<std::uint8_t> &rgb) {

    glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(xyz.size() * sizeof(float)), xyz.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(rgb.size()), rgb.data(), GL_DYNAMIC_DRAW);
    point_count_ = int(xyz.size() / 3);
    use_cuda_vbo_ = false;

    show_mesh_ = false;
    mesh_index_count_ = 0;
    mesh_area_m2_ = 0.F;
    mesh_component_areas_.clear();
    vcg_mesh_ = std::make_unique<VcgMeshStorage>();
}

void OpenGLPointCloudViewer::updateCudaFinal(const ffs_viewer::geometry::FinalCloudFrame &cloud,
                                                   const std::uint8_t* host_mask, int mask_width, int mask_height, bool show_mesh) {
    if (cloud.point_count <= 0) return;
    auto display_cloud = cloud;
    if (host_mask != nullptr && mask_width > 0 && mask_height > 0) {
        const std::size_t bytes = std::size_t(mask_width) * std::size_t(mask_height);
        if (mask_capacity_ < bytes) {
            if (d_mask_ != nullptr) cudaFree(d_mask_);
            if (cudaMalloc(reinterpret_cast<void**>(&d_mask_), bytes) != cudaSuccess)
                throw std::runtime_error("CUDA failed to allocate final-cloud mask");
            mask_capacity_ = bytes;
        }
        if (cudaMemcpyAsync(d_mask_, host_mask, bytes, cudaMemcpyHostToDevice, 0) != cudaSuccess)
            throw std::runtime_error("CUDA failed to upload final-cloud mask");
        display_cloud.d_mask = d_mask_;
        display_cloud.mask_width = mask_width;
        display_cloud.mask_height = mask_height;
    }
    if (cuda_resource_ != nullptr) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_resource_));
        cuda_resource_ = nullptr;
    }
    glBindBuffer(GL_ARRAY_BUFFER, cuda_vbo_);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(display_cloud.point_count) * sizeof(ffs_viewer::geometry::GpuPointVertex),
                 nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsResource* resource = nullptr;
    if (cudaGraphicsGLRegisterBuffer(&resource, cuda_vbo_, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess)
        throw std::runtime_error("CUDA failed to register final-cloud OpenGL VBO");
    cuda_resource_ = resource;
    if (cudaGraphicsMapResources(1, &resource, 0) != cudaSuccess)
        throw std::runtime_error("CUDA failed to map final-cloud OpenGL VBO");
    void* device_vertices = nullptr;
    std::size_t bytes = 0;
    if (cudaGraphicsResourceGetMappedPointer(&device_vertices, &bytes, resource) != cudaSuccess ||
        bytes < std::size_t(display_cloud.point_count) * sizeof(ffs_viewer::geometry::GpuPointVertex))
        throw std::runtime_error("CUDA final-cloud OpenGL VBO has unexpected size");
    ffs_viewer::geometry::writeFinalCloudVertices(display_cloud,
        static_cast<ffs_viewer::geometry::GpuPointVertex*>(device_vertices), 0);
    if (cudaStreamSynchronize(0) != cudaSuccess || cudaGraphicsUnmapResources(1, &resource, 0) != cudaSuccess)
        throw std::runtime_error("CUDA failed to finalize final-cloud OpenGL VBO");
    if (cuda_index_resource_ != nullptr) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource *>(cuda_index_resource_));
        cuda_index_resource_ = nullptr;
    }
    show_mesh_ = show_mesh && host_mask != nullptr && mask_width > 0 && mask_height > 0;
    mesh_index_count_ = 0;
    mesh_area_m2_ = 0.F;
    mesh_component_areas_.clear();
    vcg_mesh_ = std::make_unique<VcgMeshStorage>();
    if (show_mesh_) {
        const auto cpu_mesh = ffs_viewer::geometry::prepareFinalCloudMeshForCpu(display_cloud, 0);
        auto adaptive_mesh = buildContourMesh(cpu_mesh, host_mask, mask_width, mask_height,
                                              display_cloud.mesh_step,
                                              display_cloud.mesh_depth_threshold_m, vcg_mesh_->mesh);
        mesh_component_areas_ = std::move(adaptive_mesh.components);
        mesh_index_count_ = static_cast<int>(adaptive_mesh.indices.size());
        for (const MeshComponentArea &component : mesh_component_areas_)
            mesh_area_m2_ += component.area_m2;
        show_mesh_ = mesh_index_count_ > 0 && mesh_area_m2_ > 0.F;
        if (show_mesh_) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cuda_ebo_);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         GLsizeiptr(adaptive_mesh.indices.size() * sizeof(std::uint32_t)),
                         adaptive_mesh.indices.data(), GL_DYNAMIC_DRAW);
        }
    }
    point_count_ = display_cloud.point_count;
    use_cuda_vbo_ = true;
}

bool OpenGLPointCloudViewer::hasVcgMesh() const noexcept {
    return vcg_mesh_ != nullptr && vcg_mesh_->mesh.fn > 0;
}

void OpenGLPointCloudViewer::saveVcgMeshObj(const std::filesystem::path &path) const {
    if (!hasVcgMesh())
        throw std::logic_error("No VCG mesh is available to save");

    const VcgAreaMesh &mesh = vcg_mesh_->mesh;
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Unable to create OBJ file: " + path.string());
    output << "# FoundationStereo VCG mesh; coordinates are in meters" << std::string(1, 10);
    for (const VcgAreaVertex &vertex : mesh.vert) {
        const vcg::Point3f &position = vertex.cP();
        output << "v " << position[0] << " " << position[1] << " " << position[2]
               << std::string(1, 10);
    }
    for (const VcgAreaFace &face : mesh.face) {
        if (face.IsD())
            continue;
        output << "f " << vcg::tri::Index(mesh, face.cV(0)) + 1U << " "
               << vcg::tri::Index(mesh, face.cV(1)) + 1U << " "
               << vcg::tri::Index(mesh, face.cV(2)) + 1U << std::string(1, 10);
    }
    if (!output)
        throw std::runtime_error("Unable to write OBJ file: " + path.string());
}

void OpenGLPointCloudViewer::resetToLeftCameraView() {
    camera_.yaw = 0.F;
    camera_.pitch = 0.F;
    camera_.pan_x = 0.F;
    camera_.pan_y = 0.F;
}

void OpenGLPointCloudViewer::setMaxDepth(float max_depth_m) {
    max_depth_m_ = std::max(0.F, max_depth_m);
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
        camera_.distance = std::clamp(camera_.distance - wheel * .15F,
                                      -(kCloudDepthOffsetM + max_depth_m_), 30.F);
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
    glTranslatef(0.F, 0.F, -kCloudDepthOffsetM);
    // Camera-space depth is +Z; legacy OpenGL views toward -Z.
    glScalef(1.F, 1.F, -1.F);
    glPointSize(2.F);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    if (use_cuda_vbo_) {
        glBindBuffer(GL_ARRAY_BUFFER, cuda_vbo_);
        glVertexPointer(3, GL_FLOAT, sizeof(ffs_viewer::geometry::GpuPointVertex), nullptr);
        glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(ffs_viewer::geometry::GpuPointVertex), reinterpret_cast<void*>(3 * sizeof(float)));
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, xyz_vbo_);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, rgb_vbo_);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);
    }
    if (use_cuda_vbo_ && show_mesh_ && mesh_index_count_ > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, cuda_vbo_);
        glVertexPointer(3, GL_FLOAT, sizeof(ffs_viewer::geometry::GpuPointVertex), nullptr);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cuda_ebo_);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, mesh_index_count_, GL_UNSIGNED_INT, nullptr);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glDrawArrays(GL_POINTS, 0, point_count_);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_SCISSOR_TEST);
}



} // namespace ffs_viewer::ui

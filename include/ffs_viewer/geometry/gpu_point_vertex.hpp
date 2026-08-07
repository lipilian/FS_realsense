#pragma once

#include <cstdint>

namespace ffs_viewer::geometry {

struct GpuPointVertex {
    float x;
    float y;
    float z;
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

} // namespace ffs_viewer::geometry

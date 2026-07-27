#include "opengl_point_cloud_viewer.hpp"
#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/d455_stereo_source.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/highgui.hpp>
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
cv::Mat bgr(const std::vector<std::uint8_t> &v, int w, int h) {
    cv::Mat g(h, w, CV_8UC1, const_cast<std::uint8_t *>(v.data())), r;
    cv::cvtColor(g, r, cv::COLOR_GRAY2BGR);
    return r;
}
cv::Mat dispVis(const ffs_viewer::inference::DisparityFrame &d) {
    float m = 1;
    for (float x : d.values)
        if (std::isfinite(x))
            m = std::max(m, x);
    cv::Mat f(d.height, d.width, CV_32F, const_cast<float *>(d.values.data())), u, c;
    f.convertTo(u, CV_8U, 255. / m);
    cv::applyColorMap(u, c, cv::COLORMAP_TURBO);
    return c;
}
struct CloudData {
    std::vector<float> xyz;
    std::vector<std::uint8_t> rgb;
};
CloudData buildCloud(const ffs_viewer::inference::DisparityFrame &d,
                     const ffs_viewer::io::StereoFrame &f,
                     const ffs_viewer::io::StereoCalibration &k, int step, float maxz) {
    CloudData out;
    out.xyz.reserve(size_t(d.width / step) * size_t(d.height / step) * 3);
    out.rgb.reserve(out.xyz.capacity());
    float fb = k.left.fx * k.baseline_m;
    for (int y = 0; y < d.height; y += step)
        for (int x = 0; x < d.width; x += step) {
            size_t i = size_t(y) * d.width + x;
            float q = d.values[i];
            if (!std::isfinite(q) || q <= 0)
                continue;
            float z = fb / q;
            if (z < .1F || z > maxz)
                continue;
            out.xyz.insert(out.xyz.end(),
                           {(x - k.left.cx) * z / k.left.fx, -(y - k.left.cy) * z / k.left.fy, z});
            auto g = f.left_y8[i];
            out.rgb.insert(out.rgb.end(), {g, g, g});
        }
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        auto o = parse(argc, argv);
        ffs_viewer::ui::OpenGLPointCloudViewer viewer;
        cv::namedWindow("FFS Stereo Pair", cv::WINDOW_NORMAL);
        cv::namedWindow("FFS Disparity", cv::WINDOW_NORMAL);
        ffs_viewer::io::D455StereoSource s;
        s.open();
        auto k = s.calibration();
        ffs_viewer::inference::FfsRunner r(o.engine_dir);
        ffs_viewer::io::StereoFrame f;
        while (!viewer.shouldClose()) {
            s.next(f);
            auto d = r.infer(f);
            auto p = buildCloud(d, f, k, o.point_step, o.max_depth_m);
            viewer.render(p.xyz, p.rgb);
            cv::Mat stereo;
            cv::hconcat(bgr(f.left_y8, f.width, f.height), bgr(f.right_y8, f.width, f.height),
                        stereo);
            cv::imshow("FFS Stereo Pair", stereo);
            cv::imshow("FFS Disparity", dispVis(d));
            if (cv::waitKey(1) == 27)
                break;
            viewer.pollEvents();
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ffs_live_viewer: " << e.what() << '\n';
        return 1;
    }
}

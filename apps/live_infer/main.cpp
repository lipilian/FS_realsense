#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/d455_stereo_source.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void requestStop(int) {
    g_stop_requested = 1;
}

struct Options {
    std::string engine_dir = FFS_VIEWER_DEFAULT_ENGINE_DIR;
    int frames = 0;
    int report_every = 15;
    float max_depth_m = 10.0F;
};

struct DepthStats {
    std::size_t valid_pixels = 0;
    float min_depth_m = std::numeric_limits<float>::infinity();
    float max_depth_m = 0.0F;
};

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "Options:\n"
        << "  --engine-dir <dir>  FFS engine directory (default: "
        << FFS_VIEWER_DEFAULT_ENGINE_DIR << ")\n"
        << "  --frames <count>    Stop after this many pairs; 0 runs until Ctrl-C\n"
        << "  --report-every <n>  Print runtime statistics every n processed pairs (default: 15)\n"
        << "  --max-depth-m <m>   Valid-depth cutoff for console statistics (default: 10)\n"
        << "  --help              Show this help message\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--engine-dir" || argument == "--frames" ||
            argument == "--report-every" || argument == "--max-depth-m") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            const std::string value = argv[i];
            if (argument == "--engine-dir") {
                options.engine_dir = value;
            } else if (argument == "--frames") {
                options.frames = std::stoi(value);
            } else if (argument == "--report-every") {
                options.report_every = std::stoi(value);
            } else {
                options.max_depth_m = std::stof(value);
            }
            continue;
        }
        throw std::runtime_error("Unknown option: " + argument);
    }

    if (options.frames < 0) {
        throw std::runtime_error("--frames must be zero or positive");
    }
    if (options.report_every <= 0) {
        throw std::runtime_error("--report-every must be positive");
    }
    if (!std::isfinite(options.max_depth_m) || options.max_depth_m <= 0.0F) {
        throw std::runtime_error("--max-depth-m must be finite and positive");
    }
    return options;
}

DepthStats summarizeDepth(const ffs_viewer::inference::DisparityFrame& disparity,
                          const ffs_viewer::io::StereoCalibration& calibration,
                          float max_depth_m) {
    DepthStats stats;
    const float focal_baseline = calibration.left.fx * calibration.baseline_m;
    for (const float disparity_px : disparity.values) {
        if (!std::isfinite(disparity_px) || disparity_px <= 0.0F) {
            continue;
        }
        const float depth_m = focal_baseline / disparity_px;
        if (!std::isfinite(depth_m) || depth_m < 0.1F || depth_m > max_depth_m) {
            continue;
        }
        ++stats.valid_pixels;
        stats.min_depth_m = std::min(stats.min_depth_m, depth_m);
        stats.max_depth_m = std::max(stats.max_depth_m, depth_m);
    }
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        std::signal(SIGINT, requestStop);
        std::signal(SIGTERM, requestStop);

        ffs_viewer::io::D455StereoSource source;
        source.open();
        const ffs_viewer::io::StereoCalibration calibration = source.calibration();
        ffs_viewer::inference::FfsRunner runner(options.engine_dir);
        if (runner.modelWidth() != calibration.left.width ||
            runner.modelHeight() != calibration.left.height) {
            throw std::runtime_error("D455 capture dimensions do not match the FFS engine");
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "D455 live: IR1/IR2 " << calibration.left.width << 'x'
                  << calibration.left.height << " Y8 @" << calibration.fps << " Hz\n"
                  << "FFS engine: " << runner.modelWidth() << 'x' << runner.modelHeight()
                  << ", max_disparity=" << runner.maxDisparity() << "\n"
                  << "Press Ctrl-C to stop. Stale queued pairs are dropped.\n";

        ffs_viewer::io::StereoFrame frame;
        int processed = 0;
        int report_samples = 0;
        float accumulated_gpu_ms = 0.0F;
        float accumulated_host_ms = 0.0F;
        const auto start = std::chrono::steady_clock::now();
        auto report_start = start;

        while (g_stop_requested == 0 && (options.frames == 0 || processed < options.frames)) {
            source.next(frame);
            if (frame.left_frame_number != frame.right_frame_number ||
                std::abs(frame.left_timestamp_ms - frame.right_timestamp_ms) > 1.0) {
                throw std::runtime_error("D455 delivered a non-synchronized infrared pair");
            }

            const auto disparity = runner.infer(frame);
            const DepthStats depth = summarizeDepth(disparity, calibration, options.max_depth_m);
            ++processed;
            ++report_samples;
            accumulated_gpu_ms += disparity.timing.gpu_total_ms;
            accumulated_host_ms += disparity.timing.host_total_ms;

            if (report_samples == options.report_every ||
                (options.frames > 0 && processed == options.frames)) {
                const auto now = std::chrono::steady_clock::now();
                const float elapsed_s = std::chrono::duration<float>(now - report_start).count();
                const float process_fps = report_samples / elapsed_s;
                std::cout << "processed=" << processed
                          << " process_fps=" << process_fps
                          << " gpu_ms=" << accumulated_gpu_ms / report_samples
                          << " host_ms=" << accumulated_host_ms / report_samples
                          << " stale_dropped=" << source.droppedPairs()
                          << " valid_depth=" << depth.valid_pixels << '/' << disparity.values.size()
                          << " depth_m=[" << depth.min_depth_m << ", " << depth.max_depth_m << "]\n";
                report_samples = 0;
                accumulated_gpu_ms = 0.0F;
                accumulated_host_ms = 0.0F;
                report_start = now;
            }
        }

        const float total_s = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        std::cout << "stopped: processed=" << processed
                  << " average_fps=" << (processed / total_s)
                  << " stale_dropped=" << source.droppedPairs() << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ffs_live_infer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

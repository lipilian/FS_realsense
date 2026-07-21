#include "ffs_viewer/inference/ffs_runner.hpp"
#include "ffs_viewer/io/db3_stereo_source.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ffs_viewer::inference::DisparityFrame;
using ffs_viewer::inference::FfsRunner;
using ffs_viewer::inference::InferenceTiming;
using ffs_viewer::io::Db3StereoSource;
using ffs_viewer::io::StereoCalibration;
using ffs_viewer::io::StereoFrame;

namespace {

struct Options {
    fs::path input;
    int frames = 20;
    fs::path output;
    bool write_output = false;
    fs::path engine_dir = FFS_VIEWER_DEFAULT_ENGINE_DIR;
    int infer_frame = 0;
    float max_depth_m = 10.0F;
    int benchmark_frames = 0;
    int warmup_frames = 20;
};

std::string frameStem(int index) {
    std::ostringstream stream;
    stream << "frame_" << std::setw(6) << std::setfill('0') << index;
    return stream.str();
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --input <recording.db3> [options]\n"
        << "Options:\n"
        << "  --frames <count>      Number of synchronized pairs to inspect (default: 20)\n"
        << "  --output <dir>        Save Y8 PNG pairs, calibration.yaml, and metadata.csv\n"
        << "  --infer-frame <index> Run FFS on this one-based pair; requires --output\n"
        << "  --benchmark-frames <n> Time n frames after warmup; requires --output\n"
        << "  --warmup-frames <n>    Warmup count for benchmarks (default: 20)\n"
        << "  --max-depth-m <m>   Keep depth and cloud points within this distance (default: 10)\n"
        << "  --engine-dir <dir>    Engine directory (default: "
        << FFS_VIEWER_DEFAULT_ENGINE_DIR << ")\n"
        << "  --help                Show this help message\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--input" || argument == "--frames" || argument == "--output" ||
            argument == "--engine-dir" || argument == "--infer-frame" || argument == "--max-depth-m" || argument == "--benchmark-frames" || argument == "--warmup-frames") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            const std::string value = argv[i];
            if (argument == "--input") {
                options.input = value;
            } else if (argument == "--frames") {
                options.frames = std::stoi(value);
            } else if (argument == "--output") {
                options.output = value;
                options.write_output = true;
            } else if (argument == "--engine-dir") {
                options.engine_dir = value;
            } else if (argument == "--infer-frame") {
                options.infer_frame = std::stoi(value);
            } else if (argument == "--max-depth-m") {
                options.max_depth_m = std::stof(value);
            } else if (argument == "--benchmark-frames") {
                options.benchmark_frames = std::stoi(value);
            } else {
                options.warmup_frames = std::stoi(value);
            }
            continue;
        }
        throw std::runtime_error("Unknown option: " + argument);
    }

    if (options.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (options.frames <= 0) {
        throw std::runtime_error("--frames must be positive");
    }
    if (options.infer_frame < 0) {
        throw std::runtime_error("--infer-frame must be zero (disabled) or positive");
    }
    if (!std::isfinite(options.max_depth_m) || options.max_depth_m <= 0.0F) {
        throw std::runtime_error("--max-depth-m must be finite and positive");
    }

    if (options.benchmark_frames < 0 || options.warmup_frames < 0) {
        throw std::runtime_error("--benchmark-frames and --warmup-frames must be non-negative");
    }
    if (options.benchmark_frames > 0 && !options.write_output) {
        throw std::runtime_error("--benchmark-frames requires --output for CSV results");
    }

    if (options.infer_frame > 0 && !options.write_output) {
        throw std::runtime_error("--infer-frame requires --output so validation artifacts are retained");
    }
    return options;
}

void writeCamera(cv::FileStorage& storage, const char* name,
                 const ffs_viewer::io::CameraIntrinsics& intrinsics) {
    storage << name << "{";
    storage << "width" << intrinsics.width;
    storage << "height" << intrinsics.height;
    storage << "fx" << intrinsics.fx;
    storage << "fy" << intrinsics.fy;
    storage << "cx" << intrinsics.cx;
    storage << "cy" << intrinsics.cy;
    storage << "distortion_model" << intrinsics.distortion_model;
    storage << "distortion_coeffs" << "[";
    for (const float coefficient : intrinsics.distortion_coeffs) {
        storage << coefficient;
    }
    storage << "]";
    storage << "}";
}

void writeCalibration(const fs::path& output, const StereoCalibration& calibration) {
    cv::FileStorage storage(output.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!storage.isOpened()) {
        throw std::runtime_error("Cannot write calibration file: " + output.string());
    }

    writeCamera(storage, "left", calibration.left);
    writeCamera(storage, "right", calibration.right);
    storage << "right_to_left_rotation"
            << cv::Mat(3, 3, CV_32F,
                       const_cast<float*>(calibration.right_to_left_rotation.data()));
    storage << "right_to_left_translation"
            << cv::Mat(3, 1, CV_32F,
                       const_cast<float*>(calibration.right_to_left_translation.data()));
    storage << "baseline_m" << calibration.baseline_m;
    storage << "fps" << calibration.fps;
}

void writePngPair(const fs::path& output, int index, const StereoFrame& frame) {
    const std::string stem = frameStem(index);
    const cv::Mat left(frame.height, frame.width, CV_8UC1,
                       const_cast<std::uint8_t*>(frame.left_y8.data()));
    const cv::Mat right(frame.height, frame.width, CV_8UC1,
                        const_cast<std::uint8_t*>(frame.right_y8.data()));
    if (!cv::imwrite((output / (stem + "_left.png")).string(), left) ||
        !cv::imwrite((output / (stem + "_right.png")).string(), right)) {
        throw std::runtime_error("Failed to write Y8 PNG pair " + std::to_string(index));
    }
}

void writeDisparityArtifacts(const fs::path& output, int index,
                             const DisparityFrame& disparity) {
    const std::string stem = frameStem(index);
    const fs::path raw_path = output / (stem + "_disparity.f32");
    std::ofstream raw(raw_path, std::ios::binary);
    if (!raw) {
        throw std::runtime_error("Cannot write disparity file: " + raw_path.string());
    }
    raw.write(reinterpret_cast<const char*>(disparity.values.data()),
              static_cast<std::streamsize>(disparity.values.size() * sizeof(float)));
    if (!raw) {
        throw std::runtime_error("Failed while writing disparity file: " + raw_path.string());
    }

    const cv::Mat disparity_float(disparity.height, disparity.width, CV_32FC1,
                                  const_cast<float*>(disparity.values.data()));
    cv::Mat disparity_u8;
    float visualization_max = 1.0F;
    for (const float value : disparity.values) {
        if (std::isfinite(value)) {
            visualization_max = std::max(visualization_max, value);
        }
    }
    disparity_float.convertTo(disparity_u8, CV_8UC1, 255.0 / visualization_max);
    cv::Mat visualization;
    cv::applyColorMap(disparity_u8, visualization, cv::COLORMAP_TURBO);
    const fs::path visualization_path = output / (stem + "_disparity_vis.png");
    if (!cv::imwrite(visualization_path.string(), visualization)) {
        throw std::runtime_error("Cannot write disparity preview: " + visualization_path.string());
    }
}

struct PointXyzi {
    float x;
    float y;
    float z;
    float intensity;
};
static_assert(sizeof(PointXyzi) == 4 * sizeof(float));

struct DepthSummary {
    std::size_t valid_pixels = 0;
    float min_depth_m = std::numeric_limits<float>::infinity();
    float max_depth_m = 0.0F;
};

DepthSummary writeDepthAndCloudArtifacts(const fs::path& output, int index,
                                           const DisparityFrame& disparity,
                                           const StereoFrame& frame,
                                           const StereoCalibration& calibration,
                                           float max_depth_m) {
    if (disparity.width != frame.width || disparity.height != frame.height ||
        calibration.left.width != frame.width || calibration.left.height != frame.height) {
        throw std::runtime_error("Disparity, image, and left-camera dimensions must match");
    }
    if (calibration.left.fx <= 0.0F || calibration.left.fy <= 0.0F ||
        calibration.baseline_m <= 0.0F) {
        throw std::runtime_error("Stereo calibration contains invalid intrinsics or baseline");
    }

    const std::size_t pixels = static_cast<std::size_t>(disparity.width) * disparity.height;
    std::vector<float> depth_m(pixels, std::numeric_limits<float>::quiet_NaN());
    std::vector<PointXyzi> points;
    points.reserve(pixels);
    DepthSummary summary;
    const float focal_baseline = calibration.left.fx * calibration.baseline_m;

    for (int y = 0; y < disparity.height; ++y) {
        for (int x = 0; x < disparity.width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * disparity.width + x;
            const float disparity_px = disparity.values[pixel];
            if (!std::isfinite(disparity_px) || disparity_px <= 0.0F) {
                continue;
            }
            const float z = focal_baseline / disparity_px;
            if (!std::isfinite(z) || z < 0.1F || z > max_depth_m) {
                continue;
            }

            depth_m[pixel] = z;
            points.push_back({
                (static_cast<float>(x) - calibration.left.cx) * z / calibration.left.fx,
                (static_cast<float>(y) - calibration.left.cy) * z / calibration.left.fy,
                z,
                static_cast<float>(frame.left_y8[pixel]) / 255.0F,
            });
            ++summary.valid_pixels;
            summary.min_depth_m = std::min(summary.min_depth_m, z);
            summary.max_depth_m = std::max(summary.max_depth_m, z);
        }
    }

    const std::string stem = frameStem(index);
    const fs::path depth_path = output / (stem + "_depth.f32");
    std::ofstream depth_file(depth_path, std::ios::binary);
    if (!depth_file) {
        throw std::runtime_error("Cannot write depth file: " + depth_path.string());
    }
    depth_file.write(reinterpret_cast<const char*>(depth_m.data()),
                     static_cast<std::streamsize>(depth_m.size() * sizeof(float)));
    if (!depth_file) {
        throw std::runtime_error("Failed while writing depth file: " + depth_path.string());
    }

    cv::Mat depth_u8(disparity.height, disparity.width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < disparity.height; ++y) {
        for (int x = 0; x < disparity.width; ++x) {
            const float z = depth_m[static_cast<std::size_t>(y) * disparity.width + x];
            if (std::isfinite(z)) {
                const float normalized = std::clamp(1.0F - z / max_depth_m, 0.0F, 1.0F);
                depth_u8.at<std::uint8_t>(y, x) = std::max(1, static_cast<int>(std::lround(255.0F * normalized)));
            }
        }
    }
    cv::Mat depth_visualization;
    cv::applyColorMap(depth_u8, depth_visualization, cv::COLORMAP_TURBO);
    depth_visualization.setTo(cv::Scalar(0, 0, 0), depth_u8 == 0);
    const fs::path depth_vis_path = output / (stem + "_depth_vis.png");
    if (!cv::imwrite(depth_vis_path.string(), depth_visualization)) {
        throw std::runtime_error("Cannot write depth preview: " + depth_vis_path.string());
    }

    const fs::path cloud_path = output / (stem + "_cloud.ply");
    std::ofstream cloud_file(cloud_path, std::ios::binary);
    if (!cloud_file) {
        throw std::runtime_error("Cannot write point cloud: " + cloud_path.string());
    }
    cloud_file << "ply\nformat binary_little_endian 1.0\n"
               << "element vertex " << points.size() << "\n"
               << "property float x\nproperty float y\nproperty float z\n"
               << "property float intensity\nend_header\n";
    cloud_file.write(reinterpret_cast<const char*>(points.data()),
                     static_cast<std::streamsize>(points.size() * sizeof(PointXyzi)));
    if (!cloud_file) {
        throw std::runtime_error("Failed while writing point cloud: " + cloud_path.string());
    }

    const fs::path metrics_path = output / (stem + "_depth_metrics.csv");
    std::ofstream metrics(metrics_path);
    if (!metrics) {
        throw std::runtime_error("Cannot write depth metrics: " + metrics_path.string());
    }
    metrics << "valid_depth_pixels,total_pixels,min_depth_m,max_depth_m,max_depth_filter_m\n"
            << summary.valid_pixels << "," << pixels << "," << summary.min_depth_m << ","
            << summary.max_depth_m << "," << max_depth_m << "\n";
    return summary;
}

void printDepthSummary(int index, const DepthSummary& summary, std::size_t total_pixels) {
    std::cout << "depth frame " << index
              << ": valid=" << summary.valid_pixels << "/" << total_pixels
              << " range_m=[" << summary.min_depth_m << ", " << summary.max_depth_m << "]\n";
}

void printCalibration(const StereoCalibration& calibration) {
    std::cout << std::fixed << std::setprecision(6)
              << "IR1/IR2: " << calibration.left.width << "x" << calibration.left.height
              << " Y8 @" << calibration.fps << "\n"
              << "left K: fx=" << calibration.left.fx
              << " fy=" << calibration.left.fy
              << " cx=" << calibration.left.cx
              << " cy=" << calibration.left.cy << "\n"
              << "baseline_m: " << calibration.baseline_m << "\n";
}

void printDisparitySummary(int index, const DisparityFrame& disparity) {
    int finite = 0;
    int positive = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : disparity.values) {
        if (!std::isfinite(value)) {
            continue;
        }
        ++finite;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        if (value > 0.0F) {
            ++positive;
        }
    }

    std::cout << "inference frame " << index
              << ": finite=" << finite << '/' << disparity.values.size()
              << " positive=" << positive
              << " min=" << minimum
              << " max=" << maximum << "\n";
}

struct BenchmarkSample {
    int frame_index = 0;
    InferenceTiming timing;
};

struct LatencySummary {
    float mean_ms = 0.0F;
    float p50_ms = 0.0F;
    float p95_ms = 0.0F;
    float min_ms = 0.0F;
    float max_ms = 0.0F;
};

LatencySummary summarizeLatency(std::vector<float> values) {
    if (values.empty()) {
        throw std::runtime_error("Cannot summarize an empty benchmark");
    }
    std::sort(values.begin(), values.end());
    LatencySummary summary;
    for (const float value : values) {
        summary.mean_ms += value;
    }
    summary.mean_ms /= static_cast<float>(values.size());
    const auto percentile = [&values](float fraction) {
        const std::size_t position = static_cast<std::size_t>(std::ceil(fraction * values.size()));
        return values[std::max<std::size_t>(1, position) - 1];
    };
    summary.p50_ms = percentile(0.50F);
    summary.p95_ms = percentile(0.95F);
    summary.min_ms = values.front();
    summary.max_ms = values.back();
    return summary;
}

void writeBenchmarkSummary(const fs::path& output, const std::vector<BenchmarkSample>& samples) {
    std::vector<float> h2d;
    std::vector<float> inference;
    std::vector<float> d2h;
    std::vector<float> gpu_total;
    std::vector<float> host_total;
    h2d.reserve(samples.size());
    inference.reserve(samples.size());
    d2h.reserve(samples.size());
    gpu_total.reserve(samples.size());
    host_total.reserve(samples.size());
    for (const BenchmarkSample& sample : samples) {
        h2d.push_back(sample.timing.h2d_ms);
        inference.push_back(sample.timing.inference_ms);
        d2h.push_back(sample.timing.d2h_ms);
        gpu_total.push_back(sample.timing.gpu_total_ms);
        host_total.push_back(sample.timing.host_total_ms);
    }

    std::ofstream summary_file(output / "benchmark_summary.csv");
    if (!summary_file) {
        throw std::runtime_error("Cannot write benchmark_summary.csv in " + output.string());
    }
    summary_file << "stage,mean_ms,p50_ms,p95_ms,min_ms,max_ms\n";
    const auto write_row = [&summary_file](const char* stage, std::vector<float> values) {
        const LatencySummary summary = summarizeLatency(std::move(values));
        summary_file << stage << "," << summary.mean_ms << "," << summary.p50_ms << ","
                     << summary.p95_ms << "," << summary.min_ms << "," << summary.max_ms << "\n";
    };
    write_row("h2d", std::move(h2d));
    write_row("inference", std::move(inference));
    write_row("d2h", std::move(d2h));
    write_row("gpu_total", std::move(gpu_total));
    write_row("host_total", std::move(host_total));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        Db3StereoSource source(options.input.string());
        source.open();

        const StereoCalibration calibration = source.calibration();
        printCalibration(calibration);

        std::unique_ptr<FfsRunner> runner;
        if (options.infer_frame > 0 || options.benchmark_frames > 0) {
            runner = std::make_unique<FfsRunner>(options.engine_dir.string());
            std::cout << "FFS engine: " << runner->modelWidth() << 'x' << runner->modelHeight()
                      << ", max_disparity=" << runner->maxDisparity() << "\n";
        }

        std::ofstream metadata;
        if (options.write_output) {
            fs::create_directories(options.output);
            writeCalibration(options.output / "calibration.yaml", calibration);
            metadata.open(options.output / "metadata.csv");
            if (!metadata) {
                throw std::runtime_error("Cannot write metadata.csv in " + options.output.string());
            }
            metadata << "index,left_frame_number,right_frame_number,left_timestamp_ms,"
                        "right_timestamp_ms,timestamp_delta_ms\n";
        }

        std::ofstream benchmark;
        std::vector<BenchmarkSample> benchmark_samples;
        if (options.benchmark_frames > 0) {
            benchmark.open(options.output / "benchmark.csv");
            if (!benchmark) {
                throw std::runtime_error("Cannot write benchmark.csv in " + options.output.string());
            }
            benchmark << "frame_index,h2d_ms,inference_ms,d2h_ms,gpu_total_ms,host_total_ms\n";
            benchmark_samples.reserve(static_cast<std::size_t>(options.benchmark_frames));
        }

        constexpr double kMaxTimestampDeltaMs = 1.0;
        const int frames_to_read = std::max({options.frames, options.infer_frame,
            options.warmup_frames + options.benchmark_frames});
        int mismatched_pairs = 0;
        for (int index = 1; index <= frames_to_read; ++index) {
            StereoFrame frame;
            if (!source.next(frame)) {
                throw std::runtime_error("DB3 playback ended before the requested frame count");
            }

            const double timestamp_delta_ms =
                std::abs(frame.left_timestamp_ms - frame.right_timestamp_ms);
            const bool synchronized =
                frame.left_frame_number == frame.right_frame_number &&
                timestamp_delta_ms <= kMaxTimestampDeltaMs;
            if (!synchronized) {
                ++mismatched_pairs;
            }

            std::cout << "frame " << index
                      << ": left=" << frame.left_frame_number
                      << " right=" << frame.right_frame_number
                      << " timestamp_delta_ms=" << std::setprecision(3) << timestamp_delta_ms
                      << (synchronized ? " synchronized" : " MISMATCH") << "\n";

            const bool save_pair = options.write_output &&
                (options.benchmark_frames == 0 || index == options.infer_frame);
            if (save_pair) {
                writePngPair(options.output, index, frame);
                metadata << index << ',' << frame.left_frame_number << ','
                         << frame.right_frame_number << ',' << std::setprecision(17)
                         << frame.left_timestamp_ms << ',' << frame.right_timestamp_ms << ','
                         << timestamp_delta_ms << '\n';
            }
            const bool in_warmup = options.benchmark_frames > 0 &&
                index <= options.warmup_frames;
            const bool in_benchmark = options.benchmark_frames > 0 &&
                index > options.warmup_frames &&
                index <= options.warmup_frames + options.benchmark_frames;
            if (in_warmup || in_benchmark || index == options.infer_frame) {
                const DisparityFrame disparity = runner->infer(frame);
                if (in_benchmark) {
                    benchmark_samples.push_back({index, disparity.timing});
                    benchmark << index << "," << disparity.timing.h2d_ms << ","
                              << disparity.timing.inference_ms << "," << disparity.timing.d2h_ms << ","
                              << disparity.timing.gpu_total_ms << "," << disparity.timing.host_total_ms << "\n";
                }
                if (index == options.infer_frame) {
                    writeDisparityArtifacts(options.output, index, disparity);
                    const DepthSummary depth_summary = writeDepthAndCloudArtifacts(
                        options.output, index, disparity, frame, calibration, options.max_depth_m);
                    printDepthSummary(index, depth_summary, disparity.values.size());
                    printDisparitySummary(index, disparity);
                }
            }
        }

        if (options.benchmark_frames > 0) {
            writeBenchmarkSummary(options.output, benchmark_samples);
            std::cout << "benchmark_samples: " << benchmark_samples.size()
                      << " (after " << options.warmup_frames << " warmup frames)\n";
        }

        std::cout << "validated_pairs: " << frames_to_read
                  << ", synchronization_mismatches: " << mismatched_pairs << "\n";
        return mismatched_pairs == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "ffs_offline_validate: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

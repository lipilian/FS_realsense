#include "ffs_viewer/io/db3_stereo_source.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using ffs_viewer::io::Db3StereoSource;
using ffs_viewer::io::StereoCalibration;
using ffs_viewer::io::StereoFrame;

namespace {

struct Options {
    fs::path input;
    int frames = 20;
    fs::path output;
    bool write_output = false;
};

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --input <recording.db3> [options]\n"
        << "Options:\n"
        << "  --frames <count>   Number of synchronized pairs to inspect (default: 20)\n"
        << "  --output <dir>     Save Y8 PNG pairs, calibration.yaml, and metadata.csv\n"
        << "  --help             Show this help message\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--input" || argument == "--frames" || argument == "--output") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            const std::string value = argv[i];
            if (argument == "--input") {
                options.input = value;
            } else if (argument == "--frames") {
                options.frames = std::stoi(value);
            } else {
                options.output = value;
                options.write_output = true;
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
    const std::string stem = "frame_" + [&] {
        std::ostringstream stream;
        stream << std::setw(6) << std::setfill('0') << index;
        return stream.str();
    }();

    const cv::Mat left(frame.height, frame.width, CV_8UC1,
                       const_cast<std::uint8_t*>(frame.left_y8.data()));
    const cv::Mat right(frame.height, frame.width, CV_8UC1,
                        const_cast<std::uint8_t*>(frame.right_y8.data()));
    if (!cv::imwrite((output / (stem + "_left.png")).string(), left) ||
        !cv::imwrite((output / (stem + "_right.png")).string(), right)) {
        throw std::runtime_error("Failed to write Y8 PNG pair " + std::to_string(index));
    }
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        Db3StereoSource source(options.input.string());
        source.open();

        const StereoCalibration calibration = source.calibration();
        printCalibration(calibration);

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

        constexpr double kMaxTimestampDeltaMs = 1.0;
        int mismatched_pairs = 0;
        for (int index = 1; index <= options.frames; ++index) {
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

            if (options.write_output) {
                writePngPair(options.output, index, frame);
                metadata << index << ',' << frame.left_frame_number << ','
                         << frame.right_frame_number << ',' << std::setprecision(17)
                         << frame.left_timestamp_ms << ',' << frame.right_timestamp_ms << ','
                         << timestamp_delta_ms << '\n';
            }
        }

        std::cout << "validated_pairs: " << options.frames
                  << ", synchronization_mismatches: " << mismatched_pairs << "\n";
        return mismatched_pairs == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "ffs_offline_validate: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

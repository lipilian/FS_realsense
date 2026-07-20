# D455 Fast-FoundationStereo Viewer — Implementation Plan

## 1. Goal and Assumptions

Build a native Linux C++/CMake application that captures synchronized left and right infrared streams from an Intel RealSense D455, runs Fast-FoundationStereo (FFS) through TensorRT, and displays the stereo pair, disparity, and an interactive 3D point cloud in real time.

Initial target: Ubuntu, an NVIDIA RTX-class GPU, CUDA 13.2, TensorRT 10.13.2 or newer, NVIDIA driver r580 or newer, and the D455's maximum 1280×800 Y8 infrared stereo pair at 30 FPS. FFS receives the native fixed 1280×800 input directly because both dimensions are divisible by 32. Monochrome Y8 input will be replicated to three channels for FFS. FFS-derived depth will drive the point cloud; D455 hardware depth is an optional diagnostic reference and has a separate 1280×720 profile.

## 2. Technology Stack

- C++20 and CMake 3.24+
- `librealsense2` for device discovery, synchronized capture, calibration, and camera controls
- Official FFS C++ TensorRT single-engine backend, pinned to a known revision
- CUDA for preprocessing, depth conversion, and point generation
- OpenCV for rectification, validation overlays, and disparity color mapping
- GLFW, OpenGL, and Dear ImGui for a dockable RealSense Viewer-style interface
- Catch2 for unit and integration tests

Model weights and generated TensorRT engines must remain outside version control. Document expected locations and checksums. Review the NVIDIA model and source licenses before redistribution.

## 3. Proposed Repository Structure

```text
CMakeLists.txt                 Top-level options and targets
cmake/                         Dependency discovery and TensorRT helpers
include/ffs_viewer/            Interfaces and shared frame types
src/camera/                    D455 capture and calibration
src/inference/                 FFS adapter and CUDA preprocessing
src/geometry/                  Depth and point-cloud generation
src/ui/                        ImGui panels and OpenGL rendering
apps/viewer/main.cpp           Application composition and main loop
assets/shaders/                OpenGL shaders
config/default.yaml            Stream, model, and display defaults
tests/{unit,integration}/      Geometry and pipeline tests
third_party/                   Pinned external source dependencies
models/README.md               Engine generation and installation
```

Keep acquisition, inference, geometry, and rendering behind small interfaces. This allows recorded stereo input and alternate inference backends without changing the UI.

## 4. Runtime Architecture

```text
D455 capture thread
  -> latest-only synchronized stereo buffer
  -> CUDA preprocessing and TensorRT inference worker
  -> disparity, depth, and point-cloud buffer
  -> UI/OpenGL render thread
```

Use bounded queues of two or three frames and discard stale work. Each result carries its source frame number, hardware timestamp, calibration snapshot, and timing measurements. The render loop never waits for capture or inference; it reuses the latest complete result. Start with one CUDA stream and use CUDA events for profiling.

## 5. Capture and Calibration

1. Select the D455 by serial number and request IR stream indices 1 and 2 at 1280×800/Y8/30 FPS. This is the maximum native stereo/IR profile on the target D455. Its separate hardware-depth profile tops out at 1280×720.
2. Read both intrinsics and the right-to-left extrinsics at startup; derive the physical baseline from the translation.
3. Determine whether the selected profiles are already rectified. If necessary, create cached OpenCV stereo-rectification maps. Provide an epipolar-overlay mode to detect incorrect or duplicate rectification.
4. Expose emitter, laser power, exposure, gain, and auto-exposure in the UI.
5. Detect frame-number or timestamp mismatches rather than passing unsynchronized images to inference.

## 6. FFS Inference

1. Export a fixed-shape 1280×800 ONNX model in the upstream CUDA 12.4/PyTorch environment, then build the FP16 TensorRT engine and official FFS plugin in the CUDA 13.2 deployment environment. Use the native IR dimensions directly; no resize, padding, or crop is required.
2. Convert Y8 images into normalized planar RGB on the GPU using the upstream channel mean and standard deviation.
3. Begin with `max_disp=192` and `valid_iters=4`. Benchmark this full-resolution mode before changing those values: FFS recommends widths below 1000 pixels, so 1280-pixel input may not maintain 30 FPS on every RTX GPU. Later, ship several prebuilt engine profiles instead of rebuilding engines from UI settings.
4. Warm up the engine before publishing results.
5. Validate left/right orientation, output dimensions, and disparity units using a known offline stereo pair before enabling live inference.

## 7. Depth and Point Cloud

For every valid disparity `d`, calculate:

```text
Z = fx * baseline / d
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Scale the rectified intrinsics whenever input is resized or cropped. Reject non-finite and low disparities and points outside configurable near/far limits. Generate XYZ and point colors on the GPU, then upload to an OpenGL vertex buffer. Add CUDA–OpenGL interop only if profiling shows the upload is significant. Optional filtering should be introduced only after the unfiltered geometry is validated.

## 8. UI Scope

The default dock layout contains the left IR image, right IR image, colorized disparity, and 3D viewport. The cloud view supports orbit, pan, zoom, reset, point-size control, depth clipping, and coloring by IR, depth, or disparity.

A sidebar provides device and model controls, pause/resume, snapshot export, and live capture/inference/render FPS plus end-to-end latency. Camera disconnection and incompatible engine dimensions should appear in a persistent status bar instead of terminating the application.

Recording and playback are phase-two features, but the camera interface should support a recorded source from the beginning.

## 9. Delivery Phases

1. **FFS ONNX/TensorRT test:** Export an FFS ONNX model at the designed 1280×800 input size; build the FP16 TensorRT engine and FFS plugin in the CUDA 13.2 environment; run warm-up and timed inference on rectified stereo pairs. Record engine build success, engine file size, GPU memory use, mean/p50/p95 inference latency, throughput, and output validity. Gate: a valid 1280×800 disparity map and a documented performance baseline before viewer integration.
2. **D455 feasibility:** Capture synchronized D455 IR frames, confirm the 1280×800 stereo profile, check the GPU/driver/toolchain, and confirm license constraints. Repeat the Step 1 benchmark using recorded D455 pairs.
3. **CMake skeleton and camera:** Add dependency checks, interfaces, calibration, rectification diagnostics, and a recorded-image source.
4. **Inference:** Integrate TensorRT, GPU preprocessing, engine warm-up, bounded queues, and disparity visualization.
5. **Geometry and 3D:** Add metric depth, validity masks, point generation, and the interactive renderer.
6. **Viewer polish:** Add docking, settings, snapshots/PLY export, reconnect recovery, metrics, and actionable errors.
7. **Hardening:** Profile transfers and stalls, run automated and hardware tests, document setup, and optionally create a reproducible Docker development image.

## 10. Verification and Acceptance

- Unit tests cover disparity-to-depth conversion, resized intrinsics, invalid disparities, calibration conversion, and queue/drop behavior.
- Integration tests replay fixed stereo pairs and compare output statistics with checked-in golden metadata.
- Hardware tests cover start/stop, disconnect/reconnect, exposure changes, and a 30-minute stability run.
- At the selected native 1280×800 capture and inference resolution, measure sustained capture, inference, and visualization FPS on the reference GPU. A 30 FPS end-to-end rate is the target, not an assumption; retain bounded latency, no steady memory growth, and responsive UI controls even if stale frames must be dropped.
- Validate metric scale and orientation using planar targets and known distances. Set numerical tolerances after the feasibility measurements.

## 11. Decisions Required Before Implementation

1. Exact NVIDIA GPU, OS, TensorRT version, and whether Jetson support is required. CUDA 13.2 is the selected deployment toolkit.
2. Whether FFS source and weights may be redistributed or users must obtain them separately.
3. Whether 30 FPS remains required if full-resolution 1280×800 capture and FFS inference cannot meet it on the selected GPU; no lower-resolution fallback is planned unless approved.
4. Whether version one requires recording/playback, PLY export, RGB texturing, or live visualization only.

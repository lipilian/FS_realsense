# D455 Stereo Viewer — Current Status and Implementation Plan

> Last updated: 2026-08-03
> This document uses the currently working application as its baseline. Completed work is no longer listed as future implementation work.

## 1. Final Goal

The application has two processing pipelines with different purposes:

1. **Live preview pipeline:** Helps the user quickly adjust the D455 capture angle. Its priorities are stability, bounded latency, and a responsive UI.
2. **Final capture pipeline:** After the user confirms the angle and presses the final capture button, the application runs one high-quality FoundationStereo inference on a frozen stereo pair. It generates the final disparity map and point cloud. The user then draws a mask on the left image, and the points selected by that mask are converted into a Mesh and shown in the 3D view.

Complete user workflow:

```text
Live preview
  -> User adjusts the capture angle
  -> User presses "Final Capture"
  -> Freeze one synchronized 1280×800 stereo pair and its calibration
  -> Preprocess it to 960×608
  -> Run one high-quality FoundationStereo inference
  -> Generate the final disparity map and organized point cloud
  -> User draws or edits a mask on the left image
  -> Select valid 3D points inside the mask
  -> Generate and display a Mesh
  -> Accept the result or retake the image
```

## 2. Current Completed Work

### 2.1 Live Preview

- [x] Capture synchronized `1280×800` left and right images from the RealSense D455.
- [x] Run a fixed-shape Fast-FoundationStereo (FFS) model for live inference.
- [x] Keep the live processing pipeline stable at **15 Hz**.
- [x] Compute and display the disparity map in real time.
- [x] Generate a 3D point cloud from disparity and D455 calibration in real time.
- [x] Display and interact with the live point cloud in the UI.

This pipeline already satisfies the preview requirement before final capture. Except for blocking bug fixes, later work should not sacrifice the stable 15 Hz preview rate for higher preview quality.

### 2.2 Final High-Quality Model

- [x] Build a fixed-input `960×608` high-quality FoundationStereo (FS) model.
- [x] Reserve this model for single-frame processing after final capture; it does not need to run in real time.
- [x] Preload the FS TensorRT engine, context, CUDA buffers, and stream when the viewer starts; release them when it closes.
- [x] Integrate FS into the viewer's `Capture` workflow on a frozen stereo snapshot.
- [x] Mark pixels where `x_right = x_left - disparity < 0` as invalid (`-1`) and render them black in the final disparity view.
- [x] Generate and denoise the final organized point cloud on CUDA with a voxel-hash radius filter.
- [x] Send the final cloud directly from CUDA to an OpenGL VBO; final display does not copy XYZ/validity data through the CPU or use `point_step`.
- [x] Measure single-inference latency, GPU memory use, and output validity through the C++/TensorRT path.

## 3. Processing Pipelines

### 3.1 Live Preview Pipeline

```text
D455 1280×800 synchronized stereo
  -> FFS 1280×800 fast inference
  -> preview disparity
  -> preview point cloud
  -> UI at a fixed 15 Hz
```

Requirements:

- Use a latest-only or bounded buffer and drop stale frames instead of accumulating latency.
- Capture, inference, and UI rendering must not block each other for long periods.
- Every preview result must carry its source frame number, timestamp, and calibration.
- The preview point cloud is only for framing and angle selection. It is not an input to the final Mesh.

### 3.2 Final Capture Pipeline

```text
final synchronized stereo snapshot (1280×800)
  -> resize to 960×600
  -> add 8 rows of padding to produce 960×608
  -> high-quality FS inference
  -> remove or ignore padded rows
  -> CUDA invalid-disparity handling (`-1`) + validity map
  -> CUDA organized XYZ point cloud + voxel-hash denoising
  -> CUDA-to-OpenGL final point-cloud display
  -> apply user mask
  -> mesh generation
  -> 3D mesh display
```

Requirements:

- Always process the same synchronized pair that was frozen when the user pressed the button. Never mix in a newer live frame.
- The snapshot must contain the original left and right images, intrinsics, extrinsics or baseline, frame number, and timestamp.
- Final inference must run in a worker so that the UI remains responsive and can show `processing`, `error`, and `ready` states.
- In the first version, pause live FFS inference while final inference is running so the two models do not compete for GPU execution and memory. Resume it when the user returns to live preview.
- The processing status must visibly progress through `Capture queued`, `Running FS inference...`, `Denoising final point cloud...`, and `Capture complete`.
- The 2D final disparity is retained on the CPU for display; the final XYZ, validity, and grayscale point color remain on the GPU for 3D rendering.
- Final disparity, point cloud, mask, and Mesh must all be associated with the same snapshot ID.

## 4. 960×608 Image and Coordinate Mapping

The original `1280×800` image and the `960×600` content area have the same aspect ratio:

```text
scale_x = scale_y = 960 / 1280 = 600 / 800 = 0.75
```

Add eight pixels of vertical padding to produce the model's `960×608` input. The implementation must define and consistently use:

```text
pad_top + pad_bottom = 8
```

The recommended approach is to generate the final point cloud on the organized `960×600` grid after removing the padding. Use scaled intrinsics:

```text
fx' = 0.75 * fx
fy' = 0.75 * fy
cx' = 0.75 * cx
cy' = 0.75 * cy
```

If geometry is calculated in the padded `960×608` coordinate system, use `cy_padded = 0.75 * cy + pad_top`. The physical baseline does not change when the image is resized. Calculate depth and point coordinates with:

```text
Z = fx' * baseline / disparity
X = (u - cx') * Z / fx'
Y = (v - cy') * Z / fy'
```

Store the user's binary mask at the original left-image resolution of `1280×800`. Before generating the Mesh, map the mask to the unpadded `960×600` point-cloud grid using nearest-neighbor sampling. Do not use linear interpolation for a binary mask.

Offline tests must verify that:

- Resize and padding behavior exactly match the preprocessing used to build the FS model.
- Left and right images are in the correct order, and padded rows never enter the point cloud.
- Disparity is measured in pixels in the model-input coordinate system.
- Scaled intrinsics and disparity use the same coordinate system.
- A recognizable 2D mask region selects the expected 3D region.

## 5. UI State Machine

```text
LivePreview
  -> SnapshotCaptured
  -> FinalProcessing
  -> MaskEditing
  -> MeshReady

FinalProcessing -> ProcessingError
MaskEditing/MeshReady/ProcessingError -> LivePreview (Retake)
```

- **LivePreview:** Show the live stereo pair, preview disparity, and preview point cloud. Final capture is available.
- **SnapshotCaptured:** Lock the final stereo images and calibration so live frames cannot overwrite them.
- **FinalProcessing:** Run FS inference and calculate final disparity and the final point cloud.
- **MaskEditing:** Draw or erase the mask on the frozen left image and display a translucent overlay.
- **MeshReady:** Display the Mesh generated from the current mask. Regenerate it after mask changes.
- **ProcessingError:** Preserve the snapshot and error information and allow retry or retake.

Disable the final capture button during `FinalProcessing` so multiple high-quality jobs cannot be started at the same time.

## 6. Mask and Mesh

### 6.1 First Mask Version

- Brush, eraser, and brush-size controls.
- Clear and undo actions.
- Translucent mask overlay.
- Mask pixel count and corresponding valid 3D point count.
- A single-channel binary mask stored at the original `1280×800` left-image resolution.

A point may enter the Mesh only when all of the following conditions are true:

```text
inside user mask
AND valid disparity
AND finite XYZ
AND inside the configured near/far depth range
```

### 6.2 First Mesh Version

Prefer generating triangles from the 2D neighborhood structure of the organized point cloud:

- Generate at most two triangles for each adjacent `2×2` pixel cell.
- Create a triangle only when all three vertices are valid and inside the mask.
- Reject triangles with excessive depth discontinuities or long 3D edges so geometry does not bridge foreground and background surfaces.
- Use consistent triangle winding and calculate normals.
- Allow Mesh regeneration when the mask or rejection thresholds change.

After the organized-grid approach is validated, decide whether smoothing, hole filling, outlier filtering, or Poisson reconstruction is necessary.

## 7. Implementation Order

### Phase A — High-Quality Model C++ Integration

- [x] Add a separate FS engine adapter for `models/fs_960x608_iters32`.
- [x] Implement preprocessing that exactly matches model construction: `1280×800 -> 960×600 -> 960×608`.

- [x] Export and inspect raw disparity, colorized disparity, and valid-pixel statistics.
- [x] Record warmed-up inference latency and GPU memory use.

**Gate A:** C++/TensorRT consistently produces a `960×608` disparity map with correct dimensions, orientation, and reasonable values.

### Phase B — Final Capture and Final Point Cloud

- [x] Replace `Stop` with `Capture` and prevent concurrent capture jobs.
- [x] Atomically freeze the latest synchronized stereo pair and calibration snapshot.
- [x] Run high-quality FS inference through the existing pipeline worker.
- [x] Remove padding, invalidate invisible pixels, and generate an organized `960×600` cloud using scaled intrinsics in CUDA constant memory.
- [x] Apply CUDA voxel-hash radius denoising (3 cm radius; retain points with at least 30 neighbors).
- [x] Replace the disparity display with the final FS result and show final-processing status.
- [x] Upload the full final cloud to OpenGL directly from CUDA; use left IR grayscale as the current point color.
- [x] Perform a D455/GUI runtime test of CUDA–OpenGL interop, point-cloud scale/orientation, and final-cloud coloring.
- [x] Support processing errors, retry, and retake.

**Gate B:** One final capture always generates disparity and point-cloud results from one snapshot. The UI remains responsive; CUDA–OpenGL interop, metric scale, and orientation are confirmed on real hardware.

### Phase C — Left-Image Mask Editing

- [x] Add mask editing tools and a mask overlay to the frozen `1280×800` left image.
- [x] Map the mask to the `960×600` point-cloud grid with nearest-neighbor sampling.
- [x] Extract and highlight valid masked points in the 3D view.
- [x] Clear the old mask, final point cloud, and Mesh after retake.

**Gate C:** A selection in the left image matches the selected 3D points without visible scaling, padding, or vertical-flip errors.

### Phase D — Mesh Generation and Display

- [x] Generate triangle indices from the masked organized point cloud.
- [x] Add depth-discontinuity and maximum-edge-length rejection.
- [x] Calculate normals and add a Mesh rendering mode.
- [x] Regenerate the Mesh after mask changes.
- [x] Allow switching between point cloud, masked points, and Mesh views.

**Gate D:** The Mesh covers only the selected object region, does not bridge obvious depth discontinuities, and remains stable during 3D interaction.

### Phase E — Persistence, Stability, and UX

- [x] Save a reproducible final-capture package containing stereo images, calibration, disparity, mask, point cloud or Mesh, and metadata.
- [x] Add PLY or OBJ export if required by the final use case.
- [x] Handle camera disconnection, model loading failure, GPU out-of-memory errors, cancellation, and retry.
- [x] Test continuous preview, repeated final captures, and long-running stability.
- [x] Update `readme.md` with build, model placement, and runtime instructions.

## 8. Acceptance Criteria

### Live Preview

- Use synchronized `1280×800` stereo input and maintain the FFS pipeline at **15 Hz**.
- Keep latency bounded without accumulating stale frames.
- Display disparity, point cloud, and left image from the same source frame.

### Final Processing

- Use a fixed `960×608` FS input with padding that matches the model configuration.
- Each click processes only the frozen snapshot and does not retrigger on new live frames.
- Keep the UI responsive during high-quality inference.
- Validate point-cloud metric scale, horizontal and vertical orientation, and camera coordinate direction using a target with known dimensions and distance.
- Produce repeatable results from the same snapshot and configuration.

### Mask and Mesh

- Align the mask overlay with the original left image.
- Verify mask-to-`960×600` point-cloud mapping at image boundaries and recognizable feature points.
- Exclude invalid disparity, padding, and out-of-range depth values from the Mesh.
- Prevent the Mesh from bridging depth discontinuities or edges beyond configured thresholds.
- Ensure an old snapshot, mask, point cloud, or Mesh cannot leak into a new task after retake.

## 9. Decisions Still to Confirm

The following decisions do not block Phase A:

1. Whether the eight padding pixels are split equally between the top and bottom or placed entirely on one side. This must match the preprocessing used during FS model export.
2. Measured TensorRT inference time and whether the UI needs percentage progress or only the current stage and elapsed time.
3. Whether the first mask version also needs polygon selection, magic wand, or automatic segmentation.
4. Whether the Mesh is primarily for display, measurement, or manufacturing/export. This determines the priority of smoothing, hole filling, and OBJ/PLY/STL support.
5. Whether live preview should continue during final inference. The current default is to pause preview inference to prioritize GPU memory and final-processing stability.

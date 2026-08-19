# After Algorithm Evaluation Plan

**Start date:** August 21, 2026
**Owner:** Liu Hong
**Goal:** Deploy hierarchical FoundationStereo inference with TensorRT/CUDA and select the deployment GPU after end-to-end evaluation.

## Scope and Constraints

- Evaluate only one stereo-pair configuration using two SMI cameras with IMX265 sensors.
- Liu Hong will focus on FoundationStereo deployment; other cameras, sensors, and stereo configurations are out of scope.
- The inference path follows the existing hierarchical design: coarse disparity → `init_disp` → full-resolution refined disparity.
- TensorRT builds and validation must disable/avoid TF32 to keep numerical behavior controlled.

## Schedule

| Phase | Dates | Work | Deliverable |
|---|---|---|---|
| 1. CUDA padding and preprocessing | Aug 21 – Aug 27 | Implement CUDA padding and required preprocessing. Convert each `2448×2048` rectified stereo pair to `2464×2048` by replicate-padding 8 pixels on both the left and right sides. | GPU-generated full-resolution input matching Python `InputPadder` behavior. |
| 2. Coarse TensorRT engine | Aug 28 – Sep 27 | Export and deploy the coarse FoundationStereo TensorRT engine; build the static GWC TensorRT add-on/plugin; disable TF32; complete the coarse inference path. | First coarse disparity at `1232×1024` externally. The engine uses padded `1248×1024` output, followed by removal of 8 pixels from both horizontal sides. |
| 3. Coarse-to-refine connection | Sep 28 – Oct 4 | Implement the CUDA connection between the engines: coarse-output unpadding, tensor remapping, upsampling, disparity scaling, clamping, and generation of the `512×616` `init_disp` required by the refine engine. | The refine engine receives an initial disparity consistent with Python `run_hierachical`. |
| 4. Refine TensorRT engine | Oct 5 – Nov 4 | Export, build, and deploy the full-resolution refine engine. Implement TensorRT inference using `2464×2048` left/right images and `init_disp`. | Padded full-resolution disparity refined from the coarse estimate. |
| 5. CUDA post-processing | Nov 5 – Nov 11 | Implement final CUDA disparity unpadding and disparity post-processing. | Final `2448×2048` disparity ready for downstream depth and point-cloud processing. |
| 6. End-to-end evaluation and GPU decision | Nov 12 – Nov 18 | Measure complete-pipeline latency, peak memory, stability, and deviation from the Python reference; select the final deployment GPU from the results. | Performance and memory report plus a documented GPU selection decision. |

## Acceptance Criteria

- Every CUDA resize, padding, unpadding, and tensor-remapping operation matches the Python hierarchical reference in shape, coordinates, and disparity scale.
- The coarse stage exposes `1232×1024` output; refine receives `512×616` `init_disp`; final output is `2448×2048`.
- The complete TensorRT pipeline runs without TF32, produces no NaN or Inf values, and is numerically compared with the Python reference.
- The final performance report includes coarse inference, coarse-to-refine processing, refine inference, post-processing, total latency, and both per-stage and end-to-end peak GPU memory.
- GPU selection is based on full-pipeline memory headroom, stability, and measured runtime.

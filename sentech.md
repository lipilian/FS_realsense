# Sentech Dual-Camera Work Log

## Current goal

Create a Sentech dual-camera app. Phase 1 verifies that two cameras can be opened simultaneously. It does not acquire images, create a UI, or depend on Intel RealSense.

## Completed work

- Kept the existing `live_viewer` app and did not modify its source code.
- Added `apps/sentech_dual_connect/main.cpp`, built as `ffs_sentech_dual_connect`.
- Uses the Sentech StApi SDK (default path: `/opt/sentech`) to open the first two available cameras.
- Prints the display name and serial number for each connected camera. A failed connection returns a non-zero exit code.
- Does not start acquisition, so the app only tests the connection path and does not change camera settings.
- Automatically sets `GENICAM_GENTL64_PATH` only when it is absent, allowing StApi to locate `libstgentl.cti`.
- Added the Sentech include paths, libraries, and runtime RPATH to CMake.
- The connection test can be built with the existing offline/RealSense build disabled, allowing hardware validation on a system without a CUDA compiler.

## Verified hardware result

```text
Camera 1: connected | name: STC-MCS500U3V(21LJ548) | serial: 21LJ548
Camera 2: connected | name: STC-MCS500U3V(21LJ530) | serial: 21LJ530
PASS: both Sentech cameras are connected.
```

Both cameras were opened successfully.

## Build and run

```bash
cmake -S . -B build/sentech -DFFS_VIEWER_BUILD_OFFLINE_VALIDATE=OFF
cmake --build build/sentech --target ffs_sentech_dual_connect
./build/sentech/ffs_sentech_dual_connect
```

If the SDK is not installed at `/opt/sentech`, configure with:

```bash
cmake -S . -B build/sentech \
  -DFFS_VIEWER_BUILD_OFFLINE_VALIDATE=OFF \
  -DFFS_SENTECH_ROOT=<Sentech SDK path>
```

## Next phase

This Sentech app will later add CUDA, stereo acquisition, inference, and point-cloud support. Its intended capabilities will match `live_viewer`, but its data source will be two Sentech cameras and it will not depend on `realsense2`.

That phase requires separating the reusable CUDA inference and point-cloud core from the current RealSense-bound data-source code.

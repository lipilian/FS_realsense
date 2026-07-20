# Fast FoundationStereo Runtime Dependency

This directory vendors only the Route A C++ inference runtime from
[NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo).
It contains the single-engine TensorRT wrapper, the `FFSGWCVolume` plugin, and
CUDA preprocessing/disparity-to-depth kernels. It intentionally excludes ONNX
export, engine building, Route B, demos, profiling, OpenCV, and generated files.

The code currently targets TensorRT 10 because the plugin implements
`IPluginV2DynamicExt`. TensorRT 11 removes that API. The engine and its matching
YAML file remain runtime assets and are not stored in this dependency.

## Use from the parent project

```cmake
add_subdirectory(third_party/fast_foundation_stereo_runtime)
target_link_libraries(your_app PRIVATE ffs::runtime)
```

Configure the parent for the target GPU and installed SDKs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
  -DFFS_TENSORRT_ROOT=/usr
```

Include `ffs_depth_single_tensorrt.hpp` and construct
`ffs_depth::FFSSingleEngineInference` with a directory containing exactly one
YAML configuration plus `fast_foundationstereo.engine`:

```cpp
ffs_depth::FFSSingleEngineInference inference("models/ffs_1280x800");
```

Input/output pointers are CUDA device pointers; call `sync()` before consuming
results on another stream or on the CPU. See `models/README.md` for the current
engine configuration, compatibility limits, and checksums.

## Provenance and license

The runtime sources were copied from the upstream `cpp/` directory and the
single-engine header was decoupled from the unused Route B class. Upstream's
license is preserved in `LICENSE.txt`; review it before distribution or
commercial use.

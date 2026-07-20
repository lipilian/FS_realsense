# Runtime Models

Runtime inference expects the Route A engine and matching export metadata at:

```text
models/ffs_1280x800/
├── fast_foundationstereo.engine
└── onnx.yaml
```

Create the inference wrapper with this directory:

```cpp
ffs_depth::FFSSingleEngineInference inference("models/ffs_1280x800");
```

The current engine was generated for a fixed `1x3x800x1280` input with
`max_disp=192`, `valid_iters=4`, and mixed precision enabled. It was built
with TensorRT 10.16.1, CUDA 13.2, and the local SM 120 GPU environment. TensorRT
engines are not portable across arbitrary TensorRT versions or GPU platforms;
rebuild the engine when the deployment environment changes.

## Checksums

```text
4c55e19c2179539754fa3175b3ace48c607d560e3b9bc295c431b999d9f6375b  fast_foundationstereo.engine
b93104a48b5a3a8a56fc1455c2f51315eb72a3503217851ef38966d41b754e6a  onnx.yaml
```

The generated engine is intentionally ignored by Git. Keep a backed-up copy
outside the repository if the ONNX exporter, source weights, and upstream FFS
checkout will be removed.

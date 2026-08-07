# Sentech 双目 App 工作记录

## 当前结构

- UI 源码：`apps/sentech_stereo/main.cpp`
- Sentech 采集 I/O：`include/ffs_viewer/io/sentech_stereo_source.hpp` 与 `src/io/sentech_stereo_source.cpp`
- ChArUco 检测：`include/ffs_viewer/calibration/live_charuco_detector.hpp` 与 `src/calibration/live_charuco_detector.cpp`
- CMake 目标与可执行文件：`sentech_stereo`
- 原有 `live_viewer` 保留，未删除或替换。
- Sentech app 不依赖 Intel RealSense 或 `realsense2`。

## 当前功能

`sentech_stereo` 是一个最小的双相机 ImGui viewer。

`Acquisition` 控制窗口包含三个基础按钮及一个显示/推理切换按钮：

- `Start`：扫描两台可用 Sentech 相机，按源码中写死的相机名称识别左右角色，启动两路数据流，并显示左右画面。
- `Stop`：停止两台相机及主机端的数据流。
- `Quit`：退出程序。
- `Show Rectified Images` / `Show Raw Images`：切换原始图和校正图；切到校正图时自动启动 live FFS disparity 推理，并打开 `Live Disparity` 与 `Live Point Cloud` 窗口。
- `Capture`：冻结当前 rectified pair，使用 `models/fs_608x512_iters32/fs.engine` 运行 32-iteration FoundationStereo，并执行 CUDA 点云投影和半径去噪。
- `Draw`：在冻结的 `608 × 512` 左图上绘制待测表面；绘制、mesh edge 和面积计算全部使用全分辨率，不做 downsample。

独立的 `Calibration` 控制窗口提供可编辑的 `Squares X/Y`、`Square length (m)`、`Marker length (m)` 和 ArUco dictionary（`4×4`、`5×5`、`6×6`、`7×7` 常用容量）下拉框；按 `Apply ChArUco Board` 才会验证、应用并写入当前运行目录的 `sentech_stereo_charuco.json`；下次从同一目录启动时会自动读取。默认参数为 `8 × 5` squares、`54 mm` square length、`40 mm` marker length、`DICT_4X4_250`。窗口还提供 `Live ChArUco Detection` / `Stop Live ChArUco Detection` 切换按钮。启用后对左右最新 BGR 帧实时检测，并在画面上叠加 marker 与 ChArUco 角点；检测计数显示在 Calibration 面板。本机 C++ OpenCV 为 4.6，因此检测流程使用 `detectMarkers()` 后调用 `interpolateCornersCharuco()`；这是 OpenCV 4 中完成内角点插值的标准 API。`Capture One Calibration Pair` 会收集之后的 5 个左右新帧组合，使用 Sentech `GetTimestampNS()` 的绝对差挑选最小者，然后在独立的 `Calibration Pair` 窗口显示静态左右图、timestamp 差及各自的检测结果；相同 ChArUco ID 的左右角点会以黄色连线标出。最近 20 组会保留在历史记录中，可在该窗口用 `Previous` / `Next` 切换，并可用 `Delete This Pair` 删除当前组。`Finish Calibration` 会使用与当前 board 参数相同的历史 pair，先分别标定左右单目内参与畸变，再固定内参做双目标定；RMS 和有效 pair 数会显示在 Calibration 面板。成功结果会写入当前运行目录的 `sentech_stereo_calibration.json`；下次启动若文件存在则自动加载该结果及对应 board 参数。

图像通过 Sentech StApi 的像素格式转换器统一转换为 `BGR8`，再上传到 OpenGL 纹理，在 `Stereo View` 面板并排显示。左右角色不依赖设备发现顺序，而是按相机名称固定匹配：`STC-MCS500U3V(21LJ530)` 为左相机，`STC-MCS500U3V(21LJ548)` 为右相机。匹配 Sentech 的用户定义名称或显示名称；未知或重复名称会使 `Start` 失败。两路帧均不做软件旋转，按相机原始方向进入 app，输出尺寸保持为 `2448 × 2048`。除连接、采集、ChArUco 标定、图像校正、live FFS disparity 推理和 FoundationStereo 最终点云处理外，当前 app 不修改相机设置或做硬件同步。

采集采用 latest-frame 缓冲：每台相机各有一个后台线程持续取流、转换并发布最新帧；UI 线程只读取每路最新完成帧。每路使用三个可复用 BGR 帧槽，ChArUco 检测或 UI 跟不上时会丢弃旧帧，而不会使显示延迟持续累积。该机制不对两台相机做时间戳配对，也不提供硬件同步保证。

## 已验证的硬件

连接测试已成功打开两台相机：

```text
Camera 1: STC-MCS500U3V(21LJ548) | serial: 21LJ548
Camera 2: STC-MCS500U3V(21LJ530) | serial: 21LJ530
PASS: both Sentech cameras are connected.
```

当前实际采集配置（两台相机相同）：

```text
Resolution: 2448 x 2048
Raw pixel format: BayerRG8
```

Sentech app 将原始 BayerRG8 转换为同分辨率的 BGR8，用于显示和后续处理。

Acquisition 区块提供单一切换按钮：默认显示原始左右 BGR 图像；点击 `Show Rectified Images` 后，使用已保存或刚完成的双目标定结果生成并显示 stereo-rectified 图像。没有可用标定 JSON / 标定结果时不会切换，并会在界面提示。切回 raw 不影响采集、标定 pair 的候选帧或标定历史。实时 ChArUco detection 则叠加在当前显示模式的图像上。

ImGui viewer 已成功编译。当前自动化环境没有 `xvfb-run`，因此没有自动点击 `Start` 来验证桌面窗口中的实时预览；请在有显示器的桌面会话中运行确认画面。

## Sentech SDK 运行时配置

默认 Sentech SDK 根目录为 `/opt/sentech`，可通过 CMake 变量 `FFS_SENTECH_ROOT` 修改。

左右相机名称直接写死在 `src/io/sentech_stereo_source.cpp`：

```text
Left:  STC-MCS500U3V(21LJ530)
Right: STC-MCS500U3V(21LJ548)
```

不使用 CMake 参数或缓存控制左右相机映射。若实际相机名称需要变更，直接修改 I/O 源码中的 `kLeftCameraName` 和 `kRightCameraName` 后重新编译。

程序仅在 `GENICAM_GENTL64_PATH` 未设置时自动指向 SDK 的 `lib` 目录，以便 StApi 找到 `libstgentl.cti`。

CMake 已为 `sentech_stereo`：

- 链接 `StApi_TL`、`StApi_IP`、GenICam 和 SDK 自带的 `turbojpeg`；
- 设置 `/opt/sentech/lib` 与 `/opt/sentech/lib/GenICam` 的 RPATH；
- 使 `libturbojpeg.so.0` 能在启动时解析，无需手动设置 `LD_LIBRARY_PATH`。

## 构建和运行

### 与其他 app 共用 `build` 目录

在具备 CUDA、RealSense 及现有 `live_viewer` 全部依赖的机器上：

```bash
cmake -S . -B build
cmake --build build --target sentech_stereo
./build/sentech_stereo
```

`cmake -S . -B build` 仅进行配置，不会编译或运行程序；项目始终要求 CUDA，并会同时检查 CUDA、RealSense、TensorRT 等原有依赖。

同一个目录也可构建其他可执行文件：

```bash
cmake --build build --target live_viewer live_infer offline_validate
```


## Sentech FFS engine：608 × 512（当前可用）

当前已成功生成并准备优先接入 Sentech app 的 FFS engine 为 `models/ffs_608x512`。图像处理链为：先完成 stereo rectification；两张图都从左、右边缘各裁去 8 px（`2448 × 2048` → `2432 × 2048`）；再等比例四分之一 downsample，得到 `608 × 512`。该尺寸宽高均可被 32 整除，比例与裁剪后的原图一致。

在 `Fast-FoundationStereo` 目录执行以下两条命令即可生成该 engine：

```bash
python scripts/make_plugin_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path ../models/ffs_608x512 \
  --height 512 \
  --width 608 \
  --valid_iters 4 \
  --max_disp 192

cpp/build/ffs_build_single_engine \
  ../models/ffs_608x512/fast_foundationstereo_plugin.onnx \
  ../models/ffs_608x512/fast_foundationstereo.engine
```

本机已成功生成 `models/ffs_608x512/fast_foundationstereo.engine`（约 51 MB）。运行时目录必须同时保留 `fast_foundationstereo.engine` 与唯一的 `onnx.yaml`。导出过程中的 PyTorch `TracerWarning`、legacy ONNX exporter 和 constant-folding warning 对这次固定尺寸 engine 属于预期提示。`max_disp=192` 目前沿用现有 engine；之后应根据 Sentech 在 `608 × 512` 下的最近工作距离及校正后最大视差重新评估。

`sentech_stereo` 会在用户点击 `Show Rectified Images` 时启动独立的 latest-frame FFS worker。每个 rectified pair 的左右图均去掉左、右边缘各 8 px，随后缩小为 `608 × 512` BGR8 并直接送入 FFS 的 RGB 路径（runtime 内部完成 BGR→RGB channel reorder）。worker 只保留最新 pending pair：当推理慢于采集时会丢掉旧 pair，不会令 live view 累积延迟。最新 disparity 以 Turbo 伪彩色显示在 `Live Disparity` 窗口。worker 会用 `stereoRectify` 输出的投影内参、基线、共同裁剪和下采样后的坐标系在 CUDA 上把 disparity 投影为彩色点云；每帧的 `608 × 512` GPU 顶点通过 CUDA–OpenGL interop 直接写入 live viewer 的 VBO，不再把 `xyz/rgb` 回传 CPU。无效点会在 GPU 上隐藏，并过滤 `0.1–10 m` 深度；disparity 伪彩图仍会回传 CPU 用于 ImGui。`Live Point Cloud` 支持左键拖动旋转、右键拖动平移、滚轮缩放；切回 raw 图会停止该 worker。

### FoundationStereo 最终 Capture / Draw

`models/fs_608x512_iters32/fs.engine` 的输入与输出都是 `1 × 3 × 512 × 608` / `1 × 1 × 512 × 608`。按 `Capture` 时，app 会冻结当前 rectified pair，并使用与 live FFS 相同的共同裁剪（左右图各裁左、右 8 px）和缩放，得到 `608 × 512` BGR8 输入。FoundationStereo 输入保持 BGR CHW 顺序，与其 OpenCV BGR 训练/示例预处理一致。

点击 `Capture` 后，冻结该 rectified pair，然后立即停止 live FFS worker 和两台 Sentech 相机的采集，避免最终 FS 推理与 live 推理争用 GPU。之后点击 `Start` 会恢复相机采集；如果仍处于 rectified 模式且已有 calibration，也会同时恢复 live FFS。FS 完成后复用 `live_viewer` 的 CUDA `FinalCloudProcessor`：依据 rectified projection 的 `fx`、`fy`、`cx`、`cy` 和 baseline 将 disparity 投影为三维点，过滤 `0.1–10 m`，并执行半径邻域去噪。最终 pair 直接显示在既有 `Stereo View`，最终点云、mesh edge 和最大连通表面积直接显示在既有 `Live Point Cloud`；不再创建 `Final Capture` 或 `Final Point Cloud` 窗口。`Draw` 打开的 mask editor 固定使用完整 `608 × 512`，并以深度连续阈值构建所选区域的三角 mesh。

### 1216 × 1024 实验记录

`1216 × 1024`（左右图共同横向裁去 16 px 后二分之一 downsample）也已生成一个约 57 MB 的 FP16 engine，但 TensorRT 10.16 默认 builder optimization level 3 会报 Myelin `No valid tactics` / `Must have costs`。本机只能通过 `--workspace-mb 8192 --optimization-level 0 --no-compilation-cache` 完成构建；该模式可能比默认优化的 engine 慢。因此当前优先使用正常构建成功的 `608 × 512` engine，`1216 × 1024` 保留为后续性能/精度对比选项。

## 后续 CUDA 阶段

Sentech app 后续会接入 CUDA、双目采集同步、立体匹配/推理和点云显示，目标功能与 `live_viewer` 对齐，但相机数据源始终使用两台 Sentech，不链接 `realsense2`。

实现该阶段时，需要将现有 CUDA 推理和点云核心从当前 RealSense 数据源部分拆出，使其可被 `sentech_stereo` 复用。\n
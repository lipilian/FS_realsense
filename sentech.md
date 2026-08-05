# Sentech 双目 App 工作记录

## 当前结构

- 源码目录：`apps/sentech_stereo`
- CMake 目标与可执行文件：`sentech_stereo`
- 原有 `live_viewer` 保留，未删除或替换。
- Sentech app 不依赖 Intel RealSense 或 `realsense2`。

## 当前功能

`sentech_stereo` 是一个最小的双相机 ImGui viewer。

控制窗口只有三个按钮：

- `Quit`：退出程序。
- `Start`：扫描两台可用 Sentech 相机，按源码中写死的相机名称识别左右角色，启动两路数据流，并显示左右画面。
- `Stop`：停止两台相机及主机端的数据流。

图像通过 Sentech StApi 的像素格式转换器统一转换为 `BGR8`，再上传到 OpenGL 纹理，在 `Stereo View` 面板并排显示。左右角色不依赖设备发现顺序，而是按相机名称固定匹配：`STC-MCS500U3V(21LJ530)` 为左相机，`STC-MCS500U3V(21LJ548)` 为右相机。匹配 Sentech 的用户定义名称或显示名称；未知或重复名称会使 `Start` 失败。左相机帧在进入 app 时用 OpenCV 顺时针旋转 90°；右相机将原有的 180° 与顺时针 90° 合并为一次逆时针旋转 90°（等价于顺时针 270°）。两路输出帧均为 `2048 × 2448`。除连接与采集外，当前 app 不修改相机设置、不做同步、标定、推理或点云处理。

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

Sentech app 将原始 BayerRG8 转换为同分辨率的 BGR8，用于旋转、显示和后续处理。

ImGui viewer 已成功编译。当前自动化环境没有 `xvfb-run`，因此没有自动点击 `Start` 来验证桌面窗口中的实时预览；请在有显示器的桌面会话中运行确认画面。

## Sentech SDK 运行时配置

默认 Sentech SDK 根目录为 `/opt/sentech`，可通过 CMake 变量 `FFS_SENTECH_ROOT` 修改。

左右相机名称直接写死在 `apps/sentech_stereo/main.cpp`：

```text
Left:  STC-MCS500U3V(21LJ530)
Right: STC-MCS500U3V(21LJ548)
```

不使用 CMake 参数或缓存控制左右相机映射。若实际相机名称需要变更，直接修改源码中的 `kLeftCameraName` 和 `kRightCameraName` 后重新编译。

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


## 后续 CUDA 阶段

Sentech app 后续会接入 CUDA、双目采集同步、立体匹配/推理和点云显示，目标功能与 `live_viewer` 对齐，但相机数据源始终使用两台 Sentech，不链接 `realsense2`。

实现该阶段时，需要将现有 CUDA 推理和点云核心从当前 RealSense 数据源部分拆出，使其可被 `sentech_stereo` 复用。

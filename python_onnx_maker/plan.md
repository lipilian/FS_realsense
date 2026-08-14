# Mesh Measurement 展示网站计划

## 目标

把约 60 次测量按每 10 次一组组织成约 6 组。每组记录 camera angle、center/off-center 模式和其他实验条件，让局域网用户能够：

- 浏览每组及组内 10 次 measurement。
- 查看 rectified left image、masked left image 和 disparity map。
- 旋转、缩放和平移带颜色的 3D mesh。
- 切换 surface、surface + wireframe 和 wireframe-only 显示。
- 查看单次面积及该组 10 次结果的稳定性统计。

网站只读取预先生成的静态结果，不在浏览时运行 FoundationStereo 或重新生成 mesh。

## 第一阶段：准备每次 measurement 的输出

为每次测量生成以下文件：

- `left.png`：rectified left image。
- `masked_left.png`：应用选区 mask 后的 rectified left image。
- `disparity.png`：用于网页展示的 disparity 彩色图。
- `mesh.ply`：带 RGB 顶点颜色的最终 CDT mesh。
- `metrics.json`：面积、顶点数、三角形数和实验条件。

所有 disparity 图必须使用统一的最小值、最大值和 colormap，保证不同 measurement 可以直接比较。三张图片必须尺寸一致，并处于同一个 rectified 坐标系。

建议目录结构：

```text
web_data/
├── manifest.json
├── group_01/
│   ├── run_01/
│   │   ├── left.png
│   │   ├── masked_left.png
│   │   ├── disparity.png
│   │   ├── mesh.ply
│   │   └── metrics.json
│   ├── run_02/
│   └── ...
├── group_02/
└── ...
```

`metrics.json` 建议格式：

```json
{
  "group": 1,
  "run": 1,
  "mode": "center",
  "camera_angle_deg": 0.0,
  "area_m2": 0.012345,
  "vertex_count": 125000,
  "triangle_count": 240000
}
```

如果 camera angle 不只是一个数值，可以改为：

```json
{
  "camera_yaw_deg": 0.0,
  "camera_pitch_deg": 10.0,
  "camera_roll_deg": 0.0
}
```

## 第二阶段：批量导出工具

把当前 `fs_high_resolution.ipynb` 中已经验证的流程整理成批处理入口，依次处理全部 measurement：

1. 加载一组 left/right image、mask 和 calibration。
2. Stereo rectification。
3. FoundationStereo inference。
4. GPU 深度过滤和去噪。
5. 使用 `downsample_pixels=4` 构建 CDT mesh。
6. 计算所有最终三角形的 3D 总面积。
7. 保存三张展示图片、彩色 PLY 和 `metrics.json`。
8. 释放本次 measurement 的中间 tensor，继续下一次。

模型只加载一次，所有 measurement 共用同一个进程，避免重复加载约 16 GB GPU 权重和缓存。

批处理完成后进行自动检查：

- 五个输出文件全部存在且非空。
- PLY 至少包含一个三角形。
- PLY 中不存在未被三角形引用的孤立顶点。
- 顶点、三角形和面积为有效有限值。
- RGB 顶点颜色存在。
- 图片尺寸一致。
- 失败项目写入日志，不阻止剩余项目继续处理。

## 第三阶段：生成总索引和组内统计

扫描全部 `metrics.json`，生成根目录下的 `manifest.json`。每个 group 计算：

- 10 次面积的 mean。
- min 和 max。
- range。
- standard deviation。
- coefficient of variation（CV）。
- 每次 measurement 相对组均值的百分比差异。

示例：

```json
{
  "groups": [
    {
      "id": "group_01",
      "mode": "center",
      "camera_angle_deg": 0.0,
      "area_mean_m2": 0.01234,
      "area_std_m2": 0.00003,
      "area_cv_percent": 0.24,
      "runs": [
        {
          "id": "run_01",
          "area_m2": 0.01235,
          "left": "group_01/run_01/left.png",
          "masked_left": "group_01/run_01/masked_left.png",
          "disparity": "group_01/run_01/disparity.png",
          "mesh": "group_01/run_01/mesh.ply"
        }
      ]
    }
  ]
}
```

## 第四阶段：网页 UI

建议技术栈：

- TypeScript + Vite。
- Three.js 加载和显示彩色 PLY。
- FastAPI 提供网站、`manifest.json`、图片和 PLY 静态文件。
- 固定数据阶段不使用数据库。

桌面布局：

```text
┌──────────────────────────────────────────────────────────────┐
│ Mesh Measurement Dashboard       Center / Off-center filter │
├───────────────┬──────────────────────────────────────────────┤
│ Group list    │ Left image │ Disparity │ Masked left        │
│ └ Run list    ├────────────────────────────┬─────────────────┤
│               │ Rotatable colored 3D mesh │ Measurements    │
│               │ Surface / Wireframe       │ Area / Δ / CV   │
│               ├────────────────────────────┴─────────────────┤
│               │ Group area scatter plot and tolerance range │
└───────────────┴──────────────────────────────────────────────┘
```

3D viewer 控件：

- 鼠标左键旋转。
- 滚轮缩放。
- 右键平移。
- `Surface`。
- `Surface + Wireframe`。
- `Wireframe only`。
- 顶点颜色开关。
- Reset camera。
- Fullscreen。

切换 measurement 时：

1. 立即更新文字和图片。
2. 显示 mesh loading 状态。
3. 按需加载当前约 6 MB 的 PLY。
4. 新 mesh 显示后释放旧 geometry 和 material 的 GPU 资源。
5. 不同时加载全部 60 个 PLY。

## 第五阶段：局域网部署

在保存全部数据的工作站上启动服务：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

同一局域网用户访问：

```text
http://<工作站局域网IP>:8000
```

部署检查：

- 工作站防火墙允许 TCP 8000 端口。
- 用户与工作站处于同一局域网/VLAN。
- 网页不允许通过路径参数读取 `web_data` 以外的文件。
- 如果数据敏感或网络不完全可信，增加登录认证或反向代理访问控制。

## 验收标准

- 页面能识别全部 group 和 measurement，没有缺失或重复。
- 点击任意 measurement 后三张图片与 mesh 属于同一次测量。
- 彩色 PLY 能正常旋转、缩放和平移。
- surface 和 wireframe 模式可以即时切换。
- 页面显示的面积和对应 `metrics.json` 一致。
- 每组正确显示 mean、range、standard deviation 和 CV。
- 只按需加载当前 PLY，连续切换数据时浏览器内存不会持续增长。
- 局域网内至少两台其他电脑可以同时访问和操作。

## 推荐实施顺序

1. 确认 60 次数据的 group/run 命名和实验元数据。
2. 固定 disparity 的统一可视化范围。
3. 完成单次 measurement 的五文件导出。
4. 批量处理全部数据并进行自动检查。
5. 生成 `manifest.json` 和组内统计。
6. 实现网页数据导航和三张图片显示。
7. 加入 Three.js 彩色 PLY viewer 和 wireframe 控件。
8. 加入面积统计图和容差展示。
9. 在局域网部署并做多用户测试。

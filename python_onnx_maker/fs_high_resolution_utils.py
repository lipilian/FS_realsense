"""Shared setup helpers for the high-resolution FoundationStereo notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "bootstrap_notebook",
    "build_constrained_mesh",
    "configure_capture",
    "load_and_rectify_capture",
    "load_rectify_and_select_mask",
    "load_foundation_stereo_model",
    "postprocess_disparity_gpu",
    "run_foundation_stereo_inference",
    "save_surface_point_cloud",
    "save_triangle_mesh",
    "show_rectified_selection",
]


@dataclass(frozen=True)
class InferenceResult:
    disparity: Any
    seconds: float
    memory_before_mib: dict[str, float]
    memory_after_mib: dict[str, float]


@dataclass(frozen=True)
class SurfaceResult:
    xyz_map: Any
    valid_mask: Any
    keep_mask: Any


@dataclass(frozen=True)
class MeshResult:
    vertices: Any
    triangles: Any
    colors: Any


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "FoundationStereo").is_dir():
            return candidate
    raise RuntimeError("请从 FS_realsense 或其子目录启动 notebook。")


def load_foundation_stereo_model(fs_root: Path, device: Any) -> tuple[Any, Path]:
    """Load the high-resolution FoundationStereo checkpoint onto ``device``."""
    import torch
    from omegaconf import OmegaConf

    checkpoint_path = fs_root / "weights" / "23-51-11" / "model_best_bp2.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    from core.foundation_stereo import FoundationStereo

    config = OmegaConf.load(checkpoint_path.parent / "cfg.yaml")
    if "vit_size" not in config:
        config.vit_size = "vitl"
    model = FoundationStereo(config)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del checkpoint
    torch.cuda.synchronize(device)
    return model, checkpoint_path


def _gpu_memory_mib(torch: Any, device: Any) -> dict[str, float]:
    return {
        "allocated": torch.cuda.memory_allocated(device) / 2**20,
        "reserved": torch.cuda.memory_reserved(device) / 2**20,
        "peak_allocated": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def _infer_disparity(
    model: Any, left_rgb: Any, right_rgb: Any, device: Any, valid_iters: int, hiera: int
) -> Any:
    import numpy as np
    import torch
    from core.utils.utils import InputPadder

    height, width = left_rgb.shape[:2]
    left = torch.from_numpy(np.ascontiguousarray(left_rgb)).to(device, dtype=torch.float32)
    right = torch.from_numpy(np.ascontiguousarray(right_rgb)).to(device, dtype=torch.float32)
    left, right = left.permute(2, 0, 1)[None], right.permute(2, 0, 1)[None]
    padder = InputPadder(left.shape, divis_by=32, force_square=False)
    left, right = padder.pad(left, right)
    with torch.inference_mode():
        if hiera:
            disparity = model.run_hierachical(
                left, right, iters=valid_iters, test_mode=True, small_ratio=0.5
            )
        else:
            disparity = model.forward(left, right, iters=valid_iters, test_mode=True)
    disparity = padder.unpad(disparity.float())
    return disparity.squeeze().reshape(height, width)


def run_foundation_stereo_inference(
    model: Any, left_rgb: Any, right_rgb: Any, device: Any, valid_iters: int,
    hiera: int, warmup_runs: int,
) -> InferenceResult:
    """Run warmup and one timed FoundationStereo inference on a rectified pair."""
    import time

    import torch

    for _ in range(warmup_runs):
        _infer_disparity(model, left_rgb, right_rgb, device, valid_iters, hiera)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = _gpu_memory_mib(torch, device)
    started_at = time.perf_counter()
    disparity = _infer_disparity(model, left_rgb, right_rgb, device, valid_iters, hiera)
    torch.cuda.synchronize(device)
    return InferenceResult(
        disparity=disparity,
        seconds=time.perf_counter() - started_at,
        memory_before_mib=memory_before,
        memory_after_mib=_gpu_memory_mib(torch, device),
    )


def _local_xyz_denoise(
    xyz_map: Any, valid_mask: Any, selected_mask: Any, max_neighbor_distance_m: float,
    min_neighbors: int, edge_min_neighbors: int,
) -> Any:
    """Keep pixels supported by nearby, geometrically consistent 3×3 XYZ neighbours."""
    import torch
    import torch.nn.functional as functional

    if min_neighbors <= 0:
        return valid_mask
    height, width = valid_mask.shape
    xyz_padded = functional.pad(xyz_map.permute(2, 0, 1), (1, 1, 1, 1))
    valid_padded = functional.pad(valid_mask, (1, 1, 1, 1), value=False)
    selected_padded = functional.pad(selected_mask, (1, 1, 1, 1), value=False)
    neighbor_count = valid_mask.new_zeros(valid_mask.shape, dtype=xyz_map.dtype)
    selected_neighbor_count = valid_mask.new_zeros(valid_mask.shape, dtype=xyz_map.dtype)
    for offset_y in range(3):
        for offset_x in range(3):
            if offset_y == 1 and offset_x == 1:
                continue
            neighbor_xyz = xyz_padded[:, offset_y:offset_y + height, offset_x:offset_x + width]
            neighbor_xyz = neighbor_xyz.permute(1, 2, 0)
            neighbor_valid = valid_padded[offset_y:offset_y + height, offset_x:offset_x + width]
            selected_neighbor = selected_padded[offset_y:offset_y + height, offset_x:offset_x + width]
            distance = (neighbor_xyz - xyz_map).square().sum(dim=-1).sqrt()
            neighbor_count += (neighbor_valid & (distance <= max_neighbor_distance_m)).to(xyz_map.dtype)
            selected_neighbor_count += selected_neighbor.to(xyz_map.dtype)

    # A mask interior has all eight selected neighbours. At its boundary, reduce
    # the support requirement so contours and thin selected structures survive.
    interior = selected_mask & (selected_neighbor_count == 8)
    edge_requirement = selected_neighbor_count.clamp(min=1, max=edge_min_neighbors)
    required_neighbors = torch.where(
        interior,
        torch.full_like(selected_neighbor_count, min_neighbors),
        edge_requirement,
    )
    return valid_mask & (neighbor_count >= required_neighbors)


def postprocess_disparity_gpu(
    disparity: Any, rectified_k: Any, baseline_m: float, selected_mask: Any, device: Any,
    z_far_m: float, remove_invisible: bool = True, denoise: bool = True,
    max_neighbor_distance_m: float = 0.01, min_neighbors: int = 3,
    edge_min_neighbors: int = 2,
) -> SurfaceResult:
    """Filter disparity, unproject to XYZ, and denoise entirely on CUDA."""
    import torch

    if disparity.ndim != 2 or disparity.device != device:
        raise ValueError("Disparity must be a 2D CUDA tensor on the requested device")
    if z_far_m <= 0 or max_neighbor_distance_m <= 0 or min_neighbors <= 0 or edge_min_neighbors <= 0:
        raise ValueError("Depth, distance, and neighbour-count thresholds must be positive")

    height, width = disparity.shape
    selected_mask = torch.as_tensor(selected_mask, device=device, dtype=torch.bool)
    if selected_mask.shape != disparity.shape:
        raise ValueError("Selection mask dimensions must match disparity")
    fx, fy, cx, cy = (float(rectified_k[0, 0]), float(rectified_k[1, 1]),
                      float(rectified_k[0, 2]), float(rectified_k[1, 2]))
    if fx <= 0 or fy <= 0 or baseline_m <= 0:
        raise ValueError("Rectified camera intrinsics and baseline must be positive")

    u_grid = torch.arange(width, device=device, dtype=disparity.dtype)[None, :]
    v_grid = torch.arange(height, device=device, dtype=disparity.dtype)[:, None]
    valid_mask = torch.isfinite(disparity) & (disparity > 0) & selected_mask
    if remove_invisible:
        valid_mask &= u_grid >= disparity
    safe_disparity = torch.where(valid_mask, disparity, torch.ones_like(disparity)) # 不好的dispairty 会被变成 1， 避免除零
    depth_m = fx * float(baseline_m) / safe_disparity 
    valid_mask &= torch.isfinite(depth_m) & (depth_m > 0) & (depth_m <= z_far_m)
    depth_m = torch.where(valid_mask, depth_m, torch.zeros_like(depth_m))
    xyz_map = torch.stack(
        ((u_grid - cx) * depth_m / fx, (v_grid - cy) * depth_m / fy, depth_m), dim=-1
    )
    keep_mask = valid_mask
    if denoise:
        keep_mask = _local_xyz_denoise(
            xyz_map, valid_mask, selected_mask, max_neighbor_distance_m, min_neighbors,
            edge_min_neighbors,
        )
    depth_m = torch.where(keep_mask, depth_m, torch.zeros_like(depth_m))
    xyz_map = torch.where(keep_mask[..., None], xyz_map, torch.zeros_like(xyz_map))
    return SurfaceResult(xyz_map=xyz_map, valid_mask=valid_mask, keep_mask=keep_mask)


def save_surface_point_cloud(
    surface: SurfaceResult, rectified_left_rgb: Any, output_path: Path
) -> int:
    """Transfer only retained vertices and colours to CPU, then write a PLY point cloud."""
    import open3d as o3d
    import torch

    colors = torch.as_tensor(rectified_left_rgb, device=surface.xyz_map.device, dtype=torch.float32)
    if colors.shape != surface.xyz_map.shape:
        raise ValueError("Rectified-left image dimensions must match the XYZ map")
    vertices = surface.xyz_map[surface.keep_mask].detach().cpu().numpy()
    colors = (colors[surface.keep_mask] / 255.0).detach().cpu().numpy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud):
        raise RuntimeError(f"Unable to write point cloud: {output_path}")
    return len(vertices)


def _contour_loop_on_retained_vertices(contour_uv, retained_tree):
    """Snap an annotation contour to retained XYZ pixels, preserving its cyclic order."""
    import numpy as np

    _, loop = retained_tree.query(contour_uv.astype(np.float64), k=1)
    loop = np.asarray(loop, dtype=np.int64)
    loop = loop[np.r_[True, loop[1:] != loop[:-1]]]
    if len(loop) > 1 and loop[0] == loop[-1]:
        loop = loop[:-1]
    if len(np.unique(loop)) < 3:
        return None
    # A repeated, non-adjacent retained pixel makes a non-simple PSLG loop. It
    # means the denoising removed too much of this component to constrain safely.
    if len(np.unique(loop)) != len(loop):
        return None
    return loop



def build_constrained_mesh(
    xyz_map, point_mask, selected_mask, rectified_left_rgb,
    downsample_pixels=2, max_edge_m=0.02, max_depth_jump_m=0.01,
):
    """Create a per-component constrained Delaunay mesh from retained XYZ points.

    Every exterior annotation-contour segment is submitted to Triangle as a PSLG
    constraint. Triangle's `p` mode performs constrained Delaunay triangulation;
    this cell then verifies that every constraint is present before applying the
    3D edge/depth rejection filters. Denoising holes are deliberately not PSLG
    loops, so nearby valid points may bridge them.
    """
    import cv2
    import numpy as np
    from scipy.spatial import cKDTree

    try:
        import triangle
    except ImportError as error:
        raise RuntimeError(
            "CDT requires the `triangle` package in the fs environment; run `pip install triangle`."
        ) from error

    if downsample_pixels < 1 or max_edge_m <= 0 or max_depth_jump_m <= 0:
        raise ValueError("downsample_pixels, max_edge_m, and max_depth_jump_m must be positive")

    xyz = xyz_map.detach().cpu().numpy().astype(np.float64, copy=False)
    retained = point_mask.detach().cpu().numpy().astype(bool, copy=False)
    selected = np.asarray(selected_mask, dtype=bool)
    colors = np.asarray(rectified_left_rgb, dtype=np.float64) / 255.0
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or retained.shape != xyz.shape[:2]:
        raise ValueError("xyz_map must be H×W×3 and point_mask must be H×W")
    if selected.shape != retained.shape or colors.shape != xyz.shape:
        raise ValueError("mask and rectified image dimensions must match xyz_map")

    component_count, component_labels = cv2.connectedComponents(
        selected.astype(np.uint8), connectivity=8
    )
    vertex_blocks, triangle_blocks, color_blocks = [], [], []
    vertex_offset = 0

    for component_id in range(1, component_count):
        component = component_labels == component_id
        active = retained & component
        rows, cols = np.nonzero(active)
        if len(rows) < 3:
            continue
        retained_uv = np.column_stack((cols, rows)).astype(np.float64)
        retained_xyz = xyz[rows, cols]
        retained_colors = colors[rows, cols]

        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour_uv = max(contours, key=cv2.contourArea).reshape(-1, 2)
        boundary = _contour_loop_on_retained_vertices(contour_uv, cKDTree(retained_uv))
        if boundary is None:
            print(f"Skipping component {component_id}: its retained boundary is not a simple loop")
            continue

        sampled = ((retained_uv[:, 0].astype(np.int64) % downsample_pixels == 0) &
                   (retained_uv[:, 1].astype(np.int64) % downsample_pixels == 0))
        sampled[boundary] = True
        selected_indices = np.flatnonzero(sampled)
        source_to_sampled = np.full(len(retained_uv), -1, dtype=np.int64)
        source_to_sampled[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)
        boundary = source_to_sampled[boundary]
        boundary = boundary[np.r_[True, boundary[1:] != boundary[:-1]]]
        if len(boundary) < 3:
            continue

        segments = np.column_stack((boundary, np.roll(boundary, -1))).astype(np.int32)
        cdt = triangle.triangulate(
            {"vertices": retained_uv[selected_indices], "segments": segments}, "pQ"
        )
        if "triangles" not in cdt:
            continue
        cdt_uv = np.asarray(cdt["vertices"], dtype=np.float64)
        distance, cdt_to_sampled = cKDTree(retained_uv[selected_indices]).query(cdt_uv, k=1)
        if np.any(distance > 1e-6):
            raise RuntimeError("CDT inserted a vertex without a retained XYZ sample")
        faces = cdt_to_sampled[np.asarray(cdt["triangles"], dtype=np.int64)]

        # Verify the hard-constraint contract before any optional 3D filtering.
        face_edges = np.sort(np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ), axis=1)
        triangulated_edges = {tuple(edge) for edge in face_edges}
        required_edges = {tuple(sorted(edge)) for edge in segments}
        if not required_edges.issubset(triangulated_edges):
            raise RuntimeError("CDT did not preserve every exterior contour segment")

        sampled_uv = retained_uv[selected_indices]
        sampled_xyz = retained_xyz[selected_indices]
        sampled_colors = retained_colors[selected_indices]
        centroid = sampled_uv[faces].mean(axis=1)
        centroid_cols = np.clip(np.rint(centroid[:, 0]).astype(np.int64), 0, selected.shape[1] - 1)
        centroid_rows = np.clip(np.rint(centroid[:, 1]).astype(np.int64), 0, selected.shape[0] - 1)
        face_xyz = sampled_xyz[faces]
        edge_lengths = np.linalg.norm(face_xyz - np.roll(face_xyz, -1, axis=1), axis=2)
        depth_jump = np.ptp(face_xyz[..., 2], axis=1)
        keep_faces = (
            (component_labels[centroid_rows, centroid_cols] == component_id) &
            (edge_lengths.max(axis=1) <= max_edge_m) &
            (depth_jump <= max_depth_jump_m)
        )
        faces = faces[keep_faces]
        if len(faces) == 0:
            continue

        used, compact_faces = np.unique(faces, return_inverse=True)
        vertex_blocks.append(sampled_xyz[used])
        triangle_blocks.append(compact_faces.reshape(-1, 3) + vertex_offset)
        color_blocks.append(sampled_colors[used])
        vertex_offset += len(used)

    if not triangle_blocks:
        raise RuntimeError("No mesh triangles survived; relax edge/depth thresholds or inspect the mask")
    return MeshResult(
        vertices=np.concatenate(vertex_blocks),
        triangles=np.concatenate(triangle_blocks),
        colors=np.concatenate(color_blocks),
    )

def save_triangle_mesh(mesh_result: MeshResult, output_path: Path) -> None:
    """Write an indexed, coloured triangle PLY."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_result.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_result.triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_result.colors)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"Unable to write triangle mesh: {output_path}")


def configure_capture(namespace: dict[str, Any], repo_root: Path, capture_name: str) -> Path:
    """Expose the paths and dimensions for one saved raw stereo capture."""
    capture_directory = repo_root / "data" / capture_name
    paths = {
        "LEFT_PATH": capture_directory / "left.png",
        "RIGHT_PATH": capture_directory / "right.png",
        "LEFT_MASK_PATH": capture_directory / "masks" / "left_mask.png",
        "CALIBRATION_PATH": capture_directory / "calibration.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件: {path}")
    namespace.update(
        {
            "CAPTURE_DIR": capture_directory,
            **paths,
            "EXPECTED_WIDTH": 2448,
            "EXPECTED_HEIGHT": 2048,
            "POINT_CLOUD_DIR": capture_directory,
        }
    )
    return capture_directory


def _json_matrix(item: dict[str, Any]) -> Any:
    import numpy as np

    return np.asarray(item["data"], dtype=np.float64).reshape(item["rows"], item["cols"])


def load_and_rectify_capture(
    namespace: dict[str, Any], left_path: Path, right_path: Path, calibration_path: Path,
    expected_width: int, expected_height: int,
) -> tuple[Any, Any, Any, float]:
    """Load one raw pair and rectify it using its saved OpenCV calibration."""
    import json

    import cv2
    import imageio.v3 as iio

    raw_left, raw_right = iio.imread(left_path), iio.imread(right_path)
    if raw_left.ndim != 3 or raw_left.shape[2] != 3 or raw_right.shape != raw_left.shape:
        raise ValueError(f"左右图必须是同尺寸 RGB，实际为 {raw_left.shape} / {raw_right.shape}")
    height, width = raw_left.shape[:2]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"期望 {expected_width}x{expected_height}，实际为 {width}x{height}")

    calibration = json.loads(Path(calibration_path).read_text())
    k1, k2 = _json_matrix(calibration["left_camera_matrix"]), _json_matrix(calibration["right_camera_matrix"])
    d1, d2 = _json_matrix(calibration["left_distortion"]), _json_matrix(calibration["right_distortion"])
    rotation = _json_matrix(calibration["right_to_left_rotation"])
    translation = _json_matrix(calibration["right_to_left_translation"])
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY
    )
    left_map = cv2.initUndistortRectifyMap(k1, d1, r1, p1, (width, height), cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(k2, d2, r2, p2, (width, height), cv2.CV_32FC1)
    rectified_left = cv2.remap(raw_left, *left_map, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(raw_right, *right_map, cv2.INTER_LINEAR)
    rectified_k = p1[:, :3].astype("float32")
    baseline_m = abs(float(p2[0, 3] / p2[0, 0]))
    if baseline_m <= 0:
        raise RuntimeError(f"无效 baseline: {baseline_m}")

    namespace.update({"raw_left": raw_left, "raw_right": raw_right, "left_map": left_map})
    return rectified_left, rectified_right, rectified_k, baseline_m


def show_rectified_pair(rectified_left: Any, rectified_right: Any) -> None:
    """Display the rectified stereo images side by side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, image, title in zip(
        axes, (rectified_left, rectified_right), ("Rectified left", "Rectified right")
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    plt.show()


def load_rectify_and_select_mask(
    namespace: dict[str, Any], left_path: Path, right_path: Path, mask_path: Path,
    calibration_path: Path, expected_width: int, expected_height: int,
) -> tuple[Any, Any, Any, float, Any]:
    """Rectify one stereo pair and its raw-left-aligned binary selection mask."""
    import cv2
    import imageio.v3 as iio
    import numpy as np

    rectified_left, rectified_right, rectified_k, baseline_m = load_and_rectify_capture(
        namespace, left_path, right_path, calibration_path, expected_width, expected_height
    )
    raw_left_mask = iio.imread(mask_path)
    if raw_left_mask.ndim == 3:
        raw_left_mask = cv2.cvtColor(raw_left_mask[..., :3], cv2.COLOR_RGB2GRAY)
    raw_left = namespace["raw_left"]
    if raw_left_mask.ndim != 2 or raw_left_mask.shape != raw_left.shape[:2]:
        raise ValueError(
            f"left_mask.png 必须与 left.png 对齐，实际为 {raw_left_mask.shape} / {raw_left.shape[:2]}"
        )
    raw_left_mask = (raw_left_mask > 0).astype(np.uint8)
    rectified_left_mask = cv2.remap(
        raw_left_mask, *namespace["left_map"], cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    namespace.update(
        {"raw_left_mask": raw_left_mask, "rectified_left_mask": rectified_left_mask}
    )
    return rectified_left, rectified_right, rectified_k, baseline_m, rectified_left_mask


def show_rectified_selection(
    rectified_left: Any, rectified_left_mask: Any, rectified_right: Any
) -> None:
    """Show the rectified pair with the raw-left selection transformed into its left view."""
    import matplotlib.pyplot as plt

    selected_left = rectified_left.copy()
    selected_left[~rectified_left_mask] = 0
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for axis, image, title in zip(
        axes,
        (rectified_left, selected_left, rectified_right),
        ("Rectified left", "Selected rectified left", "Rectified right"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    plt.show()


def bootstrap_notebook(namespace: dict[str, Any]) -> tuple[Path, Path, Any]:
    """Load notebook dependencies, configure CUDA, and expose shared names.

    ``namespace`` should normally be the notebook's ``globals()``. Keeping the
    imports here lets the notebook focus on data, inference, and results.
    """
    import json
    import sys
    import time

    import cv2
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np
    import open3d as o3d
    import torch
    from omegaconf import OmegaConf

    repo_root = _repository_root(Path.cwd().resolve())
    fs_root = repo_root / "FoundationStereo"
    if str(fs_root) not in sys.path:
        sys.path.insert(0, str(fs_root))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    from Utils import depth2xyzmap, toOpen3dCloud, vis_disparity

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU；请确认当前 kernel 是 conda 环境 fs。")
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    def gpu_memory_mib() -> dict[str, float]:
        return {
            "allocated": torch.cuda.memory_allocated(device) / 2**20,
            "reserved": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved": torch.cuda.max_memory_reserved(device) / 2**20,
        }

    namespace.update(
        {
            "Path": Path,
            "json": json,
            "time": time,
            "cv2": cv2,
            "iio": iio,
            "plt": plt,
            "np": np,
            "o3d": o3d,
            "torch": torch,
            "OmegaConf": OmegaConf,
            "FoundationStereo": FoundationStereo,
            "InputPadder": InputPadder,
            "depth2xyzmap": depth2xyzmap,
            "toOpen3dCloud": toOpen3dCloud,
            "vis_disparity": vis_disparity,
            "gpu_memory_mib": gpu_memory_mib,
            "load_foundation_stereo_model": load_foundation_stereo_model,
        }
    )
    return repo_root, fs_root, device

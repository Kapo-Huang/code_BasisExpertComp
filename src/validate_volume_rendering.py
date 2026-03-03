import argparse
import inspect
import importlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
import torch
from matplotlib.colors import ListedColormap

from inr.cli import build_model, load_config, resolve_data_paths
from inr.data import MultiTargetVolumetricDataset, VolumetricDataset
from inr.datasets.base import parse_volume_shape
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
from tqdm import tqdm as _tqdm

GLOBAL_VMIN = -1.0
GLOBAL_VMAX = 1.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Volume rendering for prediction/GT from checkpoint inference or npy files."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml (required).")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["checkpoint", "npy"],
        required=True,
        help="checkpoint: infer and render; npy: render provided npy files.",
    )

    # Common
    parser.add_argument("--start-time", type=int, default=0, help="Start timestep (inclusive).")
    parser.add_argument(
        "--end-time",
        type=int,
        default=-1,
        help="End timestep (exclusive). -1 means T.",
    )
    parser.add_argument("--outdir", type=str, default="validate_out/volume_rendering", help="Output image directory.")
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Render image width.",
    )
    parser.add_argument("--window-height", type=int, default=960, help="Render image height.")
    parser.add_argument("--background", type=str, default="white", help="Background color.")
    parser.add_argument(
        "--vertex-shader",
        type=str,
        default="shaders/vertexShader.glsl",
        help="Path to custom vertex shader source.",
    )
    parser.add_argument(
        "--fragment-shader",
        type=str,
        default="shaders/fragmentShader.glsl",
        help="Path to custom fragment shader source.",
    )
    parser.add_argument(
        "--tf-config",
        type=str,
        default="TF_H2.config",
        help="TransferFunction config path (.config JSON with alpha/colour control points).",
    )
    parser.add_argument("--clim-low-pct", type=float, default=1.0, help="Lower percentile for clim.")
    parser.add_argument("--clim-high-pct", type=float, default=99.9, help="Upper percentile for clim.")
    parser.add_argument(
        "--opacity-unit-distance",
        type=float,
        default=0.6,
        help="Volume opacity unit distance. Smaller=more foggy, larger=sharper.",
    )
    parser.add_argument("--camera-dist-scale", type=float, default=1.15, help="Camera distance = scale * bbox diagonal.")
    parser.add_argument("--zoom", type=float, default=1.35, help="Camera zoom factor.")
    parser.add_argument(
        "--up-rotate-deg",
        type=float,
        default=-120.0,
        help="Rotate camera up vector around view direction by this degree (Rodrigues, right-hand rule).",
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel index for multi-channel arrays/outputs.")

    # Checkpoint mode
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for --mode checkpoint.")
    parser.add_argument(
        "--attrs",
        type=str,
        default="",
        help="Comma-separated attr names. Required for multi-target checkpoint mode.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cpu/cuda/cuda:0.")
    parser.add_argument("--batch-size", type=int, default=32768, help="Inference batch size in checkpoint mode.")

    # NPY mode
    parser.add_argument(
        "--npy-files",
        type=str,
        nargs="*",
        default=[],
        help="Attr-path mapping list in format attr=path.npy (used in --mode npy).",
    )
    parser.add_argument(
        "--npy-tag",
        type=str,
        default="npy",
        help="Filename prefix tag for --mode npy.",
    )

    return parser.parse_args()


def _register_numpy_core_aliases():
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias_name, target_name in aliases.items():
        if alias_name in sys.modules:
            continue
        sys.modules[alias_name] = importlib.import_module(target_name)


def _torch_load_checkpoint(path: Path):
    # Some historical checkpoints were serialized with internal numpy module paths.
    # Register aliases before loading so torch.load can resolve them directly.
    _register_numpy_core_aliases()

    # `weights_only` exists in newer torch versions. We check signature once
    # instead of relying on exception-based control flow.
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    return torch.load(str(path), **load_kwargs)


def _load_target_stats(stats_path: str, attr_paths=None):
    data = np.load(stats_path, allow_pickle=True)
    if attr_paths:
        stats = {}
        for name in attr_paths.keys():
            mk = f"{name}__mean"
            sk = f"{name}__std"
            if mk not in data or sk not in data:
                raise KeyError(f"Missing keys '{mk}'/'{sk}' in {stats_path}")
            stats[name] = {"mean": data[mk], "std": data[sk]}
        return stats
    if "mean" not in data or "std" not in data:
        raise KeyError(f"Missing keys 'mean'/'std' in {stats_path}")
    return {"mean": data["mean"], "std": data["std"]}


def _build_dataset(cfg):
    data_cfg = cfg["data"]
    data_info = resolve_data_paths(data_cfg)
    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))

    target_stats = None
    stats_path = data_cfg.get("target_stats_path")
    if stats_path and Path(stats_path).exists():
        target_stats = _load_target_stats(stats_path, attr_paths=data_info.get("attr_paths"))

    if data_info.get("attr_paths"):
        dataset = MultiTargetVolumetricDataset(
            data_info["attr_paths"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
    else:
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
    return dataset


def _resolve_volume_shape(cfg, dataset=None):
    data_cfg = cfg.get("data", {})
    shape = parse_volume_shape(data_cfg.get("volume_shape") or data_cfg.get("volume_dims"))
    if shape is not None:
        return shape
    if dataset is not None and hasattr(dataset, "volume_shape"):
        return dataset.volume_shape
    raise ValueError("Cannot resolve volume shape from config. Please set data.volume_shape (X/Y/Z/T).")


def _select_times(start_t: int, end_t: int, T: int) -> List[int]:
    if T <= 0:
        raise ValueError("Invalid T<=0.")
    s = int(start_t)
    e = int(end_t) if int(end_t) >= 0 else int(T)
    if s < 0 or s >= T:
        raise ValueError(f"start-time out of range: {s}, valid [0, {T - 1}]")
    if e < 0 or e > T:
        raise ValueError(f"end-time out of range: {e}, valid [0, {T}]")
    if e <= s:
        raise ValueError(f"Invalid time range: start={s}, end={e}. Require end > start.")
    return list(range(s, e))


def _parse_attr_list(arg: str, available: List[str]) -> List[str]:
    if not arg or not arg.strip():
        return list(available)
    req = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in req if x not in available]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {available}")
    return req


def _parse_attr_path_pairs(items: List[str]) -> Dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --npy-files item '{item}'. Expected attr=path.npy")
        k, v = item.split("=", 1)
        name = k.strip()
        path = Path(v.strip())
        if not name:
            raise ValueError(f"Invalid attr name in '{item}'")
        if not path.exists():
            raise FileNotFoundError(f"NPY file not found for attr '{name}': {path}")
        out[name] = path
    if not out:
        raise ValueError("No valid npy files parsed.")
    return out


def _resolve_attr_stats(dataset, ckpt_payload, attr_name: str, out_dim: int):
    y_mean = ckpt_payload.get("y_mean") if isinstance(ckpt_payload, dict) else None
    y_std = ckpt_payload.get("y_std") if isinstance(ckpt_payload, dict) else None
    mean_arr = None
    std_arr = None

    if isinstance(y_mean, dict) and isinstance(y_std, dict):
        if attr_name in y_mean and attr_name in y_std:
            mean_arr = np.asarray(y_mean[attr_name]).reshape(-1)
            std_arr = np.asarray(y_std[attr_name]).reshape(-1)
    elif y_mean is not None and y_std is not None and attr_name == "targets":
        mean_arr = np.asarray(y_mean).reshape(-1)
        std_arr = np.asarray(y_std).reshape(-1)

    if mean_arr is None or std_arr is None:
        if isinstance(dataset, MultiTargetVolumetricDataset):
            dataset._ensure_target_stats(attr_name)
            mean_arr = dataset.y_mean[attr_name].reshape(-1).cpu().numpy()
            std_arr = dataset.y_std[attr_name].reshape(-1).cpu().numpy()
        else:
            dataset._ensure_target_stats()
            mean_arr = dataset.y_mean.reshape(-1).cpu().numpy()
            std_arr = dataset.y_std.reshape(-1).cpu().numpy()

    mean_arr = np.asarray(mean_arr, dtype=np.float32).reshape(-1)
    std_arr = np.asarray(std_arr, dtype=np.float32).reshape(-1)
    if mean_arr.size == 1 and out_dim > 1:
        mean_arr = np.full((out_dim,), float(mean_arr[0]), dtype=np.float32)
        std_arr = np.full((out_dim,), float(std_arr[0]), dtype=np.float32)
    if mean_arr.size != out_dim or std_arr.size != out_dim:
        raise ValueError(
            f"Denorm stats dim mismatch for attr '{attr_name}': "
            f"stats=({mean_arr.size},{std_arr.size}) vs out_dim={out_dim}"
        )
    return mean_arr, std_arr


def _extract_volume_from_array(arr: np.ndarray, shape, t_idx: int, channel: int) -> np.ndarray:
    X = int(shape.X)
    Y = int(shape.Y)
    Z = int(shape.Z)
    T = int(shape.T)
    V = X * Y * Z
    if t_idx < 0 or t_idx >= T:
        raise ValueError(f"t_idx out of range: {t_idx}, valid [0, {T - 1}]")

    if arr.ndim == 5:
        cdim = int(arr.shape[-1])
        if channel < 0 or channel >= cdim:
            raise ValueError(f"channel out of range: {channel}, valid [0, {cdim - 1}]")
        vol = np.asarray(arr[t_idx, :, :, :, channel], dtype=np.float32)
        return vol

    if arr.ndim == 4:
        if channel != 0:
            raise ValueError("4D scalar array only supports channel=0.")
        vol = np.asarray(arr[t_idx, :, :, :], dtype=np.float32)
        return vol

    start = t_idx * V
    end = (t_idx + 1) * V
    if arr.ndim == 2:
        cdim = int(arr.shape[1])
        if channel < 0 or channel >= cdim:
            raise ValueError(f"channel out of range: {channel}, valid [0, {cdim - 1}]")
        flat = np.asarray(arr[start:end, channel], dtype=np.float32)
        return flat.reshape(Z, Y, X)
    if arr.ndim == 1:
        if channel != 0:
            raise ValueError("1D scalar array only supports channel=0.")
        flat = np.asarray(arr[start:end], dtype=np.float32)
        return flat.reshape(Z, Y, X)
    raise ValueError(f"Unsupported npy ndim={arr.ndim}, expected 1/2/4/5.")


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _predict_volumes_for_timestep(
    model: torch.nn.Module,
    dataset,
    attrs: List[str],
    shape,
    t_idx: int,
    batch_size: int,
    channel: int,
    denorm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    # Coordinates are generated in the same flattened order as training data:
    # x changes fastest, then y, then z.
    X = int(shape.X)
    Y = int(shape.Y)
    Z = int(shape.Z)
    V = X * Y * Z

    x_mean = dataset.x_mean.reshape(-1).cpu().numpy().astype(np.float32)
    x_std = dataset.x_std.reshape(-1).cpu().numpy().astype(np.float32)

    out_flat = {name: np.empty((V,), dtype=np.float32) for name in attrs}

    # Different model implementations may or may not support `hard_topk`.
    # Detect once, then use a direct call path in the hot loop.
    supports_hard_topk = "hard_topk" in inspect.signature(model.forward).parameters

    model.eval()
    with torch.no_grad():
        for start in range(0, V, int(batch_size)):
            end = min(start + int(batch_size), V)
            v = np.arange(start, end, dtype=np.int64)
            x = (v % X).astype(np.float32)
            y = ((v // X) % Y).astype(np.float32)
            z = (v // (X * Y)).astype(np.float32)
            t = np.full_like(x, float(t_idx), dtype=np.float32)
            coords = np.stack([x, y, z, t], axis=1)
            coords = (coords - x_mean[None, :]) / x_std[None, :]
            xb = torch.from_numpy(coords).to(device, non_blocking=True)

            if supports_hard_topk:
                pred = model(xb, hard_topk=True)
            else:
                pred = model(xb)

            if isinstance(pred, dict):
                for name in attrs:
                    if name not in pred:
                        raise KeyError(f"Model output missing attr '{name}'. Available: {list(pred.keys())}")
                    arr = pred[name].detach().cpu().numpy()
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    cdim = int(arr.shape[1])
                    if channel < 0 or channel >= cdim:
                        raise ValueError(f"channel out of range for attr '{name}': {channel}, valid [0, {cdim - 1}]")
                    m, s = denorm_stats[name]
                    out_flat[name][start:end] = arr[:, channel] * float(s[channel]) + float(m[channel])
            else:
                if len(attrs) != 1:
                    raise ValueError("Single tensor model output cannot map to multiple attrs.")
                name = attrs[0]
                arr = pred.detach().cpu().numpy()
                if arr.ndim == 1:
                    arr = arr[:, None]
                cdim = int(arr.shape[1])
                if channel < 0 or channel >= cdim:
                    raise ValueError(f"channel out of range for attr '{name}': {channel}, valid [0, {cdim - 1}]")
                m, s = denorm_stats[name]
                out_flat[name][start:end] = arr[:, channel] * float(s[channel]) + float(m[channel])

    volumes = {name: out_flat[name].reshape(Z, Y, X) for name in attrs}
    return volumes


def _make_camera_views_from_grid(grid, dist_scale: float):
    # Camera distance is scaled by the bounding-box diagonal to keep a
    # comparable framing across datasets with different physical extents.
    bounds = np.asarray(grid.bounds, dtype=np.float64)
    center = tuple(np.asarray(grid.center, dtype=np.float64).tolist())
    dx = float(bounds[1] - bounds[0])
    dy = float(bounds[3] - bounds[2])
    dz = float(bounds[5] - bounds[4])
    diag = float(np.linalg.norm([dx, dy, dz]))
    if not np.isfinite(diag) or diag <= 0:
        diag = max(dx, dy, dz, 1.0)
    dist = max(1e-6, float(dist_scale) * diag)

    c = np.asarray(center, dtype=np.float64)
    # Keep only the validated main view requested by the user.
    main_dir = np.asarray([0.92, 0.74, 0.58], dtype=np.float64)
    main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-12)
    cam_pos = tuple((c + dist * main_dir).tolist())
    views = [("main", cam_pos, (0.0, 0.0, 1.0))]
    return center, views, diag


def _rotate_up_around_view(cam_pos, center, cam_up, angle_deg: float = -90.0):
    eps = 1e-12
    cam_pos_v = np.asarray(cam_pos, dtype=np.float64)
    center_v = np.asarray(center, dtype=np.float64)
    up0 = np.asarray(cam_up, dtype=np.float64)

    view = center_v - cam_pos_v
    n_view = float(np.linalg.norm(view))
    if not np.isfinite(n_view) or n_view <= eps:
        view_dir = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        view_dir = view / n_view

    n_up = float(np.linalg.norm(up0))
    if not np.isfinite(n_up) or n_up <= eps:
        up0 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        n_up = 1.0
    up0 = up0 / n_up

    # If up is almost parallel to view_dir, build a stable perpendicular up first.
    if abs(float(np.dot(up0, view_dir))) > 0.999:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, view_dir))) > 0.999:
            ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        up0 = ref - np.dot(ref, view_dir) * view_dir
        n_up0 = float(np.linalg.norm(up0))
        if n_up0 <= eps:
            up0 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up0 = up0 / n_up0

    theta = float(np.deg2rad(angle_deg))
    k = view_dir
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    # Rodrigues rotation: rotate up0 around axis k by theta.
    up1 = up0 * cos_t + np.cross(k, up0) * sin_t + k * np.dot(k, up0) * (1.0 - cos_t)

    # Re-orthogonalize and normalize.
    up1 = up1 - np.dot(up1, view_dir) * view_dir
    n_up1 = float(np.linalg.norm(up1))
    if not np.isfinite(n_up1) or n_up1 <= eps:
        up1 = up0
        up1 = up1 - np.dot(up1, view_dir) * view_dir
        n_up1 = float(np.linalg.norm(up1))
        if n_up1 <= eps:
            up1 = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up1 = up1 / n_up1
    else:
        up1 = up1 / n_up1

    return tuple(up1.tolist())


def _volume_to_grid(volume_zyx: np.ndarray):
    # volume_zyx: (Z, Y, X) -> VTK image data in (X, Y, Z)
    vol_xyz = np.transpose(np.asarray(volume_zyx, dtype=np.float32), (2, 1, 0))
    grid = pv.ImageData(dimensions=vol_xyz.shape)
    grid.point_data["values"] = np.ascontiguousarray(vol_xyz.ravel(order="F"))
    return grid


def _compute_clim(volume: np.ndarray, low_pct: float, high_pct: float):
    lo = float(np.nanpercentile(volume, low_pct))
    hi = float(np.nanpercentile(volume, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(volume))
        hi = float(np.nanmax(volume))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
    return [lo, hi]


def _dedupe_sorted_points(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Keep the last y when duplicated x exists.
    uniq_x: List[float] = []
    uniq_y: List[np.ndarray] = []
    for x, y in zip(xs, ys):
        if uniq_x and abs(float(x) - float(uniq_x[-1])) <= 1e-12:
            uniq_y[-1] = y
        else:
            uniq_x.append(float(x))
            uniq_y.append(y)
    return np.asarray(uniq_x, dtype=np.float64), np.asarray(uniq_y, dtype=np.float64)


def _parse_colour_value(raw) -> np.ndarray:
    if isinstance(raw, dict):
        rgb = [float(raw["r"]), float(raw["g"]), float(raw["b"])]
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
        rgb = [float(raw[0]), float(raw[1]), float(raw[2])]
    else:
        raise ValueError(f"Invalid colourValue format: {raw}")
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)


def _load_transfer_function_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"TransferFunction config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    alpha_points = cfg.get("alpha_control_points")
    colour_points = cfg.get("colour_control_points")
    if not isinstance(alpha_points, list) or len(alpha_points) < 2:
        raise ValueError("alpha_control_points must be a list with at least 2 points.")
    if not isinstance(colour_points, list) or len(colour_points) < 2:
        raise ValueError("colour_control_points must be a list with at least 2 points.")

    alpha_x = np.asarray([float(p["dataValue"]) for p in alpha_points], dtype=np.float64)
    alpha_y = np.asarray([float(p["alphaValue"]) for p in alpha_points], dtype=np.float64)
    colour_x = np.asarray([float(p["dataValue"]) for p in colour_points], dtype=np.float64)
    colour_y = np.asarray([_parse_colour_value(p["colourValue"]) for p in colour_points], dtype=np.float64)

    alpha_x = np.clip(alpha_x, 0.0, 1.0)
    alpha_y = np.clip(alpha_y, 0.0, 1.0)
    colour_x = np.clip(colour_x, 0.0, 1.0)
    colour_y = np.clip(colour_y, 0.0, 1.0)

    order_a = np.argsort(alpha_x, kind="mergesort")
    alpha_x, alpha_y = _dedupe_sorted_points(alpha_x[order_a], alpha_y[order_a])
    order_c = np.argsort(colour_x, kind="mergesort")
    colour_x, colour_y = _dedupe_sorted_points(colour_x[order_c], colour_y[order_c])

    if alpha_x.size < 2 or colour_x.size < 2:
        raise ValueError("Control points become invalid after sorting/deduplication.")

    logger.info(
        "Loaded TF config '%s': alpha_points=%d, colour_points=%d",
        path,
        int(alpha_x.size),
        int(colour_x.size),
    )
    return {
        "alpha_x": alpha_x,
        "alpha_y": alpha_y,
        "colour_x": colour_x,
        "colour_y": colour_y,
    }


def _build_transfer_function_luts(tf_cfg: Dict[str, np.ndarray], bins: int = 256):
    # Sample both alpha/color control points onto a fixed LUT. VTK then maps
    # data values linearly through clim -> [0, 1] -> LUT index.
    x = np.linspace(0.0, 1.0, int(bins), dtype=np.float64)
    alpha = np.interp(x, tf_cfg["alpha_x"], tf_cfg["alpha_y"])
    colour = np.column_stack(
        [
            np.interp(x, tf_cfg["colour_x"], tf_cfg["colour_y"][:, 0]),
            np.interp(x, tf_cfg["colour_x"], tf_cfg["colour_y"][:, 1]),
            np.interp(x, tf_cfg["colour_x"], tf_cfg["colour_y"][:, 2]),
        ]
    )
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    colour = np.clip(colour, 0.0, 1.0).astype(np.float32)
    # PyVista expects a matplotlib Colormap object for custom RGB LUTs.
    colour_cmap = ListedColormap(colour, name="tf_custom")
    return colour_cmap, alpha.tolist()


def _load_shader_source(path: Path, kind: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{kind} shader not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{kind} shader is empty: {path}")
    return text


def _extract_fragment_alpha_cutoff(fragment_source: str) -> float:
    # Expected patterns from provided shader:
    # if (accumulatedColor.a >= 0.95) break;
    # if (gl_FragColor.a < 0.05) discard;
    m = re.search(r"accumulatedColor\\.a\\s*>=\\s*([0-9]*\\.?[0-9]+)", fragment_source)
    if m is not None:
        val = float(m.group(1))
        return float(np.clip(val, 0.05, 0.999))
    return 0.95


def _extract_fragment_max_steps(fragment_source: str) -> int:
    m = re.search(r"MAX_STEPS\\s*=\\s*([0-9]+)", fragment_source)
    if m is not None:
        return max(16, int(m.group(1)))
    return 887


def _apply_custom_volume_shaders(volume_actor, vertex_source: str, fragment_source: str):
    # Adapt user-provided WebGL-style shaders onto VTK's volume ray-caster by
    # injecting logic at shader replacement points.
    required_vertex_tokens = ("v_nearpos", "v_farpos", "v_position")
    for token in required_vertex_tokens:
        if token not in vertex_source:
            raise ValueError(
                f"Vertex shader is missing expected token '{token}'. "
                "Please use shaders/vertexShader.glsl-compatible source."
            )
    if "cast_volume" not in fragment_source:
        raise ValueError(
            "Fragment shader is missing expected function 'cast_volume'. "
            "Please use shaders/fragmentShader.glsl-compatible source."
        )

    alpha_cutoff = _extract_fragment_alpha_cutoff(fragment_source)
    max_steps_hint = _extract_fragment_max_steps(fragment_source)
    logger.info(
        "Applying custom shader pipeline: alpha_cutoff=%.3f, max_steps_hint=%d",
        alpha_cutoff,
        max_steps_hint,
    )
    shader_property = volume_actor.GetShaderProperty()

    shader_property.ClearAllVertexShaderReplacements()
    shader_property.ClearAllFragmentShaderReplacements()

    # Vertex adaptation from shaders/vertexShader.glsl:
    # expose near/far clip positions and object-space vertex position.
    shader_property.AddVertexShaderReplacement(
        "//VTK::Base::Dec",
        True,
        "//VTK::Base::Dec\nout vec4 v_nearpos;\nout vec4 v_farpos;\nout vec3 v_position;\n",
        False,
    )
    shader_property.AddVertexShaderReplacement(
        "//VTK::ComputeTextureCoords::Impl",
        True,
        "//VTK::ComputeTextureCoords::Impl\n"
        "  mat4 viewtransformf = in_modelViewMatrix;\n"
        "  mat4 viewtransformi = inverse(in_modelViewMatrix);\n"
        "  vec4 position4 = vec4(in_vertexPos, 1.0);\n"
        "  vec4 pos_in_cam = viewtransformf * position4;\n"
        "  pos_in_cam.z = -pos_in_cam.w;\n"
        "  v_nearpos = viewtransformi * pos_in_cam;\n"
        "  pos_in_cam.z = pos_in_cam.w;\n"
        "  v_farpos = viewtransformi * pos_in_cam;\n"
        "  v_position = in_vertexPos;\n",
        False,
    )

    # Fragment declarations for varyings and constants sourced from
    # shaders/fragmentShader.glsl.
    shader_property.AddFragmentShaderReplacement(
        "//VTK::Base::Dec",
        True,
        "//VTK::Base::Dec\nin vec4 v_nearpos;\nin vec4 v_farpos;\nin vec3 v_position;\n"
        f"const int USER_MAX_STEPS = {int(max_steps_hint)};\n",
        False,
    )

    # Replace VTK's default per-step shading with front-to-back compositing
    # style from cast_volume() in shaders/fragmentShader.glsl.
    shader_property.AddFragmentShaderReplacement(
        "//VTK::Shading::Impl",
        True,
        "    if (!g_skip)\n"
        "    {\n"
        "      vec4 scalar = texture3D(in_volume[0], g_dataPos);\n"
        "      scalar = scalar * in_volume_scale[0] + in_volume_bias[0];\n"
        "      float opacity = computeOpacity(scalar);\n"
        "      if (opacity > 0.0)\n"
        "      {\n"
        "        vec4 sampleColor = computeColor(scalar, opacity);\n"
        "        sampleColor.rgb *= sampleColor.a;\n"
        "        g_fragColor.rgb += (1.0 - g_fragColor.a) * sampleColor.rgb;\n"
        "        g_fragColor.a += (1.0 - g_fragColor.a) * sampleColor.a;\n"
        f"        if (g_fragColor.a >= {alpha_cutoff:.6f})\n"
        "        {\n"
        "          g_exit = true;\n"
        "        }\n"
        "      }\n"
        "    }\n",
        False,
    )


def _render_volume_views(
    volume_zyx: np.ndarray,
    outdir: Path,
    stem_prefix: str,
    window_size: Tuple[int, int],
    background: str,
    vertex_shader_source: str,
    fragment_shader_source: str,
    tf_colour_cmap: ListedColormap,
    tf_opacity_lut: List[float],
    low_pct: float,
    high_pct: float,
    opacity_unit_distance: float,
    camera_dist_scale: float,
    zoom: float,
    up_rotate_deg: float,
):
    # Use off-screen rendering so this script can run on headless servers.
    pv.OFF_SCREEN = True
    outdir.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(volume_zyx, dtype=np.float32)
    finite = np.isfinite(vol)

    # Replace NaN/Inf with a robust fill value to avoid transfer-function artifacts.
    if not np.any(finite):
        logger.warning("Volume '%s' has no finite values; rendering zeros.", stem_prefix)
        vol = np.zeros_like(vol, dtype=np.float32)
    else:
        fill = float(np.nanpercentile(vol[finite], 50.0))
        vol = np.where(finite, vol, fill).astype(np.float32, copy=False)

    # Normalize with fixed global range: global_vmin=-1, global_vmax=1.
    # Values outside [-1, 1] are clipped after normalization.
    src_min = float(np.min(vol))
    src_max = float(np.max(vol))
    vol = (vol - GLOBAL_VMIN) / (GLOBAL_VMAX - GLOBAL_VMIN)
    vol = np.clip(vol, 0.0, 1.0).astype(np.float32, copy=False)

    # After normalization, keep clim fixed to [0, 1].
    grid = _volume_to_grid(vol)
    clim = [0.0, 1.0]
    center, views, diag = _make_camera_views_from_grid(grid, camera_dist_scale)
    opacity_nz = float(np.mean(np.asarray(tf_opacity_lut, dtype=np.float32) > 1e-6))
    logger.info(
        "Render '%s': src_range=(%.4e, %.4e), norm_range=(%.1f, %.1f)->(0,1), clim=(%.1f, %.1f), opacity_nonzero=%.1f%%",
        stem_prefix,
        src_min,
        src_max,
        GLOBAL_VMIN,
        GLOBAL_VMAX,
        float(clim[0]),
        float(clim[1]),
        100.0 * opacity_nz,
    )

    outputs = []
    for view_name, cam_pos, cam_up in views:
        plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        plotter.set_background(str(background))

        # Visual quality options: FXAA suppresses jagged edges; depth peeling helps
        # translucent layers blend in a more stable order.
        plotter.enable_anti_aliasing("fxaa")
        plotter.enable_depth_peeling()

        # Build volume actor first, then attach shader-adapted ray marcher.
        volume_actor = plotter.add_volume(
            grid,
            scalars="values",
            cmap=tf_colour_cmap,
            opacity=tf_opacity_lut,
            shade=True,
            clim=clim,
            show_scalar_bar=False,
            ambient=0.25,
            diffuse=0.75,
            specular=0.25,
            specular_power=15.0,
            opacity_unit_distance=float(opacity_unit_distance),
            blending="composite",
        )
        _apply_custom_volume_shaders(
            volume_actor=volume_actor,
            vertex_source=vertex_shader_source,
            fragment_source=fragment_shader_source,
        )

        # Start from a grid-aware camera, then enforce our main-view orientation.
        plotter.reset_camera(bounds=grid.bounds)
        up1 = _rotate_up_around_view(
            cam_pos=cam_pos,
            center=center,
            cam_up=cam_up,
            angle_deg=float(up_rotate_deg),
        )
        plotter.camera_position = [cam_pos, center, up1]

        # Tie clipping planes to scene size: too narrow clips content, too wide
        # reduces depth precision and harms translucent composition.
        near = max(1e-3, 0.02 * float(diag))
        far = max(10.0, 4.0 * float(diag))
        plotter.camera.clipping_range = (near, far)
        plotter.camera.zoom(float(zoom))

        logger.info(
            "View '%s' camera_position=%s up_rotate_deg=%.1f",
            view_name,
            plotter.camera_position,
            float(up_rotate_deg),
        )
        out_path = outdir / f"{stem_prefix}_{view_name}.png"
        plotter.screenshot(str(out_path))
        plotter.close()
        outputs.append(out_path)
    return outputs


def _run_checkpoint_mode(
    args,
    cfg,
    tf_colour_cmap: ListedColormap,
    tf_opacity_lut: List[float],
    vertex_shader_source: str,
    fragment_shader_source: str,
):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in --mode checkpoint.")

    dataset = _build_dataset(cfg)
    shape = _resolve_volume_shape(cfg, dataset)
    times = _select_times(args.start_time, args.end_time, int(shape.T))

    model = build_model(cfg["model"], dataset)
    payload = _torch_load_checkpoint(Path(args.checkpoint))
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    if isinstance(dataset, MultiTargetVolumetricDataset):
        available_attrs = list(dataset.view_specs().keys())
    else:
        available_attrs = ["targets"]
    if not args.attrs.strip():
        raise ValueError("--attrs is required in --mode checkpoint, e.g. --attrs H+,He+,O2+")
    attrs = _parse_attr_list(args.attrs, available_attrs)

    # infer output dims from dataset
    if isinstance(dataset, MultiTargetVolumetricDataset):
        specs = dataset.view_specs()
        out_dims = {name: int(specs[name]) for name in attrs}
    else:
        out_dims = {attrs[0]: int(getattr(dataset, "_target_dim", 1))}

    denorm_stats = {}
    payload_dict = payload if isinstance(payload, dict) else {}
    for name in attrs:
        denorm_stats[name] = _resolve_attr_stats(dataset, payload_dict, name, out_dims[name])

    outdir = Path(args.outdir)
    total_frames = len(times) * len(attrs)
    pbar = _tqdm(total=total_frames, desc="volume_render_ckpt", leave=True) if _tqdm is not None else None

    for t_idx in times:
        _sync_if_needed(device)
        volumes = _predict_volumes_for_timestep(
            model=model,
            dataset=dataset,
            attrs=attrs,
            shape=shape,
            t_idx=int(t_idx),
            batch_size=int(args.batch_size),
            channel=int(args.channel),
            denorm_stats=denorm_stats,
            device=device,
        )
        for attr_name in attrs:
            stem = f"ckpt_{attr_name}_t{int(t_idx):04d}"
            _render_volume_views(
                volume_zyx=volumes[attr_name],
                outdir=outdir,
                stem_prefix=stem,
                window_size=(int(args.window_width), int(args.window_height)),
                background=str(args.background),
                vertex_shader_source=vertex_shader_source,
                fragment_shader_source=fragment_shader_source,
                tf_colour_cmap=tf_colour_cmap,
                tf_opacity_lut=tf_opacity_lut,
                low_pct=float(args.clim_low_pct),
                high_pct=float(args.clim_high_pct),
                opacity_unit_distance=float(args.opacity_unit_distance),
                camera_dist_scale=float(args.camera_dist_scale),
                zoom=float(args.zoom),
                up_rotate_deg=float(args.up_rotate_deg),
            )
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    logger.info(
        "Checkpoint rendering done. mode=checkpoint attrs=%s times=[%d,%d) views=%d outdir=%s",
        attrs,
        times[0],
        times[-1] + 1,
        1,
        outdir,
    )


def _run_npy_mode(
    args,
    cfg,
    tf_colour_cmap: ListedColormap,
    tf_opacity_lut: List[float],
    vertex_shader_source: str,
    fragment_shader_source: str,
):
    if not args.npy_files:
        raise ValueError("--npy-files is required in --mode npy.")
    attr_paths = _parse_attr_path_pairs(args.npy_files)

    shape = _resolve_volume_shape(cfg, dataset=None)
    times = _select_times(args.start_time, args.end_time, int(shape.T))
    outdir = Path(args.outdir)

    arrays = {name: np.load(str(path), mmap_mode="r") for name, path in attr_paths.items()}
    total_frames = len(times) * len(arrays)
    pbar = _tqdm(total=total_frames, desc="volume_render_npy", leave=True) if _tqdm is not None else None

    for t_idx in times:
        for attr_name, arr in arrays.items():
            volume = _extract_volume_from_array(
                arr=arr,
                shape=shape,
                t_idx=int(t_idx),
                channel=int(args.channel),
            )
            stem = f"{args.npy_tag}_{attr_name}_t{int(t_idx):04d}"
            _render_volume_views(
                volume_zyx=volume,
                outdir=outdir,
                stem_prefix=stem,
                window_size=(int(args.window_width), int(args.window_height)),
                background=str(args.background),
                vertex_shader_source=vertex_shader_source,
                fragment_shader_source=fragment_shader_source,
                tf_colour_cmap=tf_colour_cmap,
                tf_opacity_lut=tf_opacity_lut,
                low_pct=float(args.clim_low_pct),
                high_pct=float(args.clim_high_pct),
                opacity_unit_distance=float(args.opacity_unit_distance),
                camera_dist_scale=float(args.camera_dist_scale),
                zoom=float(args.zoom),
                up_rotate_deg=float(args.up_rotate_deg),
            )
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    logger.info(
        "NPY rendering done. mode=npy attrs=%s times=[%d,%d) views=%d outdir=%s",
        list(arrays.keys()),
        times[0],
        times[-1] + 1,
        1,
        outdir,
    )


def main():
    setup_logging()
    args = _parse_args()
    cfg = load_config(args.config)
    vertex_shader_source = _load_shader_source(Path(args.vertex_shader), kind="Vertex")
    fragment_shader_source = _load_shader_source(Path(args.fragment_shader), kind="Fragment")
    tf_cfg = _load_transfer_function_config(Path(args.tf_config))
    tf_colour_cmap, tf_opacity_lut = _build_transfer_function_luts(tf_cfg, bins=256)

    if args.mode == "checkpoint":
        _run_checkpoint_mode(
            args,
            cfg,
            tf_colour_cmap=tf_colour_cmap,
            tf_opacity_lut=tf_opacity_lut,
            vertex_shader_source=vertex_shader_source,
            fragment_shader_source=fragment_shader_source,
        )
    else:
        _run_npy_mode(
            args,
            cfg,
            tf_colour_cmap=tf_colour_cmap,
            tf_opacity_lut=tf_opacity_lut,
            vertex_shader_source=vertex_shader_source,
            fragment_shader_source=fragment_shader_source,
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import importlib
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from inr.cli import build_model, load_config, resolve_data_paths
from inr.data import MultiTargetVolumetricDataset, VolumetricDataset
from inr.datasets.base import (
    DEFAULT_NORMALIZATION_SCHEME,
    resolve_checkpoint_normalization_scheme,
    resolve_normalization_scheme,
)
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Full-volume PSNR validation with inference time."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=16000, help="Inference batch size.")
    parser.add_argument(
        "--attrs",
        type=str,
        default="",
        help="Comma-separated attr names for multi-target model (default: all attrs).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="validate_out/validation_psnr_results.csv",
        help="Path to csv file for results.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="validate_out/validation_psnr_timestep.log",
        help="Path to .log file for per-timestep PSNR results.",
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
    def _load():
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    try:
        return _load()
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy._core", "numpy._core.multiarray"}:
            raise
        _register_numpy_core_aliases()
        return _load()


def _load_target_stats(stats_path: str, attr_paths=None, expected_scheme: str | None = None):
    data = np.load(stats_path, allow_pickle=True)
    stats_scheme = DEFAULT_NORMALIZATION_SCHEME
    if "scheme" in data:
        raw = np.asarray(data["scheme"])
        stats_scheme = resolve_normalization_scheme(str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0]))
    if expected_scheme is not None:
        expected = resolve_normalization_scheme(expected_scheme)
        if stats_scheme != expected:
            raise ValueError(
                f"Target stats scheme mismatch in {stats_path}: file={stats_scheme!r}, expected={expected!r}"
            )
    if attr_paths:
        stats = {}
        for name in attr_paths.keys():
            mean_key = f"{name}__mean"
            std_key = f"{name}__std"
            if mean_key not in data or std_key not in data:
                raise KeyError(f"Missing keys '{mean_key}'/'{std_key}' in {stats_path}")
            stats[name] = {"mean": data[mean_key], "std": data[std_key]}
        return stats
    if "mean" not in data or "std" not in data:
        raise KeyError(f"Missing keys 'mean'/'std' in {stats_path}")
    return {"mean": data["mean"], "std": data["std"]}


def _build_dataset(cfg):
    data_cfg = cfg["data"]
    data_info = resolve_data_paths(data_cfg)
    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))
    normalization_scheme = resolve_normalization_scheme(data_cfg.get("normalization_scheme"))

    target_stats = None
    stats_path = data_cfg.get("target_stats_path")
    if stats_path and Path(stats_path).exists():
        try:
            target_stats = _load_target_stats(
                stats_path,
                attr_paths=data_info.get("attr_paths"),
                expected_scheme=normalization_scheme,
            )
        except ValueError as exc:
            logger.warning("Ignoring incompatible target stats file %s: %s", stats_path, exc)

    if data_info.get("attr_paths"):
        dataset = MultiTargetVolumetricDataset(
            data_info["attr_paths"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            normalization_scheme=normalization_scheme,
            target_stats=target_stats,
        )
        gt_paths = {name: Path(path) for name, path in data_info["attr_paths"].items()}
    else:
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            normalization_scheme=normalization_scheme,
            target_stats=target_stats,
        )
        gt_paths = {"targets": Path(data_info["y_path"])}
    return dataset, gt_paths


def _validate_checkpoint_normalization_scheme(dataset, ckpt_payload):
    ckpt_scheme = resolve_checkpoint_normalization_scheme(ckpt_payload)
    dataset_scheme = resolve_normalization_scheme(getattr(dataset, "normalization_scheme", None))
    if ckpt_scheme != dataset_scheme:
        raise ValueError(
            f"Checkpoint normalization_scheme={ckpt_scheme!r} does not match "
            f"dataset/config normalization_scheme={dataset_scheme!r}"
        )


def _parse_attrs(arg: str, available: List[str]) -> List[str]:
    if not arg or not arg.strip():
        return list(available)
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {available}")
    return requested


def _prepare_xy_base(dataset, device: torch.device):
    X = int(dataset.volume_shape.X)
    Y = int(dataset.volume_shape.Y)

    xs = torch.arange(X, device=device, dtype=torch.float32).repeat(Y)
    ys = torch.arange(Y, device=device, dtype=torch.float32).repeat_interleave(X)

    mean = dataset.x_mean.reshape(-1).to(device=device, dtype=torch.float32)
    std = dataset.x_std.reshape(-1).to(device=device, dtype=torch.float32)

    base = torch.empty((X * Y, 4), device=device, dtype=torch.float32)
    base[:, 0] = (xs - mean[0]) / std[0]
    base[:, 1] = (ys - mean[1]) / std[1]

    return base, mean, std


def _build_xy_coords_norm_from_base(
    base_xy: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    t_idx: int,
    z_idx: int,
) -> torch.Tensor:
    coords = base_xy.clone()
    coords[:, 2] = (float(z_idx) - x_mean[2]) / x_std[2]
    coords[:, 3] = (float(t_idx) - x_mean[3]) / x_std[3]
    return coords


def _prepare_xy_index(volume_shape):
    X = int(volume_shape.X)
    Y = int(volume_shape.Y)
    xs = np.tile(np.arange(X, dtype=np.int64), Y)
    ys = np.repeat(np.arange(Y, dtype=np.int64), X)
    return xs, ys


def _extract_xy_slice(arr: np.ndarray, volume_shape, t_idx: int, z_idx: int, xs=None, ys=None) -> np.ndarray:
    X = int(volume_shape.X)
    Y = int(volume_shape.Y)
    Z = int(volume_shape.Z)

    if arr.ndim == 5:
        return np.asarray(arr[t_idx, z_idx, :, :, :], dtype=np.float32)
    if arr.ndim == 4:
        return np.asarray(arr[t_idx, z_idx, :, :], dtype=np.float32)[:, :, None]

    if xs is None or ys is None:
        xs = np.tile(np.arange(X, dtype=np.int64), Y)
        ys = np.repeat(np.arange(Y, dtype=np.int64), X)

    idx = ((t_idx * Z + z_idx) * Y + ys) * X + xs

    if arr.ndim == 2:
        C = int(arr.shape[1])
        return np.asarray(arr[idx, :], dtype=np.float32).reshape(Y, X, C)
    if arr.ndim == 1:
        return np.asarray(arr[idx], dtype=np.float32).reshape(Y, X, 1)
    raise ValueError(f"Unsupported target ndim: {arr.ndim}")


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


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _predict_slice(
    model: torch.nn.Module,
    coords_norm: torch.Tensor,
    attrs: List[str],
    batch_size: int,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], float]:
    pred_chunks: Dict[str, List[torch.Tensor]] = {name: [] for name in attrs}
    _sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        n = int(coords_norm.shape[0])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = coords_norm[start:end]
            try:
                out = model(xb, hard_topk=True)
            except TypeError:
                out = model(xb)

            if isinstance(out, dict):
                for name in attrs:
                    if name not in out:
                        raise KeyError(f"Model output missing attr '{name}'. Available: {list(out.keys())}")
                    pred_chunks[name].append(out[name])
            else:
                if len(attrs) != 1:
                    raise ValueError("Single tensor output cannot be mapped to multiple attrs.")
                pred_chunks[attrs[0]].append(out)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - t0

    out_tensors: Dict[str, torch.Tensor] = {}
    for name in attrs:
        arr = torch.cat(pred_chunks[name], dim=0)
        if arr.ndim == 1:
            arr = arr[:, None]
        out_tensors[name] = arr
    return out_tensors, float(elapsed)


def _infer_output_dims(dataset, attrs: List[str]) -> Dict[str, int]:
    if isinstance(dataset, MultiTargetVolumetricDataset):
        specs = dataset.view_specs()
        return {name: int(specs[name]) for name in attrs}
    out_dim = int(getattr(dataset, "_target_dim", 1))
    return {attrs[0]: out_dim}


def _compute_psnr_from_accum(sse: float, count: int, gt_min: float, gt_max: float) -> float:
    if count <= 0:
        return float("nan")
    mse = float(sse) / float(count)
    if mse <= 0:
        return float("inf")
    data_range = float(gt_max - gt_min)
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(abs(float(gt_min)), abs(float(gt_max))) + 1e-12
    return float(10.0 * math.log10((data_range * data_range) / (mse + 1e-12)))


def _append_csv_rows(csv_path: Path, rows: List[Dict]):
    fieldnames = [
        "timestamp",
        "eval_mode",
        "config_path",
        "checkpoint_path",
        "exp_id",
        "model_name",
        "dataset_name",
        "attr",
        "psnr",
        "inference_time_sec",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_timestep_log(log_path: Path, lines: List[str], reset: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if reset else "a"
    with log_path.open(mode, encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    setup_logging()
    args = _parse_args()

    cfg = load_config(args.config)
    dataset, gt_paths = _build_dataset(cfg)
    model = build_model(cfg["model"], dataset)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = _torch_load_checkpoint(ckpt_path)
    _validate_checkpoint_normalization_scheme(dataset, payload)
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    if isinstance(dataset, MultiTargetVolumetricDataset):
        available_attrs = list(dataset.view_specs().keys())
    else:
        available_attrs = ["targets"]
    attrs = _parse_attrs(args.attrs, available_attrs)

    output_dims = _infer_output_dims(dataset, attrs)
    denorm_stats = {}
    payload_dict = payload if isinstance(payload, dict) else {}
    for name in attrs:
        denorm_stats[name] = _resolve_attr_stats(dataset, payload_dict, name, output_dims[name])
    denorm_stats_torch = {
        name: (
            torch.from_numpy(denorm_stats[name][0]).to(device=device, dtype=torch.float32),
            torch.from_numpy(denorm_stats[name][1]).to(device=device, dtype=torch.float32),
        )
        for name in attrs
    }

    gt_arrays = {name: np.load(str(gt_paths[name]), mmap_mode="r") for name in attrs}
    xy_xs, xy_ys = _prepare_xy_index(dataset.volume_shape)

    timestep_psnr: Dict[str, List[float]] = {name: [] for name in attrs}

    Y = int(dataset.volume_shape.Y)
    X = int(dataset.volume_shape.X)
    Z = int(dataset.volume_shape.Z)
    T = int(dataset.volume_shape.T)
    total_slices = T * Z
    total_points = 0
    total_infer_time = 0.0
    global_start = time.perf_counter()
    pbar = _tqdm(total=total_slices, desc="validate_psnr_full", leave=True) if _tqdm is not None else None
    base_xy, x_mean_t, x_std_t = _prepare_xy_base(dataset, device)
    log_path = Path(args.log)
    _append_timestep_log(
        log_path,
        [
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"config_path={Path(args.config).resolve()}",
            f"checkpoint_path={Path(args.checkpoint).resolve()}",
            f"attrs={','.join(attrs)}",
            "--- per-timestep psnr ---",
        ],
        reset=True,
    )

    for t_idx in range(T):
        logger.info("Evaluating full timestep %d/%d", t_idx + 1, T)
        timestep_stats = {
            name: {
                "sse": torch.zeros((), dtype=torch.float64, device=device),
                "count": 0,
                "gt_min": torch.full((), float("inf"), dtype=torch.float32, device=device),
                "gt_max": torch.full((), float("-inf"), dtype=torch.float32, device=device),
            }
            for name in attrs
        }
        timestep_infer_time = 0.0

        for z_idx in range(Z):
            coords_norm = _build_xy_coords_norm_from_base(base_xy, x_mean_t, x_std_t, t_idx, z_idx)
            pred_map, elapsed = _predict_slice(
                model=model,
                coords_norm=coords_norm,
                attrs=attrs,
                batch_size=int(args.batch_size),
                device=device,
            )
            total_infer_time += float(elapsed)
            timestep_infer_time += float(elapsed)
            total_points += int(coords_norm.shape[0])

            for name in attrs:
                pred_flat = pred_map[name]
                pred_plane = pred_flat.reshape(Y, X, output_dims[name])
                mean_t, std_t = denorm_stats_torch[name]
                pred_plane = pred_plane * std_t[None, None, :] + mean_t[None, None, :]
                gt_plane = _extract_xy_slice(
                    gt_arrays[name],
                    dataset.volume_shape,
                    t_idx,
                    z_idx,
                    xs=xy_xs,
                    ys=xy_ys,
                )
                if gt_plane.shape != pred_plane.shape:
                    raise ValueError(
                        f"Shape mismatch for attr '{name}' at t={t_idx}, z={z_idx}: "
                        f"gt={gt_plane.shape}, pred={pred_plane.shape}"
                    )

                gt_plane_t = torch.from_numpy(gt_plane).to(device=device, dtype=pred_plane.dtype, non_blocking=True)
                diff = pred_plane - gt_plane_t
                timestep_stats[name]["sse"] += torch.sum(diff * diff, dtype=torch.float64)
                timestep_stats[name]["count"] += int(diff.numel())
                timestep_stats[name]["gt_min"] = torch.minimum(timestep_stats[name]["gt_min"], torch.min(gt_plane_t))
                timestep_stats[name]["gt_max"] = torch.maximum(timestep_stats[name]["gt_max"], torch.max(gt_plane_t))

            if pbar is not None:
                pbar.update(1)

        timestep_lines = []
        timestep_vals = []
        for name in attrs:
            t_psnr = _compute_psnr_from_accum(
                sse=float(timestep_stats[name]["sse"].item()),
                count=int(timestep_stats[name]["count"]),
                gt_min=float(timestep_stats[name]["gt_min"].item()),
                gt_max=float(timestep_stats[name]["gt_max"].item()),
            )
            timestep_psnr[name].append(float(t_psnr))
            timestep_vals.append(float(t_psnr))
            timestep_lines.append(
                f"t={t_idx:04d} attr={name} psnr={float(t_psnr):.6f} "
                f"inference_time_sec={float(timestep_infer_time):.4f}"
            )

        if len(timestep_vals) > 1:
            timestep_lines.append(
                f"t={t_idx:04d} attr=__mean__ psnr={float(np.mean(timestep_vals)):.6f} "
                f"inference_time_sec={float(timestep_infer_time):.4f}"
            )

        _append_timestep_log(log_path, timestep_lines, reset=False)

    if pbar is not None:
        pbar.close()

    total_wall_time = time.perf_counter() - global_start
    logger.info(
        "Full evaluation finished: T=%d Z=%d slices=%d points=%d inference_time=%.4fs wall_time=%.4fs",
        T,
        Z,
        total_slices,
        total_points,
        float(total_infer_time),
        float(total_wall_time),
    )

    rows = []
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "timestamp": run_ts,
        "eval_mode": "timestep_avg_psnr",
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "exp_id": str(cfg.get("exp_id", "")),
        "model_name": str(cfg.get("model", {}).get("name", "")),
        "dataset_name": str(cfg.get("data", {}).get("dataset_name", "")),
    }
    for name in attrs:
        vals = [v for v in timestep_psnr[name] if np.isfinite(v)]
        psnr_val = float(np.mean(vals)) if vals else float("nan")
        row = {
            "attr": name,
            "psnr": float(psnr_val),
            "inference_time_sec": float(total_infer_time),
        }
        row.update(meta)
        rows.append(row)

    logger.info("===== Validation Result (PSNR Full Volume) =====")
    for row in rows:
        logger.info(
            "[%s] avg_timestep_PSNR=%.6f | inference_time=%.4fs",
            row["attr"],
            row["psnr"],
            row["inference_time_sec"],
        )

    if len(rows) > 1:
        mean_psnr = float(np.mean([r["psnr"] for r in rows]))
        logger.info(
            "[mean] PSNR=%.6f | inference_time=%.4fs",
            mean_psnr,
            float(total_infer_time),
        )
        mean_row = {
            "attr": "__mean__",
            "psnr": mean_psnr,
            "inference_time_sec": float(total_infer_time),
        }
        mean_row.update(meta)
        rows.append(mean_row)

    csv_path = Path(args.csv)
    _append_csv_rows(csv_path, rows)
    logger.info("Saved PSNR validation results to CSV: %s", csv_path)
    logger.info("Saved per-timestep PSNR log to: %s", log_path)


if __name__ == "__main__":
    main()

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
    parser.add_argument("--device", type=str, default=None, help="Device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=32768, help="Inference batch size.")
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


def _load_target_stats(stats_path: str, attr_paths=None):
    data = np.load(stats_path, allow_pickle=True)
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
        gt_paths = {name: Path(path) for name, path in data_info["attr_paths"].items()}
    else:
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
        gt_paths = {"targets": Path(data_info["y_path"])}
    return dataset, gt_paths


def _parse_attrs(arg: str, available: List[str]) -> List[str]:
    if not arg or not arg.strip():
        return list(available)
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {available}")
    return requested


def _build_xy_coords_norm(dataset, t_idx: int, z_idx: int) -> np.ndarray:
    X = int(dataset.volume_shape.X)
    Y = int(dataset.volume_shape.Y)
    x = np.tile(np.arange(X, dtype=np.float32), Y)
    y = np.repeat(np.arange(Y, dtype=np.float32), X)
    z = np.full((X * Y,), float(z_idx), dtype=np.float32)
    t = np.full((X * Y,), float(t_idx), dtype=np.float32)
    coords = np.stack([x, y, z, t], axis=1)

    mean = dataset.x_mean.reshape(-1).cpu().numpy().astype(np.float32)
    std = dataset.x_std.reshape(-1).cpu().numpy().astype(np.float32)
    coords = (coords - mean[None, :]) / std[None, :]
    return coords.astype(np.float32, copy=False)


def _extract_xy_slice(arr: np.ndarray, volume_shape, t_idx: int, z_idx: int) -> np.ndarray:
    X = int(volume_shape.X)
    Y = int(volume_shape.Y)
    Z = int(volume_shape.Z)

    if arr.ndim == 5:
        return np.asarray(arr[t_idx, z_idx, :, :, :], dtype=np.float32)
    if arr.ndim == 4:
        return np.asarray(arr[t_idx, z_idx, :, :], dtype=np.float32)[:, :, None]

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
    coords_norm: np.ndarray,
    attrs: List[str],
    batch_size: int,
    device: torch.device,
) -> tuple[Dict[str, np.ndarray], float]:
    pred_chunks: Dict[str, List[torch.Tensor]] = {name: [] for name in attrs}
    _sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        n = int(coords_norm.shape[0])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(coords_norm[start:end]).to(device, non_blocking=True)
            try:
                out = model(xb, hard_topk=True)
            except TypeError:
                out = model(xb)

            if isinstance(out, dict):
                for name in attrs:
                    if name not in out:
                        raise KeyError(f"Model output missing attr '{name}'. Available: {list(out.keys())}")
                    pred_chunks[name].append(out[name].detach().cpu())
            else:
                if len(attrs) != 1:
                    raise ValueError("Single tensor output cannot be mapped to multiple attrs.")
                pred_chunks[attrs[0]].append(out.detach().cpu())
    _sync_if_needed(device)
    elapsed = time.perf_counter() - t0

    out_np = {}
    for name in attrs:
        arr = torch.cat(pred_chunks[name], dim=0).numpy()
        if arr.ndim == 1:
            arr = arr[:, None]
        out_np[name] = np.asarray(arr, dtype=np.float32)
    return out_np, float(elapsed)


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

    gt_arrays = {name: np.load(str(gt_paths[name]), mmap_mode="r") for name in attrs}

    stats = {}
    for name in attrs:
        stats[name] = {
            "sse": 0.0,
            "count": 0,
            "gt_min": float("inf"),
            "gt_max": float("-inf"),
        }

    Y = int(dataset.volume_shape.Y)
    X = int(dataset.volume_shape.X)
    Z = int(dataset.volume_shape.Z)
    T = int(dataset.volume_shape.T)
    total_slices = T * Z
    total_points = 0
    total_infer_time = 0.0
    global_start = time.perf_counter()
    pbar = _tqdm(total=total_slices, desc="validate_psnr_full", leave=True) if _tqdm is not None else None

    for t_idx in range(T):
        logger.info("Evaluating full timestep %d/%d", t_idx + 1, T)
        for z_idx in range(Z):
            coords_norm = _build_xy_coords_norm(dataset, t_idx, z_idx)
            pred_map, elapsed = _predict_slice(
                model=model,
                coords_norm=coords_norm,
                attrs=attrs,
                batch_size=int(args.batch_size),
                device=device,
            )
            total_infer_time += float(elapsed)
            total_points += int(coords_norm.shape[0])

            for name in attrs:
                pred_flat = pred_map[name]
                pred_plane = pred_flat.reshape(Y, X, output_dims[name])
                mean_arr, std_arr = denorm_stats[name]
                pred_plane = pred_plane * std_arr[None, None, :] + mean_arr[None, None, :]
                gt_plane = _extract_xy_slice(gt_arrays[name], dataset.volume_shape, t_idx, z_idx)
                if gt_plane.shape != pred_plane.shape:
                    raise ValueError(
                        f"Shape mismatch for attr '{name}' at t={t_idx}, z={z_idx}: "
                        f"gt={gt_plane.shape}, pred={pred_plane.shape}"
                    )

                diff = pred_plane.astype(np.float64) - gt_plane.astype(np.float64)
                stats[name]["sse"] += float(np.sum(diff * diff, dtype=np.float64))
                stats[name]["count"] += int(diff.size)
                stats[name]["gt_min"] = min(stats[name]["gt_min"], float(np.min(gt_plane)))
                stats[name]["gt_max"] = max(stats[name]["gt_max"], float(np.max(gt_plane)))

            if pbar is not None:
                pbar.update(1)

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
        "eval_mode": "full_volume_psnr",
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "exp_id": str(cfg.get("exp_id", "")),
        "model_name": str(cfg.get("model", {}).get("name", "")),
        "dataset_name": str(cfg.get("data", {}).get("dataset_name", "")),
    }
    for name in attrs:
        psnr_val = _compute_psnr_from_accum(
            sse=float(stats[name]["sse"]),
            count=int(stats[name]["count"]),
            gt_min=float(stats[name]["gt_min"]),
            gt_max=float(stats[name]["gt_max"]),
        )
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
            "[%s] PSNR=%.6f | inference_time=%.4fs",
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


if __name__ == "__main__":
    main()

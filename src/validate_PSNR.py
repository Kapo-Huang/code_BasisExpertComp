import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from inr.cli import build_model, load_config, resolve_data_paths
from inr.data import MultiViewCoordDataset, NodeDataset
from inr.utils.logging_utils import setup_logging
from inr.utils.io import warn_if_multiview_attr_order_mismatch

logger = logging.getLogger(__name__)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Node dataset PSNR validation with inference time."
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
        default="validate_out/validation_psnr.log",
        help="Path to .log file for validation results.",
    )
    return parser.parse_args()


def _register_numpy_core_aliases():
    import importlib
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


def _parse_attrs(arg: str, available: List[str]) -> List[str]:
    if not arg or not arg.strip():
        return list(available)
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {available}")
    return requested


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
        if isinstance(dataset, MultiViewCoordDataset):
            mean_arr = dataset.y_mean[attr_name].reshape(-1).cpu().numpy()
            std_arr = dataset.y_std[attr_name].reshape(-1).cpu().numpy()
        else:
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


def _predict_batch(
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
    if isinstance(dataset, MultiViewCoordDataset):
        return {name: int(dataset.y[name].shape[-1]) for name in attrs}
    out_dim = int(dataset.y.shape[-1]) if dataset.y.ndim > 1 else 1
    return {attrs[0]: out_dim}


def _get_timestep_indices(coords_np: np.ndarray, time_col: int = -1) -> Dict[float, np.ndarray]:
    """
    Extract indices grouped by raw timestep values (assumes last column is time).
    """
    time_vals = coords_np[:, time_col]
    unique_times = np.unique(time_vals)
    indices = {}
    for t_val in unique_times:
        indices[float(t_val)] = np.where(time_vals == t_val)[0]
    return indices


def _compute_psnr_from_mse(mse: float, gt_min: float, gt_max: float) -> float:
    """Compute PSNR given MSE and data range."""
    if not np.isfinite(mse) or mse < 0:
        return float("nan")
    if mse == 0:
        return float("inf")
    data_range = float(gt_max - gt_min)
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(abs(float(gt_min)), abs(float(gt_max))) + 1e-12
    return float(10.0 * math.log10((data_range * data_range) / (mse + 1e-12)))


def _append_csv_rows(csv_path: Path, rows: List[Dict]):
    fieldnames = [
        "timestamp",
        "config_path",
        "checkpoint_path",
        "exp_id",
        "model_name",
        "dataset_name",
        "attr",
        "psnr",
        "mse",
        "inference_time_sec",
        "num_samples",
        "num_timesteps",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_log(log_path: Path, lines: List[str], reset: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if reset else "a"
    with log_path.open(mode, encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    setup_logging()
    args = _parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    data_info = resolve_data_paths(data_cfg)
    normalize = bool(data_cfg.get("normalize", True))

    if data_info.get("attr_paths"):
        dataset = MultiViewCoordDataset(
            data_info["x_path"],
            data_info["attr_paths"],
            normalize=normalize,
            stats_path=data_cfg.get("target_stats_path"),
        )
        available_attrs = list(dataset.y.keys())
    else:
        dataset = NodeDataset(
            data_info["x_path"],
            data_info["y_path"],
            normalize=normalize,
            stats_path=data_cfg.get("target_stats_path"),
        )
        available_attrs = ["targets"]

    model = build_model(cfg["model"], dataset)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = _torch_load_checkpoint(ckpt_path)
    if isinstance(dataset, MultiViewCoordDataset):
        warn_if_multiview_attr_order_mismatch(
            payload,
            dataset.view_specs().keys(),
            context=str(Path(args.config).resolve()),
            logger_override=logger,
        )
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

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
    denorm_stats_np = {
        name: (
            denorm_stats_torch[name][0].cpu().numpy()[None, :],
            denorm_stats_torch[name][1].cpu().numpy()[None, :],
        )
        for name in attrs
    }

    if isinstance(dataset, MultiViewCoordDataset):
        coords_np = dataset.x.cpu().numpy() if isinstance(dataset.x, torch.Tensor) else dataset.x
        gt_data = {name: (dataset.y[name].cpu().numpy() if isinstance(dataset.y[name], torch.Tensor) else dataset.y[name]) for name in attrs}
    else:
        coords_np = dataset.x.cpu().numpy() if isinstance(dataset.x, torch.Tensor) else dataset.x
        gt_data = {"targets": (dataset.y.cpu().numpy() if isinstance(dataset.y, torch.Tensor) else dataset.y)}

    coords_norm = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)
    n_samples = int(coords_norm.shape[0])
    
    timestep_indices = _get_timestep_indices(coords_np)
    timesteps = sorted(timestep_indices.keys())
    num_timesteps = len(timesteps)

    log_path = Path(args.log)
    _append_log(
        log_path,
        [
            f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"config_path={Path(args.config).resolve()}",
            f"checkpoint_path={Path(args.checkpoint).resolve()}",
            f"dataset_name={data_info.get('dataset_name', 'unknown')}",
            f"num_samples={n_samples}",
            f"num_timesteps={num_timesteps}",
            f"input_dim={coords_np.shape[1]}",
            f"attrs={','.join(attrs)}",
            "--- Per-Timestep PSNR ---",
        ],
        reset=True,
    )

    global_start = time.perf_counter()
    pbar = _tqdm(total=num_timesteps, desc="validate_psnr", leave=True) if _tqdm is not None else None

    timestep_psnrs: Dict[str, List[float]] = {name: [] for name in attrs}
    total_infer_time = 0.0

    for t_order, raw_time in enumerate(timesteps):
        idxs = timestep_indices[raw_time]
        coords_t = coords_norm[idxs]
        
        pred_map, elapsed = _predict_batch(
            model=model,
            coords_norm=coords_t,
            attrs=attrs,
            batch_size=int(args.batch_size),
            device=device,
        )
        total_infer_time += float(elapsed)
        
        timestep_log_lines = []
        for name in attrs:
            pred_flat = pred_map[name]
            pred_np = pred_flat.cpu().numpy()

            mean_np, std_np = denorm_stats_np[name]
            pred_denorm = pred_np * std_np + mean_np

            gt_flat = gt_data[name][idxs]
            if gt_flat.ndim == 1:
                gt_flat = gt_flat[:, None]
            gt_denorm = gt_flat * std_np + mean_np

            if pred_denorm.shape != gt_denorm.shape:
                raise ValueError(
                    f"Shape mismatch for attr '{name}' at timestep {raw_time}: "
                    f"pred={pred_denorm.shape}, gt={gt_denorm.shape}"
                )

            diff = pred_denorm - gt_denorm
            mse = float(np.mean(diff ** 2))
            gt_min = float(np.min(gt_denorm))
            gt_max = float(np.max(gt_denorm))
            psnr = _compute_psnr_from_mse(mse, gt_min, gt_max)

            timestep_psnrs[name].append(float(psnr))
            timestep_log_lines.append(
                f"t_idx={t_order:04d} raw_t={raw_time:>10.6f} attr={name:<12} "
                f"psnr={float(psnr):>10.6f} mse={float(mse):>12.6e}"
            )
        
        if pbar is not None:
            pbar.update(1)
        
        _append_log(log_path, timestep_log_lines, reset=False)

    if pbar is not None:
        pbar.close()

    total_wall_time = time.perf_counter() - global_start

    rows = []
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "timestamp": run_ts,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "exp_id": str(cfg.get("exp_id", "")),
        "model_name": str(cfg.get("model", {}).get("name", "")),
        "dataset_name": str(data_info.get("dataset_name", "")),
        "num_samples": n_samples,
        "num_timesteps": num_timesteps,
    }

    _append_log(log_path, ["--- Per-Attribute Averaged (over timesteps) ---"], reset=False)
    
    summary_lines = []
    final_attr_psnrs = []

    for name in attrs:
        vals = [v for v in timestep_psnrs[name] if np.isfinite(v)]
        avg_psnr = float(np.mean(vals)) if vals else float("nan")
        final_attr_psnrs.append(avg_psnr)
        
        row = {
            "attr": name,
            "psnr": avg_psnr,
            "mse": float("nan"),
            "inference_time_sec": float(total_infer_time),
        }
        row.update(meta)
        rows.append(row)
        
        summary_lines.append(
            f"attr={name:<12} avg_psnr={float(avg_psnr):>10.6f} "
            f"(over {len(vals)} timesteps)"
        )

    if len(rows) > 1:
        mean_psnr = float(np.mean(final_attr_psnrs))
        summary_lines.append(
            f"--- Mean PSNR across all attributes: {float(mean_psnr):.6f} ---"
        )
        mean_row = {
            "attr": "__mean__",
            "psnr": mean_psnr,
            "mse": float("nan"),
            "inference_time_sec": float(total_infer_time),
        }
        mean_row.update(meta)
        rows.append(mean_row)

    _append_log(log_path, summary_lines, reset=False)

    logger.info("===== Node Dataset Validation Results =====")
    for attr, psnr in zip(attrs, final_attr_psnrs):
        logger.info("[%s] avg_timestep_PSNR=%.6f", attr, psnr)

    if len(rows) > 1:
        logger.info("[mean] avg_timestep_PSNR=%.6f", float(np.mean(final_attr_psnrs)))

    logger.info(
        "Validation finished: samples=%d timesteps=%d inference_time=%.4fs wall_time=%.4fs",
        n_samples,
        num_timesteps,
        float(total_infer_time),
        float(total_wall_time),
    )

    csv_path = Path(args.csv)
    _append_csv_rows(csv_path, rows)
    logger.info("Saved validation results to CSV: %s", csv_path)
    logger.info("Saved validation log to: %s", log_path)


if __name__ == "__main__":
    main()

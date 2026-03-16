import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        description="Export predicted scalar volume as per-timestep 1D .npy files."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Single timestep to export. If omitted, export all timesteps.",
    )
    parser.add_argument("--attr", type=str, default="", help="Target attr name for multi-target models.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=32768, help="Inference batch size.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out/prediction_npy",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pred",
        help="Output filename prefix.",
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
    else:
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
    return dataset


def _resolve_attr_name(dataset, arg_attr: str) -> str:
    if isinstance(dataset, MultiTargetVolumetricDataset):
        available = list(dataset.view_specs().keys())
        if not arg_attr.strip():
            raise ValueError(f"--attr is required for multi-target model. Available attrs: {available}")
        if arg_attr not in available:
            raise KeyError(f"Unknown attr '{arg_attr}'. Available attrs: {available}")
        return arg_attr
    return "targets"


def _resolve_attr_stats(dataset, ckpt_payload, attr_name: str, out_dim: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _select_timesteps(timestamp: Optional[int], T: int) -> List[int]:
    if T <= 0:
        raise ValueError("Invalid volume shape: T<=0")
    if timestamp is None:
        return list(range(T))
    t = int(timestamp)
    if t < 0 or t >= T:
        raise ValueError(f"timestamp out of range: {t}, valid [0, {T - 1}]")
    return [t]


def _predict_timestep_flat_scalar(
    model: torch.nn.Module,
    dataset,
    attr_name: str,
    out_dim: int,
    t_idx: int,
    batch_size: int,
    denorm_mean: torch.Tensor,
    denorm_std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    X = int(dataset.volume_shape.X)
    Y = int(dataset.volume_shape.Y)
    Z = int(dataset.volume_shape.Z)
    V = X * Y * Z

    x_mean = dataset.x_mean.reshape(-1).to(device=device, dtype=torch.float32)
    x_std = dataset.x_std.reshape(-1).to(device=device, dtype=torch.float32)

    out_flat = np.empty((V,), dtype=np.float32)

    model.eval()
    _sync_if_needed(device)
    with torch.inference_mode():
        for start in range(0, V, int(batch_size)):
            end = min(start + int(batch_size), V)
            idx = torch.arange(start, end, device=device, dtype=torch.int64)
            x = (idx % X).to(dtype=torch.float32)
            y = ((idx // X) % Y).to(dtype=torch.float32)
            z = (idx // (X * Y)).to(dtype=torch.float32)
            t = torch.full_like(x, float(t_idx), dtype=torch.float32)

            coords = torch.empty((end - start, 4), device=device, dtype=torch.float32)
            coords[:, 0] = (x - x_mean[0]) / x_std[0]
            coords[:, 1] = (y - x_mean[1]) / x_std[1]
            coords[:, 2] = (z - x_mean[2]) / x_std[2]
            coords[:, 3] = (t - x_mean[3]) / x_std[3]

            try:
                pred = model(coords, hard_topk=True)
            except TypeError:
                pred = model(coords)

            if isinstance(pred, dict):
                if attr_name not in pred:
                    raise KeyError(f"Model output missing attr '{attr_name}'. Available: {list(pred.keys())}")
                pred_t = pred[attr_name]
            else:
                pred_t = pred

            if pred_t.ndim == 1:
                pred_t = pred_t[:, None]
            if int(pred_t.shape[1]) != int(out_dim):
                raise ValueError(
                    f"Output dim mismatch for attr '{attr_name}': "
                    f"pred_dim={int(pred_t.shape[1])}, expected={int(out_dim)}"
                )

            pred_t = pred_t * denorm_std[None, :] + denorm_mean[None, :]
            if int(pred_t.shape[1]) > 1:
                scalar = torch.linalg.norm(pred_t, dim=1)
            else:
                scalar = pred_t[:, 0]

            out_flat[start:end] = scalar.detach().cpu().numpy().astype(np.float32, copy=False)

    _sync_if_needed(device)
    return out_flat


def _build_output_path(outdir: Path, prefix: str, attr_name: str, t_idx: int) -> Path:
    if attr_name == "targets":
        filename = f"{prefix}_t{int(t_idx):04d}.npy"
    else:
        filename = f"{prefix}_{attr_name}_t{int(t_idx):04d}.npy"
    return outdir / filename


def main():
    setup_logging()
    args = _parse_args()

    cfg = load_config(args.config)
    dataset = _build_dataset(cfg)
    model = build_model(cfg["model"], dataset)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = _torch_load_checkpoint(ckpt_path)
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    attr_name = _resolve_attr_name(dataset, args.attr)
    if isinstance(dataset, MultiTargetVolumetricDataset):
        out_dim = int(dataset.view_specs()[attr_name])
    else:
        out_dim = int(getattr(dataset, "_target_dim", 1))

    payload_dict = payload if isinstance(payload, dict) else {}
    mean_arr, std_arr = _resolve_attr_stats(dataset, payload_dict, attr_name, out_dim)
    denorm_mean = torch.from_numpy(mean_arr).to(device=device, dtype=torch.float32)
    denorm_std = torch.from_numpy(std_arr).to(device=device, dtype=torch.float32)

    T = int(dataset.volume_shape.T)
    timesteps = _select_timesteps(args.timestamp, T)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pbar = _tqdm(total=len(timesteps), desc="predict_export", leave=True) if _tqdm is not None else None

    for t_idx in timesteps:
        pred_flat = _predict_timestep_flat_scalar(
            model=model,
            dataset=dataset,
            attr_name=attr_name,
            out_dim=out_dim,
            t_idx=int(t_idx),
            batch_size=int(args.batch_size),
            denorm_mean=denorm_mean,
            denorm_std=denorm_std,
            device=device,
        )

        out_path = _build_output_path(outdir, str(args.prefix), attr_name, int(t_idx))
        np.save(str(out_path), pred_flat)
        logger.info("Saved timestep=%d shape=%s to %s", int(t_idx), tuple(pred_flat.shape), out_path)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    logger.info(
        "Prediction export finished. timesteps=%d attr=%s outdir=%s",
        len(timesteps),
        attr_name,
        outdir,
    )


if __name__ == "__main__":
    main()
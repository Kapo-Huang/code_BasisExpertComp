from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import load_checkpoint_payload


def normalize_coords(coords: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, normalize: bool) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if not normalize:
        return coords
    return ((coords - np.asarray(x_mean, dtype=np.float32)) / np.asarray(x_std, dtype=np.float32)).astype(np.float32)


def denormalize_targets(values: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, normalize: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not normalize:
        return values
    return values * np.asarray(y_std, dtype=np.float32) + np.asarray(y_mean, dtype=np.float32)


def compute_time_indexers(time_values: np.ndarray) -> list[tuple[float, slice | np.ndarray]]:
    unique_times, first_indices, counts = np.unique(
        time_values,
        return_index=True,
        return_counts=True,
    )
    order = np.argsort(first_indices)
    unique_times = unique_times[order]
    first_indices = first_indices[order]
    counts = counts[order]

    contiguous = True
    for raw_time, start, count in zip(unique_times, first_indices, counts):
        block = time_values[start : start + count]
        if block.shape[0] != count or not np.all(block == raw_time):
            contiguous = False
            break

    if contiguous:
        return [
            (float(raw_time), slice(int(start), int(start + count)))
            for raw_time, start, count in zip(unique_times, first_indices, counts)
        ]

    return [
        (float(raw_time), np.flatnonzero(time_values == raw_time))
        for raw_time in unique_times
    ]


def select_indexer_block(array: np.ndarray, indexer: slice | np.ndarray) -> np.ndarray:
    return np.asarray(array[indexer])


def select_timestamps(
    time_indexers: list[tuple[float, slice | np.ndarray]],
    timestamps: str = "",
    max_timesteps: int = 0,
) -> list[int]:
    total = len(time_indexers)
    if timestamps.strip():
        selected = []
        seen = set()
        for token in timestamps.split(","):
            token = token.strip()
            if not token:
                continue
            idx = int(token)
            if idx < 0 or idx >= total:
                raise ValueError(f"Timestamp index {idx} is out of range [0, {total - 1}]")
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        return selected
    if max_timesteps > 0:
        return list(range(min(total, int(max_timesteps))))
    return list(range(total))


def unwrap_model_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def resolve_checkpoint_stats(payload: Any, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalize_inputs = bool(cfg["DATA"].get("normalize_inputs", False))
    normalize_targets = bool(cfg["DATA"].get("normalize_targets", False))
    input_dim = int(cfg["MODEL"]["in_dim"])
    target_dim = int(cfg["MODEL"]["out_dim"])

    if not normalize_inputs and not normalize_targets:
        return (
            np.zeros((1, input_dim), dtype=np.float32),
            np.ones((1, input_dim), dtype=np.float32),
            np.zeros((1, target_dim), dtype=np.float32),
            np.ones((1, target_dim), dtype=np.float32),
        )

    if isinstance(payload, dict):
        x_mean = payload.get("x_mean")
        x_std = payload.get("x_std")
        y_mean = payload.get("y_mean")
        y_std = payload.get("y_std")
        if x_mean is not None and x_std is not None and y_mean is not None and y_std is not None:
            return (
                np.asarray(x_mean, dtype=np.float32).reshape(1, -1),
                np.asarray(x_std, dtype=np.float32).reshape(1, -1),
                np.asarray(y_mean, dtype=np.float32).reshape(1, -1),
                np.asarray(y_std, dtype=np.float32).reshape(1, -1),
            )

    from datasets_loader.Mesh import _ensure_2d, _load_or_compute_stats

    source = np.load(cfg["DATA"]["source_path"], mmap_mode="r")
    target = _ensure_2d(np.load(cfg["DATA"]["target_path"], mmap_mode="r"))
    stats_path = Path(cfg["DATA"]["target_stats_path"]) if cfg["DATA"].get("target_stats_path") else None
    x_mean, x_std, y_mean, y_std = _load_or_compute_stats(
        source=source,
        target=target,
        stats_path=stats_path,
        stats_key=str(cfg["DATA"]["stats_key"]),
        input_dim=int(source.shape[1]),
        target_dim=int(target.shape[1]),
        load_input_stats=normalize_inputs,
        load_target_stats=normalize_targets,
    )
    return x_mean.numpy(), x_std.numpy(), y_mean.numpy(), y_std.numpy()


def predict_block(
    model: torch.nn.Module,
    coords_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    pred_chunks = []
    total = int(coords_norm.shape[0])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    wall_start = None
    if device.type != "cuda":
        import time

        wall_start = time.perf_counter()
    else:
        start_time.record()

    with torch.inference_mode():
        for start in range(0, total, int(batch_size)):
            end = min(start + int(batch_size), total)
            xb = torch.from_numpy(coords_norm[start:end]).to(device=device, dtype=torch.float32, non_blocking=True)
            xb = xb.unsqueeze(0)
            output = model(xb)
            if not isinstance(output, dict):
                raise TypeError(f"Unexpected model output type: {type(output)!r}")
            if "selected_nonmanifold_pnts_pred" in output:
                pred = output["selected_nonmanifold_pnts_pred"]
            elif "nonmanifold_pnts_pred" in output:
                pred = output["nonmanifold_pnts_pred"].permute(0, 2, 1)
            else:
                raise KeyError(f"Prediction tensor not found in output keys: {list(output.keys())}")

            pred = pred.detach().cpu()
            while pred.ndim > 2 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            if pred.ndim == 3 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if pred.ndim == 1:
                pred = pred[:, None]
            pred = pred.numpy().astype(np.float32, copy=False)
            pred = pred.reshape(pred.shape[0], -1)
            pred_chunks.append(pred)

    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize(device)
        elapsed = float(start_time.elapsed_time(end_time) / 1000.0)
    else:
        import time

        elapsed = float(time.perf_counter() - wall_start)

    return np.concatenate(pred_chunks, axis=0), elapsed


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mse = float(np.mean(np.square(pred - gt)))
    if mse == 0.0:
        return float("inf"), mse
    gt_min = float(np.min(gt))
    gt_max = float(np.max(gt))
    data_range = float(gt_max - gt_min)
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(abs(gt_min), abs(gt_max), 1.0)
    psnr = float(10.0 * math.log10((data_range * data_range) / (mse + 1.0e-12)))
    return psnr, mse


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Any:
    return load_checkpoint_payload(path, device=device)

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inr.utils.logging_utils import setup_logging
from validate_router_distribution import (
    _build_input_norm_stats,
    _build_model_and_inputs,
    _get_router_probs_for_batch,
    _get_view_names,
    _load_yaml,
    _parse_requested_attrs,
    _resolve_train_dir,
    _torch_load_checkpoint,
)

logger = logging.getLogger(__name__)


def _parse_volume_shape_override(raw_values: list[int] | None) -> tuple[int, int, int] | None:
    if raw_values is None:
        return None
    if len(raw_values) != 3:
        raise ValueError("--volume-shape must provide exactly 3 integers: X Y Z")
    x, y, z = (int(raw_values[0]), int(raw_values[1]), int(raw_values[2]))
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError(f"--volume-shape values must be > 0, got {(x, y, z)}")
    return x, y, z


def _resolve_analysis_shape(volume_shape, shape_override: tuple[int, int, int] | None) -> tuple[int, int, int, int]:
    inferred = (int(volume_shape.X), int(volume_shape.Y), int(volume_shape.Z), int(volume_shape.T))
    if shape_override is None:
        return inferred
    x, y, z = shape_override
    if (x, y, z) != inferred[:3]:
        logger.warning(
            "Using --volume-shape=%s instead of inferred spatial shape=%s. "
            "Ensure this matches your structured grid semantics.",
            (x, y, z),
            inferred[:3],
        )
    return x, y, z, inferred[3]


def _resolve_timestep(timestep_arg: int | None, total_t: int) -> int:
    if total_t <= 0:
        raise ValueError(f"Invalid temporal size T={total_t}")
    if timestep_arg is None:
        if total_t == 1:
            return 0
        raise ValueError("--timestep is required when T > 1.")
    timestep = int(timestep_arg)
    if timestep < 0 or timestep >= int(total_t):
        raise ValueError(f"--timestep out of range: {timestep}, valid range is [0, {int(total_t) - 1}]")
    return timestep


def _resolve_center_index(idx: int | None, size: int, axis_name: str) -> int:
    if size <= 0:
        raise ValueError(f"Invalid axis size for {axis_name}: {size}")
    if idx is None:
        if axis_name == "x":
            # Keep default center consistent with validate_basis_fun.py.
            return int(size // 1.5)
        return size // 2
    value = int(idx)
    if value < 0 or value >= int(size):
        raise ValueError(f"--center-{axis_name} out of range: {value}, valid range is [0, {int(size) - 1}]")
    return value


def _parse_attribute_names(raw_names: list[str] | None) -> list[str] | None:
    if raw_names is None:
        return None
    names = [str(name).strip() for name in raw_names if str(name).strip()]
    if not names:
        raise ValueError("--attribute-names provided but empty after stripping.")
    return names


def _build_sampling_order(n_xyz: int, samples_per_timestamp: int, rng: np.random.Generator) -> np.ndarray:
    if n_xyz <= 0:
        raise ValueError(f"n_xyz must be > 0, got {n_xyz}")
    if samples_per_timestamp <= 0:
        raise ValueError(f"samples_per_timestamp must be > 0, got {samples_per_timestamp}")
    if samples_per_timestamp >= n_xyz:
        return np.arange(n_xyz, dtype=np.int64)
    return rng.permutation(n_xyz).astype(np.int64, copy=False)


def _make_sampled_xyz_coords(flat_indices: np.ndarray, sx: int, sy: int, t: int) -> np.ndarray:
    idx = np.asarray(flat_indices, dtype=np.int64)
    x = (idx % sx).astype(np.float32)
    yz = idx // sx
    y = (yz % sy).astype(np.float32)
    z = (yz // sy).astype(np.float32)
    t_arr = np.full_like(x, float(t), dtype=np.float32)
    return np.stack([x, y, z, t_arr], axis=1)


def _stack_probs_by_attr(probs_by_attr: dict[str, torch.Tensor], selected_attrs: list[str]) -> torch.Tensor:
    probs = [probs_by_attr[attr_name] for attr_name in selected_attrs]
    weights = torch.stack(probs, dim=1)
    if weights.dim() != 3:
        raise ValueError(f"Expected stacked router weights to be 3D [N, A, E], got {tuple(weights.shape)}")
    return weights


def load_router_outputs(
    model: torch.nn.Module,
    xb: torch.Tensor,
    selected_attrs: list[str],
    device: torch.device,
    input_type: str,
) -> np.ndarray | torch.Tensor:
    probs_by_attr = _get_router_probs_for_batch(
        model=model,
        xb=xb,
        selected_attrs=selected_attrs,
        device=device,
    )
    weights = _stack_probs_by_attr(probs_by_attr, selected_attrs)
    if input_type == "numpy":
        return weights.detach().cpu().numpy().astype(np.float32, copy=False)
    return weights


def sort_experts(router_weights: np.ndarray | torch.Tensor, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rank <= 0:
        raise ValueError(f"max_rank must be > 0, got {max_rank}")

    if isinstance(router_weights, torch.Tensor):
        if router_weights.dim() != 3:
            raise ValueError(f"router_weights must be 3D [N, A, E], got {tuple(router_weights.shape)}")
        num_experts = int(router_weights.shape[2])
        keep_k = min(int(num_experts), int(max_rank) + 1)
        topk = torch.topk(router_weights, k=keep_k, dim=-1, largest=True, sorted=True)
        sorted_idx = topk.indices.detach().cpu().numpy().astype(np.int64, copy=False)
        sorted_val = topk.values.detach().cpu().numpy().astype(np.float32, copy=False)
        return sorted_idx, sorted_val

    weights_np = np.asarray(router_weights)
    if weights_np.ndim != 3:
        raise ValueError(f"router_weights must be 3D [N, A, E], got {tuple(weights_np.shape)}")
    num_experts = int(weights_np.shape[2])
    keep_k = min(int(num_experts), int(max_rank) + 1)
    order = np.argsort(-weights_np, axis=-1)
    sorted_idx = order[:, :, :keep_k].astype(np.int64, copy=False)
    sorted_val = np.take_along_axis(weights_np, sorted_idx, axis=-1).astype(np.float32, copy=False)
    return sorted_idx, sorted_val


def compute_rank_entropy(chosen_ids: np.ndarray, num_experts: int) -> np.ndarray:
    ids = np.asarray(chosen_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"chosen_ids must be 2D [N, A], got shape {tuple(ids.shape)}")
    n_coords, n_attrs = int(ids.shape[0]), int(ids.shape[1])
    if n_coords <= 0:
        return np.empty((0,), dtype=np.float32)
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")

    if np.min(ids) < 0 or np.max(ids) >= int(num_experts):
        raise ValueError(
            f"chosen_ids contain out-of-range experts: min={int(np.min(ids))}, max={int(np.max(ids))}, "
            f"num_experts={int(num_experts)}"
        )

    counts = np.zeros((n_coords, int(num_experts)), dtype=np.int32)
    rows = np.arange(n_coords, dtype=np.int64)
    for attr_idx in range(n_attrs):
        np.add.at(counts, (rows, ids[:, attr_idx]), 1)

    probs = counts.astype(np.float64, copy=False) / float(n_attrs)
    valid = probs > 0.0
    entropy = -np.sum(np.where(valid, probs * np.log(probs + 1e-12), 0.0), axis=1)
    denom = math.log(float(min(int(n_attrs), int(num_experts)))) if min(int(n_attrs), int(num_experts)) > 1 else 0.0
    if denom <= 0.0:
        return np.zeros((n_coords,), dtype=np.float32)
    entropy_norm = np.clip(entropy / denom, 0.0, 1.0)
    return entropy_norm.astype(np.float32, copy=False)


def compute_rank_confidence(sorted_scores: np.ndarray, rank: int) -> np.ndarray | None:
    scores = np.asarray(sorted_scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"sorted_scores must be 3D [N, A, K], got shape {tuple(scores.shape)}")
    if rank <= 0:
        raise ValueError(f"rank must be >= 1, got {rank}")
    left = int(rank - 1)
    right = int(rank)
    if right >= int(scores.shape[2]):
        return None
    margin = scores[:, :, left] - scores[:, :, right]
    return margin.mean(axis=1).astype(np.float32, copy=False)


def reshape_to_volume(values: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1D [N], got shape {tuple(arr.shape)}")
    expected = int(x) * int(y) * int(z)
    if int(arr.shape[0]) != expected:
        raise ValueError(f"Cannot reshape values size={int(arr.shape[0])} into volume [{x}, {y}, {z}]")
    # Flat coordinates are enumerated with X as the fastest-changing axis, so the
    # structured grid layout that matches validate_basis_fun.py is (Z, Y, X).
    return arr.reshape((int(z), int(y), int(x)))


def extract_orthogonal_slices(volume: np.ndarray, center_x: int, center_y: int, center_z: int) -> dict[str, np.ndarray]:
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D [Z, Y, X], got shape {tuple(vol.shape)}")
    z_size, y_size, x_size = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
    if center_x < 0 or center_x >= x_size:
        raise ValueError(f"center_x out of range: {center_x}")
    if center_y < 0 or center_y >= y_size:
        raise ValueError(f"center_y out of range: {center_y}")
    if center_z < 0 or center_z >= z_size:
        raise ValueError(f"center_z out of range: {center_z}")

    # Match validate_basis_fun.py orientation conventions exactly:
    # YZ @ x=mx -> shape (Z, Y), XZ @ y=my -> shape (Z, X), XY @ z=mz -> shape (Y, X)
    return {
        "YZ": np.asarray(vol[:, :, center_x], dtype=np.float32),
        "XZ": np.asarray(vol[:, center_y, :], dtype=np.float32),
        "XY": np.asarray(vol[center_z, :, :], dtype=np.float32),
    }


def _plot_three_slices(
    slices: dict[str, np.ndarray],
    out_path: Path,
    title_prefix: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    view_order = ["YZ", "XZ", "XY"]
    for ax, view_name in zip(axes, view_order):
        data_2d = np.asarray(slices[view_name], dtype=np.float32)
        im = ax.imshow(data_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(view_name)
        ax.set_xlabel("Axis-1")
        ax.set_ylabel("Axis-2")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_slices(
    entropy_volume: np.ndarray,
    rank: int,
    output_dir: Path,
    experiment_name: str,
    timestep: int,
    center_x: int,
    center_y: int,
    center_z: int,
    dpi: int,
) -> Path:
    slices = extract_orthogonal_slices(entropy_volume, center_x=center_x, center_y=center_y, center_z=center_z)
    out_path = output_dir / f"entropy_top{int(rank)}_slices.png"
    title = (
        f"{experiment_name} | cross-attribute | Entropy Top-{int(rank)} | "
        f"t={int(timestep)} | x={int(center_x)}, y={int(center_y)}, z={int(center_z)}"
    )
    _plot_three_slices(slices=slices, out_path=out_path, title_prefix=title, cmap="viridis", vmin=0.0, vmax=1.0, dpi=dpi)
    return out_path


def plot_confidence_slices(
    confidence_volume: np.ndarray,
    rank: int,
    output_dir: Path,
    experiment_name: str,
    timestep: int,
    center_x: int,
    center_y: int,
    center_z: int,
    dpi: int,
    global_vmin: float | None,
    global_vmax: float | None,
) -> Path:
    slices = extract_orthogonal_slices(confidence_volume, center_x=center_x, center_y=center_y, center_z=center_z)
    values = np.asarray(confidence_volume, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0:
        logger.info("Top-%d confidence range: min=%.6f max=%.6f", int(rank), float(np.min(finite_values)), float(np.max(finite_values)))
    else:
        logger.warning("Top-%d confidence volume has no finite values.", int(rank))

    out_path = output_dir / f"confidence_top{int(rank)}_slices.png"
    title = (
        f"{experiment_name} | cross-attribute | Confidence Top-{int(rank)} | "
        f"t={int(timestep)} | x={int(center_x)}, y={int(center_y)}, z={int(center_z)}"
    )
    _plot_three_slices(
        slices=slices,
        out_path=out_path,
        title_prefix=title,
        cmap="magma",
        vmin=global_vmin,
        vmax=global_vmax,
        dpi=dpi,
    )
    return out_path


def plot_combined_rank_panels(
    volumes_by_rank: dict[int, np.ndarray],
    metric_name: str,
    output_dir: Path,
    experiment_name: str,
    timestep: int,
    center_x: int,
    center_y: int,
    center_z: int,
    dpi: int,
    global_vmin: float | None = None,
    global_vmax: float | None = None,
) -> Path | None:
    ranks = sorted(int(rank) for rank in volumes_by_rank.keys())
    if not ranks:
        return None

    out_path = output_dir / ("entropy_all_ranks.png" if metric_name == "entropy" else "confidence_all_ranks.png")
    is_entropy = metric_name == "entropy"
    cmap = "viridis" if is_entropy else "magma"

    fig, axes = plt.subplots(len(ranks), 3, figsize=(15, 4.6 * len(ranks)), constrained_layout=True)
    if len(ranks) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, rank in enumerate(ranks):
        slices = extract_orthogonal_slices(
            volumes_by_rank[int(rank)],
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
        for col_idx, view_name in enumerate(["YZ", "XZ", "XY"]):
            ax = axes[row_idx, col_idx]
            panel = np.asarray(slices[view_name], dtype=np.float32)
            if is_entropy:
                vmin = 0.0
                vmax = 1.0
            else:
                vmin = global_vmin
                vmax = global_vmax

            im = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"Top-{int(rank)} {view_name}")
            ax.set_xlabel("Axis-1")
            ax.set_ylabel("Axis-2")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{experiment_name} | cross-attribute | {metric_name.capitalize()} | "
        f"t={int(timestep)} | x={int(center_x)}, y={int(center_y)}, z={int(center_z)}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _compute_distribution_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compute_global_confidence_scale(confidence_volumes: dict[int, np.ndarray]) -> tuple[float | None, float | None]:
    if not confidence_volumes:
        return None, None

    pieces = []
    for vol in confidence_volumes.values():
        arr = np.asarray(vol, dtype=np.float32).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            pieces.append(finite)

    if not pieces:
        logger.warning("No finite confidence values across ranks; confidence plots will use matplotlib default scale.")
        return None, None

    all_values = np.concatenate(pieces, axis=0)
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if math.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
    logger.info("Global confidence color scale: vmin=%.6f vmax=%.6f", vmin, vmax)
    return vmin, vmax


def _save_histogram(values: np.ndarray, out_prefix: Path, title: str, dpi: int) -> dict[str, Path]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if arr.size == 0:
        logger.warning("Skipping histogram for %s: no finite values.", out_prefix.name)
        return {}

    counts, bin_edges = np.histogram(arr, bins=80)
    npz_path = out_prefix.with_suffix(".npz")
    np.savez(npz_path, counts=counts, bin_edges=bin_edges)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), constrained_layout=True)
    ax.hist(arr, bins=80, color="#4C72B0", alpha=0.9, edgecolor="black", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    png_path = out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "npz": npz_path}


def save_summary_stats(summary: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return out_path


def extract_router_topk_overlap(
    config_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    attr_arg: str | None,
    timestep: int | None,
    center_x: int | None,
    center_y: int | None,
    center_z: int | None,
    volume_shape_override: tuple[int, int, int] | None,
    attribute_names: list[str] | None,
    batch_size: int,
    device_str: str | None,
    train_dir_override: str | None,
    input_type: str,
    max_rank: int,
    save_combined_figures: bool,
    save_histograms: bool,
    save_high_entropy_confidence_mask: bool,
    entropy_high_percentile: float,
    confidence_high_percentile: float,
    experiment_name_override: str | None,
    dpi: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = _load_yaml(config_path)
    exp_name = str(experiment_name_override or cfg.get("exp_id", "")).strip() or config_path.stem
    logger.info("[%s] Stage: load config = %.3fs", exp_name, time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if "model_state" not in checkpoint:
        raise KeyError(f"'model_state' missing in checkpoint: {checkpoint_path}")
    logger.info("[%s] Stage: load checkpoint = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    train_dir = _resolve_train_dir(cfg["data"], train_dir_override)
    model, volume_shape, _ = _build_model_and_inputs(cfg, checkpoint, train_dir)
    x_size, y_size, z_size, t_size = _resolve_analysis_shape(volume_shape, shape_override=volume_shape_override)
    t_idx = _resolve_timestep(timestep_arg=timestep, total_t=t_size)
    logger.info("[%s] Stage: build model/input specs = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    normalize_inputs = bool(cfg["data"].get("normalize_inputs", cfg["data"].get("normalize", True)))
    mean_np, std_np = _build_input_norm_stats(volume_shape, normalize_inputs)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    view_names = _get_view_names(model)
    selected_attrs = _parse_requested_attrs(attr_arg, view_names)
    if attribute_names is not None:
        if len(attribute_names) != len(selected_attrs):
            raise ValueError(
                f"--attribute-names length ({len(attribute_names)}) must match selected attributes ({len(selected_attrs)})"
            )
        display_attrs = list(attribute_names)
    else:
        display_attrs = list(selected_attrs)

    num_experts = int(getattr(model, "num_experts"))
    entropy_ranks = [rank for rank in range(1, min(int(max_rank), int(num_experts)) + 1)]
    confidence_ranks = [rank for rank in entropy_ranks if rank < int(num_experts)]
    if not entropy_ranks:
        raise ValueError(f"No valid entropy ranks: num_experts={num_experts}, max_rank={max_rank}")
    if int(max_rank) > int(num_experts):
        logger.warning(
            "Requested --max-rank=%d exceeds num_experts=%d. Computing entropy ranks %s only.",
            int(max_rank),
            int(num_experts),
            entropy_ranks,
        )
    if int(max_rank) >= 3 and int(num_experts) < 4:
        logger.warning("Top-3 confidence requires E>=4. It will be skipped for this model (E=%d).", int(num_experts))

    if input_type not in {"numpy", "torch"}:
        raise ValueError(f"Unsupported --input-type: {input_type}")

    center_x_resolved = _resolve_center_index(center_x, x_size, "x")
    center_y_resolved = _resolve_center_index(center_y, y_size, "y")
    center_z_resolved = _resolve_center_index(center_z, z_size, "z")

    logger.info("[%s] Stage: prepare inference = %.3fs", exp_name, time.perf_counter() - t_stage)

    sx = int(x_size)
    sy = int(y_size)
    sz = int(z_size)
    n_xyz = sx * sy * sz
    if n_xyz <= 0:
        raise ValueError(f"Invalid structured grid size: X={sx}, Y={sy}, Z={sz}")

    entropy_flat: dict[int, np.ndarray] = {
        int(rank): np.empty((n_xyz,), dtype=np.float32) for rank in entropy_ranks
    }
    confidence_flat: dict[int, np.ndarray] = {
        int(rank): np.empty((n_xyz,), dtype=np.float32) for rank in confidence_ranks
    }

    with torch.inference_mode():
        cursor = 0
        t_infer = time.perf_counter()
        while cursor < int(n_xyz):
            chunk_size = min(int(batch_size), int(n_xyz) - cursor)
            flat_indices = np.arange(cursor, cursor + chunk_size, dtype=np.int64)
            coords = _make_sampled_xyz_coords(flat_indices, sx=sx, sy=sy, t=int(t_idx))
            coords_norm = ((coords - mean_np) / std_np).astype(np.float32, copy=False)
            xb = torch.from_numpy(coords_norm).to(device, non_blocking=True)

            router_weights = load_router_outputs(
                model=model,
                xb=xb,
                selected_attrs=selected_attrs,
                device=device,
                input_type=input_type,
            )
            sorted_idx, sorted_val = sort_experts(router_weights=router_weights, max_rank=max_rank)

            for rank in entropy_ranks:
                chosen_rank_ids = sorted_idx[:, :, int(rank - 1)]
                entropy_values = compute_rank_entropy(chosen_ids=chosen_rank_ids, num_experts=num_experts)
                entropy_flat[int(rank)][cursor : cursor + chunk_size] = entropy_values

            for rank in confidence_ranks:
                confidence_values = compute_rank_confidence(sorted_scores=sorted_val, rank=rank)
                if confidence_values is None:
                    continue
                confidence_flat[int(rank)][cursor : cursor + chunk_size] = confidence_values

            cursor += chunk_size

        logger.info(
            "[%s] Stage: full-grid inference @t=%d = %.3fs (coords=%d, attrs=%d, experts=%d)",
            exp_name,
            int(t_idx),
            time.perf_counter() - t_infer,
            int(n_xyz),
            int(len(selected_attrs)),
            int(num_experts),
        )

    entropy_volumes: dict[int, np.ndarray] = {
        int(rank): reshape_to_volume(entropy_flat[int(rank)], x=sx, y=sy, z=sz) for rank in entropy_ranks
    }
    confidence_volumes: dict[int, np.ndarray] = {
        int(rank): reshape_to_volume(confidence_flat[int(rank)], x=sx, y=sy, z=sz) for rank in confidence_ranks
    }
    confidence_global_vmin, confidence_global_vmax = _compute_global_confidence_scale(confidence_volumes)

    outdir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "experiment": exp_name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "timestep": int(t_idx),
        "shape": {"X": int(sx), "Y": int(sy), "Z": int(sz), "T": int(t_size)},
        "center": {"x": int(center_x_resolved), "y": int(center_y_resolved), "z": int(center_z_resolved)},
        "input_type": input_type,
        "max_rank_requested": int(max_rank),
        "num_experts": int(num_experts),
        "selected_attrs": list(display_attrs),
        "ranks": {},
        "artifacts": {},
    }

    for rank in entropy_ranks:
        entropy_volume = entropy_volumes[int(rank)]
        entropy_npy = outdir / f"entropy_top{int(rank)}.npy"
        np.save(entropy_npy, entropy_volume)
        files[f"entropy_top{int(rank)}_npy"] = str(entropy_npy)

        entropy_fig = plot_entropy_slices(
            entropy_volume=entropy_volume,
            rank=int(rank),
            output_dir=outdir,
            experiment_name=exp_name,
            timestep=int(t_idx),
            center_x=int(center_x_resolved),
            center_y=int(center_y_resolved),
            center_z=int(center_z_resolved),
            dpi=int(dpi),
        )
        files[f"entropy_top{int(rank)}_slices_png"] = str(entropy_fig)

        rank_key = f"top{int(rank)}"
        summary["ranks"][rank_key] = {
            "entropy": _compute_distribution_stats(entropy_volume.reshape(-1)),
        }

        if save_histograms:
            hist_out = _save_histogram(
                values=entropy_volume.reshape(-1),
                out_prefix=outdir / f"entropy_top{int(rank)}_hist",
                title=f"Entropy Top-{int(rank)} Histogram",
                dpi=int(dpi),
            )
            if hist_out:
                files[f"entropy_top{int(rank)}_hist_png"] = str(hist_out["png"])
                files[f"entropy_top{int(rank)}_hist_npz"] = str(hist_out["npz"])

    for rank in confidence_ranks:
        confidence_volume = confidence_volumes[int(rank)]
        confidence_npy = outdir / f"confidence_top{int(rank)}.npy"
        np.save(confidence_npy, confidence_volume)
        files[f"confidence_top{int(rank)}_npy"] = str(confidence_npy)

        confidence_fig = plot_confidence_slices(
            confidence_volume=confidence_volume,
            rank=int(rank),
            output_dir=outdir,
            experiment_name=exp_name,
            timestep=int(t_idx),
            center_x=int(center_x_resolved),
            center_y=int(center_y_resolved),
            center_z=int(center_z_resolved),
            dpi=int(dpi),
            global_vmin=confidence_global_vmin,
            global_vmax=confidence_global_vmax,
        )
        files[f"confidence_top{int(rank)}_slices_png"] = str(confidence_fig)

        rank_key = f"top{int(rank)}"
        summary["ranks"].setdefault(rank_key, {})
        summary["ranks"][rank_key]["confidence"] = _compute_distribution_stats(confidence_volume.reshape(-1))

        if save_histograms:
            hist_out = _save_histogram(
                values=confidence_volume.reshape(-1),
                out_prefix=outdir / f"confidence_top{int(rank)}_hist",
                title=f"Confidence Top-{int(rank)} Histogram",
                dpi=int(dpi),
            )
            if hist_out:
                files[f"confidence_top{int(rank)}_hist_png"] = str(hist_out["png"])
                files[f"confidence_top{int(rank)}_hist_npz"] = str(hist_out["npz"])

    skipped_conf_ranks = [rank for rank in entropy_ranks if rank not in confidence_ranks]
    if skipped_conf_ranks:
        logger.warning("Skipping confidence for ranks without rank+1 expert: %s", skipped_conf_ranks)
        for rank in skipped_conf_ranks:
            rank_key = f"top{int(rank)}"
            summary["ranks"].setdefault(rank_key, {})
            summary["ranks"][rank_key]["confidence"] = None

    if save_high_entropy_confidence_mask:
        for rank in sorted(set(entropy_ranks).intersection(confidence_ranks)):
            entropy_vec = entropy_volumes[int(rank)].reshape(-1)
            confidence_vec = confidence_volumes[int(rank)].reshape(-1)
            entropy_thr = float(np.percentile(entropy_vec, entropy_high_percentile))
            confidence_thr = float(np.percentile(confidence_vec, confidence_high_percentile))
            mask_vec = (entropy_vec > entropy_thr) & (confidence_vec > confidence_thr)
            mask_vol = mask_vec.reshape((sz, sy, sx))

            mask_path = outdir / f"high_entropy_high_confidence_top{int(rank)}.npy"
            np.save(mask_path, mask_vol)
            files[f"high_entropy_high_confidence_top{int(rank)}_npy"] = str(mask_path)

            rank_key = f"top{int(rank)}"
            summary["ranks"].setdefault(rank_key, {})
            summary["ranks"][rank_key]["high_entropy_high_confidence_mask"] = {
                "entropy_percentile": float(entropy_high_percentile),
                "confidence_percentile": float(confidence_high_percentile),
                "entropy_threshold": entropy_thr,
                "confidence_threshold": confidence_thr,
                "true_count": int(np.sum(mask_vec)),
                "ratio": float(np.mean(mask_vec.astype(np.float32))),
            }

    if save_combined_figures:
        entropy_all = plot_combined_rank_panels(
            volumes_by_rank=entropy_volumes,
            metric_name="entropy",
            output_dir=outdir,
            experiment_name=exp_name,
            timestep=int(t_idx),
            center_x=int(center_x_resolved),
            center_y=int(center_y_resolved),
            center_z=int(center_z_resolved),
            dpi=int(dpi),
        )
        if entropy_all is not None:
            files["entropy_all_ranks_png"] = str(entropy_all)

        confidence_all = plot_combined_rank_panels(
            volumes_by_rank=confidence_volumes,
            metric_name="confidence",
            output_dir=outdir,
            experiment_name=exp_name,
            timestep=int(t_idx),
            center_x=int(center_x_resolved),
            center_y=int(center_y_resolved),
            center_z=int(center_z_resolved),
            dpi=int(dpi),
            global_vmin=confidence_global_vmin,
            global_vmax=confidence_global_vmax,
        )
        if confidence_all is not None:
            files["confidence_all_ranks_png"] = str(confidence_all)

    summary["artifacts"] = files
    summary_path = save_summary_stats(summary=summary, out_path=outdir / "summary.json")
    files["summary_json"] = str(summary_path)

    logger.info("[%s] Stage: total = %.3fs", exp_name, time.perf_counter() - t_start)
    return {
        "summary_path": summary_path,
        "attrs": list(display_attrs),
        "timestep": int(t_idx),
        "shape": (int(sx), int(sy), int(sz), int(t_size)),
        "entropy_ranks": list(entropy_ranks),
        "confidence_ranks": list(confidence_ranks),
        "files": files,
    }


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Compute rank-specific cross-attribute expert entropy/confidence maps over a structured grid at fixed timestep."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out/router_rank_entropy",
        help="Output directory for entropy/confidence npy, figures, and summary files.",
    )
    parser.add_argument("--attr", type=str, default=None, help="single attr or comma list; default=all attrs")
    parser.add_argument(
        "--attribute-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional display names for selected attributes; length must equal number of selected attrs.",
    )
    parser.add_argument("--timestep", type=int, default=None, help="Fixed timestep index t. Optional only when T=1.")
    parser.add_argument(
        "--volume-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional override for structured spatial shape X Y Z.",
    )
    parser.add_argument("--center-x", type=int, default=None, help="Slice center index on X axis; default=X//2")
    parser.add_argument("--center-y", type=int, default=None, help="Slice center index on Y axis; default=Y//2")
    parser.add_argument("--center-z", type=int, default=None, help="Slice center index on Z axis; default=Z//2")
    parser.add_argument(
        "--max-rank",
        type=int,
        default=3,
        help="Maximum rank to analyze for entropy/confidence maps (Top-1..Top-K).",
    )
    parser.add_argument("--input-type", type=str, choices=["numpy", "torch"], default="torch")
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g., cpu/cuda:0")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir containing target_*.npy")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name override in titles.")
    parser.add_argument(
        "--save-combined-figures",
        action="store_true",
        help="Save combined multi-rank panel figures for entropy and confidence.",
    )
    parser.add_argument(
        "--save-histograms",
        action="store_true",
        help="Save per-rank histogram figures and histogram bins (.npz).",
    )
    parser.add_argument(
        "--save-high-entropy-confidence-mask",
        action="store_true",
        help="Save per-rank mask where entropy/confidence exceed percentile thresholds.",
    )
    parser.add_argument(
        "--entropy-high-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for high-entropy mask criterion.",
    )
    parser.add_argument(
        "--confidence-high-percentile",
        type=float,
        default=50.0,
        help="Percentile threshold for high-confidence mask criterion.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output image dpi")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_rank <= 0:
        raise ValueError("--max-rank must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if not (0.0 <= float(args.entropy_high_percentile) <= 100.0):
        raise ValueError("--entropy-high-percentile must be in [0, 100]")
    if not (0.0 <= float(args.confidence_high_percentile) <= 100.0):
        raise ValueError("--confidence-high-percentile must be in [0, 100]")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    outdir = Path(args.outdir)
    volume_shape_override = _parse_volume_shape_override(args.volume_shape)
    attribute_names = _parse_attribute_names(args.attribute_names)

    result = extract_router_topk_overlap(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        outdir=outdir,
        attr_arg=args.attr,
        timestep=args.timestep,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        volume_shape_override=volume_shape_override,
        attribute_names=attribute_names,
        batch_size=int(args.batch_size),
        device_str=args.device,
        train_dir_override=args.train_dir,
        input_type=str(args.input_type),
        max_rank=int(args.max_rank),
        save_combined_figures=bool(args.save_combined_figures),
        save_histograms=bool(args.save_histograms),
        save_high_entropy_confidence_mask=bool(args.save_high_entropy_confidence_mask),
        entropy_high_percentile=float(args.entropy_high_percentile),
        confidence_high_percentile=float(args.confidence_high_percentile),
        experiment_name_override=args.experiment_name,
        dpi=int(args.dpi),
    )

    logger.info("Finished %s", config_path)
    logger.info("  Attrs: %s", result["attrs"])
    logger.info("  Timestep: %s", result["timestep"])
    logger.info("  Shape: %s", result["shape"])
    logger.info("  Entropy ranks: %s", result["entropy_ranks"])
    logger.info("  Confidence ranks: %s", result["confidence_ranks"])
    logger.info("  Summary: %s", result["summary_path"])


if __name__ == "__main__":
    main()

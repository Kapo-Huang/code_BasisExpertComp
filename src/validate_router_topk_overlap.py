import argparse
import csv
import logging
import time
from pathlib import Path

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


def _parse_timestamps(text: str) -> list[int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("--timestamps must contain at least one comma-separated integer.")

    values = []
    seen = set()
    duplicates = set()
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp '{part}'. --timestamps must be comma-separated integers.") from exc
        if value in seen:
            duplicates.add(value)
        seen.add(value)
        values.append(value)

    if duplicates:
        raise ValueError(f"Duplicate timestamps are not allowed: {sorted(duplicates)}")
    return sorted(values)


def _validate_timestamps(timestamps: list[int], volume_shape) -> list[int]:
    t_max = int(volume_shape.T) - 1
    invalid = [t for t in timestamps if t < 0 or t > t_max]
    if invalid:
        raise ValueError(f"--timestamps out of range: {invalid}, valid range is [0, {t_max}]")
    return list(timestamps)


def _format_timestamps_tag(timestamps: list[int]) -> str:
    return "-".join(str(int(t)) for t in timestamps)


def _sample_flat_indices(n_xyz: int, samples_per_timestamp: int, rng: np.random.Generator) -> np.ndarray:
    if n_xyz <= 0:
        raise ValueError(f"n_xyz must be > 0, got {n_xyz}")
    if samples_per_timestamp <= 0:
        raise ValueError(f"samples_per_timestamp must be > 0, got {samples_per_timestamp}")

    if samples_per_timestamp >= n_xyz:
        return np.arange(n_xyz, dtype=np.int64)

    flat_indices = rng.choice(n_xyz, size=int(samples_per_timestamp), replace=False)
    flat_indices.sort()
    return flat_indices.astype(np.int64, copy=False)


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


def _compute_batch_overlap_sum(topk_indices: torch.Tensor, top_k: int) -> torch.Tensor:
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be 3D [N, A, K], got {tuple(topk_indices.shape)}")
    if int(topk_indices.shape[-1]) != int(top_k):
        raise ValueError(f"topk_indices last dim mismatch: got {int(topk_indices.shape[-1])}, expected {int(top_k)}")

    matches = topk_indices.unsqueeze(2).unsqueeze(-1) == topk_indices.unsqueeze(1).unsqueeze(-2)
    intersections = matches.any(dim=-1).sum(dim=-1).to(dtype=torch.float64)
    return intersections.sum(dim=0) / float(top_k)


def _compute_overlap_matrix_from_topk(topk_indices: torch.Tensor) -> np.ndarray:
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be 3D [N, A, K], got {tuple(topk_indices.shape)}")
    num_coords = int(topk_indices.shape[0])
    if num_coords <= 0:
        raise ValueError("topk_indices must contain at least one coordinate.")
    top_k = int(topk_indices.shape[-1])
    overlap_sum = _compute_batch_overlap_sum(topk_indices, top_k=top_k)
    matrix = (overlap_sum / float(num_coords)).detach().cpu().numpy().astype(np.float64, copy=False)
    return _finalize_overlap_matrix(matrix)


def _finalize_overlap_matrix(matrix: np.ndarray) -> np.ndarray:
    overlap = np.asarray(matrix, dtype=np.float64)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError(f"overlap matrix must be square 2D, got shape {tuple(overlap.shape)}")
    overlap = 0.5 * (overlap + overlap.T)
    np.fill_diagonal(overlap, 1.0)
    return overlap


def _validate_overlap_matrix(matrix: np.ndarray) -> None:
    overlap = np.asarray(matrix, dtype=np.float64)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError(f"overlap matrix must be square, got {tuple(overlap.shape)}")
    if not np.all(np.isfinite(overlap)):
        raise ValueError("overlap matrix contains non-finite values.")
    if np.min(overlap) < -1e-8 or np.max(overlap) > 1.0 + 1e-8:
        raise ValueError(
            f"overlap matrix values must stay in [0, 1], got min={float(np.min(overlap))} max={float(np.max(overlap))}"
        )
    if not np.allclose(overlap, overlap.T, atol=1e-8, rtol=0.0):
        raise ValueError("overlap matrix is not symmetric.")
    if not np.allclose(np.diag(overlap), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("overlap matrix diagonal must be exactly 1.")


def _plot_overlap_heatmap(
    overlap_matrix: np.ndarray,
    attrs: list[str],
    out_path: Path,
    exp_name: str,
    timestamps: list[int],
    num_samples_total: int,
    top_k: int,
    dpi: int,
) -> None:
    attr_count = len(attrs)
    fig_size = max(8.0, 1.05 * float(attr_count) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(overlap_matrix, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Average Top-K Overlap")

    ax.set_xticks(np.arange(attr_count))
    ax.set_yticks(np.arange(attr_count))
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.set_yticklabels(attrs)
    ax.set_xlabel("Attribute")
    ax.set_ylabel("Attribute")
    ax.set_title(
        f"{exp_name} | Router Top-{int(top_k)} Overlap | "
        f"timestamps=[{','.join(str(int(t)) for t in timestamps)}] | samples={int(num_samples_total)}"
    )

    for row_idx in range(attr_count):
        for col_idx in range(attr_count):
            value = float(overlap_matrix[row_idx, col_idx])
            text_color = "white" if value < 0.5 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_overlap_csv(matrix: np.ndarray, attrs: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attr", *attrs])
        for attr_name, row in zip(attrs, np.asarray(matrix, dtype=np.float64)):
            writer.writerow([attr_name, *[f"{float(value):.8f}" for value in row]])


def extract_router_topk_overlap(
    config_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    attr_arg: str | None,
    timestamps: list[int],
    samples_per_timestamp: int,
    batch_size: int,
    device_str: str | None,
    train_dir_override: str | None,
    dpi: int,
    seed: int,
):
    t_start = time.perf_counter()
    cfg = _load_yaml(config_path)
    exp_name = str(cfg.get("exp_id", "")).strip() or config_path.stem
    logger.info("[%s] Stage: load config = %.3fs", exp_name, time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if "model_state" not in checkpoint:
        raise KeyError(f"'model_state' missing in checkpoint: {checkpoint_path}")
    logger.info("[%s] Stage: load checkpoint = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    train_dir = _resolve_train_dir(cfg["data"], train_dir_override)
    model, volume_shape, _ = _build_model_and_inputs(cfg, checkpoint, train_dir)
    timestamps = _validate_timestamps(timestamps, volume_shape)
    logger.info("[%s] Stage: build model/input specs = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    normalize_inputs = bool(cfg["data"].get("normalize_inputs", cfg["data"].get("normalize", True)))
    mean_np, std_np = _build_input_norm_stats(volume_shape, normalize_inputs)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    view_names = _get_view_names(model)
    selected_attrs = _parse_requested_attrs(attr_arg, view_names)
    num_experts = int(getattr(model, "num_experts"))
    top_k = min(3, num_experts)
    if top_k <= 0:
        raise ValueError(f"Model must have at least one expert, got num_experts={num_experts}")
    logger.info("[%s] Stage: prepare inference = %.3fs", exp_name, time.perf_counter() - t_stage)

    sx = int(volume_shape.X)
    sy = int(volume_shape.Y)
    sz = int(volume_shape.Z)
    n_xyz = sx * sy * sz
    rng = np.random.default_rng(seed)
    attr_count = len(selected_attrs)
    overlap_sum = np.zeros((attr_count, attr_count), dtype=np.float64)
    num_samples_total = 0

    with torch.inference_mode():
        for t in timestamps:
            t_timestamp = time.perf_counter()
            sampled_indices = _sample_flat_indices(n_xyz=n_xyz, samples_per_timestamp=samples_per_timestamp, rng=rng)
            coords = _make_sampled_xyz_coords(sampled_indices, sx=sx, sy=sy, t=int(t))
            coords_norm = ((coords - mean_np) / std_np).astype(np.float32, copy=False)
            t_sample_count = int(coords_norm.shape[0])

            for start in range(0, t_sample_count, int(batch_size)):
                end = min(start + int(batch_size), t_sample_count)
                xb = torch.from_numpy(coords_norm[start:end]).to(device, non_blocking=True)
                probs_by_attr = _get_router_probs_for_batch(
                    model=model,
                    xb=xb,
                    selected_attrs=selected_attrs,
                    device=device,
                )
                weights = _stack_probs_by_attr(probs_by_attr, selected_attrs)
                topk_indices = torch.topk(weights, k=top_k, dim=-1, largest=True, sorted=False).indices
                batch_overlap_sum = _compute_batch_overlap_sum(topk_indices=topk_indices, top_k=top_k)
                overlap_sum += batch_overlap_sum.detach().cpu().numpy().astype(np.float64, copy=False)

            num_samples_total += t_sample_count
            logger.info(
                "[%s] Timestamp %d done in %.3fs (samples=%d, top_k=%d)",
                exp_name,
                int(t),
                time.perf_counter() - t_timestamp,
                t_sample_count,
                top_k,
            )

    if num_samples_total <= 0:
        raise RuntimeError("No sampled coordinates were processed.")

    overlap_matrix = _finalize_overlap_matrix(overlap_sum / float(num_samples_total))
    _validate_overlap_matrix(overlap_matrix)

    outdir.mkdir(parents=True, exist_ok=True)
    timestamps_tag = _format_timestamps_tag(timestamps)
    png_path = outdir / f"router_topk_overlap_t{timestamps_tag}.png"
    csv_path = outdir / f"router_topk_overlap_t{timestamps_tag}.csv"

    _plot_overlap_heatmap(
        overlap_matrix=overlap_matrix,
        attrs=selected_attrs,
        out_path=png_path,
        exp_name=exp_name,
        timestamps=timestamps,
        num_samples_total=num_samples_total,
        top_k=top_k,
        dpi=dpi,
    )
    _write_overlap_csv(overlap_matrix, selected_attrs, csv_path)

    logger.info("[%s] Stage: total = %.3fs", exp_name, time.perf_counter() - t_start)
    return {
        "png_path": png_path,
        "csv_path": csv_path,
        "attrs": list(selected_attrs),
        "timestamps": list(timestamps),
        "num_samples_total": int(num_samples_total),
        "top_k": int(top_k),
        "overlap_matrix": overlap_matrix,
    }


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Sample router outputs and visualize pairwise Top-K expert overlap across attributes."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out/router_overlap",
        help="Output directory for overlap heatmap PNG and CSV matrix.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        required=True,
        help="Comma-separated timestamps to aggregate over, e.g. 10,30,50,70,90",
    )
    parser.add_argument("--attr", type=str, default=None, help="single attr or comma list; default=all attrs")
    parser.add_argument(
        "--samples-per-timestamp",
        type=int,
        default=160000,
        help="Number of uniformly sampled XYZ coordinates per timestamp. Uses full grid when this exceeds XYZ size.",
    )
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g., cpu/cuda:0")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir containing target_*.npy")
    parser.add_argument("--dpi", type=int, default=200, help="output image dpi")
    parser.add_argument("--seed", type=int, default=0, help="random seed for coordinate sampling")
    args = parser.parse_args()

    if args.samples_per_timestamp <= 0:
        raise ValueError("--samples-per-timestamp must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    timestamps = _parse_timestamps(args.timestamps)
    outdir = Path(args.outdir)

    result = extract_router_topk_overlap(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        outdir=outdir,
        attr_arg=args.attr,
        timestamps=timestamps,
        samples_per_timestamp=int(args.samples_per_timestamp),
        batch_size=int(args.batch_size),
        device_str=args.device,
        train_dir_override=args.train_dir,
        dpi=int(args.dpi),
        seed=int(args.seed),
    )

    logger.info("Finished %s", config_path)
    logger.info("  Attrs: %s", result["attrs"])
    logger.info("  Timestamps: %s", result["timestamps"])
    logger.info("  Samples: %s", result["num_samples_total"])
    logger.info("  Top-K: %s", result["top_k"])
    logger.info("  PNG: %s", result["png_path"])
    logger.info("  CSV: %s", result["csv_path"])


if __name__ == "__main__":
    main()

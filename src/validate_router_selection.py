import argparse
import csv
import logging
import math
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
    _build_palette,
    _get_router_probs_for_batch,
    _get_view_names,
    _load_yaml,
    _make_xyz_coords_batch,
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


def _metric_ylabel(metric: str) -> str:
    if metric == "count":
        return "Selection Count (Top-3 hits)"
    if metric == "expected":
        return "Expected Selection (sum of router probs)"
    raise ValueError(f"Unknown metric: {metric}")


def _figure_title(exp_name: str, metric: str, timestamps: list[int], top_k: int) -> str:
    ts_text = ",".join(str(int(t)) for t in timestamps)
    if metric == "count":
        mode_text = f"mode=count (Top-{top_k} membership)"
    else:
        mode_text = "mode=expected (sum of raw router probs)"
    return f"{exp_name} | Router Selection by Attribute | {mode_text} | timestamps=[{ts_text}]"


def _aggregate_selection_stats(
    model,
    volume_shape,
    selected_attrs,
    metric: str,
    timestamps: list[int],
    batch_size: int,
    device: torch.device,
    normalize_inputs: bool,
) -> np.ndarray:
    sx = int(volume_shape.X)
    sy = int(volume_shape.Y)
    sz = int(volume_shape.Z)
    n_xyz = sx * sy * sz
    num_experts = int(getattr(model, "num_experts"))
    top_k = min(3, num_experts)
    mean_np, std_np = _build_input_norm_stats(volume_shape, normalize_inputs)

    if metric == "count":
        accum = np.zeros((len(selected_attrs), num_experts), dtype=np.int64)
    elif metric == "expected":
        accum = np.zeros((len(selected_attrs), num_experts), dtype=np.float64)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    with torch.no_grad():
        for t in timestamps:
            t0 = time.perf_counter()
            for start in range(0, n_xyz, batch_size):
                end = min(start + batch_size, n_xyz)
                coords = _make_xyz_coords_batch(start=start, end=end, sx=sx, sy=sy, t=int(t))
                coords_norm = ((coords - mean_np) / std_np).astype(np.float32)
                xb = torch.from_numpy(coords_norm).to(device, non_blocking=True)
                probs_by_attr = _get_router_probs_for_batch(
                    model=model,
                    xb=xb,
                    selected_attrs=selected_attrs,
                    device=device,
                )

                for attr_idx, attr_name in enumerate(selected_attrs):
                    probs_attr = probs_by_attr[attr_name]
                    if metric == "expected":
                        values = probs_attr.detach().cpu().numpy().astype(np.float64, copy=False)
                        accum[attr_idx, :] += values.sum(axis=0, dtype=np.float64)
                    else:
                        topk_idx = torch.topk(
                            probs_attr,
                            k=top_k,
                            dim=-1,
                            largest=True,
                            sorted=False,
                        ).indices.reshape(-1)
                        counts = torch.bincount(topk_idx.detach().cpu(), minlength=num_experts)
                        accum[attr_idx, :] += counts.detach().cpu().numpy().astype(np.int64, copy=False)

            logger.info(
                "Router selection aggregation: t=%d done in %.3fs",
                int(t),
                time.perf_counter() - t0,
            )

    return accum


def _validate_aggregate_totals(
    values_by_attr_expert: np.ndarray,
    metric: str,
    volume_shape,
    timestamps: list[int],
    top_k: int,
) -> None:
    expected_total = int(len(timestamps) * int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z))
    row_sums = values_by_attr_expert.sum(axis=1)

    if metric == "count":
        target = expected_total * int(top_k)
        if not np.all(row_sums == target):
            raise RuntimeError(
                f"Count invariant failed: each attr should sum to {target}, got {row_sums.tolist()}"
            )
        return

    atol = max(1e-6 * expected_total, 1.0)
    if not np.allclose(row_sums, float(expected_total), rtol=0.0, atol=atol):
        raise RuntimeError(
            "Expected invariant failed: each attr should sum to "
            f"{expected_total}, got {row_sums.tolist()} with atol={atol}"
        )


def _plot_selection_bars(
    values_by_attr_expert: np.ndarray,
    attrs: list[str],
    metric: str,
    timestamps: list[int],
    out_path: Path,
    exp_name: str,
    dpi: int,
) -> None:
    num_attrs = len(attrs)
    num_experts = int(values_by_attr_expert.shape[1])
    ncols = min(3, max(1, num_attrs))
    nrows = int(math.ceil(num_attrs / ncols))
    palette_hex = _build_palette(num_experts)
    expert_indices = np.arange(num_experts, dtype=np.int32)
    y_label = _metric_ylabel(metric)
    top_k = min(3, num_experts)

    fig_w = max(10.0, 4.5 * ncols)
    fig_h = max(5.0, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    y_max = float(np.max(values_by_attr_expert)) if values_by_attr_expert.size else 0.0
    y_upper = y_max * 1.08 if y_max > 0 else 1.0

    for ax, attr_name, values in zip(axes, attrs, values_by_attr_expert):
        ax.bar(expert_indices, values, color=palette_hex, width=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(str(attr_name))
        ax.set_xticks(expert_indices)
        ax.set_xlabel("Expert Index")
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

    for ax in axes[num_attrs:]:
        ax.set_visible(False)

    for ax in axes[:num_attrs]:
        ax.set_ylabel(y_label)

    fig.suptitle(_figure_title(exp_name=exp_name, metric=metric, timestamps=timestamps, top_k=top_k))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_selection_csv(
    values_by_attr_expert: np.ndarray,
    attrs: list[str],
    metric: str,
    timestamps: list[int],
    out_path: Path,
) -> None:
    fieldnames = ["metric", "attr", "expert_index", "value", "timestamps", "num_timestamps"]
    timestamps_text = ",".join(str(int(t)) for t in timestamps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for attr_name, values in zip(attrs, values_by_attr_expert):
            for expert_idx, value in enumerate(values):
                row_value = int(value) if metric == "count" else float(value)
                writer.writerow(
                    {
                        "metric": metric,
                        "attr": str(attr_name),
                        "expert_index": int(expert_idx),
                        "value": row_value,
                        "timestamps": timestamps_text,
                        "num_timestamps": int(len(timestamps)),
                    }
                )


def extract_router_selection(
    config_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    attr_arg: str | None,
    timestamps: list[int],
    metric: str,
    batch_size: int,
    device_str: str | None,
    train_dir_override: str | None,
    dpi: int,
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
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    view_names = _get_view_names(model)
    selected_attrs = _parse_requested_attrs(attr_arg, view_names)
    logger.info("[%s] Stage: prepare inference = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    values_by_attr_expert = _aggregate_selection_stats(
        model=model,
        volume_shape=volume_shape,
        selected_attrs=selected_attrs,
        metric=metric,
        timestamps=timestamps,
        batch_size=batch_size,
        device=device,
        normalize_inputs=normalize_inputs,
    )
    logger.info("[%s] Stage: aggregate selection = %.3fs", exp_name, time.perf_counter() - t_stage)

    num_experts = int(getattr(model, "num_experts"))
    top_k = min(3, num_experts)
    _validate_aggregate_totals(
        values_by_attr_expert=values_by_attr_expert,
        metric=metric,
        volume_shape=volume_shape,
        timestamps=timestamps,
        top_k=top_k,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    timestamps_tag = _format_timestamps_tag(timestamps)
    png_path = outdir / f"router_selection_{metric}_t{timestamps_tag}.png"
    csv_path = outdir / f"router_selection_{metric}_t{timestamps_tag}.csv"

    _plot_selection_bars(
        values_by_attr_expert=values_by_attr_expert.astype(np.float64, copy=False),
        attrs=selected_attrs,
        metric=metric,
        timestamps=timestamps,
        out_path=png_path,
        exp_name=exp_name,
        dpi=dpi,
    )
    _write_selection_csv(
        values_by_attr_expert=values_by_attr_expert,
        attrs=selected_attrs,
        metric=metric,
        timestamps=timestamps,
        out_path=csv_path,
    )

    logger.info("[%s] Stage: total = %.3fs", exp_name, time.perf_counter() - t_start)
    return {
        "png_path": png_path,
        "csv_path": csv_path,
        "attrs": list(selected_attrs),
        "timestamps": list(timestamps),
        "metric": metric,
        "values_by_attr_expert": values_by_attr_expert,
    }


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Aggregate router expert selection counts/expectations over selected timestamps."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out/router_selection",
        help="Output directory for PNG and CSV artifacts.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        required=True,
        help="Comma-separated timestamps to aggregate over, e.g. 10,30,50,70,90",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="count",
        choices=["count", "expected"],
        help="count: Top-3 membership counts; expected: sum of raw router probabilities",
    )
    parser.add_argument("--attr", type=str, default=None, help="single attr or comma list; default=all attrs")
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g., cpu/cuda:0")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir containing target_*.npy")
    parser.add_argument("--dpi", type=int, default=180, help="output image dpi")
    args = parser.parse_args()

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

    result = extract_router_selection(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        outdir=outdir,
        attr_arg=args.attr,
        timestamps=timestamps,
        metric=str(args.metric),
        batch_size=int(args.batch_size),
        device_str=args.device,
        train_dir_override=args.train_dir,
        dpi=int(args.dpi),
    )

    logger.info("Finished %s", config_path)
    logger.info("  Metric: %s", result["metric"])
    logger.info("  Attrs: %s", result["attrs"])
    logger.info("  Timestamps: %s", result["timestamps"])
    logger.info("  PNG: %s", result["png_path"])
    logger.info("  CSV: %s", result["csv_path"])


if __name__ == "__main__":
    main()

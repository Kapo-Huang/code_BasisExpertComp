import argparse
import colorsys
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap, Normalize, to_rgb
from matplotlib.patches import Patch

from inr.cli import build_model
from inr.datasets.base import (
    compute_input_stats_analytic,
    infer_or_validate_volume_shape,
    peek_array,
    target_dim_from_array,
)
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    out = str(name).strip()
    for ch in invalid:
        out = out.replace(ch, "_")
    out = out.replace(" ", "_")
    return out if out else "unknown"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _torch_load_checkpoint(path: Path):
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


def _extract_head_names(state_dict):
    names = set()
    for key in state_dict.keys():
        if not key.startswith("heads."):
            continue
        parts = key.split(".")
        if len(parts) >= 3:
            names.add(parts[1])
    return sorted(names)


def _resolve_train_dir(data_cfg, train_dir_override: str | None):
    if train_dir_override:
        return Path(train_dir_override)

    data_root = Path(data_cfg.get("data_root", "data"))
    dataset_name = data_cfg.get("dataset_name")
    if dataset_name:
        return data_root / "raw" / str(dataset_name) / "train"

    target_dir = data_cfg.get("target_dir")
    if target_dir:
        target_path = Path(target_dir)
        if target_path.name == "train":
            return target_path
        candidate = target_path.parent / "train"
        if candidate.exists():
            return candidate
        return target_path

    raise ValueError("Cannot resolve train directory. Provide --train-dir or data.dataset_name/data_root.")


def _collect_train_attr_paths(train_dir: Path):
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    paths = {}
    for p in sorted(train_dir.glob("target_*.npy")):
        name = p.stem[len("target_") :]
        paths[name] = str(p)
    if not paths:
        raise FileNotFoundError(f"No target_*.npy files found under: {train_dir}")
    return paths


def _filter_attr_paths_by_shape(attr_paths, volume_shape):
    if volume_shape is None:
        return dict(attr_paths), {}

    kept = {}
    dropped = {}
    for name, path in attr_paths.items():
        try:
            arr = peek_array(path)
            infer_or_validate_volume_shape(arr, volume_shape)
            kept[name] = path
        except Exception as exc:
            dropped[name] = str(exc)
    return kept, dropped


def _infer_volume_shape(data_cfg, attr_paths):
    volume_shape = data_cfg.get("volume_shape") or data_cfg.get("volume_dims")
    first_path = next(iter(attr_paths.values()))
    first = peek_array(first_path)
    return infer_or_validate_volume_shape(first, volume_shape)


def _build_view_specs(attr_paths, volume_shape):
    specs = {}
    for name, path in attr_paths.items():
        arr = peek_array(path)
        infer_or_validate_volume_shape(arr, volume_shape)
        specs[name] = int(target_dim_from_array(arr))
    return specs


class _ViewSpecDataset:
    def __init__(self, view_specs):
        self._view_specs = dict(view_specs)

    def view_specs(self):
        return dict(self._view_specs)


def _normalize_model_cfg_for_volume(model_cfg):
    local = dict(model_cfg)
    if int(local.get("in_features", 4)) != 4:
        local["in_features"] = 4
    return local


def _validate_train_vs_checkpoint_attrs(train_attrs, ckpt_head_attrs):
    if not ckpt_head_attrs:
        return
    train_set = set(train_attrs)
    head_set = set(ckpt_head_attrs)
    if train_set != head_set:
        raise ValueError(
            "Attribute mismatch between train files and checkpoint heads.\n"
            f"  train attrs: {sorted(train_set)}\n"
            f"  ckpt heads: {sorted(head_set)}"
        )


def _parse_requested_attrs(attr_arg: str | None, available_attrs):
    if not attr_arg:
        return list(available_attrs)
    requested = [x.strip() for x in str(attr_arg).split(",") if x.strip()]
    if not requested:
        return list(available_attrs)
    unknown = [a for a in requested if a not in available_attrs]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {list(available_attrs)}")
    return requested


def _build_center_slices_info(volume_shape, time_index: int):
    sx = int(volume_shape.X)
    sy = int(volume_shape.Y)
    sz = int(volume_shape.Z)
    st = int(volume_shape.T)
    if time_index < 0 or time_index >= st:
        raise ValueError(f"--time-index out of range: {time_index}, valid [0, {st - 1}]")
    return {
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "st": st,
        "t": int(time_index),
        "mx": int(sx // 1.5),
        "my": int(sy // 2),
        "mz": int(sz // 2),
        "shape_x": (sz, sy),  # (z, y)
        "shape_y": (sz, sx),  # (z, x)
        "shape_z": (sy, sx),  # (y, x)
    }


def _build_slice_coords(info):
    sx = int(info["sx"])
    sy = int(info["sy"])
    sz = int(info["sz"])
    t = int(info["t"])
    mx = int(info["mx"])
    my = int(info["my"])
    mz = int(info["mz"])

    z = np.repeat(np.arange(sz, dtype=np.float32), sy)
    y = np.tile(np.arange(sy, dtype=np.float32), sz)
    x_const = np.full_like(z, float(mx), dtype=np.float32)
    t_const = np.full_like(z, float(t), dtype=np.float32)
    coords_x = np.stack([x_const, y, z, t_const], axis=1)

    z = np.repeat(np.arange(sz, dtype=np.float32), sx)
    x = np.tile(np.arange(sx, dtype=np.float32), sz)
    y_const = np.full_like(z, float(my), dtype=np.float32)
    t_const = np.full_like(z, float(t), dtype=np.float32)
    coords_y = np.stack([x, y_const, z, t_const], axis=1)

    y = np.repeat(np.arange(sy, dtype=np.float32), sx)
    x = np.tile(np.arange(sx, dtype=np.float32), sy)
    z_const = np.full_like(y, float(mz), dtype=np.float32)
    t_const = np.full_like(y, float(t), dtype=np.float32)
    coords_z = np.stack([x, y, z_const, t_const], axis=1)

    return {"x": coords_x, "y": coords_y, "z": coords_z}


def _normalize_coords(coords_by_plane, volume_shape, normalize_inputs: bool):
    if not normalize_inputs:
        return {k: v.astype(np.float32) for k, v in coords_by_plane.items()}
    x_mean, x_std = compute_input_stats_analytic(
        volume_shape,
        unbiased=True,
        dtype=np.float64,
        eps=1e-12,
    )
    mean_np = x_mean.squeeze(0).numpy().astype(np.float32)
    std_np = x_std.squeeze(0).numpy().astype(np.float32)
    return {k: ((v - mean_np) / std_np).astype(np.float32) for k, v in coords_by_plane.items()}


def _build_input_norm_stats(volume_shape, normalize_inputs: bool):
    if not normalize_inputs:
        return (
            np.zeros((4,), dtype=np.float32),
            np.ones((4,), dtype=np.float32),
        )
    x_mean, x_std = compute_input_stats_analytic(
        volume_shape,
        unbiased=True,
        dtype=np.float64,
        eps=1e-12,
    )
    mean_np = x_mean.squeeze(0).numpy().astype(np.float32)
    std_np = x_std.squeeze(0).numpy().astype(np.float32)
    return mean_np, std_np


def _get_view_names(model):
    if hasattr(model, "view_names"):
        return list(model.view_names)
    raise ValueError("Model has no view_names; cannot map router probs to attributes.")


def _get_router_probs_for_planes(model, coords_norm_by_plane, batch_size: int, device: torch.device):
    view_names = _get_view_names(model)
    probs_by_plane = {}

    with torch.no_grad():
        for plane_name, coords_norm in coords_norm_by_plane.items():
            probs_chunks = {name: [] for name in view_names}
            total = int(coords_norm.shape[0])
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                xb = torch.from_numpy(coords_norm[start:end]).to(device, non_blocking=True)

                if hasattr(model, "gating") and hasattr(model, "view_embedding"):
                    router_in = model.pos_enc(xb) if hasattr(model, "pos_enc") else xb
                    for view_idx, name in enumerate(view_names):
                        view_ids = torch.full((xb.shape[0],), view_idx, device=device, dtype=torch.long)
                        view_embed = model.view_embedding(view_ids)
                        probs, _ = model.gating(router_in, view_embed)
                        probs_chunks[name].append(probs.cpu())
                else:
                    out = model(xb, return_aux=True, hard_topk=False)
                    if not isinstance(out, (tuple, list)) or len(out) < 2:
                        raise ValueError("Model forward(return_aux=True) did not return aux.")
                    aux = out[1]
                    probs_all = aux.get("probs")
                    if probs_all is None:
                        raise ValueError("Aux has no 'probs'.")
                    for view_idx, name in enumerate(view_names):
                        probs_chunks[name].append(probs_all[:, view_idx, :].cpu())
            probs_by_plane[plane_name] = {name: torch.cat(chunks, dim=0) for name, chunks in probs_chunks.items()}

    return probs_by_plane


def _get_router_probs_for_batch(
    model,
    xb: torch.Tensor,
    selected_attrs,
    device: torch.device,
):
    view_names = _get_view_names(model)
    view_to_idx = {name: i for i, name in enumerate(view_names)}
    missing = [name for name in selected_attrs if name not in view_to_idx]
    if missing:
        raise KeyError(f"Missing attrs in model views: {missing}")

    if hasattr(model, "gating") and hasattr(model, "view_embedding"):
        router_in = model.pos_enc(xb) if hasattr(model, "pos_enc") else xb
        out = {}
        for attr_name in selected_attrs:
            view_idx = int(view_to_idx[attr_name])
            view_ids = torch.full((xb.shape[0],), view_idx, device=device, dtype=torch.long)
            view_embed = model.view_embedding(view_ids)
            probs, _ = model.gating(router_in, view_embed)
            out[attr_name] = probs
        return out

    out = model(xb, return_aux=True, hard_topk=False)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("Model forward(return_aux=True) did not return aux.")
    aux = out[1]
    probs_all = aux.get("probs")
    if probs_all is None:
        raise ValueError("Aux has no 'probs'.")
    return {attr_name: probs_all[:, int(view_to_idx[attr_name]), :] for attr_name in selected_attrs}


def _make_xyz_coords_batch(start: int, end: int, sx: int, sy: int, t: int):
    idx = np.arange(start, end, dtype=np.int64)
    x = (idx % sx).astype(np.float32)
    yz = idx // sx
    y = (yz % sy).astype(np.float32)
    z = (yz // sy).astype(np.float32)
    t_arr = np.full_like(x, float(t), dtype=np.float32)
    return np.stack([x, y, z, t_arr], axis=1)


def _compute_expected_selection_over_time(
    model,
    volume_shape,
    selected_attrs,
    num_experts: int,
    batch_size: int,
    device: torch.device,
    normalize_inputs: bool,
):
    sx = int(volume_shape.X)
    sy = int(volume_shape.Y)
    sz = int(volume_shape.Z)
    st = int(volume_shape.T)
    n_xyz = sx * sy * sz
    mean_np, std_np = _build_input_norm_stats(volume_shape, normalize_inputs)
    expected = np.zeros((len(selected_attrs), st, num_experts), dtype=np.float64)

    with torch.no_grad():
        for t in range(st):
            t0 = time.perf_counter()
            accum = np.zeros((len(selected_attrs), num_experts), dtype=np.float64)

            for start in range(0, n_xyz, batch_size):
                end = min(start + batch_size, n_xyz)
                coords = _make_xyz_coords_batch(start=start, end=end, sx=sx, sy=sy, t=t)
                coords_norm = ((coords - mean_np) / std_np).astype(np.float32)
                xb = torch.from_numpy(coords_norm).to(device, non_blocking=True)
                probs_by_attr = _get_router_probs_for_batch(
                    model=model,
                    xb=xb,
                    selected_attrs=selected_attrs,
                    device=device,
                )
                for attr_idx, attr_name in enumerate(selected_attrs):
                    accum[attr_idx, :] += (
                        probs_by_attr[attr_name].sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                    )

            expected[:, t, :] = accum
            logger.info(
                "Router temporal aggregation: t=%d/%d done in %.3fs",
                t,
                st - 1,
                time.perf_counter() - t0,
            )

    return expected


def _build_palette(num_experts: int):
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if num_experts <= len(base):
        colors = base[:num_experts]
    else:
        colors = []
        for i in range(num_experts):
            h = (i / max(1, num_experts)) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
            colors.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _plot_rank_expert_map(
    rank_maps: dict,
    num_experts: int,
    palette_hex,
    out_path: Path,
    title: str,
    info: dict,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    cmap = ListedColormap(palette_hex)
    panels = [
        ("x", f"YZ @ x={info['mx']}", "Y", "Z"),
        ("y", f"XZ @ y={info['my']}", "X", "Z"),
        ("z", f"XY @ z={info['mz']}", "X", "Y"),
    ]
    im = None
    for ax, (plane, sub_title, x_label, y_label) in zip(axes, panels):
        im = ax.imshow(
            rank_maps[plane],
            origin="lower",
            cmap=cmap,
            vmin=-0.5,
            vmax=num_experts - 0.5,
            interpolation="nearest",
        )
        ax.set_title(sub_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    ticks = np.arange(num_experts)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=ticks, shrink=0.95)
    cbar.ax.set_yticklabels([f"E{i}" for i in range(num_experts)])
    cbar.set_label("Expert ID")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _compose_rank_weight_rgb(
    rank_map: np.ndarray,
    weight_map: np.ndarray,
    palette_hex,
    weight_floor: float,
    weight_gamma: float,
):
    palette_rgb = np.array([to_rgb(c) for c in palette_hex], dtype=np.float32)
    base = palette_rgb[rank_map.astype(np.int64)]
    w = np.asarray(weight_map, dtype=np.float32)
    w = np.clip(w, 0.0, 1.0)
    gamma = max(1e-6, float(weight_gamma))
    intensity = np.power(w, gamma)
    floor = float(np.clip(weight_floor, 0.0, 1.0))
    intensity = floor + (1.0 - floor) * intensity
    rgb = (1.0 - intensity[..., None]) * 1.0 + intensity[..., None] * base
    return np.clip(rgb, 0.0, 1.0)


def _plot_rank_weighted_expert_map(
    rank_maps: dict,
    weight_maps: dict,
    num_experts: int,
    palette_hex,
    out_path: Path,
    title: str,
    info: dict,
    weight_floor: float,
    weight_gamma: float,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("x", f"YZ @ x={info['mx']}", "Y", "Z"),
        ("y", f"XZ @ y={info['my']}", "X", "Z"),
        ("z", f"XY @ z={info['mz']}", "X", "Y"),
    ]
    for ax, (plane, sub_title, x_label, y_label) in zip(axes, panels):
        rgb = _compose_rank_weight_rgb(
            rank_map=rank_maps[plane],
            weight_map=weight_maps[plane],
            palette_hex=palette_hex,
            weight_floor=weight_floor,
            weight_gamma=weight_gamma,
        )
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(sub_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    legend_handles = [
        Patch(facecolor=palette_hex[i], edgecolor="none", label=f"E{i}")
        for i in range(num_experts)
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        title="Experts",
        fontsize=8,
    )

    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="Greys")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Rank Weight")

    axes[-1].text(
        1.02,
        0.0,
        f"Encoding:\nHue=expert ID\nDepth=weight\nfloor={weight_floor:.2f}\ngamma={weight_gamma:.2f}",
        transform=axes[-1].transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
    )

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _compose_topk_overlay(topk_maps, palette_hex, overlay_alpha: float, overlay_decay: float):
    palette_rgb = np.array([to_rgb(c) for c in palette_hex], dtype=np.float32)
    rgb = palette_rgb[topk_maps[0].astype(np.int64)]
    for r in range(1, len(topk_maps)):
        alpha_r = float(overlay_alpha) * (float(overlay_decay) ** float(r - 1))
        alpha_r = max(0.0, min(1.0, alpha_r))
        overlay_rgb = palette_rgb[topk_maps[r].astype(np.int64)]
        rgb = (1.0 - alpha_r) * rgb + alpha_r * overlay_rgb
    return np.clip(rgb, 0.0, 1.0)


def _plot_topk_coverage(
    topk_maps_by_plane,
    num_experts: int,
    palette_hex,
    out_path: Path,
    title: str,
    info: dict,
    overlay_alpha: float,
    overlay_decay: float,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("x", f"YZ @ x={info['mx']}", "Y", "Z"),
        ("y", f"XZ @ y={info['my']}", "X", "Z"),
        ("z", f"XY @ z={info['mz']}", "X", "Y"),
    ]
    for ax, (plane, sub_title, x_label, y_label) in zip(axes, panels):
        rgb = _compose_topk_overlay(topk_maps_by_plane[plane], palette_hex, overlay_alpha, overlay_decay)
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(sub_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    legend_handles = [
        Patch(facecolor=palette_hex[i], edgecolor="none", label=f"E{i}")
        for i in range(num_experts)
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        title="Experts",
        fontsize=8,
    )

    info_lines = ["Blend: Top-1 base"]
    for r in range(2, len(topk_maps_by_plane["x"]) + 1):
        alpha_r = float(overlay_alpha) * (float(overlay_decay) ** float(r - 2))
        info_lines.append(f"Top-{r} alpha={alpha_r:.3f}")
    axes[-1].text(
        1.02,
        0.0,
        "\n".join(info_lines),
        transform=axes[-1].transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
    )

    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_map(
    entropy_maps: dict,
    num_experts: int,
    out_path: Path,
    title: str,
    info: dict,
    entropy_cmap: str,
    dpi: int,
):
    vmax = float(np.log(max(2, num_experts)))
    vmin = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("x", f"YZ @ x={info['mx']}", "Y", "Z"),
        ("y", f"XZ @ y={info['my']}", "X", "Z"),
        ("z", f"XY @ z={info['mz']}", "X", "Y"),
    ]
    im = None
    for ax, (plane, sub_title, x_label, y_label) in zip(axes, panels):
        im = ax.imshow(
            entropy_maps[plane],
            origin="lower",
            cmap=entropy_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(sub_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Router Entropy")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_expert_time_curve(
    values: np.ndarray,
    out_path: Path,
    title: str,
    expert_idx: int,
    color: str,
    dpi: int,
):
    t = np.arange(values.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    ax.plot(t, values, color=color, linewidth=2.0)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Expected Selection (sum of router probs over XYZ)")
    ax.set_title(f"Expert {expert_idx}")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_all_experts_time_curve_log(
    values_by_time_expert: np.ndarray,
    out_path: Path,
    title: str,
    palette_hex,
    dpi: int,
):
    if values_by_time_expert.ndim != 2:
        raise ValueError(
            f"values_by_time_expert must be 2D (T, M), got {tuple(values_by_time_expert.shape)}"
        )

    t = np.arange(values_by_time_expert.shape[0], dtype=np.int32)
    num_experts = int(values_by_time_expert.shape[1])
    eps = 1e-12

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    for expert_idx in range(num_experts):
        v = np.asarray(values_by_time_expert[:, expert_idx], dtype=np.float64)
        v = np.clip(v, eps, None)
        color = palette_hex[expert_idx % len(palette_hex)]
        ax.plot(t, v, linewidth=1.8, color=color, label=f"E{expert_idx}")

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Expected Selection (log scale)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_expected_selection_npy(
    values_by_time_expert: np.ndarray,
    out_path: Path,
):
    if values_by_time_expert.ndim != 2:
        raise ValueError(
            f"values_by_time_expert must be 2D (T, M), got {tuple(values_by_time_expert.shape)}"
        )

    t_len = int(values_by_time_expert.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_col = np.arange(t_len, dtype=np.float64).reshape(-1, 1)
    payload = np.concatenate([time_col, values_by_time_expert.astype(np.float64, copy=False)], axis=1)
    np.save(str(out_path), payload)


def _compute_topk_and_entropy(probs_attr: torch.Tensor, k: int):
    if probs_attr.dim() != 2:
        raise ValueError(f"probs_attr must be 2D (N, M), got {tuple(probs_attr.shape)}")
    m = int(probs_attr.shape[1])
    k = min(max(1, int(k)), m)
    topk_out = torch.topk(probs_attr, k=k, dim=-1, largest=True, sorted=True)
    topk = topk_out.indices  # (N, K)
    topk_weights = topk_out.values  # (N, K)
    eps = 1e-9
    entropy = -(probs_attr * (probs_attr + eps).log()).sum(dim=-1)  # (N,)
    return topk, topk_weights, entropy


def _build_model_and_inputs(cfg, checkpoint, train_dir: Path):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    attr_paths = _collect_train_attr_paths(train_dir)
    attr_paths, dropped = _filter_attr_paths_by_shape(
        attr_paths,
        data_cfg.get("volume_shape") or data_cfg.get("volume_dims"),
    )
    if dropped:
        logger.warning("Dropped by shape mismatch: %s", sorted(dropped.keys()))
    if not attr_paths:
        raise ValueError("No valid train attr_paths after volume shape filtering.")
    volume_shape = _infer_volume_shape(data_cfg, attr_paths)
    view_specs = _build_view_specs(attr_paths, volume_shape)

    head_names = _extract_head_names(checkpoint["model_state"])
    _validate_train_vs_checkpoint_attrs(view_specs.keys(), head_names)

    dataset_stub = _ViewSpecDataset(view_specs)
    model_cfg_local = _normalize_model_cfg_for_volume(model_cfg)
    model = build_model(model_cfg_local, dataset_stub)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, volume_shape, view_specs


def _resolve_run_name(cfg, config_path: Path, run_timestamp: str):
    exp_id = str(cfg.get("exp_id", "")).strip()
    base = exp_id if exp_id else config_path.stem
    return f"{_safe_name(base)}_{run_timestamp}"


def _resolve_output_dir(
    outdir: Path,
    task: str,
    slice_info: dict | None,
):
    if task == "time-curves":
        return outdir / "time_curves"
    if slice_info is None:
        raise ValueError("slice_info is required for slices/all tasks.")
    return outdir / str(int(slice_info["t"]))


def extract_router_distribution(
    config_path: Path,
    checkpoint_path: Path,
    outdir: Path,
    run_timestamp: str,
    attr_arg: str | None,
    time_index: int,
    task: str,
    top_k_override: int | None,
    batch_size: int,
    device_str: str | None,
    train_dir_override: str | None,
    overlay_alpha: float,
    overlay_decay: float,
    weight_floor: float,
    weight_gamma: float,
    entropy_cmap: str,
    dpi: int,
):
    t_start = time.perf_counter()
    cfg = _load_yaml(config_path)
    exp_name = _resolve_run_name(cfg, config_path, run_timestamp)
    logger.info("[%s] Stage: load config = %.3fs", exp_name, time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if "model_state" not in checkpoint:
        raise KeyError(f"'model_state' missing in checkpoint: {checkpoint_path}")
    logger.info("[%s] Stage: load checkpoint = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    train_dir = _resolve_train_dir(cfg["data"], train_dir_override)
    model, volume_shape, view_specs = _build_model_and_inputs(cfg, checkpoint, train_dir)
    logger.info("[%s] Stage: build model/input specs = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    normalize_inputs = bool(cfg["data"].get("normalize_inputs", cfg["data"].get("normalize", True)))
    generate_slices = task in {"slices", "all"}
    generate_time_curves = task in {"time-curves", "all"}
    if not generate_slices and not generate_time_curves:
        raise ValueError(f"Unknown task: {task}")
    slice_info = None
    coords_norm = None
    if generate_slices:
        slice_info = _build_center_slices_info(volume_shape, time_index)
        coords_by_plane = _build_slice_coords(slice_info)
        coords_norm = _normalize_coords(coords_by_plane, volume_shape, normalize_inputs)
    logger.info("[%s] Stage: prepare routing inputs = %.3fs", exp_name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    probs_dict = None
    if generate_slices:
        probs_dict = _get_router_probs_for_planes(model, coords_norm, batch_size=batch_size, device=device)
    logger.info("[%s] Stage: router slice inference = %.3fs", exp_name, time.perf_counter() - t_stage)

    num_experts = int(getattr(model, "num_experts"))
    top_k_model = int(getattr(model, "top_k", min(2, num_experts)))
    top_k = int(top_k_override) if top_k_override is not None else top_k_model
    top_k = max(1, min(top_k, num_experts))

    view_names = _get_view_names(model)
    selected_attrs = _parse_requested_attrs(attr_arg, view_names)
    if generate_slices:
        missing_probs = [name for name in selected_attrs if name not in probs_dict["x"]]
        if missing_probs:
            raise KeyError(f"Missing probs for attrs: {missing_probs}")

    expected_over_time = None
    if generate_time_curves:
        t_stage = time.perf_counter()
        expected_over_time = _compute_expected_selection_over_time(
            model=model,
            volume_shape=volume_shape,
            selected_attrs=selected_attrs,
            num_experts=num_experts,
            batch_size=batch_size,
            device=device,
            normalize_inputs=normalize_inputs,
        )
        logger.info("[%s] Stage: temporal expected selection = %.3fs", exp_name, time.perf_counter() - t_stage)
    attr_to_idx = {name: i for i, name in enumerate(selected_attrs)}

    palette_hex = _build_palette(num_experts)
    run_dir = _resolve_output_dir(outdir=outdir, task=task, slice_info=slice_info)
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for attr_name in selected_attrs:
        t_attr = time.perf_counter()
        attr_safe = _safe_name(attr_name)

        rank_paths = []
        rank_weight_paths = []
        coverage_path = None
        entropy_path = None
        if generate_slices:
            topk_maps_by_plane = {}
            topk_weights_by_plane = {}
            entropy_maps = {}
            for plane_name in ("x", "y", "z"):
                probs_attr = probs_dict[plane_name][attr_name].to(torch.float32)  # (N, M)
                topk_idx, topk_weights, entropy = _compute_topk_and_entropy(probs_attr, k=top_k)
                shape_2d = tuple(slice_info[f"shape_{plane_name}"])
                topk_maps_by_plane[plane_name] = [
                    topk_idx[:, rank].reshape(shape_2d).cpu().numpy()
                    for rank in range(top_k)
                ]
                topk_weights_by_plane[plane_name] = [
                    topk_weights[:, rank].reshape(shape_2d).cpu().numpy()
                    for rank in range(top_k)
                ]
                entropy_maps[plane_name] = entropy.reshape(shape_2d).cpu().numpy()

            for rank in range(top_k):
                rank_num = rank + 1
                rank_path = (
                    run_dir
                    / f"{attr_safe}_top{int(rank_num):02d}_expert_map_orth.png"
                )
                title = (
                    f"{exp_name} | attr={attr_name} | Top-{rank_num} Expert Map | "
                    f"t={time_index} | (mx,my,mz)=({slice_info['mx']},{slice_info['my']},{slice_info['mz']})"
                )
                _plot_rank_expert_map(
                    rank_maps={
                        "x": topk_maps_by_plane["x"][rank],
                        "y": topk_maps_by_plane["y"][rank],
                        "z": topk_maps_by_plane["z"][rank],
                    },
                    num_experts=num_experts,
                    palette_hex=palette_hex,
                    out_path=rank_path,
                    title=title,
                    info=slice_info,
                    dpi=dpi,
                )
                rank_paths.append(rank_path)

                rank_weight_path = (
                    run_dir
                    / f"{attr_safe}_top{int(rank_num):02d}_expert_weight_orth.png"
                )
                rank_weight_title = (
                    f"{exp_name} | attr={attr_name} | Top-{rank_num} Expert-Weight Heatmap | "
                    f"t={time_index} | (mx,my,mz)=({slice_info['mx']},{slice_info['my']},{slice_info['mz']})"
                )
                _plot_rank_weighted_expert_map(
                    rank_maps={
                        "x": topk_maps_by_plane["x"][rank],
                        "y": topk_maps_by_plane["y"][rank],
                        "z": topk_maps_by_plane["z"][rank],
                    },
                    weight_maps={
                        "x": topk_weights_by_plane["x"][rank],
                        "y": topk_weights_by_plane["y"][rank],
                        "z": topk_weights_by_plane["z"][rank],
                    },
                    num_experts=num_experts,
                    palette_hex=palette_hex,
                    out_path=rank_weight_path,
                    title=rank_weight_title,
                    info=slice_info,
                    weight_floor=weight_floor,
                    weight_gamma=weight_gamma,
                    dpi=dpi,
                )
                rank_weight_paths.append(rank_weight_path)

            coverage_path = (
                run_dir
                / f"{attr_safe}_top{int(top_k)}_coverage_orth.png"
            )
            coverage_title = (
                f"{exp_name} | attr={attr_name} | Top-{top_k} Coverage Overlay | "
                f"t={time_index} | (mx,my,mz)=({slice_info['mx']},{slice_info['my']},{slice_info['mz']})"
            )
            _plot_topk_coverage(
                topk_maps_by_plane=topk_maps_by_plane,
                num_experts=num_experts,
                palette_hex=palette_hex,
                out_path=coverage_path,
                title=coverage_title,
                info=slice_info,
                overlay_alpha=overlay_alpha,
                overlay_decay=overlay_decay,
                dpi=dpi,
            )

            entropy_path = (
                run_dir
                / f"{attr_safe}_entropy_orth.png"
            )
            entropy_title = (
                f"{exp_name} | attr={attr_name} | Router Entropy | "
                f"t={time_index} | (mx,my,mz)=({slice_info['mx']},{slice_info['my']},{slice_info['mz']})"
            )
            _plot_entropy_map(
                entropy_maps=entropy_maps,
                num_experts=num_experts,
                out_path=entropy_path,
                title=entropy_title,
                info=slice_info,
                entropy_cmap=entropy_cmap,
                dpi=dpi,
            )

        time_curve_paths = []
        time_curve_combined_path = None
        time_curve_npy_path = None
        if generate_time_curves:
            attr_idx = int(attr_to_idx[attr_name])
            values_te = np.asarray(expected_over_time[attr_idx], dtype=np.float64)  # (T, M)
            for expert_idx in range(num_experts):
                curve_path = run_dir / f"{attr_safe}_expert_{int(expert_idx):02d}_expected_selection_time.png"
                curve_title = (
                    f"{exp_name} | attr={attr_name} | Expert {expert_idx} | "
                    "Expected Router Selection vs Time"
                )
                _plot_expert_time_curve(
                    values=values_te[:, expert_idx],
                    out_path=curve_path,
                    title=curve_title,
                    expert_idx=expert_idx,
                    color=palette_hex[expert_idx],
                    dpi=dpi,
                )
                time_curve_paths.append(curve_path)

            time_curve_combined_path = run_dir / f"{attr_safe}_experts_expected_selection_time_log.png"
            time_curve_combined_title = (
                f"{exp_name} | attr={attr_name} | All Experts | "
                "Expected Router Selection vs Time (log y)"
            )
            _plot_all_experts_time_curve_log(
                values_by_time_expert=values_te,
                out_path=time_curve_combined_path,
                title=time_curve_combined_title,
                palette_hex=palette_hex,
                dpi=dpi,
            )

            time_curve_npy_path = run_dir / f"{attr_safe}_expected_selection_time.npy"
            _save_expected_selection_npy(
                values_by_time_expert=values_te,
                out_path=time_curve_npy_path,
            )

        outputs.append(
            {
                "attr": attr_name,
                "top_rank_paths": rank_paths,
                "top_rank_weight_paths": rank_weight_paths,
                "coverage_path": coverage_path,
                "entropy_path": entropy_path,
                "time_curve_paths": time_curve_paths,
                "time_curve_combined_path": time_curve_combined_path,
                "time_curve_npy_path": time_curve_npy_path,
            }
        )
        logger.info("[%s] Attr %s done = %.3fs", exp_name, attr_name, time.perf_counter() - t_attr)

    logger.info("[%s] Stage: total = %.3fs", exp_name, time.perf_counter() - t_start)
    return outputs


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Visualize basis expert router distribution (slices/time-curves/all)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out",
        help="base output directory; slices/all write to outdir/<time-index>, time-curves write to outdir/time_curves",
    )
    parser.add_argument("--attr", type=str, default=None, help="single attr or comma list; default=all attrs")
    parser.add_argument(
        "--task",
        type=str,
        default="slices",
        choices=["slices", "time-curves", "all"],
        help="slices: only spatial top-k/coverage/entropy; time-curves: only per-expert curves over time; all: generate both",
    )
    parser.add_argument("--timestamp", type=int, default=0, help="time index t (only used for task=slices/all)")
    parser.add_argument("--top-k", type=int, default=None, help="top-k rank count to visualize; default=model top_k")
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g., cpu/cuda:0")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir containing target_*.npy")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.35,
        help="alpha for Top-2 overlay on Top-1 base (Top-3+ use decay)",
    )
    parser.add_argument(
        "--overlay-decay",
        type=float,
        default=0.6,
        help="decay factor for deeper ranks: alpha_r = overlay_alpha * overlay_decay^(r-2)",
    )
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=0.15,
        help="minimum color intensity for weighted top-k map (0~1)",
    )
    parser.add_argument(
        "--weight-gamma",
        type=float,
        default=1.0,
        help="gamma for weighted top-k map intensity ( >0 )",
    )
    parser.add_argument("--entropy-cmap", type=str, default="magma", help="colormap for entropy heatmap")
    parser.add_argument("--dpi", type=int, default=180, help="output image dpi")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be > 0 when provided")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if args.overlay_alpha < 0 or args.overlay_alpha > 1:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if args.overlay_decay < 0 or args.overlay_decay > 1:
        raise ValueError("--overlay-decay must be in [0, 1]")
    if args.weight_floor < 0 or args.weight_floor > 1:
        raise ValueError("--weight-floor must be in [0, 1]")
    if args.weight_gamma <= 0:
        raise ValueError("--weight-gamma must be > 0")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        outputs = extract_router_distribution(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            outdir=outdir,
            run_timestamp=run_timestamp,
            attr_arg=args.attr,
            time_index=int(args.timestamp),
            task=str(args.task),
            top_k_override=args.top_k,
            batch_size=int(args.batch_size),
            device_str=args.device,
            train_dir_override=args.train_dir,
            overlay_alpha=float(args.overlay_alpha),
            overlay_decay=float(args.overlay_decay),
            weight_floor=float(args.weight_floor),
            weight_gamma=float(args.weight_gamma),
            entropy_cmap=args.entropy_cmap,
            dpi=int(args.dpi),
        )
    except Exception as exc:
        logger.exception("Failed: %s (%s)", config_path, exc)
        raise

    logger.info("Finished %s", config_path)
    for item in outputs:
        logger.info("  Attr: %s", item["attr"])
        for idx, p in enumerate(item.get("top_rank_paths", []), start=1):
            logger.info("    Top-%s expert map: %s", idx, p)
        for idx, p in enumerate(item.get("top_rank_weight_paths", []), start=1):
            logger.info("    Top-%s expert weight heatmap: %s", idx, p)
        if item.get("coverage_path") is not None:
            logger.info("    Top-k coverage: %s", item["coverage_path"])
        if item.get("entropy_path") is not None:
            logger.info("    Entropy: %s", item["entropy_path"])
        for idx, p in enumerate(item.get("time_curve_paths", [])):
            logger.info("    Expert %s expected-vs-time: %s", idx, p)
        if item.get("time_curve_combined_path") is not None:
            logger.info("    All-expert expected-vs-time (log y): %s", item["time_curve_combined_path"])
        if item.get("time_curve_npy_path") is not None:
            logger.info("    Time stats npy: %s", item["time_curve_npy_path"])


if __name__ == "__main__":
    main()

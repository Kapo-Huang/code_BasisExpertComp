import argparse
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

from inr.cli import build_model
from inr.data import MultiTargetVolumetricDataset
from inr.datasets.base import infer_or_validate_volume_shape, parse_volume_shape, peek_array
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


def _match_checkpoint(epoch: int, ckpt_files):
    if not ckpt_files:
        return None
    token = f"epoch{epoch}"
    for ckpt in ckpt_files:
        if token in ckpt.name:
            return ckpt
    return None


def _collect_experiments(exp_root: Path):
    exps = []
    for child in sorted(exp_root.iterdir()):
        if not child.is_dir():
            continue
        cfg_path = child / "configs" / "config.yaml"
        if cfg_path.exists():
            exps.append(child)
    return exps


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
    parsed_shape = parse_volume_shape(volume_shape)
    if parsed_shape is None:
        return dict(attr_paths), {}

    kept = {}
    dropped = {}
    for name, path in attr_paths.items():
        try:
            arr = peek_array(path)
            infer_or_validate_volume_shape(arr, parsed_shape)
            kept[name] = path
        except Exception as exc:
            dropped[name] = str(exc)
    return kept, dropped


def _parse_requested_attrs(attr_arg: str | None, available_attrs):
    if not attr_arg:
        return list(sorted(available_attrs))
    items = [s.strip() for s in str(attr_arg).split(",") if s.strip()]
    if not items:
        return list(sorted(available_attrs))
    unknown = [name for name in items if name not in available_attrs]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {sorted(available_attrs)}")
    return items


def _resolve_denorm_stats(dataset, checkpoint, attr_name: str, channel: int):
    y_mean = checkpoint.get("y_mean")
    y_std = checkpoint.get("y_std")
    mean_arr = None
    std_arr = None

    if isinstance(y_mean, dict) and isinstance(y_std, dict):
        if attr_name in y_mean and attr_name in y_std:
            mean_arr = np.asarray(y_mean[attr_name]).reshape(-1)
            std_arr = np.asarray(y_std[attr_name]).reshape(-1)
    elif y_mean is not None and y_std is not None and len(dataset.view_specs()) == 1:
        mean_arr = np.asarray(y_mean).reshape(-1)
        std_arr = np.asarray(y_std).reshape(-1)

    if mean_arr is None or std_arr is None:
        dataset._ensure_target_stats(attr_name)
        mean_arr = dataset.y_mean[attr_name].reshape(-1).cpu().numpy()
        std_arr = dataset.y_std[attr_name].reshape(-1).cpu().numpy()

    if channel < 0 or channel >= mean_arr.shape[0]:
        raise ValueError(
            f"--channel out of range for '{attr_name}': {channel}, valid [0, {mean_arr.shape[0] - 1}]"
        )
    return float(mean_arr[channel]), float(std_arr[channel])


def _build_center_slices_info(volume_shape, t_idx: int):
    sx = int(volume_shape.X)
    sy = int(volume_shape.Y)
    sz = int(volume_shape.Z)
    st = int(volume_shape.T)
    if t_idx < 0 or t_idx >= st:
        raise ValueError(f"--time-index out of range: {t_idx}, valid [0, {st - 1}]")
    return {
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "st": st,
        "t": int(t_idx),
        "mx": int(sx // 1.5),
        "my": sy // 2,
        "mz": sz // 2,
        "shape_x": (sz, sy),  # (z, y)
        "shape_y": (sz, sx),  # (z, x)
        "shape_z": (sy, sx),  # (y, x)
    }


def _build_slice_indices(info):
    sx = int(info["sx"])
    sy = int(info["sy"])
    sz = int(info["sz"])
    t = int(info["t"])
    mx = int(info["mx"])
    my = int(info["my"])
    mz = int(info["mz"])

    z_idx = np.arange(sz, dtype=np.int64)[:, None]
    y_idx = np.arange(sy, dtype=np.int64)[None, :]
    x_idx = np.arange(sx, dtype=np.int64)[None, :]

    idx_x = ((t * sz + z_idx) * sy + y_idx) * sx + mx  # (Z, Y)
    idx_y = ((t * sz + z_idx) * sy + my) * sx + x_idx  # (Z, X)
    idx_z = ((t * sz + mz) * sy + np.arange(sy, dtype=np.int64)[:, None]) * sx + x_idx  # (Y, X)
    return idx_x, idx_y, idx_z


def _extract_gt_slices(attr_path: str, info, channel: int):
    arr = np.load(attr_path, mmap_mode="r")
    sx = int(info["sx"])
    sy = int(info["sy"])
    sz = int(info["sz"])
    t = int(info["t"])
    mx = int(info["mx"])
    my = int(info["my"])
    mz = int(info["mz"])

    if arr.ndim == 5:
        if channel < 0 or channel >= int(arr.shape[-1]):
            raise ValueError(f"--channel out of range for file '{attr_path}': dim={int(arr.shape[-1])}")
        vol = np.asarray(arr[t, :, :, :, channel], dtype=np.float32)  # (Z, Y, X)
        return {
            "x": vol[:, :, mx],
            "y": vol[:, my, :],
            "z": vol[mz, :, :],
        }

    if arr.ndim == 4:
        if channel != 0:
            raise ValueError(f"File '{attr_path}' is single-channel; --channel must be 0.")
        vol = np.asarray(arr[t, :, :, :], dtype=np.float32)  # (Z, Y, X)
        return {
            "x": vol[:, :, mx],
            "y": vol[:, my, :],
            "z": vol[mz, :, :],
        }

    if arr.ndim not in {1, 2}:
        raise ValueError(f"Unsupported target ndim: {arr.ndim} for {attr_path}")

    cdim = 1 if arr.ndim == 1 else int(arr.shape[1])
    if channel < 0 or channel >= cdim:
        raise ValueError(f"--channel out of range for file '{attr_path}': dim={cdim}")

    idx_x, idx_y, idx_z = _build_slice_indices(info)
    flat_x = idx_x.reshape(-1)
    flat_y = idx_y.reshape(-1)
    flat_z = idx_z.reshape(-1)

    if arr.ndim == 1:
        plane_x = np.asarray(arr[flat_x], dtype=np.float32).reshape(sz, sy)
        plane_y = np.asarray(arr[flat_y], dtype=np.float32).reshape(sz, sx)
        plane_z = np.asarray(arr[flat_z], dtype=np.float32).reshape(sy, sx)
    else:
        plane_x = np.asarray(arr[flat_x, channel], dtype=np.float32).reshape(sz, sy)
        plane_y = np.asarray(arr[flat_y, channel], dtype=np.float32).reshape(sz, sx)
        plane_z = np.asarray(arr[flat_z, channel], dtype=np.float32).reshape(sy, sx)

    return {
        "x": plane_x,
        "y": plane_y,
        "z": plane_z,
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


def _normalize_coords(coords_by_plane, x_mean: np.ndarray, x_std: np.ndarray):
    out = {}
    for key, coords in coords_by_plane.items():
        out[key] = ((coords - x_mean) / x_std).astype(np.float32)
    return out


def _predict_expert_slice_outputs(
    model,
    coords_by_plane,
    shape_by_plane,
    attr_name: str,
    channel: int,
    batch_size: int,
    device: torch.device,
    mean: float | None,
    std: float | None,
):
    num_experts = len(model.experts)
    outputs = {}

    with torch.no_grad():
        for plane_name, coords in coords_by_plane.items():
            n_pts = int(coords.shape[0])
            plane_out = np.empty((n_pts, num_experts), dtype=np.float32)

            for start in range(0, n_pts, batch_size):
                end = min(start + batch_size, n_pts)
                xb = torch.from_numpy(coords[start:end]).to(device, non_blocking=True)

                x_pe = model.pos_enc(xb)
                expert_feats = torch.stack([expert(x_pe) for expert in model.experts], dim=1)
                batch_n, expert_n, feat_dim = expert_feats.shape

                shared = model.decoder(expert_feats.reshape(batch_n * expert_n, feat_dim))
                pred = model.heads[attr_name](shared).reshape(batch_n, expert_n, -1)[:, :, channel]

                if mean is not None and std is not None:
                    pred = pred * std + mean

                plane_out[start:end, :] = pred.cpu().numpy()

            shp = tuple(shape_by_plane[plane_name])
            outputs[plane_name] = plane_out.reshape(shp[0], shp[1], num_experts)
    return outputs


def _collect_color_range(gt_slices, expert_slices, clip_percentile: float | None):
    arrays = [gt_slices["x"], gt_slices["y"], gt_slices["z"]]
    for plane in ("x", "y", "z"):
        arrays.append(expert_slices[plane].reshape(-1))

    flat = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays], axis=0)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return -1.0, 1.0

    if clip_percentile is not None and 50.0 < float(clip_percentile) < 100.0:
        hi = float(np.nanpercentile(finite, float(clip_percentile)))
        lo = float(np.nanpercentile(finite, 100.0 - float(clip_percentile)))
    else:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _choose_cmap(cmap_arg: str | None, vmin: float, vmax: float):
    if cmap_arg:
        return cmap_arg, vmin, vmax
    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return "coolwarm", -bound, bound
    return "viridis", vmin, vmax


def _plot_orthogonal_slices(
    slices,
    outpath: Path,
    title: str,
    attr_name: str,
    subject_name: str,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    panels = [
        (f"{attr_name} | {subject_name} | YZ @ x=mx", slices["x"], "Y", "Z"),
        (f"{attr_name} | {subject_name} | XZ @ y=my", slices["y"], "X", "Z"),
        (f"{attr_name} | {subject_name} | XY @ z=mz", slices["z"], "X", "Y"),
    ]

    im = None
    for ax, (sub_title, arr, xlabel, ylabel) in zip(axes, panels):
        im = ax.imshow(np.asarray(arr), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(sub_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Value")
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _build_dataset_and_model(cfg, attr_paths_for_model, checkpoint):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    model_name = str(model_cfg.get("name", "")).strip().lower()
    if model_name not in {"light_basis_expert", "lightbasis_expert", "light_basisexperts"}:
        raise ValueError(f"model '{model_cfg.get('name')}' is not light_basis_expert")

    volume_shape = data_cfg.get("volume_shape")
    if volume_shape is None:
        raise ValueError("data.volume_shape is required for flattened voxel files.")

    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))

    dataset = MultiTargetVolumetricDataset(
        attr_paths_for_model,
        volume_shape=volume_shape,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
    )

    model_cfg_local = dict(model_cfg)
    if int(model_cfg_local.get("in_features", 4)) != 4:
        model_cfg_local["in_features"] = 4
    model = build_model(model_cfg_local, dataset)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return dataset, model


def process_experiment(
    exp_dir: Path,
    outdir: Path,
    epoch: int,
    attr_arg: str | None,
    time_index: int,
    channel: int,
    batch_size: int,
    device_str: str | None,
    cmap_arg: str | None,
    clip_percentile: float | None,
    dpi: int,
    train_dir_override: str | None,
):
    t_start = time.perf_counter()
    cfg = _load_yaml(exp_dir / "configs" / "config.yaml")
    logger.info("[%s] Stage: load config = %.3fs", exp_dir.name, time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    ckpt_files = sorted((exp_dir / "checkpoints").glob("*.pth"))
    ckpt_path = _match_checkpoint(epoch, ckpt_files)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {exp_dir}")
    checkpoint = _torch_load_checkpoint(ckpt_path)
    if "model_state" not in checkpoint:
        raise KeyError(f"'model_state' missing in {ckpt_path}")
    logger.info("[%s] Stage: load checkpoint = %.3fs", exp_dir.name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    train_dir = _resolve_train_dir(cfg["data"], train_dir_override)
    attr_paths = _collect_train_attr_paths(train_dir)
    volume_shape = cfg["data"].get("volume_shape")
    attr_paths, dropped = _filter_attr_paths_by_shape(attr_paths, volume_shape)
    if dropped:
        logger.warning("[%s] Dropped by shape mismatch: %s", exp_dir.name, sorted(dropped.keys()))
    if not attr_paths:
        raise ValueError(f"No valid train attr_paths for {exp_dir.name}")
    logger.info("[%s] Stage: resolve train attrs = %.3fs", exp_dir.name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    dataset, model = _build_dataset_and_model(cfg, attr_paths, checkpoint)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    logger.info("[%s] Stage: build model = %.3fs", exp_dir.name, time.perf_counter() - t_stage)

    info = _build_center_slices_info(dataset.volume_shape, time_index)
    coords = _build_slice_coords(info)
    if dataset.normalize_inputs:
        x_mean = dataset.x_mean.squeeze(0).cpu().numpy().astype(np.float32)
        x_std = dataset.x_std.squeeze(0).cpu().numpy().astype(np.float32)
        coords = _normalize_coords(coords, x_mean=x_mean, x_std=x_std)

    run_tag = (
        f"basis_e{int(epoch)}_t{int(info['t']):03d}_ch{int(channel)}"
        f"_mx{int(info['mx'])}_my{int(info['my'])}_mz{int(info['mz'])}"
    )
    run_dir = outdir / exp_dir.name / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_attrs = _parse_requested_attrs(attr_arg, attr_paths.keys())
    logger.info(
        "[%s] Start slicing: t=%s center=(mx=%s,my=%s,mz=%s), attrs=%s",
        exp_dir.name,
        info["t"],
        info["mx"],
        info["my"],
        info["mz"],
        selected_attrs,
    )

    shape_by_plane = {
        "x": info["shape_x"],
        "y": info["shape_y"],
        "z": info["shape_z"],
    }

    all_outputs = []
    for attr_name in selected_attrs:
        t_attr = time.perf_counter()
        attr_dim = int(dataset.view_specs()[attr_name])
        if channel < 0 or channel >= attr_dim:
            raise ValueError(f"--channel out of range for '{attr_name}': valid [0, {attr_dim - 1}]")

        mean = None
        std = None
        if dataset.normalize_targets:
            mean, std = _resolve_denorm_stats(dataset, checkpoint, attr_name, channel)

        gt_slices = _extract_gt_slices(attr_paths[attr_name], info, channel)
        expert_slices = _predict_expert_slice_outputs(
            model=model,
            coords_by_plane=coords,
            shape_by_plane=shape_by_plane,
            attr_name=attr_name,
            channel=channel,
            batch_size=batch_size,
            device=device,
            mean=mean,
            std=std,
        )

        vmin, vmax = _collect_color_range(gt_slices, expert_slices, clip_percentile)
        cmap, vmin, vmax = _choose_cmap(cmap_arg, vmin, vmax)

        attr_safe = _safe_name(attr_name)
        gt_path = run_dir / f"{attr_safe}_gt_orth.png"
        _plot_orthogonal_slices(
            slices=gt_slices,
            outpath=gt_path,
            title=f"{exp_dir.name} | {attr_name} | GT | t={info['t']} | (mx,my,mz)=({info['mx']},{info['my']},{info['mz']})",
            attr_name=attr_name,
            subject_name="GT",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )

        expert_paths = []
        num_experts = int(expert_slices["x"].shape[-1])
        for expert_idx in range(num_experts):
            pred_slice = {
                "x": expert_slices["x"][:, :, expert_idx],
                "y": expert_slices["y"][:, :, expert_idx],
                "z": expert_slices["z"][:, :, expert_idx],
            }
            pred_path = run_dir / f"{attr_safe}_expert_{int(expert_idx):02d}_orth.png"
            _plot_orthogonal_slices(
                slices=pred_slice,
                outpath=pred_path,
                title=f"{exp_dir.name} | {attr_name} | Expert {expert_idx} | t={info['t']} | (mx,my,mz)=({info['mx']},{info['my']},{info['mz']})",
                attr_name=attr_name,
                subject_name=f"Expert {expert_idx}",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                dpi=dpi,
            )
            expert_paths.append(pred_path)

        all_outputs.append(
            {
                "attr": attr_name,
                "gt": gt_path,
                "experts": expert_paths,
                "num_experts": num_experts,
            }
        )
        logger.info("[%s] Attr %s done = %.3fs", exp_dir.name, attr_name, time.perf_counter() - t_attr)

    logger.info("[%s] Stage: total = %.3fs", exp_dir.name, time.perf_counter() - t_start)
    return all_outputs


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Visualize per-expert basis outputs as orthogonal center slices on voxel data (router skipped)."
    )
    parser.add_argument("--experiments", type=str, default="experiments", help="experiments root directory")
    parser.add_argument("--outdir", type=str, default="validate_out", help="output directory")
    parser.add_argument("--exp-id", type=str, default=None, help="single experiment id")
    parser.add_argument("--epoch", type=int, required=True, help="checkpoint epoch")
    parser.add_argument("--attr", type=str, default=None, help="attr name or comma list; default=all attrs")
    parser.add_argument("--time-index", type=int, default=0, help="time index t")
    parser.add_argument("--channel", type=int, default=0, help="channel index for multi-channel attr")
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g. cpu/cuda:0")
    parser.add_argument("--cmap", type=str, default=None, help="matplotlib colormap; default auto")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="color range clip percentile in (50,100), default 99.5; <=50 disables",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output image dpi")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir containing target_*.npy")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    clip_percentile = args.clip_percentile
    if clip_percentile is not None and clip_percentile <= 50.0:
        clip_percentile = None

    exp_root = Path(args.experiments)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.exp_id:
        exp_dirs = [exp_root / args.exp_id]
    else:
        exp_dirs = _collect_experiments(exp_root)

    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        try:
            outputs = process_experiment(
                exp_dir=exp_dir,
                outdir=outdir,
                epoch=int(args.epoch),
                attr_arg=args.attr,
                time_index=int(args.time_index),
                channel=int(args.channel),
                batch_size=int(args.batch_size),
                device_str=args.device,
                cmap_arg=args.cmap,
                clip_percentile=clip_percentile,
                dpi=int(args.dpi),
                train_dir_override=args.train_dir,
            )
        except Exception as exc:
            logger.exception("Failed: %s (%s)", exp_dir.name, exc)
            continue

        logger.info("Finished %s", exp_dir.name)
        for item in outputs:
            logger.info("  Attr: %s", item["attr"])
            logger.info("    GT slice image: %s", item["gt"])
            logger.info("    Expert images: %s", item["num_experts"])
            for p in item["experts"]:
                logger.info("      %s", p)


if __name__ == "__main__":
    main()

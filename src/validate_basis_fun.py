import argparse
import importlib
import logging
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from inr.cli import _resolve_path, build_model, load_config, resolve_data_paths
from inr.data import MultiViewCoordDataset, NodeDataset
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_MESH_SUBDIRS = ("validate_mesh", "mesh_vtu", "wind_vtu")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render per-expert predictions and GT for one experiment directory "
            "on mesh snapshots (Node/MultiView datasets)."
        )
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to one experiment directory, e.g. ./experiments/moeinr-ocean-v",
    )
    parser.add_argument(
        "--attr",
        type=str,
        default=None,
        help="Attr name or comma-separated list. Default: all attrs.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="0-based index into sorted unique timesteps. Default: evenly render up to 20 timesteps.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index for multi-channel attrs. Default: 0.",
    )
    parser.add_argument("--batch-size", type=int, default=65536, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda:0.")
    parser.add_argument("--dpi-scale", type=float, default=1.0, help="Scale factor for render window.")
    return parser.parse_args()


def _safe_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    out = str(name).strip()
    for ch in invalid:
        out = out.replace(ch, "_")
    out = out.replace(" ", "_")
    return out if out else "unknown"


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


def _extract_epoch(path: Path) -> int | None:
    match = re.search(r"epoch(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _pick_checkpoint(ckpt_dir: Path) -> Path:
    ckpt_files = sorted(ckpt_dir.glob("*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found under: {ckpt_dir}")

    final_ckpts = [path for path in ckpt_files if _extract_epoch(path) is None]
    if final_ckpts:
        chosen = max(final_ckpts, key=lambda path: (path.stat().st_mtime, path.name))
        logger.info("Selected final checkpoint: %s", chosen)
        return chosen

    epoch_ckpts = [(path, _extract_epoch(path)) for path in ckpt_files]
    valid_epoch_ckpts = [(path, epoch) for path, epoch in epoch_ckpts if epoch is not None]
    if valid_epoch_ckpts:
        chosen = max(valid_epoch_ckpts, key=lambda item: (item[1], item[0].stat().st_mtime))[0]
        logger.info("Selected highest-epoch checkpoint: %s", chosen)
        return chosen

    chosen = max(ckpt_files, key=lambda path: (path.stat().st_mtime, path.name))
    logger.info("Selected latest checkpoint by mtime fallback: %s", chosen)
    return chosen


def _resolve_stats_path(data_cfg: dict) -> str | None:
    stats_path = (
        data_cfg.get("target_stats_path")
        or data_cfg.get("stats_path")
        or data_cfg.get("normalization_stats_path")
    )
    if not stats_path:
        return None
    return _resolve_path(str(stats_path))


def _load_dataset(cfg: dict):
    data_cfg = cfg["data"]
    data_info = resolve_data_paths(data_cfg)
    stats_path = _resolve_stats_path(data_cfg)
    normalize = bool(data_cfg.get("normalize", True))

    if data_info.get("attr_paths"):
        dataset = MultiViewCoordDataset(
            data_info["x_path"],
            data_info["attr_paths"],
            normalize=normalize,
            stats_path=stats_path,
        )
        attrs = list(dataset.y.keys())
        gt_paths = {name: Path(path) for name, path in data_info["attr_paths"].items()}
    else:
        dataset = NodeDataset(
            data_info["x_path"],
            data_info["y_path"],
            normalize=normalize,
            stats_path=stats_path,
        )
        attrs = ["targets"]
        gt_paths = {"targets": Path(data_info["y_path"])}

    return dataset, data_info, attrs, gt_paths


def _parse_requested_attrs(attr_arg: str | None, available_attrs: list[str]) -> list[str]:
    if not attr_arg:
        return list(available_attrs)
    items = [s.strip() for s in str(attr_arg).split(",") if s.strip()]
    if not items:
        return list(available_attrs)
    unknown = [name for name in items if name not in available_attrs]
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {available_attrs}")
    return items


def _compute_time_indexers(time_values: np.ndarray) -> list[tuple[float, slice | np.ndarray]]:
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

    return [(float(raw_time), np.flatnonzero(time_values == raw_time)) for raw_time in np.sort(unique_times)]


def _indexer_size(indexer: slice | np.ndarray) -> int:
    if isinstance(indexer, slice):
        return int(indexer.stop - indexer.start)
    return int(indexer.shape[0])


def _select_default_timestamps(num_timesteps: int) -> list[int]:
    if num_timesteps <= 0:
        return []
    k = min(20, num_timesteps)
    selected = []
    seen = set()
    for i in range(k):
        idx = int(math.floor(i * num_timesteps / k))
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _select_tensor_block(tensor: torch.Tensor, indexer: slice | np.ndarray) -> torch.Tensor:
    if isinstance(indexer, slice):
        return tensor[indexer]
    return tensor[torch.from_numpy(indexer.astype(np.int64, copy=False))]


def _select_array_block(array: np.ndarray, indexer: slice | np.ndarray) -> np.ndarray:
    return np.asarray(array[indexer])


@lru_cache(maxsize=None)
def _read_vtu_topology(mesh_path: str) -> tuple[int, int]:
    path = Path(mesh_path)
    with path.open("rb") as handle:
        head = handle.read(4096).decode("utf-8", errors="ignore")
    match = re.search(r'NumberOfPoints="(\d+)"\s+NumberOfCells="(\d+)"', head)
    if not match:
        raise ValueError(f"Failed to read NumberOfPoints/NumberOfCells from: {path}")
    return int(match.group(1)), int(match.group(2))


def _mesh_time_tokens(raw_time: float) -> list[str]:
    tokens = []
    if float(raw_time).is_integer():
        int_text = str(int(raw_time))
        tokens.extend([int_text, f"{int_text}_0"])
    float_text = f"{raw_time}"
    tokens.extend([float_text, float_text.replace(".", "_")])
    return list(dict.fromkeys(tokens))


def _collect_mesh_candidates(x_path: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    current = x_path.resolve().parent

    for ancestor in [current, *current.parents]:
        for subdir_name in _MESH_SUBDIRS:
            subdir = ancestor / subdir_name
            if not subdir.exists() or not subdir.is_dir():
                continue
            for path in sorted(subdir.glob("*.vtu")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)
        if ancestor.name.lower() in {"raw", "processed", "data"}:
            break

    return candidates


def _infer_preferred_association(data_info: dict) -> str | None:
    path_tokens = [Path(data_info["x_path"]).stem.lower()]
    if data_info.get("y_path"):
        path_tokens.append(Path(data_info["y_path"]).stem.lower())
    for path in (data_info.get("attr_paths") or {}).values():
        path_tokens.append(Path(path).stem.lower())

    joined = " ".join(path_tokens)
    if "cell" in joined:
        return "cell"
    if "point" in joined:
        return "point"
    return None


def _resolve_mesh_for_timestep(
    raw_time: float,
    sample_count: int,
    candidate_paths: list[Path],
    preferred_association: str | None,
) -> tuple[Path, str]:
    if not candidate_paths:
        raise FileNotFoundError(
            "No mesh candidates found. Expected a .vtu under validate_mesh/, mesh_vtu/, or wind_vtu/."
        )

    tokens = _mesh_time_tokens(raw_time)
    ranked: list[tuple[int, int, int, Path, str]] = []

    for order, path in enumerate(candidate_paths):
        num_points, num_cells = _read_vtu_topology(str(path))
        associations = []
        if num_points == sample_count:
            associations.append("point")
        if num_cells == sample_count:
            associations.append("cell")
        if not associations:
            continue

        if preferred_association in associations:
            association = preferred_association
            assoc_score = 1
        else:
            association = associations[0]
            assoc_score = 0

        stem = path.stem
        time_score = 1 if any(token and token in stem for token in tokens) else 0
        ranked.append((time_score, assoc_score, -order, path, association))

    if not ranked:
        raise FileNotFoundError(
            f"No mesh candidate matches sample count {sample_count} for raw time {raw_time}."
        )

    ranked.sort(reverse=True)
    _, _, _, chosen_path, association = ranked[0]
    return chosen_path, association


def _ensure_pyvista_available():
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required to render validation images. Install pyvista in the active Python environment."
        ) from exc


def _normalize_clim(clim: tuple[float, float] | None) -> tuple[float, float]:
    if clim is None:
        return 0.0, 1.0
    lo, hi = float(clim[0]), float(clim[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        delta = max(abs(lo), 1.0) * 1e-6
        return lo - delta, hi + delta
    return lo, hi


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _merge_range(current: tuple[float, float] | None, values: np.ndarray) -> tuple[float, float] | None:
    new_range = _finite_range(values)
    if new_range is None:
        return current
    if current is None:
        return new_range
    return min(current[0], new_range[0]), max(current[1], new_range[1])


def _resolve_channel_stats(dataset, attr_name: str, channel: int) -> tuple[float, float]:
    if isinstance(dataset, MultiViewCoordDataset):
        mean = dataset.y_mean[attr_name].reshape(-1)
        std = dataset.y_std[attr_name].reshape(-1)
    else:
        mean = dataset.y_mean.reshape(-1)
        std = dataset.y_std.reshape(-1)

    if channel < 0 or channel >= int(mean.shape[0]):
        raise ValueError(
            f"--channel out of range for '{attr_name}': {channel}, valid [0, {int(mean.shape[0]) - 1}]"
        )
    return float(mean[channel].item()), float(std[channel].item())


def _select_attr_channel(values: np.ndarray, attr_name: str, channel: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        if channel != 0:
            raise ValueError(f"Attr '{attr_name}' is single-channel; --channel must be 0.")
        return array
    if array.ndim == 2:
        if channel < 0 or channel >= int(array.shape[1]):
            raise ValueError(
                f"--channel out of range for '{attr_name}': {channel}, valid [0, {int(array.shape[1]) - 1}]"
            )
        return array[:, channel]
    raise ValueError(f"Unsupported GT array shape for '{attr_name}': {array.shape}")


def _require_expert_modules(model: torch.nn.Module):
    missing = [name for name in ["experts"] if not hasattr(model, name)]
    has_light = hasattr(model, "decoder") and hasattr(model, "heads")
    has_basis = hasattr(model, "decoders") and hasattr(model, "view_names")
    if missing or not (has_light or has_basis):
        raise ValueError(
            f"Model {type(model).__name__} is not a supported BasisExpert variant. "
            "Expected experts + (decoder/heads) or experts + (decoders/view_names)."
        )


def _expert_encoder_input(model: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "pos_enc"):
        return model.pos_enc(xb)
    return xb


def _resolve_attr_view_index(model: torch.nn.Module, attr_name: str) -> int:
    view_names = list(getattr(model, "view_names"))
    if attr_name not in view_names:
        raise KeyError(f"Attr '{attr_name}' not found in model views. Available: {view_names}")
    return int(view_names.index(attr_name))


def _predict_simple_concat_family(
    model: torch.nn.Module,
    expert_feats: torch.Tensor,
    attr_name: str,
    view_idx: int,
) -> torch.Tensor:
    bsz, num_experts, feat_dim = expert_feats.shape
    num_views = len(model.view_names)
    decoder = model.decoders[attr_name]
    mode = str(getattr(model, "fusion_mode", "concat")).strip().lower()

    if mode == "concat":
        fused = expert_feats[:, :, None, :].expand(bsz, num_experts, num_views, feat_dim)
        decoder_in = fused.reshape(bsz * num_experts, num_views * feat_dim)
    elif mode == "mean":
        decoder_in = expert_feats.reshape(bsz * num_experts, feat_dim)
    elif mode == "mlp":
        cat_in = expert_feats[:, :, None, :].expand(bsz, num_experts, num_views, feat_dim)
        cat_in = cat_in.reshape(bsz * num_experts, num_views * feat_dim)
        decoder_in = model.fusion_mlp(cat_in)
    elif mode == "none":
        view_ids = torch.full((bsz,), int(view_idx), device=expert_feats.device, dtype=torch.long)
        view_embed = model.view_embedding(view_ids)
        view_embed = view_embed[:, None, :].expand(bsz, num_experts, -1)
        decoder_in = torch.cat(
            [expert_feats, view_embed],
            dim=-1,
        ).reshape(bsz * num_experts, -1)
    else:
        raise ValueError(f"Unsupported fusion_mode '{mode}' for model {type(model).__name__}")

    pred = decoder(decoder_in)
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred.reshape(bsz, num_experts, -1)


def _predict_attention_family(
    model: torch.nn.Module,
    expert_feats: torch.Tensor,
    attr_name: str,
    view_idx: int,
) -> torch.Tensor:
    bsz, num_experts, feat_dim = expert_feats.shape
    num_views = len(model.view_names)

    # Build one token grid per expert by reusing that expert feature across all views.
    h_views = expert_feats[:, :, None, :].expand(bsz, num_experts, num_views, feat_dim).clone()

    if hasattr(model, "view_embed_proj"):
        view_ids = torch.arange(num_views, device=expert_feats.device, dtype=torch.long)
        view_embed = model.view_embedding(view_ids)
        view_bias = model.view_embed_proj(view_embed)
        h_views = h_views + view_bias[None, None, :, :]

    tokens = h_views.reshape(bsz * num_experts, num_views, feat_dim)
    ctx = model.ctx_token.expand(bsz * num_experts, -1, -1)
    fused_tokens = model.fusion(torch.cat([ctx, tokens], dim=1))

    ctx_flat = fused_tokens[:, 0, :]
    h_view = fused_tokens[:, 1 + int(view_idx), :]
    decoder_in = torch.cat([h_view, ctx_flat], dim=-1)

    pred = model.decoders[attr_name](decoder_in)
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred.reshape(bsz, num_experts, -1)


def _predict_expert_channel_block(
    model: torch.nn.Module,
    coords_block: torch.Tensor,
    attr_name: str,
    channel: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    experts = getattr(model, "experts")

    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        total = int(coords_block.shape[0])
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            xb = coords_block[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            expert_in = _expert_encoder_input(model, xb)
            expert_feats = torch.stack([expert(expert_in) for expert in experts], dim=1)

            if hasattr(model, "decoder") and hasattr(model, "heads"):
                heads = getattr(model, "heads")
                if attr_name not in heads:
                    raise KeyError(
                        f"Attr '{attr_name}' not found in model heads. Available: {list(heads.keys())}"
                    )
                batch_n, expert_n, feat_dim = expert_feats.shape
                shared = model.decoder(expert_feats.reshape(batch_n * expert_n, feat_dim))
                pred = heads[attr_name](shared)
                if pred.ndim == 1:
                    pred = pred[:, None]
                pred = pred.reshape(batch_n, expert_n, -1)
            elif hasattr(model, "decoders") and hasattr(model, "view_names"):
                if attr_name not in model.decoders:
                    raise KeyError(
                        f"Attr '{attr_name}' not found in model decoders. Available: {list(model.decoders.keys())}"
                    )
                view_idx = _resolve_attr_view_index(model, attr_name)

                if hasattr(model, "fusion") and hasattr(model, "ctx_token"):
                    pred = _predict_attention_family(model, expert_feats, attr_name, view_idx)
                else:
                    pred = _predict_simple_concat_family(model, expert_feats, attr_name, view_idx)
            else:
                raise ValueError(
                    f"Unsupported model layout for per-expert visualization: {type(model).__name__}"
                )

            if channel < 0 or channel >= int(pred.shape[2]):
                raise ValueError(
                    f"--channel out of range for model head '{attr_name}': {channel}, "
                    f"valid [0, {int(pred.shape[2]) - 1}]"
                )
            chunks.append(pred[:, :, channel].detach().cpu())

    return torch.cat(chunks, dim=0)


def _denormalize_channel_values(
    values: np.ndarray,
    dataset,
    attr_name: str,
    channel: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not bool(getattr(dataset, "normalize", False)):
        return arr
    mean, std = _resolve_channel_stats(dataset, attr_name, channel)
    return arr * std + mean


def _render_frame(
    mesh_path: Path,
    association: str,
    values: np.ndarray,
    outpath: Path,
    clim: tuple[float, float],
    dpi_scale: float,
):
    import pyvista as pv

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Expected 1D scalar array for rendering, got shape={array.shape}")

    mesh = pv.read(str(mesh_path))
    if association == "point":
        if mesh.n_points != array.shape[0]:
            raise ValueError(
                f"Point-data size mismatch for {mesh_path}: mesh points={mesh.n_points}, values={array.shape[0]}"
            )
        mesh.point_data["U_vis"] = array
    elif association == "cell":
        if mesh.n_cells != array.shape[0]:
            raise ValueError(
                f"Cell-data size mismatch for {mesh_path}: mesh cells={mesh.n_cells}, values={array.shape[0]}"
            )
        mesh.cell_data["U_vis"] = array
    else:
        raise ValueError(f"Unknown association: {association}")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    width = max(800, int(1800 * float(dpi_scale)))
    height = max(600, int(1400 * float(dpi_scale)))

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="U_vis",
        clim=list(clim),
        show_edges=False,
        show_scalar_bar=False,
        nan_color="white",
    )
    plotter.reset_camera()
    plotter.show(screenshot=str(outpath))
    plotter.close()


def _prepare_gt_output_path(out_root: Path, exp_id: str, attr_name: str, time_index: int) -> Path:
    attr_safe = _safe_name(attr_name)
    return out_root / attr_safe / f"t{time_index:04d}" / f"{exp_id}_t{time_index:04d}_gt.png"


def _prepare_expert_output_path(
    out_root: Path,
    exp_id: str,
    attr_name: str,
    time_index: int,
    expert_idx: int,
) -> Path:
    attr_safe = _safe_name(attr_name)
    return (
        out_root
        / attr_safe
        / f"t{time_index:04d}"
        / f"{exp_id}_t{time_index:04d}_expert_{expert_idx:02d}.png"
    )


def main():
    setup_logging()
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.dpi_scale <= 0:
        raise ValueError("--dpi-scale must be > 0")

    exp_dir = Path(args.experiment_path).resolve()
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_path = exp_dir / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    ckpt_path = _pick_checkpoint(exp_dir / "checkpoints")
    cfg = load_config(str(cfg_path))
    exp_id = str(cfg.get("exp_id") or exp_dir.name)

    dataset, data_info, attrs_all, gt_paths = _load_dataset(cfg)
    attrs = _parse_requested_attrs(args.attr, attrs_all)
    for attr_name in attrs:
        gt_path = gt_paths[attr_name]
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground-truth file not found for '{attr_name}': {gt_path}")

    raw_coords = np.load(data_info["x_path"], mmap_mode="r", allow_pickle=False)
    if raw_coords.ndim != 2 or raw_coords.shape[1] < 1:
        raise ValueError(f"Expected coords array with shape (N, D>=1), got {raw_coords.shape}")

    time_values = raw_coords[:, -1]
    time_indexers = _compute_time_indexers(time_values)
    num_timesteps = len(time_indexers)
    if num_timesteps == 0:
        raise ValueError("No timesteps found in coords array.")

    if args.timestamp is not None:
        if args.timestamp < 0 or args.timestamp >= num_timesteps:
            raise ValueError(
                f"--timestamp {args.timestamp} is out of range. Valid range: [0, {num_timesteps - 1}]"
            )
        selected_time_indices = [int(args.timestamp)]
    else:
        selected_time_indices = _select_default_timestamps(num_timesteps)

    selected_steps = []
    for time_index in selected_time_indices:
        raw_time, indexer = time_indexers[time_index]
        selected_steps.append(
            {
                "time_index": int(time_index),
                "raw_time": float(raw_time),
                "indexer": indexer,
                "sample_count": _indexer_size(indexer),
            }
        )

    candidate_meshes = _collect_mesh_candidates(Path(data_info["x_path"]))
    preferred_association = _infer_preferred_association(data_info)
    for step in selected_steps:
        mesh_path, association = _resolve_mesh_for_timestep(
            raw_time=step["raw_time"],
            sample_count=step["sample_count"],
            candidate_paths=candidate_meshes,
            preferred_association=preferred_association,
        )
        step["mesh_path"] = mesh_path
        step["mesh_association"] = association

    _ensure_pyvista_available()

    model = build_model(cfg["model"], dataset)
    _require_expert_modules(model)

    payload = _torch_load_checkpoint(ckpt_path)
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    gt_arrays = {
        name: np.load(str(path), mmap_mode="r", allow_pickle=False)
        for name, path in gt_paths.items()
        if name in attrs
    }

    logger.info("Experiment: %s", exp_id)
    logger.info("Config: %s", cfg_path)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Attrs: %s", attrs)
    logger.info(
        "Rendering timesteps: %s",
        [f"{step['time_index']}({step['raw_time']})" for step in selected_steps],
    )

    out_root = Path(f"expert_output_{_safe_name(exp_id)}")
    num_experts = int(len(model.experts))
    logger.info("Detected experts: %d", num_experts)

    for attr_name in attrs:
        logger.info("Attr '%s': pass 1/2 collecting color range", attr_name)
        attr_clim: tuple[float, float] | None = None

        for step in selected_steps:
            coords_block = _select_tensor_block(dataset.x, step["indexer"])
            pred_expert = _predict_expert_channel_block(
                model=model,
                coords_block=coords_block,
                attr_name=attr_name,
                channel=int(args.channel),
                batch_size=int(args.batch_size),
                device=device,
            )
            pred_expert_np = pred_expert.numpy()
            pred_expert_np = _denormalize_channel_values(pred_expert_np, dataset, attr_name, int(args.channel))

            gt_block = _select_array_block(gt_arrays[attr_name], step["indexer"])
            gt_vis = _select_attr_channel(gt_block, attr_name, int(args.channel))

            attr_clim = _merge_range(attr_clim, gt_vis)
            for expert_idx in range(num_experts):
                attr_clim = _merge_range(attr_clim, pred_expert_np[:, expert_idx])

        clim = _normalize_clim(attr_clim)
        logger.info("Attr '%s': color range = [%.6f, %.6f]", attr_name, clim[0], clim[1])

        logger.info("Attr '%s': pass 2/2 rendering", attr_name)
        for step in selected_steps:
            coords_block = _select_tensor_block(dataset.x, step["indexer"])
            pred_expert = _predict_expert_channel_block(
                model=model,
                coords_block=coords_block,
                attr_name=attr_name,
                channel=int(args.channel),
                batch_size=int(args.batch_size),
                device=device,
            )
            pred_expert_np = pred_expert.numpy()
            pred_expert_np = _denormalize_channel_values(pred_expert_np, dataset, attr_name, int(args.channel))

            gt_block = _select_array_block(gt_arrays[attr_name], step["indexer"])
            gt_vis = _select_attr_channel(gt_block, attr_name, int(args.channel))

            gt_out = _prepare_gt_output_path(out_root, exp_id, attr_name, step["time_index"])
            _render_frame(
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                values=gt_vis,
                outpath=gt_out,
                clim=clim,
                dpi_scale=float(args.dpi_scale),
            )

            for expert_idx in range(num_experts):
                pred_out = _prepare_expert_output_path(
                    out_root=out_root,
                    exp_id=exp_id,
                    attr_name=attr_name,
                    time_index=step["time_index"],
                    expert_idx=expert_idx,
                )
                _render_frame(
                    mesh_path=step["mesh_path"],
                    association=step["mesh_association"],
                    values=pred_expert_np[:, expert_idx],
                    outpath=pred_out,
                    clim=clim,
                    dpi_scale=float(args.dpi_scale),
                )

            logger.info(
                "Rendered attr=%s timestep=%s -> gt=%s experts=%d",
                attr_name,
                step["time_index"],
                gt_out,
                num_experts,
            )


if __name__ == "__main__":
    main()

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
from inr.utils.io import warn_if_multiview_attr_order_mismatch

logger = logging.getLogger(__name__)

_MESH_SUBDIRS = ("validate_mesh", "mesh_vtu", "wind_vtu")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render predictions and ground truth for one experiment directory."
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to one experiment directory, e.g. ./experiments/light_basis_expert-stress",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="0-based index into sorted unique timesteps. Default: evenly render up to 20 timesteps.",
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


def _select_tensor_block(tensor: torch.Tensor, indexer: slice | np.ndarray) -> torch.Tensor:
    if isinstance(indexer, slice):
        return tensor[indexer]
    return tensor[torch.from_numpy(indexer.astype(np.int64, copy=False))]


def _select_array_block(array: np.ndarray, indexer: slice | np.ndarray) -> np.ndarray:
    return np.asarray(array[indexer])


def _to_visual_scalar(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] in (2, 3):
        return np.linalg.norm(array, axis=1)
    return array


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


def _predict_block(
    model: torch.nn.Module,
    coords_block: torch.Tensor,
    attrs: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pred_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in attrs}
    with torch.inference_mode():
        total = int(coords_block.shape[0])
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            xb = coords_block[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            try:
                output = model(xb, hard_topk=True)
            except TypeError:
                output = model(xb)

            if isinstance(output, dict):
                for name in attrs:
                    if name not in output:
                        raise KeyError(
                            f"Model output missing attr '{name}'. Available attrs: {list(output.keys())}"
                        )
                    chunk = output[name].detach().cpu()
                    if chunk.ndim == 1:
                        chunk = chunk[:, None]
                    pred_chunks[name].append(chunk)
            else:
                if len(attrs) != 1:
                    raise ValueError("Single tensor output cannot be mapped to multiple attrs.")
                chunk = output.detach().cpu()
                if chunk.ndim == 1:
                    chunk = chunk[:, None]
                pred_chunks[attrs[0]].append(chunk)

    return {name: torch.cat(chunks, dim=0) for name, chunks in pred_chunks.items()}


def _denormalize_prediction(dataset, attr_name: str, pred_cpu: torch.Tensor) -> np.ndarray:
    if isinstance(dataset, MultiViewCoordDataset):
        return dataset.denormalize_attr(attr_name, pred_cpu).cpu().numpy()
    return dataset.denormalize_targets(pred_cpu).cpu().numpy()


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

    return [
        (float(raw_time), np.flatnonzero(time_values == raw_time))
        for raw_time in np.sort(unique_times)
    ]


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


@lru_cache(maxsize=None)
def _read_vtu_topology(mesh_path: str) -> tuple[int, int]:
    path = Path(mesh_path)
    with path.open("rb") as handle:
        head = handle.read(4096).decode("utf-8", errors="ignore")
    match = re.search(r'NumberOfPoints="(\d+)"\s+NumberOfCells="(\d+)"', head)
    if not match:
        raise ValueError(f"Failed to read NumberOfPoints/NumberOfCells from: {path}")
    return int(match.group(1)), int(match.group(2))


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


def _render_frame(
    mesh_path: Path,
    association: str,
    values: np.ndarray,
    outpath: Path,
    clim: tuple[float, float],
    zoom_factor: float,
):
    import pyvista as pv

    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
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
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1400))
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
    plotter.camera.zoom(float(zoom_factor))
    plotter.show(screenshot=str(outpath))
    plotter.close()


def _ensure_pyvista_available():
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required to render validation images. Install pyvista in the active Python environment."
        ) from exc


def _prepare_output_path(out_root: Path, exp_id: str, attr_name: str, kind: str, time_index: int) -> Path:
    return out_root / exp_id / attr_name / f"{exp_id}_t{time_index:04d}_{kind}.png"


def _render_zoom_factor(data_info: dict) -> float:
    dataset_name = str(data_info.get("dataset_name", "")).strip().lower()
    if dataset_name == "ocean":
        return 1.8
    return 1.35


def main():
    setup_logging()
    args = _parse_args()

    exp_dir = Path(args.experiment_path).resolve()
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_path = exp_dir / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    ckpt_path = _pick_checkpoint(exp_dir / "checkpoints")
    cfg = load_config(str(cfg_path))
    exp_id = str(cfg.get("exp_id") or exp_dir.name)

    dataset, data_info, attrs, gt_paths = _load_dataset(cfg)
    for attr_name, gt_path in gt_paths.items():
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
    payload = _torch_load_checkpoint(ckpt_path)
    if isinstance(dataset, MultiViewCoordDataset):
        warn_if_multiview_attr_order_mismatch(
            payload,
            dataset.view_specs().keys(),
            context=str(cfg_path),
            logger_override=logger,
        )
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch_size = int(
        cfg.get("training", {}).get(
            "pred_batch_size",
            cfg.get("training", {}).get("batch_size", 8192),
        )
    )
    gt_arrays = {
        name: np.load(str(path), mmap_mode="r", allow_pickle=False)
        for name, path in gt_paths.items()
    }

    logger.info("Experiment: %s", exp_id)
    logger.info("Config: %s", cfg_path)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Attrs: %s", attrs)
    logger.info("Render zoom factor: %.2f", _render_zoom_factor(data_info))
    logger.info(
        "Rendering timesteps: %s",
        [f"{step['time_index']}({step['raw_time']})" for step in selected_steps],
    )

    attr_clims: dict[str, tuple[float, float] | None] = {name: None for name in attrs}
    logger.info("Pass 1/2: collecting per-attr color ranges")
    for step in selected_steps:
        coords_block = _select_tensor_block(dataset.x, step["indexer"])
        pred_map = _predict_block(model, coords_block, attrs, batch_size, device)
        for attr_name in attrs:
            pred_vis = _to_visual_scalar(
                _denormalize_prediction(dataset, attr_name, pred_map[attr_name])
            )
            gt_vis = _to_visual_scalar(
                _select_array_block(gt_arrays[attr_name], step["indexer"])
            )
            attr_clims[attr_name] = _merge_range(attr_clims[attr_name], pred_vis)
            attr_clims[attr_name] = _merge_range(attr_clims[attr_name], gt_vis)

    normalized_clims = {
        attr_name: _normalize_clim(clim)
        for attr_name, clim in attr_clims.items()
    }
    zoom_factor = _render_zoom_factor(data_info)

    out_root = Path("validate_out")
    logger.info("Pass 2/2: rendering frames to %s", out_root.resolve())
    for step in selected_steps:
        coords_block = _select_tensor_block(dataset.x, step["indexer"])
        pred_map = _predict_block(model, coords_block, attrs, batch_size, device)
        for attr_name in attrs:
            pred_vis = _to_visual_scalar(
                _denormalize_prediction(dataset, attr_name, pred_map[attr_name])
            )
            gt_vis = _to_visual_scalar(
                _select_array_block(gt_arrays[attr_name], step["indexer"])
            )
            pred_out = _prepare_output_path(out_root, exp_id, attr_name, "pred", step["time_index"])
            gt_out = _prepare_output_path(out_root, exp_id, attr_name, "gt", step["time_index"])
            _render_frame(
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                values=pred_vis,
                outpath=pred_out,
                clim=normalized_clims[attr_name],
                zoom_factor=zoom_factor,
            )
            _render_frame(
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                values=gt_vis,
                outpath=gt_out,
                clim=normalized_clims[attr_name],
                zoom_factor=zoom_factor,
            )
            logger.info(
                "Rendered attr=%s timestep=%s -> pred=%s gt=%s",
                attr_name,
                step["time_index"],
                pred_out,
                gt_out,
            )


if __name__ == "__main__":
    main()

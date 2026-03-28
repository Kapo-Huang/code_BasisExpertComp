import argparse
import csv
import importlib
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inr.cli import _resolve_path, build_model, load_config, resolve_data_paths
from inr.data import MultiViewCoordDataset, NodeDataset
from inr.utils.io import warn_if_multiview_attr_order_mismatch
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_MESH_SUBDIRS = ("validate_mesh", "mesh_vtu", "wind_vtu")
_CSV_FIELDNAMES = [
    "row_type",
    "exp_id",
    "model_name",
    "dataset_name",
    "checkpoint_path",
    "attr",
    "time_index",
    "raw_time",
    "num_samples",
    "num_timesteps",
    "gt_render_path",
    "pred_render_path",
    "psnr",
    "ssim",
    "lpips",
]


@dataclass(frozen=True)
class AttrEvalSpec:
    raw_gt_path: Path
    offset: np.ndarray | None
    scale: np.ndarray | None


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset: object
    data_info: dict
    attrs: list[str]
    train_gt_paths: dict[str, Path]
    raw_coords_path: Path
    raw_gt_paths: dict[str, Path]
    attr_eval_affines: dict[str, tuple[np.ndarray, np.ndarray] | None]


@dataclass
class EvaluationSetup:
    dataset: object
    data_info: dict
    attrs: list[str]
    train_gt_paths: dict[str, Path]
    raw_coords_path: Path
    attr_specs: dict[str, AttrEvalSpec]


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
        help="0-based index into sorted unique timesteps. Default: render all timesteps.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to metrics CSV. Default: validate_out/<dataset_name>/<exp_id>/<exp_id>_metrics.csv",
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


def _resolve_raw_eval_path(path_value: str | Path) -> Path:
    path = Path(path_value).resolve()
    for suffix in ("_normalized", "_denormalized"):
        if not path.stem.endswith(suffix):
            continue
        candidate = path.with_name(f"{path.stem[:-len(suffix)]}{path.suffix}")
        if not candidate.exists():
            raise FileNotFoundError(f"Expected raw counterpart for {path}, but not found: {candidate}")
        return candidate.resolve()
    return path


def _is_normalized_artifact_path(path: Path) -> bool:
    return any(path.stem.endswith(suffix) for suffix in ("_normalized", "_denormalized"))


@lru_cache(maxsize=None)
def _load_npz_payload(npz_path: str) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return {
            key: np.array(payload[key], copy=True)
            for key in payload.files
        }


def _as_stat_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array[None, :]
    return array


def _append_unique(candidates: list[str], value: str | None) -> None:
    if value and value not in candidates:
        candidates.append(value)


def _attr_stats_key_candidates(attr_name: str, raw_gt_path: Path) -> list[str]:
    candidates: list[str] = []
    _append_unique(candidates, attr_name)
    if attr_name.startswith("data_"):
        _append_unique(candidates, attr_name[len("data_"):])

    raw_stem = raw_gt_path.stem
    for suffix in ("_normalized", "_denormalized"):
        if raw_stem.endswith(suffix):
            raw_stem = raw_stem[: -len(suffix)]
            break

    raw_token = raw_stem
    for prefix in ("target_", "targets_"):
        if raw_token.startswith(prefix):
            raw_token = raw_token[len(prefix):]
            break

    _append_unique(candidates, raw_token)
    for prefix in ("point_", "cell_", "data_"):
        if raw_token.startswith(prefix):
            _append_unique(candidates, raw_token[len(prefix):])
    return candidates


def _resolve_raw_stats_path(data_cfg: dict, raw_gt_path: Path) -> Path | None:
    configured = _resolve_stats_path(data_cfg)
    if configured:
        configured_path = Path(configured).resolve()
        if configured_path.exists():
            return configured_path

    parent = raw_gt_path.parent
    stem = raw_gt_path.stem
    candidates: list[Path] = []
    if stem.startswith("target_point_"):
        candidates.append(parent / "target_point_stats_multi.npz")
    if stem.startswith("target_"):
        candidates.append(parent / f"target_stats_{stem[len('target_'):]}.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _stats_payload_has_node_targets(stats_path: str | Path) -> bool:
    path = Path(stats_path)
    if not path.exists():
        return False
    payload = _load_npz_payload(str(path.resolve()))
    return "y_mean" in payload and "y_std" in payload


def _validation_stats_cache_path(raw_gt_path: Path) -> Path:
    return Path("validate_out") / "_stats_cache" / f"{raw_gt_path.stem}_validation_stats.npz"


def _validation_normalized_cache_path(raw_path: Path) -> Path:
    cache_root = Path("validate_out") / "_normalized_cache"
    try:
        relative_parent = raw_path.resolve().relative_to(Path.cwd().resolve()).parent
        target_parent = cache_root / relative_parent
    except ValueError:
        target_parent = cache_root / raw_path.parent.name
    return target_parent / f"{raw_path.stem}_normalized_validate_v2{raw_path.suffix}"


def _stress_minmax_denominator(raw_path: Path, raw_min: np.ndarray, raw_max: np.ndarray) -> np.ndarray:
    if raw_path.stem == "target_cell_S_IntegrationPoints":
        denom = raw_max
    else:
        denom = raw_max - raw_min
    return np.where(np.abs(denom) < 1e-12, np.ones_like(denom, dtype=np.float32), denom).astype(np.float32)


@lru_cache(maxsize=None)
def _stress_minmax_params(raw_path_value: str) -> tuple[np.ndarray, np.ndarray]:
    raw_path = Path(raw_path_value).resolve()
    raw = np.load(str(raw_path), mmap_mode="r", allow_pickle=False)
    raw_min = np.min(raw, axis=0, keepdims=True).astype(np.float32)
    raw_max = np.max(raw, axis=0, keepdims=True).astype(np.float32)
    denom = _stress_minmax_denominator(raw_path, raw_min, raw_max)
    return raw_min.astype(np.float32), denom.astype(np.float32)


@lru_cache(maxsize=None)
def _stress_minmax_affine(raw_path_value: str) -> tuple[np.ndarray, np.ndarray]:
    raw_min, denom = _stress_minmax_params(raw_path_value)
    scale = denom / 2.0
    offset = raw_min + scale
    return offset.astype(np.float32), scale.astype(np.float32)


def _ensure_stress_minmax_normalized_cache(raw_path: Path) -> Path:
    cache_path = _validation_normalized_cache_path(raw_path)
    if cache_path.exists():
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    raw = np.load(str(raw_path), mmap_mode="r", allow_pickle=False)
    raw_min, denom = _stress_minmax_params(str(raw_path))

    rows = int(raw.shape[0]) if raw.ndim > 0 else 1
    features = int(raw.shape[1]) if raw.ndim > 1 else 1
    rows_per_chunk = max(1, min(rows, max(1, 1_000_000 // features)))
    normalized = np.lib.format.open_memmap(
        str(tmp_path),
        mode="w+",
        dtype=np.float32,
        shape=raw.shape,
    )
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        block = np.asarray(raw[start:end], dtype=np.float32)
        normalized[start:end] = ((block - raw_min) / denom) * 2.0 - 1.0
    del normalized
    tmp_path.replace(cache_path)
    logger.info("Created min-max normalized cache: %s", cache_path)
    return cache_path.resolve()


def _prepare_stress_minmax_fallback(
    data_cfg: dict,
    data_info: dict,
    x_path: Path,
    y_path: Path | None,
    attr_paths: dict[str, Path],
) -> tuple[Path, Path | None, dict[str, Path], dict[str, Path], dict[str, tuple[np.ndarray, np.ndarray]]]:
    dataset_name = str(data_cfg.get("dataset_name", "")).strip().lower()
    if dataset_name != "stress" or bool(data_cfg.get("normalize", True)):
        raise ValueError("Stress min-max fallback only applies to stress configs with normalize=false.")

    raw_coords_path = _resolve_raw_eval_path(x_path)
    x_load_path = _ensure_stress_minmax_normalized_cache(raw_coords_path)

    if attr_paths:
        y_load_path = None
        load_attr_paths: dict[str, Path] = {}
        raw_gt_paths: dict[str, Path] = {}
        attr_eval_affines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for attr_name, configured_path in attr_paths.items():
            raw_gt_path = _resolve_raw_eval_path(configured_path)
            raw_gt_paths[attr_name] = raw_gt_path
            load_attr_paths[attr_name] = _ensure_stress_minmax_normalized_cache(raw_gt_path)
            attr_eval_affines[attr_name] = _stress_minmax_affine(str(raw_gt_path))
        return x_load_path, y_load_path, load_attr_paths, raw_gt_paths, attr_eval_affines

    if y_path is None:
        raise ValueError("Expected y_path for single-target stress min-max fallback.")

    attr_name = _infer_single_target_attr_name(data_info)
    raw_gt_path = _resolve_raw_eval_path(y_path)
    attr_eval_affines = {attr_name: _stress_minmax_affine(str(raw_gt_path))}
    return (
        x_load_path,
        _ensure_stress_minmax_normalized_cache(raw_gt_path),
        {},
        {attr_name: raw_gt_path},
        attr_eval_affines,
    )


def _load_attr_denorm_stats(data_cfg: dict, attr_name: str, raw_gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats_path = _resolve_raw_stats_path(data_cfg, raw_gt_path)
    if stats_path is None:
        raise FileNotFoundError(f"Failed to locate raw stats for attr '{attr_name}' from {raw_gt_path}")

    payload = _load_npz_payload(str(stats_path))
    if "y_mean" in payload and "y_std" in payload:
        return _as_stat_matrix(payload["y_mean"]), _as_stat_matrix(payload["y_std"])

    for candidate in _attr_stats_key_candidates(attr_name, raw_gt_path):
        mean_key = f"y_mean_{candidate}"
        std_key = f"y_std_{candidate}"
        if mean_key in payload and std_key in payload:
            return _as_stat_matrix(payload[mean_key]), _as_stat_matrix(payload[std_key])

    raise KeyError(
        f"Failed to locate y_mean/y_std for attr '{attr_name}' in stats file: {stats_path}"
    )


def _resolve_attr_eval_spec(
    data_cfg: dict,
    dataset,
    attr_name: str,
    train_gt_path: Path,
    raw_gt_path: Path,
    preset_affine: tuple[np.ndarray, np.ndarray] | None = None,
) -> AttrEvalSpec:
    if bool(getattr(dataset, "normalize", False)):
        if isinstance(dataset, MultiViewCoordDataset):
            offset = dataset.y_mean[attr_name].cpu().numpy()
            scale = dataset.y_std[attr_name].cpu().numpy()
        else:
            offset = dataset.y_mean.cpu().numpy()
            scale = dataset.y_std.cpu().numpy()
        return AttrEvalSpec(
            raw_gt_path=raw_gt_path,
            offset=_as_stat_matrix(offset),
            scale=_as_stat_matrix(scale),
        )

    if preset_affine is not None:
        offset, scale = preset_affine
        return AttrEvalSpec(
            raw_gt_path=raw_gt_path,
            offset=_as_stat_matrix(offset),
            scale=_as_stat_matrix(scale),
        )

    if train_gt_path.resolve() == raw_gt_path.resolve():
        return AttrEvalSpec(raw_gt_path=raw_gt_path, offset=None, scale=None)

    offset, scale = _load_attr_denorm_stats(data_cfg, attr_name, raw_gt_path)
    return AttrEvalSpec(raw_gt_path=raw_gt_path, offset=offset, scale=scale)


def _build_evaluation_setup(cfg: dict) -> EvaluationSetup:
    load_result = _load_dataset(cfg)
    attr_specs: dict[str, AttrEvalSpec] = {}
    for attr_name, train_gt_path in load_result.train_gt_paths.items():
        raw_gt_path = load_result.raw_gt_paths[attr_name]
        attr_specs[attr_name] = _resolve_attr_eval_spec(
            cfg["data"],
            load_result.dataset,
            attr_name,
            Path(train_gt_path).resolve(),
            raw_gt_path,
            preset_affine=load_result.attr_eval_affines.get(attr_name),
        )
    return EvaluationSetup(
        dataset=load_result.dataset,
        data_info=load_result.data_info,
        attrs=load_result.attrs,
        train_gt_paths=load_result.train_gt_paths,
        raw_coords_path=load_result.raw_coords_path,
        attr_specs=attr_specs,
    )


def _load_dataset(cfg: dict) -> DatasetLoadResult:
    data_cfg = cfg["data"]
    data_info = resolve_data_paths(data_cfg)
    stats_path = _resolve_stats_path(data_cfg)
    normalize = bool(data_cfg.get("normalize", True))

    x_path = Path(data_info["x_path"]).resolve()
    y_path = Path(data_info["y_path"]).resolve() if data_info.get("y_path") else None
    attr_paths = {
        name: Path(path).resolve()
        for name, path in (data_info.get("attr_paths") or {}).items()
    }

    referenced_paths = [x_path]
    if y_path is not None:
        referenced_paths.append(y_path)
    referenced_paths.extend(attr_paths.values())
    missing_paths = [path for path in referenced_paths if not path.exists()]
    raw_coords_path = _resolve_raw_eval_path(x_path)
    if attr_paths:
        raw_gt_paths = {
            name: _resolve_raw_eval_path(path)
            for name, path in attr_paths.items()
        }
    elif y_path is not None:
        raw_gt_paths = {
            _infer_single_target_attr_name(data_info): _resolve_raw_eval_path(y_path)
        }
    else:
        raw_gt_paths = {}

    if missing_paths:
        if not all(_is_normalized_artifact_path(path) for path in missing_paths):
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing dataset artifact(s): {missing_text}")

        attr_eval_affines: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        if not normalize:
            dataset_name = str(data_cfg.get("dataset_name", "")).strip().lower()
            if dataset_name != "stress":
                raise FileNotFoundError(
                    "Missing pre-normalized dataset artifacts for a normalize=false config; "
                    "automatic min-max fallback is only implemented for stress experiments."
                )
            x_path, y_path, attr_paths, raw_gt_paths, attr_eval_affines = _prepare_stress_minmax_fallback(
                data_cfg,
                data_info,
                x_path,
                y_path,
                attr_paths,
            )
            stats_path = None
            logger.info(
                "Dataset artifacts missing for config paths; rebuilt min-max normalized cache from raw data. x=%s",
                x_path,
            )
        else:
            x_path = raw_coords_path
            if y_path is not None:
                y_path = raw_gt_paths[_infer_single_target_attr_name(data_info)]
            attr_paths = {
                name: raw_gt_paths[name]
                for name in attr_paths.keys()
            }
            stats_ref_path = next(iter(attr_paths.values()), y_path)
            inferred_stats_path = (
                _resolve_raw_stats_path(data_cfg, stats_ref_path)
                if stats_ref_path is not None else None
            )
            if stats_path is None and inferred_stats_path is None:
                raise FileNotFoundError(
                    f"Failed to infer stats path for normalized-only dataset config: {missing_paths[0]}"
                )
            chosen_stats_path = Path(inferred_stats_path or stats_path).resolve()
            if y_path is not None and not attr_paths and not _stats_payload_has_node_targets(chosen_stats_path):
                chosen_stats_path = _validation_stats_cache_path(y_path).resolve()
            stats_path = str(chosen_stats_path)
            normalize = True
            logger.info(
                "Dataset artifacts missing for config paths; falling back to raw data + stats normalization. x=%s stats=%s",
                x_path,
                stats_path,
            )
    else:
        attr_eval_affines = {}

    if data_info.get("attr_paths"):
        dataset = MultiViewCoordDataset(
            str(x_path),
            {name: str(path) for name, path in attr_paths.items()},
            normalize=normalize,
            stats_path=stats_path,
        )
        attrs = list(dataset.y.keys())
        gt_paths = {name: Path(path) for name, path in attr_paths.items()}
    else:
        dataset = NodeDataset(
            str(x_path),
            str(y_path),
            normalize=normalize,
            stats_path=stats_path,
        )
        attr_name = _infer_single_target_attr_name(data_info)
        attrs = [attr_name]
        gt_paths = {attr_name: Path(y_path)}

    return DatasetLoadResult(
        dataset=dataset,
        data_info=data_info,
        attrs=attrs,
        train_gt_paths=gt_paths,
        raw_coords_path=raw_coords_path,
        raw_gt_paths=raw_gt_paths,
        attr_eval_affines=attr_eval_affines,
    )


def _infer_single_target_attr_name(data_info: dict) -> str:
    y_path = str(data_info.get("y_path") or "").strip()
    if not y_path:
        return "targets"

    stem = Path(y_path).stem.strip()
    for suffix in ("_normalized", "_denormalized"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if stem.startswith("target_"):
        stem = stem[len("target_") :]
    elif stem.startswith("targets_"):
        stem = stem[len("targets_") :]

    if not stem or stem == "targets":
        return "targets"
    if stem.startswith("data_"):
        return stem
    return f"data_{stem}"


def _select_tensor_block(tensor: torch.Tensor, indexer: slice | np.ndarray) -> torch.Tensor:
    if isinstance(indexer, slice):
        return tensor[indexer]
    return tensor[torch.from_numpy(indexer.astype(np.int64, copy=False))]


def _select_array_block(array: np.ndarray, indexer: slice | np.ndarray) -> np.ndarray:
    return np.asarray(array[indexer])


def _as_feature_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array[:, None]
    return array


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


def _denormalize_values(dataset, attr_name: str, values: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if isinstance(dataset, MultiViewCoordDataset):
        return dataset.denormalize_attr(attr_name, tensor).cpu().numpy()
    return dataset.denormalize_targets(tensor).cpu().numpy()


def _to_feature_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 1:
        return array[:, None]
    return array


def _denormalize_for_eval(values: torch.Tensor | np.ndarray, spec: AttrEvalSpec) -> np.ndarray:
    array = _to_feature_numpy(values)
    if spec.offset is None or spec.scale is None:
        return array
    return array * spec.scale + spec.offset


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
    return list(range(max(0, int(num_timesteps))))


def _compute_psnr_from_mse(mse: float, gt_min: float, gt_max: float) -> float:
    if not np.isfinite(mse) or mse < 0:
        return float("nan")
    if mse == 0:
        return float("inf")
    data_range = float(gt_max - gt_min)
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(abs(float(gt_min)), abs(float(gt_max)), 1.0)
    return float(10.0 * math.log10((data_range * data_range) / mse))


def _compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_arr = np.asarray(gt, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"PSNR shape mismatch: gt={gt_arr.shape}, pred={pred_arr.shape}")
    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    gt_min = float(np.min(gt_arr))
    gt_max = float(np.max(gt_arr))
    return _compute_psnr_from_mse(mse, gt_min, gt_max)


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


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] != 3:
        raise ValueError(f"Expected 1/3/4 image channels, got {array.shape[2]}")

    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 + 1e-6 else 1.0
        array = array * scale
    array = np.nan_to_num(array, nan=255.0, posinf=255.0, neginf=0.0)
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _render_frame(
    mesh_path: Path,
    association: str,
    values: np.ndarray,
    outpath: Path | None,
    clim: tuple[float, float],
    zoom_factor: float,
) -> np.ndarray:
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

    if outpath is not None:
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
    plotter.render()
    image = plotter.screenshot(filename=str(outpath) if outpath is not None else None, return_img=True)
    plotter.close()
    return _ensure_rgb_uint8(image)


def _ensure_runtime_dependencies():
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required to render validation images. Install pyvista in the active Python environment."
        ) from exc

    try:
        import lpips  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "lpips is required to compute LPIPS on rendered images. Install lpips in the active Python environment."
        ) from exc

    try:
        from skimage.metrics import structural_similarity as _  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required to compute SSIM on rendered images. Install scikit-image in the active Python environment."
        ) from exc


def _prepare_pred_output_path(
    out_root: Path,
    dataset_name: str,
    exp_id: str,
    attr_name: str,
    time_index: int,
) -> Path:
    return out_root / dataset_name / attr_name / exp_id / f"{exp_id}_t{time_index:04d}_pred.png"


def _prepare_gt_output_path(
    out_root: Path,
    dataset_name: str,
    attr_name: str,
    time_index: int,
) -> Path:
    return out_root / dataset_name / attr_name / f"gt_t{time_index:04d}.png"


def _default_csv_path(out_root: Path, dataset_name: str, exp_id: str) -> Path:
    return out_root / dataset_name / exp_id / f"{exp_id}_metrics.csv"


def _dataset_name(data_info: dict) -> str:
    name = str(data_info.get("dataset_name", "")).strip()
    return name or "unknown"


def _render_zoom_factor(data_info: dict) -> float:
    dataset_name = str(data_info.get("dataset_name", "")).strip().lower()
    if dataset_name == "ocean":
        return 1.8
    return 1.35


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return _ensure_rgb_uint8(np.asarray(img.convert("RGB")))


def _gt_meta_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _read_gt_cache_clim(image_path: Path) -> tuple[float, float] | None:
    meta_path = _gt_meta_path(image_path)
    if not image_path.exists() or not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    clim = payload.get("clim")
    if not isinstance(clim, list) or len(clim) != 2:
        return None
    try:
        return float(clim[0]), float(clim[1])
    except (TypeError, ValueError):
        return None


def _write_gt_cache_clim(image_path: Path, clim: tuple[float, float]) -> None:
    meta_path = _gt_meta_path(image_path)
    meta_path.write_text(
        json.dumps({"clim": [float(clim[0]), float(clim[1])]}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _can_reuse_gt_cache(image_path: Path, clim: tuple[float, float]) -> bool:
    cached_clim = _read_gt_cache_clim(image_path)
    if cached_clim is None:
        return False
    return bool(np.allclose(np.asarray(cached_clim), np.asarray(clim), rtol=1e-6, atol=1e-12))


def _compute_ssim(gt_image: np.ndarray, pred_image: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    gt_rgb = _ensure_rgb_uint8(gt_image)
    pred_rgb = _ensure_rgb_uint8(pred_image)
    if gt_rgb.shape != pred_rgb.shape:
        raise ValueError(f"SSIM image shape mismatch: gt={gt_rgb.shape}, pred={pred_rgb.shape}")
    return float(structural_similarity(gt_rgb, pred_rgb, channel_axis=-1, data_range=255))


def _build_lpips_model(device: torch.device) -> torch.nn.Module:
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def _lpips_tensor_from_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = _ensure_rgb_uint8(image).astype(np.float32) / 127.5 - 1.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)


def _compute_lpips(
    lpips_model: torch.nn.Module,
    gt_image: np.ndarray,
    pred_image: np.ndarray,
    device: torch.device,
) -> float:
    gt_tensor = _lpips_tensor_from_image(gt_image, device)
    pred_tensor = _lpips_tensor_from_image(pred_image, device)
    with torch.inference_mode():
        score = lpips_model(gt_tensor, pred_tensor)
    return float(score.detach().cpu().reshape(-1)[0])


def _mean_finite(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _write_csv_rows(csv_path: Path, rows: list[dict[str, object]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_experiment(
    experiment_path: str | Path,
    csv_path: str | Path | None = None,
    timestamp: int | None = None,
) -> Path:
    exp_dir = Path(experiment_path).resolve()
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_path = exp_dir / "configs" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    ckpt_path = _pick_checkpoint(exp_dir / "checkpoints").resolve()
    cfg = load_config(str(cfg_path))
    exp_id = str(cfg.get("exp_id") or exp_dir.name)

    eval_setup = _build_evaluation_setup(cfg)
    dataset = eval_setup.dataset
    data_info = eval_setup.data_info
    attrs = eval_setup.attrs

    for attr_name, spec in eval_setup.attr_specs.items():
        if not spec.raw_gt_path.exists():
            raise FileNotFoundError(f"Ground-truth file not found for '{attr_name}': {spec.raw_gt_path}")

    raw_coords = np.load(str(eval_setup.raw_coords_path), mmap_mode="r", allow_pickle=False)
    if raw_coords.ndim != 2 or raw_coords.shape[1] < 1:
        raise ValueError(f"Expected coords array with shape (N, D>=1), got {raw_coords.shape}")
    if int(raw_coords.shape[0]) != int(len(dataset)):
        raise ValueError(
            f"Raw coords / dataset length mismatch: raw={raw_coords.shape[0]} dataset={len(dataset)}"
        )

    time_values = raw_coords[:, -1]
    time_indexers = _compute_time_indexers(time_values)
    num_timesteps = len(time_indexers)
    if num_timesteps == 0:
        raise ValueError("No timesteps found in coords array.")

    if timestamp is not None:
        if timestamp < 0 or timestamp >= num_timesteps:
            raise ValueError(
                f"--timestamp {timestamp} is out of range. Valid range: [0, {num_timesteps - 1}]"
            )
        selected_time_indices = [int(timestamp)]
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

    candidate_meshes = _collect_mesh_candidates(eval_setup.raw_coords_path)
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

    _ensure_runtime_dependencies()

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
    lpips_model = _build_lpips_model(device)

    batch_size = int(
        cfg.get("training", {}).get(
            "pred_batch_size",
            cfg.get("training", {}).get("batch_size", 8192),
        )
    )
    gt_arrays = {
        name: np.load(str(spec.raw_gt_path), mmap_mode="r", allow_pickle=False)
        for name, spec in eval_setup.attr_specs.items()
    }

    dataset_name = _dataset_name(data_info)
    out_root = Path("validate_out")
    resolved_csv_path = (
        Path(csv_path).resolve()
        if csv_path
        else _default_csv_path(out_root, dataset_name, exp_id).resolve()
    )

    logger.info("Experiment: %s", exp_id)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Config: %s", cfg_path)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Model coords path: %s", data_info["x_path"])
    logger.info("Eval raw coords path: %s", eval_setup.raw_coords_path)
    logger.info("Attrs: %s", attrs)
    logger.info("CSV output: %s", resolved_csv_path)
    logger.info("Render zoom factor: %.2f", _render_zoom_factor(data_info))
    logger.info(
        "Rendering timesteps: %s",
        [f"{step['time_index']}({step['raw_time']})" for step in selected_steps],
    )

    attr_clims: dict[str, tuple[float, float] | None] = {name: None for name in attrs}
    psnr_map: dict[tuple[str, int], float] = {}

    logger.info("Pass 1/2: collecting per-attr color ranges and PSNR")
    for step in selected_steps:
        coords_block = _select_tensor_block(dataset.x, step["indexer"])
        pred_map = _predict_block(model, coords_block, attrs, batch_size, device)
        for attr_name in attrs:
            spec = eval_setup.attr_specs[attr_name]
            pred_denorm = _denormalize_for_eval(pred_map[attr_name], spec)
            gt_values = _as_feature_matrix(_select_array_block(gt_arrays[attr_name], step["indexer"]))
            gt_vis = _to_visual_scalar(gt_values)
            attr_clims[attr_name] = _merge_range(attr_clims[attr_name], gt_vis)
            psnr_map[(attr_name, int(step["time_index"]))] = _compute_psnr(gt_values, pred_denorm)

    normalized_clims = {
        attr_name: _normalize_clim(clim)
        for attr_name, clim in attr_clims.items()
    }
    zoom_factor = _render_zoom_factor(data_info)
    per_timestep_rows: list[dict[str, object]] = []

    logger.info("Pass 2/2: rendering frames to %s", out_root.resolve())
    for step in selected_steps:
        coords_block = _select_tensor_block(dataset.x, step["indexer"])
        pred_map = _predict_block(model, coords_block, attrs, batch_size, device)
        for attr_name in attrs:
            spec = eval_setup.attr_specs[attr_name]
            pred_denorm = _denormalize_for_eval(pred_map[attr_name], spec)
            gt_values = _as_feature_matrix(_select_array_block(gt_arrays[attr_name], step["indexer"]))
            pred_vis = _to_visual_scalar(pred_denorm)
            gt_vis = _to_visual_scalar(gt_values)
            pred_out = _prepare_pred_output_path(
                out_root,
                dataset_name,
                exp_id,
                attr_name,
                int(step["time_index"]),
            )
            gt_out = _prepare_gt_output_path(
                out_root,
                dataset_name,
                attr_name,
                int(step["time_index"]),
            )
            pred_img = _render_frame(
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                values=pred_vis,
                outpath=pred_out,
                clim=normalized_clims[attr_name],
                zoom_factor=zoom_factor,
            )
            if _can_reuse_gt_cache(gt_out, normalized_clims[attr_name]):
                gt_img = _load_image(gt_out)
                logger.info(
                    "Reused cached GT render for attr=%s timestep=%s -> %s",
                    attr_name,
                    step["time_index"],
                    gt_out,
                )
            else:
                gt_img = _render_frame(
                    mesh_path=step["mesh_path"],
                    association=step["mesh_association"],
                    values=gt_vis,
                    outpath=gt_out,
                    clim=normalized_clims[attr_name],
                    zoom_factor=zoom_factor,
                )
                _write_gt_cache_clim(gt_out, normalized_clims[attr_name])
            ssim_value = _compute_ssim(gt_img, pred_img)
            lpips_value = _compute_lpips(lpips_model, gt_img, pred_img, device)
            per_timestep_rows.append(
                {
                    "row_type": "per_timestep",
                    "exp_id": exp_id,
                    "model_name": str(cfg.get("model", {}).get("name", "")),
                    "dataset_name": dataset_name,
                    "checkpoint_path": str(ckpt_path),
                    "attr": attr_name,
                    "time_index": int(step["time_index"]),
                    "raw_time": float(step["raw_time"]),
                    "num_samples": int(step["sample_count"]),
                    "num_timesteps": int(num_timesteps),
                    "gt_render_path": str(gt_out.resolve()),
                    "pred_render_path": str(pred_out.resolve()),
                    "psnr": psnr_map[(attr_name, int(step["time_index"]))],
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }
            )
            logger.info(
                "Rendered attr=%s timestep=%s -> pred=%s gt=%s psnr=%.6f ssim=%.6f lpips=%.6f",
                attr_name,
                step["time_index"],
                pred_out,
                gt_out,
                float(psnr_map[(attr_name, int(step["time_index"]))]),
                float(ssim_value),
                float(lpips_value),
            )

    summary_rows: list[dict[str, object]] = []
    for attr_name in attrs:
        attr_rows = [row for row in per_timestep_rows if row["attr"] == attr_name]
        summary_rows.append(
            {
                "row_type": "attr_mean",
                "exp_id": exp_id,
                "model_name": str(cfg.get("model", {}).get("name", "")),
                "dataset_name": dataset_name,
                "checkpoint_path": str(ckpt_path),
                "attr": attr_name,
                "time_index": "",
                "raw_time": "",
                "num_samples": "",
                "num_timesteps": int(num_timesteps),
                "gt_render_path": "",
                "pred_render_path": "",
                "psnr": _mean_finite([float(row["psnr"]) for row in attr_rows]),
                "ssim": _mean_finite([float(row["ssim"]) for row in attr_rows]),
                "lpips": _mean_finite([float(row["lpips"]) for row in attr_rows]),
            }
        )

    summary_rows.append(
        {
            "row_type": "global_mean",
            "exp_id": exp_id,
            "model_name": str(cfg.get("model", {}).get("name", "")),
            "dataset_name": dataset_name,
            "checkpoint_path": str(ckpt_path),
            "attr": "__all__",
            "time_index": "",
            "raw_time": "",
            "num_samples": "",
            "num_timesteps": int(num_timesteps),
            "gt_render_path": "",
            "pred_render_path": "",
            "psnr": _mean_finite([float(row["psnr"]) for row in per_timestep_rows]),
            "ssim": _mean_finite([float(row["ssim"]) for row in per_timestep_rows]),
            "lpips": _mean_finite([float(row["lpips"]) for row in per_timestep_rows]),
        }
    )

    rows = per_timestep_rows + summary_rows
    _write_csv_rows(resolved_csv_path, rows)
    logger.info("Saved validation metrics to CSV: %s", resolved_csv_path)
    return resolved_csv_path


def main():
    setup_logging()
    args = _parse_args()
    validate_experiment(
        args.experiment_path,
        csv_path=(args.csv or None),
        timestamp=args.timestamp,
    )


if __name__ == "__main__":
    main()

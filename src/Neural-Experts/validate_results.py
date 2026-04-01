from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
for path in (str(THIS_DIR), str(THIS_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from runtime_limits import apply_runtime_thread_limits, configure_threading_env

configure_threading_env()

import numpy as np
import torch
from PIL import Image

THREAD_LIMITS = apply_runtime_thread_limits()

from mesh.common import REPO_ROOT, ensure_sys_path, load_config

ensure_sys_path()

from mesh.inference import (
    compute_psnr,
    compute_time_indexers,
    denormalize_targets,
    load_checkpoint,
    normalize_coords,
    predict_block,
    unwrap_model_state,
)
from models import build_model

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
class SourceSpec:
    model_source_path: Path
    raw_source_path: Path
    apply_runtime_normalize: bool


@dataclass(frozen=True)
class TargetSpec:
    model_target_path: Path
    raw_gt_path: Path
    offset: np.ndarray | None
    scale: np.ndarray | None
    apply_runtime_denormalize: bool
    raw_replacements: tuple[tuple[float, float], ...]
    psnr_data_range: float


@dataclass(frozen=True)
class ExperimentSetup:
    exp_dir: Path
    exp_id: str
    cfg_path: Path
    ckpt_path: Path
    cfg: dict[str, Any]
    payload: Any
    dataset_name: str
    attr_name: str
    association: str
    source: SourceSpec
    target: TargetSpec


@dataclass(frozen=True)
class MinMaxRule:
    axis: str
    denominator: str
    replacements: tuple[tuple[float, float], ...] = ()
    cache_flavor: str = "minmax"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render predictions and ground truth for one Neural-Experts experiment directory."
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to one Neural-Experts experiment directory, e.g. ./experiments/Neural Expert/stress_point_U",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="0-based index into sorted unique timesteps. Default: render all timesteps.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Override render zoom factor. Default: dataset-specific internal default.",
    )
    parser.add_argument(
        "--clip-max-percentile",
        type=float,
        default=None,
        help="Clip the render color upper bound to this GT percentile. Example: 99.7. Default: no clipping.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Output root for rendered images and default CSV path. Default: ./validate_out",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to metrics CSV. Default: validate_out/<dataset_name>/<exp_id>/<exp_id>_metrics.csv",
    )
    return parser.parse_args()


def _ensure_required_conda_env(expected_env: str = "compression") -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    executable = Path(sys.executable).resolve()
    sys_prefix = Path(sys.prefix).resolve()

    env_name_ok = env_name == expected_env
    prefix_ok = any(part == expected_env for part in conda_prefix.replace("\\", "/").split("/") if part)
    executable_ok = any(part == expected_env for part in executable.parts)
    sys_prefix_ok = any(part == expected_env for part in sys_prefix.parts)

    if env_name_ok or prefix_ok or executable_ok or sys_prefix_ok:
        logger.info(
            "Validated conda environment: expected=%s active_env=%s executable=%s",
            expected_env,
            env_name or "<unknown>",
            executable,
        )
        return

    raise RuntimeError(
        "Neural-Experts validation must run inside the conda environment "
        f"'{expected_env}'. Current interpreter: {executable}. "
        f"Current CONDA_DEFAULT_ENV: {env_name or '<unset>'}. "
        f"Use: conda run -n {expected_env} python Neural-Experts/validate_results.py ..."
    )


def _register_numpy_core_aliases() -> None:
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


def _load_checkpoint_with_aliases(path: Path) -> Any:
    try:
        return load_checkpoint(path, device="cpu")
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy._core", "numpy._core.multiarray"}:
            raise
        _register_numpy_core_aliases()
        return load_checkpoint(path, device="cpu")


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D array, got {array.shape}")


def _as_stat_matrix(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _is_normalized_artifact_path(path: Path) -> bool:
    return any(path.stem.endswith(suffix) for suffix in ("_normalized", "_denormalized"))


def _raw_counterpart_path(path: Path) -> Path:
    for suffix in ("_normalized", "_denormalized"):
        if path.stem.endswith(suffix):
            return path.with_name(f"{path.stem[:-len(suffix)]}{path.suffix}")
    return path


def _resolve_raw_eval_path(path_value: str | Path) -> Path:
    path = Path(path_value).resolve()
    candidate = _raw_counterpart_path(path)
    if candidate != path and not candidate.exists():
        raise FileNotFoundError(f"Expected raw counterpart for {path}, but not found: {candidate}")
    return candidate.resolve()


def _repair_path_string(path_value: str | None) -> str | None:
    if not path_value:
        return None

    raw_text = str(path_value).strip()
    if not raw_text:
        return None

    path = Path(raw_text)
    if path.exists():
        return str(path.resolve())

    normalized = raw_text.replace("\\", "/")
    if "/src/" in normalized:
        suffix = normalized.split("/src/", 1)[1]
        candidate = (REPO_ROOT / Path(suffix)).resolve()
        return str(candidate)

    if path.is_absolute():
        return str(path)
    return str(path.resolve())


def _repair_cfg_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("DATA", {})
    model_cfg = cfg.get("MODEL", {})
    train_cfg = cfg.get("TRAINING", {})
    pretrain_cfg = train_cfg.get("pretrain_assignment", {}) or {}
    validation_cfg = cfg.get("VALIDATION", {})

    for key in ("source_path", "target_path", "target_stats_path"):
        if data_cfg.get(key):
            data_cfg[key] = _repair_path_string(str(data_cfg[key]))

    if model_cfg.get("manager_pt_path"):
        model_cfg["manager_pt_path"] = _repair_path_string(str(model_cfg["manager_pt_path"]))

    if pretrain_cfg.get("cache_path"):
        pretrain_cfg["cache_path"] = _repair_path_string(str(pretrain_cfg["cache_path"]))

    if validation_cfg.get("checkpoint_path"):
        validation_cfg["checkpoint_path"] = _repair_path_string(str(validation_cfg["checkpoint_path"]))

    train_cfg["pretrain_assignment"] = pretrain_cfg
    cfg["DATA"] = data_cfg
    cfg["MODEL"] = model_cfg
    cfg["TRAINING"] = train_cfg
    cfg["VALIDATION"] = validation_cfg
    return cfg


def _pick_config_path(exp_dir: Path) -> Path:
    candidates = [
        exp_dir / "validate_artifacts" / "config.yaml",
        exp_dir / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config not found under experiment directory: {exp_dir}")


def _extract_epoch(path: Path) -> int | None:
    match = re.search(r"epoch(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _pick_checkpoint(exp_dir: Path) -> Path:
    candidate_dirs = [
        exp_dir / "validate_artifacts",
        exp_dir,
    ]
    ckpt_files: list[Path] = []
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        ckpt_files.extend(path.resolve() for path in candidate_dir.glob("*.pth"))
        if ckpt_files:
            break

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found under experiment directory: {exp_dir}")

    final_ckpts = [path for path in ckpt_files if _extract_epoch(path) is None]
    if final_ckpts:
        return max(final_ckpts, key=lambda path: (path.stat().st_mtime, path.name))

    epoch_ckpts = [(path, _extract_epoch(path)) for path in ckpt_files]
    valid_epoch_ckpts = [(path, epoch) for path, epoch in epoch_ckpts if epoch is not None]
    if valid_epoch_ckpts:
        return max(valid_epoch_ckpts, key=lambda item: (item[1], item[0].stat().st_mtime))[0]

    return max(ckpt_files, key=lambda path: (path.stat().st_mtime, path.name))


def _select_default_timestamps(num_timesteps: int) -> list[int]:
    return list(range(max(0, int(num_timesteps))))


def _indexer_size(indexer: slice | np.ndarray) -> int:
    if isinstance(indexer, slice):
        return int(indexer.stop - indexer.start)
    return int(indexer.shape[0])


def _select_array_block(array: np.ndarray, indexer: slice | np.ndarray) -> np.ndarray:
    return np.asarray(array[indexer])


def _normalized_cache_root() -> Path:
    return Path("validate_out") / "_normalized_cache"


def _stats_cache_root() -> Path:
    return Path("validate_out") / "_stats_cache"


def _cache_relative_parent(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).parent
    except ValueError:
        return Path(path.parent.name)


def _normalized_cache_path(raw_path: Path, flavor: str) -> Path:
    flavor = _sanitize_token(flavor)
    return _normalized_cache_root() / _cache_relative_parent(raw_path) / f"{raw_path.stem}_{flavor}_validate_v2{raw_path.suffix}"


def _source_minmax_rule(dataset_name: str) -> MinMaxRule:
    return MinMaxRule(axis="perdim", denominator="range", cache_flavor=f"{dataset_name}_source_perdim_minmax")


def _target_minmax_rule(dataset_name: str, attr_name: str) -> MinMaxRule:
    dataset_name = str(dataset_name).strip().lower()
    attr_name = str(attr_name).strip()
    if dataset_name == "ocean":
        replacements: tuple[tuple[float, float], ...] = ()
        if attr_name == "fort63":
            replacements = ((-99999.0, -3.8),)
        return MinMaxRule(
            axis="global",
            denominator="range",
            replacements=replacements,
            cache_flavor=f"ocean_{_sanitize_token(attr_name)}_global_minmax",
        )
    if dataset_name == "stress":
        denominator = "max" if attr_name == "cell_S_IntegrationPoints" else "range"
        return MinMaxRule(
            axis="perdim",
            denominator=denominator,
            cache_flavor=f"stress_{_sanitize_token(attr_name)}_{denominator}_perdim_minmax",
        )
    raise ValueError(f"Unsupported dataset_name for min-max rule: {dataset_name}")


def _apply_replacements(values: np.ndarray, replacements: tuple[tuple[float, float], ...]) -> np.ndarray:
    if not replacements:
        return np.asarray(values)
    out = np.array(values, copy=True)
    for old, new in replacements:
        out[out == old] = new
    return out


def _stream_minmax_stats(raw_path: Path, rule: MinMaxRule) -> tuple[np.ndarray, np.ndarray, float]:
    raw = _ensure_2d(np.load(str(raw_path), mmap_mode="r", allow_pickle=False))
    rows = int(raw.shape[0])
    dims = int(raw.shape[1])
    chunk_rows = max(1, min(rows, max(1, 1_000_000 // max(dims, 1))))

    global_min = float("inf")
    global_max = float("-inf")
    if rule.axis == "global":
        scalar_min = float("inf")
        scalar_max = float("-inf")
    else:
        scalar_min = None
        scalar_max = None

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        block = np.asarray(raw[start:end], dtype=np.float32)
        block = _apply_replacements(block, rule.replacements).astype(np.float32, copy=False)
        block_global_min = float(np.min(block))
        block_global_max = float(np.max(block))
        global_min = min(global_min, block_global_min)
        global_max = max(global_max, block_global_max)
        if rule.axis == "global":
            scalar_min = min(float(scalar_min), block_global_min)
            scalar_max = max(float(scalar_max), block_global_max)
        else:
            block_min = np.min(block, axis=0, keepdims=True)
            block_max = np.max(block, axis=0, keepdims=True)
            scalar_min = block_min if scalar_min is None else np.minimum(scalar_min, block_min)
            scalar_max = block_max if scalar_max is None else np.maximum(scalar_max, block_max)

    if rule.axis == "global":
        min_arr = np.full((1, dims), float(scalar_min), dtype=np.float32)
        max_arr = np.full((1, dims), float(scalar_max), dtype=np.float32)
    else:
        min_arr = np.asarray(scalar_min, dtype=np.float32).reshape(1, dims)
        max_arr = np.asarray(scalar_max, dtype=np.float32).reshape(1, dims)

    data_range = float(global_max - global_min)
    return min_arr, max_arr, data_range


def _denominator_from_rule(raw_min: np.ndarray, raw_max: np.ndarray, rule: MinMaxRule) -> np.ndarray:
    if rule.denominator == "max":
        denom = np.asarray(raw_max, dtype=np.float32)
    elif rule.denominator == "range":
        denom = np.asarray(raw_max, dtype=np.float32) - np.asarray(raw_min, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported denominator rule: {rule.denominator}")
    return np.where(np.abs(denom) < 1.0e-12, np.ones_like(denom, dtype=np.float32), denom).astype(np.float32)


@lru_cache(maxsize=None)
def _minmax_affine(
    raw_path_value: str,
    axis: str,
    denominator: str,
    replacements_key: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    rule = MinMaxRule(axis=axis, denominator=denominator, replacements=replacements_key, cache_flavor="cached")
    raw_min, raw_max, data_range = _stream_minmax_stats(Path(raw_path_value).resolve(), rule)
    denom = _denominator_from_rule(raw_min, raw_max, rule)
    scale = denom / 2.0
    offset = raw_min + scale
    return offset.astype(np.float32), scale.astype(np.float32), float(data_range)


def _ensure_minmax_normalized_cache(raw_path: Path, rule: MinMaxRule) -> Path:
    cache_path = _normalized_cache_path(raw_path, flavor=rule.cache_flavor)
    if cache_path.exists():
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    raw = _ensure_2d(np.load(str(raw_path), mmap_mode="r", allow_pickle=False))
    raw_min, raw_max, _ = _stream_minmax_stats(raw_path, rule)
    denom = _denominator_from_rule(raw_min, raw_max, rule)
    rows = int(raw.shape[0])
    dims = int(raw.shape[1])
    rows_per_chunk = max(1, min(rows, max(1, 1_000_000 // max(dims, 1))))
    normalized = np.lib.format.open_memmap(
        str(tmp_path),
        mode="w+",
        dtype=np.float32,
        shape=raw.shape,
    )
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        block = np.asarray(raw[start:end], dtype=np.float32)
        block = _apply_replacements(block, rule.replacements).astype(np.float32, copy=False)
        normalized[start:end] = ((block - raw_min) / denom) * 2.0 - 1.0
    del normalized
    tmp_path.replace(cache_path)
    logger.info("Created min-max normalized cache: %s", cache_path)
    return cache_path.resolve()


def _affine_cache_path(model_target_path: Path, raw_target_path: Path) -> Path:
    key = f"{model_target_path.resolve()}|{raw_target_path.resolve()}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    stem = f"{_sanitize_token(model_target_path.stem)}__to__{_sanitize_token(raw_target_path.stem)}_{digest}.npz"
    return _stats_cache_root() / stem


def _stats_path_from_cfg(cfg: dict[str, Any]) -> Path | None:
    path_value = cfg["DATA"].get("target_stats_path")
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.exists():
        return path.resolve()
    return None


@lru_cache(maxsize=None)
def _load_npz_payload(npz_path: str) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _resolve_stats_arrays(
    payload: Any,
    cfg: dict[str, Any],
    raw_source_path: Path,
    raw_target_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(payload, dict):
        x_mean = payload.get("x_mean")
        x_std = payload.get("x_std")
        y_mean = payload.get("y_mean")
        y_std = payload.get("y_std")
        if x_mean is not None and x_std is not None and y_mean is not None and y_std is not None:
            return (
                _as_stat_matrix(np.asarray(x_mean, dtype=np.float32)),
                _as_stat_matrix(np.asarray(x_std, dtype=np.float32)),
                _as_stat_matrix(np.asarray(y_mean, dtype=np.float32)),
                _as_stat_matrix(np.asarray(y_std, dtype=np.float32)),
            )

    from datasets_loader.Mesh import _load_or_compute_stats

    source = np.load(str(raw_source_path), mmap_mode="r")
    target = _ensure_2d(np.load(str(raw_target_path), mmap_mode="r"))
    stats_path = _stats_path_from_cfg(cfg)
    x_mean, x_std, y_mean, y_std = _load_or_compute_stats(
        source=source,
        target=target,
        stats_path=stats_path,
        stats_key=str(cfg["DATA"]["stats_key"]),
        input_dim=int(source.shape[1]),
        target_dim=int(target.shape[1]),
        load_input_stats=True,
        load_target_stats=True,
    )
    return x_mean.numpy(), x_std.numpy(), y_mean.numpy(), y_std.numpy()


def _ensure_zscore_normalized_cache(
    raw_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    flavor: str,
) -> Path:
    cache_path = _normalized_cache_path(raw_path, flavor=f"zscore_{flavor}")
    if cache_path.exists():
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    raw = _ensure_2d(np.load(str(raw_path), mmap_mode="r", allow_pickle=False))
    mean = _as_stat_matrix(mean)
    std = np.maximum(_as_stat_matrix(std), 1.0e-12)
    rows = int(raw.shape[0])
    dims = int(raw.shape[1])
    rows_per_chunk = max(1, min(rows, max(1, 1_000_000 // max(dims, 1))))
    normalized = np.lib.format.open_memmap(
        str(tmp_path),
        mode="w+",
        dtype=np.float32,
        shape=raw.shape,
    )
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        block = np.asarray(raw[start:end], dtype=np.float32)
        normalized[start:end] = (block - mean) / std
    del normalized
    tmp_path.replace(cache_path)
    logger.info("Created z-score normalized cache: %s", cache_path)
    return cache_path.resolve()


def _stress_minmax_denominator(raw_path: Path, raw_min: np.ndarray, raw_max: np.ndarray) -> np.ndarray:
    if raw_path.stem == "target_cell_S_IntegrationPoints":
        denom = raw_max
    else:
        denom = raw_max - raw_min
    return np.where(np.abs(denom) < 1e-12, np.ones_like(denom, dtype=np.float32), denom).astype(np.float32)


@lru_cache(maxsize=None)
def _stress_minmax_params(raw_path_value: str) -> tuple[np.ndarray, np.ndarray]:
    raw_path = Path(raw_path_value).resolve()
    raw = _ensure_2d(np.load(str(raw_path), mmap_mode="r", allow_pickle=False))
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
    cache_path = _normalized_cache_path(raw_path, flavor="stress_minmax")
    if cache_path.exists():
        return cache_path.resolve()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    raw = _ensure_2d(np.load(str(raw_path), mmap_mode="r", allow_pickle=False))
    raw_min, denom = _stress_minmax_params(str(raw_path))
    rows = int(raw.shape[0])
    dims = int(raw.shape[1])
    rows_per_chunk = max(1, min(rows, max(1, 1_000_000 // max(dims, 1))))
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
    logger.info("Created stress min-max normalized cache: %s", cache_path)
    return cache_path.resolve()


def _fit_affine_streaming(model_target_path: Path, raw_target_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _affine_cache_path(model_target_path, raw_target_path)
    if cache_path.exists():
        payload = _load_npz_payload(str(cache_path.resolve()))
        cached_model = str(payload.get("model_target_path", ""))
        cached_raw = str(payload.get("raw_target_path", ""))
        if cached_model == str(model_target_path.resolve()) and cached_raw == str(raw_target_path.resolve()):
            return _as_stat_matrix(payload["offset"]), _as_stat_matrix(payload["scale"])

    model_arr = _ensure_2d(np.load(str(model_target_path), mmap_mode="r", allow_pickle=False))
    raw_arr = _ensure_2d(np.load(str(raw_target_path), mmap_mode="r", allow_pickle=False))
    if model_arr.shape != raw_arr.shape:
        raise ValueError(
            f"Affine fit shape mismatch: model={model_arr.shape}, raw={raw_arr.shape} for "
            f"{model_target_path} -> {raw_target_path}"
        )

    rows = int(model_arr.shape[0])
    dims = int(model_arr.shape[1])
    chunk_rows = max(1, min(rows, max(1, 1_000_000 // max(dims, 1))))
    n = 0.0
    sx = np.zeros((dims,), dtype=np.float64)
    sy = np.zeros((dims,), dtype=np.float64)
    sxx = np.zeros((dims,), dtype=np.float64)
    sxy = np.zeros((dims,), dtype=np.float64)

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        x_block = np.asarray(model_arr[start:end], dtype=np.float64)
        y_block = np.asarray(raw_arr[start:end], dtype=np.float64)
        block_n = float(end - start)
        n += block_n
        sx += x_block.sum(axis=0)
        sy += y_block.sum(axis=0)
        sxx += np.square(x_block).sum(axis=0)
        sxy += (x_block * y_block).sum(axis=0)

    denom = n * sxx - sx * sx
    safe = np.abs(denom) >= 1.0e-12
    scale = np.ones((dims,), dtype=np.float64)
    scale[safe] = (n * sxy[safe] - sx[safe] * sy[safe]) / denom[safe]
    offset = (sy - scale * sx) / max(n, 1.0)

    sample_count = min(rows, 8192)
    if sample_count > 0:
        if sample_count == rows:
            sample_idx = np.arange(rows, dtype=np.int64)
        else:
            sample_idx = np.linspace(0, rows - 1, num=sample_count, dtype=np.int64)
        x_sample = np.asarray(model_arr[sample_idx], dtype=np.float64)
        y_sample = np.asarray(raw_arr[sample_idx], dtype=np.float64)
        recon = x_sample * scale[None, :] + offset[None, :]
        err = np.abs(recon - y_sample)
        y_range = np.max(y_sample, axis=0) - np.min(y_sample, axis=0)
        tol = np.maximum(5.0e-4, np.maximum(y_range * 1.0e-3, 1.0e-6))
        if np.any(np.max(err, axis=0) > tol):
            raise ValueError(
                f"Failed affine validation for {model_target_path} -> {raw_target_path}: "
                f"max_err={np.max(err, axis=0)} tol={tol}"
            )

    offset_out = offset.astype(np.float32).reshape(1, -1)
    scale_out = scale.astype(np.float32).reshape(1, -1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(cache_path),
        model_target_path=np.asarray(str(model_target_path.resolve())),
        raw_target_path=np.asarray(str(raw_target_path.resolve())),
        offset=offset_out,
        scale=scale_out,
    )
    logger.info("Cached target affine: %s", cache_path)
    return offset_out, scale_out


def _build_source_spec(
    cfg: dict[str, Any],
    dataset_name: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> SourceSpec:
    configured_source = Path(str(cfg["DATA"]["source_path"]))
    raw_source = _raw_counterpart_path(configured_source)
    raw_source_path = raw_source if raw_source.exists() else configured_source

    if configured_source.exists():
        return SourceSpec(
            model_source_path=configured_source.resolve(),
            raw_source_path=raw_source_path.resolve(),
            apply_runtime_normalize=False,
        )

    if not raw_source.exists():
        raise FileNotFoundError(f"Source data not found: configured={configured_source} raw={raw_source}")

    if _is_normalized_artifact_path(configured_source):
        model_source_path = _ensure_minmax_normalized_cache(raw_source, _source_minmax_rule(dataset_name))
        return SourceSpec(
            model_source_path=model_source_path,
            raw_source_path=raw_source.resolve(),
            apply_runtime_normalize=False,
        )

    return SourceSpec(
        model_source_path=raw_source.resolve(),
        raw_source_path=raw_source.resolve(),
        apply_runtime_normalize=bool(cfg["DATA"].get("normalize_inputs", False)),
    )


def _build_target_spec(
    cfg: dict[str, Any],
    dataset_name: str,
    attr_name: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> TargetSpec:
    configured_target = Path(str(cfg["DATA"]["target_path"]))
    raw_target = _raw_counterpart_path(configured_target)
    raw_gt_path = raw_target if raw_target.exists() else configured_target
    target_rule = _target_minmax_rule(dataset_name, attr_name)

    if configured_target.exists():
        model_target_path = configured_target.resolve()
        if raw_target.exists() and _is_normalized_artifact_path(configured_target):
            offset, scale, data_range = _minmax_affine(
                str(raw_target.resolve()),
                axis=target_rule.axis,
                denominator=target_rule.denominator,
                replacements_key=target_rule.replacements,
            )
            return TargetSpec(
                model_target_path=model_target_path,
                raw_gt_path=raw_target.resolve(),
                offset=offset,
                scale=scale,
                apply_runtime_denormalize=False,
                raw_replacements=target_rule.replacements,
                psnr_data_range=float(max(data_range, 1.0e-12)),
            )
        return TargetSpec(
            model_target_path=model_target_path,
            raw_gt_path=raw_gt_path.resolve(),
            offset=None,
            scale=None,
            apply_runtime_denormalize=bool(cfg["DATA"].get("normalize_targets", False)),
            raw_replacements=(),
            psnr_data_range=float("nan"),
        )

    if not raw_target.exists():
        raise FileNotFoundError(f"Target data not found: configured={configured_target} raw={raw_target}")

    if _is_normalized_artifact_path(configured_target):
        model_target_path = _ensure_minmax_normalized_cache(raw_target, target_rule)
        offset, scale, data_range = _minmax_affine(
            str(raw_target.resolve()),
            axis=target_rule.axis,
            denominator=target_rule.denominator,
            replacements_key=target_rule.replacements,
        )
        return TargetSpec(
            model_target_path=model_target_path,
            raw_gt_path=raw_target.resolve(),
            offset=_as_stat_matrix(offset),
            scale=_as_stat_matrix(scale),
            apply_runtime_denormalize=False,
            raw_replacements=target_rule.replacements,
            psnr_data_range=float(max(data_range, 1.0e-12)),
        )

    return TargetSpec(
        model_target_path=raw_target.resolve(),
        raw_gt_path=raw_target.resolve(),
        offset=None,
        scale=None,
        apply_runtime_denormalize=bool(cfg["DATA"].get("normalize_targets", False)),
        raw_replacements=(),
        psnr_data_range=float("nan"),
    )


def _build_experiment_setup(experiment_path: str | Path) -> ExperimentSetup:
    exp_dir = Path(experiment_path).resolve()
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_path = _pick_config_path(exp_dir)
    ckpt_path = _pick_checkpoint(exp_dir)
    cfg = _repair_cfg_paths(load_config(cfg_path))
    payload = _load_checkpoint_with_aliases(ckpt_path)

    dataset_name = str(cfg["DATA"]["dataset_name"]).strip().lower()
    attr_name = str(cfg["DATA"]["attr_name"]).strip()
    association = str(cfg["DATA"]["association"]).strip().lower()
    if dataset_name not in {"ocean", "stress"}:
        raise ValueError(f"Unsupported dataset_name in Neural-Experts validation: {dataset_name}")
    if association not in {"point", "cell"}:
        raise ValueError(f"Unsupported association in Neural-Experts validation: {association}")

    source_candidate = Path(str(cfg["DATA"]["source_path"]))
    target_candidate = Path(str(cfg["DATA"]["target_path"]))
    raw_source_path = _raw_counterpart_path(source_candidate)
    raw_target_path = _raw_counterpart_path(target_candidate)
    if not raw_source_path.exists():
        raw_source_path = source_candidate
    if not raw_target_path.exists():
        raw_target_path = target_candidate

    x_mean, x_std, y_mean, y_std = _resolve_stats_arrays(payload, cfg, raw_source_path, raw_target_path)
    source_spec = _build_source_spec(cfg, dataset_name, x_mean=x_mean, x_std=x_std)
    target_spec = _build_target_spec(cfg, dataset_name, attr_name=attr_name, y_mean=y_mean, y_std=y_std)

    return ExperimentSetup(
        exp_dir=exp_dir,
        exp_id=exp_dir.name,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        cfg=cfg,
        payload=payload,
        dataset_name=dataset_name,
        attr_name=attr_name,
        association=association,
        source=source_spec,
        target=target_spec,
    )


def _as_feature_matrix(values: np.ndarray) -> np.ndarray:
    return _ensure_2d(values)


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
        delta = max(abs(lo), 1.0) * 1.0e-6
        return lo - delta, hi + delta
    return lo, hi


def _clip_upper_clim(
    clim: tuple[float, float],
    gt_visual_blocks: list[np.ndarray],
    percentile: float | None,
) -> tuple[float, float]:
    lo, hi = _normalize_clim(clim)
    if percentile is None:
        return lo, hi

    finite_blocks: list[np.ndarray] = []
    for block in gt_visual_blocks:
        flat = np.asarray(block, dtype=np.float32).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size:
            finite_blocks.append(finite)
    if not finite_blocks:
        return lo, hi

    merged = np.concatenate(finite_blocks, axis=0)
    clipped_hi = float(np.percentile(merged, float(percentile)))
    if not np.isfinite(clipped_hi) or clipped_hi <= lo or clipped_hi >= hi:
        return lo, hi
    return lo, clipped_hi


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
        associations: list[str] = []
        if num_points == sample_count:
            associations.append("point")
        if num_cells == sample_count:
            associations.append("cell")
        if not associations:
            continue

        if preferred_association in associations:
            association = str(preferred_association)
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
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 + 1.0e-6 else 1.0
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
        cmap="viridis",
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


def _output_attr_name(attr_name: str) -> str:
    token = str(attr_name).strip()
    if not token:
        return "targets"
    if token == "targets" or token.startswith("data_"):
        return token
    return f"data_{token}"


def _render_zoom_factor(dataset_name: str) -> float:
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
    return bool(np.allclose(np.asarray(cached_clim), np.asarray(clim), rtol=1.0e-6, atol=1.0e-12))


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


def _write_csv_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_target_eval_block(
    gt_array: np.ndarray,
    indexer: slice | np.ndarray,
    target_spec: TargetSpec,
) -> np.ndarray:
    values = _as_feature_matrix(_select_array_block(gt_array, indexer).astype(np.float32))
    if target_spec.raw_replacements:
        values = _apply_replacements(values, target_spec.raw_replacements).astype(np.float32, copy=False)
    return values


def _compute_eval_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mse = float(np.mean(np.square(pred - gt)))
    if mse == 0.0:
        return float("inf")
    resolved_range = float(data_range)
    if not np.isfinite(resolved_range) or resolved_range <= 0:
        resolved_range = float(np.max(gt) - np.min(gt))
        if not np.isfinite(resolved_range) or resolved_range <= 0:
            resolved_range = max(abs(float(np.min(gt))), abs(float(np.max(gt))), 1.0)
    return float(10.0 * np.log10((resolved_range * resolved_range) / (mse + 1.0e-12)))


def _map_predictions_to_eval_space(
    pred_model_space: np.ndarray,
    target_spec: TargetSpec,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    pred = _as_feature_matrix(pred_model_space).astype(np.float32, copy=False)
    if target_spec.apply_runtime_denormalize:
        pred = denormalize_targets(pred, y_mean, y_std, True)
    if target_spec.offset is not None and target_spec.scale is not None:
        pred = pred * target_spec.scale + target_spec.offset
    return pred.astype(np.float32, copy=False)


def validate_experiment(
    experiment_path: str | Path,
    csv_path: str | Path | None = None,
    timestamp: int | None = None,
    zoom: float | None = None,
    clip_max_percentile: float | None = None,
    output_root: str | Path | None = None,
) -> Path:
    setup = _build_experiment_setup(experiment_path)
    _ensure_runtime_dependencies()

    if zoom is not None and float(zoom) <= 0.0:
        raise ValueError(f"--zoom must be positive, got {zoom}")
    if clip_max_percentile is not None and not (0.0 < float(clip_max_percentile) <= 100.0):
        raise ValueError(
            f"--clip-max-percentile must be in (0, 100], got {clip_max_percentile}"
        )

    logger.info("Thread limits active: intra_op=%d inter_op=%d", THREAD_LIMITS[0], THREAD_LIMITS[1])
    logger.info("Experiment: %s", setup.exp_id)
    logger.info("Dataset: %s", setup.dataset_name)
    logger.info("Config: %s", setup.cfg_path)
    logger.info("Checkpoint: %s", setup.ckpt_path)
    logger.info("Attr: %s", setup.attr_name)
    logger.info("Association: %s", setup.association)
    logger.info("Model source path: %s", setup.source.model_source_path)
    logger.info("Eval raw source path: %s", setup.source.raw_source_path)
    logger.info("Model target path: %s", setup.target.model_target_path)
    logger.info("Eval raw target path: %s", setup.target.raw_gt_path)

    raw_source = np.load(str(setup.source.raw_source_path), mmap_mode="r", allow_pickle=False)
    model_source = np.load(str(setup.source.model_source_path), mmap_mode="r", allow_pickle=False)
    model_target_array = _ensure_2d(np.load(str(setup.target.model_target_path), mmap_mode="r", allow_pickle=False))
    gt_array = _ensure_2d(np.load(str(setup.target.raw_gt_path), mmap_mode="r", allow_pickle=False))
    if raw_source.ndim != 2 or raw_source.shape[1] < 1:
        raise ValueError(f"Expected raw coords array with shape (N, D>=1), got {raw_source.shape}")
    if model_source.ndim != 2:
        raise ValueError(f"Expected model coords array with shape (N, D), got {model_source.shape}")
    if int(raw_source.shape[0]) != int(model_source.shape[0]):
        raise ValueError(f"Source row mismatch: raw={raw_source.shape[0]} model={model_source.shape[0]}")
    if int(raw_source.shape[0]) != int(gt_array.shape[0]):
        raise ValueError(f"Source / target row mismatch: source={raw_source.shape[0]} target={gt_array.shape[0]}")
    if int(raw_source.shape[0]) != int(model_target_array.shape[0]):
        raise ValueError(f"Source / model-target row mismatch: source={raw_source.shape[0]} target={model_target_array.shape[0]}")

    x_mean, x_std, y_mean, y_std = _resolve_stats_arrays(
        setup.payload,
        setup.cfg,
        setup.source.raw_source_path,
        setup.target.raw_gt_path,
    )

    time_values = np.asarray(raw_source[:, -1])
    time_indexers = compute_time_indexers(time_values)
    num_timesteps = len(time_indexers)
    if num_timesteps == 0:
        raise ValueError("No timesteps found in source coordinates.")

    if timestamp is not None:
        if timestamp < 0 or timestamp >= num_timesteps:
            raise ValueError(f"--timestamp {timestamp} is out of range. Valid range: [0, {num_timesteps - 1}]")
        selected_time_indices = [int(timestamp)]
    else:
        selected_time_indices = _select_default_timestamps(num_timesteps)

    candidate_meshes = _collect_mesh_candidates(setup.source.raw_source_path)
    selected_steps: list[dict[str, object]] = []
    for time_index in selected_time_indices:
        raw_time, indexer = time_indexers[time_index]
        sample_count = _indexer_size(indexer)
        mesh_path, mesh_association = _resolve_mesh_for_timestep(
            raw_time=raw_time,
            sample_count=sample_count,
            candidate_paths=candidate_meshes,
            preferred_association=setup.association,
        )
        selected_steps.append(
            {
                "time_index": int(time_index),
                "raw_time": float(raw_time),
                "indexer": indexer,
                "sample_count": int(sample_count),
                "mesh_path": mesh_path,
                "mesh_association": mesh_association,
            }
        )

    model, _ = build_model(setup.cfg, setup.cfg["LOSS"])
    model.load_state_dict(unwrap_model_state(setup.payload), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    lpips_model = _build_lpips_model(device)

    batch_size = int(setup.cfg.get("TRAINING", {}).get("pred_batch_size", 16000))
    out_root = Path(output_root).resolve() if output_root else Path("validate_out").resolve()
    resolved_csv_path = Path(csv_path).resolve() if csv_path else _default_csv_path(out_root, setup.dataset_name, setup.exp_id).resolve()
    zoom_factor = float(zoom) if zoom is not None else _render_zoom_factor(setup.dataset_name)
    render_attr_name = _output_attr_name(setup.attr_name)

    logger.info("CSV output: %s", resolved_csv_path)
    logger.info("Render zoom factor: %.2f", zoom_factor)
    logger.info("Render attr token: %s", render_attr_name)
    logger.info(
        "Render clip max percentile: %s",
        "none" if clip_max_percentile is None else f"{float(clip_max_percentile):.3f}",
    )
    logger.info(
        "Rendering timesteps: %s",
        [f"{step['time_index']}({step['raw_time']})" for step in selected_steps],
    )
    psnr_in_model_space = bool(
        _is_normalized_artifact_path(Path(str(setup.cfg["DATA"]["target_path"])))
        or setup.target.model_target_path.resolve() != setup.target.raw_gt_path.resolve()
    )
    logger.info("PSNR space: %s", "model" if psnr_in_model_space else "eval")

    attr_clim: tuple[float, float] | None = None
    gt_visual_blocks: list[np.ndarray] | None = [] if clip_max_percentile is not None else None
    psnr_map: dict[int, float] = {}
    logger.info("Pass 1/2: collecting color ranges and PSNR")
    for step in selected_steps:
        coords_block = _select_array_block(model_source, step["indexer"]).astype(np.float32)
        if setup.source.apply_runtime_normalize:
            coords_block = normalize_coords(coords_block, x_mean, x_std, True)
        gt_values = _select_target_eval_block(gt_array, step["indexer"], setup.target)
        gt_model_values = _as_feature_matrix(_select_array_block(model_target_array, step["indexer"]).astype(np.float32))
        pred_model, _ = predict_block(model, coords_block, batch_size, device)
        pred_eval = _map_predictions_to_eval_space(pred_model, setup.target, y_mean, y_std)
        gt_vis = _to_visual_scalar(gt_values)
        attr_clim = _merge_range(attr_clim, gt_vis)
        if gt_visual_blocks is not None:
            gt_visual_blocks.append(np.asarray(gt_vis, dtype=np.float32).reshape(-1).copy())
        if psnr_in_model_space:
            psnr_value, _ = compute_psnr(pred_model, gt_model_values)
        else:
            psnr_value, _ = compute_psnr(pred_eval, gt_values)
        psnr_map[int(step["time_index"])] = psnr_value

    base_clim = _normalize_clim(attr_clim)
    normalized_clim = _clip_upper_clim(base_clim, gt_visual_blocks or [], clip_max_percentile)
    if normalized_clim != base_clim:
        logger.info(
            "Attr %s color clip: base=[%.6f, %.6f] clipped=[%.6f, %.6f] percentile=%.3f",
            render_attr_name,
            base_clim[0],
            base_clim[1],
            normalized_clim[0],
            normalized_clim[1],
            float(clip_max_percentile),
        )
    per_timestep_rows: list[dict[str, object]] = []

    logger.info("Pass 2/2: rendering frames to %s", out_root)
    for step in selected_steps:
        coords_block = _select_array_block(model_source, step["indexer"]).astype(np.float32)
        if setup.source.apply_runtime_normalize:
            coords_block = normalize_coords(coords_block, x_mean, x_std, True)
        gt_values = _select_target_eval_block(gt_array, step["indexer"], setup.target)
        pred_model, _ = predict_block(model, coords_block, batch_size, device)
        pred_eval = _map_predictions_to_eval_space(pred_model, setup.target, y_mean, y_std)

        pred_vis = _to_visual_scalar(pred_eval)
        gt_vis = _to_visual_scalar(gt_values)
        pred_out = _prepare_pred_output_path(
            out_root=out_root,
            dataset_name=setup.dataset_name,
            exp_id=setup.exp_id,
            attr_name=render_attr_name,
            time_index=int(step["time_index"]),
        )
        gt_out = _prepare_gt_output_path(
            out_root=out_root,
            dataset_name=setup.dataset_name,
            attr_name=render_attr_name,
            time_index=int(step["time_index"]),
        )
        pred_img = _render_frame(
            mesh_path=step["mesh_path"],
            association=str(step["mesh_association"]),
            values=pred_vis,
            outpath=pred_out,
            clim=normalized_clim,
            zoom_factor=zoom_factor,
        )
        if _can_reuse_gt_cache(gt_out, normalized_clim):
            gt_img = _load_image(gt_out)
            logger.info("Reused cached GT render for timestep=%s -> %s", step["time_index"], gt_out)
        else:
            gt_img = _render_frame(
                mesh_path=step["mesh_path"],
                association=str(step["mesh_association"]),
                values=gt_vis,
                outpath=gt_out,
                clim=normalized_clim,
                zoom_factor=zoom_factor,
            )
            _write_gt_cache_clim(gt_out, normalized_clim)

        ssim_value = _compute_ssim(gt_img, pred_img)
        lpips_value = _compute_lpips(lpips_model, gt_img, pred_img, device)
        per_timestep_rows.append(
            {
                "row_type": "per_timestep",
                "exp_id": setup.exp_id,
                "model_name": str(setup.cfg["MODEL"]["model_name"]),
                "dataset_name": setup.dataset_name,
                "checkpoint_path": str(setup.ckpt_path),
                "attr": render_attr_name,
                "time_index": int(step["time_index"]),
                "raw_time": float(step["raw_time"]),
                "num_samples": int(step["sample_count"]),
                "num_timesteps": int(num_timesteps),
                "gt_render_path": str(gt_out.resolve()),
                "pred_render_path": str(pred_out.resolve()),
                "psnr": float(psnr_map[int(step["time_index"])]),
                "ssim": float(ssim_value),
                "lpips": float(lpips_value),
            }
        )
        logger.info(
            "Rendered attr=%s timestep=%s -> pred=%s gt=%s psnr=%.6f ssim=%.6f lpips=%.6f",
            setup.attr_name,
            step["time_index"],
            pred_out,
            gt_out,
            float(psnr_map[int(step["time_index"])]),
            float(ssim_value),
            float(lpips_value),
        )

    summary_rows = [
        {
            "row_type": "attr_mean",
            "exp_id": setup.exp_id,
            "model_name": str(setup.cfg["MODEL"]["model_name"]),
            "dataset_name": setup.dataset_name,
            "checkpoint_path": str(setup.ckpt_path),
            "attr": render_attr_name,
            "time_index": "",
            "raw_time": "",
            "num_samples": "",
            "num_timesteps": int(num_timesteps),
            "gt_render_path": "",
            "pred_render_path": "",
            "psnr": _mean_finite([float(row["psnr"]) for row in per_timestep_rows]),
            "ssim": _mean_finite([float(row["ssim"]) for row in per_timestep_rows]),
            "lpips": _mean_finite([float(row["lpips"]) for row in per_timestep_rows]),
        },
        {
            "row_type": "global_mean",
            "exp_id": setup.exp_id,
            "model_name": str(setup.cfg["MODEL"]["model_name"]),
            "dataset_name": setup.dataset_name,
            "checkpoint_path": str(setup.ckpt_path),
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
        },
    ]

    rows = per_timestep_rows + summary_rows
    _write_csv_rows(resolved_csv_path, rows)
    logger.info("Saved validation metrics to CSV: %s", resolved_csv_path)
    return resolved_csv_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _ensure_required_conda_env("compression")
    args = _parse_args()
    validate_experiment(
        args.experiment_path,
        csv_path=(args.csv or None),
        timestamp=args.timestamp,
        zoom=args.zoom,
        clip_max_percentile=args.clip_max_percentile,
        output_root=(args.output_root or None),
    )


if __name__ == "__main__":
    main()

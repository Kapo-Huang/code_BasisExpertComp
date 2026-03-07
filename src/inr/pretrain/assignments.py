from __future__ import annotations

from dataclasses import dataclass
import logging
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Subset

from inr.data import MultiViewCoordDataset, NodeDataset

logger = logging.getLogger(__name__)


@dataclass
class PretrainAssignmentConfig:
    method: str = "voxel_clustering"
    seed: int = 42
    cache_path: str = ""
    cluster_num_time_samples: int = 16
    spatial_blocks: Optional[Tuple[int, int, int]] = None
    time_block_size: int = 0


def _load_cached(cache_path: str) -> Optional[np.ndarray]:
    if not cache_path:
        return None
    cache = Path(cache_path)
    if not cache.exists():
        return None
    return np.asarray(np.load(cache), dtype=np.int64)


def _save_cached(cache_path: str, assignments: np.ndarray):
    if not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, assignments)


def _validate_assignments(assignments: np.ndarray, n_samples: int, num_experts: int) -> Tuple[bool, str]:
    arr = np.asarray(assignments)
    if arr.ndim != 1:
        return False, f"assignments ndim must be 1, got {arr.ndim}"
    if arr.shape != (int(n_samples),):
        return False, f"assignments shape must be ({n_samples},), got {arr.shape}"
    if arr.size <= 0:
        return False, "assignments must be non-empty"
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < 0 or hi >= int(num_experts):
        return False, f"assignment ids out of range: min={lo}, max={hi}, num_experts={int(num_experts)}"
    return True, ""


def _balanced_random_assignments(n_samples: int, num_experts: int, seed: int) -> np.ndarray:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    rng = np.random.default_rng(seed)
    reps = int(ceil(n_samples / float(num_experts)))
    base = np.tile(np.arange(num_experts, dtype=np.int64), reps)[:n_samples]
    rng.shuffle(base)
    return base


def _resolve_base_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, np.asarray(dataset.indices, dtype=np.int64)
    return dataset, None


def _tensor_to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return np.asarray(tensor.detach().cpu(), dtype=np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _slice_array(array: np.ndarray, indices: Optional[np.ndarray]) -> np.ndarray:
    if indices is None:
        return array
    return array[indices]


def _dataset_coords(dataset, indices: Optional[np.ndarray]) -> np.ndarray:
    coords = _tensor_to_numpy(dataset.x)
    coords = _slice_array(coords, indices)
    x_mean = getattr(dataset, "x_mean", None)
    x_std = getattr(dataset, "x_std", None)
    if x_mean is not None and x_std is not None and getattr(dataset, "normalize", False):
        coords = coords * _tensor_to_numpy(x_std) + _tensor_to_numpy(x_mean)
    return np.asarray(coords, dtype=np.float32)


def _dataset_cluster_features(dataset, indices: Optional[np.ndarray]) -> np.ndarray:
    if isinstance(dataset, NodeDataset):
        targets = _slice_array(_tensor_to_numpy(dataset.y), indices)
        return np.asarray(targets, dtype=np.float32)
    if isinstance(dataset, MultiViewCoordDataset):
        blocks = []
        for name in dataset.y.keys():
            blocks.append(_slice_array(_tensor_to_numpy(dataset.y[name]), indices))
        return np.concatenate(blocks, axis=1).astype(np.float32)
    raise TypeError(f"Unsupported dataset type for pretrain assignments: {type(dataset)}")


def _kmeans_assignments(features: np.ndarray, num_experts: int, seed: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    n_samples = int(features.shape[0])
    if n_samples < num_experts:
        raise ValueError(f"num_experts={num_experts} exceeds sample count N={n_samples}")

    chunk_size = min(50_000, n_samples)
    kmeans = MiniBatchKMeans(
        n_clusters=int(num_experts),
        random_state=int(seed),
        batch_size=max(256, chunk_size),
        n_init=3,
    )

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        kmeans.partial_fit(features[start:end])

    assignments = np.empty((n_samples,), dtype=np.int64)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        assignments[start:end] = kmeans.predict(features[start:end]).astype(np.int64)
    return assignments


def _choose_grid_dims(num_experts: int) -> Tuple[int, int, int]:
    nx = int(round(num_experts ** (1.0 / 3.0)))
    nx = max(1, nx)
    while num_experts % nx != 0 and nx > 1:
        nx -= 1
    rem = max(1, num_experts // nx)
    ny = int(round(rem ** 0.5))
    ny = max(1, ny)
    while rem % ny != 0 and ny > 1:
        ny -= 1
    nz = max(1, rem // ny)
    return int(nx), int(ny), int(nz)


def _block_indices(values: np.ndarray, num_blocks: int) -> np.ndarray:
    if num_blocks <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.int64)
    edges = np.linspace(vmin, vmax, num_blocks + 1)
    block_ids = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray(block_ids, dtype=np.int64)


def _spatial_block_assignments(coords: np.ndarray, num_experts: int, spatial_blocks: Optional[Tuple[int, int, int]]) -> np.ndarray:
    if coords.shape[1] < 3:
        raise ValueError("spatial_block assignments require at least 3 coordinate dimensions")
    if spatial_blocks is None:
        nx, ny, nz = _choose_grid_dims(num_experts)
    else:
        if len(spatial_blocks) != 3:
            raise ValueError("spatial_blocks must be a 3-tuple/list: [nx, ny, nz]")
        nx, ny, nz = (int(spatial_blocks[0]), int(spatial_blocks[1]), int(spatial_blocks[2]))
    bx = _block_indices(coords[:, 0], max(1, nx))
    by = _block_indices(coords[:, 1], max(1, ny))
    bz = _block_indices(coords[:, 2], max(1, nz))
    block_id = (bz * max(1, ny) + by) * max(1, nx) + bx
    return np.asarray(block_id % num_experts, dtype=np.int64)


def _time_block_assignments(coords: np.ndarray, num_experts: int, time_block_size: int) -> np.ndarray:
    if coords.shape[1] < 4:
        raise ValueError("time_block assignments require at least 4 coordinate dimensions")
    time_values = coords[:, 3]
    order = np.argsort(time_values, kind="stable")
    if time_block_size <= 0:
        time_block_size = int(ceil(coords.shape[0] / float(num_experts)))
    assignments = np.empty((coords.shape[0],), dtype=np.int64)
    for rank, sample_index in enumerate(order):
        assignments[sample_index] = (rank // int(time_block_size)) % num_experts
    return assignments


def compute_pretrain_assignments(dataset, num_experts: int, cfg: PretrainAssignmentConfig) -> np.ndarray:
    base_dataset, indices = _resolve_base_dataset(dataset)
    method = str(cfg.method).strip().lower()
    num_experts = int(num_experts)
    n_samples = len(dataset)

    cached = _load_cached(cfg.cache_path)
    if cached is not None:
        ok, reason = _validate_assignments(cached, n_samples, num_experts)
        if ok:
            return cached
        logger.warning(
            "Ignoring invalid pretrain assignments cache '%s': %s. Recomputing assignments.",
            cfg.cache_path,
            reason,
        )

    if method in {"voxel_clustering", "clustering", "kmeans", "sample_clustering"}:
        features = _dataset_cluster_features(base_dataset, indices)
        assignments = _kmeans_assignments(features, num_experts, int(cfg.seed))
    elif method in {"random", "uniform", "random_uniform"}:
        assignments = _balanced_random_assignments(n_samples, num_experts, int(cfg.seed))
    elif method in {"spatial_block", "spatial_blocks", "space_block"}:
        coords = _dataset_coords(base_dataset, indices)
        assignments = _spatial_block_assignments(coords, num_experts, cfg.spatial_blocks)
    elif method in {"time_block", "time_blocks", "temporal_block"}:
        coords = _dataset_coords(base_dataset, indices)
        assignments = _time_block_assignments(coords, num_experts, int(cfg.time_block_size))
    else:
        raise ValueError(f"Unknown pretrain assignment method: {cfg.method}")

    ok, reason = _validate_assignments(assignments, n_samples, num_experts)
    if not ok:
        raise ValueError(f"Invalid pretrain assignments generated by method '{cfg.method}': {reason}")
    _save_cached(cfg.cache_path, assignments)
    return assignments

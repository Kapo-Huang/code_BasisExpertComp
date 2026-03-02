from __future__ import annotations

from dataclasses import dataclass
import logging
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from inr.datasets.base import VolumeShape
from inr.datasets.volumetric import MultiTargetVolumetricDataset, VolumetricDataset
from inr.pretrain.voxel_clustering import compute_voxel_cluster_assignments

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
    arr = np.load(cache)
    return np.asarray(arr, dtype=np.int64)


def _save_cached(cache_path: str, assignments: np.ndarray):
    if not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, assignments)


def _validate_assignments(assignments: np.ndarray, volume_shape: VolumeShape, num_experts: int) -> Tuple[bool, str]:
    arr = np.asarray(assignments)
    if arr.ndim != 1:
        return False, f"assignments ndim must be 1, got {arr.ndim}"
    X, Y, Z, T = int(volume_shape.X), int(volume_shape.Y), int(volume_shape.Z), int(volume_shape.T)
    V = int(X) * int(Y) * int(Z)
    if arr.shape not in {(V,), (T,)}:
        return False, f"assignments shape must be (V,) or (T,), got {arr.shape} (V={V}, T={T})"
    if arr.size <= 0:
        return False, "assignments must be non-empty"
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < 0 or hi >= int(num_experts):
        return False, f"assignment ids out of range: min={lo}, max={hi}, num_experts={int(num_experts)}"
    return True, ""


def _balanced_random_assignments(V: int, num_experts: int, seed: int) -> np.ndarray:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    rng = np.random.default_rng(seed)
    reps = int(ceil(V / float(num_experts)))
    base = np.tile(np.arange(num_experts, dtype=np.int64), reps)[:V]
    rng.shuffle(base)
    return base


def _choose_grid_dims(num_experts: int) -> Tuple[int, int, int]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
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


def _spatial_block_assignments(volume_shape: VolumeShape, num_experts: int,
                               spatial_blocks: Optional[Tuple[int, int, int]]) -> np.ndarray:
    X, Y, Z = int(volume_shape.X), int(volume_shape.Y), int(volume_shape.Z)
    V = X * Y * Z
    if spatial_blocks is None:
        nx, ny, nz = _choose_grid_dims(num_experts)
    else:
        if len(spatial_blocks) != 3:
            raise ValueError("spatial_blocks must be a 3-tuple/list: [nx, ny, nz]")
        nx, ny, nz = (int(spatial_blocks[0]), int(spatial_blocks[1]), int(spatial_blocks[2]))
    nx = max(1, nx)
    ny = max(1, ny)
    nz = max(1, nz)
    num_blocks = nx * ny * nz
    block_x = max(1, int(ceil(X / float(nx))))
    block_y = max(1, int(ceil(Y / float(ny))))
    block_z = max(1, int(ceil(Z / float(nz))))

    assignments = np.empty((V,), dtype=np.int64)
    v = 0
    for z in range(Z):
        bz = min(nz - 1, z // block_z)
        for y in range(Y):
            by = min(ny - 1, y // block_y)
            for x in range(X):
                bx = min(nx - 1, x // block_x)
                block_id = (bz * ny + by) * nx + bx
                assignments[v] = block_id % num_experts
                v += 1
    if num_blocks < num_experts:
        # spread unused experts evenly by modulo; already handled by % num_experts
        pass
    return assignments


def _time_block_assignments(volume_shape: VolumeShape, num_experts: int, time_block_size: int) -> np.ndarray:
    T = int(volume_shape.T)
    if T <= 0:
        raise ValueError("T must be positive")
    if time_block_size <= 0:
        time_block_size = int(ceil(T / float(num_experts)))
    assignments = np.empty((T,), dtype=np.int64)
    for t in range(T):
        block_id = t // int(time_block_size)
        assignments[t] = block_id % num_experts
    return assignments


def compute_pretrain_assignments(
    dataset: VolumetricDataset | MultiTargetVolumetricDataset,
    num_experts: int,
    cfg: PretrainAssignmentConfig,
) -> np.ndarray:
    method = str(cfg.method).strip().lower()
    num_experts = int(num_experts)
    cached = _load_cached(cfg.cache_path)
    if cached is not None:
        ok, reason = _validate_assignments(cached, dataset.volume_shape, num_experts)
        if ok:
            return cached
        logger.warning(
            "Ignoring invalid pretrain assignments cache '%s': %s. Recomputing assignments.",
            cfg.cache_path,
            reason,
        )

    if method in {"voxel_clustering", "clustering", "kmeans"}:
        assignments = compute_voxel_cluster_assignments(
            dataset,
            num_experts=num_experts,
            num_time_samples=int(cfg.cluster_num_time_samples),
            seed=int(cfg.seed),
            cache_path=None,
        )
    elif method in {"random", "uniform", "random_uniform"}:
        volume_shape = dataset.volume_shape
        V = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
        assignments = _balanced_random_assignments(V, num_experts, int(cfg.seed))
    elif method in {"spatial_block", "spatial_blocks", "space_block"}:
        assignments = _spatial_block_assignments(
            dataset.volume_shape,
            num_experts,
            cfg.spatial_blocks,
        )
    elif method in {"time_block", "time_blocks", "temporal_block"}:
        assignments = _time_block_assignments(
            dataset.volume_shape,
            num_experts,
            int(cfg.time_block_size),
        )
    else:
        raise ValueError(f"Unknown pretrain assignment method: {cfg.method}")

    ok, reason = _validate_assignments(assignments, dataset.volume_shape, num_experts)
    if not ok:
        raise ValueError(f"Invalid pretrain assignments generated by method '{cfg.method}': {reason}")
    _save_cached(cfg.cache_path, assignments)
    return assignments

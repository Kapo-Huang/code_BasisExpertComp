from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from inr.datasets.base import VolumeShape, idx_to_xyzt
from inr.datasets.volumetric import MultiTargetVolumetricDataset, VolumetricDataset


def _sample_time_indices(T: int, num_time_samples: int, seed: int) -> np.ndarray:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if num_time_samples <= 0:
        raise ValueError(f"num_time_samples must be positive, got {num_time_samples}")
    rng = np.random.default_rng(seed)
    replace = T < num_time_samples
    return rng.choice(T, size=num_time_samples, replace=replace).astype(np.int64)


def _build_features_single(
    y_flat: np.ndarray,
    voxel_ids: np.ndarray,
    time_ids: np.ndarray,
    V: int,
) -> np.ndarray:
    idx = voxel_ids[:, None] + V * time_ids[None, :]
    flat = np.asarray(y_flat[idx.reshape(-1)], dtype=np.float32)
    if flat.ndim == 1:
        flat = flat.reshape(-1, 1)
    flat = flat.reshape(voxel_ids.shape[0], time_ids.shape[0], -1)
    return flat.reshape(voxel_ids.shape[0], -1)


def _build_features_multi(
    targets_flat: dict,
    voxel_ids: np.ndarray,
    time_ids: np.ndarray,
    V: int,
) -> np.ndarray:
    idx = voxel_ids[:, None] + V * time_ids[None, :]
    blocks = []
    for name in targets_flat.keys():
        flat = np.asarray(targets_flat[name][idx.reshape(-1)], dtype=np.float32)
        if flat.ndim == 1:
            flat = flat.reshape(-1, 1)
        flat = flat.reshape(voxel_ids.shape[0], time_ids.shape[0], -1)
        blocks.append(flat)
    merged = np.concatenate(blocks, axis=2)
    return merged.reshape(voxel_ids.shape[0], -1)


def _assert_mapping(volume_shape: VolumeShape, V: int, seed: int):
    rng = np.random.default_rng(seed + 1)
    v = int(rng.integers(0, V))
    t = int(rng.integers(0, volume_shape.T))
    idx = v + V * t
    x, y, z, t2 = idx_to_xyzt(idx, volume_shape)
    v2 = x + volume_shape.X * (y + volume_shape.Y * z)
    assert t2 == t, "idx-to-time mismatch"
    assert v2 == v, "idx-to-voxel mismatch"


def compute_voxel_cluster_assignments(
    dataset: VolumetricDataset | MultiTargetVolumetricDataset,
    num_experts: int,
    num_time_samples: int,
    seed: int,
    cache_path: Optional[str],
) -> np.ndarray:
    """
    Returns assignments[v] -> expert id for each voxel v in [0, V-1].
    """
    if cache_path:
        cache = Path(cache_path)
        if cache.exists():
            arr = np.load(cache)
            arr = np.asarray(arr, dtype=np.int64)
            volume_shape = dataset.volume_shape
            V = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
            assert arr.shape == (V,)
            assert arr.min() >= 0
            assert arr.max() < num_experts
            return arr

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    volume_shape = dataset.volume_shape
    X, Y, Z, T = map(int, (volume_shape.X, volume_shape.Y, volume_shape.Z, volume_shape.T))
    V = int(X) * int(Y) * int(Z)
    if V < num_experts:
        raise ValueError(f"num_experts={num_experts} exceeds voxel count V={V}")
    assert V == X * Y * Z
    _assert_mapping(volume_shape, V, seed)

    time_ids = _sample_time_indices(T, num_time_samples, seed)

    if isinstance(dataset, VolumetricDataset):
        y_flat = dataset._ensure_y_flat()
        build_features = lambda voxels: _build_features_single(y_flat, voxels, time_ids, V)
    elif isinstance(dataset, MultiTargetVolumetricDataset):
        targets_flat = dataset._ensure_targets_flat()
        build_features = lambda voxels: _build_features_multi(targets_flat, voxels, time_ids, V)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    chunk_voxels = 50_000
    kmeans = MiniBatchKMeans(
        n_clusters=int(num_experts),
        random_state=int(seed),
        batch_size=min(chunk_voxels, V),
    )

    for start in range(0, V, chunk_voxels):
        end = min(start + chunk_voxels, V)
        voxels = np.arange(start, end, dtype=np.int64)
        feats = build_features(voxels)
        kmeans.partial_fit(feats)

    assignments = np.empty((V,), dtype=np.int64)
    for start in range(0, V, chunk_voxels):
        end = min(start + chunk_voxels, V)
        voxels = np.arange(start, end, dtype=np.int64)
        feats = build_features(voxels)
        assignments[start:end] = kmeans.predict(feats).astype(np.int64)

    assert assignments.shape == (V,)
    assert assignments.min() >= 0
    assert assignments.max() < num_experts

    if cache_path:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, assignments)

    return assignments

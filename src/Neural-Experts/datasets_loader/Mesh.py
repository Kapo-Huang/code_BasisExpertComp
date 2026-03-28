from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime_limits import apply_runtime_thread_limits, configure_threading_env

configure_threading_env()

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset

apply_runtime_thread_limits()


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape={arr.shape}")


def _compute_stats_streaming(arr: np.ndarray, chunk_size: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    data = _ensure_2d(arr)
    n_samples = int(data.shape[0])
    dims = int(data.shape[1])
    total_sum = np.zeros((dims,), dtype=np.float64)
    total_sq = np.zeros((dims,), dtype=np.float64)

    for start in range(0, n_samples, int(chunk_size)):
        end = min(start + int(chunk_size), n_samples)
        block = np.asarray(data[start:end], dtype=np.float64)
        total_sum += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)

    if n_samples <= 1:
        mean = total_sum / max(n_samples, 1)
        var = np.zeros_like(mean)
    else:
        mean = total_sum / n_samples
        var = (total_sq - (total_sum * total_sum) / n_samples) / (n_samples - 1)
        var = np.maximum(var, 0.0)

    std = np.sqrt(np.maximum(var, 1.0e-12))
    return mean.astype(np.float32)[None, :], std.astype(np.float32)[None, :]


def _load_npz_stats(stats_path: Path | None) -> dict[str, np.ndarray]:
    if stats_path is None or not stats_path.exists():
        return {}
    data = np.load(str(stats_path), allow_pickle=True)
    return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}


def _validate_stats_dims(name: str, value: np.ndarray, expected_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(1, -1)
    if int(array.shape[1]) != int(expected_dim):
        raise ValueError(f"Stats dim mismatch for {name}: expected {expected_dim}, got {array.shape[1]}")
    return array


def _identity_stats(dim: int) -> tuple[np.ndarray, np.ndarray]:
    dim = int(dim)
    return np.zeros((1, dim), dtype=np.float32), np.ones((1, dim), dtype=np.float32)


def _load_or_compute_stats(
    source: np.ndarray,
    target: np.ndarray,
    stats_path: Path | None,
    stats_key: str,
    input_dim: int,
    target_dim: int,
    load_input_stats: bool = True,
    load_target_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = _load_npz_stats(stats_path)
    x_mean = x_std = y_mean = y_std = None

    if load_input_stats and "x_mean" in stats and "x_std" in stats:
        x_mean = _validate_stats_dims("x_mean", stats["x_mean"], input_dim)
        x_std = _validate_stats_dims("x_std", stats["x_std"], input_dim)

    if load_target_stats and "y_mean" in stats and "y_std" in stats:
        y_mean = _validate_stats_dims("y_mean", stats["y_mean"], target_dim)
        y_std = _validate_stats_dims("y_std", stats["y_std"], target_dim)

    stats_y_mean_key = f"y_mean_{stats_key}"
    stats_y_std_key = f"y_std_{stats_key}"
    if load_target_stats and y_mean is None and stats_y_mean_key in stats and stats_y_std_key in stats:
        y_mean = _validate_stats_dims(stats_y_mean_key, stats[stats_y_mean_key], target_dim)
        y_std = _validate_stats_dims(stats_y_std_key, stats[stats_y_std_key], target_dim)

    if load_target_stats and y_mean is None and "mean" in stats and "std" in stats:
        y_mean = _validate_stats_dims("mean", stats["mean"], target_dim)
        y_std = _validate_stats_dims("std", stats["std"], target_dim)

    if not load_input_stats:
        x_mean, x_std = _identity_stats(input_dim)
    elif x_mean is None or x_std is None:
        x_mean, x_std = _compute_stats_streaming(source)
    if not load_target_stats:
        y_mean, y_std = _identity_stats(target_dim)
    elif y_mean is None or y_std is None:
        y_mean, y_std = _compute_stats_streaming(target)

    x_std = np.maximum(x_std, 1.0e-12).astype(np.float32)
    y_std = np.maximum(y_std, 1.0e-12).astype(np.float32)

    if stats_path is not None and not stats_path.exists() and (load_input_stats or load_target_stats):
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(stats_path),
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )

    return (
        torch.from_numpy(x_mean),
        torch.from_numpy(x_std),
        torch.from_numpy(y_mean),
        torch.from_numpy(y_std),
    )


@dataclass
class _AssignmentConfig:
    method: str
    fit_samples: int
    cache_path: str
    normalize_features: bool
    random_seed: int
    chunk_size: int


class CoordinateKMeansAssignments:
    def __init__(
        self,
        source: np.ndarray,
        n_experts: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        cfg: _AssignmentConfig,
    ) -> None:
        method = str(cfg.method).strip().lower()
        if method != "coord_kmeans":
            raise ValueError(f"Unsupported pretrain assignment method: {cfg.method}")

        self.n_experts = int(n_experts)
        self.normalize_features = bool(cfg.normalize_features)
        self.random_seed = int(cfg.random_seed)
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)
        self.chunk_size = max(1024, int(cfg.chunk_size))
        self.cache_path = Path(cfg.cache_path) if cfg.cache_path else None
        self.centers = self._load_or_fit_centers(source, max(1, int(cfg.fit_samples)))

    def _feature_block(self, coords: np.ndarray) -> np.ndarray:
        features = np.asarray(coords, dtype=np.float32)
        if self.normalize_features:
            features = (features - self.x_mean) / self.x_std
        return features.astype(np.float32, copy=False)

    def _load_or_fit_centers(self, source: np.ndarray, fit_samples: int) -> np.ndarray:
        if self.cache_path is not None and self.cache_path.exists():
            try:
                data = np.load(str(self.cache_path), allow_pickle=True)
                centers = np.asarray(data["centers"], dtype=np.float32)
                if centers.shape != (self.n_experts, source.shape[1]):
                    raise ValueError(
                        f"Cached centers shape mismatch: expected {(self.n_experts, source.shape[1])}, got {centers.shape}"
                    )
                cached_normalize = bool(np.asarray(data.get("normalize_features", self.normalize_features)).reshape(()))
                if cached_normalize != self.normalize_features:
                    raise ValueError("Cached assignment centers were built with different normalize_features setting")
                return centers
            except Exception:
                pass

        n_total = int(source.shape[0])
        fit_n = min(int(fit_samples), n_total)
        if fit_n < self.n_experts:
            raise ValueError(f"pretrain fit_samples={fit_n} must be >= n_experts={self.n_experts}")
        rng = np.random.default_rng(self.random_seed)
        if fit_n >= n_total:
            sample_idx = np.arange(n_total, dtype=np.int64)
        else:
            sample_idx = np.sort(rng.choice(n_total, size=fit_n, replace=False).astype(np.int64))

        features = self._feature_block(np.asarray(source[sample_idx], dtype=np.float32))
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_experts,
            random_state=self.random_seed,
            batch_size=min(max(256, self.n_experts * 64), fit_n),
            n_init=3,
        )

        for start in range(0, fit_n, self.chunk_size):
            end = min(start + self.chunk_size, fit_n)
            kmeans.partial_fit(features[start:end])

        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(self.cache_path),
                centers=centers,
                normalize_features=np.asarray(self.normalize_features, dtype=np.bool_),
            )
        return centers

    def assign(self, coords: np.ndarray) -> np.ndarray:
        features = self._feature_block(coords)
        distances = np.square(features[:, None, :] - self.centers[None, :, :]).sum(axis=-1)
        return np.argmin(distances, axis=1).astype(np.int64)


class MeshAttributeDataset(Dataset):
    def __init__(self, cfg: dict[str, Any], file_id: str | None = None):
        super().__init__()
        data_cfg = cfg["DATA"]
        train_cfg = cfg["TRAINING"]
        model_cfg = cfg["MODEL"]

        self.seed = int(cfg.get("seed", 0))
        self.dataset_name = str(data_cfg.get("dataset_name", "")).strip().lower()
        if self.dataset_name not in {"ocean", "stress"}:
            raise ValueError(f"Unsupported mesh dataset: {self.dataset_name}")

        self.attr_name = str(data_cfg.get("attr_name") or file_id or "").strip()
        if not self.attr_name:
            raise ValueError("DATA.attr_name must be provided")
        self.association = str(data_cfg.get("association", "")).strip().lower()
        if self.association not in {"point", "cell"}:
            raise ValueError("DATA.association must be 'point' or 'cell'")

        self.source_path = Path(data_cfg["source_path"])
        self.target_path = Path(data_cfg["target_path"])
        self.target_stats_path = Path(data_cfg["target_stats_path"]) if data_cfg.get("target_stats_path") else None
        self.stats_key = str(data_cfg.get("stats_key", "")).strip()
        if not self.stats_key:
            raise ValueError("DATA.stats_key must be provided")

        self.n_points = int(train_cfg["n_points"])
        self.segmentation_mode = bool(train_cfg.get("segmentation_mode", False))
        self.normalize_inputs = bool(data_cfg.get("normalize_inputs", False))
        self.normalize_targets = bool(data_cfg.get("normalize_targets", False))
        pretrain_cfg = train_cfg.get("pretrain_assignment", {}) or {}
        self.assignment_normalize_features = bool(pretrain_cfg.get("normalize_features", self.normalize_inputs))

        self.source = np.load(str(self.source_path), mmap_mode="r")
        self.target = np.load(str(self.target_path), mmap_mode="r")

        if self.source.ndim != 2:
            raise ValueError(f"Mesh source must be 2D, got shape={self.source.shape}")
        self.input_dim = int(self.source.shape[1])
        if self.input_dim != 4:
            raise ValueError(f"Mesh source input_dim must be 4, got {self.input_dim}")
        expected_in_dim = int(model_cfg.get("in_dim", self.input_dim))
        if expected_in_dim != self.input_dim:
            raise ValueError(f"MODEL.in_dim mismatch: expected {expected_in_dim}, actual {self.input_dim}")

        target_2d = _ensure_2d(self.target)
        if int(self.source.shape[0]) != int(target_2d.shape[0]):
            raise ValueError(
                f"Sample count mismatch: source={int(self.source.shape[0])}, target={int(target_2d.shape[0])}"
            )
        self.num_samples = int(target_2d.shape[0])
        self.target_dim = int(target_2d.shape[1])
        expected_out_dim = int(model_cfg.get("out_dim", self.target_dim))
        if expected_out_dim != self.target_dim:
            raise ValueError(f"MODEL.out_dim mismatch: expected {expected_out_dim}, actual {self.target_dim}")

        self.x_mean, self.x_std, self.y_mean, self.y_std = _load_or_compute_stats(
            source=self.source,
            target=target_2d,
            stats_path=self.target_stats_path,
            stats_key=self.stats_key,
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            load_input_stats=self.normalize_inputs or (self.segmentation_mode and self.assignment_normalize_features),
            load_target_stats=self.normalize_targets,
        )
        self.x_mean_np = self.x_mean.numpy()
        self.x_std_np = self.x_std.numpy()
        self.y_mean_np = self.y_mean.numpy()
        self.y_std_np = self.y_std.numpy()

        self.assignment = None
        if self.segmentation_mode:
            self.assignment = CoordinateKMeansAssignments(
                source=self.source,
                n_experts=int(model_cfg["n_experts"]),
                x_mean=self.x_mean_np,
                x_std=self.x_std_np,
                cfg=_AssignmentConfig(
                    method=str(pretrain_cfg.get("method", "coord_kmeans")),
                    fit_samples=int(pretrain_cfg.get("fit_samples", 50_000)),
                    cache_path=str(pretrain_cfg.get("cache_path", "")),
                    normalize_features=self.assignment_normalize_features,
                    random_seed=int(pretrain_cfg.get("random_seed", self.seed)),
                    chunk_size=int(pretrain_cfg.get("chunk_size", 65_536)),
                ),
            )

    def __len__(self) -> int:
        return np.iinfo(np.int32).max

    def _sample_indices(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(idx))
        return rng.integers(0, self.num_samples, size=self.n_points, dtype=np.int64)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = self._sample_indices(int(idx))
        coords_raw = np.asarray(self.source[sample_idx], dtype=np.float32)
        coords = coords_raw
        if self.normalize_inputs:
            coords = (coords - self.x_mean_np) / self.x_std_np
        coords_t = torch.from_numpy(coords.astype(np.float32, copy=False))

        if self.segmentation_mode:
            if self.assignment is None:
                raise RuntimeError("Segmentation mode requires assignment centers")
            segments = self.assignment.assign(coords_raw)
            return {
                "nonmnfld_points": coords_t,
                "nonmnfld_segments_gt": torch.from_numpy(segments),
            }

        values = _ensure_2d(np.asarray(self.target[sample_idx], dtype=np.float32))
        if self.normalize_targets:
            values = (values - self.y_mean_np) / self.y_std_np

        return {
            "nonmnfld_points": coords_t,
            "nonmnfld_val": torch.from_numpy(values.astype(np.float32, copy=False)),
        }

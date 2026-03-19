from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _volume_shape_from_cfg(volume_shape_cfg):
    return (
        int(volume_shape_cfg["X"]),
        int(volume_shape_cfg["Y"]),
        int(volume_shape_cfg["Z"]),
        int(volume_shape_cfg["T"]),
    )


def _compute_input_stats(volume_shape):
    x_size, y_size, z_size, t_size = volume_shape
    n_total = float(x_size * y_size * z_size * t_size)
    correction = (n_total / (n_total - 1.0)) if n_total > 1.0 else 1.0

    def _mean(size):
        return (float(size) - 1.0) / 2.0

    def _var(size):
        if int(size) <= 1:
            return 0.0
        return (float(size) * float(size) - 1.0) / 12.0

    means = np.array(
        [_mean(x_size), _mean(y_size), _mean(z_size), _mean(t_size)],
        dtype=np.float32,
    )
    stds = np.sqrt(
        np.maximum(
            np.array(
                [_var(x_size), _var(y_size), _var(z_size), _var(t_size)],
                dtype=np.float64,
            )
            * correction,
            1.0e-12,
        )
    ).astype(np.float32)
    return torch.from_numpy(means[None, :]), torch.from_numpy(stds[None, :])


def _compute_target_stats(arr, chunk_size=1_000_000):
    flat = arr.reshape(-1, arr.shape[-1] if arr.ndim == 2 else 1)
    count = 0
    sum_1 = np.zeros((flat.shape[-1],), dtype=np.float64)
    sum_2 = np.zeros((flat.shape[-1],), dtype=np.float64)

    for start in range(0, flat.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), flat.shape[0])
        block = np.asarray(flat[start:end], dtype=np.float64)
        count += block.shape[0]
        sum_1 += block.sum(axis=0)
        sum_2 += (block * block).sum(axis=0)

    if count <= 1:
        mean = sum_1 / max(count, 1)
        var = np.zeros_like(mean)
    else:
        mean = sum_1 / count
        var = (sum_2 - (sum_1 * sum_1) / count) / (count - 1)
        var = np.maximum(var, 0.0)

    std = np.sqrt(np.maximum(var, 1.0e-12))
    return mean.astype(np.float32)[None, :], std.astype(np.float32)[None, :]


class IonizationINRDataset(Dataset):
    def __init__(self, cfg, file_id=None):
        super().__init__()
        data_cfg = cfg["DATA"]
        train_cfg = cfg["TRAINING"]

        self.seed = int(cfg.get("seed", 0))
        self.attr_name = str(
            data_cfg.get("attr_name") or file_id or Path(data_cfg["target_path"]).stem.replace("target_", "")
        )
        self.target_path = Path(data_cfg["target_path"])
        self.target_stats_path = str(data_cfg.get("target_stats_path", "") or "")
        self.volume_shape = _volume_shape_from_cfg(data_cfg["volume_shape"])
        self.x_size, self.y_size, self.z_size, self.t_size = self.volume_shape
        self.spatial_size = self.x_size * self.y_size * self.z_size
        self.total_size = self.spatial_size * self.t_size
        self.n_points = int(train_cfg["n_points"])
        self.segmentation_mode = bool(train_cfg.get("segmentation_mode", False))
        self.normalize_inputs = bool(data_cfg.get("normalize_inputs", True))
        self.normalize_targets = bool(data_cfg.get("normalize_targets", False))
        self.n_segments = int(data_cfg["n_segments"])
        self.grid_patch_size = int(data_cfg.get("grid_patch_size", 1))
        self.segmentation_type = str(data_cfg.get("segmentation_type", "random_balanced")).strip().lower()
        if self.segmentation_type != "random_balanced":
            raise ValueError("IonizationINRDataset only supports segmentation_type='random_balanced'")

        self.target = np.load(str(self.target_path), mmap_mode="r")
        if self.target.ndim not in (1, 2):
            raise ValueError(f"Ionization target must be flat 1D/2D array, got shape={self.target.shape}")
        if int(self.target.shape[0]) != int(self.total_size):
            raise ValueError(
                f"Target length mismatch: target N={int(self.target.shape[0])}, expected N={int(self.total_size)}"
            )
        self.target_dim = int(self.target.shape[1]) if self.target.ndim == 2 else 1
        if self.target_dim != 1:
            raise ValueError(f"IonizationINRDataset expects single-channel targets, got target_dim={self.target_dim}")

        self.x_mean, self.x_std = _compute_input_stats(self.volume_shape)
        self.y_mean, self.y_std = self._load_target_stats()
        self.segments = self._random_balanced_segmentation()

    def _load_target_stats(self):
        stats_path = Path(self.target_stats_path) if self.target_stats_path else None
        if stats_path is not None and stats_path.exists():
            data = np.load(str(stats_path), allow_pickle=True)
            if "mean" in data and "std" in data:
                mean = np.asarray(data["mean"], dtype=np.float32).reshape(1, -1)
                std = np.asarray(data["std"], dtype=np.float32).reshape(1, -1)
                return torch.from_numpy(mean), torch.from_numpy(np.maximum(std, 1.0e-12))
            attr_mean_key = f"{self.attr_name}__mean"
            attr_std_key = f"{self.attr_name}__std"
            if attr_mean_key in data and attr_std_key in data:
                mean = np.asarray(data[attr_mean_key], dtype=np.float32).reshape(1, -1)
                std = np.asarray(data[attr_std_key], dtype=np.float32).reshape(1, -1)
                return torch.from_numpy(mean), torch.from_numpy(np.maximum(std, 1.0e-12))

        multi_stats_path = self.target_path.parent / "target_stats_multi.npz"
        if multi_stats_path.exists():
            data = np.load(str(multi_stats_path), allow_pickle=True)
            attr_mean_key = f"{self.attr_name}__mean"
            attr_std_key = f"{self.attr_name}__std"
            if attr_mean_key in data and attr_std_key in data:
                mean = np.asarray(data[attr_mean_key], dtype=np.float32).reshape(1, -1)
                std = np.asarray(data[attr_std_key], dtype=np.float32).reshape(1, -1)
                if stats_path is not None:
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(str(stats_path), mean=mean, std=std)
                return torch.from_numpy(mean), torch.from_numpy(np.maximum(std, 1.0e-12))

        mean, std = _compute_target_stats(self.target)
        if stats_path is not None:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(stats_path), mean=mean, std=std)
        return torch.from_numpy(mean), torch.from_numpy(np.maximum(std, 1.0e-12))

    def _random_balanced_segmentation(self):
        side = int(np.sqrt(self.grid_patch_size))
        if side * side != int(self.grid_patch_size):
            raise ValueError(f"grid_patch_size must be a perfect square, got {self.grid_patch_size}")
        side = max(side, 1)
        rng = np.random.default_rng(self.seed)
        reduced = rng.integers(
            0,
            self.n_segments,
            size=(
                int(np.ceil(self.z_size / float(side))),
                int(np.ceil(self.y_size / float(side))),
                int(np.ceil(self.x_size / float(side))),
            ),
            dtype=np.uint16,
        )
        segments = np.repeat(np.repeat(np.repeat(reduced, side, axis=2), side, axis=1), side, axis=0)
        segments = segments[: self.z_size, : self.y_size, : self.x_size]
        return segments.reshape(-1)

    def _sample_flat_indices(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        return rng.integers(0, self.total_size, size=self.n_points, dtype=np.int64)

    def _flat_to_coords(self, flat_idx):
        x = flat_idx % self.x_size
        rem = flat_idx // self.x_size
        y = rem % self.y_size
        rem = rem // self.y_size
        z = rem % self.z_size
        t = rem // self.z_size
        coords = np.stack([x, y, z, t], axis=-1).astype(np.float32)
        spatial_idx = (x + self.x_size * (y + self.y_size * z)).astype(np.int64)
        return coords, spatial_idx

    def __len__(self):
        return np.iinfo(np.int32).max

    def __getitem__(self, idx):
        flat_idx = self._sample_flat_indices(idx)
        coords_raw, spatial_idx = self._flat_to_coords(flat_idx)
        coords = coords_raw
        if self.normalize_inputs:
            coords = (coords - self.x_mean.numpy()) / self.x_std.numpy()
        coords_t = torch.from_numpy(coords.astype(np.float32))

        if self.segmentation_mode:
            segments = torch.from_numpy(np.asarray(self.segments[spatial_idx], dtype=np.int64))
            return {
                "nonmnfld_points": coords_t,
                "nonmnfld_segments_gt": segments,
            }

        values = np.asarray(self.target[flat_idx], dtype=np.float32).reshape(-1, 1)
        if self.normalize_targets:
            values = (values - self.y_mean.numpy()) / self.y_std.numpy()

        return {
            "nonmnfld_points": coords_t,
            "nonmnfld_val": torch.from_numpy(values.astype(np.float32)),
        }

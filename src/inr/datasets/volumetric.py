from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import (
    VolumeShape,
    compute_input_stats_analytic,
    compute_target_stats_streaming,
    idx_to_xyzt,
    infer_or_validate_volume_shape,
    peek_array,
    target_dim_from_array,
)


class VolumetricDataset(Dataset):
    """
    Lazy coordinate dataset for volumetric data (single target).
    """

    def __init__(
        self,
        y_path: str,
        volume_shape: Optional[VolumeShape] = None,
        normalize_inputs: bool = True,
        normalize_targets: bool = True,
        target_stats: Optional[Dict[str, np.ndarray]] = None,
        stats_dtype: np.dtype = np.float64,
        eps: float = 1e-12,
        coords_chunk_size: Optional[int] = None,
    ):
        super().__init__()

        self.y_path = y_path
        self.eps = float(eps)
        self.normalize_inputs = True
        self.normalize_targets = True
        self._stats_dtype = stats_dtype

        y_np = peek_array(y_path)
        self.volume_shape = infer_or_validate_volume_shape(y_np, volume_shape)
        self._target_dim = target_dim_from_array(y_np)
        self._n_samples = int(self.volume_shape.N)
        del y_np

        self.x_mean, self.x_std = compute_input_stats_analytic(
            self.volume_shape, unbiased=True, dtype=stats_dtype, eps=self.eps
        )
        self._x_mean_s = self.x_mean.squeeze(0)
        self._x_std_s = self.x_std.squeeze(0)

        self.y_mean = None
        self.y_std = None
        if target_stats is not None:
            mean = np.asarray(target_stats["mean"])
            std = np.asarray(target_stats["std"])
            self.y_mean = torch.from_numpy(mean).to(torch.float32)
            self.y_std = torch.from_numpy(std).to(torch.float32)
            if self.y_mean.ndim == 1:
                self.y_mean = self.y_mean.view(1, -1)
            if self.y_std.ndim == 1:
                self.y_std = self.y_std.view(1, -1)
            self.y_std = torch.where(self.y_std == 0, torch.ones_like(self.y_std), self.y_std)

        self.y_np = None
        self._y_flat = None
        self._coords_all = None

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        return int(idx)

    def denormalize_targets(self, y_norm: torch.Tensor) -> torch.Tensor:
        self._ensure_target_stats()
        return y_norm * self.y_std.to(y_norm.device) + self.y_mean.to(y_norm.device)

    def input_stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.x_mean.numpy(), "std": self.x_std.numpy()}

    def target_stats(self) -> Dict[str, np.ndarray]:
        self._ensure_target_stats()
        return {"mean": self.y_mean.numpy(), "std": self.y_std.numpy()}

    def _ensure_y_loaded(self) -> np.ndarray:
        if self.y_np is None:
            self.y_np = np.load(self.y_path, mmap_mode="r")
        return self.y_np

    def _ensure_y_flat(self) -> np.ndarray:
        if self._y_flat is None:
            y_np = self._ensure_y_loaded()
            if y_np.ndim == 5:
                self._y_flat = y_np.reshape(-1, y_np.shape[-1])
            elif y_np.ndim == 4:
                self._y_flat = y_np.reshape(-1, 1)
            elif y_np.ndim == 2:
                self._y_flat = y_np
            elif y_np.ndim == 1:
                self._y_flat = y_np.reshape(-1, 1)
            else:
                raise ValueError(f"Unsupported target ndim: {y_np.ndim} with shape {y_np.shape}")
        return self._y_flat

    def _read_target_flat(self, idx: int) -> torch.Tensor:
        y_flat = self._ensure_y_flat()
        arr = y_flat[idx]
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.ravel(arr)
        return torch.from_numpy(arr)

    def _ensure_target_stats(self):
        if self.y_mean is None or self.y_std is None:
            y_np = self._ensure_y_loaded()
            self.y_mean, self.y_std = compute_target_stats_streaming(
                y_np, eps=self.eps, dtype=self._stats_dtype
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["y_np"] = None
        state["_y_flat"] = None
        state["_coords_all"] = None
        return state


class MultiTargetVolumetricDataset(Dataset):
    """
    Lazy coordinate dataset for volumetric data with multiple targets.
    """

    def __init__(
        self,
        attr_paths: Dict[str, str],
        volume_shape: Optional[VolumeShape] = None,
        normalize_inputs: bool = True,
        normalize_targets: bool = True,
        target_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        stats_dtype: np.dtype = np.float64,
        eps: float = 1e-12,
        coords_chunk_size: Optional[int] = None,
    ):
        if not attr_paths:
            raise ValueError("attr_paths must be a non-empty dict")

        self.attr_paths = dict(attr_paths)
        self.eps = float(eps)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self._stats_dtype = stats_dtype

        first_path = next(iter(self.attr_paths.values()))
        first = peek_array(first_path)
        self.volume_shape = infer_or_validate_volume_shape(first, volume_shape)
        self._n_samples = int(self.volume_shape.N)
        del first

        self._view_specs = {}
        for name, path in self.attr_paths.items():
            arr = peek_array(path)
            infer_or_validate_volume_shape(arr, self.volume_shape)
            self._view_specs[name] = target_dim_from_array(arr)
            del arr

        self.x_mean, self.x_std = compute_input_stats_analytic(
            self.volume_shape, unbiased=True, dtype=stats_dtype, eps=self.eps
        )
        self._x_mean_s = self.x_mean.squeeze(0)
        self._x_std_s = self.x_std.squeeze(0)

        self.y_mean = {}
        self.y_std = {}
        self._y_mean_s = {}
        self._y_std_s = {}
        if target_stats is not None:
            for name, stats in target_stats.items():
                mean = np.asarray(stats["mean"])
                std = np.asarray(stats["std"])
                mean_t = torch.from_numpy(mean).to(torch.float32)
                std_t = torch.from_numpy(std).to(torch.float32)
                if mean_t.ndim == 1:
                    mean_t = mean_t.view(1, -1)
                if std_t.ndim == 1:
                    std_t = std_t.view(1, -1)
                std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
                self.y_mean[name] = mean_t
                self.y_std[name] = std_t
                self._y_mean_s[name] = mean_t.squeeze(0)
                self._y_std_s[name] = std_t.squeeze(0)

        self.y_np = None
        self._y_flat = None
        self._coords_all = None
        self._targets_flat = None

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        return int(idx)

    def view_specs(self) -> Dict[str, int]:
        return dict(self._view_specs)

    def denormalize_attr(self, name: str, y_norm: torch.Tensor) -> torch.Tensor:
        self._ensure_target_stats(name)
        mean = self.y_mean[name].to(y_norm.device)
        std = self.y_std[name].to(y_norm.device)
        return y_norm * std + mean

    def _ensure_targets_loaded(self) -> Dict[str, np.ndarray]:
        if self.y_np is None:
            self.y_np = {name: np.load(path, mmap_mode="r") for name, path in self.attr_paths.items()}
        return self.y_np

    def _ensure_targets_flat(self) -> Dict[str, np.ndarray]:
        if self._y_flat is None:
            self._y_flat = {}
            for name, arr in self._ensure_targets_loaded().items():
                if arr.ndim == 5:
                    self._y_flat[name] = arr.reshape(-1, arr.shape[-1])
                elif arr.ndim == 4:
                    self._y_flat[name] = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    self._y_flat[name] = arr
                elif arr.ndim == 1:
                    self._y_flat[name] = arr.reshape(-1, 1)
                else:
                    raise ValueError(f"Unsupported target ndim: {arr.ndim} with shape {arr.shape}")
        self._targets_flat = self._y_flat
        return self._y_flat

    def _ensure_target_stats(self, name: str):
        if self.y_mean.get(name) is None or self.y_std.get(name) is None:
            y_np = self._ensure_targets_loaded()[name]
            mean_t, std_t = compute_target_stats_streaming(
                y_np, eps=self.eps, dtype=self._stats_dtype
            )
            self.y_mean[name] = mean_t
            self.y_std[name] = std_t
            self._y_mean_s[name] = mean_t.squeeze(0)
            self._y_std_s[name] = std_t.squeeze(0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["y_np"] = None
        state["_y_flat"] = None
        state["_coords_all"] = None
        state["_targets_flat"] = None
        return state


def _singletarget_collate(indices: List[int], dataset: VolumetricDataset):
    idx = np.asarray(indices, dtype=np.int64)
    X = int(dataset.volume_shape.X)
    Y = int(dataset.volume_shape.Y)
    Z = int(dataset.volume_shape.Z)
    x = idx % X
    idx2 = idx // X
    y = idx2 % Y
    idx3 = idx2 // Y
    z = idx3 % Z
    t = idx3 // Z
    coords = np.stack([x, y, z, t], axis=1).astype(np.float32)
    xb = torch.from_numpy(coords)
    if dataset.normalize_inputs:
        xb = (xb - dataset._x_mean_s) / dataset._x_std_s

    y_flat = dataset._y_flat if dataset._y_flat is not None else dataset._ensure_y_flat()
    block = np.asarray(y_flat[idx], dtype=np.float32)
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    yb = torch.from_numpy(block.copy())
    if dataset.normalize_targets:
        dataset._ensure_target_stats()
        yb = (yb - dataset.y_mean.squeeze(0)) / dataset.y_std.squeeze(0)

    return xb.to(torch.float32), yb.to(torch.float32)


def _multitarget_collate(indices: List[int], dataset: MultiTargetVolumetricDataset):
    idx = np.asarray(indices, dtype=np.int64)
    X = int(dataset.volume_shape.X)
    Y = int(dataset.volume_shape.Y)
    Z = int(dataset.volume_shape.Z)
    x = idx % X
    idx2 = idx // X
    y = idx2 % Y
    idx3 = idx2 // Y
    z = idx3 % Z
    t = idx3 // Z
    coords = np.stack([x, y, z, t], axis=1).astype(np.float32)
    xb = torch.from_numpy(coords)
    if dataset.normalize_inputs:
        xb = (xb - dataset._x_mean_s) / dataset._x_std_s

    targets_flat = dataset._targets_flat if dataset._targets_flat is not None else dataset._ensure_targets_flat()
    yb = {}
    for name, flat in targets_flat.items():
        if dataset.normalize_targets:
            dataset._ensure_target_stats(name)
        block = np.asarray(flat[idx], dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        target = torch.from_numpy(block.copy())
        if dataset.normalize_targets:
            target = (target - dataset._y_mean_s[name]) / dataset._y_std_s[name]
        yb[name] = target

    xb = xb.to(torch.float32)
    yb = {name: tensor.to(torch.float32) for name, tensor in yb.items()}
    return xb, yb


def make_singletarget_collate(dataset: VolumetricDataset) -> Callable[[List[int]], tuple]:
    dataset._ensure_y_flat()
    return partial(_singletarget_collate, dataset=dataset)


def make_multitarget_collate(dataset: MultiTargetVolumetricDataset) -> Callable[[List[int]], tuple]:
    dataset._ensure_targets_flat()
    return partial(_multitarget_collate, dataset=dataset)

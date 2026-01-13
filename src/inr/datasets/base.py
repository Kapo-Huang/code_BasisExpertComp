from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class VolumeShape:
    X: int
    Y: int
    Z: int
    T: int

    @property
    def N(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z) * int(self.T)


def peek_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def to_torch_copy(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))


def parse_volume_shape(
    volume_shape: Optional[Union[VolumeShape, Tuple[int, int, int, int], Dict[str, int]]]
) -> Optional[VolumeShape]:
    if volume_shape is None:
        return None
    if isinstance(volume_shape, VolumeShape):
        return volume_shape
    if isinstance(volume_shape, dict):
        return VolumeShape(
            X=int(volume_shape["X"]),
            Y=int(volume_shape["Y"]),
            Z=int(volume_shape["Z"]),
            T=int(volume_shape["T"]),
        )
    if isinstance(volume_shape, (tuple, list)) and len(volume_shape) == 4:
        return VolumeShape(*map(int, volume_shape))
    raise ValueError("volume_shape must be VolumeShape, dict with X/Y/Z/T, or a 4-tuple/list.")


def infer_or_validate_volume_shape(
    y_np: np.ndarray,
    volume_shape: Optional[Union[VolumeShape, Tuple[int, int, int, int], Dict[str, int]]],
) -> VolumeShape:
    volume_shape = parse_volume_shape(volume_shape)
    if y_np.ndim in (4, 5):
        T, Z, Y, X = map(int, y_np.shape[:4])
        inferred = VolumeShape(X=X, Y=Y, Z=Z, T=T)
        if volume_shape is not None and volume_shape != inferred:
            raise ValueError(f"volume_shape mismatch: provided={volume_shape} inferred={inferred}")
        return inferred

    if volume_shape is None:
        raise ValueError("For flattened targets (1D/2D), you must provide volume_shape=(X,Y,Z,T).")

    N = volume_shape.N
    if y_np.ndim == 2 and int(y_np.shape[0]) != N:
        raise ValueError(f"Flattened y first dim must be N={N}, got {y_np.shape[0]}")
    if y_np.ndim == 1 and int(y_np.shape[0]) != N:
        raise ValueError(f"Flattened y length must be N={N}, got {y_np.shape[0]}")
    return volume_shape


def target_dim_from_array(y: np.ndarray) -> int:
    if y.ndim == 5:
        return int(y.shape[-1])
    if y.ndim == 4:
        return 1
    if y.ndim == 2:
        return int(y.shape[1])
    if y.ndim == 1:
        return 1
    raise ValueError(f"Unsupported target ndim: {y.ndim} with shape {y.shape}")


def idx_to_xyzt(idx: int, s: VolumeShape) -> Tuple[int, int, int, int]:
    if idx < 0 or idx >= s.N:
        raise IndexError(f"idx out of range: {idx} not in [0, {s.N})")
    x = idx % s.X
    idx //= s.X
    y = idx % s.Y
    idx //= s.Y
    z = idx % s.Z
    idx //= s.Z
    t = idx
    return int(x), int(y), int(z), int(t)


def read_target_at_xyzt(
    arr: np.ndarray,
    x: int,
    y: int,
    z: int,
    t: int,
    volume_shape: VolumeShape,
) -> torch.Tensor:
    if arr.ndim == 5:
        return to_torch_copy(arr[t, z, y, x, :])
    if arr.ndim == 4:
        return to_torch_copy([arr[t, z, y, x]])

    flat_idx = ((t * volume_shape.Z + z) * volume_shape.Y + y) * volume_shape.X + x
    if arr.ndim == 2:
        return to_torch_copy(arr[flat_idx, :])
    if arr.ndim == 1:
        return to_torch_copy([arr[flat_idx]])

    raise ValueError(f"Unsupported target ndim: {arr.ndim} with shape {arr.shape}")


def compute_input_stats_analytic(
    s: VolumeShape,
    unbiased: bool,
    dtype: np.dtype,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = float(s.N)
    correction = (N / (N - 1.0)) if (unbiased and N > 1) else 1.0

    def mean_n(n: int) -> float:
        return (float(n) - 1.0) / 2.0

    def var_pop(n: int) -> float:
        if n <= 1:
            return 0.0
        return (float(n) * float(n) - 1.0) / 12.0

    means = np.array([mean_n(s.X), mean_n(s.Y), mean_n(s.Z), mean_n(s.T)], dtype=dtype)
    vars_ = np.array([var_pop(s.X), var_pop(s.Y), var_pop(s.Z), var_pop(s.T)], dtype=dtype) * correction
    stds = np.sqrt(np.maximum(vars_, eps)).astype(dtype)

    x_mean = torch.from_numpy(means.astype(np.float32)).view(1, 4)
    x_std = torch.from_numpy(stds.astype(np.float32)).view(1, 4)
    x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
    return x_mean, x_std


def compute_target_stats_streaming(
    y: np.ndarray,
    eps: float,
    chunk_voxels: int = 1_000_000,
    dtype: np.dtype = np.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C = target_dim_from_array(y)

    # flatten 视图（尽量不复制）
    if y.ndim == 5:
        flat = y.reshape(-1, y.shape[-1])
    elif y.ndim == 4:
        flat = y.reshape(-1, 1)
    elif y.ndim == 2:
        flat = y
    elif y.ndim == 1:
        flat = y.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported target ndim: {y.ndim}")

    N = int(flat.shape[0])
    count = 0
    s1 = np.zeros((C,), dtype=dtype)   # sum
    s2 = np.zeros((C,), dtype=dtype)   # sumsq

    for start in range(0, N, chunk_voxels):
        end = min(start + chunk_voxels, N)

        # 这里会把 dtype 转成 float64（通常需要一次拷贝），但没有逐元素 Python 循环了
        block = np.asarray(flat[start:end], dtype=dtype)

        count_b = block.shape[0]
        count += count_b
        s1 += block.sum(axis=0)
        s2 += (block * block).sum(axis=0)

    if count < 2:
        mean = s1 / max(count, 1)
        var = np.zeros_like(mean)
    else:
        mean = s1 / count
        # 无偏方差（ddof=1）
        var = (s2 - (s1 * s1) / count) / (count - 1)
        var = np.maximum(var, 0.0)

    std = np.sqrt(np.maximum(var, eps))

    y_mean = torch.from_numpy(mean.astype(np.float32)).view(1, -1)
    y_std = torch.from_numpy(std.astype(np.float32)).view(1, -1)
    y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)
    return y_mean, y_std

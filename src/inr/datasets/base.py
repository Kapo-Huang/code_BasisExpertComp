from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

NORMALIZATION_SCHEME_Z_SCORE = "z_score"
NORMALIZATION_SCHEME_MINMAX_NEG1_POS1 = "minmax_neg1_pos1"
DEFAULT_NORMALIZATION_SCHEME = NORMALIZATION_SCHEME_Z_SCORE


@dataclass(frozen=True)
class VolumeShape:
    X: int
    Y: int
    Z: int
    T: int

    @property
    def N(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z) * int(self.T)


def to_torch_copy(x: Union[np.ndarray, list, tuple]) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def peek_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def resolve_normalization_scheme(scheme: str | None) -> str:
    if scheme is None:
        return DEFAULT_NORMALIZATION_SCHEME

    normalized = str(scheme).strip().lower().replace("-", "_")
    aliases = {
        "z_score": NORMALIZATION_SCHEME_Z_SCORE,
        "zscore": NORMALIZATION_SCHEME_Z_SCORE,
        "standard": NORMALIZATION_SCHEME_Z_SCORE,
        "standard_score": NORMALIZATION_SCHEME_Z_SCORE,
        "minmax_neg1_pos1": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
        "range_neg1_pos1": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
        "minmax": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
        "min_max": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
        "minmax[-1,1]": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
        "min_max[-1,1]": NORMALIZATION_SCHEME_MINMAX_NEG1_POS1,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown normalization_scheme '{scheme}'. "
            f"Supported: {NORMALIZATION_SCHEME_Z_SCORE}, {NORMALIZATION_SCHEME_MINMAX_NEG1_POS1}"
        )
    return aliases[normalized]


def _coerce_stats_array(value) -> np.ndarray | None:
    if value is None or isinstance(value, dict):
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _infer_volume_shape_from_x_mean(x_mean: np.ndarray | None) -> VolumeShape | None:
    if x_mean is None or x_mean.size != 4:
        return None

    dims: list[int] = []
    for mean in x_mean:
        n = int(round(float(mean) * 2.0 + 1.0))
        if n <= 0:
            return None
        expected_mean = (float(n) - 1.0) / 2.0
        if abs(float(mean) - expected_mean) > 1e-3:
            return None
        dims.append(n)

    return VolumeShape(X=dims[0], Y=dims[1], Z=dims[2], T=dims[3])


def _infer_normalization_scheme_from_input_stats(payload) -> str | None:
    if not isinstance(payload, dict):
        return None

    x_mean = _coerce_stats_array(payload.get("x_mean"))
    x_std = _coerce_stats_array(payload.get("x_std"))
    if x_mean is None or x_std is None or x_std.size != 4:
        return None

    volume_shape = _infer_volume_shape_from_x_mean(x_mean)
    if volume_shape is None:
        return None

    best_scheme = None
    best_error = np.inf
    for scheme in (NORMALIZATION_SCHEME_Z_SCORE, NORMALIZATION_SCHEME_MINMAX_NEG1_POS1):
        _, expected_std = compute_input_stats_analytic(
            volume_shape,
            unbiased=True,
            dtype=np.float64,
            eps=1e-12,
            normalization_scheme=scheme,
        )
        expected = expected_std.numpy().reshape(-1).astype(np.float64, copy=False)
        rel_error = np.max(np.abs(x_std - expected) / np.maximum(np.abs(expected), 1e-12))
        if rel_error < best_error:
            best_error = rel_error
            best_scheme = scheme

    if best_error <= 1e-3:
        return best_scheme
    return None


def resolve_checkpoint_normalization_scheme(payload) -> str:
    raw = payload.get("normalization_scheme") if isinstance(payload, dict) else None
    if raw is not None:
        return resolve_normalization_scheme(raw)

    inferred = _infer_normalization_scheme_from_input_stats(payload)
    if inferred is not None:
        return inferred

    return DEFAULT_NORMALIZATION_SCHEME


def target_dim_from_array(arr: np.ndarray) -> int:
    if arr.ndim in (5, 2):
        return int(arr.shape[-1])
    if arr.ndim in (4, 1):
        return 1
    raise ValueError(f"Unsupported target ndim: {arr.ndim} with shape {arr.shape}")


def infer_or_validate_volume_shape(
    arr: np.ndarray,
    volume_shape: Optional[Union[VolumeShape, dict]],
) -> VolumeShape:
    if volume_shape is not None:
        if isinstance(volume_shape, VolumeShape):
            s = volume_shape
        else:
            s = VolumeShape(
                X=int(volume_shape["X"]),
                Y=int(volume_shape["Y"]),
                Z=int(volume_shape["Z"]),
                T=int(volume_shape["T"]),
            )

        if arr.ndim in (4, 5):
            expected = (int(s.T), int(s.Z), int(s.Y), int(s.X))
            got = tuple(int(v) for v in arr.shape[:4])
            if got != expected:
                raise ValueError(f"Volume shape mismatch: array={got}, expected={expected}")
        elif arr.ndim in (1, 2):
            if int(arr.shape[0]) != int(s.N):
                raise ValueError(f"Flat target size mismatch: array N={arr.shape[0]}, expected N={s.N}")
        else:
            raise ValueError(f"Unsupported target ndim: {arr.ndim} with shape {arr.shape}")
        return s

    if arr.ndim not in (4, 5):
        raise ValueError(
            "volume_shape must be provided for flat targets with ndim 1/2; "
            f"got ndim={arr.ndim}, shape={arr.shape}"
        )

    return VolumeShape(
        X=int(arr.shape[3]),
        Y=int(arr.shape[2]),
        Z=int(arr.shape[1]),
        T=int(arr.shape[0]),
    )


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
    normalization_scheme: str = DEFAULT_NORMALIZATION_SCHEME,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scheme = resolve_normalization_scheme(normalization_scheme)

    if scheme == NORMALIZATION_SCHEME_Z_SCORE:
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
    else:
        def center_n(n: int) -> float:
            return (float(n) - 1.0) / 2.0

        def half_range_n(n: int) -> float:
            if n <= 1:
                return 0.0
            return (float(n) - 1.0) / 2.0

        means = np.array([center_n(s.X), center_n(s.Y), center_n(s.Z), center_n(s.T)], dtype=dtype)
        stds = np.array(
            [half_range_n(s.X), half_range_n(s.Y), half_range_n(s.Z), half_range_n(s.T)],
            dtype=dtype,
        )
        stds = np.maximum(stds, eps).astype(dtype)

    x_mean = torch.from_numpy(means.astype(np.float32)).view(1, 4)
    x_std = torch.from_numpy(stds.astype(np.float32)).view(1, 4)
    x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
    return x_mean, x_std


def compute_target_stats_streaming(
    y: np.ndarray,
    eps: float,
    chunk_voxels: int = 1_000_000,
    dtype: np.dtype = np.float64,
    normalization_scheme: str = DEFAULT_NORMALIZATION_SCHEME,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C = target_dim_from_array(y)
    scheme = resolve_normalization_scheme(normalization_scheme)

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
    if scheme == NORMALIZATION_SCHEME_Z_SCORE:
        count = 0
        s1 = np.zeros((C,), dtype=dtype)
        s2 = np.zeros((C,), dtype=dtype)

        for start in range(0, N, chunk_voxels):
            end = min(start + chunk_voxels, N)
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
            var = (s2 - (s1 * s1) / count) / (count - 1)
            var = np.maximum(var, 0.0)

        std = np.sqrt(np.maximum(var, eps))
    else:
        mins = np.full((C,), np.inf, dtype=dtype)
        maxs = np.full((C,), -np.inf, dtype=dtype)

        for start in range(0, N, chunk_voxels):
            end = min(start + chunk_voxels, N)
            block = np.asarray(flat[start:end], dtype=dtype)
            mins = np.minimum(mins, np.min(block, axis=0))
            maxs = np.maximum(maxs, np.max(block, axis=0))

        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            raise ValueError("Failed to compute finite target min/max stats.")

        mean = (mins + maxs) / 2.0
        std = np.maximum((maxs - mins) / 2.0, eps)

    y_mean = torch.from_numpy(mean.astype(np.float32)).view(1, -1)
    y_std = torch.from_numpy(std.astype(np.float32)).view(1, -1)
    y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)
    return y_mean, y_std

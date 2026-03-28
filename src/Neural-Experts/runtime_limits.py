from __future__ import annotations

import os

_THREADPOOL_LIMITER = None
_RUNTIME_LIMITS_APPLIED = False

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


def _parse_positive_int(raw_value: str | None, default_value: int) -> int:
    if raw_value is None:
        return int(default_value)
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed = int(default_value)
    return max(1, parsed)


def get_thread_limits(default_threads: int = 1) -> tuple[int, int]:
    intra_threads = _parse_positive_int(os.environ.get("NEURAL_EXPERTS_NUM_THREADS"), default_threads)
    interop_threads = _parse_positive_int(os.environ.get("NEURAL_EXPERTS_NUM_INTEROP_THREADS"), intra_threads)
    return intra_threads, interop_threads


def configure_threading_env(default_threads: int = 1) -> tuple[int, int]:
    intra_threads, interop_threads = get_thread_limits(default_threads=default_threads)
    for key in _THREAD_ENV_KEYS:
        os.environ.setdefault(key, str(intra_threads))
    return intra_threads, interop_threads


def apply_runtime_thread_limits(default_threads: int = 1) -> tuple[int, int]:
    global _THREADPOOL_LIMITER
    global _RUNTIME_LIMITS_APPLIED

    intra_threads, interop_threads = configure_threading_env(default_threads=default_threads)
    if _RUNTIME_LIMITS_APPLIED:
        return intra_threads, interop_threads

    try:
        import torch

        try:
            torch.set_num_threads(intra_threads)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
    except Exception:
        pass

    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=intra_threads)
    except Exception:
        _THREADPOOL_LIMITER = None

    _RUNTIME_LIMITS_APPLIED = True
    return intra_threads, interop_threads

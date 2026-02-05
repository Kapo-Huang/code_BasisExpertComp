import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

EPS = 1e-12

logger = logging.getLogger(__name__)


def compute_psnr(gt, pred):
    gt_f = np.asarray(gt, dtype=np.float64)
    pred_f = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean((gt_f - pred_f) ** 2))
    data_range = float(np.max(gt_f) - np.min(gt_f))
    if data_range <= 0:
        data_range = float(np.max(np.abs(gt_f))) + EPS
    if skimage_psnr is not None:
        return float(skimage_psnr(gt_f, pred_f, data_range=data_range))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / (mse + EPS))


def to_scalar(u):
    if isinstance(u, np.ndarray) and u.ndim == 2 and u.shape[1] in (2, 3):
        return np.linalg.norm(u, axis=1)
    return u


def split_frames(arr: np.ndarray, n_frames: int) -> List[np.ndarray]:
    """
    Split arr into n_frames along axis=0.
    Preferred: equal-length split when divisible; fallback to np.array_split.
    Supports:
      - (T*N, C) or (T*N,)  -> split into n_frames chunks each ~N
      - (T, N, C)          -> split into n_frames along T
    """
    a = np.asarray(arr)
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")

    if a.ndim >= 2 and a.shape[0] == n_frames:
        return [a[i] for i in range(n_frames)]

    total = a.shape[0]
    if total % n_frames == 0:
        step = total // n_frames
        return [a[i * step : (i + 1) * step] for i in range(n_frames)]

    logger.warning(
        "total points %s not divisible by n_frames=%s. "
        "Falling back to np.array_split -> unequal frame sizes may affect metrics.",
        total,
        n_frames,
    )
    return list(np.array_split(a, n_frames, axis=0))


def top_indices_by_gt(gt_1d: np.ndarray, top_percent: Optional[float], topk: Optional[int]) -> np.ndarray:
    g = np.asarray(gt_1d).reshape(-1)
    n = g.size
    if n == 0:
        return np.array([], dtype=np.int64)

    if topk is not None and topk > 0:
        k = min(int(topk), n)
        idx = np.argpartition(g, -k)[-k:]
        idx = idx[np.argsort(g[idx])[::-1]]
        return idx

    if top_percent is None:
        raise ValueError("Either top_percent or topk must be provided.")
    p = float(top_percent)
    if p <= 0:
        return np.array([], dtype=np.int64)
    k = max(1, int(math.ceil(n * p)))
    idx = np.argpartition(g, -k)[-k:]
    idx = idx[np.argsort(g[idx])[::-1]]
    return idx


def tail_metrics(gt_1d: np.ndarray, pred_1d: np.ndarray, top_percent: float = 0.01, topk: Optional[int] = None):
    """
    Tail defined by top values of gt (top_percent or topk).
    Returns dict of tail metrics.
    """
    g = np.asarray(gt_1d, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_1d, dtype=np.float64).reshape(-1)
    if g.size != p.size:
        raise ValueError("tail_metrics: gt/pred size mismatch")

    idx = top_indices_by_gt(g, top_percent=top_percent, topk=topk)
    if idx.size == 0:
        return dict(
            tail_count=0,
            tail_mae=float("nan"),
            tail_mre=float("nan"),
            tail_max_abs=float("nan"),
            tail_max_rel=float("nan"),
            tail_p99_abs=float("nan"),
            tail_p999_abs=float("nan"),
            tail_p99_rel=float("nan"),
            tail_p999_rel=float("nan"),
        )

    err_abs = np.abs(p[idx] - g[idx])
    err_rel = err_abs / (np.abs(g[idx]) + EPS)

    return dict(
        tail_count=int(idx.size),
        tail_mae=float(np.nanmean(err_abs)),
        tail_mre=float(np.nanmean(err_rel)),
        tail_max_abs=float(np.nanmax(err_abs)),
        tail_max_rel=float(np.nanmax(err_rel)),
        tail_p99_abs=float(np.nanpercentile(err_abs, 99)),
        tail_p999_abs=float(np.nanpercentile(err_abs, 99.9)),
        tail_p99_rel=float(np.nanpercentile(err_rel, 99)),
        tail_p999_rel=float(np.nanpercentile(err_rel, 99.9)),
    )


def iou_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    if a.size != b.size:
        raise ValueError("iou_dice: mask size mismatch")

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    denom = a.sum() + b.sum()
    iou = float(inter / union) if union > 0 else 1.0
    dice = float((2 * inter) / denom) if denom > 0 else 1.0
    return iou, dice


def hotspot_metrics(gt_1d: np.ndarray, pred_1d: np.ndarray, tau: Optional[float], tau_percentile: Optional[float]):
    g = np.asarray(gt_1d, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_1d, dtype=np.float64).reshape(-1)
    if g.size != p.size:
        raise ValueError("hotspot_metrics: gt/pred size mismatch")

    if tau is None:
        if tau_percentile is None:
            raise ValueError("Either tau or tau_percentile must be provided")
        tau = float(np.nanpercentile(g, tau_percentile))

    m_g = g >= tau
    m_p = p >= tau
    iou, dice = iou_dice(m_g, m_p)
    return dict(
        hotspot_tau=float(tau),
        hotspot_iou=iou,
        hotspot_dice=dice,
        hotspot_gt_count=int(m_g.sum()),
        hotspot_pred_count=int(m_p.sum()),
    )


def pairwise_min_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each point in A, compute min Euclidean distance to any point in B.
    A: (m,3), B: (n,3)
    Returns: (m,)
    """
    if A.size == 0:
        return np.array([], dtype=np.float64)
    if B.size == 0:
        return np.full((A.shape[0],), np.inf, dtype=np.float64)

    diff = A[:, None, :] - B[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return np.sqrt(np.min(d2, axis=1))


def peak_matching_metrics(
    points_xyz: np.ndarray,
    gt_1d: np.ndarray,
    pred_1d: np.ndarray,
    peak_topk: int,
    match_radius: Optional[float],
    match_radius_factor: float,
):
    """
    Peaks: top-k points by value (separately for gt and pred).
    Matching: nearest-neighbor distances from gt peaks -> pred peaks.
    Recall@r: fraction of gt peaks with distance <= r.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    g = np.asarray(gt_1d, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_1d, dtype=np.float64).reshape(-1)
    if pts.shape[0] != g.size or g.size != p.size:
        raise ValueError("peak_matching_metrics: points/gt/pred size mismatch")

    k = int(peak_topk)
    if k <= 0:
        return dict(
            peak_k=0,
            peak_nn_mean=float("nan"),
            peak_nn_median=float("nan"),
            peak_nn_max=float("nan"),
            peak_recall=float("nan"),
        )

    idx_g = top_indices_by_gt(g, top_percent=None, topk=k)
    idx_p = top_indices_by_gt(p, top_percent=None, topk=k)

    A = pts[idx_g]
    B = pts[idx_p]

    d = pairwise_min_dist(A, B)

    if match_radius is None:
        bb_min = np.nanmin(pts, axis=0)
        bb_max = np.nanmax(pts, axis=0)
        diag = float(np.linalg.norm(bb_max - bb_min))
        r = diag * float(match_radius_factor)
    else:
        r = float(match_radius)

    recall = float(np.mean(d <= r)) if d.size > 0 else float("nan")

    return dict(
        peak_k=int(idx_g.size),
        peak_nn_mean=float(np.nanmean(d)),
        peak_nn_median=float(np.nanmedian(d)),
        peak_nn_max=float(np.nanmax(d)),
        peak_recall=recall,
        peak_match_radius=float(r),
    )

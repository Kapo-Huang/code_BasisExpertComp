import logging
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from .io_utils import (
    count_params_from_ckpt,
    load_yaml,
    match_checkpoint,
    pick_by_epoch,
    pick_pred_for_attr,
    resolve_gt_paths,
    safe_load_npy,
)
from .metrics import (
    compute_psnr,
    hotspot_metrics,
    peak_matching_metrics,
    split_frames,
    tail_metrics,
    to_scalar,
)
from .plotting import plot_pred_vs_gt, plot_rel_error_series

logger = logging.getLogger(__name__)


def validate_experiment(
    exp_dir: Path,
    outdir: Path,
    mesh_path: Path,
    n_frames: int,
    epoch: int,
    img_scale: float,
    relerr_clip_percentile: float | None,
    relerr_max: float | None,
    tail_percent: float,
    tail_topk: int | None,
    hotspot_tau: float | None,
    hotspot_tau_percentile: float | None,
    peak_topk: int,
    match_radius: float | None,
    match_radius_factor: float,
):
    t_start = time.perf_counter()
    cfg_path = exp_dir / "configs" / "config.yaml"
    cfg = load_yaml(cfg_path)
    logger.info("[%s] Stage: load config = %.3fs", exp_dir.name, time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    pred_files = sorted((exp_dir / "predictions").glob("*.npy"))
    logger.info(
        "[%s] Stage: find prediction file = %.3fs",
        exp_dir.name,
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    ckpt_files = sorted((exp_dir / "checkpoints").glob("*.pth"))
    ckpt_path = match_checkpoint(epoch, ckpt_files)
    logger.info(
        "[%s] Stage: find checkpoint file = %.3fs",
        exp_dir.name,
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    gt_paths = resolve_gt_paths(cfg)
    if not gt_paths:
        raise FileNotFoundError(f"GT not found for {exp_dir}")
    for name, gt_path in gt_paths.items():
        if gt_path is None or not gt_path.exists():
            raise FileNotFoundError(f"GT not found for {exp_dir} ({name})")
    logger.info(
        "[%s] Stage: resolve GT path = %.3fs",
        exp_dir.name,
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    model_size = ckpt_path.stat().st_size if ckpt_path is not None else None
    param_count = count_params_from_ckpt(ckpt_path) if ckpt_path is not None else None
    logger.info("[%s] Stage: model stats = %.3fs", exp_dir.name, time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    mesh_points = None
    if mesh_path.exists():
        mesh_template = pv.read(str(mesh_path))
        mesh_points = np.asarray(mesh_template.points, dtype=np.float64)
    else:
        mesh_template = None

    def _agg_mean(key: str, xs: list[dict]):
        vals = [d.get(key, float("nan")) for d in xs]
        return float(np.nanmean(np.asarray(vals, dtype=np.float64)))

    def _agg_max(key: str, xs: list[dict]):
        vals = [d.get(key, float("nan")) for d in xs]
        return float(np.nanmax(np.asarray(vals, dtype=np.float64)))

    rows = []
    pred_imgs = []
    cmp_imgs = []

    for attr_name, gt_path in gt_paths.items():
        t_stage = time.perf_counter()
        if len(gt_paths) == 1 and attr_name == "targets":
            pred_path = pick_by_epoch(pred_files, epoch)
        else:
            pred_path = pick_pred_for_attr(pred_files, epoch, attr_name)
        if pred_path is None:
            raise FileNotFoundError(f"No predictions found for {attr_name} epoch {epoch} in {exp_dir}")
        logger.info(
            "[%s] Stage: pick pred (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        t_stage = time.perf_counter()
        gt = safe_load_npy(gt_path)
        pred = safe_load_npy(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"GT and pred shape mismatch ({attr_name}): {gt.shape} vs {pred.shape}")
        logger.info(
            "[%s] Stage: load arrays (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        t_stage = time.perf_counter()
        psnr_val = compute_psnr(gt, pred)
        logger.info(
            "[%s] Stage: compute PSNR (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        t_stage = time.perf_counter()
        pred_frames = split_frames(pred, n_frames)
        gt_frames = split_frames(gt, n_frames)
        pred_series = [to_scalar(u) for u in pred_frames]
        gt_series = [to_scalar(u) for u in gt_frames]
        logger.info(
            "[%s] Stage: split series (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        t_stage = time.perf_counter()
        tail_list = []
        hotspot_list = []
        peak_list = []

        for gt_u, pred_u in zip(gt_series, pred_series):
            gt_u = np.asarray(gt_u).reshape(-1)
            pred_u = np.asarray(pred_u).reshape(-1)

            tail_list.append(tail_metrics(gt_u, pred_u, top_percent=tail_percent, topk=tail_topk))
            hotspot_list.append(hotspot_metrics(gt_u, pred_u, tau=hotspot_tau, tau_percentile=hotspot_tau_percentile))

            if mesh_points is not None and mesh_points.shape[0] == gt_u.size:
                peak_list.append(
                    peak_matching_metrics(
                        mesh_points,
                        gt_u,
                        pred_u,
                        peak_topk=peak_topk,
                        match_radius=match_radius,
                        match_radius_factor=match_radius_factor,
                    )
                )

        metrics = {}
        metrics.update(
            {
                "tail_percent": float(tail_percent),
                "tail_count_mean": _agg_mean("tail_count", tail_list),
                "tail_mae_mean": _agg_mean("tail_mae", tail_list),
                "tail_mre_mean": _agg_mean("tail_mre", tail_list),
                "tail_p99_abs_mean": _agg_mean("tail_p99_abs", tail_list),
                "tail_p999_abs_mean": _agg_mean("tail_p999_abs", tail_list),
                "tail_p99_rel_mean": _agg_mean("tail_p99_rel", tail_list),
                "tail_p999_rel_mean": _agg_mean("tail_p999_rel", tail_list),
                "tail_max_abs_worst": _agg_max("tail_max_abs", tail_list),
                "tail_max_rel_worst": _agg_max("tail_max_rel", tail_list),
            }
        )

        metrics.update(
            {
                "hotspot_tau_percentile": float(hotspot_tau_percentile) if hotspot_tau_percentile is not None else "",
                "hotspot_tau_mean": _agg_mean("hotspot_tau", hotspot_list),
                "hotspot_iou_mean": _agg_mean("hotspot_iou", hotspot_list),
                "hotspot_dice_mean": _agg_mean("hotspot_dice", hotspot_list),
                "hotspot_gt_count_mean": _agg_mean("hotspot_gt_count", hotspot_list),
                "hotspot_pred_count_mean": _agg_mean("hotspot_pred_count", hotspot_list),
            }
        )

        if peak_list:
            metrics.update(
                {
                    "peak_topk": int(peak_topk),
                    "peak_match_radius_mean": _agg_mean("peak_match_radius", peak_list),
                    "peak_nn_mean": _agg_mean("peak_nn_mean", peak_list),
                    "peak_nn_median_mean": _agg_mean("peak_nn_median", peak_list),
                    "peak_nn_max_worst": _agg_max("peak_nn_max", peak_list),
                    "peak_recall_mean": _agg_mean("peak_recall", peak_list),
                }
            )
        else:
            metrics.update(
                {
                    "peak_topk": int(peak_topk),
                    "peak_match_radius_mean": "",
                    "peak_nn_mean": "",
                    "peak_nn_median_mean": "",
                    "peak_nn_max_worst": "",
                    "peak_recall_mean": "",
                }
            )

        logger.info(
            "[%s] Stage: compute tail/hotspot/peak metrics (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        t_stage = time.perf_counter()
        outdir.mkdir(parents=True, exist_ok=True)
        pred_img = outdir / f"{exp_dir.name}_{attr_name}_relerr_series_{pred_path.stem}.png"
        cmp_img = outdir / f"{exp_dir.name}_{attr_name}_pred_vs_gt_{pred_path.stem}.png"
        logger.info(
            "[%s] Stage: prepare outputs (%s) = %.3fs",
            exp_dir.name,
            attr_name,
            time.perf_counter() - t_stage,
        )

        if mesh_path.exists():
            t_stage = time.perf_counter()
            plot_rel_error_series(
                mesh_path, pred_series, gt_series, pred_img, img_scale, relerr_clip_percentile, relerr_max
            )
            logger.info(
                "[%s] Stage: plot rel error (%s) = %.3fs",
                exp_dir.name,
                attr_name,
                time.perf_counter() - t_stage,
            )

            t_stage = time.perf_counter()
            plot_pred_vs_gt(mesh_path, pred_series, gt_series, cmp_img, img_scale)
            logger.info(
                "[%s] Stage: plot pred vs gt (%s) = %.3fs",
                exp_dir.name,
                attr_name,
                time.perf_counter() - t_stage,
            )

        db_size = gt_path.stat().st_size
        cr = float(db_size / model_size) if model_size and model_size > 0 else None

        row = {
            "exp_id": cfg.get("exp_id", exp_dir.name),
            "model_name": cfg.get("model", {}).get("name", ""),
            "dataset_name": cfg.get("data", {}).get("dataset_name", ""),
            "split": cfg.get("data", {}).get("split", ""),
            "attr_name": attr_name,
            "pred_file": str(pred_path),
            "ckpt_file": str(ckpt_path) if ckpt_path is not None else "",
            "params": param_count if param_count is not None else "",
            "psnr": psnr_val,
            "cr": cr if cr is not None else "",
            "model_size_bytes": model_size if model_size is not None else "",
            "db_size_bytes": db_size,
            **metrics,
        }
        rows.append(row)
        pred_imgs.append(pred_img)
        cmp_imgs.append(cmp_img)

    logger.info("[%s] Stage: total = %.3fs", exp_dir.name, time.perf_counter() - t_start)
    return rows, pred_imgs, cmp_imgs

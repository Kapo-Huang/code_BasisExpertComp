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
    resolve_gt_path,
    resolve_gt_paths,
    safe_load_npy,
)
from .metrics import (
    compute_psnr,
    error_stats,
    hotspot_metrics,
    peak_matching_metrics,
    split_frames,
    to_scalar,
    EPS,
)
from .plotting import plot_pred_vs_gt, plot_rel_error_series


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
    print(f"[{exp_dir.name}] Stage: load config = {time.perf_counter() - t_start:.3f}s")

    t_stage = time.perf_counter()
    pred_files = sorted((exp_dir / "predictions").glob("*.npy"))
    print(f"[{exp_dir.name}] Stage: find prediction file = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    ckpt_files = sorted((exp_dir / "checkpoints").glob("*.pth"))
    ckpt_path = match_checkpoint(epoch, ckpt_files)
    print(f"[{exp_dir.name}] Stage: find checkpoint file = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    gt_paths = resolve_gt_paths(cfg)
    if not gt_paths:
        raise FileNotFoundError(f"GT not found for {exp_dir}")
    for name, gt_path in gt_paths.items():
        if gt_path is None or not gt_path.exists():
            raise FileNotFoundError(f"GT not found for {exp_dir} ({name})")
    print(f"[{exp_dir.name}] Stage: resolve GT path = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    model_size = ckpt_path.stat().st_size if ckpt_path is not None else None
    param_count = count_params_from_ckpt(ckpt_path) if ckpt_path is not None else None
    print(f"[{exp_dir.name}] Stage: model stats = {time.perf_counter() - t_stage:.3f}s")

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
        print(f"[{exp_dir.name}] Stage: pick pred ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

        t_stage = time.perf_counter()
        gt = safe_load_npy(gt_path)
        pred = safe_load_npy(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"GT and pred shape mismatch ({attr_name}): {gt.shape} vs {pred.shape}")
        print(f"[{exp_dir.name}] Stage: load arrays ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

        t_stage = time.perf_counter()
        psnr_mean, psnr_min, psnr_per_channel = compute_psnr(gt, pred)
        print(f"[{exp_dir.name}] Stage: compute PSNR ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

        t_stage = time.perf_counter()
        pred_frames = split_frames(pred, n_frames)
        gt_frames = split_frames(gt, n_frames)
        pred_series = [to_scalar(u) for u in pred_frames]
        gt_series = [to_scalar(u) for u in gt_frames]
        print(f"[{exp_dir.name}] Stage: split series ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

        t_stage = time.perf_counter()
        err_list = []
        hotspot_list = []
        peak_list = []

        for gt_raw, pred_raw, gt_u, pred_u in zip(gt_frames, pred_frames, gt_series, pred_series):
            err_list.append(error_stats(gt_raw, pred_raw, eps=EPS))
            gt_u = np.asarray(gt_u).reshape(-1)
            pred_u = np.asarray(pred_u).reshape(-1)

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
                "mse_all_mean": _agg_mean("mse_all", err_list),
                "rmse_all_mean": _agg_mean("rmse_all", err_list),
                "mae_all_mean": _agg_mean("mae_all", err_list),
                "mre_all_mean": _agg_mean("mre_all", err_list),
                "abs_p95_mean": _agg_mean("abs_p95", err_list),
                "abs_p99_mean": _agg_mean("abs_p99", err_list),
                "abs_p999_mean": _agg_mean("abs_p999", err_list),
                "rel_p95_mean": _agg_mean("rel_p95", err_list),
                "rel_p99_mean": _agg_mean("rel_p99", err_list),
                "rel_p999_mean": _agg_mean("rel_p999", err_list),
                "abs_max_worst": _agg_max("abs_max", err_list),
                "rel_max_worst": _agg_max("rel_max", err_list),
                "rmse_ch_worst_mean": _agg_mean("rmse_ch_worst", err_list),
                "mae_ch_worst_mean": _agg_mean("mae_ch_worst", err_list),
                "mre_ch_worst_mean": _agg_mean("mre_ch_worst", err_list),
                "rmse_all_worst": _agg_max("rmse_all", err_list),
                "abs_p99_worst": _agg_max("abs_p99", err_list),
                "rel_p99_worst": _agg_max("rel_p99", err_list),
            }
        )

        metrics.update(
            {
                "hotspot_tau_percentile": float(hotspot_tau_percentile) if hotspot_tau_percentile is not None else "",
                "hotspot_tau_mean": _agg_mean("hotspot_tau", hotspot_list),
                "hotspot_iou_mean": _agg_mean("hotspot_iou", hotspot_list),
                "hotspot_gt_count_mean": _agg_mean("hotspot_gt_count", hotspot_list),
                "hotspot_pred_count_mean": _agg_mean("hotspot_pred_count", hotspot_list),
            }
        )

        if peak_list:
            metrics.update(
                {
                    "peak_topk": int(peak_topk),
                    "peak_match_radius_mean": _agg_mean("peak_match_radius", peak_list),
                    "peak_nn_median_mean": _agg_mean("peak_nn_median", peak_list),
                    "peak_recall_mean": _agg_mean("peak_recall", peak_list),
                }
            )
        else:
            metrics.update(
                {
                    "peak_topk": int(peak_topk),
                    "peak_match_radius_mean": "",
                    "peak_nn_median_mean": "",
                    "peak_recall_mean": "",
                }
            )

        print(
            f"[{exp_dir.name}] Stage: compute tail/hotspot/peak metrics ({attr_name}) = "
            f"{time.perf_counter() - t_stage:.3f}s"
        )

        t_stage = time.perf_counter()
        outdir.mkdir(parents=True, exist_ok=True)
        pred_img = outdir / f"{exp_dir.name}_{attr_name}_relerr_series_{pred_path.stem}.png"
        cmp_img = outdir / f"{exp_dir.name}_{attr_name}_pred_vs_gt_{pred_path.stem}.png"
        print(f"[{exp_dir.name}] Stage: prepare outputs ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

        if mesh_path.exists():
            t_stage = time.perf_counter()
            plot_rel_error_series(
                mesh_path, pred_series, gt_series, pred_img, img_scale, relerr_clip_percentile, relerr_max
            )
            print(f"[{exp_dir.name}] Stage: plot rel error ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")

            t_stage = time.perf_counter()
            plot_pred_vs_gt(mesh_path, pred_series, gt_series, cmp_img, img_scale)
            print(f"[{exp_dir.name}] Stage: plot pred vs gt ({attr_name}) = {time.perf_counter() - t_stage:.3f}s")
        else:
            raise Warning(f"Mesh path not found: {mesh_path}, skipping plots.")

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
            "psnr_mean": psnr_mean,
            "psnr_min": psnr_min,
            "psnr_p10": float(np.percentile(psnr_per_channel, 10)) if psnr_per_channel.size else float("nan"),
            "cr": cr if cr is not None else "",
            "model_size_bytes": model_size if model_size is not None else "",
            "db_size_bytes": db_size,
            **metrics,
        }
        rows.append(row)
        pred_imgs.append(pred_img)
        cmp_imgs.append(cmp_img)

    print(f"[{exp_dir.name}] Stage: total = {time.perf_counter() - t_start:.3f}s")
    return rows, pred_imgs, cmp_imgs

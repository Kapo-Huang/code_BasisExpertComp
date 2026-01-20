import argparse
from pathlib import Path

from .evaluator import validate_experiment
from .io_utils import append_csv_row, collect_experiments


def main():
    parser = argparse.ArgumentParser(description="Validate experiment results and generate plots.")
    parser.add_argument("--experiments", type=str, default="experiments", help="experiments root directory")
    parser.add_argument("--outdir", type=str, default="validate_out", help="output directory")
    parser.add_argument("--csv", type=str, default="validate_out/validation_results.csv", help="csv path to append")
    parser.add_argument("--exp-id", type=str, default=None, help="single experiment id to validate")
    parser.add_argument("--mesh", type=str, default="data/raw/sukong/validate_mesh/sukong_zip_0_0.vtu", help="mesh vtu path")
    parser.add_argument("--n-frames", type=int, default=10, help="number of time frames to split for plotting")
    parser.add_argument("--epoch", type=int, required=True, help="epoch number to select pred/ckpt")
    parser.add_argument("--img-scale", type=float, default=2.0, help="scale factor for plot resolution")
    parser.add_argument(
        "--relerr-clip-percentile",
        type=float,
        default=99.5,
        help="clip relerr colormap max to this percentile (set <=0 to disable)",
    )
    parser.add_argument(
        "--relerr-max",
        type=float,
        default=None,
        help="hard cap for relerr colormap max (optional)",
    )
    parser.add_argument(
        "--tail-percent",
        type=float,
        default=0.01,
        help="tail percent by GT top values (e.g., 0.01=top1%)",
    )
    parser.add_argument(
        "--tail-topk",
        type=int,
        default=None,
        help="use top-k instead of tail-percent if set",
    )
    parser.add_argument(
        "--hotspot-tau",
        type=float,
        default=None,
        help="absolute threshold tau for hotspot (overrides percentile)",
    )
    parser.add_argument(
        "--hotspot-tau-percentile",
        type=float,
        default=99.0,
        help="tau = percentile of GT if hotspot-tau is None",
    )
    parser.add_argument(
        "--peak-topk",
        type=int,
        default=100,
        help="top-k peaks for point-set matching",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=None,
        help="absolute radius for peak matching (same units as mesh points)",
    )
    parser.add_argument(
        "--match-radius-factor",
        type=float,
        default=0.01,
        help="if match-radius is None, r = bbox_diag * factor",
    )
    args = parser.parse_args()

    if args.n_frames <= 0:
        raise ValueError("--n-frames must be > 0")
    if args.img_scale <= 0:
        raise ValueError("--img-scale must be > 0")
    if args.relerr_clip_percentile is not None and args.relerr_clip_percentile <= 0:
        args.relerr_clip_percentile = None

    exp_root = Path(args.experiments)
    outdir = Path(args.outdir)
    csv_path = Path(args.csv)
    mesh_path = Path(args.mesh)

    if args.exp_id:
        exp_dirs = [exp_root / args.exp_id]
        outdir = outdir / args.exp_id
    else:
        exp_dirs = collect_experiments(exp_root)

    fieldnames = [
        "exp_id",
        "model_name",
        "dataset_name",
        "split",
        "attr_name",
        "pred_file",
        "ckpt_file",
        "params",
        "psnr",
        "cr",
        "model_size_bytes",
        "db_size_bytes",
        "tail_percent",
        "tail_count_mean",
        "tail_mae_mean",
        "tail_mre_mean",
        "tail_p99_abs_mean",
        "tail_p999_abs_mean",
        "tail_p99_rel_mean",
        "tail_p999_rel_mean",
        "tail_max_abs_worst",
        "tail_max_rel_worst",
        "hotspot_tau_percentile",
        "hotspot_tau_mean",
        "hotspot_iou_mean",
        "hotspot_dice_mean",
        "hotspot_gt_count_mean",
        "hotspot_pred_count_mean",
        "peak_topk",
        "peak_match_radius_mean",
        "peak_nn_mean",
        "peak_nn_median_mean",
        "peak_nn_max_worst",
        "peak_recall_mean",
    ]

    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        rows, pred_imgs, cmp_imgs = validate_experiment(
            exp_dir,
            outdir,
            mesh_path,
            args.n_frames,
            args.epoch,
            args.img_scale,
            args.relerr_clip_percentile,
            args.relerr_max,
            args.tail_percent,
            args.tail_topk,
            args.hotspot_tau,
            args.hotspot_tau_percentile,
            args.peak_topk,
            args.match_radius,
            args.match_radius_factor,
        )
        for row in rows:
            append_csv_row(csv_path, row, fieldnames)
        print(f"Validated {exp_dir.name}")
        for pred_img, cmp_img, row in zip(pred_imgs, cmp_imgs, rows):
            print(f"  Attr: {row.get('attr_name', '')}")
            print(f"  Pred plot: {pred_img}")
            print(f"  Compare plot: {cmp_img}")


if __name__ == "__main__":
    main()

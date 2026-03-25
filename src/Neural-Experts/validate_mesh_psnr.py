from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
for path in (str(THIS_DIR), str(THIS_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mesh.common import ensure_sys_path, load_config

ensure_sys_path()

from mesh.inference import (
    compute_psnr,
    compute_time_indexers,
    denormalize_targets,
    load_checkpoint,
    normalize_coords,
    predict_block,
    resolve_checkpoint_stats,
    select_indexer_block,
    select_timestamps,
    unwrap_model_state,
    write_csv,
)
from models import build_model

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="PSNR validation for Neural-Experts Mesh models.")
    parser.add_argument("--config", required=True, type=str, help="Path to mesh config yaml")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (.pth)")
    parser.add_argument("--device", default="cuda", type=str, help="Device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", default=16000, type=int, help="Inference batch size")
    parser.add_argument("--timestamps", default="", type=str, help="Optional comma-separated timestep indices")
    parser.add_argument("--max-timesteps", default=0, type=int, help="Optional cap on timesteps to validate")
    parser.add_argument("--csv", default="validate_out/neural_experts_mesh_psnr.csv", type=str, help="CSV output path")
    parser.add_argument("--log", default="validate_out/neural_experts_mesh_psnr.log", type=str, help="Log output path")
    return parser.parse_args()


def _append_log(path: str | Path, lines: list[str], reset: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if reset else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    cfg = load_config(args.config)
    payload = load_checkpoint(args.checkpoint, device="cpu")

    model, _ = build_model(cfg, cfg["LOSS"])
    model.load_state_dict(unwrap_model_state(payload), strict=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()

    source = np.load(cfg["DATA"]["source_path"], mmap_mode="r")
    target = np.load(cfg["DATA"]["target_path"], mmap_mode="r")
    if target.ndim == 1:
        target = target.reshape(-1, 1)

    time_indexers = compute_time_indexers(np.asarray(source[:, -1]))
    selected = select_timestamps(time_indexers, timestamps=args.timestamps, max_timesteps=args.max_timesteps)
    x_mean, x_std, y_mean, y_std = resolve_checkpoint_stats(payload, cfg)

    _append_log(
        args.log,
        [
            f"config={Path(args.config).resolve()}",
            f"checkpoint={Path(args.checkpoint).resolve()}",
            f"attr_name={cfg['DATA']['attr_name']}",
            f"dataset_name={cfg['DATA']['dataset_name']}",
            f"association={cfg['DATA']['association']}",
            "--- per timestep ---",
        ],
        reset=True,
    )

    rows = []
    psnr_values = []
    total_inference = 0.0
    for t_index in selected:
        raw_time, indexer = time_indexers[t_index]
        coords = select_indexer_block(source, indexer).astype(np.float32)
        gt = select_indexer_block(target, indexer).astype(np.float32)
        coords_norm = normalize_coords(coords, x_mean, x_std, bool(cfg["DATA"].get("normalize_inputs", True)))
        pred_norm, elapsed = predict_block(model, coords_norm, int(args.batch_size), device)
        pred = denormalize_targets(pred_norm, y_mean, y_std, bool(cfg["DATA"].get("normalize_targets", False)))
        psnr, mse = compute_psnr(pred, gt)
        psnr_values.append(psnr)
        total_inference += elapsed
        line = (
            f"timestep={t_index:04d} raw_t={raw_time} samples={coords.shape[0]} "
            f"psnr={psnr:.6f} mse={mse:.6e} infer_sec={elapsed:.6f}"
        )
        _append_log(args.log, [line], reset=False)
        rows.append(
            {
                "attr_name": cfg["DATA"]["attr_name"],
                "dataset_name": cfg["DATA"]["dataset_name"],
                "association": cfg["DATA"]["association"],
                "timestamp_index": t_index,
                "raw_time": raw_time,
                "num_samples": int(coords.shape[0]),
                "psnr": psnr,
                "mse": mse,
                "inference_time_sec": elapsed,
                "config_path": str(Path(args.config).resolve()),
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
            }
        )
        logger.info(line)

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    summary_line = f"avg_psnr={avg_psnr:.6f} timesteps={len(psnr_values)} total_inference_sec={total_inference:.6f}"
    _append_log(args.log, ["--- summary ---", summary_line], reset=False)
    logger.info(summary_line)

    rows.append(
        {
            "attr_name": cfg["DATA"]["attr_name"],
            "dataset_name": cfg["DATA"]["dataset_name"],
            "association": cfg["DATA"]["association"],
            "timestamp_index": "__mean__",
            "raw_time": "",
            "num_samples": "",
            "psnr": avg_psnr,
            "mse": "",
            "inference_time_sec": total_inference,
            "config_path": str(Path(args.config).resolve()),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
        }
    )
    write_csv(
        args.csv,
        fieldnames=[
            "attr_name",
            "dataset_name",
            "association",
            "timestamp_index",
            "raw_time",
            "num_samples",
            "psnr",
            "mse",
            "inference_time_sec",
            "config_path",
            "checkpoint_path",
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
for path in (str(THIS_DIR), str(THIS_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from runtime_limits import apply_runtime_thread_limits, configure_threading_env

configure_threading_env()

import numpy as np
import torch

THREAD_LIMITS = apply_runtime_thread_limits()

from mesh.common import ensure_sys_path, load_config

ensure_sys_path()

from mesh.inference import (
    compute_time_indexers,
    denormalize_targets,
    load_checkpoint,
    normalize_coords,
    predict_block,
    resolve_checkpoint_stats,
    select_indexer_block,
    select_timestamps,
    unwrap_model_state,
)
from models import build_model

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="Predict Mesh attribute values with Neural-Experts.")
    parser.add_argument("--config", required=True, type=str, help="Path to mesh config yaml")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (.pth)")
    parser.add_argument("--device", default="cuda", type=str, help="Device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", default=16000, type=int, help="Prediction batch size")
    parser.add_argument("--timestamps", default="", type=str, help="Optional comma-separated timestep indices")
    parser.add_argument("--max-timesteps", default=0, type=int, help="Optional cap on timesteps to export")
    parser.add_argument("--outdir", default="validate_out/neural_experts_predictions", type=str, help="Output directory")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    logger.info("Thread limits active: intra_op=%d inter_op=%d", THREAD_LIMITS[0], THREAD_LIMITS[1])

    cfg = load_config(args.config)
    payload = load_checkpoint(args.checkpoint, device="cpu")

    model, _ = build_model(cfg, cfg["LOSS"])
    model.load_state_dict(unwrap_model_state(payload), strict=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(device).eval()

    source = np.load(cfg["DATA"]["source_path"], mmap_mode="r")
    time_indexers = compute_time_indexers(np.asarray(source[:, -1]))
    selected = select_timestamps(time_indexers, timestamps=args.timestamps, max_timesteps=args.max_timesteps)
    x_mean, x_std, y_mean, y_std = resolve_checkpoint_stats(payload, cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    attr_name = str(cfg["DATA"]["attr_name"])

    for t_index in selected:
        raw_time, indexer = time_indexers[t_index]
        coords = select_indexer_block(source, indexer).astype(np.float32)
        coords_norm = normalize_coords(coords, x_mean, x_std, bool(cfg["DATA"].get("normalize_inputs", False)))
        pred_norm, elapsed = predict_block(model, coords_norm, int(args.batch_size), device)
        pred = denormalize_targets(pred_norm, y_mean, y_std, bool(cfg["DATA"].get("normalize_targets", False)))
        out_path = outdir / f"{attr_name}_t{t_index:04d}.npy"
        np.save(str(out_path), pred.astype(np.float32))
        logger.info("Saved timestep=%d raw_t=%s pred=%s inference_time=%.4fs", t_index, raw_time, out_path, elapsed)


if __name__ == "__main__":
    main()

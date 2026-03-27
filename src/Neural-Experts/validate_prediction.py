from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent

for path in (str(THIS_DIR), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_render_module():
    module_path = THIS_DIR / "validate_prediction_render.py"
    module_name = "neural_experts_validate_prediction_render"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_render = _load_render_module()
build_model_for_run = _render.build_model_for_run
discover_run_artifacts = _render.discover_run_artifacts
load_yaml = _render.load_yaml
predict_timestep_flat_scalar = _render.predict_timestep_flat_scalar
resolve_cli_path = _render.resolve_cli_path
resolve_device = _render.resolve_device
resolve_input_stats = _render.resolve_input_stats
resolve_output_stats = _render.resolve_output_stats
select_timesteps = _render.select_timesteps
setup_logging = _render.setup_logging
volume_shape_from_config = _render.volume_shape_from_config

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Neural-Experts ionization predictions as per-timestep 1D .npy files."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Single Neural-Experts run directory, e.g. result/Ionization/Neural Expert/PD",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Single timestep to export. If omitted, export all timesteps.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default="",
        help="Comma-separated timesteps to export, e.g. '10,30,50,70,90'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=60000, help="Inference batch size.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="validate_out/prediction_npy/NeuralExpert",
        help="Output directory for exported .npy files.",
    )
    parser.add_argument("--prefix", type=str, default="pred", help="Output filename prefix.")
    return parser.parse_args()


def build_output_path(outdir: Path, prefix: str, t_idx: int) -> Path:
    return outdir / f"{prefix}_t{int(t_idx):04d}.npy"


def main() -> int:
    setup_logging()
    args = parse_args()

    run_dir = resolve_cli_path(args.run_dir)
    if run_dir is None:
        raise ValueError("--run-dir is required")
    artifact = discover_run_artifacts(run_dir)

    model_config = load_yaml(artifact.model_config_path)
    validate_cfg = load_yaml(artifact.validate_config_path)
    volume_shape = volume_shape_from_config(validate_cfg["data"]["volume_shape"])
    timesteps = select_timesteps(args.timestamp, args.timestamps, volume_shape.T)

    outdir = resolve_cli_path(args.outdir)
    if outdir is None:
        raise ValueError("Unable to resolve output directory")
    outdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, checkpoint_payload = build_model_for_run(model_config, artifact.checkpoint_path, device)
    normalize_inputs = bool(validate_cfg["data"].get("normalize_inputs", True))
    normalize_targets = bool(validate_cfg["data"].get("normalize_targets", False))
    x_mean, x_std = resolve_input_stats(volume_shape, validate_cfg, checkpoint_payload)
    y_mean, y_std = resolve_output_stats(artifact.case_name, validate_cfg, checkpoint_payload)

    logger.info("Run directory: %s", artifact.run_dir)
    logger.info("Case=%s", artifact.case_name)
    logger.info("Model config: %s", artifact.model_config_path)
    logger.info("Validate config: %s", artifact.validate_config_path)
    logger.info("Checkpoint: %s", artifact.checkpoint_path)
    logger.info("Using device=%s batch_size=%d timesteps=%s", device, int(args.batch_size), timesteps)
    logger.info("Output directory: %s", outdir)
    total_inference_seconds = 0.0
    pbar = _tqdm(total=len(timesteps), desc="predict_export", leave=True) if _tqdm is not None else None

    for t_idx in timesteps:
        pred_flat, infer_time = predict_timestep_flat_scalar(
            model=model,
            volume_shape=volume_shape,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            t_idx=int(t_idx),
            batch_size=int(args.batch_size),
            device=device,
        )
        total_inference_seconds += infer_time

        out_path = build_output_path(outdir, str(args.prefix), int(t_idx))
        np.save(str(out_path), pred_flat.astype(np.float32, copy=False))
        logger.info(
            "Saved timestep=%d shape=%s to %s (inference time: %.3f s)",
            int(t_idx),
            tuple(pred_flat.shape),
            out_path,
            infer_time,
        )

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    avg_inference_seconds = total_inference_seconds / len(timesteps) if timesteps else 0.0
    logger.info(
        "Prediction export finished. case=%s timesteps=%d voxels_per_timestep=%d total_inference_time=%.3f s avg_inference_time=%.3f s",
        artifact.case_name,
        len(timesteps),
        volume_shape.voxels_per_timestep,
        total_inference_seconds,
        avg_inference_seconds,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate_prediction.py failed")
        print(f"validate_prediction.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inr.cli import build_model, load_config, resolve_data_paths
from inr.data import MultiTargetVolumetricDataset
from inr.utils.logging_utils import setup_logging
from validate_prediction import (
    _build_dataset,
    _build_output_path,
    _predict_timestep_flat_scalar,
    _resolve_attr_name,
    _select_timesteps,
    _torch_load_checkpoint,
)

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RENDER_TASK_PATH = REPO_ROOT / "Vis" / "volume-visual" / "render_task.py"
IMAGE_VALIDATION_PATH = REPO_ROOT / "Vis" / "image_level_validation.py"
DEFAULT_RESULT_ROOT = REPO_ROOT / "Vis" / "volume-visual" / "result"
DEFAULT_TMP_ROOT = DEFAULT_RESULT_ROOT / ".tmp" / "predict_render"
DEFAULT_GT_TARGET_ROOT = REPO_ROOT / "ScientificCompressionWithINR" / "src" / "data" / "raw" / "ionization" / "train"
DEFAULT_TF_ROOT = REPO_ROOT / "Vis" / "volume-visual" / "config" / "Ionization"
CASE_ALIASES = {
    "gt": "GT",
    "pd": "PD",
    "he": "He",
    "h2": "H2",
    "h+": "H+",
    "hplus": "H+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-storage prediction -> render -> image-level validation pipeline. "
            "Predicted .npy files are rendered immediately and then cleaned up."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
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
        help="Comma-separated timesteps to export, e.g. '0,10,20,30'.",
    )
    parser.add_argument("--attr", type=str, default="", help="Target attr name for multi-target models.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device: cpu / cuda / cuda:0")
    parser.add_argument("--batch-size", type=int, default=16000, help="Inference batch size.")
    parser.add_argument("--prefix", type=str, default="pred", help="Output filename prefix.")
    parser.add_argument("--case", type=str, default="", help="Case name, e.g. GT / PD / H+ / H2 / He.")
    parser.add_argument(
        "--method-name",
        type=str,
        default="",
        help="Output subdirectory under result/<CASE>. Defaults to a normalized model-specific name.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
        help="Root directory for rendered PNGs and metrics.",
    )
    parser.add_argument(
        "--tmp-root",
        type=str,
        default=str(DEFAULT_TMP_ROOT),
        help="Temporary directory root for transient .npy files.",
    )
    parser.add_argument(
        "--gt-target-path",
        type=str,
        default="",
        help="Optional GT target .npy path. Defaults to the dataset target file for the selected case.",
    )
    parser.add_argument(
        "--gt-render-strategy",
        choices=("missing", "existing", "always"),
        default="missing",
        help="How to prepare GT PNGs: reuse existing, require existing, or always regenerate.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="Optional CSV output path. Defaults to <method_dir>/image_metrics.csv",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="",
        help="Optional summary JSON output path. Defaults to <method_dir>/image_metrics_summary.json",
    )
    parser.add_argument("--width", type=int, default=2048, help="Rendered PNG width in pixels.")
    parser.add_argument("--height", type=int, default=2048, help="Rendered PNG height in pixels.")
    parser.add_argument("--settle-frames", type=int, default=4, help="Render frames to wait before export.")
    parser.add_argument("--sample-rate", type=float, default=5.0, help="Volume ray-marching sample rate.")
    parser.add_argument("--ambient", type=float, default=0.2, help="Ambient lighting factor.")
    parser.add_argument("--diffuse", type=float, default=1.0, help="Diffuse lighting factor.")
    parser.add_argument("--specular", type=float, default=1.0, help="Specular lighting factor.")
    parser.add_argument("--shininess", type=float, default=128.0, help="Specular highlight exponent.")
    parser.add_argument("--contrast", type=float, default=0.2, help="Transfer-function contrast factor.")
    parser.add_argument(
        "--gpu-mode",
        choices=("auto", "hardware", "swiftshader"),
        default="auto",
        help="GPU backend for render_task.py.",
    )
    parser.add_argument("--browser", type=str, default="", help="Optional Chrome/Edge executable path.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Render timeout in seconds.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep transient run directory for debugging.")
    return parser.parse_args()


def resolve_cli_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def canonicalize_case_name(raw_case: str) -> str:
    value = str(raw_case).strip()
    if not value:
        return value
    normalized = value.replace("_sub", "").strip()
    alias = CASE_ALIASES.get(normalized.lower())
    return alias if alias else normalized


def infer_case_name(args_case: str, data_cfg: dict[str, Any], data_info: dict[str, Any], attr_name: str) -> str:
    if args_case.strip():
        return canonicalize_case_name(args_case)

    if data_info.get("attr_paths") and attr_name and attr_name != "targets":
        return canonicalize_case_name(attr_name)

    candidates = [
        data_cfg.get("target_path"),
        data_cfg.get("y_path"),
        data_info.get("y_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        stem = Path(str(candidate)).stem
        if stem.startswith("target_"):
            return canonicalize_case_name(stem[len("target_"):])
        return canonicalize_case_name(stem)

    raise ValueError("Unable to infer case name. Pass --case explicitly.")


def sanitize_path_component(value: str) -> str:
    sanitized = str(value).strip()
    for bad_char in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        sanitized = sanitized.replace(bad_char, "_")
    return sanitized or "run"


def infer_model_display_name(model_cfg: dict[str, Any], exp_id: str | None) -> str | None:
    text = " ".join([str(model_cfg.get("name", "")), str(exp_id or "")]).lower()
    if "basis" in text and "expert" in text:
        return "BasisExpert"
    if "moe" in text:
        return "MoE-INR"
    if "coordnet" in text or "coord_net" in text:
        return "CoordNet"
    if "siren" in text:
        return "SIREN"
    print("Warning: Unable to infer model display name from config. Using fallback naming.", file=sys.stderr)
    return None


def infer_method_name(args_method_name: str, model_cfg: dict[str, Any], cfg: dict[str, Any], case_name: str) -> str:
    if args_method_name.strip():
        return sanitize_path_component(args_method_name)

    exp_id = cfg.get("exp_id")
    display = infer_model_display_name(model_cfg, exp_id)
    if display:
        return f"{display}_Ionization_{case_name}"

    fallback = exp_id or model_cfg.get("name") or "model"
    return sanitize_path_component(str(fallback))


def resolve_gt_target_path(
    cli_value: str,
    data_info: dict[str, Any],
    attr_name: str,
    case_name: str,
) -> Path:
    cli_path = resolve_cli_path(cli_value)
    if cli_path is not None:
        return cli_path

    attr_paths = data_info.get("attr_paths") or {}
    if attr_name and attr_name in attr_paths:
        return resolve_cli_path(str(attr_paths[attr_name])) or Path(attr_paths[attr_name])

    y_path = data_info.get("y_path")
    if y_path:
        return resolve_cli_path(str(y_path)) or Path(y_path)

    return (DEFAULT_GT_TARGET_ROOT / f"target_{case_name}.npy").resolve()


def resolve_transfer_function_path(case_name: str) -> Path:
    path = DEFAULT_TF_ROOT / case_name / "transferfunction.json"
    if not path.is_file():
        raise FileNotFoundError(f"Transfer-function JSON not found for case '{case_name}': {path}")
    return path


def resolve_viewport_path(case_name: str) -> Path:
    if case_name == "PD":
        path = DEFAULT_TF_ROOT / "PD" / "viewport_PD.json"
    else:
        path = DEFAULT_TF_ROOT / "viewport.json"
    if not path.is_file():
        raise FileNotFoundError(f"Viewport JSON not found for case '{case_name}': {path}")
    return path


def load_validation_module():
    spec = importlib.util.spec_from_file_location("validate_image_level_runtime", str(IMAGE_VALIDATION_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load image validation module: {IMAGE_VALIDATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_image_pair"):
        raise AttributeError(f"{IMAGE_VALIDATION_PATH} does not define validate_image_pair().")
    return module


class GroundTruthVolumeSource:
    def __init__(self, target_path: Path, volume_shape) -> None:
        self.target_path = target_path
        self.volume_shape = volume_shape
        self._array: np.ndarray | None = None

    @property
    def voxels_per_timestep(self) -> int:
        return int(self.volume_shape.X) * int(self.volume_shape.Y) * int(self.volume_shape.Z)

    def _ensure_array(self) -> np.ndarray:
        if self._array is None:
            if not self.target_path.is_file():
                raise FileNotFoundError(f"GT target file not found: {self.target_path}")
            self._array = np.load(self.target_path, mmap_mode="r")
            if self._array.ndim in (1, 2) and int(self._array.shape[0]) != int(self.volume_shape.N):
                raise ValueError(
                    f"GT target size mismatch for {self.target_path}: "
                    f"{int(self._array.shape[0])} vs expected {int(self.volume_shape.N)}"
                )
        return self._array

    def extract_scalar_timestep(self, t_idx: int) -> np.ndarray:
        array = self._ensure_array()
        t_idx = int(t_idx)
        per_timestep = self.voxels_per_timestep
        if array.ndim == 5:
            block = np.asarray(array[t_idx], dtype=np.float32).reshape(-1, array.shape[-1])
            return self._reduce_target_block(block)
        if array.ndim == 4:
            return np.asarray(array[t_idx], dtype=np.float32).reshape(-1)
        start = t_idx * per_timestep
        end = start + per_timestep
        if array.ndim == 2:
            block = np.asarray(array[start:end], dtype=np.float32)
            return self._reduce_target_block(block)
        if array.ndim == 1:
            return np.asarray(array[start:end], dtype=np.float32).reshape(-1)
        raise ValueError(f"Unsupported GT target ndim: {array.ndim} with shape {array.shape}")

    @staticmethod
    def _reduce_target_block(block: np.ndarray) -> np.ndarray:
        if block.ndim == 1:
            return block.reshape(-1).astype(np.float32, copy=False)
        if block.shape[1] == 1:
            return block[:, 0].astype(np.float32, copy=False)
        return np.linalg.norm(block, axis=1).astype(np.float32, copy=False)

    def write_timestep_npy(self, t_idx: int, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.extract_scalar_timestep(t_idx))
        return out_path


def build_run_directory(tmp_root: Path, method_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = tmp_root / f"{sanitize_path_component(method_name)}_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cleanup_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.debug("Failed to delete temporary file: %s", path, exc_info=True)


def run_render_task(
    args: argparse.Namespace,
    volume_path: Path,
    transfer_function_path: Path,
    viewport_path: Path,
    dims_xyz: tuple[int, int, int],
) -> Path:
    command = [
        sys.executable,
        str(RENDER_TASK_PATH),
        "--volume",
        str(volume_path),
        "--transfer-function",
        str(transfer_function_path),
        "--viewport",
        str(viewport_path),
        "--dims",
        str(int(dims_xyz[0])),
        str(int(dims_xyz[1])),
        str(int(dims_xyz[2])),
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--settle-frames",
        str(int(args.settle_frames)),
        "--sample-rate",
        str(float(args.sample_rate)),
        "--ambient",
        str(float(args.ambient)),
        "--diffuse",
        str(float(args.diffuse)),
        "--specular",
        str(float(args.specular)),
        "--shininess",
        str(float(args.shininess)),
        "--contrast",
        str(float(args.contrast)),
        "--gpu-mode",
        str(args.gpu_mode),
        "--timeout",
        str(float(args.timeout)),
    ]
    browser_path = resolve_cli_path(args.browser)
    if browser_path is not None:
        command.extend(["--browser", str(browser_path)])

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "no stderr/stdout captured"
        raise RuntimeError(f"render_task.py failed for {volume_path.name}: {detail}")

    png_path = volume_path.with_suffix(".png")
    if not png_path.is_file():
        raise FileNotFoundError(f"render_task.py did not produce expected PNG: {png_path}")
    if result.stdout.strip():
        logger.debug("render_task.py output for %s: %s", volume_path.name, result.stdout.strip())
    return png_path


def ensure_ground_truth_png(
    args: argparse.Namespace,
    gt_strategy: str,
    gt_png_path: Path,
    gt_volume_source: GroundTruthVolumeSource | None,
    run_dir: Path,
    case_name: str,
    t_idx: int,
    transfer_function_path: Path,
    viewport_path: Path,
    dims_xyz: tuple[int, int, int],
) -> tuple[Path, Path | None]:
    if gt_strategy == "existing":
        if gt_png_path.is_file():
            return gt_png_path, None
        raise FileNotFoundError(f"GT PNG not found and --gt-render-strategy=existing: {gt_png_path}")

    if gt_strategy == "missing" and gt_png_path.is_file():
        return gt_png_path, None

    if gt_volume_source is None:
        raise FileNotFoundError("GT PNG generation requires a valid --gt-target-path or inferable GT target file.")

    gt_temp_npy = run_dir / "gt" / f"gt_{case_name}_t{int(t_idx):04d}.npy"
    gt_volume_source.write_timestep_npy(t_idx, gt_temp_npy)
    gt_temp_png = run_render_task(
        args=args,
        volume_path=gt_temp_npy,
        transfer_function_path=transfer_function_path,
        viewport_path=viewport_path,
        dims_xyz=dims_xyz,
    )

    gt_png_path.parent.mkdir(parents=True, exist_ok=True)
    gt_temp_png.replace(gt_png_path)
    return gt_png_path, gt_temp_npy


def write_metrics_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "case",
        "timestep",
        "pred_path",
        "gt_path",
        "ssim",
        "lpips",
        "status",
        "error",
        "inference_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "method": row.get("method", ""),
                    "case": row.get("case", ""),
                    "timestep": row.get("timestep", ""),
                    "pred_path": row.get("pred_path", ""),
                    "gt_path": row.get("gt_path", ""),
                    "ssim": "" if row.get("ssim") is None else f"{float(row['ssim']):.8f}",
                    "lpips": "" if row.get("lpips") is None else f"{float(row['lpips']):.8f}",
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                    "inference_seconds": (
                        "" if row.get("inference_seconds") is None else f"{float(row['inference_seconds']):.6f}"
                    ),
                }
            )


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary(
    results: list[dict[str, Any]],
    case_name: str,
    method_name: str,
    method_dir: Path,
    total_inference_seconds: float,
) -> dict[str, Any]:
    success_rows = [row for row in results if row.get("status") == "ok"]
    failed_rows = [row for row in results if row.get("status") != "ok"]
    return {
        "case": case_name,
        "method": method_name,
        "result_dir": str(method_dir),
        "total_timesteps": len(results),
        "successful_timesteps": [int(row["timestep"]) for row in success_rows],
        "failed_timesteps": [int(row["timestep"]) for row in failed_rows],
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "ssim": summarize_metric(success_rows, "ssim"),
        "lpips": summarize_metric(success_rows, "lpips"),
        "total_inference_seconds": float(total_inference_seconds),
        "avg_inference_seconds": float(total_inference_seconds / len(results)) if results else 0.0,
    }


def write_summary_json(summary: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def log_summary(summary: dict[str, Any]) -> None:
    logger.info(
        "Validation summary for %s/%s: success=%d failure=%d",
        summary["case"],
        summary["method"],
        int(summary["success_count"]),
        int(summary["failure_count"]),
    )
    if summary.get("ssim"):
        logger.info(
            "SSIM mean=%.6f std=%.6f min=%.6f max=%.6f",
            summary["ssim"]["mean"],
            summary["ssim"]["std"],
            summary["ssim"]["min"],
            summary["ssim"]["max"],
        )
    if summary.get("lpips"):
        logger.info(
            "LPIPS mean=%.6f std=%.6f min=%.6f max=%.6f",
            summary["lpips"]["mean"],
            summary["lpips"]["std"],
            summary["lpips"]["min"],
            summary["lpips"]["max"],
        )


def main() -> int:
    setup_logging()
    args = parse_args()

    if not RENDER_TASK_PATH.is_file():
        raise FileNotFoundError(f"render_task.py not found: {RENDER_TASK_PATH}")
    if not IMAGE_VALIDATION_PATH.is_file():
        raise FileNotFoundError(f"image_level_validation.py not found: {IMAGE_VALIDATION_PATH}")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    data_info = resolve_data_paths(data_cfg)
    dataset = _build_dataset(cfg)
    model = build_model(model_cfg, dataset)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = _torch_load_checkpoint(checkpoint_path)
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)

    inference_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(inference_device).eval()

    attr_name = _resolve_attr_name(dataset, args.attr)
    if isinstance(dataset, MultiTargetVolumetricDataset):
        out_dim = int(dataset.view_specs()[attr_name])
    else:
        out_dim = int(getattr(dataset, "_target_dim", 1))

    timesteps = _select_timesteps(args.timestamp, args.timestamps, int(dataset.volume_shape.T))
    case_name = infer_case_name(args.case, data_cfg, data_info, attr_name)
    method_name = infer_method_name(args.method_name, model_cfg, cfg, case_name)
    result_root = resolve_cli_path(args.result_root) or DEFAULT_RESULT_ROOT
    tmp_root = resolve_cli_path(args.tmp_root) or DEFAULT_TMP_ROOT
    method_dir = result_root / case_name / method_name
    gt_png_dir = result_root / case_name
    method_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = resolve_cli_path(args.metrics_csv) or (method_dir / "image_metrics.csv")
    metrics_json_path = resolve_cli_path(args.metrics_json) or (method_dir / "image_metrics_summary.json")
    transfer_function_path = resolve_transfer_function_path(case_name)
    viewport_path = resolve_viewport_path(case_name)
    gt_target_path = resolve_gt_target_path(args.gt_target_path, data_info, attr_name, case_name)
    gt_source: GroundTruthVolumeSource | None = None
    if args.gt_render_strategy in {"missing", "always"}:
        gt_source = GroundTruthVolumeSource(gt_target_path, dataset.volume_shape)

    validation_module = load_validation_module()
    validate_image_pair = validation_module.validate_image_pair
    use_lpips = getattr(validation_module, "lpips", None) is not None
    metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not use_lpips:
        logger.warning("lpips is not installed. Falling back to SSIM-only image validation.")
    dims_xyz = (
        int(dataset.volume_shape.X),
        int(dataset.volume_shape.Y),
        int(dataset.volume_shape.Z),
    )
    run_dir = build_run_directory(tmp_root, method_name)

    logger.info("Case=%s Method=%s Timesteps=%s", case_name, method_name, timesteps)
    logger.info("Prediction PNG output directory: %s", method_dir)
    logger.info("GT PNG directory: %s", gt_png_dir)
    logger.info("Temporary run directory: %s", run_dir)

    denorm_mean = torch.zeros(out_dim, device=inference_device)
    denorm_std = torch.ones(out_dim, device=inference_device)
    total_inference_seconds = 0.0
    results: list[dict[str, Any]] = []

    try:
        for t_idx in timesteps:
            pred_temp_npy = _build_output_path(run_dir / "pred", args.prefix, attr_name, int(t_idx))
            pred_temp_png = pred_temp_npy.with_suffix(".png")
            final_pred_png = method_dir / pred_temp_png.name
            gt_png_path = gt_png_dir / f"GT_{case_name}_{int(t_idx)}.png"
            gt_temp_npy: Path | None = None
            stage = "predict"
            row: dict[str, Any] = {
                "method": method_name,
                "case": case_name,
                "timestep": int(t_idx),
                "pred_path": str(final_pred_png),
                "gt_path": str(gt_png_path),
                "ssim": None,
                "lpips": None,
                "status": "pending",
                "error": "",
                "inference_seconds": None,
            }

            logger.info("Processing timestep %d", int(t_idx))
            try:
                pred_temp_npy.parent.mkdir(parents=True, exist_ok=True)
                pred_flat, infer_time = _predict_timestep_flat_scalar(
                    model=model,
                    dataset=dataset,
                    attr_name=attr_name,
                    out_dim=out_dim,
                    t_idx=int(t_idx),
                    batch_size=int(args.batch_size),
                    denorm_mean=denorm_mean,
                    denorm_std=denorm_std,
                    device=inference_device,
                )
                row["inference_seconds"] = float(infer_time)
                total_inference_seconds += infer_time
                np.save(pred_temp_npy, pred_flat)

                stage = "pred_render"
                pred_rendered_png = run_render_task(
                    args=args,
                    volume_path=pred_temp_npy,
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=dims_xyz,
                )
                final_pred_png.parent.mkdir(parents=True, exist_ok=True)
                pred_rendered_png.replace(final_pred_png)

                stage = "gt_render"
                gt_png_path, gt_temp_npy = ensure_ground_truth_png(
                    args=args,
                    gt_strategy=args.gt_render_strategy,
                    gt_png_path=gt_png_path,
                    gt_volume_source=gt_source,
                    run_dir=run_dir,
                    case_name=case_name,
                    t_idx=int(t_idx),
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=dims_xyz,
                )
                row["gt_path"] = str(gt_png_path)

                stage = "metrics"
                metrics = validate_image_pair(
                    str(gt_png_path),
                    str(final_pred_png),
                    use_lpips=use_lpips,
                    device=metrics_device,
                )
                row["ssim"] = float(metrics["ssim"])
                row["lpips"] = None if metrics["lpips"] is None else float(metrics["lpips"])
                row["status"] = "ok"
                logger.info(
                    "t=%d SSIM=%.6f%s",
                    int(t_idx),
                    row["ssim"],
                    "" if row["lpips"] is None else f" LPIPS={row['lpips']:.6f}",
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"{stage}_failed"
                row["error"] = str(exc)
                logger.exception("Failed timestep %d during %s", int(t_idx), stage)
            finally:
                if not args.keep_temp:
                    cleanup_file(pred_temp_npy)
                    cleanup_file(pred_temp_png)
                    cleanup_file(gt_temp_npy)
                    cleanup_file(None if gt_temp_npy is None else gt_temp_npy.with_suffix(".png"))
            results.append(row)
    finally:
        if not args.keep_temp:
            shutil.rmtree(run_dir, ignore_errors=True)

    write_metrics_csv(results, metrics_csv_path)
    summary = build_summary(
        results=results,
        case_name=case_name,
        method_name=method_name,
        method_dir=method_dir,
        total_inference_seconds=total_inference_seconds,
    )
    write_summary_json(summary, metrics_json_path)
    log_summary(summary)
    logger.info("Metrics CSV written to %s", metrics_csv_path)
    logger.info("Metrics summary JSON written to %s", metrics_json_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate_prediction_render.py failed")
        print(f"validate_prediction_render.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

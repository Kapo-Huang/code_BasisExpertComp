from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from compression_cli_common import (
    build_sz3_decompress_command,
    build_tthresh_decompress_command,
    build_zfp_decompress_command,
    get_sz3_dtype_args,
    run_command,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

CASE_ALIASES = {
    "gt": "GT",
    "pd": "PD",
    "he": "He",
    "h2": "H2",
    "h+": "H+",
    "hplus": "H+",
}

METHOD_DISPLAY_NAMES = {
    "sz3": "SZ3",
    "zfp": "ZFP",
    "tthresh": "TTHRESH",
}

METHOD_ALIASES = {
    "sz3": "sz3",
    "zfp": "zfp",
    "tthresh": "tthresh",
}

RESULT_JSON_PATTERN = re.compile(
    r"^target_(?P<case>.+?)_(?P<method>sz3|zfp|tthresh)_result\.json$",
    re.IGNORECASE,
)

CSV_FIELDNAMES = [
    "method",
    "case",
    "timestep",
    "pred_path",
    "gt_path",
    "psnr",
    "ssim",
    "lpips",
    "status",
    "error",
    "inference_seconds",
    "decode_seconds",
    "render_seconds",
    "metric_seconds",
    "total_pipeline_seconds",
    "compression_ratio",
    "compressed_path",
    "source_result_json",
]


@dataclass(frozen=True)
class VolumeShape:
    X: int
    Y: int
    Z: int
    T: int

    @property
    def voxels_per_timestep(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z)

    @property
    def total_values(self) -> int:
        return self.voxels_per_timestep * int(self.T)


@dataclass(frozen=True)
class CompressionArtifact:
    case_token: str
    case_name: str
    method_key: str
    method_display: str
    method_name: str
    compressed_path: Path
    result_json_path: Path
    dtype: np.dtype[Any]
    volume_shape: VolumeShape
    compression_ratio: float | None
    compressed_nbytes: int | None
    compression_time_seconds: float | None
    source_payload: dict[str, Any]


@dataclass(frozen=True)
class TimestepContext:
    timestep: int
    pred_png_path: Path
    gt_png_path: Path
    cached_row: dict[str, Any] | None
    can_reuse_complete: bool
    needs_decode: bool


@dataclass(frozen=True)
class RuntimePaths:
    artifacts_root: Path
    gt_root: Path
    result_root: Path
    tmp_root: Path
    render_task_path: Path
    image_validation_path: Path
    transfer_function_root: Path
    viewport_root: Path


def setup_logging() -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch decompression -> render -> image-level validation pipeline for "
            "precomputed compression artifacts."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        required=True,
        help="Directory containing compressed artifacts and *_result.json metadata.",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help="Directory containing target_<CASE>.npy ground-truth volumes.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Root directory for rendered PNGs and metrics.",
    )
    parser.add_argument(
        "--tmp-root",
        type=str,
        default="",
        help="Optional temporary directory root. Defaults to <result-root>/.tmp/compression_render.",
    )
    parser.add_argument(
        "--render-script",
        type=str,
        required=True,
        help="Path to render_task.py or a compatible rendering entrypoint.",
    )
    parser.add_argument(
        "--image-validation-script",
        type=str,
        required=True,
        help="Path to image_level_validation.py or a compatible image-metrics module.",
    )
    parser.add_argument(
        "--transfer-function-root",
        type=str,
        required=True,
        help="Directory containing <CASE>/transferfunction.json files.",
    )
    parser.add_argument(
        "--viewport-root",
        type=str,
        default="",
        help="Directory containing viewport.json and optional PD/viewport_PD.json. Defaults to --transfer-function-root.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Optional comma-separated case filter, e.g. 'GT,PD,H+'.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Optional comma-separated method filter, e.g. 'SZ3,ZFP'.",
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
        help="Comma-separated timesteps to export, e.g. '0,10,20,30'.",
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
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary run directories for debugging.")
    return parser.parse_args(argv)


def resolve_cli_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    artifacts_root = resolve_cli_path(args.artifacts_root)
    gt_root = resolve_cli_path(args.gt_root)
    result_root = resolve_cli_path(args.result_root)
    render_task_path = resolve_cli_path(args.render_script)
    image_validation_path = resolve_cli_path(args.image_validation_script)
    transfer_function_root = resolve_cli_path(args.transfer_function_root)

    required_paths = {
        "--artifacts-root": artifacts_root,
        "--gt-root": gt_root,
        "--result-root": result_root,
        "--render-script": render_task_path,
        "--image-validation-script": image_validation_path,
        "--transfer-function-root": transfer_function_root,
    }
    missing_options = [option_name for option_name, path in required_paths.items() if path is None]
    if missing_options:
        raise ValueError("Missing required path arguments: " + ", ".join(sorted(missing_options)))

    assert artifacts_root is not None
    assert gt_root is not None
    assert result_root is not None
    assert render_task_path is not None
    assert image_validation_path is not None
    assert transfer_function_root is not None

    viewport_root = resolve_cli_path(args.viewport_root) or transfer_function_root
    tmp_root = resolve_cli_path(args.tmp_root) or (result_root / ".tmp" / "compression_render")

    if not render_task_path.is_file():
        raise FileNotFoundError(f"--render-script not found: {render_task_path}")
    if not image_validation_path.is_file():
        raise FileNotFoundError(f"--image-validation-script not found: {image_validation_path}")
    if not transfer_function_root.is_dir():
        raise FileNotFoundError(f"--transfer-function-root not found: {transfer_function_root}")
    if not viewport_root.is_dir():
        raise FileNotFoundError(f"--viewport-root not found: {viewport_root}")

    return RuntimePaths(
        artifacts_root=artifacts_root.resolve(),
        gt_root=gt_root.resolve(),
        result_root=result_root.resolve(),
        tmp_root=tmp_root.resolve(),
        render_task_path=render_task_path.resolve(),
        image_validation_path=image_validation_path.resolve(),
        transfer_function_root=transfer_function_root.resolve(),
        viewport_root=viewport_root.resolve(),
    )


def canonicalize_case_name(raw_case: str) -> str:
    value = str(raw_case).strip()
    if not value:
        return value
    normalized = value.replace("_sub", "").strip()
    alias = CASE_ALIASES.get(normalized.lower())
    return alias if alias else normalized


def canonicalize_method_key(raw_method: str) -> str:
    value = str(raw_method).strip().lower()
    if not value:
        return value
    alias = METHOD_ALIASES.get(value)
    if alias is None:
        raise ValueError(f"Unsupported method filter: {raw_method}")
    return alias


def sanitize_path_component(value: str) -> str:
    sanitized = str(value).strip()
    for bad_char in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        sanitized = sanitized.replace(bad_char, "_")
    return sanitized or "run"


def parse_case_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {canonicalize_case_name(token) for token in raw.split(",") if token.strip()}


def parse_method_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {canonicalize_method_key(token) for token in raw.split(",") if token.strip()}


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def paths_match(cached_path: str, expected_path: Path) -> bool:
    cached_text = str(cached_path).strip()
    if not cached_text:
        return False
    try:
        return Path(cached_text).resolve() == expected_path.resolve()
    except OSError:
        return False


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


def has_required_metrics(row: dict[str, Any], require_lpips: bool = True) -> bool:
    if row.get("status") != "ok":
        return False
    if row.get("psnr") is None:
        return False
    if row.get("ssim") is None:
        return False
    if require_lpips and row.get("lpips") is None:
        return False
    return True


def get_missing_image_metrics(row: dict[str, Any], require_lpips: bool = True) -> list[str]:
    metric_names: list[str] = []
    if row.get("ssim") is None:
        metric_names.append("ssim")
    if require_lpips and row.get("lpips") is None:
        metric_names.append("lpips")
    return metric_names


def get_missing_metrics(row: dict[str, Any], require_lpips: bool = True) -> list[str]:
    metric_names: list[str] = []
    if row.get("psnr") is None:
        metric_names.append("psnr")
    metric_names.extend(get_missing_image_metrics(row, require_lpips=require_lpips))
    return metric_names


def compute_psnr(pred_flat: np.ndarray, gt_flat: np.ndarray) -> float:
    if pred_flat.shape != gt_flat.shape:
        raise ValueError(f"PSNR shape mismatch: pred={pred_flat.shape}, gt={gt_flat.shape}")

    diff = np.asarray(pred_flat, dtype=np.float64) - np.asarray(gt_flat, dtype=np.float64)
    if int(diff.size) <= 0:
        return float("nan")

    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")

    gt_min = float(np.min(gt_flat))
    gt_max = float(np.max(gt_flat))
    data_range = gt_max - gt_min
    if (not np.isfinite(data_range)) or data_range <= 0.0:
        data_range = max(abs(gt_min), abs(gt_max)) + 1e-12

    return float(10.0 * math.log10((data_range * data_range) / (mse + 1e-12)))


def select_timesteps(timestamp: int | None, timestamps: str, total_t: int) -> list[int]:
    if total_t <= 0:
        raise ValueError("Invalid volume shape: T<=0")
    if timestamps.strip():
        selected: list[int] = []
        seen: set[int] = set()
        for token in timestamps.split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value < 0 or value >= total_t:
                raise ValueError(f"timestamp out of range: {value}, valid [0, {total_t - 1}]")
            if value not in seen:
                seen.add(value)
                selected.append(value)
        if not selected:
            raise ValueError("--timestamps was provided but no valid timestep was parsed")
        return selected
    if timestamp is None:
        return list(range(total_t))
    value = int(timestamp)
    if value < 0 or value >= total_t:
        raise ValueError(f"timestamp out of range: {value}, valid [0, {total_t - 1}]")
    return [value]


def resolve_transfer_function_path(case_name: str, transfer_function_root: Path) -> Path:
    path = transfer_function_root / case_name / "transferfunction.json"
    if not path.is_file():
        raise FileNotFoundError(f"Transfer-function JSON not found for case '{case_name}': {path}")
    return path


def resolve_viewport_path(case_name: str, viewport_root: Path) -> Path:
    if case_name == "PD":
        path = viewport_root / "PD" / "viewport_PD.json"
    else:
        path = viewport_root / "viewport.json"
    if not path.is_file():
        raise FileNotFoundError(f"Viewport JSON not found for case '{case_name}': {path}")
    return path


def load_validation_module(image_validation_path: Path):
    spec = importlib.util.spec_from_file_location("validate_image_level_runtime", str(image_validation_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load image validation module: {image_validation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_image_pair"):
        raise AttributeError(f"{image_validation_path} does not define validate_image_pair().")
    if getattr(module, "lpips", None) is None:
        raise ImportError(
            "lpips is not installed in the current environment. "
            "Run this script with `conda run -n compression python ...`."
        )
    return module


class GroundTruthVolumeSource:
    def __init__(self, target_path: Path, volume_shape: VolumeShape) -> None:
        self.target_path = target_path
        self.volume_shape = volume_shape
        self._array: np.ndarray | None = None

    def _ensure_array(self) -> np.ndarray:
        if self._array is None:
            if not self.target_path.is_file():
                raise FileNotFoundError(f"GT target file not found: {self.target_path}")
            self._array = np.load(self.target_path, mmap_mode="r")
            if self._array.ndim in (1, 2) and int(self._array.shape[0]) != int(self.volume_shape.total_values):
                raise ValueError(
                    f"GT target size mismatch for {self.target_path}: "
                    f"{int(self._array.shape[0])} vs expected {int(self.volume_shape.total_values)}"
                )
        return self._array

    def extract_scalar_timestep(self, timestep: int) -> np.ndarray:
        array = self._ensure_array()
        timestep = int(timestep)
        per_timestep = self.volume_shape.voxels_per_timestep
        if array.ndim == 5:
            block = np.asarray(array[timestep], dtype=np.float32).reshape(-1, array.shape[-1])
            return self._reduce_target_block(block)
        if array.ndim == 4:
            return np.asarray(array[timestep], dtype=np.float32).reshape(-1)

        start = timestep * per_timestep
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

    def write_timestep_npy(self, timestep: int, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.extract_scalar_timestep(timestep))
        return out_path


def build_run_directory(tmp_root: Path, method_name: str, case_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = tmp_root / f"{sanitize_path_component(method_name)}_{sanitize_path_component(case_name)}_{stamp}_{os.getpid()}"
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
    render_task_path: Path,
    volume_path: Path,
    transfer_function_path: Path,
    viewport_path: Path,
    dims_xyz: tuple[int, int, int],
) -> Path:
    command = [
        sys.executable,
        str(render_task_path),
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
        cwd=str(render_task_path.parent),
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
    return png_path


def ensure_ground_truth_png(
    args: argparse.Namespace,
    gt_png_path: Path,
    gt_volume_source: GroundTruthVolumeSource,
    run_dir: Path,
    case_name: str,
    timestep: int,
    render_task_path: Path,
    transfer_function_path: Path,
    viewport_path: Path,
    dims_xyz: tuple[int, int, int],
) -> tuple[Path, Path | None, bool]:
    if gt_png_path.is_file():
        return gt_png_path, None, False

    gt_temp_npy = run_dir / "gt" / f"gt_{case_name}_t{int(timestep):04d}.npy"
    gt_volume_source.write_timestep_npy(timestep, gt_temp_npy)
    gt_temp_png = run_render_task(
        args=args,
        render_task_path=render_task_path,
        volume_path=gt_temp_npy,
        transfer_function_path=transfer_function_path,
        viewport_path=viewport_path,
        dims_xyz=dims_xyz,
    )
    gt_png_path.parent.mkdir(parents=True, exist_ok=True)
    gt_temp_png.replace(gt_png_path)
    return gt_png_path, gt_temp_npy, True


def ensure_prediction_png(
    args: argparse.Namespace,
    render_task_path: Path,
    pred_flat: np.ndarray | None,
    pred_temp_npy: Path,
    final_pred_png: Path,
    transfer_function_path: Path,
    viewport_path: Path,
    dims_xyz: tuple[int, int, int],
) -> tuple[Path, bool]:
    if final_pred_png.is_file():
        return final_pred_png, False

    if pred_flat is None:
        raise ValueError(f"Prediction render requested but no prediction array is available for {final_pred_png.name}")

    pred_temp_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_temp_npy, np.asarray(pred_flat, dtype=np.float32).reshape(-1))
    pred_temp_png = run_render_task(
        args=args,
        render_task_path=render_task_path,
        volume_path=pred_temp_npy,
        transfer_function_path=transfer_function_path,
        viewport_path=viewport_path,
        dims_xyz=dims_xyz,
    )
    final_pred_png.parent.mkdir(parents=True, exist_ok=True)
    pred_temp_png.replace(final_pred_png)
    return final_pred_png, True


def write_metrics_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "method": row.get("method", ""),
                    "case": row.get("case", ""),
                    "timestep": row.get("timestep", ""),
                    "pred_path": row.get("pred_path", ""),
                    "gt_path": row.get("gt_path", ""),
                    "psnr": "" if row.get("psnr") is None else f"{float(row['psnr']):.8f}",
                    "ssim": "" if row.get("ssim") is None else f"{float(row['ssim']):.8f}",
                    "lpips": "" if row.get("lpips") is None else f"{float(row['lpips']):.8f}",
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                    "inference_seconds": (
                        "" if row.get("inference_seconds") is None else f"{float(row['inference_seconds']):.6f}"
                    ),
                    "decode_seconds": (
                        "" if row.get("decode_seconds") is None else f"{float(row['decode_seconds']):.6f}"
                    ),
                    "render_seconds": (
                        "" if row.get("render_seconds") is None else f"{float(row['render_seconds']):.6f}"
                    ),
                    "metric_seconds": (
                        "" if row.get("metric_seconds") is None else f"{float(row['metric_seconds']):.6f}"
                    ),
                    "total_pipeline_seconds": (
                        ""
                        if row.get("total_pipeline_seconds") is None
                        else f"{float(row['total_pipeline_seconds']):.6f}"
                    ),
                    "compression_ratio": (
                        "" if row.get("compression_ratio") is None else f"{float(row['compression_ratio']):.8f}"
                    ),
                    "compressed_path": row.get("compressed_path", ""),
                    "source_result_json": row.get("source_result_json", ""),
                }
            )


def load_cached_metric_rows(csv_path: Path, method_name: str, case_name: str) -> dict[int, dict[str, Any]]:
    if not csv_path.is_file():
        return {}

    cached_rows: dict[int, dict[str, Any]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            timestep_text = str(raw_row.get("timestep", "")).strip()
            if not timestep_text:
                continue
            try:
                timestep = int(timestep_text)
            except ValueError:
                logger.warning("Skipping cached metrics row with invalid timestep '%s' in %s", timestep_text, csv_path)
                continue

            row_method = str(raw_row.get("method", "")).strip()
            row_case = str(raw_row.get("case", "")).strip()
            if row_method and row_method != method_name:
                continue
            if row_case and row_case != case_name:
                continue

            cached_rows[timestep] = {
                "method": row_method or method_name,
                "case": row_case or case_name,
                "timestep": timestep,
                "pred_path": str(raw_row.get("pred_path", "")).strip(),
                "gt_path": str(raw_row.get("gt_path", "")).strip(),
                "psnr": parse_optional_float(raw_row.get("psnr")),
                "ssim": parse_optional_float(raw_row.get("ssim")),
                "lpips": parse_optional_float(raw_row.get("lpips")),
                "status": str(raw_row.get("status", "")).strip(),
                "error": str(raw_row.get("error", "")).strip(),
                "inference_seconds": parse_optional_float(raw_row.get("inference_seconds")),
                "decode_seconds": parse_optional_float(raw_row.get("decode_seconds")),
                "render_seconds": parse_optional_float(raw_row.get("render_seconds")),
                "metric_seconds": parse_optional_float(raw_row.get("metric_seconds")),
                "total_pipeline_seconds": parse_optional_float(raw_row.get("total_pipeline_seconds")),
                "compression_ratio": parse_optional_float(raw_row.get("compression_ratio")),
                "compressed_path": str(raw_row.get("compressed_path", "")).strip(),
                "source_result_json": str(raw_row.get("source_result_json", "")).strip(),
                "cache_hit": True,
            }
    return cached_rows


def cached_row_matches_context(
    cached_row: dict[str, Any] | None,
    method_name: str,
    case_name: str,
    pred_png_path: Path,
    gt_png_path: Path,
) -> bool:
    if not cached_row:
        return False
    if cached_row.get("method") != method_name:
        return False
    if cached_row.get("case") != case_name:
        return False
    if not paths_match(str(cached_row.get("pred_path", "")), pred_png_path):
        return False
    if not paths_match(str(cached_row.get("gt_path", "")), gt_png_path):
        return False
    return True


def resolve_local_compressed_path(
    compressed_value: str | None,
    artifacts_root: Path,
    case_token: str,
    method_key: str,
) -> Path:
    candidates: list[Path] = []
    if compressed_value:
        candidates.append((artifacts_root / Path(str(compressed_value)).name).resolve())
    suffix = {
        "sz3": ".sz3pkg",
        "zfp": ".zfp",
        "tthresh": ".tthresh",
    }[method_key]
    candidates.append((artifacts_root / f"target_{case_token}{suffix}").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Unable to resolve local compressed artifact for case '{case_token}' method '{method_key}'. "
        f"Searched: {searched}"
    )


def parse_artifact(result_json_path: Path, artifacts_root: Path) -> CompressionArtifact:
    match = RESULT_JSON_PATTERN.match(result_json_path.name)
    if match is None:
        raise ValueError(f"Unsupported artifact result filename: {result_json_path.name}")

    payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    case_token = str(match.group("case")).strip()
    case_name = canonicalize_case_name(case_token)
    method_key = canonicalize_method_key(match.group("method"))
    method_display = METHOD_DISPLAY_NAMES[method_key]
    method_name = f"{method_display}_Ionization_{case_name}"

    dtype_value = payload.get("dtype")
    if not dtype_value:
        raise ValueError(f"Missing dtype in {result_json_path}")
    dtype = np.dtype(str(dtype_value))

    used_shape_raw = payload.get("used_shape")
    if not isinstance(used_shape_raw, list) or len(used_shape_raw) != 4:
        raise ValueError(f"Expected 4D used_shape in {result_json_path}, got {used_shape_raw!r}")
    shape = tuple(int(value) for value in used_shape_raw)
    volume_shape = VolumeShape(X=shape[0], Y=shape[1], Z=shape[2], T=shape[3])

    compressed_path = resolve_local_compressed_path(
        compressed_value=str(payload.get("compressed", "")).strip(),
        artifacts_root=artifacts_root,
        case_token=case_token,
        method_key=method_key,
    )

    compressed_nbytes = None
    if compressed_path.is_file():
        compressed_nbytes = int(compressed_path.stat().st_size)

    compression_ratio = parse_optional_float(payload.get("compression_ratio"))
    compression_time_seconds = parse_optional_float(payload.get("compression_time_seconds"))
    return CompressionArtifact(
        case_token=case_token,
        case_name=case_name,
        method_key=method_key,
        method_display=method_display,
        method_name=method_name,
        compressed_path=compressed_path,
        result_json_path=result_json_path.resolve(),
        dtype=dtype,
        volume_shape=volume_shape,
        compression_ratio=compression_ratio,
        compressed_nbytes=compressed_nbytes,
        compression_time_seconds=compression_time_seconds,
        source_payload=payload,
    )


def discover_artifacts(
    artifacts_root: Path,
    case_filter: set[str],
    method_filter: set[str],
) -> list[CompressionArtifact]:
    if not artifacts_root.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_root}")

    artifacts: list[CompressionArtifact] = []
    for result_json_path in sorted(artifacts_root.glob("*_result.json")):
        artifact = parse_artifact(result_json_path.resolve(), artifacts_root.resolve())
        if case_filter and artifact.case_name not in case_filter:
            continue
        if method_filter and artifact.method_key not in method_filter:
            continue
        artifacts.append(artifact)

    if not artifacts:
        raise FileNotFoundError(
            f"No matching artifacts found in {artifacts_root} for cases={sorted(case_filter)} methods={sorted(method_filter)}"
        )
    return artifacts


def resolve_binary_path(method_key: str) -> Path:
    candidates = {
        "sz3": [
            SCRIPT_DIR / "SZ3" / "build" / "tools" / "sz3" / "sz3.exe",
            SCRIPT_DIR / "SZ3" / "build" / "tools" / "sz3" / "sz3",
        ],
        "zfp": [
            SCRIPT_DIR / "zfp" / "build" / "bin" / "zfp.exe",
            SCRIPT_DIR / "zfp" / "build" / "bin" / "zfp",
        ],
        "tthresh": [
            SCRIPT_DIR / "tthresh" / "build" / "tthresh.exe",
            SCRIPT_DIR / "tthresh" / "build" / "tthresh",
        ],
    }[method_key]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Unable to locate binary for method '{method_key}'. Searched: {searched}")


def unpack_sz3_payload(package_path: Path, unpack_dir: Path) -> Path:
    with zipfile.ZipFile(package_path, "r") as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        meta_payload_name: str | None = None
        if "meta.json" in names:
            meta = json.loads(archive.read("meta.json").decode("utf-8"))
            raw_payload_name = str(meta.get("payload", "")).strip()
            if raw_payload_name:
                meta_payload_name = raw_payload_name
        payload_name = meta_payload_name
        if not payload_name or payload_name not in names:
            payload_candidates = [name for name in names if Path(name).name != "meta.json"]
            if len(payload_candidates) != 1:
                raise ValueError(f"Unable to determine SZ3 payload in package: {package_path}")
            payload_name = payload_candidates[0]
        archive.extract(payload_name, path=unpack_dir)
        return (unpack_dir / payload_name).resolve()


def decompress_artifact(
    artifact: CompressionArtifact,
    run_dir: Path,
) -> tuple[Path, float]:
    raw_output_path = run_dir / f"{artifact.method_key}_{sanitize_path_component(artifact.case_token)}.raw"
    binary_path = resolve_binary_path(artifact.method_key)

    if artifact.method_key == "sz3":
        payload_dir = run_dir / "sz3_payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = unpack_sz3_payload(artifact.compressed_path, payload_dir)
        dtype_args, _ = get_sz3_dtype_args(artifact.dtype)
        command = build_sz3_decompress_command(
            binary_path=binary_path,
            compressed_path=payload_path,
            raw_output_path=raw_output_path,
            dtype_args=dtype_args,
            shape=(
                int(artifact.volume_shape.X),
                int(artifact.volume_shape.Y),
                int(artifact.volume_shape.Z),
                int(artifact.volume_shape.T),
            ),
        )
    elif artifact.method_key == "zfp":
        command = build_zfp_decompress_command(
            binary_path=binary_path,
            compressed_path=artifact.compressed_path,
            raw_output_path=raw_output_path,
        )
    else:
        command = build_tthresh_decompress_command(
            binary_path=binary_path,
            compressed_path=artifact.compressed_path,
            raw_output_path=raw_output_path,
        )

    result = run_command(command)
    return raw_output_path.resolve(), float(result.elapsed_seconds)


def open_raw_memmap(raw_output_path: Path, artifact: CompressionArtifact) -> np.memmap:
    expected_size = int(artifact.volume_shape.total_values) * int(artifact.dtype.itemsize)
    actual_size = int(raw_output_path.stat().st_size)
    if actual_size != expected_size:
        raise ValueError(
            f"Raw decompression size mismatch for {artifact.compressed_path.name}: "
            f"expected {expected_size} bytes, got {actual_size}"
        )
    return np.memmap(
        raw_output_path,
        dtype=artifact.dtype,
        mode="r",
        shape=(int(artifact.volume_shape.total_values),),
    )


def close_memmap(raw_memmap: np.memmap | None) -> None:
    if raw_memmap is None:
        return
    mmap_obj = getattr(raw_memmap, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def extract_flat_timestep_from_memmap(
    raw_memmap: np.memmap,
    volume_shape: VolumeShape,
    timestep: int,
) -> np.ndarray:
    timestep = int(timestep)
    start = timestep * int(volume_shape.voxels_per_timestep)
    end = start + int(volume_shape.voxels_per_timestep)
    return np.array(raw_memmap[start:end], dtype=np.float32, copy=True).reshape(-1)


def build_summary(
    results: list[dict[str, Any]],
    artifact: CompressionArtifact,
    method_dir: Path,
    decode_seconds: float,
) -> dict[str, Any]:
    success_rows = [row for row in results if row.get("status") == "ok"]
    failed_rows = [row for row in results if row.get("status") != "ok"]
    executed_rows = [row for row in results if row.get("inference_seconds") is not None and not row.get("cache_hit", False)]
    cache_hits = sum(1 for row in results if row.get("cache_hit", False))

    total_inference_seconds = float(
        sum(float(row.get("inference_seconds") or 0.0) for row in executed_rows)
    )
    total_render_seconds = float(sum(float(row.get("render_seconds") or 0.0) for row in executed_rows))
    total_metric_seconds = float(sum(float(row.get("metric_seconds") or 0.0) for row in executed_rows))
    total_pipeline_seconds = float(decode_seconds + total_inference_seconds)
    executed_count = int(len(executed_rows))

    return {
        "case": artifact.case_name,
        "method": artifact.method_name,
        "result_dir": str(method_dir),
        "total_timesteps": len(results),
        "successful_timesteps": [int(row["timestep"]) for row in success_rows],
        "failed_timesteps": [int(row["timestep"]) for row in failed_rows],
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "psnr": summarize_metric(success_rows, "psnr"),
        "ssim": summarize_metric(success_rows, "ssim"),
        "lpips": summarize_metric(success_rows, "lpips"),
        "cache_hit_count": int(cache_hits),
        "executed_inference_count": executed_count,
        "total_inference_seconds": total_inference_seconds,
        "avg_inference_seconds": float(total_inference_seconds / executed_count) if executed_count else 0.0,
        "compression": {
            "compressed_path": str(artifact.compressed_path),
            "source_result_json": str(artifact.result_json_path),
            "compression_ratio": artifact.compression_ratio,
            "compressed_nbytes": artifact.compressed_nbytes,
            "compression_time_seconds": artifact.compression_time_seconds,
        },
        "runtime": {
            "decode_seconds": float(decode_seconds),
            "render_seconds_total": total_render_seconds,
            "metric_seconds_total": total_metric_seconds,
            "total_pipeline_seconds": total_pipeline_seconds,
            "avg_render_seconds": float(total_render_seconds / executed_count) if executed_count else 0.0,
            "avg_metric_seconds": float(total_metric_seconds / executed_count) if executed_count else 0.0,
            "avg_total_pipeline_seconds": float(total_pipeline_seconds / executed_count) if executed_count else 0.0,
        },
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
    if summary.get("psnr"):
        logger.info(
            "PSNR mean=%.6f std=%.6f min=%.6f max=%.6f",
            summary["psnr"]["mean"],
            summary["psnr"]["std"],
            summary["psnr"]["min"],
            summary["psnr"]["max"],
        )
    if summary.get("lpips"):
        logger.info(
            "LPIPS mean=%.6f std=%.6f min=%.6f max=%.6f",
            summary["lpips"]["mean"],
            summary["lpips"]["std"],
            summary["lpips"]["min"],
            summary["lpips"]["max"],
        )
    runtime = summary.get("runtime") or {}
    logger.info(
        "Runtime decode=%.6fs render_total=%.6fs metric_total=%.6fs total_pipeline=%.6fs",
        float(runtime.get("decode_seconds") or 0.0),
        float(runtime.get("render_seconds_total") or 0.0),
        float(runtime.get("metric_seconds_total") or 0.0),
        float(runtime.get("total_pipeline_seconds") or 0.0),
    )


def build_row_template(
    artifact: CompressionArtifact,
    timestep: int,
    pred_png_path: Path,
    gt_png_path: Path,
) -> dict[str, Any]:
    return {
        "method": artifact.method_name,
        "case": artifact.case_name,
        "timestep": int(timestep),
        "pred_path": str(pred_png_path),
        "gt_path": str(gt_png_path),
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "status": "pending",
        "error": "",
        "inference_seconds": None,
        "decode_seconds": None,
        "render_seconds": None,
        "metric_seconds": None,
        "total_pipeline_seconds": None,
        "compression_ratio": artifact.compression_ratio,
        "compressed_path": str(artifact.compressed_path),
        "source_result_json": str(artifact.result_json_path),
        "cache_hit": False,
    }


def build_timestep_contexts(
    artifact: CompressionArtifact,
    timesteps: list[int],
    method_dir: Path,
    gt_png_dir: Path,
    cached_metric_rows: dict[int, dict[str, Any]],
) -> list[TimestepContext]:
    contexts: list[TimestepContext] = []
    for timestep in timesteps:
        pred_png_path = method_dir / f"pred_t{int(timestep):04d}.png"
        gt_png_path = gt_png_dir / f"GT_{artifact.case_name}_{int(timestep)}.png"
        raw_cached_row = cached_metric_rows.get(int(timestep))
        cache_context_matches = cached_row_matches_context(
            cached_row=raw_cached_row,
            method_name=artifact.method_name,
            case_name=artifact.case_name,
            pred_png_path=pred_png_path,
            gt_png_path=gt_png_path,
        )
        cached_row = raw_cached_row if cache_context_matches else None
        can_reuse_complete = (
            cache_context_matches
            and cached_row is not None
            and has_required_metrics(cached_row, require_lpips=True)
            and pred_png_path.is_file()
            and gt_png_path.is_file()
        )
        needs_decode = not can_reuse_complete and (
            pred_png_path.is_file() is False
            or cached_row is None
            or cached_row.get("psnr") is None
        )
        contexts.append(
            TimestepContext(
                timestep=int(timestep),
                pred_png_path=pred_png_path,
                gt_png_path=gt_png_path,
                cached_row=cached_row,
                can_reuse_complete=bool(can_reuse_complete),
                needs_decode=bool(needs_decode),
            )
        )
    return contexts


def process_artifact(
    artifact: CompressionArtifact,
    args: argparse.Namespace,
    validation_module: Any,
    runtime_paths: RuntimePaths,
) -> list[dict[str, Any]]:
    gt_target_path = (runtime_paths.gt_root / f"target_{artifact.case_name}.npy").resolve()
    gt_volume_source = GroundTruthVolumeSource(gt_target_path, artifact.volume_shape)
    transfer_function_path = resolve_transfer_function_path(
        artifact.case_name,
        runtime_paths.transfer_function_root,
    )
    viewport_path = resolve_viewport_path(artifact.case_name, runtime_paths.viewport_root)
    dims_xyz = (
        int(artifact.volume_shape.X),
        int(artifact.volume_shape.Y),
        int(artifact.volume_shape.Z),
    )
    method_dir = (runtime_paths.result_root / artifact.case_name / artifact.method_name).resolve()
    gt_png_dir = (runtime_paths.result_root / artifact.case_name).resolve()
    method_dir.mkdir(parents=True, exist_ok=True)
    runtime_paths.tmp_root.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = method_dir / "image_metrics.csv"
    metrics_json_path = method_dir / "image_metrics_summary.json"
    timesteps = select_timesteps(args.timestamp, args.timestamps, int(artifact.volume_shape.T))
    validate_image_pair = validation_module.validate_image_pair
    torch_module = getattr(validation_module, "torch", None)
    metrics_device = "cuda" if torch_module is not None and torch_module.cuda.is_available() else "cpu"

    cached_metric_rows = load_cached_metric_rows(metrics_csv_path, artifact.method_name, artifact.case_name)
    if cached_metric_rows:
        logger.info(
            "[%s/%s] Loaded %d cached rows from %s",
            artifact.case_name,
            artifact.method_display,
            len(cached_metric_rows),
            metrics_csv_path,
        )

    contexts = build_timestep_contexts(
        artifact=artifact,
        timesteps=timesteps,
        method_dir=method_dir,
        gt_png_dir=gt_png_dir,
        cached_metric_rows=cached_metric_rows,
    )

    decode_needed = any(context.needs_decode for context in contexts)
    run_dir: Path | None = None
    raw_output_path: Path | None = None
    raw_memmap: np.memmap | None = None
    decode_seconds = 0.0
    decode_error: Exception | None = None
    decode_recorded = False
    results: list[dict[str, Any]] = []

    if decode_needed:
        run_dir = build_run_directory(runtime_paths.tmp_root, artifact.method_name, artifact.case_name)
        logger.info(
            "[%s/%s] Decompressing %s",
            artifact.case_name,
            artifact.method_display,
            artifact.compressed_path.name,
        )
        try:
            raw_output_path, decode_seconds = decompress_artifact(artifact, run_dir)
            raw_memmap = open_raw_memmap(raw_output_path, artifact)
        except Exception as exc:  # noqa: BLE001
            decode_error = exc
            logger.exception(
                "[%s/%s] Failed to decompress %s",
                artifact.case_name,
                artifact.method_display,
                artifact.compressed_path.name,
            )

    try:
        for context in contexts:
            row = build_row_template(
                artifact=artifact,
                timestep=context.timestep,
                pred_png_path=context.pred_png_path,
                gt_png_path=context.gt_png_path,
            )
            if context.cached_row is not None:
                row.update(
                    {
                        "psnr": context.cached_row.get("psnr"),
                        "ssim": context.cached_row.get("ssim"),
                        "lpips": context.cached_row.get("lpips"),
                        "inference_seconds": context.cached_row.get("inference_seconds"),
                        "decode_seconds": context.cached_row.get("decode_seconds"),
                        "render_seconds": context.cached_row.get("render_seconds"),
                        "metric_seconds": context.cached_row.get("metric_seconds"),
                        "total_pipeline_seconds": context.cached_row.get("total_pipeline_seconds"),
                    }
                )

            if context.can_reuse_complete:
                row["status"] = "ok"
                row["error"] = ""
                row["cache_hit"] = True
                results.append(row)
                logger.info(
                    "[%s/%s] Skipping timestep %d because cached metrics and PNGs already exist",
                    artifact.case_name,
                    artifact.method_display,
                    int(context.timestep),
                )
                continue

            if context.needs_decode and decode_error is not None:
                row["status"] = "decode_failed"
                row["error"] = str(decode_error)
                results.append(row)
                continue

            stage = "prepare"
            gt_temp_npy: Path | None = None
            pred_temp_npy: Path | None = None
            pred_temp_png: Path | None = None
            timestep_start = time.perf_counter()
            render_seconds = 0.0
            metric_seconds = 0.0

            try:
                pred_flat: np.ndarray | None = None
                if context.needs_decode:
                    if raw_memmap is None:
                        raise RuntimeError("Decoded artifact buffer is not available")
                    stage = "psnr"
                    pred_flat = extract_flat_timestep_from_memmap(raw_memmap, artifact.volume_shape, context.timestep)
                    gt_flat = gt_volume_source.extract_scalar_timestep(context.timestep)
                    row["psnr"] = compute_psnr(pred_flat, gt_flat)
                elif row["psnr"] is None:
                    raise RuntimeError(
                        f"PSNR is unavailable for timestep {int(context.timestep)} without decoding. "
                        "Delete the stale CSV row or allow the artifact to be decoded."
                    )

                stage = "pred_render"
                pred_temp_npy = (run_dir or runtime_paths.tmp_root) / "pred" / f"pred_t{int(context.timestep):04d}.npy"
                pred_temp_png = pred_temp_npy.with_suffix(".png")
                render_start = time.perf_counter()
                _, did_render_pred = ensure_prediction_png(
                    args=args,
                    render_task_path=runtime_paths.render_task_path,
                    pred_flat=pred_flat,
                    pred_temp_npy=pred_temp_npy,
                    final_pred_png=context.pred_png_path,
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=dims_xyz,
                )
                if did_render_pred:
                    render_seconds += time.perf_counter() - render_start

                stage = "gt_render"
                render_start = time.perf_counter()
                gt_png_path, gt_temp_npy, did_render_gt = ensure_ground_truth_png(
                    args=args,
                    gt_png_path=context.gt_png_path,
                    gt_volume_source=gt_volume_source,
                    run_dir=run_dir or runtime_paths.tmp_root,
                    case_name=artifact.case_name,
                    timestep=context.timestep,
                    render_task_path=runtime_paths.render_task_path,
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=dims_xyz,
                )
                row["gt_path"] = str(gt_png_path)
                if did_render_gt:
                    render_seconds += time.perf_counter() - render_start

                image_metrics_to_compute = get_missing_image_metrics(row, require_lpips=True)
                if image_metrics_to_compute:
                    stage = "metrics"
                    metric_start = time.perf_counter()
                    metrics = validate_image_pair(
                        str(gt_png_path),
                        str(context.pred_png_path),
                        use_lpips=True,
                        device=metrics_device,
                        requested_metrics=image_metrics_to_compute,
                    )
                    metric_seconds += time.perf_counter() - metric_start
                    if "ssim" in image_metrics_to_compute and metrics["ssim"] is not None:
                        row["ssim"] = float(metrics["ssim"])
                    if "lpips" in image_metrics_to_compute:
                        row["lpips"] = None if metrics["lpips"] is None else float(metrics["lpips"])

                if not has_required_metrics({**row, "status": "ok"}, require_lpips=True):
                    raise RuntimeError(
                        "Missing metrics after processing: " + ", ".join(get_missing_metrics(row, require_lpips=True))
                    )

                row_elapsed = time.perf_counter() - timestep_start
                row["status"] = "ok"
                row["error"] = ""
                row["render_seconds"] = float(render_seconds)
                row["metric_seconds"] = float(metric_seconds)
                row["inference_seconds"] = float(row_elapsed)
                if decode_needed and not decode_recorded:
                    row["decode_seconds"] = float(decode_seconds)
                    decode_recorded = True
                else:
                    row["decode_seconds"] = 0.0
                row["total_pipeline_seconds"] = float(row_elapsed + float(row["decode_seconds"] or 0.0))
                logger.info(
                    "[%s/%s] t=%d PSNR=%.6f SSIM=%.6f LPIPS=%.6f",
                    artifact.case_name,
                    artifact.method_display,
                    int(context.timestep),
                    float(row["psnr"]),
                    float(row["ssim"]),
                    float(row["lpips"]),
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"{stage}_failed"
                row["error"] = str(exc)
                if row["render_seconds"] is None:
                    row["render_seconds"] = float(render_seconds)
                if row["metric_seconds"] is None:
                    row["metric_seconds"] = float(metric_seconds)
                if row["inference_seconds"] is None:
                    row["inference_seconds"] = float(time.perf_counter() - timestep_start)
                if decode_needed and not decode_recorded:
                    row["decode_seconds"] = float(decode_seconds)
                    decode_recorded = True
                else:
                    row["decode_seconds"] = 0.0
                row["total_pipeline_seconds"] = float(
                    float(row.get("inference_seconds") or 0.0) + float(row.get("decode_seconds") or 0.0)
                )
                logger.exception(
                    "[%s/%s] Failed timestep %d during %s",
                    artifact.case_name,
                    artifact.method_display,
                    int(context.timestep),
                    stage,
                )
            finally:
                if not args.keep_temp:
                    cleanup_file(pred_temp_npy)
                    cleanup_file(pred_temp_png)
                    cleanup_file(gt_temp_npy)
                    cleanup_file(None if gt_temp_npy is None else gt_temp_npy.with_suffix(".png"))
            results.append(row)
    finally:
        close_memmap(raw_memmap)
        if not args.keep_temp and run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)

    write_metrics_csv(results, metrics_csv_path)
    summary = build_summary(results=results, artifact=artifact, method_dir=method_dir, decode_seconds=decode_seconds)
    write_summary_json(summary, metrics_json_path)
    log_summary(summary)
    logger.info("[%s/%s] Metrics CSV written to %s", artifact.case_name, artifact.method_display, metrics_csv_path)
    logger.info(
        "[%s/%s] Metrics summary JSON written to %s",
        artifact.case_name,
        artifact.method_display,
        metrics_json_path,
    )
    return results


def main() -> int:
    setup_logging()
    args = parse_args()
    runtime_paths = resolve_runtime_paths(args)
    case_filter = parse_case_filter(args.cases)
    method_filter = parse_method_filter(args.methods)

    validation_module = load_validation_module(runtime_paths.image_validation_path)
    artifacts = discover_artifacts(
        artifacts_root=runtime_paths.artifacts_root,
        case_filter=case_filter,
        method_filter=method_filter,
    )

    logger.info("Discovered %d artifacts under %s", len(artifacts), runtime_paths.artifacts_root)
    for artifact in artifacts:
        logger.info(
            "Processing artifact case=%s method=%s compressed=%s",
            artifact.case_name,
            artifact.method_display,
            artifact.compressed_path.name,
        )
        process_artifact(
            artifact=artifact,
            args=args,
            validation_module=validation_module,
            runtime_paths=runtime_paths,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate_compression_render.py failed")
        print(f"validate_compression_render.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

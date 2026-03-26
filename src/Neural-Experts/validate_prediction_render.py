from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
INR_ROOT = PROJECT_ROOT.parent

for path in (str(THIS_DIR), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models import build_model

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
RENDER_TASK_PATH = INR_ROOT / "Vis" / "volume-visual" / "render_task.py"
IMAGE_VALIDATION_PATH = INR_ROOT / "Vis" / "image_level_validation.py"
DEFAULT_RUNS_ROOT = SRC_ROOT / "result" / "Ionization" / "Neural Expert"
DEFAULT_RESULT_ROOT = INR_ROOT / "Vis" / "volume-visual" / "result"
DEFAULT_TMP_ROOT = DEFAULT_RESULT_ROOT / ".tmp" / "neural_expert_predict_render"
DEFAULT_TF_ROOT = INR_ROOT / "Vis" / "volume-visual" / "config" / "Ionization"
CASE_ALIASES = {
    "gt": "GT",
    "pd": "PD",
    "he": "He",
    "h2": "H2",
    "h+": "H+",
    "hplus": "H+",
}


@dataclass(frozen=True)
class VolumeShape:
    X: int
    Y: int
    Z: int
    T: int

    @property
    def N(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z) * int(self.T)

    @property
    def voxels_per_timestep(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z)

    @property
    def dims_xyz(self) -> tuple[int, int, int]:
        return int(self.X), int(self.Y), int(self.Z)


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    case_name: str
    model_config_path: Path
    validate_config_path: Path
    checkpoint_path: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Neural-Experts ionization validation pipeline: predict -> render -> "
            "compute metrics, then immediately clean up temporary prediction volumes."
        )
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Single Neural-Experts run directory, e.g. result/Ionization/Neural Expert/H+",
    )
    source_group.add_argument(
        "--runs-root",
        type=str,
        default="",
        help="Root directory that contains case subdirectories. Defaults to result/Ionization/Neural Expert.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Optional comma-separated case filter when using --runs-root, e.g. 'GT,H+,PD'.",
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
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=60000, help="Inference batch size.")
    parser.add_argument("--prefix", type=str, default="pred", help="Temporary/output filename prefix.")
    parser.add_argument(
        "--method-name",
        type=str,
        default="NeuralExpert_Ionization_{case}",
        help="Output folder name template under result/<CASE>. Supports {case}.",
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
        help="Optional GT target .npy path. Only supported in single-run mode.",
    )
    parser.add_argument(
        "--gt-render-strategy",
        choices=("missing", "existing", "always"),
        default="missing",
        help="How to prepare GT PNGs: reuse existing, require existing, or always regenerate.",
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


def resolve_src_relative(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (SRC_ROOT / path).resolve()


def canonicalize_case_name(raw_case: str) -> str:
    value = str(raw_case).strip()
    if not value:
        return value
    normalized = value.replace("_sub", "").strip()
    alias = CASE_ALIASES.get(normalized.lower())
    return alias if alias else normalized


def parse_case_filter(raw_cases: str) -> set[str]:
    if not raw_cases.strip():
        return set()
    return {canonicalize_case_name(token) for token in raw_cases.split(",") if token.strip()}


def sanitize_path_component(value: str) -> str:
    sanitized = str(value).strip()
    for bad_char in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        sanitized = sanitized.replace(bad_char, "_")
    return sanitized or "run"


def build_method_name(template: str, case_name: str) -> str:
    rendered = str(template).format(case=case_name)
    return sanitize_path_component(rendered)


def _register_numpy_core_aliases() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias_name, target_name in aliases.items():
        if alias_name in sys.modules:
            continue
        sys.modules[alias_name] = importlib.import_module(target_name)


def torch_load_checkpoint(path: Path):
    def _load():
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    try:
        return _load()
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy._core", "numpy._core.multiarray"}:
            raise
        _register_numpy_core_aliases()
        return _load()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def volume_shape_from_config(shape_cfg: dict[str, Any]) -> VolumeShape:
    return VolumeShape(
        X=int(shape_cfg["X"]),
        Y=int(shape_cfg["Y"]),
        Z=int(shape_cfg["Z"]),
        T=int(shape_cfg["T"]),
    )


def compute_input_stats(volume_shape: VolumeShape) -> tuple[np.ndarray, np.ndarray]:
    n_total = float(volume_shape.N)
    correction = (n_total / (n_total - 1.0)) if n_total > 1.0 else 1.0

    def _mean(size: int) -> float:
        return (float(size) - 1.0) / 2.0

    def _var(size: int) -> float:
        if int(size) <= 1:
            return 0.0
        return (float(size) * float(size) - 1.0) / 12.0

    means = np.array(
        [
            _mean(volume_shape.X),
            _mean(volume_shape.Y),
            _mean(volume_shape.Z),
            _mean(volume_shape.T),
        ],
        dtype=np.float32,
    )
    stds = np.sqrt(
        np.maximum(
            np.array(
                [
                    _var(volume_shape.X),
                    _var(volume_shape.Y),
                    _var(volume_shape.Z),
                    _var(volume_shape.T),
                ],
                dtype=np.float64,
            )
            * correction,
            1.0e-12,
        )
    ).astype(np.float32)
    return means, stds


def compute_target_stats(arr: np.ndarray, chunk_size: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(-1, arr.shape[-1] if arr.ndim == 2 else 1)
    count = 0
    sum_1 = np.zeros((flat.shape[-1],), dtype=np.float64)
    sum_2 = np.zeros((flat.shape[-1],), dtype=np.float64)

    for start in range(0, flat.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), flat.shape[0])
        block = np.asarray(flat[start:end], dtype=np.float64)
        count += block.shape[0]
        sum_1 += block.sum(axis=0)
        sum_2 += (block * block).sum(axis=0)

    if count <= 1:
        mean = sum_1 / max(count, 1)
        var = np.zeros_like(mean)
    else:
        mean = sum_1 / count
        var = (sum_2 - (sum_1 * sum_1) / count) / (count - 1)
        var = np.maximum(var, 0.0)

    std = np.sqrt(np.maximum(var, 1.0e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def load_target_stats(stats_path: Path, attr_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(stats_path), allow_pickle=True)
    if "mean" in data and "std" in data:
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(data["std"], dtype=np.float32).reshape(-1)
        return mean, np.maximum(std, 1.0e-12)

    attr_mean_key = f"{attr_name}__mean"
    attr_std_key = f"{attr_name}__std"
    if attr_mean_key in data and attr_std_key in data:
        mean = np.asarray(data[attr_mean_key], dtype=np.float32).reshape(-1)
        std = np.asarray(data[attr_std_key], dtype=np.float32).reshape(-1)
        return mean, np.maximum(std, 1.0e-12)

    raise KeyError(f"Unable to find target stats for case '{attr_name}' in {stats_path}")


def resolve_device(requested: str) -> torch.device:
    if requested.strip():
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{requested}' but CUDA is not available.")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
            t_idx = int(token)
            if t_idx < 0 or t_idx >= total_t:
                raise ValueError(f"timestamp out of range: {t_idx}, valid [0, {total_t - 1}]")
            if t_idx not in seen:
                seen.add(t_idx)
                selected.append(t_idx)
        if not selected:
            raise ValueError("--timestamps was provided but no valid timestep was parsed")
        return selected

    if timestamp is None:
        return list(range(total_t))
    t_idx = int(timestamp)
    if t_idx < 0 or t_idx >= total_t:
        raise ValueError(f"timestamp out of range: {t_idx}, valid [0, {total_t - 1}]")
    return [t_idx]


def build_output_path(outdir: Path, prefix: str, t_idx: int) -> Path:
    return outdir / f"{prefix}_t{int(t_idx):04d}.npy"


def infer_case_name(model_cfg: dict[str, Any], validate_cfg: dict[str, Any], run_dir: Path) -> str:
    candidates = [
        model_cfg.get("DATA", {}).get("attr_name"),
        validate_cfg.get("data", {}).get("target_path"),
        run_dir.name,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        stem = Path(text).stem
        if stem.startswith("target_"):
            stem = stem[len("target_") :]
        return canonicalize_case_name(stem)
    raise ValueError(f"Unable to infer case name from {run_dir}")


def find_run_checkpoint(run_dir: Path) -> Path | None:
    preferred = sorted(run_dir.glob("*_model_final.pth"))
    if preferred:
        return preferred[0]

    trained_models_dir = run_dir / "trained_models"
    preferred = sorted(trained_models_dir.glob("*_model_final.pth"))
    if preferred:
        return preferred[0]

    candidates = sorted(p for p in run_dir.glob("*.pth") if p.is_file())
    if candidates:
        return candidates[0]

    candidates = sorted(trained_models_dir.glob("*.pth"))
    if candidates:
        return candidates[-1]
    return None


def discover_run_artifacts(run_dir: Path) -> RunArtifacts:
    run_dir = run_dir.resolve()
    model_config_path = run_dir / "config.yaml"
    validate_dir = run_dir / "validate_artifacts"
    validate_config_path = validate_dir / "config.yaml"

    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not validate_config_path.is_file():
        raise FileNotFoundError(f"Validate config not found: {validate_config_path}")

    validate_ckpts = sorted(validate_dir.glob("*.pth"))
    checkpoint_path = validate_ckpts[0] if validate_ckpts else find_run_checkpoint(run_dir)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Unable to locate checkpoint under {run_dir}")

    model_cfg = load_yaml(model_config_path)
    validate_cfg = load_yaml(validate_config_path)
    case_name = infer_case_name(model_cfg, validate_cfg, run_dir)
    return RunArtifacts(
        run_dir=run_dir,
        case_name=case_name,
        model_config_path=model_config_path,
        validate_config_path=validate_config_path,
        checkpoint_path=checkpoint_path.resolve(),
    )


def discover_run_dirs(args: argparse.Namespace) -> list[RunArtifacts]:
    if args.run_dir.strip():
        run_dirs = [resolve_cli_path(args.run_dir)]
    else:
        runs_root = resolve_cli_path(args.runs_root) or DEFAULT_RUNS_ROOT.resolve()
        if not runs_root.is_dir():
            raise FileNotFoundError(f"Runs root not found: {runs_root}")
        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())

    artifacts: list[RunArtifacts] = []
    case_filter = parse_case_filter(args.cases)
    for run_dir in run_dirs:
        if run_dir is None:
            continue
        artifact = discover_run_artifacts(run_dir)
        if case_filter and artifact.case_name not in case_filter:
            continue
        artifacts.append(artifact)

    if not artifacts:
        raise ValueError("No valid Neural-Experts run directories were discovered.")
    return artifacts


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
    def __init__(self, target_path: Path, volume_shape: VolumeShape) -> None:
        self.target_path = target_path
        self.volume_shape = volume_shape
        self._array: np.ndarray | None = None

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
        per_timestep = self.volume_shape.voxels_per_timestep
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
        cwd=str(INR_ROOT),
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
        raise FileNotFoundError("GT PNG generation requires a valid GT target file.")

    gt_temp_npy = run_dir / "gt" / f"GT_{case_name}_{int(t_idx):04d}.npy"
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


def compute_psnr(pred_flat: np.ndarray, gt_flat: np.ndarray) -> float:
    if pred_flat.shape != gt_flat.shape:
        raise ValueError(f"PSNR shape mismatch: pred={pred_flat.shape}, gt={gt_flat.shape}")

    diff = np.asarray(pred_flat, dtype=np.float64) - np.asarray(gt_flat, dtype=np.float64)
    if diff.size <= 0:
        return float("nan")

    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")

    gt_min = float(np.min(gt_flat))
    gt_max = float(np.max(gt_flat))
    data_range = gt_max - gt_min
    if (not np.isfinite(data_range)) or data_range <= 0.0:
        data_range = max(abs(gt_min), abs(gt_max)) + 1.0e-12
    return float(10.0 * math.log10((data_range * data_range) / (mse + 1.0e-12)))


def write_metrics_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
                    "psnr": "" if row.get("psnr") is None else f"{float(row['psnr']):.8f}",
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
    run_dir: Path,
    checkpoint_path: Path,
    total_inference_seconds: float,
) -> dict[str, Any]:
    success_rows = [row for row in results if row.get("status") == "ok"]
    failed_rows = [row for row in results if row.get("status") != "ok"]
    return {
        "case": case_name,
        "method": method_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "result_dir": str(method_dir),
        "total_timesteps": len(results),
        "successful_timesteps": [int(row["timestep"]) for row in success_rows],
        "failed_timesteps": [int(row["timestep"]) for row in failed_rows],
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "psnr": summarize_metric(success_rows, "psnr"),
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
    if summary.get("psnr"):
        logger.info(
            "PSNR mean=%.6f std=%.6f min=%.6f max=%.6f",
            summary["psnr"]["mean"],
            summary["psnr"]["std"],
            summary["psnr"]["min"],
            summary["psnr"]["max"],
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


def resolve_target_paths(validate_cfg: dict[str, Any]) -> tuple[Path, Path | None]:
    data_cfg = validate_cfg["data"]
    target_path = resolve_src_relative(data_cfg.get("target_path"))
    if target_path is None:
        raise ValueError("validate_artifacts/config.yaml does not contain data.target_path")
    target_stats_path = resolve_src_relative(data_cfg.get("target_stats_path"))
    return target_path, target_stats_path


def resolve_output_stats(
    case_name: str,
    validate_cfg: dict[str, Any],
    checkpoint_payload: Any,
) -> tuple[np.ndarray, np.ndarray]:
    normalize_targets = bool(validate_cfg["data"].get("normalize_targets", False))
    if not normalize_targets:
        return np.zeros((1,), dtype=np.float32), np.ones((1,), dtype=np.float32)

    if isinstance(checkpoint_payload, dict):
        y_mean = checkpoint_payload.get("y_mean")
        y_std = checkpoint_payload.get("y_std")
        if y_mean is not None and y_std is not None:
            mean_arr = np.asarray(y_mean, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(y_std, dtype=np.float32).reshape(-1)
            return mean_arr, np.maximum(std_arr, 1.0e-12)

    target_path, target_stats_path = resolve_target_paths(validate_cfg)
    if target_stats_path is not None and target_stats_path.is_file():
        return load_target_stats(target_stats_path, case_name)

    target_array = np.load(str(target_path), mmap_mode="r")
    return compute_target_stats(target_array)


def resolve_input_stats(
    volume_shape: VolumeShape,
    validate_cfg: dict[str, Any],
    checkpoint_payload: Any,
) -> tuple[np.ndarray, np.ndarray]:
    normalize_inputs = bool(validate_cfg["data"].get("normalize_inputs", True))
    if not normalize_inputs:
        return np.zeros((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)

    if isinstance(checkpoint_payload, dict):
        x_mean = checkpoint_payload.get("x_mean")
        x_std = checkpoint_payload.get("x_std")
        if x_mean is not None and x_std is not None:
            mean_arr = np.asarray(x_mean, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(x_std, dtype=np.float32).reshape(-1)
            return mean_arr, np.maximum(std_arr, 1.0e-12)

    return compute_input_stats(volume_shape)


def extract_prediction_tensor(model_output: Any) -> torch.Tensor:
    if not isinstance(model_output, dict):
        raise TypeError(f"Unexpected model output type: {type(model_output)!r}")

    if "selected_nonmanifold_pnts_pred" in model_output:
        pred = model_output["selected_nonmanifold_pnts_pred"]
    elif "nonmanifold_pnts_pred" in model_output:
        pred = model_output["nonmanifold_pnts_pred"]
    else:
        raise KeyError(f"Unable to find prediction tensor in model output keys: {list(model_output.keys())}")

    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred)
    return pred


def predict_timestep_flat_scalar(
    model: torch.nn.Module,
    volume_shape: VolumeShape,
    normalize_inputs: bool,
    normalize_targets: bool,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    t_idx: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    x_mean_t = torch.as_tensor(x_mean, device=device, dtype=torch.float32).reshape(-1)
    x_std_t = torch.as_tensor(x_std, device=device, dtype=torch.float32).reshape(-1)
    y_mean_t = torch.as_tensor(y_mean, device=device, dtype=torch.float32).reshape(-1)
    y_std_t = torch.as_tensor(y_std, device=device, dtype=torch.float32).reshape(-1)

    voxel_count = volume_shape.voxels_per_timestep
    out_flat = np.empty((voxel_count,), dtype=np.float32)

    model.eval()
    sync_if_needed(device)
    infer_start = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, voxel_count, int(batch_size)):
            end = min(start + int(batch_size), voxel_count)
            idx = torch.arange(start, end, device=device, dtype=torch.int64)

            x = (idx % volume_shape.X).to(dtype=torch.float32)
            y = ((idx // volume_shape.X) % volume_shape.Y).to(dtype=torch.float32)
            z = (idx // (volume_shape.X * volume_shape.Y)).to(dtype=torch.float32)
            t = torch.full_like(x, float(t_idx), dtype=torch.float32)

            coords = torch.empty((1, end - start, 4), device=device, dtype=torch.float32)
            if normalize_inputs:
                coords[0, :, 0] = (x - x_mean_t[0]) / x_std_t[0]
                coords[0, :, 1] = (y - x_mean_t[1]) / x_std_t[1]
                coords[0, :, 2] = (z - x_mean_t[2]) / x_std_t[2]
                coords[0, :, 3] = (t - x_mean_t[3]) / x_std_t[3]
            else:
                coords[0, :, 0] = x
                coords[0, :, 1] = y
                coords[0, :, 2] = z
                coords[0, :, 3] = t

            pred = extract_prediction_tensor(model(coords))
            pred = pred.detach().to(dtype=torch.float32).squeeze()
            pred = pred.reshape(-1)
            if pred.numel() != (end - start):
                raise ValueError(
                    f"Prediction size mismatch at timestep {t_idx}: "
                    f"{pred.numel()} vs expected {end - start}"
                )

            if normalize_targets:
                pred = pred * y_std_t[0] + y_mean_t[0]

            out_flat[start:end] = pred.cpu().numpy().astype(np.float32, copy=False)

    sync_if_needed(device)
    infer_time = time.perf_counter() - infer_start
    return out_flat, infer_time


def build_model_for_run(
    model_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    cfg_copy = yaml.safe_load(yaml.safe_dump(model_config))
    model, _ = build_model(cfg_copy, cfg_copy["LOSS"])
    payload = torch_load_checkpoint(checkpoint_path)
    model_state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(model_state, strict=True)
    model = model.to(device).eval()
    return model, payload


def run_single_artifact(
    artifact: RunArtifacts,
    args: argparse.Namespace,
    validate_image_pair,
    use_lpips: bool,
    metrics_device: str,
) -> dict[str, Any]:
    model_config = load_yaml(artifact.model_config_path)
    validate_cfg = load_yaml(artifact.validate_config_path)
    case_name = artifact.case_name
    method_name = build_method_name(args.method_name, case_name)
    result_root = resolve_cli_path(args.result_root) or DEFAULT_RESULT_ROOT
    tmp_root = resolve_cli_path(args.tmp_root) or DEFAULT_TMP_ROOT
    if args.gt_target_path and not args.run_dir.strip():
        raise ValueError("--gt-target-path is only supported in single-run mode.")

    target_path, _ = resolve_target_paths(validate_cfg)
    volume_shape = volume_shape_from_config(validate_cfg["data"]["volume_shape"])
    transfer_function_path = resolve_transfer_function_path(case_name)
    viewport_path = resolve_viewport_path(case_name)
    method_dir = result_root / case_name / method_name
    gt_png_dir = result_root / case_name
    metrics_csv_path = method_dir / "image_metrics.csv"
    metrics_json_path = method_dir / "image_metrics_summary.json"
    method_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, checkpoint_payload = build_model_for_run(model_config, artifact.checkpoint_path, device)
    normalize_inputs = bool(validate_cfg["data"].get("normalize_inputs", True))
    normalize_targets = bool(validate_cfg["data"].get("normalize_targets", False))
    x_mean, x_std = resolve_input_stats(volume_shape, validate_cfg, checkpoint_payload)
    y_mean, y_std = resolve_output_stats(case_name, validate_cfg, checkpoint_payload)
    timesteps = select_timesteps(args.timestamp, args.timestamps, volume_shape.T)

    gt_target_path = resolve_cli_path(args.gt_target_path) if args.gt_target_path else target_path
    gt_metrics_source: GroundTruthVolumeSource | None = None
    if gt_target_path is not None and gt_target_path.is_file():
        gt_metrics_source = GroundTruthVolumeSource(gt_target_path, volume_shape)
    else:
        logger.warning("GT target not found, PSNR will be skipped: %s", gt_target_path)

    gt_render_source: GroundTruthVolumeSource | None = None
    if args.gt_render_strategy in {"missing", "always"}:
        if gt_target_path is None or not gt_target_path.is_file():
            raise FileNotFoundError(
                f"GT target file required for --gt-render-strategy={args.gt_render_strategy}: {gt_target_path}"
            )
        gt_render_source = gt_metrics_source or GroundTruthVolumeSource(gt_target_path, volume_shape)

    run_dir = build_run_directory(tmp_root, method_name)
    total_inference_seconds = 0.0
    results: list[dict[str, Any]] = []

    logger.info("Run directory: %s", artifact.run_dir)
    logger.info("Case=%s Method=%s Timesteps=%s", case_name, method_name, timesteps)
    logger.info("Prediction PNG output directory: %s", method_dir)
    logger.info("GT PNG directory: %s", gt_png_dir)
    logger.info("Temporary run directory: %s", run_dir)

    try:
        for t_idx in timesteps:
            pred_temp_npy = build_output_path(run_dir / "pred", args.prefix, int(t_idx))
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
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "status": "pending",
                "error": "",
                "inference_seconds": None,
            }

            logger.info("Processing case=%s timestep=%d", case_name, int(t_idx))
            try:
                pred_temp_npy.parent.mkdir(parents=True, exist_ok=True)
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
                row["inference_seconds"] = float(infer_time)
                total_inference_seconds += infer_time
                np.save(pred_temp_npy, pred_flat)

                if gt_metrics_source is not None:
                    stage = "psnr"
                    gt_flat = gt_metrics_source.extract_scalar_timestep(int(t_idx))
                    row["psnr"] = compute_psnr(pred_flat, gt_flat)

                stage = "pred_render"
                pred_rendered_png = run_render_task(
                    args=args,
                    volume_path=pred_temp_npy,
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=volume_shape.dims_xyz,
                )
                final_pred_png.parent.mkdir(parents=True, exist_ok=True)
                pred_rendered_png.replace(final_pred_png)

                stage = "gt_render"
                gt_png_path, gt_temp_npy = ensure_ground_truth_png(
                    args=args,
                    gt_strategy=args.gt_render_strategy,
                    gt_png_path=gt_png_path,
                    gt_volume_source=gt_render_source,
                    run_dir=run_dir,
                    case_name=case_name,
                    t_idx=int(t_idx),
                    transfer_function_path=transfer_function_path,
                    viewport_path=viewport_path,
                    dims_xyz=volume_shape.dims_xyz,
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
                    "case=%s t=%d PSNR=%s SSIM=%.6f%s",
                    case_name,
                    int(t_idx),
                    "N/A" if row["psnr"] is None else f"{float(row['psnr']):.6f}",
                    row["ssim"],
                    "" if row["lpips"] is None else f" LPIPS={row['lpips']:.6f}",
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"{stage}_failed"
                row["error"] = str(exc)
                logger.exception("Failed case=%s timestep=%d during %s", case_name, int(t_idx), stage)
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
        run_dir=artifact.run_dir,
        checkpoint_path=artifact.checkpoint_path,
        total_inference_seconds=total_inference_seconds,
    )
    write_summary_json(summary, metrics_json_path)
    log_summary(summary)
    logger.info("Metrics CSV written to %s", metrics_csv_path)
    logger.info("Metrics summary JSON written to %s", metrics_json_path)
    return summary


def main() -> int:
    setup_logging()
    args = parse_args()

    if not RENDER_TASK_PATH.is_file():
        raise FileNotFoundError(f"render_task.py not found: {RENDER_TASK_PATH}")
    if not IMAGE_VALIDATION_PATH.is_file():
        raise FileNotFoundError(f"image_level_validation.py not found: {IMAGE_VALIDATION_PATH}")

    artifacts = discover_run_dirs(args)
    validation_module = load_validation_module()
    validate_image_pair = validation_module.validate_image_pair
    use_lpips = getattr(validation_module, "lpips", None) is not None
    metrics_device = "cuda" if torch.cuda.is_available() else "cpu"

    if not use_lpips:
        logger.warning("lpips is not installed. Falling back to SSIM-only image validation.")

    summaries: list[dict[str, Any]] = []
    for artifact in artifacts:
        summaries.append(
            run_single_artifact(
                artifact=artifact,
                args=args,
                validate_image_pair=validate_image_pair,
                use_lpips=use_lpips,
                metrics_device=metrics_device,
            )
        )

    logger.info("Processed %d Neural-Experts run(s).", len(summaries))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate_prediction_render.py failed")
        print(f"validate_prediction_render.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

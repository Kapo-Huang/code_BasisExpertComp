from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


TTHRESH_PSNR_OFFSET = 20.0 * math.log10(2.0)


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


class CommandExecutionError(RuntimeError):
    def __init__(self, result: CommandResult):
        self.result = result
        super().__init__(
            f"Command failed with exit code {result.returncode}: {' '.join(result.command)}"
        )


def resolve_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_shape(shape_value: Any) -> tuple[int, ...] | None:
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(item.strip()) for item in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(item) for item in shape_value)
    raise ValueError("shape must be a comma-separated string or a list of integers")


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    return config


def load_psnr_config(
    script_dir: Path,
    config_arg: str,
    binary_key: str,
    legacy_keys: Iterable[str] = ("cr", "rate"),
) -> dict[str, Any]:
    config_path = resolve_path(script_dir, config_arg)
    config = load_yaml_config(config_path)

    present_legacy_keys = [key for key in legacy_keys if key in config]
    if present_legacy_keys:
        joined = ", ".join(sorted(present_legacy_keys))
        raise ValueError(f"Unsupported config keys: {joined}. Use 'psnr' instead.")

    required_keys = ("input", binary_key, "psnr", "compressed", "recon")
    missing = [key for key in required_keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config keys: {joined}")

    result_json_value = config.get("result_json")
    return {
        "config_path": config_path,
        "input_path": resolve_path(script_dir, config["input"]),
        "binary_path": resolve_path(script_dir, config[binary_key]),
        "compressed_path": resolve_path(script_dir, config["compressed"]),
        "recon_path": resolve_path(script_dir, config["recon"]),
        "result_json_path": resolve_path(script_dir, result_json_value)
        if result_json_value
        else None,
        "shape": parse_shape(config.get("shape")),
        "target_psnr": float(config["psnr"]),
        "raw_config": config,
    }


def load_array(
    input_path: Path,
    shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    array = np.load(input_path, allow_pickle=False)
    if array.dtype == object:
        raise TypeError("object dtype .npy files are not supported")

    loaded_shape = tuple(int(item) for item in array.shape)
    if shape is not None:
        if int(np.prod(shape)) != int(array.size):
            raise ValueError(f"shape {shape} does not match data size {array.size}")
        array = array.reshape(shape)

    used_shape = tuple(int(item) for item in array.shape)
    return np.ascontiguousarray(array), loaded_shape, used_shape


def run_command(command: Sequence[str | Path]) -> CommandResult:
    normalized_command = tuple(str(item) for item in command)
    completed = subprocess.run(
        normalized_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result = CommandResult(
        command=normalized_command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
    if result.returncode != 0:
        raise CommandExecutionError(result)
    return result


def format_command_error(error: CommandExecutionError) -> str:
    result = error.result
    lines = [
        "Command failed:",
        " ".join(result.command),
    ]
    if result.stdout:
        lines.append("STDOUT:")
        lines.append(result.stdout.rstrip())
    if result.stderr:
        lines.append("STDERR:")
        lines.append(result.stderr.rstrip())
    return "\n".join(lines) + "\n"


def get_sz3_dtype_args(dtype: np.dtype[Any] | type[np.generic]) -> tuple[list[str], str]:
    mapping: dict[np.dtype[Any], tuple[list[str], str]] = {
        np.dtype(np.float32): (["-f"], "float32"),
        np.dtype(np.float64): (["-d"], "float64"),
        np.dtype(np.int32): (["-I", "32"], "int32"),
        np.dtype(np.int64): (["-I", "64"], "int64"),
    }
    dtype_obj = np.dtype(dtype)
    if dtype_obj not in mapping:
        raise TypeError("SZ3 PSNR mode supports float32, float64, int32, and int64 inputs")
    return mapping[dtype_obj]


def get_tthresh_dtype(dtype: np.dtype[Any] | type[np.generic]) -> tuple[str, str]:
    mapping: dict[np.dtype[Any], tuple[str, str]] = {
        np.dtype(np.uint8): ("uchar", "uint8"),
        np.dtype(np.uint16): ("ushort", "uint16"),
        np.dtype(np.int32): ("int", "int32"),
        np.dtype(np.float32): ("float", "float32"),
        np.dtype(np.float64): ("double", "float64"),
    }
    dtype_obj = np.dtype(dtype)
    if dtype_obj not in mapping:
        raise TypeError("TTHRESH supports uint8, uint16, int32, float32, and float64 inputs")
    return mapping[dtype_obj]


def get_zfp_dtype_flag(dtype: np.dtype[Any] | type[np.generic]) -> tuple[str, str]:
    mapping: dict[np.dtype[Any], tuple[str, str]] = {
        np.dtype(np.float32): ("-f", "float32"),
        np.dtype(np.float64): ("-d", "float64"),
    }
    dtype_obj = np.dtype(dtype)
    if dtype_obj not in mapping:
        raise TypeError("ZFP PSNR mode supports only float32 and float64 inputs")
    return mapping[dtype_obj]


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    original_f64 = np.asarray(original, dtype=np.float64)
    reconstructed_f64 = np.asarray(reconstructed, dtype=np.float64)
    if original_f64.shape != reconstructed_f64.shape:
        raise ValueError(
            f"shape mismatch: original {original_f64.shape} vs reconstructed {reconstructed_f64.shape}"
        )

    diff = original_f64 - reconstructed_f64
    mse = float(np.mean(diff**2)) if diff.size else 0.0
    rmse = float(math.sqrt(mse))
    max_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    data_range = float(np.max(original_f64) - np.min(original_f64)) if original_f64.size else 0.0

    if mse == 0.0:
        measured_psnr = float("inf")
    elif data_range == 0.0:
        measured_psnr = float("-inf")
    else:
        measured_psnr = float(20.0 * math.log10(data_range / rmse))

    return {
        "measured_psnr": measured_psnr,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "data_range": data_range,
    }


def tthresh_native_psnr(target_psnr: float) -> float:
    return float(target_psnr - TTHRESH_PSNR_OFFSET)


def zfp_tolerance_from_psnr(data_range: float, target_psnr: float) -> float:
    return float(data_range * (10.0 ** (-target_psnr / 20.0)))


def format_number(value: float) -> str:
    return f"{float(value):.17g}"


def build_sz3_compress_command(
    binary_path: Path,
    raw_input_path: Path,
    compressed_path: Path,
    dtype_args: Sequence[str],
    shape: Sequence[int],
    target_psnr: float,
) -> list[str]:
    return [
        str(binary_path),
        *dtype_args,
        "-i",
        str(raw_input_path),
        "-z",
        str(compressed_path),
        f"-{len(shape)}",
        *map(str, shape),
        "-M",
        "PSNR",
        format_number(target_psnr),
    ]


def build_sz3_decompress_command(
    binary_path: Path,
    compressed_path: Path,
    raw_output_path: Path,
    dtype_args: Sequence[str],
    shape: Sequence[int],
) -> list[str]:
    return [
        str(binary_path),
        *dtype_args,
        "-z",
        str(compressed_path),
        "-o",
        str(raw_output_path),
        f"-{len(shape)}",
        *map(str, shape),
    ]


def build_tthresh_command(
    binary_path: Path,
    raw_input_path: Path,
    compressed_path: Path,
    raw_output_path: Path,
    io_type: str,
    shape: Sequence[int],
    native_psnr: float,
) -> list[str]:
    return [
        str(binary_path),
        "-i",
        str(raw_input_path),
        "-t",
        io_type,
        "-s",
        *map(str, shape),
        "-p",
        format_number(native_psnr),
        "-c",
        str(compressed_path),
        "-o",
        str(raw_output_path),
    ]


def build_zfp_compress_command(
    binary_path: Path,
    raw_input_path: Path,
    compressed_path: Path,
    dtype_flag: str,
    shape: Sequence[int],
    tolerance: float,
) -> list[str]:
    return [
        str(binary_path),
        dtype_flag,
        f"-{len(shape)}",
        *map(str, shape),
        "-i",
        str(raw_input_path),
        "-z",
        str(compressed_path),
        "-a",
        format_number(tolerance),
        "-h",
    ]


def build_zfp_decompress_command(
    binary_path: Path,
    compressed_path: Path,
    raw_output_path: Path,
) -> list[str]:
    return [
        str(binary_path),
        "-z",
        str(compressed_path),
        "-o",
        str(raw_output_path),
        "-h",
    ]


def build_result(
    *,
    method: str,
    input_path: Path,
    compressed_path: Path,
    recon_path: Path,
    loaded_shape: Sequence[int],
    used_shape: Sequence[int],
    dtype_label: str,
    target_psnr: float,
    native_mode: str,
    native_value: float,
    original: np.ndarray,
    reconstructed: np.ndarray,
    compress_result: CommandResult | None = None,
    decompress_result: CommandResult | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics(original, reconstructed)
    compressed_nbytes = int(compressed_path.stat().st_size)
    original_nbytes = int(original.nbytes)
    compression_ratio = float("inf") if compressed_nbytes == 0 else float(original_nbytes / compressed_nbytes)

    return {
        "method": method,
        "input": str(input_path),
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "loaded_shape": list(loaded_shape),
        "used_shape": list(used_shape),
        "dtype": dtype_label,
        "target_psnr": float(target_psnr),
        "native_mode": native_mode,
        "native_value": float(native_value),
        "measured_psnr": metrics["measured_psnr"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "max_error": metrics["max_error"],
        "original_nbytes": original_nbytes,
        "compressed_nbytes": compressed_nbytes,
        "compression_ratio": compression_ratio,
        "compress_stdout": compress_result.stdout if compress_result else "",
        "compress_stderr": compress_result.stderr if compress_result else "",
        "decompress_stdout": decompress_result.stdout if decompress_result else "",
        "decompress_stderr": decompress_result.stderr if decompress_result else "",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

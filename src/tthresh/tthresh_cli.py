import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


REPO_DIR = Path(__file__).resolve().parent


def resolve_path(path_value):
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (REPO_DIR / p).resolve()


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.stderr.write("CMD FAILED:\n")
        sys.stderr.write(" ".join(map(str, cmd)) + "\n")
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        sys.exit(p.returncode)
    return p


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def dtype_to_tthresh(arr):
    if arr.dtype == np.uint8:
        return arr.astype(np.uint8, copy=False), np.uint8, "uchar", "uint8"
    if arr.dtype == np.float32:
        return arr.astype(np.float32, copy=False), np.float32, "float", "float32"
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
        return arr, np.float32, "float", "float32"
    arr = arr.astype(np.float32)
    return arr, np.float32, "float", "float32"


def parse_shape(shape_value):
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(v) for v in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(v) for v in shape_value)
    raise ValueError("shape 必须是字符串 '600,248,248' 或列表 [600,248,248]")


def main():
    p = argparse.ArgumentParser(description="Compress numpy array using TTHRESH with config from YAML")
    p.add_argument("--config", required=True, help="YAML configuration file path")
    args = p.parse_args()

    config_path = resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_path = resolve_path(cfg["input"])
    tthresh_path = resolve_path(cfg["tthresh"])
    psnr = float(cfg["psnr"])
    compressed_path = resolve_path(cfg["compressed"])
    recon_path = resolve_path(cfg["recon"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None

    shape_cfg = parse_shape(cfg.get("shape"))

    ensure_parent(compressed_path)
    ensure_parent(recon_path)
    if result_json is not None:
        ensure_parent(result_json)

    arr = np.load(input_path, allow_pickle=False)
    if arr.dtype == object:
        raise TypeError("object dtype .npy is not supported")

    loaded_shape = tuple(int(x) for x in arr.shape)

    if arr.ndim < 3:
        if shape_cfg is None:
            raise ValueError(
                f"input npy is ndim={arr.ndim}, loaded shape={loaded_shape}. "
                f"TTHRESH needs 3+ dims. Please provide 'shape' in YAML like [600, 248, 248]"
            )
        if int(np.prod(shape_cfg)) != int(arr.size):
            raise ValueError(f"shape {shape_cfg} does not match data size {arr.size}")
        arr = arr.reshape(shape_cfg)

    arr, np_dtype, ttype, dtype_name = dtype_to_tthresh(arr)
    arr = np.ascontiguousarray(arr)

    shape = tuple(int(x) for x in arr.shape)
    if len(shape) < 3:
        raise ValueError(f"TTHRESH needs 3 or more dimensions, got ndim={len(shape)}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "input.raw"
        raw_out = td / "recon.raw"

        arr.tofile(raw_in)

        out = run([
            str(tthresh_path),
            "-i", str(raw_in),
            "-t", ttype,
            "-s", *map(str, shape),
            "-p", str(psnr),
            "-c", str(compressed_path),
            "-o", str(raw_out),
        ])

        recon = np.fromfile(raw_out, dtype=np_dtype).reshape(shape)
        np.save(recon_path, recon)

    result = {
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "loaded_shape": list(loaded_shape),
        "used_shape": list(shape),
        "dtype": dtype_name,
        "psnr": psnr,
        "stdout": out.stdout.strip(),
    }

    if result_json is not None:
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
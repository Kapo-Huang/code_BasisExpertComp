import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml


REPO_DIR = Path(__file__).resolve().parent


def log(message):
    now = time.strftime('%H:%M:%S')
    print(f"[{now}] {message}", flush=True)


def resolve_path(path_value):
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (REPO_DIR / p).resolve()


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run(cmd, desc=None):
    if desc:
        log(desc)
    start = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start
    if p.returncode != 0:
        sys.stderr.write("CMD FAILED:\n")
        sys.stderr.write(" ".join(map(str, cmd)) + "\n")
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        sys.exit(p.returncode)
    if desc:
        log(f"完成: {desc} (耗时 {elapsed:.2f}s)")
    return p


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


def write_archive(archive_path, payload_path, payload_name, meta):
    ensure_parent(archive_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(payload_path, arcname=payload_name)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def read_archive(archive_path, extract_dir):
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        if "meta.json" not in names:
            raise ValueError("压缩文件中缺少 meta.json")
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        payload_name = meta.get("payload_name", "payload.tth")
        if payload_name not in names:
            raise ValueError(f"压缩文件中缺少 {payload_name}")
        payload_path = Path(extract_dir) / payload_name
        with open(payload_path, "wb") as f:
            f.write(zf.read(payload_name))
    return meta, payload_path


def save_result(result, result_json):
    if result_json is not None:
        ensure_parent(result_json)
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    return cfg


def compress_mode(cfg):
    input_path = resolve_path(cfg["input"])
    tthresh_path = resolve_path(cfg["tthresh"])
    psnr = float(cfg["psnr"])
    compressed_path = resolve_path(cfg["compressed"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None
    shape_cfg = parse_shape(cfg.get("shape"))

    file_size = input_path.stat().st_size
    log(f"开始读取原始数据: {input_path} ({file_size / (1024 ** 2):.2f} MiB)")
    arr = np.load(input_path, allow_pickle=False)
    log(f"读取完成: shape={arr.shape}, dtype={arr.dtype}")

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

    log(f"准备压缩: used_shape={shape}, dtype={dtype_name}, psnr={psnr}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "input.raw"
        payload_path = td / "payload.tth"

        log("写入临时 raw 输入文件")
        arr.tofile(raw_in)
        log("临时 raw 输入文件写入完成")

        out = run([
            str(tthresh_path),
            "-i", str(raw_in),
            "-t", ttype,
            "-s", *map(str, shape),
            "-p", str(psnr),
            "-c", str(payload_path),
        ], desc="调用 TTHRESH 压缩")

        meta = {
            "format": "tthresh_archive_v2",
            "payload_name": "payload.tth",
            "input": str(input_path),
            "loaded_shape": list(loaded_shape),
            "used_shape": list(shape),
            "dtype": dtype_name,
            "numpy_dtype": np.dtype(np_dtype).name,
            "tthresh_type": ttype,
            "psnr": psnr,
            "original_nbytes": int(arr.nbytes),
            "compressed_nbytes": int(payload_path.stat().st_size),
            "stdout": out.stdout.strip(),
            "stderr": out.stderr.strip(),
        }

        log(f"写出压缩归档: {compressed_path}")
        write_archive(compressed_path, payload_path, "payload.tth", meta)
        log("压缩归档写出完成")

    result = {
        "mode": "compress",
        "compressed": str(compressed_path),
        "loaded_shape": list(loaded_shape),
        "shape": list(shape),
        "dtype": dtype_name,
        "psnr": psnr,
    }
    save_result(result, result_json)


def decompress_mode(cfg):
    compressed_path = resolve_path(cfg["compressed"])
    recon_path = resolve_path(cfg["recon"])
    tthresh_path = resolve_path(cfg["tthresh"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None

    ensure_parent(recon_path)

    file_size = compressed_path.stat().st_size
    log(f"开始读取压缩文件: {compressed_path} ({file_size / (1024 ** 2):.2f} MiB)")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_out = td / "recon.raw"

        meta, payload_path = read_archive(compressed_path, td)
        shape = tuple(int(x) for x in meta["used_shape"])
        np_dtype = np.dtype(meta["numpy_dtype"])

        run([
            str(tthresh_path),
            "-c", str(payload_path),
            "-o", str(raw_out),
        ], desc="调用 TTHRESH 解压")

        log("读取重建 raw 并保存为 npy")
        recon = np.fromfile(raw_out, dtype=np_dtype).reshape(shape)
        np.save(recon_path, recon)
        log(f"重建文件已保存: {recon_path}")

    result = {
        "mode": "decompress",
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "shape": list(shape),
        "dtype": str(np_dtype),
        "psnr": meta.get("psnr"),
    }
    save_result(result, result_json)


def main():
    p = argparse.ArgumentParser(description="TTHRESH CLI with progress output and split compress/decompress modes")
    p.add_argument("--config", required=True, help="YAML configuration file path")
    p.add_argument("--mode", required=True, choices=["compress", "decompress"], help="运行模式")
    args = p.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    if args.mode == "compress":
        compress_mode(cfg)
    else:
        decompress_mode(cfg)


if __name__ == "__main__":
    main()

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
SUPPORTED_METRICS = {"ABS", "REL", "PSNR"}


def log(message):
    now = time.strftime("%H:%M:%S")
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


def choose_dtype(arr):
    if arr.dtype == np.float32:
        return arr, np.float32, "-f", "float32"
    if arr.dtype == np.float64:
        return arr, np.float64, "-d", "float64"
    arr = arr.astype(np.float32)
    return arr, np.float32, "-f", "float32"


def parse_shape(shape_value):
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(x) for x in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(x) for x in shape_value)
    raise ValueError("shape 必须是字符串 '600,248,248' 或列表 [600,248,248]")


def parse_metric_name(metric_name):
    if metric_name is None:
        raise ValueError("compress 模式下必须提供 metric")

    metric_name = str(metric_name).strip().upper()
    aliases = {
        "ABSOLUTE": "ABS",
        "ABS_ERROR": "ABS",
        "RELATIVE": "REL",
    }
    metric_name = aliases.get(metric_name, metric_name)
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"不支持的 metric: {metric_name}，目前只支持 {sorted(SUPPORTED_METRICS)}")
    return metric_name


def build_metric_spec(cfg):
    if cfg.get("cr") is not None:
        raise ValueError("最终版只支持 ABS / REL / PSNR 直控模式，不再支持 cr 搜索模式")

    metric = parse_metric_name(cfg.get("metric") or cfg.get("error_mode"))
    value = cfg.get("metric_value")
    if value is None:
        value = cfg.get("error_value")
    if value is None:
        value = cfg.get("eb")

    if value is None:
        raise ValueError("compress 模式下必须提供 metric_value")

    metric_value = float(value)
    if metric_value <= 0:
        raise ValueError("metric_value 必须 > 0")

    return {
        "metric": metric,
        "metric_value": metric_value,
    }


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    return cfg


def compress_once(sz3_path, raw_path, cmp_path, dtype_flag, shape, metric, metric_value):
    cmd = [
        str(sz3_path),
        dtype_flag,
        "-i", str(raw_path),
        "-z", str(cmp_path),
        f"-{len(shape)}",
        *map(str, shape),
        "-M", metric, f"{metric_value:.18g}",
    ]
    run(cmd, desc=f"调用 SZ3 压缩 (metric={metric}, value={metric_value:.6g})")
    return os.path.getsize(cmp_path)


def decompress_once(sz3_path, cmp_path, raw_out_path, dtype_flag, shape):
    cmd = [
        str(sz3_path),
        dtype_flag,
        "-z", str(cmp_path),
        "-o", str(raw_out_path),
        f"-{len(shape)}",
        *map(str, shape),
    ]
    run(cmd, desc="调用 SZ3 解压")


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
        payload_name = meta.get("payload_name", "payload.sz")
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


def compress_mode(cfg):
    input_path = resolve_path(cfg["input"])
    sz3_path = resolve_path(cfg["sz3"])
    compressed_path = resolve_path(cfg["compressed"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None
    shape_cfg = parse_shape(cfg.get("shape"))
    metric_spec = build_metric_spec(cfg)

    file_size = input_path.stat().st_size
    log(f"开始读取原始数据: {input_path} ({file_size / (1024 ** 2):.2f} MiB)")
    arr = np.load(input_path, allow_pickle=False)
    log(f"读取完成: shape={arr.shape}, dtype={arr.dtype}")

    if arr.dtype == object:
        raise TypeError("object dtype not supported")

    loaded_shape = tuple(int(x) for x in arr.shape)
    arr, np_dtype, dtype_flag, dtype_name = choose_dtype(arr)

    if shape_cfg is not None:
        if int(np.prod(shape_cfg)) != int(arr.size):
            raise ValueError(f"shape {shape_cfg} 和数据 size {arr.size} 不匹配")
        arr = arr.reshape(shape_cfg)
        shape = tuple(int(x) for x in shape_cfg)
    else:
        shape = tuple(int(x) for x in arr.shape)

    if len(shape) < 1 or len(shape) > 4:
        raise ValueError(f"SZ3 CLI only supports 1D-4D arrays, got ndim={len(shape)}")

    arr = np.ascontiguousarray(arr)
    original_nbytes = int(arr.nbytes)
    log(f"准备压缩: used_shape={shape}, dtype={dtype_name}, 原始大小={original_nbytes} bytes")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "input.raw"
        cmp_path = td / "payload.sz"

        log("写入临时 raw 输入文件")
        arr.tofile(raw_in)
        log("临时 raw 输入文件写入完成")

        cmp_size = compress_once(
            sz3_path=sz3_path,
            raw_path=raw_in,
            cmp_path=cmp_path,
            dtype_flag=dtype_flag,
            shape=shape,
            metric=metric_spec["metric"],
            metric_value=metric_spec["metric_value"],
        )
        actual_cr = original_nbytes / cmp_size
        log(f"压缩完成: compressed={cmp_size} bytes, actual_cr={actual_cr:.6g}")

        meta = {
            "format": "sz3_archive_final",
            "payload_name": "payload.sz",
            "input": str(input_path),
            "loaded_shape": list(loaded_shape),
            "used_shape": list(shape),
            "dtype": dtype_name,
            "numpy_dtype": np.dtype(np_dtype).name,
            "sz3_dtype_flag": dtype_flag,
            "metric": metric_spec["metric"],
            "metric_value": metric_spec["metric_value"],
            "actual_cr": actual_cr,
            "original_nbytes": original_nbytes,
            "compressed_nbytes": cmp_size,
        }

        log(f"写出压缩归档: {compressed_path}")
        write_archive(compressed_path, cmp_path, "payload.sz", meta)
        log("压缩归档写出完成")

    result = {
        "mode": "compress",
        "compressed": str(compressed_path),
        "loaded_shape": list(loaded_shape),
        "shape": list(shape),
        "dtype": dtype_name,
        "metric": metric_spec["metric"],
        "metric_value": metric_spec["metric_value"],
        "actual_cr": actual_cr,
    }
    save_result(result, result_json)


def decompress_mode(cfg):
    compressed_path = resolve_path(cfg["compressed"])
    recon_path = resolve_path(cfg["recon"])
    sz3_path = resolve_path(cfg["sz3"])
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
        dtype_flag = meta["sz3_dtype_flag"]

        log(f"解压参数: shape={shape}, dtype={np_dtype}, dtype_flag={dtype_flag}")
        decompress_once(sz3_path, payload_path, raw_out, dtype_flag, shape)

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
        "metric": meta.get("metric"),
        "metric_value": meta.get("metric_value"),
        "actual_cr": meta.get("actual_cr"),
    }
    save_result(result, result_json)


def main():
    ap = argparse.ArgumentParser(description="SZ3 CLI with progress output and split compress/decompress modes")
    ap.add_argument("--config", required=True, help="config.yaml 路径")
    ap.add_argument("--mode", required=True, choices=["compress", "decompress"], help="运行模式")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    if args.mode == "compress":
        compress_mode(cfg)
    else:
        decompress_mode(cfg)


if __name__ == "__main__":
    main()

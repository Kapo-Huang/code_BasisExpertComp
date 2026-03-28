import argparse
import json
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
        sys.stderr.write("FAILED: " + " ".join(map(str, cmd)) + "\n")
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        sys.exit(1)
    if desc:
        log(f"完成: {desc} (耗时 {elapsed:.2f}s)")
    return p


def parse_shape(shape_value):
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(v) for v in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(v) for v in shape_value)
    raise ValueError("shape 必须是字符串 '600,248,248' 或列表 [600,248,248]")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    return cfg


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
        payload_name = meta.get("payload_name", "payload.zfp")
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
    zfp_path = resolve_path(cfg["zfp"])
    compressed_path = resolve_path(cfg["compressed"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None
    shape = parse_shape(cfg.get("shape"))
    rate = float(cfg["rate"])

    if shape is None:
        raise ValueError("compress 模式必须在 YAML 中提供 shape")

    file_size = input_path.stat().st_size
    log(f"开始读取原始数据: {input_path} ({file_size / (1024 ** 2):.2f} MiB)")
    arr = np.load(input_path, allow_pickle=False).astype(np.float32)
    log(f"读取完成: shape={arr.shape}, dtype={arr.dtype}")

    if int(np.prod(shape)) != arr.size:
        raise ValueError("shape 和数据 size 不匹配")

    arr = arr.reshape(shape)
    arr = np.ascontiguousarray(arr)
    log(f"准备压缩: used_shape={shape}, dtype=float32, rate={rate}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "in.raw"
        payload_path = td / "payload.zfp"

        log("写入临时 raw 输入文件")
        arr.tofile(raw_in)
        log("临时 raw 输入文件写入完成")

        compress_cmd = [
            str(zfp_path),
            "-f",
            f"-{len(shape)}",
            *map(str, shape),
            "-i", str(raw_in),
            "-z", str(payload_path),
            "-r", str(rate),
            "-h",
        ]
        compress_out = run(compress_cmd, desc="调用 ZFP 压缩")

        meta = {
            "format": "zfp_archive_v2",
            "payload_name": "payload.zfp",
            "input": str(input_path),
            "used_shape": list(shape),
            "dtype": "float32",
            "numpy_dtype": "float32",
            "rate": rate,
            "original_nbytes": int(arr.nbytes),
            "compressed_nbytes": int(payload_path.stat().st_size),
            "stdout": compress_out.stdout.strip(),
            "stderr": compress_out.stderr.strip(),
        }

        log(f"写出压缩归档: {compressed_path}")
        write_archive(compressed_path, payload_path, "payload.zfp", meta)
        log("压缩归档写出完成")

    result = {
        "mode": "compress",
        "compressed": str(compressed_path),
        "shape": list(shape),
        "dtype": "float32",
        "rate": rate,
    }
    save_result(result, result_json)


def decompress_mode(cfg):
    compressed_path = resolve_path(cfg["compressed"])
    recon_path = resolve_path(cfg["recon"])
    zfp_path = resolve_path(cfg["zfp"])
    result_json = resolve_path(cfg["result_json"]) if cfg.get("result_json") else None

    ensure_parent(recon_path)

    file_size = compressed_path.stat().st_size
    log(f"开始读取压缩文件: {compressed_path} ({file_size / (1024 ** 2):.2f} MiB)")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_out = td / "out.raw"

        meta, payload_path = read_archive(compressed_path, td)
        shape = tuple(int(x) for x in meta["used_shape"])
        np_dtype = np.dtype(meta["numpy_dtype"])

        decompress_cmd = [
            str(zfp_path),
            "-z", str(payload_path),
            "-o", str(raw_out),
            "-h",
        ]
        run(decompress_cmd, desc="调用 ZFP 解压")

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
        "rate": meta.get("rate"),
    }
    save_result(result, result_json)


def main():
    parser = argparse.ArgumentParser(description="ZFP CLI with progress output and split compress/decompress modes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["compress", "decompress"], help="运行模式")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    if args.mode == "compress":
        compress_mode(cfg)
    else:
        decompress_mode(cfg)


if __name__ == "__main__":
    main()

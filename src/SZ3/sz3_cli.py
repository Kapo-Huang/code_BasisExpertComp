import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
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


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def compress_once(sz3_path, raw_path, cmp_path, dtype_flag, shape, eb):
    cmd = [
        str(sz3_path),
        dtype_flag,
        "-i", str(raw_path),
        "-z", str(cmp_path),
        f"-{len(shape)}",
        *map(str, shape),
        "-M", "ABS", f"{eb:.18g}",
    ]
    run(cmd)
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
    run(cmd)


def choose_dtype(arr):
    if arr.dtype == np.float32:
        return arr, np.float32, "-f", "float32"
    if arr.dtype == np.float64:
        return arr, np.float64, "-d", "float64"
    arr = arr.astype(np.float32)
    return arr, np.float32, "-f", "float32"


def make_tmp_sz():
    fd, path = tempfile.mkstemp(suffix=".sz")
    os.close(fd)
    return Path(path)


def parse_shape(shape_value):
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(x) for x in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(x) for x in shape_value)
    raise ValueError("shape 必须是字符串 '600,248,248' 或列表 [600,248,248]")


def search_error_bound(sz3_path, raw_path, dtype_flag, shape, original_nbytes, target_cr, data_min, data_max):
    if target_cr <= 1:
        raise ValueError("cr must be > 1")

    value_range = float(data_max - data_min)

    def trial(eb):
        path = make_tmp_sz()
        try:
            size = compress_once(sz3_path, raw_path, path, dtype_flag, shape, eb)
            cr = original_nbytes / size
            return path, size, cr
        except Exception:
            if path.exists():
                path.unlink()
            raise

    if value_range == 0.0:
        eb = 1e-12
        path, size, cr = trial(eb)
        return eb, path, size, cr

    low = max(value_range * 1e-12, 1e-18)
    high = max(value_range * 1e-6, 1e-12)

    best_path = None
    best_size = None
    best_cr = None
    best_eb = None

    for _ in range(60):
        path, size, cr = trial(high)
        if cr >= target_cr:
            best_path, best_size, best_cr, best_eb = path, size, cr, high
            break
        path.unlink()
        high *= 2.0

    if best_path is None:
        raise RuntimeError("cannot reach target compression ratio")

    for _ in range(60):
        mid = (low + high) / 2.0
        path, size, cr = trial(mid)

        if cr >= target_cr:
            if best_path is not None and best_path.exists():
                best_path.unlink()
            best_path, best_size, best_cr, best_eb = path, size, cr, mid
            high = mid
        else:
            path.unlink()
            low = mid

    return best_eb, best_path, best_size, best_cr


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.yaml 路径")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    input_path = resolve_path(cfg["input"])
    sz3_path = resolve_path(cfg["sz3"])
    target_cr = float(cfg["cr"])
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

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "input.raw"
        raw_out = td / "recon.raw"
        best_cmp_copy = td / "payload.sz"

        arr.tofile(raw_in)

        eb, cmp_path, cmp_size, actual_cr = search_error_bound(
            sz3_path=sz3_path,
            raw_path=raw_in,
            dtype_flag=dtype_flag,
            shape=shape,
            original_nbytes=original_nbytes,
            target_cr=target_cr,
            data_min=float(arr.min()),
            data_max=float(arr.max()),
        )

        meta = {
            "input": str(input_path),
            "loaded_shape": list(loaded_shape),
            "used_shape": list(shape),
            "dtype": dtype_name,
            "sz3_dtype_flag": dtype_flag,
            "target_cr": target_cr,
            "actual_cr": actual_cr,
            "used_error_bound": eb,
            "original_nbytes": original_nbytes,
            "compressed_nbytes": cmp_size,
        }

        os.replace(cmp_path, best_cmp_copy)

        with zipfile.ZipFile(compressed_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(best_cmp_copy, arcname="payload.sz")
            zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

        decompress_once(sz3_path, best_cmp_copy, raw_out, dtype_flag, shape)

        recon = np.fromfile(raw_out, dtype=np_dtype).reshape(shape)
        np.save(recon_path, recon)

    output = {
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "loaded_shape": list(loaded_shape),
        "shape": list(shape),
        "target_cr": target_cr,
        "actual_cr": actual_cr,
        "used_error_bound": eb,
    }

    if result_json is not None:
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
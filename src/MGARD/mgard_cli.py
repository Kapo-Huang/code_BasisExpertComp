import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


REPO_DIR = Path(__file__).resolve().parent


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


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


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    return cfg


def parse_shape(shape_value):
    if shape_value is None:
        return None
    if isinstance(shape_value, str):
        return tuple(int(x) for x in shape_value.split(","))
    if isinstance(shape_value, (list, tuple)):
        return tuple(int(x) for x in shape_value)
    raise ValueError("shape 必须是字符串 '600,248,248' 或列表 [600,248,248]")


def choose_dtype(arr):
    if arr.dtype == np.float32:
        return arr, np.float32, "s", "float32"
    if arr.dtype == np.float64:
        return arr, np.float64, "d", "float64"
    arr = arr.astype(np.float32)
    return arr, np.float32, "s", "float32"


def dtype_from_name(dtype_name):
    name = str(dtype_name).strip().lower()
    if name in ("float32", "single", "s"):
        return np.float32, "s", "float32"
    if name in ("float64", "double", "d"):
        return np.float64, "d", "float64"
    raise ValueError(f"unsupported dtype: {dtype_name}")


def normalize_smoothness(s):
    if isinstance(s, str):
        ss = s.strip().lower()
        if ss in ("inf", "infinity"):
            return "infinity"
        try:
            float(ss)
            return ss
        except ValueError as e:
            raise ValueError(f"invalid smoothness: {s}") from e
    return str(s)


def maybe_resolve_optional_path(cfg, key):
    value = cfg.get(key)
    if value in (None, ""):
        return None
    return resolve_path(value)


def write_json(path, obj):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_common_runtime_args(cfg):
    cmd = []

    device = cfg.get("device")
    if device:
        cmd += ["-d", str(device)]

    verbose = cfg.get("verbose")
    if verbose is not None:
        cmd += ["-v", str(int(verbose))]

    num_devices = cfg.get("num_devices")
    if num_devices is not None:
        cmd += ["-g", str(int(num_devices))]

    prefetch = cfg.get("prefetch")
    if prefetch is not None:
        cmd += ["-h", "1" if bool(prefetch) else "0"]

    return cmd


def build_compress_args(cfg):
    cmd = []

    mode = str(cfg.get("error_mode", "abs")).lower()
    if mode not in ("abs", "rel"):
        raise ValueError("error_mode 必须是 abs 或 rel")
    cmd += ["-em", mode]

    if "error_bound" not in cfg:
        raise ValueError("compress 模式必须提供 error_bound")
    cmd += ["-e", str(cfg["error_bound"])]

    smoothness = normalize_smoothness(cfg.get("smoothness", 0))
    cmd += ["-s", smoothness]

    reorder = cfg.get("reorder")
    if reorder is not None:
        cmd += ["-r", str(int(reorder))]

    block = cfg.get("domain_decomposition")
    if block is not None:
        cmd += ["-b", str(int(block))]

    max_memory_footprint = cfg.get("max_memory_footprint")
    if max_memory_footprint is not None:
        cmd += ["-f", str(int(max_memory_footprint))]

    lossless = cfg.get("lossless")
    if lossless is not None:
        cmd += ["-l", str(int(lossless))]

    coords = cfg.get("coordinates")
    if coords:
        coords_path = resolve_path(coords)
        cmd += ["-u", str(coords_path)]

    return cmd


def infer_mode(cli_mode, cfg):
    if cli_mode:
        return cli_mode
    mode = str(cfg.get("mode", "compress")).lower()
    if mode not in ("compress", "decompress"):
        raise ValueError("mode 必须是 compress / decompress")
    return mode


def load_array_for_compress(cfg):
    input_path = resolve_path(cfg["input"])
    log(f"读取原始数据: {input_path}")
    arr = np.load(input_path, allow_pickle=False)

    if arr.dtype == object:
        raise TypeError("object dtype not supported")

    loaded_shape = tuple(int(x) for x in arr.shape)
    arr, np_dtype, dtype_flag, dtype_name = choose_dtype(arr)

    shape_cfg = parse_shape(cfg.get("shape"))
    if shape_cfg is not None:
        if int(np.prod(shape_cfg)) != int(arr.size):
            raise ValueError(f"shape {shape_cfg} 和数据 size {arr.size} 不匹配")
        arr = arr.reshape(shape_cfg)
        shape = tuple(int(x) for x in shape_cfg)
    else:
        shape = tuple(int(x) for x in arr.shape)

    if len(shape) < 1 or len(shape) > 5:
        raise ValueError(f"MGARD-X CLI supports 1D-5D arrays, got ndim={len(shape)}")

    arr = np.ascontiguousarray(arr)
    log(f"数据读取完成: loaded_shape={loaded_shape}, used_shape={shape}, dtype={dtype_name}, nbytes={arr.nbytes}")
    return {
        "input_path": input_path,
        "arr": arr,
        "loaded_shape": loaded_shape,
        "shape": shape,
        "np_dtype": np_dtype,
        "dtype_flag": dtype_flag,
        "dtype_name": dtype_name,
    }


def load_recon_spec(cfg):
    meta_path = maybe_resolve_optional_path(cfg, "meta_json")
    meta = None
    if meta_path is not None and meta_path.exists():
        log(f"读取 meta_json: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    recon_shape = parse_shape(cfg.get("recon_shape"))
    if recon_shape is None and meta is not None:
        recon_shape = tuple(int(x) for x in meta["used_shape"])
    if recon_shape is None:
        raise ValueError("decompress 模式需要 recon_shape，或者提供存在的 meta_json")

    recon_dtype = cfg.get("recon_dtype")
    if recon_dtype is None and meta is not None:
        recon_dtype = meta["dtype"]
    if recon_dtype is None:
        raise ValueError("decompress 模式需要 recon_dtype，或者提供存在的 meta_json")

    np_dtype, dtype_flag, dtype_name = dtype_from_name(str(recon_dtype))
    return recon_shape, np_dtype, dtype_flag, dtype_name, meta_path, meta


def compress(cfg):
    mgard_path = resolve_path(cfg["mgard_x"])
    compressed_path = resolve_path(cfg["compressed"])
    meta_path = maybe_resolve_optional_path(cfg, "meta_json")
    result_json = maybe_resolve_optional_path(cfg, "result_json")

    ensure_parent(compressed_path)
    if meta_path is not None:
        ensure_parent(meta_path)
    if result_json is not None:
        ensure_parent(result_json)

    info = load_array_for_compress(cfg)
    arr = info["arr"]
    shape = info["shape"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "input.raw"
        log(f"写入临时 raw 输入文件: {raw_in}")
        arr.tofile(raw_in)

        cmd = [
            str(mgard_path),
            "-z",
            "-i", str(raw_in),
            "-o", str(compressed_path),      # 你本地二进制要求 -o
            "-dt", info["dtype_flag"],       # 你本地二进制要求 -dt
            "-dim", str(len(shape)),
            *map(str, shape),
        ]
        cmd += build_compress_args(cfg)
        cmd += build_common_runtime_args(cfg)

        log("开始压缩")
        run(cmd)
        log("压缩完成")

    meta = {
        "input": str(info["input_path"]),
        "loaded_shape": list(info["loaded_shape"]),
        "used_shape": list(shape),
        "dtype": info["dtype_name"],
        "mgard_dtype_flag": info["dtype_flag"],
        "compressed": str(compressed_path),
        "compressed_nbytes": int(os.path.getsize(compressed_path)),
        "error_mode": str(cfg.get("error_mode", "abs")).lower(),
        "error_bound": float(cfg["error_bound"]),
        "smoothness": normalize_smoothness(cfg.get("smoothness", 0)),
        "device": cfg.get("device", "auto"),
        "coordinates": str(resolve_path(cfg["coordinates"])) if cfg.get("coordinates") else None,
    }

    if meta_path is not None:
        log(f"写入 meta_json: {meta_path}")
        write_json(meta_path, meta)

    output = {
        "mode": "compress",
        "compressed": str(compressed_path),
        "compressed_nbytes": int(os.path.getsize(compressed_path)),
        "loaded_shape": list(info["loaded_shape"]),
        "shape": list(shape),
        "dtype": info["dtype_name"],
        "error_mode": meta["error_mode"],
        "error_bound": meta["error_bound"],
        "smoothness": meta["smoothness"],
        "meta_json": str(meta_path) if meta_path is not None else None,
    }

    if result_json is not None:
        log(f"写入 result_json: {result_json}")
        write_json(result_json, output)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def decompress(cfg):
    mgard_path = resolve_path(cfg["mgard_x"])
    compressed_path = resolve_path(cfg["compressed"])
    recon_path = resolve_path(cfg["recon"])
    result_json = maybe_resolve_optional_path(cfg, "result_json")

    ensure_parent(recon_path)
    if result_json is not None:
        ensure_parent(result_json)

    recon_shape, np_dtype, dtype_flag, dtype_name, meta_path, meta = load_recon_spec(cfg)

    log(f"准备解压: {compressed_path}")
    log(f"重建信息: shape={recon_shape}, dtype={dtype_name}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_out = td / "recon.raw"

        cmd = [
            str(mgard_path),
            "-x",
            "-i", str(compressed_path),
            "-o", str(raw_out),
        ]
        cmd += build_common_runtime_args(cfg)

        log("开始解压")
        run(cmd)
        log("解压完成")

        recon = np.fromfile(raw_out, dtype=np_dtype)
        expected = int(np.prod(recon_shape))
        if recon.size != expected:
            raise ValueError(
                f"解压得到的数据元素数为 {recon.size}，但 recon_shape={recon_shape} 需要 {expected} 个元素；"
                "请检查 recon_shape / recon_dtype 或 meta_json 是否正确"
            )
        recon = recon.reshape(recon_shape)

        log(f"写出重建文件: {recon_path}")
        np.save(recon_path, recon)

    output = {
        "mode": "decompress",
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "shape": list(recon_shape),
        "dtype": dtype_name,
        "compressed_nbytes": int(os.path.getsize(compressed_path)),
        "recon_nbytes": int(recon.nbytes),
        "meta_json": str(meta_path) if meta_path is not None else None,
    }

    if result_json is not None:
        log(f"写入 result_json: {result_json}")
        write_json(result_json, output)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.yaml 路径")
    ap.add_argument("--mode", choices=["compress", "decompress"], help="覆盖 YAML 中的 mode")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    log(f"读取配置文件: {config_path}")
    cfg = load_config(config_path)
    mode = infer_mode(args.mode, cfg)
    log(f"运行模式: {mode}")

    if mode == "compress":
        compress(cfg)
    elif mode == "decompress":
        decompress(cfg)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
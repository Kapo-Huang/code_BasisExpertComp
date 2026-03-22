import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("FAILED:", " ".join(map(str, cmd)))
        print(p.stdout)
        print(p.stderr)
        sys.exit(1)
    return p


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    input_path = Path(cfg["input"])
    zfp_path = Path(cfg["zfp"])
    compressed_path = Path(cfg["compressed"])
    recon_path = Path(cfg["recon"])
    shape = tuple(cfg["shape"])
    rate = cfg["rate"]

    ensure_parent(compressed_path)
    ensure_parent(recon_path)

    # 读取数据
    arr = np.load(input_path).astype(np.float32)

    if int(np.prod(shape)) != arr.size:
        raise ValueError("shape 和数据 size 不匹配")

    arr = arr.reshape(shape)
    arr = np.ascontiguousarray(arr)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "in.raw"
        raw_out = td / "out.raw"

        # npy → raw
        arr.tofile(raw_in)

        # 压缩（只用 -r）
        compress_cmd = [
            str(zfp_path),
            "-f",
            f"-{len(shape)}",
            *map(str, shape),
            "-i", str(raw_in),
            "-z", str(compressed_path),
            "-r", str(rate),
            "-h",
        ]

        compress_out = run(compress_cmd)

        # 解压
        decompress_cmd = [
            str(zfp_path),
            "-z", str(compressed_path),
            "-o", str(raw_out),
            "-h",
        ]

        decompress_out = run(decompress_cmd)

        # raw → numpy
        recon = np.fromfile(raw_out, dtype=np.float32).reshape(shape)
        np.save(recon_path, recon)

    # 计算指标
    mse = float(np.mean((arr - recon) ** 2))
    max_err = float(np.max(np.abs(arr - recon)))
    psnr = float(20 * np.log10(np.max(arr)) - 10 * np.log10(mse))

    result = {
        "input": str(input_path),
        "compressed": str(compressed_path),
        "recon": str(recon_path),
        "shape": list(shape),
        "rate": rate,
        "mse": mse,
        "max_error": max_err,
        "psnr": psnr,
        "zfp_log": compress_out.stderr.strip(),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compression_cli_common import (
    build_result,
    build_sz3_compress_command,
    build_sz3_decompress_command,
    ensure_parent,
    format_command_error,
    get_sz3_dtype_args,
    load_array,
    load_psnr_config,
    log_progress,
    run_command,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress data with SZ3 using a target PSNR")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    try:
        config = load_psnr_config(SCRIPT_DIR, args.config, "sz3")
        input_path = config["input_path"]
        sz3_path = config["binary_path"]
        compressed_path = config["compressed_path"]
        recon_path = config["recon_path"]
        result_json_path = config["result_json_path"]
        target_psnr = config["target_psnr"]

        ensure_parent(compressed_path)
        ensure_parent(recon_path)
        if result_json_path is not None:
            ensure_parent(result_json_path)

        log_progress("sz3", f"Loading input array from {input_path}")
        array, loaded_shape, used_shape = load_array(input_path, config["shape"])
        if not 1 <= len(used_shape) <= 4:
            raise ValueError(f"SZ3 supports only 1D-4D arrays, got ndim={len(used_shape)}")

        dtype_args, dtype_label = get_sz3_dtype_args(array.dtype)
        log_progress("sz3", f"Input ready: dtype={dtype_label}, shape={used_shape}")

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            raw_input_path = tmp_dir / "input.raw"
            raw_output_path = tmp_dir / "recon.raw"
            payload_path = tmp_dir / "payload.sz"
            array.tofile(raw_input_path)

            log_progress("sz3", "Running native compression")
            compress_result = run_command(
                build_sz3_compress_command(
                    sz3_path,
                    raw_input_path,
                    payload_path,
                    dtype_args,
                    used_shape,
                    target_psnr,
                )
            )
            log_progress("sz3", f"Compression finished in {compress_result.elapsed_seconds:.3f}s")

            package_meta = {
                "method": "sz3",
                "input": str(input_path),
                "used_shape": list(used_shape),
                "dtype": dtype_label,
                "target_psnr": target_psnr,
                "native_mode": "psnr",
                "native_value": target_psnr,
                "payload": "payload.sz",
            }
            log_progress("sz3", "Packaging compressed payload")
            with zipfile.ZipFile(compressed_path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.write(payload_path, arcname="payload.sz")
                archive.writestr("meta.json", json.dumps(package_meta, indent=2))

            log_progress("sz3", "Running native decompression")
            decompress_result = run_command(
                build_sz3_decompress_command(
                    sz3_path,
                    payload_path,
                    raw_output_path,
                    dtype_args,
                    used_shape,
                )
            )
            log_progress("sz3", f"Decompression finished in {decompress_result.elapsed_seconds:.3f}s")

            reconstructed = np.fromfile(raw_output_path, dtype=array.dtype).reshape(used_shape)
            log_progress("sz3", f"Saving reconstruction to {recon_path}")
            np.save(recon_path, reconstructed)

        result = build_result(
            method="sz3",
            input_path=input_path,
            compressed_path=compressed_path,
            recon_path=recon_path,
            loaded_shape=loaded_shape,
            used_shape=used_shape,
            dtype_label=dtype_label,
            target_mode="psnr",
            target_value=target_psnr,
            target_psnr=target_psnr,
            native_mode="psnr",
            native_value=target_psnr,
            original=array,
            reconstructed=reconstructed,
            compress_result=compress_result,
            decompress_result=decompress_result,
        )

        if result_json_path is not None:
            log_progress("sz3", f"Writing result JSON to {result_json_path}")
            write_json(result_json_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        if hasattr(error, "result"):
            sys.stderr.write(format_command_error(error))
        else:
            sys.stderr.write(f"{error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

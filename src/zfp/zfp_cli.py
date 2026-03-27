import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compression_cli_common import (
    build_result,
    build_zfp_compress_command,
    build_zfp_decompress_command,
    compute_metrics,
    ensure_parent,
    format_command_error,
    get_zfp_dtype_flag,
    load_array,
    load_psnr_config,
    run_command,
    write_json,
    zfp_tolerance_from_psnr,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress data with ZFP using a target PSNR")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    try:
        config = load_psnr_config(SCRIPT_DIR, args.config, "zfp")
        input_path = config["input_path"]
        zfp_path = config["binary_path"]
        compressed_path = config["compressed_path"]
        recon_path = config["recon_path"]
        result_json_path = config["result_json_path"]
        target_psnr = config["target_psnr"]

        ensure_parent(compressed_path)
        ensure_parent(recon_path)
        if result_json_path is not None:
            ensure_parent(result_json_path)

        array, loaded_shape, used_shape = load_array(input_path, config["shape"])
        if not 1 <= len(used_shape) <= 4:
            raise ValueError(f"ZFP supports only 1D-4D arrays, got ndim={len(used_shape)}")

        dtype_flag, dtype_label = get_zfp_dtype_flag(array.dtype)
        data_range = compute_metrics(array, array)["data_range"]
        tolerance = zfp_tolerance_from_psnr(data_range, target_psnr)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            raw_input_path = tmp_dir / "input.raw"
            raw_output_path = tmp_dir / "recon.raw"
            array.tofile(raw_input_path)

            compress_result = run_command(
                build_zfp_compress_command(
                    zfp_path,
                    raw_input_path,
                    compressed_path,
                    dtype_flag,
                    used_shape,
                    tolerance,
                )
            )

            decompress_result = run_command(
                build_zfp_decompress_command(
                    zfp_path,
                    compressed_path,
                    raw_output_path,
                )
            )

            reconstructed = np.fromfile(raw_output_path, dtype=array.dtype).reshape(used_shape)
            np.save(recon_path, reconstructed)

        result = build_result(
            method="zfp",
            input_path=input_path,
            compressed_path=compressed_path,
            recon_path=recon_path,
            loaded_shape=loaded_shape,
            used_shape=used_shape,
            dtype_label=dtype_label,
            target_psnr=target_psnr,
            native_mode="accuracy",
            native_value=tolerance,
            original=array,
            reconstructed=reconstructed,
            compress_result=compress_result,
            decompress_result=decompress_result,
        )

        if result_json_path is not None:
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

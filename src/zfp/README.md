# ZFP Accuracy Wrapper

This wrapper keeps the native `zfp` binary unchanged and exposes a YAML interface that accepts either `psnr` or absolute-error `tolerance`.

## Build

```bash
cd zfp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Expected binary path:

```text
build/bin/zfp
```

## Config

Required keys:

```yaml
input: ../path/to/volRendering_H2.npy
zfp: build/bin/zfp

psnr: 40.0
# tolerance: 0.01
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.zfp
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

Notes:

- External PSNR is always `20 * log10((max(original) - min(original)) / rmse)`.
- `psnr` and `tolerance` are mutually exclusive; provide exactly one.
- ZFP has no native PSNR mode, so when `psnr` is provided the wrapper converts it into native fixed-accuracy mode:
  `tolerance = data_range * 10^(-psnr / 20)`.
- When `tolerance` is provided, the wrapper passes it directly to native `-a`.
- Compression and decompression both use `-h`, so the `.zfp` file keeps the native header.
- Only `float32` and `float64` inputs are supported in this fixed-accuracy wrapper.
- Legacy `rate` is no longer accepted.
- Progress logs are printed to `stderr`; the final JSON result stays on `stdout`.

Tolerance-only example:

```yaml
input: ../path/to/volRendering_H2.npy
zfp: build/bin/zfp
tolerance: 0.01
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.zfp
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

## Run

```bash
python zfp_cli.py --config configs/volRendering_H2.yaml
```

## Result Schema

```json
{
  "method": "zfp",
  "input": "/abs/path/input.npy",
  "compressed": "/abs/path/output.zfp",
  "recon": "/abs/path/recon.npy",
  "loaded_shape": [36902400],
  "used_shape": [600, 248, 248],
  "dtype": "float32",
  "target_mode": "psnr",
  "target_value": 40.0,
  "target_psnr": 40.0,
  "native_mode": "accuracy",
  "native_value": 0.01,
  "measured_psnr": 39.12,
  "mse": 0.00015,
  "rmse": 0.01225,
  "max_error": 0.11,
  "original_nbytes": 147609600,
  "compressed_nbytes": 18451216,
  "compression_ratio": 7.99999306278784,
  "compression_time_seconds": 0.45,
  "decompression_time_seconds": 0.18,
  "total_time_seconds": 0.63,
  "compress_stdout": "",
  "compress_stderr": "",
  "decompress_stdout": "",
  "decompress_stderr": ""
}
```

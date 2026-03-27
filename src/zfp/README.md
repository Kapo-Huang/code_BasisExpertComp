# ZFP PSNR Wrapper

This wrapper keeps the native `zfp` binary unchanged and exposes a PSNR-only YAML interface.

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
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.zfp
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

Notes:

- External PSNR is always `20 * log10((max(original) - min(original)) / rmse)`.
- ZFP has no native PSNR mode, so the wrapper converts the requested PSNR into native fixed-accuracy mode:
  `tolerance = data_range * 10^(-psnr / 20)`.
- Compression and decompression both use `-h`, so the `.zfp` file keeps the native header.
- Only `float32` and `float64` inputs are supported in this PSNR-only wrapper.
- Legacy `rate` is no longer accepted.

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
  "compress_stdout": "",
  "compress_stderr": "",
  "decompress_stdout": "",
  "decompress_stderr": ""
}
```

# TTHRESH PSNR Wrapper

This wrapper keeps the native `tthresh` binary unchanged and exposes the same PSNR-only interface as the SZ3 and ZFP wrappers.

## Build

```bash
cd tthresh
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Expected binary path:

```text
build/tthresh
```

## Config

Required keys:

```yaml
input: ../path/to/volRendering_H2.npy
tthresh: build/tthresh
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.tthresh
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

Notes:

- External PSNR is always `20 * log10((max(original) - min(original)) / rmse)`.
- TTHRESH uses a native PSNR definition with an extra `/ 2` term in the denominator, so the wrapper converts:
  `native_psnr = target_psnr - 20 * log10(2)`.
- TTHRESH requires at least 3 dimensions after reshape.
- Supported dtypes are `uint8`, `uint16`, `int32`, `float32`, and `float64`.
- Progress logs are printed to `stderr`; the final JSON result stays on `stdout`.

## Run

```bash
python tthresh_cli.py --config configs/volRendering_H2.yaml
```

## Result Schema

```json
{
  "method": "tthresh",
  "input": "/abs/path/input.npy",
  "compressed": "/abs/path/output.tthresh",
  "recon": "/abs/path/recon.npy",
  "loaded_shape": [36902400],
  "used_shape": [600, 248, 248],
  "dtype": "float32",
  "target_mode": "psnr",
  "target_value": 40.0,
  "target_psnr": 40.0,
  "native_mode": "psnr",
  "native_value": 33.979400086720375,
  "measured_psnr": 39.90,
  "mse": 0.00010,
  "rmse": 0.01011,
  "max_error": 0.082,
  "original_nbytes": 147609600,
  "compressed_nbytes": 54826,
  "compression_ratio": 2692.32845730128,
  "compression_time_seconds": 0.82,
  "decompression_time_seconds": 0.0,
  "total_time_seconds": 0.82,
  "compress_stdout": "oldbits = ...",
  "compress_stderr": "",
  "decompress_stdout": "",
  "decompress_stderr": ""
}
```

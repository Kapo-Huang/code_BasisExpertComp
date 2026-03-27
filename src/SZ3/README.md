# SZ3 PSNR Wrapper

This wrapper keeps the native `sz3` binary unchanged and exposes a PSNR-only YAML interface.

## Build

```bash
cd SZ3
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Expected binary path:

```text
build/tools/sz3/sz3
```

## Config

Required keys:

```yaml
input: ../path/to/volRendering_H2.npy
sz3: build/tools/sz3/sz3
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.sz3pkg
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

Notes:

- `psnr` always means `20 * log10((max(original) - min(original)) / rmse)`.
- The wrapper calls SZ3 in native PSNR mode: `-M PSNR <psnr>`.
- `.sz3pkg` stores the native `.sz` payload plus wrapper metadata.
- Legacy `cr` is no longer accepted.

## Run

```bash
python sz3_cli.py --config configs/volRendering_H2.yaml
```

## Result Schema

`result_json` and stdout use the same fields:

```json
{
  "method": "sz3",
  "input": "/abs/path/input.npy",
  "compressed": "/abs/path/output.sz3pkg",
  "recon": "/abs/path/recon.npy",
  "loaded_shape": [36902400],
  "used_shape": [600, 248, 248],
  "dtype": "float32",
  "target_psnr": 40.0,
  "native_mode": "psnr",
  "native_value": 40.0,
  "measured_psnr": 39.98,
  "mse": 0.00012,
  "rmse": 0.01095,
  "max_error": 0.083,
  "original_nbytes": 147609600,
  "compressed_nbytes": 15353,
  "compression_ratio": 9614.38155409366,
  "compress_stdout": "",
  "compress_stderr": "",
  "decompress_stdout": "",
  "decompress_stderr": ""
}
```

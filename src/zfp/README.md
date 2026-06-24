# ZFP Wrapper

## 用途

`zfp_cli.py` 为原生 `zfp` 二进制提供统一的 YAML 接口，支持三种目标模式：

- `psnr`
- `tolerance`
- `rate`

## 构建

```bash
cmake -S . -B build
cmake --build build --config Release
```

可执行文件通常位于 `build/bin/zfp`。

## 配置

示例：`configs/volRendering_H2.yaml`

```yaml
input: ../path/to/volRendering_H2.npy
zfp: build/bin/zfp
psnr: 40.0
# tolerance: 0.01
# rate: 8.0
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.zfp
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

说明：

- `psnr`、`tolerance`、`rate` 必须且只能提供一个
- 当使用 `psnr` 时，包装层会换算成原生 `accuracy` 模式
- 当前包装器仅支持 `float32` 与 `float64`

## 运行

```bash
python zfp_cli.py --config configs/volRendering_H2.yaml
```

## 输出

- `compressed`：原生 `.zfp`
- `recon`：解压后的 `.npy`
- `result_json`：统一结构的统计结果

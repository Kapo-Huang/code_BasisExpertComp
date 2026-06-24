# SZ3 Wrapper

## 用途

`sz3_cli.py` 为原生 `sz3` 二进制提供一个基于 YAML 的 PSNR 驱动包装层。

## 构建

```bash
cmake -S . -B build
cmake --build build --config Release
```

可执行文件通常位于 `build/tools/sz3/sz3`。

## 配置

示例：`configs/volRendering_H2.yaml`

```yaml
input: ../path/to/volRendering_H2.npy
sz3: build/tools/sz3/sz3
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.sz3pkg
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

说明：

- 仅接受 `psnr`
- `shape` 用于把输入 `.npy` 重排为原生压缩器所需形状
- 输出目录会自动创建

## 运行

```bash
python sz3_cli.py --config configs/volRendering_H2.yaml
```

## 输出

- `compressed`：包装后的 `.sz3pkg`
- `recon`：解压后的 `.npy`
- `result_json`：统一结构的统计结果

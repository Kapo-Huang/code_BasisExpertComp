# TTHRESH Wrapper

## 用途

`tthresh_cli.py` 为原生 `tthresh` 二进制提供一个基于 YAML 的 PSNR 包装层。

## 构建

```bash
cmake -S . -B build
cmake --build build --config Release
```

可执行文件通常位于 `build/tthresh`。

## 配置

示例：`configs/volRendering_H2.yaml`

```yaml
input: ../path/to/volRendering_H2.npy
tthresh: build/tthresh
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.tthresh
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

说明：

- 仅接受 `psnr`
- 包装层会把外部 PSNR 换算为 `tthresh` 原生定义
- 输入在 reshape 后至少要有 3 个维度

## 运行

```bash
python tthresh_cli.py --config configs/volRendering_H2.yaml
```

## 输出

- `compressed`：原生 `.tthresh`
- `recon`：解压后的 `.npy`
- `result_json`：统一结构的统计结果

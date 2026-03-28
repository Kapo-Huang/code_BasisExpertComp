# SZ3 CLI 压缩工具指南

在 WSL 环境下编译 SZ3。

## 1. 环境与依赖

首先，进入 **WSL 环境**。

**安装 Python 依赖：**

```bash
pip install numpy pyyaml
```

---

## 2. 编译 SZ3

在 `src` 目录下，进入 `SZ3` 源码目录并执行以下命令进行编译：

```bash
cd SZ3
rm -rf build
mkdir build
cd build
cmake ..
make -j
```

> **⚠️ 常见问题 (Troubleshooting):**
> 如果在 `cmake` 或编译过程中卡住或报错，通常是因为缺少系统级依赖。请执行以下命令安装后重新尝试编译：
> ```bash
> sudo apt update
> sudo apt install -y pkg-config libzstd-dev
> ```

编译完成后，回到 SZ3 根目录：
```bash
cd ..
```

---

## 3. 配置文件 (YAML)

运行脚本前，需要准备 YAML 配置文件。

参数说明：
SZ3 支持 ABS / REL / PSNR 这三种误差参数
1. ABS ：控制点对点绝对误差。
也就是每个重建值和原值的差，按绝对值看，不能超过你给的阈值。

2. REL ：value-range-based relative error，也就是误差阈值按全局数据范围来定：
有效绝对误差 = relative × (max - min)。
例如：
数据范围是 [100, 110]，range = 10
REL = 0.01
那有效的绝对误差其实就是 0.1

3. PSNR
PSNR 控制的是整体失真质量，衡量的是 overall data distortion，而不是 maximum error。
适合更关心整体质量

具体参数：
- metric: ABS
- metric: REL
- metric: PSNR
搭配
- metric_value: ...


**示例：** `configs/volRendering_H2.yaml`

```yaml
input: data/volRendering_datasets/volRendering_H2.npy
sz3: build/tools/sz3/sz3
metric: ABS
metric_value: 1e-4
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.sz3pkg
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

---

## 4. 运行测试

在主目录下，通过 Python 调用 CLI 脚本并传入配置文件：

压缩：
```bash
python3 sz3_cli.py --config configs/volRendering_H2.yaml --mode compress
```
解压：
```bash
python3 sz3_cli.py --config configs/volRendering_H2.yaml --mode decompress
```

---

## 5. 输出结果
**示例：**

```json
[16:42:12] 开始读取原始数据: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/data/volRendering_datasets/volRendering_H2.npy (140.77 MiB)
[16:42:13] 读取完成: shape=(36902400,), dtype=float32
[16:42:13] 准备压缩: used_shape=(600, 248, 248), dtype=float32, 原始大小=147609600 bytes
[16:42:13] 写入临时 raw 输入文件
[16:42:13] 临时 raw 输入文件写入完成
[16:42:13] 调用 SZ3 压缩 (metric=ABS, value=0.0001)
[16:42:14] 完成: 调用 SZ3 压缩 (metric=ABS, value=0.0001) (耗时 0.81s)
[16:42:14] 压缩完成: compressed=3511644 bytes, actual_cr=42.0343
[16:42:14] 写出压缩归档: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2.sz3pkg
[16:42:15] 压缩归档写出完成
{
  "mode": "compress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2.sz3pkg",
  "loaded_shape": [
    36902400
  ],
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "metric": "ABS",
  "metric_value": 0.0001,
  "actual_cr": 42.03432922016013
}
```

```json
[16:44:07] 开始读取压缩文件: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2.sz3pkg (3.35 MiB)
[16:44:07] 解压参数: shape=(600, 248, 248), dtype=float32, dtype_flag=-f
[16:44:07] 调用 SZ3 解压
[16:44:07] 完成: 调用 SZ3 解压 (耗时 0.41s)
[16:44:07] 读取重建 raw 并保存为 npy
[16:44:09] 重建文件已保存: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2_recon.npy
{
  "mode": "decompress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2.sz3pkg",
  "recon": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2_recon.npy",
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "metric": "ABS",
  "metric_value": 0.0001,
  "actual_cr": 42.03432922016013
}
```

运行结束后，生成的文件将保存在 `outputs/` 目录中：

* `volRendering_H2.sz3pkg`：SZ3 压缩后的二进制数据包。
* `volRendering_H2_recon.npy`：解压重建后的 numpy 数组数据。
* `volRendering_H2_result.json`：包含压缩率、PSNR、MSE 等评估指标的测试结果。

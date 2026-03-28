# ZFP Compression CLI (Repro Pipeline)

## 1. 环境要求

* Linux / WSL
* Python ≥ 3.8
* `numpy`
* `pyyaml`
* `cmake`

**安装依赖：**

```bash
pip3 install numpy pyyaml
```

---

## 2. 编译 zfp

```bash
cd zfp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**生成路径：** `build/bin/zfp`

---

## 3. 项目结构

```text
zfp/
├── data/
│   └── volRendering_datasets/
│       └── *.npy
│
├── configs/
│   └── *.yaml
│
├── outputs/
│
├── build/
│   └── bin/
│       └── zfp
│
├── zfp_cli.py
└── README.md
```

---

## 4. 配置文件（YAML）

**示例：** `configs/volRendering_H2.yaml`
```yaml
input: data/volRendering_datasets/volRendering_H2.npy
zfp: build/bin/zfp

compressed: outputs/volRendering_H2.zfp
recon: outputs/volRendering_H2_recon.npy

shape: [600, 248, 248]

rate: 4
```

---

## 5. 运行

在 `zfp/` 目录执行以下命令：

压缩：
```bash
python3 zfp_cli.py --config configs/volRendering_H2.yaml --mode compress
```
解压：
```bash
python3 zfp_cli.py --config configs/volRendering_H2.yaml --mode decompress
```

---

## 6. 输出结果

**示例：**

```json
[16:57:04] 开始读取原始数据: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/data/volRendering_datasets/volRendering_H2.npy (140.77 MiB)
[16:57:06] 读取完成: shape=(36902400,), dtype=float32
[16:57:06] 准备压缩: used_shape=(600, 248, 248), dtype=float32, rate=0.0001
[16:57:06] 写入临时 raw 输入文件
[16:57:06] 临时 raw 输入文件写入完成
[16:57:06] 调用 ZFP 压缩
[16:57:06] 完成: 调用 ZFP 压缩 (耗时 0.22s)
[16:57:06] 写出压缩归档: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/outputs/volRendering_H2.zfp
[16:57:06] 压缩归档写出完成
{
  "mode": "compress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/outputs/volRendering_H2.zfp",
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "rate": 0.0001
}
```

```json
[16:57:04] 开始读取原始数据: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/data/volRendering_datasets/volRendering_H2.npy (140.77 MiB)
[16:57:06] 读取完成: shape=(36902400,), dtype=float32
[16:57:06] 准备压缩: used_shape=(600, 248, 248), dtype=float32, rate=0.0001
[16:57:06] 写入临时 raw 输入文件
[16:57:06] 临时 raw 输入文件写入完成
[16:57:06] 调用 ZFP 压缩
[16:57:06] 完成: 调用 ZFP 压缩 (耗时 0.22s)
[16:57:06] 写出压缩归档: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/outputs/volRendering_H2.zfp
[16:57:06] 压缩归档写出完成
{
  "mode": "compress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/zfp/outputs/volRendering_H2.zfp",
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "rate": 0.0001
}
```

---

## 7. rate参数说明

`rate` 表示 **每个数据值分配的压缩位数**，单位是 **bits/value**，在命令行里对应 ZFP 的 `-r` 选项。

### 1. 基本含义

ZFP 会把数据按块压缩。对于 `d` 维数据，每个块包含 `4^d` 个值；每个块使用固定数量的压缩比特
也就是说，`rate` 越大，每个值保留的比特越多，重建质量通常越高，但压缩率越低；`rate` 越小，压缩越强，但误差通常越大。


### 2. rate不是任意小都有效

对于 floating-point 数据，ZFP 每个块至少还要存一些控制信息，例如空块标志和共同指数，因此每块的 `maxbits` 有最小限制：

* `float32`：`maxbits >= 9`
* `float64`：`maxbits >= 12` 

所以最小可用 `rate` 为：

* `float32`： 
  {min-rate} = {9}/{4^d}
  
* `float64`：
  {min-rate} = {12}/{4^d}
  

例如，对 **3D float32** 数据，`d=3`，每块有 `4^3 = 64` 个值，因此：

{min rate} = 9/64 = 0.140625


这意味着如果你设置：

* `rate = 0.01`
* `rate = 0.0001`

它们其实都低于最小可支持值，最终会落到同一个实际有效 rate：0.140625，压缩和解压结果完全一样。

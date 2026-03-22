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

```bash
python3 zfp_cli.py --config configs/volRendering_H2.yaml
```

---

## 6. 输出结果

**示例：**

```json
{
  "input": "data/volRendering_datasets/volRendering_H2.npy",
  "compressed": "outputs/volRendering_H2.zfp",
  "recon": "outputs/volRendering_H2_recon.npy",
  "shape": [
    600,
    248,
    248
  ],
  "rate": 4,
  "mse": 2.287594270455884e-06,
  "max_error": 0.2856958508491516,
  "psnr": 56.40620999864703,
  "zfp_log": "type=float nx=600 ny=248 nz=248 nw=1 raw=147609600 zfp=18451216 ratio=8 rate=4"
}
```

---

## 7. 参数说明

```yaml
rate: 4
```

**取值：**

| rate | 压缩率  | 质量 |
| :--- | :---- | :--- |
| 16   | ~2x   | 高   |
| 8    | ~4x   | 好   |
| 4    | ~8x   | 中   |
| 2    | ~16x  | 差   |
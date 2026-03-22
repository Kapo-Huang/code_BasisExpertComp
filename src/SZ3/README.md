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

运行脚本前，需要准备 YAML 配置文件。注意：cr参数是压缩率可自行调节

**示例：** `configs/volRendering_H2.yaml`

```yaml
input: data/volRendering_datasets/volRendering_H2.npy
sz3: build/tools/sz3/sz3
cr: 10000
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.sz3pkg
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```

---

## 4. 运行测试

在主目录下，通过 Python 调用 CLI 脚本并传入配置文件：

```bash
python3 sz3_cli.py --config configs/volRendering_H2.yaml
```

---

## 5. 输出结果
**示例：**

```json
{
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2.sz3pkg",
  "recon": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/SZ3/outputs/volRendering_H2_recon.npy",
  "loaded_shape": [
    36902400
  ],
  "shape": [
    600,
    248,
    248
  ],
  "target_cr": 10000.0,
  "actual_cr": 10025.782788833798,
  "used_error_bound": 0.5364989910958684
}
```
运行结束后，生成的文件将保存在 `outputs/` 目录中：

* `volRendering_H2.sz3pkg`：SZ3 压缩后的二进制数据包。
* `volRendering_H2_recon.npy`：解压重建后的 numpy 数组数据。
* `volRendering_H2_result.json`：包含压缩率、PSNR、MSE 等评估指标的测试结果。

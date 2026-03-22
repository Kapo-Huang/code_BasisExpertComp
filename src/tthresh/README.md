# TTHRESH CLI 压缩工具

## 1. 编译 TTHRESH

在 `src` 目录下，进入 `tthresh` 源码目录并执行以下命令：

```bash
cd tthresh
rm -rf build
mkdir build
cd build
cmake .. 
make -j
```

> **⚠️ 常见问题 (Troubleshooting):**
> 如果运行 `cmake ..` 时发生报错，通常是由于 CMake 版本策略引起的。请尝试加上版本最低要求参数重新配置：
> ```bash
> cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
> ```

编译完成后，回到 `tthresh` 根目录：

```bash
cd ..
```

---

## 2. 安装依赖

请确保你的 Python 环境已安装以下依赖包：

```bash
pip install numpy pyyaml
```

---

## 3. 配置文件 (YAML)

在运行前编写 YAML 配置文件。psnr越小，压缩率越高。通过调节psnr，控制压缩率

**示例：** `configs/volRendering_H2.yaml`

```yaml
input: data/volRendering_datasets/volRendering_H2.npy
tthresh: build/tthresh
psnr: 40
shape: [600, 248, 248]

compressed: outputs/volRendering_H2.tthresh
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json
```
*(注：这里的 `input` 使用了相对路径引用了 SZ3 目录下的测试数据，请确保该路径下的数据真实存在。)*

---

## 4. 运行测试

在主目录下，运行 Python 脚本并指定配置文件：

```bash
python3 tthresh_cli.py --config configs/volRendering_H2.yaml
```

---

## 5. 输出结果

**示例：**

```json
{
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2.tthresh",
  "recon": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2_recon.npy",
  "loaded_shape": [
    36902400
  ],
  "used_shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "psnr": 40.0,
  "stdout": "oldbits = 1180876800, newbits = 438608, compressionratio = 2692.33, bpv = 0.0118856\neps = 0.0101889, rmse = 0.0101154, psnr = 39.9003"
}
```

执行完毕后，所有生成的文件将保存在 `outputs/` 目录中：

* `volRendering_H2.tthresh`：TTHRESH 压缩后的文件。
* `volRendering_H2_recon.npy`：解压并重建后的 NumPy 数组数据。
* `volRendering_H2_result.json`：包含压缩耗时、PSNR 以及其他误差指标的测试结果。


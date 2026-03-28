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
压缩：
```bash
python3 tthresh_cli.py --config configs/volRendering_H2.yaml --mode compress
```
解压：
```bash
python3 tthresh_cli.py --config configs/volRendering_H2.yaml --mode decompress
```
---

## 5. 输出结果

**示例：**

```json
[16:51:54] 开始读取原始数据: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/data/volRendering_datasets/volRendering_H2.npy (140.77 MiB)
[16:51:56] 读取完成: shape=(36902400,), dtype=float32
[16:51:56] 准备压缩: used_shape=(600, 248, 248), dtype=float32, psnr=40.0
[16:51:56] 写入临时 raw 输入文件
[16:51:56] 临时 raw 输入文件写入完成
[16:51:56] 调用 TTHRESH 压缩
[16:52:03] 完成: 调用 TTHRESH 压缩 (耗时 7.53s)
[16:52:03] 写出压缩归档: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2.tthresh
[16:52:03] 压缩归档写出完成
{
  "mode": "compress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2.tthresh",
  "loaded_shape": [
    36902400
  ],
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "psnr": 40.0
}
```

```json
[16:52:47] 开始读取压缩文件: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2.tthresh (0.05 MiB)
[16:52:47] 调用 TTHRESH 解压
[16:52:49] 完成: 调用 TTHRESH 解压 (耗时 1.60s)
[16:52:49] 读取重建 raw 并保存为 npy
[16:52:50] 重建文件已保存: /mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2_recon.npy
{
  "mode": "decompress",
  "compressed": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2.tthresh",
  "recon": "/mnt/d/Lab/data_compression/code_BasisExpertComp/src/tthresh/outputs/volRendering_H2_recon.npy",
  "shape": [
    600,
    248,
    248
  ],
  "dtype": "float32",
  "psnr": 40.0
}
```


执行完毕后，所有生成的文件将保存在 `outputs/` 目录中：

* `volRendering_H2.tthresh`：TTHRESH 压缩后的文件。
* `volRendering_H2_recon.npy`：解压并重建后的 NumPy 数组数据。
* `volRendering_H2_result.json`：包含压缩耗时、PSNR 以及其他误差指标的测试结果。


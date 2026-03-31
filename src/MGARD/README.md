# MGARD CLI Wrapper (For structured Dataset)

- 从 **一个 YAML 文件** 读取参数
- 通过 `mode` 在 `compress` / `decompress` 两种模式之间切换
- 支持进度信息输出
- 输入输出统一用 `.npy`
- 内部自动转成 MGARD-X CLI 需要的 raw 二进制数据

---

## 1. 功能

```yaml
mode: compress
```

或者：

```yaml
mode: decompress
```

### compress

- 输入：原始 `.npy`
- 输出：压缩文件 `.mgard`
- 可选输出：
  - `meta_json`
  - `result_json`

### decompress

- 输入：压缩文件 `.mgard`
- 输出：重建文件 `.npy`
- 需要能确定重建数组的 `shape` 和 `dtype`


---

## 2. 进度信息

脚本会在关键阶段打印进度，例如：

- 读取配置文件
- 读取原始数据
- 写入临时 raw 文件
- 开始压缩 / 解压
- 写出 `meta_json`
- 写出 `result_json`
- 写出重建 `.npy`

如果 MGARD-X 本身有输出，脚本也会实时透传到终端，而不是等命令结束后再一次性打印。

---

## 3. 依赖

### Python 依赖

```bash
pip install numpy pyyaml
```

### Ubuntu / WSL 常用构建依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  zlib1g-dev \
  libzstd-dev \
  libprotobuf-dev \
  libprotoc-dev \
  protobuf-compiler \
  libtclap-dev
```

---

## 4. 构建 MGARD-X

在 MGARD 仓库根目录执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMGARD_ENABLE_SERIAL=ON \
  -DMGARD_ENABLE_OPENMP=ON \
  -DMGARD_ENABLE_CLI=ON

cmake --build build -j"$(nproc)"
```

得到的可执行文件路径：

```bash
./build/bin/mgard-x
```

---

## 5. 文件说明

- `mgard_cli.py`：主脚本
- `/configs/####.yaml`：配置文件

---

## 6. YAML 示例

```yaml
# 单一配置文件，通过修改 mode 在 compress / decompress 两种模式之间切换
mode: compress   # 可选: compress / decompress

mgard_x: ./build/bin/mgard-x

# compress 模式使用下面这些字段
input: ./data/volRendering_H2.npy
compressed: ./outputs/volRendering_H2.mgard
meta_json: ./outputs/volRendering_H2_meta.json
result_json: ./outputs/volRendering_H2_result.json

shape: [600, 248, 248]

error_mode: abs
error_bound: 1e-2
smoothness: 0

# decompress 模式至少需要这些字段
# - compressed
# - recon
# 并且二选一:
#   1) meta_json 已存在
#   2) 手动提供 recon_shape + recon_dtype
recon: ./outputs/volRendering_H2_recon.npy

# 如果没有 meta_json，就取消下面两行注释
# recon_shape: [600, 248, 248]
# recon_dtype: float32

# 通用运行参数
device: openmp
verbose: 2
num_devices: 1
prefetch: true

# 以下主要用于 compress
reorder: 0
domain_decomposition: 0
max_memory_footprint:
lossless: 2
coordinates:
```

---

## 7. 运行命令

```bash
python3 mgard_cli.py --config mgard_example.yaml
```

你要压缩时，把 `mode` 改成：

```yaml
mode: compress
```

你要解压时，把 `mode` 改成：

```yaml
mode: decompress
```

也可以用命令行临时覆盖：

```bash
python3 mgard_cli.py --config configs/volRendering_H2.yaml --mode compress
python3 mgard_cli.py --config configs/volRendering_H2.yaml --mode decompress
```

---

## 8. 参数说明

官方只提供：
- abs/rel ：绝对误差 / 相对误差
- error bound ：允许的最大误差
- smoothness ：平滑度参数

没有 --psnr、--mse 这类输入选项。

使用示例 yaml文件：

```yaml
error_mode: abs
error_bound: 1e-2
smoothness: 0
```
error_mode可选：abs/rel
abs/rel 决定“误差怎么定义”
error_bound 误差上限的数值



# MGARD 非结构化数据（Unstructured Dataset）支持说明

##  重要说明

- MGARD **不提供**非结构化数据的命令行工具（CLI）
- 非结构化支持属于 **实验性功能**
- 仅提供 **C++库接口（library API）**
- 必须 **自行编写 C++ 代码调用**

---

## 1. 支持范围

结构化数据（规则网格 / volumetric） 完全支持（有 CLI：mgard-x） 
非结构化网格（mesh） 实验性（仅 C++，无 CLI）

---

## 2. 依赖安装

必须安装以下系统依赖：

```bash
sudo apt update
sudo apt install libblas-dev liblapack-dev
sudo apt install gfortran
````

---

## 3. 安装 MOAB（必须）

MOAB 是 MGARD 非结构化支持的核心依赖。

```bash
git clone https://bitbucket.org/fathomteam/moab.git
cd moab

mkdir build && cd build

cmake .. \
  -DENABLE_MPI=OFF \
  -DENABLE_HDF5=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j
sudo make install
```

---

## 4. 编译 MGARD（开启 unstructured）

```bash
cd MGARD

cmake -S . -B build-unstructured \
  -DCMAKE_BUILD_TYPE=Release \
  -DMGARD_ENABLE_SERIAL=ON \
  -DMGARD_ENABLE_UNSTRUCTURED=ON

cmake --build build-unstructured -j
```

---

## 5. 编译结果验证

必须看到：

```text
MOAB:      1
unstructured: ON
```

同时你会看到：

```text
CLI: OFF
```

不会生成：

```bash
build/bin/mgard-x
```

---

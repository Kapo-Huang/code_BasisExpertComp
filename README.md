# Classic Compression Toolkit

面向 `SZ3`、`ZFP`、`TTHRESH` 的压缩运行与验证仓库。

## 项目内容

- 保留三个压缩器的上游源码目录：`src/SZ3`、`src/zfp`、`src/tthresh`
- 保留统一的 Python 包装层：`src/SZ3/sz3_cli.py`、`src/zfp/zfp_cli.py`、`src/tthresh/tthresh_cli.py`
- 保留公共工具与测试：`src/compression_cli_common.py`、`src/validate_compression_render.py`、`src/tests`
- 已移除 INR 训练/推理代码、旧实验配置、样例输出和构建缓存

## 目录结构

```text
classicCompression/
├─ README.md
├─ requirements.txt
└─ src/
   ├─ compression_cli_common.py
   ├─ validate_compression_render.py
   ├─ tests/
   ├─ SZ3/
   ├─ zfp/
   └─ tthresh/
```

## 环境

运行时依赖：

```bash
pip install -r requirements.txt
```

开发/测试依赖：

```bash
pip install pytest
```

`src/validate_compression_render.py` 依赖外部渲染与图像评估脚本；如果要计算 `LPIPS`，还需要在对应环境中额外安装其脚本所需依赖（通常包括 `torch`、`lpips` 及渲染侧依赖）。

## 构建压缩器

### SZ3

```bash
cmake -S src/SZ3 -B src/SZ3/build
cmake --build src/SZ3/build --config Release
```

可执行文件通常位于 `src/SZ3/build/tools/sz3/sz3`（Windows 下为 `sz3.exe`）。

### ZFP

```bash
cmake -S src/zfp -B src/zfp/build
cmake --build src/zfp/build --config Release
```

可执行文件通常位于 `src/zfp/build/bin/zfp`（Windows 下为 `zfp.exe`）。

### TTHRESH

```bash
cmake -S src/tthresh -B src/tthresh/build
cmake --build src/tthresh/build --config Release
```

可执行文件通常位于 `src/tthresh/build/tthresh`（Windows 下为 `tthresh.exe`）。

## 运行压缩

三个包装器都会：

- 读取 `.npy` 输入
- 调用原生压缩器完成压缩/解压
- 输出重建 `.npy`
- 生成统一结构的结果 JSON

输出目录会按需自动创建，默认不再跟踪到 Git。

### SZ3

```bash
python src/SZ3/sz3_cli.py --config src/SZ3/configs/volRendering_H2.yaml
```

配置字段：

```yaml
input: ../path/to/data.npy
sz3: build/tools/sz3/sz3
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/case.sz3pkg
recon: outputs/case_recon.npy
result_json: outputs/case_result.json
```

### ZFP

```bash
python src/zfp/zfp_cli.py --config src/zfp/configs/volRendering_H2.yaml
```

配置字段：

```yaml
input: ../path/to/data.npy
zfp: build/bin/zfp
psnr: 40.0
# tolerance: 0.01
# rate: 8.0
shape: [600, 248, 248]

compressed: outputs/case.zfp
recon: outputs/case_recon.npy
result_json: outputs/case_result.json
```

`psnr`、`tolerance`、`rate` 三选一。

### TTHRESH

```bash
python src/tthresh/tthresh_cli.py --config src/tthresh/configs/volRendering_H2.yaml
```

配置字段：

```yaml
input: ../path/to/data.npy
tthresh: build/tthresh
psnr: 40.0
shape: [600, 248, 248]

compressed: outputs/case.tthresh
recon: outputs/case_recon.npy
result_json: outputs/case_result.json
```

## 结果验证

`src/validate_compression_render.py` 用于：

1. 从压缩结果 JSON 发现 artifact
2. 调用对应原生压缩器解压
3. 调用外部 `render_task.py` 渲染预测图和 GT 图
4. 调用外部 `image_level_validation.py` 计算 `PSNR/SSIM/LPIPS`

该脚本不再硬编码旧 INR/Vis 仓库路径，运行时必须显式传入外部资源路径。

### artifact 目录约定

建议先把待验证文件集中到一个目录，例如：

```text
artifacts/
├─ target_GT_SZ3_result.json
├─ target_GT.sz3pkg
├─ target_HPlus_ZFP_result.json
├─ target_HPlus.zfp
└─ ...
```

### 示例命令

```bash
python src/validate_compression_render.py \
  --artifacts-root artifacts \
  --gt-root /path/to/gt \
  --result-root /path/to/render_results \
  --render-script /path/to/render_task.py \
  --image-validation-script /path/to/image_level_validation.py \
  --transfer-function-root /path/to/render_config/Ionization
```

可选参数：

- `--tmp-root`：临时目录；默认是 `<result-root>/.tmp/compression_render`
- `--viewport-root`：视口配置根目录；默认复用 `--transfer-function-root`
- `--cases`、`--methods`：过滤 case 和压缩器
- `--timestamp` / `--timestamps`：只验证指定时间步
- `--keep-temp`：保留中间文件

## 测试

```bash
pytest src/tests
```

当前测试覆盖：

- 配置解析
- 压缩命令构造
- `ZFP` 包装器模式切换
- `validate_compression_render.py` 的 artifact 发现、缓存判定和路径解析

## 说明

- 本仓库不再包含 INR 训练/推理能力。
- 构建目录、输出目录和压缩产物默认视为本地产物，不再纳入版本控制。
- 三个压缩器目录仍保留完整上游源码树，便于后续编译、比对和二次开发。

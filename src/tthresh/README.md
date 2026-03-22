# TTHRESH CLI

在 src 目录下：

cd tthresh
rm -rf build
mkdir build
cd build
cmake .. 
make -j

如果运行 cmake .. 报错，可尝试换成 cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5

回到 tthresh 目录：

cd ..

安装依赖：

pip install numpy pyyaml

写配置文件（configs/volRendering_H2.yaml）：

input: ../SZ3/build/data/volRendering_datasets/volRendering_H2.npy
tthresh: build/tthresh
psnr: 40
shape: [600, 248, 248]
compressed: outputs/volRendering_H2.tthresh
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json

运行：

python3 tthresh_cli.py --config configs/volRendering_H2.yaml

结果在：

outputs/volRendering_H2.tthresh
outputs/volRendering_H2_recon.npy
outputs/volRendering_H2_result.json
# SZ3 CLI 

进入wsl
在 src 目录下：

cd SZ3
rm -rf build
mkdir build
cd build
cmake ..
make -j

上述步骤如果卡住了，可尝试： 
sudo apt update
sudo apt install -y pkg-config libzstd-dev

回到 SZ3 目录：
cd ..

安装依赖：

pip install numpy pyyaml

写配置文件
以 configs/volRendering_H2.yaml 为例：

input: build/data/volRendering_datasets/volRendering_H2.npy
sz3: build/tools/sz3/sz3
cr: 10000
shape: [600, 248, 248]
compressed: outputs/volRendering_H2.sz3pkg
recon: outputs/volRendering_H2_recon.npy
result_json: outputs/volRendering_H2_result.json

运行：

python3 sz3_cli.py --config configs/volRendering_H2.yaml

结果保存在：

outputs/volRendering_H2.sz3pkg
outputs/volRendering_H2_recon.npy
outputs/volRendering_H2_result.json
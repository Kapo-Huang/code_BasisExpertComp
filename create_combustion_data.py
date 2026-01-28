import numpy as np
import os
# 创建combustion模拟数据
data_dir = "data/combustion/train/"
os.makedirs(data_dir, exist_ok=True)

# 4个变量，每个3个时间步，分辨率 32x32x32
variables = ['velocity', 'rate', 'mixture', 'OH']

for var in variables:
    # 创建数据 [3, 32, 32, 32]
    data = np.random.randn(3, 32, 32, 32).astype(np.float32)
    file_path = f"{data_dir}target_{var}_sub.npy"
    np.save(file_path, data)
    print(f"创建: {file_path}, 形状: {data.shape}")

print("\n✅ 完成！数据保存在: data/combustion/train/")

import torch
import torch.nn as nn
import numpy as np
import math

# ==========================================
# 1. 基础层定义 (Layers & Blocks)
# ==========================================

class SineLayer(nn.Module):
    """Standard Sine Layer for SIREN-based networks."""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ResBlock(nn.Module):
    """Residual Block: Input -> Sine -> Sine -> Residual (+ Projection)"""
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.net = []
        
        # 内部层 is_first=False
        self.net.append(SineLayer(in_features, out_features, is_first=False))
        self.net.append(SineLayer(out_features, out_features, is_first=False))
        self.net = nn.Sequential(*self.net)

        self.flag = in_features != out_features
        if self.flag:
            self.transform = SineLayer(in_features, out_features, is_first=False)
        
    def forward(self, features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        # 0.5 系数用于稳定深层训练
        return 0.5 * (outputs + features)


# ==========================================
# 2. 核心网络架构 (CoordNetB)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies=6, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.out_dim = in_features * (int(include_input) + 2 * num_frequencies)
        
        freq_bands = 2.0 ** torch.arange(num_frequencies) * math.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        angles = x.unsqueeze(-1) * self.freq_bands
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        encoded = encoded.reshape(x.shape[0], -1)
        if self.include_input:
            return torch.cat([x, encoded], dim=-1)
        return encoded


class CoordNetB(nn.Module):
    """
    MC-INR Core Network: Branched CoordNet
    GFE (5 layers) + LFE (6 layers)
    """
    def __init__(self, cluster_id, in_features=4, out_vars=3, hidden_features=64, gfe_layers=5, lfe_layers=6):
        super().__init__()
        self.cluster_id = cluster_id
        
        # 1. PE
        self.pe = PositionalEncoding(in_features, num_frequencies=6, include_input=True)
        pe_dim = self.pe.out_dim

        # 2. GFE (5 layers total)
        gfe_net = []
        gfe_net.append(ResBlock(pe_dim, hidden_features)) # Layer 1
        for _ in range(gfe_layers - 1):
            gfe_net.append(ResBlock(hidden_features, hidden_features)) # Layer 2-5
        self.gfe = nn.Sequential(*gfe_net)

        # 3. LFE (6 layers total)
        self.lfes = nn.ModuleList()
        for _ in range(out_vars):
            lfe_net = []
            # Layer 1-5
            for _ in range(lfe_layers - 1):
                lfe_net.append(ResBlock(hidden_features, hidden_features))
            # Layer 6 (Output)
            lfe_net.append(nn.Linear(hidden_features, 1))
            self.lfes.append(nn.Sequential(*lfe_net))

    def forward(self, coords):
        h0 = self.pe(coords)
        h_global = self.gfe(h0)
        outputs = [lfe(h_global) for lfe in self.lfes]
        return torch.cat(outputs, dim=-1)


# ==========================================
# 3. 管理器模型 (MC-INR Manager)
# ==========================================

class mc_inr(nn.Module):
    """
    MC-INR 主模型
    功能：
    1. 维护多个 CoordNetB。
    2. 维护每个 Cluster 的 Centroids (用于推理时路由)。
    3. 支持动态分裂。
    """
    def __init__(
        self,
        cluster_num: int,
        initial_centroids: torch.Tensor, # Must provide initial centroids [K, 3]
        in_features: int,
        out_vars: int,
        hidden_features: int = 64,
        gfe_layers: int = 5,
        lfe_layers: int = 6,
    ):
        super().__init__()
        
        self.hyperparams = {
            "in_features": in_features,
            "out_vars": out_vars,
            "hidden_features": hidden_features,
            "gfe_layers": gfe_layers,
            "lfe_layers": lfe_layers,
        }
        
        # 注册 centroids 为 buffer，使其随模型保存 (state_dict)
        # initial_centroids shape: [K, 3] (x, y, z)
        self.register_buffer("centroids", initial_centroids.clone())
        
        # 初始化 K 个网络
        self.clusters = nn.ModuleList([
            CoordNetB(cluster_id=i, **self.hyperparams)
            for i in range(cluster_num)
        ])

    def forward(self, coords, cluster_idx=None):
        """
        Args:
            coords: [B, 4] (x, y, z, t)
            cluster_idx: [B] Optional. 
                         If provided (Training), use labels directly.
                         If None (Inference), find nearest centroid.
        """
        # --- Inference Mode (Calculate Nearest Centroid) ---
        if cluster_idx is None:
            # 提取空间坐标 [B, 3]
            spatial_coords = coords[:, :3] 
            
            # 计算距离: ||x - mu_i||^2
            # spatial_coords: [B, 3]
            # centroids:      [K, 3]
            # cdist 计算每对点之间的距离
            dists = torch.cdist(spatial_coords, self.centroids)
            
            # 找到最近的 cluster index
            # 注意：已分裂的父节点 centroids 会被设为 Inf，不会被选中
            cluster_idx = torch.argmin(dists, dim=1)

        # --- Forward Pass ---
        outputs = torch.zeros(
            coords.shape[0], 
            self.hyperparams['out_vars'], 
            device=coords.device
        )
        
        # 遍历所有有效的 cluster 进行计算
        unique_indices = torch.unique(cluster_idx)
        for i in unique_indices:
            i = i.item()
            if i < len(self.clusters):
                mask = (cluster_idx == i)
                if mask.any():
                    outputs[mask] = self.clusters[i](coords[mask])
                    
        return outputs

    def split_specific_cluster(self, target_idx, new_centroids):
        """
        动态分裂机制：
        1. 继承权重。
        2. 更新 Centroids 列表。
        
        Args:
            target_idx: 被分裂的旧 Cluster 索引
            new_centroids: [2, 3] Tensor, 两个新子 Cluster 的中心点
        """
        old_model = self.clusters[target_idx]
        current_count = len(self.clusters)
        
        # 1. 创建两个新网络
        new_net_1 = CoordNetB(cluster_id=current_count, **self.hyperparams)
        new_net_2 = CoordNetB(cluster_id=current_count + 1, **self.hyperparams)
        
        # 2. 继承参数
        new_net_1.load_state_dict(old_model.state_dict())
        new_net_2.load_state_dict(old_model.state_dict())
        
        # 3. 移动到正确设备
        try:
            device = next(old_model.parameters()).device
        except:
            device = torch.device('cpu')
        new_net_1.to(device)
        new_net_2.to(device)
        
        # 4. 追加到 ModuleList
        self.clusters.append(new_net_1)
        self.clusters.append(new_net_2)
        
        # 5. 更新 Centroids Buffer
        # 确保 new_centroids 在同一设备
        new_centroids = new_centroids.to(self.centroids.device)
        
        # 将旧的 target_idx 的 centroid 设为 Inf，使其在推理时永远不被选中
        self.centroids[target_idx] = float('inf')
        
        # 追加新的 centroids
        self.centroids = torch.cat([self.centroids, new_centroids], dim=0)
        
        return current_count, current_count + 1
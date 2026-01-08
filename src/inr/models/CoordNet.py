import math
import torch
import torch.nn as nn

from .siren import SineLayer


class SirenResBlock(nn.Module):
    """
    对应你参考代码里的 ResBlock（sine 版本）：
      - 两个 SineLayer 串联
      - 如果 in_dim != out_dim，再加一个 transform（SineLayer）
      - forward: y = 0.5 * (outputs + features_or_transformed)
    """
    def __init__(self, in_dim: int, out_dim: int,
                 omega_0: float = 30.0,
                 first_layer: bool = False):
        super().__init__()

        # 主分支：两个 SineLayer
        self.fc1 = SineLayer(
            in_features=in_dim,
            out_features=out_dim,
            is_first=first_layer,
            omega_0=omega_0,
        )
        self.fc2 = SineLayer(
            in_features=out_dim,
            out_features=out_dim,
            is_first=False,
            omega_0=omega_0,
        )

        # 是否需要改变通道数
        self.need_transform = (in_dim != out_dim)
        if self.need_transform:
            # 这里也用 SineLayer，对齐你参考的 ResBlock 实现
            self.transform = SineLayer(
                in_features=in_dim,
                out_features=out_dim,
                is_first=False,      # 一般不把这个当 first layer
                omega_0=omega_0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支
        out = self.fc1(x)
        out = self.fc2(out)

        # 残差分支
        if self.need_transform:
            res = self.transform(x)
        else:
            res = x

        # 对齐：return 0.5 * (outputs + features)
        return 0.5 * (out + res)


class CoordNet(nn.Module):
    """
    CoordNet 风格的 SIREN 残差网络：
      - in_dim -> init_features -> 2*init_features -> 4*init_features
      - 若干个 4*init_features -> 4*init_features 的残差块
      - 最后一个 4*init_features -> out_dim 的残差块
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 24,
        init_features: int = 64,
        num_res: int = 10,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        blocks = []

        # 对应 CoordNet:
        # self.net.append(ResBlock(in_features, init_features))
        blocks.append(
            SirenResBlock(
                in_dim=in_features,
                out_dim=init_features,
                omega_0=first_omega_0,
                first_layer=True,   # 第一个 block 视为 first layer
            )
        )

        # self.net.append(ResBlock(init_features, 2*init_features))
        blocks.append(
            SirenResBlock(
                in_dim=init_features,
                out_dim=2 * init_features,
                omega_0=hidden_omega_0,
                first_layer=False,
            )
        )

        # self.net.append(ResBlock(2*init_features, 4*init_features))
        blocks.append(
            SirenResBlock(
                in_dim=2 * init_features,
                out_dim=4 * init_features,
                omega_0=hidden_omega_0,
                first_layer=False,
            )
        )

        # for i in range(self.num_res):
        #     self.net.append(ResBlock(4*init_features,4*init_features))
        for _ in range(num_res):
            blocks.append(
                SirenResBlock(
                    in_dim=4 * init_features,
                    out_dim=4 * init_features,
                    omega_0=hidden_omega_0,
                    first_layer=False,
                )
            )

        # 最后一个 ResBlock(4*init_features, out_features)
        blocks.append(
            SirenResBlock(
                in_dim=4 * init_features,
                out_dim=out_features,
                omega_0=hidden_omega_0,
                first_layer=False,
            )
        )

        # 和 CoordNet 一样，直接打包成 Sequential
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet_from_config(model_cfg) -> CoordNet:
    """
    从配置构建模型，保留你原来 first_omega_0 / hidden_omega_0 的接口，
    同时支持可选的 init_features / num_res / in_dim / out_dim。
    """
    return CoordNet(
        in_dim=int(model_cfg.get("in_features", 4)),
        out_dim=int(model_cfg.get("out_features", 24)),
        init_features=int(model_cfg.get("init_features", 64)),
        num_res=int(model_cfg.get("num_res", 10)),
        first_omega_0=float(model_cfg.get("first_omega_0", 30.0)),
        hidden_omega_0=float(model_cfg.get("hidden_omega_0", 30.0)),
    )
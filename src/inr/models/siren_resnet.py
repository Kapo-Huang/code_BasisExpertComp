import math
import torch
import torch.nn as nn

from .siren import SineLayer


class SirenResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, first_layer: bool = False):
        super().__init__()
        self.need_skip = in_dim != out_dim
        self.fc1 = SineLayer(in_dim, out_dim, omega_0=omega_0, is_first=first_layer)
        self.fc2 = SineLayer(out_dim, out_dim, omega_0=omega_0)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        if self.need_skip:
            self.skip = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                bound = 1.0 / in_dim
                self.skip.weight.uniform_(-bound, bound)
                if self.skip.bias is not None:
                    self.skip.bias.zero_()

    def forward(self, x):
        identity = x if not self.need_skip else self.skip(x)
        h = self.fc1(x)
        h = self.fc2(h)
        return identity + self.alpha * h


class SirenResNet(nn.Module):
    """Residual SIREN encoder followed by linear head."""

    def __init__(self, first_omega_0: float = 30.0, hidden_omega_0: float = 30.0, in_features: int = 4, out_features: int = 24):
        super().__init__()
        encoder_dims = [
            (in_features, 64),
            (64, 128),
            (128, 256),
        ]
        encoder_dims += [(256, 256)] * 10  # RB4-RB13

        blocks = []
        for idx, (din, dout) in enumerate(encoder_dims):
            is_first = idx == 0
            omega = first_omega_0 if is_first else hidden_omega_0
            blocks.append(SirenResBlock(din, dout, omega_0=omega, first_layer=is_first))
        self.feature = nn.Sequential(*blocks)

        self.final = nn.Linear(256, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / 256) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, x):
        x = self.feature(x)
        return self.final(x)


def build_resnet_from_config(model_cfg) -> SirenResNet:
    return SirenResNet(
        first_omega_0=float(model_cfg.get("first_omega_0", 30.0)),
        hidden_omega_0=float(model_cfg.get("hidden_omega_0", 30.0)),
        in_features=int(model_cfg.get("in_features", 4)),
        out_features=int(model_cfg.get("out_features", 24)),
    )
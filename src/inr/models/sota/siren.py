import math
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = True,
    ):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.zero_()
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_siren_from_config(model_cfg) -> Siren:
    return Siren(
        in_features=int(model_cfg["in_features"]),
        out_features=int(model_cfg["out_features"]),
        hidden_features=int(model_cfg.get("hidden_features", 256)),
        hidden_layers=int(model_cfg.get("hidden_layers", 3)),
        first_omega_0=float(model_cfg.get("first_omega_0", 30.0)),
        hidden_omega_0=float(model_cfg.get("hidden_omega_0", 30.0)),
        outermost_linear=bool(model_cfg.get("outermost_linear", True)),
    )

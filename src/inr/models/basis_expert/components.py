import math
from typing import Tuple

import torch
import torch.nn as nn

from ..sota.siren import SineLayer


class PositionalEncoding(nn.Module):
    """Learnable Fourier positional encoding (dimension-aligned with legacy PE)."""

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        _ = log_sampling
        target_out_dim = in_features * (int(include_input) + 2 * num_frequencies)
        if target_out_dim % 2 != 0:
            raise ValueError(f"PositionalEncoding out_dim must be even, got {target_out_dim}.")
        self.lin = nn.Linear(in_features, target_out_dim // 2, bias=True)
        self.in_features = in_features
        self.include_input = include_input
        self.out_dim = 2 * self.lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        return: (B, out_dim)
        """
        u = self.lin(x)
        return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)


class SirenMLP(nn.Module):
    """Simple SIREN MLP used by experts, gating, and decoder heads."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        layers = [SineLayer(in_dim, hidden_dim, omega_0=first_omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0))

        self.mlp = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        return self.final(h)


class ExpertEncoder(nn.Module):
    """Expert encoder that maps coords to latent features."""

    def __init__(
        self,
        in_features: int,
        feature_dim: int,
        use_positional_encoding: bool = True,
        num_frequencies: int = 6,
        include_input: bool = True,
        hidden_dim: int = 256,
        num_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(
                in_features=in_features,
                num_frequencies=num_frequencies,
                include_input=include_input,
            )
            mlp_in = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            mlp_in = in_features

        self.mlp = SirenMLP(
            in_dim=mlp_in,
            out_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        self.out_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        return self.mlp(x)


class ViewGating(nn.Module):
    """View-conditioned gating that outputs expert routing probabilities."""

    def __init__(
        self,
        in_features: int,
        view_embed_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.gate = SirenMLP(
            in_dim=in_features + view_embed_dim,
            out_dim=num_experts,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, coords: torch.Tensor, view_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_in = torch.cat([coords, view_embed], dim=-1)
        logits = self.gate(gate_in)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits

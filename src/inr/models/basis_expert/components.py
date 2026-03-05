import math
from typing import Tuple

import torch
import torch.nn as nn

from ..sota.siren import SineLayer


# class PositionalEncoding(nn.Module):
#     """Learnable Fourier positional encoding (dimension-aligned with legacy PE)."""

#     def __init__(
#         self,
#         in_features: int,
#         mapping_size: int = 0,
#         num_frequencies: int = 6,
#         include_input: bool = True,
#         log_sampling: bool = True,
#     ):
#         super().__init__()
#         _ = log_sampling
#         if mapping_size is not None and int(mapping_size) > 0:
#             mapping_size = int(mapping_size)
#         else:
#             target_out_dim = in_features * (int(include_input) + 2 * num_frequencies)
#             if target_out_dim % 2 != 0:
#                 raise ValueError(f"PositionalEncoding out_dim must be even, got {target_out_dim}.")
#             mapping_size = target_out_dim // 2
#         self.lin = nn.Linear(in_features, mapping_size, bias=True)
#         self.in_features = in_features
#         self.include_input = include_input
#         self.mapping_size = mapping_size
#         self.out_dim = 2 * self.lin.out_features

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, in_features)
#         return: (B, out_dim)
#         """
#         u = self.lin(x)
#         return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)

class PositionalEncoding(nn.Module):
    """Learnable Fourier positional encoding."""
    def __init__(
        self,
        in_features: int,
        mapping_size: int,
    ):
        super().__init__()
        if int(mapping_size) <= 0:
            raise ValueError("mapping_size must be > 0")
        self.in_features = in_features
        self.lin = nn.Linear(in_features, int(mapping_size), bias=True)

    @property
    def out_dim(self) -> int:
        return 2 * self.lin.out_features

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


class PackedSirenExperts(nn.Module):
    """Packed experts with shared architecture and stacked parameters."""

    def __init__(
        self,
        num_experts: int,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        *,
        init_weights: bool = True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("in_dim, hidden_dim, out_dim must be >= 1")

        self.num_experts = int(num_experts)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_dim = int(out_dim)
        self.first_omega_0 = float(first_omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)
        self.num_sine_layers = self.num_layers - 1

        self.sine_weights = nn.ParameterList()
        self.sine_biases = nn.ParameterList()

        self.sine_weights.append(
            nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.in_dim))
        )
        self.sine_biases.append(
            nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        )
        for _ in range(1, self.num_sine_layers):
            self.sine_weights.append(
                nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.hidden_dim))
            )
            self.sine_biases.append(
                nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
            )

        self.final_weight = nn.Parameter(
            torch.empty(self.num_experts, self.out_dim, self.hidden_dim)
        )
        self.final_bias = nn.Parameter(torch.empty(self.num_experts, self.out_dim))

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            first_bound = 1.0 / float(self.in_dim)
            self.sine_weights[0].uniform_(-first_bound, first_bound)
            self.sine_biases[0].zero_()

            hidden_bound = math.sqrt(6.0 / float(self.hidden_dim)) / float(self.hidden_omega_0)
            for idx in range(1, self.num_sine_layers):
                self.sine_weights[idx].uniform_(-hidden_bound, hidden_bound)
                self.sine_biases[idx].zero_()

            self.final_weight.uniform_(-hidden_bound, hidden_bound)
            self.final_bias.zero_()

    @classmethod
    def from_expert_list(cls, experts) -> "PackedSirenExperts":
        if experts is None or len(experts) == 0:
            raise ValueError("experts must be a non-empty sequence")

        first = experts[0]
        if getattr(first, "pos_enc", None) is not None:
            raise ValueError("PackedSirenExperts expects experts with use_positional_encoding=False")

        first_mlp = first.mlp
        first_sine_layers = list(first_mlp.mlp)
        if len(first_sine_layers) < 1:
            raise ValueError("Each expert must have at least one sine layer")

        in_dim = int(first_sine_layers[0].linear.in_features)
        hidden_dim = int(first_sine_layers[0].linear.out_features)
        num_layers = int(len(first_sine_layers) + 1)
        out_dim = int(first_mlp.final.out_features)
        first_omega_0 = float(first_sine_layers[0].omega_0)
        hidden_omega_0 = (
            float(first_sine_layers[1].omega_0)
            if len(first_sine_layers) > 1
            else float(first_sine_layers[0].omega_0)
        )

        packed = cls(
            num_experts=len(experts),
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            init_weights=False,
        )

        with torch.no_grad():
            for expert_idx, expert in enumerate(experts):
                if getattr(expert, "pos_enc", None) is not None:
                    raise ValueError("All experts must disable positional encoding for packing")
                mlp = expert.mlp
                sine_layers = list(mlp.mlp)
                if len(sine_layers) != packed.num_sine_layers:
                    raise ValueError("Inconsistent number of sine layers across experts")
                if int(mlp.final.out_features) != packed.out_dim:
                    raise ValueError("Inconsistent out_dim across experts")
                for layer_idx, layer in enumerate(sine_layers):
                    if int(layer.linear.in_features) != (
                        packed.in_dim if layer_idx == 0 else packed.hidden_dim
                    ):
                        raise ValueError("Inconsistent expert in_features across experts")
                    if int(layer.linear.out_features) != packed.hidden_dim:
                        raise ValueError("Inconsistent expert hidden_dim across experts")
                    packed.sine_weights[layer_idx][expert_idx].copy_(layer.linear.weight)
                    if layer.linear.bias is None:
                        raise ValueError("PackedSirenExperts requires expert layers with bias=True")
                    packed.sine_biases[layer_idx][expert_idx].copy_(layer.linear.bias)
                packed.final_weight[expert_idx].copy_(mlp.final.weight)
                if mlp.final.bias is None:
                    raise ValueError("PackedSirenExperts requires final linear bias=True")
                packed.final_bias[expert_idx].copy_(mlp.final.bias)
        return packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bi,moi->bmo", x, self.sine_weights[0])
        h = h + self.sine_biases[0].unsqueeze(0)
        h = torch.sin(self.first_omega_0 * h)

        for layer_idx in range(1, self.num_sine_layers):
            h = torch.einsum("bmi,moi->bmo", h, self.sine_weights[layer_idx])
            h = h + self.sine_biases[layer_idx].unsqueeze(0)
            h = torch.sin(self.hidden_omega_0 * h)

        out = torch.einsum("bmi,moi->bmo", h, self.final_weight)
        out = out + self.final_bias.unsqueeze(0)
        return out


class SmallMLPHead(nn.Module):
    """Two-layer lightweight head: Linear -> ReLU -> Linear."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(8, int(in_dim // 4))
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


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
            mapping_size = in_features * (int(include_input) + 2 * num_frequencies) // 2
            self.pos_enc = PositionalEncoding(
                in_features=in_features,
                mapping_size=mapping_size,
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

import math
from typing import Optional

import torch
import torch.nn as nn

from .siren import SineLayer


class PositionalEncoding(nn.Module):
    """NeRF-style positional encoding."""
    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 6,
        include_input: bool = False,
        log_sampling: bool = True,
    ):
        super().__init__()
        if log_sampling:
            freq_bands = 2.0 ** torch.arange(num_frequencies) * math.pi
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies) * math.pi
        self.register_buffer("freq_bands", freq_bands)
        self.in_features = in_features
        self.include_input = include_input
        self.out_dim = in_features * (int(include_input) + 2 * num_frequencies)
        print(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        return: (B, out_dim)
        """
        angles = x.unsqueeze(-1) * self.freq_bands
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        encoded = encoded.reshape(x.shape[0], -1)
        if self.include_input:
            return torch.cat([x, encoded], dim=-1)
        return encoded


class SirenMLP(nn.Module):
    """Simple SIREN MLP used by the policy network and expert decoders."""
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

        layers = []
        layers.append(SineLayer(in_dim, hidden_dim, omega_0=first_omega_0, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0))

        self.mlp = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)

        # SIREN-style init for the linear head
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        return self.final(h)


class BottleneckResBlock(nn.Module):
    """
    Residual block with bottleneck:
    in_dim -> bottleneck_dim -> bottleneck_dim -> out_dim,
    activations after the first two linear layers, and activation after residual add.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bottleneck_dim: Optional[int] = None,
        activation_factory=None,
    ):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(1, min(in_dim, out_dim) // 2)
        if activation_factory is None:
            activation_factory = lambda _: nn.ReLU()

        self.fc1 = nn.Linear(in_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, out_dim)

        self.act1 = activation_factory(bottleneck_dim)
        self.act2 = activation_factory(bottleneck_dim)
        self.act_out = activation_factory(out_dim)

        self.shortcut = None
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act1(h)
        h = self.fc2(h)
        h = self.act2(h)
        h = self.fc3(h)

        shortcut = self.shortcut(x) if self.shortcut is not None else x
        h = h + shortcut
        return self.act_out(h)


class SharedSirenEncoder(nn.Module):
    """
    Encoder with positional encoding -> sine -> sine -> residual SIREN block.
    """

    def __init__(
        self,
        in_features: int = 4,
        feature_dim: int = 256,
        base_dim: Optional[int] = None,
        num_frequencies: int = 6,
        include_input: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(
            in_features=in_features,
            num_frequencies=num_frequencies,
            include_input=include_input,
        )
        pos_dim = self.pos_enc.out_dim
        self.pe_proj = None
        if base_dim is not None:
            target_pos_dim = 2 * base_dim
            if pos_dim != target_pos_dim:
                self.pe_proj = nn.Linear(pos_dim, target_pos_dim)
                pos_dim = target_pos_dim
            sine1_dim = 4 * base_dim
            sine2_dim = 8 * base_dim
            res_dim = 8 * base_dim
        else:
            sine1_dim = feature_dim
            sine2_dim = feature_dim
            res_dim = feature_dim

        self.sine1 = SineLayer(pos_dim, sine1_dim, omega_0=first_omega_0, is_first=True)
        self.sine2 = SineLayer(sine1_dim, sine2_dim, omega_0=hidden_omega_0)
        self.res_block = BottleneckResBlock(sine2_dim, res_dim)
        self.out_dim = res_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(x)
        if self.pe_proj is not None:
            h = self.pe_proj(h)
        h = self.sine1(h)
        h = self.sine2(h)
        return self.res_block(h)


class PolicyNetwork(nn.Module):
    """
    Policy network that produces routing probabilities and a latent feature
    to fuse with the shared encoder output.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_experts: int = 7,
        gate_in_dim: Optional[int] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        assert num_layers >= 2

        layers = []
        layers.append(SineLayer(in_features, hidden_dim, omega_0=first_omega_0, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0))
        self.feature = nn.Sequential(*layers)
        if gate_in_dim is None:
            gate_in_dim = hidden_dim
        self.gate = nn.Linear(gate_in_dim, num_experts)

        with torch.no_grad():
            bound = math.sqrt(6.0 / gate_in_dim) / hidden_omega_0
            self.gate.weight.uniform_(-bound, bound)
            if self.gate.bias is not None:
                self.gate.bias.zero_()

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.gate_in_dim = gate_in_dim

    def forward(self, x: torch.Tensor, encoder_feat: Optional[torch.Tensor] = None):
        feat = self.feature(x)
        gate_input = feat
        if encoder_feat is not None:
            gate_input = torch.cat([encoder_feat, feat], dim=-1)
        logits = self.gate(gate_input)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits, feat


class ExpertDecoder(nn.Module):
    """Single expert decoder that receives fused encoder/policy features."""

    def __init__(
        self,
        in_dim: int,
        out_features: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.mlp = SirenMLP(
            in_dim=in_dim,
            out_dim=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MoEINR(nn.Module):
    """
    Mixture-of-Experts INR:
    - shared encoder extracts high-dimensional features
    - policy network outputs routing probabilities
    - multiple expert decoders predict the target, combined via soft/hard routing
    """

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 1,
        num_experts: int = 7,
        encoder_feature_dim: int = 256,
        base_dim: Optional[int] = None,
        encoder_first_omega_0: float = 30.0,
        encoder_hidden_omega_0: float = 30.0,
        policy_hidden_dim: int = 128,
        policy_num_layers: int = 3,
        policy_first_omega_0: float = 30.0,
        policy_hidden_omega_0: float = 30.0,
        expert_hidden_dim: int = 256,
        expert_num_layers: int = 3,
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            feature_dim=encoder_feature_dim,
            base_dim=base_dim,
            num_frequencies=int(base_dim / in_features),
            include_input=False,
            first_omega_0=encoder_first_omega_0,
            hidden_omega_0=encoder_hidden_omega_0,
        )

        self.policy = PolicyNetwork(
            in_features=in_features,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            num_experts=num_experts,
            gate_in_dim=encoder_feature_dim + policy_hidden_dim,
            first_omega_0=policy_first_omega_0,
            hidden_omega_0=policy_hidden_omega_0,
        )

        fused_dim = encoder_feature_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertDecoder(
                    in_dim=fused_dim,
                    out_features=out_features,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )
        self.out_features = out_features

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_routing: bool = False,
        return_all: bool = False,
    ):
        """
        x: [B, in_features]
        """
        enc_feat = self.encoder(x)
        probs, logits, pol_feat = self.policy(x, enc_feat)
        fused = enc_feat

        preds = [expert(fused) for expert in self.experts]
        preds_all = torch.stack(preds, dim=1)  # [B, K, out_features]

        if hard_routing:
            indices = torch.argmax(probs, dim=-1)
            y = preds_all[torch.arange(x.shape[0], device=x.device), indices]
        else:
            y = torch.sum(preds_all * probs.unsqueeze(-1), dim=1)

        if return_all:
            return y, preds_all, probs, logits
        return y


def build_moe_inr_from_config(cfg) -> MoEINR:
    """Helper to construct MoEINR from a plain dict/YAML config."""
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        encoder_feature_dim = 8 * base_dim
        policy_hidden_dim = base_dim
        expert_hidden_dim = 8 * base_dim
    else:
        encoder_feature_dim = int(cfg.get("encoder_feature_dim", 256))
        policy_hidden_dim = int(cfg.get("policy_hidden_dim", 128))
        expert_hidden_dim = int(cfg.get("expert_hidden_dim", 256))
    return MoEINR(
        in_features=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        num_experts=int(cfg.get("num_experts", 7)),
        encoder_feature_dim=encoder_feature_dim,
        base_dim=base_dim,
        encoder_first_omega_0=float(cfg.get("encoder_first_omega_0", 30.0)),
        encoder_hidden_omega_0=float(cfg.get("encoder_hidden_omega_0", 30.0)),
        policy_hidden_dim=policy_hidden_dim,
        policy_num_layers=int(cfg.get("policy_num_layers", 3)),
        policy_first_omega_0=float(cfg.get("policy_first_omega_0", 30.0)),
        policy_hidden_omega_0=float(cfg.get("policy_hidden_omega_0", 30.0)),
        expert_hidden_dim=expert_hidden_dim,
        expert_num_layers=int(cfg.get("expert_num_layers", 3)),
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
    )

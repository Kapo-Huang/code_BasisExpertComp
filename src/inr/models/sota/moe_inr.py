import math
import logging
from typing import Optional

import torch
import torch.nn as nn

from .siren import SineLayer

logger = logging.getLogger(__name__)


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class PositionalEncoding(nn.Module):
    """Learnable Fourier positional encoding."""
    def __init__(
        self,
        in_features: int,
        mapping_size: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.lin = nn.Linear(in_features, mapping_size, bias=True)

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
            bottleneck_dim = max(1, min(in_dim, out_dim) // 4)
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
        include_input: bool = False,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if include_input:
            raise ValueError("Learnable Fourier PE does not support include_input=True.")
        target_pos_dim = 2 * base_dim
        pe_mapping_dim = base_dim
        sine1_dim = 4 * base_dim
        sine2_dim = 8 * base_dim
        res_dim = 8 * base_dim

        self.pos_enc = PositionalEncoding(
            in_features=in_features,
            mapping_size=pe_mapping_dim,
        )
        pos_dim = self.pos_enc.out_dim
        if pos_dim != target_pos_dim:
            raise RuntimeError(
                f"Learnable Fourier PE dim mismatch: got {pos_dim}, expected {target_pos_dim}."
            )

        self.sine1 = SineLayer(pos_dim, sine1_dim, omega_0=first_omega_0, is_first=True)
        self.sine2 = SineLayer(sine1_dim, sine2_dim, omega_0=hidden_omega_0)
        self.res_block = BottleneckResBlock(sine2_dim, res_dim)
        self.out_dim = res_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(x)
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
        # self.mlp = SirenMLP(
        #     in_dim=in_dim,
        #     out_dim=out_features,
        #     hidden_dim=hidden_dim,
        #     num_layers=num_layers,
        #     first_omega_0=first_omega_0,
        #     hidden_omega_0=hidden_omega_0,
        # )
        self.mlp = nn.Linear(in_dim, out_features)

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
        top_k: int = 1,
    ):
        super().__init__()

        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            feature_dim=encoder_feature_dim,
            base_dim=base_dim,
            num_frequencies=(int(base_dim / in_features) if base_dim is not None else 6),
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
        self.top_k = max(1, min(int(top_k), int(num_experts)))
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

        policy_network_params = _count_parameters(self.policy)
        experts_params = _count_parameters(self.experts)
        shared_encoder_params = _count_parameters(self.encoder)
        logger.info(
            "MoEINR init params: policy_network=%s experts=%s shared_encoder=%s",
            f"{policy_network_params:,}",
            f"{experts_params:,}",
            f"{shared_encoder_params:,}",
        )

    def _topk_mask(self, probs: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(probs, k=self.top_k, dim=-1)
        mask = torch.zeros_like(probs)
        return mask.scatter_(1, indices, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_routing: bool = False,
        hard_topk: bool = True,
        return_aux: bool = False,
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
            mask = self._topk_mask(probs)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            weights = masked_probs if hard_topk else probs
            y = torch.sum(preds_all * weights.unsqueeze(-1), dim=1)

        if return_aux:
            aux = {
                "probs": probs.unsqueeze(1),
                "masks": self._topk_mask(probs).unsqueeze(1),
                "expert_feats": preds_all,
                "encoder_feat": enc_feat,
                "policy_feat": pol_feat,
                "logits": logits,
            }
            return y, aux

        if return_all:
            return y, preds_all, probs, logits
        return y

    def pretrain_forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feat = self.encoder(x)
        _, logits, _ = self.policy(x, enc_feat)
        return logits

    def pretrain_parameters(self):
        return list(self.encoder.parameters()) + list(self.policy.parameters())


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
        top_k=int(cfg.get("top_k", 1)),
    )

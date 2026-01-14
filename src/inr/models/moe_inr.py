import math
from typing import Optional

import torch
import torch.nn as nn

from .siren import SineLayer


class LearnableFourierPE(nn.Module):
    """Learnable Fourier features: Linear -> [sin, cos]."""

    def __init__(self, in_features: int, base_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features, base_dim)
        self.out_dim = 2 * base_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj(x)
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
    Encoder with learnable Fourier features -> sine -> sine -> residual block.
    """

    def __init__(
        self,
        in_features: int = 4,
        base_dim: int = 16,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.pe = LearnableFourierPE(in_features=in_features, base_dim=self.base_dim)
        self.sine1 = SineLayer(2 * self.base_dim, 4 * self.base_dim, omega_0=first_omega_0, is_first=True)
        self.sine2 = SineLayer(4 * self.base_dim, 8 * self.base_dim, omega_0=hidden_omega_0)
        self.res_block = BottleneckResBlock(
            8 * self.base_dim,
            8 * self.base_dim,
            bottleneck_dim=2 * self.base_dim,
        )
        self.out_dim = 8 * self.base_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pe(x)
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
        base_dim: int = 16,
        num_experts: int = 7,
        gate_in_dim: Optional[int] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.feature1 = SineLayer(in_features, self.base_dim, omega_0=first_omega_0, is_first=True)
        self.feature2 = SineLayer(self.base_dim, self.base_dim, omega_0=hidden_omega_0)
        if gate_in_dim is None:
            gate_in_dim = self.base_dim
        self.gate = nn.Linear(gate_in_dim, num_experts)

        with torch.no_grad():
            bound = math.sqrt(6.0 / gate_in_dim) / hidden_omega_0
            self.gate.weight.uniform_(-bound, bound)
            if self.gate.bias is not None:
                self.gate.bias.zero_()

        self.num_experts = num_experts
        self.hidden_dim = self.base_dim
        self.gate_in_dim = gate_in_dim

    def forward(self, x: torch.Tensor, encoder_feat: Optional[torch.Tensor] = None):
        feat = self.feature2(self.feature1(x))
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
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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
        base_dim: int = 16,
        encoder_first_omega_0: float = 30.0,
        encoder_hidden_omega_0: float = 30.0,
        policy_first_omega_0: float = 30.0,
        policy_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.base_dim = int(base_dim)

        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            base_dim=self.base_dim,
            first_omega_0=encoder_first_omega_0,
            hidden_omega_0=encoder_hidden_omega_0,
        )

        encoder_dim = 8 * self.base_dim
        self.policy = PolicyNetwork(
            in_features=in_features,
            base_dim=self.base_dim,
            num_experts=num_experts,
            gate_in_dim=encoder_dim + self.base_dim,
            first_omega_0=policy_first_omega_0,
            hidden_omega_0=policy_hidden_omega_0,
        )

        fused_dim = encoder_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertDecoder(
                    in_dim=fused_dim,
                    out_features=out_features,
                )
                for _ in range(num_experts)
            ]
        )
        self.out_features = out_features
        assert self.encoder.out_dim == 8 * self.base_dim
        assert self.policy.hidden_dim == self.base_dim
        assert self.policy.gate_in_dim == 9 * self.base_dim
        for expert in self.experts:
            assert expert.linear.in_features == 8 * self.base_dim

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_routing: Optional[bool] = None,
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

        use_hard = hard_routing if hard_routing is not None else (not self.training)
        if use_hard:
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
    if base_dim is None:
        raise ValueError("MoEINR requires base_dim to be set.")
    base_dim = int(base_dim)
    return MoEINR(
        in_features=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        num_experts=int(cfg.get("num_experts", 7)),
        base_dim=base_dim,
        encoder_first_omega_0=float(cfg.get("encoder_first_omega_0", 30.0)),
        encoder_hidden_omega_0=float(cfg.get("encoder_hidden_omega_0", 30.0)),
        policy_first_omega_0=float(cfg.get("policy_first_omega_0", 30.0)),
        policy_hidden_omega_0=float(cfg.get("policy_hidden_omega_0", 30.0)),
    )

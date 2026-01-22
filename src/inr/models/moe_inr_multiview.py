from typing import Dict, Optional

import torch
import torch.nn as nn

from .moe_inr import ExpertDecoder, PolicyNetwork, SharedSirenEncoder


class MoEINRMultiView(nn.Module):
    """
    Multi-view MoE-INR with a shared MoE decoder:
    - shared encoder extracts high-dimensional features
    - shared policy routes to expert decoders
    - experts output the concatenated attributes, then slice per view
    """

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        num_experts: int = 7,
        encoder_feature_dim: int = 256,
        base_dim: Optional[int] = None,
        top_k: int = 2,
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
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.total_out_dim = int(sum(self.view_dims.values()))
        self.view_slices = {}
        offset = 0
        for name in self.view_names:
            width = int(self.view_dims[name])
            self.view_slices[name] = slice(offset, offset + width)
            offset += width

        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            feature_dim=encoder_feature_dim,
            base_dim=base_dim,
            num_frequencies=int(base_dim / in_features) if base_dim is not None else 6,
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

        self.experts = nn.ModuleList(
            [
                ExpertDecoder(
                    in_dim=encoder_feature_dim,
                    out_features=self.total_out_dim,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )

    def _topk_mask(self, probs: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(probs, k=self.top_k, dim=-1)
        mask = torch.zeros_like(probs)
        return mask.scatter_(1, indices, 1.0)

    def forward(
        self,
        coords: torch.Tensor,
        request: Optional[str] = None,
        *,
        hard_routing: bool = False,
        return_aux: bool = False,
    ):
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        enc_feat = self.encoder(coords)
        probs, _logits, _feat = self.policy(coords, enc_feat)
        mask = self._topk_mask(probs)

        expert_outs = [expert(enc_feat) for expert in self.experts]
        preds_all = torch.stack(expert_outs, dim=1)

        if hard_routing:
            indices = torch.argmax(probs, dim=-1)
            pred_all = preds_all[torch.arange(coords.shape[0], device=coords.device), indices]
        else:
            pred_all = torch.sum(preds_all * probs.unsqueeze(-1), dim=1)

        preds = {name: pred_all[:, self.view_slices[name]] for name in self.view_names}

        output = preds if request is None else preds[request]
        if return_aux:
            probs_view = probs.unsqueeze(1).repeat(1, self.num_views, 1)
            masks_view = mask.unsqueeze(1).repeat(1, self.num_views, 1)
            aux = {
                "probs": probs_view,  # (B, V, M)
                "masks": masks_view,  # (B, V, M)
                "expert_feats": enc_feat.unsqueeze(1).repeat(1, self.num_experts, 1),
            }
            return output, aux
        return output


def build_moe_inr_multiview_from_config(cfg, view_specs: Dict[str, int]) -> MoEINRMultiView:
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
    return MoEINRMultiView(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        num_experts=int(cfg.get("num_experts", 7)),
        encoder_feature_dim=encoder_feature_dim,
        base_dim=base_dim,
        top_k=int(cfg.get("top_k", 2)),
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

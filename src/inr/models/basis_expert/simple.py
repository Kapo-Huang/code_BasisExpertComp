from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .components import ExpertEncoder, SirenMLP, ViewGating


class BasisExpertSimpleConcat(nn.Module):
    """
    MoE-INR with view-conditioned gating and selectable fusion:
    - concat: concatenate all view features
    - mean: average view features
    - mlp: MLP over concatenated view features
    - none: per-view decoder with view embedding only
    """

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        num_experts: int = 7,
        expert_feature_dim: int = 128,
        top_k: int = 2,
        view_embed_dim: int = 16,
        expert_use_positional_encoding: bool = True,
        expert_num_frequencies: int = 6,
        expert_hidden_dim: int = 128,
        expert_num_layers: int = 3,
        gate_hidden_dim: int = 128,
        gate_num_layers: int = 3,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        fusion_mode: str = "concat",
        fusion_hidden_dim: int = 128,
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
        gate_first_omega_0: float = 30.0,
        gate_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")

        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_feature_dim = expert_feature_dim
        self.view_embed_dim = view_embed_dim
        self.fusion_mode = self._normalize_fusion_mode(fusion_mode)

        self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
        self.gating = ViewGating(
            in_features=in_features,
            view_embed_dim=view_embed_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_num_layers,
            first_omega_0=gate_first_omega_0,
            hidden_omega_0=gate_hidden_omega_0,
        )

        self.experts = nn.ModuleList(
            [
                ExpertEncoder(
                    in_features=in_features,
                    feature_dim=expert_feature_dim,
                    use_positional_encoding=expert_use_positional_encoding,
                    num_frequencies=expert_num_frequencies,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )

        if self.fusion_mode == "concat":
            fused_dim = self.num_views * expert_feature_dim
            self.fusion_mlp = None
        elif self.fusion_mode == "mean":
            fused_dim = expert_feature_dim
            self.fusion_mlp = None
        elif self.fusion_mode == "mlp":
            fused_dim = expert_feature_dim
            fusion_in_dim = self.num_views * expert_feature_dim
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_in_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim, fused_dim),
            )
        elif self.fusion_mode == "none":
            fused_dim = expert_feature_dim + view_embed_dim
            self.fusion_mlp = None
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        self.decoders = nn.ModuleDict(
            {
                name: SirenMLP(
                    in_dim=fused_dim,
                    out_dim=out_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    first_omega_0=decoder_first_omega_0,
                    hidden_omega_0=decoder_hidden_omega_0,
                )
                for name, out_dim in self.view_dims.items()
            }
        )

    @staticmethod
    def _normalize_fusion_mode(mode: str) -> str:
        key = (mode or "concat").strip().lower()
        if key in {"concat", "simple_concat"}:
            return "concat"
        if key in {"mean", "simple_mean", "avg", "average"}:
            return "mean"
        if key in {"mlp", "simple_mlp"}:
            return "mlp"
        if key in {"none", "no_fusion", "nofusion"}:
            return "none"
        return key

    def _topk_mask(self, probs: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(probs, k=self.top_k, dim=-1)
        mask = torch.zeros_like(probs)
        return mask.scatter_(1, indices, 1.0)

    def forward(
        self,
        coords: torch.Tensor,
        request: Optional[str] = None,
        *,
        hard_topk: bool = True,
        return_aux: bool = False,
    ):
        """
        coords: (B, in_features)
        request: optional view name to return a single attribute prediction
        """
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        expert_feats = torch.stack([expert(coords) for expert in self.experts], dim=1)  # (B, M, F)

        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        h_views: List[torch.Tensor] = []
        view_embeds: List[torch.Tensor] = []

        for view_idx, _name in enumerate(self.view_names):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            probs, _ = self.gating(coords, view_embed)
            mask = self._topk_mask(probs)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            weights = masked_probs if hard_topk else probs
            h_v = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)

            probs_list.append(probs)
            masks_list.append(mask)
            h_views.append(h_v)
            view_embeds.append(view_embed)

        h_views_tensor = torch.stack(h_views, dim=1)  # (B, V, F)
        h_fused = None
        if self.fusion_mode == "concat":
            h_fused = torch.cat(h_views, dim=-1)  # (B, V * F)
        elif self.fusion_mode == "mean":
            h_fused = h_views_tensor.mean(dim=1)  # (B, F)
        elif self.fusion_mode == "mlp":
            h_flat = torch.cat(h_views, dim=-1)
            h_fused = self.fusion_mlp(h_flat)

        if self.fusion_mode == "none":
            preds = {}
            for view_idx, name in enumerate(self.view_names):
                decoder_in = torch.cat([h_views[view_idx], view_embeds[view_idx]], dim=-1)
                preds[name] = self.decoders[name](decoder_in)
        else:
            preds = {name: self.decoders[name](h_fused) for name in self.view_names}
        output = preds if request is None else preds[request]

        if return_aux:
            aux = {
                "probs": torch.stack(probs_list, dim=1),  # (B, V, M)
                "masks": torch.stack(masks_list, dim=1),  # (B, V, M)
                "H_views": h_views_tensor,
                "H_fused": h_fused if h_fused is not None else h_views_tensor,
                "expert_feats": expert_feats,
            }
            return output, aux
        return output


def build_basisExpert_simple_concat_from_config(
    cfg: Dict, view_specs: Dict[str, int]
) -> BasisExpertSimpleConcat:
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        expert_feature_dim = 8 * base_dim
        view_embed_dim = base_dim
        expert_hidden_dim = 8 * base_dim
        gate_hidden_dim = 8 * base_dim
        decoder_hidden_dim = 8 * base_dim
        fusion_hidden_dim = int(cfg.get("fusion_hidden_dim", 8 * base_dim))
    else:
        expert_feature_dim = int(cfg.get("expert_feature_dim", 128))
        view_embed_dim = int(cfg.get("view_embed_dim", 16))
        expert_hidden_dim = int(cfg.get("expert_hidden_dim", 128))
        gate_hidden_dim = int(cfg.get("gate_hidden_dim", 128))
        decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 128))
        fusion_hidden_dim = int(cfg.get("fusion_hidden_dim", decoder_hidden_dim))
    return BasisExpertSimpleConcat(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        num_experts=int(cfg.get("num_experts", 7)),
        expert_feature_dim=expert_feature_dim,
        top_k=int(cfg.get("top_k", 2)),
        view_embed_dim=view_embed_dim,
        expert_use_positional_encoding=bool(cfg.get("expert_use_positional_encoding", True)),
        expert_num_frequencies=int(cfg.get("expert_num_frequencies", 6)),
        expert_hidden_dim=expert_hidden_dim,
        expert_num_layers=int(cfg.get("expert_num_layers", 3)),
        gate_hidden_dim=gate_hidden_dim,
        gate_num_layers=int(cfg.get("gate_num_layers", 3)),
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=int(cfg.get("decoder_num_layers", 3)),
        fusion_mode=str(cfg.get("fusion_mode", "concat")),
        fusion_hidden_dim=fusion_hidden_dim,
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        gate_first_omega_0=float(cfg.get("gate_first_omega_0", 30.0)),
        gate_hidden_omega_0=float(cfg.get("gate_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
    )

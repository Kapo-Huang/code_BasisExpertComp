from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .components import ExpertEncoder, SirenMLP, ViewGating


class TinyTransformerFusion(nn.Module):
    """Lightweight transformer encoder for cross-view fusion."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        ff_dim = int(feature_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class BasisExpertsAttention(nn.Module):
    """
    MoE-INR with cross-view transformer fusion and fixed decoder input size.
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
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
        gate_first_omega_0: float = 30.0,
        gate_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
        fusion_num_layers: int = 2,
        fusion_num_heads: Optional[int] = None,
        fusion_mlp_ratio: float = 4.0,
        fusion_dropout: float = 0.1,
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

        self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
        self.view_embed_proj = nn.Linear(view_embed_dim, expert_feature_dim, bias=False)
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

        num_heads = self._resolve_num_heads(expert_feature_dim, fusion_num_heads)
        self.ctx_token = nn.Parameter(torch.zeros(1, 1, expert_feature_dim))
        nn.init.normal_(self.ctx_token, std=0.02)
        self.fusion = TinyTransformerFusion(
            feature_dim=expert_feature_dim,
            num_layers=fusion_num_layers,
            num_heads=num_heads,
            mlp_ratio=fusion_mlp_ratio,
            dropout=fusion_dropout,
        )

        fused_dim = expert_feature_dim * 2
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
    def _resolve_num_heads(feature_dim: int, requested: Optional[int]) -> int:
        if requested is not None:
            if feature_dim % requested != 0:
                raise ValueError("fusion_num_heads must divide expert_feature_dim")
            return requested
        preferred = 8 if feature_dim >= 256 else 4
        for candidate in (preferred, 4, 2, 1, 8):
            if feature_dim % candidate == 0:
                return candidate
        return 1

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
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        expert_feats = torch.stack([expert(coords) for expert in self.experts], dim=1)  # (B, M, F)

        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        h_views: List[torch.Tensor] = []

        for view_idx, _name in enumerate(self.view_names):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            probs, _ = self.gating(coords, view_embed)
            mask = self._topk_mask(probs)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            weights = masked_probs if hard_topk else probs

            h_v = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
            h_v = h_v + self.view_embed_proj(view_embed)
            probs_list.append(probs)
            masks_list.append(mask)
            h_views.append(h_v)

        h_views_tensor = torch.stack(h_views, dim=1)  # (B, V, F)
        ctx = self.ctx_token.expand(coords.shape[0], -1, -1)
        tokens = torch.cat([ctx, h_views_tensor], dim=1)
        tokens = self.fusion(tokens)

        ctx_out = tokens[:, :1, :]
        h_views_fused = tokens[:, 1:, :]
        ctx_flat = ctx_out.squeeze(1)

        preds = {}
        for view_idx, name in enumerate(self.view_names):
            z_v = torch.cat([h_views_fused[:, view_idx, :], ctx_flat], dim=-1)
            preds[name] = self.decoders[name](z_v)

        output = preds if request is None else preds[request]
        if return_aux:
            aux = {
                "probs": torch.stack(probs_list, dim=1),  # (B, V, M)
                "masks": torch.stack(masks_list, dim=1),  # (B, V, M)
                "H_views": h_views_tensor,
                "H_views_fused": h_views_fused,
                "CTX": ctx_out,
                "expert_feats": expert_feats,
            }
            return output, aux
        return output


def build_basisExperts_attention_from_config(cfg: Dict, view_specs: Dict[str, int]) -> BasisExpertsAttention:
    fusion_num_heads_raw = cfg.get("fusion_num_heads")
    fusion_num_heads = int(fusion_num_heads_raw) if fusion_num_heads_raw is not None else None
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        expert_feature_dim = 8 * base_dim
        view_embed_dim = base_dim
        expert_hidden_dim = 8 * base_dim
        gate_hidden_dim = 8 * base_dim
        decoder_hidden_dim = 8 * base_dim
    else:
        expert_feature_dim = int(cfg.get("expert_feature_dim", 128))
        view_embed_dim = int(cfg.get("view_embed_dim", 16))
        expert_hidden_dim = int(cfg.get("expert_hidden_dim", 128))
        gate_hidden_dim = int(cfg.get("gate_hidden_dim", 128))
        decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 128))
    return BasisExpertsAttention(
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
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        gate_first_omega_0=float(cfg.get("gate_first_omega_0", 30.0)),
        gate_hidden_omega_0=float(cfg.get("gate_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
        fusion_num_layers=int(cfg.get("fusion_num_layers", 2)),
        fusion_num_heads=fusion_num_heads,
        fusion_mlp_ratio=float(cfg.get("fusion_mlp_ratio", 4.0)),
        fusion_dropout=float(cfg.get("fusion_dropout", 0.1)),
    )

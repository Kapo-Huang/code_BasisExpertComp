# import logging
# from typing import Dict, List, Optional

# import torch
# import torch.nn as nn

# from ..sota.siren import SineLayer
# from .components import (
#     ExpertEncoder,
#     PositionalEncoding,
#     SirenMLP,
#     ViewGating,
# )

# logger = logging.getLogger(__name__)


# def _count_parameters(module: nn.Module) -> int:
#     return sum(p.numel() for p in module.parameters())


# class DecoderBottleneckResBlock(nn.Module):
#     """Residual bottleneck block: D -> D/2 -> D."""

#     def __init__(self, dim: int, omega_0: float = 30.0):
#         super().__init__()
#         bottleneck_dim = max(1, dim // 2)
#         self.down = SineLayer(dim, bottleneck_dim, omega_0=omega_0)
#         self.up = SineLayer(bottleneck_dim, dim, omega_0=omega_0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.up(self.down(x))


# class LightDecoder(nn.Module):
#     """
#     Shared decoder:
#     - proj_in: SineLayer(F -> D)
#     - bottleneck residual blocks on D
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         *,
#         first_omega_0: float = 30.0,
#         hidden_omega_0: float = 30.0,
#         num_res_blocks: int = 1,
#     ):
#         super().__init__()
#         if num_res_blocks < 1:
#             raise ValueError("num_res_blocks must be >= 1")
#         self.proj_in = SineLayer(in_dim, out_dim, is_first=True, omega_0=first_omega_0)
#         self.res_blocks = nn.Sequential(
#             *[DecoderBottleneckResBlock(out_dim, omega_0=hidden_omega_0) for _ in range(num_res_blocks)]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.proj_in(x)
#         return self.res_blocks(h)


# class LightBasisExpert(nn.Module):
#     """
#     Light MoE-INR:
#     - experts + view-conditioned gating -> h_v
#     - shared decoder maps h_v to a shared feature
#     - per-view heads map shared feature to each attribute
#     """

#     def __init__(
#         self,
#         in_features: int,
#         view_specs: Dict[str, int],
#         num_experts: int = 7,
#         expert_feature_dim: int = 128,
#         top_k: int = 2,
#         view_embed_dim: int = 16,
#         pe_mapping_size: Optional[int] = None,
#         expert_hidden_dim: int = 128,
#         expert_num_layers: int = 3,
#         gate_hidden_dim: int = 128,
#         gate_num_layers: int = 3,
#         decoder_feature_dim: int = 128,
#         decoder_num_res_blocks: int = 1,
#         expert_first_omega_0: float = 30.0,
#         expert_hidden_omega_0: float = 30.0,
#         gate_first_omega_0: float = 30.0,
#         gate_hidden_omega_0: float = 30.0,
#         decoder_first_omega_0: float = 30.0,
#         decoder_hidden_omega_0: float = 30.0,
#     ):
#         super().__init__()
#         if top_k < 1:
#             raise ValueError("top_k must be >= 1")
#         if not view_specs:
#             raise ValueError("view_specs must be a non-empty dict")

#         self.view_names = list(view_specs.keys())
#         self.view_dims = dict(view_specs)
#         self.num_views = len(self.view_names)
#         self.num_experts = num_experts
#         self.top_k = min(top_k, num_experts)
#         self.expert_feature_dim = int(expert_feature_dim)
#         self.decoder_feature_dim = int(decoder_feature_dim)
#         if self.expert_feature_dim < 1:
#             raise ValueError("expert_feature_dim must be >= 1")
#         if self.decoder_feature_dim < 1:
#             raise ValueError("decoder_feature_dim must be >= 1")

#         self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
#         resolved_pe_mapping_size = int(pe_mapping_size) if pe_mapping_size is not None else int(in_features)
#         self.pos_enc = PositionalEncoding(
#             in_features=in_features,
#             mapping_size=resolved_pe_mapping_size,
#         )
#         pe_dim = self.pos_enc.out_dim
#         self.gating = ViewGating(
#             in_features=pe_dim,
#             view_embed_dim=view_embed_dim,
#             num_experts=num_experts,
#             hidden_dim=gate_hidden_dim,
#             num_layers=gate_num_layers,
#             first_omega_0=gate_first_omega_0,
#             hidden_omega_0=gate_hidden_omega_0,
#         )

#         self.experts = nn.ModuleList(
#             [
#                 ExpertEncoder(
#                     in_features=pe_dim,
#                     feature_dim=expert_feature_dim,
#                     use_positional_encoding=False,
#                     hidden_dim=expert_hidden_dim,
#                     num_layers=expert_num_layers,
#                     first_omega_0=expert_first_omega_0,
#                     hidden_omega_0=expert_hidden_omega_0,
#                 )
#                 for _ in range(num_experts)
#             ]
#         )

#         decoder_in_dim = self.expert_feature_dim
#         self.decoder = LightDecoder(
#             in_dim=decoder_in_dim,
#             out_dim=self.decoder_feature_dim,
#             first_omega_0=decoder_first_omega_0,
#             hidden_omega_0=decoder_hidden_omega_0,
#             num_res_blocks=int(decoder_num_res_blocks),
#         )

#         head_hidden_dim = max(8, self.decoder_feature_dim // 4)
#         self.heads = nn.ModuleDict(
#             {
#                 name: SirenMLP(
#                     in_dim=self.decoder_feature_dim,
#                     out_dim=out_dim,
#                     hidden_dim=head_hidden_dim,
#                     num_layers=3,
#                     first_omega_0=decoder_first_omega_0,
#                     hidden_omega_0=decoder_hidden_omega_0,
#                 )
#                 for name, out_dim in self.view_dims.items()
#             }
#         )

#         pos_enc_params = _count_parameters(self.pos_enc)
#         view_embedding_params = _count_parameters(self.view_embedding)
#         gating_params = _count_parameters(self.gating)
#         experts_params = _count_parameters(self.experts)
#         decoder_params = _count_parameters(self.decoder)
#         heads_params = _count_parameters(self.heads)
#         logger.info(
#             "LightBasisExpert init params: pos_enc=%s view_embedding=%s gating=%s experts=%s decoder=%s heads=%s",
#             f"{pos_enc_params:,}",
#             f"{view_embedding_params:,}",
#             f"{gating_params:,}",
#             f"{experts_params:,}",
#             f"{decoder_params:,}",
#             f"{heads_params:,}",
#         )

#     def _topk_mask(self, probs: torch.Tensor) -> torch.Tensor:
#         _, indices = torch.topk(probs, k=self.top_k, dim=-1)
#         mask = torch.zeros_like(probs)
#         return mask.scatter_(1, indices, 1.0)

#     def forward(
#         self,
#         coords: torch.Tensor,
#         request: Optional[str] = None,
#         *,
#         hard_topk: bool = True,
#         return_aux: bool = False,
#     ):
#         if request is not None and request not in self.view_dims:
#             raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

#         x_pe = self.pos_enc(coords)
#         expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)  # (B, M, F)

#         preds = {}
#         probs_list: List[torch.Tensor] = []
#         masks_list: List[torch.Tensor] = []
#         h_views: List[torch.Tensor] = []
#         shared_feats: List[torch.Tensor] = []

#         for view_idx, name in enumerate(self.view_names):
#             view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
#             view_embed = self.view_embedding(view_ids)
#             probs, _ = self.gating(x_pe, view_embed)
#             mask = self._topk_mask(probs)
#             masked_probs = probs * mask
#             masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
#             weights = masked_probs if hard_topk else probs

#             h_v = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
#             decoder_in = h_v
#             shared_feat = self.decoder(decoder_in)
#             preds[name] = self.heads[name](shared_feat)

#             probs_list.append(probs)
#             masks_list.append(mask)
#             h_views.append(h_v)
#             shared_feats.append(shared_feat)

#         output = preds if request is None else preds[request]
#         if return_aux:
#             aux = {
#                 "probs": torch.stack(probs_list, dim=1),  # (B, V, M)
#                 "masks": torch.stack(masks_list, dim=1),  # (B, V, M)
#                 "H_views": torch.stack(h_views, dim=1),
#                 "H_shared": torch.stack(shared_feats, dim=1),
#                 "expert_feats": expert_feats,
#             }
#             return output, aux
#         return output

#     def pretrain_forward(self, coords: torch.Tensor) -> torch.Tensor:
#         x_pe = self.pos_enc(coords)
#         logits_list: List[torch.Tensor] = []
#         for view_idx in range(self.num_views):
#             view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
#             view_embed = self.view_embedding(view_ids)
#             _, logits = self.gating(x_pe, view_embed)
#             logits_list.append(logits)
#         return torch.stack(logits_list, dim=0).mean(dim=0)

#     def pretrain_parameters(self):
#         return list(self.gating.parameters()) + list(self.view_embedding.parameters())

#     def pretrain_teacher_shared_feat(self, coords: torch.Tensor, teacher_mode: str = "random_topk"):
#         x_pe = self.pos_enc(coords)
#         expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)  # (B, M, F)
#         mode = (teacher_mode or "uniform").strip().lower()
#         if mode == "uniform":
#             weights = torch.full(
#                 (expert_feats.shape[0], self.num_experts),
#                 1.0 / float(self.num_experts),
#                 device=expert_feats.device,
#                 dtype=expert_feats.dtype,
#             )
#         elif mode == "random_topk":
#             weights = torch.zeros(
#                 (expert_feats.shape[0], self.num_experts),
#                 device=expert_feats.device,
#                 dtype=expert_feats.dtype,
#             )
#             k = max(1, min(self.top_k, self.num_experts))
#             idx = torch.randint(0, self.num_experts, (expert_feats.shape[0], k), device=expert_feats.device)
#             weights.scatter_(1, idx, 1.0 / float(k))
#         else:
#             raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
#         h_teacher = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
#         shared_teacher = self.decoder(h_teacher)
#         return shared_teacher

#     def pretrain_stage1_parameters(self):
#         return self.experts.parameters()

#     def pretrain_stage1_expert_feats(self, coords: torch.Tensor) -> torch.Tensor:
#         x_pe = self.pos_enc(coords)
#         return torch.stack([expert(x_pe) for expert in self.experts], dim=1)

#     def pretrain_stage2_parameters(self):
#         return list(self.gating.parameters()) + list(self.view_embedding.parameters())

#     def pretrain_stage2_router(self, coords: torch.Tensor, temperature: float = 1.0):
#         x_pe = self.pos_enc(coords)
#         expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)
#         probs_list: List[torch.Tensor] = []
#         masks_list: List[torch.Tensor] = []
#         for view_idx in range(self.num_views):
#             view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
#             view_embed = self.view_embedding(view_ids)
#             _probs, logits = self.gating(x_pe, view_embed)
#             temp = float(temperature) if temperature is not None else 1.0
#             if temp <= 0:
#                 temp = 1.0
#             probs = torch.softmax(logits / temp, dim=-1)
#             probs_list.append(probs)
#             masks_list.append(torch.ones_like(probs))
#         probs = torch.stack(probs_list, dim=1)
#         masks = torch.stack(masks_list, dim=1)
#         return probs, masks, expert_feats


# def build_light_basis_expert_from_config(cfg: Dict, view_specs: Dict[str, int]) -> LightBasisExpert:
#     base_dim_raw = cfg.get("base_dim")
#     if base_dim_raw is None:
#         raise ValueError("light_basis_expert requires 'base_dim' in model config.")

#     base_dim = int(base_dim_raw)
#     pe_mapping_raw = cfg.get("pe_mapping_size")
#     decoder_feature_raw = cfg.get("decoder_feature_dim")

#     pe_mapping_size = int(pe_mapping_raw) if pe_mapping_raw is not None else base_dim
#     expert_feature_dim = int(cfg.get("expert_feature_dim", 8 * base_dim))
#     view_embed_dim = int(cfg.get("view_embed_dim", base_dim))
#     expert_hidden_dim = int(cfg.get("expert_hidden_dim", 4 * base_dim))
#     gate_hidden_dim = int(cfg.get("gate_hidden_dim", 8 * base_dim))
#     decoder_feature_dim = (
#         int(decoder_feature_raw) if decoder_feature_raw is not None else expert_feature_dim
#     )
#     decoder_num_res_blocks = int(cfg.get("decoder_num_res_blocks", 1))
#     return LightBasisExpert(
#         in_features=int(cfg.get("in_features", 4)),
#         view_specs=view_specs,
#         num_experts=int(cfg.get("num_experts", 7)),
#         expert_feature_dim=expert_feature_dim,
#         top_k=int(cfg.get("top_k", 2)),
#         view_embed_dim=view_embed_dim,
#         pe_mapping_size=pe_mapping_size,
#         expert_hidden_dim=expert_hidden_dim,
#         expert_num_layers=int(cfg.get("expert_num_layers", 3)),
#         gate_hidden_dim=gate_hidden_dim,
#         gate_num_layers=int(cfg.get("gate_num_layers", 3)),
#         decoder_feature_dim=decoder_feature_dim,
#         decoder_num_res_blocks=decoder_num_res_blocks,
#         expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
#         expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
#         gate_first_omega_0=float(cfg.get("gate_first_omega_0", 30.0)),
#         gate_hidden_omega_0=float(cfg.get("gate_hidden_omega_0", 30.0)),
#         decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
#         decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
#     )

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .components import ExpertEncoder, SirenMLP, ViewGating, PositionalEncoding


class LightBasisExpert(nn.Module):
    """
    Light MoE-INR:
    - experts + view-conditioned gating -> h_v
    - shared decoder maps h_v to a shared feature
    - per-view heads map shared feature to each attribute
    """

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        num_experts: int = 7,
        positional_out_dim: int = 128,
        expert_feature_dim: int = 128,
        top_k: int = 2,
        view_embed_dim: int = 16,
        expert_num_frequencies: int = 6,
        expert_hidden_dim: int = 128,
        expert_num_layers: int = 3,
        gate_hidden_dim: int = 128,
        gate_num_layers: int = 3,
        decoder_feature_dim: int = 128,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        head_hidden_dim: Optional[int] = None,
        head_num_layers: int = 2,
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
        gate_first_omega_0: float = 30.0,
        gate_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
        head_first_omega_0: float = 30.0,
        head_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")
        if head_num_layers < 2:
            raise ValueError("head_num_layers must be >= 2")

        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_feature_dim = expert_feature_dim

        self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
        self.pos_enc = PositionalEncoding(
            in_features=in_features,
        )
        pe_dim = self.pos_enc.out_dim
        self.gating = ViewGating(
            in_features=pe_dim,
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
                    in_features=pe_dim,
                    feature_dim=expert_feature_dim,
                    use_positional_encoding=False,
                    num_frequencies=expert_num_frequencies,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )

        decoder_in_dim = expert_feature_dim
        self.decoder = SirenMLP(
            in_dim=decoder_in_dim,
            out_dim=decoder_feature_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            first_omega_0=decoder_first_omega_0,
            hidden_omega_0=decoder_hidden_omega_0,
        )

        head_hidden_dim = decoder_feature_dim if head_hidden_dim is None else int(head_hidden_dim)
        self.heads = nn.ModuleDict(
            {
                name: SirenMLP(
                    in_dim=decoder_feature_dim,
                    out_dim=out_dim,
                    hidden_dim=head_hidden_dim,
                    num_layers=head_num_layers,
                    first_omega_0=head_first_omega_0,
                    hidden_omega_0=head_hidden_omega_0,
                )
                for name, out_dim in self.view_dims.items()
            }
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
        hard_topk: bool = True,
        return_aux: bool = False,
    ):
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        x_pe = self.pos_enc(coords)
        expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)  # (B, M, F)

        preds = {}
        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        h_views: List[torch.Tensor] = []
        shared_feats: List[torch.Tensor] = []

        for view_idx, name in enumerate(self.view_names):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            probs, _ = self.gating(x_pe, view_embed)
            mask = self._topk_mask(probs)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            weights = masked_probs if hard_topk else probs

            h_v = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
            decoder_in = h_v
            shared_feat = self.decoder(decoder_in)
            preds[name] = self.heads[name](shared_feat)

            probs_list.append(probs)
            masks_list.append(mask)
            h_views.append(h_v)
            shared_feats.append(shared_feat)

        output = preds if request is None else preds[request]
        if return_aux:
            aux = {
                "probs": torch.stack(probs_list, dim=1),  # (B, V, M)
                "masks": torch.stack(masks_list, dim=1),  # (B, V, M)
                "H_views": torch.stack(h_views, dim=1),
                "H_shared": torch.stack(shared_feats, dim=1),
                "expert_feats": expert_feats,
            }
            return output, aux
        return output

    def pretrain_forward(self, coords: torch.Tensor) -> torch.Tensor:
        x_pe = self.pos_enc(coords)
        logits_list: List[torch.Tensor] = []
        for view_idx in range(self.num_views):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            _, logits = self.gating(x_pe, view_embed)
            logits_list.append(logits)
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def pretrain_parameters(self):
        return list(self.gating.parameters()) + list(self.view_embedding.parameters())

    def pretrain_teacher_shared_feat(self, coords: torch.Tensor, teacher_mode: str = "random_topk"):
        x_pe = self.pos_enc(coords)
        expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)  # (B, M, F)
        mode = (teacher_mode or "uniform").strip().lower()
        if mode == "uniform":
            weights = torch.full(
                (expert_feats.shape[0], self.num_experts),
                1.0 / float(self.num_experts),
                device=expert_feats.device,
                dtype=expert_feats.dtype,
            )
        elif mode == "random_topk":
            weights = torch.zeros(
                (expert_feats.shape[0], self.num_experts),
                device=expert_feats.device,
                dtype=expert_feats.dtype,
            )
            k = max(1, min(self.top_k, self.num_experts))
            idx = torch.randint(0, self.num_experts, (expert_feats.shape[0], k), device=expert_feats.device)
            weights.scatter_(1, idx, 1.0 / float(k))
        else:
            raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
        h_teacher = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
        shared_teacher = self.decoder(h_teacher)
        return shared_teacher

    def pretrain_stage1_parameters(self):
        return self.experts.parameters()

    def pretrain_stage1_expert_feats(self, coords: torch.Tensor) -> torch.Tensor:
        x_pe = self.pos_enc(coords)
        return torch.stack([expert(x_pe) for expert in self.experts], dim=1)

    def pretrain_stage2_parameters(self):
        return list(self.gating.parameters()) + list(self.view_embedding.parameters())

    def pretrain_stage2_router(self, coords: torch.Tensor, temperature: float = 1.0):
        x_pe = self.pos_enc(coords)
        expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)
        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        for view_idx in range(self.num_views):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            _probs, logits = self.gating(x_pe, view_embed)
            temp = float(temperature) if temperature is not None else 1.0
            if temp <= 0:
                temp = 1.0
            probs = torch.softmax(logits / temp, dim=-1)
            probs_list.append(probs)
            masks_list.append(torch.ones_like(probs))
        probs = torch.stack(probs_list, dim=1)
        masks = torch.stack(masks_list, dim=1)
        return probs, masks, expert_feats


def build_light_basis_expert_from_config(cfg: Dict, view_specs: Dict[str, int]) -> LightBasisExpert:
    base_dim = cfg.get("base_dim")
    head_hidden_raw = cfg.get("head_hidden_dim")
    decoder_feature_raw = cfg.get("decoder_feature_dim")
    
    base_dim = int(base_dim)
    positional_out_dim = 1 * base_dim
    expert_feature_dim = 8 * base_dim
    view_embed_dim = base_dim
    expert_hidden_dim = 4 * base_dim
    gate_hidden_dim = 8 * base_dim
    decoder_hidden_dim = 8 * base_dim
    decoder_feature_dim = (
        int(decoder_feature_raw) if decoder_feature_raw is not None else expert_feature_dim
    )
    head_hidden_dim = (
        int(head_hidden_raw) if head_hidden_raw is not None else decoder_feature_dim
    )
    return LightBasisExpert(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        num_experts=int(cfg.get("num_experts", 7)),
        expert_feature_dim=expert_feature_dim,
        positional_out_dim=positional_out_dim,
        top_k=int(cfg.get("top_k", 2)),
        view_embed_dim=view_embed_dim,
        expert_num_frequencies=int(cfg.get("expert_num_frequencies", 6)),
        expert_hidden_dim=expert_hidden_dim,
        expert_num_layers=int(cfg.get("expert_num_layers", 3)),
        gate_hidden_dim=gate_hidden_dim,
        gate_num_layers=int(cfg.get("gate_num_layers", 3)),
        decoder_feature_dim=decoder_feature_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=int(cfg.get("decoder_num_layers", 3)),
        head_hidden_dim=head_hidden_dim,
        head_num_layers=int(cfg.get("head_num_layers", 2)),
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        gate_first_omega_0=float(cfg.get("gate_first_omega_0", 30.0)),
        gate_hidden_omega_0=float(cfg.get("gate_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
        head_first_omega_0=float(cfg.get("head_first_omega_0", 30.0)),
        head_hidden_omega_0=float(cfg.get("head_hidden_omega_0", 30.0)),
    )
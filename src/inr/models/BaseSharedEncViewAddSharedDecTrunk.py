import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .moe_inr import SirenMLP


class PositionalEncoding(nn.Module):
    """NeRF-style positional encoding."""

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 6,
        include_input: bool = True,
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


class SharedEncoder(nn.Module):
    """Shared encoder that maps coords to latent features."""

    def __init__(
        self,
        in_features: int,
        feature_dim: int,
        use_positional_encoding: bool = True,
        num_frequencies: int = 6,
        include_input: bool = True,
        hidden_dim: int = 128,
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


class BaseSharedEncViewAddSharedDecTrunk(nn.Module):
    """
    Minimal shared-encoder baseline:
    - single shared encoder maps coords -> h
    - per-view embedding added to h (no gating, no fusion)
    - per-view decoders map h_view -> attribute
    """

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        shared_feature_dim: int = 128,
        view_embed_dim: int = 16,
        shared_use_positional_encoding: bool = True,
        shared_num_frequencies: int = 6,
        shared_hidden_dim: int = 128,
        shared_num_layers: int = 3,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        shared_first_omega_0: float = 30.0,
        shared_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")

        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
        self.view_embed_proj = nn.Linear(view_embed_dim, shared_feature_dim, bias=False)

        self.shared_encoder = SharedEncoder(
            in_features=in_features,
            feature_dim=shared_feature_dim,
            use_positional_encoding=shared_use_positional_encoding,
            num_frequencies=shared_num_frequencies,
            hidden_dim=shared_hidden_dim,
            num_layers=shared_num_layers,
            first_omega_0=shared_first_omega_0,
            hidden_omega_0=shared_hidden_omega_0,
        )

        self.decoders = nn.ModuleDict(
            {
                name: SirenMLP(
                    in_dim=shared_feature_dim,
                    out_dim=out_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    first_omega_0=decoder_first_omega_0,
                    hidden_omega_0=decoder_hidden_omega_0,
                )
                for name, out_dim in self.view_dims.items()
            }
        )

    def forward(
        self,
        coords: torch.Tensor,
        request: Optional[str] = None,
        *,
        return_aux: bool = False,
    ):
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        shared_feat = self.shared_encoder(coords)  # (B, F)

        preds = {}
        h_views: List[torch.Tensor] = []
        for view_idx, name in enumerate(self.view_names):
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            h_view = shared_feat + self.view_embed_proj(view_embed)
            h_views.append(h_view)
            preds[name] = self.decoders[name](h_view)

        output = preds if request is None else preds[request]
        if return_aux:
            aux = {
                "H_views": torch.stack(h_views, dim=1),
                "shared_feat": shared_feat,
            }
            return output, aux
        return output


def build_base_shared_enc_view_add_shared_dec_trunk_from_config(
    cfg: Dict, view_specs: Dict[str, int]
) -> BaseSharedEncViewAddSharedDecTrunk:
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        shared_feature_dim = 8 * base_dim
        view_embed_dim = base_dim
        shared_hidden_dim = 8 * base_dim
        decoder_hidden_dim = 8 * base_dim
    else:
        shared_feature_dim = int(cfg.get("shared_feature_dim", 128))
        view_embed_dim = int(cfg.get("view_embed_dim", 16))
        shared_hidden_dim = int(cfg.get("shared_hidden_dim", 128))
        decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 128))
    return BaseSharedEncViewAddSharedDecTrunk(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        shared_feature_dim=shared_feature_dim,
        view_embed_dim=view_embed_dim,
        shared_use_positional_encoding=bool(cfg.get("shared_use_positional_encoding", True)),
        shared_num_frequencies=int(cfg.get("shared_num_frequencies", 6)),
        shared_hidden_dim=shared_hidden_dim,
        shared_num_layers=int(cfg.get("shared_num_layers", 3)),
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=int(cfg.get("decoder_num_layers", 3)),
        shared_first_omega_0=float(cfg.get("shared_first_omega_0", 30.0)),
        shared_hidden_omega_0=float(cfg.get("shared_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
    )

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .components import PositionalEncoding, SirenMLP


class SharedEncINR(nn.Module):
    """
    Ablation of LightBasisExpert with a single shared encoder.

    - positional encoding + shared encoder -> enc_feat
    - per-view embedding projection produces h_v
    - shared decoder maps h_v to a shared feature
    - per-view heads map shared feature to each attribute
    """

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        enc_feature_dim: int = 128,
        view_embed_dim: int = 16,
        enc_num_frequencies: int = 6,
        enc_hidden_dim: int = 128,
        enc_layer_num: int = 3,
        decoder_feature_dim: int = 128,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        head_hidden_dim: Optional[int] = None,
        head_num_layers: int = 2,
        enc_first_omega_0: float = 30.0,
        enc_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
        head_first_omega_0: float = 30.0,
        head_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")
        if enc_layer_num < 2:
            raise ValueError("enc_layer_num must be >= 2")
        if head_num_layers < 2:
            raise ValueError("head_num_layers must be >= 2")

        self.view_names = list(view_specs.keys())
        self.view_name_to_idx = {name: idx for idx, name in enumerate(self.view_names)}
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.enc_feature_dim = enc_feature_dim

        self.view_embedding = nn.Embedding(self.num_views, view_embed_dim)
        self.view_embed_proj = nn.Linear(view_embed_dim, enc_feature_dim, bias=False)

        self.pos_enc = PositionalEncoding(
            in_features=in_features,
            num_frequencies=enc_num_frequencies,
        )
        pe_dim = self.pos_enc.out_dim
        self.encoder = SirenMLP(
            in_dim=pe_dim,
            out_dim=enc_feature_dim,
            hidden_dim=enc_hidden_dim,
            num_layers=enc_layer_num,
            first_omega_0=enc_first_omega_0,
            hidden_omega_0=enc_hidden_omega_0,
        )

        self.decoder = SirenMLP(
            in_dim=enc_feature_dim,
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

        _ = hard_topk
        x_pe = self.pos_enc(coords)
        enc_feat = self.encoder(x_pe)

        preds = {}
        h_views: List[torch.Tensor] = []
        shared_feats: List[torch.Tensor] = []

        if request is None:
            view_items = enumerate(self.view_names)
        else:
            view_items = [(self.view_name_to_idx[request], request)]

        for view_idx, name in view_items:
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            h_v = enc_feat + self.view_embed_proj(view_embed)
            shared_feat = self.decoder(h_v)
            preds[name] = self.heads[name](shared_feat)
            h_views.append(h_v)
            shared_feats.append(shared_feat)

        output = preds if request is None else preds[request]
        if return_aux:
            aux = {
                "H_views": torch.stack(h_views, dim=1),
                "H_shared": torch.stack(shared_feats, dim=1),
                "enc_feat": enc_feat,
            }
            return output, aux
        return output


def build_shared_enc_inr_from_config(cfg: Dict, view_specs: Dict[str, int]) -> SharedEncINR:
    base_dim = cfg.get("base_dim")
    enc_base_dim_raw = cfg.get("enc_base_dim", base_dim)
    dec_base_dim_raw = cfg.get("dec_base_dim", base_dim)
    if enc_base_dim_raw is None or dec_base_dim_raw is None:
        raise ValueError(
            "shared_enc_inr requires model.enc_base_dim and model.dec_base_dim, or model.base_dim"
        )

    head_hidden_raw = cfg.get("head_hidden_dim")
    decoder_feature_raw = cfg.get("decoder_feature_dim")

    enc_base_dim = int(enc_base_dim_raw)
    dec_base_dim = int(dec_base_dim_raw)
    decoder_feature_dim = (
        int(decoder_feature_raw) if decoder_feature_raw is not None else 8 * dec_base_dim
    )
    enc_feature_dim = decoder_feature_dim
    view_embed_dim = enc_base_dim
    enc_hidden_dim = 8 * enc_base_dim
    decoder_hidden_dim = 8 * dec_base_dim
    head_hidden_dim = (
        int(head_hidden_raw) if head_hidden_raw is not None else decoder_feature_dim
    )

    return SharedEncINR(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        enc_feature_dim=enc_feature_dim,
        view_embed_dim=view_embed_dim,
        enc_num_frequencies=int(cfg.get("expert_num_frequencies", 6)),
        enc_hidden_dim=enc_hidden_dim,
        enc_layer_num=int(cfg.get("enc_layer_num", cfg.get("expert_num_layers", 3))),
        decoder_feature_dim=decoder_feature_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=int(cfg.get("decoder_num_layers", 3)),
        head_hidden_dim=head_hidden_dim,
        head_num_layers=int(cfg.get("head_num_layers", 2)),
        enc_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        enc_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
        head_first_omega_0=float(cfg.get("head_first_omega_0", 30.0)),
        head_hidden_omega_0=float(cfg.get("head_hidden_omega_0", 30.0)),
    )

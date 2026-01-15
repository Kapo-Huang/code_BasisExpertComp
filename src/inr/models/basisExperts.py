import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .siren import SineLayer


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

class BasisExperts(nn.Module):
    """
    DMVC-CE style MoE-INR:
    - experts pool as a soft-shared encoder
    - view-conditioned gating for sparse routing
    - concatenated multi-view representation
    - view-specific decoder heads
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

        fused_dim = self.num_views * expert_feature_dim
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

        # Compute expert features
        expert_feats = torch.stack([expert(coords) for expert in self.experts], dim=1)  # (B, M, F)

        probs_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        h_views: List[torch.Tensor] = []

        for view_idx, _name in enumerate(self.view_names):
            # Get gating probabilities for this view
            view_ids = torch.full((coords.shape[0],), view_idx, device=coords.device, dtype=torch.long)
            view_embed = self.view_embedding(view_ids)
            probs, _ = self.gating(coords, view_embed)
            mask = self._topk_mask(probs)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
            # Fuse expert features for this view
            weights = masked_probs if hard_topk else probs
            h_v = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)

            probs_list.append(probs)
            masks_list.append(mask)
            h_views.append(h_v)

        h_views_tensor = torch.stack(h_views, dim=1)  # (B, V, F)
        h_fused = torch.cat(h_views, dim=-1)  # (B, V * F)

        preds = {name: self.decoders[name](h_fused) for name in self.view_names}
        output = preds if request is None else preds[request]

        if return_aux:
            aux = {
                "probs": torch.stack(probs_list, dim=1),  # (B, V, M)
                "masks": torch.stack(masks_list, dim=1),  # (B, V, M)
                "H_views": h_views_tensor,
                "H_fused": h_fused,
                "expert_feats": expert_feats,
            }
            return output, aux
        return output

class MultiViewCoordDataset(Dataset):
    """
    Dataset for shared coords with multiple attribute targets.

    coords.npy: (N, D)
    y_attr.npy: (N, C_attr)
    """

    def __init__(
        self,
        coords_path: str,
        attr_paths: Dict[str, str],
        normalize: bool = True,
        stats_path: Optional[str] = None,
    ):
        coords = np.load(coords_path, mmap_mode="r")
        if not attr_paths:
            raise ValueError("attr_paths must be a non-empty dict")

        attrs = {}
        for name, path in attr_paths.items():
            data = np.load(path, mmap_mode="r")
            if data.shape[0] != coords.shape[0]:
                raise ValueError(f"Mismatched samples for {name}: {data.shape[0]} vs {coords.shape[0]}")
            attrs[name] = data

        # self.x = torch.from_numpy(coords.astype(np.float32))
        # self.y = {name: torch.from_numpy(arr.astype(np.float32)) for name, arr in attrs.items()}
        self.x = torch.from_numpy(coords)
        self.y = {name: torch.from_numpy(arr) for name, arr in attrs.items()}
        self.normalize = normalize
        self.stats_path = stats_path

        if normalize:
            stats_loaded = False
            if stats_path and Path(stats_path).exists():
                stats = np.load(stats_path)
                try:
                    self.x_mean = torch.from_numpy(stats["x_mean"]).to(torch.float32)
                    self.x_std = torch.from_numpy(stats["x_std"]).to(torch.float32)
                    self.y_mean = {}
                    self.y_std = {}
                    for name in self.y.keys():
                        mean_key = f"y_mean_{name}"
                        std_key = f"y_std_{name}"
                        self.y_mean[name] = torch.from_numpy(stats[mean_key]).to(torch.float32)
                        self.y_std[name] = torch.from_numpy(stats[std_key]).to(torch.float32)
                        self.y_std[name] = torch.where(
                            self.y_std[name] == 0, torch.ones_like(self.y_std[name]), self.y_std[name]
                        )
                    self.x_std = torch.where(self.x_std == 0, torch.ones_like(self.x_std), self.x_std)
                    stats_loaded = True
                except KeyError:
                    stats_loaded = False

            if not stats_loaded:
                self.x_mean, self.x_std = self._compute_stats(self.x)
                self.y_mean = {}
                self.y_std = {}
                for name, tensor in self.y.items():
                    mean, std = self._compute_stats(tensor)
                    self.y_mean[name] = mean
                    self.y_std[name] = std
                if stats_path:
                    self._save_stats(stats_path)

            self.x = (self.x - self.x_mean) / self.x_std
            for name, tensor in self.y.items():
                self.y[name] = (tensor - self.y_mean[name]) / self.y_std[name]
        else:
            self.x_mean = torch.zeros_like(self.x[:1])
            self.x_std = torch.ones_like(self.x[:1])
            self.y_mean = {name: torch.zeros_like(tensor[:1]) for name, tensor in self.y.items()}
            self.y_std = {name: torch.ones_like(tensor[:1]) for name, tensor in self.y.items()}

    @staticmethod
    def _compute_stats(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = tensor.mean(0, keepdim=True)
        std = tensor.std(0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    def _save_stats(self, stats_path: str) -> None:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x_mean": self.x_mean.numpy(),
            "x_std": self.x_std.numpy(),
        }
        for name in self.y.keys():
            payload[f"y_mean_{name}"] = self.y_mean[name].numpy()
            payload[f"y_std_{name}"] = self.y_std[name].numpy()
        np.savez(stats_path, **payload)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        xb = self.x[idx]
        yb = {name: tensor[idx] for name, tensor in self.y.items()}
        return xb, yb

    def view_specs(self) -> Dict[str, int]:
        return {name: int(tensor.shape[1]) for name, tensor in self.y.items()}

    def denormalize_attr(self, name: str, y_norm: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return y_norm
        mean = self.y_mean[name].to(y_norm.device)
        std = self.y_std[name].to(y_norm.device)
        return y_norm * std + mean

# 重建损失
def reconstruction_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    if loss_type not in ("mse", "l1"):
        raise ValueError("loss_type must be 'mse' or 'l1'")

    total = 0.0
    for name, pred in preds.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        target = targets[name]
        if loss_type == "mse":
            total = total + weight * F.mse_loss(pred, target)
        else:
            total = total + weight * F.l1_loss(pred, target)
    return total

# 专家负载均衡
def load_balance_loss(probs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    rho = masks.mean(dim=(0, 1))
    rho_hat = probs.mean(dim=(0, 1))
    return torch.mean(rho * rho_hat)

# 专家之间的多样性
def diversity_loss(expert_feats: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    num_experts = expert_feats.shape[1]
    if num_experts < 2:
        return torch.zeros((), device=expert_feats.device)
    mean_e = expert_feats.mean(dim=0)  # (M, F)
    dists = torch.cdist(mean_e, mean_e, p=2)
    sim = torch.exp(-(dists ** 2) / (2.0 * sigma * sigma))
    mask = 1.0 - torch.eye(num_experts, device=expert_feats.device)
    return (sim * mask).sum() / (mask.sum() + 1e-9)


# def example_train_step(
#     model: BasisExperts,
#     batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
#     *,
#     weights: Optional[Dict[str, float]] = None,
#     lam_eq: float = 0.0,
#     gam_div: float = 0.0,
#     loss_type: str = "mse",
# ) -> Dict[str, torch.Tensor]:
#     coords, targets = batch
#     preds, aux = model(coords, return_aux=True)
#     loss_recon = reconstruction_loss(preds, targets, weights=weights, loss_type=loss_type)
#     loss = loss_recon

#     if lam_eq > 0.0:
#         loss_eq = load_balance_loss(aux["probs"], aux["masks"])
#         loss = loss + lam_eq * loss_eq
#     else:
#         loss_eq = torch.zeros((), device=coords.device)

#     if gam_div > 0.0:
#         loss_div = diversity_loss(aux["expert_feats"])
#         loss = loss + gam_div * loss_div
#     else:
#         loss_div = torch.zeros((), device=coords.device)

#     return {
#         "loss": loss,
#         "loss_recon": loss_recon,
#         "loss_eq": loss_eq,
#         "loss_div": loss_div,
#     }


def build_basisExperts_from_config(cfg: Dict, view_specs: Dict[str, int]) -> BasisExperts:
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
    return BasisExperts(
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
    )

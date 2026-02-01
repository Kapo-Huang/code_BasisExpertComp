from typing import Dict, Optional

import torch
import torch.nn.functional as F


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


def load_balance_loss(probs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    rho = masks.mean(dim=(0, 1))
    rho_hat = probs.mean(dim=(0, 1))
    return torch.mean(rho * rho_hat)


def diversity_loss(expert_feats: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    num_experts = expert_feats.shape[1]
    if num_experts < 2:
        return torch.zeros((), device=expert_feats.device)
    mean_e = expert_feats.mean(dim=0)  # (M, F)
    dists = torch.cdist(mean_e, mean_e, p=2)
    sim = torch.exp(-(dists ** 2) / (2.0 * sigma * sigma))
    mask = 1.0 - torch.eye(num_experts, device=expert_feats.device)
    return (sim * mask).sum() / (mask.sum() + 1e-9)


def orth_loss(expert_feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if expert_feats.dim() != 3:
        raise ValueError(f"expert_feats must be 3D (B, M, F), got shape {tuple(expert_feats.shape)}")
    bsz, num_experts, feat_dim = expert_feats.shape
    if bsz < 2 or num_experts < 2:
        return expert_feats.new_zeros(())
    z = expert_feats - expert_feats.mean(dim=0, keepdim=True)
    z = z / (expert_feats.std(dim=0, keepdim=True) + eps)
    z = z.permute(0, 2, 1).reshape(-1, num_experts)
    c = (z.t() @ z) / float(z.shape[0])
    off = c - torch.eye(num_experts, device=c.device, dtype=c.dtype)
    return (off * off).sum() / (num_experts * (num_experts - 1))

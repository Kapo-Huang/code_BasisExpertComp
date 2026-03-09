from typing import Dict, Optional, Tuple

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


def reconstruction_loss_with_breakdown(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
    loss_type: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]]]:
    """
    Returns:
      - weighted_attr_sum_loss: sum_i w_i * L_i
      - weighted_dim_normalized_loss: (sum_i w_i * d_i * L_i) / (sum_i w_i * d_i)
      - details[name]:
          dim: output dimension d_i (per sample)
          weight: w_i
          loss_per_dim: L_i (mean over batch and all output dims)
          loss_sum_dims: d_i * L_i (mean over batch, summed over output dims)
    """
    if loss_type not in ("mse", "l1"):
        raise ValueError("loss_type must be 'mse' or 'l1'")
    if not preds:
        raise ValueError("preds must be non-empty")

    first_tensor = next(iter(preds.values()))
    weighted_attr_sum_loss = torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)
    weighted_dim_numer = torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)
    weighted_dim_denom = torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)
    details: Dict[str, Dict[str, float]] = {}

    for name, pred in preds.items():
        if name not in targets:
            raise KeyError(f"Missing target for prediction key '{name}'")
        target = targets[name]
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[0] <= 0:
            raise ValueError(f"Invalid batch size for '{name}': {pred.shape[0]}")

        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        loss_per_dim = F.mse_loss(pred, target) if loss_type == "mse" else F.l1_loss(pred, target)
        out_dim = int(pred[0].numel())
        loss_sum_dims = loss_per_dim * float(out_dim)

        weighted_attr_sum_loss = weighted_attr_sum_loss + weight * loss_per_dim
        weighted_dim_numer = weighted_dim_numer + weight * loss_sum_dims
        weighted_dim_denom = weighted_dim_denom + (weight * float(out_dim))

        details[name] = {
            "dim": float(out_dim),
            "weight": float(weight),
            "loss_per_dim": float(loss_per_dim.detach().item()),
            "loss_sum_dims": float(loss_sum_dims.detach().item()),
        }

    weighted_dim_normalized_loss = weighted_dim_numer / (weighted_dim_denom + 1e-12)
    return weighted_attr_sum_loss, weighted_dim_normalized_loss, details


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

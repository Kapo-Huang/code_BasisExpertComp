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
    weighted_attr_sum_loss, _, _, _ = reconstruction_loss_with_breakdown(
        preds,
        targets,
        weights=weights,
        loss_type=loss_type,
    )
    return weighted_attr_sum_loss


def reconstruction_loss_with_breakdown(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
    loss_type: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object], Dict[str, torch.Tensor]]:
    if loss_type not in ("mse", "l1"):
        raise ValueError("loss_type must be 'mse' or 'l1'")

    weighted_attr_sum_loss = None
    weighted_dim_numer = None
    weighted_dim_denom = 0.0
    details = {
        "selected_mode": "attr_sum",
        "selected_loss": 0.0,
        "weighted_attr_sum_loss": 0.0,
        "weighted_dim_normalized_loss": 0.0,
        "per_view": {},
        "weight_mode": "static",
    }
    per_view_loss_tensors: Dict[str, torch.Tensor] = {}

    for name, pred in preds.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        target = targets[name]
        if loss_type == "mse":
            loss_per_dim = F.mse_loss(pred, target)
        else:
            loss_per_dim = F.l1_loss(pred, target)

        out_dim = int(pred[0].numel())
        loss_sum_dims = loss_per_dim * float(out_dim)

        if weighted_attr_sum_loss is None:
            weighted_attr_sum_loss = weight * loss_per_dim
            weighted_dim_numer = weight * loss_sum_dims
        else:
            weighted_attr_sum_loss = weighted_attr_sum_loss + weight * loss_per_dim
            weighted_dim_numer = weighted_dim_numer + weight * loss_sum_dims
        weighted_dim_denom += weight * float(out_dim)

        per_view_loss_tensors[name] = loss_per_dim
        details["per_view"][name] = {
            "dim": float(out_dim),
            "weight": float(weight),
            "loss_sum_dims": float(loss_sum_dims.detach().item()),
            "loss_per_dim": float(loss_per_dim.detach().item()),
        }

    if weighted_attr_sum_loss is None or weighted_dim_numer is None:
        raise ValueError("reconstruction_loss requires at least one prediction-target pair")

    weighted_dim_normalized_loss = weighted_dim_numer / (weighted_dim_denom + 1e-12)

    details["selected_loss"] = float(weighted_attr_sum_loss.detach().item())
    details["weighted_attr_sum_loss"] = float(weighted_attr_sum_loss.detach().item())
    details["weighted_dim_normalized_loss"] = float(weighted_dim_normalized_loss.detach().item())
    return weighted_attr_sum_loss, weighted_dim_normalized_loss, details, per_view_loss_tensors

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

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAttrEMALoss(nn.Module):
    def __init__(
        self,
        attr_names: Iterable[str],
        beta: float = 0.95,
        eps: float = 1e-8,
        w_min: float = 0.2,
        w_max: float = 5.0,
        warmup_steps: int = 0,
        loss_type: str = "mse",
    ):
        super().__init__()
        attr_names = list(attr_names)
        if not attr_names:
            raise ValueError("attr_names must be non-empty")
        if len(set(attr_names)) != len(attr_names):
            raise ValueError("attr_names must not contain duplicates")
        if loss_type not in ("mse", "l1"):
            raise ValueError("loss_type must be 'mse' or 'l1'")

        self.attr_names: List[str] = attr_names
        self.beta = float(beta)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.warmup_steps = int(warmup_steps)
        self.loss_type = str(loss_type)

        self.register_buffer("ema", torch.ones(len(self.attr_names)))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def _compute_per_attr_losses(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for name in self.attr_names:
            if name not in preds:
                raise KeyError(f"Missing prediction for key '{name}'")
            if name not in targets:
                raise KeyError(f"Missing target for key '{name}'")
            pred = preds[name]
            tgt = targets[name]
            if pred.shape != tgt.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': pred={tuple(pred.shape)} target={tuple(tgt.shape)}"
                )
            loss_value = F.mse_loss(pred, tgt) if self.loss_type == "mse" else F.l1_loss(pred, tgt)
            losses.append(loss_value)
        return torch.stack(losses, dim=0)

    @torch.no_grad()
    def _update_ema(self, losses_vec: torch.Tensor):
        self.ema.mul_(self.beta).add_(losses_vec * (1.0 - self.beta))

    @torch.no_grad()
    def _compute_weights_from_ema(self) -> torch.Tensor:
        weights = 1.0 / (self.ema + self.eps)
        weights = weights / (weights.mean() + self.eps)
        return torch.clamp(weights, self.w_min, self.w_max)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        *,
        return_details: bool = False,
        return_tensors: bool = False,
        update_ema: bool = True,
    ):
        losses_t = self._compute_per_attr_losses(preds, targets)
        step_now = int(self.step.detach().item())

        if update_ema:
            self._update_ema(losses_t.detach())
        if step_now >= self.warmup_steps:
            with torch.no_grad():
                weights = self._compute_weights_from_ema()
        else:
            weights = torch.ones_like(losses_t)

        if update_ema:
            with torch.no_grad():
                self.step.add_(1)

        weights = weights.to(device=losses_t.device, dtype=losses_t.dtype)
        total = torch.sum(weights * losses_t)

        if not return_details and not return_tensors:
            return total

        details = None
        if return_details:
            details = {
                "per_attr_loss": {
                    name: float(losses_t[index].detach().item())
                    for index, name in enumerate(self.attr_names)
                },
                "ema": {
                    name: float(self.ema[index].detach().item())
                    for index, name in enumerate(self.attr_names)
                },
                "weights": {
                    name: float(weights[index].detach().item())
                    for index, name in enumerate(self.attr_names)
                },
                "total": float(total.detach().item()),
                "step": int(self.step.detach().item()),
                "warmup_steps": int(self.warmup_steps),
            }

        if return_details and return_tensors:
            return total, losses_t, weights, details
        if return_details:
            return total, details
        return total, losses_t, weights


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
    mean_e = expert_feats.mean(dim=0)
    dists = torch.cdist(mean_e, mean_e, p=2)
    sim = torch.exp(-(dists ** 2) / (2.0 * sigma * sigma))
    mask = 1.0 - torch.eye(num_experts, device=expert_feats.device)
    return (sim * mask).sum() / (mask.sum() + 1e-9)


def orth_loss(expert_feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if expert_feats.dim() != 3:
        raise ValueError(f"expert_feats must be 3D (B, M, F), got shape {tuple(expert_feats.shape)}")
    batch_size, num_experts, _ = expert_feats.shape
    if batch_size < 2 or num_experts < 2:
        return expert_feats.new_zeros(())
    z = expert_feats - expert_feats.mean(dim=0, keepdim=True)
    z = z / (expert_feats.std(dim=0, keepdim=True) + eps)
    z = z.permute(0, 2, 1).reshape(-1, num_experts)
    corr = (z.t() @ z) / float(z.shape[0])
    off_diag = corr - torch.eye(num_experts, device=corr.device, dtype=corr.dtype)
    return (off_diag * off_diag).sum() / (num_experts * (num_experts - 1))

from typing import Optional, TYPE_CHECKING

import torch

from inr.training.losses import (
    MultiAttrEMALoss,
    reconstruction_loss_with_breakdown,
)

if TYPE_CHECKING:
    from inr.training.loops import TrainingConfig


def compute_multiview_loss(
    model,
    xb,
    yb,
    cfg: "TrainingConfig",
    return_aux: bool = False,
    return_breakdown: bool = False,
    hard_topk: bool = True,
    ema_recon_loss: Optional[MultiAttrEMALoss] = None,
    update_ema_loss: bool = True,
):
    try:
        preds, aux = model(xb, return_aux=True, hard_topk=hard_topk)
    except TypeError:
        try:
            preds, aux = model(xb, return_aux=True)
        except TypeError:
            preds = model(xb)
            aux = {}

    recon_mode = str(getattr(cfg, "multiview_recon_reduction", "attr_sum") or "attr_sum").strip().lower()
    weight_mode = "static"
    ema_state = None

    if ema_recon_loss is not None:
        _, losses_t, dynamic_w, ema_details = ema_recon_loss(
            preds,
            yb,
            return_details=True,
            return_tensors=True,
            update_ema=update_ema_loss,
        )
        attr_names = list(ema_recon_loss.attr_names)
        effective_w = dynamic_w
        if cfg.view_loss_weights:
            base_w = torch.tensor(
                [float(cfg.view_loss_weights.get(name, 1.0)) for name in attr_names],
                device=losses_t.device,
                dtype=losses_t.dtype,
            )
            effective_w = effective_w * base_w
            effective_w = effective_w / (effective_w.mean() + 1e-12)

        dims = torch.tensor(
            [float(preds[name][0].numel()) for name in attr_names],
            device=losses_t.device,
            dtype=losses_t.dtype,
        )
        loss_recon_attr_sum = torch.sum(effective_w * losses_t)
        weighted_dim_numer = torch.sum(effective_w * dims * losses_t)
        weighted_dim_denom = torch.sum(effective_w * dims)
        loss_recon_dim_norm = weighted_dim_numer / (weighted_dim_denom + 1e-12)

        recon_details = {}
        for index, name in enumerate(attr_names):
            loss_per_dim = float(losses_t[index].detach().item())
            out_dim = float(dims[index].detach().item())
            weight = float(effective_w[index].detach().item())
            recon_details[name] = {
                "dim": out_dim,
                "weight": weight,
                "loss_per_dim": loss_per_dim,
                "loss_sum_dims": loss_per_dim * out_dim,
            }

        weight_mode = "ema"
        ema_state = {
            "step": int(ema_details["step"]),
            "warmup_steps": int(ema_details["warmup_steps"]),
            "dynamic_weights": dict(ema_details["weights"]),
            "effective_weights": {
                name: float(effective_w[index].detach().item())
                for index, name in enumerate(attr_names)
            },
        }
    else:
        loss_recon_attr_sum, loss_recon_dim_norm, recon_details = reconstruction_loss_with_breakdown(
            preds,
            yb,
            weights=cfg.view_loss_weights or None,
            loss_type=cfg.loss_type,
        )

    if recon_mode in {"dim_mean", "dim_normalized", "fair"}:
        loss_recon = loss_recon_dim_norm
    else:
        recon_mode = "attr_sum"
        loss_recon = loss_recon_attr_sum

    loss = loss_recon

    if return_aux and return_breakdown:
        recon_breakdown = {
            "selected_mode": recon_mode,
            "selected_loss": float(loss_recon.detach().item()),
            "weighted_attr_sum_loss": float(loss_recon_attr_sum.detach().item()),
            "weighted_dim_normalized_loss": float(loss_recon_dim_norm.detach().item()),
            "per_view": recon_details,
            "weight_mode": weight_mode,
            "ema_state": ema_state,
        }
        return loss, aux, recon_breakdown
    if return_aux:
        return loss, aux
    if return_breakdown:
        recon_breakdown = {
            "selected_mode": recon_mode,
            "selected_loss": float(loss_recon.detach().item()),
            "weighted_attr_sum_loss": float(loss_recon_attr_sum.detach().item()),
            "weighted_dim_normalized_loss": float(loss_recon_dim_norm.detach().item()),
            "per_view": recon_details,
            "weight_mode": weight_mode,
            "ema_state": ema_state,
        }
        return loss, recon_breakdown
    return loss
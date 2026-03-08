from typing import Optional, TYPE_CHECKING

import torch

from inr.training.batches import is_multiview_target, unpack_batch
from inr.training.objectives import compute_multiview_loss

if TYPE_CHECKING:
    from inr.training.loops import TrainingConfig
    from inr.training.losses import MultiAttrEMALoss


def evaluate(
    model,
    loader,
    criterion,
    device,
    n_samples: int,
    cfg: "TrainingConfig",
    hard_topk: bool = True,
    ema_recon_loss: Optional["MultiAttrEMALoss"] = None,
    progress_factory=None,
):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        iterator = loader
        if progress_factory is not None:
            iterator = progress_factory(loader, desc="eval", leave=False)
        for batch in iterator:
            xb, yb = unpack_batch(batch)
            xb = xb.to(device)
            if is_multiview_target(yb):
                yb = {name: tensor.to(device) for name, tensor in yb.items()}
                loss = compute_multiview_loss(
                    model,
                    xb,
                    yb,
                    cfg,
                    hard_topk=hard_topk,
                    ema_recon_loss=ema_recon_loss,
                    update_ema_loss=False,
                )
            else:
                yb = yb.to(device)
                try:
                    pred = model(xb, hard_topk=hard_topk)
                except TypeError:
                    pred = model(xb)
                loss = criterion(pred, yb)
            loss_sum += float(loss.item()) * xb.shape[0]
    return loss_sum / max(n_samples, 1)
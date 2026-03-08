import logging
import time
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from inr.training.pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from inr.training.pretrain.loaders import build_pretrain_loader

if TYPE_CHECKING:
    from inr.training.loops import TrainingConfig

logger = logging.getLogger(__name__)


def run_pretrain(
    model: torch.nn.Module,
    dataset: Dataset,
    cfg: "TrainingConfig",
    device: torch.device,
    progress_factory: Optional[object] = None,
) -> None:
    if not cfg.pretrain.enabled:
        return
    if cfg.pretrain.epochs <= 0:
        return
    if cfg.pretrain.batch_size <= 0:
        raise ValueError("pretrain.batch_size must be positive")
    if not hasattr(model, "num_experts"):
        raise ValueError("Pretrain requires model.num_experts")
    if not hasattr(model, "pretrain_parameters") or not callable(getattr(model, "pretrain_parameters")):
        raise ValueError("Pretrain requires model.pretrain_parameters()")
    if not hasattr(model, "pretrain_forward") or not callable(getattr(model, "pretrain_forward")):
        raise ValueError("Pretrain requires model.pretrain_forward(x)")

    num_experts = int(getattr(model, "num_experts"))
    assignments_cfg = PretrainAssignmentConfig(
        method=cfg.pretrain.assignments_method,
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=str(cfg.pretrain.assignments_cache_path or ""),
        cluster_num_time_samples=int(cfg.pretrain.cluster_num_time_samples),
        spatial_blocks=cfg.pretrain.spatial_blocks,
        time_block_size=int(cfg.pretrain.time_block_size),
    )
    assignments = compute_pretrain_assignments(dataset, num_experts=num_experts, cfg=assignments_cfg)
    pretrain_loader = build_pretrain_loader(dataset, cfg, assignments)

    optimizer = torch.optim.Adam(list(model.pretrain_parameters()), lr=cfg.pretrain.lr, betas=(0.9, 0.999))
    start_time = time.time()
    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        counts = torch.zeros(num_experts, dtype=torch.long)
        iterator = pretrain_loader
        if progress_factory is not None:
            iterator = progress_factory(pretrain_loader, desc=f"pretrain {epoch}/{cfg.pretrain.epochs}", leave=False)
        for xb, expert_id in iterator:
            xb = xb.to(device)
            expert_id = expert_id.to(device)
            logits = model.pretrain_forward(xb)
            if logits.ndim != 2:
                raise ValueError(f"pretrain_forward must return shape (B, C), got {tuple(logits.shape)}")
            num_classes = int(logits.shape[-1])
            target_min = int(expert_id.min().item())
            target_max = int(expert_id.max().item())
            if target_min < 0 or target_max >= num_classes:
                raise ValueError(
                    "Invalid pretrain target labels: "
                    f"min={target_min}, max={target_max}, num_classes={num_classes}. "
                    "Clear pretrain.assignments_cache_path and rerun."
                )
            loss = F.cross_entropy(logits, expert_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == expert_id).sum().item())
            total += int(expert_id.numel())
            counts += torch.bincount(preds.detach().cpu(), minlength=num_experts)

        logger.info(
            "Pretrain epoch %s/%s loss=%.6e acc=%.4f counts=%s time=%.1fs",
            epoch,
            cfg.pretrain.epochs,
            epoch_loss / max(total, 1),
            correct / max(total, 1),
            counts.tolist(),
            time.time() - start_time,
        )
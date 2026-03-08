from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import RandomSampler

from inr.training.loaders import build_loader
from inr.training.pretrain.datasets import NodePretrainDataset

if TYPE_CHECKING:
    from inr.training.loops import TrainingConfig


def build_pretrain_loader(dataset, cfg: "TrainingConfig", assignments: np.ndarray):
    pretrain_ds = NodePretrainDataset(dataset, assignments)
    sampler = None
    if cfg.batches_per_epoch_budget > 0:
        sampler = RandomSampler(
            pretrain_ds,
            replacement=True,
            num_samples=int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size),
        )
    return build_loader(pretrain_ds, cfg.pretrain.batch_size, cfg.num_workers, shuffle=sampler is None, sampler=sampler)
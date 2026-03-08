from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from inr.training.batches import unpack_batch
from inr.training.loaders import build_loader

if TYPE_CHECKING:
    from inr.training.loops import TrainingConfig


def predict_full(model, dataset, cfg: "TrainingConfig", device, suffix: str = "", hard_topk: bool = True, progress_factory=None):
    model.eval()
    loader = build_loader(dataset, cfg.pred_batch_size, cfg.num_workers, shuffle=False)
    n_samples = len(dataset)
    pred_arrays = None
    offset = 0
    with torch.no_grad():
        iterator = loader
        if progress_factory is not None:
            iterator = progress_factory(loader, desc="predict_full", leave=False)
        for batch in iterator:
            xb, _ = unpack_batch(batch)
            xb = xb.to(device)
            batch_size = xb.shape[0]
            try:
                pred = model(xb, hard_topk=hard_topk)
            except TypeError:
                pred = model(xb)
            if isinstance(pred, dict):
                blocks = {}
                for name, tensor in pred.items():
                    block = tensor
                    if hasattr(dataset, "denormalize_attr"):
                        block = dataset.denormalize_attr(name, block)
                    blocks[name] = block.cpu().numpy()
                if pred_arrays is None:
                    base = cfg.save_pred[:-4] if cfg.save_pred.endswith(".npy") else cfg.save_pred
                    pred_arrays = {}
                    for name, block in blocks.items():
                        out_shape = (n_samples,) + tuple(block.shape[1:])
                        save_path = f"{base}_{name}{suffix}.npy"
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        pred_arrays[name] = np.lib.format.open_memmap(save_path, mode="w+", dtype=block.dtype, shape=out_shape)
                for name, block in blocks.items():
                    pred_arrays[name][offset:offset + batch_size] = block
            else:
                block = pred
                if hasattr(dataset, "denormalize_targets"):
                    block = dataset.denormalize_targets(block)
                block = block.cpu().numpy()
                if pred_arrays is None:
                    out_shape = (n_samples,) + tuple(block.shape[1:])
                    save_path = cfg.save_pred if suffix == "" else f"{cfg.save_pred[:-4]}{suffix}.npy"
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    pred_arrays = np.lib.format.open_memmap(save_path, mode="w+", dtype=block.dtype, shape=out_shape)
                pred_arrays[offset:offset + batch_size] = block
            offset += batch_size
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inr.data import NodeDataset
from inr.models.basisExperts import diversity_loss, load_balance_loss, reconstruction_loss
from inr.utils.io import save_checkpoint
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 65536
    pred_batch_size: int = 65536
    num_workers: int = 4
    lr: float = 5e-5
    log_every: int = 4
    save_every: int = 0
    early_stop_patience: int = 0
    seed: int = 42
    save_model: str = "outputs/model.pth"
    save_pred: str = "outputs/pred.npy"
    device: Optional[str] = None
    data_x_path: str = ""
    data_y_path: str = ""
    exp_dir: str = ""
    exp_id: str = ""
    loss_type: str = "mse"
    lam_eq: float = 0.0
    gam_div: float = 0.0
    view_loss_weights: Optional[dict] = field(default_factory=dict)


def _unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb
    raise ValueError(f"Unexpected batch structure: {len(batch)}")


def _is_multiview_target(targets) -> bool:
    return isinstance(targets, dict)


def _compute_multiview_loss(model, xb, yb, cfg: TrainingConfig):
    preds, aux = model(xb, return_aux=True)
    loss_recon = reconstruction_loss(
        preds,
        yb,
        weights=cfg.view_loss_weights or None,
        loss_type=cfg.loss_type,
    )
    loss = loss_recon

    if cfg.lam_eq > 0.0:
        loss_eq = load_balance_loss(aux["probs"], aux["masks"])
        loss = loss + cfg.lam_eq * loss_eq
    else:
        loss_eq = torch.zeros((), device=xb.device)

    if cfg.gam_div > 0.0:
        loss_div = diversity_loss(aux["expert_feats"])
        loss = loss + cfg.gam_div * loss_div
    else:
        loss_div = torch.zeros((), device=xb.device)

    return loss, loss_recon, loss_eq, loss_div


def build_dataloaders(
    dataset: Dataset, cfg: TrainingConfig
) -> Tuple[DataLoader, Optional[NodeDataset]]:
    train_ds = dataset
    train_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        train_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **train_kwargs,
    )
    return train_loader, train_ds


def train_model(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig):
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, train_ds = build_dataloaders(dataset, cfg)
    is_multiview = hasattr(dataset, "view_specs")

    model = model.to(device)
    _print_model_size(model)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in iterator:
            xb, yb = _unpack_batch(batch)
            xb = xb.to(device)
            if _is_multiview_target(yb):
                yb = {name: tensor.to(device) for name, tensor in yb.items()}
                loss, loss_recon, loss_eq, loss_div = _compute_multiview_loss(model, xb, yb, cfg)
            else:
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss_recon = loss
                loss_eq = torch.zeros((), device=loss.device)
                loss_div = torch.zeros((), device=loss.device)
                if hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                elif hasattr(model, "indicator_regularization"):
                    reg = model.indicator_regularization()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= len(train_ds)

        if epoch % cfg.log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            # calculate PSNR on full train set
            if not is_multiview:
                psnr_val = _compute_psnr_streaming_single(model, train_loader, dataset, device)
                print(
                    f"Epoch {epoch}/{cfg.epochs} loss={epoch_loss:.6e} PSNR={psnr_val:.2f} time={elapsed:.1f}s"
                )
            else:
                psnr_vals = _compute_psnr_streaming_multiview(model, train_loader, dataset, device)
                psnr_parts = [f"{name}={psnr_vals[name]:.2f}" for name in psnr_vals.keys()]
                psnr_text = " ".join(psnr_parts)
                print(
                    f"Epoch {epoch}/{cfg.epochs} loss={epoch_loss:.6e} PSNR[{psnr_text}] time={elapsed:.1f}s"
                )
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(model, dataset, cfg.save_model, suffix=f"_epoch{epoch}")
            predict_full(model, dataset, cfg, device, suffix=f"_epoch{epoch}")

    save_checkpoint(model, dataset, cfg.save_model)
    predict_full(model, dataset, cfg, device)


def predict_full(model, dataset: Dataset, cfg: TrainingConfig, device, suffix: str = ""):
    model.eval()
    loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        batch_size=cfg.pred_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    n_samples = len(dataset)
    pred_arrays = None
    offset = 0
    with torch.no_grad():
        for batch in loader:
            xb, _ = _unpack_batch(batch)
            xb = xb.to(device)
            batch_size = xb.shape[0]
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
                        pred_arrays[name] = np.lib.format.open_memmap(
                            save_path, mode="w+", dtype=block.dtype, shape=out_shape
                        )
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
                    pred_arrays = np.lib.format.open_memmap(
                        save_path, mode="w+", dtype=block.dtype, shape=out_shape
                    )
                pred_arrays[offset:offset + batch_size] = block
            offset += batch_size


def _print_model_size(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_mb = total_bytes / (1024 ** 2)
    print(
        f"Model params: {total_params:,} ({trainable_params:,} trainable), size={total_mb:.2f} MB"
    )

def _compute_psnr_streaming_single(model, loader, dataset, device) -> float:
    model.eval()
    total_se = 0.0
    total_count = 0
    gt_min = float("inf")
    gt_max = float("-inf")
    with torch.no_grad():
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="psnr", leave=False)
        for batch in iterator:
            xb, yb = _unpack_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            if hasattr(dataset, "denormalize_targets"):
                pred = dataset.denormalize_targets(pred)
                yb = dataset.denormalize_targets(yb)
            se = torch.sum((pred - yb) ** 2)
            total_se += float(se.item())
            total_count += int(pred.numel())
            gt_min = min(gt_min, float(yb.min().item()))
            gt_max = max(gt_max, float(yb.max().item()))
    data_range = gt_max - gt_min
    if data_range <= 0:
        data_range = 1.0
    if total_count == 0:
        return float("nan")
    mse = total_se / total_count
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def _compute_psnr_streaming_multiview(model, loader, dataset, device) -> dict:
    model.eval()
    total_se = {}
    total_count = {}
    gt_min = {}
    gt_max = {}
    with torch.no_grad():
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="psnr", leave=False)
        for batch in iterator:
            xb, yb = _unpack_batch(batch)
            xb = xb.to(device)
            preds = model(xb)
            for name, pred in preds.items():
                target = yb[name].to(device)
                if hasattr(dataset, "denormalize_attr"):
                    pred = dataset.denormalize_attr(name, pred)
                    target = dataset.denormalize_attr(name, target)
                se = torch.sum((pred - target) ** 2)
                total_se[name] = total_se.get(name, 0.0) + float(se.item())
                total_count[name] = total_count.get(name, 0) + int(pred.numel())
                cur_min = float(target.min().item())
                cur_max = float(target.max().item())
                gt_min[name] = min(gt_min.get(name, cur_min), cur_min)
                gt_max[name] = max(gt_max.get(name, cur_max), cur_max)

    psnr_vals = {}
    for name in total_se.keys():
        data_range = gt_max[name] - gt_min[name]
        if data_range <= 0:
            data_range = 1.0
        mse = total_se[name] / max(1, total_count[name])
        if mse <= 0:
            psnr_vals[name] = float("inf")
        else:
            psnr_vals[name] = 10.0 * math.log10((data_range ** 2) / mse)
    return psnr_vals

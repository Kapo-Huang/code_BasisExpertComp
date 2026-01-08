import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inr.data import NodeDataset
from inr.models.moe_inr_experts_pools import diversity_loss, load_balance_loss, reconstruction_loss
from inr.utils.io import save_checkpoint
from skimage.metrics import peak_signal_noise_ratio as psnr


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 65536
    pred_batch_size: int = 65536
    lr: float = 5e-5
    val_split: float = 0.1
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
) -> Tuple[DataLoader, Optional[DataLoader], Optional[NodeDataset], Optional[NodeDataset]]:
    val_ratio = max(0.0, min(0.5, float(cfg.val_split)))
    n_total = len(dataset)
    n_val = int(round(n_total * val_ratio))
    n_val = max(1 if val_ratio > 0 else 0, n_val)
    n_train = n_total - n_val
    if n_val > 0 and n_train <= 0:
        n_val = max(0, n_val - 1)
        n_train = n_total - n_val

    if n_val > 0:
        g = torch.Generator().manual_seed(int(cfg.seed))
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
        )
    return train_loader, val_loader, train_ds, val_ds


def train_model(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig):
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(dataset, cfg)
    is_multiview = hasattr(dataset, "view_specs")

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
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

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device, len(val_ds), cfg)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} "
                    f"val={val_loss:.6e} time={elapsed:.1f}s (early stop)"
                )
                break

        if epoch % cfg.log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            # calculate PSNR
            if val_loader is not None and not is_multiview:
                with torch.no_grad():
                    preds, gts = [], []
                    for batch in val_loader:
                        xb, yb = _unpack_batch(batch)
                        xb = xb.to(device)
                        pred = model(xb)
                        preds.append(pred.cpu())
                        gts.append(yb)
                    pred_all = torch.cat(preds, dim=0)
                    gt_all = torch.cat(gts, dim=0)
                gt_denorm = dataset.denormalize_targets(gt_all)
                pred_denorm = dataset.denormalize_targets(pred_all)
                data_range = float(torch.max(gt_denorm) - torch.min(gt_denorm))
                data_range = data_range if data_range > 0 else 1.0
                psnr_val = psnr(gt_denorm.numpy(), pred_denorm.numpy(), data_range=data_range)
                print(f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} PSNR={psnr_val:.2f} time={elapsed:.1f}s")
            elif val_loader is not None and is_multiview:
                with torch.no_grad():
                    pred_dict = {}
                    gt_dict = {}
                    for batch in val_loader:
                        xb, yb = _unpack_batch(batch)
                        xb = xb.to(device)
                        preds = model(xb)
                        for name, pred in preds.items():
                            pred_dict.setdefault(name, []).append(pred.cpu())
                        for name, target in yb.items():
                            gt_dict.setdefault(name, []).append(target.cpu())

                psnr_parts = []
                for name in pred_dict.keys():
                    pred_all = torch.cat(pred_dict[name], dim=0)
                    gt_all = torch.cat(gt_dict[name], dim=0)
                    if hasattr(dataset, "denormalize_attr"):
                        pred_all = dataset.denormalize_attr(name, pred_all)
                        gt_all = dataset.denormalize_attr(name, gt_all)
                    data_range = float(torch.max(gt_all) - torch.min(gt_all))
                    data_range = data_range if data_range > 0 else 1.0
                    psnr_val = psnr(gt_all.numpy(), pred_all.numpy(), data_range=data_range)
                    psnr_parts.append(f"{name}={psnr_val:.2f}")
                psnr_text = " ".join(psnr_parts)
                print(
                    f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} "
                    f"val={val_loss:.6e} PSNR[{psnr_text}] time={elapsed:.1f}s"
                )
            elif val_loader is not None:
                print(
                    f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} "
                    f"val={val_loss:.6e} time={elapsed:.1f}s"
                )
            else:
                print(f"Epoch {epoch}/{cfg.epochs} loss={epoch_loss:.6e} time={elapsed:.1f}s")
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(model, dataset, cfg.save_model, suffix=f"_epoch{epoch}")
            predict_full(model, dataset, cfg, device, suffix=f"_epoch{epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, dataset, cfg.save_model)
    predict_full(model, dataset, cfg, device)


def evaluate(model, loader, criterion, device, n_samples: int, cfg: TrainingConfig):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            xb, yb = _unpack_batch(batch)
            xb = xb.to(device)
            if _is_multiview_target(yb):
                yb = {name: tensor.to(device) for name, tensor in yb.items()}
                loss, _, _, _ = _compute_multiview_loss(model, xb, yb, cfg)
            else:
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
            loss_sum += loss.item() * xb.shape[0]
    return loss_sum / n_samples


def predict_full(model, dataset: Dataset, cfg: TrainingConfig, device, suffix: str = ""):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=cfg.pred_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            xb, _ = _unpack_batch(batch)
            xb = xb.to(device)
            pred = model(xb)
            if isinstance(pred, dict):
                preds.append({name: tensor.cpu() for name, tensor in pred.items()})
            else:
                preds.append(pred.cpu())

    if preds and isinstance(preds[0], dict):
        pred_dict = {}
        for name in preds[0].keys():
            pred_all = torch.cat([p[name] for p in preds], dim=0)
            if hasattr(dataset, "denormalize_attr"):
                pred_all = dataset.denormalize_attr(name, pred_all)
            pred_dict[name] = pred_all.numpy()

        base = cfg.save_pred[:-4] if cfg.save_pred.endswith(".npy") else cfg.save_pred
        for name, arr in pred_dict.items():
            save_path = f"{base}_{name}{suffix}.npy"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, arr)
    else:
        pred_all = torch.cat(preds, dim=0)
        if hasattr(dataset, "denormalize_targets"):
            pred_all = dataset.denormalize_targets(pred_all)
        pred_all = pred_all.numpy()

        save_path = cfg.save_pred if suffix == "" else f"{cfg.save_pred[:-4]}{suffix}.npy"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, pred_all)

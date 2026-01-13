import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from inr.datasets.volumetric import (
    MultiTargetVolumetricDataset,
    VolumetricDataset,
    make_multitarget_collate,
    make_singletarget_collate,
)
from inr.models.basisExperts import diversity_loss, load_balance_loss, reconstruction_loss
from inr.utils.io import save_checkpoint
from skimage.metrics import peak_signal_noise_ratio as psnr
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8000
    pred_batch_size: int = 8000
    num_workers: int = 4
    lr: float = 5e-5
    val_split: float = 0.1
    log_every: int = 4
    save_every: int = 0
    early_stop_patience: int = 0
    seed: int = 42
    save_model: str = "outputs/model.pth"
    save_pred: str = "outputs/pred.npy"
    device: Optional[str] = None
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
    try:
        preds, aux = model(xb, return_aux=True)
    except TypeError:
        preds = model(xb)
        aux = {}
    loss_recon = reconstruction_loss(
        preds,
        yb,
        weights=cfg.view_loss_weights or None,
        loss_type=cfg.loss_type,
    )
    loss = loss_recon

    loss_eq = torch.zeros((), device=xb.device)
    if cfg.lam_eq > 0.0 and "probs" in aux and "masks" in aux:
        loss_eq = load_balance_loss(aux["probs"], aux["masks"])
        loss = loss + cfg.lam_eq * loss_eq

    loss_div = torch.zeros((), device=xb.device)
    if cfg.gam_div > 0.0 and "expert_feats" in aux:
        loss_div = diversity_loss(aux["expert_feats"])
        loss = loss + cfg.gam_div * loss_div

    return loss, loss_recon, loss_eq, loss_div


def build_dataloaders(
    dataset: Dataset, cfg: TrainingConfig
) -> Tuple[DataLoader, Optional[DataLoader], Dataset, Optional[Dataset]]:
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

    def _resolve_collate(ds: Dataset):
        base_ds = ds.dataset if isinstance(ds, Subset) else ds
        if isinstance(base_ds, MultiTargetVolumetricDataset):
            return make_multitarget_collate(base_ds)
        if isinstance(base_ds, VolumetricDataset):
            return make_singletarget_collate(base_ds)
        return None

    train_collate = _resolve_collate(train_ds)
    train_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        train_kwargs["prefetch_factor"] = 4
    if train_collate is not None:
        train_kwargs["collate_fn"] = train_collate
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **train_kwargs,
    )
    val_loader = None
    if val_ds is not None:
        val_collate = _resolve_collate(val_ds)
        val_kwargs = {
            "pin_memory": True,
            "num_workers": cfg.num_workers,
            "persistent_workers": cfg.num_workers > 0,
        }
        if cfg.num_workers > 0:
            val_kwargs["prefetch_factor"] = 4
        if val_collate is not None:
            val_kwargs["collate_fn"] = val_collate
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            **val_kwargs,
        )
    return train_loader, val_loader, train_ds, val_ds


def train_model(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig):
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(dataset, cfg)
    print(f"DataLoader build: {time.perf_counter() - t0:.2f}s")
    is_multiview = hasattr(dataset, "view_specs")

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    if cfg.loss_type == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    start_time = time.time()
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        data_time = 0.0
        compute_time = 0.0
        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        prev_time = time.perf_counter()
        for batch in iterator:
            data_time += time.perf_counter() - prev_time
            step_start = time.perf_counter()
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
            compute_time += time.perf_counter() - step_start
            prev_time = time.perf_counter()
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
            if val_loader is not None and not is_multiview:
                with torch.no_grad():
                    preds, gts = [], []
                    val_iter = val_loader
                    if tqdm is not None:
                        val_iter = tqdm(val_loader, desc=f"val {epoch}/{cfg.epochs}", leave=False)
                    for batch in val_iter:
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
                    val_iter = val_loader
                    if tqdm is not None:
                        val_iter = tqdm(val_loader, desc=f"val {epoch}/{cfg.epochs}", leave=False)
                    for batch in val_iter:
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
            else:
                print(f"Epoch {epoch}/{cfg.epochs} loss={epoch_loss:.6e} time={elapsed:.1f}s")
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch} timing: data={data_time:.2f}s compute={compute_time:.2f}s")
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
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="eval", leave=False)
        for batch in iterator:
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
    pred_collate = None
    if isinstance(dataset, MultiTargetVolumetricDataset):
        pred_collate = make_multitarget_collate(dataset)
    elif isinstance(dataset, VolumetricDataset):
        pred_collate = make_singletarget_collate(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.pred_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
        collate_fn=pred_collate,
    )

    preds = []
    with torch.no_grad():
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="predict_full", leave=False)
        for batch in iterator:
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

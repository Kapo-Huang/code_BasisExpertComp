import copy
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, random_split

from inr.training.losses import (
    MultiAttrEMALoss,
    diversity_loss,
    load_balance_loss,
    orth_loss,
    reconstruction_loss_with_breakdown,
)
from inr.pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from inr.utils.io import load_checkpoint, save_checkpoint

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

logger = logging.getLogger(__name__)


def _resolve_tqdm():
    if _tqdm is None:
        return None
    flag = os.getenv("INR_TQDM")
    if flag is not None and flag.strip().lower() in {"0", "false", "no", "off"}:
        return None
    disable = os.getenv("TQDM_DISABLE")
    if disable is not None and disable.strip().lower() in {"1", "true", "yes", "on"}:
        return None
    auto = os.getenv("INR_TQDM_AUTO", "1").strip().lower()
    if auto in {"1", "true", "yes", "on"}:
        try:
            if not sys.stderr.isatty():
                return None
        except Exception:
            return None
    return _tqdm


tqdm = _resolve_tqdm()


@dataclass
class PretrainConfig:
    enabled: bool = False
    epochs: int = 0
    lr: float = 5e-5
    batch_size: int = 8000
    cluster_num_time_samples: int = 16
    cluster_seed: int = 42
    assignments_cache_path: str = ""
    assignments_method: str = "voxel_clustering"
    spatial_blocks: Optional[Tuple[int, int, int]] = None
    time_block_size: int = 0
    mode: str = "router_classification"
    stage1_epochs: int = 0
    stage2_epochs: int = 0
    stage1_gam_div: float = 1e-4
    stage1_gam_orth: float = 1e-4
    stage1_orth_eps: float = 1e-6
    stage1_div_sigma: float = 1.0
    stage2_entropy_weight: float = 0.0
    stage2_lam_eq: float = 0.0
    stage2_temperature: float = 1.0


@dataclass
class TimeStepCurriculumConfig:
    enabled: bool = False
    mode: str = "linear"
    start_timesteps: int = 0
    end_timesteps: int = 0
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    stride_groups: int = 0
    epochs_per_group: int = 0


@dataclass
class MultiAttrEMALossConfig:
    enabled: bool = False
    beta: float = 0.95
    eps: float = 1e-8
    w_min: float = 0.2
    w_max: float = 5.0
    warmup_steps: int = 0


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8000
    pred_batch_size: int = 8000
    num_workers: int = 4
    batches_per_epoch_budget: int = 0
    lr: float = 5e-5
    val_split: float = 0.0
    log_every: int = 4
    log_psnr_every: int = 5
    psnr_sample_ratio: float = 1.0
    save_every: int = 0
    early_stop_patience: int = 0
    seed: int = 42
    save_model: str = "outputs/model.pth"
    save_pred: str = "outputs/pred.npy"
    device: Optional[str] = None
    exp_dir: str = ""
    exp_id: str = ""
    run_timestamp: str = ""
    loss_type: str = "mse"
    lam_eq: float = 0.0
    gam_div: float = 0.0
    gam_orth: float = 0.0
    orth_eps: float = 1e-6
    div_sigma: float = 1.0
    view_loss_weights: Optional[dict] = field(default_factory=dict)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    timestep_curriculum: TimeStepCurriculumConfig = field(default_factory=TimeStepCurriculumConfig)
    lr_decay_rate: float = 0.0
    lr_decay_step: int = 0
    freeze_router_at: float = 0.8
    hard_topk_warmup_epochs: int = 0
    multiview_recon_reduction: str = "attr_sum"
    multiview_ema_loss: MultiAttrEMALossConfig = field(default_factory=MultiAttrEMALossConfig)
    resume_path: Optional[str] = None


class _NodePretrainDataset(Dataset):
    def __init__(self, dataset: Dataset, assignments: np.ndarray):
        if len(dataset) != int(len(assignments)):
            raise ValueError("assignments length must match dataset length")
        self.dataset = dataset
        self.assignments = np.asarray(assignments, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        coords, _ = self.dataset[idx]
        return coords, int(self.assignments[idx])


class _CoordOnlyDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        coords, _ = self.dataset[idx]
        return coords


def _unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb
    raise ValueError(f"Unexpected batch structure: {len(batch)}")


def _is_multiview_target(targets) -> bool:
    return isinstance(targets, dict)


def _compute_multiview_loss(
    model,
    xb,
    yb,
    cfg: TrainingConfig,
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
    loss_eq = torch.zeros((), device=xb.device)
    if cfg.lam_eq > 0.0 and "probs" in aux and "masks" in aux:
        loss_eq = load_balance_loss(aux["probs"], aux["masks"])
        loss = loss + cfg.lam_eq * loss_eq

    loss_div = torch.zeros((), device=xb.device)
    if cfg.gam_div > 0.0 and "expert_feats" in aux:
        loss_div = diversity_loss(aux["expert_feats"], sigma=cfg.div_sigma)
        loss = loss + cfg.gam_div * loss_div

    loss_orth = torch.zeros((), device=xb.device)
    if cfg.gam_orth > 0.0 and "expert_feats" in aux:
        loss_orth = orth_loss(aux["expert_feats"], eps=cfg.orth_eps)
        loss = loss + cfg.gam_orth * loss_orth

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
        return loss, loss_recon, loss_eq, loss_div, loss_orth, aux, recon_breakdown
    if return_aux:
        return loss, loss_recon, loss_eq, loss_div, loss_orth, aux
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
        return loss, loss_recon, loss_eq, loss_div, loss_orth, recon_breakdown
    return loss, loss_recon, loss_eq, loss_div, loss_orth


def _build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, sampler=None) -> DataLoader:
    kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **kwargs,
    )


def _build_pretrain_loader(dataset: Dataset, cfg: TrainingConfig, assignments: np.ndarray) -> DataLoader:
    pretrain_ds = _NodePretrainDataset(dataset, assignments)
    sampler = None
    if cfg.batches_per_epoch_budget > 0:
        sampler = RandomSampler(
            pretrain_ds,
            replacement=True,
            num_samples=int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size),
        )
    return _build_loader(pretrain_ds, cfg.pretrain.batch_size, cfg.num_workers, shuffle=sampler is None, sampler=sampler)


def _build_pretrain_coords_loader(dataset: Dataset, cfg: TrainingConfig) -> DataLoader:
    coord_ds = _CoordOnlyDataset(dataset)
    sampler = None
    if cfg.batches_per_epoch_budget > 0:
        sampler = RandomSampler(
            coord_ds,
            replacement=True,
            num_samples=int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size),
        )
    return _build_loader(coord_ds, cfg.pretrain.batch_size, cfg.num_workers, shuffle=sampler is None, sampler=sampler)


def _maybe_run_pretrain(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig, device: torch.device):
    if not cfg.pretrain.enabled:
        return
    mode = str(cfg.pretrain.mode).strip().lower()
    if mode != "basis_two_stage" and cfg.pretrain.epochs <= 0:
        return
    if cfg.pretrain.batch_size <= 0:
        raise ValueError("pretrain.batch_size must be positive")
    if not hasattr(model, "num_experts"):
        raise ValueError("Pretrain requires model.num_experts")
    num_experts = int(getattr(model, "num_experts"))

    if mode == "basis_two_stage":
        stage1_epochs = int(cfg.pretrain.stage1_epochs)
        stage2_epochs = int(cfg.pretrain.stage2_epochs)
        if stage1_epochs <= 0 and stage2_epochs <= 0:
            raise ValueError("basis_two_stage requires stage1_epochs and/or stage2_epochs > 0")

        coord_loader = _build_pretrain_coords_loader(dataset, cfg)
        if stage1_epochs > 0:
            if not hasattr(model, "pretrain_stage1_parameters") or not callable(getattr(model, "pretrain_stage1_parameters")):
                raise ValueError("basis_two_stage requires model.pretrain_stage1_parameters()")
            if not hasattr(model, "pretrain_stage1_expert_feats") or not callable(getattr(model, "pretrain_stage1_expert_feats")):
                raise ValueError("basis_two_stage requires model.pretrain_stage1_expert_feats(x)")
            optimizer = torch.optim.Adam(list(model.pretrain_stage1_parameters()), lr=cfg.pretrain.lr, betas=(0.9, 0.999))
            start_time = time.time()
            for epoch in range(1, stage1_epochs + 1):
                model.train()
                epoch_loss = 0.0
                steps = 0
                iterator = coord_loader
                if tqdm is not None:
                    iterator = tqdm(coord_loader, desc=f"pretrain_s1 {epoch}/{stage1_epochs}", leave=False)
                for xb in iterator:
                    xb = xb.to(device)
                    expert_feats = model.pretrain_stage1_expert_feats(xb)
                    loss_div = diversity_loss(expert_feats, sigma=cfg.pretrain.stage1_div_sigma)
                    loss_orth = orth_loss(expert_feats, eps=cfg.pretrain.stage1_orth_eps)
                    loss = cfg.pretrain.stage1_gam_div * loss_div + cfg.pretrain.stage1_gam_orth * loss_orth
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    steps += 1
                logger.info(
                    "Pretrain stage1 %s/%s loss=%.6e time=%.1fs",
                    epoch,
                    stage1_epochs,
                    epoch_loss / max(steps, 1),
                    time.time() - start_time,
                )

        if stage2_epochs > 0:
            if not hasattr(model, "pretrain_stage2_parameters") or not callable(getattr(model, "pretrain_stage2_parameters")):
                raise ValueError("basis_two_stage requires model.pretrain_stage2_parameters()")
            if not hasattr(model, "pretrain_stage2_router") or not callable(getattr(model, "pretrain_stage2_router")):
                raise ValueError("basis_two_stage requires model.pretrain_stage2_router(x)")
            if not hasattr(model, "pretrain_teacher_shared_feat") or not callable(getattr(model, "pretrain_teacher_shared_feat")):
                raise ValueError("basis_two_stage requires model.pretrain_teacher_shared_feat(x)")
            optimizer = torch.optim.Adam(list(model.pretrain_stage2_parameters()), lr=cfg.pretrain.lr, betas=(0.9, 0.999))
            start_time = time.time()
            for epoch in range(1, stage2_epochs + 1):
                model.train()
                epoch_loss = 0.0
                steps = 0
                iterator = coord_loader
                if tqdm is not None:
                    iterator = tqdm(coord_loader, desc=f"pretrain_s2 {epoch}/{stage2_epochs}", leave=False)
                for xb in iterator:
                    xb = xb.to(device)
                    shared_teacher = model.pretrain_teacher_shared_feat(xb).detach()
                    probs, masks, expert_feats = model.pretrain_stage2_router(xb, temperature=cfg.pretrain.stage2_temperature)
                    expert_feats = expert_feats.detach()
                    h_router = torch.sum(probs.unsqueeze(-1) * expert_feats.unsqueeze(1), dim=2)
                    batch_size, num_views, feat_dim = h_router.shape
                    shared_router = model.decoder(h_router.reshape(-1, feat_dim)).reshape(batch_size, num_views, -1)
                    teacher = shared_teacher.unsqueeze(1).expand_as(shared_router)
                    loss = F.mse_loss(shared_router, teacher)
                    if cfg.pretrain.stage2_entropy_weight > 0.0:
                        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                        loss = loss - cfg.pretrain.stage2_entropy_weight * entropy
                    if cfg.pretrain.stage2_lam_eq > 0.0:
                        loss_eq = load_balance_loss(probs, masks)
                        loss = loss + cfg.pretrain.stage2_lam_eq * loss_eq
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    steps += 1
                logger.info(
                    "Pretrain stage2 %s/%s loss=%.6e time=%.1fs",
                    epoch,
                    stage2_epochs,
                    epoch_loss / max(steps, 1),
                    time.time() - start_time,
                )
        return

    assignments_cfg = PretrainAssignmentConfig(
        method=cfg.pretrain.assignments_method,
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=str(cfg.pretrain.assignments_cache_path or ""),
        cluster_num_time_samples=int(cfg.pretrain.cluster_num_time_samples),
        spatial_blocks=cfg.pretrain.spatial_blocks,
        time_block_size=int(cfg.pretrain.time_block_size),
    )
    assignments = compute_pretrain_assignments(dataset, num_experts=num_experts, cfg=assignments_cfg)
    pretrain_loader = _build_pretrain_loader(dataset, cfg, assignments)

    if not hasattr(model, "pretrain_parameters") or not callable(getattr(model, "pretrain_parameters")):
        raise ValueError("Pretrain requires model.pretrain_parameters()")
    if not hasattr(model, "pretrain_forward") or not callable(getattr(model, "pretrain_forward")):
        raise ValueError("Pretrain requires model.pretrain_forward(x)")

    optimizer = torch.optim.Adam(list(model.pretrain_parameters()), lr=cfg.pretrain.lr, betas=(0.9, 0.999))
    start_time = time.time()
    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        counts = torch.zeros(num_experts, dtype=torch.long)
        iterator = pretrain_loader
        if tqdm is not None:
            iterator = tqdm(pretrain_loader, desc=f"pretrain {epoch}/{cfg.pretrain.epochs}", leave=False)
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


def build_dataloaders(dataset: Dataset, cfg: TrainingConfig):
    train_ds = dataset
    val_ds = None
    if 0.0 < float(cfg.val_split) < 1.0 and len(dataset) > 1:
        train_size = max(1, int(round(len(dataset) * (1.0 - float(cfg.val_split)))))
        val_size = len(dataset) - train_size
        if val_size > 0:
            generator = torch.Generator().manual_seed(int(cfg.seed))
            train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_sampler = None
    if cfg.batches_per_epoch_budget > 0:
        train_sampler = RandomSampler(
            train_ds,
            replacement=True,
            num_samples=int(cfg.batches_per_epoch_budget) * int(cfg.batch_size),
        )
    train_loader = _build_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = None
    if val_ds is not None:
        val_loader = _build_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False)
    return train_loader, val_loader, train_ds, val_ds


def _freeze_router_modules(model: torch.nn.Module) -> int:
    frozen = 0
    for attr in ("gating", "view_embedding"):
        if not hasattr(model, attr):
            continue
        module = getattr(model, attr)
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen += 1
    return frozen


def train_model(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig):
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(dataset, cfg)
    logger.info("DataLoader build: %.2fs", time.perf_counter() - t0)
    is_multiview = hasattr(dataset, "view_specs")

    model = model.to(device)
    _maybe_run_pretrain(model, train_ds, cfg, device)
    current_lr = float(cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
    criterion = torch.nn.L1Loss() if cfg.loss_type == "l1" else torch.nn.MSELoss()

    start_epoch = 1
    if cfg.resume_path:
        checkpoint = load_checkpoint(cfg.resume_path, model)
        resume_epoch = int(checkpoint.get("epoch", 0) or 0)
        optim_state = checkpoint.get("optimizer_state")
        if optim_state is not None:
            try:
                optimizer.load_state_dict(optim_state)
            except Exception as exc:
                logger.warning("Failed to load optimizer state from %s: %s", cfg.resume_path, exc)
        if resume_epoch > 0:
            start_epoch = resume_epoch + 1
        if start_epoch > cfg.epochs:
            logger.info("Resume epoch %s >= total epochs %s; skipping training.", resume_epoch, cfg.epochs)
            return

    multiview_ema_recon_loss: Optional[MultiAttrEMALoss] = None
    if is_multiview and cfg.multiview_ema_loss.enabled:
        attr_names = list(dataset.view_specs().keys())
        if not attr_names:
            raise ValueError("multiview_ema_loss requires at least one attribute in dataset.view_specs().")
        multiview_ema_recon_loss = MultiAttrEMALoss(
            attr_names=attr_names,
            beta=cfg.multiview_ema_loss.beta,
            eps=cfg.multiview_ema_loss.eps,
            w_min=cfg.multiview_ema_loss.w_min,
            w_max=cfg.multiview_ema_loss.w_max,
            warmup_steps=cfg.multiview_ema_loss.warmup_steps,
            loss_type=cfg.loss_type,
        ).to(device)
        logger.info(
            "Enable multiview EMA loss balancing: attrs=%s beta=%.4f warmup_steps=%d w_min=%.3f w_max=%.3f",
            attr_names,
            cfg.multiview_ema_loss.beta,
            cfg.multiview_ema_loss.warmup_steps,
            cfg.multiview_ema_loss.w_min,
            cfg.multiview_ema_loss.w_max,
        )
        if cfg.view_loss_weights:
            logger.info("view_loss_weights detected: static weights are multiplied with EMA weights and renormalized.")

    best_val = float("inf")
    best_state = None
    no_improve = 0
    start_time = time.time()
    freeze_epoch = None
    router_frozen = False
    freeze_at = float(cfg.freeze_router_at)
    if freeze_at > 0:
        freeze_epoch = max(1, int(math.ceil(cfg.epochs * freeze_at))) if freeze_at < 1.0 else int(freeze_at)

    for epoch in range(start_epoch, cfg.epochs + 1):
        hard_topk = not (cfg.hard_topk_warmup_epochs > 0 and epoch <= int(cfg.hard_topk_warmup_epochs))
        if not router_frozen and freeze_epoch is not None and epoch >= freeze_epoch:
            frozen_params = _freeze_router_modules(model)
            if frozen_params > 0:
                logger.info("Freeze router modules at epoch %s (params frozen: %s)", epoch, frozen_params)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=current_lr, betas=(0.9, 0.999))
                router_frozen = True

        if cfg.lr_decay_step > 0 and cfg.lr_decay_rate > 0.0:
            steps = (epoch - 1) // int(cfg.lr_decay_step)
            current_lr = float(cfg.lr) * (float(cfg.lr_decay_rate) ** steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        model.train()
        epoch_loss = 0.0
        steps_seen = 0
        expert_select_counts = None
        multiview_recon_attr_sum_acc = 0.0
        multiview_recon_dim_norm_acc = 0.0
        multiview_recon_selected_acc = 0.0
        multiview_per_view_acc: Dict[str, Dict[str, float]] = {}
        multiview_selected_mode = "attr_sum"
        multiview_weight_mode = "static"
        multiview_last_ema_state = None

        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in iterator:
            xb, yb = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            if _is_multiview_target(yb):
                yb = {name: tensor.to(device, non_blocking=True) for name, tensor in yb.items()}
                loss, _, _, _, _, aux, recon_breakdown = _compute_multiview_loss(
                    model,
                    xb,
                    yb,
                    cfg,
                    return_aux=True,
                    return_breakdown=True,
                    hard_topk=hard_topk,
                    ema_recon_loss=multiview_ema_recon_loss,
                    update_ema_loss=True,
                )
                multiview_recon_attr_sum_acc += float(recon_breakdown["weighted_attr_sum_loss"])
                multiview_recon_dim_norm_acc += float(recon_breakdown["weighted_dim_normalized_loss"])
                multiview_recon_selected_acc += float(recon_breakdown["selected_loss"])
                multiview_selected_mode = str(recon_breakdown["selected_mode"])
                multiview_weight_mode = str(recon_breakdown.get("weight_mode", "static"))
                multiview_last_ema_state = recon_breakdown.get("ema_state")
                for name, stats in recon_breakdown["per_view"].items():
                    if name not in multiview_per_view_acc:
                        multiview_per_view_acc[name] = {
                            "dim": float(stats["dim"]),
                            "weight_sum": 0.0,
                            "loss_sum_dims": 0.0,
                            "loss_per_dim": 0.0,
                        }
                    multiview_per_view_acc[name]["weight_sum"] += float(stats["weight"])
                    multiview_per_view_acc[name]["loss_sum_dims"] += float(stats["loss_sum_dims"])
                    multiview_per_view_acc[name]["loss_per_dim"] += float(stats["loss_per_dim"])
            else:
                yb = yb.to(device, non_blocking=True)
                try:
                    pred, aux = model(xb, return_aux=True, hard_topk=hard_topk)
                except TypeError:
                    try:
                        pred, aux = model(xb, return_aux=True)
                    except TypeError:
                        pred = model(xb)
                        aux = {}
                loss = criterion(pred, yb)
                if cfg.lam_eq > 0.0 and "probs" in aux and "masks" in aux:
                    loss = loss + cfg.lam_eq * load_balance_loss(aux["probs"], aux["masks"])
                if cfg.gam_div > 0.0 and "expert_feats" in aux:
                    loss = loss + cfg.gam_div * diversity_loss(aux["expert_feats"], sigma=cfg.div_sigma)
                if cfg.gam_orth > 0.0 and "expert_feats" in aux:
                    loss = loss + cfg.gam_orth * orth_loss(aux["expert_feats"], eps=cfg.orth_eps)
                if hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                elif hasattr(model, "indicator_regularization"):
                    reg = model.indicator_regularization()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps_seen += 1

            if aux and "probs" in aux:
                probs = aux["probs"].detach()
                reduce_dims = tuple(range(probs.dim() - 1))
                counts = probs.float().sum(dim=reduce_dims).cpu()
                if expert_select_counts is None:
                    expert_select_counts = torch.zeros_like(counts)
                expert_select_counts += counts

        epoch_loss /= max(steps_seen, 1)

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device,
                len(val_ds),
                cfg,
                hard_topk=hard_topk,
                ema_recon_loss=multiview_ema_recon_loss,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                logger.info("Epoch %s/%s train=%.6e val=%.6e time=%.1fs (early stop)", epoch, cfg.epochs, epoch_loss, val_loss, time.time() - start_time)
                break

        if epoch % cfg.log_every == 0 or epoch == start_epoch:
            if val_loss is None:
                logger.info("Epoch %s/%s train=%.6e time=%.1fs", epoch, cfg.epochs, epoch_loss, time.time() - start_time)
            else:
                logger.info("Epoch %s/%s train=%.6e val=%.6e time=%.1fs", epoch, cfg.epochs, epoch_loss, val_loss, time.time() - start_time)
            if steps_seen > 0 and multiview_per_view_acc:
                metric_tag = "mse" if str(cfg.loss_type).strip().lower() == "mse" else "l1"
                logger.info(
                    "Recon loss (%s): attr_sum=%.6e dim_normalized=%.6e selected[%s]=%.6e",
                    multiview_weight_mode,
                    multiview_recon_attr_sum_acc / float(steps_seen),
                    multiview_recon_dim_norm_acc / float(steps_seen),
                    multiview_selected_mode,
                    multiview_recon_selected_acc / float(steps_seen),
                )
                for name in sorted(multiview_per_view_acc.keys()):
                    stats = multiview_per_view_acc[name]
                    logger.info(
                        "Recon breakdown [%s]: dim=%d weight(avg)=%.3f %s(sum_dims)=%.6e %s(per_dim_avg)=%.6e",
                        name,
                        int(round(stats["dim"])),
                        stats["weight_sum"] / float(steps_seen),
                        metric_tag,
                        stats["loss_sum_dims"] / float(steps_seen),
                        metric_tag,
                        stats["loss_per_dim"] / float(steps_seen),
                    )
                if multiview_weight_mode == "ema" and multiview_last_ema_state:
                    logger.info(
                        "EMA balance state: step=%d warmup_steps=%d effective_weights=%s",
                        int(multiview_last_ema_state.get("step", 0)),
                        int(multiview_last_ema_state.get("warmup_steps", 0)),
                        multiview_last_ema_state.get("effective_weights", {}),
                    )
            if expert_select_counts is not None:
                total_counts = expert_select_counts.sum().item()
                counts_text = " ".join(
                    f"E{i}={count:.2f} ({count / total_counts:.2%})"
                    for i, count in enumerate(expert_select_counts.tolist())
                )
                logger.info("Expert utilization rate: %s", counts_text)

        if cfg.log_psnr_every > 0 and epoch % cfg.log_psnr_every == 0:
            psnr_ds = dataset
            if 0.0 < float(cfg.psnr_sample_ratio) < 1.0:
                n_total = len(dataset)
                n_sample = max(1, int(round(n_total * float(cfg.psnr_sample_ratio))))
                generator = torch.Generator().manual_seed(int(cfg.seed))
                indices = torch.randperm(n_total, generator=generator)[:n_sample].tolist()
                psnr_ds = Subset(dataset, indices)
            psnr_loader = _build_loader(psnr_ds, cfg.pred_batch_size, cfg.num_workers, shuffle=False)
            if is_multiview:
                psnr_vals = _compute_psnr_streaming_multiview(model, psnr_loader, dataset, device, hard_topk=hard_topk)
                psnr_text = " ".join(f"{name}={value:.2f}" for name, value in psnr_vals.items())
                logger.info("PSNR epoch %s/%s: %s time=%.1fs", epoch, cfg.epochs, psnr_text, time.time() - start_time)
            else:
                psnr_val = _compute_psnr_streaming_single(model, psnr_loader, dataset, device, hard_topk=hard_topk)
                logger.info("PSNR epoch %s/%s: %.2f time=%.1fs", epoch, cfg.epochs, psnr_val, time.time() - start_time)

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(model, dataset, cfg.save_model, suffix=f"_epoch{epoch}", run_timestamp=cfg.run_timestamp, epoch=epoch, optimizer=optimizer)
            predict_full(model, dataset, cfg, device, suffix=f"_epoch{epoch}", hard_topk=hard_topk)

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, dataset, cfg.save_model, run_timestamp=cfg.run_timestamp, epoch=cfg.epochs, optimizer=optimizer)
    predict_full(model, dataset, cfg, device)


def evaluate(
    model,
    loader,
    criterion,
    device,
    n_samples: int,
    cfg: TrainingConfig,
    hard_topk: bool = True,
    ema_recon_loss: Optional[MultiAttrEMALoss] = None,
):
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
                loss, _, _, _, _ = _compute_multiview_loss(
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


def predict_full(model, dataset: Dataset, cfg: TrainingConfig, device, suffix: str = "", hard_topk: bool = True):
    model.eval()
    loader = _build_loader(dataset, cfg.pred_batch_size, cfg.num_workers, shuffle=False)
    n_samples = len(dataset)
    pred_arrays = None
    offset = 0
    with torch.no_grad():
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc="predict_full", leave=False)
        for batch in iterator:
            xb, _ = _unpack_batch(batch)
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


def _compute_psnr_streaming_single(model, loader, dataset, device, hard_topk: bool = True) -> float:
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
            try:
                pred = model(xb, hard_topk=hard_topk)
            except TypeError:
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


def _compute_psnr_streaming_multiview(model, loader, dataset, device, hard_topk: bool = True) -> dict:
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
            try:
                preds = model(xb, hard_topk=hard_topk)
            except TypeError:
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
        psnr_vals[name] = float("inf") if mse <= 0 else 10.0 * math.log10((data_range ** 2) / mse)
    return psnr_vals

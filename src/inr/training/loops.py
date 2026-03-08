import copy
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, RandomSampler, Subset, random_split

from inr.training.batches import is_multiview_target, unpack_batch
from inr.training.evaluation import evaluate
from inr.training.loaders import build_loader
from inr.training.metrics import compute_psnr_streaming_multiview, compute_psnr_streaming_single
from inr.training.losses import (
    MultiAttrEMALoss,
)
from inr.training.objectives import compute_multiview_loss
from inr.training.prediction import predict_full
from inr.training.pretrain import PretrainConfig, run_pretrain
from inr.utils.timing import EpochTimeBreakdown, log_epoch_time_breakdown, timing_elapsed, timing_start
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
    log_time_breakdown: bool = True
    time_breakdown_cuda_sync: bool = False


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
    train_loader = build_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = None
    if val_ds is not None:
        val_loader = build_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False)
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
    timing_enabled = bool(getattr(cfg, "log_time_breakdown", True))
    sync_timing = bool(getattr(cfg, "time_breakdown_cuda_sync", False))
    t0 = time.perf_counter()
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(dataset, cfg)
    logger.info("DataLoader build: %.2fs", time.perf_counter() - t0)
    is_multiview = hasattr(dataset, "view_specs")

    model = model.to(device)
    run_pretrain(model, train_ds, cfg, device, progress_factory=tqdm)
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
        epoch_timer = EpochTimeBreakdown()
        epoch_started_at = time.perf_counter()
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
        iterator = iter(iterator)
        batch_fetch_started_at = time.perf_counter()
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if timing_enabled:
                epoch_timer.data_loading += time.perf_counter() - batch_fetch_started_at

            stage_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            xb, yb = unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            if is_multiview_target(yb):
                yb = {name: tensor.to(device, non_blocking=True) for name, tensor in yb.items()}
                if timing_enabled:
                    epoch_timer.device_transfer += timing_elapsed(stage_started_at, device, sync_timing)
                    stage_started_at = timing_start(device, sync_timing)
                loss, aux, recon_breakdown = compute_multiview_loss(
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
                if timing_enabled:
                    epoch_timer.forward_loss += timing_elapsed(stage_started_at, device, sync_timing)
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
                if timing_enabled:
                    epoch_timer.device_transfer += timing_elapsed(stage_started_at, device, sync_timing)
                    stage_started_at = timing_start(device, sync_timing)
                try:
                    pred, aux = model(xb, return_aux=True, hard_topk=hard_topk)
                except TypeError:
                    try:
                        pred, aux = model(xb, return_aux=True)
                    except TypeError:
                        pred = model(xb)
                        aux = {}
                loss = criterion(pred, yb)
                if hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                elif hasattr(model, "indicator_regularization"):
                    reg = model.indicator_regularization()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                if timing_enabled:
                    epoch_timer.forward_loss += timing_elapsed(stage_started_at, device, sync_timing)

            stage_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            optimizer.zero_grad()
            if timing_enabled:
                epoch_timer.optimizer += timing_elapsed(stage_started_at, device, sync_timing)

            stage_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            loss.backward()
            if timing_enabled:
                epoch_timer.backward += timing_elapsed(stage_started_at, device, sync_timing)

            stage_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            optimizer.step()
            if timing_enabled:
                epoch_timer.optimizer += timing_elapsed(stage_started_at, device, sync_timing)
            epoch_loss += float(loss.item())
            steps_seen += 1

            stage_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            if aux and "probs" in aux:
                probs = aux["probs"].detach()
                reduce_dims = tuple(range(probs.dim() - 1))
                counts = probs.float().sum(dim=reduce_dims).cpu()
                if expert_select_counts is None:
                    expert_select_counts = torch.zeros_like(counts)
                expert_select_counts += counts
            if timing_enabled:
                epoch_timer.expert_stats += timing_elapsed(stage_started_at, device, sync_timing)
            batch_fetch_started_at = time.perf_counter()

        epoch_loss /= max(steps_seen, 1)

        val_loss = None
        if val_loader is not None:
            val_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device,
                len(val_ds),
                cfg,
                hard_topk=hard_topk,
                ema_recon_loss=multiview_ema_recon_loss,
                progress_factory=tqdm,
            )
            if timing_enabled:
                epoch_timer.validation += timing_elapsed(val_started_at, device, sync_timing)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                if timing_enabled:
                    log_started_at = time.perf_counter()
                logger.info("Epoch %s/%s train=%.6e val=%.6e time=%.1fs (early stop)", epoch, cfg.epochs, epoch_loss, val_loss, time.time() - start_time)
                if timing_enabled:
                    epoch_timer.logging += time.perf_counter() - log_started_at
                    log_epoch_time_breakdown(epoch, cfg.epochs, time.perf_counter() - epoch_started_at, epoch_timer)
                break

        if epoch % cfg.log_every == 0 or epoch == start_epoch:
            if timing_enabled:
                log_started_at = time.perf_counter()
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
            if timing_enabled:
                epoch_timer.logging += time.perf_counter() - log_started_at

        if cfg.log_psnr_every > 0 and epoch % cfg.log_psnr_every == 0:
            psnr_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            psnr_ds = dataset
            if 0.0 < float(cfg.psnr_sample_ratio) < 1.0:
                n_total = len(dataset)
                n_sample = max(1, int(round(n_total * float(cfg.psnr_sample_ratio))))
                generator = torch.Generator().manual_seed(int(cfg.seed))
                indices = torch.randperm(n_total, generator=generator)[:n_sample].tolist()
                psnr_ds = Subset(dataset, indices)
            psnr_loader = build_loader(psnr_ds, cfg.pred_batch_size, cfg.num_workers, shuffle=False)
            if is_multiview:
                psnr_vals = compute_psnr_streaming_multiview(model, psnr_loader, dataset, device, hard_topk=hard_topk, progress_factory=tqdm)
                psnr_text = " ".join(f"{name}={value:.2f}" for name, value in psnr_vals.items())
                logger.info("PSNR epoch %s/%s: %s time=%.1fs", epoch, cfg.epochs, psnr_text, time.time() - start_time)
            else:
                psnr_val = compute_psnr_streaming_single(model, psnr_loader, dataset, device, hard_topk=hard_topk, progress_factory=tqdm)
                logger.info("PSNR epoch %s/%s: %.2f time=%.1fs", epoch, cfg.epochs, psnr_val, time.time() - start_time)
            if timing_enabled:
                epoch_timer.psnr += timing_elapsed(psnr_started_at, device, sync_timing)

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            save_checkpoint(model, dataset, cfg.save_model, suffix=f"_epoch{epoch}", run_timestamp=cfg.run_timestamp, epoch=epoch, optimizer=optimizer)
            predict_full(model, dataset, cfg, device, suffix=f"_epoch{epoch}", hard_topk=hard_topk, progress_factory=tqdm)
            if timing_enabled:
                epoch_timer.checkpoint += timing_elapsed(save_started_at, device, sync_timing)

        if timing_enabled:
            log_epoch_time_breakdown(epoch, cfg.epochs, time.perf_counter() - epoch_started_at, epoch_timer)

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, dataset, cfg.save_model, run_timestamp=cfg.run_timestamp, epoch=cfg.epochs, optimizer=optimizer)
    predict_full(model, dataset, cfg, device, progress_factory=tqdm)

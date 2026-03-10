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
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from inr.datasets.volumetric import (
    MultiTargetVolumetricDataset,
    VolumetricDataset,
    make_multitarget_collate,
    make_singletarget_collate,
)
from inr.training.losses import (
    reconstruction_loss,
    reconstruction_loss_with_breakdown,
)
from inr.training.gradient_diagnostics import diagnose_multitask_gradients
from inr.training.gradient_balancer import GradNormBalancer, _apply_multitask_gradient
from inr.pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from inr.utils.io import save_checkpoint
from skimage.metrics import peak_signal_noise_ratio as psnr
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
class GradientBalancerConfig:
    enabled: bool = False
    method: str = "pcgrad"
    cagrad_c: float = 0.4
    solver_max_iter: int = 50
    solver_lr: float = 0.25
    gradnorm_alpha: float = 0.5
    gradnorm_lr: float = 1e-3
    gradnorm_every_n_steps: int = 100


@dataclass
class GradientDiagConfig:
    enabled: bool = False
    every_n_steps: int = 200
    max_layers_to_log: int = 10


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8000
    pred_batch_size: int = 8000
    num_workers: int = 4
    batches_per_epoch_budget: int = 0
    lr: float = 5e-5
    val_split: float = 0.1
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
    gradient_balancer: GradientBalancerConfig = field(default_factory=GradientBalancerConfig)
    gradient_diag: GradientDiagConfig = field(default_factory=GradientDiagConfig)


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
    return_task_losses: bool = False,
    hard_topk: bool = True,
):
    try:
        preds, aux = model(xb, return_aux=True, hard_topk=hard_topk)
    except TypeError:
        try:
            preds, aux = model(xb, return_aux=True)
        except TypeError:
            preds = model(xb)
            aux = {}
    weighted_attr_sum_loss, weighted_dim_normalized_loss, recon_breakdown, task_losses = reconstruction_loss_with_breakdown(
        preds,
        yb,
        weights=cfg.view_loss_weights or None,
        loss_type=cfg.loss_type,
    )
    loss_recon = weighted_attr_sum_loss
    loss = loss_recon

    recon_breakdown["selected_mode"] = "attr_sum"
    recon_breakdown["selected_loss"] = float(weighted_attr_sum_loss.detach().item())
    recon_breakdown["weighted_attr_sum_loss"] = float(weighted_attr_sum_loss.detach().item())
    recon_breakdown["weighted_dim_normalized_loss"] = float(weighted_dim_normalized_loss.detach().item())

    if return_aux and return_breakdown and return_task_losses:
        return loss, loss_recon, aux, recon_breakdown, task_losses
    if return_aux and return_breakdown:
        return loss, loss_recon, aux, recon_breakdown
    if return_aux and return_task_losses:
        return loss, loss_recon, aux, task_losses
    if return_aux:
        return loss, loss_recon, aux
    if return_breakdown and return_task_losses:
        return loss, loss_recon, recon_breakdown, task_losses
    if return_breakdown:
        return loss, loss_recon, recon_breakdown
    if return_task_losses:
        return loss, loss_recon, task_losses
    return loss, loss_recon


def _resolve_collate(ds: Dataset):
    base_ds = ds.dataset if isinstance(ds, Subset) else ds
    if isinstance(base_ds, MultiTargetVolumetricDataset):
        return make_multitarget_collate(base_ds)
    if isinstance(base_ds, VolumetricDataset):
        return make_singletarget_collate(base_ds)
    return None


def _resolve_base_dataset(ds: Dataset):
    return ds.dataset if isinstance(ds, Subset) else ds


class TimeStepRandomSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        volume_shape,
        total_samples_budget: int,
        generator=None,
        active_timesteps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.volume_shape = volume_shape
        self.total_samples_budget = int(total_samples_budget)
        self.samples_per_timestep = 0
        self.total_samples = 0
        self._active_timesteps = None
        self._active_indices = None
        self.set_active_timesteps(active_timesteps)
        self.generator = generator

        self._per_t_indices = None
        if isinstance(dataset, Subset):
            indices = np.asarray(dataset.indices, dtype=np.int64)
            V = int(self.volume_shape.X) * int(self.volume_shape.Y) * int(self.volume_shape.Z)
            t = indices // V
            self._per_t_indices = []
            for ti in range(int(self.volume_shape.T)):
                mask = t == ti
                if not np.any(mask):
                    raise ValueError(f"Subset has no samples for timestep t={ti}")
                self._per_t_indices.append(indices[mask])

    @property
    def active_timesteps(self) -> Optional[int]:
        return self._active_timesteps

    def _set_active_count(self, count: int):
        count = int(count)
        if count <= 0 or self.total_samples_budget <= 0:
            self.samples_per_timestep = 0
            self.total_samples = 0
            return
        self.samples_per_timestep = self.total_samples_budget // count
        self.total_samples = self.samples_per_timestep * count

    def set_active_timesteps(self, active_timesteps: Optional[int]):
        self._active_indices = None
        total_timesteps = int(self.volume_shape.T)
        if active_timesteps is None:
            self._active_timesteps = None
            self._set_active_count(total_timesteps)
            return
        active_timesteps = int(active_timesteps)
        if active_timesteps <= 0:
            raise ValueError("active_timesteps must be positive")
        if active_timesteps > total_timesteps:
            active_timesteps = total_timesteps
        self._active_timesteps = active_timesteps
        self._set_active_count(self._active_timesteps)

    def set_active_indices(self, indices: Optional[list]):
        self._active_timesteps = None
        if indices is None:
            self._active_indices = None
            self._set_active_count(int(self.volume_shape.T))
            return
        if not indices:
            raise ValueError("active_indices must be a non-empty list")
        total_timesteps = int(self.volume_shape.T)
        self._active_indices = [int(t) for t in indices if 0 <= int(t) < total_timesteps]
        if not self._active_indices:
            raise ValueError("active_indices must contain at least one valid timestep")
        self._set_active_count(len(self._active_indices))

    def __iter__(self):
        V = int(self.volume_shape.X) * int(self.volume_shape.Y) * int(self.volume_shape.Z)
        T = int(self.volume_shape.T)
        active_indices = None
        if self._active_indices is not None:
            active_indices = [t for t in self._active_indices if 0 <= t < T]
        elif self._active_timesteps is not None:
            active_T = min(self._active_timesteps, T)
            active_indices = list(range(active_T))
        else:
            active_indices = list(range(T))
        if not active_indices:
            return

        active_tensor = torch.tensor(active_indices, dtype=torch.long)
        total = int(self.total_samples)
        chunk_size = 1_000_000
        remaining = total
        while remaining > 0:
            cur = min(chunk_size, remaining)
            t_choice = torch.randint(0, len(active_indices), (cur,), generator=self.generator)
            t_vals = active_tensor[t_choice]
            if self._per_t_indices is None:
                offsets = torch.randint(0, V, (cur,), generator=self.generator)
                indices = t_vals * V + offsets
                for idx in indices.tolist():
                    yield int(idx)
            else:
                t_vals_np = t_vals.cpu().numpy()
                out = np.empty(cur, dtype=np.int64)
                for t in np.unique(t_vals_np):
                    pos = np.where(t_vals_np == t)[0]
                    pool = self._per_t_indices[int(t)]
                    picks = torch.randint(0, len(pool), (len(pos),), generator=self.generator).cpu().numpy()
                    out[pos] = pool[picks]
                for idx in out.tolist():
                    yield int(idx)
            remaining -= cur

    def __len__(self) -> int:
        T = int(self.volume_shape.T)
        if self._active_indices is not None:
            active = len([t for t in self._active_indices if 0 <= t < T])
        elif self._active_timesteps is not None:
            active = min(int(self._active_timesteps), T)
        else:
            active = T
        return self.samples_per_timestep * active



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


def _resolve_curriculum_timesteps(cfg: TrainingConfig, epoch: int, total_timesteps: int) -> int:
    cur = cfg.timestep_curriculum
    if not cur.enabled:
        return total_timesteps
    if cur.start_timesteps <= 0 or cur.end_timesteps <= 0:
        raise ValueError("timestep_curriculum requires positive start_timesteps and end_timesteps.")
    start = min(int(cur.start_timesteps), total_timesteps)
    end = min(int(cur.end_timesteps), total_timesteps)
    if end < start:
        raise ValueError("timestep_curriculum end_timesteps must be >= start_timesteps.")
    warmup = max(0, int(cur.warmup_epochs))
    ramp = max(0, int(cur.ramp_epochs))
    if epoch <= warmup:
        return start
    if ramp <= 0:
        return end
    progress = min(1.0, (epoch - warmup) / float(ramp))
    active = int(round(start + (end - start) * progress))
    return max(start, min(end, active))


def _resolve_stride_groups(cfg: TrainingConfig, epoch: int, total_timesteps: int) -> Tuple[list, int, int]:
    cur = cfg.timestep_curriculum
    if not cur.enabled:
        return list(range(total_timesteps)), total_timesteps, 0
    G = int(cur.stride_groups)
    if G <= 0:
        raise ValueError("stride_groups must be positive for stride curriculum.")
    group_span = int(cur.epochs_per_group)
    if group_span <= 0:
        raise ValueError("epochs_per_group must be positive for stride curriculum.")
    active_groups = min(G, 1 + (epoch - 1) // group_span)
    active_indices = [t for t in range(total_timesteps) if (t % G) < active_groups]
    return active_indices, active_groups, G

def _build_pretrain_loader(dataset: Dataset, cfg: TrainingConfig, assignments: np.ndarray) -> DataLoader:
    base_ds = _resolve_base_dataset(dataset)
    if isinstance(base_ds, MultiTargetVolumetricDataset):
        collate_fn = make_multitarget_collate(
            base_ds,
            assignments=assignments,
            return_expert_id=True,
            read_targets=False,
        )
    elif isinstance(base_ds, VolumetricDataset):
        collate_fn = make_singletarget_collate(
            base_ds,
            assignments=assignments,
            return_expert_id=True,
            read_targets=False,
        )
    else:
        raise TypeError(f"Unsupported dataset type: {type(base_ds)}")

    sampler = None
    shuffle = True
    if cfg.batches_per_epoch_budget > 0:
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        total_samples_budget = int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size)
        sampler = TimeStepRandomSampler(dataset, base_ds.volume_shape, total_samples_budget)
        shuffle = False

    pretrain_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        pretrain_kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        **pretrain_kwargs,
    )


def _maybe_run_pretrain(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig, device: torch.device):
    if not cfg.pretrain.enabled:
        return
    if cfg.pretrain.epochs <= 0:
        return
    if cfg.pretrain.batch_size <= 0:
        raise ValueError("pretrain.batch_size must be positive")
    if str(cfg.pretrain.assignments_method).strip().lower() in {"voxel_clustering", "clustering", "kmeans"}:
        if cfg.pretrain.cluster_num_time_samples <= 0:
            raise ValueError("pretrain.cluster_num_time_samples must be positive")
    if not hasattr(model, "num_experts"):
        raise ValueError("Pretrain requires model.num_experts")
    num_experts = getattr(model, "num_experts", None)
    if num_experts is None:
        raise ValueError("Pretrain requires model.num_experts")

    base_ds = _resolve_base_dataset(dataset)
    assignments_cfg = PretrainAssignmentConfig(
        method=cfg.pretrain.assignments_method,
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=str(cfg.pretrain.assignments_cache_path or ""),
        cluster_num_time_samples=int(cfg.pretrain.cluster_num_time_samples),
        spatial_blocks=cfg.pretrain.spatial_blocks,
        time_block_size=int(cfg.pretrain.time_block_size),
    )
    assignments = compute_pretrain_assignments(
        base_ds,
        num_experts=int(num_experts),
        cfg=assignments_cfg,
    )
    pretrain_loader = _build_pretrain_loader(dataset, cfg, assignments)

    if hasattr(model, "pretrain_parameters") and callable(getattr(model, "pretrain_parameters")):
        pretrain_params = list(model.pretrain_parameters())
    else:
        raise ValueError("Pretrain requires model.pretrain_parameters()")

    opt = torch.optim.Adam(
        pretrain_params,
        lr=cfg.pretrain.lr,
        betas=(0.9, 0.999),
    )

    start_time = time.time()
    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        counts = torch.zeros(int(num_experts), dtype=torch.long)
        iterator = pretrain_loader
        if tqdm is not None:
            iterator = tqdm(pretrain_loader, desc=f"pretrain {epoch}/{cfg.pretrain.epochs}", leave=False)
        for xb, expert_id in iterator:
            xb = xb.to(device)
            expert_id = expert_id.to(device)
            if hasattr(model, "pretrain_forward") and callable(getattr(model, "pretrain_forward")):
                logits = model.pretrain_forward(xb)
            else:
                raise ValueError("Pretrain requires model.pretrain_forward(x)")
            if logits.ndim != 2:
                raise ValueError(f"pretrain_forward must return shape (B, C), got {tuple(logits.shape)}")
            num_classes = int(logits.shape[-1])
            target_min = int(expert_id.min().item())
            target_max = int(expert_id.max().item())
            if target_min < 0 or target_max >= num_classes:
                raise ValueError(
                    "Invalid pretrain target labels: "
                    f"min={target_min}, max={target_max}, num_classes={num_classes}. "
                    "This usually means cached assignments were generated with a different num_experts. "
                    "Clear pretrain.assignments_cache_path (or set it to empty) and rerun."
                )
            loss = F.cross_entropy(logits, expert_id)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * xb.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == expert_id).sum().item()
            total += int(expert_id.numel())
            counts += torch.bincount(preds.detach().cpu(), minlength=int(num_experts))

        epoch_loss /= max(total, 1)
        acc = correct / max(total, 1)
        elapsed = time.time() - start_time
        logger.info(
            "Pretrain epoch %s/%s loss=%.6e acc=%.4f counts=%s time=%.1fs",
            epoch,
            cfg.pretrain.epochs,
            epoch_loss,
            acc,
            counts.tolist(),
            elapsed,
        )


def build_dataloaders(
    dataset: Dataset, cfg: TrainingConfig
) -> Tuple[DataLoader, Optional[DataLoader], Dataset, Optional[Dataset]]:
    train_ds, val_ds = dataset, None

    train_collate = _resolve_collate(train_ds)
    sampler = None
    shuffle = True
    if cfg.batches_per_epoch_budget > 0:
        base_ds = _resolve_base_dataset(train_ds)
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        total_samples_budget = int(cfg.batches_per_epoch_budget) * int(cfg.batch_size)
        sampler = TimeStepRandomSampler(train_ds, base_ds.volume_shape, total_samples_budget)
        shuffle = False
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
        shuffle=shuffle,
        sampler=sampler,
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
    logger.info("DataLoader build: %.2fs", time.perf_counter() - t0)
    is_multiview = hasattr(dataset, "view_specs")
    psnr_collate = _resolve_collate(dataset)
    psnr_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        psnr_kwargs["prefetch_factor"] = 4
    if psnr_collate is not None:
        psnr_kwargs["collate_fn"] = psnr_collate

    model = model.to(device)
    _maybe_run_pretrain(model, dataset, cfg, device)
    current_lr = float(cfg.lr)
    optim = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
    if cfg.loss_type == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    start_time = time.time()
    best_val = float("inf")
    best_state = None
    no_improve = 0
    expert_select_counts = None
    gradnorm_balancer = None
    sampler = getattr(train_loader, "sampler", None)
    timestep_sampler = sampler if isinstance(sampler, TimeStepRandomSampler) else None
    total_timesteps = None
    last_active_timesteps = None
    last_stride_groups = None
    if cfg.timestep_curriculum.enabled:
        if timestep_sampler is None:
            raise ValueError("timestep_curriculum requires batches_per_epoch_budget > 0.")
        total_timesteps = int(timestep_sampler.volume_shape.T)

    freeze_epoch = None
    router_frozen = False
    freeze_at = float(cfg.freeze_router_at)
    if freeze_at > 0:
        if freeze_at < 1.0:
            freeze_epoch = max(1, int(math.ceil(cfg.epochs * freeze_at)))
        else:
            freeze_epoch = int(freeze_at)
    if (
        cfg.gradient_balancer.enabled
        and str(cfg.gradient_balancer.method).strip().lower() == "gradnorm"
        and int(cfg.gradient_balancer.gradnorm_every_n_steps) <= 0
    ):
        raise ValueError("gradient_balancer.gradnorm_every_n_steps must be positive when GradNorm is enabled")
    if cfg.gradient_diag.enabled and int(cfg.gradient_diag.every_n_steps) <= 0:
        raise ValueError("gradient_diag.every_n_steps must be positive when gradient diagnostics are enabled")

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        hard_topk = True
        if cfg.hard_topk_warmup_epochs > 0 and epoch <= int(cfg.hard_topk_warmup_epochs):
            hard_topk = False
        if not router_frozen and freeze_epoch is not None and epoch >= freeze_epoch:
            frozen_params = _freeze_router_modules(model)
            logger.info("Freeze router modules at epoch %s (params frozen: %s)", epoch, frozen_params)
            optim = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=current_lr,
                betas=(0.9, 0.999),
            )
            router_frozen = True
        if cfg.lr_decay_step > 0 and cfg.lr_decay_rate > 0.0:
            steps = (epoch - 1) // int(cfg.lr_decay_step)
            current_lr = float(cfg.lr) * (float(cfg.lr_decay_rate) ** steps)
            # logger.info("Epoch %s: setting learning rate to %.6e", epoch, current_lr)
            for group in optim.param_groups:
                group["lr"] = current_lr
        if cfg.timestep_curriculum.enabled and timestep_sampler is not None:
            if cfg.timestep_curriculum.mode.lower() == "stride":
                active_indices, active_groups, group_total = _resolve_stride_groups(
                    cfg, epoch, total_timesteps
                )
                timestep_sampler.set_active_indices(active_indices)
                if last_stride_groups != active_groups:
                    logger.info("Curriculum stride groups: %s/%s", active_groups, group_total)
                    last_stride_groups = active_groups
            else:
                active_timesteps = _resolve_curriculum_timesteps(cfg, epoch, total_timesteps)
                timestep_sampler.set_active_timesteps(active_timesteps)
                if last_active_timesteps != active_timesteps:
                    logger.info("Curriculum timesteps: %s/%s", active_timesteps, total_timesteps)
                    last_active_timesteps = active_timesteps
        model.train()
        epoch_loss = 0.0
        steps_seen = 0
        multiview_recon_attr_sum_acc = 0.0
        multiview_recon_dim_norm_acc = 0.0
        multiview_recon_selected_acc = 0.0
        multiview_per_view_acc: Dict[str, Dict[str, float]] = {}
        multiview_selected_mode = "attr_sum"
        multiview_weight_mode = "static"
        last_grad_balance_state = None
        last_gradnorm_state = None
        last_grad_diag_state = None
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
            xb = xb.to(device, non_blocking=True)
            task_losses = None
            is_multiview_batch = _is_multiview_target(yb)
            if is_multiview_batch:
                yb = {name: tensor.to(device, non_blocking=True) for name, tensor in yb.items()}
                loss, loss_recon, aux, recon_breakdown, task_losses = _compute_multiview_loss(
                    model,
                    xb,
                    yb,
                    cfg,
                    return_aux=True,
                    return_breakdown=True,
                    return_task_losses=True,
                    hard_topk=hard_topk,
                )
                multiview_recon_attr_sum_acc += float(recon_breakdown["weighted_attr_sum_loss"])
                multiview_recon_dim_norm_acc += float(recon_breakdown["weighted_dim_normalized_loss"])
                multiview_recon_selected_acc += float(recon_breakdown["selected_loss"])
                multiview_selected_mode = str(recon_breakdown["selected_mode"])
                multiview_weight_mode = str(recon_breakdown.get("weight_mode", "static"))
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
                if (
                    cfg.gradient_diag.enabled
                    and task_losses is not None
                    and len(task_losses) > 1
                    and global_step % int(cfg.gradient_diag.every_n_steps) == 0
                ):
                    last_grad_diag_state = diagnose_multitask_gradients(task_losses, model)
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
                loss_recon = loss
                if hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                elif hasattr(model, "indicator_regularization"):
                    reg = model.indicator_regularization()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
            loss_for_update = loss
            if is_multiview_batch:
                balancer_method = str(cfg.gradient_balancer.method).strip().lower()
                if cfg.gradient_balancer.enabled and task_losses is not None and len(task_losses) > 1:
                    if balancer_method in {"pcgrad", "mgda", "cagrad"}:
                        optim.zero_grad()
                        last_grad_balance_state = _apply_multitask_gradient(model, task_losses, cfg)
                    elif balancer_method == "gradnorm":
                        if gradnorm_balancer is None:
                            gradnorm_balancer = GradNormBalancer(task_losses.keys(), cfg.gradient_balancer, device)
                        if global_step % int(cfg.gradient_balancer.gradnorm_every_n_steps) == 0:
                            last_gradnorm_state = gradnorm_balancer.update(task_losses, model)
                        weighted_loss = gradnorm_balancer.build_weighted_loss_only(task_losses)
                        loss_for_update = weighted_loss
                        optim.zero_grad()
                        weighted_loss.backward()
                    else:
                        raise ValueError(f"Unsupported gradient balancer method: {balancer_method}")
                else:
                    optim.zero_grad()
                    loss.backward()
            else:
                optim.zero_grad()
                loss.backward()
            optim.step()
            epoch_loss += loss_for_update.item()
            steps_seen += 1
            global_step += 1
            # loss_orth is only for logging; zero if not used
            if aux and "probs" in aux:
                probs = aux["probs"].detach()
                reduce_dims = tuple(range(probs.dim() - 1))
                counts = probs.float().sum(dim=reduce_dims).cpu()
                if expert_select_counts is None:
                    expert_select_counts = torch.zeros_like(counts)
                expert_select_counts += counts
            compute_time += time.perf_counter() - step_start
            prev_time = time.perf_counter()
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
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                elapsed = time.time() - start_time
                logger.info(
                    "Epoch %s/%s train=%.6e val=%.6e time=%.1fs (early stop)",
                    epoch,
                    cfg.epochs,
                    epoch_loss,
                    val_loss,
                    elapsed,
                )
                break

        if epoch % cfg.log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            if not is_multiview:
                logger.info(
                    "Epoch %s/%s train=%.6e time=%.1fs",
                    epoch,
                    cfg.epochs,
                    epoch_loss,
                    elapsed,
                )
            else:
                if val_loss is None:
                    logger.info(
                        "Epoch %s/%s train=%.6e time=%.1fs",
                        epoch,
                        cfg.epochs,
                        epoch_loss,
                        elapsed,
                    )
                else:
                    logger.info(
                        "Epoch %s/%s train=%.6e val=%.6e time=%.1fs",
                        epoch,
                        cfg.epochs,
                        epoch_loss,
                        val_loss,
                        elapsed,
                    )
                if steps_seen > 0:
                    logger.info(
                        "Recon loss (%s): attr_sum=%.6e dim_normalized=%.6e selected[%s]=%.6e",
                        multiview_weight_mode,
                        multiview_recon_attr_sum_acc / float(steps_seen),
                        multiview_recon_dim_norm_acc / float(steps_seen),
                        multiview_selected_mode,
                        multiview_recon_selected_acc / float(steps_seen),
                    )
                if last_grad_balance_state is not None:
                    method = str(last_grad_balance_state.get("method", "")).lower()
                    if method in {"mgda", "cagrad"}:
                        task_names = list(last_grad_balance_state.get("task_names") or [])
                        task_weights = list(last_grad_balance_state.get("task_weights") or [])
                        weight_text = " ".join(
                            f"{name}={float(weight):.6f}"
                            for name, weight in zip(task_names, task_weights)
                        )
                        logger.info("Gradient balancer [%s]: %s", method, weight_text)
                    elif method == "pcgrad":
                        logger.info("Gradient balancer [pcgrad]: applied")
                if last_gradnorm_state is not None:
                    gradnorm_weights = last_gradnorm_state.get("weights") or {}
                    weights_text = " ".join(
                        f"{name}={float(value):.6f}"
                        for name, value in gradnorm_weights.items()
                    )
                    logger.info("GradNorm weights: %s", weights_text)

                    gradnorm_grad_norms = last_gradnorm_state.get("grad_norms") or {}
                    grad_norms_text = " ".join(
                        f"{name}={float(value):.6f}"
                        for name, value in gradnorm_grad_norms.items()
                    )
                    logger.info("GradNorm grad_norms: %s", grad_norms_text)

                    gradnorm_relative_rates = last_gradnorm_state.get("relative_rates") or {}
                    relative_rates_text = " ".join(
                        f"{name}={float(value):.6f}"
                        for name, value in gradnorm_relative_rates.items()
                    )
                    logger.info("GradNorm relative_rates: %s", relative_rates_text)
                    logger.info("GradNorm grad_loss: %.6f", float(last_gradnorm_state.get("grad_loss", 0.0)))
                if last_grad_diag_state is not None:
                    grad_norms = last_grad_diag_state.get("grad_norms") or {}
                    grad_norm_text = " ".join(
                        f"{name}={float(value):.6f}"
                        for name, value in grad_norms.items()
                    )
                    logger.info("Gradient norms: %s", grad_norm_text)

                    pairwise_cos = last_grad_diag_state.get("pairwise_cos") or {}
                    pairwise_text = " ".join(
                        f"{task_i}-{task_j}={float(value):.6f}"
                        for (task_i, task_j), value in pairwise_cos.items()
                    )
                    logger.info("Pairwise cosine similarity: %s", pairwise_text)

                    max_layers_to_log = int(cfg.gradient_diag.max_layers_to_log)
                    layer_conflicts = list(last_grad_diag_state.get("layer_conflicts") or [])[:max_layers_to_log]
                    layer_text = " | ".join(
                        f"{item['layer']}: avg_cos={float(item['avg_cos']):.6f}, min_cos={float(item['min_cos']):.6f}, neg_ratio={float(item['neg_ratio']):.6f}"
                        for item in layer_conflicts
                    )
                    logger.info("Most conflicting layers: %s", layer_text)
            if expert_select_counts is not None:
                sumCount = expert_select_counts.sum().item()
                counts_text = " ".join(
                    f"E{i}={count:.2f} ({count / sumCount:.2%})"
                    for i, count in enumerate(expert_select_counts.tolist())
                )
                logger.info("Expert utilization rate: %s", counts_text)
                expert_select_counts = torch.zeros_like(expert_select_counts)
        if (cfg.log_psnr_every > 0 and (epoch % cfg.log_psnr_every == 0)):
            psnr_ds = dataset
            if cfg.psnr_sample_ratio < 1.0:
                n_total = len(dataset)
                n_sample = max(1, int(round(n_total * float(cfg.psnr_sample_ratio))))
                gen = torch.Generator()
                gen.manual_seed(int(cfg.seed))
                indices = torch.randperm(n_total, generator=gen)[:n_sample].tolist()
                psnr_ds = Subset(dataset, indices)
            psnr_loader = DataLoader(
                psnr_ds,
                batch_size=cfg.pred_batch_size,
                shuffle=False,
                **psnr_kwargs,
            )
            if not is_multiview:
                with torch.no_grad():
                    preds, gts = [], []
                    val_iter = psnr_loader
                    if tqdm is not None:
                        val_iter = tqdm(psnr_loader, desc=f"psnr {epoch}/{cfg.epochs}", leave=False)
                    for batch in val_iter:
                        xb, yb = _unpack_batch(batch)
                        xb = xb.to(device, non_blocking=True)
                        try:
                            pred = model(xb, hard_topk=hard_topk)
                        except TypeError:
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
                elapsed = time.time() - start_time
                logger.info(
                    "PSNR epoch %s/%s: %.2f time=%.1fs",
                    epoch,
                    cfg.epochs,
                    psnr_val,
                    elapsed,
                )
            elif is_multiview:
                with torch.no_grad():
                    pred_dict = {}
                    gt_dict = {}
                    val_iter = psnr_loader
                    if tqdm is not None:
                        val_iter = tqdm(psnr_loader, desc=f"psnr {epoch}/{cfg.epochs}", leave=False)
                    for batch in val_iter:
                        xb, yb = _unpack_batch(batch)
                        xb = xb.to(device, non_blocking=True)
                        try:
                            preds = model(xb, hard_topk=hard_topk)
                        except TypeError:
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
                elapsed = time.time() - start_time
                logger.info(
                    "PSNR epoch %s/%s: %s time=%.1fs",
                    epoch,
                    cfg.epochs,
                    psnr_text,
                    elapsed,
                )
        # if epoch % cfg.log_every == 0 or epoch == 1:
        #     print(f"Epoch {epoch} timing: data={data_time:.2f}s compute={compute_time:.2f}s")
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                model,
                dataset,
                cfg.save_model,
                suffix=f"_epoch{epoch}",
                run_timestamp=cfg.run_timestamp,
            )
            print(f"Debug: No predict_full is set, only checkpoint save at epoch {epoch}.")
            # predict_full(model, dataset, cfg, device, suffix=f"_epoch{epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, dataset, cfg.save_model, run_timestamp=cfg.run_timestamp)
    print(f"Debug: No predict_full is set, only checkpoint save at the end of training.")
    # predict_full(model, dataset, cfg, device)


def evaluate(
    model,
    loader,
    criterion,
    device,
    n_samples: int,
    cfg: TrainingConfig,
    hard_topk: bool = True,
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
                loss, _ = _compute_multiview_loss(
                    model,
                    xb,
                    yb,
                    cfg,
                    hard_topk=hard_topk,
                )
            else:
                yb = yb.to(device)
                try:
                    pred = model(xb, hard_topk=hard_topk)
                except TypeError:
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

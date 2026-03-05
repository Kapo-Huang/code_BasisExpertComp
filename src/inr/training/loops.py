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
    MultiAttrEMALoss,
    diversity_loss,
    load_balance_loss,
    orth_loss,
    reconstruction_loss_with_breakdown,
)
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
        for i, name in enumerate(attr_names):
            loss_per_dim = float(losses_t[i].detach().item())
            out_dim = float(dims[i].detach().item())
            weight = float(effective_w[i].detach().item())
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
            "effective_weights": {name: float(effective_w[i].detach().item()) for i, name in enumerate(attr_names)},
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


def _resolve_collate(ds: Dataset):
    base_ds = ds.dataset if isinstance(ds, Subset) else ds
    if isinstance(base_ds, MultiTargetVolumetricDataset):
        return make_multitarget_collate(base_ds)
    if isinstance(base_ds, VolumetricDataset):
        return make_singletarget_collate(base_ds)
    return None


def _resolve_base_dataset(ds: Dataset):
    return ds.dataset if isinstance(ds, Subset) else ds


class TimeStepRandomBatchSampler(Sampler[list]):
    def __init__(
        self,
        dataset: Dataset,
        volume_shape,
        total_samples_budget: int,
        batch_size: int,
        generator=None,
        active_timesteps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.volume_shape = volume_shape
        self.total_samples_budget = int(total_samples_budget)
        self.batch_size = int(batch_size)
        self.generator = generator

        self.samples_per_timestep = 0
        self.total_samples = 0
        self._active_timesteps = None
        self._active_indices = None
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
        self.set_active_timesteps(active_timesteps)

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
        X = int(self.volume_shape.X)
        Y = int(self.volume_shape.Y)
        Z = int(self.volume_shape.Z)
        V = X * Y * Z
        T = int(self.volume_shape.T)

        if self._active_indices is not None:
            active_indices = [t for t in self._active_indices if 0 <= t < T]
        elif self._active_timesteps is not None:
            active_indices = list(range(min(self._active_timesteps, T)))
        else:
            active_indices = list(range(T))

        if not active_indices:
            return

        active_tensor = torch.tensor(active_indices, dtype=torch.long)

        total = int(self.total_samples)
        bs = self.batch_size
        n_batches = total // bs
        if n_batches <= 0:
            return

        for _ in range(n_batches):
            t_choice = torch.randint(0, len(active_indices), (bs,), generator=self.generator)
            t_vals = active_tensor[t_choice]

            if self._per_t_indices is None:
                offsets = torch.randint(0, V, (bs,), generator=self.generator)
                batch_idx = (t_vals * V + offsets).cpu().numpy().astype(np.int64, copy=False)
                yield batch_idx
            else:
                t_vals_np = t_vals.cpu().numpy()
                out = np.empty(bs, dtype=np.int64)
                for t in np.unique(t_vals_np):
                    pos = np.where(t_vals_np == t)[0]
                    pool = self._per_t_indices[int(t)]
                    picks = torch.randint(0, len(pool), (len(pos),), generator=self.generator).cpu().numpy()
                    out[pos] = pool[picks]
                yield out

    def __len__(self) -> int:
        total = int(self.total_samples)
        bs = int(self.batch_size)
        return total // max(bs, 1)



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

    batch_sampler = None
    shuffle = True
    if cfg.batches_per_epoch_budget > 0:
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        total_samples_budget = int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size)
        batch_sampler = TimeStepRandomBatchSampler(
            dataset,
            base_ds.volume_shape,
            total_samples_budget,
            batch_size=cfg.pretrain.batch_size,
        )
        shuffle = False

    pretrain_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        pretrain_kwargs["prefetch_factor"] = 4
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            **pretrain_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=shuffle,
        **pretrain_kwargs,
    )


def _build_pretrain_coords_loader(dataset: Dataset, cfg: TrainingConfig) -> DataLoader:
    base_ds = _resolve_base_dataset(dataset)
    if isinstance(base_ds, MultiTargetVolumetricDataset):
        collate_fn = make_multitarget_collate(base_ds, read_targets=False)
    elif isinstance(base_ds, VolumetricDataset):
        collate_fn = make_singletarget_collate(base_ds, read_targets=False)
    else:
        raise TypeError(f"Unsupported dataset type: {type(base_ds)}")

    batch_sampler = None
    shuffle = True
    if cfg.batches_per_epoch_budget > 0:
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        total_samples_budget = int(cfg.batches_per_epoch_budget) * int(cfg.pretrain.batch_size)
        batch_sampler = TimeStepRandomBatchSampler(
            dataset,
            base_ds.volume_shape,
            total_samples_budget,
            batch_size=cfg.pretrain.batch_size,
        )
        shuffle = False

    pretrain_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": collate_fn,
    }
    if cfg.num_workers > 0:
        pretrain_kwargs["prefetch_factor"] = 4
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            **pretrain_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=shuffle,
        **pretrain_kwargs,
    )


def _maybe_run_pretrain(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig, device: torch.device):
    if not cfg.pretrain.enabled:
        return
    mode = str(cfg.pretrain.mode).strip().lower()
    if mode != "basis_two_stage" and cfg.pretrain.epochs <= 0:
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
            opt = torch.optim.Adam(
                list(model.pretrain_stage1_parameters()),
                lr=cfg.pretrain.lr,
                betas=(0.9, 0.999),
            )
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
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    steps += 1
                epoch_loss /= max(steps, 1)
                elapsed = time.time() - start_time
                logger.info(
                    "Pretrain stage1 %s/%s loss=%.6e div=%.3e orth=%.3e time=%.1fs",
                    epoch,
                    stage1_epochs,
                    epoch_loss,
                    loss_div.item(),
                    loss_orth.item(),
                    elapsed,
                )

        if stage2_epochs > 0:
            if not hasattr(model, "pretrain_stage2_parameters") or not callable(getattr(model, "pretrain_stage2_parameters")):
                raise ValueError("basis_two_stage requires model.pretrain_stage2_parameters()")
            if not hasattr(model, "pretrain_stage2_router") or not callable(getattr(model, "pretrain_stage2_router")):
                raise ValueError("basis_two_stage requires model.pretrain_stage2_router(x)")
            opt = torch.optim.Adam(
                list(model.pretrain_stage2_parameters()),
                lr=cfg.pretrain.lr,
                betas=(0.9, 0.999),
            )
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
                    if not hasattr(model, "pretrain_teacher_shared_feat") or not callable(getattr(model, "pretrain_teacher_shared_feat")):
                        raise ValueError("basis_two_stage requires model.pretrain_teacher_shared_feat(x)")
                    shared_teacher = model.pretrain_teacher_shared_feat(xb).detach()
                    probs, masks, expert_feats = model.pretrain_stage2_router(
                        xb,
                        temperature=cfg.pretrain.stage2_temperature,
                    )
                    expert_feats = expert_feats.detach()
                    h_router = torch.sum(probs.unsqueeze(-1) * expert_feats.unsqueeze(1), dim=2)  # (B, V, F)
                    bsz, num_views, feat_dim = h_router.shape
                    shared_router = model.decoder(h_router.reshape(-1, feat_dim)).reshape(bsz, num_views, -1)
                    shared_teacher_exp = shared_teacher.unsqueeze(1).expand_as(shared_router)
                    loss_recon = F.mse_loss(shared_router, shared_teacher_exp)
                    loss = loss_recon
                    if cfg.pretrain.stage2_entropy_weight > 0.0:
                        eps = 1e-9
                        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
                        loss = loss - cfg.pretrain.stage2_entropy_weight * ent
                    if cfg.pretrain.stage2_lam_eq > 0.0:
                        loss_eq = load_balance_loss(probs, masks)
                        loss = loss + cfg.pretrain.stage2_lam_eq * loss_eq
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    steps += 1
                epoch_loss /= max(steps, 1)
                elapsed = time.time() - start_time
                logger.info(
                    "Pretrain stage2 %s/%s loss=%.6e time=%.1fs",
                    epoch,
                    stage2_epochs,
                    epoch_loss,
                    elapsed,
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
    batch_sampler = None
    shuffle = True
    if cfg.batches_per_epoch_budget > 0:
        base_ds = _resolve_base_dataset(train_ds)
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        total_samples_budget = int(cfg.batches_per_epoch_budget) * int(cfg.batch_size)
        batch_sampler = TimeStepRandomBatchSampler(
            train_ds,
            base_ds.volume_shape,
            total_samples_budget,
            batch_size=cfg.batch_size,
        )
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
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            shuffle=False,
            **train_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
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
    multiview_ema_recon_loss: Optional[MultiAttrEMALoss] = None
    if is_multiview and cfg.multiview_ema_loss.enabled:
        if not hasattr(dataset, "view_specs") or not callable(getattr(dataset, "view_specs")):
            raise ValueError("multiview_ema_loss requires dataset.view_specs() for attribute names.")
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

    start_time = time.time()
    best_val = float("inf")
    best_state = None
    no_improve = 0
    expert_select_counts = None
    train_batch_sampler = getattr(train_loader, "batch_sampler", None)
    timestep_sampler = (
        train_batch_sampler if isinstance(train_batch_sampler, TimeStepRandomBatchSampler) else None
    )
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
        multiview_last_ema_state = None
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
            if _is_multiview_target(yb):
                yb = {name: tensor.to(device, non_blocking=True) for name, tensor in yb.items()}
                loss, loss_recon, loss_eq, loss_div, loss_orth, aux, recon_breakdown = _compute_multiview_loss(
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
                loss_recon = loss
                loss_eq = torch.zeros((), device=loss.device)
                loss_div = torch.zeros((), device=loss.device)
                loss_orth = torch.zeros((), device=loss.device)
                if cfg.gam_orth > 0.0 and "expert_feats" in aux:
                    loss_orth = orth_loss(aux["expert_feats"], eps=cfg.orth_eps)
                    loss = loss + cfg.gam_orth * loss_orth
                if hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
                elif hasattr(model, "indicator_regularization"):
                    reg = model.indicator_regularization()
                    loss = loss + (reg if torch.is_tensor(reg) else torch.tensor(reg, device=loss.device, dtype=loss.dtype))
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            steps_seen += 1
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
                ema_recon_loss=multiview_ema_recon_loss,
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

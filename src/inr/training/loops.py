import copy
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

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
from inr.training.losses import diversity_loss, load_balance_loss, orth_loss, reconstruction_loss
from inr.pretrain.voxel_clustering import compute_voxel_cluster_assignments
from inr.utils.io import save_checkpoint
from skimage.metrics import peak_signal_noise_ratio as psnr
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


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
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8000
    pred_batch_size: int = 8000
    num_workers: int = 4
    batches_per_timestep: int = 0
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


def _unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb
    raise ValueError(f"Unexpected batch structure: {len(batch)}")


def _is_multiview_target(targets) -> bool:
    return isinstance(targets, dict)


def _compute_multiview_loss(model, xb, yb, cfg: TrainingConfig, return_aux: bool = False):
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
        loss_div = diversity_loss(aux["expert_feats"], sigma=cfg.div_sigma)
        loss = loss + cfg.gam_div * loss_div

    loss_orth = torch.zeros((), device=xb.device)
    if cfg.gam_orth > 0.0 and "expert_feats" in aux:
        loss_orth = orth_loss(aux["expert_feats"], eps=cfg.orth_eps)
        loss = loss + cfg.gam_orth * loss_orth

    if return_aux:
        return loss, loss_recon, loss_eq, loss_div, loss_orth, aux
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


class TimeStepRandomSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        volume_shape,
        samples_per_timestep: int,
        generator=None,
        active_timesteps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.volume_shape = volume_shape
        self.samples_per_timestep = int(samples_per_timestep)
        self._active_timesteps = None
        self._active_indices = None
        self.total_samples = 0
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

    def set_active_timesteps(self, active_timesteps: Optional[int]):
        self._active_indices = None
        if active_timesteps is None:
            self._active_timesteps = None
            self.total_samples = self.samples_per_timestep * int(self.volume_shape.T)
            return
        active_timesteps = int(active_timesteps)
        if active_timesteps <= 0:
            raise ValueError("active_timesteps must be positive")
        max_t = int(self.volume_shape.T)
        if active_timesteps > max_t:
            active_timesteps = max_t
        self._active_timesteps = active_timesteps
        self.total_samples = self.samples_per_timestep * int(self._active_timesteps)

    def set_active_indices(self, indices: Optional[list]):
        self._active_timesteps = None
        if indices is None:
            self._active_indices = None
            self.total_samples = self.samples_per_timestep * int(self.volume_shape.T)
            return
        if not indices:
            raise ValueError("active_indices must be a non-empty list")
        self._active_indices = [int(t) for t in indices]
        self.total_samples = self.samples_per_timestep * len(self._active_indices)

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
        if self._per_t_indices is None:
            for t in active_indices:
                offsets = torch.randint(0, V, (self.samples_per_timestep,), generator=self.generator)
                base = t * V
                for offset in offsets.tolist():
                    yield base + offset
        else:
            for t in active_indices:
                pool = self._per_t_indices[t]
                picks = torch.randint(0, len(pool), (self.samples_per_timestep,), generator=self.generator)
                for pick in picks.tolist():
                    yield int(pool[pick])

    def __len__(self) -> int:
        return self.total_samples


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
        shuffle=True,
        **pretrain_kwargs,
    )


def _maybe_run_pretrain(model: torch.nn.Module, dataset: Dataset, cfg: TrainingConfig, device: torch.device):
    if not cfg.pretrain.enabled or cfg.pretrain.epochs <= 0:
        return
    if cfg.pretrain.batch_size <= 0:
        raise ValueError("pretrain.batch_size must be positive")
    if cfg.pretrain.cluster_num_time_samples <= 0:
        raise ValueError("pretrain.cluster_num_time_samples must be positive")
    if not (hasattr(model, "encoder") and hasattr(model, "policy") and hasattr(model, "experts")):
        raise ValueError("Pretrain requires model.encoder, model.policy, and model.experts")
    num_experts = getattr(model, "num_experts", None)
    if num_experts is None:
        raise ValueError("Pretrain requires model.num_experts")

    base_ds = _resolve_base_dataset(dataset)
    assignments = compute_voxel_cluster_assignments(
        base_ds,
        num_experts=int(num_experts),
        num_time_samples=int(cfg.pretrain.cluster_num_time_samples),
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=cfg.pretrain.assignments_cache_path or None,
    )
    pretrain_loader = _build_pretrain_loader(dataset, cfg, assignments)

    expert_flags = [p.requires_grad for p in model.experts.parameters()]
    for p in model.experts.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.policy.parameters()),
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
            enc_feat = model.encoder(xb)
            _, logits, _ = model.policy(xb, enc_feat)
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
        print(
            f"Pretrain epoch {epoch}/{cfg.pretrain.epochs} "
            f"loss={epoch_loss:.6e} acc={acc:.4f} "
            f"counts={counts.tolist()} time={elapsed:.1f}s"
        )

    for p, flag in zip(model.experts.parameters(), expert_flags):
        p.requires_grad = flag


def build_dataloaders(
    dataset: Dataset, cfg: TrainingConfig
) -> Tuple[DataLoader, Optional[DataLoader], Dataset, Optional[Dataset]]:
    train_ds, val_ds = dataset, None

    train_collate = _resolve_collate(train_ds)
    sampler = None
    shuffle = True
    if cfg.batches_per_timestep > 0:
        base_ds = _resolve_base_dataset(train_ds)
        if not hasattr(base_ds, "volume_shape") or base_ds.volume_shape is None:
            raise ValueError("Time-step sampling requires dataset.volume_shape with T dimension.")
        samples_per_timestep = int(cfg.batches_per_timestep) * int(cfg.batch_size)
        sampler = TimeStepRandomSampler(train_ds, base_ds.volume_shape, samples_per_timestep)
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
    print(f"DataLoader build: {time.perf_counter() - t0:.2f}s")
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
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    if cfg.loss_type == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    start_time = time.time()
    best_val = float("inf")
    best_state = None
    no_improve = 0
    expert_select_counts = None
    sampler = getattr(train_loader, "sampler", None)
    timestep_sampler = sampler if isinstance(sampler, TimeStepRandomSampler) else None
    total_timesteps = None
    last_active_timesteps = None
    last_stride_groups = None
    if cfg.timestep_curriculum.enabled:
        if timestep_sampler is None:
            raise ValueError("timestep_curriculum requires batches_per_timestep > 0.")
        total_timesteps = int(timestep_sampler.volume_shape.T)

    for epoch in range(1, cfg.epochs + 1):
        if cfg.lr_decay_step > 0 and cfg.lr_decay_rate > 0.0:
            steps = (epoch - 1) // int(cfg.lr_decay_step)
            lr = float(cfg.lr) * (float(cfg.lr_decay_rate) ** steps)
            print(f"Epoch {epoch}: setting learning rate to {lr:.6e}")
            for group in optim.param_groups:
                group["lr"] = lr
        if cfg.timestep_curriculum.enabled and timestep_sampler is not None:
            if cfg.timestep_curriculum.mode.lower() == "stride":
                active_indices, active_groups, group_total = _resolve_stride_groups(
                    cfg, epoch, total_timesteps
                )
                timestep_sampler.set_active_indices(active_indices)
                if last_stride_groups != active_groups:
                    print(f"Curriculum stride groups: {active_groups}/{group_total}")
                    last_stride_groups = active_groups
            else:
                active_timesteps = _resolve_curriculum_timesteps(cfg, epoch, total_timesteps)
                timestep_sampler.set_active_timesteps(active_timesteps)
                if last_active_timesteps != active_timesteps:
                    print(f"Curriculum timesteps: {active_timesteps}/{total_timesteps}")
                    last_active_timesteps = active_timesteps
        model.train()
        epoch_loss = 0.0
        steps_seen = 0
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
                loss, loss_recon, loss_eq, loss_div, loss_orth, aux = _compute_multiview_loss(
                    model, xb, yb, cfg, return_aux=True
                )
            else:
                yb = yb.to(device, non_blocking=True)
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
            if aux and "masks" in aux:
                masks = aux["masks"].detach()
                reduce_dims = tuple(range(masks.dim() - 1))
                counts = masks.float().sum(dim=reduce_dims).to(torch.long).cpu()
                if expert_select_counts is None:
                    expert_select_counts = torch.zeros_like(counts)
                expert_select_counts += counts
            compute_time += time.perf_counter() - step_start
            prev_time = time.perf_counter()
        epoch_loss /= max(steps_seen, 1)

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
            if not is_multiview:
                print(f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} time={elapsed:.1f}s")
            else:
                if val_loss is None:
                    print(f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} time={elapsed:.1f}s")
                else:
                    print(f"Epoch {epoch}/{cfg.epochs} train={epoch_loss:.6e} val={val_loss:.6e} time={elapsed:.1f}s")
            if expert_select_counts is not None:
                sumCount = expert_select_counts.sum().item()
                counts_text = " ".join(
                    f"E{i}={count} ({count / sumCount:.2%})" for i, count in enumerate(expert_select_counts.tolist())
                )
                print(f"Expert utilization rate: {counts_text}")
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
                print(f"PSNR epoch {epoch}/{cfg.epochs}: {psnr_val:.2f} time={elapsed:.1f}s")
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
                print(f"PSNR epoch {epoch}/{cfg.epochs}: {psnr_text} time={elapsed:.1f}s")
        # if epoch % cfg.log_every == 0 or epoch == 1:
        #     print(f"Epoch {epoch} timing: data={data_time:.2f}s compute={compute_time:.2f}s")
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
                loss, _, _, _, _ = _compute_multiview_loss(model, xb, yb, cfg)
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

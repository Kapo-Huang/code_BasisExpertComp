import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, TensorDataset

from inr.data import MultiViewCoordDataset, NodeDataset
from inr.models.sota.mc_inr import mc_inr
from inr.utils.io import save_checkpoint

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class MCTrainingConfig:
    # --- Basic ---
    epochs: int = 200
    batch_size: int = 65536
    num_workers: int = 4
    lr: float = 5e-5
    lr_decay_step: int = 30
    lr_decay_gamma: float = 0.92
    log_every: int = 1
    save_every: int = 20
    seed: int = 42
    device: Optional[str] = None
    resume_path: Optional[str] = None

    # --- Paths ---
    data_x_path: str = ""
    data_y_path: str = ""
    data_attr_paths: Optional[Dict[str, str]] = None
    data_normalize: bool = True
    data_stats_path: Optional[str] = None
    save_model: str = "outputs/mc_inr_model.pth"
    save_pred: str = "outputs/pred.npy"

    # --- Model Arch ---
    hidden_features: int = 64
    gfe_layers: int = 5
    lfe_layers: int = 6

    # --- Clustering / Training ---
    initial_k: int = 20
    sampling_ratio: float = 0.3
    split_threshold: float = 5e-4
    split_check_interval: int = 30
    min_split_points: int = 2
    max_recluster_rounds: int = 8

    # --- Convergence ---
    convergence_patience: int = 30
    convergence_delta: float = 0.0

    # --- Fine-tuning ---
    finetune_epochs: int = 200
    finetune_lr: Optional[float] = None
    finetune_sampling_ratio: float = 1.0

    # --- Compatibility switches ---
    split_after_meta: bool = False
    recluster_after_finetune: bool = False


class MCINRDataset(Dataset):
    def __init__(self, coords, targets, cluster_ids):
        self.coords = coords
        self.targets = targets
        self.cluster_ids = cluster_ids

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.targets[idx], self.cluster_ids[idx]

    def update_labels(self, indices, new_labels):
        self.cluster_ids[indices] = new_labels.to(self.cluster_ids.device)


def load_mc_checkpoint(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_coords_and_targets(cfg: MCTrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.data_attr_paths:
        if len(cfg.data_attr_paths) == 0:
            raise ValueError("data_attr_paths is provided but empty.")
        ds = MultiViewCoordDataset(
            cfg.data_x_path,
            cfg.data_attr_paths,
            normalize=bool(cfg.data_normalize),
            stats_path=cfg.data_stats_path,
        )
        coords = ds.x
        ordered_names = list(cfg.data_attr_paths.keys())
        target_blocks = [ds.y[name] for name in ordered_names]
        targets = torch.cat(target_blocks, dim=1)
        target_dims = {name: int(ds.y[name].shape[1]) for name in ordered_names}
        print(
            f"Loaded multi-attribute targets from attr_paths: {ordered_names} "
            f"with dims={target_dims} -> out_vars={targets.shape[1]}"
        )
    else:
        if not cfg.data_y_path:
            raise ValueError("Either `data_y_path` or `data_attr_paths` must be provided.")
        ds = NodeDataset(
            cfg.data_x_path,
            cfg.data_y_path,
            normalize=bool(cfg.data_normalize),
            stats_path=cfg.data_stats_path,
        )
        coords = ds.x
        targets = ds.y

    if coords.ndim == 1:
        coords = coords[:, None]
    if targets.ndim == 1:
        targets = targets[:, None]

    coords = coords.to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)
    print(
        f"Data paths | x={Path(cfg.data_x_path).resolve()} "
        f"| y={Path(cfg.data_y_path).resolve() if cfg.data_y_path else 'attr_paths'} "
        f"| normalize={bool(cfg.data_normalize)}"
    )
    return coords, targets


def _print_model_size(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} ({trainable_params:,} trainable)")


def _sample_indices_by_cluster(cluster_ids: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
    sampled_indices = []
    unique_cluster_ids = torch.unique(cluster_ids)

    for cluster_id in unique_cluster_ids:
        cluster_indices = torch.nonzero(cluster_ids == cluster_id, as_tuple=True)[0]
        if cluster_indices.numel() == 0:
            continue

        take = max(1, int(math.ceil(cluster_indices.numel() * sampling_ratio)))
        take = min(take, cluster_indices.numel())
        local_perm = torch.randperm(cluster_indices.numel())[:take]
        sampled_indices.append(cluster_indices[local_perm])

    if not sampled_indices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(sampled_indices, dim=0)


def perform_kmeans_clustering(coords, K=20, seed=42):
    """
    Returns:
        cluster_indices_list: list[np.ndarray]
        centroids: np.ndarray [K, 3]
    """
    if hasattr(coords, "cpu"):
        spatial_coords = coords[:, :3].cpu().numpy()
    else:
        spatial_coords = coords[:, :3]

    print(f"Executing K-Means (K={K})...")
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(spatial_coords)
    centroids = kmeans.cluster_centers_

    clusters = []
    for i in range(K):
        indices = np.where(labels == i)[0]
        clusters.append(indices)

    return clusters, centroids


def _build_initial_labels_from_kmeans(coords: torch.Tensor, cfg: MCTrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    cluster_indices_list, initial_centroids_np = perform_kmeans_clustering(
        coords, K=cfg.initial_k, seed=cfg.seed
    )
    initial_centroids = torch.tensor(initial_centroids_np, dtype=torch.float32)

    labels = torch.zeros(coords.shape[0], dtype=torch.long)
    for c_id, indices in enumerate(cluster_indices_list):
        if len(indices) > 0:
            labels[indices] = c_id
    return labels, initial_centroids


def _infer_labels_from_centroids(coords: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(coords[:, :3], centroids)
    return torch.argmin(dists, dim=1).to(dtype=torch.long)


def _build_model_and_dataset_from_checkpoint(
    cfg: MCTrainingConfig,
    payload: Dict[str, Any],
    coords: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Tuple[mc_inr, MCINRDataset]:
    if "model_state" not in payload:
        raise KeyError("Checkpoint missing `model_state` for resume.")
    model_state = payload["model_state"]

    centroids = model_state.get("centroids")
    if centroids is None:
        labels, centroids = _build_initial_labels_from_kmeans(coords, cfg)
    else:
        centroids = torch.as_tensor(centroids, dtype=torch.float32)
        labels = None

    cluster_num = int(centroids.shape[0])
    in_features = int(coords.shape[1])
    out_vars = int(targets.shape[1])
    model = mc_inr(
        cluster_num=cluster_num,
        initial_centroids=centroids,
        in_features=in_features,
        out_vars=out_vars,
        hidden_features=cfg.hidden_features,
        gfe_layers=cfg.gfe_layers,
        lfe_layers=cfg.lfe_layers,
    ).to(device)
    model.load_state_dict(model_state)

    cluster_ids = payload.get("cluster_ids")
    if cluster_ids is not None:
        cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long).cpu()
        if cluster_ids.numel() != coords.shape[0]:
            print(
                "Warning: checkpoint cluster_ids length mismatch; recalculating labels by nearest centroid."
            )
            cluster_ids = _infer_labels_from_centroids(coords, model.centroids.detach().cpu())
    else:
        if labels is None:
            cluster_ids = _infer_labels_from_centroids(coords, model.centroids.detach().cpu())
        else:
            cluster_ids = labels

    dataset = MCINRDataset(coords, targets, cluster_ids)
    return model, dataset


def restore_model_dataset_from_checkpoint(
    cfg: MCTrainingConfig,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[mc_inr, MCINRDataset, Dict[str, Any]]:
    device = device or torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = load_mc_checkpoint(checkpoint_path)

    coords, targets = _load_coords_and_targets(cfg)

    model, dataset = _build_model_and_dataset_from_checkpoint(cfg, payload, coords, targets, device)
    return model, dataset, payload


def perform_final_split(model, dataset, cfg, device) -> int:
    """
    Split clusters whose full-data MSE is above threshold.
    Returns:
        split_count: number of clusters actually split.
    """
    print("\n[Re-Clustering] Checking cluster residuals...")

    cluster_sses = {}
    cluster_counts = {}

    full_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()

    with torch.no_grad():
        iterator = full_loader
        if tqdm:
            iterator = tqdm(full_loader, desc="Calculating cluster MSE", leave=False)
        for xb, yb, cid in iterator:
            xb, yb, cid = xb.to(device), yb.to(device), cid.to(device)
            preds = model(xb, cluster_idx=cid)
            point_mse = torch.mean((preds - yb) ** 2, dim=-1)

            for c in torch.unique(cid):
                c_item = c.item()
                mask = cid == c
                count = int(mask.sum().item())
                if count == 0:
                    continue
                cluster_sses[c_item] = cluster_sses.get(c_item, 0.0) + point_mse[mask].sum().item()
                cluster_counts[c_item] = cluster_counts.get(c_item, 0) + count

    split_targets = []
    current_cluster_count = len(model.clusters)

    for c_id in range(current_cluster_count):
        count = cluster_counts.get(c_id, 0)
        if count < cfg.min_split_points:
            continue

        mse = cluster_sses[c_id] / count
        if mse > cfg.split_threshold:
            print(f"  -> Cluster {c_id}: MSE {mse:.6f} > {cfg.split_threshold}. Split.")
            split_targets.append(c_id)

    if not split_targets:
        print("  -> No cluster exceeds split threshold.")
        return 0

    split_count = 0
    for old_cid in split_targets:
        indices = torch.nonzero(dataset.cluster_ids == old_cid, as_tuple=True)[0]
        if indices.numel() < cfg.min_split_points:
            continue

        subset_coords = dataset.coords[indices, :3].cpu().numpy()
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=cfg.seed)
        sub_labels = kmeans.fit_predict(subset_coords)
        new_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        new_id_1, new_id_2 = model.split_specific_cluster(old_cid, new_centroids)

        new_labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=dataset.cluster_ids.device)
        mapped_labels = torch.where(new_labels_tensor == 0, new_id_1, new_id_2)
        dataset.update_labels(indices, mapped_labels)
        split_count += 1

    print(f"  -> Split complete. split_count={split_count}, total_clusters={len(model.clusters)}")
    return split_count


def train_mc_inr(cfg: MCTrainingConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Starting MC-INR meta stage on {device}")

    print(f"Loading data from {cfg.data_x_path}...")
    coords, targets = _load_coords_and_targets(cfg)
    N = coords.shape[0]
    in_features = coords.shape[1]
    out_vars = targets.shape[1]
    print(f"Data: {N} samples | Input: {in_features} | Output: {out_vars}")

    resume_payload = None
    stage = None
    if cfg.resume_path:
        print(f"Resume requested from: {cfg.resume_path}")
        resume_payload = load_mc_checkpoint(cfg.resume_path)
        stage = resume_payload.get("mc_stage")

    if resume_payload is not None:
        model, dataset = _build_model_and_dataset_from_checkpoint(cfg, resume_payload, coords, targets, device)
        if stage == "finetune":
            print("Resume checkpoint is from finetune stage; skipping meta training stage.")
            return model, dataset
        print(f"Resumed meta stage checkpoint. Stage={stage!r}")
    else:
        print(f"Phase 1: Initial Clustering (K={cfg.initial_k})...")
        initial_labels, initial_centroids = _build_initial_labels_from_kmeans(coords, cfg)
        dataset = MCINRDataset(coords, targets, initial_labels)

        print("Phase 2: Initializing Model...")
        model = mc_inr(
            cluster_num=cfg.initial_k,
            initial_centroids=initial_centroids,
            in_features=in_features,
            out_vars=out_vars,
            hidden_features=cfg.hidden_features,
            gfe_layers=cfg.gfe_layers,
            lfe_layers=cfg.lfe_layers,
        ).to(device)

    _print_model_size(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_gamma
    )
    criterion = nn.MSELoss()

    start_epoch = 1
    best_loss = float("inf")
    epochs_no_improve = 0
    if resume_payload is not None and stage in (None, "meta"):
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            print("Loaded optimizer state for meta resume.")
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
            print("Loaded scheduler state for meta resume.")
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_loss = float(resume_payload.get("best_loss", best_loss))
        epochs_no_improve = int(resume_payload.get("epochs_no_improve", 0))

    if start_epoch > cfg.epochs:
        print(
            f"Meta stage already reached epoch {start_epoch - 1} (target epochs={cfg.epochs}). "
            "Skipping meta training."
        )
        return model, dataset

    print("Phase 3: Meta Stage Training (cluster-wise random sampling)...")
    start_time = time.time()
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        last_epoch = epoch

        subset_indices = _sample_indices_by_cluster(dataset.cluster_ids, cfg.sampling_ratio)
        subset_size = int(subset_indices.numel())
        if subset_size == 0:
            raise RuntimeError("No sampled points for meta training. Check sampling_ratio.")

        batch_coords = dataset.coords[subset_indices]
        batch_targets = dataset.targets[subset_indices]
        batch_cids = dataset.cluster_ids[subset_indices]

        train_loader = DataLoader(
            TensorDataset(batch_coords, batch_targets, batch_cids),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        epoch_loss = 0.0
        iterator = train_loader
        if tqdm:
            iterator = tqdm(train_loader, desc=f"Meta Epoch {epoch}/{cfg.epochs}", leave=False)

        for xb, yb, cid in iterator:
            xb, yb, cid = xb.to(device), yb.to(device), cid.to(device)

            preds = model(xb, cluster_idx=cid)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.shape[0]

        epoch_loss /= subset_size
        scheduler.step()

        if best_loss - epoch_loss > cfg.convergence_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % cfg.log_every == 0:
            elapsed = time.time() - start_time
            lr_curr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{cfg.epochs} | Loss: {epoch_loss:.6e} | "
                f"Best: {best_loss:.6e} | NoImprove: {epochs_no_improve} | "
                f"LR: {lr_curr:.2e} | Time: {elapsed:.1f}s"
            )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                model,
                dataset,
                cfg.save_model,
                suffix=f"_meta_epoch{epoch}",
                epoch=epoch,
                optimizer=optimizer,
                extra_payload={
                    "mc_stage": "meta",
                    "cluster_ids": dataset.cluster_ids.detach().cpu(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_loss": float(best_loss),
                    "epochs_no_improve": int(epochs_no_improve),
                },
            )

        if epochs_no_improve >= cfg.convergence_patience:
            print(
                f"Early stop at epoch {epoch}: no loss improvement for "
                f"{cfg.convergence_patience} epochs."
            )
            break

    if cfg.split_after_meta:
        print("\nPhase 4: Optional split after meta stage...")
        perform_final_split(model, dataset, cfg, device)

    save_checkpoint(
        model,
        dataset,
        cfg.save_model,
        suffix="_meta",
        epoch=last_epoch,
        optimizer=optimizer,
        extra_payload={
            "mc_stage": "meta",
            "cluster_ids": dataset.cluster_ids.detach().cpu(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": float(best_loss),
            "epochs_no_improve": int(epochs_no_improve),
        },
    )
    return model, dataset


def predict_full(model, dataset, cfg, device, epoch: Optional[int] = None):
    """
    Inference on full dataset using automatic nearest-centroid routing.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_preds = []

    with torch.no_grad():
        iterator = loader
        if tqdm:
            iterator = tqdm(loader, desc="Predicting (Auto Routing)")
        for xb, _, _ in iterator:
            xb = xb.to(device)
            pred = model(xb, cluster_idx=None)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    save_path = Path(cfg.save_pred)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(cfg, "data_attr_paths", None):
        attr_paths = cfg.data_attr_paths
        ordered_names = list(attr_paths.keys())
        dims = []
        for name in ordered_names:
            arr = np.load(attr_paths[name], mmap_mode="r", allow_pickle=False)
            dim = 1 if arr.ndim == 1 else arr.shape[1]
            dims.append(dim)

        if sum(dims) != all_preds.shape[1]:
            print(
                "Warning: attr dims do not sum to prediction width. "
                "Saving combined prediction file only."
            )
            pred_name = save_path.stem
            if epoch is not None:
                pred_name = f"{pred_name}_epoch{epoch}"
            combined_path = save_path.with_name(f"{pred_name}.npy")
            np.save(combined_path, all_preds)
            print(f"Predictions saved to {combined_path}")
            return

        start = 0
        for name, dim in zip(ordered_names, dims):
            end = start + dim
            pred_chunk = all_preds[:, start:end]
            pred_name = save_path.stem
            if epoch is not None:
                pred_name = f"{pred_name}_epoch{epoch}"
            pred_name = f"{pred_name}_{name}.npy"
            pred_path = save_path.with_name(pred_name)
            np.save(pred_path, pred_chunk)
            print(f"Predictions saved to {pred_path}")
            start = end
    else:
        pred_name = save_path.stem
        if epoch is not None:
            pred_name = f"{pred_name}_epoch{epoch}"
        pred_path = save_path.with_name(f"{pred_name}.npy")
        np.save(pred_path, all_preds)
        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    print("Please run via configuration yaml.")

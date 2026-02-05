import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from inr.training.mc_meta import _sample_indices_by_cluster, load_mc_checkpoint
from inr.utils.io import save_checkpoint

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _build_optimizer_and_scheduler(model, cfg):
    finetune_lr = cfg.finetune_lr if cfg.finetune_lr is not None else cfg.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_gamma
    )
    return optimizer, scheduler, finetune_lr


def train_full_finetuning(model, dataset, cfg):
    """
    Full-data fine-tuning stage.
    No cluster splitting is performed in this stage.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FULL DATA FINE-TUNING")
    print("=" * 60)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    finetune_epochs = int(getattr(cfg, "finetune_epochs", cfg.epochs))
    patience = max(1, int(getattr(cfg, "convergence_patience", 30)))
    delta = float(getattr(cfg, "convergence_delta", 0.0))

    criterion = nn.MSELoss()
    sampling_ratio = 1.0
    if sampling_ratio <= 0:
        raise ValueError("finetune_sampling_ratio must be > 0.")
    if sampling_ratio > 1.0:
        sampling_ratio = 1.0
    start_time = time.time()

    optimizer, scheduler, lr = _build_optimizer_and_scheduler(model, cfg)
    print(f"Fine-tuning LR: {lr:.2e}")

    start_epoch = 1
    best_loss = float("inf")
    no_improve_epochs = 0

    resume_path = getattr(cfg, "resume_path", None)
    if resume_path:
        payload = load_mc_checkpoint(resume_path)
        stage = payload.get("mc_stage")
        if stage == "finetune":
            print(f"Resuming fine-tuning from checkpoint: {resume_path}")
            model.load_state_dict(payload["model_state"])

            cluster_ids = payload.get("cluster_ids")
            if cluster_ids is not None:
                cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long)
                if cluster_ids.numel() == dataset.cluster_ids.numel():
                    dataset.cluster_ids = cluster_ids.cpu()
                else:
                    print("Warning: resume cluster_ids length mismatch; keeping current dataset labels.")

            if "optimizer_state" in payload:
                optimizer.load_state_dict(payload["optimizer_state"])
                print("Loaded optimizer state for fine-tune resume.")
            if "scheduler_state" in payload:
                scheduler.load_state_dict(payload["scheduler_state"])
                print("Loaded scheduler state for fine-tune resume.")

            last_epoch = int(payload.get("ft_epoch", payload.get("epoch", 0)))
            start_epoch = last_epoch + 1
            best_loss = float(payload.get("best_loss", best_loss))
            no_improve_epochs = int(payload.get("epochs_no_improve", 0))

    if start_epoch > finetune_epochs:
        print(
            f"Fine-tuning already reached epoch {start_epoch - 1} "
            f"(target epochs={finetune_epochs}). Skipping fine-tuning."
        )
        return model, start_epoch - 1

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, finetune_epochs + 1):
        final_epoch = epoch
        model.train()

        if sampling_ratio < 1.0:
            subset_indices = _sample_indices_by_cluster(dataset.cluster_ids, sampling_ratio)
            subset_size = int(subset_indices.numel())
            if subset_size == 0:
                raise RuntimeError("No sampled points for fine-tuning. Check finetune_sampling_ratio.")

            batch_coords = dataset.coords[subset_indices]
            batch_targets = dataset.targets[subset_indices]
            batch_cids = dataset.cluster_ids[subset_indices]
            loader = DataLoader(
                TensorDataset(batch_coords, batch_targets, batch_cids),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )

        epoch_loss = 0.0
        total_samples = 0
        iterator = loader
        if tqdm:
            iterator = tqdm(loader, desc=f"FT Epoch {epoch}/{finetune_epochs}", leave=False)

        for xb, yb, cid in iterator:
            xb, yb, cid = xb.to(device), yb.to(device), cid.to(device)
            preds = model(xb, cluster_idx=cid)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = epoch_loss / max(total_samples, 1)
        scheduler.step()

        if best_loss - avg_loss > delta:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch % cfg.log_every == 0:
            elapsed = time.time() - start_time
            lr_curr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{finetune_epochs} | Loss: {avg_loss:.6e} | "
                f"Best: {best_loss:.6e} | NoImprove: {no_improve_epochs} | "
                f"LR: {lr_curr:.2e} | Time: {elapsed:.1f}s"
            )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                model,
                dataset,
                cfg.save_model,
                suffix=f"_ft_epoch{epoch}",
                epoch=epoch,
                optimizer=optimizer,
                extra_payload={
                    "mc_stage": "finetune",
                    "cluster_ids": dataset.cluster_ids.detach().cpu(),
                    "scheduler_state": scheduler.state_dict(),
                    "ft_epoch": int(epoch),
                    "best_loss": float(best_loss),
                    "epochs_no_improve": int(no_improve_epochs),
                },
            )

        if no_improve_epochs >= patience:
            print(
                f"Fine-tuning converged: no loss improvement for {patience} epochs."
            )
            break
        
    epoch_path = cfg.save_model.replace(".pth", f"_ft_epoch{final_epoch}.pth")
    save_checkpoint(
            model,
            dataset,
            epoch_path,
            epoch=final_epoch,
            optimizer=optimizer,
            extra_payload=payload,
    )
    return model, final_epoch

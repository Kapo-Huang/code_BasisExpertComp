import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class EpochTimeBreakdown:
    data_loading: float = 0.0
    device_transfer: float = 0.0
    forward_loss: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0
    expert_stats: float = 0.0
    validation: float = 0.0
    psnr: float = 0.0
    checkpoint: float = 0.0
    logging: float = 0.0


def maybe_sync_timing(device: torch.device, sync_cuda: bool) -> None:
    if not sync_cuda:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timing_start(device: torch.device, sync_cuda: bool) -> float:
    maybe_sync_timing(device, sync_cuda)
    return time.perf_counter()


def timing_elapsed(start_time: float, device: torch.device, sync_cuda: bool) -> float:
    maybe_sync_timing(device, sync_cuda)
    return time.perf_counter() - start_time


def format_timing_ratio(value: float, total: float) -> str:
    if total <= 0.0:
        return "0.0%"
    return f"{(value / total) * 100.0:.1f}%"


def log_epoch_time_breakdown(epoch: int, total_epochs: int, epoch_total: float, timing: EpochTimeBreakdown) -> None:
    train_total = (
        timing.data_loading
        + timing.device_transfer
        + timing.forward_loss
        + timing.backward
        + timing.optimizer
        + timing.expert_stats
    )
    accounted_total = train_total + timing.validation + timing.psnr + timing.checkpoint + timing.logging
    other_total = max(epoch_total - accounted_total, 0.0)

    logger.info(
        "Epoch %s/%s timing(total): total=%.2fs train=%.2fs (%s) val=%.2fs (%s) psnr=%.2fs (%s) save=%.2fs (%s) log=%.2fs (%s) other=%.2fs (%s)",
        epoch,
        total_epochs,
        epoch_total,
        train_total,
        format_timing_ratio(train_total, epoch_total),
        timing.validation,
        format_timing_ratio(timing.validation, epoch_total),
        timing.psnr,
        format_timing_ratio(timing.psnr, epoch_total),
        timing.checkpoint,
        format_timing_ratio(timing.checkpoint, epoch_total),
        timing.logging,
        format_timing_ratio(timing.logging, epoch_total),
        other_total,
        format_timing_ratio(other_total, epoch_total),
    )
    logger.info(
        "Epoch %s/%s timing(train): data=%.2fs (%s) transfer=%.2fs (%s) forward+loss=%.2fs (%s) backward=%.2fs (%s) optimizer=%.2fs (%s) router_stats=%.2fs (%s)",
        epoch,
        total_epochs,
        timing.data_loading,
        format_timing_ratio(timing.data_loading, train_total),
        timing.device_transfer,
        format_timing_ratio(timing.device_transfer, train_total),
        timing.forward_loss,
        format_timing_ratio(timing.forward_loss, train_total),
        timing.backward,
        format_timing_ratio(timing.backward, train_total),
        timing.optimizer,
        format_timing_ratio(timing.optimizer, train_total),
        timing.expert_stats,
        format_timing_ratio(timing.expert_stats, train_total),
    )
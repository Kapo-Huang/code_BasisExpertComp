import os
import logging
from pathlib import Path
from typing import Any, Dict

import torch


logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_checkpoint_multiview_attr_order(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("y_mean", "y_std"):
        mapping = payload.get(key)
        if isinstance(mapping, dict) and mapping:
            return [str(name) for name in mapping.keys()]
    return None


def warn_if_multiview_attr_order_mismatch(
    payload: Any,
    current_attr_names,
    *,
    context: str,
    logger_override: logging.Logger | None = None,
) -> bool:
    checkpoint_attr_names = get_checkpoint_multiview_attr_order(payload)
    current_names = [str(name) for name in current_attr_names]
    if checkpoint_attr_names is None or not current_names:
        return False
    if checkpoint_attr_names == current_names:
        return False
    if set(checkpoint_attr_names) != set(current_names):
        return False

    active_logger = logger_override or logger
    active_logger.warning(
        "Checkpoint multiview attr order differs from current config order at %s. "
        "This can silently swap view-conditioned outputs. checkpoint_order=%s current_order=%s. "
        "Check configs/config.yaml and the training-time attr_paths order.",
        context,
        checkpoint_attr_names,
        current_names,
    )
    return True


def save_checkpoint(
    model: torch.nn.Module,
    dataset,
    path: str,
    suffix: str = "",
    run_timestamp: str | None = None,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    save_path = path if suffix == "" else f"{path[:-4]}{suffix}.pth"
    ensure_dir(save_path)
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: v.numpy() for k, v in value.items()}
        return value.numpy()

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "x_mean": _to_numpy(getattr(dataset, "x_mean", None)),
        "x_std": _to_numpy(getattr(dataset, "x_std", None)),
        "y_mean": _to_numpy(getattr(dataset, "y_mean", None)),
        "y_std": _to_numpy(getattr(dataset, "y_std", None)),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if run_timestamp:
        payload["run_timestamp"] = run_timestamp
    torch.save(payload, save_path)
    logger.info("Saved checkpoint to %s", save_path)


def load_checkpoint(path: str, model: torch.nn.Module):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # PyTorch 2.6 defaults weights_only=True; we need full dict payload for our checkpoints.
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    return data

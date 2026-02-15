import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_checkpoint_path(path: str, suffix: str = "", run_timestamp: str = "") -> str:
    path_obj = Path(path)
    ext = path_obj.suffix or ".pth"
    stem = path_obj.stem if path_obj.suffix else path_obj.name
    run_tag = (run_timestamp or "").strip()
    if run_tag and run_tag not in stem:
        stem = f"{stem}_{run_tag}"
    if suffix:
        stem = f"{stem}{suffix}"
    return str(path_obj.with_name(f"{stem}{ext}"))


def save_checkpoint(
    model: torch.nn.Module,
    dataset,
    path: str,
    suffix: str = "",
    run_timestamp: str = "",
):
    save_path = _resolve_checkpoint_path(path, suffix=suffix, run_timestamp=run_timestamp)
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

import os
from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    dataset,
    path: str,
    suffix: str = "",
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra_payload: Dict[str, Any] | None = None,
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
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, save_path)
    print(f"Saved checkpoint to {save_path}")


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

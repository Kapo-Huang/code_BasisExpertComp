from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
NEURAL_EXPERTS_ROOT = THIS_DIR.parent
REPO_ROOT = NEURAL_EXPERTS_ROOT.parent


def ensure_sys_path() -> None:
    for path in (str(NEURAL_EXPERTS_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_path(path_str: str | None, config_dir: Path) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path.resolve())
    return str((config_dir / path).resolve())


def _deep_copy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(cfg))


def resolve_config_paths(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    cfg = _deep_copy_cfg(cfg)
    config_dir = Path(config_path).resolve().parent

    data_cfg = cfg.get("DATA", {})
    model_cfg = cfg.get("MODEL", {})
    train_cfg = cfg.get("TRAINING", {})
    pretrain_cfg = train_cfg.get("pretrain_assignment", {}) or {}

    for key in ("source_path", "target_path", "target_stats_path"):
        if data_cfg.get(key):
            data_cfg[key] = resolve_path(str(data_cfg[key]), config_dir)

    if model_cfg.get("manager_pt_path"):
        model_cfg["manager_pt_path"] = resolve_path(str(model_cfg["manager_pt_path"]), config_dir)

    if pretrain_cfg.get("cache_path"):
        pretrain_cfg["cache_path"] = resolve_path(str(pretrain_cfg["cache_path"]), config_dir)

    train_cfg["pretrain_assignment"] = pretrain_cfg
    cfg["DATA"] = data_cfg
    cfg["MODEL"] = model_cfg
    cfg["TRAINING"] = train_cfg
    return cfg


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_config_paths(cfg, path)


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def to_device(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def load_state_dict_payload(path: str | Path, device: torch.device | str = "cpu") -> Any:
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def load_checkpoint_payload(path: str | Path, device: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)

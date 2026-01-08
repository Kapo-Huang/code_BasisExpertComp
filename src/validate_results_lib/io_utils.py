import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_by_epoch(files, epoch: int):
    if not files:
        return None
    token = f"epoch{epoch}"
    for p in files:
        if token in p.name:
            return p
    return None


def match_checkpoint(epoch: int, ckpt_files):
    if not ckpt_files:
        return None
    token = f"epoch{epoch}"
    for ckpt in ckpt_files:
        if token in ckpt.name:
            return ckpt
    return None


def safe_load_npy(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(str(path), mmap_mode="r", allow_pickle=False)


def resolve_gt_path(cfg):
    data_cfg = cfg.get("data", {})
    y_path = data_cfg.get("y_path")
    if y_path:
        return Path(y_path).resolve() if not os.path.isabs(y_path) else Path(y_path)
    data_root = Path(data_cfg.get("data_root", "data"))
    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split", "train")
    if dataset_name is None:
        return None
    return (data_root / "raw" / dataset_name / split / "targets.npy").resolve()


def resolve_gt_paths(cfg):
    data_cfg = cfg.get("data", {})
    attr_paths = data_cfg.get("attr_paths")
    if attr_paths:
        resolved = {}
        for name, path in attr_paths.items():
            resolved[name] = Path(path).resolve() if not os.path.isabs(path) else Path(path)
        return resolved
    single = resolve_gt_path(cfg)
    return {"targets": single} if single is not None else {}


def pick_pred_for_attr(files, epoch: int, attr_name: str):
    if not files:
        return None
    token = f"epoch{epoch}"
    for p in files:
        if token in p.name and attr_name in p.name:
            return p
    return None


def collect_experiments(exp_root: Path):
    exps = []
    for child in sorted(exp_root.iterdir()):
        if not child.is_dir():
            continue
        cfg_path = child / "configs" / "config.yaml"
        if cfg_path.exists():
            exps.append(child)
    return exps


def append_csv_row(csv_path: Path, row, fieldnames):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def count_params_from_ckpt(path: Path):
    try:
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(str(path), map_location="cpu")
        state = data.get("model_state", {})
        return int(sum(v.numel() for v in state.values()))
    except Exception:
        return None

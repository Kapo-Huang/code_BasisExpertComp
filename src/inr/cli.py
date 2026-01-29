import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from inr.data import (
    MultiTargetVolumetricDataset,
    VolumetricDataset,
)
from inr.datasets.base import compute_target_stats_streaming
from inr.models.moe_inr import build_moe_inr_from_config
from inr.models.basisExpert_simple_concat import build_basisExpert_simple_concat_from_config
from inr.models.siren import build_siren_from_config
from inr.training.loops import PretrainConfig, TrainingConfig, train_model

def parse_args():
    p = argparse.ArgumentParser(description="Train Implicit Neural Representations (SIREN variants)")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda:0")
    return p.parse_args()

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(model_cfg, dataset=None):
    name = model_cfg["name"].lower()
    if name == "siren":
        return build_siren_from_config(model_cfg)
    if name in {"moe_inr", "moeinr", "moe-inr"}:
        return build_moe_inr_from_config(model_cfg)
    if name in {"basisexperts", "basis_experts", "basis-experts"}:
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExperts requires a MultiTargetVolumetricDataset with view_specs().")
        return build_basisExpert_simple_concat_from_config(model_cfg, dataset.view_specs())
    if name in {"basisexperts_attention", "basis_experts_attention", "basis-experts-attention"}:
        from inr.models.basisExperts_attention import build_basisExperts_attention_from_config
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExperts_attention requires a MultiTargetVolumetricDataset with view_specs().")
        return build_basisExperts_attention_from_config(model_cfg, dataset.view_specs())
    if name == "coordnet":
        from inr.models.CoordNet import build_coordnet_from_config
        return build_coordnet_from_config(model_cfg)
    raise ValueError(f"Unknown model name: {name}")


def resolve_data_paths(data_cfg):
    """
    Normalizes data paths to the volumetric structure:
    - raw data:   data/raw/<dataset>/<split>/targets.npy (single)
    - raw data:   data/raw/<dataset>/<split>/target_*.npy (multi-attr)
    - processed:  data/processed/<dataset>/<version>/<split>/targets.npy
    Explicit y_path/attr_paths override everything.
    """
    data_root = Path(data_cfg.get("data_root", "data"))
    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split", "train")
    processed_version = data_cfg.get("processed_version")
    volume_shape = data_cfg.get("volume_shape") or data_cfg.get("volume_dims")

    y_path = data_cfg.get("y_path") or data_cfg.get("target_path")
    attr_paths = data_cfg.get("attr_paths")
    target_dir = data_cfg.get("target_dir")
    attr_names = data_cfg.get("attr_names")
    attr_file_template = data_cfg.get("attr_file_template", "target_{name}.npy")
    attr_prefix = data_cfg.get("attr_prefix", "target_")

    if attr_paths is None and target_dir:
        target_root = Path(target_dir)
        if attr_names:
            attr_paths = {
                str(name): str(target_root / attr_file_template.format(name=name))
                for name in attr_names
            }
        else:
            files = sorted(target_root.glob("*.npy"))
            if not files:
                raise ValueError(f"No .npy files found in target_dir={target_dir}")
            attr_paths = {}
            for path in files:
                name = path.stem
                if attr_prefix and name.startswith(attr_prefix):
                    name = name[len(attr_prefix):]
                attr_paths[name] = str(path)

    if dataset_name and (y_path is None and not attr_paths):
        base_root = data_root / ("processed" if processed_version else "raw")
        base = base_root / dataset_name
        if processed_version:
            base = base / processed_version
        base = base / split
        y_path = str(base / "targets.npy")

    if y_path is None and not attr_paths:
        raise ValueError("y_path or attr_paths must be provided or inferrable.")

    return {
        "y_path": str(y_path) if y_path is not None else None,
        "attr_paths": attr_paths,
        "dataset_name": dataset_name,
        "split": split,
        "volume_shape": volume_shape,
    }


def _load_target_stats(stats_path: str, attr_names=None):
    stats = {}
    data = np.load(stats_path, allow_pickle=True)
    if attr_names:
        names = list(attr_names.keys()) if isinstance(attr_names, dict) else list(attr_names)
        for name in names:
            mean_key = f"{name}__mean"
            std_key = f"{name}__std"
            if mean_key in data and std_key in data:
                stats[name] = {
                    "mean": data[mean_key],
                    "std": data[std_key],
                }
            elif "mean" in data and "std" in data and len(names) == 1:
                stats[name] = {
                    "mean": data["mean"],
                    "std": data["std"],
                }
            else:
                available = list(data.keys())
                raise KeyError(f"Missing stats for '{name}' in {stats_path}. Keys: {available}")
    else:
        if "mean" in data and "std" in data:
            stats = {"mean": data["mean"], "std": data["std"]}
        else:
            mean_keys = [k for k in data.keys() if k.endswith("__mean")]
            std_keys = [k for k in data.keys() if k.endswith("__std")]
            if len(mean_keys) == 1 and len(std_keys) == 1:
                stats = {"mean": data[mean_keys[0]], "std": data[std_keys[0]]}
            else:
                available = list(data.keys())
                raise KeyError(f"Missing mean/std in {stats_path}. Keys: {available}")
    return stats


def _save_target_stats(stats_path: str, stats, attr_names=None):
    payload = {}
    if attr_names:
        names = list(attr_names.keys()) if isinstance(attr_names, dict) else list(attr_names)
        for name in names:
            payload[f"{name}__mean"] = np.asarray(stats[name]["mean"])
            payload[f"{name}__std"] = np.asarray(stats[name]["std"])
    else:
        payload["mean"] = np.asarray(stats["mean"])
        payload["std"] = np.asarray(stats["std"])
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, **payload)


def _compute_target_stats(y_path=None, attr_paths=None, eps: float = 1e-12):
    if y_path:
        y_np = np.load(y_path, mmap_mode="r")
        mean, std = compute_target_stats_streaming(y_np, eps=eps)
        return {"mean": mean.numpy(), "std": std.numpy()}
    stats = {}
    for name, path in attr_paths.items():
        arr = np.load(path, mmap_mode="r")
        mean, std = compute_target_stats_streaming(arr, eps=eps)
        stats[name] = {"mean": mean.numpy(), "std": std.numpy()}
    return stats


def build_experiment_layout(cfg, model_cfg, data_info):
    """
    Standardizes experiment outputs under:
    experiments/<exp_id>/{configs,checkpoints,predictions,logs}
    """
    exp_root = Path(cfg.get("experiment_root", "experiments"))
    exp_id = cfg.get("exp_id")
    if exp_id is None:
        dataset_token = data_info.get("dataset_name")
        if dataset_token is None:
            y_path = data_info.get("y_path")
            dataset_token = (
                Path(y_path).parent.name or Path(y_path).stem if y_path else "dataset"
            )
        exp_id = f"{model_cfg['name'].lower()}-{dataset_token}"

    exp_dir = exp_root / exp_id
    ckpt_path = exp_dir / "checkpoints" / f"{exp_id}.pth"
    pred_path = exp_dir / "predictions" / f"{exp_id}-full.npy"
    cfg_snapshot = exp_dir / "configs" / "config.yaml"

    for d in ["configs", "checkpoints", "predictions", "logs"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    # Persist a copy of the launched config for reproducibility
    cfg_snapshot.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return {
        "exp_dir": str(exp_dir),
        "exp_id": exp_id,
        "save_model": str(ckpt_path),
        "save_pred": str(pred_path),
        "config_snapshot": str(cfg_snapshot),
    }

def _format_num(num: int) -> str:
    return f"{num:,}"

def _model_size_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)

def _format_bytes_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def main():
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    print(f"Config load: {time.perf_counter() - t0:.2f}s")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg_raw = cfg["training"]

    t1 = time.perf_counter()
    data_info = resolve_data_paths(data_cfg)
    print(f"Resolve data paths: {time.perf_counter() - t1:.2f}s")
    exp_layout = build_experiment_layout(cfg, model_cfg, data_info)

    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))
    target_stats = None
    stats_path = data_cfg.get("target_stats_path")
    if stats_path and Path(stats_path).exists():
        print(f"Loading target stats: {stats_path}")
        target_stats = _load_target_stats(stats_path, attr_names=data_info.get("attr_paths"))
    elif stats_path and data_cfg.get("compute_target_stats", False):
        print(f"Computing target stats: {stats_path}")
        target_stats = _compute_target_stats(
            y_path=data_info.get("y_path"),
            attr_paths=data_info.get("attr_paths"),
            eps=float(data_cfg.get("stats_eps", 1e-12)),
        )
        if data_info.get("attr_paths"):
            _save_target_stats(stats_path, target_stats, attr_names=data_info.get("attr_paths"))
        else:
            _save_target_stats(stats_path, target_stats, attr_names=None)

    if data_info.get("attr_paths"):
        t2 = time.perf_counter()
        dataset = MultiTargetVolumetricDataset(
            data_info["attr_paths"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
        print(f"Dataset init (multi-target): {time.perf_counter() - t2:.2f}s")
    else:
        t2 = time.perf_counter()
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
        print(f"Dataset init (single-target): {time.perf_counter() - t2:.2f}s")
    print(f"Dataset size: {len(dataset)} samples")
    t3 = time.perf_counter()
    model = build_model(model_cfg, dataset)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = _model_size_bytes(model)
    print(
        "Model size: "
        f"params={_format_num(n_params)} "
        f"trainable={_format_num(n_trainable)} "
        f"size={_format_bytes_mb(size_bytes)}"
    )
    print(f"Model build: {time.perf_counter() - t3:.2f}s")

    pretrain_raw = train_cfg_raw.get("pretrain", {}) or {}
    pretrain_cfg = PretrainConfig(
        enabled=bool(pretrain_raw.get("enabled", False)),
        epochs=int(pretrain_raw.get("epochs", 0)),
        lr=float(pretrain_raw.get("lr", train_cfg_raw.get("lr", 5e-5))),
        batch_size=int(pretrain_raw.get("batch_size", train_cfg_raw.get("batch_size", 65536))),
        cluster_num_time_samples=int(pretrain_raw.get("cluster_num_time_samples", 16)),
        cluster_seed=int(pretrain_raw.get("cluster_seed", train_cfg_raw.get("seed", 42))),
        assignments_cache_path=str(pretrain_raw.get("assignments_cache_path", "")),
    )

    train_cfg = TrainingConfig(
        epochs=int(train_cfg_raw.get("epochs", 100)),
        batch_size=int(train_cfg_raw.get("batch_size", 65536)),
        pred_batch_size=int(train_cfg_raw.get("pred_batch_size", train_cfg_raw.get("batch_size", 65536))),
        num_workers=int(train_cfg_raw.get("num_workers", 4)),
        lr=float(train_cfg_raw.get("lr", 5e-5)),
        val_split=float(train_cfg_raw.get("val_split", 0.1)),
        log_every=int(train_cfg_raw.get("log_every", 4)),
        save_every=int(train_cfg_raw.get("save_every", 0)),
        early_stop_patience=int(train_cfg_raw.get("early_stop_patience", 0)),
        seed=int(train_cfg_raw.get("seed", 42)),
        save_model=train_cfg_raw.get("save_model", exp_layout["save_model"]),
        save_pred=train_cfg_raw.get("save_pred", exp_layout["save_pred"]),
        device=args.device,
        exp_dir=exp_layout["exp_dir"],
        exp_id=exp_layout["exp_id"],
        loss_type=str(train_cfg_raw.get("loss_type", "mse")),
        lam_eq=float(train_cfg_raw.get("lam_eq", 0.0)),
        gam_div=float(train_cfg_raw.get("gam_div", 0.0)),
        view_loss_weights=train_cfg_raw.get("view_loss_weights"),
        pretrain=pretrain_cfg,
    )

    print("Train start.")
    train_model(model, dataset, train_cfg)


if __name__ == "__main__":
    main()

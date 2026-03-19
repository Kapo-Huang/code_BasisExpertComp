import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")
os.environ.setdefault("MKL_NUM_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "64")

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from inr.data import (
    MultiTargetVolumetricDataset,
    VolumetricDataset,
)
from inr.datasets.base import compute_target_stats_streaming
from inr.models.baseline.base_moe_enc_view_add_dec_trunk import (
    build_base_moe_enc_view_add_dec_trunk_from_config,
)
from inr.models.baseline.base_shared_enc_view_add_shared_dec_trunk import (
    build_base_shared_enc_view_add_shared_dec_trunk_from_config,
)
from inr.models.baseline.base_shared_enc_view_attention_fused_dec_trunk import (
    build_base_shared_enc_view_attention_fused_dec_trunk_from_config,
)
from inr.models.basis_expert.experts_attention import build_basisExperts_attention_from_config
from inr.models.basis_expert.experts_attention_light_pe import (
    build_basisExperts_attention_light_pe_from_config,
)
from inr.models.basis_expert.light_basis_expert import build_light_basis_expert_from_config
from inr.models.basis_expert.simple import build_basisExpert_simple_concat_from_config
from inr.models.sota.coordnet import build_coordnet_from_config
from inr.models.sota.moe_inr import build_moe_inr_from_config
from inr.models.sota.neural_expert import build_neural_expert_from_config
from inr.models.sota.siren import build_siren_from_config
from inr.models.sota.stsr_inr import (
    build_stsr_inr_from_config,
    build_stsr_inr_multiview_from_config,
)
from inr.training.loops import (
    GradientBalancerConfig,
    GradientDiagConfig,
    MultiAttrEMALossConfig,
    PretrainConfig,
    TimeStepCurriculumConfig,
    TrainingConfig,
    train_model,
)
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Train Implicit Neural Representations (SIREN variants)")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda:0")
    return p.parse_args()

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_model_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _require_view_specs(dataset, model_name: str):
    if dataset is None or not hasattr(dataset, "view_specs"):
        raise ValueError(f"{model_name} requires a MultiTargetVolumetricDataset with view_specs().")
    return dataset.view_specs()


def build_model(model_cfg, dataset=None):
    name_raw = model_cfg["name"]
    name = _normalize_model_name(name_raw)

    if name == "siren":
        return build_siren_from_config(model_cfg)
    if name in {"moe_inr", "moeinr"}:
        return build_moe_inr_from_config(model_cfg)
    if name in {"neural_expert", "neuralexpert"}:
        return build_neural_expert_from_config(model_cfg)
    if name in {"coordnet", "coord_net"}:
        return build_coordnet_from_config(model_cfg)
    if name in {"stsr_inr", "stsrinr"}:
        return build_stsr_inr_from_config(model_cfg)
    if name in {"stsr_inr_multiview", "stsrinr_multiview"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_stsr_inr_multiview_from_config(model_cfg, view_specs)

    if name in {"basis_experts", "basisexperts", "basis_expert"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_basisExpert_simple_concat_from_config(model_cfg, view_specs)
    if name in {"basis_experts_attention", "basisexperts_attention"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_basisExperts_attention_from_config(model_cfg, view_specs)
    if name in {"basis_experts_attention_light_pe", "basisexperts_attention_light_pe"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_basisExperts_attention_light_pe_from_config(model_cfg, view_specs)
    if name in {"light_basis_expert", "lightbasis_expert", "light_basisexperts"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_light_basis_expert_from_config(model_cfg, view_specs)

    if name in {
        "base_shared_enc_view_add_shared_dec_trunk",
        "baseline_shared_enc_view_add_shared_dec_trunk",
    }:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_base_shared_enc_view_add_shared_dec_trunk_from_config(model_cfg, view_specs)
    if name in {
        "base_shared_enc_view_attention_fused_dec_trunk",
        "baseline_shared_enc_view_attention_fused_dec_trunk",
    }:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_base_shared_enc_view_attention_fused_dec_trunk_from_config(model_cfg, view_specs)
    if name in {"base_moe_enc_view_add_dec_trunk", "baseline_moe_enc_view_add_dec_trunk"}:
        view_specs = _require_view_specs(dataset, name_raw)
        return build_base_moe_enc_view_add_dec_trunk_from_config(model_cfg, view_specs)

    raise ValueError(f"Unknown model name: {name_raw}")


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

def _weight_bias_numel(model) -> int:
    total = 0
    for name, p in model.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            total += p.numel()
    return int(total)

def _model_size_bytes(model) -> int:
    fp16_bytes = 2
    return _weight_bias_numel(model) * fp16_bytes

def _format_bytes_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def main():
    setup_logging()
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    logger.info("Config load: %.2fs", time.perf_counter() - t0)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg_raw = cfg["training"]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t1 = time.perf_counter()
    data_info = resolve_data_paths(data_cfg)
    logger.info("Resolve data paths: %.2fs", time.perf_counter() - t1)
    exp_layout = build_experiment_layout(cfg, model_cfg, data_info)
    setup_logging(log_dir=Path(exp_layout["exp_dir"]) / "logs", run_timestamp=run_timestamp)

    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))
    target_stats = None
    stats_path = data_cfg.get("target_stats_path")
    if stats_path and Path(stats_path).exists():
        logger.info("Loading target stats: %s", stats_path)
        target_stats = _load_target_stats(stats_path, attr_names=data_info.get("attr_paths"))
    elif stats_path and data_cfg.get("compute_target_stats", False):
        logger.info("Computing target stats: %s", stats_path)
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
        logger.info("Dataset init (multi-target): %.2fs", time.perf_counter() - t2)
    else:
        t2 = time.perf_counter()
        dataset = VolumetricDataset(
            data_info["y_path"],
            volume_shape=data_info.get("volume_shape"),
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            target_stats=target_stats,
        )
        logger.info("Dataset init (single-target): %.2fs", time.perf_counter() - t2)
    logger.info("Dataset size: %s samples", len(dataset))
    t3 = time.perf_counter()
    model = build_model(model_cfg, dataset)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = _model_size_bytes(model)
    logger.info(
        "Model size: params=%s trainable=%s size(fp16, weights+bias)=%s",
        _format_num(n_params),
        _format_num(n_trainable),
        _format_bytes_mb(size_bytes),
    )
    logger.info("Model build: %.2fs", time.perf_counter() - t3)

    pretrain_raw = train_cfg_raw.get("pretrain", {}) or {}
    pretrain_cfg = PretrainConfig(
        enabled=bool(pretrain_raw.get("enabled", False)),
        epochs=int(pretrain_raw.get("epochs", 0)),
        lr=float(pretrain_raw.get("lr", train_cfg_raw.get("lr", 5e-5))),
        batch_size=int(pretrain_raw.get("batch_size", train_cfg_raw.get("batch_size", 65536))),
        cluster_num_time_samples=int(pretrain_raw.get("cluster_num_time_samples", 16)),
        cluster_seed=int(pretrain_raw.get("cluster_seed", train_cfg_raw.get("seed", 42))),
        assignments_cache_path=str(pretrain_raw.get("assignments_cache_path", "")),
        assignments_method=str(pretrain_raw.get("assignments_method", "voxel_clustering")),
        spatial_blocks=tuple(pretrain_raw.get("spatial_blocks", [])) or None,
        time_block_size=int(pretrain_raw.get("time_block_size", 0)),
    )

    timestep_curriculum_raw = train_cfg_raw.get("timestep_curriculum", {}) or {}
    timestep_curriculum_cfg = TimeStepCurriculumConfig(
        enabled=bool(timestep_curriculum_raw.get("enabled", False)),
        mode=str(timestep_curriculum_raw.get("mode", "linear")),
        start_timesteps=int(timestep_curriculum_raw.get("start_timesteps", 0)),
        end_timesteps=int(timestep_curriculum_raw.get("end_timesteps", 0)),
        warmup_epochs=int(timestep_curriculum_raw.get("warmup_epochs", 0)),
        ramp_epochs=int(timestep_curriculum_raw.get("ramp_epochs", 0)),
        stride_groups=int(timestep_curriculum_raw.get("stride_groups", 0)),
        epochs_per_group=int(timestep_curriculum_raw.get("epochs_per_group", 0)),
    )
    gradient_balancer_raw = train_cfg_raw.get("gradient_balancer", {}) or {}
    gradient_balancer_cfg = GradientBalancerConfig(
        enabled=bool(gradient_balancer_raw.get("enabled", False)),
        method=str(gradient_balancer_raw.get("method", "pcgrad")),
        cagrad_c=float(gradient_balancer_raw.get("cagrad_c", 0.4)),
        solver_max_iter=int(gradient_balancer_raw.get("solver_max_iter", 50)),
        solver_lr=float(gradient_balancer_raw.get("solver_lr", 0.25)),
        gradnorm_alpha=float(gradient_balancer_raw.get("gradnorm_alpha", 0.5)),
        gradnorm_lr=float(gradient_balancer_raw.get("gradnorm_lr", 1.0e-3)),
        gradnorm_every_n_steps=int(gradient_balancer_raw.get("gradnorm_every_n_steps", 100)),
    )
    gradient_diag_raw = train_cfg_raw.get("gradient_diag", {}) or {}
    gradient_diag_cfg = GradientDiagConfig(
        enabled=bool(gradient_diag_raw.get("enabled", False)),
        every_n_steps=int(gradient_diag_raw.get("every_n_steps", 200)),
        max_layers_to_log=int(gradient_diag_raw.get("max_layers_to_log", 10)),
    )
    multiview_ema_raw = train_cfg_raw.get("multiview_ema_loss", {}) or {}
    multiview_ema_cfg = MultiAttrEMALossConfig(
        enabled=bool(multiview_ema_raw.get("enabled", False)),
        beta=float(multiview_ema_raw.get("beta", 0.95)),
        eps=float(multiview_ema_raw.get("eps", 1e-8)),
        w_min=float(multiview_ema_raw.get("w_min", 0.2)),
        w_max=float(multiview_ema_raw.get("w_max", 5.0)),
        warmup_steps=int(multiview_ema_raw.get("warmup_steps", 0)),
    )
    train_cfg = TrainingConfig(
        epochs=int(train_cfg_raw.get("epochs", 100)),
        batch_size=int(train_cfg_raw.get("batch_size", 65536)),
        pred_batch_size=int(train_cfg_raw.get("pred_batch_size", train_cfg_raw.get("batch_size", 65536))),
        num_workers=int(train_cfg_raw.get("num_workers", 4)),
        batches_per_epoch_budget=int(train_cfg_raw.get("batches_per_epoch_budget", 0)),
        lr=float(train_cfg_raw.get("lr", 5e-5)),
        val_split=float(train_cfg_raw.get("val_split", 0.1)),
        log_every=int(train_cfg_raw.get("log_every", 4)),
        log_psnr_every=int(train_cfg_raw.get("log_psnr_every", train_cfg_raw.get("log_PSNR_every", 5))),
        psnr_sample_ratio=float(train_cfg_raw.get("psnr_sample_ratio", 1.0)),
        save_every=int(train_cfg_raw.get("save_every", 0)),
        early_stop_patience=int(train_cfg_raw.get("early_stop_patience", 0)),
        seed=int(train_cfg_raw.get("seed", 42)),
        save_model=train_cfg_raw.get("save_model", exp_layout["save_model"]),
        save_pred=train_cfg_raw.get("save_pred", exp_layout["save_pred"]),
        device=args.device,
        exp_dir=exp_layout["exp_dir"],
        exp_id=exp_layout["exp_id"],
        run_timestamp=run_timestamp,
        loss_type=str(train_cfg_raw.get("loss_type", "mse")),
        view_loss_weights=train_cfg_raw.get("view_loss_weights"),
        pretrain=pretrain_cfg,
        timestep_curriculum=timestep_curriculum_cfg,
        lr_decay_rate=float(train_cfg_raw.get("lr_decay_rate", 0.0)),
        lr_decay_step=int(train_cfg_raw.get("lr_decay_step", 0)),
        freeze_router_at=float(train_cfg_raw.get("freeze_router_at", 0.8)),
        hard_topk_warmup_epochs=int(train_cfg_raw.get("hard_topk_warmup_epochs", 0)),
        gradient_balancer=gradient_balancer_cfg,
        gradient_diag=gradient_diag_cfg,
        multiview_ema_loss=multiview_ema_cfg,
    )

    logger.info("Training config:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    logger.info("Train start.")
    train_model(model, dataset, train_cfg)


if __name__ == "__main__":
    main()

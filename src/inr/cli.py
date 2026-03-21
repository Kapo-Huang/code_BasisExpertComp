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

import yaml

from inr.data import MultiViewCoordDataset, NodeDataset
from inr.models.basis_expert.light_basis_expert import build_light_basis_expert_from_config
from inr.models.basis_expert.simple import build_basisExpert_simple_concat_from_config
from inr.models.sota.moe_inr import build_moe_inr_from_config
from inr.models.sota.siren import build_siren_from_config
from inr.training.pretrain import PretrainConfig
from inr.training.loops import (
    MultiAttrEMALossConfig,
    TimeStepCurriculumConfig,
    TrainingConfig,
    train_model,
)
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent

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
        raise ValueError(f"{model_name} requires a MultiViewCoordDataset with view_specs().")
    return dataset.view_specs()


def _resolve_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    candidates = [
        Path(path_value),
        REPO_ROOT / path,
        SRC_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if path.parts and (SRC_ROOT / path.parts[0]).exists():
        return str(SRC_ROOT / path)
    return str(REPO_ROOT / path)


def _resolve_mapping_paths(path_mapping):
    if not path_mapping:
        return path_mapping
    return {str(name): _resolve_path(path) for name, path in path_mapping.items()}


def build_model(model_cfg, dataset=None):
    name_raw = model_cfg["name"]
    name = _normalize_model_name(name_raw)
    if name == "siren":
        return build_siren_from_config(model_cfg)
    if name in {"moe_inr", "moeinr"}:
        return build_moe_inr_from_config(model_cfg)
    if name in {"basisexpert_simple_concat", "basis_expert", "basis_experts", "basisexperts"}:
        return build_basisExpert_simple_concat_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"basisexperts_attention", "basis_experts_attention"}:
        from inr.models.basis_expert.experts_attention import build_basisExperts_attention_from_config

        return build_basisExperts_attention_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"basisexperts_attention_light_pe", "basis_experts_attention_light_pe", "basisexperts_attention_lightpe"}:
        from inr.models.basis_expert.experts_attention_light_pe import (
            build_basisExperts_attention_light_pe_from_config,
        )

        return build_basisExperts_attention_light_pe_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"lightbasisexpert", "light_basis_expert", "light_basisexperts", "light_basis_expert_pe"}:
        return build_light_basis_expert_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"stsr_inr", "stsrinr"}:
        from inr.models.sota.stsr_inr import (
            build_stsr_inr_from_config,
            build_stsr_inr_multiview_from_config,
        )
        if dataset is not None and hasattr(dataset, "view_specs"):
            return build_stsr_inr_multiview_from_config(model_cfg, dataset.view_specs())
        return build_stsr_inr_from_config(model_cfg)
    if name in {"base_shared_enc_view_add_shared_dec_trunk", "baseline_shared_enc_view_add_shared_dec_trunk", "sharedenc_viewadd"}:
        from inr.models.baseline.base_shared_enc_view_add_shared_dec_trunk import (
            build_base_shared_enc_view_add_shared_dec_trunk_from_config,
        )

        return build_base_shared_enc_view_add_shared_dec_trunk_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"base_shared_enc_view_attention_fused_dec_trunk", "baseline_shared_enc_view_attention_fused_dec_trunk", "sharedenc_viewattn"}:
        from inr.models.baseline.base_shared_enc_view_attention_fused_dec_trunk import (
            build_base_shared_enc_view_attention_fused_dec_trunk_from_config,
        )

        return build_base_shared_enc_view_attention_fused_dec_trunk_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"base_moe_enc_view_add_dec_trunk", "baseline_moe_enc_view_add_dec_trunk", "moeenc_viewadd"}:
        from inr.models.baseline.base_moe_enc_view_add_dec_trunk import (
            build_base_moe_enc_view_add_dec_trunk_from_config,
        )

        return build_base_moe_enc_view_add_dec_trunk_from_config(model_cfg, _require_view_specs(dataset, name_raw))
    if name in {"coordnet", "coord_net"}:
        from inr.models.sota.coordnet import build_coordnet_from_config

        return build_coordnet_from_config(model_cfg)
    raise ValueError(f"Unknown model name: {name_raw}")

def resolve_data_paths(data_cfg):
    """
    Normalizes data paths to the new structure:
    - raw data:   data/raw/<dataset>/<split>/coords.npy, targets.npy
    - processed:  data/processed/<dataset>/<version>/<split>/coords.npy, targets.npy
    Explicit x_path/y_path override everything.
    """
    data_root = Path(_resolve_path(str(data_cfg.get("data_root", "data"))))
    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split", "train")
    processed_version = data_cfg.get("processed_version")

    x_path = _resolve_path(data_cfg.get("x_path"))
    y_path = _resolve_path(data_cfg.get("y_path"))
    attr_paths = _resolve_mapping_paths(data_cfg.get("attr_paths"))
    if dataset_name and (x_path is None or (y_path is None and not attr_paths)):
        base_root = data_root / ("processed" if processed_version else "raw")
        base = base_root / dataset_name
        if processed_version:
            base = base / processed_version
        base = base / split
        x_path = x_path or _resolve_path(str(base / "coords.npy"))
        if not attr_paths:
            y_path = y_path or _resolve_path(str(base / "targets.npy"))

    if x_path is None or (y_path is None and not attr_paths):
        raise ValueError("x_path and y_path (or attr_paths) must be provided or inferrable.")

    return {
        "x_path": _resolve_path(str(x_path)),
        "y_path": _resolve_path(str(y_path)) if y_path is not None else None,
        "attr_paths": attr_paths,
        "dataset_name": dataset_name,
        "split": split,
    }


def build_experiment_layout(cfg, model_cfg, data_info, run_timestamp: str):
    """
    Standardizes experiment outputs under:
    experiments/<exp_id>/{configs,checkpoints,predictions,logs}
    """
    exp_root = Path(cfg.get("experiment_root", "experiments"))
    exp_id = cfg.get("exp_id")
    if exp_id is None:
        dataset_token = data_info.get("dataset_name") or Path(data_info["x_path"]).parent.name or Path(data_info["x_path"]).stem
        exp_id = f"{model_cfg['name'].lower()}-{dataset_token}"

    exp_dir = exp_root / exp_id
    ckpt_path = exp_dir / "checkpoints" / f"{exp_id}_{run_timestamp}.pth"
    pred_path = exp_dir / "predictions" / f"{exp_id}-full.npy"
    cfg_snapshot = exp_dir / "configs" / f"config_{run_timestamp}.yaml"
    cfg_snapshot_legacy = exp_dir / "configs" / "config.yaml"

    for d in ["configs", "checkpoints", "predictions", "logs"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    # Persist a copy of the launched config for reproducibility.
    cfg_snapshot_payload = dict(cfg)
    cfg_snapshot_payload["run_timestamp"] = run_timestamp
    cfg_text = yaml.safe_dump(cfg_snapshot_payload)
    cfg_snapshot.write_text(cfg_text, encoding="utf-8")
    # Keep legacy path for downstream scripts that still read configs/config.yaml.
    cfg_snapshot_legacy.write_text(cfg_text, encoding="utf-8")

    return {
        "exp_dir": str(exp_dir),
        "exp_id": exp_id,
        "save_model": str(ckpt_path),
        "save_pred": str(pred_path),
        "config_snapshot": str(cfg_snapshot),
        "config_snapshot_legacy": str(cfg_snapshot_legacy),
    }


def _format_num(num: int) -> str:
    return f"{num:,}"


def _weight_bias_numel(model) -> int:
    total = 0
    for name, param in model.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            total += param.numel()
    return int(total)


def _model_size_bytes(model) -> int:
    return _weight_bias_numel(model) * 2


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
    exp_layout = build_experiment_layout(cfg, model_cfg, data_info, run_timestamp)
    setup_logging(log_dir=Path(exp_layout["exp_dir"]) / "logs", run_timestamp=run_timestamp)

    stats_path = _resolve_path(
        data_cfg.get("target_stats_path")
        or data_cfg.get("stats_path")
        or data_cfg.get("normalization_stats_path")
    )
    normalize = bool(data_cfg.get("normalize", True))

    if data_info.get("attr_paths"):
        t2 = time.perf_counter()
        dataset = MultiViewCoordDataset(
            data_info["x_path"],
            data_info["attr_paths"],
            normalize=normalize,
            stats_path=stats_path,
        )
        logger.info("Dataset init (multi-target): %.2fs", time.perf_counter() - t2)
    else:
        t2 = time.perf_counter()
        dataset = NodeDataset(
            data_info["x_path"],
            data_info["y_path"],
            normalize=normalize,
            stats_path=stats_path,
        )
        logger.info("Dataset init (single-target): %.2fs", time.perf_counter() - t2)
    logger.info("Dataset size: %s samples", len(dataset))

    t3 = time.perf_counter()
    model = build_model(model_cfg, dataset)
    n_params = sum(param.numel() for param in model.parameters())
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(
        "Model size: params=%s trainable=%s size(fp16, weights+bias)=%s",
        _format_num(n_params),
        _format_num(n_trainable),
        _format_bytes_mb(_model_size_bytes(model)),
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
        assignments_cache_path=str(_resolve_path(pretrain_raw.get("assignments_cache_path", "")) or ""),
        assignments_method=str(pretrain_raw.get("assignments_method", "voxel_clustering")),
        spatial_blocks=tuple(pretrain_raw.get("spatial_blocks", [])) or None,
        time_block_size=int(pretrain_raw.get("time_block_size", 0)),
        mode=str(pretrain_raw.get("mode", "router_classification")),
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
        val_split=float(train_cfg_raw.get("val_split", 0.0)),
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
        multiview_recon_reduction=str(train_cfg_raw.get("multiview_recon_reduction", "attr_sum")),
        multiview_ema_loss=multiview_ema_cfg,
        resume_path=train_cfg_raw.get("resume_path"),
        log_time_breakdown=bool(train_cfg_raw.get("log_time_breakdown", True)),
        time_breakdown_cuda_sync=bool(train_cfg_raw.get("time_breakdown_cuda_sync", False)),
    )

    logger.info("Training config:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    logger.info("Train start.")
    train_model(model, dataset, train_cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _bootstrap_repo_root() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    for path in (repo_root, package_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_repo_root()

import yaml

from inr.cli import (
    _compute_target_stats,
    _load_target_stats,
    _save_target_stats,
    build_experiment_layout,
    build_model,
    resolve_data_paths,
)
from inr.data import VolumetricDataset
from inr.training.loops import PretrainConfig, TimeStepCurriculumConfig, TrainingConfig, train_model
from inr.utils.io import _resolve_checkpoint_path
from inr.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Switch-NeRF as a single-target INR on ionization data."
    )
    parser.add_argument(
        "--config",
        "--config_file",
        dest="config",
        required=True,
        help="Path to the Switch-NeRF INR yaml config.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional override for experiment id or experiment directory path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional override for ionization dataset root. Both data/raw/ionization and data/raw/ionization/train are accepted.",
    )
    parser.add_argument(
        "--target_attr",
        type=str,
        default="",
        help="Optional override for single target attribute, e.g. GT / H+ / H2 / He / PD.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="",
        help="Optional override for the concrete single-target .npy file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, e.g. cpu / cuda / cuda:0.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training.epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for training.batch_size.")
    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=None,
        help="Optional override for training.pred_batch_size.",
    )
    parser.add_argument(
        "--batches_per_epoch_budget",
        type=int,
        default=None,
        help="Optional override for training.batches_per_epoch_budget.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for training.num_workers.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Optional override for training.save_every.",
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="Disable router pretraining regardless of the yaml config.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a yaml mapping: {path}")
    return data


def _resolve_path(path_value: str | None, *, base: Path = REPO_ROOT) -> str | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _resolve_dataset_dir(dataset_path: str, split: str) -> str:
    resolved = Path(_resolve_path(dataset_path) or "")
    if not resolved:
        raise ValueError("dataset_path must not be empty")
    candidate = resolved / split
    if candidate.is_dir():
        return str(candidate)
    return str(resolved)


def _find_moe_layer_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    layers = model_cfg.get("layers") or {}
    if not isinstance(layers, dict):
        raise ValueError("model.layers must be a yaml mapping")

    if "0" in layers:
        return layers["0"]

    for key, value in layers.items():
        if isinstance(value, dict) and str(value.get("type", "")).strip().lower() == "moe":
            return value
    raise KeyError("No MoE layer found in model.layers")


def _resolve_exp_name(exp_name: str, cfg: Dict[str, Any]) -> Tuple[str | None, str | None]:
    text = str(exp_name or "").strip()
    if not text:
        return None, None

    candidate = Path(text)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        resolved = Path(_resolve_path(text) or "")
        return str(resolved.parent), resolved.name
    return None, text


def _convert_switch_nerf_config(raw_cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_cfg = copy.deepcopy(raw_cfg)
    data_cfg = dict(source_cfg.get("data") or {})
    training_raw = dict(source_cfg.get("training") or {})
    model_cfg = dict(source_cfg.get("model") or {})

    split = str(source_cfg.get("split", data_cfg.get("split", "train"))).strip() or "train"
    dataset_root_cfg = args.dataset_path or source_cfg.get("dataset_path") or "./data/raw/ionization"
    dataset_dir = _resolve_dataset_dir(str(dataset_root_cfg), split)

    attr_from_cfg = str(
        source_cfg.get("target_attr")
        or data_cfg.get("target_attr")
        or source_cfg.get("attr_name")
        or ""
    ).strip()
    target_attr = str(args.target_attr or attr_from_cfg).strip()
    target_path_cfg = data_cfg.get("target_path") or source_cfg.get("target_path")
    stats_path_cfg = data_cfg.get("target_stats_path") or source_cfg.get("target_stats_path")

    if args.target_path:
        target_path = _resolve_path(args.target_path)
    elif args.target_attr or not target_path_cfg:
        if not target_attr:
            raise ValueError("target_attr is required when target_path is not explicitly provided.")
        target_path = str(Path(dataset_dir) / f"target_{target_attr}.npy")
    else:
        target_path = _resolve_path(str(target_path_cfg))

    if args.target_attr or not stats_path_cfg:
        if not target_attr:
            raise ValueError("target_attr is required when target_stats_path is not explicitly provided.")
        target_stats_path = str(Path(dataset_dir) / f"target_stats_{target_attr}.npz")
    else:
        target_stats_path = _resolve_path(str(stats_path_cfg))

    if not target_path or not Path(target_path).exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    volume_shape = data_cfg.get("volume_shape") or source_cfg.get("volume_shape")
    if not volume_shape:
        raise ValueError("data.volume_shape is required for ionization INR training.")

    xyz_cfg = (model_cfg.get("layers") or {}).get("xyz") or {}
    moe_layer_cfg = _find_moe_layer_cfg(model_cfg)
    gate_cfg = (model_cfg.get("layers") or {}).get("moe_external_gate") or {}

    hidden_dim = _as_int(xyz_cfg.get("out_ch"), 256)
    coord_pos_freqs = _as_int(source_cfg.get("pos_xyz_dim"), 12)
    num_experts = _as_int(source_cfg.get("moe_expert_num"), 8)
    use_external_gate = bool(source_cfg.get("use_moe_external_gate", "moe_external_gate" in (model_cfg.get("layers") or {})))
    use_gate_input_norm = bool(source_cfg.get("use_gate_input_norm", "gate_input_norm" in (model_cfg.get("layers") or {})))
    router_noise_std = max(_as_float(source_cfg.get("gate_noise"), 0.0), 0.0)

    budget = _as_int(training_raw.get("batches_per_epoch_budget"), 0)
    epochs = training_raw.get("epochs")
    if epochs is None:
        train_iterations = _as_int(training_raw.get("train_iterations"), 0)
        if train_iterations > 0 and budget > 0:
            epochs = int(math.ceil(train_iterations / float(budget)))
        else:
            epochs = 600

    save_every = _as_int(training_raw.get("save_every"), 0)
    if save_every <= 0:
        ckpt_interval = _as_int(training_raw.get("ckpt_interval"), 0)
        if ckpt_interval > 0 and budget > 0:
            save_every = max(1, int(math.ceil(ckpt_interval / float(budget))))

    experiment_root = _resolve_path(str(source_cfg.get("experiment_root") or "./result/Ionization/Switch-NeRF"))
    exp_id = str(source_cfg.get("exp_id") or f"switch_nerf-ionization-{target_attr or 'scalar'}").strip()
    exp_root_override, exp_id_override = _resolve_exp_name(args.exp_name, source_cfg)
    if exp_root_override:
        experiment_root = exp_root_override
    if exp_id_override:
        exp_id = exp_id_override

    converted_cfg = {
        "experiment": "switch_nerf_inr",
        "experiment_root": experiment_root,
        "exp_id": exp_id,
        "data": {
            "dataset_name": "ionization",
            "split": split,
            "target_path": target_path,
            "target_stats_path": target_stats_path,
            "compute_target_stats": bool(data_cfg.get("compute_target_stats", True)),
            "volume_shape": volume_shape,
            "normalize": bool(data_cfg.get("normalize", True)),
        },
        "model": {
            "name": "switch_nerf",
            "in_features": 4,
            "out_features": 1,
            "num_experts": num_experts,
            "coord_pos_freqs": coord_pos_freqs,
            "hidden_dim": hidden_dim,
            "moe_layer_num": 1,
            "expert_layer_num": _as_int(moe_layer_cfg.get("num"), 7),
            "expert_skips": list(moe_layer_cfg.get("skips", [3])),
            "gate_hidden_dim": _as_int(gate_cfg.get("out_ch"), hidden_dim),
            "gate_layer_num": _as_int(gate_cfg.get("num"), 2),
            "top_k": _as_int(moe_layer_cfg.get("k"), 1),
            "use_external_gate": use_external_gate,
            "use_gate_input_norm": use_gate_input_norm,
            "router_temperature": 1.0,
            "router_noise_std": router_noise_std,
            "balance_loss_weight": _as_float(source_cfg.get("moe_l_aux_wt"), 5.0e-4),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": _as_int(training_raw.get("batch_size"), 16000),
            "pred_batch_size": _as_int(training_raw.get("pred_batch_size"), _as_int(training_raw.get("batch_size"), 16000)),
            "num_workers": _as_int(training_raw.get("num_workers"), 2),
            "batches_per_epoch_budget": budget,
            "lr": _as_float(training_raw.get("lr"), 5.0e-5),
            "val_split": _as_float(training_raw.get("val_split"), 0.0),
            "log_every": _as_int(training_raw.get("log_every"), 1),
            "log_psnr_every": _as_int(training_raw.get("log_psnr_every"), 100),
            "psnr_sample_ratio": _as_float(training_raw.get("psnr_sample_ratio"), 0.1),
            "save_every": save_every,
            "early_stop_patience": _as_int(training_raw.get("early_stop_patience"), 0),
            "seed": _as_int(training_raw.get("seed"), _as_int(source_cfg.get("random_seed"), 42)),
            "loss_type": str(training_raw.get("loss_type", "mse")),
            "lr_decay_rate": _as_float(training_raw.get("lr_decay_rate"), 0.0),
            "lr_decay_step": _as_int(training_raw.get("lr_decay_step"), 0),
            "freeze_router_at": _as_float(training_raw.get("freeze_router_at"), 0.8),
            "hard_topk_warmup_epochs": _as_int(training_raw.get("hard_topk_warmup_epochs"), 0),
            "pretrain": copy.deepcopy(training_raw.get("pretrain") or {}),
            "timestep_curriculum": copy.deepcopy(training_raw.get("timestep_curriculum") or {}),
        },
    }

    if args.epochs is not None:
        converted_cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        converted_cfg["training"]["batch_size"] = int(args.batch_size)
    if args.pred_batch_size is not None:
        converted_cfg["training"]["pred_batch_size"] = int(args.pred_batch_size)
    if args.batches_per_epoch_budget is not None:
        converted_cfg["training"]["batches_per_epoch_budget"] = int(args.batches_per_epoch_budget)
    if args.num_workers is not None:
        converted_cfg["training"]["num_workers"] = int(args.num_workers)
    if args.save_every is not None:
        converted_cfg["training"]["save_every"] = int(args.save_every)
    if args.no_pretrain:
        converted_cfg["training"].setdefault("pretrain", {})
        converted_cfg["training"]["pretrain"]["enabled"] = False
        converted_cfg["training"]["pretrain"]["epochs"] = 0

    resolved_source = copy.deepcopy(source_cfg)
    resolved_source["dataset_path"] = dataset_dir
    resolved_source["split"] = split
    resolved_source["target_attr"] = target_attr
    resolved_source.setdefault("data", {})
    resolved_source["data"] = data_cfg
    resolved_source["data"]["target_path"] = target_path
    resolved_source["data"]["target_stats_path"] = target_stats_path
    resolved_source["data"]["volume_shape"] = volume_shape
    resolved_source["experiment_root"] = experiment_root
    resolved_source["exp_id"] = exp_id

    return converted_cfg, resolved_source


def _build_dataset(data_cfg: Dict[str, Any]) -> Tuple[VolumetricDataset, Dict[str, Any]]:
    data_info = resolve_data_paths(data_cfg)
    if data_info.get("attr_paths"):
        raise ValueError("Switch-NeRF INR trainer only supports a single target_path.")

    normalize_inputs = bool(data_cfg.get("normalize_inputs", data_cfg.get("normalize", True)))
    normalize_targets = bool(data_cfg.get("normalize_targets", data_cfg.get("normalize", True)))
    target_stats = None
    stats_path = data_cfg.get("target_stats_path")
    if stats_path and Path(stats_path).exists():
        logger.info("Loading target stats: %s", stats_path)
        target_stats = _load_target_stats(stats_path)
    elif stats_path and data_cfg.get("compute_target_stats", False):
        logger.info("Computing target stats: %s", stats_path)
        target_stats = _compute_target_stats(
            y_path=data_info.get("y_path"),
            attr_paths=None,
            eps=float(data_cfg.get("stats_eps", 1.0e-12)),
        )
        _save_target_stats(stats_path, target_stats, attr_names=None)

    dataset = VolumetricDataset(
        data_info["y_path"],
        volume_shape=data_info.get("volume_shape"),
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
        target_stats=target_stats,
    )
    return dataset, data_info


def _build_training_cfg(
    cfg: Dict[str, Any],
    exp_layout: Dict[str, str],
    run_timestamp: str,
    device: str | None,
) -> TrainingConfig:
    training_raw = cfg["training"]
    pretrain_raw = training_raw.get("pretrain", {}) or {}
    timestep_curriculum_raw = training_raw.get("timestep_curriculum", {}) or {}

    pretrain_cfg = PretrainConfig(
        enabled=bool(pretrain_raw.get("enabled", False)),
        epochs=_as_int(pretrain_raw.get("epochs"), 0),
        lr=_as_float(pretrain_raw.get("lr"), training_raw.get("lr", 5.0e-5)),
        batch_size=_as_int(pretrain_raw.get("batch_size"), training_raw.get("batch_size", 16000)),
        cluster_num_time_samples=_as_int(pretrain_raw.get("cluster_num_time_samples"), 16),
        cluster_seed=_as_int(pretrain_raw.get("cluster_seed"), training_raw.get("seed", 42)),
        assignments_cache_path=_resolve_path(pretrain_raw.get("assignments_cache_path")),
        assignments_method=str(pretrain_raw.get("assignments_method", "voxel_clustering")),
        spatial_blocks=tuple(pretrain_raw.get("spatial_blocks", [])) or None,
        time_block_size=_as_int(pretrain_raw.get("time_block_size"), 0),
    )
    timestep_curriculum_cfg = TimeStepCurriculumConfig(
        enabled=bool(timestep_curriculum_raw.get("enabled", False)),
        mode=str(timestep_curriculum_raw.get("mode", "linear")),
        start_timesteps=_as_int(timestep_curriculum_raw.get("start_timesteps"), 0),
        end_timesteps=_as_int(timestep_curriculum_raw.get("end_timesteps"), 0),
        warmup_epochs=_as_int(timestep_curriculum_raw.get("warmup_epochs"), 0),
        ramp_epochs=_as_int(timestep_curriculum_raw.get("ramp_epochs"), 0),
        stride_groups=_as_int(timestep_curriculum_raw.get("stride_groups"), 0),
        epochs_per_group=_as_int(timestep_curriculum_raw.get("epochs_per_group"), 0),
    )

    return TrainingConfig(
        epochs=_as_int(training_raw.get("epochs"), 600),
        batch_size=_as_int(training_raw.get("batch_size"), 16000),
        pred_batch_size=_as_int(training_raw.get("pred_batch_size"), training_raw.get("batch_size", 16000)),
        num_workers=_as_int(training_raw.get("num_workers"), 2),
        batches_per_epoch_budget=_as_int(training_raw.get("batches_per_epoch_budget"), 0),
        lr=_as_float(training_raw.get("lr"), 5.0e-5),
        val_split=_as_float(training_raw.get("val_split"), 0.0),
        log_every=_as_int(training_raw.get("log_every"), 1),
        log_psnr_every=_as_int(training_raw.get("log_psnr_every"), 100),
        psnr_sample_ratio=_as_float(training_raw.get("psnr_sample_ratio"), 0.1),
        save_every=_as_int(training_raw.get("save_every"), 0),
        early_stop_patience=_as_int(training_raw.get("early_stop_patience"), 0),
        seed=_as_int(training_raw.get("seed"), 42),
        save_model=str(training_raw.get("save_model") or exp_layout["save_model"]),
        save_pred=str(training_raw.get("save_pred") or exp_layout["save_pred"]),
        device=device,
        exp_dir=exp_layout["exp_dir"],
        exp_id=exp_layout["exp_id"],
        run_timestamp=run_timestamp,
        loss_type=str(training_raw.get("loss_type", "mse")),
        pretrain=pretrain_cfg,
        timestep_curriculum=timestep_curriculum_cfg,
        lr_decay_rate=_as_float(training_raw.get("lr_decay_rate"), 0.0),
        lr_decay_step=_as_int(training_raw.get("lr_decay_step"), 0),
        freeze_router_at=_as_float(training_raw.get("freeze_router_at"), 0.8),
        hard_topk_warmup_epochs=_as_int(training_raw.get("hard_topk_warmup_epochs"), 0),
    )


def _write_source_snapshot(exp_layout: Dict[str, str], source_cfg: Dict[str, Any]) -> str:
    source_path = Path(exp_layout["exp_dir"]) / "configs" / "switch_nerf_source.yaml"
    source_path.write_text(
        yaml.safe_dump(source_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(source_path)


def main() -> None:
    args = _parse_args()
    setup_logging()

    source_config_path = Path(args.config)
    if not source_config_path.is_absolute():
        source_config_path = (Path.cwd() / source_config_path).resolve()
    raw_cfg = _load_yaml(source_config_path)
    converted_cfg, resolved_source = _convert_switch_nerf_config(raw_cfg, args)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset, data_info = _build_dataset(converted_cfg["data"])
    exp_layout = build_experiment_layout(converted_cfg, converted_cfg["model"], data_info)
    setup_logging(log_dir=Path(exp_layout["exp_dir"]) / "logs", run_timestamp=run_timestamp)
    source_snapshot = _write_source_snapshot(exp_layout, resolved_source)

    model = build_model(converted_cfg["model"], dataset)
    train_cfg = _build_training_cfg(converted_cfg, exp_layout, run_timestamp, args.device)

    n_params = sum(param.numel() for param in model.parameters())
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info("Switch-NeRF INR source config: %s", source_config_path)
    logger.info("Resolved source snapshot: %s", source_snapshot)
    logger.info("validate_prediction config: %s", exp_layout["config_snapshot"])
    logger.info("Dataset size: %s samples", f"{len(dataset):,}")
    logger.info("Model parameters: total=%s trainable=%s", f"{n_params:,}", f"{n_trainable:,}")
    logger.info("Training config:\n%s", yaml.safe_dump(converted_cfg, sort_keys=False, allow_unicode=True))

    train_model(model, dataset, train_cfg)

    final_ckpt = _resolve_checkpoint_path(train_cfg.save_model, run_timestamp=train_cfg.run_timestamp)
    logger.info("Final checkpoint: %s", final_ckpt)
    logger.info("validate_prediction ready config: %s", exp_layout["config_snapshot"])


if __name__ == "__main__":
    main()

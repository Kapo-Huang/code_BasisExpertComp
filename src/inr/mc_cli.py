import argparse
from pathlib import Path
from typing import Any, Dict
import yaml
import warnings
warnings.filterwarnings("ignore")
from inr.training.mc_train_pipeline import run_full_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train MC-INR with a two-stage pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda:0")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from (meta or finetune stage).",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_data_paths(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same path resolution style as the main CLI:
    - raw data: data/raw/<dataset>/<split>/{coords.npy,targets.npy}
    - processed: data/processed/<dataset>/<version>/<split>/{coords.npy,targets.npy}
    """
    data_root = Path(data_cfg.get("data_root", "data"))
    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split", "train")
    processed_version = data_cfg.get("processed_version")

    x_path = data_cfg.get("x_path")
    y_path = data_cfg.get("y_path")
    attr_paths = data_cfg.get("attr_paths")

    if dataset_name and (x_path is None or (y_path is None and not attr_paths)):
        base_root = data_root / ("processed" if processed_version else "raw")
        base = base_root / dataset_name
        if processed_version:
            base = base / processed_version
        base = base / split
        x_path = x_path or str(base / "coords.npy")
        if not attr_paths:
            y_path = y_path or str(base / "targets.npy")

    if x_path is None or (y_path is None and not attr_paths):
        raise ValueError(
            "MC-INR config must provide data paths via x_path plus one of y_path/attr_paths."
        )

    normalized_attr_paths = None
    if attr_paths:
        normalized_attr_paths = {str(k): str(v) for k, v in attr_paths.items()}

    return {
        "x_path": str(x_path),
        "y_path": str(y_path) if y_path is not None else None,
        "attr_paths": normalized_attr_paths,
        "dataset_name": dataset_name,
        "split": split,
    }


def build_experiment_layout(cfg: Dict[str, Any], data_info: Dict[str, Any], model_name: str = "mc_inr") -> Dict[str, str]:
    exp_root = Path(cfg.get("experiment_root", "experiments"))
    exp_id = cfg.get("exp_id")
    if exp_id is None:
        dataset_token = (
            data_info.get("dataset_name")
            or Path(data_info["x_path"]).parent.name
            or Path(data_info["x_path"]).stem
        )
        exp_id = f"{model_name.lower()}-{dataset_token}"

    exp_dir = exp_root / exp_id
    ckpt_path = exp_dir / "checkpoints" / f"{exp_id}.pth"
    pred_path = exp_dir / "predictions" / f"{exp_id}-full.npy"
    cfg_snapshot = exp_dir / "configs" / "config.yaml"

    for d in ["configs", "checkpoints", "predictions", "logs"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    cfg_snapshot.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"Experiment ID: {exp_id}")

    return {
        "exp_dir": str(exp_dir),
        "exp_id": exp_id,
        "save_model": str(ckpt_path),
        "save_pred": str(pred_path),
        "config_snapshot": str(cfg_snapshot),
    }


def normalize_mc_config(raw_cfg: Dict[str, Any], device_override: str | None = None) -> Dict[str, Any]:
    """
    Accepts both:
    1) Flat MC-INR config (current style in src/configs/Mc_INR/exp_data.yaml)
    2) Nested config with keys: data/model/training (similar to main cli.py)
    """
    is_nested = "training" in raw_cfg or "data" in raw_cfg or "model" in raw_cfg
    if not is_nested:
        cfg = dict(raw_cfg)
        if device_override is not None:
            cfg["device"] = device_override
        if "data_normalize" not in cfg and "normalize" in cfg:
            cfg["data_normalize"] = bool(cfg["normalize"])
        return cfg

    data_cfg = raw_cfg.get("data", {})
    model_cfg = raw_cfg.get("model", {})
    train_cfg = raw_cfg.get("training", {})

    data_info = resolve_data_paths(data_cfg)
    model_name = model_cfg.get("name", "mc_inr")
    model_name_norm = str(model_name).lower()
    if model_name_norm not in {"mc_inr", "mcinr", "mc-inr"}:
        raise ValueError(
            f"mc_cli only supports MC-INR configs, but got model.name={model_name!r}. "
            "Use `python -m inr.cli --config ...` for non-MC models."
        )
    exp_layout = build_experiment_layout(raw_cfg, data_info, model_name=model_name)

    cfg: Dict[str, Any] = {}

    # Data
    cfg["data_x_path"] = data_info["x_path"]
    if data_info.get("attr_paths"):
        cfg["data_attr_paths"] = data_info["attr_paths"]
    else:
        cfg["data_y_path"] = data_info["y_path"]
    cfg["data_normalize"] = bool(data_cfg.get("normalize", True))
    if data_cfg.get("stats_path") is not None:
        cfg["data_stats_path"] = data_cfg.get("stats_path")

    # Model-specific fields for MC-INR
    if "hidden_features" in model_cfg:
        cfg["hidden_features"] = model_cfg["hidden_features"]
    if "gfe_layers" in model_cfg:
        cfg["gfe_layers"] = model_cfg["gfe_layers"]
    if "lfe_layers" in model_cfg:
        cfg["lfe_layers"] = model_cfg["lfe_layers"]

    # Training and clustering fields (reuse names directly when present)
    direct_keys = [
        "epochs",
        "batch_size",
        "num_workers",
        "lr",
        "lr_decay_step",
        "lr_decay_gamma",
        "log_every",
        "save_every",
        "seed",
        "device",
        "initial_k",
        "sampling_ratio",
        "split_threshold",
        "split_check_interval",
        "min_split_points",
        "max_recluster_rounds",
        "convergence_patience",
        "convergence_delta",
        "finetune_epochs",
        "finetune_lr",
        "finetune_sampling_ratio",
        "split_after_meta",
        "recluster_after_finetune",
        "resume_path",
    ]
    for key in direct_keys:
        if key in train_cfg:
            cfg[key] = train_cfg[key]
        elif key in raw_cfg:
            cfg[key] = raw_cfg[key]

    # Output paths
    cfg["save_model"] = train_cfg.get("save_model", raw_cfg.get("save_model", exp_layout["save_model"]))
    cfg["save_pred"] = train_cfg.get("save_pred", raw_cfg.get("save_pred", exp_layout["save_pred"]))

    if device_override is not None:
        cfg["device"] = device_override

    return cfg


def main():
    args = parse_args()
    raw_cfg = load_config(args.config)
    mc_cfg = normalize_mc_config(raw_cfg, device_override=args.device)
    if args.resume is not None:
        mc_cfg["resume_path"] = args.resume
    run_full_pipeline(mc_cfg)


if __name__ == "__main__":
    main()

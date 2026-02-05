import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml

from inr.training.mc_full_finetuning import train_full_finetuning
from inr.training.mc_meta import (
    MCTrainingConfig,
    load_mc_checkpoint,
    predict_full,
    perform_final_split,
    restore_model_dataset_from_checkpoint,
    train_mc_inr,
)


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _filter_mc_cfg(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    valid_keys = {f.name for f in fields(MCTrainingConfig)}
    filtered = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    ignored = sorted(k for k in raw_cfg.keys() if k not in valid_keys)
    if ignored:
        print(f"[MC Pipeline] Ignoring unsupported config keys: {ignored}")
    return filtered


def _coerce_mc_cfg(
    cfg_or_path: Union[MCTrainingConfig, Dict[str, Any], str, Path],
    device_override: str | None = None,
) -> MCTrainingConfig:
    if isinstance(cfg_or_path, MCTrainingConfig):
        cfg = cfg_or_path
    elif isinstance(cfg_or_path, (str, Path)):
        print("Loading configuration...")
        raw_cfg = _load_yaml(cfg_or_path)
        cfg = MCTrainingConfig(**_filter_mc_cfg(raw_cfg))
    elif isinstance(cfg_or_path, dict):
        cfg = MCTrainingConfig(**_filter_mc_cfg(cfg_or_path))
    else:
        raise TypeError(
            f"Unsupported config type: {type(cfg_or_path)}. "
            "Expected MCTrainingConfig | dict | str | Path."
        )

    if device_override is not None:
        cfg.device = device_override
    return cfg


def run_full_pipeline(
    cfg_or_path: Union[MCTrainingConfig, Dict[str, Any], str, Path],
    device_override: str | None = None,
):
    cfg = _coerce_mc_cfg(cfg_or_path, device_override=device_override)
    resume_stage = None
    if cfg.resume_path:
        resume_payload = load_mc_checkpoint(cfg.resume_path)
        resume_stage = resume_payload.get("mc_stage")

    if resume_stage == "finetune":
        print("\n" + "#" * 60)
        print("RESUME: LOAD FINETUNE CHECKPOINT")
        print("#" * 60 + "\n")
        model, dataset, _ = restore_model_dataset_from_checkpoint(cfg, cfg.resume_path)
    else:
        print("\n" + "#" * 60)
        print("PHASE 1: META STAGE")
        print("#" * 60 + "\n")
        model, dataset = train_mc_inr(cfg)

    finetuned_model, finetune_epoch = train_full_finetuning(model, dataset, cfg)

    # if bool(getattr(cfg, "recluster_after_finetune", False)):
    #     max_rounds = int(getattr(cfg, "max_recluster_rounds", 1))
    #     if max_rounds < 1:
    #         max_rounds = 1
    #     device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    #     for round_idx in range(max_rounds):
    #         print("\n" + "#" * 60)
    #         print(f"RE-CLUSTER ROUND {round_idx + 1}/{max_rounds}")
    #         print("#" * 60)
    #         split_count = perform_final_split(finetuned_model, dataset, cfg, device)
    #         if split_count <= 0:
    #             break
    #         finetuned_model, finetune_epoch = train_full_finetuning(finetuned_model, dataset, cfg)

    print("\n" + "#" * 60)
    print("PHASE 3: FINAL INFERENCE")
    print("#" * 60 + "\n")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predict_full(finetuned_model, dataset, cfg, device, epoch=finetune_epoch)

    print("\nAll phases completed.\n")
    return finetuned_model, dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Run MC-INR full two-stage pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to MC-INR yaml config")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., cpu or cuda:0")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg_or_path: Union[Dict[str, Any], str, Path] = args.config
    if args.resume is not None:
        raw_cfg = _load_yaml(args.config)
        raw_cfg["resume_path"] = args.resume
        cfg_or_path = raw_cfg
    run_full_pipeline(cfg_or_path, device_override=args.device)

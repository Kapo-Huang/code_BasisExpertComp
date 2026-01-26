import argparse
import yaml
from pathlib import Path
from inr.data import NodeDataset
from inr.models.moe_inr import build_moe_inr_from_config
from inr.models.basisExpert_simple_concat import (
    MultiViewCoordDataset,
    build_basisExpert_simple_concat_from_config,
)
from inr.models.siren import build_siren_from_config
from inr.training.loops import TrainingConfig, train_model

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
    if name in {"moe_inr_multiview", "moeinr_multiview", "moe-inr-multiview"}:
        from inr.models.moe_inr_multiview import build_moe_inr_multiview_from_config
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("moe_inr_multiview requires a MultiViewCoordDataset with view_specs().")
        return build_moe_inr_multiview_from_config(model_cfg, dataset.view_specs())
    if name in {"basisexpert_simple_concat"}:
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExpert_simple_concat requires a MultiViewCoordDataset with view_specs().")
        return build_basisExpert_simple_concat_from_config(model_cfg, dataset.view_specs())
    if name in {"baisiexpert_nomoe", "baisiexpert_no_moe", "basisexpert_no_moe", "basisexpert_nomoe"}:
        from inr.models.baisiExpert_NoMoE import build_baisiExpert_NoMoE_from_config
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("baisiExpert_NoMoE requires a MultiViewCoordDataset with view_specs().")
        return build_baisiExpert_NoMoE_from_config(model_cfg, dataset.view_specs())
    if name in {"basisexperts_attention"}:
        from inr.models.basisExperts_attention import build_basisExperts_attention_from_config
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExperts_attention requires a MultiViewCoordDataset with view_specs().")
        return build_basisExperts_attention_from_config(model_cfg, dataset.view_specs())
    if name in {"basisexperts_attention_light_pe", "basisexperts_attention_lightpe"}:
        from inr.models.basisExperts_attention_light_PE import (
            build_basisExperts_attention_light_pe_from_config,
        )
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExperts_attention_light_PE requires a MultiViewCoordDataset with view_specs().")
        return build_basisExperts_attention_light_pe_from_config(model_cfg, dataset.view_specs())
    if name in {"basisexperts_attention_light_viewembed"}:
        from inr.models.basisExperts_attention_light_ViewEmbed import (
            build_basisExperts_attention_light_viewembed_from_config,
        )
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("basisExperts_attention_light_ViewEmbed requires a MultiViewCoordDataset with view_specs().")
        return build_basisExperts_attention_light_viewembed_from_config(model_cfg, dataset.view_specs())
    if name in {"lightbasisexpert", "light_basis_expert", "light_basis_expert_pe"}:
        from inr.models.LightBasisExpert import build_light_basis_expert_from_config
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError("LightBasisExpert requires a MultiViewCoordDataset with view_specs().")
        return build_light_basis_expert_from_config(model_cfg, dataset.view_specs())
    if name in {"stsr_inr", "stsrinr", "stsr-inr"}:
        from inr.models.STSR_inr import (
            build_stsr_inr_from_config,
            build_stsr_inr_multiview_from_config,
        )
        if dataset is not None and hasattr(dataset, "view_specs"):
            return build_stsr_inr_multiview_from_config(model_cfg, dataset.view_specs())
        return build_stsr_inr_from_config(model_cfg)
    if name in {
        "basesharedencviewaddshareddectrunk",
        "base_shared_enc_view_add_shared_dec_trunk",
        "sharedenc_viewadd",
    }:
        from inr.models.BaseSharedEncViewAddSharedDecTrunk import (
            build_base_shared_enc_view_add_shared_dec_trunk_from_config,
        )
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError(
                "BaseSharedEncViewAddSharedDecTrunk requires a MultiViewCoordDataset with view_specs()."
            )
        return build_base_shared_enc_view_add_shared_dec_trunk_from_config(
            model_cfg, dataset.view_specs()
        )
    if name in {
        "basesharedencviewattentionfuseddectrunk",
        "base_shared_enc_view_attention_fused_dec_trunk",
        "sharedenc_viewattn",
    }:
        from inr.models.BaseSharedEncViewAttentionFusedDecTrunk import (
            build_base_shared_enc_view_attention_fused_dec_trunk_from_config,
        )
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError(
                "BaseSharedEncViewAttentionFusedDecTrunk requires a MultiViewCoordDataset with view_specs()."
            )
        return build_base_shared_enc_view_attention_fused_dec_trunk_from_config(
            model_cfg, dataset.view_specs()
        )
    if name in {
        "basemoeencviewadddectrunk",
        "base_moe_enc_view_add_dec_trunk",
        "moeenc_viewadd",
    }:
        from inr.models.BaseMoEEncViewAddDecTrunk import (
            build_base_moe_enc_view_add_dec_trunk_from_config,
        )
        if dataset is None or not hasattr(dataset, "view_specs"):
            raise ValueError(
                "BaseMoEEncViewAddDecTrunk requires a MultiViewCoordDataset with view_specs()."
            )
        return build_base_moe_enc_view_add_dec_trunk_from_config(
            model_cfg, dataset.view_specs()
        )
    if name == "coordnet":
        from inr.models.CoordNet import build_coordnet_from_config
        return build_coordnet_from_config(model_cfg)
    raise ValueError(f"Unknown model name: {name}")

def resolve_data_paths(data_cfg):
    """
    Normalizes data paths to the new structure:
    - raw data:   data/raw/<dataset>/<split>/coords.npy, targets.npy
    - processed:  data/processed/<dataset>/<version>/<split>/coords.npy, targets.npy
    Explicit x_path/y_path override everything.
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
        raise ValueError("x_path and y_path (or attr_paths) must be provided or inferrable.")

    return {
        "x_path": str(x_path),
        "y_path": str(y_path) if y_path is not None else None,
        "attr_paths": attr_paths,
        "dataset_name": dataset_name,
        "split": split,
    }


def build_experiment_layout(cfg, model_cfg, data_info):
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
    ckpt_path = exp_dir / "checkpoints" / f"{exp_id}.pth"
    pred_path = exp_dir / "predictions" / f"{exp_id}-full.npy"
    cfg_snapshot = exp_dir / "configs" / "config.yaml"

    for d in ["configs", "checkpoints", "predictions", "logs"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    # Persist a copy of the launched config for reproducibility
    cfg_snapshot.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    print(f"Experiment ID: {exp_id}")

    return {
        "exp_dir": str(exp_dir),
        "exp_id": exp_id,
        "save_model": str(ckpt_path),
        "save_pred": str(pred_path),
        "config_snapshot": str(cfg_snapshot),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg_raw = cfg["training"]

    data_info = resolve_data_paths(data_cfg)
    exp_layout = build_experiment_layout(cfg, model_cfg, data_info)
    stats_path = data_cfg.get("stats_path")

    if data_info.get("attr_paths"):
        dataset = MultiViewCoordDataset(
            data_info["x_path"],
            data_info["attr_paths"],
            normalize=bool(data_cfg.get("normalize", True)),
            stats_path=stats_path,
        )
    else:
        dataset = NodeDataset(
            data_info["x_path"],
            data_info["y_path"],
            normalize=bool(data_cfg.get("normalize", True)),
            stats_path=stats_path,
        )
    model = build_model(model_cfg, dataset)

    train_cfg = TrainingConfig(
        epochs=int(train_cfg_raw.get("epochs", 100)),
        batch_size=int(train_cfg_raw.get("batch_size", 65536)),
        pred_batch_size=int(train_cfg_raw.get("pred_batch_size", train_cfg_raw.get("batch_size", 65536))),
        num_workers=int(train_cfg_raw.get("num_workers", 4)),
        lr=float(train_cfg_raw.get("lr", 5e-5)),
        log_every=int(train_cfg_raw.get("log_every", 4)),
        save_every=int(train_cfg_raw.get("save_every", 0)),
        early_stop_patience=int(train_cfg_raw.get("early_stop_patience", 0)),
        seed=int(train_cfg_raw.get("seed", 42)),
        save_model=train_cfg_raw.get("save_model", exp_layout["save_model"]),
        save_pred=train_cfg_raw.get("save_pred", exp_layout["save_pred"]),
        device=args.device,
        data_x_path=data_info["x_path"],
        data_y_path=data_info.get("y_path") or "",
        exp_dir=exp_layout["exp_dir"],
        exp_id=exp_layout["exp_id"],
        loss_type=str(train_cfg_raw.get("loss_type", "mse")),
        lam_eq=float(train_cfg_raw.get("lam_eq", 0.0)),
        gam_div=float(train_cfg_raw.get("gam_div", 0.0)),
        view_loss_weights=train_cfg_raw.get("view_loss_weights"),
        resume_path=train_cfg_raw.get("resume_path"),
    )

    train_model(model, dataset, train_cfg)


if __name__ == "__main__":
    main()

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

THIS_DIR = Path(__file__).resolve().parent
NEURAL_EXPERTS_ROOT = THIS_DIR.parent
REPO_ROOT = NEURAL_EXPERTS_ROOT.parent
for path in (str(NEURAL_EXPERTS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from datasets import build_dataloader
from models import build_model
from models.stage_handler import TrainingStageHandler
import utils.utils as utils

try:
    import wandb
except Exception:
    class _DummyRun:
        def __init__(self, project):
            self.id = "disabled"
            self.project = project
            self.entity = "disabled"
            self.name = ""

        def log_code(self, *args, **kwargs):
            return None

    class _DummyConfig:
        def update(self, *args, **kwargs):
            return None

    class _DummyWandb:
        def __init__(self):
            self.run = _DummyRun(project="disabled")
            self.config = _DummyConfig()

        def init(self, project=None, entity=None, save_code=None, dir=None, mode=None):
            self.run = _DummyRun(project=project or "disabled")
            return self.run

        def define_metric(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _DummyWandb()


def lossdict2str(loss_dict):
    string = ""
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            val = value.item()
        else:
            val = value
        if val == 0.0:
            continue
        if key == "lr" or abs(val) < 1.0e-4:
            string += f"{key}: {val:.4e}, "
        else:
            string += f"{key}: {val:.8f}, "
    return string


def _estimate_model_size_fp32(model):
    parameter_count = sum(param.numel() for param in model.parameters())
    size_bytes = parameter_count * 4
    return parameter_count, size_bytes


def _format_size_bytes(size_bytes):
    size_mib = size_bytes / (1024 ** 2)
    size_mb = size_bytes / 1.0e6
    return f"{size_bytes} bytes ({size_mib:.2f} MiB, {size_mb:.2f} MB)"


def _resolve_path(path_str, config_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _normalize_relative_to_repo(path):
    path = Path(path).resolve()
    try:
        rel = path.relative_to(REPO_ROOT.resolve())
        return "./" + rel.as_posix()
    except Exception:
        return path.as_posix()


def _resolve_config_paths(cfg, config_path):
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    config_dir = Path(config_path).resolve().parent
    cfg["DATA"]["target_path"] = str(_resolve_path(cfg["DATA"]["target_path"], config_dir))
    if cfg["DATA"].get("target_stats_path"):
        cfg["DATA"]["target_stats_path"] = str(_resolve_path(cfg["DATA"]["target_stats_path"], config_dir))
    if cfg["MODEL"].get("manager_pt_path"):
        cfg["MODEL"]["manager_pt_path"] = str(_resolve_path(cfg["MODEL"]["manager_pt_path"], config_dir))
    return cfg


def _format_duration(seconds):
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _to_device(data, device):
    out = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _load_state_dict_payload(path, device):
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def _save_validate_checkpoint(model, dataset, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "x_mean": dataset.x_mean.cpu().numpy(),
        "x_std": dataset.x_std.cpu().numpy(),
        "y_mean": dataset.y_mean.cpu().numpy(),
        "y_std": dataset.y_std.cpu().numpy(),
    }
    torch.save(payload, str(out_path))


def _build_validate_config(cfg, validate_ckpt_path):
    data_cfg = cfg["DATA"]
    model_cfg = cfg["MODEL"]
    volume_shape = data_cfg["volume_shape"]
    return {
        "experiment": f"exp_data_ionization_neural_expert_{data_cfg['attr_name']}",
        "exp_id": f"neural_expert-ionization-{data_cfg['attr_name']}",
        "experiment_root": "experiments",
        "data": {
            "dataset_name": "ionization",
            "split": "train",
            "data_root": "./data",
            "target_path": _normalize_relative_to_repo(data_cfg["target_path"]),
            "target_stats_path": _normalize_relative_to_repo(data_cfg["target_stats_path"]),
            "compute_target_stats": False,
            "volume_shape": {
                "X": int(volume_shape["X"]),
                "Y": int(volume_shape["Y"]),
                "Z": int(volume_shape["Z"]),
                "T": int(volume_shape["T"]),
            },
            "normalize_inputs": bool(data_cfg.get("normalize_inputs", True)),
            "normalize_targets": bool(data_cfg.get("normalize_targets", False)),
        },
        "model": {
            "name": "neural_expert",
            "in_features": int(model_cfg["in_dim"]),
            "out_features": int(model_cfg["out_dim"]),
            "num_experts": int(model_cfg["n_experts"]),
            "top_k": int(model_cfg["top_k"]),
            "decoder_hidden_dim": int(model_cfg["decoder_hidden_dim"]),
            "decoder_n_hidden_layers": int(model_cfg["decoder_n_hidden_layers"]),
            "decoder_input_encoding": str(model_cfg["decoder_input_encoding"]),
            "decoder_nl": str(model_cfg["decoder_nl"]),
            "decoder_init_type": str(model_cfg["decoder_init_type"]),
            "decoder_freqs": float(model_cfg["decoder_freqs"]),
            "decoder_trainable_freqs": bool(model_cfg.get("decoder_trainable_freqs", False)),
            "manager_hidden_dim": int(model_cfg["manager_hidden_dim"]),
            "manager_n_hidden_layers": int(model_cfg["manager_n_hidden_layers"]),
            "manager_input_encoding": str(model_cfg["manager_input_encoding"]),
            "manager_nl": str(model_cfg["manager_nl"]),
            "manager_init": str(model_cfg["manager_init"]),
            "manager_softmax_temperature": float(model_cfg["manager_softmax_temperature"]),
            "manager_softmax_temp_trainable": bool(model_cfg["manager_softmax_temp_trainable"]),
            "manager_q_activation": str(model_cfg["manager_q_activation"]),
            "manager_clamp_q": float(model_cfg["manager_clamp_q"]),
            "manager_conditioning": str(model_cfg["manager_conditioning"]),
            "manager_type": str(model_cfg.get("manager_type", "standard")),
            "shared_encoder": bool(model_cfg.get("shared_encoder", False)),
        },
        "training": {
            "epochs": int(cfg["TRAINING"]["num_epochs"]),
            "batch_size": int(cfg["TRAINING"]["n_points"]),
            "pred_batch_size": int(cfg["TRAINING"]["n_points"]),
            "num_workers": 0,
            "lr": float(cfg["TRAINING"]["lr"]),
            "save_model": str(Path(validate_ckpt_path).as_posix()),
        },
    }


def _export_validate_artifacts(cfg, args, dataset, model):
    validate_dir = Path(args.logdir) / "validate_artifacts"
    validate_dir.mkdir(parents=True, exist_ok=True)
    validate_ckpt_path = validate_dir / f"{args.identifier}.pth"
    validate_cfg_path = validate_dir / "config.yaml"
    _save_validate_checkpoint(model, dataset, validate_ckpt_path)
    validate_cfg = _build_validate_config(cfg, validate_ckpt_path)
    validate_cfg_path.write_text(yaml.safe_dump(validate_cfg, sort_keys=False), encoding="utf-8")
    return validate_cfg_path, validate_ckpt_path


def main(args):
    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(os.path.join(args.logdir, "wandb"), exist_ok=True)
        os.makedirs(os.path.join(args.logdir, "trained_models"), exist_ok=True)
        shutil.copyfile(args.config, os.path.join(args.logdir, "config.yaml"))
        shutil.copytree(NEURAL_EXPERTS_ROOT / "models", Path(args.logdir) / "models", dirs_exist_ok=True)
        shutil.copy(__file__, os.path.join(args.logdir, Path(__file__).name))

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = _resolve_config_paths(cfg, args.config)
    cfg["TRAINING"]["n_samples"] = cfg["TRAINING"]["num_epochs"]
    if args.attribute:
        cfg["DATA"]["attr_name"] = args.attribute

    if cfg["DATA"]["n_segments"] > cfg["MODEL"]["n_experts"]:
        raise ValueError("DATA.n_segments must be <= MODEL.n_experts")

    wandb_run = wandb.init(
        project=f"{cfg['wandb_project']}_{cfg['DATA']['attr_name']}",
        entity="anu-cvml",
        save_code=True,
        dir=os.path.join(args.logdir, "wandb"),
        mode="disabled",
    )
    cfg["WANDB"] = {"id": wandb_run.id, "project": wandb_run.project, "entity": wandb_run.entity}
    wandb_run.name = args.identifier
    wandb.config.update(cfg)
    if hasattr(wandb, "run") and hasattr(wandb.run, "log_code"):
        wandb.run.log_code(".")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    log_file = open(os.path.join(args.logdir, "out.log"), "w", encoding="utf-8")
    timing_cfg = cfg.get("TRAINING", {}).get("timing", {})
    timing_enabled = bool(timing_cfg.get("enabled", True))
    timing_start = time.perf_counter()
    completed_epochs = 0
    print(args)
    print("torch version: ", torch.__version__)

    try:
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_dataloader, train_set = build_dataloader(cfg, cfg["DATA"]["attr_name"], training=True)
        cfg["MODEL"]["out_dim"] = 1

        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        cfg["device"] = device

        SINR, _ = build_model(cfg, cfg["LOSS"])
        n_parameters = utils.count_parameters(SINR)
        total_parameters, model_size_bytes = _estimate_model_size_fp32(SINR)
        wandb.log(
            {
                "number of paramters": n_parameters,
                "model_size_bytes_fp32": model_size_bytes,
                "model_size_mib_fp32": model_size_bytes / (1024 ** 2),
            }
        )
        utils.log_string(f"Number of parameters in the current model:{n_parameters}", log_file)
        utils.log_string(
            f"Model size assuming float32 parameters: {_format_size_bytes(model_size_bytes)} "
            f"(total parameters: {total_parameters})",
            log_file,
        )

        training_stage_handler = TrainingStageHandler(cfg["TRAINING"]["stages"], SINR, cfg)
        criterion = training_stage_handler.criterion
        lr = cfg["TRAINING"]["lr"] if isinstance(cfg["TRAINING"]["lr"], float) else cfg["TRAINING"]["lr"]["all"]
        optimizer = optim.Adam(training_stage_handler.get_trainable_params(), lr=lr, betas=(0.9, 0.999))
        if "moe" in cfg["MODEL"]["model_name"]:
            training_stage_handler.freeze_params()
        scheduler = training_stage_handler.get_scheduler(optimizer)

        if cfg["MODEL"].get("load_pt_manager", False):
            manager_pt_checkpoint_path = Path(cfg["MODEL"]["manager_pt_path"])
            if not manager_pt_checkpoint_path.exists():
                raise FileNotFoundError(f"Missing manager pretrain checkpoint: {manager_pt_checkpoint_path}")
            SINR.load_state_dict(_load_state_dict_payload(manager_pt_checkpoint_path, device), strict=True)
            utils.log_string(f"Loaded pretrained manager from {manager_pt_checkpoint_path}", log_file)

        SINR.to(device)

        model_outdir = Path(args.logdir) / "trained_models"
        save_interval = int(cfg["TRAINING"].get("save_every", 100) or 100)
        max_epochs = int(cfg["TRAINING"]["num_epochs"])

        for step, data in enumerate(train_dataloader):
            if step >= max_epochs:
                break

            epoch_timer_start = time.perf_counter() if timing_enabled else None

            if step % save_interval == 0:
                torch.save(SINR.state_dict(), str(model_outdir / f"{cfg['MODEL']['model_name']}_model_{step}.pth"))

            data = _to_device(data, device)
            SINR.zero_grad()
            SINR.train()

            output_pred = SINR(data["nonmnfld_points"])
            output_pred["step"] = step
            output_pred["logdir"] = args.logdir
            loss_dict = criterion(output_pred=output_pred, data=data, dataset=train_set)

            lr_t = torch.tensor(optimizer.param_groups[0]["lr"])
            loss_dict["lr"] = lr_t
            if "moe" in cfg["MODEL"]["model_name"] and cfg["MODEL"]["manager_q_activation"] == "softmax" and cfg["MODEL"]["manager_softmax_temp_trainable"]:
                loss_dict["softmax_temp"] = SINR.manager_net.q_activation.temperature.item()

            utils.log_losses_wandb(step, -1, 1, loss_dict, 1, criterion.weight_dict)
            if step % 100 == 0:
                utils.log_string(f"{step:05d} " + lossdict2str(loss_dict), log_file)

            loss_dict["loss"].backward()
            optimizer.step()
            scheduler.step()

            if step > training_stage_handler.get_end_iteration():
                utils.log_string("Moved to the next training stage...", log_file)
                training_stage_handler.move_to_the_next_training_stage(optimizer, scheduler)
                criterion = training_stage_handler.criterion

            completed_epochs = step + 1
            if timing_enabled and epoch_timer_start is not None:
                epoch_elapsed = time.perf_counter() - epoch_timer_start
                utils.log_string(
                    f"[timing] epoch {completed_epochs:05d}/{max_epochs:05d}: "
                    f"{epoch_elapsed:.3f}s ({_format_duration(epoch_elapsed)})",
                    log_file,
                )

        final_state_path = model_outdir / f"{cfg['MODEL']['model_name']}_model_final.pth"
        torch.save(SINR.state_dict(), str(final_state_path))
        utils.log_string(f"Saved final state dict to {final_state_path}", log_file)

        if cfg["TRAINING"].get("segmentation_mode", False) and cfg["MODEL"].get("manager_pt_path"):
            manager_pt_path = Path(cfg["MODEL"]["manager_pt_path"])
            manager_pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(SINR.state_dict(), str(manager_pt_path))
            utils.log_string(f"Exported manager pretrain checkpoint to {manager_pt_path}", log_file)

        validate_cfg_path, validate_ckpt_path = _export_validate_artifacts(cfg, args, train_set, SINR)
        utils.log_string(f"Exported validate config to {validate_cfg_path}", log_file)
        utils.log_string(f"Exported validate checkpoint to {validate_ckpt_path}", log_file)
    finally:
        if timing_enabled:
            total_elapsed = time.perf_counter() - timing_start
            avg_epoch_time = total_elapsed / completed_epochs if completed_epochs > 0 else 0.0
            utils.log_string(
                f"[timing] training summary: epochs={completed_epochs}, "
                f"total={total_elapsed:.3f}s ({_format_duration(total_elapsed)}), "
                f"avg_epoch={avg_epoch_time:.3f}s",
                log_file,
            )
        log_file.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Neural-Experts ionization training")
    parser.add_argument(
        "--config",
        default=str(NEURAL_EXPERTS_ROOT / "configs" / "ionization" / "config_ionization_Hplus.yaml"),
        type=str,
        help="config file",
    )
    parser.add_argument("--logdir", default="./log", type=str, help="path to log directory")
    parser.add_argument("--identifier", default="debug_ionization", type=str, help="unique identifier for this experiment")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index to use")
    parser.add_argument("--attribute", default="", type=str, help="optional attribute override")
    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.identifier)
    return args


if __name__ == "__main__":
    main(parse_args())

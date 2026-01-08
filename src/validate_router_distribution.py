import argparse
import math
import time
from pathlib import Path
import colorsys

import numpy as np
import torch
import yaml
import pyvista as pv
from torch.utils.data import DataLoader

from inr.cli import build_model, resolve_data_paths
from inr.data import NodeDataset
from inr.utils.io import load_checkpoint


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _match_checkpoint(epoch: int, ckpt_files):
    if not ckpt_files:
        return None
    token = f"epoch{epoch}"
    for ckpt in ckpt_files:
        if token in ckpt.name:
            return ckpt
    return None


def _collect_experiments(exp_root: Path):
    exps = []
    for child in sorted(exp_root.iterdir()):
        if not child.is_dir():
            continue
        cfg_path = child / "configs" / "config.yaml"
        if cfg_path.exists():
            exps.append(child)
    return exps


def _unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb
    raise ValueError(f"Unexpected batch structure: {len(batch)}")


def _router_probs(model, xb, hard_routing: bool):
    if not hasattr(model, "policy"):
        raise ValueError("Model has no policy network; cannot extract router outputs.")
    out = model(xb, hard_routing=hard_routing, return_all=True)
    if not isinstance(out, (tuple, list)) or len(out) < 3:
        raise ValueError("Model did not return routing probabilities.")
    return out[2]


def _build_palette(num_experts: int):
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if num_experts <= len(base):
        return base[:num_experts]
    colors = []
    for i in range(num_experts):
        h = (i / max(1, num_experts)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
        colors.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _plot_expert_series(
    mesh_path: Path,
    expert_series,
    outpath: Path,
    img_scale: float,
    num_experts: int,
):
    mesh_template = pv.read(str(mesh_path))
    n_frames = len(expert_series)
    ncols = min(5, n_frames) if n_frames > 0 else 1
    nrows = int(math.ceil(n_frames / ncols)) if n_frames > 0 else 1
    width = max(800, 500 * ncols)
    height = max(600, 500 * nrows)
    width = int(width * img_scale)
    height = int(height * img_scale)

    palette = _build_palette(num_experts)
    clim = [-0.5, num_experts - 0.5]

    plotter = pv.Plotter(shape=(nrows, ncols), window_size=(width, height), off_screen=True)
    plotter.set_background("SlateGray")

    for idx, expert_idx in enumerate(expert_series):
        row = idx // ncols
        col = idx % ncols
        plotter.subplot(row, col)
        mesh = mesh_template.copy()
        expert_idx = np.asarray(expert_idx).ravel()
        if expert_idx.shape[0] == mesh.n_points:
            mesh.point_data["expert_id"] = expert_idx
            scalars = "expert_id"
        elif expert_idx.shape[0] == mesh.n_cells:
            mesh.cell_data["expert_id"] = expert_idx
            scalars = "expert_id"
        else:
            raise ValueError(
                f"Expert idx length {expert_idx.shape[0]} does not match mesh points {mesh.n_points} or cells {mesh.n_cells}."
            )
        plotter.add_mesh(
            mesh,
            scalars=scalars,
            clim=clim,
            cmap=palette,
            show_edges=False,
            scalar_bar_args=dict(title="Expert", n_labels=num_experts),
            nan_color="white",
        )
        plotter.add_text(f"Expert {idx}", font_size=10)

    plotter.show(screenshot=str(outpath))


def extract_router_distribution(
    exp_dir: Path,
    outdir: Path,
    mesh_path: Path,
    n_frames: int,
    epoch: int,
    batch_size: int,
    device: str | None,
    img_scale: float,
    hard_routing: bool,
    force_soft: bool,
):
    t_start = time.perf_counter()
    cfg_path = exp_dir / "configs" / "config.yaml"
    cfg = _load_yaml(cfg_path)
    print(f"[{exp_dir.name}] Stage: load config = {time.perf_counter() - t_start:.3f}s")

    t_stage = time.perf_counter()
    ckpt_files = sorted((exp_dir / "checkpoints").glob("*.pth"))
    ckpt_path = _match_checkpoint(epoch, ckpt_files)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {exp_dir}")
    print(f"[{exp_dir.name}] Stage: find checkpoint file = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    data_info = resolve_data_paths(data_cfg)
    dataset = NodeDataset(
        data_info["x_path"],
        data_info["y_path"],
        normalize=bool(data_cfg.get("normalize", True)),
    )
    inferred_in = int(getattr(dataset.x, "shape", [None, None])[1] or 0)
    cfg_in = int(model_cfg.get("in_features", inferred_in or 0))
    if inferred_in and cfg_in and inferred_in != cfg_in:
        print(
            f"[{exp_dir.name}] Warning: model in_features={cfg_in} but dataset has {inferred_in}; "
            "overriding to match dataset."
        )
        model_cfg = dict(model_cfg)
        model_cfg["in_features"] = inferred_in
    model = build_model(model_cfg, dataset)
    print(f"[{exp_dir.name}] Stage: build model/dataset = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    load_checkpoint(str(ckpt_path), model)
    print(f"[{exp_dir.name}] Stage: load checkpoint = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    if force_soft and hasattr(model, "hard_routing_at_eval"):
        model.hard_routing_at_eval = False
    print(f"[{exp_dir.name}] Stage: prepare device = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    print(f"[{exp_dir.name}] Stage: build loader = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    expert_ids = []
    with torch.no_grad():
        for batch in loader:
            xb, _ = _unpack_batch(batch)
            xb = xb.to(device)
            probs = _router_probs(model, xb, hard_routing=hard_routing)
            idx = torch.argmax(probs, dim=-1)
            expert_ids.append(idx.cpu())
    expert_all = torch.cat(expert_ids, dim=0).numpy()
    print(f"[{exp_dir.name}] Stage: router inference = {time.perf_counter() - t_stage:.3f}s")

    t_stage = time.perf_counter()
    if n_frames <= 0:
        raise ValueError("--n-frames must be > 0")
    expert_series = np.array_split(expert_all, n_frames)
    outdir.mkdir(parents=True, exist_ok=True)
    img_path = outdir / f"{exp_dir.name}_router_experts_epoch{epoch}.png"
    print(f"[{exp_dir.name}] Stage: prepare outputs = {time.perf_counter() - t_stage:.3f}s")

    if mesh_path.exists():
        t_stage = time.perf_counter()
        num_experts = int(getattr(model, "num_experts", np.max(expert_all) + 1))
        _plot_expert_series(mesh_path, expert_series, img_path, img_scale, num_experts)
        print(f"[{exp_dir.name}] Stage: plot experts = {time.perf_counter() - t_stage:.3f}s")

    print(f"[{exp_dir.name}] Stage: total = {time.perf_counter() - t_start:.3f}s")
    return img_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MoE router expert assignments on a mesh.")
    parser.add_argument("--experiments", type=str, default="experiments", help="experiments root directory")
    parser.add_argument("--outdir", type=str, default="validate_out", help="output directory")
    parser.add_argument("--exp-id", type=str, default=None, help="single experiment id to visualize")
    parser.add_argument("--mesh", type=str, default="data/raw/mesh_vtu/sukong_zip_0_0.vtu", help="mesh vtu path")
    parser.add_argument("--n-frames", type=int, default=10, help="number of time frames to split for plotting")
    parser.add_argument("--epoch", type=int, required=True, help="epoch number to select checkpoint")
    parser.add_argument("--batch-size", type=int, default=65536, help="inference batch size")
    parser.add_argument("--device", type=str, default=None, help="force device, e.g., cpu or cuda:0")
    parser.add_argument("--img-scale", type=float, default=2.0, help="scale factor for plot resolution")
    parser.add_argument("--hard-routing", action="store_true", help="force hard routing in model forward")
    parser.add_argument("--force-soft", action="store_true", help="disable hard routing at eval if supported")
    args = parser.parse_args()

    exp_root = Path(args.experiments)
    outdir = Path(args.outdir)
    mesh_path = Path(args.mesh)

    if args.exp_id:
        exp_dirs = [exp_root / args.exp_id]
    else:
        exp_dirs = _collect_experiments(exp_root)

    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        img_path = extract_router_distribution(
            exp_dir,
            outdir,
            mesh_path,
            args.n_frames,
            args.epoch,
            args.batch_size,
            args.device,
            args.img_scale,
            args.hard_routing,
            args.force_soft,
        )
        print(f"Visualized {exp_dir.name}")
        print(f"  Expert plot: {img_path}")


if __name__ == "__main__":
    main()

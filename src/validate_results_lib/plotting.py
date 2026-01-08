import math

import numpy as np
import pyvista as pv

from .metrics import EPS


def plot_rel_error_series(
    mesh_path,
    pred_series,
    gt_series,
    outpath,
    img_scale: float,
    relerr_clip_percentile: float | None,
    relerr_max: float | None,
):
    mesh_template = pv.read(str(mesh_path))
    n_frames = len(pred_series)
    ncols = min(5, n_frames) if n_frames > 0 else 1
    nrows = int(math.ceil(n_frames / ncols)) if n_frames > 0 else 1
    width = max(800, 500 * ncols)
    height = max(600, 500 * nrows)
    width = int(width * img_scale)
    height = int(height * img_scale)

    rel_errors = [
        (np.abs(pred - gt) / (np.abs(gt) + EPS)) * 100.0
        for pred, gt in zip(pred_series, gt_series)
    ]
    vmin = min(float(np.nanmin(u)) for u in rel_errors)
    vmax = max(float(np.nanmax(u)) for u in rel_errors)
    if relerr_clip_percentile is not None:
        flat = np.concatenate([u.ravel() for u in rel_errors])
        vmax = float(np.nanpercentile(flat, relerr_clip_percentile))
    if relerr_max is not None:
        vmax = float(relerr_max)

    plotter = pv.Plotter(shape=(nrows, ncols), window_size=(width, height), off_screen=True)
    plotter.set_background("SlateGray")

    for idx, u_vis in enumerate(rel_errors):
        row = idx // ncols
        col = idx % ncols
        plotter.subplot(row, col)
        mesh = mesh_template.copy()
        mesh["U_vis"] = u_vis
        plotter.add_mesh(
            mesh,
            scalars="U_vis",
            clim=[vmin, vmax],
            show_edges=False,
            scalar_bar_args=dict(title="RelErr (%)"),
            nan_color="white",
        )
        plotter.add_text(f"RelErr {idx}", font_size=10)

    plotter.show(screenshot=str(outpath))


def plot_pred_vs_gt(mesh_path, pred_series, gt_series, outpath, img_scale: float):
    mesh_template = pv.read(str(mesh_path))
    n_frames = len(pred_series)
    ncols = min(5, n_frames) if n_frames > 0 else 1
    nrows = int(math.ceil(n_frames / ncols))
    plot_rows = max(2, 2 * nrows)

    width = max(800, 500 * ncols)
    height = max(800, 500 * plot_rows)
    width = int(width * img_scale)
    height = int(height * img_scale)

    vmin = min(float(np.nanmin(u)) for u in list(pred_series) + list(gt_series))
    vmax = max(float(np.nanmax(u)) for u in list(pred_series) + list(gt_series))

    plotter = pv.Plotter(shape=(plot_rows, ncols), window_size=(width, height), off_screen=True)
    plotter.set_background("SlateGray")

    for idx, (pred_u, gt_u) in enumerate(zip(pred_series, gt_series)):
        row = (idx // ncols) * 2
        col = idx % ncols

        plotter.subplot(row, col)
        mesh_pred = mesh_template.copy()
        mesh_pred["U_vis"] = pred_u
        plotter.add_mesh(
            mesh_pred,
            scalars="U_vis",
            clim=[vmin, vmax],
            show_edges=False,
            scalar_bar_args=dict(title="|U|"),
            nan_color="white",
        )
        plotter.add_text(f"Pred {idx}", font_size=10)

        plotter.subplot(row + 1, col)
        mesh_gt = mesh_template.copy()
        mesh_gt["U_vis"] = gt_u
        plotter.add_mesh(
            mesh_gt,
            scalars="U_vis",
            clim=[vmin, vmax],
            show_edges=False,
            scalar_bar_args=dict(title="|U|"),
            nan_color="white",
        )
        plotter.add_text(f"GT {idx}", font_size=10)

    plotter.show(screenshot=str(outpath))

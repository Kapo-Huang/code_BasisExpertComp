import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inr.utils.logging_utils import setup_logging
from validate_results import (
    _align_eval_shapes,
    _build_lpips_model,
    _clip_upper_clim,
    _collect_mesh_candidates,
    _compute_lpips,
    _compute_psnr,
    _compute_ssim,
    _compute_time_indexers,
    _ensure_runtime_dependencies,
    _indexer_size,
    _load_image,
    _mean_finite,
    _merge_range,
    _normalize_clim,
    _render_frame,
    _render_zoom_factor,
    _resolve_mesh_for_timestep,
    _select_default_timestamps,
    _to_visual_scalar,
)

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES = [
    "row_type",
    "method",
    "dataset_name",
    "attr",
    "time_index",
    "raw_time",
    "num_samples",
    "num_timesteps",
    "gt_render_path",
    "pred_render_path",
    "psnr",
    "ssim",
    "lpips",
    "recon_path",
    "compressed_path",
    "result_json_path",
    "compression_ratio",
    "target_mode",
    "target_value",
    "shape_transform",
    "recon_shape",
    "restored_shape",
]
_METHOD_COMPRESSED_EXT = {
    "SZ3": ".sz3pkg",
    "ZFP": ".zfp",
}
_RECON_SUFFIXES = ("_recon.npy", "recon.npy")
_RESULT_SUFFIXES = ("_result.json", "result.json")
_GT_CACHE_VERSION = 3


@dataclass(frozen=True)
class AttrRegistryEntry:
    stem: str
    dataset_name: str
    gt_path: Path
    coords_path: Path
    preferred_association: str
    raw_replacements: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ArtifactCandidate:
    path: Path
    is_canonical: bool


@dataclass(frozen=True)
class ArtifactRecord:
    method: str
    spec: AttrRegistryEntry
    compressed_path: Path | None
    recon_path: Path | None
    result_json_path: Path | None
    compressed_aliases: tuple[Path, ...]
    recon_aliases: tuple[Path, ...]
    result_aliases: tuple[Path, ...]


@dataclass(frozen=True)
class ShapeRestoreResult:
    array: np.ndarray
    transform: str
    recon_shape: tuple[int, ...]
    restored_shape: tuple[int, ...]


def _default_attr_registry() -> dict[str, AttrRegistryEntry]:
    root = Path("data/raw").resolve()
    return {
        "target_fort63": AttrRegistryEntry(
            stem="target_fort63",
            dataset_name="ocean",
            gt_path=(root / "ocean/train/target_fort63.npy").resolve(),
            coords_path=(root / "ocean/train/source_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_fort64": AttrRegistryEntry(
            stem="target_fort64",
            dataset_name="ocean",
            gt_path=(root / "ocean/train/target_fort64.npy").resolve(),
            coords_path=(root / "ocean/train/source_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_fort73": AttrRegistryEntry(
            stem="target_fort73",
            dataset_name="ocean",
            gt_path=(root / "ocean/train/target_fort73.npy").resolve(),
            coords_path=(root / "ocean/train/source_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_speed": AttrRegistryEntry(
            stem="target_speed",
            dataset_name="ocean",
            gt_path=(root / "ocean/train/target_speed.npy").resolve(),
            coords_path=(root / "ocean/train/source_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_v": AttrRegistryEntry(
            stem="target_v",
            dataset_name="ocean",
            gt_path=(root / "ocean/train/target_v.npy").resolve(),
            coords_path=(root / "ocean/train/source_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_cell_E": AttrRegistryEntry(
            stem="target_cell_E",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_cell_E.npy").resolve(),
            coords_path=(root / "stress/train/source_cell_XYZT.npy").resolve(),
            preferred_association="cell",
        ),
        "target_cell_E_IntegrationPoints": AttrRegistryEntry(
            stem="target_cell_E_IntegrationPoints",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_cell_E_IntegrationPoints.npy").resolve(),
            coords_path=(root / "stress/train/source_cell_XYZT.npy").resolve(),
            preferred_association="cell",
        ),
        "target_cell_S": AttrRegistryEntry(
            stem="target_cell_S",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_cell_S.npy").resolve(),
            coords_path=(root / "stress/train/source_cell_XYZT.npy").resolve(),
            preferred_association="cell",
        ),
        "target_cell_S_IntegrationPoints": AttrRegistryEntry(
            stem="target_cell_S_IntegrationPoints",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_cell_S_IntegrationPoints.npy").resolve(),
            coords_path=(root / "stress/train/source_cell_XYZT.npy").resolve(),
            preferred_association="cell",
        ),
        "target_point_RF": AttrRegistryEntry(
            stem="target_point_RF",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_point_RF.npy").resolve(),
            coords_path=(root / "stress/train/source_point_XYZT.npy").resolve(),
            preferred_association="point",
        ),
        "target_point_U": AttrRegistryEntry(
            stem="target_point_U",
            dataset_name="stress",
            gt_path=(root / "stress/train/target_point_U.npy").resolve(),
            coords_path=(root / "stress/train/source_point_XYZT.npy").resolve(),
            preferred_association="point",
        ),
    }


ATTR_REGISTRY = _default_attr_registry()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate pre-reconstructed compression results under pred_datasets."
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default="pred_datasets",
        help="Root directory containing method subdirectories with reconstructed files.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated methods to validate. Default: scan all method directories.",
    )
    parser.add_argument(
        "--attrs",
        type=str,
        default="",
        help="Comma-separated target stems to validate. Default: validate all available stems.",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="0-based timestep index. Default: validate all timesteps.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Override render zoom factor. Default: dataset-specific value.",
    )
    parser.add_argument(
        "--clip-max-percentile",
        type=float,
        default=None,
        help="Clip the render color upper bound to this GT percentile.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Output root for rendered images and summary files. Default: ./validate_out/recon",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Metrics CSV output path. Default: <output-root>/recon_metrics.csv",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Manifest JSON output path. Default: <output-root>/pred_datasets_manifest.json",
    )
    return parser.parse_args()


def _normalize_method_name(method_name: str) -> str:
    return str(method_name).strip().upper()


def _canonical_relpath(method_name: str, filename: str) -> str:
    return str(Path(method_name) / filename).replace("\\", "/")


def _expected_files_for_stem(method_name: str, stem: str) -> dict[str, str]:
    ext = _METHOD_COMPRESSED_EXT.get(method_name)
    out = {
        "recon": _canonical_relpath(method_name, f"{stem}_recon.npy"),
        "result": _canonical_relpath(method_name, f"{stem}_result.json"),
    }
    out["compressed"] = _canonical_relpath(method_name, f"{stem}{ext}") if ext else ""
    return out


def _strip_known_suffix(name: str, suffixes: tuple[str, ...]) -> tuple[str, bool] | None:
    for index, suffix in enumerate(suffixes):
        if not name.endswith(suffix):
            continue
        return name[: -len(suffix)], index == 0
    return None


def _parse_artifact_file(method_name: str, path: Path) -> tuple[str, str, bool] | None:
    name = path.name
    parsed = _strip_known_suffix(name, _RECON_SUFFIXES)
    if parsed is not None:
        stem, is_canonical = parsed
        return stem, "recon", is_canonical

    parsed = _strip_known_suffix(name, _RESULT_SUFFIXES)
    if parsed is not None:
        stem, is_canonical = parsed
        return stem, "result", is_canonical

    ext = _METHOD_COMPRESSED_EXT.get(method_name)
    if ext and name.endswith(ext):
        return name[: -len(ext)], "compressed", True
    return None


def _select_candidate(candidates: list[ArtifactCandidate]) -> tuple[Path | None, tuple[Path, ...]]:
    if not candidates:
        return None, ()
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.is_canonical else 1,
            str(item.path).lower(),
        ),
    )
    chosen = ranked[0]
    aliases = tuple(item.path for item in ranked[1:] if item.path != chosen.path)
    return chosen.path.resolve(), aliases


def _discover_method_records(
    method_dir: Path,
    registry: dict[str, AttrRegistryEntry],
) -> dict[str, ArtifactRecord]:
    method_name = _normalize_method_name(method_dir.name)
    discovered: dict[str, dict[str, list[ArtifactCandidate]]] = {
        stem: {"compressed": [], "recon": [], "result": []}
        for stem in registry
    }

    for path in method_dir.rglob("*"):
        if not path.is_file():
            continue
        parsed = _parse_artifact_file(method_name, path)
        if parsed is None:
            continue
        stem, kind, is_canonical = parsed
        if stem not in registry:
            logger.warning("Ignoring unregistered artifact stem '%s' in %s", stem, path)
            continue
        discovered[stem][kind].append(ArtifactCandidate(path=path.resolve(), is_canonical=is_canonical))

    records: dict[str, ArtifactRecord] = {}
    for stem, spec in registry.items():
        compressed_path, compressed_aliases = _select_candidate(discovered[stem]["compressed"])
        recon_path, recon_aliases = _select_candidate(discovered[stem]["recon"])
        result_json_path, result_aliases = _select_candidate(discovered[stem]["result"])
        records[stem] = ArtifactRecord(
            method=method_name,
            spec=spec,
            compressed_path=compressed_path,
            recon_path=recon_path,
            result_json_path=result_json_path,
            compressed_aliases=compressed_aliases,
            recon_aliases=recon_aliases,
            result_aliases=result_aliases,
        )
    return records


def _normalize_requested_methods(pred_root: Path, methods_arg: str | None) -> list[Path]:
    method_dirs = [path.resolve() for path in sorted(pred_root.iterdir()) if path.is_dir()]
    if not methods_arg:
        return method_dirs

    requested = {
        _normalize_method_name(item)
        for item in str(methods_arg).split(",")
        if item.strip()
    }
    selected = [path for path in method_dirs if _normalize_method_name(path.name) in requested]
    missing = sorted(requested - {_normalize_method_name(path.name) for path in selected})
    if missing:
        available = sorted(_normalize_method_name(path.name) for path in method_dirs)
        raise KeyError(f"Unknown methods: {missing}. Available methods: {available}")
    return selected


def _normalize_requested_attrs(attrs_arg: str | None, registry: dict[str, AttrRegistryEntry]) -> list[str]:
    if not attrs_arg:
        return list(sorted(registry.keys()))

    alias_map = {stem: stem for stem in registry}
    alias_map.update({stem[len("target_"):]: stem for stem in registry if stem.startswith("target_")})
    requested: list[str] = []
    unknown: list[str] = []
    for item in (token.strip() for token in str(attrs_arg).split(",") if token.strip()):
        stem = alias_map.get(item)
        if stem is None:
            unknown.append(item)
            continue
        if stem not in requested:
            requested.append(stem)
    if unknown:
        raise KeyError(f"Unknown attrs: {unknown}. Available attrs: {sorted(registry.keys())}")
    return requested


def _read_result_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_replacements(values: np.ndarray, replacements: tuple[tuple[float, float], ...]) -> np.ndarray:
    if not replacements:
        return np.asarray(values)
    out = np.array(values, copy=True)
    for old, new in replacements:
        out[out == old] = new
    return out


def _select_eval_block(
    array: np.ndarray,
    indexer: slice | np.ndarray,
    replacements: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    values = np.asarray(array[indexer], dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if replacements:
        values = _apply_replacements(values, replacements).astype(np.float32, copy=False)
    return values


def _restore_prediction_shape(
    recon_array: np.ndarray,
    gt_shape: tuple[int, ...],
    result_payload: dict[str, Any] | None = None,
) -> ShapeRestoreResult:
    recon_shape = tuple(int(dim) for dim in recon_array.shape)
    target_shape = tuple(int(dim) for dim in gt_shape)
    if recon_shape == target_shape:
        return ShapeRestoreResult(
            array=recon_array,
            transform="identity",
            recon_shape=recon_shape,
            restored_shape=target_shape,
        )

    payload = result_payload or {}
    loaded_shape = tuple(int(dim) for dim in payload.get("loaded_shape") or ())
    used_shape = tuple(int(dim) for dim in payload.get("used_shape") or ())

    if loaded_shape and int(np.prod(loaded_shape)) == int(np.prod(recon_shape)):
        flat = np.reshape(recon_array, (-1,), order="C")
        loaded_view = np.reshape(flat, loaded_shape, order="C")
        if loaded_shape == target_shape:
            return ShapeRestoreResult(
                array=loaded_view,
                transform="reshape_from_used_to_loaded" if used_shape else "reshape_to_loaded",
                recon_shape=recon_shape,
                restored_shape=target_shape,
            )
        if int(np.prod(loaded_shape)) == int(np.prod(target_shape)):
            return ShapeRestoreResult(
                array=np.reshape(loaded_view, target_shape, order="C"),
                transform="reshape_from_used_to_loaded_to_gt" if used_shape else "reshape_loaded_to_gt",
                recon_shape=recon_shape,
                restored_shape=target_shape,
            )

    if int(np.prod(recon_shape)) == int(np.prod(target_shape)):
        return ShapeRestoreResult(
            array=np.reshape(recon_array, target_shape, order="C"),
            transform="reshape_to_gt",
            recon_shape=recon_shape,
            restored_shape=target_shape,
        )

    raise ValueError(
        "Failed to restore reconstructed array shape: "
        f"recon={recon_shape} gt={target_shape} loaded_shape={loaded_shape or None} used_shape={used_shape or None}"
    )


def _default_csv_path(output_root: Path) -> Path:
    return (output_root / "recon_metrics.csv").resolve()


def _default_manifest_path(output_root: Path) -> Path:
    return (output_root / "pred_datasets_manifest.json").resolve()


def _prepare_pred_output_path(
    out_root: Path,
    method: str,
    dataset_name: str,
    attr_name: str,
    time_index: int,
) -> Path:
    return out_root / method / dataset_name / attr_name / f"{method}_{attr_name}_t{time_index:04d}_pred.png"


def _prepare_gt_output_path(
    out_root: Path,
    method: str,
    dataset_name: str,
    attr_name: str,
    time_index: int,
) -> Path:
    return out_root / method / dataset_name / attr_name / f"gt_t{time_index:04d}.png"


def _stringify_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _method_target_mode(payload: dict[str, Any]) -> str:
    mode = payload.get("target_mode")
    if mode is None:
        mode = payload.get("native_mode")
    return "" if mode is None else str(mode)


def _method_target_value(payload: dict[str, Any]) -> Any:
    if payload.get("target_value") is not None:
        return payload["target_value"]
    if payload.get("target_psnr") is not None:
        return payload["target_psnr"]
    if payload.get("native_value") is not None:
        return payload["native_value"]
    return ""


def _write_csv_rows(csv_path: Path, rows: list[dict[str, Any]]):
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _gt_meta_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _read_gt_cache_meta(image_path: Path) -> dict[str, Any] | None:
    meta_path = _gt_meta_path(image_path)
    if not image_path.exists() or not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_gt_cache_meta(
    image_path: Path,
    clim: tuple[float, float],
    mesh_path: Path,
    association: str,
    raw_time: float,
    num_samples: int,
) -> None:
    payload = {
        "cache_version": _GT_CACHE_VERSION,
        "clim": [float(clim[0]), float(clim[1])],
        "mesh_path": str(mesh_path.resolve()),
        "association": str(association),
        "raw_time": float(raw_time),
        "num_samples": int(num_samples),
    }
    _gt_meta_path(image_path).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _can_reuse_gt_cache(
    image_path: Path,
    clim: tuple[float, float],
    mesh_path: Path,
    association: str,
    raw_time: float,
    num_samples: int,
) -> bool:
    payload = _read_gt_cache_meta(image_path)
    if payload is None:
        return False
    if int(payload.get("cache_version", -1)) != _GT_CACHE_VERSION:
        return False
    cached_clim = payload.get("clim")
    if not isinstance(cached_clim, list) or len(cached_clim) != 2:
        return False
    if not np.allclose(np.asarray(cached_clim, dtype=np.float64), np.asarray(clim, dtype=np.float64), rtol=1e-6, atol=1e-12):
        return False
    if str(payload.get("mesh_path", "")) != str(mesh_path.resolve()):
        return False
    if str(payload.get("association", "")) != str(association):
        return False
    if int(payload.get("num_samples", -1)) != int(num_samples):
        return False
    try:
        cached_time = float(payload.get("raw_time"))
    except (TypeError, ValueError):
        return False
    return bool(np.isclose(cached_time, float(raw_time), rtol=1e-9, atol=1e-12))


def _manifest_status_for_record(record: ArtifactRecord) -> str:
    paths = {
        "compressed": record.compressed_path,
        "recon": record.recon_path,
        "result": record.result_json_path,
    }
    missing = [name for name, path in paths.items() if path is None]
    expected = _expected_files_for_stem(record.method, record.spec.stem)
    selected_noncanonical = any(
        path is not None and path.name != Path(expected[kind]).name
        for kind, path in (
            ("compressed", record.compressed_path),
            ("recon", record.recon_path),
            ("result", record.result_json_path),
        )
    )
    has_alias = bool(
        selected_noncanonical
        or record.compressed_aliases
        or record.recon_aliases
        or record.result_aliases
    )
    if not missing and has_alias:
        return "complete_with_aliases"
    if not missing:
        return "complete"
    if len(missing) == len(paths):
        return "missing"
    return "partial"


def _relative_to(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _build_manifest_payload(
    pred_root: Path,
    method_dirs: list[Path],
    registry: dict[str, AttrRegistryEntry],
    selected_attrs: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, ArtifactRecord]]]:
    manifest: dict[str, Any] = {
        "pred_root": str(pred_root.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_attrs": sorted(selected_attrs),
        "methods": {},
    }
    discovered: dict[str, dict[str, ArtifactRecord]] = {}
    for method_dir in method_dirs:
        method_name = _normalize_method_name(method_dir.name)
        records = _discover_method_records(method_dir, registry)
        discovered[method_name] = records
        method_payload: dict[str, Any] = {
            "compressed_ext": _METHOD_COMPRESSED_EXT.get(method_name, ""),
            "available_stems": [],
            "missing_stems": [],
            "records": {},
        }
        for stem, record in records.items():
            expected = _expected_files_for_stem(method_name, stem)
            found_files = {
                "compressed": _relative_to(record.compressed_path, pred_root),
                "recon": _relative_to(record.recon_path, pred_root),
                "result": _relative_to(record.result_json_path, pred_root),
            }
            aliases = {
                "compressed": [_relative_to(path, pred_root) for path in record.compressed_aliases],
                "recon": [_relative_to(path, pred_root) for path in record.recon_aliases],
                "result": [_relative_to(path, pred_root) for path in record.result_aliases],
            }
            missing_files = [
                relpath
                for kind, relpath in expected.items()
                if relpath and not found_files.get(kind)
            ]
            if record.recon_path is not None:
                method_payload["available_stems"].append(stem)
            else:
                method_payload["missing_stems"].append(stem)
            method_payload["records"][stem] = {
                "dataset_name": record.spec.dataset_name,
                "selected_for_validation": stem in selected_attrs,
                "status": _manifest_status_for_record(record),
                "expected_files": expected,
                "found_files": found_files,
                "aliases": aliases,
                "missing_files": missing_files,
            }
        manifest["methods"][method_name] = method_payload
    return manifest, discovered


def _write_manifest(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _selected_time_steps(
    coords_path: Path,
    timestamp: int | None,
    preferred_association: str,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    coords = np.load(str(coords_path), mmap_mode="r", allow_pickle=False)
    if coords.ndim != 2 or coords.shape[1] < 1:
        raise ValueError(f"Expected coords array with shape (N, D>=1), got {coords.shape}")

    time_values = coords[:, -1]
    time_indexers = _compute_time_indexers(time_values)
    num_timesteps = len(time_indexers)
    if num_timesteps == 0:
        raise ValueError(f"No timesteps found in coords array: {coords_path}")

    if timestamp is not None:
        if timestamp < 0 or timestamp >= num_timesteps:
            raise ValueError(
                f"--timestamp {timestamp} is out of range for {coords_path}. "
                f"Valid range: [0, {num_timesteps - 1}]"
            )
        selected_time_indices = [int(timestamp)]
    else:
        selected_time_indices = _select_default_timestamps(num_timesteps)

    candidate_meshes = _collect_mesh_candidates(coords_path)
    steps: list[dict[str, Any]] = []
    for time_index in selected_time_indices:
        raw_time, indexer = time_indexers[time_index]
        sample_count = _indexer_size(indexer)
        mesh_path, association = _resolve_mesh_for_timestep(
            raw_time=float(raw_time),
            sample_count=sample_count,
            candidate_paths=candidate_meshes,
            preferred_association=preferred_association,
        )
        steps.append(
            {
                "time_index": int(time_index),
                "raw_time": float(raw_time),
                "indexer": indexer,
                "sample_count": int(sample_count),
                "mesh_path": mesh_path,
                "mesh_association": association,
            }
        )
    return coords, steps, num_timesteps


def _close_array_handles(*arrays: object) -> None:
    seen: set[int] = set()
    for array in arrays:
        current = array
        while isinstance(current, np.ndarray):
            mmap_obj = getattr(current, "_mmap", None)
            if mmap_obj is not None and id(mmap_obj) not in seen:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
                seen.add(id(mmap_obj))
            base = getattr(current, "base", None)
            if base is None or base is current:
                break
            current = base


def _validate_record(
    record: ArtifactRecord,
    output_root: Path,
    timestamp: int | None,
    zoom: float | None,
    clip_max_percentile: float | None,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if record.recon_path is None:
        raise FileNotFoundError(f"No reconstruction file found for {record.method}/{record.spec.stem}")

    gt_array = np.load(str(record.spec.gt_path), mmap_mode="r", allow_pickle=False)
    recon_array = np.load(str(record.recon_path), mmap_mode="r", allow_pickle=False)
    coords = None
    restored = None
    try:
        result_payload = _read_result_payload(record.result_json_path)
        restored = _restore_prediction_shape(recon_array, tuple(int(dim) for dim in gt_array.shape), result_payload)
        coords, selected_steps, num_timesteps = _selected_time_steps(
            record.spec.coords_path,
            timestamp=timestamp,
            preferred_association=record.spec.preferred_association,
        )
        if int(coords.shape[0]) != int(gt_array.shape[0]):
            raise ValueError(
                f"Coords / GT length mismatch for {record.spec.stem}: coords={coords.shape[0]} gt={gt_array.shape[0]}"
            )
        if int(coords.shape[0]) != int(restored.array.shape[0]):
            raise ValueError(
                f"Coords / recon length mismatch for {record.spec.stem}: coords={coords.shape[0]} "
                f"recon={restored.array.shape[0]}"
            )

        dataset_name = record.spec.dataset_name
        zoom_factor = float(zoom) if zoom is not None else _render_zoom_factor({"dataset_name": dataset_name})
        raw_replacements = record.spec.raw_replacements

        logger.info(
            "Evaluating %s/%s from %s with shape transform=%s (%s -> %s)",
            record.method,
            record.spec.stem,
            record.recon_path,
            restored.transform,
            restored.recon_shape,
            restored.restored_shape,
        )

        attr_clim: tuple[float, float] | None = None
        gt_visual_blocks: list[np.ndarray] = []
        psnr_map: dict[int, float] = {}

        for step in selected_steps:
            gt_values = _select_eval_block(gt_array, step["indexer"], replacements=raw_replacements)
            pred_values = _select_eval_block(restored.array, step["indexer"], replacements=raw_replacements)
            gt_values, pred_values = _align_eval_shapes(gt_values, pred_values, record.spec.stem)
            gt_vis = _to_visual_scalar(gt_values)
            attr_clim = _merge_range(attr_clim, gt_vis)
            if clip_max_percentile is not None:
                gt_visual_blocks.append(np.asarray(gt_vis, dtype=np.float32).reshape(-1).copy())
            psnr_map[int(step["time_index"])] = _compute_psnr(gt_values, pred_values)

        normalized_clim = _normalize_clim(attr_clim)
        normalized_clim = _clip_upper_clim(normalized_clim, gt_visual_blocks, clip_max_percentile)

        per_timestep_rows: list[dict[str, Any]] = []
        for step in selected_steps:
            gt_values = _select_eval_block(gt_array, step["indexer"], replacements=raw_replacements)
            pred_values = _select_eval_block(restored.array, step["indexer"], replacements=raw_replacements)
            gt_values, pred_values = _align_eval_shapes(gt_values, pred_values, record.spec.stem)
            gt_vis = _to_visual_scalar(gt_values)
            pred_vis = _to_visual_scalar(pred_values)

            pred_out = _prepare_pred_output_path(
                output_root,
                record.method,
                dataset_name,
                record.spec.stem,
                int(step["time_index"]),
            )
            gt_out = _prepare_gt_output_path(
                output_root,
                record.method,
                dataset_name,
                record.spec.stem,
                int(step["time_index"]),
            )

            pred_img = _render_frame(
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                values=pred_vis,
                outpath=pred_out,
                clim=normalized_clim,
                zoom_factor=zoom_factor,
            )
            if _can_reuse_gt_cache(
                gt_out,
                normalized_clim,
                mesh_path=step["mesh_path"],
                association=step["mesh_association"],
                raw_time=float(step["raw_time"]),
                num_samples=int(step["sample_count"]),
            ):
                gt_img = _load_image(gt_out)
            else:
                gt_img = _render_frame(
                    mesh_path=step["mesh_path"],
                    association=step["mesh_association"],
                    values=gt_vis,
                    outpath=gt_out,
                    clim=normalized_clim,
                    zoom_factor=zoom_factor,
                )
                _write_gt_cache_meta(
                    gt_out,
                    normalized_clim,
                    mesh_path=step["mesh_path"],
                    association=step["mesh_association"],
                    raw_time=float(step["raw_time"]),
                    num_samples=int(step["sample_count"]),
                )

            ssim_value = _compute_ssim(gt_img, pred_img)
            lpips_value = _compute_lpips(lpips_model, gt_img, pred_img, device)
            per_timestep_rows.append(
                {
                    "row_type": "per_timestep",
                    "method": record.method,
                    "dataset_name": dataset_name,
                    "attr": record.spec.stem,
                    "time_index": int(step["time_index"]),
                    "raw_time": float(step["raw_time"]),
                    "num_samples": int(step["sample_count"]),
                    "num_timesteps": int(num_timesteps),
                    "gt_render_path": str(gt_out.resolve()),
                    "pred_render_path": str(pred_out.resolve()),
                    "psnr": psnr_map[int(step["time_index"])],
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                    "recon_path": str(record.recon_path.resolve()) if record.recon_path else "",
                    "compressed_path": str(record.compressed_path.resolve()) if record.compressed_path else "",
                    "result_json_path": str(record.result_json_path.resolve()) if record.result_json_path else "",
                    "compression_ratio": result_payload.get("compression_ratio", ""),
                    "target_mode": _method_target_mode(result_payload),
                    "target_value": _method_target_value(result_payload),
                    "shape_transform": restored.transform,
                    "recon_shape": _stringify_shape(restored.recon_shape),
                    "restored_shape": _stringify_shape(restored.restored_shape),
                }
            )

        summary_rows = [
            {
                "row_type": "attr_mean",
                "method": record.method,
                "dataset_name": dataset_name,
                "attr": record.spec.stem,
                "time_index": "",
                "raw_time": "",
                "num_samples": "",
                "num_timesteps": int(num_timesteps),
                "gt_render_path": "",
                "pred_render_path": "",
                "psnr": _mean_finite([float(row["psnr"]) for row in per_timestep_rows]),
                "ssim": _mean_finite([float(row["ssim"]) for row in per_timestep_rows]),
                "lpips": _mean_finite([float(row["lpips"]) for row in per_timestep_rows]),
                "recon_path": str(record.recon_path.resolve()) if record.recon_path else "",
                "compressed_path": str(record.compressed_path.resolve()) if record.compressed_path else "",
                "result_json_path": str(record.result_json_path.resolve()) if record.result_json_path else "",
                "compression_ratio": result_payload.get("compression_ratio", ""),
                "target_mode": _method_target_mode(result_payload),
                "target_value": _method_target_value(result_payload),
                "shape_transform": restored.transform,
                "recon_shape": _stringify_shape(restored.recon_shape),
                "restored_shape": _stringify_shape(restored.restored_shape),
            }
        ]
        return per_timestep_rows, summary_rows
    finally:
        _close_array_handles(
            gt_array,
            recon_array,
            getattr(restored, "array", None) if restored is not None else None,
            coords,
        )


def validate_pred_root(
    pred_root: str | Path = "pred_datasets",
    methods: str | None = None,
    attrs: str | None = None,
    timestamp: int | None = None,
    zoom: float | None = None,
    clip_max_percentile: float | None = None,
    output_root: str | Path | None = None,
    csv_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    pred_root_path = Path(pred_root).resolve()
    if not pred_root_path.exists() or not pred_root_path.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {pred_root_path}")

    if zoom is not None and float(zoom) <= 0.0:
        raise ValueError(f"--zoom must be positive, got {zoom}")
    if clip_max_percentile is not None and not (0.0 < float(clip_max_percentile) <= 100.0):
        raise ValueError(
            f"--clip-max-percentile must be in (0, 100], got {clip_max_percentile}"
        )

    selected_attrs = set(_normalize_requested_attrs(attrs, ATTR_REGISTRY))
    method_dirs = _normalize_requested_methods(pred_root_path, methods)

    output_root_path = Path(output_root).resolve() if output_root else (Path("validate_out") / "recon").resolve()
    resolved_csv_path = Path(csv_path).resolve() if csv_path else _default_csv_path(output_root_path)
    resolved_manifest_path = Path(manifest_path).resolve() if manifest_path else _default_manifest_path(output_root_path)

    manifest_payload, discovered = _build_manifest_payload(
        pred_root=pred_root_path,
        method_dirs=method_dirs,
        registry=ATTR_REGISTRY,
        selected_attrs=selected_attrs,
    )
    _write_manifest(resolved_manifest_path, manifest_payload)
    logger.info("Saved prediction manifest to %s", resolved_manifest_path)

    _ensure_runtime_dependencies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = _build_lpips_model(device)

    per_timestep_rows: list[dict[str, Any]] = []
    attr_summary_rows: list[dict[str, Any]] = []
    global_summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for method_dir in method_dirs:
        method_name = _normalize_method_name(method_dir.name)
        method_records = discovered[method_name]
        method_per_timestep_rows: list[dict[str, Any]] = []
        for stem in sorted(selected_attrs):
            record = method_records[stem]
            if record.recon_path is None:
                logger.info("Skipping %s/%s because no recon file is present.", method_name, stem)
                continue
            try:
                record_per_timestep, record_summary = _validate_record(
                    record=record,
                    output_root=output_root_path,
                    timestamp=timestamp,
                    zoom=zoom,
                    clip_max_percentile=clip_max_percentile,
                    lpips_model=lpips_model,
                    device=device,
                )
            except Exception as exc:
                failures.append(f"{method_name}/{stem}: {exc}")
                logger.exception("Failed to validate %s/%s", method_name, stem)
                continue
            per_timestep_rows.extend(record_per_timestep)
            attr_summary_rows.extend(record_summary)
            method_per_timestep_rows.extend(record_per_timestep)

        if method_per_timestep_rows:
            global_summary_rows.append(
                {
                    "row_type": "global_mean",
                    "method": method_name,
                    "dataset_name": "__all__",
                    "attr": "__all__",
                    "time_index": "",
                    "raw_time": "",
                    "num_samples": "",
                    "num_timesteps": "",
                    "gt_render_path": "",
                    "pred_render_path": "",
                    "psnr": _mean_finite([float(row["psnr"]) for row in method_per_timestep_rows]),
                    "ssim": _mean_finite([float(row["ssim"]) for row in method_per_timestep_rows]),
                    "lpips": _mean_finite([float(row["lpips"]) for row in method_per_timestep_rows]),
                    "recon_path": "",
                    "compressed_path": "",
                    "result_json_path": "",
                    "compression_ratio": "",
                    "target_mode": "",
                    "target_value": "",
                    "shape_transform": "",
                    "recon_shape": "",
                    "restored_shape": "",
                }
            )

    rows = per_timestep_rows + attr_summary_rows + global_summary_rows
    if not rows:
        failure_text = "; ".join(failures) if failures else "No reconstruction files matched the requested filters."
        raise RuntimeError(f"No validation results were generated. {failure_text}")

    _write_csv_rows(resolved_csv_path, rows)
    logger.info("Saved reconstruction validation metrics to %s", resolved_csv_path)
    if failures:
        logger.warning("Validation completed with %d failure(s).", len(failures))
    return resolved_csv_path, resolved_manifest_path


def main():
    setup_logging()
    args = _parse_args()
    validate_pred_root(
        pred_root=args.pred_root,
        methods=(args.methods or None),
        attrs=(args.attrs or None),
        timestamp=args.timestamp,
        zoom=args.zoom,
        clip_max_percentile=args.clip_max_percentile,
        output_root=(args.output_root or None),
        csv_path=(args.csv or None),
        manifest_path=(args.manifest or None),
    )


if __name__ == "__main__":
    main()

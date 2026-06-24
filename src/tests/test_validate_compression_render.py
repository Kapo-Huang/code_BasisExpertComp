import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compression_cli_common import build_tthresh_decompress_command
from validate_compression_render import (
    CompressionArtifact,
    RuntimePaths,
    VolumeShape,
    build_timestep_contexts,
    canonicalize_case_name,
    discover_artifacts,
    extract_flat_timestep_from_memmap,
    load_validation_module,
    parse_artifact,
    parse_args,
    resolve_runtime_paths,
    resolve_local_compressed_path,
)


def write_result_json(
    path: Path,
    *,
    method: str,
    compressed: str,
    dtype: str = "float32",
    used_shape: list[int] | None = None,
    compression_ratio: float = 123.5,
) -> None:
    payload = {
        "method": method,
        "input": "/stale/server/input.npy",
        "compressed": compressed,
        "recon": "/stale/server/recon.npy",
        "loaded_shape": [24, 1],
        "used_shape": used_shape or [2, 3, 1, 4],
        "dtype": dtype,
        "compression_ratio": compression_ratio,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_artifact(tmp_path: Path) -> CompressionArtifact:
    compressed_path = tmp_path / "target_GT.sz3pkg"
    compressed_path.write_bytes(b"sz3")
    result_json_path = tmp_path / "target_GT_SZ3_result.json"
    write_result_json(result_json_path, method="sz3", compressed="/stale/server/target_GT.sz3pkg")
    return parse_artifact(result_json_path, tmp_path)


def test_canonicalize_case_name_normalizes_hplus_alias() -> None:
    assert canonicalize_case_name("HPlus") == "H+"
    assert canonicalize_case_name("H+") == "H+"


def test_resolve_local_compressed_path_ignores_stale_server_path(tmp_path: Path) -> None:
    local_artifact = tmp_path / "target_GT.zfp"
    local_artifact.write_bytes(b"zfp")

    resolved = resolve_local_compressed_path(
        compressed_value="/mnt/server/run/target_GT.zfp",
        artifacts_root=tmp_path,
        case_token="GT",
        method_key="zfp",
    )

    assert resolved == local_artifact.resolve()


def test_parse_artifact_normalizes_hplus_and_resolves_local_payload(tmp_path: Path) -> None:
    local_artifact = tmp_path / "target_HPlus.tthresh"
    local_artifact.write_bytes(b"tthresh")
    result_json_path = tmp_path / "target_HPlus_TTHRESH_result.json"
    write_result_json(
        result_json_path,
        method="tthresh",
        compressed="/mnt/server/run/target_HPlus.tthresh",
    )

    artifact = parse_artifact(result_json_path, tmp_path)

    assert artifact.case_token == "HPlus"
    assert artifact.case_name == "H+"
    assert artifact.method_key == "tthresh"
    assert artifact.method_display == "TTHRESH"
    assert artifact.method_name == "TTHRESH_Ionization_H+"
    assert artifact.compressed_path == local_artifact.resolve()
    assert artifact.compression_ratio == 123.5


def test_discover_artifacts_reads_output_layout_and_filters(tmp_path: Path) -> None:
    (tmp_path / "target_GT.sz3pkg").write_bytes(b"sz3")
    write_result_json(
        tmp_path / "target_GT_SZ3_result.json",
        method="sz3",
        compressed="/stale/server/target_GT.sz3pkg",
    )

    (tmp_path / "target_HPlus.zfp").write_bytes(b"zfp")
    write_result_json(
        tmp_path / "target_HPlus_ZFP_result.json",
        method="zfp",
        compressed="/stale/server/target_HPlus.zfp",
    )

    artifacts = discover_artifacts(tmp_path, case_filter=set(), method_filter=set())
    filtered = discover_artifacts(tmp_path, case_filter={"H+"}, method_filter={"zfp"})

    assert [(artifact.case_name, artifact.method_key) for artifact in artifacts] == [
        ("GT", "sz3"),
        ("H+", "zfp"),
    ]
    assert len(filtered) == 1
    assert filtered[0].case_name == "H+"
    assert filtered[0].method_display == "ZFP"


def test_extract_flat_timestep_from_memmap_returns_contiguous_block(tmp_path: Path) -> None:
    raw_path = tmp_path / "decompressed.raw"
    expected = np.arange(24, dtype=np.float32)
    expected.tofile(raw_path)
    memmap = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(24,))

    try:
        timestep = extract_flat_timestep_from_memmap(memmap, VolumeShape(X=2, Y=3, Z=1, T=4), 2)
    finally:
        memmap._mmap.close()

    assert np.array_equal(timestep, np.arange(12, 18, dtype=np.float32))


def test_build_timestep_contexts_requires_decode_when_cached_row_paths_mismatch(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path)
    method_dir = tmp_path / artifact.case_name / artifact.method_name
    gt_png_dir = tmp_path / artifact.case_name
    method_dir.mkdir(parents=True, exist_ok=True)
    gt_png_dir.mkdir(parents=True, exist_ok=True)

    pred_png_path = method_dir / "pred_t0000.png"
    gt_png_path = gt_png_dir / "GT_GT_0.png"
    pred_png_path.write_bytes(b"png")
    gt_png_path.write_bytes(b"png")

    contexts = build_timestep_contexts(
        artifact=artifact,
        timesteps=[0],
        method_dir=method_dir,
        gt_png_dir=gt_png_dir,
        cached_metric_rows={
            0: {
                "method": artifact.method_name,
                "case": artifact.case_name,
                "pred_path": str(tmp_path / "stale_pred.png"),
                "gt_path": str(gt_png_path),
                "psnr": 42.0,
                "ssim": 0.99,
                "lpips": 0.01,
                "status": "ok",
            }
        },
    )

    assert len(contexts) == 1
    assert contexts[0].cached_row is None
    assert contexts[0].needs_decode is True
    assert contexts[0].can_reuse_complete is False


def test_build_tthresh_decompress_command_uses_decode_only_flags() -> None:
    command = build_tthresh_decompress_command(
        Path("tthresh"),
        Path("input.tthresh"),
        Path("output.raw"),
    )

    assert command == [
        "tthresh",
        "-c",
        "input.tthresh",
        "-o",
        "output.raw",
    ]


def test_resolve_runtime_paths_defaults_tmp_and_viewport_root(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    gt_root = tmp_path / "gt"
    result_root = tmp_path / "results"
    render_script = tmp_path / "render_task.py"
    image_validation_script = tmp_path / "image_level_validation.py"
    transfer_function_root = tmp_path / "render_config"

    artifacts_root.mkdir()
    gt_root.mkdir()
    result_root.mkdir()
    (transfer_function_root / "GT").mkdir(parents=True)
    render_script.write_text("print('render')\n", encoding="utf-8")
    image_validation_script.write_text(
        "lpips = object()\n"
        "def validate_image_pair(*args, **kwargs):\n"
        "    return {'ssim': 1.0, 'lpips': 0.0}\n",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--artifacts-root",
            str(artifacts_root),
            "--gt-root",
            str(gt_root),
            "--result-root",
            str(result_root),
            "--render-script",
            str(render_script),
            "--image-validation-script",
            str(image_validation_script),
            "--transfer-function-root",
            str(transfer_function_root),
        ]
    )

    runtime_paths = resolve_runtime_paths(args)

    assert isinstance(runtime_paths, RuntimePaths)
    assert runtime_paths.artifacts_root == artifacts_root.resolve()
    assert runtime_paths.gt_root == gt_root.resolve()
    assert runtime_paths.result_root == result_root.resolve()
    assert runtime_paths.tmp_root == (result_root / ".tmp" / "compression_render").resolve()
    assert runtime_paths.transfer_function_root == transfer_function_root.resolve()
    assert runtime_paths.viewport_root == transfer_function_root.resolve()


def test_resolve_runtime_paths_rejects_missing_render_script(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    gt_root = tmp_path / "gt"
    result_root = tmp_path / "results"
    image_validation_script = tmp_path / "image_level_validation.py"
    transfer_function_root = tmp_path / "render_config"

    artifacts_root.mkdir()
    gt_root.mkdir()
    result_root.mkdir()
    transfer_function_root.mkdir()
    image_validation_script.write_text(
        "lpips = object()\n"
        "def validate_image_pair(*args, **kwargs):\n"
        "    return {'ssim': 1.0, 'lpips': 0.0}\n",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--artifacts-root",
            str(artifacts_root),
            "--gt-root",
            str(gt_root),
            "--result-root",
            str(result_root),
            "--render-script",
            str(tmp_path / "missing_render_task.py"),
            "--image-validation-script",
            str(image_validation_script),
            "--transfer-function-root",
            str(transfer_function_root),
        ]
    )

    with pytest.raises(FileNotFoundError, match="--render-script"):
        resolve_runtime_paths(args)


def test_load_validation_module_uses_explicit_path(tmp_path: Path) -> None:
    validation_script = tmp_path / "image_level_validation.py"
    validation_script.write_text(
        "lpips = object()\n"
        "def validate_image_pair(*args, **kwargs):\n"
        "    return {'ssim': 0.9, 'lpips': 0.1}\n",
        encoding="utf-8",
    )

    module = load_validation_module(validation_script)

    assert callable(module.validate_image_pair)
    assert module.validate_image_pair("gt.png", "pred.png", use_lpips=True) == {"ssim": 0.9, "lpips": 0.1}

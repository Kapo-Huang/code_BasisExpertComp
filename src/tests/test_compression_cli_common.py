import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compression_cli_common import (
    CommandResult,
    build_result,
    build_sz3_compress_command,
    build_tthresh_command,
    build_zfp_compress_command,
    compute_metrics,
    get_zfp_dtype_flag,
    load_array,
    load_psnr_config,
    prepare_command_env,
    load_zfp_config,
    tthresh_native_psnr,
    zfp_tolerance_from_psnr,
)


def test_compute_metrics_uses_full_range_psnr():
    original = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    reconstructed = np.array([0.0, 1.0, 2.0, 2.0], dtype=np.float32)

    metrics = compute_metrics(original, reconstructed)

    assert metrics["mse"] == pytest.approx(0.25)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["max_error"] == pytest.approx(1.0)
    assert metrics["data_range"] == pytest.approx(3.0)
    assert metrics["measured_psnr"] == pytest.approx(20.0 * math.log10(3.0 / 0.5))


def test_tthresh_native_psnr_applies_half_range_offset():
    native_psnr = tthresh_native_psnr(40.0)
    assert native_psnr == pytest.approx(33.979400086720375)


def test_zfp_tolerance_uses_direct_psnr_conversion():
    tolerance = zfp_tolerance_from_psnr(200.0, 40.0)
    assert tolerance == pytest.approx(2.0)


def test_load_psnr_config_rejects_legacy_keys(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "compressed": "out.zfp",
                "recon": "recon.npy",
                "rate": 4,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported config keys: rate"):
        load_psnr_config(tmp_path, config_path.name, "zfp")


def test_load_psnr_config_returns_target_descriptor(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "sz3": "build/bin/sz3",
                "psnr": 40.0,
                "compressed": "out.sz",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    config = load_psnr_config(tmp_path, config_path.name, "sz3")

    assert config["target_mode"] == "psnr"
    assert config["target_value"] == pytest.approx(40.0)
    assert config["target_psnr"] == pytest.approx(40.0)


def test_load_psnr_config_rejects_tolerance_key(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "sz3": "build/bin/sz3",
                "tolerance": 0.01,
                "compressed": "out.sz",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tolerance"):
        load_psnr_config(tmp_path, config_path.name, "sz3")


def test_load_psnr_config_rejects_rate_key(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "sz3": "build/bin/sz3",
                "rate": 8.0,
                "compressed": "out.sz",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported config keys: rate"):
        load_psnr_config(tmp_path, config_path.name, "sz3")


def test_load_zfp_config_accepts_psnr(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "psnr": 40.0,
                "compressed": "out.zfp",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    config = load_zfp_config(tmp_path, config_path.name)

    assert config["target_mode"] == "psnr"
    assert config["target_value"] == pytest.approx(40.0)
    assert config["target_psnr"] == pytest.approx(40.0)


def test_load_zfp_config_accepts_tolerance(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "tolerance": 0.0125,
                "compressed": "out.zfp",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    config = load_zfp_config(tmp_path, config_path.name)

    assert config["target_mode"] == "tolerance"
    assert config["target_value"] == pytest.approx(0.0125)
    assert config["target_psnr"] is None


def test_load_zfp_config_accepts_rate(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "rate": 8.5,
                "compressed": "out.zfp",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    config = load_zfp_config(tmp_path, config_path.name)

    assert config["target_mode"] == "rate"
    assert config["target_value"] == pytest.approx(8.5)
    assert config["target_psnr"] is None


@pytest.mark.parametrize(
    "payload",
    (
        {
            "input": "input.npy",
            "zfp": "build/bin/zfp",
            "compressed": "out.zfp",
            "recon": "recon.npy",
        },
        {
            "input": "input.npy",
            "zfp": "build/bin/zfp",
            "psnr": 40.0,
            "tolerance": 0.01,
            "compressed": "out.zfp",
            "recon": "recon.npy",
        },
        {
            "input": "input.npy",
            "zfp": "build/bin/zfp",
            "psnr": 40.0,
            "rate": 8.0,
            "compressed": "out.zfp",
            "recon": "recon.npy",
        },
        {
            "input": "input.npy",
            "zfp": "build/bin/zfp",
            "tolerance": 0.01,
            "rate": 8.0,
            "compressed": "out.zfp",
            "recon": "recon.npy",
        },
        {
            "input": "input.npy",
            "zfp": "build/bin/zfp",
            "psnr": 40.0,
            "tolerance": 0.01,
            "rate": 8.0,
            "compressed": "out.zfp",
            "recon": "recon.npy",
        },
    ),
)
def test_load_zfp_config_requires_exactly_one_target_key(tmp_path: Path, payload: dict[str, object]):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one of 'psnr', 'tolerance', or 'rate'"):
        load_zfp_config(tmp_path, config_path.name)


@pytest.mark.parametrize("tolerance", (0.0, -0.1))
def test_load_zfp_config_rejects_non_positive_tolerance(tmp_path: Path, tolerance: float):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "tolerance": tolerance,
                "compressed": "out.zfp",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive finite"):
        load_zfp_config(tmp_path, config_path.name)


@pytest.mark.parametrize("rate", (0.0, -0.1, float("inf"), float("nan")))
def test_load_zfp_config_rejects_invalid_rate(tmp_path: Path, rate: float):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": "input.npy",
                "zfp": "build/bin/zfp",
                "rate": rate,
                "compressed": "out.zfp",
                "recon": "recon.npy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive finite"):
        load_zfp_config(tmp_path, config_path.name)


def test_load_array_reshapes_when_shape_is_provided(tmp_path: Path):
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.arange(12, dtype=np.float32))

    array, loaded_shape, used_shape = load_array(input_path, (3, 4))

    assert loaded_shape == (12,)
    assert used_shape == (3, 4)
    assert array.shape == (3, 4)


def test_prepare_command_env_prepends_runtime_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    tool_path = tool_dir / "codec.exe"
    tool_path.write_bytes(b"")

    conda_prefix = tmp_path / "conda-env"
    conda_bin = conda_prefix / "Library" / "bin"
    conda_mingw = conda_prefix / "Library" / "mingw-w64" / "bin"
    conda_bin.mkdir(parents=True)
    conda_mingw.mkdir(parents=True)

    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))

    env = prepare_command_env((tool_path,))
    path_entries = env["PATH"].split(";")

    assert path_entries[0] == str(tool_dir.resolve())
    assert str(conda_bin.resolve()) in path_entries
    assert str(conda_mingw.resolve()) in path_entries
    assert r"C:\Windows\System32" in path_entries


def test_command_builders_use_expected_native_flags():
    sz3_command = build_sz3_compress_command(
        Path("sz3"),
        Path("input.raw"),
        Path("payload.sz"),
        ["-f"],
        (8, 8, 8),
        42.0,
    )
    tthresh_command = build_tthresh_command(
        Path("tthresh"),
        Path("input.raw"),
        Path("out.tthresh"),
        Path("recon.raw"),
        "float",
        (8, 8, 8),
        35.0,
    )
    zfp_command = build_zfp_compress_command(
        Path("zfp"),
        Path("input.raw"),
        Path("out.zfp"),
        get_zfp_dtype_flag(np.float32)[0],
        (8, 8, 8),
        "accuracy",
        0.001,
    )
    zfp_rate_command = build_zfp_compress_command(
        Path("zfp"),
        Path("input.raw"),
        Path("out.zfp"),
        get_zfp_dtype_flag(np.float32)[0],
        (8, 8, 8),
        "rate",
        8.0,
    )

    assert "-M" in sz3_command and "PSNR" in sz3_command
    assert "-p" in tthresh_command and "35" in tthresh_command
    assert "-a" in zfp_command and "-h" in zfp_command
    assert "-r" in zfp_rate_command and "-h" in zfp_rate_command


def test_build_zfp_compress_command_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported ZFP native mode"):
        build_zfp_compress_command(
            Path("zfp"),
            Path("input.raw"),
            Path("out.zfp"),
            get_zfp_dtype_flag(np.float32)[0],
            (8, 8, 8),
            "precision",
            12.0,
        )


def test_build_result_emits_unified_schema(tmp_path: Path):
    compressed_path = tmp_path / "artifact.bin"
    compressed_path.write_bytes(b"1234")
    recon_path = tmp_path / "recon.npy"

    result = build_result(
        method="zfp",
        input_path=tmp_path / "input.npy",
        compressed_path=compressed_path,
        recon_path=recon_path,
        loaded_shape=(4,),
        used_shape=(2, 2),
        dtype_label="float32",
        target_mode="psnr",
        target_value=40.0,
        target_psnr=40.0,
        native_mode="accuracy",
        native_value=0.01,
        original=np.ones((2, 2), dtype=np.float32),
        reconstructed=np.ones((2, 2), dtype=np.float32),
        compress_result=CommandResult(("zfp",), "compress", "", 0),
        decompress_result=CommandResult(("zfp",), "decompress", "", 0),
    )

    assert set(result) == {
        "method",
        "input",
        "compressed",
        "recon",
        "loaded_shape",
        "used_shape",
        "dtype",
        "target_mode",
        "target_value",
        "target_psnr",
        "native_mode",
        "native_value",
        "measured_psnr",
        "mse",
        "rmse",
        "max_error",
        "original_nbytes",
        "compressed_nbytes",
        "compression_ratio",
        "compression_time_seconds",
        "decompression_time_seconds",
        "total_time_seconds",
        "compress_stdout",
        "compress_stderr",
        "decompress_stdout",
        "decompress_stderr",
    }
    assert result["target_mode"] == "psnr"
    assert result["target_value"] == pytest.approx(40.0)
    assert result["target_psnr"] == pytest.approx(40.0)


def test_build_result_allows_null_target_psnr(tmp_path: Path):
    compressed_path = tmp_path / "artifact.bin"
    compressed_path.write_bytes(b"1234")

    result = build_result(
        method="zfp",
        input_path=tmp_path / "input.npy",
        compressed_path=compressed_path,
        recon_path=tmp_path / "recon.npy",
        loaded_shape=(2,),
        used_shape=(2,),
        dtype_label="float32",
        target_mode="tolerance",
        target_value=0.01,
        target_psnr=None,
        native_mode="accuracy",
        native_value=0.01,
        original=np.ones((2,), dtype=np.float32),
        reconstructed=np.ones((2,), dtype=np.float32),
        compress_result=CommandResult(("zfp",), "", "", 0),
        decompress_result=CommandResult(("zfp",), "", "", 0),
    )

    assert result["target_mode"] == "tolerance"
    assert result["target_value"] == pytest.approx(0.01)
    assert result["target_psnr"] is None


def test_build_result_includes_timing_fields(tmp_path: Path):
    compressed_path = tmp_path / "artifact.bin"
    compressed_path.write_bytes(b"1234")

    result = build_result(
        method="sz3",
        input_path=tmp_path / "input.npy",
        compressed_path=compressed_path,
        recon_path=tmp_path / "recon.npy",
        loaded_shape=(2,),
        used_shape=(2,),
        dtype_label="float32",
        target_mode="psnr",
        target_value=40.0,
        target_psnr=40.0,
        native_mode="psnr",
        native_value=40.0,
        original=np.ones((2,), dtype=np.float32),
        reconstructed=np.ones((2,), dtype=np.float32),
        compress_result=CommandResult(("sz3",), "", "", 0, elapsed_seconds=1.25),
        decompress_result=CommandResult(("sz3",), "", "", 0, elapsed_seconds=0.75),
    )

    assert result["compression_time_seconds"] == pytest.approx(1.25)
    assert result["decompression_time_seconds"] == pytest.approx(0.75)
    assert result["total_time_seconds"] == pytest.approx(2.0)

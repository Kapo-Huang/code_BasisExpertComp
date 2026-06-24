import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compression_cli_common import CommandResult
import zfp.zfp_cli as zfp_cli


def test_zfp_cli_uses_direct_tolerance_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    input_array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    input_path = tmp_path / "input.npy"
    compressed_path = tmp_path / "out.zfp"
    recon_path = tmp_path / "recon.npy"
    result_json_path = tmp_path / "result.json"
    np.save(input_path, input_array)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": str(input_path),
                "zfp": str(tmp_path / "fake-zfp"),
                "tolerance": 0.125,
                "shape": [2, 2, 2],
                "compressed": str(compressed_path),
                "recon": str(recon_path),
                "result_json": str(result_json_path),
            }
        ),
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fail_psnr_conversion(_data_range: float, _target_psnr: float) -> float:
        raise AssertionError("PSNR conversion should not run for tolerance mode")

    def fake_run_command(command: list[str]) -> CommandResult:
        commands.append(command)
        if "-i" in command:
            compressed_path.write_bytes(b"zfp")
            return CommandResult(tuple(command), "", "", 0, elapsed_seconds=0.1)

        input_array.tofile(Path(command[command.index("-o") + 1]))
        return CommandResult(tuple(command), "", "", 0, elapsed_seconds=0.05)

    monkeypatch.setattr(zfp_cli, "zfp_tolerance_from_psnr", fail_psnr_conversion)
    monkeypatch.setattr(zfp_cli, "run_command", fake_run_command)
    monkeypatch.setattr(sys, "argv", ["zfp_cli.py", "--config", str(config_path)])

    exit_code = zfp_cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert len(commands) == 2
    assert commands[0][commands[0].index("-a") + 1] == "0.125"

    result = json.loads(captured.out)
    assert result["target_mode"] == "tolerance"
    assert result["target_value"] == pytest.approx(0.125)
    assert result["target_psnr"] is None
    assert result["native_mode"] == "accuracy"
    assert result["native_value"] == pytest.approx(0.125)
    assert json.loads(result_json_path.read_text(encoding="utf-8"))["target_mode"] == "tolerance"


def test_zfp_cli_uses_direct_rate_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    input_array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    input_path = tmp_path / "input.npy"
    compressed_path = tmp_path / "out.zfp"
    recon_path = tmp_path / "recon.npy"
    result_json_path = tmp_path / "result.json"
    np.save(input_path, input_array)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": str(input_path),
                "zfp": str(tmp_path / "fake-zfp"),
                "rate": 8.0,
                "shape": [2, 2, 2],
                "compressed": str(compressed_path),
                "recon": str(recon_path),
                "result_json": str(result_json_path),
            }
        ),
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fail_psnr_conversion(_data_range: float, _target_psnr: float) -> float:
        raise AssertionError("PSNR conversion should not run for rate mode")

    def fake_run_command(command: list[str]) -> CommandResult:
        commands.append(command)
        if "-i" in command:
            compressed_path.write_bytes(b"zfp")
            return CommandResult(tuple(command), "", "", 0, elapsed_seconds=0.1)

        input_array.tofile(Path(command[command.index("-o") + 1]))
        return CommandResult(tuple(command), "", "", 0, elapsed_seconds=0.05)

    monkeypatch.setattr(zfp_cli, "zfp_tolerance_from_psnr", fail_psnr_conversion)
    monkeypatch.setattr(zfp_cli, "run_command", fake_run_command)
    monkeypatch.setattr(sys, "argv", ["zfp_cli.py", "--config", str(config_path)])

    exit_code = zfp_cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert len(commands) == 2
    assert "-r" in commands[0]
    assert commands[0][commands[0].index("-r") + 1] == "8"

    result = json.loads(captured.out)
    assert result["target_mode"] == "rate"
    assert result["target_value"] == pytest.approx(8.0)
    assert result["target_psnr"] is None
    assert result["native_mode"] == "rate"
    assert result["native_value"] == pytest.approx(8.0)
    assert json.loads(result_json_path.read_text(encoding="utf-8"))["target_mode"] == "rate"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import simple_ai_trading.round74_segmented_development_cli as subject


def _arguments(tmp_path: Path) -> argparse.Namespace:
    repository = tmp_path / "repository"
    repository.mkdir()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subject.register_round74_segmented_development_command(subparsers)
    return parser.parse_args(
        [
            "binance-round74-develop",
            "--repository",
            str(repository),
            "--database",
            "shard-01.duckdb",
            "--database",
            "shard-02.duckdb",
            "--target-assemblies",
            "targets",
            "--source-artifacts",
            "sources",
            "--model-output",
            "model",
            "--qualification-output",
            "qualification",
            "--terminal-observed-wall-ns",
            "123",
        ]
    )


def test_round74_development_cli_passes_repeatable_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _arguments(tmp_path)
    observed: dict[str, object] = {}

    def run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(subject, "run_round74_segmented_development", run)

    assert subject.command_binance_round74_develop(args) == 0
    repository = Path(args.repository).resolve()
    assert observed["database_paths"] == (
        repository / "shard-01.duckdb",
        repository / "shard-02.duckdb",
    )
    assert observed["terminal_observed_wall_ns"] == 123
    assert observed["enable_ai"] is True
    assert observed["supervised_device_group_policy"] == "auto"
    assert observed["supervised_device_run_group_size"] == 8
    assert observed["device_group_preflight_timeout_seconds"] == 300.0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_round74_development_cli_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setattr(
        subject,
        "run_round74_segmented_development",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("route differs")),
    )

    assert subject.command_binance_round74_develop(args) == 2
    error = capsys.readouterr().err
    assert "ValueError" in error
    assert "route differs" in error

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from simple_ai_trading import entrypoint
from simple_ai_trading import polymarket_round21_cli as cli_module
from simple_ai_trading.command_contract import command_specs, workflow_commands


def test_round21_corpus_command_is_in_shared_cli_windows_contract() -> None:
    specs = {spec.name: spec for spec in command_specs()}
    workflow = {item.name: item for item in workflow_commands()}

    assert "polymarket-round21-corpus" in specs
    assert workflow["polymarket-round21-corpus"].page == "Research"
    assert workflow["polymarket-round21-corpus"].group == "Polymarket evidence"
    assert {
        option.dest for option in specs["polymarket-round21-corpus"].options
    } >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
    }


def test_round21_corpus_command_publishes_through_exact_library_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    expected = {"manifest_sha256": "b" * 64}
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda path: transport if path.name == "terminal.json" else None,
    )

    def fake_publish(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(cli_module, "publish_round21_core_corpus", fake_publish)
    args = entrypoint._parse_args(
        [
            "polymarket-round21-corpus",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--repository",
            str(tmp_path),
            "--observed-at-ms",
            "1900000000000",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "repository": tmp_path,
        "source_database": tmp_path / "source.duckdb",
        "terminal_transport_manifest": transport,
        "publication_directory": tmp_path / "publication",
        "observed_at_ms": 1_900_000_000_000,
    }
    assert "authority=false" in capsys.readouterr().out


def test_round21_corpus_command_fails_closed_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: (_ for _ in ()).throw(ValueError("capture is active")),
    )
    args = argparse.Namespace(
        source_database=str(tmp_path / "source.duckdb"),
        terminal_transport_manifest=str(tmp_path / "terminal.json"),
        publication_directory=str(tmp_path / "publication"),
        repository=str(tmp_path),
        observed_at_ms=None,
        json=False,
    )

    assert cli_module.command_polymarket_round21_corpus(args) == 2
    assert "capture is active" in capsys.readouterr().err
    assert not (tmp_path / "publication").exists()

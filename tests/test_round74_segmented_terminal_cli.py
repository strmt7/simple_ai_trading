from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from simple_ai_trading.entrypoint import _build_parser
import simple_ai_trading.round74_segmented_terminal_cli as subject


def _required_arguments(tmp_path: Path) -> list[str]:
    return [
        "binance-round74-sealed-evaluate",
        "--database",
        str(tmp_path / "capture-a.duckdb"),
        "--database",
        str(tmp_path / "capture-b.duckdb"),
        "--test-target-assemblies",
        str(tmp_path / "targets"),
        "--source-artifacts",
        str(tmp_path / "sources"),
        "--development-bundle",
        str(tmp_path / "development.json"),
        "--pretest-policy",
        str(tmp_path / "policy.json"),
        "--ai-qualification",
        str(tmp_path / "qualification.json"),
        "--one-use-store",
        str(tmp_path / "one-use.sqlite3"),
        "--sealed-ledger",
        str(tmp_path / "sealed.sqlite3"),
        "--output",
        str(tmp_path / "result.json"),
    ]


def test_round74_terminal_parser_requires_explicit_one_use_acknowledgement(
    tmp_path: Path,
) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_arguments(tmp_path))
    parsed = parser.parse_args(
        [*_required_arguments(tmp_path), "--acknowledge-one-use-test-access"]
    )
    assert parsed.profile == "conservative"
    assert parsed.database == [
        str(tmp_path / "capture-a.duckdb"),
        str(tmp_path / "capture-b.duckdb"),
    ]
    assert parsed.acknowledge_one_use_test_access is True


def test_round74_terminal_handler_resolves_paths_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def run(**kwargs):
        observed.update(kwargs)
        return {"status": "complete", "result_sha256": "1" * 64}

    monkeypatch.setattr(subject, "run_round74_segmented_terminal_evaluation", run)
    args = _build_parser().parse_args(
        [
            *_required_arguments(tmp_path),
            "--repository",
            str(tmp_path),
            "--acknowledge-one-use-test-access",
            "--terminal-observed-wall-ns",
            "1800000000000000000",
        ]
    )

    assert args.func(args) == 0
    assert observed["repository"] == tmp_path.resolve()
    assert observed["database_paths"] == (
        tmp_path / "capture-a.duckdb",
        tmp_path / "capture-b.duckdb",
    )
    assert observed["profile"] == "conservative"
    assert observed["terminal_observed_wall_ns"] == 1_800_000_000_000_000_000
    assert callable(observed["progress"])


def test_round74_terminal_handler_rejects_direct_acknowledgement_bypass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(repository=".", acknowledge_one_use_test_access=False)

    assert subject.command_binance_round74_sealed_evaluate(args) == 2
    assert "one-use acknowledgement is required" in capsys.readouterr().err


def test_round74_terminal_recovery_handler_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def recover(**kwargs):
        observed.update(kwargs)
        return {"status": "recovered", "result_sha256": "2" * 64}

    monkeypatch.setattr(subject, "recover_round74_segmented_terminal_result", recover)
    args = _build_parser().parse_args(
        [
            "binance-round74-recover-sealed",
            "--one-use-store",
            str(tmp_path / "one-use.sqlite3"),
            "--output",
            str(tmp_path / "recovered.json"),
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "one_use_store_path": tmp_path / "one-use.sqlite3",
        "output_path": tmp_path / "recovered.json",
    }

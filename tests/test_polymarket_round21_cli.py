from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simple_ai_trading import entrypoint
from simple_ai_trading import polymarket_round21_cli as cli_module
from simple_ai_trading.command_contract import command_specs, workflow_commands
from simple_ai_trading.polymarket_round21_shadow_store import (
    Round21ProspectiveShadowStore,
)

from polymarket_round21_support import (
    SHADOW_MODEL_SHA,
    SHADOW_SEALED_SHA,
    START_MS,
)


RUN_ID = "1" * 32


def test_round21_corpus_command_is_in_shared_cli_windows_contract() -> None:
    specs = {spec.name: spec for spec in command_specs()}
    workflow = {item.name: item for item in workflow_commands()}

    assert "polymarket-round21-terminal" in specs
    assert workflow["polymarket-round21-terminal"].page == "Research"
    assert workflow["polymarket-round21-terminal"].group == "Polymarket evidence"
    assert {option.dest for option in specs["polymarket-round21-terminal"].options} >= {
        "campaign_plan",
        "state_root",
        "output",
        "repository",
        "observed_at_ms",
    }
    assert "polymarket-round21-sidecar-terminal" in specs
    assert workflow["polymarket-round21-sidecar-terminal"].page == "Research"
    assert (
        workflow["polymarket-round21-sidecar-terminal"].group == "Polymarket evidence"
    )
    assert {
        option.dest for option in specs["polymarket-round21-sidecar-terminal"].options
    } >= {"campaign_plan", "state_root", "output", "observed_at_ms"}
    assert "polymarket-round21-corpus" in specs
    assert workflow["polymarket-round21-corpus"].page == "Research"
    assert workflow["polymarket-round21-corpus"].group == "Polymarket evidence"
    assert {option.dest for option in specs["polymarket-round21-corpus"].options} >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
    }
    assert "polymarket-round21-ablate-basis" in specs
    assert workflow["polymarket-round21-ablate-basis"].page == "Research"
    assert workflow["polymarket-round21-ablate-basis"].group == "Polymarket models"
    assert {
        option.dest for option in specs["polymarket-round21-ablate-basis"].options
    } >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
        "repository",
        "output",
    }
    assert "polymarket-round21-fit-core" in specs
    assert workflow["polymarket-round21-fit-core"].page == "Research"
    assert workflow["polymarket-round21-fit-core"].group == "Polymarket models"
    assert {option.dest for option in specs["polymarket-round21-fit-core"].options} >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
        "basis_ablation_result",
        "output",
        "compute_backend",
    }
    assert "polymarket-round21-fit-matched" in specs
    assert workflow["polymarket-round21-fit-matched"].page == "Research"
    assert workflow["polymarket-round21-fit-matched"].group == "Polymarket models"
    assert {
        option.dest for option in specs["polymarket-round21-fit-matched"].options
    } >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
        "sidecar_database",
        "sidecar_terminal_manifest",
        "basis_ablation_result",
        "output",
        "compute_backend",
    }
    assert "polymarket-round21-evaluate-development" in specs
    assert (
        workflow["polymarket-round21-evaluate-development"].group == "Polymarket models"
    )
    assert {
        option.dest
        for option in specs["polymarket-round21-evaluate-development"].options
    } >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
        "model_artifact",
        "selected_layer",
        "sidecar_database",
        "sidecar_terminal_manifest",
        "output",
    }
    assert "polymarket-round21-ai-development" in specs
    assert workflow["polymarket-round21-ai-development"].group == "Polymarket models"
    assert {
        option.dest for option in specs["polymarket-round21-ai-development"].options
    } >= {
        "source_database",
        "terminal_transport_manifest",
        "publication_directory",
        "model_artifact",
        "selected_layer",
        "sidecar_database",
        "sidecar_terminal_manifest",
        "risk_benchmark_evidence",
        "qwen3_5_9b_digest",
        "fin_r1_8b_digest",
        "fino1_8b_digest",
        "ai_cache_database",
        "ollama_url",
        "output",
        "acknowledge_one_use_test_access",
        "repository",
        "one_use_store",
        "pretest_output",
        "claim_output",
        "sealed_output",
    }
    assert "polymarket-round21-recover-sealed" in specs
    assert workflow["polymarket-round21-recover-sealed"].group == "Polymarket evidence"
    assert {
        option.dest for option in specs["polymarket-round21-recover-sealed"].options
    } >= {"one_use_store", "output"}
    assert "polymarket-round21-shadow" in specs
    assert workflow["polymarket-round21-shadow"].page == "Research"
    assert workflow["polymarket-round21-shadow"].group == "Polymarket evidence"
    assert {option.dest for option in specs["polymarket-round21-shadow"].options} >= {
        "action",
        "shadow_database",
        "run_id",
        "model_artifact",
        "model_file_sha256",
        "evaluation_report",
        "evaluation_file_sha256",
        "duration_seconds",
    }


def test_round21_terminal_command_seals_through_exact_library_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"manifest_sha256": "a" * 64, "eligible_run_ids": ["b" * 32]}
    observed: dict[str, object] = {}

    def fake_build(**kwargs):
        observed.update(kwargs)
        return manifest

    def fake_write(path: Path, value: object) -> None:
        observed["write_path"] = path
        observed["write_value"] = value

    monkeypatch.setattr(
        cli_module,
        "build_round21_terminal_transport_manifest",
        fake_build,
    )
    monkeypatch.setattr(
        cli_module,
        "write_round21_terminal_transport_manifest",
        fake_write,
    )
    args = entrypoint._parse_args(
        [
            "polymarket-round21-terminal",
            "--campaign-plan",
            str(tmp_path / "plan.json"),
            "--state-root",
            str(tmp_path / "state"),
            "--output",
            str(tmp_path / "terminal.json"),
            "--repository",
            str(tmp_path),
            "--observed-at-ms",
            "1900000000000",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "repository": tmp_path,
        "plan_path": tmp_path / "plan.json",
        "state_root": tmp_path / "state",
        "observed_at_ms": 1_900_000_000_000,
        "write_path": tmp_path / "terminal.json",
        "write_value": manifest,
    }
    output = capsys.readouterr().out
    assert "eligible_runs=1" in output
    assert "authority=false" in output


def test_round21_terminal_command_fails_closed_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_round21_terminal_transport_manifest",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("capture is active")),
    )
    output = tmp_path / "terminal.json"
    args = argparse.Namespace(
        campaign_plan=str(tmp_path / "plan.json"),
        state_root=str(tmp_path / "state"),
        output=str(output),
        repository=str(tmp_path),
        observed_at_ms=None,
        json=False,
    )

    assert cli_module.command_polymarket_round21_terminal(args) == 2
    assert "capture is active" in capsys.readouterr().err
    assert not output.exists()


def test_round21_sidecar_terminal_uses_independent_public_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = {"manifest_sha256": "a" * 64, "eligible_run_ids": ["b" * 32]}
    observed: dict[str, object] = {}

    def fake_build(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return manifest

    def fake_write(path: Path, value: object) -> None:
        observed["write_path"] = path
        observed["write_value"] = value

    monkeypatch.setattr(
        cli_module,
        "build_round21_sidecar_terminal_manifest",
        fake_build,
    )
    monkeypatch.setattr(
        cli_module,
        "write_round21_sidecar_terminal_manifest",
        fake_write,
    )
    args = entrypoint._parse_args(
        [
            "polymarket-round21-sidecar-terminal",
            "--campaign-plan",
            str(tmp_path / "sidecar-plan.json"),
            "--state-root",
            str(tmp_path / "sidecar-state"),
            "--output",
            str(tmp_path / "sidecar-terminal.json"),
            "--observed-at-ms",
            "1900000000000",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "plan_path": tmp_path / "sidecar-plan.json",
        "state_root": tmp_path / "sidecar-state",
        "observed_at_ms": 1_900_000_000_000,
        "write_path": tmp_path / "sidecar-terminal.json",
        "write_value": manifest,
    }
    assert "authority=false" in capsys.readouterr().out


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


def test_round21_fit_core_writes_source_bound_non_authoritative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    basis_ablation = {"result_sha256": "d" * 64, "basis_accepted": True}
    artifact = {"artifact_sha256": "b" * 64, "payload": [1, 2, 3]}
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda path: transport if path.name == "terminal.json" else None,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_probability_basis_ablation_result",
        lambda path: basis_ablation if path.name == "basis.json" else None,
    )

    def fake_fit(**kwargs):
        observed.update(kwargs)
        return artifact

    monkeypatch.setattr(cli_module, "fit_round21_core_baseline", fake_fit)
    output = tmp_path / "model.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-fit-core",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--basis-ablation-result",
            str(tmp_path / "basis.json"),
            "--output",
            str(output),
            "--compute-backend",
            "cpu",
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "publication_directory": tmp_path / "publication",
        "source_database": tmp_path / "source.duckdb",
        "terminal_transport_manifest": transport,
        "basis_ablation_result": basis_ablation,
        "compute_backend": "cpu",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    result = json.loads(capsys.readouterr().out)
    assert result["artifact_sha256"] == "b" * 64
    assert len(result["artifact_file_sha256"]) == 64
    assert result["optional_binance_comparison_completed"] is False
    assert result["sealed_test_accessed"] is False
    assert result["live_trading_authority"] is False


def test_round21_basis_ablation_writes_immutable_non_authoritative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    ablation = {
        "result_sha256": "b" * 64,
        "basis_accepted": False,
        "next_action": "reject_basis_and_supersede_model_design_v6_before_full_fit",
        "profitability_claim": False,
        "live_trading_authority": False,
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda path: transport if path.name == "terminal.json" else None,
    )

    def fake_evaluate(**kwargs):
        observed.update(kwargs)
        return ablation

    monkeypatch.setattr(
        cli_module,
        "evaluate_round21_core_probability_basis",
        fake_evaluate,
    )
    output = tmp_path / "basis-ablation.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-ablate-basis",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--repository",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "repository": tmp_path,
        "publication_directory": tmp_path / "publication",
        "source_database": tmp_path / "source.duckdb",
        "terminal_transport_manifest": transport,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == ablation
    result = json.loads(capsys.readouterr().out)
    assert result["result_sha256"] == "b" * 64
    assert len(result["result_file_sha256"]) == 64
    assert result["basis_accepted"] is False
    assert result["sealed_test_accessed"] is False
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False

    assert args.func(args) == 2
    assert "output already exists" in capsys.readouterr().err
    assert json.loads(output.read_text(encoding="utf-8")) == ablation


def test_round21_fit_core_fails_without_creating_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: (_ for _ in ()).throw(ValueError("manifest differs")),
    )
    output = tmp_path / "model.json"
    args = argparse.Namespace(
        source_database=str(tmp_path / "source.duckdb"),
        terminal_transport_manifest=str(tmp_path / "terminal.json"),
        publication_directory=str(tmp_path / "publication"),
        basis_ablation_result=str(tmp_path / "basis.json"),
        output=str(output),
        compute_backend="cpu",
        json=False,
    )

    assert cli_module.command_polymarket_round21_fit_core(args) == 2
    assert "manifest differs" in capsys.readouterr().err
    assert not output.exists()

    output.write_text("existing evidence", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: pytest.fail("existing output must block before manifest access"),
    )
    assert cli_module.command_polymarket_round21_fit_core(args) == 2
    assert "output already exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "existing evidence"


def test_round21_fit_matched_writes_predictive_non_authoritative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    sidecar = {"manifest_sha256": "b" * 64}
    basis_ablation = {"result_sha256": "d" * 64, "basis_accepted": True}
    artifact = {
        "artifact_sha256": "c" * 64,
        "trained_layers": ["core", "core_spot", "core_spot_usdm"],
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: transport,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_sidecar_terminal_manifest",
        lambda _path: sidecar,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_probability_basis_ablation_result",
        lambda _path: basis_ablation,
    )

    def fake_fit(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return artifact

    monkeypatch.setattr(
        cli_module,
        "fit_round21_matched_optional_candidate",
        fake_fit,
    )
    output = tmp_path / "matched.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-fit-matched",
            "--source-database",
            str(tmp_path / "core.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "core-terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--sidecar-database",
            str(tmp_path / "sidecar.duckdb"),
            "--sidecar-terminal-manifest",
            str(tmp_path / "sidecar-terminal.json"),
            "--basis-ablation-result",
            str(tmp_path / "basis.json"),
            "--output",
            str(output),
            "--compute-backend",
            "directml",
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed == {
        "publication_directory": tmp_path / "publication",
        "source_database": tmp_path / "core.duckdb",
        "terminal_transport_manifest": transport,
        "sidecar_database": tmp_path / "sidecar.duckdb",
        "sidecar_terminal_manifest": sidecar,
        "basis_ablation_result": basis_ablation,
        "compute_backend": "directml",
    }
    result = json.loads(capsys.readouterr().out)
    assert result["matched_predictive_comparison_completed"] is True
    assert result["economic_evaluation_completed"] is False
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False


def test_round21_development_economics_uses_shared_non_authoritative_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    artifact = {
        "artifact_sha256": "b" * 64,
        "trained_layers": ["core"],
    }
    assembly = SimpleNamespace(
        partition_policy="policy",
        train="train",
        tune_calibration="calibration",
        tune_selection="selection",
        publication_manifest_sha256="c" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: transport,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_development_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        cli_module,
        "assemble_round21_core_development",
        lambda **_kwargs: assembly,
    )

    def fake_replay(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        progress = kwargs["progress"]
        assert callable(progress)
        progress("prepared", {"condition_count": 123, "ledger_count": 81})
        return SimpleNamespace(
            result_sha256="d" * 64,
            source_condition_count=123,
            development_gate_passed=False,
            asdict=lambda: {
                "result_sha256": "d" * 64,
                "profitability_claim": False,
                "live_trading_authority": False,
            },
        )

    monkeypatch.setattr(
        cli_module,
        "replay_round21_development_economics",
        fake_replay,
    )
    output = tmp_path / "economics.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-evaluate-development",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed["source_database"] == tmp_path / "source.duckdb"
    assert observed["development_panels"] == ("train", "calibration", "selection")
    assert observed["selected_population_layer"] == "core"
    assert json.loads(output.read_text(encoding="utf-8"))["result_sha256"] == "d" * 64
    captured = capsys.readouterr()
    assert "condition_count=123" in captured.err
    assert json.loads(captured.out)["profitability_claim"] is False


def test_round21_matched_economics_requires_independent_sidecar_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: {"manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_development_artifact",
        lambda _path: {
            "artifact_sha256": "b" * 64,
            "trained_layers": ["core", "core_spot", "core_spot_usdm"],
        },
    )
    output = tmp_path / "economics.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-evaluate-development",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--selected-layer",
            "core_spot",
            "--output",
            str(output),
        ]
    )

    assert args.func(args) == 2
    assert "require the exact sidecar" in capsys.readouterr().err
    assert not output.exists()


def test_round21_ai_development_runs_one_shared_program_and_compact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    artifact = {
        "artifact_sha256": "b" * 64,
        "trained_layers": ["core"],
    }
    assembly = SimpleNamespace(
        partition_policy="policy",
        train="train",
        tune_calibration="calibration",
        tune_selection="selection",
        publication_manifest_sha256="c" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: transport,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_development_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        cli_module,
        "assemble_round21_core_development",
        lambda **_kwargs: assembly,
    )

    class FakeCache:
        def __init__(self, path, **kwargs):
            observed["cache_path"] = path
            observed["cache_options"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(cli_module, "PolymarketEvidenceStore", FakeCache)
    economic = SimpleNamespace(source_condition_count=321)
    selection = SimpleNamespace(
        qualified_candidate_count=0,
        nominated_model=None,
    )
    result = SimpleNamespace(
        result_sha256="d" * 64,
        economic_result=economic,
        replay_result=SimpleNamespace(comparisons=(1, 2, 3)),
        candidate_selection=selection,
        asdict=lambda: {
            "result_sha256": "d" * 64,
            "profitability_claim": False,
            "live_trading_authority": False,
        },
    )

    def fake_program(**kwargs):
        observed.update(kwargs)
        kwargs["progress"]("ai_candidate_started", {"model": "qwen3.5:9b"})
        return result

    monkeypatch.setattr(
        cli_module,
        "run_round21_development_ai_program",
        fake_program,
    )
    risk = tmp_path / "risk.json"
    risk.write_text('{"passed":false}', encoding="ascii")
    output = tmp_path / "ai-development.json"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-ai-development",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--risk-benchmark-evidence",
            str(risk),
            "--qwen3-5-9b-digest",
            "1" * 64,
            "--fin-r1-8b-digest",
            "2" * 64,
            "--fino1-8b-digest",
            "3" * 64,
            "--ai-cache-database",
            str(tmp_path / "ai-cache.duckdb"),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed["source_database"] == tmp_path / "source.duckdb"
    assert observed["development_panels"] == (
        "train",
        "calibration",
        "selection",
    )
    assert tuple(config.model for config in observed["configs"]) == (
        "qwen3.5:9b",
        "fin-r1:8b",
        "fino1:8b",
    )
    assert observed["expected_model_digests"] == {
        "qwen3.5:9b": "1" * 64,
        "fin-r1:8b": "2" * 64,
        "fino1:8b": "3" * 64,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["result_sha256"] == ("d" * 64)
    captured = capsys.readouterr()
    assert "ai_candidate_started model=qwen3.5:9b" in captured.err
    assert json.loads(captured.out)["candidate_count"] == 3


def test_round21_ai_development_runs_explicit_one_use_sealed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = {"manifest_sha256": "a" * 64}
    artifact = {
        "artifact_sha256": "b" * 64,
        "trained_layers": ["core"],
    }
    assembly = SimpleNamespace(
        partition_policy="policy",
        train="train",
        tune_calibration="calibration",
        tune_selection="selection",
        publication_manifest_sha256="c" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_round21_terminal_transport_manifest",
        lambda _path: transport,
    )
    monkeypatch.setattr(
        cli_module,
        "load_round21_development_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        cli_module,
        "assemble_round21_core_development",
        lambda **_kwargs: assembly,
    )

    class FakeCache:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(cli_module, "PolymarketEvidenceStore", FakeCache)
    economic = SimpleNamespace(
        source_condition_count=321,
        selected_matrix="matrix",
        optional_comparison=None,
    )
    selection = SimpleNamespace(
        qualified_candidate_count=0,
        nominated_model=None,
    )
    result = SimpleNamespace(
        result_sha256="d" * 64,
        economic_result=economic,
        replay_result=SimpleNamespace(comparisons=()),
        candidate_selection=selection,
        asdict=lambda: {"result_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        cli_module,
        "run_round21_development_ai_program",
        lambda **_kwargs: result,
    )
    pretest = SimpleNamespace(
        asdict=lambda: {"manifest_sha256": "e" * 64},
    )
    claim = SimpleNamespace(
        asdict=lambda: {"claim_sha256": "f" * 64},
    )

    def build_pretest(*args, **kwargs):
        observed["pretest_args"] = args
        observed["pretest_kwargs"] = kwargs
        return pretest

    monkeypatch.setattr(cli_module, "build_round21_pretest_manifest", build_pretest)
    monkeypatch.setattr(
        cli_module,
        "create_round21_one_use_claim",
        lambda value: claim if value is pretest else pytest.fail("pretest differs"),
    )
    sealed_result = SimpleNamespace(
        candidate_accepted=False,
        ai_enabled_candidate=False,
    )
    sealed_outcome = SimpleNamespace(result=sealed_result)

    def evaluate_sealed(**kwargs):
        observed["sealed_kwargs"] = kwargs
        return sealed_outcome

    monkeypatch.setattr(
        cli_module,
        "evaluate_round21_terminal_sealed_once",
        evaluate_sealed,
    )
    monkeypatch.setattr(
        cli_module,
        "build_round21_sealed_result_bundle",
        lambda value: (
            {"candidate_accepted": False, "live_trading_authority": False}
            if value is sealed_result
            else pytest.fail("sealed result differs")
        ),
    )
    risk = tmp_path / "risk.json"
    risk.write_text("{}", encoding="ascii")
    output = tmp_path / "ai-development.json"
    pretest_output = tmp_path / "pretest.json"
    claim_output = tmp_path / "claim.json"
    sealed_output = tmp_path / "sealed.json"
    one_use_store = tmp_path / "one-use.sqlite3"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-ai-development",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--risk-benchmark-evidence",
            str(risk),
            "--qwen3-5-9b-digest",
            "1" * 64,
            "--fin-r1-8b-digest",
            "2" * 64,
            "--fino1-8b-digest",
            "3" * 64,
            "--ai-cache-database",
            str(tmp_path / "ai-cache.duckdb"),
            "--output",
            str(output),
            "--acknowledge-one-use-test-access",
            "--repository",
            str(tmp_path),
            "--one-use-store",
            str(one_use_store),
            "--pretest-output",
            str(pretest_output),
            "--claim-output",
            str(claim_output),
            "--sealed-output",
            str(sealed_output),
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed["pretest_args"] == (str(tmp_path),)
    assert observed["pretest_kwargs"] == {
        "selected_population_layer": "core",
        "core_corpus_publication_directory": str(tmp_path / "publication"),
        "optional_campaign_terminal_sha256": None,
        "development_model_artifact": artifact,
        "development_economic_matrix": "matrix",
        "development_optional_comparison": None,
        "development_ai_selection": selection,
    }
    sealed_kwargs = observed["sealed_kwargs"]
    assert sealed_kwargs["store_path"] == one_use_store
    assert sealed_kwargs["claim"] is claim
    assert sealed_kwargs["pretest"] is pretest
    assert sealed_kwargs["development_ai_report"] is None
    assert sealed_kwargs["development_ai_comparison"] is None
    assert json.loads(pretest_output.read_text(encoding="utf-8")) == {
        "manifest_sha256": "e" * 64
    }
    assert json.loads(claim_output.read_text(encoding="utf-8")) == {
        "claim_sha256": "f" * 64
    }
    assert json.loads(sealed_output.read_text(encoding="utf-8")) == {
        "candidate_accepted": False,
        "live_trading_authority": False,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["target_accessed"] is True
    assert summary["sealed_candidate_accepted"] is False
    assert summary["sealed_ai_enabled_candidate"] is False
    assert summary["live_trading_authority"] is False


def test_round21_ai_development_rejects_source_as_cache_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "run_round21_development_ai_program",
        lambda **_kwargs: pytest.fail("AI program must not run"),
    )
    risk = tmp_path / "risk.json"
    risk.write_text("{}", encoding="ascii")
    shared = tmp_path / "source.duckdb"
    args = entrypoint._parse_args(
        [
            "polymarket-round21-ai-development",
            "--source-database",
            str(shared),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--risk-benchmark-evidence",
            str(risk),
            "--qwen3-5-9b-digest",
            "1" * 64,
            "--fin-r1-8b-digest",
            "2" * 64,
            "--fino1-8b-digest",
            "3" * 64,
            "--ai-cache-database",
            str(shared),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.func(args) == 2
    assert "must be separate" in capsys.readouterr().err


def test_round21_ai_development_rejects_partial_sealed_inputs_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "run_round21_development_ai_program",
        lambda **_kwargs: pytest.fail("AI program must not run"),
    )
    risk = tmp_path / "risk.json"
    risk.write_text("{}", encoding="ascii")
    args = entrypoint._parse_args(
        [
            "polymarket-round21-ai-development",
            "--source-database",
            str(tmp_path / "source.duckdb"),
            "--terminal-transport-manifest",
            str(tmp_path / "terminal.json"),
            "--publication-directory",
            str(tmp_path / "publication"),
            "--model-artifact",
            str(tmp_path / "model.json"),
            "--risk-benchmark-evidence",
            str(risk),
            "--qwen3-5-9b-digest",
            "1" * 64,
            "--fin-r1-8b-digest",
            "2" * 64,
            "--fino1-8b-digest",
            "3" * 64,
            "--ai-cache-database",
            str(tmp_path / "ai-cache.duckdb"),
            "--output",
            str(tmp_path / "result.json"),
            "--sealed-output",
            str(tmp_path / "sealed.json"),
        ]
    )

    assert args.func(args) == 2
    assert (
        "sealed evaluation paths require --acknowledge-one-use-test-access"
        in capsys.readouterr().err
    )


def test_round21_sealed_recovery_exports_completed_bundle_without_reopening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "one-use.sqlite3"
    store.touch()
    output = tmp_path / "sealed.json"
    bundle = {
        "result": {
            "result_sha256": "a" * 64,
            "candidate_accepted": False,
            "ai_enabled_candidate": False,
        },
        "bundle_sha256": "b" * 64,
    }
    observed: list[Path] = []

    def load(path):
        observed.append(path)
        return bundle

    monkeypatch.setattr(
        cli_module,
        "load_round21_completed_sealed_bundle",
        load,
    )
    args = entrypoint._parse_args(
        [
            "polymarket-round21-recover-sealed",
            "--one-use-store",
            str(store),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert observed == [store]
    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    summary = json.loads(capsys.readouterr().out)
    assert summary["result_sha256"] == "a" * 64
    assert summary["test_access_reopened"] is False
    assert summary["live_trading_authority"] is False


def test_round21_sealed_recovery_rejects_existing_output_before_store_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "one-use.sqlite3"
    store.touch()
    output = tmp_path / "sealed.json"
    output.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        cli_module,
        "load_round21_completed_sealed_bundle",
        lambda _path: pytest.fail("completed store must not be read"),
    )
    args = entrypoint._parse_args(
        [
            "polymarket-round21-recover-sealed",
            "--one-use-store",
            str(store),
            "--output",
            str(output),
        ]
    )

    assert args.func(args) == 2
    assert "output already exists" in capsys.readouterr().err


def _terminal_shadow_audit(path: Path):
    with Round21ProspectiveShadowStore(path) as store:
        store.start_run(
            run_id=RUN_ID,
            source_model_artifact_sha256=SHADOW_MODEL_SHA,
            sealed_result_sha256=SHADOW_SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS,
        )
        store.terminate_run(
            RUN_ID,
            status="complete",
            finished_at_ms=START_MS + 1,
        )
        return store.audit_run(RUN_ID)


def test_round21_shadow_audit_is_offline_and_non_authoritative(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "shadow.sqlite3"
    _terminal_shadow_audit(database)
    args = entrypoint._parse_args(
        [
            "polymarket-round21-shadow",
            "--action",
            "audit",
            "--shadow-database",
            str(database),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == RUN_ID
    assert payload["terminal"]["status"] == "complete"
    assert payload["prediction_count"] == 0
    assert payload["credentials_used"] is False
    assert payload["grants_execution_authority"] is False
    assert payload["live_trading_authority"] is False


def test_round21_shadow_run_uses_exact_public_runtime_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit = _terminal_shadow_audit(tmp_path / "audit-source.sqlite3")
    public_client = SimpleNamespace(session=Mock())
    observed: dict[str, object] = {}

    class Runner:
        async def run(self, stop, *, scheduled_end_ms):
            observed["stop"] = stop
            observed["scheduled_end_ms"] = scheduled_end_ms
            return audit

    stack = SimpleNamespace(runner=Runner(), close=Mock())
    monkeypatch.setattr(
        cli_module,
        "PolymarketPublicClient",
        lambda: public_client,
    )

    def build_stack(**kwargs):
        observed.update(kwargs)
        return stack

    monkeypatch.setattr(
        cli_module,
        "build_polymarket_round21_shadow_runtime_stack",
        build_stack,
    )
    monkeypatch.setattr(cli_module.time, "time_ns", lambda: START_MS * 1_000_000)
    args = entrypoint._parse_args(
        [
            "polymarket-round21-shadow",
            "--action",
            "run",
            "--shadow-database",
            str(tmp_path / "run.sqlite3"),
            "--run-id",
            RUN_ID,
            "--model-artifact",
            "model.json",
            "--model-file-sha256",
            "a" * 64,
            "--evaluation-report",
            "evaluation.json",
            "--evaluation-file-sha256",
            "b" * 64,
            "--duration-seconds",
            "5",
            "--discovery-seconds",
            "2",
            "--poll-seconds",
            "0.1",
            "--queue-capacity",
            "4000",
        ]
    )

    assert args.func(args) == 0
    assert observed["public_client"] is public_client
    assert observed["model_artifact_path"] == "model.json"
    assert observed["expected_model_file_sha256"] == "a" * 64
    assert observed["evaluation_report_path"] == "evaluation.json"
    assert observed["expected_evaluation_file_sha256"] == "b" * 64
    assert observed["run_id"] == RUN_ID
    assert observed["discovery_interval_seconds"] == 2.0
    assert observed["poll_interval_seconds"] == 0.1
    assert observed["queue_capacity"] == 4_000
    assert observed["scheduled_end_ms"] == START_MS + 5_000
    assert "authority=false" in capsys.readouterr().out
    stack.close.assert_called_once_with()
    public_client.session.close.assert_called_once_with()


def test_round21_shadow_command_fails_closed_without_required_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = entrypoint._parse_args(
        [
            "polymarket-round21-shadow",
            "--action",
            "audit",
            "--shadow-database",
            str(tmp_path / "missing.sqlite3"),
        ]
    )

    assert args.func(args) == 2
    assert "shadow database is unavailable" in capsys.readouterr().err
    assert not (tmp_path / "missing.sqlite3").exists()


def test_round21_shadow_audit_requires_id_when_database_has_multiple_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "multiple.sqlite3"
    _terminal_shadow_audit(database)
    with Round21ProspectiveShadowStore(database) as store:
        store.start_run(
            run_id="2" * 32,
            source_model_artifact_sha256=SHADOW_MODEL_SHA,
            sealed_result_sha256=SHADOW_SEALED_SHA,
            population_layer="core",
            started_at_ms=START_MS + 10,
        )
        store.terminate_run(
            "2" * 32,
            status="complete",
            finished_at_ms=START_MS + 11,
        )
    args = entrypoint._parse_args(
        [
            "polymarket-round21-shadow",
            "--action",
            "audit",
            "--shadow-database",
            str(database),
        ]
    )

    assert args.func(args) == 2
    assert "exactly one shadow run" in capsys.readouterr().err

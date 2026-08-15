from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_ai_sealed as sealed_ai
import tools.run_polymarket_round28_ai_sealed as sealed_operator
from simple_ai_trading.polymarket_round28_ai_cases import Round28AICasePanel
from simple_ai_trading.polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_MODEL_IDS,
)
from simple_ai_trading.polymarket_round28_ai_sealed import (
    build_round28_ai_sealed_terminal_result,
)
from simple_ai_trading.polymarket_round28_ai_selection import (
    Round28AICandidateSelection,
)
from tools.run_polymarket_round28_ai_sealed import _parser as _economic_parser
from tools.run_polymarket_round28_ai_sealed_cases import (
    _parser as _case_parser,
)
from tools.run_polymarket_round28_ai_sealed_cases import _result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _selection() -> Round28AICandidateSelection:
    coverage = tuple(
        {
            "model_id": model_id,
            "status": "evaluated",
            "host_evidence_sha256": format(index + 1, "064x"),
            "runtime_digest": format(index + 11, "064x"),
            "economic_report_sha256": format(index + 21, "064x"),
        }
        for index, model_id in enumerate(POLYMARKET_ROUND28_AI_MODEL_IDS)
    )
    provisional = Round28AICandidateSelection(
        case_panel_sha256="a" * 64,
        round28_economic_report_sha256="b" * 64,
        candidate_coverage=coverage,
        economic_report_sha256=tuple(
            sorted(str(item["economic_report_sha256"]) for item in coverage)
        ),
        nominated_model_id=str(coverage[0]["model_id"]),
        nominated_runtime_digest=str(coverage[0]["runtime_digest"]),
        nominated_report_sha256=str(coverage[0]["economic_report_sha256"]),
        selection_sha256="",
    )
    return replace(
        provisional,
        selection_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _panel() -> Round28AICasePanel:
    provisional = Round28AICasePanel(
        partition_role="sealed",
        source_run_id="sealed-run",
        model_name="l2_offset_logistic",
        model_sha256="2" * 64,
        selection_claim_sha256="d" * 64,
        source_audit_sha256="f" * 64,
        economic_config={},
        evaluated_condition_count=0,
        evaluated_condition_ids_sha256="3" * 64,
        baseline_candidate_population_sha256="4" * 64,
        selection_reason_counts={},
        cases=(),
        panel_sha256="",
    )
    return replace(
        provisional,
        panel_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def test_round28_ai_sealed_case_phase_exposes_no_target_or_outcome_path() -> None:
    destinations = {
        action.dest for action in _case_parser()._actions if action.dest != "help"
    }

    assert "sealed_source_database" in destinations
    assert "ai_selection_claim" in destinations
    assert "nominated_host_report" in destinations
    assert all(
        fragment not in destination
        for destination in destinations
        for fragment in ("target", "outcome", "resolution")
    )
    assert all("model" not in destination for destination in destinations)


def test_round28_ai_sealed_economic_phase_cannot_select_or_change_model() -> None:
    destinations = {
        action.dest
        for action in _economic_parser()._actions
        if action.dest != "help"
    }

    assert {
        "round27_target_store",
        "sealed_source_database",
        "ai_selection_claim",
        "nominated_host_report",
        "sealed_case_panel",
        "sealed_inference_report",
        "sealed_round28_economic_report",
        "sealed_ai_economic_report",
        "terminal_ai_result",
    } <= destinations
    assert all("model" not in destination for destination in destinations)


def test_round28_ai_sealed_case_result_supports_no_nomination() -> None:
    result = _result(
        ai_selection_sha256="a" * 64,
        status="no_candidate_nominated",
        model_id=None,
        panel_sha256=None,
        inference_report_sha256=None,
    )

    assert result["status"] == "no_candidate_nominated"
    assert result["target_accessed"] is False
    assert result["orders_submitted"] is False
    assert len(result["result_sha256"]) == 64


def test_round28_ai_sealed_terminal_binds_nomination_and_denies_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    panel = _panel()
    baseline = {
        "report_sha256": "5" * 64,
        "selection_claim_sha256": panel.selection_claim_sha256,
    }
    inference = {
        "candidate": {
            "model_id": selection.nominated_model_id,
            "runtime_digest": selection.nominated_runtime_digest,
        },
        "report_sha256": "6" * 64,
    }
    ai_report = {
        "candidate": dict(inference["candidate"]),
        "partition_role": "sealed",
        "case_panel_sha256": panel.panel_sha256,
        "inference_report_sha256": inference["report_sha256"],
        "round28_economic_report_sha256": baseline["report_sha256"],
        "selection_claim_sha256": panel.selection_claim_sha256,
        "matched_after_cost_uplift_gate_passed": True,
        "report_sha256": "7" * 64,
    }
    monkeypatch.setattr(
        sealed_ai,
        "validate_round28_sealed_economic_report",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        sealed_ai,
        "validate_round28_ai_economic_report",
        lambda value: dict(value),
    )

    terminal = build_round28_ai_sealed_terminal_result(
        ai_selection=selection,
        panel=panel,
        inference_report=inference,
        sealed_round28_economic_report=baseline,
        sealed_ai_economic_report=ai_report,
    )

    assert terminal["observed_after_cost_ai_uplift"] is True
    assert terminal["model_prompt_or_threshold_changed_after_selection"] is False
    assert terminal["edge_claim"] is False
    assert terminal["profitability_claim"] is False
    assert terminal["orders_submitted"] is False
    assert terminal["trading_authority"] is False

    ai_report["case_panel_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="terminal binding differs"):
        build_round28_ai_sealed_terminal_result(
            ai_selection=selection,
            panel=panel,
            inference_report=inference,
            sealed_round28_economic_report=baseline,
            sealed_ai_economic_report=ai_report,
        )


def test_completed_terminal_rerun_does_not_open_target_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "target.duckdb",
        "source.duckdb",
        "selection.json",
        "host.json",
        "panel.json",
        "inference.json",
        "case-result.json",
        "baseline.json",
        "economics.json",
        "terminal.json",
    )
    paths = {name: tmp_path / name for name in names}
    for path in paths.values():
        path.write_text("{}", encoding="ascii")
    selection = _selection()
    panel = _panel()
    candidate = SimpleNamespace(
        model_id=selection.nominated_model_id,
        runtime_digest=selection.nominated_runtime_digest,
    )
    inference = SimpleNamespace(
        candidate={
            "model_id": candidate.model_id,
            "runtime_digest": candidate.runtime_digest,
        },
        report_sha256="6" * 64,
        asdict=lambda: {
            "candidate": {
                "model_id": candidate.model_id,
                "runtime_digest": candidate.runtime_digest,
            },
            "report_sha256": "6" * 64,
        },
    )
    terminal = {"terminal": True}
    monkeypatch.setattr(sealed_operator, "load_round28_ai_contract", lambda _root: {})
    monkeypatch.setattr(
        sealed_operator,
        "round28_ai_candidate_selection_from_mapping",
        lambda _value: selection,
    )
    monkeypatch.setattr(
        sealed_operator,
        "validate_round28_ai_host_report",
        lambda _value, *, contract: ({}, candidate),
    )
    monkeypatch.setattr(
        sealed_operator,
        "round28_ai_case_panel_from_mapping",
        lambda _value: panel,
    )
    monkeypatch.setattr(
        sealed_operator,
        "round28_ai_inference_report_from_mapping",
        lambda _value: inference,
    )
    monkeypatch.setattr(
        sealed_operator,
        "validate_round28_ai_inference_report",
        lambda *_args, **_kwargs: inference,
    )
    monkeypatch.setattr(sealed_operator, "_validate_case_result", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sealed_operator,
        "validate_round28_sealed_economic_report",
        lambda _value: {},
    )
    monkeypatch.setattr(
        sealed_operator,
        "validate_round28_ai_economic_report",
        lambda _value: {},
    )
    monkeypatch.setattr(
        sealed_operator,
        "build_round28_ai_sealed_terminal_result",
        lambda **_kwargs: terminal,
    )
    monkeypatch.setattr(
        sealed_operator,
        "Round27TargetStore",
        lambda *_args, **_kwargs: pytest.fail("target store was opened"),
    )
    paths["terminal.json"].write_text(json.dumps(terminal), encoding="ascii")

    exit_code = sealed_operator.main(
        [
            "--repository",
            str(tmp_path),
            "--round27-target-store",
            "target.duckdb",
            "--sealed-source-database",
            "source.duckdb",
            "--ai-selection-claim",
            "selection.json",
            "--nominated-host-report",
            "host.json",
            "--sealed-case-panel",
            "panel.json",
            "--sealed-inference-report",
            "inference.json",
            "--sealed-case-result",
            "case-result.json",
            "--sealed-round28-economic-report",
            "baseline.json",
            "--sealed-ai-economic-report",
            "economics.json",
            "--terminal-ai-result",
            "terminal.json",
        ]
    )

    assert exit_code == 0

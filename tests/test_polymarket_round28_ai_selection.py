from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

import simple_ai_trading.polymarket_round28_ai_selection as selection_module
from simple_ai_trading.polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_MODEL_IDS,
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    round28_ai_candidate_from_contract,
)
from simple_ai_trading.polymarket_round28_ai_selection import (
    build_round28_ai_host_failure,
    round28_ai_candidate_selection_from_mapping,
    select_round28_ai_candidate,
    validate_round28_ai_host_failure,
)
from test_polymarket_round28_ai_inference import _host_report


ROOT = Path(__file__).resolve().parents[1]
_PANEL_SHA256 = "c" * 64
_BASELINE_SHA256 = "d" * 64


def _economic_report(candidate, host_report, marker: str, *, passed: bool):
    lower = "0.2" if marker == "1" else "0.1"
    return {
        "report_sha256": marker * 64,
        "partition_role": "selection",
        "candidate": asdict(candidate),
        "host_qualification_report_sha256": host_report["report_sha256"],
        "case_panel_sha256": _PANEL_SHA256,
        "round28_economic_report_sha256": _BASELINE_SHA256,
        "paired_scenarios": [
            {
                "base_delay_ms": delay,
                "paired_condition_bootstrap": {"ci95_lower": lower},
                "paired_mean_net_pnl_delta_quote": lower,
                "maximum_drawdown_delta_fraction": "0",
            }
            for delay in (250, 500, 1_000, 2_000)
        ],
        "matched_after_cost_uplift_gate_passed": passed,
    }


def _failure(contract, model_id: str, marker: str):
    return build_round28_ai_host_failure(
        contract=contract,
        model_id=model_id,
        phase="artifact_download",
        error_code="artifact_unavailable",
        private_detail_sha256=marker * 64,
        observed_at_ms=1,
    )


def test_round28_ai_selection_accounts_for_every_candidate_and_ranks_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_round28_ai_contract(ROOT)
    qwen = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    oda = round28_ai_candidate_from_contract(
        contract,
        model_id="OpenDataArena/ODA-Fin-SFT-8B",
    )
    qwen_host = _host_report(contract, qwen)
    oda_host = _host_report(contract, oda)
    qwen_economic = _economic_report(qwen, qwen_host, "1", passed=True)
    oda_economic = _economic_report(oda, oda_host, "2", passed=True)
    challenger_failure = _failure(
        contract,
        "OpenDataArena/ODA-Fin-RL-8B",
        "3",
    )
    monkeypatch.setattr(
        selection_module,
        "validate_round28_ai_economic_report",
        lambda value: dict(value),
    )

    selection = select_round28_ai_candidate(
        contract=contract,
        host_qualification_reports=(qwen_host, oda_host),
        host_failure_reports=(challenger_failure,),
        economic_reports=(qwen_economic, oda_economic),
        case_panel_sha256=_PANEL_SHA256,
        round28_economic_report_sha256=_BASELINE_SHA256,
    )

    assert tuple(
        item["model_id"] for item in selection.candidate_coverage
    ) == POLYMARKET_ROUND28_AI_MODEL_IDS
    assert [item["status"] for item in selection.candidate_coverage] == [
        "evaluated",
        "evaluated",
        "host_rejected",
    ]
    assert selection.nominated_model_id == qwen.model_id
    assert selection.nominated_runtime_digest == qwen.runtime_digest
    assert round28_ai_candidate_selection_from_mapping(selection.asdict()) == selection


def test_round28_ai_selection_requires_two_evaluated_candidates_for_nomination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_round28_ai_contract(ROOT)
    qwen = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    qwen_host = _host_report(contract, qwen)
    qwen_economic = _economic_report(qwen, qwen_host, "1", passed=True)
    failures = tuple(
        _failure(contract, model_id, str(index + 4))
        for index, model_id in enumerate(POLYMARKET_ROUND28_AI_MODEL_IDS[1:])
    )
    monkeypatch.setattr(
        selection_module,
        "validate_round28_ai_economic_report",
        lambda value: dict(value),
    )

    selection = select_round28_ai_candidate(
        contract=contract,
        host_qualification_reports=(qwen_host,),
        host_failure_reports=failures,
        economic_reports=(qwen_economic,),
        case_panel_sha256=_PANEL_SHA256,
        round28_economic_report_sha256=_BASELINE_SHA256,
    )

    assert selection.nominated_model_id is None
    assert selection.nominated_report_sha256 is None


def test_round28_ai_selection_rejects_incomplete_candidate_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_round28_ai_contract(ROOT)
    qwen = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    qwen_host = _host_report(contract, qwen)
    monkeypatch.setattr(
        selection_module,
        "validate_round28_ai_economic_report",
        lambda value: dict(value),
    )

    with pytest.raises(ValueError, match="coverage is incomplete"):
        select_round28_ai_candidate(
            contract=contract,
            host_qualification_reports=(qwen_host,),
            host_failure_reports=(),
            economic_reports=(
                _economic_report(qwen, qwen_host, "1", passed=False),
            ),
            case_panel_sha256=_PANEL_SHA256,
            round28_economic_report_sha256=_BASELINE_SHA256,
        )


def test_round28_ai_host_failure_is_sanitized_and_tamper_evident() -> None:
    contract = load_round28_ai_contract(ROOT)
    report = _failure(contract, "OpenDataArena/ODA-Fin-RL-8B", "a")

    assert validate_round28_ai_host_failure(report, contract=contract) == report
    assert "message" not in report
    serialized = json.dumps(report).lower()
    assert all(
        marker not in serialized
        for marker in ("c:\\", "http://", "https://", "ghp_", "api_key")
    )

    tampered = dict(report)
    tampered["error_code"] = "provider_unavailable"
    with pytest.raises(ValueError, match="host failure report differs"):
        validate_round28_ai_host_failure(tampered, contract=contract)

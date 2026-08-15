from __future__ import annotations

from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_ai_sealed_access as access_module
from simple_ai_trading.polymarket_round28_ai_sealed_access import (
    build_round28_ai_sealed_access_receipt,
    validate_round28_ai_sealed_access_receipt,
)
from tools.authorize_polymarket_round28_ai_sealed_cases import (
    _parser as _authorization_parser,
)
from tools.run_polymarket_round28_ai_sealed_cases_enveloped import (
    _parser as _sealed_case_parser,
)
from tools.run_polymarket_round28_ai_sealed_cases_enveloped import _result


def _inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    pair = SimpleNamespace(
        model_family="l2_offset_logistic",
        augmented_model=SimpleNamespace(
            model_name="l2_offset_logistic",
            model_sha256="7" * 64,
        ),
    )
    panel = SimpleNamespace(
        partition_role="selection",
        panel_sha256="5" * 64,
        selection_claim_sha256="4" * 64,
        source_audit_sha256="3" * 64,
        model_name="l2_offset_logistic",
        model_sha256="7" * 64,
    )
    selection = SimpleNamespace(
        case_panel_sha256=panel.panel_sha256,
        round28_economic_report_sha256="6" * 64,
        selection_sha256="8" * 64,
        nominated_model_id="OpenDataArena/ODA-Fin-SFT-8B",
        nominated_runtime_digest="9" * 64,
    )
    manifest = {"manifest_sha256": "3" * 64}
    monkeypatch.setattr(
        access_module,
        "_validate_selection_lineage",
        lambda **_kwargs: (manifest, pair, panel, selection),
    )
    monkeypatch.setattr(
        access_module,
        "validate_round28_sealed_access_artifacts",
        lambda **_kwargs: pair,
    )
    return {
        "contract": {"contract_sha256": "1" * 64},
        "preregistration": {"preregistration_sha256": "2" * 64},
        "selection_input_manifest": manifest,
        "selection_claim": {"claim_sha256": "4" * 64},
        "selection_economic_report": {
            "report_sha256": "6" * 64,
            "resolution_evidence_sha256": "a" * 64,
        },
        "selection_ai_case_panel": {"panel_sha256": "5" * 64},
        "ai_selection_claim": {"selection_sha256": "8" * 64},
        "selection_resolution_evidence_sha256": "a" * 64,
    }


def test_sealed_access_receipt_exposes_identities_not_selection_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)

    receipt = build_round28_ai_sealed_access_receipt(**inputs)

    assert receipt["selection_after_cost_gate_passed"] is True
    assert receipt["selection_metrics_exposed_to_case_process"] is False
    assert receipt["target_data_exposed_to_case_process"] is False
    assert receipt["orders_submitted"] is False
    assert all(
        fragment not in key
        for key in receipt
        for fragment in (
            "return",
            "profit",
            "drawdown",
            "trade_count",
            "win_rate",
        )
    )
    validation_inputs = dict(inputs)
    validation_inputs.pop("selection_economic_report")
    validation_inputs.pop("selection_resolution_evidence_sha256")
    validated, _pair, _panel, _selection = (
        validate_round28_ai_sealed_access_receipt(
            receipt,
            **validation_inputs,
        )
    )
    assert validated == receipt


def test_sealed_access_receipt_rejects_semantic_tampering_with_fresh_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    receipt = build_round28_ai_sealed_access_receipt(**inputs)
    receipt["selection_after_cost_gate_passed"] = False
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = access_module._canonical_sha256(receipt)
    validation_inputs = dict(inputs)
    validation_inputs.pop("selection_economic_report")
    validation_inputs.pop("selection_resolution_evidence_sha256")

    with pytest.raises(ValueError, match="access receipt differs"):
        validate_round28_ai_sealed_access_receipt(
            receipt,
            **validation_inputs,
        )


def test_sealed_case_process_has_no_sensitive_evaluation_input() -> None:
    destinations = {
        action.dest
        for action in _sealed_case_parser()._actions
        if action.dest != "help"
    }
    authorization_destinations = {
        action.dest
        for action in _authorization_parser()._actions
        if action.dest != "help"
    }

    assert "sealed_access_receipt" in destinations
    assert "selection_economic_report" in authorization_destinations
    assert all(
        fragment not in destination
        for destination in destinations
        for fragment in ("target", "outcome", "resolution", "economic", "pnl")
    )


def test_enveloped_no_nomination_result_binds_access_receipt() -> None:
    result = _result(
        access_receipt_sha256="a" * 64,
        ai_selection_sha256="b" * 64,
        status="no_candidate_nominated",
        model_id=None,
        prompt_envelope_report_sha256=None,
        panel_sha256=None,
        inference_report_sha256=None,
    )

    assert result["status"] == "no_candidate_nominated"
    assert result["sealed_access_receipt_sha256"] == "a" * 64
    assert result["selection_metrics_accessed"] is False
    assert result["orders_submitted"] is False
    assert len(result["result_sha256"]) == 64

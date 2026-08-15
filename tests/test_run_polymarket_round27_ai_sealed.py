from __future__ import annotations

from tools.run_polymarket_round27_ai_sealed import _parser, _terminal_result


def test_round27_sealed_ai_operator_requires_frozen_receipts_and_targets() -> None:
    destinations = {
        action.dest
        for action in _parser()._actions
        if action.dest != "help"
    }

    assert {
        "target_store",
        "sealed_source_database",
        "ai_selection_claim",
        "sealed_case_panel",
        "sealed_inference_report",
        "baseline_sealed_economic_report",
        "sealed_ai_economic_report",
        "terminal_ai_result",
    } <= destinations
    assert all("model" not in destination for destination in destinations)


def test_round27_sealed_ai_terminal_never_grants_authority() -> None:
    terminal = _terminal_result(
        ai_selection_sha256="a" * 64,
        nominated_model_id="model-id",
        panel_sha256="b" * 64,
        inference_report_sha256="c" * 64,
        baseline_report_sha256="d" * 64,
        ai_report={
            "report_sha256": "e" * 64,
            "matched_after_cost_uplift_gate_passed": True,
        },
    )

    assert terminal["sealed_matched_after_cost_uplift_gate_passed"] is True
    assert terminal["observed_after_cost_ai_uplift"] is True
    assert terminal["model_prompt_or_threshold_changed_after_selection"] is False
    assert terminal["edge_claim"] is False
    assert terminal["profitability_claim"] is False
    assert terminal["orders_submitted"] is False
    assert terminal["trading_authority"] is False

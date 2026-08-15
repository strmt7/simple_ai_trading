from __future__ import annotations

from tools.run_polymarket_round27_ai_sealed_cases import _parser, _result


def test_round27_sealed_ai_case_operator_exposes_no_target_path() -> None:
    destinations = {
        action.dest
        for action in _parser()._actions
        if action.dest != "help"
    }

    assert "feature_store" in destinations
    assert "sealed_source_database" in destinations
    assert "ai_selection_claim" in destinations
    assert all(
        fragment not in destination
        for destination in destinations
        for fragment in ("target", "outcome", "resolution", "economic_report")
    )


def test_round27_sealed_ai_case_result_represents_no_nomination() -> None:
    result = _result(
        selection_sha256="a" * 64,
        status="no_candidate_nominated",
        model_id=None,
        panel_sha256=None,
        inference_report_sha256=None,
    )

    assert result["status"] == "no_candidate_nominated"
    assert result["target_accessed"] is False
    assert result["orders_submitted"] is False
    assert len(result["result_sha256"]) == 64

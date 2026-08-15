from __future__ import annotations

from tools.run_polymarket_round27_sealed import _terminal_result
from tools.run_polymarket_round27_selection import _canonical_sha256


def _artifact(field: str, value: str) -> dict[str, object]:
    return {field: value}


def test_terminal_result_requires_both_frozen_sealed_gates() -> None:
    terminal = _terminal_result(
        contract_sha256="a" * 64,
        selection_claim=_artifact("claim_sha256", "b" * 64),
        selection_economic_claim=_artifact("claim_sha256", "c" * 64),
        selection_economic_report=_artifact("report_sha256", "d" * 64),
        sealed_prediction={
            "result_sha256": "e" * 64,
            "prediction_edge_gate_passed": True,
        },
        sealed_economics={
            "report_sha256": "f" * 64,
            "economic_edge_gate_passed": False,
        },
    )

    assert terminal["prediction_edge_gate_passed"] is True
    assert terminal["economic_edge_gate_passed"] is False
    assert terminal["observed_after_cost_edge_gate_passed"] is False
    assert terminal["model_or_threshold_changed_after_selection"] is False
    assert terminal["edge_claim"] is False
    assert terminal["orders_submitted"] is False
    body = dict(terminal)
    claimed = body.pop("result_sha256")
    assert claimed == _canonical_sha256(body)

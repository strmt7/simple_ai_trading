from __future__ import annotations

import pytest

from tools.run_polymarket_round27_ai_selection import _parser, _target_role


def test_round27_ai_selection_operator_joins_only_persisted_ai_receipts() -> None:
    destinations = {
        action.dest
        for action in _parser()._actions
        if action.dest != "help"
    }

    assert {
        "target_store",
        "selection_source_database",
        "case_panel",
        "qwen_inference_report",
        "oda_inference_report",
        "baseline_economic_report",
        "qwen_economic_report",
        "oda_economic_report",
        "ai_selection_claim",
    } <= destinations
    assert all("model" not in destination for destination in destinations)


def test_round27_ai_selection_operator_requires_exact_target_role() -> None:
    role = _target_role(
        {
            "roles": [
                {
                    "role": "selection",
                    "evidence_chain_sha256": "a" * 64,
                }
            ]
        }
    )
    assert role["role"] == "selection"

    with pytest.raises(ValueError, match="target role differs"):
        _target_role({"roles": []})

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
    load_round27_model_amendment,
    validate_round27_model_amendment,
)


_ROOT = Path(__file__).resolve().parents[1]


def test_round27_model_amendment_is_exact_and_pre_target() -> None:
    amendment = load_round27_model_amendment(_ROOT)

    assert amendment["amendment_sha256"] == POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
    assert amendment["knowledge_at_freeze"] == {
        "ai_assist_economic_metrics_computed": False,
        "model_fitted_on_stage1": False,
        "official_outcomes_accessed": False,
        "performance_metrics_computed": False,
        "sealed_partition_accessed": False,
        "selection_partition_accessed": False,
        "stage1_capture_started": True,
        "stage1_feature_rows_accessed_or_materialized": False,
    }
    assert (
        amendment["correction"][
            "economic_or_prediction_gate_thresholds_changed"
        ]
        is False
    )
    assert (
        amendment["correction"]["confidence_interval_method"]
        == "stationary_bootstrap_block_length_sensitivity_envelope"
    )
    assert amendment["predecessor_amendment_sha256"] == (
        "e3ce6285cea10337f50383cdd2b89dd048d8f015f889adaa9cc0045088a44833"
    )
    assert amendment["correction"]["expected_block_lengths_conditions"] == [
        1,
        4,
        12,
    ]
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_model.py"
        ]["corrected"]
        == "b01a98cfa846aa98882ba381610256029a9d4e05aaec4d4a4c4c0531142987c8"
    )


def test_round27_model_amendment_rejects_tampering() -> None:
    amendment = load_round27_model_amendment(_ROOT)
    tampered = copy.deepcopy(amendment)
    tampered["authority"]["edge_claim"] = True

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_model_amendment(tampered)

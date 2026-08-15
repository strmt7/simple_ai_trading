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
            "prediction_gate_numeric_thresholds_changed"
        ]
        is False
    )
    assert amendment["correction"]["brier_score_confidence_gate_added"] is True
    assert amendment["correction"]["prediction_gate_became_stricter"] is True
    assert amendment["predecessor_amendment_sha256"] == (
        "3d23b811f964df8d91f2f08fc5e5088293770ec2d758f4aff252173d30a425c0"
    )
    assert amendment["correction"][
        "fixed_window_price_path_anchor"
    ] == "last_receipt_at_or_before_window_start_when_available"
    assert set(amendment["superseded_source_text_sha256"]) == {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py",
        "src/simple_ai_trading/polymarket_round27_ai_economics.py",
        "src/simple_ai_trading/polymarket_round27_economics.py",
        "src/simple_ai_trading/polymarket_round27_experiment.py",
        "src/simple_ai_trading/polymarket_round27_features.py",
        "src/simple_ai_trading/polymarket_round27_model.py",
    }
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_model.py"
        ]["corrected"]
        == "200b95a41b984625e80da840b0db3695a44896763581ce1b59cc93bac7cd3177"
    )


def test_round27_model_amendment_rejects_tampering() -> None:
    amendment = load_round27_model_amendment(_ROOT)
    tampered = copy.deepcopy(amendment)
    tampered["authority"]["edge_claim"] = True

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_model_amendment(tampered)

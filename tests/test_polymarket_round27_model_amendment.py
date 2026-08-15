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
    assert amendment["correction"][
        "same_millisecond_receipts_treated_as_ordering_ambiguous"
    ] is True
    assert amendment["correction"][
        "economic_execution_cutoff_became_stricter"
    ] is True
    assert amendment["predecessor_amendment_sha256"] == (
        "a2ee7a5c3f89b1bca66ca3f8dd673c760719b3f47154f678b5186ee825ae3b1e"
    )
    assert amendment["correction"][
        "execution_and_markout_receipt_cutoff"
    ] == "strictly_after_target_wall_ms"
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
        == "73de58ec5c5a1c1b79119779ff2035c7d73eabca3807aff83c07755f14123774"
    )
    assert amendment["superseded_source_text_sha256"][
        "src/simple_ai_trading/polymarket_round27_features.py"
    ]["corrected"] == (
        "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
    )


def test_round27_model_amendment_rejects_tampering() -> None:
    amendment = load_round27_model_amendment(_ROOT)
    tampered = copy.deepcopy(amendment)
    tampered["authority"]["edge_claim"] = True

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_model_amendment(tampered)

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
    assert amendment["correction"]["economic_gate_thresholds_changed"] is False
    assert (
        amendment["correction"][
            "execution_limit_revalidated_against_execution_book_active_tick_size"
        ]
        is True
    )
    assert amendment["predecessor_amendment_sha256"] == (
        "8c4c7e48062446d9b6d87c716c22004fa729be094388ce6202480cc6e2098afd"
    )
    assert amendment["correction"]["ai_prompt_fields_changed"] is False
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_economics.py"
        ]["corrected"]
        == "e7d465cdbca29b5f3d94d7f3c3d4be80409a961ef31139f846b757ac6ebf4714"
    )


def test_round27_model_amendment_rejects_tampering() -> None:
    amendment = load_round27_model_amendment(_ROOT)
    tampered = copy.deepcopy(amendment)
    tampered["authority"]["edge_claim"] = True

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_model_amendment(tampered)

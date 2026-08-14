from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_round25_forensic_model import (
    validate_round25_forensic_model_fit,
    validate_round25_forensic_prediction_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
MODEL_FIT = RESEARCH / "round-025-v2-forensic-model-fit-2026-08-14.json"
PREDICTIONS = (
    RESEARCH / "round-025-v2-forensic-selection-predictions-2026-08-14.json"
)
MODEL_FIT_FILE_SHA256 = (
    "2114f71bbfdca8b325c78e1a2506045ec8fb5ae5e0377753515c0d7849c4bb09"
)
PREDICTIONS_FILE_SHA256 = (
    "235e5ddddb3692089a2ee43a2d8472a94ecd5d9a053c9bdeead55a028906615b"
)


def test_forensic_model_fit_is_hash_bound_and_truthful() -> None:
    assert hashlib.sha256(MODEL_FIT.read_bytes()).hexdigest() == (
        MODEL_FIT_FILE_SHA256
    )
    value = validate_round25_forensic_model_fit(
        json.loads(MODEL_FIT.read_text(encoding="ascii"))
    )

    assert value["model_fit_sha256"] == (
        "ee7ac00435020847dffa0eac14bc7d76fc88298abf60ab3e69786a210588c03c"
    )
    assert value["selected_candidate_id"] == "market-prior-v1"
    assert value["train_condition_count"] == 42
    assert value["calibration_condition_count"] == 12
    assert value["candidate_metrics"]["market-prior-v1"][
        "condition_equal_log_loss"
    ] == 0.38231583160091825
    assert value["candidate_metrics"]["l2-logistic-residual-v1"][
        "condition_equal_log_loss"
    ] == 0.5872658077149272
    assert value["selection_targets_accessed"] is False
    assert value["profitability_claim"] is False


def test_forensic_selection_predictions_were_frozen_before_targets() -> None:
    assert hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest() == (
        PREDICTIONS_FILE_SHA256
    )
    value = validate_round25_forensic_prediction_artifact(
        json.loads(PREDICTIONS.read_text(encoding="ascii"))
    )

    assert value["prediction_artifact_sha256"] == (
        "773bef00cde4b7e7c73d01bfcc2ed809a8a45d4c47dbd31b71047ffe1d13a271"
    )
    assert value["model_fit_sha256"] == (
        "ee7ac00435020847dffa0eac14bc7d76fc88298abf60ab3e69786a210588c03c"
    )
    assert value["access_freeze"]["selected_candidate_id"] == "market-prior-v1"
    assert value["access_freeze"]["condition_count"] == 14
    assert value["selection_targets_accessed"] is False
    assert all(row["action"] == "abstain" for row in value["trade_policy"])
    assert value["profitability_claim"] is False

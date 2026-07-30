from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round17_model import (
    Round17DevelopmentPanel,
    fit_round17_development_pretest,
    predict_round17_candidate,
    validate_round17_pretest_artifact,
)


START_MS = 1_800_000_000_000
DATASET_SHA256 = "d" * 64


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _panel(
    role: str,
    *,
    first_event_start_ms: int,
    condition_count: int,
) -> Round17DevelopmentPanel:
    condition_ids: list[str] = []
    event_starts: list[int] = []
    decisions: list[int] = []
    labels: list[float] = []
    features: list[np.ndarray] = []
    signal_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("chainlink_log_return_1000ms")
    structural_index = POLYMARKET_ROUND17_FEATURE_NAMES.index(
        "structural_probability_up"
    )
    prior_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")
    for condition_index in range(condition_count):
        event_start = first_event_start_ms + condition_index * 300_000
        label = float(condition_index % 2)
        condition = "0x" + _sha256([role, condition_index])
        for row_index, offset in enumerate((30_000, 60_000, 90_000)):
            row = np.zeros(len(POLYMARKET_ROUND17_FEATURE_NAMES), dtype=np.float64)
            row[structural_index] = 0.5
            row[prior_index] = 0.5
            row[signal_index] = (2.0 if label else -2.0) + row_index * 0.01
            condition_ids.append(condition)
            event_starts.append(event_start)
            decisions.append(event_start + offset)
            labels.append(label)
            features.append(row)
    return Round17DevelopmentPanel(
        role=role,
        condition_ids=np.asarray(condition_ids, dtype=object),
        event_start_ms=np.asarray(event_starts, dtype=np.int64),
        decision_time_ms=np.asarray(decisions, dtype=np.int64),
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
        dataset_sha256=DATASET_SHA256,
    ).validate()


def _development_panels() -> tuple[
    Round17DevelopmentPanel,
    Round17DevelopmentPanel,
    Round17DevelopmentPanel,
]:
    train = _panel("train", first_event_start_ms=START_MS, condition_count=24)
    train_end = int(np.max(train.event_start_ms)) + 300_000
    calibration = _panel(
        "tune_calibration",
        first_event_start_ms=train_end + 3_600_000,
        condition_count=10,
    )
    calibration_end = int(np.max(calibration.event_start_ms)) + 300_000
    selection = _panel(
        "tune_selection",
        first_event_start_ms=calibration_end + 3_600_000,
        condition_count=10,
    )
    return train, calibration, selection


def test_round17_pretest_is_test_blind_hash_bound_and_identifiable() -> None:
    train, calibration, selection = _development_panels()

    artifact = fit_round17_development_pretest(
        train,
        calibration,
        selection,
        compute_backend="cpu",
    )
    verified = validate_round17_pretest_artifact(artifact)

    assert len(verified["candidate_ledger"]) == 15
    assert verified["test_features_accessed"] is False
    assert verified["test_targets_accessed"] is False
    assert verified["execution_simulation_completed"] is False
    assert verified["profitability_claim"] is False
    assert verified["live_trading_authority"] is False
    assert verified["compute"]["lightgbm_backend_kind"] == "cpu"
    assert verified["development_accepted"] is True
    selection_metrics = verified["candidate_ledger"][0]["selection"]
    assert {
        "condition_balanced_brier",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "condition_weighted_balanced_accuracy",
        "condition_weighted_matthews_correlation",
    }.issubset(selection_metrics)
    selected = verified["selected_candidate"]
    prediction = predict_round17_candidate(selected, selection)
    assert prediction.shape == selection.labels.shape
    assert np.all(np.isfinite(prediction))
    assert np.mean((prediction >= 0.5) == selection.labels) == 1.0


def test_round17_pretest_rejects_artifact_tampering() -> None:
    train, calibration, selection = _development_panels()
    artifact = fit_round17_development_pretest(
        train,
        calibration,
        selection,
        compute_backend="cpu",
    )
    artifact["profitability_claim"] = True

    with pytest.raises(ValueError, match="integrity differs"):
        validate_round17_pretest_artifact(artifact)


def test_round17_pretest_rejects_rehashed_semantic_drift() -> None:
    train, calibration, selection = _development_panels()
    artifact = fit_round17_development_pretest(
        train,
        calibration,
        selection,
        compute_backend="cpu",
    )
    artifact["development_accepted"] = not artifact["development_accepted"]
    artifact.pop("pretest_sha256")
    artifact["pretest_sha256"] = _sha256(artifact)

    with pytest.raises(ValueError, match="integrity differs"):
        validate_round17_pretest_artifact(artifact)


def test_round17_pretest_rejects_short_embargo() -> None:
    train, calibration, selection = _development_panels()
    shifted = Round17DevelopmentPanel(
        role=calibration.role,
        condition_ids=calibration.condition_ids,
        event_start_ms=calibration.event_start_ms - 3_600_000,
        decision_time_ms=calibration.decision_time_ms - 3_600_000,
        features=calibration.features,
        labels=calibration.labels,
        dataset_sha256=calibration.dataset_sha256,
    ).validate()

    with pytest.raises(ValueError, match="embargo is too short"):
        fit_round17_development_pretest(
            train,
            shifted,
            selection,
            compute_backend="cpu",
        )

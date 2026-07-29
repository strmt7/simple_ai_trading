from __future__ import annotations

import hashlib

import numpy as np
import pytest

from simple_ai_trading.polymarket_historical_dataset import FEATURE_NAMES
from simple_ai_trading.polymarket_historical_model import (
    HistoricalModelPanel,
    _paired_block_bootstrap,
    condition_balanced_binary_metrics,
    fit_historical_pretest_candidates,
    load_historical_model_panel,
    predict_historical_candidate,
)


DATASET_SHA = hashlib.sha256(b"synthetic-historical-panel").hexdigest()


def _panel(
    *,
    role: str,
    conditions: int,
    seed: int,
    start_ms: int,
) -> HistoricalModelPanel:
    generator = np.random.default_rng(seed)
    rows = conditions * 8
    latent = generator.normal(size=conditions)
    labels = (latent > 0.0).astype(np.float64)
    features = generator.normal(scale=0.2, size=(rows, len(FEATURE_NAMES))).astype(
        np.float32
    )
    condition_ids = np.asarray(
        [
            f"{role}-{condition:04d}"
            for condition in range(conditions)
            for _ in range(8)
        ],
        dtype=object,
    )
    repeated_latent = np.repeat(latent, 8)
    repeated_labels = np.repeat(labels, 8)
    features[:, 0] = repeated_latent + generator.normal(scale=0.05, size=rows)
    features[:, -4:] = generator.normal(size=(rows, 4))
    event_start_ms = np.repeat(
        start_ms + np.arange(conditions, dtype=np.int64) * 300_000,
        8,
    )
    decision_time_ms = event_start_ms + np.tile(
        np.arange(30, 241, 30, dtype=np.int64) * 1_000,
        conditions,
    )
    panel = HistoricalModelPanel(
        condition_ids=condition_ids,
        roles=np.full(rows, role, dtype=object),
        event_start_ms=event_start_ms,
        decision_time_ms=decision_time_ms,
        features=features,
        labels=repeated_labels,
        dataset_sha256=DATASET_SHA,
    )
    panel.validate(expected_roles=(role,))
    return panel


def test_frozen_candidates_detect_a_real_binance_feature() -> None:
    train = _panel(role="train", conditions=120, seed=1, start_ms=1_000_000)
    tune = _panel(role="tune", conditions=48, seed=2, start_ms=100_000_000)

    candidates = fit_historical_pretest_candidates(
        train,
        tune,
        compute_backend="cpu",
    )

    assert len(candidates) == 4
    assert {candidate["family"] for candidate in candidates} == {
        "training_prevalence",
        "calendar_ridge_logistic",
        "binance_ridge_logistic",
        "binance_shallow_lightgbm",
    }
    control_loss = min(
        float(candidate["tune_metrics"]["log_loss"])
        for candidate in candidates
        if candidate["kind"] == "control"
    )
    challenger = min(
        (candidate for candidate in candidates if candidate["kind"] == "challenger"),
        key=lambda candidate: float(candidate["tune_metrics"]["log_loss"]),
    )
    prediction = predict_historical_candidate(challenger, tune.features)
    metrics = condition_balanced_binary_metrics(tune, prediction)

    assert float(metrics["log_loss"]) < control_loss
    assert float(metrics["balanced_accuracy"]) > 0.9
    assert np.all((_EPSILON <= prediction) & (prediction <= 1.0 - _EPSILON))


def test_condition_balanced_metrics_do_not_reward_duplicate_rows() -> None:
    panel = _panel(role="test", conditions=20, seed=4, start_ms=500_000_000)
    probability = np.where(panel.labels == 1.0, 0.8, 0.2)
    metrics = condition_balanced_binary_metrics(panel, probability)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.04)


def test_test_panel_is_rejected_before_one_use_state() -> None:
    class Store:
        state = "pretest_complete"

        def connect(self):
            raise AssertionError("test target SQL must not run before authorization")

    with pytest.raises(ValueError, match="not authorized"):
        load_historical_model_panel(Store(), roles=("test",))  # type: ignore[arg-type]


def test_paired_bootstrap_obeys_contract_sample_and_repetition_gates() -> None:
    values = np.linspace(-0.01, 0.02, 3_500, dtype=np.float64)

    result = _paired_block_bootstrap(
        values,
        repetitions=100,
        minimum_conditions=3_500,
    )

    assert result["repetitions"] == 100
    assert result["block_conditions"] == 12
    assert float(result["lower_95"]) < float(result["upper_95"])
    with pytest.raises(ValueError, match="bootstrap inputs"):
        _paired_block_bootstrap(
            values[:-1],
            repetitions=100,
            minimum_conditions=3_500,
        )


_EPSILON = 1e-6

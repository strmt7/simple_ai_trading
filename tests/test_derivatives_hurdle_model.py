from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from simple_ai_trading.derivatives_hurdle_data import (
    DerivativesHurdleDataset,
    FundingState,
    _funding_in_holding_window,
    _grid_age,
)
from simple_ai_trading.derivatives_hurdle_model import (
    _temperature_scale,
    classification_metrics,
    replay_actions,
)


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def test_premium_age_is_causal_and_never_backfills_from_the_future() -> None:
    observed = np.asarray([True, False, False, True, False], dtype=bool)
    assert _grid_age(observed).tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_funding_window_is_strict_after_entry_and_inclusive_at_exit() -> None:
    entry = 1_000_000
    exit_time = entry + 15 * 60_000
    state = FundingState(
        event_time_ms=np.asarray(
            [entry, entry + 5 * 60_000, exit_time, exit_time + 1], dtype=np.int64
        ),
        event_rate=np.asarray([0.0001, 0.0002, 0.0003, 0.0004]),
        event_interval_hours=np.asarray([8, 8, 8, 8]),
        last_rate_bps=np.empty(0),
        last_interval_hours=np.empty(0),
        age_minutes=np.empty(0),
        settled_sum_24h_bps=np.empty(0),
        settled_sum_72h_bps=np.empty(0),
        settled_sum_168h_bps=np.empty(0),
        event_mean_30_bps=np.empty(0),
        event_zscore_30=np.empty(0),
    )
    result = _funding_in_holding_window(
        state,
        np.asarray([entry], dtype=np.int64),
        np.asarray([exit_time], dtype=np.int64),
    )
    assert result.tolist() == [5.0]


def test_temperature_scaling_and_classification_metrics_are_finite() -> None:
    probabilities = np.asarray(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]],
        dtype=np.float64,
    )
    scaled = _temperature_scale(probabilities, 1.5)
    metrics = classification_metrics(np.asarray([0, 1, 2]), scaled)

    assert np.allclose(np.sum(scaled, axis=1), 1.0)
    assert metrics.rows == 3
    assert metrics.class_counts == (1, 1, 1)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.multiclass_log_loss > 0.0
    assert metrics.nonfinite_probabilities == 0


def test_action_replay_uses_direction_specific_net_utility_and_nonoverlap() -> None:
    start = _ms("2024-10-01T00:00:00")
    decision_time = np.asarray(
        [
            start,
            start + 5 * 60_000,
            start,
            start + 5 * 60_000,
            start,
            start + 5 * 60_000,
        ],
        dtype=np.int64,
    )
    symbol_index = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    role = np.ones(6, dtype=bool)
    probabilities = np.asarray(
        [
            [0.05, 0.10, 0.85],
            [0.05, 0.10, 0.85],
            [0.80, 0.10, 0.10],
            [0.80, 0.10, 0.10],
            [0.05, 0.10, 0.85],
            [0.05, 0.10, 0.85],
        ],
        dtype=np.float32,
    )
    dataset = DerivativesHurdleDataset(
        feature_names=("x",),
        price_flow_feature_count=1,
        features=np.zeros((6, 1), dtype=np.float32),
        decision_time_ms=decision_time,
        symbol_index=symbol_index,
        target_class={15: np.asarray([2, 2, 0, 0, 2, 2], dtype=np.int8)},
        long_net_utility_bps={
            15: np.asarray([10, 20, -5, -5, 30, 40], dtype=np.float32)
        },
        short_net_utility_bps={
            15: np.asarray([-5, -5, 12, 14, -5, -5], dtype=np.float32)
        },
        funding_cash_flow_bps={
            15: np.asarray([1, 1, 2, 2, -1, -1], dtype=np.float32)
        },
        role_masks={15: {"calibration": role}},
        source_evidence=None,  # type: ignore[arg-type]
        source_exclusions={},
    )
    outcome = replay_actions(
        dataset,
        probabilities,
        horizon=15,
        role="calibration",
        maximum_action_probability=0.6,
        direction_probability_margin=0.2,
        bootstrap_samples=0,
        bootstrap_seed=1,
    )

    assert outcome.metrics.total_trades == 3
    assert outcome.metrics.trades_by_symbol == {
        "BTCUSDT": 1,
        "ETHUSDT": 1,
        "SOLUSDT": 1,
    }
    assert outcome.metrics.overlap_rejections == 3
    assert outcome.net_return_bps.tolist() == [10.0, 12.0, 30.0]
    assert outcome.metrics.total_funding_cash_flow_bps == 2.0
    assert outcome.metrics.maximum_single_symbol_fraction == 1.0 / 3.0

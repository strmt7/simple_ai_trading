from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from simple_ai_trading.causal_meta_label_model import (
    EVALUATION_MONTHS,
    MAXIMUM_ENTRIES_PER_SYMBOL_DAY,
    META_FEATURE_COUNT,
    binary_classification_metrics,
    build_meta_features,
    frozen_meta_schedules,
    replay_meta_actions,
)


class _FeatureDataset:
    def __init__(self) -> None:
        self.features = np.arange(5 * 71, dtype=np.float32).reshape(5, 71)
        self.symbol_index = np.asarray([0, 1, 2, 0, 1], dtype=np.int8)

    def feature_view(self, feature_set: str) -> np.ndarray:
        assert feature_set == "price_flow_only"
        return self.features


def test_round40_schedules_are_chronological_and_disjoint() -> None:
    schedules = frozen_meta_schedules()

    assert tuple(schedule.evaluation_month for schedule in schedules) == (
        EVALUATION_MONTHS
    )
    first = schedules[0]
    assert first.base_training_start == "2023-01-01"
    assert first.base_training_end == "2024-03-31"
    assert first.base_early_stop_start == "2024-04-01"
    assert first.base_early_stop_end == "2024-04-30"
    assert first.meta_fit_start == "2024-05-01"
    assert first.meta_fit_end == "2024-05-20"
    assert first.meta_early_stop_start == "2024-05-21"
    assert first.meta_early_stop_end == "2024-05-31"
    assert first.threshold_calibration_start == "2024-06-01"
    assert first.threshold_calibration_end == "2024-06-30"
    assert first.evaluation_start == "2024-07-01"
    assert first.evaluation_end == "2024-07-31"


def test_meta_features_have_frozen_shape_side_and_symbol_context() -> None:
    dataset = _FeatureDataset()
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    indices = np.asarray([0, 1, 2], dtype=np.int64)

    features = build_meta_features(dataset, probabilities, indices)  # type: ignore[arg-type]

    assert features.shape == (3, META_FEATURE_COUNT)
    np.testing.assert_allclose(features[:, 71:74], probabilities[:3])
    np.testing.assert_allclose(features[:, 77], [1.0, -1.0, -1.0])
    np.testing.assert_allclose(features[:, 78:], np.eye(3, dtype=np.float32))
    assert np.isfinite(features).all()


def test_binary_metrics_handle_ties_without_optimistic_auc() -> None:
    metrics = binary_classification_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray([0.1, 0.5, 0.5, 0.9], dtype=np.float64),
    )

    assert metrics["rows"] == 4
    assert metrics["positive_rows"] == 2
    assert metrics["positive_fraction"] == 0.5
    assert metrics["roc_auc"] == 0.875
    assert float(metrics["log_loss"]) > 0.0
    assert float(metrics["brier_score"]) > 0.0


def test_capacity_replay_is_timestamp_ordered_and_hard_capped() -> None:
    start_ms = int(
        datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000
    )
    per_symbol = 10
    rows = per_symbol * 3
    decision_time_ms = np.concatenate(
        [
            start_ms + np.arange(per_symbol, dtype=np.int64) * 31 * 60_000
            for _ in range(3)
        ]
    )
    symbol_index = np.repeat(np.arange(3, dtype=np.int8), per_symbol)
    dataset = SimpleNamespace(
        rows=rows,
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        long_net_utility_bps={30: np.ones(rows, dtype=np.float32)},
        short_net_utility_bps={30: -np.ones(rows, dtype=np.float32)},
        funding_cash_flow_bps={30: np.zeros(rows, dtype=np.float32)},
    )
    primary = np.tile(
        np.asarray([[0.1, 0.1, 0.8]], dtype=np.float32), (rows, 1)
    )
    meta = np.full(rows, 0.9, dtype=np.float32)

    replay = replay_meta_actions(
        dataset,  # type: ignore[arg-type]
        primary,
        meta,
        eligibility_mask=np.ones(rows, dtype=bool),
        period_start="2024-01-01",
        period_end="2024-01-01",
        meta_probability_threshold=0.8,
        primary_margin_threshold=0.1,
        bootstrap_samples=0,
        bootstrap_seed=1,
    )

    assert replay.threshold_candidate_rows == rows
    assert replay.overlap_rejections == 0
    assert replay.capacity_rejections == 6
    assert replay.maximum_entries_in_one_symbol_day == (
        MAXIMUM_ENTRIES_PER_SYMBOL_DAY
    )
    assert replay.outcome.metrics.total_trades == 24
    assert replay.outcome.metrics.trades_by_symbol == {
        "BTCUSDT": 8,
        "ETHUSDT": 8,
        "SOLUSDT": 8,
    }
    expected = np.concatenate(
        [np.arange(offset, offset + 8) for offset in (0, 10, 20)]
    )
    np.testing.assert_array_equal(
        np.sort(replay.outcome.selected_indices), expected
    )

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.ablate_action_value_scores import (
    _average_label_uniqueness,
    _causal_cusum_events,
    _select_score_threshold,
)


def test_causal_cusum_events_reset_after_each_utc_day() -> None:
    dataset = SimpleNamespace(
        feature_names=("return_5s_bps", "realized_volatility_60s_bps"),
        features=np.asarray(
            [[0.6, 0.0], [0.5, 0.0], [0.6, 0.0], [0.5, 0.0]],
            dtype=np.float64,
        ),
        decision_time_ms=np.asarray(
            [0, 5_000, 86_400_000, 86_405_000],
            dtype=np.int64,
        ),
        rows=4,
    )

    events = _causal_cusum_events(dataset, volatility_multiplier=1.0)

    assert events.tolist() == [False, True, False, True]


def test_average_label_uniqueness_downweights_overlapping_intervals() -> None:
    dataset = SimpleNamespace(
        decision_time_ms=np.asarray([0, 5_000, 10_000], dtype=np.int64),
        long_exit_time_ms=np.asarray([10_000, 10_000, 10_000], dtype=np.int64),
        short_exit_time_ms=np.asarray([10_000, 10_000, 10_000], dtype=np.int64),
        rows=3,
    )

    weights = _average_label_uniqueness(
        dataset,
        np.asarray([0, 1], dtype=np.int64),
        side="long",
    )

    assert np.mean(weights) == np.float32(1.0)
    assert weights[0] > weights[1] > 0.0


def test_score_policy_selects_only_profitable_nonoverlapping_tail() -> None:
    rows = 100
    timestamps = np.arange(rows, dtype=np.int64) * 10_000
    targets = np.where(np.arange(rows) >= 75, 10.0, -10.0)
    dataset = SimpleNamespace(
        decision_time_ms=timestamps,
        long_exit_time_ms=timestamps + 1_000,
        short_exit_time_ms=timestamps + 1_000,
        long_net_bps=targets,
        short_net_bps=np.full(rows, -10.0),
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.zeros(rows, dtype=bool),
    )
    indexes = np.arange(rows, dtype=np.int64)

    policy = _select_score_threshold(
        dataset=dataset,
        indexes=indexes,
        long_scores=np.arange(rows, dtype=np.float64),
        short_scores=np.zeros(rows, dtype=np.float64),
        risk_level="aggressive",
    )

    assert policy["accepted"] is True
    assert policy["best_observed_metrics"]["trades"] == 25
    assert policy["best_observed_metrics"]["total_net_bps"] == 250.0

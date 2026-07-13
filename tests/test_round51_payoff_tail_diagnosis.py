from __future__ import annotations

import numpy as np

from tools.diagnose_round51_payoff_tail import (
    _ranked_action,
    _raw_rank_metrics,
    _score_batch,
)


def test_ranked_action_selects_best_side_and_excludes_ties() -> None:
    endpoints = np.array([2, 4, 6, 8], dtype=np.int64)
    side, score, order = _ranked_action(
        np.array([1.0, 3.0, 2.0, -1.0]),
        np.array([2.0, 1.0, 2.0, 0.0]),
        endpoints,
    )

    assert np.array_equal(side, np.array([-1, 1, 0, -1], dtype=np.int8))
    assert np.array_equal(score, np.array([2.0, 3.0, 2.0, 0.0]))
    assert np.array_equal(order, np.array([1, 0, 3], dtype=np.int64))

    batch = _score_batch(endpoint_indexes=endpoints, side=side, selected=order[:2])
    assert np.array_equal(batch.side, np.array([-1, 1, 0, 0], dtype=np.int8))
    assert np.array_equal(batch.eligible, np.array([True, True, False, False]))
    assert np.array_equal(batch.strength_bps, np.array([1.0, 1.0, 0.0, 0.0]))


def test_raw_rank_metrics_use_chosen_side_and_exact_explicit_cost_bridge() -> None:
    metrics = _raw_rank_metrics(
        selected_positions=np.array([2, 0], dtype=np.int64),
        side=np.array([1, -1, -1], dtype=np.int8),
        base_long=np.array([5.0, -2.0, 99.0]),
        base_short=np.array([-4.0, 3.0, -1.0]),
        base_long_outcome=np.array([2, 1, 2], dtype=np.int8),
        base_short_outcome=np.array([1, 2, 0], dtype=np.int8),
        long_liquidity_eligible=np.array([True, False, True]),
        short_liquidity_eligible=np.array([True, True, False]),
        decision_time_ms=np.array([0, 10_000, 86_400_000], dtype=np.int64),
        explicit_round_trip_cost_bps=12.0,
    )

    assert metrics["rows"] == 2
    assert metrics["long_rows"] == 1
    assert metrics["short_rows"] == 1
    assert metrics["executable_rows"] == 1
    assert metrics["executable_ratio"] == 0.5
    assert metrics["executable_mean_net_bps"] == 5.0
    assert metrics["positive_rows"] == 1
    assert metrics["mean_net_bps"] == 2.0
    assert metrics["profit_factor"] == 5.0
    assert metrics["mean_after_adding_frozen_explicit_cost_bps"] == 14.0
    assert metrics["break_even_explicit_round_trip_cost_bps"] == 14.0
    assert metrics["outcomes"] == {
        "horizon": 1,
        "stop": 0,
        "take": 1,
        "ambiguous_stop": 0,
        "protection_gap_stop": 0,
    }
    assert [row["rows"] for row in metrics["daily"]] == [1, 1]

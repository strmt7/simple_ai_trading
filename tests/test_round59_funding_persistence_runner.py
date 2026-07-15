from __future__ import annotations

import numpy as np
import pytest

from tools.run_round59_funding_persistence_feasibility import (
    _episode_metrics,
    _episodes,
    _sign_transition,
    _stationary_bootstrap_means,
)


HOUR_MS = 60 * 60 * 1000


def _row(hour: int, rate: float) -> dict[str, object]:
    return {
        "calc_time": hour * HOUR_MS,
        "funding_interval_hours": 8,
        "funding_rate": rate,
    }


def test_episode_accounting_excludes_current_settlement_and_allows_boundary_reentry() -> (
    None
):
    rows = [
        _row(0, 0.0002),
        _row(8, 0.0001),
        _row(16, -0.00005),
        _row(24, 0.0002),
        _row(32, 0.0001),
        _row(40, 0.0001),
        _row(48, -0.0001),
    ]
    episodes = _episodes(
        rows,
        trigger={"operator": "greater_or_equal", "value": 2.0},
        horizon_hours=24,
    )

    assert [item["decision_time_ms"] for item in episodes] == [0, 24 * HOUR_MS]
    assert episodes[0]["future_settlements"] == 3
    assert episodes[0]["gross_future_funding_bps"] == pytest.approx(2.5)
    assert episodes[1]["future_settlements"] == 3
    assert episodes[1]["gross_future_funding_bps"] == pytest.approx(1.0)


def test_stationary_bootstrap_is_seed_deterministic() -> None:
    values = np.asarray([1.0, -2.0, 3.0, 4.0], dtype=np.float64)
    first = _stationary_bootstrap_means(
        values, samples=200, mean_block_length=2.0, seed=5901
    )
    second = _stationary_bootstrap_means(
        values, samples=200, mean_block_length=2.0, seed=5901
    )

    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_empty_cells_and_one_sided_sign_support_remain_reportable() -> None:
    metrics = _episode_metrics(
        [],
        costs={"stress": 32.0},
        uncertainty={
            "bootstrap_samples": 20,
            "mean_block_length_episodes": 2.0,
            "confidence_lower_quantile": 0.025,
            "confidence_upper_quantile": 0.975,
        },
        seed=5901,
    )
    transition = _sign_transition([_row(0, 0.0001), _row(8, 0.0002)])

    assert metrics["episodes"] == 0
    assert metrics["cost_comparisons"]["stress"]["mean_net_reference_bps"] is None
    assert transition["next_positive_given_positive"] == 1.0
    assert transition["next_positive_given_nonpositive"] is None

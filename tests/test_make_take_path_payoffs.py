from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.make_take_path_payoffs import build_action_path_payoffs


def _path() -> dict[str, np.ndarray]:
    times = np.arange(0, 305_000, 100, dtype=np.int64)
    bid = np.full(times.size, 100.0)
    ask = np.full(times.size, 100.1)
    return {
        "path_time_ms": times,
        "path_min_bid": bid.copy(),
        "path_max_bid": bid.copy(),
        "path_close_bid": bid.copy(),
        "path_min_ask": ask.copy(),
        "path_max_ask": ask.copy(),
        "path_close_ask": ask.copy(),
    }


def _evaluate(path: dict[str, np.ndarray], *, scenario: str = "base"):
    return build_action_path_payoffs(
        scenario=scenario,
        **path,
        entry_time_ms=[2_000, 2_000],
        action_side=[1, -1],
        entry_price=[100.0, 100.1],
        entry_cost_bps=[3.0, 3.0],
        exit_cost_bps=[6.0, 6.0],
        stop_bps=[80.0, 80.0],
        take_bps=[120.0, 120.0],
    )


def test_path_payoff_uses_exact_ratio_scaled_costs_and_horizon_quote() -> None:
    path = _path()
    final_bucket = (302_000 - 100) // 100
    path["path_min_bid"][final_bucket] = 100.5
    path["path_max_bid"][final_bucket] = 100.5
    path["path_close_bid"][final_bucket] = 100.5
    path["path_min_ask"][final_bucket] = 100.6
    path["path_max_ask"][final_bucket] = 100.6
    path["path_close_ask"][final_bucket] = 100.6

    result = _evaluate(path)

    assert result.valid.tolist() == [True, True]
    assert result.outcome.tolist() == [0, 0]
    assert result.exit_time_ms.tolist() == [302_000, 302_000]
    assert result.net_bps[0] == pytest.approx(50.0 - 3.0 - 6.0 * 1.005)
    short_ratio = 100.6 / 100.1
    assert result.net_bps[1] == pytest.approx(
        (1.0 - short_ratio) * 10_000.0 - 3.0 - 6.0 * short_ratio
    )
    assert len(result.source_path_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        result.net_bps[0] = 0.0


def test_same_bucket_collision_is_stop_first_and_stress_uses_adverse_extreme() -> None:
    path = _path()
    collision = 3_000 // 100
    path["path_min_bid"][collision] = 98.0
    path["path_max_bid"][collision] = 102.0
    path["path_close_bid"][collision] = 100.0

    base = _evaluate(path, scenario="base")
    stress = _evaluate(path, scenario="stress")

    assert base.outcome[0] == 3
    assert stress.outcome[0] == 4
    assert base.exit_time_ms[0] == 3_100
    assert stress.exit_time_ms[0] == 3_100
    assert stress.net_bps[0] < base.net_bps[0]


def test_path_payoff_fails_closed_when_protection_quote_is_stale() -> None:
    path = _path()
    keep = (path["path_time_ms"] < 1_000) | (path["path_time_ms"] >= 4_000)
    for key in tuple(path):
        path[key] = path[key][keep]

    result = _evaluate(path)

    assert result.valid.tolist() == [False, False]
    assert np.isnan(result.net_bps).all()


def test_path_payoff_fails_closed_when_required_markout_quote_is_stale() -> None:
    path = _path()
    keep = (path["path_time_ms"] < 5_500) | (path["path_time_ms"] >= 8_000)
    for key in tuple(path):
        path[key] = path[key][keep]

    result = _evaluate(path)

    assert result.valid.tolist() == [False, False]
    assert np.isnan(result.net_bps).all()
    assert np.isnan(result.markout_5s_bps).all()
    assert np.isnan(result.markout_15s_bps).all()


def test_short_take_uses_ask_path_and_markout_excludes_incomplete_bucket() -> None:
    path = _path()
    take_bucket = 3_000 // 100
    path["path_min_bid"][take_bucket] = 97.9
    path["path_max_bid"][take_bucket] = 100.0
    path["path_close_bid"][take_bucket] = 98.4
    path["path_min_ask"][take_bucket] = 98.0
    path["path_max_ask"][take_bucket] = 100.1
    path["path_close_ask"][take_bucket] = 98.5
    incomplete_at_markout = 7_000 // 100
    path["path_min_bid"][incomplete_at_markout] = 89.9
    path["path_max_bid"][incomplete_at_markout] = 100.0
    path["path_close_bid"][incomplete_at_markout] = 90.0
    path["path_min_ask"][incomplete_at_markout] = 90.1
    path["path_max_ask"][incomplete_at_markout] = 100.1
    path["path_close_ask"][incomplete_at_markout] = 90.1

    result = _evaluate(path)

    assert result.outcome[1] == 2
    assert result.net_bps[1] > 0.0
    assert result.markout_5s_bps[0] == pytest.approx(0.0)
    assert result.markout_5s_bps[1] == pytest.approx(0.0)

from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.make_take_scenario_entries import (
    build_make_take_scenario_entries,
)
from simple_ai_trading.make_take_targets import (
    MAKE_TAKE_UNFILLED_OUTCOME,
    build_make_take_targets,
)
from simple_ai_trading.queue_censored_actions import build_passive_fill_result


def _entries():
    decisions = np.asarray([1_000], dtype=np.int64)
    arrival = decisions + 750
    common = {
        "arrival_time_ms": arrival,
        "queue_ahead_quantity": [100.0],
        "order_notional_quote": 1_000.0,
        "trade_id": [1],
        "trade_time_ms": [2_750],
        "trade_price": [100.0],
        "trade_quantity": [120.0],
        "trade_buyer_is_maker": [True],
    }
    long_fill = build_passive_fill_result(
        placement_price=[100.0],
        buyer_is_maker=True,
        **common,
    )
    short_fill = build_passive_fill_result(
        placement_price=[100.1],
        buyer_is_maker=False,
        **common,
    )
    return build_make_take_scenario_entries(
        scenario="base",
        decision_time_ms=decisions,
        bid_price=[100.0],
        ask_price=[100.1],
        bid_quantity=[100.0],
        ask_quantity=[100.0],
        long_fill=long_fill,
        short_fill=short_fill,
    )


def _path(_day_start_ms: int):
    times = np.arange(0, 320_000, 100, dtype=np.int64)
    bid = np.full(times.size, 100.0)
    ask = np.full(times.size, 100.1)
    return {
        "path_time_ms": times,
        "path_min_bid": bid,
        "path_max_bid": bid,
        "path_close_bid": bid,
        "path_min_ask": ask,
        "path_max_ask": ask,
        "path_close_ask": ask,
    }


def test_targets_keep_unfilled_zero_separate_from_conditional_payoffs() -> None:
    targets = build_make_take_targets(
        symbol="BTCUSDT",
        source_dataset_sha256="a" * 64,
        entries=_entries(),
        event_stop_bps=[80.0],
        event_take_bps=[120.0],
        load_day_path=_path,
    )

    assert targets.filled.tolist() == [True, False, True, True]
    assert targets.conditional_payoff_valid.tolist() == [True, False, True, True]
    assert targets.realized_valid.tolist() == [True, True, True, True]
    assert np.isnan(targets.conditional_net_bps[1])
    assert targets.realized_net_bps[1] == 0.0
    assert targets.outcome[1] == MAKE_TAKE_UNFILLED_OUTCOME
    assert targets.terminal_time_ms[1] == 16_750
    assert targets.terminal_time_ms[0] == 302_750
    assert targets.terminal_time_ms[2] == 301_750
    assert len(targets.target_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        targets.realized_net_bps[0] = 0.0


def test_targets_reject_unbounded_or_inverted_barriers() -> None:
    with pytest.raises(ValueError, match="barrier contract"):
        build_make_take_targets(
            symbol="BTCUSDT",
            source_dataset_sha256="a" * 64,
            entries=_entries(),
            event_stop_bps=[81.0],
            event_take_bps=[80.0],
            load_day_path=_path,
        )


def test_targets_sort_overlapping_passive_fills_then_restore_action_order() -> None:
    decisions = np.asarray([1_000, 11_000], dtype=np.int64)
    arrivals = decisions + 750
    common = {
        "arrival_time_ms": arrivals,
        "queue_ahead_quantity": [100.0, 100.0],
        "order_notional_quote": 1_000.0,
        "trade_id": [1],
        "trade_time_ms": [13_750],
        "trade_price": [100.0],
        "trade_quantity": [120.0],
        "trade_buyer_is_maker": [True],
    }
    long_fill = build_passive_fill_result(
        placement_price=[100.0, 100.0],
        buyer_is_maker=True,
        **common,
    )
    short_fill = build_passive_fill_result(
        placement_price=[100.1, 100.1],
        buyer_is_maker=False,
        **common,
    )
    entries = build_make_take_scenario_entries(
        scenario="base",
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[100.1, 100.1],
        bid_quantity=[100.0, 100.0],
        ask_quantity=[100.0, 100.0],
        long_fill=long_fill,
        short_fill=short_fill,
    )

    targets = build_make_take_targets(
        symbol="BTCUSDT",
        source_dataset_sha256="b" * 64,
        entries=entries,
        event_stop_bps=[80.0, 80.0],
        event_take_bps=[120.0, 120.0],
        load_day_path=_path,
    )

    assert entries.entry_time_ms[0] > entries.entry_time_ms[6]
    assert targets.terminal_time_ms[0] == 313_750
    assert targets.terminal_time_ms[6] == 311_750
    assert np.count_nonzero(targets.conditional_payoff_valid) == 6

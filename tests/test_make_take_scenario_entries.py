from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.make_take_scenario_entries import (
    build_make_take_scenario_entries,
)
from simple_ai_trading.queue_censored_actions import build_passive_fill_result


def _fills(*, latency_ms: int):
    decisions = np.asarray([10_000, 30_000], dtype=np.int64)
    arrivals = decisions + latency_ms
    trades = {
        "trade_id": [1, 2],
        "trade_time_ms": [arrivals[0] + 1_000, arrivals[1] + 1_000],
        "trade_price": [100.0, 101.0],
        "trade_quantity": [20.0, 20.0],
    }
    long_fill = build_passive_fill_result(
        arrival_time_ms=arrivals,
        placement_price=[100.0, 100.0],
        queue_ahead_quantity=[5.0, 5.0],
        buyer_is_maker=True,
        order_notional_quote=1_000.0,
        trade_buyer_is_maker=[True, False],
        **trades,
    )
    short_fill = build_passive_fill_result(
        arrival_time_ms=arrivals,
        placement_price=[101.0, 101.0],
        queue_ahead_quantity=[5.0, 5.0],
        buyer_is_maker=False,
        order_notional_quote=1_000.0,
        trade_buyer_is_maker=[True, False],
        **trades,
    )
    return decisions, long_fill, short_fill


def _entries(scenario: str):
    latency = 750 if scenario == "base" else 1_500
    decisions, long_fill, short_fill = _fills(latency_ms=latency)
    return build_make_take_scenario_entries(
        scenario=scenario,
        decision_time_ms=decisions,
        bid_price=[100.0, 100.0],
        ask_price=[101.0, 101.0],
        bid_quantity=[5.0, 5.0],
        ask_quantity=[5.0, 5.0],
        long_fill=long_fill,
        short_fill=short_fill,
    )


def test_entry_panel_preserves_unfilled_passive_orders_and_aggressive_entries() -> None:
    batch = _entries("base")

    assert batch.action_code.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert batch.filled.tolist() == [True, False, True, True, False, True, True, True]
    assert batch.fill_bucket.tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert batch.entry_time_ms[1] == -1
    assert batch.unfilled_expiry_time_ms[1] == 25_750
    assert batch.entry_time_ms[2] == 10_750
    assert batch.entry_price[:4].tolist() == [100.0, 101.0, 101.0, 100.0]
    assert not np.any(batch.eligible)
    assert batch.entry_cost_bps[:4].tolist() == [3.0, 3.0, 6.0, 6.0]
    assert np.all(batch.exit_cost_bps == 6.0)


def test_stress_entry_panel_raises_maker_cost_and_slippage_without_relabeling() -> None:
    batch = _entries("stress")

    assert batch.placement_latency_ms == 1_500
    assert batch.additional_slippage_bps_per_side == 3.0
    assert np.all(batch.entry_cost_bps == 8.0)
    assert np.all(batch.exit_cost_bps == 8.0)
    assert len(batch.batch_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        batch.filled[0] = False


def test_entry_panel_rejects_passive_fill_price_drift() -> None:
    decisions, long_fill, short_fill = _fills(latency_ms=750)
    with pytest.raises(ValueError, match="source contract"):
        build_make_take_scenario_entries(
            scenario="base",
            decision_time_ms=decisions,
            bid_price=[99.0, 99.0],
            ask_price=[101.0, 101.0],
            bid_quantity=[5.0, 5.0],
            ask_quantity=[5.0, 5.0],
            long_fill=long_fill,
            short_fill=short_fill,
        )

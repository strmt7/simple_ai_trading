from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.barrier_payoff_data import (
    STOP_EVENT,
    TAKE_PROFIT_EVENT,
    TIMEOUT_EVENT,
    BarrierSpecification,
    _simulate_side,
)
from simple_ai_trading.cross_asset_cost_data import MinuteSeries
from simple_ai_trading.derivatives_hurdle_data import FundingState


def _series(
    *,
    opens: tuple[float, ...],
    highs: tuple[float, ...],
    lows: tuple[float, ...],
) -> MinuteSeries:
    size = len(opens)
    zeros = np.zeros(size, dtype=np.float64)
    return MinuteSeries(
        symbol="BTCUSDT",
        open_time_ms=np.arange(size, dtype=np.int64) * 60_000,
        open=np.asarray(opens, dtype=np.float64),
        high=np.asarray(highs, dtype=np.float64),
        low=np.asarray(lows, dtype=np.float64),
        close=np.asarray(opens, dtype=np.float64),
        volume=zeros,
        quote_volume=zeros,
        trade_count=zeros,
        taker_buy_base_volume=zeros,
        taker_buy_quote_volume=zeros,
    )


def _funding(*, event_time_ms: int = 10_000_000, event_rate: float = 0.0) -> FundingState:
    zeros = np.zeros(5, dtype=np.float64)
    return FundingState(
        event_time_ms=np.asarray([event_time_ms], dtype=np.int64),
        event_rate=np.asarray([event_rate], dtype=np.float64),
        event_interval_hours=np.asarray([8.0], dtype=np.float64),
        last_rate_bps=zeros,
        last_interval_hours=zeros,
        age_minutes=zeros,
        settled_sum_24h_bps=zeros,
        settled_sum_72h_bps=zeros,
        settled_sum_168h_bps=zeros,
        event_mean_30_bps=zeros,
        event_zscore_30=zeros,
    )


def _specification() -> BarrierSpecification:
    return BarrierSpecification(
        horizon_minutes=2,
        stop_volatility_multiple=1.0,
        take_profit_to_stop_ratio=2.0,
        minimum_stop_bps=24.0,
        maximum_stop_bps=80.0,
        round_trip_execution_charge_bps=12.0,
    )


def test_same_minute_stop_and_take_is_conservatively_filled_at_stop() -> None:
    outputs = _simulate_side(
        _series(
            opens=(100.0, 100.0, 100.0, 100.0, 100.0),
            highs=(100.0, 102.0, 100.0, 100.0, 100.0),
            lows=(100.0, 99.0, 100.0, 100.0, 100.0),
        ),
        _funding(),
        np.asarray([0], dtype=np.int64),
        np.asarray([80.0]),
        np.asarray([160.0]),
        side=1,
        specification=_specification(),
    )

    assert outputs[0].tolist() == [STOP_EVENT]
    assert outputs[1].tolist() == [1]
    assert outputs[2][0] == pytest.approx(-80.0)
    assert outputs[4][0] == pytest.approx(-92.0)
    assert outputs[5][0] == pytest.approx(0.0)
    assert outputs[6].tolist() == [True]


def test_gap_through_long_stop_uses_worse_open_fill() -> None:
    outputs = _simulate_side(
        _series(
            opens=(100.0, 100.0, 98.5, 100.0, 100.0),
            highs=(100.0, 100.2, 99.0, 100.0, 100.0),
            lows=(100.0, 99.8, 98.0, 100.0, 100.0),
        ),
        _funding(),
        np.asarray([0], dtype=np.int64),
        np.asarray([80.0]),
        np.asarray([160.0]),
        side=1,
        specification=_specification(),
    )

    assert outputs[0].tolist() == [STOP_EVENT]
    assert outputs[1].tolist() == [2]
    assert outputs[2][0] == pytest.approx(-150.0)
    assert outputs[4][0] == pytest.approx(-162.0)
    assert outputs[5][0] == pytest.approx(70.0)


def test_timeout_and_short_take_apply_funding_with_correct_sign() -> None:
    timeout = _simulate_side(
        _series(
            opens=(100.0, 100.0, 100.0, 101.0, 101.0),
            highs=(100.0, 100.2, 100.5, 101.0, 101.0),
            lows=(100.0, 99.8, 99.7, 101.0, 101.0),
        ),
        _funding(event_time_ms=180_000, event_rate=0.001),
        np.asarray([0], dtype=np.int64),
        np.asarray([80.0]),
        np.asarray([160.0]),
        side=1,
        specification=_specification(),
    )
    short_take = _simulate_side(
        _series(
            opens=(100.0, 100.0, 100.0, 100.0, 100.0),
            highs=(100.0, 100.2, 100.0, 100.0, 100.0),
            lows=(100.0, 98.0, 100.0, 100.0, 100.0),
        ),
        _funding(event_time_ms=120_000, event_rate=0.001),
        np.asarray([0], dtype=np.int64),
        np.asarray([80.0]),
        np.asarray([160.0]),
        side=-1,
        specification=_specification(),
    )

    assert timeout[0].tolist() == [TIMEOUT_EVENT]
    assert timeout[1].tolist() == [2]
    assert timeout[2][0] == pytest.approx(100.0)
    assert timeout[3][0] == pytest.approx(10.0)
    assert timeout[4][0] == pytest.approx(78.0)
    assert short_take[0].tolist() == [TAKE_PROFIT_EVENT]
    assert short_take[1].tolist() == [1]
    assert short_take[2][0] == pytest.approx(160.0)
    assert short_take[3][0] == pytest.approx(10.0)
    assert short_take[4][0] == pytest.approx(158.0)

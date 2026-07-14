from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.cross_asset_cost_data import (
    MINUTE_MS,
    MinuteSeries,
    SYMBOLS,
    load_verified_minute_panel_window,
)
from simple_ai_trading.derivatives_hurdle_data import FundingState
from simple_ai_trading.stop_time_payoff_data import (
    STOP_EVENT,
    TIMEOUT_EVENT,
    StopTimeSpecification,
    build_stop_time_payoff_dataset,
)


def _series(symbol: str, opens: list[float], highs: list[float], lows: list[float]) -> MinuteSeries:
    rows = len(opens)
    open_values = np.asarray(opens, dtype=np.float64)
    return MinuteSeries(
        symbol=symbol,
        open_time_ms=np.arange(rows, dtype=np.int64) * MINUTE_MS,
        open=open_values,
        high=np.asarray(highs, dtype=np.float64),
        low=np.asarray(lows, dtype=np.float64),
        close=open_values.copy(),
        volume=np.ones(rows),
        quote_volume=np.ones(rows),
        trade_count=np.ones(rows, dtype=np.int64),
        taker_buy_base_volume=np.full(rows, 0.5),
        taker_buy_quote_volume=np.full(rows, 0.5),
    )


def _funding(rate: float = 0.001) -> FundingState:
    event_time = np.asarray([2 * MINUTE_MS], dtype=np.int64)
    event_rate = np.asarray([rate], dtype=np.float64)
    zeros = np.zeros(8, dtype=np.float64)
    return FundingState(
        event_time_ms=event_time,
        event_rate=event_rate,
        event_interval_hours=np.asarray([8], dtype=np.int16),
        last_rate_bps=zeros.copy(),
        last_interval_hours=zeros.copy(),
        age_minutes=zeros.copy(),
        settled_sum_24h_bps=zeros.copy(),
        settled_sum_72h_bps=zeros.copy(),
        settled_sum_168h_bps=zeros.copy(),
        event_mean_30_bps=np.asarray([np.nan]),
        event_zscore_30=np.asarray([np.nan]),
    )


def _specification() -> StopTimeSpecification:
    return StopTimeSpecification(
        horizon_minutes=3,
        stop_volatility_multiple=1.0,
        minimum_stop_bps=100.0,
        maximum_stop_bps=100.0,
        round_trip_execution_charge_bps=10.0,
    )


def test_stop_time_payoff_uses_gap_fills_funding_signs_and_timeout() -> None:
    series = _series(
        "BTCUSDT",
        [100.0, 100.0, 98.0, 100.0, 101.0],
        [100.0, 100.4, 98.5, 100.5, 101.0],
        [100.0, 99.5, 97.5, 99.5, 101.0],
    )
    panel = {symbol: series for symbol in SYMBOLS}
    funding = {symbol: _funding() for symbol in SYMBOLS}
    dataset = build_stop_time_payoff_dataset(
        panel,
        funding,
        np.asarray([0], dtype=np.int64),
        np.ones((1, len(SYMBOLS))),
        source_dataset_sha256="a" * 64,
        specification=_specification(),
    )

    assert np.all(dataset.long_event_code == STOP_EVENT)
    assert np.all(dataset.long_event_minute == 2)
    assert dataset.long_price_return_bps[0, 0] == pytest.approx(-200.0)
    assert dataset.long_gap_through_slippage_bps[0, 0] == pytest.approx(100.0)
    assert dataset.long_funding_cash_flow_bps[0, 0] == pytest.approx(-10.0)
    assert dataset.long_net_payoff_bps[0, 0] == pytest.approx(-220.0)

    assert np.all(dataset.short_event_code == TIMEOUT_EVENT)
    assert np.all(dataset.short_event_minute == 3)
    assert dataset.short_price_return_bps[0, 0] == pytest.approx(-100.0)
    assert dataset.short_funding_cash_flow_bps[0, 0] == pytest.approx(10.0)
    assert dataset.short_net_payoff_bps[0, 0] == pytest.approx(-100.0)


def test_stop_time_payoff_rejects_nonmatching_decision_grid() -> None:
    series = _series(
        "BTCUSDT",
        [100.0] * 6,
        [101.0] * 6,
        [99.0] * 6,
    )
    with pytest.raises(ValueError, match="decision timestamps differ"):
        build_stop_time_payoff_dataset(
            {symbol: series for symbol in SYMBOLS},
            {symbol: _funding(0.0) for symbol in SYMBOLS},
            np.asarray([MINUTE_MS // 2], dtype=np.int64),
            np.ones((1, len(SYMBOLS))),
            source_dataset_sha256="b" * 64,
            specification=_specification(),
        )


def test_verified_panel_window_rejects_rows_beyond_frozen_source() -> None:
    with pytest.raises(ValueError, match="outside the certified source window"):
        load_verified_minute_panel_window(
            "unused.sqlite",
            materialization_end="2025-07-01",
        )

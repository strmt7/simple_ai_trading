from __future__ import annotations

import numpy as np

from simple_ai_trading.derivatives_hurdle_data import FundingState
from simple_ai_trading.second_flow_data import START_MS, SecondFlowSeries
from simple_ai_trading.second_flow_execution_model import (
    DELAYS_SECONDS,
    TimingDataset,
    _SeriesCache,
    _capacity_selection,
    _funding_bps,
    _proposal_features,
)


def _series(symbol: str, *, mutate_from: int | None = None) -> SecondFlowSeries:
    rows = 4_000
    index = np.arange(rows, dtype=np.float64)
    close = 100.0 * np.exp(index * 1e-7)
    open_price = close.copy()
    high = close * 1.00001
    low = close * 0.99999
    volume = 2.0 + np.sin(index / 17.0) ** 2
    quote = volume * close
    taker_base = volume * (0.45 + 0.05 * np.sin(index / 11.0))
    taker_quote = taker_base * close
    if mutate_from is not None:
        close[mutate_from:] *= 1.5
        open_price[mutate_from:] *= 1.5
        high[mutate_from:] *= 1.5
        low[mutate_from:] *= 1.5
        volume[mutate_from:] *= 10.0
        quote[mutate_from:] *= 15.0
        taker_base[mutate_from:] *= 10.0
        taker_quote[mutate_from:] *= 15.0
    return SecondFlowSeries(
        symbol=symbol,
        open_time_ms=START_MS + np.arange(rows, dtype=np.int64) * 1_000,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        quote_volume=quote,
        trade_count=np.full(rows, 3, dtype=np.int64),
        taker_buy_base_volume=taker_base,
        taker_buy_quote_volume=taker_quote,
        source=("test",),
    )


def _timing_dataset(entry_times: np.ndarray, symbols: np.ndarray) -> TimingDataset:
    proposals = int(entry_times.size)
    option_rows = proposals * len(DELAYS_SECONDS)
    probabilities = np.tile(
        np.asarray([[0.2, 0.2, 0.6]], dtype=np.float32), (proposals, 1)
    )
    return TimingDataset(
        feature_names=("test_feature",),
        features=np.zeros((option_rows, 1), dtype=np.float32),
        proposal_source_index=np.arange(proposals, dtype=np.int64),
        proposal_decision_time_ms=entry_times - 60_000,
        proposal_entry_time_ms=entry_times,
        proposal_symbol_index=symbols.astype(np.int8),
        proposal_side=np.ones(proposals, dtype=np.int8),
        proposal_primary_probabilities=probabilities,
        proposal_margin=np.full(proposals, 0.4, dtype=np.float32),
        proposal_day=entry_times // 86_400_000,
        proposal_weight=np.ones(proposals, dtype=np.float32),
        option_proposal_index=np.repeat(
            np.arange(proposals, dtype=np.int64), len(DELAYS_SECONDS)
        ),
        option_delay_seconds=np.tile(
            np.asarray(DELAYS_SECONDS, dtype=np.int16), proposals
        ),
        option_base_net_bps=np.zeros(option_rows, dtype=np.float32),
        option_stress_net_bps=np.full(option_rows, -4.0, dtype=np.float32),
        option_funding_bps=np.zeros(option_rows, dtype=np.float32),
        primary_artifacts=(),
        proposal_exclusions={},
    )


def test_second_flow_features_exclude_the_entry_second_and_future() -> None:
    entry_index = 2_000
    original = {
        symbol: _SeriesCache(_series(symbol))
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    }
    future_mutated = {
        symbol: _SeriesCache(_series(symbol, mutate_from=entry_index))
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    }
    arguments = {
        "entry_time_ms": START_MS + entry_index * 1_000,
        "symbol_index": 0,
        "side": 1,
        "probabilities": np.asarray([0.2, 0.2, 0.6], dtype=np.float32),
        "margin": 0.4,
    }

    before = _proposal_features(caches=original, **arguments)
    after = _proposal_features(caches=future_mutated, **arguments)

    np.testing.assert_array_equal(before, after)


def test_funding_cash_flow_excludes_entry_and_exit_boundaries() -> None:
    empty = np.empty(0, dtype=np.float64)
    funding = FundingState(
        event_time_ms=np.asarray([1_000, 2_000, 3_000, 4_000], dtype=np.int64),
        event_rate=np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64),
        event_interval_hours=empty,
        last_rate_bps=empty,
        last_interval_hours=empty,
        age_minutes=empty,
        settled_sum_24h_bps=empty,
        settled_sum_72h_bps=empty,
        settled_sum_168h_bps=empty,
        event_mean_30_bps=empty,
        event_zscore_30=empty,
    )

    assert _funding_bps(funding, entry_time_ms=1_000, exit_time_ms=4_000) == 500.0


def test_capacity_output_is_ordered_by_actual_delayed_entry() -> None:
    dataset = _timing_dataset(
        np.asarray([START_MS + 10_000, START_MS + 30_000], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int8),
    )
    proposed = np.asarray([3, 4], dtype=np.int64)

    proposals, options, overlap, capacity = _capacity_selection(
        dataset,
        role_mask=np.asarray([True, True]),
        proposed_option=proposed,
    )

    assert proposals.tolist() == [1, 0]
    assert options.tolist() == [4, 3]
    assert overlap == 0
    assert capacity == 0

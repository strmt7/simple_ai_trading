from __future__ import annotations

import duckdb
import numpy as np

from simple_ai_trading.microstructure_features import (
    MicrostructureDataset,
    _completed_path_index_bounds,
    apply_path_aware_lifecycle_targets,
)


def test_path_bounds_exclude_partial_entry_and_exit_buckets() -> None:
    path_times = np.asarray([1_000, 2_000, 3_000, 4_000], dtype=np.int64)
    entry_arrivals = np.asarray([1_750], dtype=np.int64)
    exit_arrivals = np.asarray([3_750], dtype=np.int64)

    starts, ends = _completed_path_index_bounds(
        path_times,
        entry_arrivals,
        exit_arrivals,
        resolution_ms=1_000,
    )

    assert starts.tolist() == [1]
    assert ends.tolist() == [2]
    assert path_times[starts[0] : ends[0]].tolist() == [2_000]


def test_path_targets_can_cross_a_utc_archive_boundary() -> None:
    boundary = 1_704_153_600_000
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE current_book_ticker_path_1s (
            symbol VARCHAR, second_ms BIGINT, min_bid DOUBLE, max_bid DOUBLE,
            close_bid DOUBLE, min_ask DOUBLE, max_ask DOUBLE, close_ask DOUBLE
        )
        """
    )
    connection.executemany(
        "INSERT INTO current_book_ticker_path_1s VALUES ('BTCUSDT', ?, ?, ?, ?, ?, ?, ?)",
        [
            (boundary - 1_000, 99.9, 100.0, 100.0, 100.1, 100.2, 100.1),
            (boundary, 100.0, 100.1, 100.0, 100.1, 100.2, 100.1),
            (boundary + 1_000, 100.0, 100.1, 100.0, 100.1, 100.2, 100.1),
        ],
    )

    class _Warehouse:
        def connect(self):
            return connection

    ones = np.ones(1, dtype=np.float64)
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version="test",
        feature_names=("x",),
        horizon_seconds=4,
        total_latency_ms=0,
        taker_fee_bps=5.0,
        reference_order_notional_quote=1.0,
        max_l1_participation=0.10,
        decision_cadence_seconds=1,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=np.asarray([boundary - 2_000], dtype=np.int64),
        long_exit_time_ms=np.asarray([boundary + 2_000], dtype=np.int64),
        short_exit_time_ms=np.asarray([boundary + 2_000], dtype=np.int64),
        features=np.zeros((1, 1), dtype=np.float32),
        long_net_bps=ones,
        short_net_bps=ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(1, dtype=np.int64),
        exit_quote_age_ms=np.zeros(1, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=100.1 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=100.1 * ones,
        entry_bid_qty=10.0 * ones,
        entry_ask_qty=10.0 * ones,
        fixed_exit_bid_qty=10.0 * ones,
        fixed_exit_ask_qty=10.0 * ones,
        long_l1_participation=0.001 * ones,
        short_l1_participation=0.001 * ones,
        long_liquidity_eligible=np.ones(1, dtype=bool),
        short_liquidity_eligible=np.ones(1, dtype=bool),
    )
    try:
        output, evidence = apply_path_aware_lifecycle_targets(
            _Warehouse(),  # type: ignore[arg-type]
            dataset,
            stop_loss_bps=500.0,
            take_profit_bps=500.0,
            trigger_execution_slippage_bps=1.0,
        )
    finally:
        connection.close()

    assert output.target_mode == "exchange_trigger_market_exit_1s_adverse_first"
    assert evidence.rows == 1
    assert evidence.long_horizon_count == 1

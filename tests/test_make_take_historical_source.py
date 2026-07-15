from __future__ import annotations

from types import SimpleNamespace

import duckdb
import numpy as np

from simple_ai_trading.make_take_historical_source import (
    load_historical_day_path,
    load_historical_placement_quotes,
    load_historical_trade_chunk,
    select_role_decision_indexes,
    utc_day_chunks,
)


DAY_MS = 86_400_000


def test_role_selection_keeps_complete_warm_and_same_day_lifecycles() -> None:
    start = 10 * DAY_MS
    end = start + 2 * DAY_MS
    warmup = 3_601_000
    lifecycle = 316_500
    decisions = np.asarray(
        [
            start + warmup - 1,
            start + warmup,
            start + DAY_MS - lifecycle,
            start + DAY_MS - lifecycle + 1,
            start + DAY_MS + warmup,
            end - lifecycle,
            end - lifecycle + 1,
        ],
        dtype=np.int64,
    )
    dataset = SimpleNamespace(decision_time_ms=decisions)

    indexes = select_role_decision_indexes(
        dataset,
        role_start_ms=start,
        role_end_ms_exclusive=end,
        feature_warmup_ms=warmup,
        maximum_lifecycle_ms=lifecycle,
    )

    np.testing.assert_array_equal(indexes, np.asarray([1, 2, 4, 5]))
    assert utc_day_chunks(start, end) == (
        (start, start + DAY_MS),
        (start + DAY_MS, end),
    )


def test_duckdb_source_reads_are_causal_stale_aware_and_exact() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE current_book_ticker_100ms (
            symbol VARCHAR,
            bucket_ms BIGINT,
            available_time_ms BIGINT,
            min_bid DOUBLE,
            max_bid DOUBLE,
            close_bid DOUBLE,
            close_bid_qty DOUBLE,
            min_ask DOUBLE,
            max_ask DOUBLE,
            close_ask DOUBLE,
            close_ask_qty DOUBLE,
            last_transaction_time_ms BIGINT
        )
        """
    )
    connection.executemany(
        "INSERT INTO current_book_ticker_100ms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("BTCUSDT", 1_000, 1_100, 99.0, 101.0, 100.0, 20.0, 101.0, 103.0, 102.0, 21.0, 1_050),
            ("BTCUSDT", 3_400, 3_500, 100.0, 102.0, 101.0, 22.0, 102.0, 104.0, 103.0, 23.0, 3_450),
        ],
    )
    connection.execute(
        """
        CREATE TABLE current_trade_raw (
            symbol VARCHAR,
            trade_id BIGINT,
            trade_time_ms BIGINT,
            price DOUBLE,
            qty DOUBLE,
            buyer_is_maker BOOLEAN
        )
        """
    )
    connection.executemany(
        "INSERT INTO current_trade_raw VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("BTCUSDT", 7, 1_150, 101.0, 0.5, True),
            ("BTCUSDT", 8, 1_250, 102.0, 0.7, False),
        ],
    )

    quotes = load_historical_placement_quotes(
        connection,
        symbol="BTCUSDT",
        decision_time_ms=np.asarray([450, 2_250, 2_850], dtype=np.int64),
        placement_latency_ms=750,
        max_quote_age_ms=1_000,
    )

    np.testing.assert_array_equal(quotes.valid, np.asarray([True, False, True]))
    assert quotes.bid_price[0] == 100.0
    assert quotes.bid_price[2] == 101.0
    assert quotes.quote_available_time_ms[0] <= quotes.arrival_time_ms[0]
    selected = quotes.select_rows(np.asarray([0, 2], dtype=np.int64))
    assert np.all(selected.valid)

    trades = load_historical_trade_chunk(
        connection,
        symbol="BTCUSDT",
        start_ms=1_100,
        end_ms_exclusive=1_300,
    )
    np.testing.assert_array_equal(trades["trade_id"], np.asarray([7, 8]))
    np.testing.assert_array_equal(
        trades["trade_buyer_is_maker"], np.asarray([True, False])
    )

    path = load_historical_day_path(
        connection,
        symbol="BTCUSDT",
        day_start_ms=0,
    )
    np.testing.assert_array_equal(path["path_time_ms"], np.asarray([1_000, 3_400]))
    connection.close()

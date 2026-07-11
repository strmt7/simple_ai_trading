from __future__ import annotations

from dataclasses import dataclass
import math

import duckdb
import numpy as np
import pytest

from simple_ai_trading.microstructure_features import (
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_runtime import (
    MicrostructureSecond,
    StreamingMicrostructureFeatureEngine,
)


def _second(index: int, *, second_ms: int | None = None) -> MicrostructureSecond:
    timestamp = index * 1_000 if second_ms is None else second_ms
    mid = 100.0 + index * 0.001 + math.sin(index / 17.0) * 0.02
    prior_mid = (
        100.0 + max(0, index - 1) * 0.001 + math.sin(max(0, index - 1) / 17.0) * 0.02
    )
    open_mid = prior_mid
    high_mid = max(open_mid, mid) + 0.006 + (index % 3) * 0.0002
    low_mid = min(open_mid, mid) - 0.005 - (index % 5) * 0.0001
    half_spread = 0.006 + (index % 7) * 0.0001
    bid = mid - half_spread
    ask = mid + half_spread
    bid_qty = 1.3 + (index % 11) * 0.03
    ask_qty = 1.1 + (index % 13) * 0.025
    close_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    spread_bps = (ask - bid) * 10_000.0 / mid
    microprice = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
    microprice_offset = (microprice / mid - 1.0) * 10_000.0
    base_volume = 0.8 + (index % 19) * 0.04
    buy_volume = base_volume * (0.5 + 0.18 * math.sin(index / 23.0))
    sell_volume = base_volume - buy_volume
    trade_imbalance = (buy_volume - sell_volume) / base_volume
    return MicrostructureSecond(
        symbol="BTCUSDT",
        second_ms=timestamp,
        open_mid=open_mid,
        high_mid=high_mid,
        low_mid=low_mid,
        close_mid=mid,
        close_bid=bid,
        close_ask=ask,
        close_bid_qty=bid_qty,
        close_ask_qty=ask_qty,
        spread_bps=spread_bps,
        max_spread_bps=spread_bps * 1.15,
        l1_imbalance=close_imbalance * 0.92,
        close_l1_imbalance=close_imbalance,
        microprice_offset_bps=microprice_offset,
        quote_updates=40 + index % 29,
        event_delay_p50_ms=6.0 + index % 4,
        event_delay_p99_ms=18.0 + index % 9,
        trade_close=mid * (1.0 + math.sin(index / 13.0) * 1e-6),
        base_volume=base_volume,
        quote_volume=base_volume * mid,
        aggressive_buy_volume=buy_volume,
        aggressive_sell_volume=sell_volume,
        trade_imbalance=trade_imbalance,
        trade_count=15 + index % 17,
    )


@dataclass
class _ConnectionWarehouse:
    connection: duckdb.DuckDBPyConnection

    def connect(self) -> duckdb.DuckDBPyConnection:
        return self.connection


def _warehouse_for(rows: list[MicrostructureSecond]) -> _ConnectionWarehouse:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE current_book_ticker_1s (
            symbol VARCHAR, second_ms BIGINT, open_mid DOUBLE, high_mid DOUBLE,
            low_mid DOUBLE, close_mid DOUBLE, close_bid DOUBLE, close_ask DOUBLE,
            close_bid_qty DOUBLE, close_ask_qty DOUBLE,
            event_weighted_spread_bps DOUBLE, max_spread_bps DOUBLE,
            event_weighted_l1_imbalance DOUBLE, close_l1_imbalance DOUBLE,
            event_weighted_microprice_offset_bps DOUBLE, quote_updates BIGINT,
            event_delay_p50_ms DOUBLE, event_delay_p99_ms DOUBLE
        );
        CREATE TABLE current_trade_1s (
            symbol VARCHAR, second_ms BIGINT, close DOUBLE, base_volume DOUBLE,
            quote_volume DOUBLE, aggressive_buy_volume DOUBLE,
            aggressive_sell_volume DOUBLE, trade_imbalance DOUBLE, trade_count BIGINT
        );
        CREATE TABLE current_book_ticker_100ms (
            symbol VARCHAR, available_time_ms BIGINT, last_transaction_time_ms BIGINT,
            close_bid DOUBLE, close_ask DOUBLE, close_bid_qty DOUBLE, close_ask_qty DOUBLE
        );
        """
    )
    quote_rows = []
    book_rows = []
    trade_rows = []
    for row in rows:
        book_rows.append(
            (
                row.symbol,
                row.second_ms,
                row.open_mid,
                row.high_mid,
                row.low_mid,
                row.close_mid,
                row.close_bid,
                row.close_ask,
                row.close_bid_qty,
                row.close_ask_qty,
                row.spread_bps,
                row.max_spread_bps,
                row.l1_imbalance,
                row.close_l1_imbalance,
                row.microprice_offset_bps,
                row.quote_updates,
                row.event_delay_p50_ms,
                row.event_delay_p99_ms,
            )
        )
        trade_rows.append(
            (
                row.symbol,
                row.second_ms,
                row.trade_close,
                row.base_volume,
                row.quote_volume,
                row.aggressive_buy_volume,
                row.aggressive_sell_volume,
                row.trade_imbalance,
                row.trade_count,
            )
        )
        available = row.second_ms + 1_000
        quote_rows.append(
            (
                row.symbol,
                available,
                available,
                row.close_bid,
                row.close_ask,
                row.close_bid_qty,
                row.close_ask_qty,
            )
        )
    connection.executemany(
        "INSERT INTO current_book_ticker_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        book_rows,
    )
    connection.executemany(
        "INSERT INTO current_trade_1s VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        trade_rows,
    )
    connection.executemany(
        "INSERT INTO current_book_ticker_100ms VALUES (?, ?, ?, ?, ?, ?, ?)",
        quote_rows,
    )
    return _ConnectionWarehouse(connection)


def test_streaming_features_match_offline_duckdb_contract() -> None:
    seconds = [_second(index) for index in range(4_150)]
    warehouse = _warehouse_for(seconds)
    try:
        offline = build_executable_microstructure_dataset(
            warehouse,  # type: ignore[arg-type]
            symbol="BTCUSDT",
            horizon_seconds=1,
            total_latency_ms=0,
            taker_fee_bps=5.0,
            reference_order_notional_quote=1.0,
            max_l1_participation=0.50,
            decision_cadence_seconds=1,
        )
        engine = StreamingMicrostructureFeatureEngine(
            "BTCUSDT", decision_cadence_seconds=1
        )
        online = {}
        for second in seconds:
            emitted = engine.append(second)
            if emitted is not None:
                online[emitted.decision_time_ms] = emitted

        matching = [
            (index, online[int(timestamp)])
            for index, timestamp in enumerate(offline.decision_time_ms)
            if int(timestamp) in online
        ]
        assert len(matching) >= 100
        for index, emitted in (matching[0], matching[len(matching) // 2], matching[-1]):
            assert emitted.feature_names == offline.feature_names
            np.testing.assert_allclose(
                emitted.features,
                offline.features[index],
                rtol=3e-5,
                atol=3e-5,
            )
        latest = matching[-1][1]
        source_index = latest.source_second_ms // 1_000
        assert latest.as_mapping()["trade_imbalance"] == pytest.approx(
            seconds[source_index - 1].trade_imbalance
        )
        assert latest.as_mapping()["trade_imbalance"] != pytest.approx(
            seconds[source_index].trade_imbalance
        )
        mapping = latest.as_mapping()
        signed_flow_10s = sum(
            row.aggressive_buy_volume - row.aggressive_sell_volume
            for row in seconds[source_index - 10 : source_index]
        )
        opposing_depth = (
            latest.close_ask_qty if signed_flow_10s >= 0.0 else latest.close_bid_qty
        )
        expected_pressure = (
            math.copysign(
                math.log1p(abs(signed_flow_10s) / opposing_depth), signed_flow_10s
            )
            if signed_flow_10s
            else 0.0
        )
        assert mapping["signed_pressure_to_opposing_depth_10s"] == pytest.approx(
            expected_pressure,
            rel=3e-5,
            abs=3e-5,
        )
        assert mapping["log_bid_l1_depth_quote"] == pytest.approx(
            math.log1p(latest.close_bid * latest.close_bid_qty), rel=3e-5
        )
    finally:
        warehouse.connection.close()


def test_streaming_engine_resets_warmup_on_gap_and_rejects_crossed_quotes() -> None:
    engine = StreamingMicrostructureFeatureEngine("BTCUSDT")
    assert engine.append(_second(0)) is None
    assert engine.append(_second(1)) is None
    assert engine.append(_second(3)) is None
    assert engine.gap_resets == 1
    assert engine.warmup_remaining_seconds == 3_600

    crossed = _second(4)
    crossed = MicrostructureSecond(
        **{**crossed.__dict__, "close_bid": crossed.close_ask}
    )
    with pytest.raises(ValueError, match="crossed"):
        engine.append(crossed)


def test_streaming_week_context_uses_utc_exchange_time() -> None:
    saturday_start_ms = 1_694_217_600_000
    engine = StreamingMicrostructureFeatureEngine(
        "BTCUSDT",
        decision_cadence_seconds=1,
    )
    emitted = None
    for index in range(3_601):
        emitted = engine.append(
            _second(index, second_ms=saturday_start_ms + index * 1_000)
        )

    assert emitted is not None
    values = emitted.as_mapping()
    assert values["weekend_flag"] == 1.0
    assert values["utc_week_sin"] ** 2 + values["utc_week_cos"] ** 2 == pytest.approx(
        1.0,
        abs=1e-6,
    )

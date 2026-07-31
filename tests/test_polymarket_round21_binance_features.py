from __future__ import annotations

import json
import math

import pytest

from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_round21_binance_features import (
    POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    Round21BinanceBookTicker,
    Round21BinanceTrade,
    Round21IndependentBinanceFeatureEngine,
    parse_round21_binance_record,
)


EPOCH_MS = 1_800_000_000_000


def _record(
    market: str,
    sequence: int,
    payload: object | str,
    *,
    wall_offset_ms: int,
    connection_suffix: str = "a" * 32,
) -> CaptureFrameRecord:
    stream = "binance_spot" if market == "spot" else "binance_futures"
    lane = "binance:spot:btc" if market == "spot" else "binance:futures:btc"
    return CaptureFrameRecord(
        stream=stream,
        connection_id=f"{lane}:{connection_suffix}",
        sequence_number=sequence,
        received_wall_ms=EPOCH_MS + wall_offset_ms,
        received_monotonic_ns=(EPOCH_MS + wall_offset_ms) * 1_000_000 + sequence,
        raw_text=(
            payload
            if isinstance(payload, str)
            else json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ),
    )


def _spot_book(update_id: int, *, bid: str = "60000", ask: str = "60001") -> dict:
    return {
        "stream": "btcusdt@bookTicker",
        "data": {
            "u": update_id,
            "s": "BTCUSDT",
            "b": bid,
            "B": "2",
            "a": ask,
            "A": "3",
        },
    }


def _usdm_book(
    update_id: int,
    *,
    time_offset_ms: int,
    bid: str = "60002",
    ask: str = "60003",
) -> dict:
    return {
        "stream": "btcusdt@bookTicker",
        "data": {
            "e": "bookTicker",
            "E": EPOCH_MS + time_offset_ms,
            "T": EPOCH_MS + time_offset_ms - 1,
            "u": update_id,
            "s": "BTCUSDT",
            "b": bid,
            "B": "4",
            "a": ask,
            "A": "5",
        },
    }


def _trade(
    trade_id: int,
    *,
    time_offset_ms: int,
    price: str,
    buyer_is_maker: bool,
) -> dict:
    return {
        "stream": "btcusdt@trade",
        "data": {
            "e": "trade",
            "E": EPOCH_MS + time_offset_ms,
            "s": "BTCUSDT",
            "t": trade_id,
            "p": price,
            "q": "0.25",
            "T": EPOCH_MS + time_offset_ms - 1,
            "m": buyer_is_maker,
            "M": True,
        },
    }


def test_round21_parses_official_spot_and_usdm_wire_shapes() -> None:
    spot = parse_round21_binance_record(
        _record("spot", 1, _spot_book(10), wall_offset_ms=100)
    )
    futures = parse_round21_binance_record(
        _record(
            "usdm",
            1,
            _usdm_book(20, time_offset_ms=100),
            wall_offset_ms=101,
        )
    )
    trade = parse_round21_binance_record(
        _record(
            "spot",
            2,
            _trade(
                30,
                time_offset_ms=102,
                price="60000.5",
                buyer_is_maker=False,
            ),
            wall_offset_ms=103,
        )
    )

    assert isinstance(spot, Round21BinanceBookTicker)
    assert spot.event_time_ms is None
    assert spot.transaction_time_ms is None
    assert isinstance(futures, Round21BinanceBookTicker)
    assert futures.event_time_ms == EPOCH_MS + 100
    assert isinstance(trade, Round21BinanceTrade)
    assert trade.quote_notional == pytest.approx(15_000.125)
    assert trade.trading_authority is False


def test_round21_parser_rejects_duplicate_keys_schema_drift_and_crossed_book() -> None:
    duplicate = (
        '{"stream":"btcusdt@bookTicker","data":'
        '{"u":1,"u":2,"s":"BTCUSDT","b":"1","B":"1","a":"2","A":"1"}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        parse_round21_binance_record(
            _record("spot", 1, duplicate, wall_offset_ms=1)
        )

    changed = _spot_book(1)
    changed["data"]["new_field"] = "drift"
    with pytest.raises(ValueError, match="schema drifted"):
        parse_round21_binance_record(
            _record("spot", 1, changed, wall_offset_ms=1)
        )

    with pytest.raises(ValueError, match="book ticker is invalid"):
        parse_round21_binance_record(
            _record(
                "spot",
                1,
                _spot_book(1, bid="60001", ask="60000"),
                wall_offset_ms=1,
            )
        )


def test_round21_engine_builds_matched_spot_and_usdm_features() -> None:
    engine = Round21IndependentBinanceFeatureEngine()
    records = (
        _record("spot", 1, _spot_book(10), wall_offset_ms=100),
        _record(
            "spot",
            2,
            _trade(
                30,
                time_offset_ms=110,
                price="60000",
                buyer_is_maker=False,
            ),
            wall_offset_ms=111,
        ),
        _record(
            "spot",
            3,
            _trade(
                31,
                time_offset_ms=120,
                price="60001",
                buyer_is_maker=True,
            ),
            wall_offset_ms=121,
        ),
        _record(
            "spot",
            4,
            _spot_book(11, bid="60001", ask="60002"),
            wall_offset_ms=130,
        ),
        _record(
            "usdm",
            1,
            _usdm_book(20, time_offset_ms=101),
            wall_offset_ms=102,
        ),
        _record(
            "usdm",
            2,
            _trade(
                40,
                time_offset_ms=112,
                price="60002",
                buyer_is_maker=False,
            ),
            wall_offset_ms=113,
        ),
        _record(
            "usdm",
            3,
            _trade(
                41,
                time_offset_ms=122,
                price="60003",
                buyer_is_maker=True,
            ),
            wall_offset_ms=123,
        ),
        _record(
            "usdm",
            4,
            _usdm_book(
                21,
                time_offset_ms=131,
                bid="60003",
                ask="60004",
            ),
            wall_offset_ms=132,
        ),
    )
    for record in records:
        engine.ingest_record(record)

    features = engine.build(EPOCH_MS + 140)

    assert features.spot_available is True
    assert features.usdm_available is True
    assert len(features.spot_values) == len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES)
    assert len(features.usdm_values) == len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features.usdm_values)
    assert features.usdm_values[-9] > 0.0
    assert features.spot_source_chain_sha256 != "0" * 64
    assert features.usdm_source_chain_sha256 != features.spot_source_chain_sha256
    assert features.trading_authority is False
    assert engine.credentials_used is False
    assert engine.account_connected is False
    assert engine.execution_connected is False


def test_round21_stale_spot_blocks_only_optional_layers() -> None:
    engine = Round21IndependentBinanceFeatureEngine()
    engine.ingest_record(
        _record("spot", 1, _spot_book(10), wall_offset_ms=100)
    )
    engine.ingest_record(
        _record(
            "usdm",
            1,
            _usdm_book(20, time_offset_ms=100),
            wall_offset_ms=100,
        )
    )

    features = engine.build(EPOCH_MS + 1_101)

    assert features.spot_available is False
    assert features.usdm_available is False
    assert not any(features.spot_values)
    assert not any(features.usdm_values)
    assert features.spot_maximum_receipt_ms == 0
    assert features.usdm_maximum_receipt_ms == 0


def test_round21_reconnect_requires_explicit_reset_and_carries_no_state() -> None:
    engine = Round21IndependentBinanceFeatureEngine()
    engine.ingest_record(
        _record("spot", 1, _spot_book(10), wall_offset_ms=100)
    )
    replacement = _record(
        "spot",
        1,
        _spot_book(50, bid="61000", ask="61001"),
        wall_offset_ms=200,
        connection_suffix="b" * 32,
    )

    with pytest.raises(ValueError, match="explicit epoch reset"):
        engine.ingest_record(replacement)
    engine.reset_market("spot", replacement.connection_id)
    engine.ingest_record(replacement)
    features = engine.build(EPOCH_MS + 201)

    assert features.spot_available is True
    trade_count_indices = [
        index
        for index, name in enumerate(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES)
        if "trade_count" in name
    ]
    assert all(features.spot_values[index] == 0.0 for index in trade_count_indices)


def test_round21_engine_rejects_sequence_loss_and_future_materialization() -> None:
    engine = Round21IndependentBinanceFeatureEngine()
    engine.ingest_record(
        _record("spot", 1, _spot_book(10), wall_offset_ms=100)
    )
    with pytest.raises(ValueError, match="connection epoch differs"):
        engine.ingest_record(
            _record("spot", 3, _spot_book(11), wall_offset_ms=101)
        )
    with pytest.raises(ValueError, match="future receipts"):
        engine.build(EPOCH_MS + 99)


def test_round21_control_frames_consume_capture_sequence_without_features() -> None:
    engine = Round21IndependentBinanceFeatureEngine()
    engine.ingest_record(_record("spot", 1, "PING", wall_offset_ms=10))
    engine.ingest_record(
        _record("spot", 2, _spot_book(10), wall_offset_ms=20)
    )

    features = engine.build(EPOCH_MS + 21)

    assert features.spot_available is True

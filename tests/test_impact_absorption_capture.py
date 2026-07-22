from __future__ import annotations

import json

import pytest

from simple_ai_trading.impact_absorption import parse_book_ticker
from simple_ai_trading.impact_absorption_capture import (
    ImpactCaptureConfig,
    _ImpactFrameWriter,
    _market_stream_url,
    _public_stream_url,
    _request_weight_limit,
    _tick_sizes,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
    ImpactCaptureMessage,
)
from simple_ai_trading.impact_capture_frame import ImpactCaptureFrameRecord


RUN_ID = "a" * 32
SEGMENT_ID = "b" * 32


def _message() -> ImpactCaptureMessage:
    payload = {
        "e": "bookTicker",
        "E": 1_010,
        "T": 1_009,
        "s": "BTCUSDT",
        "u": 103,
        "b": "100.0",
        "B": "4",
        "a": "100.1",
        "A": "1",
        "st": 1,
        "ps": "BTCUSDT",
    }
    raw = json.dumps(
        {"stream": "btcusdt@bookTicker", "data": payload},
        separators=(",", ":"),
    )
    return ImpactCaptureMessage(
        record=ImpactCaptureFrameRecord(
            stream="binance_futures_public",
            connection_id="binance-public:test",
            sequence_number=0,
            received_wall_ns=1_784_058_600_000_000_000,
            received_monotonic_ns=100,
            raw_text=raw,
        ),
        event=parse_book_ticker(payload, symbol="BTCUSDT", receive_time_ns=100),
        segment_id=SEGMENT_ID,
    )


def _start_store(database) -> None:
    with ImpactAbsorptionStore(database) as store:
        store.start_run(
            run_id=RUN_ID,
            started_wall_ns=1,
            started_monotonic_ns=1,
            config={"mode": "probe"},
        )
        store.start_segment(
            run_id=RUN_ID,
            segment_id=SEGMENT_ID,
            symbol="BTCUSDT",
            started_wall_ns=2,
            started_monotonic_ns=2,
            snapshot_update_id=100,
            tick_size=0.1,
            clock_offset_ns=0,
            clock_rtt_ns=1,
            cooldown_until_wall_ns=0,
        )


def test_capture_config_separates_bounded_probe_from_hour_qualification() -> None:
    ImpactCaptureConfig(mode="probe", duration_seconds=300).validate()
    ImpactCaptureConfig(mode="qualification", duration_seconds=3_600).validate()

    with pytest.raises(ValueError, match="capped at 300"):
        ImpactCaptureConfig(mode="probe", duration_seconds=301).validate()
    with pytest.raises(ValueError, match="at least 3600"):
        ImpactCaptureConfig(mode="qualification", duration_seconds=3_599).validate()
    with pytest.raises(ValueError, match="queue capacity"):
        ImpactCaptureConfig(queue_capacity_messages=65_537).validate()


def test_stream_topology_is_three_symbols_only_and_uses_specific_liquidations() -> None:
    public_url = _public_stream_url()
    market_url = _market_stream_url()

    for symbol in IMPACT_CAPTURE_SYMBOLS:
        lowered = symbol.lower()
        assert f"{lowered}@depth@100ms" in public_url
        assert f"{lowered}@bookTicker" in public_url
        assert f"{lowered}@aggTrade" in market_url
        assert f"{lowered}@markPrice@1s" in market_url
        assert f"{lowered}@forceOrder" in market_url
    assert "!forceOrder@arr" not in market_url


def test_one_writer_flushes_atomically_without_blocking_the_caller(tmp_path) -> None:
    database = tmp_path / "impact.duckdb"
    _start_store(database)
    config = ImpactCaptureConfig(
        database=str(database),
        duration_seconds=1,
        queue_capacity_messages=8,
        frame_message_limit=1,
        frame_flush_seconds=0.01,
    )
    config.validate()
    writer = _ImpactFrameWriter(config, RUN_ID)
    writer.start()

    writer.put(_message())
    assert writer.stop(timeout_seconds=5) is True

    assert writer.failed.is_set() is False
    assert writer.frame_count == 1
    assert writer.message_count == 1
    with ImpactAbsorptionStore(database, read_only=True) as store:
        assert store.audit_run(RUN_ID).passed is True


def test_writer_queue_overflow_is_explicit_before_silent_drop(tmp_path) -> None:
    writer = _ImpactFrameWriter(
        ImpactCaptureConfig(
            database=str(tmp_path / "unused.duckdb"),
            duration_seconds=1,
            queue_capacity_messages=1,
        ),
        RUN_ID,
    )
    writer.put(_message())
    with pytest.raises(RuntimeError, match="queue overflow"):
        writer.put(_message())


def test_exchange_metadata_requires_one_unambiguous_minute_limit_and_tick_filter() -> (
    None
):
    exchange_info = {
        "rateLimits": [
            {
                "rateLimitType": "REQUEST_WEIGHT",
                "interval": "MINUTE",
                "intervalNum": 1,
                "limit": 2400,
            }
        ],
        "symbols": [
            {
                "symbol": symbol,
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}],
            }
            for symbol in IMPACT_CAPTURE_SYMBOLS
        ],
    }

    assert _request_weight_limit(exchange_info) == 2400
    assert _tick_sizes(exchange_info) == {
        symbol: "0.10" for symbol in IMPACT_CAPTURE_SYMBOLS
    }
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _request_weight_limit({**exchange_info, "rateLimits": []})

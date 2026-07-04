from __future__ import annotations

from simple_ai_trading.api import Candle
from simple_ai_trading.data_coverage import describe_candle_coverage, iso_utc


def _candle(open_time: int, close_time: int) -> Candle:
    return Candle(
        open_time=open_time,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1.0,
        close_time=close_time,
    )


def test_describe_candle_coverage_records_full_history_span_and_gaps() -> None:
    candles = [
        _candle(0, 60_000),
        _candle(60_000, 120_000),
        _candle(240_000, 300_000),
    ]

    report = describe_candle_coverage(
        symbol="btcusdc",
        market_type="spot",
        interval="1m",
        available_candles=candles,
        used_candles=candles,
        rows_used=2,
    )

    assert report.symbol == "BTCUSDC"
    assert report.full_history_requested is True
    assert report.full_available_history_used is True
    assert report.candles_available == 3
    assert report.rows_used == 2
    assert report.gap_count == 1
    assert report.largest_gap_intervals == 3.0
    assert report.integrity_status == "fail"
    assert "prices_from_timestamped_closed_candles" in report.truth_basis
    assert "coverage_gaps_detected" in report.notes


def test_describe_candle_coverage_marks_bounded_window() -> None:
    available = [_candle(0, 60_000), _candle(60_000, 120_000), _candle(120_000, 180_000)]
    used = available[1:]

    report = describe_candle_coverage(
        symbol="ETHUSDC",
        market_type="futures",
        interval="1m",
        available_candles=available,
        used_candles=used,
        rows_used=1,
        requested_start_ms=120_000,
        requested_end_ms=180_000,
    )

    assert report.full_history_requested is False
    assert report.full_available_history_used is False
    assert report.candles_used == 2
    assert report.used_start_utc == iso_utc(120_000)
    assert report.integrity_status == "warn"
    assert "operator_requested_bounded_window" in report.notes


def test_describe_candle_coverage_marks_recent_api_limit_as_not_full_history() -> None:
    candles = [_candle(0, 60_000), _candle(60_000, 120_000)]

    report = describe_candle_coverage(
        symbol="BTCUSDC",
        market_type="spot",
        interval="1m",
        available_candles=candles,
        used_candles=candles,
        rows_used=1,
        source_scope="binance_recent_limit",
    )

    assert report.source_scope == "binance_recent_limit"
    assert report.full_available_history_used is False
    assert "recent_api_limit_not_full_history" in report.integrity_warnings

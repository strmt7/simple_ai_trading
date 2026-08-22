from __future__ import annotations

import math
from pathlib import Path

import pytest

from simple_ai_trading.market_store import MarketDataStore, _missing_candle_count


def test_missing_candle_count_ignores_null_timestamp_rows() -> None:
    assert (
        _missing_candle_count(
            [{"open_time": None}, {"open_time": 0}, {"open_time": 120_000}],
            interval_ms=60_000,
            span_start=0,
            span_end=120_000,
            first_open_time=0,
            last_open_time=120_000,
        )
        == 1
    )


def test_market_data_store_reports_empty_bounded_coverage(tmp_path: Path) -> None:
    with MarketDataStore(tmp_path / "empty-window.sqlite") as store:
        quality = store.coverage_quality(
            "BTCUSDC",
            "spot",
            "1m",
            60_000,
            start_ms=0,
            end_ms=180_000,
        )
        inverted = store.coverage_quality(
            "BTCUSDC",
            "spot",
            "1m",
            60_000,
            start_ms=180_000,
            end_ms=0,
        )

    assert quality.expected_count == 4
    assert quality.gap_count == 4
    assert quality.coverage_ratio == 0.0
    assert inverted.expected_count == 0
    assert inverted.gap_count == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bidPrice", float("nan")),
        ("askPrice", float("inf")),
        ("bidQty", float("-inf")),
        ("askQty", 0.0),
        ("askQty", -1.0),
    ],
)
def test_market_data_store_rejects_non_finite_or_non_positive_top_of_book_values(
    tmp_path: Path,
    key: str,
    value: float,
) -> None:
    payload = {"bidPrice": 99.0, "bidQty": 2.0, "askPrice": 101.0, "askQty": 3.0}
    payload[key] = value

    with MarketDataStore(tmp_path / f"invalid-{key}-{value!s}.sqlite") as store:
        with pytest.raises(ValueError, match=rf"{key} must be finite and positive"):
            store.insert_top_of_book_snapshot("binance", "BTCUSDT", "spot", payload)


def test_market_data_store_keeps_extreme_top_of_book_metrics_finite(
    tmp_path: Path,
) -> None:
    with MarketDataStore(tmp_path / "extreme-book.sqlite") as store:
        assert (
            store.insert_top_of_book_snapshot(
                "binance",
                "BTCUSDT",
                "spot",
                {
                    "bidPrice": 1e308,
                    "bidQty": 1e-308,
                    "askPrice": 1e308,
                    "askQty": 1e-308,
                },
            )
            == 1
        )
        snapshot = store.latest_top_of_book("BTCUSDT", "spot")
        assert snapshot is not None
        assert snapshot.mid_price == 1e308
        assert math.isfinite(snapshot.depth_notional)

        with pytest.raises(ValueError, match="depth notional must be finite"):
            store.insert_top_of_book_snapshot(
                "binance",
                "BTCUSDT",
                "spot",
                {
                    "bidPrice": 1e308,
                    "bidQty": 1e308,
                    "askPrice": 1e308,
                    "askQty": 1e308,
                },
                ts_ms=1,
            )
        assert len(store.fetch_top_of_book("BTCUSDT", "spot")) == 1

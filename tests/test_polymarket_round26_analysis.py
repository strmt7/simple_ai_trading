from __future__ import annotations

from decimal import Decimal
import json

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_round26_analysis import (
    _BookSeries,
    _PricePoint,
    _Signal,
    _configuration_result,
    _execute_taker_trade,
    _maximum_drawdown,
    _return_bps,
)


_START_SECONDS = 1_784_058_600
_START_MS = _START_SECONDS * 1_000


def _market():
    return parse_polymarket_five_minute_market(
        {
            "id": "round26-market",
            "question": "BTC Up or Down",
            "conditionId": "0x" + "7" * 64,
            "slug": f"btc-updown-5m-{_START_SECONDS}",
            "eventStartTime": "2026-07-14T19:50:00Z",
            "endDate": "2026-07-14T19:55:00Z",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "acceptingOrders": True,
            "clobTokenIds": json.dumps(["7" * 40, "7" * 39 + "1"]),
            "outcomes": '["Up", "Down"]',
            "orderPriceMinTickSize": 0.01,
            "orderMinSize": 5,
            "feesEnabled": True,
            "feeSchedule": {
                "exponent": 1,
                "rate": 0.07,
                "takerOnly": True,
                "rebateRate": 0.2,
            },
            "liquidityNum": 20_000,
            "volumeNum": 50_000,
            "resolutionSource": (
                "https://data.chain.link/streams/btc-usd-twap-60s-streams"
            ),
            "cryptoMarketConfigId": "btc-5m-twap-60",
            "cryptoMarketConfig": {
                "asset": "btc",
                "duration": "5m",
                "id": "btc-5m-twap-60",
                "twapEnabled": True,
                "twapLookbackSeconds": 60,
            },
        }
    )


def _book(market, *, monotonic_ns: int, bid: str, ask: str):
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=market.up_token_id,
        bids=(BookLevel(Decimal(bid), Decimal("5")),),
        asks=(BookLevel(Decimal(ask), Decimal("5")),),
        source_time_ms=_START_MS + monotonic_ns // 1_000_000,
        received_wall_ms=_START_MS + monotonic_ns // 1_000_000,
        received_monotonic_ns=monotonic_ns,
        source_payload_sha256="a" * 64,
    ).validated()
    return PolymarketRecordedBook(
        run_id="round26",
        event_id=f"event-{monotonic_ns}",
        event_type="book",
        connection_id="connection",
        segment_id="segment",
        sequence_number=monotonic_ns,
        sub_index=0,
        market=market,
        outcome="Up",
        tick_size=Decimal("0.01"),
        snapshot=snapshot,
    )


def _series(*books: PolymarketRecordedBook) -> _BookSeries:
    return _BookSeries(
        times_ns=tuple(book.received_monotonic_ns for book in books),
        books=books,
    )


def test_return_bps_is_causal_and_rejects_stale_receipts() -> None:
    points = (
        _PricePoint(1_000_000_000, 1_000, 100.0),
        _PricePoint(1_100_000_000, 1_100, 101.0),
        _PricePoint(1_200_000_000, 1_200, 500.0),
    )
    times = tuple(point.received_monotonic_ns for point in points)

    value = _return_bps(
        points,
        times,
        now_ns=1_100_000_000,
        lookback_ms=100,
    )

    assert value == pytest.approx(99.5033085317)
    assert (
        _return_bps(
            points,
            times,
            now_ns=1_500_000_000,
            lookback_ms=100,
        )
        is None
    )


def test_book_lookup_enforces_maximum_observation_delay() -> None:
    market = _market()
    book = _book(market, monotonic_ns=1_300_000_000, bid="0.49", ask="0.50")
    series = _series(book)

    assert series.first_at_or_after(1_100_000_000, maximum_delay_ms=200) == book
    assert series.first_at_or_after(1_099_999_999, maximum_delay_ms=200) is None


def test_taker_round_trip_delays_both_legs_and_charges_exact_fees() -> None:
    market = _market()
    entry = _book(market, monotonic_ns=1_300_000_000, bid="0.49", ask="0.50")
    optimistic_early_exit = _book(
        market,
        monotonic_ns=1_850_000_000,
        bid="0.60",
        ask="0.61",
    )
    delayed_exit = _book(
        market,
        monotonic_ns=2_100_000_000,
        bid="0.49",
        ask="0.50",
    )

    trade = _execute_taker_trade(
        _Signal(1_000_000_000, _START_MS + 1_000, 2.0),
        market,
        _series(entry, optimistic_early_exit, delayed_exit),
        mode="momentum",
        delay_ms=250,
        hold_ms=500,
    )

    assert trade is not None
    assert trade.entry_monotonic_ns == 1_300_000_000
    assert trade.exit_monotonic_ns == 2_100_000_000
    assert trade.gross_pnl == Decimal("-0.05")
    assert trade.fees == Decimal("0.17497")
    assert trade.net_pnl == Decimal("-0.22497")


def test_configuration_prevents_overlap_until_observed_exit() -> None:
    market = _market()
    books = _series(
        _book(market, monotonic_ns=1_300_000_000, bid="0.49", ask="0.50"),
        _book(market, monotonic_ns=2_100_000_000, bid="0.51", ask="0.52"),
        _book(market, monotonic_ns=2_200_000_000, bid="0.52", ask="0.53"),
        _book(market, monotonic_ns=3_000_000_000, bid="0.54", ask="0.55"),
    )
    signals = (
        _Signal(1_000_000_000, _START_MS + 1_000, 2.0),
        _Signal(1_900_000_000, _START_MS + 1_900, 2.0),
    )

    result, trades = _configuration_result(
        signals,
        (market,),
        (market.event_start_ms,),
        {market.up_token_id: books},
        lookback_ms=100,
        threshold_bps=1.0,
        mode="momentum",
        delay_ms=250,
        hold_ms=500,
    )

    assert result["eligible_signal_count"] == 1
    assert result["trade_count"] == 1
    assert len(trades) == 1


def test_maximum_drawdown_uses_running_equity_peak() -> None:
    assert _maximum_drawdown(
        (Decimal("1"), Decimal("-0.4"), Decimal("0.2"), Decimal("-1.1"))
    ) == Decimal("1.3")

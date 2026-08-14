from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_replay import PolymarketResolutionEvidence
from simple_ai_trading.polymarket_round26_analysis import (
    _BookSeries,
    _PricePoint,
    _Signal,
    _configuration_result,
    _execute_settlement_trade,
    _execute_taker_trade,
    _maximum_drawdown,
    _return_bps,
    _settlement_configuration_result,
    _settlement_mechanism_audit,
    _source_points,
    _twap_point,
)
from simple_ai_trading.polymarket_twap60 import PolymarketTwap60Tick


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


def _resolution(market, *, winning_outcome: str) -> PolymarketResolutionEvidence:
    return PolymarketResolutionEvidence(
        run_id="round26",
        event_id="resolution",
        condition_id=market.condition_id,
        winning_asset_id=(
            market.up_token_id if winning_outcome == "Up" else market.down_token_id
        ),
        winning_outcome=winning_outcome,
        resolved_at_ms=market.end_ms + 1_000,
        received_wall_ms=market.end_ms + 2_000,
        received_monotonic_ns=10_000_000_000,
        event_sha256="b" * 64,
        source="clob_market",
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


def test_source_points_use_prevalidated_segmented_source_reconstruction() -> None:
    class Store:
        def iter_public_events(self, run_id, **controls):
            assert run_id == "round26"
            assert controls["verified_source"] is False
            yield SimpleNamespace(
                stream="binance_spot",
                event_type="trade",
                symbol="BTC",
                received_monotonic_ns=1_000,
                received_wall_ms=1,
                event={"data": {"p": "100"}},
            )

    spot, futures, twap, event_count = _source_points(Store(), "round26")

    assert len(spot) == 1
    assert futures == ()
    assert twap == ()
    assert event_count == 1


def test_twap_point_preserves_exact_e18_value() -> None:
    event = SimpleNamespace(
        event_type="crypto_prices_twap_sixty:update",
        received_wall_ms=_START_MS + 3_000,
        received_monotonic_ns=3_000_000_000,
        event={
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "timestamp": _START_MS + 1_500,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": _START_MS,
                "value": 63947.08172696262,
                "window_s": 60,
                "full_accuracy_value": "63947081726962622791680",
            },
        },
    )

    point = _twap_point(event)

    assert point is not None
    assert point.exact_e18 == 63_947_081_726_962_622_791_680


def test_settlement_audit_uses_exact_boundaries_and_official_outcome() -> None:
    market = _market()
    resolution = PolymarketResolutionEvidence(
        run_id="round26",
        event_id="resolution",
        condition_id=market.condition_id,
        winning_asset_id=market.up_token_id,
        winning_outcome="Up",
        resolved_at_ms=market.end_ms + 1_000,
        received_wall_ms=market.end_ms + 2_000,
        received_monotonic_ns=10_000_000_000,
        event_sha256="b" * 64,
        source="clob_market",
    )
    points = (
        PolymarketTwap60Tick(
            market.event_start_ms,
            market.event_start_ms + 1_500,
            market.event_start_ms + 3_000,
            1_000_000_000,
            100 * 10**18,
            "a" * 64,
        ),
        PolymarketTwap60Tick(
            market.end_ms,
            market.end_ms + 1_500,
            market.end_ms + 3_000,
            2_000_000_000,
            101 * 10**18,
            "b" * 64,
        ),
    )

    audit = _settlement_mechanism_audit((market,), (resolution,), points)

    assert audit["evaluated_market_count"] == 1
    assert audit["agreement_count"] == 1
    assert audit["agreement_rate"] == 1.0
    assert audit["minimum_eight_markets_met"] is False
    assert audit["exact_boundary_rule_supported_in_pilot"] is False
    assert audit["edge_claim"] is False


def test_settlement_audit_rejects_conflicting_source_timestamp() -> None:
    market = _market()
    first = PolymarketTwap60Tick(
        market.event_start_ms,
        market.event_start_ms,
        market.event_start_ms,
        1,
        100 * 10**18,
        "a" * 64,
    )

    with pytest.raises(ValueError, match="duplicate source values conflict"):
        _settlement_mechanism_audit(
            (market,),
            (),
            (first, replace(first, exact_e18=101 * 10**18)),
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


def test_settlement_trade_charges_one_fee_and_uses_official_payout() -> None:
    market = _market()
    entry = _book(market, monotonic_ns=1_300_000_000, bid="0.49", ask="0.50")

    trade = _execute_settlement_trade(
        _Signal(1_000_000_000, _START_MS + 1_000, 2.0),
        market,
        _resolution(market, winning_outcome="Up"),
        _series(entry),
        mode="momentum",
        delay_ms=250,
    )

    assert trade is not None
    assert trade.payout_per_share == Decimal("1")
    assert trade.gross_pnl == Decimal("2.50")
    assert trade.fees == Decimal("0.0875")
    assert trade.net_pnl == Decimal("2.4125")


def test_settlement_configuration_opens_at_most_once_per_market() -> None:
    market = _market()
    book = _book(market, monotonic_ns=1_300_000_000, bid="0.49", ask="0.50")
    signals = (
        _Signal(1_000_000_000, _START_MS + 1_000, 2.0),
        _Signal(1_100_000_000, _START_MS + 1_100, 3.0),
    )

    result, trades = _settlement_configuration_result(
        signals,
        (market,),
        (market.event_start_ms,),
        {market.condition_id: _resolution(market, winning_outcome="Down")},
        {market.up_token_id: _series(book)},
        lookback_ms=250,
        threshold_bps=1.0,
        mode="momentum",
        delay_ms=250,
    )

    assert result["eligible_signal_count"] == 1
    assert result["trade_count"] == 1
    assert len(trades) == 1
    assert trades[0].net_pnl == Decimal("-2.5875")


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

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    Round27FeatureRow,
    Round27PublicSourceSeries,
    Round27TradePoint,
    Round27TradeSeries,
    _book_flow,
    build_round27_condition_features,
    load_round27_public_source_series,
)
from simple_ai_trading.polymarket_twap60 import PolymarketTwap60Tick


_START_SECONDS = 1_786_784_400
_START_MS = _START_SECONDS * 1_000


def _market():
    return parse_polymarket_five_minute_market(
        {
            "id": "round27-market",
            "question": "BTC Up or Down",
            "conditionId": "0x" + "7" * 64,
            "slug": f"btc-updown-5m-{_START_SECONDS}",
            "eventStartTime": "2026-08-15T09:00:00Z",
            "endDate": "2026-08-15T09:05:00Z",
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


def _book(
    market,
    *,
    outcome: str,
    offset_ms: int,
    bid: str,
    ask: str,
    segment: str = "segment-1",
) -> PolymarketRecordedBook:
    token = market.up_token_id if outcome == "Up" else market.down_token_id
    wall_ms = _START_MS + offset_ms
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=token,
        bids=(
            BookLevel(Decimal(bid), Decimal("8")),
            BookLevel(Decimal(bid) - Decimal("0.01"), Decimal("12")),
            BookLevel(Decimal(bid) - Decimal("0.02"), Decimal("15")),
            BookLevel(Decimal(bid) - Decimal("0.03"), Decimal("18")),
            BookLevel(Decimal(bid) - Decimal("0.04"), Decimal("20")),
        ),
        asks=(
            BookLevel(Decimal(ask), Decimal("6")),
            BookLevel(Decimal(ask) + Decimal("0.01"), Decimal("10")),
            BookLevel(Decimal(ask) + Decimal("0.02"), Decimal("14")),
            BookLevel(Decimal(ask) + Decimal("0.03"), Decimal("17")),
            BookLevel(Decimal(ask) + Decimal("0.04"), Decimal("19")),
        ),
        source_time_ms=wall_ms,
        received_wall_ms=wall_ms,
        received_monotonic_ns=offset_ms * 1_000_000 + 1,
        source_payload_sha256=hashlib.sha256(
            f"{outcome}:{offset_ms}:{segment}".encode("ascii")
        ).hexdigest(),
    ).validated()
    return PolymarketRecordedBook(
        run_id="round27",
        event_id=f"{outcome}-{offset_ms}-{segment}",
        event_type="book",
        connection_id=f"connection-{segment}",
        segment_id=segment,
        sequence_number=offset_ms,
        sub_index=0,
        market=market,
        outcome=outcome,
        tick_size=Decimal("0.01"),
        snapshot=snapshot,
    )


def _trade(stream: str, offset_ms: int, price: float) -> Round27TradePoint:
    return Round27TradePoint(
        stream=stream,
        received_wall_ms=_START_MS + offset_ms,
        received_monotonic_ns=(offset_ms + 2_000_000) * 1_000_000 + 1,
        price=price,
        quantity=0.25,
        buyer_is_maker=offset_ms % 2 == 0,
        event_sha256=hashlib.sha256(
            f"{stream}:{offset_ms}:{price}".encode("ascii")
        ).hexdigest(),
    )


def _source() -> Round27PublicSourceSeries:
    offsets = tuple(range(-1_200_000, 30_001, 250))
    spot = tuple(
        _trade("binance_spot", offset, 100_000.0 + offset / 10_000.0)
        for offset in offsets
    )
    usdm = tuple(
        _trade("binance_futures", offset, 100_010.0 + offset / 11_000.0)
        for offset in offsets
    )
    twap = tuple(
        PolymarketTwap60Tick(
            source_time_ms=_START_MS + offset,
            publisher_time_ms=_START_MS + offset,
            received_wall_ms=_START_MS + offset,
            received_monotonic_ns=offset * 1_000_000 + 1,
            exact_e18=(100_000 * 10**18) + offset * 10**13,
            source_payload_sha256=hashlib.sha256(
                f"twap:{offset}".encode("ascii")
            ).hexdigest(),
        )
        for offset in range(0, 30_001, 1_000)
    )
    return Round27PublicSourceSeries(
        run_id="round27",
        spot=Round27TradeSeries.from_points("binance_spot", spot),
        usdm=Round27TradeSeries.from_points("binance_futures", usdm),
        twap=twap,
        source_chain_sha256="a" * 64,
    ).validated()


def test_build_round27_condition_features_is_complete_and_target_blind() -> None:
    market = _market()
    books = (
        _book(market, outcome="Up", offset_ms=29_000, bid="0.48", ask="0.50"),
        _book(market, outcome="Down", offset_ms=29_000, bid="0.50", ask="0.52"),
        _book(market, outcome="Up", offset_ms=30_000, bid="0.49", ask="0.51"),
        _book(market, outcome="Down", offset_ms=30_000, bid="0.49", ask="0.51"),
    )

    rows = build_round27_condition_features(
        market=market,
        books=books,
        source=_source(),
        eligible_intervals=(
            {
                "eligible": True,
                "interval_start_ms": _START_MS,
                "interval_end_ms": _START_MS + 30_000,
            },
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.decision_time_ms == _START_MS + 30_000
    assert row.feature_names_sha256 == POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
    assert len(row.values) == len(POLYMARKET_ROUND27_FEATURE_NAMES)
    assert row.target_accessed is False
    assert row.trading_authority is False
    assert row.maximum_receipt_wall_ms <= row.decision_time_ms


def test_round27_condition_features_require_complete_twenty_minute_context() -> None:
    market = _market()
    complete = _source()
    offsets = tuple(range(0, 30_001, 250))
    short = Round27PublicSourceSeries(
        run_id="round27",
        spot=Round27TradeSeries.from_points(
            "binance_spot",
            tuple(
                _trade("binance_spot", offset, 100_000.0 + offset / 10_000.0)
                for offset in offsets
            ),
        ),
        usdm=Round27TradeSeries.from_points(
            "binance_futures",
            tuple(
                _trade(
                    "binance_futures",
                    offset,
                    100_010.0 + offset / 11_000.0,
                )
                for offset in offsets
            ),
        ),
        twap=complete.twap,
        source_chain_sha256="b" * 64,
    ).validated()
    books = (
        _book(market, outcome="Up", offset_ms=30_000, bid="0.49", ask="0.51"),
        _book(market, outcome="Down", offset_ms=30_000, bid="0.49", ask="0.51"),
    )

    rows = build_round27_condition_features(
        market=market,
        books=books,
        source=short,
        eligible_intervals=(
            {
                "eligible": True,
                "interval_start_ms": _START_MS + 30_000,
                "interval_end_ms": _START_MS + 30_000,
            },
        ),
    )

    assert rows == ()


def test_book_flow_does_not_cross_connection_segments() -> None:
    market = _market()
    books = (
        _book(
            market,
            outcome="Up",
            offset_ms=29_000,
            bid="0.40",
            ask="0.42",
            segment="segment-1",
        ),
        _book(
            market,
            outcome="Up",
            offset_ms=30_000,
            bid="0.60",
            ask="0.62",
            segment="segment-2",
        ),
    )

    flow = _book_flow(
        books,
        tuple(book.received_wall_ms for book in books),
        decision_ms=_START_MS + 30_000,
        window_ms=5_000,
    )

    assert flow.update_count == 0
    assert flow.mid_log_return == 0.0


def test_feature_row_rejects_a_future_receipt() -> None:
    with pytest.raises(ValueError, match="feature row differs"):
        Round27FeatureRow.create(
            run_id="round27",
            condition_id="0x" + "7" * 64,
            event_start_ms=_START_MS,
            decision_time_ms=_START_MS + 30_000,
            market_prior_probability=0.5,
            values=(0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES),
            maximum_receipt_wall_ms=_START_MS + 30_001,
            source_chain_sha256="a" * 64,
        )


def test_source_loader_accepts_only_documented_trade_profiles() -> None:
    def trade_event(stream: str, event_type: str, ordinal: int):
        return SimpleNamespace(
            stream=stream,
            event_type=event_type,
            symbol="BTC",
            event={"data": {"p": "100000", "q": "0.5", "m": False}},
            event_sha256=hashlib.sha256(f"trade:{ordinal}".encode()).hexdigest(),
            received_wall_ms=_START_MS + ordinal,
            received_monotonic_ns=ordinal * 1_000_000,
        )

    tick_event = SimpleNamespace(
        stream="polymarket_rtds",
        event_type="crypto_prices_twap_sixty:update",
        symbol="BTC",
        event={
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "timestamp": _START_MS,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": _START_MS,
                "value": 100_000.0,
                "window_s": 60,
                "full_accuracy_value": str(100_000 * 10**18),
            },
        },
        received_wall_ms=_START_MS,
        received_monotonic_ns=1,
    )

    class Store:
        def iter_public_events(self, run_id, **controls):
            assert run_id == "round27"
            assert controls["verified_source"] is False
            yield tick_event
            yield trade_event("binance_spot", "trade", 1)
            yield trade_event("binance_futures", "aggTrade", 2)

    source = load_round27_public_source_series(Store(), run_id="round27")

    assert source.spot.count == 1
    assert source.usdm.count == 1
    assert len(source.twap) == 1
    assert source.target_accessed is False


def test_trade_parser_rejects_profile_drift() -> None:
    event = SimpleNamespace(
        stream="binance_futures",
        event_type="trade",
        symbol="BTC",
        event={"data": {"p": "100000", "q": "1", "m": False}},
        event_sha256="a" * 64,
        received_wall_ms=_START_MS,
        received_monotonic_ns=1,
    )

    with pytest.raises(ValueError, match="identity differs"):
        Round27TradePoint.from_event(event)

from __future__ import annotations

import pytest

from simple_ai_trading.impact_absorption import (
    DepthSequenceGapError,
    DepthSnapshotBridgeError,
    ImpactFeedIntegrityError,
    SynchronizedDepthBook,
    parse_aggregate_trade,
    parse_book_ticker,
    parse_liquidation_snapshot,
    parse_mark_price,
)


def _snapshot() -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{100.0 - index * 0.1:.1f}", "2"] for index in range(25)],
        "asks": [[f"{100.1 + index * 0.1:.1f}", "3"] for index in range(25)],
    }


def _depth(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "e": "depthUpdate",
        "E": 1_001,
        "T": 1_000,
        "s": "BTCUSDT",
        "U": 101,
        "u": 102,
        "pu": 100,
        "b": [["100.0", "4"], ["99.9", "0"]],
        "a": [["100.1", "1"]],
        "st": 1,
        "ps": "BTCUSDT",
    }
    payload.update(changes)
    return payload


def test_depth_book_bridges_snapshot_and_emits_displayed_changes() -> None:
    book = SynchronizedDepthBook("BTCUSDT", "0.1")
    book.initialize(_snapshot())

    update = book.apply(_depth(), receive_time_ns=1_100_000_000)
    assert update.stale is False
    assert update.final_update_id == 102
    assert [
        (item.side, item.added_qty, item.removed_qty) for item in update.changes
    ] == [
        ("bid", 2.0, 0.0),
        ("bid", 0.0, 2.0),
        ("ask", 0.0, 2.0),
    ]
    state = book.state()
    assert state.best_bid == pytest.approx(100.0)
    assert state.best_ask == pytest.approx(100.1)
    assert state.spread_bps > 0.0
    assert len(state.bid_levels) == 20
    assert len(state.ask_levels) == 20
    assert -1.0 <= state.imbalance_20 <= 1.0


def test_depth_book_rejects_missing_bridge_and_subsequent_gap() -> None:
    book = SynchronizedDepthBook("BTCUSDT", "0.1")
    book.initialize(_snapshot())
    with pytest.raises(DepthSnapshotBridgeError):
        book.apply(_depth(U=110, u=111, pu=109), receive_time_ns=1)

    book.apply(_depth(), receive_time_ns=2)
    with pytest.raises(DepthSequenceGapError):
        book.apply(_depth(U=103, u=104, pu=101), receive_time_ns=3)


def test_depth_book_ignores_only_fully_stale_events() -> None:
    book = SynchronizedDepthBook("BTCUSDT", "0.1")
    book.initialize(_snapshot())
    update = book.apply(_depth(U=90, u=100, pu=89), receive_time_ns=1)
    assert update.stale is True
    assert update.changes == ()
    assert book.stale_event_count == 1


def test_depth_book_rejects_off_tick_crossed_and_wrong_product_events() -> None:
    book = SynchronizedDepthBook("BTCUSDT", "0.1")
    with pytest.raises(ImpactFeedIntegrityError, match="tick size"):
        book.initialize(
            {"lastUpdateId": 1, "bids": [["100.05", "1"]], "asks": [["101", "1"]]}
        )

    book.initialize(_snapshot())
    with pytest.raises(ImpactFeedIntegrityError, match="product symbol"):
        book.apply(_depth(ps="ETHUSDT"), receive_time_ns=1)
    with pytest.raises(ImpactFeedIntegrityError, match="crossed"):
        book.apply(_depth(b=[["100.2", "1"]]), receive_time_ns=1)
    assert book.last_update_id == 100
    assert book.bridged is False
    assert book.state().best_bid == pytest.approx(100.0)

    accepted = book.apply(_depth(), receive_time_ns=2)
    assert accepted.final_update_id == 102


def test_current_2026_stream_payloads_parse_with_strict_semantics() -> None:
    ticker = parse_book_ticker(
        {
            "e": "bookTicker",
            "E": 1_010,
            "T": 1_009,
            "s": "BTCUSDT",
            "u": 101,
            "b": "100.0",
            "B": "2",
            "a": "100.1",
            "A": "3",
            "st": 1,
            "ps": "BTCUSDT",
        },
        symbol="BTCUSDT",
        receive_time_ns=2_000,
    )
    assert ticker.bid == pytest.approx(100.0)

    trade = parse_aggregate_trade(
        {
            "e": "aggTrade",
            "E": 1_020,
            "T": 1_019,
            "s": "BTCUSDT",
            "a": 500,
            "p": "100.1",
            "q": "2",
            "nq": "2",
            "f": 700,
            "l": 702,
            "m": False,
            "st": 1,
        },
        symbol="BTCUSDT",
        receive_time_ns=3_000,
    )
    assert trade.aggressive_side == "buy"
    assert trade.quote_notional == pytest.approx(200.2)

    mark = parse_mark_price(
        {
            "e": "markPriceUpdate",
            "E": 1_030,
            "T": 2_000,
            "s": "BTCUSDT",
            "p": "100.0",
            "i": "99.9",
            "P": "0",
            "r": "-0.0001",
            "st": 1,
        },
        symbol="BTCUSDT",
        receive_time_ns=4_000,
    )
    assert mark.funding_rate == pytest.approx(-0.0001)
    assert mark.estimated_settlement_price is None
    assert mark.next_funding_time_ms == 2_000


def test_liquidation_snapshot_is_observation_not_complete_flow() -> None:
    event = parse_liquidation_snapshot(
        {
            "e": "forceOrder",
            "E": 2_000,
            "st": 1,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "o": "LIMIT",
                "f": "IOC",
                "q": "4",
                "p": "99",
                "ap": "98.5",
                "X": "FILLED",
                "l": "1",
                "z": "4",
                "T": 1_999,
            },
        },
        symbol="BTCUSDT",
        receive_time_ns=5_000,
    )
    assert event.side == "SELL"
    assert event.observed_filled_quote == pytest.approx(394.0)


def test_parsers_reject_wrong_stream_type_and_non_boolean_maker() -> None:
    with pytest.raises(ImpactFeedIntegrityError, match="stream type mismatch"):
        parse_book_ticker(
            {
                "e": "bookTicker",
                "E": 2,
                "T": 1,
                "s": "BTCUSDT",
                "u": 1,
                "b": "1",
                "B": "1",
                "a": "2",
                "A": "1",
                "st": 2,
                "ps": "BTCUSDT",
            },
            symbol="BTCUSDT",
            receive_time_ns=1,
        )
    with pytest.raises(ImpactFeedIntegrityError, match="maker flag"):
        parse_aggregate_trade(
            {
                "e": "aggTrade",
                "E": 2,
                "T": 1,
                "s": "BTCUSDT",
                "a": 1,
                "p": "1",
                "q": "1",
                "f": 1,
                "l": 1,
                "m": 1,
                "st": 1,
            },
            symbol="BTCUSDT",
            receive_time_ns=1,
        )


def test_parsers_reject_non_integral_ids_and_missing_symbols() -> None:
    with pytest.raises(ImpactFeedIntegrityError, match="integer"):
        parse_book_ticker(
            {
                "e": "bookTicker",
                "E": 2,
                "T": 1,
                "s": "BTCUSDT",
                "u": 1.5,
                "b": "1",
                "B": "1",
                "a": "2",
                "A": "1",
                "st": 1,
                "ps": "BTCUSDT",
            },
            symbol="BTCUSDT",
            receive_time_ns=1,
        )
    with pytest.raises(ImpactFeedIntegrityError, match="stream symbol"):
        parse_aggregate_trade(
            {
                "e": "aggTrade",
                "E": 2,
                "T": 1,
                "a": 1,
                "p": "1",
                "q": "1",
                "f": 1,
                "l": 1,
                "m": False,
                "st": 1,
            },
            symbol="BTCUSDC",
            receive_time_ns=1,
        )

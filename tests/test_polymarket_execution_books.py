from __future__ import annotations

import hashlib
import json

import pytest

from simple_ai_trading.polymarket_execution_books import (
    build_polymarket_execution_books,
)
from simple_ai_trading.polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
)


CONDITION = "0x" + "7" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
START_MS = 1_800_000_000_000


def _canonical(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _book(token: str, *, offset_ms: int, bid: str, ask: str) -> dict[str, object]:
    return {
        "asks": [{"price": ask, "size": "9"}],
        "asset_id": token,
        "bids": [{"price": bid, "size": "8"}],
        "event_type": "book",
        "hash": hashlib.sha256(f"{token}:{offset_ms}".encode("ascii")).hexdigest(),
        "market": CONDITION,
        "timestamp": str(START_MS + offset_ms),
    }


def _change(token: str, *, offset_ms: int) -> dict[str, object]:
    return {
        "event_type": "price_change",
        "market": CONDITION,
        "price_changes": [
            {
                "asset_id": token,
                "best_ask": "0.51",
                "best_bid": "0.49",
                "hash": hashlib.sha256(f"change:{token}".encode("ascii")).hexdigest(),
                "price": "0.49",
                "side": "BUY",
                "size": "11",
            }
        ],
        "timestamp": str(START_MS + offset_ms),
    }


def _events() -> tuple:
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=2_000)
    payloads = (
        (10, _book(UP_TOKEN, offset_ms=10, bid="0.49", ask="0.51")),
        (20, _book(DOWN_TOKEN, offset_ms=20, bid="0.48", ask="0.52")),
        (30, _change(UP_TOKEN, offset_ms=30)),
    )
    sequence = 0
    for offset_ms, payload in payloads:
        sequence += 1
        builder.add(
            PolymarketClobLaneReceipt(
                lane_id="clob-a",
                connection_id="clob-a:" + "a" * 32,
                sequence_number=sequence,
                received_wall_ms=START_MS + offset_ms + 5,
                received_monotonic_ns=(START_MS + offset_ms + 5) * 1_000_000,
                raw_text=_canonical(payload),
            )
        )
    union, _audit = builder.finish()
    return union


def test_execution_books_reconstruct_exact_receipt_order() -> None:
    books = build_polymarket_execution_books(
        condition_id=CONDITION,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        union_events=_events(),
        admitted_gap_free=True,
    )

    assert [book.asset_id for book in books] == [UP_TOKEN, DOWN_TOKEN, UP_TOKEN]
    assert books[-1].bids[0].quantity == 11
    assert all(book.market_id == CONDITION for book in books)
    assert all(book.gap_free and book.connected for book in books)


def test_execution_books_reject_non_gap_free_or_regressed_receipts() -> None:
    events = _events()
    with pytest.raises(ValueError, match="identity differs"):
        build_polymarket_execution_books(
            condition_id=CONDITION,
            up_token_id=UP_TOKEN,
            down_token_id=DOWN_TOKEN,
            union_events=events,
            admitted_gap_free=False,
        )
    with pytest.raises(ValueError, match="receipt order regressed"):
        build_polymarket_execution_books(
            condition_id=CONDITION,
            up_token_id=UP_TOKEN,
            down_token_id=DOWN_TOKEN,
            union_events=(events[1], events[0]),
            admitted_gap_free=True,
        )

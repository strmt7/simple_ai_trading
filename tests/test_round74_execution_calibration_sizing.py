from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from simple_ai_trading.round74_execution_calibration_sizing import (
    prepare_round74_execution_sizing,
)


def _sources() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    exchange = {
        "schema_version": "round-074-execution-exchange-information-v1",
        "symbol": "BTCUSDT",
        "source_payload_sha256": "1" * 64,
        "symbol_payload": {
            "symbol": "BTCUSDT",
            "pair": "BTCUSDT",
            "contractType": "PERPETUAL",
            "status": "TRADING",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "filters": [
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.01",
                    "maxQty": "10",
                    "stepSize": "0.01",
                },
                {
                    "filterType": "MIN_NOTIONAL",
                    "notional": "5",
                },
            ],
        },
    }
    mark = {
        "schema_version": "round-074-execution-mark-price-v1",
        "symbol": "BTCUSDT",
        "mark_price": "100",
        "source_payload_sha256": "2" * 64,
    }
    book = {
        "schema_version": "round-074-execution-book-state-v1",
        "symbol": "BTCUSDT",
        "update_id": 42,
        "bids": [["99", "1"], ["98", "2"]],
        "asks": [["100", "1"], ["101", "2"]],
        "source_payload_sha256": "3" * 64,
    }
    return exchange, mark, book


def test_sizing_walks_book_and_floors_to_market_quantity_lattice() -> None:
    exchange, mark, book = _sources()

    plan = prepare_round74_execution_sizing(
        symbol="BTCUSDT",
        entry_side="BUY",
        target_quote_notional=Decimal("250"),
        exchange_information=exchange,
        mark_price=mark,
        book=book,
    )

    assert plan.quantity == Decimal("2.48")
    assert plan.reference_quote_notional == Decimal("249.48")
    assert plan.reference_quote_notional <= plan.target_quote_notional
    assert plan.expected_vwap == Decimal("249.48") / Decimal("2.48")
    assert plan.worst_price == Decimal("101")
    assert plan.expected_book_impact_bps > 0
    assert plan.as_dict()["quantity"] == "2.48"


def test_sell_sizing_uses_bid_depth_and_non_positive_impact() -> None:
    exchange, mark, book = _sources()

    plan = prepare_round74_execution_sizing(
        symbol="BTCUSDT",
        entry_side="SELL",
        target_quote_notional=Decimal("200"),
        exchange_information=exchange,
        mark_price=mark,
        book=book,
    )

    assert plan.quantity == Decimal("2.03")
    assert plan.reference_quote_notional == Decimal("199.94")
    assert plan.best_price == Decimal("99")
    assert plan.worst_price == Decimal("98")
    assert plan.expected_book_impact_bps > 0


def test_sizing_rejects_target_below_mark_price_minimum_notional() -> None:
    exchange, mark, book = _sources()
    exchange["symbol_payload"]["filters"][1]["notional"] = "50"

    with pytest.raises(ValueError, match="below minimum notional"):
        prepare_round74_execution_sizing(
            symbol="BTCUSDT",
            entry_side="BUY",
            target_quote_notional=Decimal("49"),
            exchange_information=exchange,
            mark_price=mark,
            book=book,
        )


def test_sizing_rejects_insufficient_captured_depth() -> None:
    exchange, mark, book = _sources()

    with pytest.raises(ValueError, match="exceeds captured book depth"):
        prepare_round74_execution_sizing(
            symbol="BTCUSDT",
            entry_side="BUY",
            target_quote_notional=Decimal("1000"),
            exchange_information=exchange,
            mark_price=mark,
            book=book,
        )


def test_sizing_rejects_target_above_exchange_maximum_quantity() -> None:
    exchange, mark, book = _sources()
    exchange["symbol_payload"]["filters"][0]["maxQty"] = "1"

    with pytest.raises(ValueError, match="maximum market quantity"):
        prepare_round74_execution_sizing(
            symbol="BTCUSDT",
            entry_side="BUY",
            target_quote_notional=Decimal("250"),
            exchange_information=exchange,
            mark_price=mark,
            book=book,
        )


def test_sizing_rejects_quantity_too_coarse_for_shared_notional() -> None:
    exchange, mark, book = _sources()
    market_lot = exchange["symbol_payload"]["filters"][0]
    market_lot["minQty"] = "1"
    market_lot["stepSize"] = "1"

    with pytest.raises(ValueError, match="notional tolerance"):
        prepare_round74_execution_sizing(
            symbol="BTCUSDT",
            entry_side="BUY",
            target_quote_notional=Decimal("150"),
            exchange_information=exchange,
            mark_price=mark,
            book=book,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("symbol_payload", "status"), "BREAK", "contract eligibility"),
        (("symbol_payload", "contractType"), "CURRENT_QUARTER", "contract eligibility"),
        (("mark_price",), "NaN", "mark price differs"),
        (("source_payload_sha256",), "not-a-hash", "exchange information hash"),
    ],
)
def test_sizing_rejects_ineligible_or_malformed_sources(
    path: tuple[str, ...],
    value: str,
    message: str,
) -> None:
    exchange, mark, book = _sources()
    targets = {
        "symbol_payload": exchange["symbol_payload"],
        "mark_price": mark,
        "source_payload_sha256": exchange,
    }
    target = targets[path[0]]
    assert isinstance(target, dict)
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        prepare_round74_execution_sizing(
            symbol="BTCUSDT",
            entry_side="BUY",
            target_quote_notional=Decimal("100"),
            exchange_information=deepcopy(exchange),
            mark_price=deepcopy(mark),
            book=deepcopy(book),
        )

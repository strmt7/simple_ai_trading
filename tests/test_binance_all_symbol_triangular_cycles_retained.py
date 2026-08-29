from __future__ import annotations

from decimal import Decimal

from tools.adjudicate_binance_all_symbol_triangular_cycles_retained import (
    build_cycles,
    evaluate_cycle,
)
from tools.screen_binance_indirect_internal_conversions import build_edges, parse_books


def _symbol(symbol: str, base: str, quote: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
        "orderTypes": ["MARKET"],
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.000001"}],
    }


def test_cycle_is_positive_only_in_the_zero_fee_upper_bound() -> None:
    exchange_info = {
        "symbols": [
            _symbol("AUSDT", "A", "USDT"),
            _symbol("BUSDT", "B", "USDT"),
            _symbol("CUSDT", "C", "USDT"),
            _symbol("AB", "A", "B"),
            _symbol("BC", "B", "C"),
            _symbol("CA", "C", "A"),
        ]
    }
    books = parse_books(
        [
            {"symbol": "AUSDT", "bidPrice": "1", "bidQty": "10000", "askPrice": "1.001", "askQty": "10000"},
            {"symbol": "BUSDT", "bidPrice": "0.5", "bidQty": "10000", "askPrice": "0.501", "askQty": "10000"},
            {"symbol": "CUSDT", "bidPrice": "0.25", "bidQty": "10000", "askPrice": "0.251", "askQty": "10000"},
            {"symbol": "AB", "bidPrice": "2", "bidQty": "10000", "askPrice": "2.001", "askQty": "10000"},
            {"symbol": "BC", "bidPrice": "2", "bidQty": "10000", "askPrice": "2.001", "askQty": "10000"},
            {"symbol": "CA", "bidPrice": "0.2507", "bidQty": "10000", "askPrice": "0.2508", "askQty": "10000"},
        ]
    )
    edges = build_edges(exchange_info)
    cycle = next(
        row
        for row in build_cycles(edges)
        if [edge.source for edge in row] == ["A", "B", "C"]
    )
    zero_fee = evaluate_cycle(
        cycle,
        start_usdt=Decimal("1000"),
        edges=edges,
        books=books,
        fee_rate=Decimal(0),
        stress_bips=Decimal(3),
    )
    vip0 = evaluate_cycle(
        cycle,
        start_usdt=Decimal("1000"),
        edges=edges,
        books=books,
        fee_rate=Decimal("0.001"),
        stress_bips=Decimal(3),
    )
    assert zero_fee is not None and vip0 is not None
    assert zero_fee["capacity_ok"] is True
    assert zero_fee["positive_after_gates"] is True
    assert zero_fee["stressed_profit_bips"] > 0
    assert vip0["positive_after_gates"] is False
    assert vip0["stressed_profit_bips"] < 0

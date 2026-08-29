from __future__ import annotations

from decimal import Decimal

from tools.screen_binance_indirect_internal_conversions import (
    build_edges,
    build_routes,
    evaluate_route,
    parse_books,
)


def _symbol(symbol: str, base: str, quote: str, step: str = "0.001") -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
        "orderTypes": ["LIMIT", "MARKET"],
        "filters": [
            {"filterType": "MARKET_LOT_SIZE", "stepSize": step},
            {"filterType": "LOT_SIZE", "stepSize": step},
        ],
    }


def test_indirect_route_applies_extra_fee_rounding_capacity_and_stress() -> None:
    exchange_info = {
        "symbols": [
            _symbol("AUSDT", "A", "USDT"),
            _symbol("BUSDT", "B", "USDT"),
            _symbol("XUSDT", "X", "USDT"),
            _symbol("AB", "A", "B"),
            _symbol("AX", "A", "X"),
            _symbol("XB", "X", "B"),
        ]
    }
    books = parse_books(
        [
            {"symbol": "AUSDT", "bidPrice": "9.99", "bidQty": "1000", "askPrice": "10.01", "askQty": "1000"},
            {"symbol": "BUSDT", "bidPrice": "0.999", "bidQty": "10000", "askPrice": "1.001", "askQty": "10000"},
            {"symbol": "XUSDT", "bidPrice": "1.999", "bidQty": "10000", "askPrice": "2.001", "askQty": "10000"},
            {"symbol": "AB", "bidPrice": "9.90", "bidQty": "1000", "askPrice": "9.92", "askQty": "1000"},
            {"symbol": "AX", "bidPrice": "5.00", "bidQty": "1000", "askPrice": "5.01", "askQty": "1000"},
            {"symbol": "XB", "bidPrice": "2.00", "bidQty": "1000", "askPrice": "2.01", "askQty": "1000"},
        ]
    )
    edges = build_edges(exchange_info)
    route = next(
        route
        for route in build_routes(edges)
        if route[0].source == "A"
        and route[0].target == "B"
        and route[1].target == "X"
    )
    result = evaluate_route(
        route,
        start_usdt=Decimal("100"),
        edges=edges,
        books=books,
        fee_rate=Decimal("0.001"),
        stress_bips=Decimal("3"),
    )
    assert result is not None
    assert result["route_id"] == "A->X->B_vs_A->B"
    assert result["capacity_ok"] is True
    assert result["savings_bips"] > Decimal("0")
    assert result["stressed_savings_bips"] > Decimal("0")
    assert result["indirect_residual_bips"] <= Decimal("1")
    assert result["positive_after_gates"] is True

from decimal import Decimal

from tools.screen_binance_spot_sor_liquidity_overlay import (
    _evaluate_group,
    _parse_exchange_info,
)


def _symbol(symbol: str, base: str, quote: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "baseAsset": base,
        "quoteAsset": quote,
        "status": "TRADING",
        "isSpotTradingAllowed": True,
        "orderTypes": ["LIMIT", "MARKET"],
    }


def test_exchange_info_selects_only_scoped_valid_sor_groups() -> None:
    payload = {
        "symbols": [
            _symbol("BTCUSDT", "BTC", "USDT"),
            _symbol("BTCUSDC", "BTC", "USDC"),
            _symbol("BNBUSDT", "BNB", "USDT"),
            _symbol("BNBUSDC", "BNB", "USDC"),
        ],
        "sors": [
            {"baseAsset": "BTC", "symbols": ["BTCUSDT", "BTCUSDC"]},
            {"baseAsset": "BNB", "symbols": ["BNBUSDT", "BNBUSDC"]},
        ],
    }
    groups, symbols = _parse_exchange_info(payload, {"BTC", "ETH", "SOL"})
    assert groups == [
        {
            "base_asset": "BTC",
            "symbols": ["BTCUSDT", "BTCUSDC"],
            "quote_assets": ["USDT", "USDC"],
        }
    ]
    assert len(symbols) == 4


def test_sor_top_level_evaluation_weakly_dominates_and_enforces_capacity() -> None:
    group = {"base_asset": "BTC", "symbols": ["BTCUSDT", "BTCUSDC"]}
    books = {
        "BTCUSDT": {
            "bidPrice": Decimal("99"),
            "bidQty": Decimal("20"),
            "askPrice": Decimal("101"),
            "askQty": Decimal("20"),
        },
        "BTCUSDC": {
            "bidPrice": Decimal("100"),
            "bidQty": Decimal("20"),
            "askPrice": Decimal("100"),
            "askQty": Decimal("20"),
        },
    }
    rows = _evaluate_group(group, books, [Decimal("100")], Decimal("1"))
    by_key = {(row["submitted_symbol"], row["side"]): row for row in rows}
    assert Decimal(by_key[("BTCUSDT", "BUY")]["gross_improvement_bips"]) > 1
    assert Decimal(by_key[("BTCUSDT", "SELL")]["gross_improvement_bips"]) > 1
    assert by_key[("BTCUSDC", "BUY")]["gross_improvement_bips"] == "0"
    assert by_key[("BTCUSDC", "SELL")]["gross_improvement_bips"] == "0"

    books["BTCUSDT"]["askQty"] = Decimal("0.1")
    incomplete = _evaluate_group(group, books, [Decimal("100")], Decimal("1"))
    direct_buy = next(
        row
        for row in incomplete
        if row["submitted_symbol"] == "BTCUSDT" and row["side"] == "BUY"
    )
    assert direct_buy["top_level_comparison_complete"] is False
    assert direct_buy["public_candidate"] is False

from __future__ import annotations

from simple_ai_trading.market_universe import assess_symbol_liquidity, select_tradeable_universe
from simple_ai_trading.types import StrategyConfig


class _Client:
    market_type = "spot"

    def __init__(self) -> None:
        self.info = {
            "symbols": [
                {"symbol": "ETHUSDC", "status": "TRADING"},
                {"symbol": "LOWUSDC", "status": "TRADING"},
                {"symbol": "BADUPUSDC", "status": "TRADING"},
            ]
        }

    def get_exchange_info(self):
        return self.info

    def get_ticker_24h(self, symbol: str):
        if symbol == "ETHUSDC":
            return {"quoteVolume": "200000000", "count": "300000"}
        return {"quoteVolume": "1000", "count": "20"}

    def get_book_ticker(self, symbol: str):
        if symbol == "ETHUSDC":
            return {"bidPrice": "3000.00", "askPrice": "3000.30"}
        return {"bidPrice": "1.00", "askPrice": "1.10"}


def test_liquidity_gate_accepts_measured_liquid_symbol() -> None:
    strategy = StrategyConfig()
    result = assess_symbol_liquidity(_Client(), "ETHUSDC", strategy)

    assert result.eligible is True
    assert result.liquidity_score >= strategy.min_liquidity_score
    profile = result.execution_profile(latency_ms=750, liquidity_haircut=0.5)
    assert profile.symbol == "ETHUSDC"
    assert profile.spread_bps > 0


def test_liquidity_gate_rejects_illiquid_and_structurally_dangerous_symbols() -> None:
    strategy = StrategyConfig()
    low = assess_symbol_liquidity(_Client(), "LOWUSDC", strategy)
    dangerous = assess_symbol_liquidity(_Client(), "BADUPUSDC", strategy)

    assert low.eligible is False
    assert "quote_volume_below_threshold" in low.reasons
    assert dangerous.eligible is False
    assert "leveraged_or_inverse_token_pattern" in dangerous.reasons


def test_universe_selection_requires_diversification() -> None:
    strategy = StrategyConfig(min_diversified_assets=2)
    selection = select_tradeable_universe(_Client(), ["ETHUSDC", "LOWUSDC"], strategy)

    assert selection.allowed is False
    assert selection.symbols == ("ETHUSDC",)
    assert len(selection.rejected) == 1

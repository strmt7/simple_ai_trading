from __future__ import annotations

from simple_ai_trading.market_universe import assess_symbol_liquidity, rank_high_liquidity_universe, select_tradeable_universe
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


class _AutoRankClient:
    market_type = "spot"

    def get_exchange_info(self):
        return {
            "symbols": [
                {"symbol": "BTCUSDC", "status": "TRADING"},
                {"symbol": "ETHUSDC", "status": "TRADING"},
                {"symbol": "SOLUSDC", "status": "TRADING"},
                {"symbol": "FDUSDUSDC", "status": "TRADING"},
                {"symbol": "BNBUSDC", "status": "TRADING"},
                {"symbol": "MICROUSDC", "status": "TRADING"},
                {"symbol": "BADUPUSDC", "status": "TRADING"},
            ]
        }

    def get_all_tickers_24h(self):
        return [
            {"symbol": "BTCUSDC", "quoteVolume": "22000000", "count": "20000"},
            {"symbol": "ETHUSDC", "quoteVolume": "16000000", "count": "22000"},
            {"symbol": "SOLUSDC", "quoteVolume": "14000000", "count": "7000"},
            {"symbol": "FDUSDUSDC", "quoteVolume": "30000000", "count": "30000", "lastPrice": "1.0001", "highPrice": "1.0008", "lowPrice": "0.9994"},
            {"symbol": "BNBUSDC", "quoteVolume": "2000000", "count": "2500"},
            {"symbol": "MICROUSDC", "quoteVolume": "90000", "count": "300"},
            {"symbol": "BADUPUSDC", "quoteVolume": "40000000", "count": "90000"},
        ]

    def get_all_book_tickers(self):
        return [
            {"symbol": "BTCUSDC", "bidPrice": "100.00", "askPrice": "100.01"},
            {"symbol": "ETHUSDC", "bidPrice": "90.00", "askPrice": "90.01"},
            {"symbol": "SOLUSDC", "bidPrice": "80.00", "askPrice": "80.02"},
            {"symbol": "FDUSDUSDC", "bidPrice": "1.0000", "askPrice": "1.0001"},
            {"symbol": "BNBUSDC", "bidPrice": "70.00", "askPrice": "70.02"},
            {"symbol": "MICROUSDC", "bidPrice": "1.00", "askPrice": "1.10"},
            {"symbol": "BADUPUSDC", "bidPrice": "50.00", "askPrice": "50.01"},
        ]


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


def test_auto_rank_uses_relative_liquidity_without_admitting_illiquid_pairs() -> None:
    strategy = StrategyConfig(min_diversified_assets=3)
    selection = rank_high_liquidity_universe(_AutoRankClient(), strategy, max_symbols=5, max_scan=10)

    assert selection.allowed is True
    assert selection.symbols == ("BTCUSDC", "ETHUSDC", "SOLUSDC")
    rejected = {item.symbol: item for item in selection.rejected}
    assert "stable_or_pegged_pair_pattern" in rejected["FDUSDUSDC"].reasons
    assert "quote_volume_below_threshold" in rejected["BNBUSDC"].reasons
    assert "quote_volume_below_threshold" in rejected["MICROUSDC"].reasons
    assert "leveraged_or_inverse_token_pattern" in rejected["BADUPUSDC"].reasons

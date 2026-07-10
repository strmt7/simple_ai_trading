"""Market-aware commission assumptions and authenticated verification."""

from __future__ import annotations

from dataclasses import dataclass

from .api import BinanceClient, CommissionRates
from .types import (
    DEFAULT_FUTURES_TAKER_FEE_BPS,
    DEFAULT_SPOT_TAKER_FEE_BPS,
    StrategyConfig,
)


@dataclass(frozen=True)
class CommissionAssumption:
    market_type: str
    symbol: str
    configured_taker_fee_bps: float
    exchange_taker_fee_bps: float | None
    modeled_taker_fee_bps: float
    source: str
    verified: bool

    def asdict(self) -> dict[str, object]:
        return {
            "market_type": self.market_type,
            "symbol": self.symbol,
            "configured_taker_fee_bps": self.configured_taker_fee_bps,
            "exchange_taker_fee_bps": self.exchange_taker_fee_bps,
            "modeled_taker_fee_bps": self.modeled_taker_fee_bps,
            "source": self.source,
            "verified": self.verified,
        }


def documented_taker_fee_floor_bps(market_type: str) -> float:
    normalized = str(market_type or "").strip().lower()
    if normalized == "futures":
        return DEFAULT_FUTURES_TAKER_FEE_BPS
    if normalized == "spot":
        return DEFAULT_SPOT_TAKER_FEE_BPS
    raise ValueError("market_type must be 'spot' or 'futures'")


def _strategy_with_taker_fee(strategy: StrategyConfig, taker_fee_bps: float) -> StrategyConfig:
    return StrategyConfig(**{**strategy.asdict(), "taker_fee_bps": max(0.0, float(taker_fee_bps))})


def apply_offline_commission_floor(
    strategy: StrategyConfig,
    *,
    market_type: str,
    symbol: str = "",
) -> tuple[StrategyConfig, CommissionAssumption]:
    configured = max(0.0, float(strategy.taker_fee_bps))
    documented_floor = documented_taker_fee_floor_bps(market_type)
    modeled = max(configured, documented_floor)
    source = "configured_stress_fee" if configured > documented_floor else "binance_documented_fee_floor"
    assumption = CommissionAssumption(
        market_type=str(market_type).strip().lower(),
        symbol=str(symbol).strip().upper(),
        configured_taker_fee_bps=configured,
        exchange_taker_fee_bps=None,
        modeled_taker_fee_bps=modeled,
        source=source,
        verified=False,
    )
    return _strategy_with_taker_fee(strategy, modeled), assumption


def apply_verified_commission_rate(
    strategy: StrategyConfig,
    *,
    client: BinanceClient,
    symbol: str,
) -> tuple[StrategyConfig, CommissionAssumption, CommissionRates]:
    rates = client.get_commission_rates(symbol)
    configured = max(0.0, float(strategy.taker_fee_bps))
    exchange_bps = max(0.0, float(rates.taker_bps))
    modeled = max(configured, exchange_bps)
    source = rates.source if modeled == exchange_bps else "configured_stress_fee_above_exchange_rate"
    assumption = CommissionAssumption(
        market_type=rates.market_type,
        symbol=rates.symbol,
        configured_taker_fee_bps=configured,
        exchange_taker_fee_bps=exchange_bps,
        modeled_taker_fee_bps=modeled,
        source=source,
        verified=True,
    )
    return _strategy_with_taker_fee(strategy, modeled), assumption, rates


__all__ = [
    "CommissionAssumption",
    "apply_offline_commission_floor",
    "apply_verified_commission_rate",
    "documented_taker_fee_floor_bps",
]

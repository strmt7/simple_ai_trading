"""Pessimistic execution realism for day-trading simulations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .types import StrategyConfig


@dataclass(frozen=True)
class SymbolExecutionProfile:
    symbol: str
    spread_bps: float
    quote_volume: float
    trade_count: int
    liquidity_score: float
    latency_ms: int
    liquidity_haircut: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionAssumptions:
    spread_bps: float
    latency_ms: int
    liquidity_haircut: float
    impact_coefficient: float
    volatility_buffer_bps: float
    testnet_to_live_buffer_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedFill:
    reference_price: float
    fill_price: float
    side_sign: int
    total_cost_bps: float
    spread_cost_bps: float
    latency_cost_bps: float
    impact_cost_bps: float
    testnet_buffer_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def execution_assumptions_from_strategy(cfg: StrategyConfig) -> ExecutionAssumptions:
    """Build pessimistic assumptions from strategy settings.

    Sources used for the model are documented in docs/LIVE_MARKET_SIMULATION.md:
    bid/ask spread, order-book liquidity, latency drift, fees, and testnet/live
    drift are modeled as additive adverse bps costs unless better L2 replay data
    is available.
    """

    spread = max(float(cfg.slippage_bps), float(cfg.max_spread_bps))
    latency_ms = max(0, int(cfg.latency_buffer_ms))
    liquidity_haircut = min(1.0, max(0.0, float(cfg.testnet_liquidity_haircut)))
    return ExecutionAssumptions(
        spread_bps=spread,
        latency_ms=latency_ms,
        liquidity_haircut=liquidity_haircut,
        impact_coefficient=18.0,
        volatility_buffer_bps=max(1.0, spread * 0.50),
        testnet_to_live_buffer_bps=max(2.0, spread * (1.0 + liquidity_haircut)),
    )


def execution_assumptions_for_symbol(
    cfg: StrategyConfig,
    profile: SymbolExecutionProfile | None = None,
) -> ExecutionAssumptions:
    base = execution_assumptions_from_strategy(cfg)
    if profile is None:
        return base
    liquidity_penalty = 1.0 + max(0.0, 1.0 - profile.liquidity_score)
    spread = max(base.spread_bps, profile.spread_bps * liquidity_penalty)
    haircut = min(1.0, max(base.liquidity_haircut, profile.liquidity_haircut))
    return ExecutionAssumptions(
        spread_bps=spread,
        latency_ms=max(base.latency_ms, int(profile.latency_ms)),
        liquidity_haircut=haircut,
        impact_coefficient=base.impact_coefficient * liquidity_penalty,
        volatility_buffer_bps=max(base.volatility_buffer_bps, spread * 0.50),
        testnet_to_live_buffer_bps=max(base.testnet_to_live_buffer_bps, spread * (1.0 + haircut)),
    )


def _safe_positive(value: float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def simulate_market_fill(
    price: float,
    side_sign: int,
    notional: float,
    cfg: StrategyConfig,
    *,
    bar_volume_notional: float | None = None,
    assumptions: ExecutionAssumptions | None = None,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> SimulatedFill:
    """Return an adverse fill price for backtests and paper simulations."""

    assumptions = assumptions or execution_assumptions_for_symbol(cfg, symbol_profile)
    reference = _safe_positive(price)
    if reference <= 0:
        return SimulatedFill(reference, 0.0, side_sign, 0.0, 0.0, 0.0, 0.0, 0.0)
    direction = 1 if side_sign >= 0 else -1
    bar_volume = _safe_positive(bar_volume_notional or 0.0)
    participation = min(1.0, abs(float(notional)) / bar_volume) if bar_volume > 0 else 1.0
    impact_cost = assumptions.impact_coefficient * (participation ** 0.5)
    latency_cost = assumptions.volatility_buffer_bps * min(10.0, assumptions.latency_ms / 1000.0)
    spread_cost = assumptions.spread_bps / 2.0
    total = spread_cost + latency_cost + impact_cost + assumptions.testnet_to_live_buffer_bps
    fill = reference * (1.0 + direction * (total / 10_000.0))
    return SimulatedFill(
        reference_price=reference,
        fill_price=fill,
        side_sign=direction,
        total_cost_bps=total,
        spread_cost_bps=spread_cost,
        latency_cost_bps=latency_cost,
        impact_cost_bps=impact_cost,
        testnet_buffer_bps=assumptions.testnet_to_live_buffer_bps,
    )

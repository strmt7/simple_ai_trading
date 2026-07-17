"""Pessimistic execution realism for day-trading simulations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .types import StrategyConfig


EXECUTION_MODEL_VERSION = "causal-adv-square-root-v2"
EXECUTION_ACTIVITY_ESTIMATOR = "causal-trailing-24h-volume; annualized-after-5m; no-future-data"


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


def configured_round_trip_cost_floor_bps(cfg: StrategyConfig) -> float:
    """Return the deterministic fee-plus-spread hurdle for one round trip.

    Dynamic latency, impact, and testnet-to-live stress remain additional
    backtest costs. This floor is the minimum gross move a training label must
    clear before a trade can have positive net value under configured costs.
    """

    assumptions = execution_assumptions_from_strategy(cfg)
    taker_fee_bps = max(0.0, float(cfg.taker_fee_bps))
    return float(2.0 * taker_fee_bps + max(0.0, assumptions.spread_bps))


def execution_assumptions_for_symbol(
    cfg: StrategyConfig,
    profile: SymbolExecutionProfile | None = None,
) -> ExecutionAssumptions:
    base = execution_assumptions_from_strategy(cfg)
    if profile is None:
        return base
    liquidity_score = min(1.0, max(0.0, float(profile.liquidity_score)))
    liquidity_penalty = 1.0 + max(0.0, 1.0 - liquidity_score)
    observed_spread = max(0.0, float(profile.spread_bps)) * liquidity_penalty
    configured_spread = max(0.0, float(base.spread_bps))
    # If a real symbol profile exists, use it.  The generic strategy spread is a
    # fallback floor for unknown/weak liquidity, not a hard floor over highly
    # liquid BTC/ETH/SOL books with observed sub-bps spreads.
    liquidity_floor = configured_spread * max(0.0, 1.0 - liquidity_score)
    microstructure_floor = min(configured_spread, 0.50)
    spread = max(observed_spread, liquidity_floor, microstructure_floor)
    haircut = min(1.0, max(base.liquidity_haircut, profile.liquidity_haircut))
    live_buffer = max(
        2.0,
        spread * (1.0 + haircut),
        base.testnet_to_live_buffer_bps * max(0.0, 1.0 - liquidity_score),
    )
    return ExecutionAssumptions(
        spread_bps=spread,
        latency_ms=max(base.latency_ms, int(profile.latency_ms)),
        liquidity_haircut=haircut,
        impact_coefficient=base.impact_coefficient * liquidity_penalty,
        volatility_buffer_bps=max(base.volatility_buffer_bps, spread * 0.50),
        testnet_to_live_buffer_bps=live_buffer,
    )


def _safe_positive(value: float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _safe_finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def market_row_quote_volume_notional(row: object, price: float | None = None) -> float:
    """Return observed quote notional, falling back to base volume times price."""

    quote_volume = _safe_finite(getattr(row, "quote_volume", 0.0), 0.0)
    if quote_volume > 0.0:
        return float(quote_volume)
    reference_price = _safe_finite(
        getattr(row, "close", 0.0) if price is None else price,
        0.0,
    )
    base_volume = _safe_finite(getattr(row, "volume", 0.0), 0.0)
    return float(max(0.0, base_volume) * max(0.0, reference_price))


def market_row_reported_quote_volume_notional(row: object) -> float:
    """Return only exchange-reported quote notional for a market row."""

    return float(max(0.0, _safe_finite(getattr(row, "quote_volume", 0.0), 0.0)))


def market_row_trailing_quote_volume_24h_estimate(row: object) -> float:
    """Return the causal trailing-24h quote-volume estimate attached to a row."""

    return float(
        max(
            0.0,
            _safe_finite(getattr(row, "trailing_quote_volume_24h_estimate", 0.0), 0.0),
        )
    )


def market_row_trade_count(row: object) -> int:
    return max(0, int(_safe_finite(getattr(row, "trade_count", 0), 0.0)))


def market_row_range_bps(row: object, price: float | None = None) -> float:
    reference_price = max(
        0.0,
        _safe_finite(getattr(row, "close", 0.0) if price is None else price, 0.0),
    )
    high = _safe_finite(getattr(row, "high", reference_price), reference_price)
    low = _safe_finite(getattr(row, "low", reference_price), reference_price)
    upper = max(high, low, reference_price)
    lower = min(high, low, reference_price)
    midpoint = max(1e-12, (upper + lower) / 2.0)
    return float(max(0.0, min(10_000.0, (upper - lower) / midpoint * 10_000.0)))


def market_row_execution_assumptions(
    row: object,
    cfg: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
    include_range: bool = True,
) -> ExecutionAssumptions:
    """Build adverse assumptions from the same row evidence used by replay.

    This remains a candle/L1 proxy rather than an order-book queue simulator.
    Keeping it in the execution layer prevents training labels and backtests
    from silently assigning different spread, volatility, and impact costs.
    """

    price = max(0.0, _safe_finite(getattr(row, "close", 0.0), 0.0))
    base = execution_assumptions_for_symbol(cfg, symbol_profile)
    has_activity_evidence = (
        _safe_finite(getattr(row, "quote_volume", 0.0), 0.0) > 0.0
        or market_row_trade_count(row) > 0
    )
    range_bps = (
        market_row_range_bps(row, price)
        if include_range and has_activity_evidence and price > 0.0
        else 0.0
    )
    if range_bps <= 0.0 and not has_activity_evidence:
        return base

    trade_count = market_row_trade_count(row)
    if not has_activity_evidence:
        sparse_trade_multiplier = 1.0
    elif trade_count <= 0:
        sparse_trade_multiplier = 2.0
    else:
        sparse_trade_multiplier = 1.0 + min(1.0, 1.0 / math.sqrt(float(trade_count)))

    spread_bps = max(base.spread_bps, min(250.0, range_bps * 0.12))
    volatility_buffer_bps = max(base.volatility_buffer_bps, min(500.0, range_bps * 0.35))
    liquidity_haircut = max(
        base.liquidity_haircut,
        min(1.0, base.liquidity_haircut + max(0.0, sparse_trade_multiplier - 1.0) * 0.25),
    )
    impact_coefficient = max(base.impact_coefficient, base.impact_coefficient * sparse_trade_multiplier)
    testnet_to_live_buffer_bps = max(
        base.testnet_to_live_buffer_bps,
        spread_bps * (1.0 + liquidity_haircut),
    )
    return ExecutionAssumptions(
        spread_bps=float(spread_bps),
        latency_ms=int(base.latency_ms),
        liquidity_haircut=float(liquidity_haircut),
        impact_coefficient=float(impact_coefficient),
        volatility_buffer_bps=float(volatility_buffer_bps),
        testnet_to_live_buffer_bps=float(testnet_to_live_buffer_bps),
    )


def simulate_market_fill(
    price: float,
    side_sign: int,
    notional: float,
    cfg: StrategyConfig,
    *,
    bar_volume_notional: float | None = None,
    daily_volume_notional: float | None = None,
    assumptions: ExecutionAssumptions | None = None,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> SimulatedFill:
    """Return an adverse fill price for backtests and paper simulations."""

    assumptions = assumptions or execution_assumptions_for_symbol(cfg, symbol_profile)
    reference = _safe_positive(price)
    if reference <= 0:
        return SimulatedFill(reference, 0.0, side_sign, 0.0, 0.0, 0.0, 0.0, 0.0)
    direction = 1 if side_sign >= 0 else -1
    daily_volume = _safe_positive(daily_volume_notional or 0.0)
    if daily_volume > 0.0:
        # Square-root impact is conventionally scaled by participation in ADV,
        # not by prints in the exact second. A liquidity haircut keeps this
        # proxy conservative when historical order-book depth is unavailable.
        executable_daily_volume = daily_volume * max(0.05, 1.0 - assumptions.liquidity_haircut)
        participation = min(1.0, abs(float(notional)) / executable_daily_volume)
    else:
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

"""Portfolio-level risk gates for model-lab acceptance."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .api import Candle
from .types import StrategyConfig


@dataclass(frozen=True)
class PortfolioRiskPolicy:
    min_symbols: int
    min_observations: int
    max_pairwise_correlation: float
    max_cluster_weight: float
    max_portfolio_cvar_95: float
    max_portfolio_drawdown: float
    max_symbol_weight: float
    min_effective_symbols: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolRiskMetrics:
    symbol: str
    observations: int
    mean_return: float
    volatility: float
    downside_volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    weight: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioRiskReport:
    accepted: bool
    reason: str | None
    preliminary_symbols: list[str]
    accepted_symbols: list[str]
    observations: int
    deployed_weight: float
    reserve_weight: float
    effective_symbol_count: float
    max_pairwise_correlation: float
    weighted_average_correlation: float
    portfolio_var_95: float
    portfolio_cvar_95: float
    portfolio_max_drawdown: float
    cluster_count: int
    max_cluster_weight: float
    clusters: list[list[str]]
    weights: dict[str, float]
    metrics: list[SymbolRiskMetrics] = field(default_factory=list)
    policy: PortfolioRiskPolicy | None = None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = [item.asdict() for item in self.metrics]
        payload["policy"] = self.policy.asdict() if self.policy is not None else None
        return payload


def policy_for_strategy(strategy: StrategyConfig, *, min_symbols: int | None = None) -> PortfolioRiskPolicy:
    risk_level = str(strategy.risk_level or "conservative").lower()
    configured_min = max(1, int(min_symbols if min_symbols is not None else strategy.min_diversified_assets))
    if risk_level == "aggressive":
        max_corr = 0.97
        max_cluster = 0.85
        cvar_multiplier = 2.50
        min_effective = max(1.0, configured_min * 0.60)
    elif risk_level == "regular":
        max_corr = 0.93
        max_cluster = 0.70
        cvar_multiplier = 1.75
        min_effective = max(1.0, configured_min * 0.75)
    else:
        max_corr = 0.90
        max_cluster = 0.55
        cvar_multiplier = 1.25
        min_effective = float(configured_min)
    return PortfolioRiskPolicy(
        min_symbols=configured_min,
        min_observations=40,
        max_pairwise_correlation=max_corr,
        max_cluster_weight=max_cluster,
        max_portfolio_cvar_95=max(0.001, float(strategy.max_portfolio_risk_pct) * cvar_multiplier),
        max_portfolio_drawdown=max(0.001, float(strategy.max_drawdown_limit)),
        max_symbol_weight=min(1.0, max(0.01, float(strategy.max_asset_allocation_pct))),
        min_effective_symbols=min_effective,
    )


def _finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(value) else default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(percentile * len(ordered)) - 1)))
    return ordered[index]


def _tail_risk(returns: Sequence[float]) -> tuple[float, float]:
    losses = [-float(value) for value in returns]
    var_95 = max(0.0, _percentile(losses, 0.95))
    tail = [loss for loss in losses if loss >= var_95]
    cvar_95 = max(0.0, _mean(tail or [var_95]))
    return var_95, cvar_95


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= max(0.0, 1.0 + float(value))
        peak = max(peak, equity)
        if peak > 0.0:
            worst = max(worst, (peak - equity) / peak)
    return _finite(worst)


def _returns_by_timestamp(candles: Sequence[Candle]) -> dict[int, float]:
    ordered = sorted(list(candles), key=lambda candle: int(candle.close_time))
    returns: dict[int, float] = {}
    previous_close: float | None = None
    for candle in ordered:
        close = float(candle.close)
        if not math.isfinite(close) or close <= 0.0:
            previous_close = None
            continue
        if previous_close is not None and previous_close > 0.0:
            returns[int(candle.close_time)] = _finite((close - previous_close) / previous_close)
        previous_close = close
    return returns


def _align_returns(candles_by_symbol: Mapping[str, Sequence[Candle]]) -> dict[str, list[float]]:
    series = {
        str(symbol): _returns_by_timestamp(candles)
        for symbol, candles in candles_by_symbol.items()
        if candles
    }
    if not series:
        return {}
    common_times: set[int] | None = None
    for values in series.values():
        keys = set(values)
        common_times = keys if common_times is None else common_times & keys
    if common_times:
        times = sorted(common_times)
        return {symbol: [values[time] for time in times] for symbol, values in series.items()}
    raw = {symbol: list(values.values()) for symbol, values in series.items()}
    length = min((len(values) for values in raw.values()), default=0)
    if length <= 0:
        return {}
    return {symbol: values[-length:] for symbol, values in raw.items()}


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    n = min(len(left), len(right))
    if n < 2:
        return 0.0
    x = list(left)[-n:]
    y = list(right)[-n:]
    mean_x = _mean(x)
    mean_y = _mean(y)
    denom_x = sum((value - mean_x) ** 2 for value in x)
    denom_y = sum((value - mean_y) ** 2 for value in y)
    denom = math.sqrt(denom_x * denom_y)
    if denom <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / denom))


def _inverse_vol_weights(aligned: Mapping[str, Sequence[float]], max_symbol_weight: float) -> dict[str, float]:
    symbols = list(aligned)
    if not symbols:
        return {}
    cap = min(1.0, max(0.0, float(max_symbol_weight)))
    if cap <= 0.0:
        return {symbol: 0.0 for symbol in symbols}
    inv: dict[str, float] = {}
    for symbol in symbols:
        vol = _std(aligned[symbol])
        inv[symbol] = 1.0 / max(vol, 1e-8)
    total = sum(inv.values())
    if total <= 0.0:
        raw = {symbol: 1.0 / len(symbols) for symbol in symbols}
    else:
        raw = {symbol: value / total for symbol, value in inv.items()}
    target_deployed = min(1.0, cap * len(symbols))
    weights = {symbol: 0.0 for symbol in symbols}
    remaining = list(symbols)
    remaining_budget = target_deployed
    while remaining and remaining_budget > 1e-12:
        raw_total = sum(max(0.0, raw[symbol]) for symbol in remaining)
        if raw_total <= 0.0:
            provisional = {symbol: remaining_budget / len(remaining) for symbol in remaining}
        else:
            provisional = {
                symbol: remaining_budget * max(0.0, raw[symbol]) / raw_total
                for symbol in remaining
            }
        capped = [symbol for symbol, weight in provisional.items() if weight > cap + 1e-12]
        if not capped:
            for symbol, weight in provisional.items():
                weights[symbol] = min(cap, max(0.0, weight))
            break
        capped_set = set(capped)
        for symbol in capped:
            weights[symbol] = cap
            remaining_budget = max(0.0, remaining_budget - cap)
        remaining = [symbol for symbol in remaining if symbol not in capped_set]
    return weights


def _normalised_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return {symbol: 0.0 for symbol in weights}
    return {symbol: max(0.0, float(value)) / total for symbol, value in weights.items()}


def _clusters(symbols: Sequence[str], correlations: Mapping[tuple[str, str], float], threshold: float) -> list[list[str]]:
    parent = {symbol: symbol for symbol in symbols}

    def find(symbol: str) -> str:
        while parent[symbol] != symbol:
            parent[symbol] = parent[parent[symbol]]
            symbol = parent[symbol]
        return symbol

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for (left, right), corr in correlations.items():
        if corr >= threshold:
            union(left, right)
    grouped: dict[str, list[str]] = {}
    for symbol in symbols:
        grouped.setdefault(find(symbol), []).append(symbol)
    return [sorted(group) for group in grouped.values()]


def _symbol_metrics(symbol: str, returns: Sequence[float], weight: float) -> SymbolRiskMetrics:
    losses = [value for value in returns if value < 0.0]
    var_95, cvar_95 = _tail_risk(returns)
    return SymbolRiskMetrics(
        symbol=symbol,
        observations=len(returns),
        mean_return=_finite(_mean(returns)),
        volatility=_finite(_std(returns)),
        downside_volatility=_finite(_std(losses)),
        var_95=_finite(var_95),
        cvar_95=_finite(cvar_95),
        max_drawdown=_finite(_max_drawdown(returns)),
        weight=_finite(weight),
    )


def build_portfolio_risk_report(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    strategy: StrategyConfig,
    *,
    min_symbols: int | None = None,
) -> PortfolioRiskReport:
    """Measure combined model-lab candidates before declaring portfolio acceptance."""

    policy = policy_for_strategy(strategy, min_symbols=min_symbols)
    preliminary = sorted(str(symbol) for symbol in candles_by_symbol if candles_by_symbol[symbol])
    if not preliminary:
        return PortfolioRiskReport(
            accepted=False,
            reason="no_preliminary_symbols",
            preliminary_symbols=[],
            accepted_symbols=[],
            observations=0,
            deployed_weight=0.0,
            reserve_weight=1.0,
            effective_symbol_count=0.0,
            max_pairwise_correlation=0.0,
            weighted_average_correlation=0.0,
            portfolio_var_95=0.0,
            portfolio_cvar_95=0.0,
            portfolio_max_drawdown=0.0,
            cluster_count=0,
            max_cluster_weight=0.0,
            clusters=[],
            weights={},
            metrics=[],
            policy=policy,
        )

    aligned = _align_returns({symbol: candles_by_symbol[symbol] for symbol in preliminary})
    observations = min((len(values) for values in aligned.values()), default=0)
    weights = _inverse_vol_weights(aligned, policy.max_symbol_weight)
    relative_weights = _normalised_weights(weights)
    symbols = list(aligned)
    correlations: dict[tuple[str, str], float] = {}
    max_corr = 0.0
    weighted_corr_num = 0.0
    weighted_corr_den = 0.0
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1:]:
            corr = _correlation(aligned[left], aligned[right])
            correlations[(left, right)] = corr
            max_corr = max(max_corr, corr)
            pair_weight = relative_weights.get(left, 0.0) * relative_weights.get(right, 0.0)
            weighted_corr_num += corr * pair_weight
            weighted_corr_den += pair_weight
    clusters = _clusters(symbols, correlations, policy.max_pairwise_correlation)
    cluster_weights = [
        sum(weights.get(symbol, 0.0) for symbol in cluster)
        for cluster in clusters
    ]
    max_cluster_weight = max(cluster_weights, default=0.0)
    portfolio_returns: list[float] = []
    for index in range(observations):
        portfolio_returns.append(sum(weights.get(symbol, 0.0) * aligned[symbol][index] for symbol in symbols))
    var_95, cvar_95 = _tail_risk(portfolio_returns)
    effective_symbols = 0.0
    weight_square_sum = sum(weight * weight for weight in relative_weights.values())
    if weight_square_sum > 0.0:
        effective_symbols = 1.0 / weight_square_sum
    metrics = [_symbol_metrics(symbol, aligned[symbol], weights.get(symbol, 0.0)) for symbol in symbols]
    reasons: list[str] = []
    if len(symbols) < policy.min_symbols:
        reasons.append(f"symbols<{policy.min_symbols}")
    if observations < policy.min_observations:
        reasons.append(f"observations<{policy.min_observations}")
    if effective_symbols + 1e-3 < policy.min_effective_symbols:
        reasons.append(f"effective_symbols<{policy.min_effective_symbols:.2f}")
    if max_cluster_weight > policy.max_cluster_weight:
        reasons.append(f"cluster_weight>{policy.max_cluster_weight:.2f}")
    if cvar_95 > policy.max_portfolio_cvar_95:
        reasons.append(f"cvar95>{policy.max_portfolio_cvar_95:.4f}")
    portfolio_drawdown = _max_drawdown(portfolio_returns)
    if portfolio_drawdown > policy.max_portfolio_drawdown:
        reasons.append(f"drawdown>{policy.max_portfolio_drawdown:.2f}")
    accepted = not reasons
    return PortfolioRiskReport(
        accepted=accepted,
        reason="; ".join(reasons) if reasons else None,
        preliminary_symbols=preliminary,
        accepted_symbols=symbols if accepted else [],
        observations=observations,
        deployed_weight=_finite(sum(weights.values())),
        reserve_weight=_finite(max(0.0, 1.0 - sum(weights.values()))),
        effective_symbol_count=_finite(effective_symbols),
        max_pairwise_correlation=_finite(max_corr),
        weighted_average_correlation=_finite(weighted_corr_num / weighted_corr_den if weighted_corr_den > 0 else 0.0),
        portfolio_var_95=_finite(var_95),
        portfolio_cvar_95=_finite(cvar_95),
        portfolio_max_drawdown=_finite(portfolio_drawdown),
        cluster_count=len(clusters),
        max_cluster_weight=_finite(max_cluster_weight),
        clusters=clusters,
        weights={symbol: _finite(weight) for symbol, weight in weights.items()},
        metrics=metrics,
        policy=policy,
    )

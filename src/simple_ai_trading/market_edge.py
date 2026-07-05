"""Market-edge evidence for day-trading model validation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .backtest import BacktestResult, closed_trades_per_day, risk_gate_skip_count, trade_activity_satisfies
from .financial_sanity import blocking_reasons, build_backtest_financial_sanity_report
from .objective import ObjectiveSpec, get_objective


_BOOTSTRAP_SEED = 90210


@dataclass(frozen=True)
class MarketEdgeReport:
    objective: str
    accepted: bool
    reason: str | None
    failed_checks: tuple[str, ...]
    benchmark_name: str
    starting_cash: float
    realized_pnl: float
    benchmark_pnl: float
    net_edge: float
    net_edge_pct: float
    realized_return_pct: float
    benchmark_return_pct: float
    min_net_edge_pct: float | None
    closed_trades: int
    min_closed_trades: int
    trades_per_day: float
    min_trades_per_day: float
    risk_gate_skip_count: int
    activity_gate_risk_explained: bool
    stopped_by_liquidation: bool
    liquidation_events: int
    liquidation_loss: float
    profit_factor: float
    min_profit_factor: float | None
    expectancy: float
    min_expectancy: float | None
    evidence_unit: str
    sample_count: int
    min_sample_count: int
    trade_return_count: int
    trade_pnl_count: int
    positive_sample_count: int
    positive_sample_rate: float
    mean_sample_return: float
    median_sample_return: float
    sample_return_stdev: float
    downside_sample_stdev: float
    downside_return_risk_ratio: float
    min_downside_return_risk_ratio: float
    sign_test_p_value: float
    max_sign_test_p_value: float
    bootstrap_confidence: float
    bootstrap_samples: int
    bootstrap_lower_mean_return: float
    min_bootstrap_lower_mean_return: float
    financial_sanity_allowed: bool
    financial_sanity_block_count: int
    financial_sanity_blocking_reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clean_numbers(values: object) -> list[float]:
    if not isinstance(values, (tuple, list)):
        return []
    cleaned: list[float] = []
    for value in values:
        parsed = _finite(value, default=float("nan"))
        if math.isfinite(parsed):
            cleaned.append(parsed)
    return cleaned


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _downside_stdev(values: list[float]) -> float:
    if not values:
        return 0.0
    downside_squares = [min(0.0, value) ** 2 for value in values]
    return math.sqrt(max(0.0, sum(downside_squares) / len(values)))


def _downside_return_risk_ratio(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_return = sum(values) / len(values)
    downside = _downside_stdev(values)
    if downside <= 0.0:
        return downside, 999.0 if mean_return > 0.0 else 0.0
    return downside, mean_return / downside


def _binomial_upper_tail(trials: int, successes: int) -> float:
    if trials <= 0 or successes <= 0:
        return 1.0
    probability = 0.0
    for k in range(successes, trials + 1):
        probability += math.comb(trials, k) * (0.5 ** trials)
    return min(1.0, max(0.0, probability))


def _bootstrap_lower_mean_return(values: list[float], *, confidence: float, samples: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rng = random.Random(_BOOTSTRAP_SEED + len(values))
    means: list[float] = []
    sample_count = max(1, int(samples))
    for _ in range(sample_count):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    return _quantile(means, 1.0 - max(0.0, min(1.0, confidence)))


def _max_sign_test_p_value(objective: ObjectiveSpec) -> float:
    if objective.name == "conservative":
        return 0.30
    if objective.name == "aggressive":
        return 0.40
    return 0.35


def _min_downside_return_risk_ratio(objective: ObjectiveSpec) -> float:
    if objective.name == "conservative":
        return 0.45
    if objective.name == "aggressive":
        return 0.20
    return 0.30


def _sample_returns(result: BacktestResult, *, cash: float) -> tuple[str, list[float], int, int]:
    trade_returns = _clean_numbers(getattr(result, "trade_returns", ()))
    trade_pnls = _clean_numbers(getattr(result, "trade_pnls", ()))
    if trade_returns:
        return "trade_return", trade_returns, len(trade_returns), len(trade_pnls)
    if trade_pnls:
        return "trade_pnl_pct_of_capital", [value / cash for value in trade_pnls], 0, len(trade_pnls)
    return "none", [], 0, 0


def build_market_edge_report(
    result: BacktestResult,
    objective: str | ObjectiveSpec,
    *,
    benchmark_name: str = "same_symbol_buy_hold_after_costs",
    bootstrap_confidence: float = 0.90,
    bootstrap_samples: int = 512,
    min_bootstrap_lower_mean_return: float = 0.0,
) -> MarketEdgeReport:
    """Build auditable evidence that the strategy beat a passive market baseline."""

    spec = get_objective(objective) if isinstance(objective, str) else objective
    cash = max(1.0, abs(_finite(getattr(result, "starting_cash", 0.0), 0.0)))
    realized_pnl = _finite(getattr(result, "realized_pnl", 0.0))
    benchmark_pnl = _finite(getattr(result, "buy_hold_pnl", 0.0))
    net_edge = _finite(getattr(result, "edge_vs_buy_hold", realized_pnl - benchmark_pnl))
    net_edge_pct = net_edge / cash
    closed_trades = int(max(0.0, _finite(getattr(result, "closed_trades", 0), 0.0)))
    stopped_by_liquidation = bool(getattr(result, "stopped_by_liquidation", False))
    liquidation_events = int(max(0.0, _finite(getattr(result, "liquidation_events", 0), 0.0)))
    liquidation_loss = max(0.0, _finite(getattr(result, "liquidation_loss", 0.0)))
    min_closed_trades = max(0, int(spec.min_closed_trades))
    trades_per_day = closed_trades_per_day(result)
    skip_count = risk_gate_skip_count(result)
    activity_ok = trade_activity_satisfies(
        result,
        min_closed_trades=min_closed_trades,
        min_trades_per_day=float(spec.min_trades_per_day),
    )
    activity_gate_risk_explained = bool(
        activity_ok
        and (
            closed_trades < min_closed_trades
            or (
                float(spec.min_trades_per_day) > 0.0
                and trades_per_day < float(spec.min_trades_per_day)
            )
        )
        and skip_count > 0
    )
    min_sample_count = 3 if activity_gate_risk_explained else max(3, min_closed_trades)
    evidence_unit, samples, trade_return_count, trade_pnl_count = _sample_returns(result, cash=cash)
    positive_count = sum(1 for value in samples if value > 0.0)
    mean_return = sum(samples) / len(samples) if samples else 0.0
    median_return = _quantile(samples, 0.5)
    downside_stdev, downside_risk_ratio = _downside_return_risk_ratio(samples)
    min_downside_risk_ratio = _min_downside_return_risk_ratio(spec)
    lower_mean = _bootstrap_lower_mean_return(
        samples,
        confidence=bootstrap_confidence,
        samples=bootstrap_samples,
    )
    max_sign_p = _max_sign_test_p_value(spec)
    sign_p = _binomial_upper_tail(len(samples), positive_count)
    profit_factor = _finite(getattr(result, "profit_factor", 0.0))
    expectancy = _finite(getattr(result, "expectancy", 0.0))
    financial_sanity = build_backtest_financial_sanity_report(result)
    financial_sanity_blocks = tuple(blocking_reasons(financial_sanity)[:8])

    failed: list[str] = []
    if financial_sanity_blocks:
        failed.append("financial_sanity_failed")
    if stopped_by_liquidation or liquidation_events > 0:
        failed.append("liquidation_events>0")
    if realized_pnl <= 0.0:
        failed.append("realized_pnl<=0.0")
    if spec.min_market_edge_pct is not None and net_edge_pct < float(spec.min_market_edge_pct):
        failed.append(f"net_edge_pct<{float(spec.min_market_edge_pct):.6f}")
    if not activity_ok and closed_trades < min_closed_trades:
        failed.append(f"closed_trades<{min_closed_trades}")
    elif not activity_ok and float(spec.min_trades_per_day) > 0.0:
        failed.append(f"trades_per_day<{float(spec.min_trades_per_day):.6f}")
    if len(samples) < min_sample_count:
        failed.append(f"sample_count<{min_sample_count}")
    if samples and sign_p > max_sign_p:
        failed.append(f"sign_test_p_value>{max_sign_p:.4f}")
    if samples and lower_mean < min_bootstrap_lower_mean_return:
        failed.append(f"bootstrap_lower_mean_return<{min_bootstrap_lower_mean_return:.4f}")
    if samples and downside_risk_ratio < min_downside_risk_ratio:
        failed.append(f"downside_return_risk_ratio<{min_downside_risk_ratio:.4f}")
    if spec.min_profit_factor is not None and profit_factor > 0.0 and profit_factor < float(spec.min_profit_factor):
        failed.append(f"profit_factor<{float(spec.min_profit_factor):.6f}")
    if spec.min_expectancy is not None and expectancy != 0.0 and expectancy <= float(spec.min_expectancy):
        failed.append(f"expectancy<={float(spec.min_expectancy):.6f}")

    failed_checks = tuple(failed)
    return MarketEdgeReport(
        objective=spec.name,
        accepted=not failed_checks,
        reason="; ".join(failed_checks) if failed_checks else None,
        failed_checks=failed_checks,
        benchmark_name=benchmark_name,
        starting_cash=float(cash),
        realized_pnl=float(realized_pnl),
        benchmark_pnl=float(benchmark_pnl),
        net_edge=float(net_edge),
        net_edge_pct=float(net_edge_pct),
        realized_return_pct=float(realized_pnl / cash),
        benchmark_return_pct=float(benchmark_pnl / cash),
        min_net_edge_pct=float(spec.min_market_edge_pct) if spec.min_market_edge_pct is not None else None,
        closed_trades=closed_trades,
        min_closed_trades=min_closed_trades,
        trades_per_day=float(trades_per_day),
        min_trades_per_day=float(spec.min_trades_per_day),
        risk_gate_skip_count=int(skip_count),
        activity_gate_risk_explained=bool(activity_gate_risk_explained),
        stopped_by_liquidation=bool(stopped_by_liquidation),
        liquidation_events=int(liquidation_events),
        liquidation_loss=float(liquidation_loss),
        profit_factor=float(profit_factor),
        min_profit_factor=float(spec.min_profit_factor) if spec.min_profit_factor is not None else None,
        expectancy=float(expectancy),
        min_expectancy=float(spec.min_expectancy) if spec.min_expectancy is not None else None,
        evidence_unit=evidence_unit,
        sample_count=len(samples),
        min_sample_count=min_sample_count,
        trade_return_count=trade_return_count,
        trade_pnl_count=trade_pnl_count,
        positive_sample_count=positive_count,
        positive_sample_rate=(positive_count / len(samples) if samples else 0.0),
        mean_sample_return=float(mean_return),
        median_sample_return=float(median_return),
        sample_return_stdev=float(_stdev(samples)),
        downside_sample_stdev=float(downside_stdev),
        downside_return_risk_ratio=float(downside_risk_ratio),
        min_downside_return_risk_ratio=float(min_downside_risk_ratio),
        sign_test_p_value=float(sign_p),
        max_sign_test_p_value=float(max_sign_p),
        bootstrap_confidence=float(bootstrap_confidence),
        bootstrap_samples=max(1, int(bootstrap_samples)),
        bootstrap_lower_mean_return=float(lower_mean),
        min_bootstrap_lower_mean_return=float(min_bootstrap_lower_mean_return),
        financial_sanity_allowed=financial_sanity.allowed,
        financial_sanity_block_count=financial_sanity.block_count,
        financial_sanity_blocking_reasons=financial_sanity_blocks,
    )


__all__ = ["MarketEdgeReport", "build_market_edge_report"]

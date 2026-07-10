"""Backtesting engine for autonomous day-trading strategies."""

from __future__ import annotations
import math
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Mapping, Sequence

from .assets import MAX_AUTONOMOUS_LEVERAGE
from .compute import BackendInfo, resolve_backend
from .execution_simulation import (
    ExecutionAssumptions,
    SimulatedFill,
    SymbolExecutionProfile,
    market_row_execution_assumptions,
    market_row_quote_volume_notional,
    market_row_reported_quote_volume_notional,
    market_row_trailing_quote_volume_24h_estimate,
    simulate_market_fill,
)
from .features import ModelRow
from .financial_sanity import blocking_reasons, build_backtest_financial_sanity_report
from .liquidity_session import LiquiditySessionAdjustment, apply_liquidity_session_meta, liquidity_session_adjustment
from .meta_label import MetaLabelDecision, apply_meta_label_policy
from .position_lifecycle import evaluate_position_exit
from .model import (
    MAX_SERIALIZED_LIGHTGBM_DEPTH,
    MAX_SERIALIZED_LIGHTGBM_NODES,
    MAX_SERIALIZED_LIGHTGBM_TREES,
    TrainedModel,
    confidence_adjusted_probability,
    effective_training_backend_name,
    market_direction_from_probability,
    model_direction_thresholds,
    model_decision_threshold,
)
from .regime import classify_market_regime
from .risk_controls import (
    market_regime_unpredictability,
    regime_unpredictability_requires_cooldown,
    stop_loss_sized_notional_pct,
)
from .trade_tape_features import TRADE_TAPE_FEATURES_PER_WINDOW
from .types import StrategyConfig


@dataclass
class BacktestResult:
    starting_cash: float
    ending_cash: float
    realized_pnl: float
    win_rate: float
    trades: int
    max_drawdown: float
    closed_trades: int
    gross_exposure: float
    total_fees: float
    stopped_by_drawdown: bool
    max_exposure: float
    trades_per_day_cap_hit: int
    buy_hold_pnl: float = 0.0
    edge_vs_buy_hold: float = 0.0
    scoring_backend_requested: str = "cpu"
    scoring_backend_kind: str = "cpu"
    scoring_backend_device: str = "cpu"
    scoring_backend_reason: str = ""
    equity_curve: tuple[dict[str, float | int], ...] = ()
    trade_pnls: tuple[float, ...] = ()
    trade_returns: tuple[float, ...] = ()
    trade_log: tuple[dict[str, object], ...] = ()
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_trade_return: float = 0.0
    trade_return_stdev: float = 0.0
    max_consecutive_losses: int = 0
    meta_label_skips: int = 0
    meta_label_downsizes: int = 0
    regime_entry_skips: int = 0
    regime_entry_downsizes: int = 0
    stopped_by_liquidation: bool = False
    liquidation_events: int = 0
    liquidation_loss: float = 0.0


@dataclass(frozen=True)
class ThresholdBacktestCalibration:
    threshold: float
    accepted: bool
    score: float
    realized_pnl: float
    total_fees: float
    max_drawdown: float
    win_rate: float
    closed_trades: int
    edge_vs_buy_hold: float
    baseline_threshold: float
    baseline_score: float
    baseline_realized_pnl: float
    baseline_closed_trades: int
    best_threshold: float
    best_score: float
    best_realized_pnl: float
    best_total_fees: float
    best_max_drawdown: float
    best_win_rate: float
    best_closed_trades: int
    best_edge_vs_buy_hold: float
    evaluated_thresholds: int
    rows: int
    stopped_by_liquidation: bool = False
    liquidation_events: int = 0
    liquidation_loss: float = 0.0
    best_stopped_by_liquidation: bool = False
    best_liquidation_events: int = 0
    best_liquidation_loss: float = 0.0
    scoring_backend_requested: str = "cpu"
    scoring_backend_kind: str = "cpu"
    scoring_backend_device: str = "cpu"
    scoring_backend_reason: str = ""
    min_closed_trades: int = 1
    min_trades_per_day: float = 0.0
    trades_per_day: float = 0.0
    best_trades_per_day: float = 0.0
    long_threshold: float | None = None
    short_threshold: float | None = None
    best_long_threshold: float | None = None
    best_short_threshold: float | None = None

    def asdict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def _finite_float(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    parsed = _finite_float(value, default)
    return low if parsed < low else (high if parsed > high else parsed)


def row_span_days(rows: Sequence[ModelRow]) -> float:
    """Return the chronological span of model rows in calendar days."""

    timestamps = [int(row.timestamp) for row in rows]
    if len(timestamps) < 2:
        return 1.0
    span_ms = max(timestamps) - min(timestamps)
    return max(1.0, float(span_ms) / 86_400_000.0)


def backtest_result_duration_days(result: object) -> float:
    """Return the observed result span in days from equity/trade timestamps."""

    timestamps: list[int] = []
    equity_curve = getattr(result, "equity_curve", ())
    if isinstance(equity_curve, (tuple, list)):
        for point in equity_curve:
            if isinstance(point, dict) and "timestamp" in point:
                timestamps.append(int(_finite_float(point.get("timestamp"), 0.0)))
    trade_log = getattr(result, "trade_log", ())
    if isinstance(trade_log, (tuple, list)):
        for trade in trade_log:
            if isinstance(trade, dict):
                if "opened_at" in trade:
                    timestamps.append(int(_finite_float(trade.get("opened_at"), 0.0)))
                if "closed_at" in trade:
                    timestamps.append(int(_finite_float(trade.get("closed_at"), 0.0)))
    timestamps = [value for value in timestamps if value > 0]
    if len(timestamps) < 2:
        return 1.0
    span_ms = max(timestamps) - min(timestamps)
    return max(1.0, float(span_ms) / 86_400_000.0)


def closed_trades_per_day(result: object, *, duration_days: float | None = None) -> float:
    days = duration_days if duration_days is not None else backtest_result_duration_days(result)
    days = max(1.0, _finite_float(days, 1.0))
    return max(0.0, _finite_float(getattr(result, "closed_trades", 0), 0.0)) / days


def risk_gate_skip_count(result: object) -> int:
    """Return entries skipped or downsized by explicit risk/regime gates."""

    return max(0, int(_finite_float(getattr(result, "regime_entry_skips", 0), 0.0))) + max(
        0,
        int(_finite_float(getattr(result, "meta_label_skips", 0), 0.0)),
    )


def risk_gated_activity_explains_shortfall(
    result: object,
    *,
    min_closed_trades: int,
    min_trades_per_day: float,
    duration_days: float | None = None,
) -> bool:
    """Return True when explicit risk gates justify missing activity targets.

    Trade-count targets prove a repeatable day-trading edge, but they must not
    become forced-entry quotas. A sparse result can pass only when the backtest
    records enough regime/meta-label vetoes to explain the missing trades.
    """

    closed_trades = max(0, int(_finite_float(getattr(result, "closed_trades", 0), 0.0)))
    min_closed = max(0, int(min_closed_trades))
    min_daily = max(0.0, _finite_float(min_trades_per_day, 0.0))
    days = max(1.0, _finite_float(duration_days, backtest_result_duration_days(result)))
    required_for_daily = int(math.ceil(min_daily * days)) if min_daily > 0.0 else 0
    required_closed = max(min_closed, required_for_daily)
    missing = max(0, required_closed - closed_trades)
    if missing <= 0:
        return False
    skip_count = risk_gate_skip_count(result)
    return skip_count >= max(5, missing * 2, closed_trades * 2)


def trade_activity_satisfies(
    result: object,
    *,
    min_closed_trades: int,
    min_trades_per_day: float,
    duration_days: float | None = None,
    allow_risk_gated_low_activity: bool = True,
) -> bool:
    """Return whether a result has enough activity without forcing bad trades.

    Sparse trading is only accepted when explicit risk/regime gates explain the
    inactivity. That keeps optimization from promoting one random trade while
    still allowing the bot to sit out genuinely unpredictable markets.
    """

    closed_trades = max(0, int(_finite_float(getattr(result, "closed_trades", 0), 0.0)))
    min_closed = max(0, int(min_closed_trades))
    min_daily = max(0.0, _finite_float(min_trades_per_day, 0.0))
    closed_floor_met = closed_trades >= min_closed
    daily_floor_met = min_daily <= 0.0 or closed_trades_per_day(result, duration_days=duration_days) >= min_daily
    if closed_floor_met and daily_floor_met:
        return True
    if not allow_risk_gated_low_activity:
        return False
    return risk_gated_activity_explains_shortfall(
        result,
        min_closed_trades=min_closed,
        min_trades_per_day=min_daily,
        duration_days=duration_days,
    )


def risk_adjusted_backtest_score(result: object, *, starting_cash: float = 1000.0) -> float:
    realized = _finite_float(getattr(result, "realized_pnl", 0.0))
    total_fees = max(0.0, _finite_float(getattr(result, "total_fees", 0.0)))
    max_drawdown = max(0.0, _finite_float(getattr(result, "max_drawdown", 0.0)))
    closed_trades = int(_finite_float(getattr(result, "closed_trades", 0), 0.0))
    stopped_by_drawdown = bool(getattr(result, "stopped_by_drawdown", False))
    stopped_by_liquidation = bool(getattr(result, "stopped_by_liquidation", False))
    liquidation_events = max(0, int(_finite_float(getattr(result, "liquidation_events", 0), 0.0)))
    cash = max(1.0, _finite_float(starting_cash, 1000.0))

    score = realized - total_fees - (max_drawdown * cash)
    if stopped_by_drawdown:
        score -= cash * 0.5
    if stopped_by_liquidation or liquidation_events > 0:
        score -= cash * max(1.0, float(liquidation_events))
    if closed_trades <= 0:
        score -= cash * 0.05
    return float(score)


def threshold_backtest_selection_score(
    result: object,
    *,
    starting_cash: float = 1000.0,
    min_closed_trades: int = 0,
    min_trades_per_day: float = 0.0,
    duration_days: float | None = None,
) -> float:
    """Score threshold candidates while discouraging one-trade overfit."""

    score = risk_adjusted_backtest_score(result, starting_cash=starting_cash)
    cash = max(1.0, _finite_float(starting_cash, 1000.0))
    closed = max(0, int(_finite_float(getattr(result, "closed_trades", 0), 0.0)))
    days = max(1.0, _finite_float(duration_days, backtest_result_duration_days(result)))
    min_closed = max(0, int(min_closed_trades))
    min_daily = max(0.0, _finite_float(min_trades_per_day, 0.0))
    required_daily = int(math.ceil(min_daily * days)) if min_daily > 0.0 else 0
    required = max(min_closed, required_daily)
    missing = max(0, required - closed)
    if missing > 0 and not risk_gated_activity_explains_shortfall(
        result,
        min_closed_trades=min_closed,
        min_trades_per_day=min_daily,
        duration_days=days,
    ):
        score -= cash * (0.02 + 0.006 * float(missing))
    if required >= 3 and closed == 1:
        score -= cash * 0.035
    realized = _finite_float(getattr(result, "realized_pnl", 0.0), 0.0)
    if closed > 0 and realized <= 0.0:
        score -= cash * 0.015
    profit_factor = _finite_float(getattr(result, "profit_factor", 0.0), 0.0)
    if closed > 0 and 0.0 < profit_factor < 1.0:
        score -= cash * 0.015
    return float(score)


def _bps_to_rate(bps: float) -> float:
    return max(0.0, bps) / 10_000.0


def _fill_price(
    price: float,
    side_sign: int,
    cfg: StrategyConfig,
    *,
    notional: float = 0.0,
    volume: float = 0.0,
    daily_volume: float = 0.0,
    symbol_profile: SymbolExecutionProfile | None = None,
    assumptions: ExecutionAssumptions | None = None,
) -> float:
    return _simulate_fill(
        price,
        side_sign,
        cfg,
        notional=notional,
        volume=volume,
        daily_volume=daily_volume,
        symbol_profile=symbol_profile,
        assumptions=assumptions,
    ).fill_price


def _simulate_fill(
    price: float,
    side_sign: int,
    cfg: StrategyConfig,
    *,
    notional: float = 0.0,
    volume: float = 0.0,
    daily_volume: float = 0.0,
    symbol_profile: SymbolExecutionProfile | None = None,
    assumptions: ExecutionAssumptions | None = None,
) -> SimulatedFill:
    return simulate_market_fill(
        price,
        side_sign,
        notional,
        cfg,
        bar_volume_notional=volume,
        daily_volume_notional=daily_volume,
        symbol_profile=symbol_profile,
        assumptions=assumptions,
    )


def _row_quote_volume_notional(row: ModelRow, price: float) -> float:
    return market_row_quote_volume_notional(row, price)


def _row_reported_quote_volume_notional(row: ModelRow) -> float:
    return market_row_reported_quote_volume_notional(row)


def _row_trailing_quote_volume_24h_estimate(row: ModelRow) -> float:
    return market_row_trailing_quote_volume_24h_estimate(row)


def _row_execution_assumptions(
    row: ModelRow,
    cfg: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
    include_range: bool = True,
) -> ExecutionAssumptions:
    return market_row_execution_assumptions(
        row,
        cfg,
        symbol_profile=symbol_profile,
        include_range=include_range,
    )


def _normalize_market_direction(
    signal_score: float,
    threshold: float | None,
    market_type: str,
    *,
    short_threshold: float | None = None,
) -> int:
    return market_direction_from_probability(
        signal_score,
        threshold,
        market_type=market_type,
        short_threshold=short_threshold,
    )


def _regime_soft_gate_size_multiplier(
    *,
    regime_score: float,
    regime_limit: float,
    signal_score: float,
    direction: int,
    long_threshold: float | None,
    short_threshold: float | None,
    market_type: str,
) -> float:
    """Return a reduced entry size for borderline unpredictable regimes."""

    score = _finite_float(regime_score, float("nan"))
    limit = _finite_float(regime_limit, float("nan"))
    if not math.isfinite(score) or not math.isfinite(limit):
        return 0.0
    score = max(0.0, min(1.0, score))
    limit = max(0.0, min(1.0, limit))
    if score <= limit:
        return 1.0
    if regime_unpredictability_requires_cooldown(score, limit):
        return 0.0
    if direction == 0:
        return 0.0

    if direction > 0:
        threshold = max(0.5, float(long_threshold)) if long_threshold is not None else None
        margin = _finite_float(signal_score, 0.5) - float(threshold if threshold is not None else 1.0)
    elif str(market_type).lower() == "futures":
        threshold = min(0.5, float(short_threshold)) if short_threshold is not None else None
        margin = float(threshold if threshold is not None else 0.0) - _finite_float(signal_score, 0.5)
    else:
        return 0.0
    if threshold is None:
        return 0.0

    severity = (score - limit) / max(1e-9, 1.0 - limit)
    severity = max(0.0, min(1.0, severity))
    required_margin = 0.01 + 0.06 * severity
    if margin < required_margin:
        return 0.0
    return max(0.20, 1.0 - 0.70 * severity)


def _close_position(
    position_side: int,
    price: float,
    entry_price: float,
    qty: float,
    notional: float,
    margin_used: float,
    cfg: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
    fill_volume_notional: float = 0.0,
    daily_volume_notional: float = 0.0,
    execution_assumptions: ExecutionAssumptions | None = None,
) -> tuple[float, float, float, float, float]:
    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    exit_fill = _simulate_fill(
        price,
        -position_side,
        cfg,
        notional=abs(notional),
        volume=fill_volume_notional if fill_volume_notional > 0.0 else abs(notional) * 20.0,
        daily_volume=daily_volume_notional,
        symbol_profile=symbol_profile,
        assumptions=execution_assumptions,
    )
    exit_price = exit_fill.fill_price
    realized = position_side * (exit_price - entry_price) * qty
    exit_fee = abs(exit_price * qty) * fee_rate
    return margin_used + realized - exit_fee, realized, exit_fee, exit_fill.total_cost_bps, exit_price


def _futures_liquidation_state(
    *,
    market_type: str,
    position_side: int,
    price: float,
    entry_price: float,
    qty: float,
    margin_used: float,
    cfg: StrategyConfig,
) -> tuple[bool, float, float]:
    """Return liquidation state using a conservative isolated-margin proxy."""

    if market_type != "futures" or position_side == 0 or qty <= 0.0 or entry_price <= 0.0:
        return False, 0.0, 0.0
    mark = max(0.0, _finite_float(price, 0.0))
    current_notional = abs(mark * qty)
    if current_notional <= 0.0:
        return True, 0.0, 0.0
    unrealized = position_side * (mark - entry_price) * qty
    margin_balance = margin_used + unrealized
    maintenance_margin = current_notional * max(0.0, _finite_float(cfg.liquidation_buffer_pct, 0.0))
    return margin_balance <= maintenance_margin, float(margin_balance), float(maintenance_margin)


def _bar_bounds(row: ModelRow, close: float) -> tuple[float, float]:
    high = _finite_float(getattr(row, "high", close), close)
    low = _finite_float(getattr(row, "low", close), close)
    return max(close, high, low), min(close, high, low)


def _adverse_mark_price(position_side: int, close: float, high: float, low: float) -> float:
    if position_side > 0:
        return low
    if position_side < 0:
        return high
    return close


def _intrabar_exit(
    *,
    position_side: int,
    entry_price: float,
    high: float,
    low: float,
    cfg: StrategyConfig,
) -> tuple[bool, float, str]:
    if position_side == 0 or entry_price <= 0.0:
        return False, entry_price, ""
    stop_pct = max(0.0, _finite_float(cfg.stop_loss_pct, 0.0))
    take_pct = max(0.0, _finite_float(cfg.take_profit_pct, 0.0))
    if position_side > 0:
        stop_price = entry_price * (1.0 - stop_pct)
        take_price = entry_price * (1.0 + take_pct)
        stop_hit = low <= stop_price
        take_hit = high >= take_price
    else:
        stop_price = entry_price * (1.0 + stop_pct)
        take_price = max(0.0, entry_price * (1.0 - take_pct))
        stop_hit = high >= stop_price
        take_hit = low <= take_price
    if stop_hit:
        reason = "intrabar_stop_loss"
        if take_hit:
            reason = "intrabar_stop_loss_ambiguous"
        return True, stop_price, reason
    if take_hit:
        return True, take_price, "intrabar_take_profit"
    return False, entry_price, ""


def _safe_day(ts_ms: int) -> int:
    return int(ts_ms // (24 * 60 * 60 * 1000))


def _equity_point(timestamp: int, equity: float, drawdown: float, position_side: int) -> dict[str, float | int]:
    return {
        "timestamp": int(timestamp),
        "equity": float(equity),
        "drawdown": float(max(0.0, drawdown)),
        "position_side": int(position_side),
    }


def _trade_return(net_pnl: float, equity_reference: float) -> float:
    reference = max(1.0, abs(_finite_float(equity_reference, 1.0)))
    return float(net_pnl) / reference


def _max_consecutive_losses(trade_pnls: list[float]) -> int:
    longest = 0
    current = 0
    for pnl in trade_pnls:
        if pnl < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _path_quality_metrics(
    trade_pnls: list[float],
    trade_returns: list[float],
) -> dict[str, float | int]:
    clean_pnls = [float(value) for value in trade_pnls if math.isfinite(float(value))]
    clean_returns = [float(value) for value in trade_returns if math.isfinite(float(value))]
    gross_profit = sum(value for value in clean_pnls if value > 0.0)
    gross_loss = abs(sum(value for value in clean_pnls if value < 0.0))
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    expectancy = sum(clean_pnls) / len(clean_pnls) if clean_pnls else 0.0
    average_return = sum(clean_returns) / len(clean_returns) if clean_returns else 0.0
    if len(clean_returns) >= 2:
        variance = sum((value - average_return) ** 2 for value in clean_returns) / (len(clean_returns) - 1)
        return_stdev = math.sqrt(max(0.0, variance))
    else:
        return_stdev = 0.0
    return {
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(min(999.0, max(0.0, profit_factor))),
        "expectancy": float(expectancy),
        "average_trade_return": float(average_return),
        "trade_return_stdev": float(return_stdev),
        "max_consecutive_losses": int(_max_consecutive_losses(clean_pnls)),
    }


def _buy_hold_pnl(
    rows: List[ModelRow],
    starting_cash: float,
    cfg: StrategyConfig,
    *,
    market_type: str,
    leverage: float,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    """Return fee/slippage-aware same-notional buy-and-hold baseline P&L."""

    if not rows or starting_cash <= 0:
        return 0.0
    first = rows[0].close
    last = rows[-1].close
    if first <= 0 or last <= 0:
        return 0.0
    baseline_notional = starting_cash * stop_loss_sized_notional_pct(cfg, market_type, leverage=leverage)
    if baseline_notional <= 0.0:
        return 0.0
    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    first_row = rows[0]
    last_row = rows[-1]
    first_volume_notional = _row_reported_quote_volume_notional(first_row)
    last_volume_notional = _row_reported_quote_volume_notional(last_row)
    entry = _fill_price(
        first,
        1,
        cfg,
        notional=baseline_notional,
        volume=first_volume_notional if first_volume_notional > 0.0 else baseline_notional * 20.0,
        daily_volume=_row_trailing_quote_volume_24h_estimate(first_row),
        symbol_profile=symbol_profile,
        assumptions=_row_execution_assumptions(first_row, cfg, symbol_profile=symbol_profile),
    )
    exit_price = _fill_price(
        last,
        -1,
        cfg,
        notional=baseline_notional,
        volume=last_volume_notional if last_volume_notional > 0.0 else baseline_notional * 20.0,
        daily_volume=_row_trailing_quote_volume_24h_estimate(last_row),
        symbol_profile=symbol_profile,
        assumptions=_row_execution_assumptions(last_row, cfg, symbol_profile=symbol_profile),
    )
    if entry <= 0 or exit_price <= 0:
        return 0.0
    qty = baseline_notional / entry
    entry_fee = baseline_notional * fee_rate
    exit_notional = qty * exit_price
    exit_fee = exit_notional * fee_rate
    return exit_notional - exit_fee - baseline_notional - entry_fee


def _clamp_threshold(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, value))


def _threshold_confidence(long_threshold: float | None, short_threshold: float | None, fallback: float) -> float:
    values: list[float] = []
    if long_threshold is not None:
        values.append(float(long_threshold))
    if short_threshold is not None:
        values.append(1.0 - float(short_threshold))
    if not values:
        values.append(float(fallback))
    return _clamp_threshold(max(values))


def _adjusted_side_thresholds(
    long_threshold: float | None,
    short_threshold: float | None,
    threshold_add: float,
) -> tuple[float | None, float | None]:
    add = max(-1.0, min(1.0, _finite_float(threshold_add, 0.0)))
    adjusted_long = None
    adjusted_short = None
    if long_threshold is not None:
        adjusted_long = max(0.0, min(1.0, float(long_threshold) + add))
    if short_threshold is not None:
        adjusted_short = max(0.0, min(1.0, float(short_threshold) - add))
    return adjusted_long, adjusted_short


def _score_backend_payload(backend: BackendInfo) -> dict[str, str]:
    return {
        "scoring_backend_requested": backend.requested,
        "scoring_backend_kind": backend.kind,
        "scoring_backend_device": backend.device,
        "scoring_backend_reason": backend.reason,
    }


def _fallback_score_backend(requested: BackendInfo, reason: str) -> BackendInfo:
    return BackendInfo(
        requested=requested.requested,
        kind="cpu",
        device="cpu",
        vendor="Python stdlib",
        reason=reason[:240],
    )


def _position_size_from_risk(
    cash: float,
    cfg: StrategyConfig,
    *,
    market_type: str,
    leverage: float,
) -> tuple[float, float]:
    """Return gross notional and margin sized from stop-loss risk budget."""

    if cash <= 0.0:
        return 0.0, 0.0
    notional_pct = stop_loss_sized_notional_pct(cfg, market_type, leverage=leverage)
    gross = cash * notional_pct
    if market_type == "spot":
        return max(0.0, gross), max(0.0, gross)

    effective_leverage = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, _finite_float(leverage, 1.0)))
    margin = gross / effective_leverage if effective_leverage > 0.0 else gross
    return max(0.0, gross), max(0.0, margin)


def _torch_device_for_backend(backend: BackendInfo):  # pragma: no cover - optional GPU runtime
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


def _batch_probabilities_torch(  # pragma: no cover - exercised by host GPU smoke verification
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    backend: BackendInfo,
    batch_size: int,
) -> list[float]:
    import torch  # type: ignore

    device = _torch_device_for_backend(backend)
    batch = max(1, int(batch_size or 8192))
    probabilities: list[float] = []
    temperature = max(1e-6, float(getattr(model, "probability_temperature", 1.0) or 1.0))

    def mlp_spec_from_params(params: Mapping[str, object], input_dim_default: int):
        raw_layers = params.get("layers")
        if not isinstance(raw_layers, list) or not raw_layers:
            return None
        input_dim = max(1, min(
            int(_clamp_float(params.get("input_dim", input_dim_default), 1, max(1, model.feature_dim), input_dim_default)),
            max(1, model.feature_dim),
        ))
        expected_inputs = input_dim
        layer_specs = []
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict):
                return None
            raw_weights = raw_layer.get("weights")
            raw_bias = raw_layer.get("bias")
            if not isinstance(raw_weights, list) or not isinstance(raw_bias, list):
                return None
            try:
                bias_values = [float(value) for value in raw_bias]
            except (TypeError, ValueError, OverflowError):
                return None
            output_dim = len(bias_values)
            if output_dim <= 0 or len(raw_weights) != expected_inputs:
                return None
            matrix: list[list[float]] = []
            try:
                for raw_column in raw_weights:
                    if not isinstance(raw_column, list) or len(raw_column) < output_dim:
                        return None
                    column = [float(raw_column[index]) for index in range(output_dim)]
                    if any(not math.isfinite(value) for value in column):
                        return None
                    matrix.append(column)
            except (TypeError, ValueError, OverflowError):
                return None
            if any(not math.isfinite(value) for value in bias_values):
                return None
            layer_specs.append((
                torch.tensor(matrix, dtype=torch.float32, device=device),
                torch.tensor(bias_values, dtype=torch.float32, device=device),
                str(raw_layer.get("activation", "relu") or "relu").lower(),
            ))
            expected_inputs = output_dim
        return input_dim, layer_specs

    def mlp_output(input_values, layer_specs):
        values = input_values
        for weights_t, bias_t, activation in layer_specs:
            values = torch.clamp(values.matmul(weights_t) + bias_t, min=-50.0, max=50.0)
            if activation == "sigmoid":
                values = torch.sigmoid(values)
            elif activation == "tanh":
                values = torch.tanh(values)
            elif activation == "linear":
                values = values
            else:
                values = torch.relu(values)
        return values[:, 0]

    def lightgbm_spec_from_params(
        params: Mapping[str, object],
        input_dim_default: int,
        *,
        tree_key: str = "tree_info",
        average_output_key: str = "average_output",
    ):
        tree_info = params.get(tree_key)
        if (
            not isinstance(tree_info, list)
            or not tree_info
            or len(tree_info) > MAX_SERIALIZED_LIGHTGBM_TREES
        ):
            return None
        try:
            input_dim = max(
                1,
                min(
                    int(params.get("input_dim", input_dim_default) or input_dim_default),
                    int(model.feature_dim),
                ),
            )
        except (TypeError, ValueError, OverflowError):
            return None

        serialized_trees: list[list[tuple[int, float, int, int, bool, bool, float]]] = []
        max_nodes = 0
        max_depth = 0
        total_nodes = 0
        for tree in tree_info:
            if not isinstance(tree, dict) or not isinstance(tree.get("tree_structure"), dict):
                return None
            pending: list[tuple[dict[str, object], int]] = [(tree["tree_structure"], 0)]
            nodes: list[tuple[int, float, int, int, bool, bool, float]] = []
            cursor = 0
            while cursor < len(pending):
                node, depth = pending[cursor]
                cursor += 1
                total_nodes += 1
                if total_nodes > MAX_SERIALIZED_LIGHTGBM_NODES or depth >= MAX_SERIALIZED_LIGHTGBM_DEPTH:
                    return None
                max_depth = max(max_depth, depth)
                if "leaf_value" in node:
                    try:
                        leaf_value = float(node["leaf_value"])
                    except (TypeError, ValueError, OverflowError):
                        return None
                    if not math.isfinite(leaf_value):
                        return None
                    nodes.append((0, 0.0, len(nodes), len(nodes), True, True, leaf_value))
                    continue
                try:
                    split_feature = int(node["split_feature"])
                    threshold = float(node["threshold"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return None
                if (
                    split_feature < 0
                    or split_feature >= input_dim
                    or not math.isfinite(threshold)
                    or str(node.get("decision_type", "<=") or "<=") != "<="
                ):
                    return None
                left = node.get("left_child")
                right = node.get("right_child")
                if not isinstance(left, dict) or not isinstance(right, dict):
                    return None
                left_index = len(pending)
                pending.append((left, depth + 1))
                right_index = len(pending)
                pending.append((right, depth + 1))
                nodes.append(
                    (
                        split_feature,
                        threshold,
                        left_index,
                        right_index,
                        bool(node.get("default_left", True)),
                        False,
                        0.0,
                    )
                )
            serialized_trees.append(nodes)
            max_nodes = max(max_nodes, len(nodes))
        if not serialized_trees or max_nodes <= 0:
            return None

        split_features: list[int] = []
        thresholds: list[float] = []
        left_children: list[int] = []
        right_children: list[int] = []
        default_left: list[float] = []
        is_leaf: list[float] = []
        leaf_values: list[float] = []
        for nodes in serialized_trees:
            padded = [*nodes]
            padded.extend([(0, 0.0, 0, 0, True, True, 0.0)] * (max_nodes - len(nodes)))
            for feature, threshold, left, right, default, leaf, leaf_value in padded:
                split_features.append(int(feature))
                thresholds.append(float(threshold))
                left_children.append(int(left))
                right_children.append(int(right))
                default_left.append(1.0 if default else 0.0)
                is_leaf.append(1.0 if leaf else 0.0)
                leaf_values.append(float(leaf_value))

        return {
            "input_dim": int(input_dim),
            "tree_count": int(len(serialized_trees)),
            "max_nodes": int(max_nodes),
            "max_depth": int(max_depth),
            "tree_offsets": torch.arange(
                len(serialized_trees),
                dtype=torch.int64,
                device=device,
            ) * int(max_nodes),
            "split_features": torch.tensor(split_features, dtype=torch.int64, device=device),
            "thresholds": torch.tensor(thresholds, dtype=torch.float32, device=device),
            "left_children": torch.tensor(left_children, dtype=torch.int64, device=device),
            "right_children": torch.tensor(right_children, dtype=torch.int64, device=device),
            "default_left": torch.tensor(default_left, dtype=torch.float32, device=device),
            "is_leaf": torch.tensor(is_leaf, dtype=torch.float32, device=device),
            "leaf_values": torch.tensor(leaf_values, dtype=torch.float32, device=device),
            "average_output": bool(params.get(average_output_key, False)),
        }

    def lightgbm_output(expert_features, spec):
        row_count = int(expert_features.shape[0])
        tree_count = int(spec["tree_count"])
        node_indexes = torch.zeros(
            (row_count, tree_count),
            dtype=torch.int64,
            device=device,
        )
        offsets = spec["tree_offsets"].reshape(1, -1)
        for _depth in range(int(spec["max_depth"]) + 1):
            flat_indexes = node_indexes + offsets
            leaf_mask = spec["is_leaf"][flat_indexes] > 0.5
            split_indexes = spec["split_features"][flat_indexes]
            feature_values = torch.gather(
                expert_features[:, : int(spec["input_dim"])],
                1,
                split_indexes,
            )
            thresholds = spec["thresholds"][flat_indexes]
            default_left = spec["default_left"][flat_indexes] > 0.5
            go_left = torch.where(torch.isfinite(feature_values), feature_values <= thresholds, default_left)
            next_indexes = torch.where(
                go_left,
                spec["left_children"][flat_indexes],
                spec["right_children"][flat_indexes],
            )
            node_indexes = torch.where(leaf_mask, node_indexes, next_indexes)
        flat_indexes = node_indexes + offsets
        final_leaf_mask = spec["is_leaf"][flat_indexes] > 0.5
        if not bool(torch.all(final_leaf_mask).detach().cpu().item()):
            raise RuntimeError("LightGBM tensor traversal did not terminate at leaves")
        output = torch.sum(spec["leaf_values"][flat_indexes], dim=1)
        if bool(spec["average_output"]):
            output = output / float(max(1, tree_count))
        return output

    members = list(model.ensemble_members)
    model_specs = []
    if members:
        for member in members:
            model_specs.append((member.weights, member.bias, member.feature_means, member.feature_stds))
    else:
        model_specs.append((model.weights, model.bias, model.feature_means, model.feature_stds))
    model_tensors = []
    for weights, bias, means, stds in model_specs:
        w = torch.tensor(list(weights), dtype=torch.float32, device=device)
        b = torch.tensor(float(bias), dtype=torch.float32, device=device)
        mean_t = torch.tensor(list(means), dtype=torch.float32, device=device)
        std_t = torch.tensor(list(stds), dtype=torch.float32, device=device)
        std_t = torch.where(torch.abs(std_t) > 0.0, std_t, torch.ones_like(std_t))
        model_tensors.append((w, b, mean_t, std_t))

    base_mean_t = torch.tensor(list(model.feature_means), dtype=torch.float32, device=device)
    base_std_t = torch.tensor(list(model.feature_stds), dtype=torch.float32, device=device)
    base_std_t = torch.where(torch.abs(base_std_t) > 0.0, base_std_t, torch.ones_like(base_std_t))
    hybrid_specs = []
    positive_weight_expert_count = 0
    for expert in getattr(model, "hybrid_experts", []) or []:
        expert_weight = max(0.0, float(expert.weight))
        if expert_weight <= 0.0:
            continue
        positive_weight_expert_count += 1
        if expert.kind in {"lorentzian_knn", "rational_quadratic_kernel"}:
            prototypes = [
                prototype
                for prototype in expert.prototypes
                if len(prototype.features) == model.feature_dim
            ]
            if not prototypes:
                raise ValueError(f"Accelerated scorer cannot load prototypes for {expert.name}")
            proto_features = torch.tensor(
                [prototype.features for prototype in prototypes],
                dtype=torch.float32,
                device=device,
            )
            proto_labels = torch.tensor(
                [float(1 if prototype.label else 0) for prototype in prototypes],
                dtype=torch.float32,
                device=device,
            )
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                proto_features,
                proto_labels,
                max(1, min(int(expert.k), len(prototypes))),
                max(1e-6, float(expert.bandwidth)),
                max(1e-6, float(expert.alpha)),
                max(1, int(expert.feature_count)),
                dict(getattr(expert, "params", {}) or {}),
            ))
        elif expert.kind == "technical_confluence":
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                None,
                None,
                0,
                1.0,
                1.0,
                max(1, int(expert.feature_count)),
                dict(getattr(expert, "params", {}) or {}),
            ))
        elif expert.kind == "rule_alpha":
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                None,
                None,
                0,
                1.0,
                1.0,
                max(1, int(expert.feature_count)),
                dict(getattr(expert, "params", {}) or {}),
            ))
        elif expert.kind in {"dense_mlp", "signed_payoff_mlp_ranker"}:
            params = dict(getattr(expert, "params", {}) or {})
            spec = mlp_spec_from_params(params, max(1, min(int(expert.feature_count), model.feature_dim)))
            if spec is None:
                raise ValueError(f"Accelerated scorer cannot load MLP expert {expert.name}")
            input_dim, layer_specs = spec
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                layer_specs,
                None,
                0,
                1.0,
                1.0,
                input_dim,
                params,
            ))
        elif expert.kind == "signed_payoff_ranker":
            params = dict(getattr(expert, "params", {}) or {})
            weights = params.get("weights")
            if not isinstance(weights, list) or not weights:
                raise ValueError(f"Accelerated scorer cannot load payoff expert {expert.name}")
            input_dim = max(1, min(int(params.get("input_dim", len(weights)) or len(weights)), model.feature_dim, len(weights)))
            payoff_weights = torch.tensor([float(value) for value in weights[:input_dim]], dtype=torch.float32, device=device)
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                payoff_weights,
                None,
                0,
                1.0,
                1.0,
                input_dim,
                params,
            ))
        elif expert.kind == "signed_payoff_lightgbm_ranker":
            params = dict(getattr(expert, "params", {}) or {})
            default_input_dim = max(1, min(int(expert.feature_count), model.feature_dim))
            payoff_tree_schema = str(params.get("payoff_tree_schema", "") or "")
            if payoff_tree_schema == "action_value_hurdle_v1":
                long_spec = lightgbm_spec_from_params(
                    params,
                    default_input_dim,
                    tree_key="long_classifier_tree_info",
                    average_output_key="long_classifier_average_output",
                )
                short_spec = lightgbm_spec_from_params(
                    params,
                    default_input_dim,
                    tree_key="short_classifier_tree_info",
                    average_output_key="short_classifier_average_output",
                )
                spec = (
                    {
                        "input_dim": int(long_spec["input_dim"]),
                        "long": long_spec,
                        "short": short_spec,
                        "action_hurdle": True,
                    }
                    if long_spec is not None
                    and short_spec is not None
                    and int(long_spec["input_dim"]) == int(short_spec["input_dim"])
                    else None
                )
            elif payoff_tree_schema == "action_value_v1":
                long_spec = lightgbm_spec_from_params(
                    params,
                    default_input_dim,
                    tree_key="long_tree_info",
                    average_output_key="long_average_output",
                )
                short_spec = lightgbm_spec_from_params(
                    params,
                    default_input_dim,
                    tree_key="short_tree_info",
                    average_output_key="short_average_output",
                )
                spec = (
                    {
                        "input_dim": int(long_spec["input_dim"]),
                        "long": long_spec,
                        "short": short_spec,
                        "action_value": True,
                    }
                    if long_spec is not None
                    and short_spec is not None
                    and int(long_spec["input_dim"]) == int(short_spec["input_dim"])
                    else None
                )
            else:
                spec = lightgbm_spec_from_params(params, default_input_dim)
            if spec is None:
                raise ValueError(f"Accelerated scorer cannot load LightGBM expert {expert.name}")
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                spec,
                None,
                0,
                1.0,
                1.0,
                int(spec["input_dim"]),
                params,
            ))
        else:
            raise ValueError(
                f"Accelerated scorer does not support positive-weight expert kind {expert.kind}"
            )
    if len(hybrid_specs) != positive_weight_expert_count:
        raise RuntimeError(
            f"Accelerated scorer expert coverage mismatch: {len(hybrid_specs)}/{positive_weight_expert_count}"
        )

    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        features = torch.tensor([row.features for row in chunk], dtype=torch.float32, device=device)
        chunk_probs = None
        for w, b, mean_t, std_t in model_tensors:
            normalized = (features - mean_t) / std_t
            logits = normalized.matmul(w.reshape(-1, 1)).reshape(-1) + b
            logits = torch.clamp(logits / temperature, min=-50.0, max=50.0)
            probs = torch.sigmoid(logits)
            chunk_probs = probs if chunk_probs is None else chunk_probs + probs
        if chunk_probs is None:  # pragma: no cover - model_specs is always populated above
            raise RuntimeError("No model probabilities were produced.")
        if len(model_specs) > 1:
            chunk_probs = chunk_probs / float(len(model_specs))
        if hybrid_specs:
            base_weight = max(0.0, float(getattr(model, "hybrid_base_weight", 1.0)))
            weighted = chunk_probs * base_weight
            total = torch.full_like(chunk_probs, base_weight)
            expert_features = (features - base_mean_t) / base_std_t
            for kind, expert_weight, proto_features, proto_labels, k, bandwidth, alpha, feature_count, params in hybrid_specs:
                expert_probs = None
                if kind == "lorentzian_knn" and proto_features is not None and proto_labels is not None:
                    distances = torch.log1p(torch.abs(expert_features[:, None, :] - proto_features[None, :, :]))
                    distances = distances.sum(dim=2) / float(max(1, model.feature_dim))
                    nearest_distance, nearest_index = torch.topk(distances, k=int(k), dim=1, largest=False)
                    label_matrix = proto_labels.reshape(1, -1).expand(features.shape[0], -1)
                    nearest_labels = torch.gather(label_matrix, 1, nearest_index)
                    neighbor_weights = 1.0 / torch.clamp(nearest_distance + 1e-6, min=1e-9)
                    expert_probs = (neighbor_weights * nearest_labels).sum(dim=1) / neighbor_weights.sum(dim=1)
                elif kind == "rational_quadratic_kernel" and proto_features is not None and proto_labels is not None:
                    deltas = expert_features[:, None, :] - proto_features[None, :, :]
                    scaled = torch.sum(deltas * deltas, dim=2) / float(max(1, model.feature_dim))
                    kernel_weights = torch.pow(
                        1.0 + scaled / float(2.0 * alpha * bandwidth * bandwidth),
                        float(-alpha),
                    )
                    expert_probs = (kernel_weights * proto_labels.reshape(1, -1)).sum(dim=1) / kernel_weights.sum(dim=1)
                elif kind == "technical_confluence":
                    count = max(1, min(int(feature_count), int(features.shape[1]), 13))
                    values = features[:, :count]
                    if count < 13:
                        padding = torch.zeros((features.shape[0], 13 - count), dtype=torch.float32, device=device)
                        values = torch.cat((values, padding), dim=1)
                    else:
                        values = values[:, :13]
                    momentum_1 = values[:, 0]
                    momentum_3 = values[:, 1]
                    momentum_10 = values[:, 2]
                    momentum_20 = values[:, 3]
                    ema_spread = values[:, 4]
                    rsi = torch.clamp(values[:, 5], min=0.0, max=1.0)
                    ema_gap = values[:, 6]
                    relative_atr = torch.abs(values[:, 7])
                    volatility_20 = torch.abs(values[:, 8])
                    volume_ratio = values[:, 9]
                    trend_acceleration = values[:, 10]
                    gap_to_vwap = values[:, 11]
                    volume_trend = values[:, 12]
                    trend = (
                        0.24 * torch.tanh(momentum_20 * 80.0)
                        + 0.20 * torch.tanh(momentum_10 * 100.0)
                        + 0.14 * torch.tanh(momentum_3 * 140.0)
                        - 0.16 * torch.tanh(ema_spread * 90.0)
                        + 0.10 * torch.tanh(trend_acceleration * 240.0)
                        + 0.06 * torch.tanh(volume_trend * 4.0)
                    )
                    mean_reversion = (
                        0.18 * torch.tanh((0.38 - rsi) * 5.0)
                        - 0.14 * torch.tanh(gap_to_vwap * 150.0)
                        - 0.08 * torch.tanh(momentum_1 * 180.0)
                    )
                    breakout = (
                        0.10 * torch.tanh(volume_ratio * 2.5)
                        + 0.10 * torch.tanh((relative_atr + volatility_20) * 80.0)
                        + 0.08 * torch.tanh((momentum_10 + momentum_20) * 80.0)
                        - 0.04 * torch.tanh(torch.abs(ema_gap) * 150.0)
                    )
                    expert_probs = torch.clamp(torch.sigmoid((trend + mean_reversion + breakout) * 2.2), min=0.0, max=1.0)
                elif kind == "dense_mlp" and proto_features is not None:
                    count = max(1, min(int(feature_count), int(expert_features.shape[1])))
                    raw_output = mlp_output(expert_features[:, :count], proto_features)
                    output_activation = str(params.get("output_activation", "sigmoid") or "sigmoid").lower()
                    if output_activation == "sigmoid":
                        expert_probs = torch.clamp(raw_output, min=0.0, max=1.0)
                    else:
                        expert_probs = torch.clamp(torch.sigmoid(raw_output), min=0.0, max=1.0)
                elif kind == "signed_payoff_ranker" and proto_features is not None:
                    count = max(1, min(int(feature_count), int(expert_features.shape[1]), int(proto_features.shape[0])))
                    raw_score = expert_features[:, :count].matmul(proto_features[:count].reshape(-1, 1)).reshape(-1)
                    raw_score = raw_score + float(_clamp_float(params.get("bias", 0.0), -10.0, 10.0, 0.0))
                    clipped = torch.clamp(raw_score, min=-1.0, max=1.0)
                    clip_bps = _clamp_float(params.get("clip_bps", 25.0), 0.1, 10_000.0, 25.0)
                    deadband_bps = _clamp_float(params.get("deadband_bps", 0.0), 0.0, clip_bps, 0.0)
                    deadband = min(0.95, float(deadband_bps) / max(1e-9, float(clip_bps)))
                    magnitude = torch.abs(clipped)
                    adjusted = torch.where(
                        magnitude <= float(deadband),
                        torch.zeros_like(clipped),
                        torch.sign(clipped) * ((magnitude - float(deadband)) / max(1e-9, 1.0 - float(deadband))),
                    )
                    sensitivity = _clamp_float(params.get("sensitivity", 6.0), 0.1, 30.0, 6.0)
                    probability_bias = _clamp_float(params.get("probability_bias", 0.0), -5.0, 5.0, 0.0)
                    expert_probs = torch.clamp(
                        torch.sigmoid(adjusted * float(sensitivity) + float(probability_bias)),
                        min=0.0,
                        max=1.0,
                    )
                elif kind == "signed_payoff_mlp_ranker" and proto_features is not None:
                    count = max(1, min(int(feature_count), int(expert_features.shape[1])))
                    raw_score = torch.clamp(mlp_output(expert_features[:, :count], proto_features), min=-1.0, max=1.0)
                    clip_bps = _clamp_float(params.get("clip_bps", 25.0), 0.1, 10_000.0, 25.0)
                    deadband_bps = _clamp_float(params.get("deadband_bps", 0.0), 0.0, clip_bps, 0.0)
                    deadband = min(0.95, float(deadband_bps) / max(1e-9, float(clip_bps)))
                    magnitude = torch.abs(raw_score)
                    adjusted = torch.where(
                        magnitude <= float(deadband),
                        torch.zeros_like(raw_score),
                        torch.sign(raw_score) * ((magnitude - float(deadband)) / max(1e-9, 1.0 - float(deadband))),
                    )
                    sensitivity = _clamp_float(params.get("sensitivity", 6.0), 0.1, 30.0, 6.0)
                    probability_bias = _clamp_float(params.get("probability_bias", 0.0), -5.0, 5.0, 0.0)
                    expert_probs = torch.clamp(
                        torch.sigmoid(adjusted * float(sensitivity) + float(probability_bias)),
                        min=0.0,
                        max=1.0,
                    )
                elif kind == "signed_payoff_lightgbm_ranker" and proto_features is not None:
                    clip_bps = _clamp_float(params.get("clip_bps", 25.0), 0.1, 10_000.0, 25.0)
                    deadband_bps = _clamp_float(params.get("deadband_bps", 0.0), 0.0, clip_bps, 0.0)
                    deadband = min(0.95, float(deadband_bps) / max(1e-9, float(clip_bps)))
                    sensitivity = _clamp_float(params.get("sensitivity", 6.0), 0.1, 30.0, 6.0)
                    probability_bias = _clamp_float(params.get("probability_bias", 0.0), -5.0, 5.0, 0.0)
                    if bool(proto_features.get("action_hurdle", False)):
                        long_margin = lightgbm_output(expert_features, proto_features["long"])
                        short_margin = lightgbm_output(expert_features, proto_features["short"])

                        def hurdle_action_value(side, margin):
                            if not bool(params.get(f"{side}_enabled", False)):
                                return torch.full_like(margin, -1.0)
                            slope = _clamp_float(
                                params.get(f"{side}_calibration_slope", 0.0),
                                0.0,
                                100.0,
                                0.0,
                            )
                            intercept = _clamp_float(
                                params.get(f"{side}_calibration_intercept", 0.0),
                                -100.0,
                                100.0,
                                0.0,
                            )
                            positive_mean = _clamp_float(
                                params.get(f"{side}_positive_mean", 0.0),
                                0.0,
                                1.0,
                                0.0,
                            )
                            nonpositive_mean = _clamp_float(
                                params.get(f"{side}_nonpositive_mean", -1.0),
                                -1.0,
                                0.0,
                                -1.0,
                            )
                            profitable_probability = torch.sigmoid(
                                margin * float(slope) + float(intercept)
                            )
                            return torch.clamp(
                                profitable_probability * float(positive_mean)
                                + (1.0 - profitable_probability) * float(nonpositive_mean),
                                min=-1.0,
                                max=1.0,
                            )

                        long_score = hurdle_action_value("long", long_margin)
                        short_score = hurdle_action_value("short", short_margin)
                        best_score = torch.maximum(long_score, short_score)
                        actionable = (best_score > float(deadband)) & (torch.abs(long_score - short_score) > 1e-12)
                        adjusted = (best_score - float(deadband)) / max(1e-9, 1.0 - float(deadband))
                        action_confidence = torch.clamp(
                            torch.sigmoid(adjusted * float(sensitivity) + float(probability_bias)),
                            min=0.5,
                            max=1.0,
                        )
                        directional_probability = torch.where(
                            long_score > short_score,
                            action_confidence,
                            1.0 - action_confidence,
                        )
                        expert_probs = torch.where(
                            actionable,
                            directional_probability,
                            torch.full_like(directional_probability, 0.5),
                        )
                    elif bool(proto_features.get("action_value", False)):
                        long_score = torch.clamp(
                            lightgbm_output(expert_features, proto_features["long"]),
                            min=-1.0,
                            max=1.0,
                        )
                        short_score = torch.clamp(
                            lightgbm_output(expert_features, proto_features["short"]),
                            min=-1.0,
                            max=1.0,
                        )
                        best_score = torch.maximum(long_score, short_score)
                        actionable = (best_score > float(deadband)) & (torch.abs(long_score - short_score) > 1e-12)
                        adjusted = (best_score - float(deadband)) / max(1e-9, 1.0 - float(deadband))
                        action_confidence = torch.clamp(
                            torch.sigmoid(adjusted * float(sensitivity) + float(probability_bias)),
                            min=0.5,
                            max=1.0,
                        )
                        directional_probability = torch.where(
                            long_score > short_score,
                            action_confidence,
                            1.0 - action_confidence,
                        )
                        expert_probs = torch.where(
                            actionable,
                            directional_probability,
                            torch.full_like(directional_probability, 0.5),
                        )
                    else:
                        raw_score = torch.clamp(
                            lightgbm_output(expert_features, proto_features),
                            min=-1.0,
                            max=1.0,
                        )
                        magnitude = torch.abs(raw_score)
                        adjusted = torch.where(
                            magnitude <= float(deadband),
                            torch.zeros_like(raw_score),
                            torch.sign(raw_score)
                            * ((magnitude - float(deadband)) / max(1e-9, 1.0 - float(deadband))),
                        )
                        expert_probs = torch.clamp(
                            torch.sigmoid(adjusted * float(sensitivity) + float(probability_bias)),
                            min=0.0,
                            max=1.0,
                        )
                elif kind == "rule_alpha":
                    count = max(1, min(int(feature_count), int(features.shape[1])))
                    raw_values = features[:, :count]
                    if count < 13:
                        padding = torch.zeros((features.shape[0], 13 - count), dtype=torch.float32, device=device)
                        values = torch.cat((raw_values, padding), dim=1)
                    else:
                        values = raw_values[:, :13]
                    family = str(params.get("family", "momentum_breakout")).strip().lower()
                    sensitivity = _clamp_float(params.get("sensitivity", 7.0), 0.1, 30.0, 7.0)
                    bias = _clamp_float(params.get("bias", 0.0), -5.0, 5.0, 0.0)
                    deadband = _clamp_float(params.get("deadband", 0.04), 0.0, 0.95, 0.04)
                    if family == "empirical_feature_edge":
                        max_feature_index = max(0, int(raw_values.shape[1]) - 1)
                        feature_index = int(_clamp_float(params.get("feature_index", 0), 0, max_feature_index, 0))
                        threshold_value = _clamp_float(params.get("feature_threshold", 0.0), -1e12, 1e12, 0.0)
                        feature_scale = max(1e-9, abs(_clamp_float(params.get("feature_scale", 1.0), 1e-12, 1e12, 1.0)))
                        tail_direction = 1.0 if _clamp_float(params.get("tail_direction", 1.0), -1.0, 1.0, 1.0) >= 0.0 else -1.0
                        trade_side = 1.0 if _clamp_float(params.get("trade_side", 1.0), -1.0, 1.0, 1.0) >= 0.0 else -1.0
                        confidence = _clamp_float(params.get("edge_confidence", 1.0), 0.0, 1.0, 1.0)
                        slope = _clamp_float(params.get("edge_slope", 1.0), 0.1, 20.0, 1.0)
                        feature_values = raw_values[:, feature_index]
                        feature_values = torch.where(
                            torch.isfinite(feature_values),
                            feature_values,
                            torch.full_like(feature_values, float(threshold_value)),
                        )
                        delta = float(tail_direction) * (feature_values - float(threshold_value)) / float(feature_scale)
                        if "second_feature_index" in params:
                            second_index = int(_clamp_float(params.get("second_feature_index", 0), 0, max_feature_index, 0))
                            second_threshold = _clamp_float(params.get("second_feature_threshold", 0.0), -1e12, 1e12, 0.0)
                            second_scale = max(1e-9, abs(_clamp_float(params.get("second_feature_scale", 1.0), 1e-12, 1e12, 1.0)))
                            second_tail = 1.0 if _clamp_float(params.get("second_tail_direction", 1.0), -1.0, 1.0, 1.0) >= 0.0 else -1.0
                            second_values = raw_values[:, second_index]
                            second_values = torch.where(
                                torch.isfinite(second_values),
                                second_values,
                                torch.full_like(second_values, float(second_threshold)),
                            )
                            second_delta = float(second_tail) * (second_values - float(second_threshold)) / float(second_scale)
                            delta = torch.minimum(delta, second_delta)
                        score = float(trade_side) * torch.clamp(torch.tanh(torch.clamp(delta, min=0.0) * float(slope)), min=0.0) * float(confidence)
                        magnitude = torch.abs(score)
                        adjusted = torch.where(
                            magnitude <= float(deadband),
                            torch.zeros_like(score),
                            torch.sign(score) * ((magnitude - float(deadband)) / max(1e-9, 1.0 - float(deadband))),
                        )
                        adjusted = torch.clamp(adjusted, min=-1.0, max=1.0)
                        expert_probs = torch.clamp(torch.sigmoid(adjusted * sensitivity + bias), min=0.0, max=1.0)
                        weighted = weighted + float(expert_weight) * expert_probs
                        total = total + float(expert_weight)
                        continue
                    momentum_1 = values[:, 0]
                    momentum_3 = values[:, 1]
                    momentum_10 = values[:, 2]
                    momentum_20 = values[:, 3]
                    ema_spread = values[:, 4]
                    rsi = torch.clamp(values[:, 5], min=0.0, max=1.0)
                    ema_gap = values[:, 6]
                    relative_atr = torch.abs(values[:, 7])
                    volatility_20 = torch.abs(values[:, 8])
                    volume_ratio = values[:, 9]
                    trend_acceleration = values[:, 10]
                    gap_to_vwap = values[:, 11]
                    volume_trend = values[:, 12]
                    order_start = int(_clamp_float(params.get("order_flow_start", -1), -1, 100000, -1))
                    order_width = int(_clamp_float(params.get("order_flow_width", 9), 1, 64, 9))
                    order_count = int(_clamp_float(params.get("order_flow_window_count", 0), 0, 32, 0))
                    htf_start = int(_clamp_float(params.get("higher_timeframe_start", -1), -1, 100000, -1))
                    htf_width = int(_clamp_float(params.get("higher_timeframe_width", 8), 1, 64, 8))
                    htf_count = int(_clamp_float(params.get("higher_timeframe_window_count", 0), 0, 32, 0))
                    tape_start = int(_clamp_float(params.get("trade_tape_start", -1), -1, 100000, -1))
                    tape_width = int(_clamp_float(
                        params.get("trade_tape_width", TRADE_TAPE_FEATURES_PER_WINDOW),
                        1,
                        64,
                        TRADE_TAPE_FEATURES_PER_WINDOW,
                    ))
                    tape_count = int(_clamp_float(params.get("trade_tape_window_count", 0), 0, 32, 0))
                    taker_buy_ratio = torch.full_like(momentum_1, 0.5)
                    signed_base = torch.zeros_like(momentum_1)
                    signed_quote = torch.zeros_like(momentum_1)
                    trade_impulse = torch.zeros_like(momentum_1)
                    quote_impulse = torch.zeros_like(momentum_1)
                    quote_per_trade_impulse = torch.zeros_like(momentum_1)
                    no_trade_ratio = torch.zeros_like(momentum_1)
                    flow_return_alignment = torch.zeros_like(momentum_1)
                    signed_ratio_delta = torch.zeros_like(momentum_1)
                    mean_abs_signed_ratio = torch.zeros_like(momentum_1)
                    flow_persistence = torch.zeros_like(momentum_1)
                    flow_acceleration = torch.zeros_like(momentum_1)
                    price_flow_divergence = torch.zeros_like(momentum_1)
                    htf_return = torch.zeros_like(momentum_1)
                    htf_mean_gap = torch.zeros_like(momentum_1)
                    htf_realized_volatility = torch.zeros_like(momentum_1)
                    htf_range = torch.zeros_like(momentum_1)
                    htf_drawdown = torch.zeros_like(momentum_1)
                    htf_bounce = torch.zeros_like(momentum_1)
                    htf_volume_impulse = torch.zeros_like(momentum_1)
                    htf_trade_impulse = torch.zeros_like(momentum_1)
                    tape_available = False
                    tape_buy_notional_ratio = torch.full_like(momentum_1, 0.5)
                    tape_signed_notional = torch.zeros_like(momentum_1)
                    tape_count_signed = torch.zeros_like(momentum_1)
                    tape_notional_impulse = torch.zeros_like(momentum_1)
                    tape_count_impulse = torch.zeros_like(momentum_1)
                    tape_large_share = torch.zeros_like(momentum_1)
                    tape_vwap_gap = torch.zeros_like(momentum_1)
                    tape_micro_drift = torch.zeros_like(momentum_1)
                    tape_no_tape = torch.ones_like(momentum_1)
                    tape_signed_acceleration = torch.zeros_like(momentum_1)
                    tape_flow_return_alignment = torch.zeros_like(momentum_1)
                    flow_groups = []
                    if order_start >= 0 and order_count > 0:
                        for order_index in range(order_count):
                            group_start = order_start + order_index * order_width
                            group_end = group_start + order_width
                            if group_end <= raw_values.shape[1]:
                                flow_groups.append(raw_values[:, group_start:group_end])
                    if flow_groups:
                        flow_stack = torch.stack(flow_groups, dim=2)
                        taker_buy_ratio = torch.clamp(torch.mean(flow_stack[:, 0, :], dim=1), min=0.0, max=1.0)
                        no_trade_ratio = torch.clamp(torch.mean(flow_stack[:, 6, :], dim=1), min=0.0, max=1.0) if order_width > 6 else no_trade_ratio
                        flow_quality = 1.0 - no_trade_ratio
                        signed_base = torch.clamp(torch.mean(flow_stack[:, 1, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 1 else signed_base
                        signed_quote = torch.clamp(torch.mean(flow_stack[:, 2, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 2 else signed_quote
                        trade_impulse = torch.clamp(torch.mean(flow_stack[:, 3, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 3 else trade_impulse
                        quote_impulse = torch.clamp(torch.mean(flow_stack[:, 4, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 4 else quote_impulse
                        quote_per_trade_impulse = torch.clamp(torch.mean(flow_stack[:, 5, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 5 else quote_per_trade_impulse
                        flow_return_alignment = torch.clamp(torch.mean(flow_stack[:, 7, :], dim=1), min=-1.0, max=1.0) if order_width > 7 else flow_return_alignment
                        signed_ratio_delta = torch.clamp(torch.mean(flow_stack[:, 8, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 8 else signed_ratio_delta
                        mean_abs_signed_ratio = torch.clamp(torch.mean(flow_stack[:, 9, :], dim=1), min=0.0, max=1.0) if order_width > 9 else mean_abs_signed_ratio
                        flow_persistence = torch.clamp(torch.mean(flow_stack[:, 10, :], dim=1), min=-1.0, max=1.0) if order_width > 10 else flow_persistence
                        flow_acceleration = torch.clamp(torch.mean(flow_stack[:, 11, :], dim=1) * flow_quality, min=-1.0, max=1.0) if order_width > 11 else flow_acceleration
                        price_flow_divergence = torch.clamp(torch.mean(flow_stack[:, 12, :], dim=1), min=-1.0, max=1.0) if order_width > 12 else price_flow_divergence
                    htf_groups = []
                    if htf_start >= 0 and htf_count > 0:
                        for htf_index in range(htf_count):
                            group_start = htf_start + htf_index * htf_width
                            group_end = group_start + htf_width
                            if group_end <= raw_values.shape[1]:
                                htf_groups.append(raw_values[:, group_start:group_end])
                    if htf_groups:
                        htf_stack = torch.stack(htf_groups, dim=2)
                        htf_return = torch.clamp(torch.mean(htf_stack[:, 0, :], dim=1), min=-1.0, max=1.0)
                        htf_mean_gap = torch.clamp(torch.mean(htf_stack[:, 1, :], dim=1), min=-1.0, max=1.0) if htf_width > 1 else htf_mean_gap
                        htf_realized_volatility = torch.clamp(torch.abs(torch.mean(htf_stack[:, 2, :], dim=1)), min=0.0, max=1.0) if htf_width > 2 else htf_realized_volatility
                        htf_range = torch.clamp(torch.abs(torch.mean(htf_stack[:, 3, :], dim=1)), min=0.0, max=2.0) if htf_width > 3 else htf_range
                        htf_drawdown = torch.clamp(torch.mean(htf_stack[:, 4, :], dim=1), min=-1.0, max=0.0) if htf_width > 4 else htf_drawdown
                        htf_bounce = torch.clamp(torch.mean(htf_stack[:, 5, :], dim=1), min=0.0, max=2.0) if htf_width > 5 else htf_bounce
                        htf_volume_impulse = torch.clamp(torch.mean(htf_stack[:, 6, :], dim=1), min=-5.0, max=5.0) if htf_width > 6 else htf_volume_impulse
                        htf_trade_impulse = torch.clamp(torch.mean(htf_stack[:, 7, :], dim=1), min=-5.0, max=5.0) if htf_width > 7 else htf_trade_impulse
                    tape_groups = []
                    if tape_start >= 0 and tape_count > 0:
                        for tape_index in range(tape_count):
                            group_start = tape_start + tape_index * tape_width
                            group_end = group_start + tape_width
                            if group_end <= raw_values.shape[1]:
                                tape_groups.append(raw_values[:, group_start:group_end])
                    if tape_groups:
                        tape_stack = torch.stack(tape_groups, dim=2)
                        tape_available = True
                        tape_no_tape = torch.clamp(torch.mean(tape_stack[:, 9, :], dim=1), min=0.0, max=1.0) if tape_width > 9 else tape_no_tape
                        tape_quality = 1.0 - tape_no_tape
                        tape_buy_notional_ratio = torch.clamp(torch.mean(tape_stack[:, 0, :], dim=1), min=0.0, max=1.0)
                        tape_signed_notional = torch.clamp(torch.mean(tape_stack[:, 1, :], dim=1) * tape_quality, min=-1.0, max=1.0) if tape_width > 1 else tape_signed_notional
                        tape_count_signed = torch.clamp(torch.mean(tape_stack[:, 2, :], dim=1) * tape_quality, min=-1.0, max=1.0) if tape_width > 2 else tape_count_signed
                        tape_notional_impulse = torch.clamp(torch.mean(tape_stack[:, 3, :], dim=1) * tape_quality, min=-1.0, max=1.0) if tape_width > 3 else tape_notional_impulse
                        tape_count_impulse = torch.clamp(torch.mean(tape_stack[:, 4, :], dim=1) * tape_quality, min=-1.0, max=1.0) if tape_width > 4 else tape_count_impulse
                        tape_large_share = torch.clamp(torch.mean(tape_stack[:, 5, :], dim=1), min=0.0, max=1.0) if tape_width > 5 else tape_large_share
                        tape_vwap_gap = torch.clamp(torch.mean(tape_stack[:, 6, :], dim=1), min=-1.0, max=1.0) if tape_width > 6 else tape_vwap_gap
                        tape_micro_drift = torch.clamp(torch.mean(tape_stack[:, 8, :], dim=1), min=-1.0, max=1.0) if tape_width > 8 else tape_micro_drift
                        tape_signed_acceleration = torch.clamp(torch.mean(tape_stack[:, 10, :], dim=1) * tape_quality, min=-1.0, max=1.0) if tape_width > 10 else tape_signed_acceleration
                        tape_flow_return_alignment = torch.clamp(torch.mean(tape_stack[:, 11, :], dim=1), min=-1.0, max=1.0) if tape_width > 11 else tape_flow_return_alignment
                    if tape_available:
                        tape_quality = 1.0 - tape_no_tape
                        tape_weight = torch.clamp(0.35 + 0.45 * tape_quality, min=0.0, max=0.80)

                        def blend(left, right):
                            return left * (1.0 - tape_weight) + right * tape_weight

                        taker_buy_ratio = torch.clamp(blend(taker_buy_ratio, tape_buy_notional_ratio), min=0.0, max=1.0)
                        signed_base = torch.clamp(blend(signed_base, tape_signed_notional), min=-1.0, max=1.0)
                        signed_quote = torch.clamp(blend(signed_quote, tape_signed_notional), min=-1.0, max=1.0)
                        trade_impulse = torch.clamp(blend(trade_impulse, tape_count_impulse), min=-1.0, max=1.0)
                        quote_impulse = torch.clamp(blend(quote_impulse, tape_notional_impulse), min=-1.0, max=1.0)
                        quote_per_trade_impulse = torch.clamp(blend(quote_per_trade_impulse, tape_large_share), min=-1.0, max=1.0)
                        no_trade_ratio = torch.minimum(no_trade_ratio, tape_no_tape)
                        flow_return_alignment = torch.clamp(blend(flow_return_alignment, tape_flow_return_alignment), min=-1.0, max=1.0)
                        signed_ratio_delta = torch.clamp(blend(signed_ratio_delta, tape_signed_acceleration), min=-1.0, max=1.0)
                        mean_abs_signed_ratio = torch.clamp(torch.maximum(mean_abs_signed_ratio, torch.abs(tape_signed_notional)), min=0.0, max=1.0)
                        flow_acceleration = torch.clamp(blend(flow_acceleration, tape_signed_acceleration), min=-1.0, max=1.0)
                        tape_divergence = torch.tanh(
                            (tape_signed_notional * 2.0)
                            - (tape_micro_drift * 1.4)
                            - (tape_vwap_gap * 0.8)
                        )
                        price_flow_divergence = torch.clamp(blend(price_flow_divergence, tape_divergence), min=-1.0, max=1.0)
                    if family == "mean_reversion_vwap":
                        score = (
                            0.36 * torch.tanh((0.42 - rsi) * 5.4)
                            - 0.30 * torch.tanh(gap_to_vwap * 135.0)
                            - 0.16 * torch.tanh(momentum_3 * 185.0)
                            + 0.10 * torch.tanh(volume_ratio * 1.7)
                            - 0.08 * torch.tanh(ema_gap * 110.0)
                        )
                    elif family == "trend_pullback":
                        trend = (
                            0.46 * torch.tanh(momentum_20 * 95.0)
                            + 0.26 * torch.tanh(momentum_10 * 115.0)
                            + 0.16 * torch.tanh(ema_gap * 125.0)
                            + 0.12 * torch.tanh(volume_trend * 4.0)
                        )
                        pullback = (
                            -0.34 * torch.tanh(momentum_3 * 180.0)
                            -0.18 * torch.tanh(momentum_1 * 240.0)
                        )
                        score = trend + pullback + 0.10 * torch.tanh(trend_acceleration * 260.0)
                    elif family == "volatility_breakout":
                        direction = (
                            0.44 * torch.tanh(momentum_1 * 320.0)
                            + 0.34 * torch.tanh(momentum_3 * 190.0)
                            + 0.14 * torch.tanh(momentum_10 * 120.0)
                            + 0.08 * torch.tanh(trend_acceleration * 260.0)
                        )
                        expansion = 0.55 + 0.45 * torch.tanh(
                            (relative_atr + volatility_20) * 95.0 + volume_ratio * 1.1
                        )
                        score = direction * expansion
                    elif family == "volume_flow_proxy":
                        flow = (
                            0.34 * torch.tanh(volume_ratio * 2.3)
                            + 0.28 * torch.tanh(volume_trend * 4.5)
                            + 0.22 * torch.tanh(trend_acceleration * 260.0)
                        )
                        direction = (
                            0.42 * torch.tanh(momentum_1 * 300.0)
                            + 0.34 * torch.tanh(momentum_3 * 180.0)
                            + 0.24 * torch.tanh(momentum_10 * 120.0)
                        )
                        score = direction * (0.55 + 0.45 * torch.tanh(flow))
                    elif family == "order_flow_momentum":
                        flow_pressure = (
                            0.34 * torch.tanh(signed_base * 2.6)
                            + 0.30 * torch.tanh(signed_quote * 2.6)
                            + 0.16 * torch.tanh(signed_ratio_delta * 3.4)
                            + 0.10 * torch.tanh(flow_return_alignment * 2.0)
                            + 0.06 * torch.tanh(trade_impulse * 1.6)
                            + 0.04 * torch.tanh(quote_impulse * 1.6)
                        )
                        price_confirmation = (
                            0.36 * torch.tanh(momentum_1 * 260.0)
                            + 0.30 * torch.tanh(momentum_3 * 170.0)
                            + 0.18 * torch.tanh(momentum_10 * 105.0)
                            + 0.16 * torch.tanh(ema_gap * 105.0)
                        )
                        liquidity_penalty = 0.18 * torch.tanh(no_trade_ratio * 4.0)
                        score = 0.62 * flow_pressure + 0.38 * price_confirmation - liquidity_penalty
                    elif family == "flow_reversion":
                        exhaustion = (
                            0.36 * torch.tanh(signed_base * 2.4)
                            + 0.30 * torch.tanh(signed_quote * 2.4)
                            + 0.16 * torch.tanh(quote_per_trade_impulse * 1.8)
                        )
                        stretched_price = (
                            0.32 * torch.tanh(gap_to_vwap * 145.0)
                            + 0.24 * torch.tanh(momentum_3 * 170.0)
                            + 0.18 * torch.tanh((rsi - 0.50) * 4.6)
                        )
                        score = -0.58 * exhaustion * torch.abs(stretched_price) - 0.42 * stretched_price
                    elif family == "flow_consensus_breakout":
                        consensus = (
                            0.22 * torch.tanh(signed_base * 2.8)
                            + 0.20 * torch.tanh(signed_quote * 2.8)
                            + 0.16 * torch.tanh(signed_ratio_delta * 3.2)
                            + 0.14 * torch.tanh(flow_acceleration * 3.0)
                            + 0.12 * torch.tanh(flow_persistence * 2.2)
                            + 0.08 * torch.tanh(flow_return_alignment * 2.0)
                            + 0.08 * torch.tanh((taker_buy_ratio - 0.5) * 5.0)
                        )
                        price_confirmation = (
                            0.38 * torch.tanh(momentum_1 * 280.0)
                            + 0.30 * torch.tanh(momentum_3 * 190.0)
                            + 0.18 * torch.tanh(momentum_10 * 120.0)
                            + 0.14 * torch.tanh(trend_acceleration * 250.0)
                        )
                        flow_quality = 1.0 - torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                        flow_strength = 0.55 + 0.45 * torch.tanh(mean_abs_signed_ratio * 4.0)
                        score = flow_quality * flow_strength * (0.68 * consensus + 0.32 * price_confirmation)
                    elif family == "liquidity_absorption_reversal":
                        absorption = (
                            0.34 * torch.tanh(price_flow_divergence * 2.6)
                            + 0.24 * torch.tanh(signed_base * 2.0)
                            + 0.20 * torch.tanh(signed_quote * 2.0)
                            - 0.12 * torch.tanh(flow_return_alignment * 2.0)
                            + 0.10 * torch.tanh(mean_abs_signed_ratio * 3.0)
                        )
                        stretch = (
                            0.34 * torch.tanh(gap_to_vwap * 150.0)
                            + 0.24 * torch.tanh(momentum_3 * 180.0)
                            + 0.18 * torch.tanh(momentum_10 * 120.0)
                            + 0.14 * torch.tanh((rsi - 0.50) * 4.8)
                        )
                        score = -0.58 * absorption - 0.42 * stretch
                    elif family == "micro_flow_scalp":
                        flow_pressure = (
                            0.24 * torch.tanh(signed_base * 3.2)
                            + 0.22 * torch.tanh(signed_quote * 3.0)
                            + 0.18 * torch.tanh(signed_ratio_delta * 4.0)
                            + 0.14 * torch.tanh(flow_acceleration * 3.2)
                            + 0.10 * torch.tanh((taker_buy_ratio - 0.5) * 6.0)
                            + 0.08 * torch.tanh(trade_impulse * 2.0)
                            + 0.04 * torch.tanh(quote_impulse * 2.0)
                        )
                        price_tape = (
                            0.40 * torch.tanh(momentum_1 * 360.0)
                            + 0.30 * torch.tanh(momentum_3 * 220.0)
                            + 0.18 * torch.tanh(trend_acceleration * 300.0)
                            + 0.12 * torch.tanh(ema_gap * 120.0)
                        )
                        liquidity_quality = 1.0 - torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                        score = liquidity_quality * (0.68 * flow_pressure + 0.32 * price_tape)
                    elif family == "vwap_snapback_scalp":
                        stretch = (
                            0.42 * torch.tanh(gap_to_vwap * 180.0)
                            + 0.24 * torch.tanh(momentum_3 * 210.0)
                            + 0.18 * torch.tanh((rsi - 0.50) * 5.2)
                            + 0.16 * torch.tanh(ema_gap * 130.0)
                        )
                        exhaustion = (
                            0.30 * torch.tanh(price_flow_divergence * 2.8)
                            + 0.22 * torch.tanh(signed_ratio_delta * 3.0)
                            - 0.18 * torch.tanh(flow_return_alignment * 2.4)
                            + 0.16 * torch.tanh(mean_abs_signed_ratio * 3.2)
                        )
                        participation = 0.72 + 0.28 * torch.tanh(mean_abs_signed_ratio * 3.0)
                        score = -participation * (0.64 * stretch + 0.36 * exhaustion)
                    elif family == "liquidity_sweep_reversal":
                        sweep = (
                            0.28 * torch.tanh(signed_base * 3.2)
                            + 0.24 * torch.tanh(signed_quote * 3.0)
                            + 0.18 * torch.tanh(trade_impulse * 2.2)
                            + 0.16 * torch.tanh(quote_impulse * 2.0)
                            + 0.14 * torch.tanh(flow_acceleration * 2.8)
                        )
                        price_stretch = (
                            0.34 * torch.tanh(gap_to_vwap * 165.0)
                            + 0.26 * torch.tanh(momentum_3 * 190.0)
                            + 0.20 * torch.tanh(momentum_10 * 130.0)
                            + 0.20 * torch.tanh((rsi - 0.50) * 4.8)
                        )
                        divergence = torch.tanh(price_flow_divergence * 3.0)
                        score = -0.52 * sweep * torch.abs(price_stretch) - 0.30 * price_stretch - 0.18 * divergence
                    elif family == "compression_breakout_scalp":
                        direction = (
                            0.42 * torch.tanh(momentum_1 * 360.0)
                            + 0.28 * torch.tanh(momentum_3 * 230.0)
                            + 0.18 * torch.tanh(signed_base * 2.6)
                            + 0.12 * torch.tanh(flow_acceleration * 2.8)
                        )
                        compression = 1.0 - torch.tanh((relative_atr + volatility_20) * 75.0)
                        participation = (
                            0.46
                            + 0.24 * torch.tanh(volume_ratio * 1.8)
                            + 0.18 * torch.tanh(torch.abs(signed_base) * 2.4)
                            + 0.12 * torch.tanh(mean_abs_signed_ratio * 3.0)
                        )
                        liquidity_quality = 1.0 - 0.5 * torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                        score = direction * (0.55 + 0.45 * compression) * participation * liquidity_quality
                    elif family == "volume_synchronized_flow":
                        flow_direction = (
                            0.24 * torch.tanh(signed_base * 2.7)
                            + 0.22 * torch.tanh(signed_quote * 2.7)
                            + 0.16 * torch.tanh(signed_ratio_delta * 3.2)
                            + 0.14 * torch.tanh(flow_acceleration * 2.8)
                            + 0.10 * torch.tanh((taker_buy_ratio - 0.5) * 5.0)
                            + 0.08 * torch.tanh(flow_persistence * 2.0)
                            + 0.06 * torch.tanh(quote_per_trade_impulse * 1.8)
                        )
                        price_direction = (
                            0.30 * torch.tanh(momentum_1 * 280.0)
                            + 0.24 * torch.tanh(momentum_3 * 190.0)
                            + 0.18 * torch.tanh(momentum_10 * 115.0)
                            + 0.14 * torch.tanh(trend_acceleration * 240.0)
                            + 0.08 * torch.tanh(ema_gap * 105.0)
                            - 0.06 * torch.tanh(gap_to_vwap * 125.0)
                        )
                        participation = (
                            0.42
                            + 0.22 * torch.tanh(volume_ratio * 1.6)
                            + 0.18 * torch.tanh(trade_impulse * 1.8)
                            + 0.18 * torch.tanh(quote_impulse * 1.8)
                        )
                        synchronization = 0.50 + 0.50 * torch.tanh(flow_return_alignment * 2.4)
                        flow_strength = 0.58 + 0.42 * torch.tanh(mean_abs_signed_ratio * 3.4)
                        liquidity_quality = 1.0 - torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                        divergence_penalty = 0.18 * torch.tanh(torch.abs(price_flow_divergence) * 2.8)
                        penalty_direction = torch.sign(torch.where(flow_direction != 0.0, flow_direction, price_direction))
                        score = (
                            liquidity_quality
                            * flow_strength
                            * (0.62 * synchronization * flow_direction + 0.38 * price_direction)
                            * (0.72 + 0.28 * participation)
                        ) - divergence_penalty * penalty_direction
                    elif family == "adaptive_tape_regime":
                        trend = (
                            0.30 * torch.tanh(momentum_1 * 310.0)
                            + 0.24 * torch.tanh(momentum_3 * 210.0)
                            + 0.18 * torch.tanh(momentum_10 * 130.0)
                            + 0.18 * torch.tanh(signed_base * 2.8)
                            + 0.10 * torch.tanh(flow_acceleration * 2.8)
                        )
                        reversion = -(
                            0.36 * torch.tanh(gap_to_vwap * 155.0)
                            + 0.24 * torch.tanh(momentum_3 * 175.0)
                            + 0.20 * torch.tanh((rsi - 0.50) * 4.6)
                            + 0.20 * torch.tanh(price_flow_divergence * 2.4)
                        )
                        persistence = torch.tanh(flow_persistence * 2.4 + flow_return_alignment * 1.8)
                        trend_weight = 0.5 + 0.5 * persistence
                        score = trend_weight * trend + (1.0 - trend_weight) * reversion
                    elif family == "higher_timeframe_alignment":
                        if not htf_groups:
                            score = torch.zeros_like(momentum_1)
                        else:
                            broad_direction = (
                                0.30 * torch.tanh(htf_return * 105.0)
                                + 0.22 * torch.tanh(htf_mean_gap * 125.0)
                                + 0.14 * torch.tanh(htf_bounce * 32.0)
                                + 0.14 * torch.tanh(htf_drawdown * 32.0)
                                + 0.10 * torch.tanh(htf_volume_impulse * 1.5)
                                + 0.10 * torch.tanh(htf_trade_impulse * 1.5)
                            )
                            local_direction = (
                                0.24 * torch.tanh(momentum_1 * 300.0)
                                + 0.22 * torch.tanh(momentum_3 * 210.0)
                                + 0.18 * torch.tanh(momentum_10 * 130.0)
                                + 0.14 * torch.tanh(ema_gap * 115.0)
                                + 0.12 * torch.tanh(signed_base * 2.4)
                                + 0.10 * torch.tanh(flow_acceleration * 2.4)
                            )
                            same_direction = broad_direction * local_direction
                            alignment = 0.42 + 0.58 * (0.5 + 0.5 * torch.tanh(same_direction * 5.0))
                            volatility_penalty = 0.35 * torch.tanh(
                                htf_realized_volatility * 130.0 + htf_range * 32.0
                            )
                            participation = (
                                0.64
                                + 0.18 * torch.tanh(htf_volume_impulse * 1.4)
                                + 0.18 * torch.tanh(volume_ratio * 1.8)
                            )
                            flow_quality = 1.0 - 0.5 * torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                            score = (
                                alignment
                                * torch.clamp(1.0 - volatility_penalty, min=0.20)
                                * participation
                                * flow_quality
                                * (0.54 * local_direction + 0.46 * broad_direction)
                            )
                    elif family == "directional_regime_rider":
                        broad_direction = (
                            0.28 * torch.tanh(momentum_20 * 115.0)
                            + 0.20 * torch.tanh(momentum_10 * 135.0)
                            + 0.16 * torch.tanh(ema_gap * 125.0)
                        )
                        if htf_groups:
                            broad_direction = (
                                0.42 * torch.tanh(htf_return * 130.0)
                                + 0.18 * torch.tanh(htf_mean_gap * 150.0)
                                + 0.12 * torch.tanh(htf_volume_impulse * 1.4)
                                + 0.12 * torch.tanh(htf_trade_impulse * 1.4)
                                + 0.16 * broad_direction
                            )
                        local_confirmation = (
                            0.30 * torch.tanh(momentum_3 * 220.0)
                            + 0.20 * torch.tanh(momentum_1 * 320.0)
                            + 0.18 * torch.tanh(signed_base * 2.6)
                            + 0.14 * torch.tanh(flow_acceleration * 2.8)
                            + 0.10 * torch.tanh(flow_persistence * 2.0)
                            + 0.08 * torch.tanh((taker_buy_ratio - 0.5) * 5.0)
                        )
                        agreement = 0.55 + 0.45 * torch.tanh((broad_direction * local_confirmation) * 5.0)
                        liquidity_quality = 1.0 - torch.clamp(no_trade_ratio, min=0.0, max=1.0)
                        volatility_drag = 0.30 * torch.tanh(
                            (relative_atr + volatility_20 + htf_realized_volatility) * 90.0
                        )
                        score = (
                            torch.clamp(1.0 - volatility_drag, min=0.25)
                            * (0.70 * broad_direction + 0.30 * local_confirmation)
                            * (0.70 + 0.30 * agreement)
                            * (0.55 + 0.45 * liquidity_quality)
                        )
                    else:
                        score = (
                            0.32 * torch.tanh(momentum_20 * 90.0)
                            + 0.27 * torch.tanh(momentum_10 * 115.0)
                            + 0.18 * torch.tanh(momentum_3 * 165.0)
                            - 0.11 * torch.tanh(ema_spread * 95.0)
                            + 0.08 * torch.tanh(volume_ratio * 2.0)
                            + 0.04 * torch.tanh((relative_atr + volatility_20) * 85.0)
                        )
                    magnitude = torch.abs(score)
                    adjusted = torch.where(
                        magnitude <= deadband,
                        torch.zeros_like(score),
                        torch.sign(score) * ((magnitude - deadband) / max(1e-9, 1.0 - deadband)),
                    )
                    adjusted = torch.clamp(adjusted, min=-1.0, max=1.0)
                    expert_probs = torch.clamp(torch.sigmoid(adjusted * sensitivity + bias), min=0.0, max=1.0)
                if expert_probs is None:
                    raise RuntimeError(f"Accelerated scorer produced no output for expert kind {kind}")
                weighted = weighted + float(expert_weight) * expert_probs
                total = total + float(expert_weight)
            chunk_probs = weighted / torch.clamp(total, min=1e-12)
        if bool(getattr(model, "probability_inverted", False)):
            chunk_probs = 1.0 - chunk_probs
        chunk_probs = torch.clamp(chunk_probs, min=0.0, max=1.0)
        probabilities.extend(float(value) for value in chunk_probs.detach().cpu().tolist())
    return probabilities


def _backtest_probabilities(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    compute_backend: str | None,
    batch_size: int,
) -> tuple[list[float], BackendInfo]:
    backend = resolve_backend(effective_training_backend_name(compute_backend))
    if backend.kind == "cpu":
        return [model.predict_proba(row.features) for row in rows], backend
    try:
        return _batch_probabilities_torch(rows, model, backend=backend, batch_size=batch_size), backend
    except Exception as exc:
        fallback = _fallback_score_backend(
            backend,
            f"{backend.kind} backtest scoring failed ({exc.__class__.__name__}); fell back to CPU",
        )
        return [model.predict_proba(row.features) for row in rows], fallback


def _threshold_grid(start: float, end: float, steps: int, baseline: float) -> list[float]:
    if steps <= 1:
        return [_clamp_threshold(baseline)]
    start = _clamp_threshold(float(start))
    end = _clamp_threshold(float(end))
    if end <= start:
        end = min(1.0, start + 0.01)
    values = [start + (end - start) * i / (steps - 1) for i in range(steps)]
    values.append(_clamp_threshold(baseline))
    return sorted(set(round(_clamp_threshold(value), 12) for value in values))


def _probability_adaptive_threshold_grid(
    probabilities: Sequence[float],
    *,
    start: float,
    end: float,
    baseline: float,
    market_type: str,
    max_thresholds: int,
    base_thresholds: Sequence[float] = (),
) -> list[float]:
    """Add model-rank thresholds so calibration searches active trade bands.

    The fixed grid is stable and reproducible, but it can be almost empty when
    a model's scores are tightly compressed.  This helper adds cutoffs at the
    empirical score tails that would actually produce entries, while preserving
    the original grid and the caller's baseline.
    """

    clean = sorted(
        _clamp_threshold(float(value))
        for value in probabilities
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    )
    start = _clamp_threshold(start)
    end = _clamp_threshold(end)
    if str(market_type).lower() == "futures":
        start = max(0.5, start)
    if end <= start:
        end = min(1.0, start + 0.01)
    thresholds = {round(_clamp_threshold(value), 12) for value in base_thresholds}
    thresholds.add(round(_clamp_threshold(baseline), 12))
    if not clean:
        return sorted(thresholds)
    row_count = len(clean)
    tail_rates = (
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
    )
    for rate in tail_rates:
        tail_count = max(1, min(row_count, int(math.ceil(row_count * rate))))
        long_threshold = clean[max(0, row_count - tail_count)]
        if start <= long_threshold <= end:
            thresholds.add(round(_clamp_threshold(long_threshold), 12))
        if str(market_type).lower() == "futures":
            short_probability_cutoff = clean[min(row_count - 1, tail_count - 1)]
            symmetric_threshold = 1.0 - short_probability_cutoff
            if start <= symmetric_threshold <= end:
                thresholds.add(round(_clamp_threshold(symmetric_threshold), 12))
    ordered = sorted(thresholds)
    limit = max(1, int(max_thresholds))
    if len(ordered) <= limit:
        return ordered
    keep = {ordered[0], ordered[-1], round(_clamp_threshold(baseline), 12)}
    remaining = max(0, limit - len(keep))
    if remaining > 0:
        denominator = max(1, remaining - 1)
        for slot in range(remaining):
            index = int(round(slot * (len(ordered) - 1) / denominator))
            keep.add(ordered[min(len(ordered) - 1, max(0, index))])
    if len(keep) < limit:
        for value in ordered:
            keep.add(value)
            if len(keep) >= limit:
                break
    return sorted(keep)


def _approximate_threshold_signal_count(
    probabilities: Sequence[float],
    *,
    cfg: StrategyConfig,
    market_type: str,
    long_threshold: float | None,
    short_threshold: float | None,
    regime_scores: Sequence[float] | None = None,
    liquidity_adjustments: Sequence[tuple[float, float, bool, bool]] | None = None,
    timestamps: Sequence[int] | None = None,
) -> int:
    """Cheaply count rows that could enter before running full lifecycle replay."""

    count = 0
    futures = str(market_type).lower() == "futures"
    max_daily: int | None = int(cfg.max_trades_per_day)
    if max_daily <= 0:
        max_daily = None
    daily_counts: dict[int, int] = {}
    cooldown_ms = max(0, int(cfg.cooldown_minutes)) * 60 * 1000
    last_entry_timestamp: int | None = None
    regime_cooldown_ms = max(0, int(cfg.unpredictability_cooldown_minutes)) * 60 * 1000
    regime_cooldown_until: int | None = None
    regime_limit = max(0.0, min(1.0, float(cfg.max_regime_unpredictability)))
    for index, raw_probability in enumerate(probabilities):
        timestamp = (
            int(timestamps[index])
            if timestamps is not None and index < len(timestamps)
            else int(index)
        )
        if regime_cooldown_until is not None and timestamp < regime_cooldown_until:
            continue
        regime_score = 0.0
        if regime_scores is not None and index < len(regime_scores):
            regime_score = max(0.0, min(1.0, _finite_float(regime_scores[index], 1.0)))
        threshold_add = 0.0
        size_multiplier = 1.0
        if liquidity_adjustments is not None and index < len(liquidity_adjustments):
            threshold_add = _finite_float(liquidity_adjustments[index][0], 0.0)
            size_multiplier = _finite_float(liquidity_adjustments[index][1], 1.0)
            if size_multiplier <= 0.0:
                continue
        adjusted_long, adjusted_short = _adjusted_side_thresholds(
            long_threshold,
            short_threshold,
            threshold_add,
        )
        probability = confidence_adjusted_probability(_finite_float(raw_probability, 0.5), cfg.confidence_beta)
        direction = _normalize_market_direction(
            probability,
            adjusted_long,
            market_type,
            short_threshold=adjusted_short,
        )
        if direction == 0:
            continue
        if not futures and direction < 0:
            continue
        if regime_score > regime_limit:
            if (
                regime_cooldown_ms > 0
                and regime_unpredictability_requires_cooldown(regime_score, regime_limit)
            ):
                regime_cooldown_until = max(
                    int(regime_cooldown_until or timestamp),
                    int(timestamp) + regime_cooldown_ms,
                )
                continue
            regime_size_multiplier = _regime_soft_gate_size_multiplier(
                regime_score=regime_score,
                regime_limit=regime_limit,
                signal_score=probability,
                direction=direction,
                long_threshold=adjusted_long,
                short_threshold=adjusted_short,
                market_type=market_type,
            )
            if regime_size_multiplier <= 0.0:
                continue
        if cooldown_ms > 0 and last_entry_timestamp is not None and timestamp - last_entry_timestamp < cooldown_ms:
            continue
        day = _safe_day(timestamp)
        if max_daily is not None and daily_counts.get(day, 0) >= max_daily:
            continue
        daily_counts[day] = daily_counts.get(day, 0) + 1
        last_entry_timestamp = timestamp
        if long_threshold is not None and direction > 0:
            count += 1
        elif futures and short_threshold is not None and direction < 0:
            count += 1
    return count


def _result_payload(result: BacktestResult) -> dict[str, float | int | bool]:
    return {
        "realized_pnl": float(result.realized_pnl),
        "total_fees": float(result.total_fees),
        "max_drawdown": float(result.max_drawdown),
        "win_rate": float(result.win_rate),
        "closed_trades": int(result.closed_trades),
        "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
        "stopped_by_liquidation": bool(getattr(result, "stopped_by_liquidation", False)),
        "liquidation_events": int(getattr(result, "liquidation_events", 0)),
        "liquidation_loss": float(getattr(result, "liquidation_loss", 0.0)),
        "profit_factor": float(getattr(result, "profit_factor", 0.0)),
        "expectancy": float(getattr(result, "expectancy", 0.0)),
        "average_trade_return": float(getattr(result, "average_trade_return", 0.0)),
        "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0)),
        "regime_entry_skips": int(getattr(result, "regime_entry_skips", 0)),
        "regime_entry_downsizes": int(getattr(result, "regime_entry_downsizes", 0)),
    }


def precompute_backtest_regime_scores(rows: Sequence[ModelRow], cfg: StrategyConfig) -> list[float]:
    row_list = list(rows)
    if not row_list:
        return []
    closes = [_finite_float(getattr(row, "close", float("nan")), float("nan")) for row in row_list]
    if any(not math.isfinite(value) or value <= 0.0 for value in closes):
        return _precompute_backtest_regime_scores_slow(row_list, cfg)
    regime_gate_min_rows = max(8, min(len(row_list), int(cfg.liquidity_lookback_bars)))
    lookback = max(8, int(cfg.liquidity_lookback_bars))
    scores: list[float] = [0.0] * len(row_list)
    returns = [0.0] * len(row_list)
    for index in range(1, len(closes)):
        returns[index] = (closes[index] / closes[index - 1]) - 1.0 if closes[index - 1] > 0.0 else 0.0

    def prefix(values: Sequence[float]) -> list[float]:
        total = 0.0
        output = [0.0]
        for value in values:
            total += float(value)
            output.append(total)
        return output

    def range_sum(values: Sequence[float], start: int, end: int) -> float:
        if end < start:
            return 0.0
        return values[end + 1] - values[start]

    return_prefix = prefix(returns)
    return_square_prefix = prefix([value * value for value in returns])
    abs_return_prefix = prefix([abs(value) for value in returns])
    adjacent_product_prefix = prefix([
        returns[index] * returns[index + 1] if 1 <= index < len(returns) - 1 else 0.0
        for index in range(len(returns))
    ])

    for row_index in range(len(row_list)):
        if row_index + 1 < regime_gate_min_rows:
            continue
        regime_window_start = max(0, row_index + 1 - lookback)
        return_start = regime_window_start + 1
        return_end = row_index
        return_count = max(0, return_end - return_start + 1)
        if return_count <= 0:
            scores[row_index] = market_regime_unpredictability("insufficient_data", 0.0, ("no_valid_returns",))
            continue
        return_sum = range_sum(return_prefix, return_start, return_end)
        return_square_sum = range_sum(return_square_prefix, return_start, return_end)
        abs_return_sum = range_sum(abs_return_prefix, return_start, return_end)
        trend_return = (closes[row_index] / closes[regime_window_start]) - 1.0
        if return_count < 2:
            volatility = 0.0
        else:
            variance = (return_square_sum - (return_sum * return_sum / return_count)) / max(1, return_count - 1)
            volatility = math.sqrt(max(0.0, variance))
        mean_abs = abs_return_sum / return_count

        positive = 0
        negative = 0
        reversals = 0
        previous_sign = 0
        for return_index in range(return_start, return_end + 1):
            sign = 1 if returns[return_index] > 0.0 else (-1 if returns[return_index] < 0.0 else 0)
            if sign == 0:
                continue
            if sign > 0:
                positive += 1
            else:
                negative += 1
            if previous_sign and sign != previous_sign:
                reversals += 1
            previous_sign = sign
        sign_count = positive + negative
        direction_consistency = max(positive, negative) / sign_count if sign_count else 0.0
        reversal_rate = reversals / max(1, sign_count - 1) if sign_count else 0.0

        if return_count < 3:
            autocorrelation = 0.0
        else:
            pair_count = return_count - 1
            left_start = return_start
            left_end = return_end - 1
            right_start = return_start + 1
            right_end = return_end
            left_sum = range_sum(return_prefix, left_start, left_end)
            right_sum = range_sum(return_prefix, right_start, right_end)
            left_square_sum = range_sum(return_square_prefix, left_start, left_end)
            right_square_sum = range_sum(return_square_prefix, right_start, right_end)
            pair_sum = range_sum(adjacent_product_prefix, left_start, left_end)
            left_mean = left_sum / pair_count
            right_mean = right_sum / pair_count
            covariance = pair_sum - left_mean * right_sum - right_mean * left_sum + pair_count * left_mean * right_mean
            left_var = left_square_sum - 2.0 * left_mean * left_sum + pair_count * left_mean * left_mean
            right_var = right_square_sum - 2.0 * right_mean * right_sum + pair_count * right_mean * right_mean
            denominator = math.sqrt(max(0.0, left_var * right_var))
            autocorrelation = 0.0 if denominator <= 1e-12 else max(-1.0, min(1.0, covariance / denominator))

        noise_floor = max(1e-9, mean_abs * math.sqrt(return_count), volatility * math.sqrt(return_count))
        trend_strength = abs(trend_return) / noise_floor
        volatility_floor = max(0.0005, mean_abs * 1.8)
        notes: list[str] = []
        if volatility >= volatility_floor and reversal_rate >= 0.55:
            regime = "volatile_chop"
            confidence = min(1.0, 0.45 + 0.35 * reversal_rate + 0.20 * min(2.0, volatility / volatility_floor) / 2.0)
        elif trend_strength >= 1.15 and direction_consistency >= 0.55:
            regime = "trend_up" if trend_return > 0.0 else "trend_down"
            confidence = min(1.0, 0.40 + 0.35 * min(2.0, trend_strength) / 2.0 + 0.25 * direction_consistency)
        elif reversal_rate >= 0.50 and trend_strength < 0.85:
            regime = "range_bound"
            confidence = min(1.0, 0.45 + 0.35 * reversal_rate + 0.20 * (1.0 - min(1.0, trend_strength)))
        elif abs(autocorrelation) >= 0.35:
            regime = "serial_correlation"
            confidence = min(1.0, 0.50 + 0.50 * abs(autocorrelation))
        else:
            regime = "mixed"
            confidence = max(0.20, min(0.70, 0.35 + 0.20 * direction_consistency + 0.15 * min(1.0, trend_strength)))
            notes.append("low_regime_separation")
        if mean_abs <= 1e-9:
            notes.append("flat_returns")
        if return_count < 10:
            notes.append("short_window")
        scores[row_index] = market_regime_unpredictability(regime, confidence, notes)
    return scores


def _precompute_backtest_regime_scores_slow(rows: Sequence[ModelRow], cfg: StrategyConfig) -> list[float]:
    row_list = list(rows)
    if not row_list:
        return []
    regime_gate_min_rows = max(8, min(len(row_list), int(cfg.liquidity_lookback_bars)))
    lookback = max(8, int(cfg.liquidity_lookback_bars))
    scores: list[float] = [0.0] * len(row_list)
    for row_index in range(len(row_list)):
        if row_index + 1 < regime_gate_min_rows:
            continue
        regime_window_start = max(0, row_index + 1 - lookback)
        regime_evidence = classify_market_regime(row_list[regime_window_start:row_index + 1])
        scores[row_index] = market_regime_unpredictability(
            regime_evidence.dominant_regime,
            regime_evidence.confidence,
            regime_evidence.notes,
        )
    return scores


def precompute_backtest_liquidity_adjustments(
    rows: Sequence[ModelRow],
    cfg: StrategyConfig,
) -> list[tuple[float, float, bool, bool]]:
    """Cache threshold-neutral liquidity/session flags for repeated replays."""

    row_list = list(rows)
    base_threshold = 0.5
    adjustments: list[tuple[float, float, bool, bool]] = []
    for row_index in range(len(row_list)):
        adjustment = liquidity_session_adjustment(row_list, row_index, cfg, base_threshold)
        threshold_add = max(-1.0, min(1.0, float(adjustment.threshold) - base_threshold))
        adjustments.append((
            threshold_add,
            float(adjustment.size_multiplier),
            bool(adjustment.low_liquidity),
            bool(adjustment.low_dynamic_session),
        ))
    return adjustments


def calibrate_threshold_for_backtest(
    rows: List[ModelRow],
    model: TrainedModel,
    cfg: StrategyConfig,
    *,
    starting_cash: float = 1000.0,
    market_type: str = "spot",
    baseline_threshold: float | None = None,
    start: float = 0.05,
    end: float = 0.95,
    steps: int = 31,
    min_score_delta: float = 0.0,
    min_closed_trades: int = 1,
    min_trades_per_day: float = 0.0,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    symbol_profile: SymbolExecutionProfile | None = None,
    adaptive_probability_thresholds: bool = False,
    max_adaptive_thresholds: int = 96,
    allowed_sides: str = "both",
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> ThresholdBacktestCalibration:
    baseline_threshold = _clamp_threshold(
        _finite_float(baseline_threshold, model_decision_threshold(model, cfg.signal_threshold))
        if baseline_threshold is not None
        else model_decision_threshold(model, cfg.signal_threshold)
    )
    if market_type == "futures":
        baseline_threshold = max(0.5, baseline_threshold)
        start = max(0.5, float(start))
    probabilities, score_backend = _backtest_probabilities(
        rows,
        model,
        compute_backend=compute_backend,
        batch_size=score_batch_size,
    )
    if status_callback is not None:
        status_callback(
            "threshold_probability_scoring_complete",
            {
                "rows": int(len(rows)),
                "probability_count": int(len(probabilities)),
                "scoring_backend_kind": str(score_backend.kind),
                "scoring_backend_device": str(score_backend.device),
            },
        )
    regime_scores = precompute_backtest_regime_scores(rows, cfg)
    liquidity_adjustments = precompute_backtest_liquidity_adjustments(rows, cfg)
    if status_callback is not None:
        status_callback(
            "threshold_precompute_complete",
            {
                "rows": int(len(rows)),
                "regime_scores": int(len(regime_scores)),
                "liquidity_adjustments": int(len(liquidity_adjustments)),
            },
        )
    baseline_long, baseline_short = model_direction_thresholds(
        replace(model, decision_threshold=baseline_threshold),
        cfg.signal_threshold,
        market_type=market_type,
    )
    baseline_model = replace(
        model,
        decision_threshold=baseline_threshold,
        long_decision_threshold=getattr(model, "long_decision_threshold", None),
        short_decision_threshold=getattr(model, "short_decision_threshold", None),
    )
    baseline_result = run_backtest(
        rows,
        baseline_model,
        cfg,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=symbol_profile,
        precomputed_probabilities=probabilities,
        precomputed_score_backend=score_backend,
        precomputed_regime_scores=regime_scores,
        precomputed_liquidity_adjustments=liquidity_adjustments,
    )
    span_days = row_span_days(rows)
    min_closed = max(0, int(min_closed_trades))
    min_daily = max(0.0, _finite_float(min_trades_per_day, 0.0))
    baseline_score = threshold_backtest_selection_score(
        baseline_result,
        starting_cash=starting_cash,
        min_closed_trades=min_closed,
        min_trades_per_day=min_daily,
        duration_days=span_days,
    )
    if status_callback is not None:
        status_callback(
            "threshold_baseline_complete",
            {
                "baseline_realized_pnl": float(baseline_result.realized_pnl),
                "baseline_closed_trades": int(baseline_result.closed_trades),
                "baseline_score": float(baseline_score),
            },
        )
    best_threshold = _threshold_confidence(baseline_long, baseline_short, baseline_threshold)
    best_long_threshold = baseline_long
    best_short_threshold = baseline_short
    best_score = baseline_score
    best_result = baseline_result
    fixed_thresholds = _threshold_grid(start, end, steps, baseline_threshold)
    thresholds = (
        _probability_adaptive_threshold_grid(
            probabilities,
            start=start,
            end=end,
            baseline=baseline_threshold,
            market_type=market_type,
            max_thresholds=max_adaptive_thresholds,
            base_thresholds=fixed_thresholds,
        )
        if adaptive_probability_thresholds
        else fixed_thresholds
    )
    seen_variants: set[tuple[float | None, float | None]] = set()
    replayed_variants = 0
    skipped_sparse_variants = 0
    required_signal_floor = max(min_closed, int(math.ceil(min_daily * span_days)) if min_daily > 0.0 else 0)
    total_thresholds = len(thresholds)
    row_timestamps = [int(row.timestamp) for row in rows]
    if status_callback is not None:
        status_callback(
            "threshold_grid_complete",
            {
                "threshold_count": int(total_thresholds),
                "required_signal_floor": int(required_signal_floor),
                "market_type": str(market_type),
                "allowed_sides": str(allowed_sides or "both"),
            },
        )
    for threshold in thresholds:
        if market_type == "futures":
            threshold = max(0.5, float(threshold))
            side_mode = str(allowed_sides or "both").strip().lower()
            if side_mode == "long":
                variants = ((threshold, None),)
            elif side_mode == "short":
                variants = ((None, 1.0 - threshold),)
            else:
                variants = (
                    (threshold, 1.0 - threshold),
                    (threshold, None),
                    (None, 1.0 - threshold),
                )
        else:
            variants = ((float(threshold), None),)
        for long_threshold, short_threshold in variants:
            key = (
                round(long_threshold, 12) if long_threshold is not None else None,
                round(short_threshold, 12) if short_threshold is not None else None,
            )
            if key in seen_variants:
                continue
            seen_variants.add(key)
            if required_signal_floor > 0:
                approximate_signals = _approximate_threshold_signal_count(
                    probabilities,
                    cfg=cfg,
                    market_type=market_type,
                    long_threshold=long_threshold if market_type == "futures" else float(threshold),
                    short_threshold=short_threshold if market_type == "futures" else None,
                    regime_scores=regime_scores,
                    liquidity_adjustments=liquidity_adjustments,
                    timestamps=row_timestamps,
                )
                if approximate_signals < required_signal_floor:
                    skipped_sparse_variants += 1
                    continue
            candidate_model = replace(
                model,
                decision_threshold=_threshold_confidence(long_threshold, short_threshold, threshold),
                long_decision_threshold=long_threshold if market_type == "futures" else None,
                short_decision_threshold=short_threshold if market_type == "futures" else None,
            )
            replayed_variants += 1
            if status_callback is not None and (replayed_variants == 1 or replayed_variants % 5 == 0):
                status_callback(
                    "threshold_calibration_progress",
                    {
                        "threshold_count": int(total_thresholds),
                        "seen_variants": int(len(seen_variants)),
                        "replayed_variants": int(replayed_variants),
                        "skipped_sparse_variants": int(skipped_sparse_variants),
                        "current_threshold": float(threshold),
                        "long_threshold": long_threshold,
                        "short_threshold": short_threshold,
                    },
                )
            result = run_backtest(
                rows,
                candidate_model,
                cfg,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                symbol_profile=symbol_profile,
                precomputed_probabilities=probabilities,
                precomputed_score_backend=score_backend,
                precomputed_regime_scores=regime_scores,
                precomputed_liquidity_adjustments=liquidity_adjustments,
            )
            score = threshold_backtest_selection_score(
                result,
                starting_cash=starting_cash,
                min_closed_trades=min_closed,
                min_trades_per_day=min_daily,
                duration_days=span_days,
            )
            if (
                score > best_score + 1e-12
                or (
                    abs(score - best_score) <= 1e-12
                    and result.realized_pnl > best_result.realized_pnl
                )
            ):
                best_threshold = _threshold_confidence(long_threshold, short_threshold, threshold)
                best_long_threshold = long_threshold if market_type == "futures" else None
                best_short_threshold = short_threshold if market_type == "futures" else None
                best_score = score
                best_result = result

    best_trades_per_day = closed_trades_per_day(best_result, duration_days=span_days)
    has_profit_backed_result = best_result.closed_trades > 0 and best_result.realized_pnl > 0.0
    has_trade_density = trade_activity_satisfies(
        best_result,
        min_closed_trades=min_closed,
        min_trades_per_day=min_daily,
        duration_days=span_days,
    )
    has_no_liquidation = (
        not bool(getattr(best_result, "stopped_by_liquidation", False))
        and int(getattr(best_result, "liquidation_events", 0)) <= 0
    )
    accepted = (
        has_profit_backed_result
        and has_trade_density
        and has_no_liquidation
        and best_score > baseline_score + max(0.0, _finite_float(min_score_delta, 0.0))
    )
    selected_threshold = best_threshold if accepted else _threshold_confidence(
        baseline_long,
        baseline_short,
        baseline_threshold,
    )
    selected_long_threshold = best_long_threshold if accepted else baseline_long
    selected_short_threshold = best_short_threshold if accepted else baseline_short
    selected_score = best_score if accepted else baseline_score
    selected_result = best_result if accepted else baseline_result
    payload = _result_payload(selected_result)
    best_payload = _result_payload(best_result)
    selected_trades_per_day = closed_trades_per_day(selected_result, duration_days=span_days)
    return ThresholdBacktestCalibration(
        threshold=float(selected_threshold),
        accepted=bool(accepted),
        score=float(selected_score),
        realized_pnl=float(payload["realized_pnl"]),
        total_fees=float(payload["total_fees"]),
        max_drawdown=float(payload["max_drawdown"]),
        win_rate=float(payload["win_rate"]),
        closed_trades=int(payload["closed_trades"]),
        edge_vs_buy_hold=float(payload["edge_vs_buy_hold"]),
        baseline_threshold=float(baseline_threshold),
        baseline_score=float(baseline_score),
        baseline_realized_pnl=float(baseline_result.realized_pnl),
        baseline_closed_trades=int(baseline_result.closed_trades),
        best_threshold=float(best_threshold),
        best_score=float(best_score),
        best_realized_pnl=float(best_payload["realized_pnl"]),
        best_total_fees=float(best_payload["total_fees"]),
        best_max_drawdown=float(best_payload["max_drawdown"]),
        best_win_rate=float(best_payload["win_rate"]),
        best_closed_trades=int(best_payload["closed_trades"]),
        best_edge_vs_buy_hold=float(best_payload["edge_vs_buy_hold"]),
        evaluated_thresholds=int(replayed_variants),
        rows=len(rows),
        stopped_by_liquidation=bool(payload["stopped_by_liquidation"]),
        liquidation_events=int(payload["liquidation_events"]),
        liquidation_loss=float(payload["liquidation_loss"]),
        best_stopped_by_liquidation=bool(best_payload["stopped_by_liquidation"]),
        best_liquidation_events=int(best_payload["liquidation_events"]),
        best_liquidation_loss=float(best_payload["liquidation_loss"]),
        scoring_backend_requested=str(score_backend.requested),
        scoring_backend_kind=str(score_backend.kind),
        scoring_backend_device=str(score_backend.device),
        scoring_backend_reason=str(score_backend.reason),
        min_closed_trades=int(min_closed),
        min_trades_per_day=float(min_daily),
        trades_per_day=float(selected_trades_per_day),
        best_trades_per_day=float(best_trades_per_day),
        long_threshold=selected_long_threshold,
        short_threshold=selected_short_threshold,
        best_long_threshold=best_long_threshold,
        best_short_threshold=best_short_threshold,
    )


def run_backtest(
    rows: List[ModelRow],
    model: TrainedModel,
    cfg: StrategyConfig,
    *,
    starting_cash: float = 1000.0,
    market_type: str = "spot",
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    symbol_profile: SymbolExecutionProfile | None = None,
    precomputed_probabilities: Sequence[float] | None = None,
    precomputed_score_backend: BackendInfo | None = None,
    precomputed_regime_scores: Sequence[float] | None = None,
    precomputed_liquidity_adjustments: Sequence[tuple[float, float, bool, bool]] | None = None,
) -> BacktestResult:
    score_backend = resolve_backend(effective_training_backend_name(compute_backend))
    if not rows:
        return BacktestResult(
            starting_cash=starting_cash,
            ending_cash=starting_cash,
            realized_pnl=0.0,
            win_rate=0.0,
            trades=0,
            max_drawdown=0.0,
            stopped_by_drawdown=False,
            closed_trades=0,
            gross_exposure=0.0,
            total_fees=0.0,
            max_exposure=0.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=0.0,
            **_score_backend_payload(score_backend),
        )

    cash = float(starting_cash)
    equity_peak = cash
    max_drawdown = 0.0
    stopped_by_drawdown = False
    wins = 0
    closed_trades = 0
    total_fees = 0.0
    max_exposure = 0.0
    cap_hits = 0
    equity_curve: list[dict[str, float | int]] = []
    trade_pnls: list[float] = []
    trade_returns: list[float] = []
    trade_log: list[dict[str, object]] = []
    meta_label_skips = 0
    meta_label_downsizes = 0
    regime_entry_skips = 0
    regime_entry_downsizes = 0
    stopped_by_liquidation = False
    liquidation_events = 0
    liquidation_loss = 0.0

    position_side = 0
    notional = 0.0
    qty = 0.0
    entry_price = 0.0
    margin_used = 0.0
    entry_fee_paid = 0.0
    entry_execution_cost_bps = 0.0
    entry_equity_reference = cash
    entry_timestamp = int(rows[0].timestamp)
    entry_row_index = 0
    flat_signal_streak = 0
    entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "initial")
    pending_signal = 0
    pending_signal_score = 0.5
    pending_meta = MetaLabelDecision(False, "no_signal", 0.0, 0.0, "initial")
    last_close_timestamp: int | None = None
    cooldown_ms = max(0, int(cfg.cooldown_minutes)) * 60 * 1000
    min_position_hold_bars = max(0, int(getattr(cfg, "min_position_hold_bars", 0) or 0))
    flat_signal_exit_grace_bars = max(0, int(getattr(cfg, "flat_signal_exit_grace_bars", 0) or 0))
    max_position_hold_bars = max(0, int(getattr(cfg, "max_position_hold_bars", 0) or 0))
    unpredictability_cooldown_ms = max(0, int(cfg.unpredictability_cooldown_minutes)) * 60 * 1000
    regime_cooldown_until: int | None = None
    final_mark_price = rows[-1].close

    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    leverage = 1.0 if market_type == "spot" else cfg.leverage
    if leverage < 1:
        leverage = 1.0
    if market_type == "futures" and leverage > MAX_AUTONOMOUS_LEVERAGE:
        leverage = MAX_AUTONOMOUS_LEVERAGE
    decision_threshold = model_decision_threshold(model, cfg.signal_threshold)
    long_threshold, short_threshold = model_direction_thresholds(model, cfg.signal_threshold, market_type=market_type)

    daily_trade_count: Dict[int, int] = {}
    max_daily: int | None = int(cfg.max_trades_per_day)
    if max_daily <= 0:
        max_daily = None

    max_open_positions = int(cfg.max_open_positions)
    regime_gate_min_rows = max(8, min(len(rows), int(cfg.liquidity_lookback_bars)))
    if precomputed_probabilities is not None:
        probabilities = precomputed_probabilities
        if len(probabilities) != len(rows):
            raise ValueError(
                f"precomputed_probabilities length mismatch: {len(probabilities)}/{len(rows)}"
            )
        if precomputed_score_backend is not None:
            score_backend = precomputed_score_backend
    else:
        probabilities, score_backend = _backtest_probabilities(
            rows,
            model,
            compute_backend=compute_backend,
            batch_size=score_batch_size,
        )
    if precomputed_regime_scores is not None:
        regime_scores = precomputed_regime_scores
        if len(regime_scores) != len(rows):
            raise ValueError(
                f"precomputed_regime_scores length mismatch: {len(regime_scores)}/{len(rows)}"
            )
    else:
        regime_scores = precompute_backtest_regime_scores(rows, cfg)
    if precomputed_liquidity_adjustments is not None:
        liquidity_adjustments = precomputed_liquidity_adjustments
        if len(liquidity_adjustments) != len(rows):
            raise ValueError(
                f"precomputed_liquidity_adjustments length mismatch: {len(liquidity_adjustments)}/{len(rows)}"
            )
    else:
        liquidity_adjustments = precompute_backtest_liquidity_adjustments(rows, cfg)

    for row_index, (row, raw_score) in enumerate(zip(rows, probabilities, strict=True)):
        entry_opened_this_bar = False
        execution_signal = pending_signal
        execution_signal_score = pending_signal_score
        execution_meta = pending_meta
        score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        threshold_add, size_multiplier, low_liquidity, low_dynamic_session = liquidity_adjustments[row_index]
        adjusted_long_threshold, adjusted_short_threshold = _adjusted_side_thresholds(
            long_threshold,
            short_threshold,
            threshold_add,
        )
        display_threshold = _threshold_confidence(
            adjusted_long_threshold,
            adjusted_short_threshold,
            decision_threshold,
        )
        liquidity_adjustment = LiquiditySessionAdjustment(
            threshold=display_threshold,
            size_multiplier=max(0.0, min(1.0, float(size_multiplier))),
            low_liquidity=bool(low_liquidity),
            low_dynamic_session=bool(low_dynamic_session),
        )
        pending_signal = _normalize_market_direction(
            score,
            adjusted_long_threshold,
            market_type,
            short_threshold=adjusted_short_threshold,
        )
        pending_signal_score = score
        meta_threshold = (
            1.0 - adjusted_short_threshold
            if market_type == "futures" and pending_signal < 0 and adjusted_short_threshold is not None
            else display_threshold
        )
        base_pending_meta = apply_meta_label_policy(
            getattr(model, "meta_label_policy", {}),
            adjusted_probability=score,
            threshold=meta_threshold,
            side=pending_signal,
            market_type=market_type,
        )
        pending_meta = apply_liquidity_session_meta(base_pending_meta, liquidity_adjustment) if pending_signal != 0 else base_pending_meta
        price = row.close
        final_mark_price = price
        bar_high, bar_low = _bar_bounds(row, price)
        regime_gate_ready = row_index + 1 >= regime_gate_min_rows
        regime_score = float(regime_scores[row_index]) if regime_gate_ready else 0.0
        regime_limit = float(cfg.max_regime_unpredictability)
        regime_score_over_limit = regime_score > regime_limit
        if (
            regime_score_over_limit
            and unpredictability_cooldown_ms > 0
            and regime_unpredictability_requires_cooldown(regime_score, regime_limit)
        ):
            regime_cooldown_until = max(
                int(regime_cooldown_until or row.timestamp),
                int(row.timestamp) + unpredictability_cooldown_ms,
            )
        regime_cooldown_active = regime_cooldown_until is not None and int(row.timestamp) < regime_cooldown_until
        regime_size_multiplier = _regime_soft_gate_size_multiplier(
            regime_score=regime_score,
            regime_limit=regime_limit,
            signal_score=execution_signal_score,
            direction=execution_signal,
            long_threshold=adjusted_long_threshold,
            short_threshold=adjusted_short_threshold,
            market_type=market_type,
        )
        day = _safe_day(row.timestamp)
        if day not in daily_trade_count:
            daily_trade_count[day] = 0
        trade_cap_reached = max_daily is not None and daily_trade_count[day] >= max_daily
        cooldown_active = (
            last_close_timestamp is not None
            and cooldown_ms > 0
            and row.timestamp - last_close_timestamp < cooldown_ms
        )
        if position_side == 0 and execution_signal != 0:
            if execution_meta.size_multiplier <= 0.0:
                if execution_meta.enabled:
                    meta_label_skips += 1
                continue
            if regime_cooldown_active or regime_size_multiplier <= 0.0:
                regime_entry_skips += 1
                continue
            if cooldown_active:
                continue
            if trade_cap_reached:
                cap_hits += 1
                continue
            gross, effective_margin = _position_size_from_risk(
                cash,
                cfg,
                market_type=market_type,
                leverage=leverage,
            )
            if execution_meta.enabled and execution_meta.action == "downsize":
                meta_label_downsizes += 1
                multiplier = max(0.0, min(1.0, float(execution_meta.size_multiplier)))
                gross *= multiplier
                effective_margin *= multiplier
            if regime_score_over_limit and regime_size_multiplier < 1.0:
                regime_entry_downsizes += 1
                gross *= regime_size_multiplier
                effective_margin *= regime_size_multiplier

            if gross <= 0 or effective_margin >= cash:
                continue

            if max_open_positions <= 0:
                cap_hits += 1
                continue

            side_sign = 1 if execution_signal > 0 else -1
            row_volume_notional = _row_quote_volume_notional(row, price)
            row_execution_assumptions = _row_execution_assumptions(
                row,
                cfg,
                symbol_profile=symbol_profile,
                include_range=False,
            )
            entry_fill = _simulate_fill(
                price,
                side_sign,
                cfg,
                notional=gross,
                volume=row_volume_notional,
                daily_volume=_row_trailing_quote_volume_24h_estimate(row),
                symbol_profile=symbol_profile,
                assumptions=row_execution_assumptions,
            )
            entry = entry_fill.fill_price
            if entry <= 0:
                continue

            fee = gross * fee_rate
            total_cost = effective_margin + fee
            if cash < total_cost:
                continue

            entry_equity_reference = cash
            entry_timestamp = int(row.timestamp)
            entry_row_index = int(row_index)
            flat_signal_streak = 0
            entry_fee_paid = fee
            entry_execution_cost_bps = float(entry_fill.total_cost_bps)
            cash -= total_cost
            total_fees += fee
            position_side = side_sign
            notional = side_sign * gross
            qty = abs(gross / entry)
            entry_price = entry
            margin_used = effective_margin
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
            entry_meta = execution_meta
            entry_opened_this_bar = True

            max_exposure = max(max_exposure, abs(notional))

        elif position_side != 0:
            adverse_mark_price = _adverse_mark_price(position_side, price, bar_high, bar_low)
            liquidated, margin_balance, maintenance_margin = _futures_liquidation_state(
                market_type=market_type,
                position_side=position_side,
                price=adverse_mark_price,
                entry_price=entry_price,
                qty=qty,
                margin_used=margin_used,
                cfg=cfg,
            )

            if liquidated:
                closed_side = position_side
                closed_notional = abs(notional)
                closed_entry_price = entry_price
                liquidation_events += 1
                stopped_by_liquidation = True
                liquidation_loss += margin_used
                closed_trades += 1
                realized = -margin_used
                net_pnl = realized - entry_fee_paid
                return_pct = _trade_return(net_pnl, entry_equity_reference)
                trade_pnls.append(float(net_pnl))
                trade_returns.append(float(return_pct))
                trade_log.append({
                    "opened_at": int(entry_timestamp),
                    "closed_at": int(row.timestamp),
                    "side": int(closed_side),
                    "gross_notional": float(closed_notional),
                    "entry_price": float(closed_entry_price),
                    "exit_mark_price": float(adverse_mark_price),
                    "realized_pnl": float(realized),
                    "net_pnl": float(net_pnl),
                    "return_pct": float(return_pct),
                    "entry_fee": float(entry_fee_paid),
                    "exit_fee": 0.0,
                    "entry_execution_cost_bps": float(entry_execution_cost_bps),
                    "exit_execution_cost_bps": 0.0,
                    "exit_reason": "liquidation",
                    "liquidated": True,
                    "liquidation_margin_balance": float(margin_balance),
                    "liquidation_maintenance_margin": float(maintenance_margin),
                    "liquidation_buffer_pct": float(cfg.liquidation_buffer_pct),
                    "meta_label_action": str(entry_meta.action),
                    "meta_label_size_multiplier": float(entry_meta.size_multiplier),
                    "meta_label_signal_strength": float(entry_meta.signal_strength),
                    "meta_label_reason": str(entry_meta.reason),
                })

                position_side = 0
                notional = 0.0
                qty = 0.0
                entry_price = 0.0
                margin_used = 0.0
                entry_fee_paid = 0.0
                entry_execution_cost_bps = 0.0
                entry_equity_reference = cash
                flat_signal_streak = 0
                entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "liquidation")
                last_close_timestamp = row.timestamp
            else:
                current_pnl_pct = (price - entry_price) / entry_price if position_side > 0 else (entry_price - price) / entry_price
                intrabar_close, close_mark_price, close_reason = _intrabar_exit(
                    position_side=position_side,
                    entry_price=entry_price,
                    high=bar_high,
                    low=bar_low,
                    cfg=cfg,
                )
                bars_held = max(0, int(row_index) - int(entry_row_index))
                lifecycle_exit = evaluate_position_exit(
                    position_side=position_side,
                    signal_direction=execution_signal,
                    current_pnl_pct=current_pnl_pct,
                    bars_held=bars_held,
                    flat_signal_streak=flat_signal_streak,
                    stop_loss_pct=cfg.stop_loss_pct,
                    take_profit_pct=cfg.take_profit_pct,
                    min_position_hold_bars=min_position_hold_bars,
                    flat_signal_exit_grace_bars=flat_signal_exit_grace_bars,
                    max_position_hold_bars=max_position_hold_bars,
                )
                flat_signal_streak = lifecycle_exit.flat_signal_streak
                if not intrabar_close:
                    close_mark_price = price
                    close_reason = lifecycle_exit.reason
                should_close = bool(intrabar_close or lifecycle_exit.should_close)

                if should_close:
                    closed_side = position_side
                    closed_notional = abs(notional)
                    closed_entry_price = entry_price
                    row_quote_volume_notional = _row_reported_quote_volume_notional(row)
                    row_execution_assumptions = _row_execution_assumptions(row, cfg, symbol_profile=symbol_profile)
                    cash_delta, realized, exit_fee, exit_execution_cost_bps, exit_price = _close_position(
                        position_side=position_side,
                        price=close_mark_price,
                        entry_price=entry_price,
                        qty=qty,
                        notional=notional,
                        margin_used=margin_used,
                        cfg=cfg,
                        symbol_profile=symbol_profile,
                        fill_volume_notional=row_quote_volume_notional,
                        daily_volume_notional=_row_trailing_quote_volume_24h_estimate(row),
                        execution_assumptions=row_execution_assumptions,
                    )
                    cash += cash_delta
                    total_fees += exit_fee
                    closed_trades += 1
                    net_pnl = realized - entry_fee_paid - exit_fee
                    return_pct = _trade_return(net_pnl, entry_equity_reference)
                    trade_pnls.append(float(net_pnl))
                    trade_returns.append(float(return_pct))
                    trade_log.append({
                        "opened_at": int(entry_timestamp),
                        "closed_at": int(row.timestamp),
                        "side": int(closed_side),
                        "gross_notional": float(closed_notional),
                        "entry_price": float(closed_entry_price),
                        "exit_mark_price": float(close_mark_price),
                        "exit_price": float(exit_price),
                        "realized_pnl": float(realized),
                        "net_pnl": float(net_pnl),
                        "return_pct": float(return_pct),
                        "entry_fee": float(entry_fee_paid),
                        "exit_fee": float(exit_fee),
                        "entry_execution_cost_bps": float(entry_execution_cost_bps),
                        "exit_execution_cost_bps": float(exit_execution_cost_bps),
                        "exit_reason": str(close_reason or "signal_exit"),
                        "bars_held": int(bars_held),
                        "flat_signal_streak": int(flat_signal_streak),
                        "meta_label_action": str(entry_meta.action),
                        "meta_label_size_multiplier": float(entry_meta.size_multiplier),
                        "meta_label_signal_strength": float(entry_meta.signal_strength),
                        "meta_label_reason": str(entry_meta.reason),
                    })
                    if net_pnl > 0:
                        wins += 1

                    position_side = 0
                    notional = 0.0
                    qty = 0.0
                    entry_price = 0.0
                    margin_used = 0.0
                    entry_fee_paid = 0.0
                    entry_execution_cost_bps = 0.0
                    entry_equity_reference = cash
                    flat_signal_streak = 0
                    entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "reset")
                    last_close_timestamp = row.timestamp

        # mark-to-market drawdown control with unrealized exposure
        if position_side != 0:
            drawdown_high, drawdown_low = (price, price) if entry_opened_this_bar else (bar_high, bar_low)
            drawdown_mark_price = _adverse_mark_price(position_side, price, drawdown_high, drawdown_low)
            unrealized = position_side * (drawdown_mark_price - entry_price) * qty
            equity = cash + margin_used + unrealized
        else:
            equity = cash

        if equity > equity_peak:
            equity_peak = equity
        dd = 1.0 if equity <= 0.0 and equity_peak > 0.0 else ((equity_peak - equity) / equity_peak if equity_peak else 0.0)
        if dd > max_drawdown:
            max_drawdown = dd
        equity_curve.append(_equity_point(int(row.timestamp), equity, dd, position_side))

        if stopped_by_liquidation:
            break

        if cfg.max_drawdown_limit > 0.0 and dd >= cfg.max_drawdown_limit:
            stopped_by_drawdown = True
            if position_side != 0:
                closed_side = position_side
                closed_notional = abs(notional)
                closed_entry_price = entry_price
                row_quote_volume_notional = _row_reported_quote_volume_notional(row)
                row_execution_assumptions = _row_execution_assumptions(row, cfg, symbol_profile=symbol_profile)
                drawdown_delta, drawdown_realized, drawdown_fee, drawdown_execution_cost_bps, drawdown_exit_price = _close_position(
                    position_side=position_side,
                    price=drawdown_mark_price,
                    entry_price=entry_price,
                    qty=qty,
                    notional=notional,
                    margin_used=margin_used,
                    cfg=cfg,
                    symbol_profile=symbol_profile,
                    fill_volume_notional=row_quote_volume_notional,
                    daily_volume_notional=_row_trailing_quote_volume_24h_estimate(row),
                    execution_assumptions=row_execution_assumptions,
                )
                cash += drawdown_delta
                total_fees += drawdown_fee
                closed_trades += 1
                net_pnl = drawdown_realized - entry_fee_paid - drawdown_fee
                return_pct = _trade_return(net_pnl, entry_equity_reference)
                trade_pnls.append(float(net_pnl))
                trade_returns.append(float(return_pct))
                trade_log.append({
                    "opened_at": int(entry_timestamp),
                    "closed_at": int(row.timestamp),
                    "side": int(closed_side),
                    "gross_notional": float(closed_notional),
                    "entry_price": float(closed_entry_price),
                    "exit_mark_price": float(drawdown_mark_price),
                    "exit_price": float(drawdown_exit_price),
                    "realized_pnl": float(drawdown_realized),
                    "net_pnl": float(net_pnl),
                    "return_pct": float(return_pct),
                    "entry_fee": float(entry_fee_paid),
                    "exit_fee": float(drawdown_fee),
                    "entry_execution_cost_bps": float(entry_execution_cost_bps),
                    "exit_execution_cost_bps": float(drawdown_execution_cost_bps),
                    "exit_reason": "drawdown_limit",
                    "drawdown": float(dd),
                    "meta_label_action": str(entry_meta.action),
                    "meta_label_size_multiplier": float(entry_meta.size_multiplier),
                    "meta_label_signal_strength": float(entry_meta.signal_strength),
                    "meta_label_reason": str(entry_meta.reason),
                })
                if net_pnl > 0:
                    wins += 1
                position_side = 0
                notional = 0.0
                qty = 0.0
                entry_price = 0.0
                margin_used = 0.0
                entry_fee_paid = 0.0
                entry_execution_cost_bps = 0.0
                entry_equity_reference = cash
                flat_signal_streak = 0
                entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "drawdown_limit")
                last_close_timestamp = row.timestamp
                close_dd = 1.0 if cash <= 0.0 and equity_peak > 0.0 else ((equity_peak - cash) / equity_peak if equity_peak else 0.0)
                if close_dd > max_drawdown:
                    max_drawdown = close_dd
                equity_curve.append(_equity_point(int(row.timestamp), cash, close_dd, position_side))
            break

    # force close residual position at final mark
    if position_side != 0:
        closed_side = position_side
        closed_notional = abs(notional)
        closed_entry_price = entry_price
        final_row = rows[-1]
        final_volume_notional = _row_reported_quote_volume_notional(final_row)
        final_execution_assumptions = _row_execution_assumptions(
            final_row,
            cfg,
            symbol_profile=symbol_profile,
            include_range=int(entry_row_index) != len(rows) - 1,
        )
        final_delta, final_realized, final_fee, final_execution_cost_bps, final_exit_price = _close_position(
            position_side=position_side,
            price=final_mark_price,
            entry_price=entry_price,
            qty=qty,
            notional=notional,
            margin_used=margin_used,
            cfg=cfg,
            symbol_profile=symbol_profile,
            fill_volume_notional=final_volume_notional,
            daily_volume_notional=_row_trailing_quote_volume_24h_estimate(final_row),
            execution_assumptions=final_execution_assumptions,
        )
        cash += final_delta
        total_fees += final_fee
        closed_trades += 1
        net_pnl = final_realized - entry_fee_paid - final_fee
        return_pct = _trade_return(net_pnl, entry_equity_reference)
        trade_pnls.append(float(net_pnl))
        trade_returns.append(float(return_pct))
        trade_log.append({
            "opened_at": int(entry_timestamp),
            "closed_at": int(rows[-1].timestamp),
            "side": int(closed_side),
            "gross_notional": float(closed_notional),
            "entry_price": float(closed_entry_price),
            "exit_mark_price": float(final_mark_price),
            "exit_price": float(final_exit_price),
            "realized_pnl": float(final_realized),
            "net_pnl": float(net_pnl),
            "return_pct": float(return_pct),
            "entry_fee": float(entry_fee_paid),
            "exit_fee": float(final_fee),
            "entry_execution_cost_bps": float(entry_execution_cost_bps),
            "exit_execution_cost_bps": float(final_execution_cost_bps),
            "exit_reason": "final_mark",
            "meta_label_action": str(entry_meta.action),
            "meta_label_size_multiplier": float(entry_meta.size_multiplier),
            "meta_label_signal_strength": float(entry_meta.signal_strength),
            "meta_label_reason": str(entry_meta.reason),
        })
        if net_pnl > 0:
            wins += 1
        position_side = 0
        notional = 0.0
        qty = 0.0
        entry_price = 0.0
        margin_used = 0.0
        flat_signal_streak = 0
        if cash > equity_peak:
            equity_peak = cash
        final_dd = 1.0 if cash <= 0.0 and equity_peak > 0.0 else ((equity_peak - cash) / equity_peak if equity_peak else 0.0)
        if final_dd > max_drawdown:
            max_drawdown = final_dd
        equity_curve.append(_equity_point(int(rows[-1].timestamp), cash, final_dd, position_side))

    realized_pnl = cash - starting_cash
    win_rate = wins / closed_trades if closed_trades else 0.0

    trades = closed_trades

    buy_hold_pnl = _buy_hold_pnl(
        rows,
        starting_cash,
        cfg,
        market_type=market_type,
        leverage=leverage,
        symbol_profile=symbol_profile,
    )
    path_quality = _path_quality_metrics(trade_pnls, trade_returns)

    result = BacktestResult(
        starting_cash=starting_cash,
        ending_cash=cash,
        realized_pnl=realized_pnl,
        win_rate=win_rate,
        trades=trades,
        max_drawdown=max_drawdown,
        stopped_by_drawdown=stopped_by_drawdown,
        closed_trades=closed_trades,
        gross_exposure=max_exposure,
        total_fees=total_fees,
        max_exposure=max_exposure,
        trades_per_day_cap_hit=cap_hits,
        buy_hold_pnl=buy_hold_pnl,
        edge_vs_buy_hold=realized_pnl - buy_hold_pnl,
        equity_curve=tuple(equity_curve),
        trade_pnls=tuple(trade_pnls),
        trade_returns=tuple(trade_returns),
        trade_log=tuple(trade_log),
        gross_profit=float(path_quality["gross_profit"]),
        gross_loss=float(path_quality["gross_loss"]),
        profit_factor=float(path_quality["profit_factor"]),
        expectancy=float(path_quality["expectancy"]),
        average_trade_return=float(path_quality["average_trade_return"]),
        trade_return_stdev=float(path_quality["trade_return_stdev"]),
        max_consecutive_losses=int(path_quality["max_consecutive_losses"]),
        meta_label_skips=int(meta_label_skips),
        meta_label_downsizes=int(meta_label_downsizes),
        regime_entry_skips=int(regime_entry_skips),
        regime_entry_downsizes=int(regime_entry_downsizes),
        stopped_by_liquidation=bool(stopped_by_liquidation),
        liquidation_events=int(liquidation_events),
        liquidation_loss=float(liquidation_loss),
        **_score_backend_payload(score_backend),
    )
    sanity = build_backtest_financial_sanity_report(result, reject_liquidation=False)
    blocks = blocking_reasons(sanity)
    if blocks:
        raise ValueError(f"backtest financial sanity failed: {'; '.join(blocks[:5])}")
    return result

"""Local risk policy checks shared by operator preflight and live runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .types import RuntimeConfig, StrategyConfig
from .assets import MAX_AUTONOMOUS_LEVERAGE
from .execution_simulation import execution_assumptions_from_strategy
from .model import ModelLoadError
from .model_readiness import load_model_readiness_report


@dataclass(frozen=True)
class RiskCheck:
    status: str
    label: str
    detail: str
    metric: float | int | str | None = None
    limit: float | int | str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskPolicyReport:
    checks: tuple[RiskCheck, ...]
    effective_dry_run: bool
    leverage: float
    notional_cap_pct: float
    max_loss_per_trade_pct: float

    @property
    def allowed(self) -> bool:
        return all(check.status != "block" for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    @property
    def block_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "block")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed"] = self.allowed
        payload["warning_count"] = self.warning_count
        payload["block_count"] = self.block_count
        return payload


@dataclass(frozen=True)
class EntryRiskDecision:
    allowed: bool
    code: str
    detail: str
    metrics: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    metric: float | int | str | None = None,
    limit: float | int | str | None = None,
) -> RiskCheck:
    return RiskCheck(status, label, detail, metric=metric, limit=limit)


def _environment(runtime: RuntimeConfig) -> str:
    if getattr(runtime, "demo", False):
        return "demo"
    return "testnet" if runtime.testnet else "mainnet"


def _effective_leverage(strategy: StrategyConfig, market_type: str, requested: float | None = None) -> float:
    if market_type != "futures":
        return 1.0
    try:
        raw = float(strategy.leverage if requested is None else requested)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(raw):
        return 1.0
    return max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, raw))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _metric_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


_UNPREDICTABLE_REGIMES = frozenset({"volatile_chop", "mixed", "insufficient_data"})


def market_regime_unpredictability(
    regime: str | None,
    confidence: float | int | None = None,
    notes: Sequence[str] | None = None,
) -> float:
    """Return a deterministic 0-1 score for regimes where new entries should wait.

    This intentionally uses only point-in-time regime evidence. It is a risk
    gate, not a predictive alpha model: high values mean the market state is too
    noisy, under-separated, or data-poor for a fresh autonomous entry.
    """

    name = str(regime or "").strip().lower()
    conf = _metric_float(confidence)
    if not math.isfinite(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    note_set = {str(note).strip().lower() for note in (notes or ())}
    if name == "insufficient_data":
        score = 1.0
    elif name == "volatile_chop":
        score = 0.72 + 0.24 * conf
    elif name == "mixed":
        score = 0.58 + 0.30 * (1.0 - conf)
    elif name == "range_bound":
        score = 0.35 + 0.20 * (1.0 - conf)
    elif name == "serial_correlation":
        score = 0.30 + 0.15 * (1.0 - conf)
    elif name in {"trend_up", "trend_down"}:
        score = 0.18 + 0.22 * (1.0 - conf)
    else:
        score = 0.65
    if "low_regime_separation" in note_set:
        score += 0.18
    if "short_window" in note_set:
        score += 0.08
    if "flat_returns" in note_set:
        score += 0.05
    return max(0.0, min(1.0, float(score)))


def stop_loss_sized_notional_pct(
    strategy: StrategyConfig,
    market_type: str,
    *,
    leverage: float | None = None,
) -> float:
    """Return gross notional exposure as an equity percentage from stop-loss risk."""

    effective_leverage = _effective_leverage(strategy, market_type, leverage)
    risk_budget_pct = max(0.0, _finite(strategy.risk_per_trade))
    loss_at_stop_pct = stop_loss_effective_loss_pct(strategy)
    risk_sized_notional_pct = (
        risk_budget_pct / loss_at_stop_pct
        if loss_at_stop_pct > 0.0
        else risk_budget_pct
    )
    max_position_pct = max(0.0, _finite(strategy.max_position_pct))
    max_asset_allocation_pct = max(
        0.0,
        min(1.0, _finite(strategy.max_asset_allocation_pct, 1.0)),
    )
    if market_type == "futures":
        return max(
            0.0,
            min(
                risk_sized_notional_pct,
                max_position_pct * effective_leverage,
                max_asset_allocation_pct,
                effective_leverage,
            ),
        )
    return max(
        0.0,
        min(risk_sized_notional_pct, max_position_pct, max_asset_allocation_pct, 1.0),
    )


def stop_loss_effective_loss_pct(strategy: StrategyConfig) -> float:
    """Return estimated equity loss per dollar notional when a stop is hit."""

    stop_loss_pct = max(0.0, _finite(strategy.stop_loss_pct))
    fee_rate = max(0.0, _finite(strategy.taker_fee_bps)) / 10_000.0
    assumptions = execution_assumptions_from_strategy(strategy)
    latency_seconds = min(10.0, max(0.0, float(assumptions.latency_ms)) / 1000.0)
    fill_cost_bps = (
        max(0.0, _finite(assumptions.spread_bps)) / 2.0
        + max(0.0, _finite(assumptions.volatility_buffer_bps)) * latency_seconds
        + max(0.0, _finite(assumptions.impact_coefficient))
        + max(0.0, _finite(assumptions.testnet_to_live_buffer_bps))
    )
    adverse_exit_fill_rate = fill_cost_bps / 10_000.0
    return max(0.0, stop_loss_pct + fee_rate + fee_rate + adverse_exit_fill_rate)


def build_risk_policy_report(
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    *,
    effective_dry_run: bool | None = None,
    leverage: float | None = None,
    model_path: str | Path | None = None,
    require_model_candidate_search: bool = False,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    min_live_data_years: float = 1.0,
    min_live_coverage_ratio: float = 0.995,
    max_live_gap_count: int = 0,
) -> RiskPolicyReport:
    """Return deterministic local risk checks without network access."""

    dry_run = bool(runtime.dry_run if effective_dry_run is None else effective_dry_run)
    effective_leverage = _effective_leverage(strategy, runtime.market_type, leverage)
    notional_cap_pct = stop_loss_sized_notional_pct(
        strategy,
        runtime.market_type,
        leverage=effective_leverage,
    )
    loss_at_stop_pct = stop_loss_effective_loss_pct(strategy)
    max_loss_per_trade_pct = notional_cap_pct * loss_at_stop_pct
    checks: list[RiskCheck] = []

    symbols = tuple(getattr(runtime, "symbols", ()) or (runtime.symbol,))
    checks.append(_check("ok", "primary symbol", runtime.symbol))
    checks.append(
        _check(
            "ok" if len(set(symbols)) >= max(1, strategy.min_diversified_assets) else "block",
            "mandatory diversification",
            f"{len(set(symbols))} configured symbols",
            metric=len(set(symbols)),
            limit=f">={strategy.min_diversified_assets}",
        )
    )
    checks.append(
        _check(
            "ok" if runtime.market_type in {"spot", "futures"} else "block",
            "market type",
            runtime.market_type,
        )
    )
    environment = _environment(runtime)
    safe_endpoint = runtime.testnet or getattr(runtime, "demo", False)
    checks.append(
        _check(
            "ok" if safe_endpoint else "block",
            "execution environment",
            f"{environment} endpoint",
        )
    )
    if dry_run:
        checks.append(_check("ok", "order mode", "paper/dry-run"))
    else:
        has_credentials = bool(runtime.api_key and runtime.api_secret)
        checks.append(
            _check(
                "ok" if has_credentials else "block",
                "order credentials",
                "configured" if has_credentials else "missing API key/secret",
            )
        )

    cash = _finite(getattr(runtime, "managed_usdc", 0.0))
    cash_status = "ok" if cash > 0.0 else ("warn" if dry_run else "block")
    checks.append(_check(cash_status, "managed USDC", f"{cash:.2f}", metric=cash, limit=">0"))
    checks.append(
        _check(
            "ok" if effective_leverage <= MAX_AUTONOMOUS_LEVERAGE else "block",
            "effective leverage",
            f"{effective_leverage:.1f}x",
            metric=effective_leverage,
            limit=f"<={MAX_AUTONOMOUS_LEVERAGE:.0f} hard",
        )
    )
    if runtime.market_type == "futures":
        liquidation_buffer = _finite(strategy.liquidation_buffer_pct)
        if liquidation_buffer <= 0.0:
            checks.append(
                _check(
                    "warn" if dry_run else "block",
                    "liquidation buffer",
                    "disabled",
                    metric=liquidation_buffer,
                    limit=">0 futures maintenance-plus-buffer proxy",
                )
            )
        else:
            checks.append(
                _check(
                    "ok" if liquidation_buffer >= 0.01 else "warn",
                    "liquidation buffer",
                    f"{liquidation_buffer:.2%}",
                    metric=liquidation_buffer,
                    limit=">=1% preferred",
                )
            )
    risk_per_trade = _finite(strategy.risk_per_trade)
    checks.append(
        _check(
            "block" if risk_per_trade <= 0.0 else ("ok" if risk_per_trade <= 0.02 else "warn"),
            "risk per trade",
            f"{risk_per_trade:.2%}",
            metric=risk_per_trade,
            limit=0.02,
        )
    )
    max_position = _finite(strategy.max_position_pct)
    checks.append(
        _check(
            "block" if max_position <= 0.0 else ("ok" if max_position <= 0.50 else ("warn" if dry_run or max_position <= 0.75 else "block")),
            "max position",
            f"{max_position:.2%}",
            metric=max_position,
            limit="<=75% hard",
        )
    )
    checks.append(
        _check(
            "ok" if notional_cap_pct <= 0.50 else ("warn" if dry_run or notional_cap_pct <= 0.75 else "block"),
            "entry notional cap",
            f"{notional_cap_pct:.2%} of equity",
            metric=notional_cap_pct,
            limit="<=75% hard",
        )
    )
    stop_loss = _finite(strategy.stop_loss_pct)
    take_profit = _finite(strategy.take_profit_pct)
    checks.append(
        _check(
            ("ok" if stop_loss > 0.0 else ("warn" if dry_run else "block")),
            "stop loss",
            f"{stop_loss:.2%}" if stop_loss > 0.0 else "disabled",
            metric=stop_loss,
            limit=">0",
        )
    )
    if stop_loss >= 1.0:
        checks.append(
            _check(
                "warn" if dry_run else "block",
                "stop-loss geometry",
                f"{stop_loss:.2%} would not produce a positive long-side stop price",
                metric=stop_loss,
                limit="<100%",
            )
        )
    checks.append(
        _check(
            "warn" if take_profit <= 0.0 else "ok",
            "take profit",
            f"{take_profit:.2%}",
            metric=take_profit,
            limit=">0",
        )
    )
    checks.append(
        _check(
            "ok" if max_loss_per_trade_pct <= 0.02 else "warn",
            "estimated loss at stop",
            f"{max_loss_per_trade_pct:.2%} of equity",
            metric=max_loss_per_trade_pct,
            limit=0.02,
        )
    )
    portfolio_risk_budget = _finite(strategy.max_portfolio_risk_pct)
    checks.append(
        _check(
            "ok" if portfolio_risk_budget <= 0.05 else ("warn" if dry_run else "block"),
            "portfolio risk budget",
            f"{portfolio_risk_budget:.2%}",
            metric=portfolio_risk_budget,
            limit="<=5%",
        )
    )
    daily_loss_budget = _finite(getattr(strategy, "max_daily_loss_pct", 0.0))
    if daily_loss_budget <= 0.0:
        checks.append(_check("block" if not dry_run else "warn", "daily loss budget", "disabled", metric=0.0, limit=">0"))
    else:
        checks.append(
            _check(
                "ok" if daily_loss_budget <= 0.02 else ("warn" if dry_run or daily_loss_budget <= 0.04 else "block"),
                "daily loss budget",
                f"{daily_loss_budget:.2%} of reference equity",
                metric=daily_loss_budget,
                limit="<=4% hard",
            )
        )
    session_loss_budget = _finite(getattr(strategy, "max_session_loss_pct", 0.0))
    if session_loss_budget <= 0.0:
        checks.append(_check("block" if not dry_run else "warn", "session loss budget", "disabled", metric=0.0, limit=">0"))
    else:
        checks.append(
            _check(
                "ok" if session_loss_budget <= 0.05 else ("warn" if dry_run or session_loss_budget <= 0.10 else "block"),
                "session loss budget",
                f"{session_loss_budget:.2%} of reference equity",
                metric=session_loss_budget,
                limit="<=10% hard",
            )
        )
    active_hard_budgets = [
        (label, value)
        for label, value in (
            ("daily loss budget", daily_loss_budget),
            ("session loss budget", session_loss_budget),
            ("portfolio risk budget", portfolio_risk_budget),
        )
        if value > 0.0
    ]
    if active_hard_budgets:
        binding_label, binding_budget = min(active_hard_budgets, key=lambda item: item[1])
        coherent = max_loss_per_trade_pct <= binding_budget + 1e-12
        checks.append(
            _check(
                "ok" if coherent else ("warn" if dry_run else "block"),
                "loss budget coherence",
                (
                    f"stop-loss estimate {max_loss_per_trade_pct:.2%} vs "
                    f"{binding_label} {binding_budget:.2%}"
                ),
                metric=max_loss_per_trade_pct,
                limit=f"<={binding_label} {binding_budget:.2%}",
            )
        )
    max_losses = int(getattr(strategy, "max_consecutive_losses", 0) or 0)
    checks.append(
        _check(
            "ok" if 1 <= max_losses <= 5 else ("warn" if dry_run or max_losses <= 10 else "block"),
            "loss-streak lockout",
            f"{max_losses} consecutive losses" if max_losses > 0 else "disabled",
            metric=max_losses,
            limit="1-5 preferred",
        )
    )
    max_network_errors = int(getattr(strategy, "max_network_errors", 0) or 0)
    checks.append(
        _check(
            "ok" if 1 <= max_network_errors <= 5 else ("warn" if dry_run or max_network_errors <= 10 else "block"),
            "network interruption halt",
            f"{max_network_errors} consecutive API errors",
            metric=max_network_errors,
            limit="1-5 preferred",
        )
    )
    recovery_cooldown = int(getattr(strategy, "recovery_cooldown_seconds", 0) or 0)
    checks.append(
        _check(
            "ok" if recovery_cooldown >= 5 else ("warn" if dry_run else "block"),
            "reconnect recovery cooldown",
            f"{recovery_cooldown}s observe-before-resume",
            metric=recovery_cooldown,
            limit=">=5s",
        )
    )
    checks.append(
        _check(
            "ok" if strategy.max_asset_allocation_pct <= 0.35 else ("warn" if dry_run else "block"),
            "single asset allocation cap",
            f"{strategy.max_asset_allocation_pct:.2%}",
            metric=strategy.max_asset_allocation_pct,
            limit="<=35%",
        )
    )
    checks.append(
        _check(
            "warn" if strategy.reinvest_profits else "ok",
            "profit reinvestment",
            "enabled - compounds both gains and losses" if strategy.reinvest_profits else "disabled",
        )
    )
    checks.append(
        _check(
            "ok" if strategy.min_liquidity_score >= 0.60 else "warn",
            "minimum liquidity score",
            f"{strategy.min_liquidity_score:.2f}",
            metric=strategy.min_liquidity_score,
            limit=">=0.60",
        )
    )
    checks.append(
        _check(
            "ok" if strategy.max_regime_unpredictability <= 0.85 else "warn",
            "regime unpredictability gate",
            f"max score {strategy.max_regime_unpredictability:.2f}; cooldown {strategy.unpredictability_cooldown_minutes}m",
            metric=strategy.max_regime_unpredictability,
            limit="<=0.85",
        )
    )

    if strategy.max_trades_per_day <= 0:
        checks.append(_check("warn", "daily trade cap", "disabled", metric=0, limit=">0"))
    else:
        checks.append(
            _check("ok", "daily trade cap", str(strategy.max_trades_per_day), metric=strategy.max_trades_per_day)
        )
    drawdown_limit = _finite(strategy.max_drawdown_limit)
    if drawdown_limit <= 0.0:
        checks.append(_check("warn", "drawdown stop", "disabled", metric=0.0, limit=">0"))
    else:
        checks.append(
            _check(
                "ok" if drawdown_limit <= 0.35 else ("warn" if dry_run or drawdown_limit <= 0.50 else "block"),
                "drawdown stop",
                f"{drawdown_limit:.2%}",
                metric=drawdown_limit,
                limit="<=50% hard",
            )
        )
    slippage_bps = _finite(strategy.slippage_bps)
    checks.append(
        _check(
            "ok" if slippage_bps <= 100.0 else "warn",
            "slippage assumption",
            f"{slippage_bps:.1f} bps",
            metric=slippage_bps,
            limit=100.0,
        )
    )
    taker_fee_bps = _finite(strategy.taker_fee_bps)
    checks.append(
        _check(
            "ok" if taker_fee_bps <= 100.0 else "warn",
            "fee assumption",
            f"{taker_fee_bps:.1f} bps",
            metric=taker_fee_bps,
            limit=100.0,
        )
    )
    if strategy.external_signals_enabled:
        checks.append(
            _check(
                "ok",
                "external signal quorum",
                f"enabled min_providers={strategy.external_signal_min_providers}",
                metric=strategy.external_signal_min_providers,
            )
        )
    else:
        checks.append(_check("warn", "external signal quorum", "disabled"))

    if model_path is not None:
        path = Path(model_path)
        exists = path.exists()
        checks.append(
            _check(
                "ok" if dry_run or exists else "block",
                "model path",
                str(path) if exists else f"missing {path}",
            )
        )
        if exists:
            try:
                readiness = load_model_readiness_report(
                    path,
                    require_model_candidate_search=require_model_candidate_search,
                    require_accelerator_evidence=require_accelerator_evidence,
                    require_live_data_evidence=require_live_data_evidence,
                    expected_symbol=expected_symbol,
                    expected_market_type=expected_market_type,
                    expected_interval=expected_interval,
                    min_live_data_years=min_live_data_years,
                    min_live_coverage_ratio=min_live_coverage_ratio,
                    max_live_gap_count=max_live_gap_count,
                )
                status = "ok" if readiness.allowed else ("warn" if dry_run else "block")
                detail = "passed" if readiness.allowed else "; ".join(
                    f"{check.label}: {check.detail}"
                    for check in readiness.checks
                    if check.status == "block"
                )
                checks.append(_check(status, "model promotion evidence", detail or "failed"))
            except (OSError, ValueError, ModelLoadError) as exc:
                checks.append(
                    _check(
                        "warn" if dry_run else "block",
                        "model promotion evidence",
                        f"unreadable model evidence: {exc}",
                    )
                )

    return RiskPolicyReport(
        checks=tuple(checks),
        effective_dry_run=dry_run,
        leverage=effective_leverage,
        notional_cap_pct=notional_cap_pct,
        max_loss_per_trade_pct=max_loss_per_trade_pct,
    )


def assess_entry_risk(
    *,
    direction: int,
    position_side: int,
    max_open_positions: int,
    max_daily_trades: int,
    daily_trade_count: int,
    cash: float,
    price: float,
    drawdown: float,
    drawdown_limit: float,
    daily_loss: float = 0.0,
    daily_loss_limit: float = 0.0,
    session_loss: float = 0.0,
    session_loss_limit: float = 0.0,
    consecutive_losses: int = 0,
    max_consecutive_losses: int = 0,
    network_errors: int = 0,
    max_network_errors: int = 0,
    recovery_pending: bool = False,
    regime: str | None = None,
    regime_confidence: float | None = None,
    regime_notes: Sequence[str] | None = None,
    regime_unpredictability_score: float | None = None,
    max_regime_unpredictability: float = 1.0,
    regime_cooldown_active: bool = False,
) -> EntryRiskDecision:
    computed_regime_score = (
        market_regime_unpredictability(regime, regime_confidence, regime_notes)
        if regime_unpredictability_score is None
        else _metric_float(regime_unpredictability_score)
    )
    metrics: dict[str, Any] = {
        "direction": int(direction),
        "position_side": int(position_side),
        "max_open_positions": int(max_open_positions),
        "max_daily_trades": int(max_daily_trades),
        "daily_trade_count": int(daily_trade_count),
        "cash": _metric_float(cash),
        "price": _metric_float(price),
        "drawdown": _metric_float(drawdown),
        "drawdown_limit": _metric_float(drawdown_limit),
        "daily_loss": _metric_float(daily_loss),
        "daily_loss_limit": _metric_float(daily_loss_limit),
        "session_loss": _metric_float(session_loss),
        "session_loss_limit": _metric_float(session_loss_limit),
        "consecutive_losses": int(consecutive_losses),
        "max_consecutive_losses": int(max_consecutive_losses),
        "network_errors": int(network_errors),
        "max_network_errors": int(max_network_errors),
        "recovery_pending": bool(recovery_pending),
        "regime": str(regime or ""),
        "regime_confidence": _metric_float(regime_confidence),
        "regime_unpredictability_score": computed_regime_score,
        "max_regime_unpredictability": _metric_float(max_regime_unpredictability),
        "regime_cooldown_active": bool(regime_cooldown_active),
    }
    finite_keys = (
        "cash",
        "price",
        "drawdown",
        "drawdown_limit",
        "daily_loss",
        "daily_loss_limit",
        "session_loss",
        "session_loss_limit",
        "regime_unpredictability_score",
        "max_regime_unpredictability",
    )
    if not all(math.isfinite(float(metrics[key])) for key in finite_keys):
        return EntryRiskDecision(False, "nonfinite", "non-finite risk input", metrics)
    regime_score = float(metrics["regime_unpredictability_score"])
    max_regime_score = float(metrics["max_regime_unpredictability"])
    if not 0.0 <= regime_score <= 1.0:
        return EntryRiskDecision(False, "invalid_regime_score", "market-regime score must be normalized 0-1", metrics)
    if not 0.0 <= max_regime_score <= 1.0:
        return EntryRiskDecision(False, "invalid_regime_limit", "market-regime limit must be normalized 0-1", metrics)
    if recovery_pending:
        return EntryRiskDecision(False, "recovery_pending", "post-interruption recovery must finish before entry", metrics)
    if regime_cooldown_active:
        return EntryRiskDecision(False, "regime_cooldown", "market-regime unpredictability cooldown is active", metrics)
    if regime_score > max_regime_score:
        return EntryRiskDecision(False, "unpredictable_regime", "market regime is too unpredictable for a new entry", metrics)
    if max_network_errors > 0 and network_errors >= max_network_errors:
        return EntryRiskDecision(False, "network_halt", "network interruption halt is active", metrics)
    if direction == 0:
        return EntryRiskDecision(False, "no_signal", "no actionable entry signal", metrics)
    if position_side != 0:
        return EntryRiskDecision(False, "position_open", "position already open", metrics)
    if max_open_positions <= 0:
        return EntryRiskDecision(False, "max_open_positions", "entry disabled by max_open_positions", metrics)
    if max_daily_trades > 0 and daily_trade_count >= max_daily_trades:
        return EntryRiskDecision(False, "trade_cap", "daily trade cap reached", metrics)
    if cash <= 0.0:
        return EntryRiskDecision(False, "cash", "no managed cash available", metrics)
    if price <= 0.0:
        return EntryRiskDecision(False, "price", "invalid market price", metrics)
    if daily_loss_limit > 0.0 and daily_loss >= daily_loss_limit:
        return EntryRiskDecision(False, "daily_loss", "daily loss budget reached", metrics)
    if session_loss_limit > 0.0 and session_loss >= session_loss_limit:
        return EntryRiskDecision(False, "session_loss", "session loss budget reached", metrics)
    if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
        return EntryRiskDecision(False, "loss_streak", "consecutive loss lockout reached", metrics)
    if drawdown_limit > 0.0 and drawdown >= drawdown_limit:
        return EntryRiskDecision(False, "drawdown", "drawdown limit already reached", metrics)
    return EntryRiskDecision(True, "allowed", "entry risk checks passed", metrics)


def render_risk_policy_report(report: RiskPolicyReport) -> str:
    lines = [
        "Risk policy report",
        (
            f"allowed={report.allowed} warnings={report.warning_count} blocks={report.block_count} "
            f"dry_run={report.effective_dry_run} leverage={report.leverage:.1f}x "
            f"notional_cap={report.notional_cap_pct:.2%} stop_loss_equity={report.max_loss_per_trade_pct:.2%}"
        ),
    ]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.label}: {check.detail}")
    return "\n".join(lines)

"""Deterministic evidence gate for AI-assisted model uplift."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

from .ai_runtime import estimate_model_parameters_b


_PNL_KEYS = ("realized_pnl", "net_pnl", "pnl")
_ROI_KEYS = ("roi_pct", "roi", "return_pct", "net_return_pct")
_DRAWDOWN_KEYS = ("max_drawdown", "max_drawdown_pct", "drawdown")
_EXPECTANCY_KEYS = ("expectancy", "edge", "mean_trade_pnl")
_PROFIT_FACTOR_KEYS = ("profit_factor",)
_TRADES_KEYS = ("closed_trades", "trade_count", "trades")
_WIN_RATE_KEYS = ("win_rate", "win_rate_pct")


@dataclass(frozen=True)
class AIUpliftPolicy:
    """Minimum evidence required before AI-assisted alpha can be promoted."""

    min_model_parameters_b: float = 2.0
    min_ai_closed_trades: int = 5
    min_pnl_delta: float = 0.0
    min_expectancy_delta: float = 0.0
    max_drawdown_delta: float = 0.0
    require_positive_ai_pnl: bool = True

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AIUpliftReport:
    """AI-vs-ML holdout result with fail-closed promotion status."""

    accepted: bool
    advisory_only: bool
    model_name: str
    model_parameters_b: float | None
    baseline: dict[str, float]
    ai: dict[str, float]
    deltas: dict[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    policy: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _first_metric(metrics: Mapping[str, object], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in metrics:
            return _finite(metrics[key])
    return 0.0


def normalize_uplift_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Normalize common backtest metric names into the AI uplift contract."""

    return {
        "realized_pnl": _first_metric(metrics, _PNL_KEYS),
        "roi_pct": _first_metric(metrics, _ROI_KEYS),
        "max_drawdown": abs(_first_metric(metrics, _DRAWDOWN_KEYS)),
        "expectancy": _first_metric(metrics, _EXPECTANCY_KEYS),
        "profit_factor": _first_metric(metrics, _PROFIT_FACTOR_KEYS),
        "closed_trades": max(0.0, _first_metric(metrics, _TRADES_KEYS)),
        "win_rate": _first_metric(metrics, _WIN_RATE_KEYS),
    }


def assess_ai_uplift(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    *,
    model_name: str = "",
    model_parameters_b: float | None = None,
    policy: AIUpliftPolicy | None = None,
) -> AIUpliftReport:
    """Return whether AI-assisted evidence beats the non-AI ML baseline."""

    cfg = policy or AIUpliftPolicy()
    baseline = normalize_uplift_metrics(baseline_metrics)
    ai = normalize_uplift_metrics(ai_metrics)
    parameters_b = model_parameters_b
    if parameters_b is None:
        parameters_b = estimate_model_parameters_b(model_name)
    deltas = {
        "realized_pnl": ai["realized_pnl"] - baseline["realized_pnl"],
        "roi_pct": ai["roi_pct"] - baseline["roi_pct"],
        "max_drawdown": ai["max_drawdown"] - baseline["max_drawdown"],
        "expectancy": ai["expectancy"] - baseline["expectancy"],
        "profit_factor": ai["profit_factor"] - baseline["profit_factor"],
        "closed_trades": ai["closed_trades"] - baseline["closed_trades"],
        "win_rate": ai["win_rate"] - baseline["win_rate"],
    }
    reasons: list[str] = []
    if parameters_b is None:
        reasons.append("model_parameter_count_unknown")
    elif parameters_b < max(0.0, float(cfg.min_model_parameters_b)):
        reasons.append(
            f"model_parameters<{float(cfg.min_model_parameters_b):.2f}B"
        )
    if cfg.require_positive_ai_pnl and ai["realized_pnl"] <= 0.0:
        reasons.append("ai_realized_pnl<=0")
    if ai["closed_trades"] < max(0, int(cfg.min_ai_closed_trades)):
        reasons.append(f"ai_closed_trades<{int(cfg.min_ai_closed_trades)}")
    if deltas["realized_pnl"] <= float(cfg.min_pnl_delta):
        reasons.append("ai_pnl_not_above_baseline")
    if deltas["expectancy"] <= float(cfg.min_expectancy_delta):
        reasons.append("ai_expectancy_not_above_baseline")
    if deltas["max_drawdown"] > float(cfg.max_drawdown_delta):
        reasons.append("ai_drawdown_worse_than_baseline")
    accepted = not reasons
    return AIUpliftReport(
        accepted=accepted,
        advisory_only=not accepted,
        model_name=str(model_name or ""),
        model_parameters_b=parameters_b,
        baseline=baseline,
        ai=ai,
        deltas=deltas,
        reasons=tuple(reasons),
        policy=cfg.asdict(),
    )


__all__ = [
    "AIUpliftPolicy",
    "AIUpliftReport",
    "assess_ai_uplift",
    "normalize_uplift_metrics",
]

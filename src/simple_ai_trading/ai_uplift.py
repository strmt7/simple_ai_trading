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
_LIQUIDATION_KEYS = ("liquidation_events", "liquidations")
_LOSS_STREAK_KEYS = ("max_consecutive_losses", "loss_streak", "consecutive_losses")
_DOWNSIDE_RETURN_RISK_KEYS = (
    "downside_return_risk_ratio",
    "return_risk_ratio",
    "profit_drawdown_ratio",
    "calmar_ratio",
)
_RETURN_SAMPLE_KEYS = ("trade_returns", "returns", "return_samples")
_PNL_SAMPLE_KEYS = ("trade_pnls", "pnl_samples", "net_pnls")
_PAIRED_DELTA_KEYS = (
    "paired_return_deltas",
    "return_deltas",
    "trade_return_deltas",
    "uplift_return_deltas",
)


@dataclass(frozen=True)
class AIUpliftPolicy:
    """Minimum evidence required before AI-assisted alpha can be promoted."""

    min_model_parameters_b: float = 2.0
    min_ai_closed_trades: int = 5
    min_paired_samples: int = 8
    min_positive_delta_rate: float = 0.55
    max_sign_test_p_value: float = 0.40
    min_pnl_delta: float = 0.0
    min_expectancy_delta: float = 0.0
    min_mean_sample_delta: float = 0.0
    max_drawdown_delta: float = 0.0
    min_downside_return_risk_delta: float = 0.0
    max_loss_streak_delta: float = 0.0
    max_ai_liquidation_events: int = 0
    require_non_degrading_profit_factor: bool = True
    require_non_degrading_win_rate: bool = True
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
    statistical_evidence: dict[str, object]
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


def _numeric_sequence(metrics: Mapping[str, object], keys: tuple[str, ...]) -> tuple[float, ...]:
    for key in keys:
        raw = metrics.get(key)
        if not isinstance(raw, (tuple, list)):
            continue
        values: list[float] = []
        for value in raw:
            parsed = _finite(value, default=float("nan"))
            if not math.isfinite(parsed):
                values = []
                break
            values.append(float(parsed))
        if values:
            return tuple(values)
    return ()


def _paired_sample_deltas(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
) -> tuple[tuple[float, ...], bool, str]:
    direct = _numeric_sequence(ai_metrics, _PAIRED_DELTA_KEYS)
    if direct:
        return direct, False, "paired_trade_return_delta"
    baseline_returns = _numeric_sequence(baseline_metrics, _RETURN_SAMPLE_KEYS)
    ai_returns = _numeric_sequence(ai_metrics, _RETURN_SAMPLE_KEYS)
    if baseline_returns and ai_returns:
        count = min(len(baseline_returns), len(ai_returns))
        return (
            tuple(ai_returns[index] - baseline_returns[index] for index in range(count)),
            len(baseline_returns) != len(ai_returns),
            "paired_trade_return_delta",
        )
    baseline_pnls = _numeric_sequence(baseline_metrics, _PNL_SAMPLE_KEYS)
    ai_pnls = _numeric_sequence(ai_metrics, _PNL_SAMPLE_KEYS)
    if baseline_pnls and ai_pnls:
        count = min(len(baseline_pnls), len(ai_pnls))
        return (
            tuple(ai_pnls[index] - baseline_pnls[index] for index in range(count)),
            len(baseline_pnls) != len(ai_pnls),
            "paired_trade_pnl_delta",
        )
    return (), False, "none"


def _binomial_upper_tail(trials: int, successes: int, p: float = 0.5) -> float:
    n = max(0, int(trials))
    k = max(0, min(n, int(successes)))
    probability = max(0.0, min(1.0, float(p)))
    if n <= 0:
        return 1.0
    total = 0.0
    for hits in range(k, n + 1):
        total += math.comb(n, hits) * (probability ** hits) * ((1.0 - probability) ** (n - hits))
    return max(0.0, min(1.0, total))


def _median(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _statistical_evidence(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    policy: AIUpliftPolicy,
) -> dict[str, object]:
    deltas, length_mismatch, evidence_unit = _paired_sample_deltas(baseline_metrics, ai_metrics)
    sample_count = len(deltas)
    positive_count = sum(1 for value in deltas if value > 0.0)
    sign_p_value = _binomial_upper_tail(sample_count, positive_count)
    mean_delta = sum(deltas) / sample_count if sample_count else 0.0
    positive_rate = positive_count / sample_count if sample_count else 0.0
    reasons: list[str] = []
    if length_mismatch:
        reasons.append("ai_uplift_paired_sample_length_mismatch")
    if sample_count < max(0, int(policy.min_paired_samples)):
        reasons.append(f"ai_uplift_paired_samples<{int(policy.min_paired_samples)}")
    if positive_rate < max(0.0, min(1.0, float(policy.min_positive_delta_rate))):
        reasons.append(f"ai_uplift_positive_delta_rate<{float(policy.min_positive_delta_rate):.2f}")
    if sign_p_value > max(0.0, min(1.0, float(policy.max_sign_test_p_value))):
        reasons.append(f"ai_uplift_sign_test_p_value>{float(policy.max_sign_test_p_value):.4f}")
    if mean_delta <= float(policy.min_mean_sample_delta):
        reasons.append(f"ai_uplift_mean_sample_delta<={float(policy.min_mean_sample_delta):g}")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "evidence_unit": evidence_unit if deltas else "none",
        "paired_sample_length_mismatch": length_mismatch,
        "sample_count": sample_count,
        "min_sample_count": max(0, int(policy.min_paired_samples)),
        "positive_delta_count": positive_count,
        "positive_delta_rate": positive_rate,
        "min_positive_delta_rate": max(0.0, min(1.0, float(policy.min_positive_delta_rate))),
        "sign_test_p_value": sign_p_value,
        "max_sign_test_p_value": max(0.0, min(1.0, float(policy.max_sign_test_p_value))),
        "mean_delta": mean_delta,
        "median_delta": _median(deltas),
        "min_mean_sample_delta": float(policy.min_mean_sample_delta),
    }


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
        "liquidation_events": max(0.0, _first_metric(metrics, _LIQUIDATION_KEYS)),
        "max_consecutive_losses": max(0.0, _first_metric(metrics, _LOSS_STREAK_KEYS)),
        "downside_return_risk_ratio": _first_metric(metrics, _DOWNSIDE_RETURN_RISK_KEYS),
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
        "liquidation_events": ai["liquidation_events"] - baseline["liquidation_events"],
        "max_consecutive_losses": ai["max_consecutive_losses"] - baseline["max_consecutive_losses"],
        "downside_return_risk_ratio": ai["downside_return_risk_ratio"] - baseline["downside_return_risk_ratio"],
    }
    statistical = _statistical_evidence(baseline_metrics, ai_metrics, cfg)
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
    if not bool(statistical.get("accepted")):
        reasons.extend(str(reason) for reason in statistical.get("reasons", ()) if str(reason))
    if deltas["realized_pnl"] <= float(cfg.min_pnl_delta):
        reasons.append("ai_pnl_not_above_baseline")
    if deltas["expectancy"] <= float(cfg.min_expectancy_delta):
        reasons.append("ai_expectancy_not_above_baseline")
    if deltas["max_drawdown"] > float(cfg.max_drawdown_delta):
        reasons.append("ai_drawdown_worse_than_baseline")
    if ai["liquidation_events"] > max(0, int(cfg.max_ai_liquidation_events)):
        reasons.append("ai_liquidation_events>0")
    if deltas["max_consecutive_losses"] > float(cfg.max_loss_streak_delta):
        reasons.append("ai_loss_streak_worse_than_baseline")
    if (
        cfg.require_non_degrading_profit_factor
        and (baseline["profit_factor"] > 0.0 or ai["profit_factor"] > 0.0)
        and deltas["profit_factor"] < 0.0
    ):
        reasons.append("ai_profit_factor_worse_than_baseline")
    if (
        cfg.require_non_degrading_win_rate
        and (baseline["win_rate"] > 0.0 or ai["win_rate"] > 0.0)
        and deltas["win_rate"] < 0.0
    ):
        reasons.append("ai_win_rate_worse_than_baseline")
    if (
        (baseline["downside_return_risk_ratio"] > 0.0 or ai["downside_return_risk_ratio"] > 0.0)
        and deltas["downside_return_risk_ratio"] < float(cfg.min_downside_return_risk_delta)
    ):
        reasons.append("ai_downside_return_risk_not_above_baseline")
    accepted = not reasons
    return AIUpliftReport(
        accepted=accepted,
        advisory_only=not accepted,
        model_name=str(model_name or ""),
        model_parameters_b=parameters_b,
        baseline=baseline,
        ai=ai,
        deltas=deltas,
        statistical_evidence=statistical,
        reasons=tuple(reasons),
        policy=cfg.asdict(),
    )


__all__ = [
    "AIUpliftPolicy",
    "AIUpliftReport",
    "assess_ai_uplift",
    "normalize_uplift_metrics",
]

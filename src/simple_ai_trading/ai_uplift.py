"""Deterministic evidence gate for AI-assisted model uplift."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

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
_LEGACY_UNBOUND_SAMPLE_KEYS = (
    "trade_returns",
    "returns",
    "return_samples",
    "trade_pnls",
    "pnl_samples",
    "net_pnls",
    "paired_return_deltas",
    "return_deltas",
    "trade_return_deltas",
    "uplift_return_deltas",
)
_MIN_MODEL_PARAMETERS_B = 2.0
_MIN_PAIRED_SAMPLES = 30
_MAX_SIGN_TEST_P_VALUE = 0.05
_MIN_POSITIVE_DELTA_RATE = 0.55
_MIN_BLOCK_BOOTSTRAP_SAMPLES = 2_000
_MIN_BLOCK_BOOTSTRAP_CONFIDENCE = 0.95
_MIN_EVALUATION_SPAN_DAYS = 90
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class AIUpliftPolicy:
    """Minimum evidence required before AI-assisted alpha can be promoted."""

    min_model_parameters_b: float = 2.0
    min_ai_closed_trades: int = 5
    min_paired_samples: int = 30
    min_positive_delta_rate: float = 0.55
    max_sign_test_p_value: float = 0.05
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
    block_bootstrap_samples: int = 2_000
    block_bootstrap_confidence: float = 0.95
    min_bootstrap_mean_delta_lower: float = 0.0
    min_evaluation_span_days: int = 90
    require_evidence_binding: bool = True

    def __post_init__(self) -> None:
        numeric_values = (
            self.min_model_parameters_b,
            self.min_positive_delta_rate,
            self.max_sign_test_p_value,
            self.min_pnl_delta,
            self.min_expectancy_delta,
            self.min_mean_sample_delta,
            self.max_drawdown_delta,
            self.min_downside_return_risk_delta,
            self.max_loss_streak_delta,
            self.block_bootstrap_confidence,
            self.min_bootstrap_mean_delta_lower,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("AI uplift policy values must be finite")
        if self.min_model_parameters_b < _MIN_MODEL_PARAMETERS_B:
            raise ValueError("AI uplift model-size policy cannot weaken the 2B floor")
        if (
            self.min_ai_closed_trades < 5
            or self.min_paired_samples < _MIN_PAIRED_SAMPLES
        ):
            raise ValueError("AI uplift sample policy cannot weaken built-in floors")
        if self.min_evaluation_span_days < _MIN_EVALUATION_SPAN_DAYS:
            raise ValueError("AI uplift evaluation span cannot be shorter than 90 days")
        if not 0.0 <= self.max_sign_test_p_value <= _MAX_SIGN_TEST_P_VALUE:
            raise ValueError("AI uplift sign-test policy cannot exceed 0.05")
        if not _MIN_POSITIVE_DELTA_RATE <= self.min_positive_delta_rate <= 1.0:
            raise ValueError("AI uplift positive-delta policy cannot weaken 0.55")
        if self.block_bootstrap_samples < _MIN_BLOCK_BOOTSTRAP_SAMPLES:
            raise ValueError(
                "AI uplift bootstrap policy cannot use fewer than 2000 samples"
            )
        if not _MIN_BLOCK_BOOTSTRAP_CONFIDENCE <= self.block_bootstrap_confidence < 1.0:
            raise ValueError("AI uplift bootstrap confidence cannot be below 0.95")
        if self.min_bootstrap_mean_delta_lower < 0.0:
            raise ValueError(
                "AI uplift bootstrap lower-bound requirement cannot be negative"
            )
        if (
            self.min_pnl_delta < 0.0
            or self.min_expectancy_delta < 0.0
            or self.min_mean_sample_delta < 0.0
            or self.max_ai_liquidation_events > 0
        ):
            raise ValueError("AI uplift improvement policy cannot permit degradation")
        if self.max_drawdown_delta > 0.0 or self.max_loss_streak_delta > 0.0:
            raise ValueError("AI uplift tail-risk policy cannot permit deterioration")
        if self.min_downside_return_risk_delta < 0.0:
            raise ValueError(
                "AI uplift downside-risk policy cannot permit deterioration"
            )
        if not (
            self.require_non_degrading_profit_factor
            and self.require_non_degrading_win_rate
            and self.require_positive_ai_pnl
            and self.require_evidence_binding
        ):
            raise ValueError("AI uplift mandatory safety gates cannot be disabled")

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
    evidence_binding: dict[str, object]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    policy: dict[str, object] = field(default_factory=dict)
    schema_version: str = "ai-uplift-v3"
    trading_authority: bool = False
    profitability_claim: bool = False

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


def _required_source_metric_reasons(
    metrics: Mapping[str, object],
    prefix: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for name, keys in (
        ("realized_pnl", _PNL_KEYS),
        ("roi_pct", _ROI_KEYS),
        ("max_drawdown", _DRAWDOWN_KEYS),
        ("expectancy", _EXPECTANCY_KEYS),
        ("profit_factor", _PROFIT_FACTOR_KEYS),
        ("closed_trades", _TRADES_KEYS),
        ("win_rate", _WIN_RATE_KEYS),
        ("liquidation_events", _LIQUIDATION_KEYS),
        ("max_consecutive_losses", _LOSS_STREAK_KEYS),
        ("downside_return_risk_ratio", _DOWNSIDE_RETURN_RISK_KEYS),
    ):
        key = next((candidate for candidate in keys if candidate in metrics), None)
        if key is None:
            reasons.append(f"ai_uplift_{prefix}_{name}_missing")
            continue
        if isinstance(metrics[key], bool):
            reasons.append(f"ai_uplift_{prefix}_{name}_nonfinite")
            continue
        try:
            parsed = float(metrics[key])
        except (TypeError, ValueError, OverflowError):
            parsed = float("nan")
        if not math.isfinite(parsed):
            reasons.append(f"ai_uplift_{prefix}_{name}_nonfinite")
    return tuple(reasons)


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _matched_period_deltas(
    periods: Sequence[Mapping[str, object]] | None,
) -> tuple[tuple[float, ...], dict[str, object], tuple[str, ...]]:
    rows = tuple(periods or ())
    reasons: list[str] = []
    canonical: list[dict[str, object]] = []
    expected_scope = ""
    expected_duration = 0
    previous_end = -1
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            reasons.append(f"ai_uplift_period_{index}_not_mapping")
            continue
        scope = str(raw.get("scope") or "").strip()
        try:
            start_ms = int(raw.get("period_start_ms"))
            end_ms = int(raw.get("period_end_ms"))
            baseline_return = float(raw.get("baseline_return"))
            ai_return = float(raw.get("ai_return"))
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"ai_uplift_period_{index}_invalid")
            continue
        if (
            not scope
            or start_ms < 0
            or end_ms <= start_ms
            or not math.isfinite(baseline_return)
            or not math.isfinite(ai_return)
        ):
            reasons.append(f"ai_uplift_period_{index}_invalid")
            continue
        duration = end_ms - start_ms
        if not expected_scope:
            expected_scope = scope
            expected_duration = duration
        if scope != expected_scope:
            reasons.append("ai_uplift_period_scope_mismatch")
        if duration != expected_duration:
            reasons.append("ai_uplift_period_duration_mismatch")
        if previous_end >= 0 and start_ms != previous_end:
            reasons.append("ai_uplift_periods_not_contiguous")
        previous_end = end_ms
        canonical.append(
            {
                "scope": scope,
                "period_start_ms": start_ms,
                "period_end_ms": end_ms,
                "baseline_return": baseline_return,
                "ai_return": ai_return,
            }
        )
    if len(canonical) != len(rows):
        reasons.append("ai_uplift_period_rows_invalid")
    if not canonical:
        reasons.append("ai_uplift_matched_periods_missing")
        fingerprint = ""
    else:
        encoded = json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
        fingerprint = hashlib.sha256(encoded).hexdigest()
    deltas = tuple(
        float(row["ai_return"]) - float(row["baseline_return"]) for row in canonical
    )
    binding = {
        "evidence_unit": "matched_fixed_period_return_delta",
        "scope": expected_scope,
        "sample_count": len(canonical),
        "period_duration_ms": expected_duration,
        "first_period_start_ms": int(canonical[0]["period_start_ms"])
        if canonical
        else None,
        "last_period_end_ms": int(canonical[-1]["period_end_ms"])
        if canonical
        else None,
        "paired_samples_sha256": fingerprint,
    }
    return deltas, binding, tuple(dict.fromkeys(reasons))


def _binomial_upper_tail(trials: int, successes: int, p: float = 0.5) -> float:
    n = max(0, int(trials))
    k = max(0, min(n, int(successes)))
    probability = max(0.0, min(1.0, float(p)))
    if n <= 0:
        return 1.0
    if k <= 0 or probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return 0.0
    if k == n:
        return probability**n

    def interval_probability(start: int, end: int) -> float:
        log_probability = math.log(probability)
        log_complement = math.log1p(-probability)
        hits = start
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(hits + 1)
            - math.lgamma(n - hits + 1)
            + hits * log_probability
            + (n - hits) * log_complement
        )
        log_total = -math.inf
        while hits <= end:
            if log_total == -math.inf:
                log_total = log_term
            else:
                upper = max(log_total, log_term)
                lower = min(log_total, log_term)
                log_total = upper + math.log1p(math.exp(lower - upper))
            if hits < end:
                log_term += (
                    math.log(n - hits)
                    - math.log(hits + 1)
                    + log_probability
                    - log_complement
                )
            hits += 1
        return math.exp(log_total)

    mode = int(math.floor((n + 1) * probability))
    if k <= mode:
        tail = 1.0 - interval_probability(0, k - 1)
    else:
        tail = interval_probability(k, n)
    return max(0.0, min(1.0, tail))


def _median(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _moving_block_bootstrap(
    deltas: tuple[float, ...],
    *,
    samples: int,
    confidence: float,
    seed_material: str,
) -> dict[str, object]:
    count = len(deltas)
    repetitions = max(200, int(samples))
    confidence_level = max(0.80, min(0.999, float(confidence)))
    if count <= 0:
        return {
            "samples": repetitions,
            "confidence": confidence_level,
            "block_length": 0,
            "mean_delta_ci_lower": 0.0,
            "mean_delta_ci_upper": 0.0,
            "positive_mean_probability": 0.0,
        }
    block_length = max(1, min(count, int(round(math.sqrt(count)))))
    maximum_start = max(0, count - block_length)
    seed = int(hashlib.sha256(seed_material.encode("ascii")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    prefix = [0.0]
    for value in deltas:
        prefix.append(prefix[-1] + value)
    complete_blocks, remainder = divmod(count, block_length)
    means: list[float] = []
    for _ in range(repetitions):
        block_totals: list[float] = []
        for _block in range(complete_blocks):
            start = rng.randint(0, maximum_start) if maximum_start else 0
            block_totals.append(prefix[start + block_length] - prefix[start])
        if remainder:
            start = rng.randint(0, maximum_start) if maximum_start else 0
            block_totals.append(prefix[start + remainder] - prefix[start])
        means.append(math.fsum(block_totals) / count)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "samples": repetitions,
        "confidence": confidence_level,
        "block_length": block_length,
        "mean_delta_ci_lower": _quantile(means, tail),
        "mean_delta_ci_upper": _quantile(means, 1.0 - tail),
        "positive_mean_probability": sum(value > 0.0 for value in means) / repetitions,
    }


def _statistical_evidence(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    policy: AIUpliftPolicy,
    matched_periods: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    deltas, period_binding, period_reasons = _matched_period_deltas(matched_periods)
    sample_count = len(deltas)
    positive_count = sum(1 for value in deltas if value > 0.0)
    negative_count = sum(1 for value in deltas if value < 0.0)
    effective_sample_count = positive_count + negative_count
    tie_count = sample_count - effective_sample_count
    sign_p_value = _binomial_upper_tail(effective_sample_count, positive_count)
    mean_delta = sum(deltas) / sample_count if sample_count else 0.0
    positive_rate = (
        positive_count / effective_sample_count if effective_sample_count else 0.0
    )
    bootstrap = _moving_block_bootstrap(
        deltas,
        samples=policy.block_bootstrap_samples,
        confidence=policy.block_bootstrap_confidence,
        seed_material=str(period_binding["paired_samples_sha256"] or "missing"),
    )
    reasons = list(period_reasons)
    first_period_ms = period_binding.get("first_period_start_ms")
    last_period_ms = period_binding.get("last_period_end_ms")
    evaluation_span_ms = (
        int(last_period_ms) - int(first_period_ms)
        if first_period_ms is not None and last_period_ms is not None
        else 0
    )
    minimum_span_ms = int(policy.min_evaluation_span_days) * _DAY_MS
    if evaluation_span_ms < minimum_span_ms:
        reasons.append(
            f"ai_uplift_evaluation_span_days<{int(policy.min_evaluation_span_days)}"
        )
    if matched_periods is None and any(
        key in baseline_metrics or key in ai_metrics
        for key in _LEGACY_UNBOUND_SAMPLE_KEYS
    ):
        reasons.append("ai_uplift_unbound_trade_sequence_rejected")
    if effective_sample_count < max(0, int(policy.min_paired_samples)):
        reasons.append(f"ai_uplift_non_tied_pairs<{int(policy.min_paired_samples)}")
    if positive_rate < max(0.0, min(1.0, float(policy.min_positive_delta_rate))):
        reasons.append(
            f"ai_uplift_positive_delta_rate<{float(policy.min_positive_delta_rate):.2f}"
        )
    if sign_p_value > max(0.0, min(1.0, float(policy.max_sign_test_p_value))):
        reasons.append(
            f"ai_uplift_sign_test_p_value>{float(policy.max_sign_test_p_value):.4f}"
        )
    if mean_delta <= float(policy.min_mean_sample_delta):
        reasons.append(
            f"ai_uplift_mean_sample_delta<={float(policy.min_mean_sample_delta):g}"
        )
    bootstrap_lower = float(bootstrap["mean_delta_ci_lower"])
    if bootstrap_lower <= float(policy.min_bootstrap_mean_delta_lower):
        reasons.append(
            "ai_uplift_block_bootstrap_lower_mean_delta<="
            f"{float(policy.min_bootstrap_mean_delta_lower):g}"
        )
    return {
        "accepted": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        **period_binding,
        "paired_sample_length_mismatch": False,
        "sample_count": sample_count,
        "effective_sample_count": effective_sample_count,
        "min_effective_sample_count": max(0, int(policy.min_paired_samples)),
        "positive_delta_count": positive_count,
        "negative_delta_count": negative_count,
        "tie_count": tie_count,
        "positive_delta_rate": positive_rate,
        "min_positive_delta_rate": max(
            0.0, min(1.0, float(policy.min_positive_delta_rate))
        ),
        "sign_test_p_value": sign_p_value,
        "max_sign_test_p_value": max(
            0.0, min(1.0, float(policy.max_sign_test_p_value))
        ),
        "mean_delta": mean_delta,
        "median_delta": _median(deltas),
        "min_mean_sample_delta": float(policy.min_mean_sample_delta),
        "block_bootstrap_samples": int(bootstrap["samples"]),
        "block_bootstrap_confidence": float(bootstrap["confidence"]),
        "block_length": int(bootstrap["block_length"]),
        "mean_delta_ci_lower": bootstrap_lower,
        "mean_delta_ci_upper": float(bootstrap["mean_delta_ci_upper"]),
        "positive_mean_probability": float(bootstrap["positive_mean_probability"]),
        "min_bootstrap_mean_delta_lower": float(policy.min_bootstrap_mean_delta_lower),
        "evaluation_span_ms": evaluation_span_ms,
        "min_evaluation_span_ms": minimum_span_ms,
    }


def _uplift_evidence_binding(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    *,
    model_artifact_sha256: str,
    paired_samples_sha256: object,
) -> dict[str, object]:
    baseline_dataset = str(baseline_metrics.get("dataset_fingerprint") or "").lower()
    ai_dataset = str(ai_metrics.get("dataset_fingerprint") or "").lower()
    baseline_artifact = str(baseline_metrics.get("evidence_sha256") or "").lower()
    ai_artifact = str(ai_metrics.get("evidence_sha256") or "").lower()
    model_artifact = str(model_artifact_sha256 or "").lower()
    paired_artifact = str(paired_samples_sha256 or "").lower()
    reasons: list[str] = []
    for label, value in (
        ("baseline_dataset_fingerprint", baseline_dataset),
        ("ai_dataset_fingerprint", ai_dataset),
        ("baseline_evidence_sha256", baseline_artifact),
        ("ai_evidence_sha256", ai_artifact),
        ("model_artifact_sha256", model_artifact),
        ("paired_samples_sha256", paired_artifact),
    ):
        if not _is_sha256(value):
            reasons.append(f"ai_uplift_{label}_invalid")
    if baseline_dataset != ai_dataset:
        reasons.append("ai_uplift_dataset_fingerprint_mismatch")
    return {
        "accepted": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "dataset_fingerprint": baseline_dataset
        if baseline_dataset == ai_dataset
        else "",
        "baseline_evidence_sha256": baseline_artifact,
        "ai_evidence_sha256": ai_artifact,
        "model_artifact_sha256": model_artifact,
        "paired_samples_sha256": paired_artifact,
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
        "downside_return_risk_ratio": _first_metric(
            metrics, _DOWNSIDE_RETURN_RISK_KEYS
        ),
    }


def assess_ai_uplift(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    *,
    model_name: str = "",
    model_parameters_b: float | None = None,
    model_artifact_sha256: str = "",
    matched_periods: Sequence[Mapping[str, object]] | None = None,
    policy: AIUpliftPolicy | None = None,
) -> AIUpliftReport:
    """Return whether AI-assisted evidence beats the non-AI ML baseline."""

    cfg = policy or AIUpliftPolicy()
    baseline = normalize_uplift_metrics(baseline_metrics)
    ai = normalize_uplift_metrics(ai_metrics)
    parameters_b = model_parameters_b
    if parameters_b is None:
        parameters_b = estimate_model_parameters_b(model_name)
    try:
        parsed_parameters_b = float(parameters_b) if parameters_b is not None else None
    except (TypeError, ValueError, OverflowError):
        parsed_parameters_b = None
    parameters_b = (
        parsed_parameters_b
        if parsed_parameters_b is not None and math.isfinite(parsed_parameters_b)
        else None
    )
    deltas = {
        "realized_pnl": ai["realized_pnl"] - baseline["realized_pnl"],
        "roi_pct": ai["roi_pct"] - baseline["roi_pct"],
        "max_drawdown": ai["max_drawdown"] - baseline["max_drawdown"],
        "expectancy": ai["expectancy"] - baseline["expectancy"],
        "profit_factor": ai["profit_factor"] - baseline["profit_factor"],
        "closed_trades": ai["closed_trades"] - baseline["closed_trades"],
        "win_rate": ai["win_rate"] - baseline["win_rate"],
        "liquidation_events": ai["liquidation_events"] - baseline["liquidation_events"],
        "max_consecutive_losses": ai["max_consecutive_losses"]
        - baseline["max_consecutive_losses"],
        "downside_return_risk_ratio": ai["downside_return_risk_ratio"]
        - baseline["downside_return_risk_ratio"],
    }
    statistical = _statistical_evidence(
        baseline_metrics,
        ai_metrics,
        cfg,
        matched_periods,
    )
    evidence_binding = _uplift_evidence_binding(
        baseline_metrics,
        ai_metrics,
        model_artifact_sha256=model_artifact_sha256,
        paired_samples_sha256=statistical.get("paired_samples_sha256"),
    )
    reasons = list(_required_source_metric_reasons(baseline_metrics, "baseline"))
    reasons.extend(_required_source_metric_reasons(ai_metrics, "ai"))
    if parameters_b is None:
        reasons.append("model_parameter_count_unknown")
    elif parameters_b < max(0.0, float(cfg.min_model_parameters_b)):
        reasons.append(f"model_parameters<{float(cfg.min_model_parameters_b):.2f}B")
    if cfg.require_positive_ai_pnl and ai["realized_pnl"] <= 0.0:
        reasons.append("ai_realized_pnl<=0")
    if ai["closed_trades"] < max(0, int(cfg.min_ai_closed_trades)):
        reasons.append(f"ai_closed_trades<{int(cfg.min_ai_closed_trades)}")
    if not bool(statistical.get("accepted")):
        reasons.extend(
            str(reason) for reason in statistical.get("reasons", ()) if str(reason)
        )
    if cfg.require_evidence_binding and not bool(evidence_binding.get("accepted")):
        reasons.extend(
            str(reason) for reason in evidence_binding.get("reasons", ()) if str(reason)
        )
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
        baseline["downside_return_risk_ratio"] > 0.0
        or ai["downside_return_risk_ratio"] > 0.0
    ) and deltas["downside_return_risk_ratio"] < float(
        cfg.min_downside_return_risk_delta
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
        evidence_binding=evidence_binding,
        reasons=tuple(dict.fromkeys(reasons)),
        policy=cfg.asdict(),
    )


__all__ = [
    "AIUpliftPolicy",
    "AIUpliftReport",
    "assess_ai_uplift",
    "normalize_uplift_metrics",
]

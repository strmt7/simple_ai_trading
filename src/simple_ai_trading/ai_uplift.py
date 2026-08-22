"""Deterministic evidence gate for AI-assisted model uplift."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence, SupportsFloat, SupportsIndex, TypedDict

from .ai_runtime import estimate_model_parameters_b
from .statistical_resampling import moving_block_bootstrap_mean


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


class _MatchedPeriodRow(TypedDict):
    scope: str
    period_start_ms: int
    period_end_ms: int
    baseline_return: float
    ai_return: float


class _PeriodBinding(TypedDict):
    evidence_unit: str
    scope: str
    sample_count: int
    period_duration_ms: int
    first_period_start_ms: int | None
    last_period_end_ms: int | None
    paired_samples_sha256: str


class _BootstrapEvidence(TypedDict):
    samples: int
    confidence: float
    block_length: int
    mean_delta_ci_lower: float
    mean_delta_ci_upper: float
    positive_mean_probability: float


class _PairedStatistics(TypedDict):
    sample_count: int
    effective_sample_count: int
    positive_count: int
    negative_count: int
    tie_count: int
    positive_rate: float
    sign_p_value: float
    mean_delta: float


@dataclass(frozen=True)
class AIUpliftPolicy:
    """Minimum evidence required before AI-assisted alpha can be promoted."""

    min_model_parameters_b: float = 2.0
    min_ai_closed_trades: int = 5
    min_paired_samples: int = 30
    min_positive_delta_rate: float = 0.55
    max_sign_test_p_value: float = 0.05
    min_pnl_delta: float = 0.0
    min_roi_delta: float = 0.0
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
        _validate_policy_numeric_values(self)
        _validate_policy_evidence_floors(self)
        _validate_policy_improvement_gates(self)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _validate_policy_numeric_values(policy: AIUpliftPolicy) -> None:
    values = (
        policy.min_model_parameters_b,
        policy.min_positive_delta_rate,
        policy.max_sign_test_p_value,
        policy.min_pnl_delta,
        policy.min_roi_delta,
        policy.min_expectancy_delta,
        policy.min_mean_sample_delta,
        policy.max_drawdown_delta,
        policy.min_downside_return_risk_delta,
        policy.max_loss_streak_delta,
        policy.block_bootstrap_confidence,
        policy.min_bootstrap_mean_delta_lower,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("AI uplift policy values must be finite")


def _validate_policy_evidence_floors(policy: AIUpliftPolicy) -> None:
    if policy.min_model_parameters_b < _MIN_MODEL_PARAMETERS_B:
        raise ValueError("AI uplift model-size policy cannot weaken the 2B floor")
    if (
        policy.min_ai_closed_trades < 5
        or policy.min_paired_samples < _MIN_PAIRED_SAMPLES
    ):
        raise ValueError("AI uplift sample policy cannot weaken built-in floors")
    if policy.min_evaluation_span_days < _MIN_EVALUATION_SPAN_DAYS:
        raise ValueError("AI uplift evaluation span cannot be shorter than 90 days")
    if not 0.0 <= policy.max_sign_test_p_value <= _MAX_SIGN_TEST_P_VALUE:
        raise ValueError("AI uplift sign-test policy cannot exceed 0.05")
    if not _MIN_POSITIVE_DELTA_RATE <= policy.min_positive_delta_rate <= 1.0:
        raise ValueError("AI uplift positive-delta policy cannot weaken 0.55")
    if policy.block_bootstrap_samples < _MIN_BLOCK_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "AI uplift bootstrap policy cannot use fewer than 2000 samples"
        )
    if not _MIN_BLOCK_BOOTSTRAP_CONFIDENCE <= policy.block_bootstrap_confidence < 1.0:
        raise ValueError("AI uplift bootstrap confidence cannot be below 0.95")
    if policy.min_bootstrap_mean_delta_lower < 0.0:
        raise ValueError(
            "AI uplift bootstrap lower-bound requirement cannot be negative"
        )


def _validate_policy_improvement_gates(policy: AIUpliftPolicy) -> None:
    degradation_allowed = any(
        (
            policy.min_pnl_delta < 0.0,
            policy.min_roi_delta < 0.0,
            policy.min_expectancy_delta < 0.0,
            policy.min_mean_sample_delta < 0.0,
            policy.max_ai_liquidation_events > 0,
        )
    )
    if degradation_allowed:
        raise ValueError("AI uplift improvement policy cannot permit degradation")
    tail_risk_deterioration_allowed = any(
        (
            policy.max_drawdown_delta > 0.0,
            policy.max_loss_streak_delta > 0.0,
        )
    )
    if tail_risk_deterioration_allowed:
        raise ValueError("AI uplift tail-risk policy cannot permit deterioration")
    if policy.min_downside_return_risk_delta < 0.0:
        raise ValueError("AI uplift downside-risk policy cannot permit deterioration")
    mandatory_gates = (
        policy.require_non_degrading_profit_factor,
        policy.require_non_degrading_win_rate,
        policy.require_positive_ai_pnl,
        policy.require_evidence_binding,
    )
    if not all(mandatory_gates):
        raise ValueError("AI uplift mandatory safety gates cannot be disabled")


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
    schema_version: str = "ai-uplift-v5"
    trading_authority: bool = False
    profitability_claim: bool = False

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (str, bytes, bytearray, SupportsFloat, SupportsIndex),
    ):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _integer(value: object) -> int | None:
    parsed = _finite(value, default=float("nan"))
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def _reason_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return ()
    return tuple(str(reason) for reason in value if str(reason))


def _first_metric(metrics: Mapping[str, object], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in metrics:
            return _finite(metrics[key])
    return 0.0


def _metric_source_keys(metrics: Mapping[str, object]) -> dict[str, str]:
    """Record the exact source key used for every normalized financial metric."""

    return {
        name: next((key for key in keys if key in metrics), "")
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
        )
    }


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
        parsed = _finite(metrics[key], default=float("nan"))
        if not math.isfinite(parsed):
            reasons.append(f"ai_uplift_{prefix}_{name}_nonfinite")
            continue
        if name in {
            "closed_trades",
            "liquidation_events",
            "max_consecutive_losses",
        } and (parsed < 0.0 or not parsed.is_integer()):
            reasons.append(f"ai_uplift_{prefix}_{name}_invalid_count")
        elif name in {"max_drawdown", "profit_factor"} and parsed < 0.0:
            reasons.append(f"ai_uplift_{prefix}_{name}_invalid_range")
        elif name == "win_rate":
            upper = 100.0 if key == "win_rate_pct" else 1.0
            if not 0.0 <= parsed <= upper:
                reasons.append(f"ai_uplift_{prefix}_{name}_invalid_range")
    return tuple(reasons)


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _parse_matched_period(
    index: int,
    raw: object,
) -> tuple[_MatchedPeriodRow | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, f"ai_uplift_period_{index}_not_mapping"
    scope = str(raw.get("scope") or "").strip()
    start_ms = _integer(raw.get("period_start_ms"))
    end_ms = _integer(raw.get("period_end_ms"))
    baseline_return = _finite(raw.get("baseline_return"), default=float("nan"))
    ai_return = _finite(raw.get("ai_return"), default=float("nan"))
    if start_ms is None or end_ms is None:
        return None, f"ai_uplift_period_{index}_invalid"
    valid = all(
        (
            bool(scope),
            start_ms >= 0,
            end_ms > start_ms,
            math.isfinite(baseline_return),
            math.isfinite(ai_return),
        )
    )
    if not valid:
        return None, f"ai_uplift_period_{index}_invalid"
    return (
        {
            "scope": scope,
            "period_start_ms": start_ms,
            "period_end_ms": end_ms,
            "baseline_return": baseline_return,
            "ai_return": ai_return,
        },
        None,
    )


def _period_consistency(
    row: _MatchedPeriodRow,
    *,
    expected_scope: str,
    expected_duration: int,
    previous_end: int,
) -> tuple[str, int, int, tuple[str, ...]]:
    scope = row["scope"]
    start_ms = row["period_start_ms"]
    end_ms = row["period_end_ms"]
    duration = end_ms - start_ms
    if not expected_scope:
        expected_scope = scope
        expected_duration = duration
    reasons: list[str] = []
    if scope != expected_scope:
        reasons.append("ai_uplift_period_scope_mismatch")
    if duration != expected_duration:
        reasons.append("ai_uplift_period_duration_mismatch")
    if previous_end >= 0 and start_ms != previous_end:
        reasons.append("ai_uplift_periods_not_contiguous")
    return expected_scope, expected_duration, end_ms, tuple(reasons)


def _matched_period_fingerprint(canonical: Sequence[_MatchedPeriodRow]) -> str:
    if not canonical:
        return ""
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _matched_period_deltas(
    periods: Sequence[Mapping[str, object]] | None,
) -> tuple[tuple[float, ...], _PeriodBinding, tuple[str, ...]]:
    rows = tuple(periods or ())
    reasons: list[str] = []
    canonical: list[_MatchedPeriodRow] = []
    expected_scope = ""
    expected_duration = 0
    previous_end = -1
    for index, raw in enumerate(rows):
        parsed, parse_reason = _parse_matched_period(index, raw)
        if parsed is None:
            reasons.append(str(parse_reason))
            continue
        expected_scope, expected_duration, previous_end, consistency_reasons = (
            _period_consistency(
                parsed,
                expected_scope=expected_scope,
                expected_duration=expected_duration,
                previous_end=previous_end,
            )
        )
        reasons.extend(consistency_reasons)
        canonical.append(parsed)
    if len(canonical) != len(rows):
        reasons.append("ai_uplift_period_rows_invalid")
    if not canonical:
        reasons.append("ai_uplift_matched_periods_missing")
    fingerprint = _matched_period_fingerprint(canonical)
    deltas = tuple(row["ai_return"] - row["baseline_return"] for row in canonical)
    binding: _PeriodBinding = {
        "evidence_unit": "matched_fixed_period_return_delta",
        "scope": expected_scope,
        "sample_count": len(canonical),
        "period_duration_ms": expected_duration,
        "first_period_start_ms": canonical[0]["period_start_ms"] if canonical else None,
        "last_period_end_ms": canonical[-1]["period_end_ms"] if canonical else None,
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


def _moving_block_bootstrap(
    deltas: tuple[float, ...],
    *,
    samples: int,
    confidence: float,
    seed_material: str,
) -> _BootstrapEvidence:
    evidence = moving_block_bootstrap_mean(
        deltas,
        samples=samples,
        confidence=confidence,
        seed_material=seed_material,
    )
    parsed_samples = _integer(evidence.get("samples"))
    parsed_block_length = _integer(evidence.get("block_length"))
    parsed_values = {
        key: _finite(evidence.get(key), default=float("nan"))
        for key in (
            "confidence",
            "mean_ci_lower",
            "mean_ci_upper",
            "positive_mean_probability",
        )
    }
    if (
        parsed_samples is None
        or parsed_block_length is None
        or any(not math.isfinite(value) for value in parsed_values.values())
    ):
        raise ValueError("moving-block bootstrap returned invalid evidence")
    return {
        "samples": parsed_samples,
        "confidence": parsed_values["confidence"],
        "block_length": parsed_block_length,
        "mean_delta_ci_lower": parsed_values["mean_ci_lower"],
        "mean_delta_ci_upper": parsed_values["mean_ci_upper"],
        "positive_mean_probability": parsed_values["positive_mean_probability"],
    }


def _paired_statistics(deltas: Sequence[float]) -> _PairedStatistics:
    sample_count = len(deltas)
    positive_count = sum(1 for value in deltas if value > 0.0)
    negative_count = sum(1 for value in deltas if value < 0.0)
    effective_sample_count = positive_count + negative_count
    return {
        "sample_count": sample_count,
        "effective_sample_count": effective_sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "tie_count": sample_count - effective_sample_count,
        "positive_rate": (
            positive_count / effective_sample_count if effective_sample_count else 0.0
        ),
        "sign_p_value": _binomial_upper_tail(effective_sample_count, positive_count),
        "mean_delta": sum(deltas) / sample_count if sample_count else 0.0,
    }


def _evaluation_span_ms(period_binding: _PeriodBinding) -> int:
    first_period_ms = period_binding["first_period_start_ms"]
    last_period_ms = period_binding["last_period_end_ms"]
    if first_period_ms is None or last_period_ms is None:
        return 0
    return last_period_ms - first_period_ms


def _statistical_rejection_reasons(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    policy: AIUpliftPolicy,
    matched_periods: Sequence[Mapping[str, object]] | None,
    period_reasons: Sequence[str],
    statistics: _PairedStatistics,
    *,
    evaluation_span_ms: int,
    minimum_span_ms: int,
    bootstrap_lower: float,
) -> tuple[str, ...]:
    reasons = list(period_reasons)
    if evaluation_span_ms < minimum_span_ms:
        reasons.append(
            f"ai_uplift_evaluation_span_days<{int(policy.min_evaluation_span_days)}"
        )
    if matched_periods is None and any(
        key in baseline_metrics or key in ai_metrics
        for key in _LEGACY_UNBOUND_SAMPLE_KEYS
    ):
        reasons.append("ai_uplift_unbound_trade_sequence_rejected")
    if statistics["effective_sample_count"] < max(0, int(policy.min_paired_samples)):
        reasons.append(f"ai_uplift_non_tied_pairs<{int(policy.min_paired_samples)}")
    minimum_positive_rate = max(
        0.0,
        min(1.0, float(policy.min_positive_delta_rate)),
    )
    if statistics["positive_rate"] < minimum_positive_rate:
        reasons.append(
            f"ai_uplift_positive_delta_rate<{float(policy.min_positive_delta_rate):.2f}"
        )
    maximum_sign_p_value = max(
        0.0,
        min(1.0, float(policy.max_sign_test_p_value)),
    )
    if statistics["sign_p_value"] > maximum_sign_p_value:
        reasons.append(
            f"ai_uplift_sign_test_p_value>{float(policy.max_sign_test_p_value):.4f}"
        )
    if statistics["mean_delta"] <= float(policy.min_mean_sample_delta):
        reasons.append(
            f"ai_uplift_mean_sample_delta<={float(policy.min_mean_sample_delta):g}"
        )
    if bootstrap_lower <= float(policy.min_bootstrap_mean_delta_lower):
        reasons.append(
            "ai_uplift_block_bootstrap_lower_mean_delta<="
            f"{float(policy.min_bootstrap_mean_delta_lower):g}"
        )
    return tuple(dict.fromkeys(reasons))


def _statistical_evidence(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    policy: AIUpliftPolicy,
    matched_periods: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    deltas, period_binding, period_reasons = _matched_period_deltas(matched_periods)
    statistics = _paired_statistics(deltas)
    bootstrap = _moving_block_bootstrap(
        deltas,
        samples=policy.block_bootstrap_samples,
        confidence=policy.block_bootstrap_confidence,
        seed_material=str(period_binding["paired_samples_sha256"] or "missing"),
    )
    evaluation_span_ms = _evaluation_span_ms(period_binding)
    minimum_span_ms = int(policy.min_evaluation_span_days) * _DAY_MS
    bootstrap_lower = bootstrap["mean_delta_ci_lower"]
    reasons = _statistical_rejection_reasons(
        baseline_metrics,
        ai_metrics,
        policy,
        matched_periods,
        period_reasons,
        statistics,
        evaluation_span_ms=evaluation_span_ms,
        minimum_span_ms=minimum_span_ms,
        bootstrap_lower=bootstrap_lower,
    )
    return {
        "accepted": not reasons,
        "reasons": list(reasons),
        **period_binding,
        "paired_sample_length_mismatch": False,
        "sample_count": statistics["sample_count"],
        "effective_sample_count": statistics["effective_sample_count"],
        "min_effective_sample_count": max(0, int(policy.min_paired_samples)),
        "positive_delta_count": statistics["positive_count"],
        "negative_delta_count": statistics["negative_count"],
        "tie_count": statistics["tie_count"],
        "positive_delta_rate": statistics["positive_rate"],
        "min_positive_delta_rate": max(
            0.0, min(1.0, float(policy.min_positive_delta_rate))
        ),
        "sign_test_p_value": statistics["sign_p_value"],
        "max_sign_test_p_value": max(
            0.0, min(1.0, float(policy.max_sign_test_p_value))
        ),
        "mean_delta": statistics["mean_delta"],
        "median_delta": _median(deltas),
        "min_mean_sample_delta": float(policy.min_mean_sample_delta),
        "block_bootstrap_samples": bootstrap["samples"],
        "block_bootstrap_confidence": bootstrap["confidence"],
        "block_length": bootstrap["block_length"],
        "mean_delta_ci_lower": bootstrap_lower,
        "mean_delta_ci_upper": bootstrap["mean_delta_ci_upper"],
        "positive_mean_probability": bootstrap["positive_mean_probability"],
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
    baseline_sources = _metric_source_keys(baseline_metrics)
    ai_sources = _metric_source_keys(ai_metrics)
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
    for name in baseline_sources:
        if (
            baseline_sources[name]
            and ai_sources[name]
            and baseline_sources[name] != ai_sources[name]
        ):
            reasons.append(f"ai_uplift_{name}_source_key_mismatch")
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
        "baseline_metric_sources": baseline_sources,
        "ai_metric_sources": ai_sources,
    }


def normalize_uplift_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Normalize common backtest metric names into the AI uplift contract."""

    drawdown_source = next(
        (key for key in _DRAWDOWN_KEYS if key in metrics),
        "",
    )
    normalized_drawdown = abs(_first_metric(metrics, _DRAWDOWN_KEYS))
    if drawdown_source == "max_drawdown_pct":
        normalized_drawdown /= 100.0
    win_rate_source = next((key for key in _WIN_RATE_KEYS if key in metrics), "")
    normalized_win_rate = _first_metric(metrics, _WIN_RATE_KEYS)
    if win_rate_source == "win_rate_pct":
        normalized_win_rate /= 100.0
    return {
        "realized_pnl": _first_metric(metrics, _PNL_KEYS),
        "roi_pct": _first_metric(metrics, _ROI_KEYS),
        "max_drawdown": normalized_drawdown,
        "expectancy": _first_metric(metrics, _EXPECTANCY_KEYS),
        "profit_factor": _first_metric(metrics, _PROFIT_FACTOR_KEYS),
        "closed_trades": max(0.0, _first_metric(metrics, _TRADES_KEYS)),
        "win_rate": normalized_win_rate,
        "liquidation_events": max(0.0, _first_metric(metrics, _LIQUIDATION_KEYS)),
        "max_consecutive_losses": max(0.0, _first_metric(metrics, _LOSS_STREAK_KEYS)),
        "downside_return_risk_ratio": _first_metric(
            metrics, _DOWNSIDE_RETURN_RISK_KEYS
        ),
    }


def _model_parameters_b(model_name: str, supplied: float | None) -> float | None:
    candidate = supplied
    if candidate is None:
        candidate = estimate_model_parameters_b(model_name)
    try:
        parsed = float(candidate) if candidate is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def _uplift_deltas(
    baseline: Mapping[str, float],
    ai: Mapping[str, float],
) -> dict[str, float]:
    return {
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


def _model_and_evidence_reasons(
    baseline_metrics: Mapping[str, object],
    ai_metrics: Mapping[str, object],
    ai: Mapping[str, float],
    policy: AIUpliftPolicy,
    parameters_b: float | None,
    statistical: Mapping[str, object],
    evidence_binding: Mapping[str, object],
) -> tuple[str, ...]:
    reasons = list(_required_source_metric_reasons(baseline_metrics, "baseline"))
    reasons.extend(_required_source_metric_reasons(ai_metrics, "ai"))
    if parameters_b is None:
        reasons.append("model_parameter_count_unknown")
    elif parameters_b < max(0.0, float(policy.min_model_parameters_b)):
        reasons.append(f"model_parameters<{float(policy.min_model_parameters_b):.2f}B")
    if policy.require_positive_ai_pnl and ai["realized_pnl"] <= 0.0:
        reasons.append("ai_realized_pnl<=0")
    if ai["closed_trades"] < max(0, int(policy.min_ai_closed_trades)):
        reasons.append(f"ai_closed_trades<{int(policy.min_ai_closed_trades)}")
    if not bool(statistical.get("accepted")):
        reasons.extend(_reason_strings(statistical.get("reasons")))
    if policy.require_evidence_binding and not bool(evidence_binding.get("accepted")):
        reasons.extend(_reason_strings(evidence_binding.get("reasons")))
    return tuple(reasons)


def _aggregate_improvement_reasons(
    deltas: Mapping[str, float],
    policy: AIUpliftPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if deltas["realized_pnl"] <= float(policy.min_pnl_delta):
        reasons.append("ai_pnl_not_above_baseline")
    if deltas["roi_pct"] <= float(policy.min_roi_delta):
        reasons.append("ai_roi_not_above_baseline")
    if deltas["expectancy"] <= float(policy.min_expectancy_delta):
        reasons.append("ai_expectancy_not_above_baseline")
    if deltas["max_drawdown"] > float(policy.max_drawdown_delta):
        reasons.append("ai_drawdown_worse_than_baseline")
    return tuple(reasons)


def _tail_risk_reasons(
    baseline: Mapping[str, float],
    ai: Mapping[str, float],
    deltas: Mapping[str, float],
    policy: AIUpliftPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if ai["liquidation_events"] > max(0, int(policy.max_ai_liquidation_events)):
        reasons.append("ai_liquidation_events>0")
    if deltas["max_consecutive_losses"] > float(policy.max_loss_streak_delta):
        reasons.append("ai_loss_streak_worse_than_baseline")
    profit_factor_observed = any(
        (baseline["profit_factor"] > 0.0, ai["profit_factor"] > 0.0)
    )
    if (
        policy.require_non_degrading_profit_factor
        and profit_factor_observed
        and deltas["profit_factor"] < 0.0
    ):
        reasons.append("ai_profit_factor_worse_than_baseline")
    win_rate_observed = any((baseline["win_rate"] > 0.0, ai["win_rate"] > 0.0))
    if (
        policy.require_non_degrading_win_rate
        and win_rate_observed
        and deltas["win_rate"] < 0.0
    ):
        reasons.append("ai_win_rate_worse_than_baseline")
    downside_risk_observed = any(
        (
            baseline["downside_return_risk_ratio"] > 0.0,
            ai["downside_return_risk_ratio"] > 0.0,
        )
    )
    if downside_risk_observed and deltas["downside_return_risk_ratio"] < float(
        policy.min_downside_return_risk_delta
    ):
        reasons.append("ai_downside_return_risk_not_above_baseline")
    return tuple(reasons)


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
    parameters_b = _model_parameters_b(model_name, model_parameters_b)
    deltas = _uplift_deltas(baseline, ai)
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
    reasons = list(
        _model_and_evidence_reasons(
            baseline_metrics,
            ai_metrics,
            ai,
            cfg,
            parameters_b,
            statistical,
            evidence_binding,
        )
    )
    reasons.extend(_aggregate_improvement_reasons(deltas, cfg))
    reasons.extend(_tail_risk_reasons(baseline, ai, deltas, cfg))
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

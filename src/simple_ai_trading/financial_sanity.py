"""Financial sanity checks for model and model-lab artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, SupportsFloat, SupportsIndex, cast

from .model import TrainedModel
from .terminal_holdout_ledger import (
    reservation_evidence_passed,
    terminal_result_fingerprint,
)

_PREFERRED_PROBABILITY_BRIER_MAX = 0.30
_HARD_PROBABILITY_BRIER_MAX = 0.35
_PREFERRED_PROBABILITY_ECE_MAX = 0.15
_HARD_PROBABILITY_ECE_MAX = 0.20
_AI_UPLIFT_REQUIRED_METRICS = (
    "realized_pnl",
    "roi_pct",
    "max_drawdown",
    "expectancy",
    "profit_factor",
    "closed_trades",
    "win_rate",
    "liquidation_events",
    "max_consecutive_losses",
    "downside_return_risk_ratio",
)
_AI_UPLIFT_SOURCE_KEYS = {
    "realized_pnl": ("realized_pnl", "net_pnl", "pnl"),
    "roi_pct": ("roi_pct", "roi", "return_pct", "net_return_pct"),
    "max_drawdown": ("max_drawdown", "max_drawdown_pct", "drawdown"),
    "expectancy": ("expectancy", "edge", "mean_trade_pnl"),
    "profit_factor": ("profit_factor",),
    "closed_trades": ("closed_trades", "trade_count", "trades"),
    "win_rate": ("win_rate", "win_rate_pct"),
    "liquidation_events": ("liquidation_events", "liquidations"),
    "max_consecutive_losses": (
        "max_consecutive_losses",
        "loss_streak",
        "consecutive_losses",
    ),
    "downside_return_risk_ratio": (
        "downside_return_risk_ratio",
        "return_risk_ratio",
        "profit_drawdown_ratio",
        "calmar_ratio",
    ),
}
_AI_UPLIFT_DEFAULT_MIN_MODEL_PARAMETERS_B = 2.0
_AI_UPLIFT_DEFAULT_MIN_AI_CLOSED_TRADES = 5
_AI_UPLIFT_DEFAULT_MIN_PAIRED_SAMPLES = 30
_AI_UPLIFT_DEFAULT_MAX_SIGN_TEST_P = 0.05
_AI_UPLIFT_DEFAULT_MIN_POSITIVE_DELTA_RATE = 0.55
_AI_UPLIFT_DEFAULT_MIN_PNL_DELTA = 0.0
_AI_UPLIFT_DEFAULT_MIN_ROI_DELTA = 0.0
_AI_UPLIFT_DEFAULT_MIN_EXPECTANCY_DELTA = 0.0
_AI_UPLIFT_DEFAULT_MAX_DRAWDOWN_DELTA = 0.0
_AI_UPLIFT_DEFAULT_MIN_MEAN_SAMPLE_DELTA = 0.0
_AI_UPLIFT_DEFAULT_BOOTSTRAP_SAMPLES = 2_000
_AI_UPLIFT_DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
_AI_UPLIFT_DEFAULT_MIN_BOOTSTRAP_LOWER = 0.0
_AI_UPLIFT_DEFAULT_MIN_EVALUATION_SPAN_DAYS = 90
_DAY_MS = 86_400_000
_REQUIRED_DATA_COVERAGE_TRUTH_BASIS = (
    "prices_from_timestamped_closed_candles",
    "coverage_measured_from_candle_close_time",
    "execution_results_are_simulated_not_exchange_fills",
)
_BLOCKED_DATA_SOURCE_TOKENS = (
    "synthetic",
    "fake",
    "mock",
    "demo",
    "sample",
    "placeholder",
    "generated",
)


@dataclass(frozen=True)
class FinancialSanityCheck:
    status: str
    label: str
    detail: str
    path: str = ""
    metric: float | int | str | None = None
    limit: float | int | str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinancialSanityReport:
    checks: tuple[FinancialSanityCheck, ...]
    source: str = ""

    @property
    def allowed(self) -> bool:
        return all(check.status != "block" for check in self.checks)

    @property
    def block_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "block")

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed"] = self.allowed
        payload["block_count"] = self.block_count
        payload["warning_count"] = self.warning_count
        return payload


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    path: str = "",
    metric: float | int | str | None = None,
    limit: float | int | str | None = None,
) -> FinancialSanityCheck:
    return FinancialSanityCheck(
        status, label, detail, path=path, metric=metric, limit=limit
    )


def _finite(value: object) -> float | None:
    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _primitive_metric(
    value: object,
    *,
    default: str = "missing",
) -> float | int | str | None:
    if value is None:
        return default
    if isinstance(value, (float, int, str)):
        return value
    return f"unsupported:{type(value).__name__}"


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _binomial_upper_tail(trials: int, successes: int, p: float = 0.5) -> float:
    n = max(0, int(trials))
    k = max(0, min(n, int(successes)))
    probability = max(0.0, min(1.0, float(p)))
    if n <= 0:
        return 1.0
    total = 0.0
    for hits in range(k, n + 1):
        total += (
            math.comb(n, hits)
            * (probability**hits)
            * ((1.0 - probability) ** (n - hits))
        )
    return max(0.0, min(1.0, total))


def _ai_uplift_period_evidence_checks(
    statistical: Mapping[str, object],
    evidence_binding: object,
    *,
    prefix: str,
    min_bootstrap_samples: int,
    min_bootstrap_confidence: float,
    min_bootstrap_lower: float,
    min_evaluation_span_ms: int,
) -> list[FinancialSanityCheck]:
    checks = _ai_uplift_period_contract_checks(
        statistical,
        evidence_binding,
        prefix=prefix,
        min_evaluation_span_ms=min_evaluation_span_ms,
    )
    checks.extend(
        _ai_uplift_bootstrap_checks(
            statistical,
            prefix=prefix,
            min_bootstrap_samples=min_bootstrap_samples,
            min_bootstrap_confidence=min_bootstrap_confidence,
            min_bootstrap_lower=min_bootstrap_lower,
        )
    )
    return checks


def _ai_uplift_period_contract_checks(
    statistical: Mapping[str, object],
    evidence_binding: object,
    *,
    prefix: str,
    min_evaluation_span_ms: int,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    base_path = f"{prefix}.ai_uplift.statistical_evidence"
    if statistical.get("evidence_unit") != "matched_fixed_period_return_delta":
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift does not use fixed-period matched returns",
                path=f"{base_path}.evidence_unit",
                metric=_primitive_metric(statistical.get("evidence_unit")),
                limit="matched_fixed_period_return_delta",
            )
        )
    paired_sha256 = statistical.get("paired_samples_sha256")
    if not _is_sha256(paired_sha256):
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has no paired-period table fingerprint",
                path=f"{base_path}.paired_samples_sha256",
                metric=_primitive_metric(paired_sha256),
                limit="64 lowercase hex characters",
            )
        )
    if isinstance(evidence_binding, Mapping) and paired_sha256 != evidence_binding.get(
        "paired_samples_sha256"
    ):
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "paired-period table hash disagrees with evidence binding",
                path=f"{base_path}.paired_samples_sha256",
                metric=_primitive_metric(paired_sha256),
                limit=_primitive_metric(evidence_binding.get("paired_samples_sha256")),
            )
        )
    sample_count = _finite(statistical.get("sample_count"))
    duration_ms = _finite(statistical.get("period_duration_ms"))
    first_ms = _finite(statistical.get("first_period_start_ms"))
    last_ms = _finite(statistical.get("last_period_end_ms"))
    scope = str(statistical.get("scope") or "")
    period_contract_valid = (
        bool(scope)
        and sample_count is not None
        and sample_count > 0.0
        and float(sample_count).is_integer()
        and duration_ms is not None
        and duration_ms > 0.0
        and float(duration_ms).is_integer()
        and first_ms is not None
        and last_ms is not None
        and last_ms > first_ms
        and abs((last_ms - first_ms) - sample_count * duration_ms) <= 0.5
        and last_ms - first_ms >= min_evaluation_span_ms
    )
    if not period_contract_valid:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has inconsistent fixed-period coverage",
                path=f"{base_path}.period_duration_ms",
                metric=duration_ms if duration_ms is not None else "missing",
                limit="contiguous sample_count * positive fixed duration",
            )
        )
    return checks


def _ai_uplift_bootstrap_checks(
    statistical: Mapping[str, object],
    *,
    prefix: str,
    min_bootstrap_samples: int,
    min_bootstrap_confidence: float,
    min_bootstrap_lower: float,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    base_path = f"{prefix}.ai_uplift.statistical_evidence"
    bootstrap_samples = _finite(statistical.get("block_bootstrap_samples"))
    bootstrap_confidence = _finite(statistical.get("block_bootstrap_confidence"))
    bootstrap_lower = _finite(statistical.get("mean_delta_ci_lower"))
    bootstrap_upper = _finite(statistical.get("mean_delta_ci_upper"))
    bootstrap_positive = _finite(statistical.get("positive_mean_probability"))
    if (
        bootstrap_samples is None
        or not float(bootstrap_samples).is_integer()
        or bootstrap_samples < min_bootstrap_samples
    ):
        checks.append(
            _check(
                "block",
                "AI uplift block bootstrap",
                "accepted AI uplift has too few block-bootstrap resamples",
                path=f"{base_path}.block_bootstrap_samples",
                metric=bootstrap_samples
                if bootstrap_samples is not None
                else "missing",
                limit=f">={min_bootstrap_samples}",
            )
        )
    if (
        bootstrap_confidence is None
        or bootstrap_confidence < min_bootstrap_confidence
        or bootstrap_confidence >= 1.0
    ):
        checks.append(
            _check(
                "block",
                "AI uplift block bootstrap",
                "accepted AI uplift confidence level is too weak",
                path=f"{base_path}.block_bootstrap_confidence",
                metric=bootstrap_confidence
                if bootstrap_confidence is not None
                else "missing",
                limit=f"[{min_bootstrap_confidence:g},1)",
            )
        )
    if (
        bootstrap_lower is None
        or bootstrap_lower <= min_bootstrap_lower
        or bootstrap_upper is None
        or bootstrap_upper < bootstrap_lower
    ):
        checks.append(
            _check(
                "block",
                "AI uplift block bootstrap",
                "accepted AI uplift confidence interval does not prove positive mean uplift",
                path=f"{base_path}.mean_delta_ci_lower",
                metric=bootstrap_lower if bootstrap_lower is not None else "missing",
                limit=f">{min_bootstrap_lower:g}",
            )
        )
    if (
        bootstrap_positive is None
        or bootstrap_positive < min_bootstrap_confidence
        or bootstrap_positive > 1.0
    ):
        checks.append(
            _check(
                "block",
                "AI uplift block bootstrap",
                "accepted AI uplift has weak positive-mean probability",
                path=f"{base_path}.positive_mean_probability",
                metric=bootstrap_positive
                if bootstrap_positive is not None
                else "missing",
                limit=f">={min_bootstrap_confidence:g}",
            )
        )
    return checks


def _finite_sequence(
    values: Sequence[object], *, path: str, label: str
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for index, value in enumerate(values):
        parsed = _finite(value)
        if parsed is None:
            checks.append(
                _check(
                    "block",
                    label,
                    "non-finite numeric parameter",
                    path=f"{path}[{index}]",
                )
            )
        elif abs(parsed) > 1e9:
            checks.append(
                _check(
                    "block",
                    label,
                    "implausibly large numeric parameter",
                    path=f"{path}[{index}]",
                    metric=parsed,
                    limit="abs<=1e9",
                )
            )
        elif abs(parsed) > 1e6:
            checks.append(
                _check(
                    "warn",
                    label,
                    "large numeric parameter requires review",
                    path=f"{path}[{index}]",
                    metric=parsed,
                    limit="abs<=1e6 preferred",
                )
            )
    return checks


def _range_check(
    value: object,
    *,
    path: str,
    label: str,
    low: float,
    high: float,
    hard_low: float | None = None,
    hard_high: float | None = None,
) -> FinancialSanityCheck:
    parsed = _finite(value)
    if parsed is None:
        return _check(
            "block",
            label,
            "missing or non-finite value",
            path=path,
            metric="missing",
            limit=f"{low:g}-{high:g}",
        )
    hard_low = low if hard_low is None else hard_low
    hard_high = high if hard_high is None else hard_high
    if parsed < hard_low or parsed > hard_high:
        return _check(
            "block",
            label,
            "outside hard financial bounds",
            path=path,
            metric=parsed,
            limit=f"{hard_low:g}-{hard_high:g}",
        )
    if parsed < low or parsed > high:
        return _check(
            "warn",
            label,
            "outside preferred financial bounds",
            path=path,
            metric=parsed,
            limit=f"{low:g}-{high:g}",
        )
    return _check(
        "ok",
        label,
        "within financial bounds",
        path=path,
        metric=parsed,
        limit=f"{low:g}-{high:g}",
    )


def _has_promotion_evidence(model: TrainedModel) -> bool:
    selection_risk = getattr(model, "selection_risk", None)
    execution_validation = getattr(model, "execution_validation", None)
    return (isinstance(selection_risk, Mapping) and bool(selection_risk)) or (
        isinstance(execution_validation, Mapping) and bool(execution_validation)
    )


@dataclass(frozen=True)
class _ProbabilityCalibrationEvidence:
    sample_size: float | None
    brier_before: float | None
    brier_after: float | None
    ece_before: float | None
    ece_after: float | None
    log_loss_before: float | None
    log_loss_after: float | None

    @classmethod
    def from_model(cls, model: TrainedModel) -> _ProbabilityCalibrationEvidence:
        return cls(
            sample_size=_finite(getattr(model, "probability_calibration_size", 0)),
            brier_before=_finite(getattr(model, "probability_brier_before", None)),
            brier_after=_finite(getattr(model, "probability_brier_after", None)),
            ece_before=_finite(getattr(model, "probability_ece_before", None)),
            ece_after=_finite(getattr(model, "probability_ece_after", None)),
            log_loss_before=_finite(
                getattr(model, "probability_log_loss_before", None)
            ),
            log_loss_after=_finite(getattr(model, "probability_log_loss_after", None)),
        )

    @property
    def any_metric(self) -> bool:
        return any(
            value is not None
            for value in (
                self.brier_before,
                self.brier_after,
                self.ece_before,
                self.ece_after,
                self.log_loss_before,
                self.log_loss_after,
            )
        )


def _probability_sample_checks(
    evidence: _ProbabilityCalibrationEvidence,
    *,
    promoted: bool,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if promoted and (evidence.sample_size is None or evidence.sample_size <= 0):
        checks.append(
            _check(
                "block",
                "probability calibration evidence",
                "promoted model is missing calibration sample evidence",
                path="probability_calibration_size",
                metric=(
                    evidence.sample_size
                    if evidence.sample_size is not None
                    else "missing"
                ),
                limit=">0",
            )
        )
    elif evidence.sample_size is not None and evidence.sample_size > 0:
        checks.append(
            _check(
                "ok",
                "probability calibration evidence",
                f"rows={int(evidence.sample_size)}",
                path="probability_calibration_size",
                metric=int(evidence.sample_size),
                limit=">0",
            )
        )
    return checks


def _probability_range_checks(
    evidence: _ProbabilityCalibrationEvidence,
    *,
    promoted: bool,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if promoted and evidence.brier_after is None:
        checks.append(
            _check(
                "block",
                "probability Brier score",
                "promoted model is missing calibrated Brier score",
                path="probability_brier_after",
                metric="missing",
                limit=f"<={_HARD_PROBABILITY_BRIER_MAX:g}",
            )
        )
    elif evidence.brier_after is not None:
        checks.append(
            _range_check(
                evidence.brier_after,
                path="probability_brier_after",
                label="probability Brier score",
                low=0.0,
                high=_PREFERRED_PROBABILITY_BRIER_MAX,
                hard_low=0.0,
                hard_high=_HARD_PROBABILITY_BRIER_MAX,
            )
        )

    if promoted and evidence.ece_after is None:
        checks.append(
            _check(
                "block",
                "probability calibration error",
                "promoted model is missing expected calibration error",
                path="probability_ece_after",
                metric="missing",
                limit=f"<={_HARD_PROBABILITY_ECE_MAX:g}",
            )
        )
    elif evidence.ece_after is not None:
        checks.append(
            _range_check(
                evidence.ece_after,
                path="probability_ece_after",
                label="probability calibration error",
                low=0.0,
                high=_PREFERRED_PROBABILITY_ECE_MAX,
                hard_low=0.0,
                hard_high=_HARD_PROBABILITY_ECE_MAX,
            )
        )
    return checks


def _probability_deterioration_checks(
    evidence: _ProbabilityCalibrationEvidence,
    *,
    promoted: bool,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if (
        evidence.brier_before is not None
        and evidence.brier_after is not None
        and evidence.brier_after > evidence.brier_before + 1e-9
    ):
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability Brier score",
                "calibration worsened Brier score",
                path="probability_brier_after",
                metric=evidence.brier_after,
                limit=f"<={evidence.brier_before:g}",
            )
        )
    if (
        evidence.log_loss_before is not None
        and evidence.log_loss_after is not None
        and evidence.log_loss_after > evidence.log_loss_before + 1e-9
    ):
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability log loss",
                "calibration worsened log loss",
                path="probability_log_loss_after",
                metric=evidence.log_loss_after,
                limit=f"<={evidence.log_loss_before:g}",
            )
        )
    if (
        evidence.any_metric
        and not promoted
        and evidence.brier_after is None
        and evidence.ece_after is None
    ):
        checks.append(
            _check(
                "warn",
                "probability calibration evidence",
                "partial calibration metrics are present without calibrated Brier or ECE",
                path="probability_brier_after",
            )
        )
    if (
        evidence.ece_before is not None
        and evidence.ece_after is not None
        and evidence.ece_after > evidence.ece_before + 1e-9
    ):
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability calibration error",
                "calibration increased expected calibration error",
                path="probability_ece_after",
                metric=evidence.ece_after,
                limit=f"<={evidence.ece_before:g}",
            )
        )
    return checks


def _probability_calibration_checks(model: TrainedModel) -> list[FinancialSanityCheck]:
    evidence = _ProbabilityCalibrationEvidence.from_model(model)
    promoted = _has_promotion_evidence(model)
    checks = _probability_sample_checks(evidence, promoted=promoted)
    checks.extend(_probability_range_checks(evidence, promoted=promoted))
    checks.extend(_probability_deterioration_checks(evidence, promoted=promoted))
    return checks


def build_model_financial_sanity_report(
    model: TrainedModel, *, source: str = "model"
) -> FinancialSanityReport:
    checks: list[FinancialSanityCheck] = []
    feature_dim = int(getattr(model, "feature_dim", 0) or 0)
    checks.append(
        _check(
            "ok" if feature_dim > 0 else "block",
            "feature dimension",
            f"{feature_dim}",
            path="feature_dim",
            metric=feature_dim,
            limit=">0",
        )
    )
    for attr in ("weights", "feature_means", "feature_stds"):
        values = list(getattr(model, attr, []) or [])
        checks.append(
            _check(
                "ok" if len(values) == feature_dim and feature_dim > 0 else "block",
                attr,
                f"length={len(values)} expected={feature_dim}",
                path=attr,
                metric=len(values),
                limit=feature_dim,
            )
        )
        checks.extend(_finite_sequence(values, path=attr, label=attr))
    checks.extend(
        _finite_sequence([getattr(model, "bias", None)], path="bias", label="bias")
    )
    checks.append(
        _range_check(
            getattr(model, "learning_rate", None),
            path="learning_rate",
            label="learning rate",
            low=1e-6,
            high=0.5,
            hard_low=1e-9,
            hard_high=1.0,
        )
    )
    checks.append(
        _range_check(
            getattr(model, "l2_penalty", None),
            path="l2_penalty",
            label="L2 penalty",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=10.0,
        )
    )
    checks.append(
        _range_check(
            getattr(model, "probability_temperature", None),
            path="probability_temperature",
            label="probability temperature",
            low=0.25,
            high=4.0,
            hard_low=1e-6,
            hard_high=10.0,
        )
    )
    checks.extend(_probability_calibration_checks(model))
    threshold = getattr(model, "decision_threshold", None)
    if threshold is not None:
        checks.append(
            _range_check(
                threshold,
                path="decision_threshold",
                label="decision threshold",
                low=0.50,
                high=0.99,
                hard_low=0.01,
                hard_high=0.99,
            )
        )
    long_threshold = getattr(model, "long_decision_threshold", None)
    if long_threshold is not None:
        checks.append(
            _range_check(
                long_threshold,
                path="long_decision_threshold",
                label="long decision threshold",
                low=0.50,
                high=0.99,
                hard_low=0.01,
                hard_high=0.99,
            )
        )
    short_threshold = getattr(model, "short_decision_threshold", None)
    if short_threshold is not None:
        checks.append(
            _range_check(
                short_threshold,
                path="short_decision_threshold",
                label="short decision threshold",
                low=0.01,
                high=0.50,
                hard_low=0.01,
                hard_high=0.99,
            )
        )
    for attr in ("class_weight_pos", "class_weight_neg"):
        checks.append(
            _range_check(
                getattr(model, attr, None),
                path=attr,
                label=attr.replace("_", " "),
                low=0.01,
                high=25.0,
                hard_low=1e-9,
                hard_high=100.0,
            )
        )
    checks.append(
        _range_check(
            getattr(model, "hybrid_base_weight", 1.0),
            path="hybrid_base_weight",
            label="hybrid base weight",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        )
    )
    for index, expert in enumerate(getattr(model, "hybrid_experts", []) or []):
        checks.append(
            _range_check(
                getattr(expert, "weight", None),
                path=f"hybrid_experts[{index}].weight",
                label="hybrid expert weight",
                low=0.0,
                high=1.0,
                hard_low=0.0,
                hard_high=1.0,
            )
        )
        checks.append(
            _range_check(
                getattr(expert, "k", 1),
                path=f"hybrid_experts[{index}].k",
                label="hybrid neighbor count",
                low=1.0,
                high=501.0,
                hard_low=1.0,
                hard_high=5001.0,
            )
        )
    execution = getattr(model, "execution_validation", None)
    if isinstance(execution, Mapping) and execution:
        coverage = execution.get("data_coverage")
        if isinstance(coverage, Mapping) and coverage.get("integrity_status") == "fail":
            checks.append(
                _check(
                    "block",
                    "data coverage",
                    "execution validation contains failed coverage",
                    path="execution_validation.data_coverage",
                )
            )
    return FinancialSanityReport(tuple(checks), source=source)


def _accepted_outcomes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        return []
    return [
        item
        for item in outcomes
        if isinstance(item, Mapping) and item.get("accepted") is True
    ]


def _symbol_sequence(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    symbols: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        symbol = item.strip().upper()
        if not symbol:
            return None
        symbols.append(symbol)
    return symbols


def _duplicate_symbols(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for symbol in symbols:
        if symbol in seen and symbol not in duplicates:
            duplicates.append(symbol)
        seen.add(symbol)
    return duplicates


def _required_metric_checks(
    payload: object,
    *,
    keys: Sequence[str],
    path: str,
    label: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if not isinstance(payload, Mapping):
        return [
            _check(
                "block",
                label,
                "missing accepted metric group",
                path=path,
                metric="missing",
                limit="mapping",
            )
        ]
    for key in keys:
        parsed = _finite(payload.get(key))
        if parsed is None:
            checks.append(
                _check(
                    "block",
                    label,
                    "missing or non-finite accepted metric",
                    path=f"{path}.{key}",
                    metric="missing",
                    limit="finite",
                )
            )
    return checks


def _selection_risk_report_for_objective(
    raw: Mapping[str, Any],
    objective: str,
) -> Mapping[str, Any] | None:
    candidate = raw.get(objective)
    if isinstance(candidate, Mapping):
        return candidate
    if len(raw) == 1:
        only_value = next(iter(raw.values()))
        if isinstance(only_value, Mapping):
            return only_value
    if "passed" in raw or "deflated_score" in raw:
        return raw
    return None


def _walk_forward_report_for_objective(
    raw: Mapping[str, Any],
    objective: str,
) -> Mapping[str, Any] | None:
    candidate = raw.get(objective)
    if isinstance(candidate, Mapping):
        return candidate
    if len(raw) == 1:
        only_value = next(iter(raw.values()))
        if isinstance(only_value, Mapping):
            return only_value
    if "passed" in raw or "fold_count" in raw:
        return raw
    return None


def _walk_forward_gate_checks(
    outcome: Mapping[str, Any],
    *,
    objectives: Sequence[str],
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    raw = outcome.get("walk_forward_gate")
    if not isinstance(raw, Mapping) or not raw:
        return [
            _check(
                "block",
                "purged walk-forward",
                "accepted outcome is missing purged walk-forward evidence",
                path=f"{prefix}.walk_forward_gate",
                metric="missing",
                limit="passed purged folds",
            )
        ]
    for objective in objectives:
        report = _walk_forward_report_for_objective(raw, str(objective))
        report_path = f"{prefix}.walk_forward_gate.{objective}"
        if not isinstance(report, Mapping):
            checks.append(
                _check(
                    "block",
                    "purged walk-forward",
                    "missing accepted objective purged walk-forward report",
                    path=report_path,
                    metric="missing",
                    limit="passed purged folds",
                )
            )
            continue
        if report.get("passed") is not True:
            checks.append(
                _check(
                    "block",
                    "purged walk-forward",
                    "accepted objective failed purged walk-forward evidence",
                    path=f"{report_path}.passed",
                    metric=report.get("passed"),
                    limit=True,
                )
            )
        reason = report.get("reason")
        if reason not in (None, ""):
            checks.append(
                _check(
                    "block",
                    "purged walk-forward",
                    "accepted purged walk-forward report contains rejection reason",
                    path=f"{report_path}.reason",
                    metric=str(reason),
                    limit="empty",
                )
            )
        fold_count = _finite(report.get("fold_count"))
        accepted_folds = _finite(report.get("accepted_folds"))
        if (
            fold_count is None
            or fold_count <= 0.0
            or not float(fold_count).is_integer()
        ):
            checks.append(
                _check(
                    "block",
                    "purged walk-forward",
                    "accepted purged walk-forward report has no real folds",
                    path=f"{report_path}.fold_count",
                    metric=fold_count if fold_count is not None else "missing",
                    limit="positive integer",
                )
            )
        if (
            accepted_folds is None
            or accepted_folds < 0.0
            or not float(accepted_folds).is_integer()
            or (fold_count is not None and accepted_folds != fold_count)
        ):
            checks.append(
                _check(
                    "block",
                    "purged walk-forward",
                    "accepted purged walk-forward fold count is inconsistent",
                    path=f"{report_path}.accepted_folds",
                    metric=accepted_folds if accepted_folds is not None else "missing",
                    limit="accepted_folds==fold_count",
                )
            )
        worst_score = _finite(report.get("worst_score"))
        checks.append(
            _check(
                "ok" if worst_score is not None and worst_score > 0.0 else "block",
                "purged walk-forward",
                "accepted purged walk-forward worst score",
                path=f"{report_path}.worst_score",
                metric=worst_score if worst_score is not None else "missing",
                limit=">0",
            )
        )
        worst_pnl = _finite(report.get("worst_realized_pnl"))
        checks.append(
            _check(
                "ok" if worst_pnl is not None and worst_pnl > 0.0 else "block",
                "purged walk-forward",
                "accepted purged walk-forward worst realized P&L",
                path=f"{report_path}.worst_realized_pnl",
                metric=worst_pnl if worst_pnl is not None else "missing",
                limit=">0",
            )
        )
        checks.append(
            _range_check(
                report.get("worst_max_drawdown"),
                path=f"{report_path}.worst_max_drawdown",
                label="purged walk-forward",
                low=0.0,
                high=1.0,
                hard_low=0.0,
                hard_high=1.0,
            )
        )
    return checks


def _selection_risk_header_checks(
    report: Mapping[str, Any],
    *,
    report_path: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if report.get("passed") is not True:
        checks.append(
            _check(
                "block",
                "selection risk",
                "accepted objective failed selection-risk evidence",
                path=f"{report_path}.passed",
                metric=report.get("passed"),
                limit=True,
            )
        )
    reasons = report.get("reasons")
    if (
        isinstance(reasons, Sequence)
        and not isinstance(reasons, (str, bytes))
        and reasons
    ):
        checks.append(
            _check(
                "block",
                "selection risk",
                "accepted selection-risk report contains rejection reasons",
                path=f"{report_path}.reasons",
                metric=len(reasons),
                limit=0,
            )
        )
    reason = report.get("reason")
    if reason not in (None, ""):
        checks.append(
            _check(
                "block",
                "selection risk",
                "accepted selection-risk report contains rejection reason",
                path=f"{report_path}.reason",
                metric=str(reason),
                limit="empty",
            )
        )
    deflated_score = _finite(report.get("deflated_score"))
    checks.append(
        _check(
            "ok" if deflated_score is not None and deflated_score > 0.0 else "block",
            "selection risk",
            "accepted selection-risk deflated score",
            path=f"{report_path}.deflated_score",
            metric=deflated_score if deflated_score is not None else "missing",
            limit=">0",
        )
    )
    effective_trials = _finite(report.get("effective_trials"))
    checks.append(
        _check(
            "ok"
            if effective_trials is not None and effective_trials >= 1.0
            else "block",
            "selection risk",
            "accepted selection-risk effective trial count",
            path=f"{report_path}.effective_trials",
            metric=effective_trials if effective_trials is not None else "missing",
            limit=">=1",
        )
    )
    return checks


def _terminal_holdout_check(
    report: Mapping[str, Any],
    *,
    objective: str,
    report_path: str,
) -> FinancialSanityCheck:
    terminal = report.get("terminal_holdout")
    terminal_path = f"{report_path}.terminal_holdout"
    terminal_mapping = terminal if isinstance(terminal, Mapping) else None
    terminal_result = (
        terminal_mapping.get("result") if terminal_mapping is not None else None
    )
    terminal_score = (
        _finite(terminal_mapping.get("score")) if terminal_mapping is not None else None
    )
    terminal_rows = (
        _finite(terminal_mapping.get("rows")) if terminal_mapping is not None else None
    )
    terminal_first = (
        _finite(terminal_mapping.get("start_timestamp"))
        if terminal_mapping is not None
        else None
    )
    terminal_last = (
        _finite(terminal_mapping.get("end_timestamp"))
        if terminal_mapping is not None
        else None
    )
    terminal_pnl = (
        _finite(terminal_result.get("realized_pnl"))
        if isinstance(terminal_result, Mapping)
        else None
    )
    terminal_fingerprint = (
        str(terminal_mapping.get("dataset_fingerprint") or "").lower()
        if terminal_mapping is not None
        else ""
    )
    try:
        result_fingerprint = terminal_result_fingerprint(terminal)
    except (TypeError, ValueError, OverflowError):
        result_fingerprint = ""
    terminal_reservation_passed = _terminal_reservation_passed(
        terminal_mapping,
        dataset_fingerprint=terminal_fingerprint,
        result_fingerprint=result_fingerprint,
        rows=terminal_rows,
        first_timestamp=terminal_first,
        last_timestamp=terminal_last,
        objective=objective,
    )
    terminal_passed = bool(
        terminal_mapping is not None
        and all(
            (
                terminal_mapping.get("schema_version") == "terminal-holdout-v1",
                terminal_mapping.get("passed") is True,
                terminal_mapping.get("reason") in (None, ""),
                _finite(terminal_mapping.get("evaluation_count")) == 1.0,
                terminal_rows is not None and terminal_rows > 0.0,
                terminal_first is not None and terminal_first >= 0.0,
                terminal_last is not None
                and terminal_first is not None
                and terminal_last >= terminal_first,
                terminal_score is not None and terminal_score > 0.0,
                _is_sha256(terminal_fingerprint),
                isinstance(terminal_result, Mapping),
                isinstance(terminal_result, Mapping)
                and terminal_result.get("accepted") is True,
                terminal_pnl is not None and terminal_pnl > 0.0,
                isinstance(terminal_result, Mapping)
                and terminal_result.get("stopped_by_liquidation") is False,
                isinstance(terminal_result, Mapping)
                and _finite(terminal_result.get("liquidation_events")) == 0.0,
                terminal_reservation_passed,
            )
        )
    )
    return _check(
        "ok" if terminal_passed else "block",
        "sealed terminal holdout",
        "accepted selection-risk terminal evidence",
        path=terminal_path,
        metric=terminal_score if terminal_score is not None else "missing",
        limit="single accepted positive fingerprinted ledger-reserved evaluation",
    )


def _terminal_reservation_passed(
    terminal: Mapping[str, Any] | None,
    *,
    dataset_fingerprint: str,
    result_fingerprint: str,
    rows: float | None,
    first_timestamp: float | None,
    last_timestamp: float | None,
    objective: str,
) -> bool:
    if terminal is None or any(
        value is None for value in (rows, first_timestamp, last_timestamp)
    ):
        return False
    checked_rows = cast(float, rows)
    checked_first_timestamp = cast(float, first_timestamp)
    checked_last_timestamp = cast(float, last_timestamp)
    return reservation_evidence_passed(
        terminal.get("reservation"),
        expected_dataset_fingerprint=dataset_fingerprint,
        expected_result_fingerprint=result_fingerprint,
        expected_rows=int(checked_rows),
        expected_first_timestamp=int(checked_first_timestamp),
        expected_last_timestamp=int(checked_last_timestamp),
        expected_objective=objective,
    )


def _selection_overfit_checks(
    report: Mapping[str, Any],
    *,
    report_path: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    overfit = report.get("overfit_diagnostics")
    if not isinstance(overfit, Mapping):
        return [
            _check(
                "block",
                "selection risk",
                "accepted selection-risk report is missing overfit diagnostics",
                path=f"{report_path}.overfit_diagnostics",
                metric="missing",
                limit="passed diagnostics",
            )
        ]
    if overfit.get("passed") is not True:
        checks.append(
            _check(
                "block",
                "selection risk",
                "accepted selection-risk overfit diagnostics failed",
                path=f"{report_path}.overfit_diagnostics.passed",
                metric=overfit.get("passed"),
                limit=True,
            )
        )
    status = str(overfit.get("status") or "")
    if status == "available":
        probability = _finite(overfit.get("probability_backtest_overfit"))
        max_probability = _finite(overfit.get("max_probability_backtest_overfit"))
        if (
            probability is None
            or max_probability is None
            or probability > max_probability
        ):
            checks.append(
                _check(
                    "block",
                    "selection risk",
                    "accepted selection-risk PBO exceeds limit",
                    path=(
                        f"{report_path}.overfit_diagnostics."
                        "probability_backtest_overfit"
                    ),
                    metric=probability if probability is not None else "missing",
                    limit=max_probability if max_probability is not None else "missing",
                )
            )
    elif status != "skipped":
        checks.append(
            _check(
                "block",
                "selection risk",
                "accepted selection-risk overfit diagnostics has unknown status",
                path=f"{report_path}.overfit_diagnostics.status",
                metric=status or "missing",
                limit="available|skipped",
            )
        )
    return checks


def _selection_risk_report_checks(
    report: Mapping[str, Any],
    *,
    objective: str,
    report_path: str,
) -> list[FinancialSanityCheck]:
    checks = _selection_risk_header_checks(report, report_path=report_path)
    checks.append(
        _terminal_holdout_check(
            report,
            objective=objective,
            report_path=report_path,
        )
    )
    checks.extend(_selection_overfit_checks(report, report_path=report_path))
    return checks


def _selection_risk_checks(
    outcome: Mapping[str, Any],
    *,
    objectives: Sequence[str],
    prefix: str,
) -> list[FinancialSanityCheck]:
    raw = outcome.get("selection_risk")
    if not isinstance(raw, Mapping) or not raw:
        return [
            _check(
                "block",
                "selection risk",
                "accepted outcome is missing selection-risk evidence",
                path=f"{prefix}.selection_risk",
                metric="missing",
                limit="passed selection-risk report",
            )
        ]
    checks: list[FinancialSanityCheck] = []
    for objective_value in objectives:
        objective = str(objective_value)
        report = _selection_risk_report_for_objective(raw, objective)
        report_path = f"{prefix}.selection_risk.{objective}"
        if not isinstance(report, Mapping):
            checks.append(
                _check(
                    "block",
                    "selection risk",
                    "missing accepted objective selection-risk report",
                    path=report_path,
                    metric="missing",
                    limit="passed selection-risk report",
                )
            )
            continue
        checks.extend(
            _selection_risk_report_checks(
                report,
                objective=objective,
                report_path=report_path,
            )
        )
    return checks


def _positive_numeric_check(
    payload: Mapping[str, Any],
    *,
    key: str,
    path: str,
    label: str,
) -> FinancialSanityCheck:
    parsed = _finite(payload.get(key))
    return _check(
        "ok" if parsed is not None and parsed > 0.0 else "block",
        label,
        f"{key}={parsed}",
        path=f"{path}.{key}",
        metric=parsed if parsed is not None else "missing",
        limit=">0",
    )


def _nonnegative_numeric_check(
    payload: Mapping[str, Any],
    *,
    key: str,
    path: str,
    label: str,
) -> FinancialSanityCheck:
    parsed = _finite(payload.get(key))
    return _check(
        "ok" if parsed is not None and parsed >= 0.0 else "block",
        label,
        f"{key}={parsed}",
        path=f"{path}.{key}",
        metric=parsed if parsed is not None else "missing",
        limit=">=0",
    )


def _stress_validation_checks(
    payload: Mapping[str, Any], *, path: str
) -> list[FinancialSanityCheck]:
    checks = [
        _positive_numeric_check(
            payload, key="scenario_count", path=path, label="stress validation"
        ),
        _nonnegative_numeric_check(
            payload, key="worst_realized_pnl", path=path, label="stress validation"
        ),
        _range_check(
            payload.get("worst_max_drawdown"),
            path=f"{path}.worst_max_drawdown",
            label="worst_max_drawdown",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        ),
    ]
    accepted_objectives = _finite(payload.get("accepted_objectives"))
    objective_count = _finite(payload.get("objective_count"))
    if accepted_objectives is not None:
        checks.append(
            _check(
                "ok" if accepted_objectives > 0.0 else "block",
                "stress validation",
                f"accepted_objectives={accepted_objectives}",
                path=f"{path}.accepted_objectives",
                metric=accepted_objectives,
                limit=">0",
            )
        )
    if accepted_objectives is not None and objective_count is not None:
        checks.append(
            _check(
                "ok" if 0.0 <= accepted_objectives <= objective_count else "block",
                "stress validation",
                "accepted objectives within objective count",
                path=f"{path}.accepted_objectives",
                metric=accepted_objectives,
                limit=f"0-{objective_count:g}",
            )
        )
    return checks


def _robustness_validation_checks(
    payload: Mapping[str, Any], *, path: str
) -> list[FinancialSanityCheck]:
    checks = [
        _positive_numeric_check(
            payload, key="window_count", path=path, label="temporal robustness"
        ),
        _positive_numeric_check(
            payload, key="accepted_windows", path=path, label="temporal robustness"
        ),
        _range_check(
            payload.get("accepted_window_rate"),
            path=f"{path}.accepted_window_rate",
            label="accepted_window_rate",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        ),
        _range_check(
            payload.get("worst_max_drawdown"),
            path=f"{path}.worst_max_drawdown",
            label="worst_max_drawdown",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        ),
        _range_check(
            payload.get("worst_sign_test_p_value"),
            path=f"{path}.worst_sign_test_p_value",
            label="worst_sign_test_p_value",
            low=0.0,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        ),
    ]
    if payload.get("statistical_edge_accepted") is not True:
        checks.append(
            _check(
                "block",
                "temporal robustness",
                "accepted outcome lacks accepted statistical-edge evidence",
                path=f"{path}.statistical_edge_accepted",
                metric=payload.get("statistical_edge_accepted"),
                limit=True,
            )
        )
    checks.append(
        _nonnegative_numeric_check(
            payload, key="worst_realized_pnl", path=path, label="temporal robustness"
        )
    )
    bootstrap_lower = _finite(payload.get("worst_bootstrap_lower_mean_return"))
    checks.append(
        _check(
            "ok" if bootstrap_lower is not None else "block",
            "temporal robustness",
            f"worst_bootstrap_lower_mean_return={bootstrap_lower}",
            path=f"{path}.worst_bootstrap_lower_mean_return",
            metric=bootstrap_lower if bootstrap_lower is not None else "missing",
            limit="finite",
        )
    )
    window_count = _finite(payload.get("window_count"))
    accepted_windows = _finite(payload.get("accepted_windows"))
    if window_count is not None and accepted_windows is not None:
        checks.append(
            _check(
                "ok" if 0.0 <= accepted_windows <= window_count else "block",
                "temporal robustness",
                "accepted windows within window count",
                path=f"{path}.accepted_windows",
                metric=accepted_windows,
                limit=f"0-{window_count:g}",
            )
        )
    return checks


def _iter_market_edge_reports(
    payload: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    reports: list[tuple[str, Mapping[str, Any]]] = []
    direct = payload.get("market_edge")
    if isinstance(direct, Mapping):
        reports.append(("market_edge", direct))
    objectives = payload.get("objectives")
    if not isinstance(objectives, list):
        return reports
    for objective_index, objective in enumerate(objectives):
        if not isinstance(objective, Mapping):
            continue
        objective_edge = objective.get("market_edge")
        if isinstance(objective_edge, Mapping):
            reports.append(
                (f"objectives[{objective_index}].market_edge", objective_edge)
            )
        for collection_name in ("results", "windows"):
            collection = objective.get(collection_name)
            if not isinstance(collection, list):
                continue
            for item_index, item in enumerate(collection):
                if not isinstance(item, Mapping):
                    continue
                result = item.get("result")
                if not isinstance(result, Mapping):
                    continue
                edge = result.get("market_edge")
                if isinstance(edge, Mapping):
                    reports.append(
                        (
                            f"objectives[{objective_index}].{collection_name}[{item_index}].result.market_edge",
                            edge,
                        )
                    )
    return reports


def _market_edge_checks(
    payload: Mapping[str, Any], *, path: str
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    reports = _iter_market_edge_reports(payload)
    summary_accepted = payload.get("market_edge_accepted")
    if summary_accepted is False:
        checks.append(
            _check(
                "block",
                "market edge",
                "summary reports failed market-edge evidence",
                path=f"{path}.market_edge_accepted",
            )
        )
    if (
        payload.get("accepted") is True
        and isinstance(payload.get("objectives"), list)
        and not reports
    ):
        checks.append(
            _check(
                "block",
                "market edge",
                "accepted validation report is missing market-edge evidence",
                path=path,
            )
        )
    for relative_path, report in reports:
        accepted = report.get("accepted")
        net_edge_pct = _finite(report.get("net_edge_pct"))
        min_edge_pct = _finite(report.get("min_net_edge_pct"))
        reason = str(report.get("reason") or "accepted")[:240]
        full_path = f"{path}.{relative_path}"
        checks.append(
            _check(
                "ok" if accepted is True else "block",
                "market edge",
                reason,
                path=full_path,
                metric=net_edge_pct if net_edge_pct is not None else "missing",
                limit=f">={min_edge_pct:g}"
                if min_edge_pct is not None
                else "positive audited edge",
            )
        )
        if net_edge_pct is None:
            checks.append(
                _check(
                    "block",
                    "market edge pct",
                    "missing or non-finite net edge",
                    path=f"{full_path}.net_edge_pct",
                )
            )
        liquidation_events = _finite(report.get("liquidation_events"))
        if (
            accepted is True
            and liquidation_events is not None
            and liquidation_events > 0
        ):
            checks.append(
                _check(
                    "block",
                    "liquidation evidence",
                    "accepted market-edge report contains liquidation events",
                    path=f"{full_path}.liquidation_events",
                    metric=liquidation_events,
                    limit=0,
                )
            )
        min_downside_ratio = _finite(report.get("min_downside_return_risk_ratio"))
        downside_ratio = _finite(report.get("downside_return_risk_ratio"))
        if accepted is True and min_downside_ratio is not None:
            if downside_ratio is None or downside_ratio < min_downside_ratio:
                checks.append(
                    _check(
                        "block",
                        "market edge downside risk",
                        "accepted market-edge report fails downside return/risk evidence",
                        path=f"{full_path}.downside_return_risk_ratio",
                        metric=downside_ratio
                        if downside_ratio is not None
                        else "missing",
                        limit=f">={min_downside_ratio:g}",
                    )
                )
    return checks


def _truth_basis_values(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _positive_count_check(
    value: object, *, path: str, label: str
) -> FinancialSanityCheck:
    parsed = _finite(value)
    metric: float | int | str
    if parsed is None:
        metric = "missing"
    elif parsed.is_integer():
        metric = int(parsed)
    else:
        metric = parsed
    return _check(
        "ok" if parsed is not None and parsed > 0 else "block",
        label,
        f"{label}={parsed}",
        path=path,
        metric=metric,
        limit=">0",
    )


def _data_coverage_checks(coverage: object, *, path: str) -> list[FinancialSanityCheck]:
    if not isinstance(coverage, Mapping):
        return [
            _check(
                "block",
                "data coverage",
                "accepted outcome is missing data-coverage evidence",
                path=path,
                metric="missing",
                limit="complete data_coverage object",
            )
        ]

    checks: list[FinancialSanityCheck] = []
    integrity_status = str(coverage.get("integrity_status") or "").strip().lower()
    if integrity_status == "fail":
        checks.append(
            _check("block", "data coverage", "coverage integrity failed", path=path)
        )
    elif integrity_status in {"ok", "warn"}:
        checks.append(
            _check(
                "ok",
                "data coverage",
                f"integrity_status={integrity_status}",
                path=f"{path}.integrity_status",
            )
        )
    else:
        checks.append(
            _check(
                "block",
                "data coverage",
                "missing or unknown coverage integrity status",
                path=f"{path}.integrity_status",
                metric=integrity_status or "missing",
                limit="ok|warn",
            )
        )

    source_scope = str(coverage.get("source_scope") or "").strip()
    source_scope_lc = source_scope.lower()
    if not source_scope:
        checks.append(
            _check(
                "block",
                "data source",
                "accepted model-lab result is missing source scope evidence",
                path=f"{path}.source_scope",
                metric="missing",
                limit="Binance market-data source scope",
            )
        )
    elif (
        any(token in source_scope_lc for token in _BLOCKED_DATA_SOURCE_TOKENS)
        or "binance" not in source_scope_lc
    ):
        checks.append(
            _check(
                "block",
                "data source",
                "accepted model-lab result must name a real Binance market-data source scope",
                path=f"{path}.source_scope",
                metric=source_scope,
                limit="source scope containing binance and no synthetic/fake/mock markers",
            )
        )
    else:
        checks.append(
            _check(
                "ok",
                "data source",
                f"source_scope={source_scope}",
                path=f"{path}.source_scope",
            )
        )

    truth_basis = _truth_basis_values(coverage.get("truth_basis"))
    missing_basis = [
        item for item in _REQUIRED_DATA_COVERAGE_TRUTH_BASIS if item not in truth_basis
    ]
    if missing_basis:
        checks.append(
            _check(
                "block",
                "data truth basis",
                "accepted outcome is missing required truth-basis evidence",
                path=f"{path}.truth_basis",
                metric=",".join(missing_basis),
                limit=",".join(_REQUIRED_DATA_COVERAGE_TRUTH_BASIS),
            )
        )
    else:
        checks.append(
            _check(
                "ok",
                "data truth basis",
                "required truth basis present",
                path=f"{path}.truth_basis",
            )
        )

    checks.append(
        _positive_count_check(
            coverage.get("candles_used"),
            path=f"{path}.candles_used",
            label="coverage candles",
        )
    )
    checks.append(
        _positive_count_check(
            coverage.get("rows_used"), path=f"{path}.rows_used", label="coverage rows"
        )
    )
    checks.append(
        _range_check(
            coverage.get("coverage_ratio"),
            path=f"{path}.coverage_ratio",
            label="coverage ratio",
            low=0.995,
            high=1.0,
            hard_low=0.0,
            hard_high=1.0,
        )
    )
    gap_count = _finite(coverage.get("gap_count"))
    checks.append(
        _check(
            "ok" if gap_count == 0 else "block",
            "coverage gaps",
            f"gap_count={gap_count}",
            path=f"{path}.gap_count",
            metric=gap_count if gap_count is not None else "missing",
            limit=0,
        )
    )
    return checks


def _portfolio_presence_checks(
    portfolio: object,
    accepted_outcomes: Sequence[Mapping[str, Any]],
) -> list[FinancialSanityCheck]:
    if not accepted_outcomes:
        return []
    if not isinstance(portfolio, Mapping):
        return [
            _check(
                "block",
                "portfolio risk",
                "accepted outcomes require a portfolio-risk report",
                path="portfolio_risk",
                metric="missing",
                limit="accepted portfolio-risk report",
            )
        ]
    if portfolio.get("accepted") is not True:
        return [
            _check(
                "block",
                "portfolio risk",
                "accepted outcomes require accepted portfolio-risk evidence",
                path="portfolio_risk.accepted",
                metric=_primitive_metric(portfolio.get("accepted")),
                limit=True,
            )
        ]
    return []


def _top_level_symbol_checks(
    symbols: Sequence[str] | None,
    accepted_outcomes: Sequence[Mapping[str, Any]],
) -> list[FinancialSanityCheck]:
    if not symbols:
        return []
    checks: list[FinancialSanityCheck] = []
    duplicates = _duplicate_symbols(symbols)
    if duplicates:
        checks.append(
            _check(
                "block",
                "accepted symbols",
                "top-level accepted symbols contain duplicates",
                path="accepted_symbols",
                metric=",".join(duplicates),
                limit="unique symbols",
            )
        )
    if not accepted_outcomes:
        checks.append(
            _check(
                "block",
                "accepted symbols",
                "top-level accepted symbols have no accepted outcome records",
                path="accepted_symbols",
                metric=",".join(symbols),
                limit="matching accepted outcomes",
            )
        )
    return checks


def _accepted_portfolio_symbol_checks(
    portfolio: Mapping[str, Any],
    accepted_outcomes: Sequence[Mapping[str, Any]],
    top_level_symbols: Sequence[str] | None,
) -> tuple[list[FinancialSanityCheck], list[str]]:
    checks: list[FinancialSanityCheck] = []
    portfolio_symbols = _symbol_sequence(portfolio.get("accepted_symbols"))
    outcome_symbols = [
        str(outcome.get("symbol")).strip().upper()
        for outcome in accepted_outcomes
        if isinstance(outcome.get("symbol"), str) and str(outcome.get("symbol")).strip()
    ]
    if not accepted_outcomes:
        checks.append(
            _check(
                "block",
                "accepted outcomes",
                "accepted portfolio has no accepted outcome records",
                path="outcomes",
                metric=0,
                limit=">=1 accepted outcome",
            )
        )
    elif len(outcome_symbols) != len(accepted_outcomes):
        checks.append(
            _check(
                "block",
                "accepted outcomes",
                "accepted outcome is missing symbol evidence",
                path="outcomes",
                metric=len(outcome_symbols),
                limit=len(accepted_outcomes),
            )
        )
    if not portfolio_symbols:
        checks.append(
            _check(
                "block",
                "portfolio symbols",
                "accepted portfolio is missing accepted symbol evidence",
                path="portfolio_risk.accepted_symbols",
                metric="missing",
                limit="non-empty symbol list",
            )
        )
        portfolio_symbols = []
    else:
        duplicates = _duplicate_symbols(portfolio_symbols)
        if duplicates:
            checks.append(
                _check(
                    "block",
                    "portfolio symbols",
                    "accepted portfolio contains duplicate symbols",
                    path="portfolio_risk.accepted_symbols",
                    metric=",".join(duplicates),
                    limit="unique symbols",
                )
            )
    normalized_top_level = list(top_level_symbols or [])
    if not normalized_top_level:
        checks.append(
            _check(
                "block",
                "accepted symbols",
                "accepted report is missing top-level accepted symbol evidence",
                path="accepted_symbols",
                metric="missing",
                limit="non-empty symbol list",
            )
        )
    symbol_sets = (
        (
            portfolio_symbols,
            normalized_top_level,
            "portfolio symbols",
            "portfolio symbols differ from top-level accepted symbols",
            "portfolio_risk.accepted_symbols",
        ),
        (
            portfolio_symbols,
            outcome_symbols,
            "portfolio symbols",
            "portfolio symbols differ from accepted outcome symbols",
            "portfolio_risk.accepted_symbols",
        ),
        (
            normalized_top_level,
            outcome_symbols,
            "accepted symbols",
            "top-level accepted symbols differ from accepted outcome symbols",
            "accepted_symbols",
        ),
    )
    for actual, expected, label, detail, path in symbol_sets:
        if actual and expected and set(actual) != set(expected):
            checks.append(
                _check(
                    "block",
                    label,
                    detail,
                    path=path,
                    metric=",".join(actual),
                    limit=",".join(expected),
                )
            )
    return checks, portfolio_symbols or normalized_top_level or outcome_symbols


def _accepted_portfolio_metric_checks(
    portfolio: Mapping[str, Any],
    accepted_symbols: Sequence[str],
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    accepted_symbol_limit = float(len(accepted_symbols)) if accepted_symbols else 100.0
    for key in (
        "effective_symbol_count",
        "correlation_adjusted_effective_symbol_count",
    ):
        checks.append(
            _range_check(
                portfolio.get(key),
                path=f"portfolio_risk.{key}",
                label=key,
                low=1.0,
                high=max(1.0, accepted_symbol_limit),
                hard_low=1.0,
                hard_high=max(1.0, accepted_symbol_limit),
            )
        )
    for key in (
        "portfolio_cvar_95",
        "portfolio_max_drawdown",
        "deployed_weight",
        "max_pairwise_correlation",
        "max_cluster_weight",
    ):
        checks.append(
            _range_check(
                portfolio.get(key),
                path=f"portfolio_risk.{key}",
                label=key,
                low=0.0,
                high=1.0,
                hard_low=-1.0 if key == "max_pairwise_correlation" else 0.0,
                hard_high=1.0,
            )
        )
    return checks


def _model_lab_portfolio_checks(
    payload: Mapping[str, Any],
    accepted_outcomes: Sequence[Mapping[str, Any]],
) -> list[FinancialSanityCheck]:
    portfolio = payload.get("portfolio_risk")
    top_level_symbols = _symbol_sequence(payload.get("accepted_symbols"))
    checks = _portfolio_presence_checks(portfolio, accepted_outcomes)
    checks.extend(_top_level_symbol_checks(top_level_symbols, accepted_outcomes))
    if isinstance(portfolio, Mapping) and portfolio.get("accepted") is True:
        symbol_checks, accepted_symbols = _accepted_portfolio_symbol_checks(
            portfolio,
            accepted_outcomes,
            top_level_symbols,
        )
        checks.extend(symbol_checks)
        checks.extend(_accepted_portfolio_metric_checks(portfolio, accepted_symbols))
    return checks


def _model_lab_objective_checks(
    outcome: Mapping[str, Any],
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    rows = _finite(outcome.get("rows"))
    checks.append(
        _check(
            "ok" if rows is not None and rows > 0 else "block",
            "accepted rows",
            f"rows={rows}",
            path=f"{prefix}.rows",
            metric=rows if rows is not None else "missing",
            limit=">0",
        )
    )
    scores = outcome.get("objective_scores")
    accepted_objectives: list[str] = []
    if not isinstance(scores, Mapping) or not scores:
        checks.append(
            _check(
                "block",
                "objective scores",
                "missing accepted objective scores",
                path=f"{prefix}.objective_scores",
            )
        )
    else:
        for objective, value in scores.items():
            accepted_objectives.append(str(objective))
            parsed = _finite(value)
            checks.append(
                _check(
                    "ok" if parsed is not None and parsed > 0.0 else "block",
                    "objective score",
                    f"{objective}={parsed}",
                    path=f"{prefix}.objective_scores.{objective}",
                    metric=parsed if parsed is not None else "missing",
                    limit=">0",
                )
            )
    if accepted_objectives:
        checks.extend(
            _walk_forward_gate_checks(
                outcome,
                objectives=accepted_objectives,
                prefix=prefix,
            )
        )
        checks.extend(
            _selection_risk_checks(
                outcome,
                objectives=accepted_objectives,
                prefix=prefix,
            )
        )
    checks.extend(
        _data_coverage_checks(
            outcome.get("data_coverage"),
            path=f"{prefix}.data_coverage",
        )
    )
    return checks


def _model_lab_validation_checks(
    outcome: Mapping[str, Any],
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for field_name, field_value in (
        ("stress_validation", outcome.get("stress_validation")),
        ("robustness_validation", outcome.get("robustness_validation")),
    ):
        if not isinstance(field_value, Mapping):
            checks.append(
                _check(
                    "block",
                    field_name,
                    "missing accepted validation report",
                    path=f"{prefix}.{field_name}",
                )
            )
            continue
        if field_value.get("accepted") is not True:
            checks.append(
                _check(
                    "block",
                    field_name,
                    "accepted outcome has failed validation",
                    path=f"{prefix}.{field_name}.accepted",
                )
            )
        validation_path = f"{prefix}.{field_name}"
        if field_name == "stress_validation":
            checks.extend(_stress_validation_checks(field_value, path=validation_path))
        if field_name == "robustness_validation":
            checks.extend(
                _robustness_validation_checks(field_value, path=validation_path)
            )
        checks.extend(_market_edge_checks(field_value, path=validation_path))
    return checks


def _policy_min_float(
    policy: Mapping[str, object],
    key: str,
    default: float,
    *,
    clamp_unit: bool = False,
) -> float:
    parsed = _finite(policy.get(key))
    if parsed is None:
        return default
    candidate = max(0.0, min(1.0, parsed)) if clamp_unit else parsed
    return max(default, candidate)


def _policy_max_float(
    policy: Mapping[str, object],
    key: str,
    default: float,
    *,
    clamp_unit: bool = False,
) -> float:
    parsed = _finite(policy.get(key))
    if parsed is None:
        return default
    candidate = max(0.0, min(1.0, parsed)) if clamp_unit else parsed
    return min(default, candidate)


def _policy_min_int(
    policy: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    parsed = _finite(policy.get(key))
    return default if parsed is None else max(default, int(parsed))


@dataclass(frozen=True)
class _AIUpliftThresholds:
    min_parameters_b: float
    min_ai_closed_trades: int
    min_paired_samples: int
    max_sign_test_p: float
    min_positive_delta_rate: float
    min_pnl_delta: float
    min_roi_delta: float
    min_expectancy_delta: float
    max_drawdown_delta: float
    min_mean_sample_delta: float
    min_bootstrap_samples: int
    min_bootstrap_confidence: float
    min_bootstrap_lower: float
    min_evaluation_span_days: int

    @classmethod
    def from_policy(cls, value: object) -> _AIUpliftThresholds:
        policy = value if isinstance(value, Mapping) else {}
        return cls(
            min_parameters_b=_policy_min_float(
                policy,
                "min_model_parameters_b",
                _AI_UPLIFT_DEFAULT_MIN_MODEL_PARAMETERS_B,
            ),
            min_ai_closed_trades=_policy_min_int(
                policy,
                "min_ai_closed_trades",
                _AI_UPLIFT_DEFAULT_MIN_AI_CLOSED_TRADES,
            ),
            min_paired_samples=_policy_min_int(
                policy,
                "min_paired_samples",
                _AI_UPLIFT_DEFAULT_MIN_PAIRED_SAMPLES,
            ),
            max_sign_test_p=_policy_max_float(
                policy,
                "max_sign_test_p_value",
                _AI_UPLIFT_DEFAULT_MAX_SIGN_TEST_P,
                clamp_unit=True,
            ),
            min_positive_delta_rate=_policy_min_float(
                policy,
                "min_positive_delta_rate",
                _AI_UPLIFT_DEFAULT_MIN_POSITIVE_DELTA_RATE,
                clamp_unit=True,
            ),
            min_pnl_delta=_policy_min_float(
                policy,
                "min_pnl_delta",
                _AI_UPLIFT_DEFAULT_MIN_PNL_DELTA,
            ),
            min_roi_delta=_policy_min_float(
                policy,
                "min_roi_delta",
                _AI_UPLIFT_DEFAULT_MIN_ROI_DELTA,
            ),
            min_expectancy_delta=_policy_min_float(
                policy,
                "min_expectancy_delta",
                _AI_UPLIFT_DEFAULT_MIN_EXPECTANCY_DELTA,
            ),
            max_drawdown_delta=_policy_max_float(
                policy,
                "max_drawdown_delta",
                _AI_UPLIFT_DEFAULT_MAX_DRAWDOWN_DELTA,
            ),
            min_mean_sample_delta=_policy_min_float(
                policy,
                "min_mean_sample_delta",
                _AI_UPLIFT_DEFAULT_MIN_MEAN_SAMPLE_DELTA,
            ),
            min_bootstrap_samples=_policy_min_int(
                policy,
                "block_bootstrap_samples",
                _AI_UPLIFT_DEFAULT_BOOTSTRAP_SAMPLES,
            ),
            min_bootstrap_confidence=_policy_min_float(
                policy,
                "block_bootstrap_confidence",
                _AI_UPLIFT_DEFAULT_BOOTSTRAP_CONFIDENCE,
            ),
            min_bootstrap_lower=_policy_min_float(
                policy,
                "min_bootstrap_mean_delta_lower",
                _AI_UPLIFT_DEFAULT_MIN_BOOTSTRAP_LOWER,
            ),
            min_evaluation_span_days=_policy_min_int(
                policy,
                "min_evaluation_span_days",
                _AI_UPLIFT_DEFAULT_MIN_EVALUATION_SPAN_DAYS,
            ),
        )


def _ai_uplift_evidence_binding_checks(
    evidence_binding: object,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if not isinstance(evidence_binding, Mapping):
        checks.append(
            _check(
                "block",
                "AI uplift evidence binding",
                "accepted AI uplift is missing hash-bound evidence",
                path=f"{prefix}.ai_uplift.evidence_binding",
                metric="missing",
                limit="accepted hash binding",
            )
        )
    else:
        if evidence_binding.get("accepted") is not True:
            checks.append(
                _check(
                    "block",
                    "AI uplift evidence binding",
                    "accepted AI uplift has a failed evidence binding",
                    path=f"{prefix}.ai_uplift.evidence_binding.accepted",
                    metric=evidence_binding.get("accepted"),
                    limit=True,
                )
            )
        binding_reasons = evidence_binding.get("reasons")
        if isinstance(binding_reasons, list) and binding_reasons:
            checks.append(
                _check(
                    "block",
                    "AI uplift evidence binding",
                    "accepted AI uplift binding contains rejection reasons",
                    path=f"{prefix}.ai_uplift.evidence_binding.reasons",
                )
            )
        for hash_name in (
            "dataset_fingerprint",
            "baseline_evidence_sha256",
            "ai_evidence_sha256",
            "model_artifact_sha256",
            "paired_samples_sha256",
        ):
            if not _is_sha256(evidence_binding.get(hash_name)):
                checks.append(
                    _check(
                        "block",
                        "AI uplift evidence binding",
                        "accepted AI uplift has an invalid SHA-256 binding",
                        path=f"{prefix}.ai_uplift.evidence_binding.{hash_name}",
                        metric=evidence_binding.get(hash_name, "missing"),
                        limit="64 lowercase hex characters",
                    )
                )
        metric_sources: dict[str, Mapping[str, object]] = {}
        for source_group in (
            "baseline_metric_sources",
            "ai_metric_sources",
        ):
            source_payload = evidence_binding.get(source_group)
            if not isinstance(source_payload, Mapping):
                checks.append(
                    _check(
                        "block",
                        "AI uplift evidence binding",
                        "accepted AI uplift is missing metric-unit provenance",
                        path=(f"{prefix}.ai_uplift.evidence_binding.{source_group}"),
                        metric="missing",
                        limit="supported source key for every metric",
                    )
                )
                continue
            metric_sources[source_group] = source_payload
            for (
                metric_name,
                allowed_sources,
            ) in _AI_UPLIFT_SOURCE_KEYS.items():
                source_name = source_payload.get(metric_name)
                if source_name not in allowed_sources:
                    checks.append(
                        _check(
                            "block",
                            "AI uplift evidence binding",
                            "accepted AI uplift has invalid metric-unit provenance",
                            path=(
                                f"{prefix}.ai_uplift.evidence_binding."
                                f"{source_group}.{metric_name}"
                            ),
                            metric=_primitive_metric(source_name),
                            limit=" or ".join(allowed_sources),
                        )
                    )
        baseline_sources = metric_sources.get("baseline_metric_sources")
        ai_sources = metric_sources.get("ai_metric_sources")
        if baseline_sources is not None and ai_sources is not None:
            for metric_name in _AI_UPLIFT_REQUIRED_METRICS:
                if baseline_sources.get(metric_name) != ai_sources.get(metric_name):
                    checks.append(
                        _check(
                            "block",
                            "AI uplift evidence binding",
                            "baseline and AI metrics use different units",
                            path=(
                                f"{prefix}.ai_uplift.evidence_binding."
                                f"ai_metric_sources.{metric_name}"
                            ),
                            metric=_primitive_metric(ai_sources.get(metric_name)),
                            limit=_primitive_metric(baseline_sources.get(metric_name)),
                        )
                    )

    return checks


@dataclass(frozen=True)
class _PairedOutcomeCounts:
    sample: float | None
    effective: float | None
    positive: float | None
    negative: float | None
    ties: float | None
    sample_is_integer: bool
    effective_is_integer: bool
    positive_is_integer: bool

    @classmethod
    def from_statistical(
        cls, statistical: Mapping[str, object]
    ) -> _PairedOutcomeCounts:
        sample = _finite(statistical.get("sample_count"))
        effective = _finite(statistical.get("effective_sample_count"))
        positive = _finite(statistical.get("positive_delta_count"))
        return cls(
            sample=sample,
            effective=effective,
            positive=positive,
            negative=_finite(statistical.get("negative_delta_count")),
            ties=_finite(statistical.get("tie_count")),
            sample_is_integer=bool(
                sample is not None and sample >= 0.0 and float(sample).is_integer()
            ),
            effective_is_integer=bool(
                effective is not None
                and effective >= 0.0
                and float(effective).is_integer()
            ),
            positive_is_integer=bool(
                positive is not None
                and positive >= 0.0
                and float(positive).is_integer()
            ),
        )

    @property
    def consistent(self) -> bool:
        return all(
            (
                self.sample_is_integer,
                self.effective_is_integer,
                self.positive_is_integer,
                self.negative is not None,
                self.negative is not None and self.negative >= 0.0,
                self.negative is not None and float(self.negative).is_integer(),
                self.ties is not None,
                self.ties is not None and self.ties >= 0.0,
                self.ties is not None and float(self.ties).is_integer(),
                self.positive is not None,
                self.effective is not None,
                self.sample is not None,
                self.positive is not None
                and self.negative is not None
                and self.effective is not None
                and self.positive + self.negative == self.effective,
                self.effective is not None
                and self.ties is not None
                and self.sample is not None
                and self.effective + self.ties == self.sample,
            )
        )


def _ai_uplift_statistical_header_checks(
    statistical: Mapping[str, object],
    evidence_binding: object,
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks = _ai_uplift_period_evidence_checks(
        statistical,
        evidence_binding,
        prefix=prefix,
        min_bootstrap_samples=thresholds.min_bootstrap_samples,
        min_bootstrap_confidence=thresholds.min_bootstrap_confidence,
        min_bootstrap_lower=thresholds.min_bootstrap_lower,
        min_evaluation_span_ms=thresholds.min_evaluation_span_days * _DAY_MS,
    )
    statistical_reasons = statistical.get("reasons")
    if isinstance(statistical_reasons, list) and statistical_reasons:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift statistics contain rejection reasons",
                path=f"{prefix}.ai_uplift.statistical_evidence.reasons",
            )
        )
    if statistical.get("accepted") is not True:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has failed paired-sample evidence",
                path=f"{prefix}.ai_uplift.statistical_evidence.accepted",
                metric=_primitive_metric(statistical.get("accepted")),
                limit=True,
            )
        )
    if statistical.get("paired_sample_length_mismatch") is True:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has unpaired sample lengths",
                path=(
                    f"{prefix}.ai_uplift.statistical_evidence."
                    "paired_sample_length_mismatch"
                ),
                metric=True,
                limit=False,
            )
        )
    return checks


def _ai_uplift_count_checks(
    counts: _PairedOutcomeCounts,
    *,
    prefix: str,
    min_paired_samples: int,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    path = f"{prefix}.ai_uplift.statistical_evidence"
    if counts.sample is None or counts.sample < min_paired_samples:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has too few matched holdout periods",
                path=f"{path}.sample_count",
                metric=counts.sample if counts.sample is not None else "missing",
                limit=f">={min_paired_samples}",
            )
        )
    if counts.sample is not None and not counts.sample_is_integer:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift sample count must be a nonnegative integer",
                path=f"{path}.sample_count",
                metric=counts.sample,
                limit="nonnegative integer",
            )
        )
    if (
        counts.effective is None
        or counts.effective < min_paired_samples
        or not counts.effective_is_integer
        or (
            counts.sample is not None
            and counts.effective is not None
            and counts.effective > counts.sample
        )
    ):
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift has too few non-tied paired outcomes",
                path=f"{path}.effective_sample_count",
                metric=(
                    counts.effective if counts.effective is not None else "missing"
                ),
                limit=f"integer in [{min_paired_samples},sample_count]",
            )
        )
    if (
        counts.positive is None
        or counts.positive < 0.0
        or (
            counts.effective is not None
            and counts.positive is not None
            and counts.positive > counts.effective
        )
        or not counts.positive_is_integer
    ):
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift positive-delta count is inconsistent",
                path=f"{path}.positive_delta_count",
                metric=counts.positive if counts.positive is not None else "missing",
                limit="integer and 0<=positive_delta_count<=effective_sample_count",
            )
        )
    if not counts.consistent:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift paired-outcome counts are inconsistent",
                path=f"{path}.effective_sample_count",
                metric=(
                    counts.effective if counts.effective is not None else "missing"
                ),
                limit="positive+negative=effective and effective+ties=sample_count",
            )
        )
    return checks


def _ai_uplift_rate_checks(
    statistical: Mapping[str, object],
    counts: _PairedOutcomeCounts,
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    path = f"{prefix}.ai_uplift.statistical_evidence"
    positive_rate = _finite(statistical.get("positive_delta_rate"))
    if positive_rate is None or positive_rate < thresholds.min_positive_delta_rate:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift positive-delta rate is too weak",
                path=f"{path}.positive_delta_rate",
                metric=positive_rate if positive_rate is not None else "missing",
                limit=f">={thresholds.min_positive_delta_rate:g}",
            )
        )
    rate_inputs_valid = all(
        (
            counts.effective is not None,
            counts.effective is not None and counts.effective > 0.0,
            counts.positive is not None,
            counts.effective_is_integer,
            counts.positive_is_integer,
            counts.positive is not None
            and counts.effective is not None
            and counts.positive <= counts.effective,
            positive_rate is not None,
        )
    )
    if rate_inputs_valid:
        positive_count = cast(float, counts.positive)
        effective_count = cast(float, counts.effective)
        checked_positive_rate = cast(float, positive_rate)
        expected_rate = positive_count / effective_count
        if abs(checked_positive_rate - expected_rate) > 1e-6:
            checks.append(
                _check(
                    "block",
                    "AI uplift statistical evidence",
                    "accepted AI uplift positive-delta rate does not match counts",
                    path=f"{path}.positive_delta_rate",
                    metric=checked_positive_rate,
                    limit=f"{expected_rate:g}",
                )
            )
    sign_p = _finite(statistical.get("sign_test_p_value"))
    if sign_p is None or sign_p > thresholds.max_sign_test_p:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift sign test is too weak",
                path=f"{path}.sign_test_p_value",
                metric=sign_p if sign_p is not None else "missing",
                limit=f"<={thresholds.max_sign_test_p:g}",
            )
        )
    sign_inputs_valid = all(
        (
            counts.effective is not None,
            counts.positive is not None,
            counts.effective_is_integer,
            counts.positive_is_integer,
            counts.positive is not None
            and counts.effective is not None
            and counts.positive <= counts.effective,
            sign_p is not None,
        )
    )
    if sign_inputs_valid:
        effective_count = cast(float, counts.effective)
        positive_count = cast(float, counts.positive)
        checked_sign_p = cast(float, sign_p)
        expected_sign_p = _binomial_upper_tail(
            int(effective_count),
            int(positive_count),
        )
        if abs(checked_sign_p - expected_sign_p) > 1e-9:
            checks.append(
                _check(
                    "block",
                    "AI uplift statistical evidence",
                    "accepted AI uplift sign-test p-value does not match counts",
                    path=f"{path}.sign_test_p_value",
                    metric=checked_sign_p,
                    limit=f"{expected_sign_p:g}",
                )
            )
    mean_delta = _finite(statistical.get("mean_delta"))
    if mean_delta is None or mean_delta <= thresholds.min_mean_sample_delta:
        checks.append(
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift mean paired delta is too weak",
                path=f"{path}.mean_delta",
                metric=mean_delta if mean_delta is not None else "missing",
                limit=f">{thresholds.min_mean_sample_delta:g}",
            )
        )
    return checks


def _ai_uplift_statistical_checks(
    ai_uplift: Mapping[str, Any],
    evidence_binding: object,
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    statistical = ai_uplift.get("statistical_evidence")
    if not isinstance(statistical, Mapping):
        return [
            _check(
                "block",
                "AI uplift statistical evidence",
                "accepted AI uplift is missing paired holdout statistical evidence",
                path=f"{prefix}.ai_uplift.statistical_evidence",
                metric="missing",
                limit="accepted paired-sample evidence",
            )
        ]
    checks = _ai_uplift_statistical_header_checks(
        statistical,
        evidence_binding,
        thresholds,
        prefix=prefix,
    )
    counts = _PairedOutcomeCounts.from_statistical(statistical)
    checks.extend(
        _ai_uplift_count_checks(
            counts,
            prefix=prefix,
            min_paired_samples=thresholds.min_paired_samples,
        )
    )
    checks.extend(
        _ai_uplift_rate_checks(
            statistical,
            counts,
            thresholds,
            prefix=prefix,
        )
    )
    return checks


def _accepted_ai_uplift_contract_checks(
    ai_uplift: Mapping[str, Any],
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if (
        ai_uplift.get("schema_version") != "ai-uplift-v5"
        or ai_uplift.get("trading_authority") is not False
        or ai_uplift.get("profitability_claim") is not False
    ):
        checks.append(
            _check(
                "block",
                "AI uplift evidence",
                "accepted AI uplift has an unsafe or unsupported authority contract",
                path=f"{prefix}.ai_uplift.schema_version",
                metric=ai_uplift.get("schema_version", "missing"),
                limit="ai-uplift-v5 with no authority or profitability claim",
            )
        )
    model_parameters_b = _finite(ai_uplift.get("model_parameters_b"))
    if model_parameters_b is None or model_parameters_b < thresholds.min_parameters_b:
        checks.append(
            _check(
                "block",
                "AI uplift evidence",
                "accepted AI uplift is missing required model-size evidence",
                path=f"{prefix}.ai_uplift.model_parameters_b",
                metric=(
                    model_parameters_b if model_parameters_b is not None else "missing"
                ),
                limit=f">={thresholds.min_parameters_b:g}",
            )
        )
    evidence_binding = ai_uplift.get("evidence_binding")
    checks.extend(
        _ai_uplift_evidence_binding_checks(
            evidence_binding,
            prefix=prefix,
        )
    )
    for group_name in ("baseline", "ai", "deltas"):
        checks.extend(
            _required_metric_checks(
                ai_uplift.get(group_name),
                keys=_AI_UPLIFT_REQUIRED_METRICS,
                path=f"{prefix}.ai_uplift.{group_name}",
                label="AI uplift evidence",
            )
        )
    checks.extend(
        _ai_uplift_statistical_checks(
            ai_uplift,
            evidence_binding,
            thresholds,
            prefix=prefix,
        )
    )
    return checks


def _ai_uplift_nonfinite_delta_checks(
    deltas: Mapping[str, Any],
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for key, value in deltas.items():
        if _finite(value) is None:
            checks.append(
                _check(
                    "block",
                    "AI uplift delta",
                    "non-finite delta",
                    path=f"{prefix}.ai_uplift.deltas.{key}",
                )
            )
    return checks


def _ai_uplift_reported_delta_checks(
    ai_uplift: Mapping[str, Any],
    deltas: Mapping[str, Any],
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    baseline_metrics = ai_uplift.get("baseline")
    ai_metrics = ai_uplift.get("ai")
    if not isinstance(baseline_metrics, Mapping) or not isinstance(
        ai_metrics,
        Mapping,
    ):
        return checks
    for key in _AI_UPLIFT_REQUIRED_METRICS:
        baseline_value = _finite(baseline_metrics.get(key))
        ai_value = _finite(ai_metrics.get(key))
        reported_delta = _finite(deltas.get(key))
        if (
            baseline_value is not None
            and ai_value is not None
            and reported_delta is not None
            and not math.isclose(
                reported_delta,
                ai_value - baseline_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            checks.append(
                _check(
                    "block",
                    "AI uplift delta",
                    "reported delta does not match AI minus baseline",
                    path=f"{prefix}.ai_uplift.deltas.{key}",
                    metric=reported_delta,
                    limit=ai_value - baseline_value,
                )
            )
    ai_pnl = _finite(ai_metrics.get("realized_pnl"))
    if ai_pnl is not None and ai_pnl <= 0.0:
        checks.append(
            _check(
                "block",
                "AI uplift evidence",
                "accepted AI uplift has nonpositive realized PnL",
                path=f"{prefix}.ai_uplift.ai.realized_pnl",
                metric=ai_pnl,
                limit=">0",
            )
        )
    ai_closed_trades = _finite(ai_metrics.get("closed_trades"))
    if (
        ai_closed_trades is not None
        and ai_closed_trades < thresholds.min_ai_closed_trades
    ):
        checks.append(
            _check(
                "block",
                "AI uplift evidence",
                "accepted AI uplift has too few closed trades",
                path=f"{prefix}.ai_uplift.ai.closed_trades",
                metric=ai_closed_trades,
                limit=f">={thresholds.min_ai_closed_trades}",
            )
        )
    return checks


def _ai_uplift_improvement_checks(
    deltas: Mapping[str, Any],
    thresholds: _AIUpliftThresholds,
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for key, threshold in (
        ("realized_pnl", thresholds.min_pnl_delta),
        ("roi_pct", thresholds.min_roi_delta),
        ("expectancy", thresholds.min_expectancy_delta),
    ):
        parsed = _finite(deltas.get(key))
        if parsed is not None and parsed <= threshold:
            checks.append(
                _check(
                    "block",
                    "AI uplift delta",
                    "accepted AI uplift does not strictly improve the metric",
                    path=f"{prefix}.ai_uplift.deltas.{key}",
                    metric=parsed,
                    limit=f">{threshold:g}",
                )
            )
    drawdown_delta = _finite(deltas.get("max_drawdown"))
    if drawdown_delta is not None and drawdown_delta > thresholds.max_drawdown_delta:
        checks.append(
            _check(
                "block",
                "AI uplift tail risk",
                "accepted AI uplift worsens maximum drawdown",
                path=f"{prefix}.ai_uplift.deltas.max_drawdown",
                metric=drawdown_delta,
                limit=f"<={thresholds.max_drawdown_delta:g}",
            )
        )
    loss_streak_delta = _finite(deltas.get("max_consecutive_losses"))
    if loss_streak_delta is not None and loss_streak_delta > 0.0:
        checks.append(
            _check(
                "block",
                "AI uplift tail risk",
                "accepted AI uplift worsens loss-streak risk",
                path=f"{prefix}.ai_uplift.deltas.max_consecutive_losses",
                metric=loss_streak_delta,
                limit="<=0",
            )
        )
    for key in ("profit_factor", "win_rate", "downside_return_risk_ratio"):
        parsed = _finite(deltas.get(key))
        if parsed is not None and parsed < 0.0:
            checks.append(
                _check(
                    "block",
                    "AI uplift tail risk",
                    "accepted AI uplift degrades risk-adjusted quality",
                    path=f"{prefix}.ai_uplift.deltas.{key}",
                    metric=parsed,
                    limit=">=0",
                )
            )
    return checks


def _ai_uplift_liquidation_checks(
    ai_uplift: Mapping[str, Any],
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    ai_metrics = ai_uplift.get("ai")
    if not isinstance(ai_metrics, Mapping):
        return []
    liquidations = _finite(ai_metrics.get("liquidation_events"))
    if liquidations is None or liquidations <= 0.0:
        return []
    return [
        _check(
            "block",
            "AI uplift liquidation risk",
            "accepted AI uplift contains liquidation events",
            path=f"{prefix}.ai_uplift.ai.liquidation_events",
            metric=liquidations,
            limit=0,
        )
    ]


def _model_lab_ai_uplift_checks(
    ai_uplift: Mapping[str, Any],
    *,
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    accepted = ai_uplift.get("accepted") is True
    reasons = ai_uplift.get("reasons")
    if accepted and isinstance(reasons, list) and reasons:
        checks.append(
            _check(
                "block",
                "AI uplift",
                "accepted AI uplift contains rejection reasons",
                path=f"{prefix}.ai_uplift.reasons",
            )
        )
    thresholds = _AIUpliftThresholds.from_policy(
        ai_uplift.get("policy") if accepted else None
    )
    if accepted:
        checks.extend(
            _accepted_ai_uplift_contract_checks(
                ai_uplift,
                thresholds,
                prefix=prefix,
            )
        )
    deltas = ai_uplift.get("deltas")
    if isinstance(deltas, Mapping):
        checks.extend(_ai_uplift_nonfinite_delta_checks(deltas, prefix=prefix))
        if accepted:
            checks.extend(
                _ai_uplift_reported_delta_checks(
                    ai_uplift,
                    deltas,
                    thresholds,
                    prefix=prefix,
                )
            )
            checks.extend(
                _ai_uplift_improvement_checks(
                    deltas,
                    thresholds,
                    prefix=prefix,
                )
            )
    if accepted:
        checks.extend(_ai_uplift_liquidation_checks(ai_uplift, prefix=prefix))
    return checks


def _model_lab_outcome_checks(
    outcome: Mapping[str, Any],
    *,
    outcome_index: int,
) -> list[FinancialSanityCheck]:
    prefix = f"outcomes[{outcome_index}]"
    checks = _model_lab_objective_checks(outcome, prefix=prefix)
    checks.extend(_model_lab_validation_checks(outcome, prefix=prefix))
    ai_uplift = outcome.get("ai_uplift")
    if isinstance(ai_uplift, Mapping):
        checks.extend(_model_lab_ai_uplift_checks(ai_uplift, prefix=prefix))
    return checks


def build_model_lab_financial_sanity_report(
    payload: Mapping[str, Any],
    *,
    source: str = "model_lab",
) -> FinancialSanityReport:
    accepted_outcomes = _accepted_outcomes(payload)
    checks = _model_lab_portfolio_checks(payload, accepted_outcomes)
    for outcome_index, outcome in enumerate(accepted_outcomes):
        checks.extend(
            _model_lab_outcome_checks(
                outcome,
                outcome_index=outcome_index,
            )
        )
    return FinancialSanityReport(tuple(checks), source=source)


def _numeric_sequence(value: object) -> list[float] | None:
    if not isinstance(value, (tuple, list)):
        return None
    output: list[float] = []
    for item in value:
        parsed = _finite(item)
        if parsed is None:
            return None
        output.append(parsed)
    return output


def _approx_equal(left: float, right: float, *, scale: float = 1.0) -> bool:
    tolerance = max(1e-6, abs(scale) * 1e-9, abs(left) * 1e-9, abs(right) * 1e-9)
    return abs(left - right) <= tolerance


def _sample_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _max_consecutive_losses(values: Sequence[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _expected_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss > 0.0:
        return min(999.0, max(0.0, gross_profit / gross_loss))
    if gross_profit > 0.0:
        return 999.0
    return 0.0


def _integer_count_check(
    checks: list[FinancialSanityCheck],
    value: float | None,
    *,
    path: str,
    label: str,
) -> int | None:
    if value is None:
        checks.append(_check("block", label, "missing or non-finite count", path=path))
        return None
    if value < 0.0:
        checks.append(
            _check(
                "block", label, "negative count", path=path, metric=value, limit=">=0"
            )
        )
        return None
    count = int(value)
    checks.append(
        _check(
            "ok" if abs(value - count) <= 1e-9 else "block",
            label,
            "integer count",
            path=path,
            metric=value,
            limit=count,
        )
    )
    return count if abs(value - count) <= 1e-9 else None


@dataclass(frozen=True)
class _BacktestEvidence:
    starting_cash: float | None
    ending_cash: float | None
    realized_pnl: float | None
    total_fees: float | None
    buy_hold_pnl: float | None
    edge_vs_buy_hold: float | None
    win_rate: float | None
    max_drawdown: float | None
    trades: float | None
    closed_trades: float | None
    gross_exposure: float | None
    max_exposure: float | None
    gross_profit: float | None
    gross_loss: float | None
    profit_factor: float | None
    expectancy: float | None
    average_trade_return: float | None
    trade_return_stdev: float | None
    max_consecutive_losses: float | None
    stopped_by_liquidation: bool
    liquidation_events: float | None
    liquidation_loss: float | None
    equity_curve: object
    trade_log: object
    raw_trade_pnls: object
    raw_trade_returns: object
    trade_pnls: list[float] | None
    trade_returns: list[float] | None

    @classmethod
    def from_result(cls, result: object) -> _BacktestEvidence:
        raw_trade_pnls = getattr(result, "trade_pnls", ())
        raw_trade_returns = getattr(result, "trade_returns", ())
        return cls(
            starting_cash=_finite(getattr(result, "starting_cash", None)),
            ending_cash=_finite(getattr(result, "ending_cash", None)),
            realized_pnl=_finite(getattr(result, "realized_pnl", None)),
            total_fees=_finite(getattr(result, "total_fees", None)),
            buy_hold_pnl=_finite(getattr(result, "buy_hold_pnl", 0.0)),
            edge_vs_buy_hold=_finite(getattr(result, "edge_vs_buy_hold", 0.0)),
            win_rate=_finite(getattr(result, "win_rate", None)),
            max_drawdown=_finite(getattr(result, "max_drawdown", None)),
            trades=_finite(getattr(result, "trades", None)),
            closed_trades=_finite(getattr(result, "closed_trades", None)),
            gross_exposure=_finite(getattr(result, "gross_exposure", None)),
            max_exposure=_finite(getattr(result, "max_exposure", None)),
            gross_profit=_finite(getattr(result, "gross_profit", 0.0)),
            gross_loss=_finite(getattr(result, "gross_loss", 0.0)),
            profit_factor=_finite(getattr(result, "profit_factor", 0.0)),
            expectancy=_finite(getattr(result, "expectancy", 0.0)),
            average_trade_return=_finite(getattr(result, "average_trade_return", 0.0)),
            trade_return_stdev=_finite(getattr(result, "trade_return_stdev", 0.0)),
            max_consecutive_losses=_finite(
                getattr(result, "max_consecutive_losses", 0)
            ),
            stopped_by_liquidation=bool(
                getattr(result, "stopped_by_liquidation", False)
            ),
            liquidation_events=_finite(getattr(result, "liquidation_events", 0)),
            liquidation_loss=_finite(getattr(result, "liquidation_loss", 0.0)),
            equity_curve=getattr(result, "equity_curve", ()),
            trade_log=getattr(result, "trade_log", ()),
            raw_trade_pnls=raw_trade_pnls,
            raw_trade_returns=raw_trade_returns,
            trade_pnls=_numeric_sequence(raw_trade_pnls),
            trade_returns=_numeric_sequence(raw_trade_returns),
        )

    @property
    def cash_scale(self) -> float:
        return max(
            1.0,
            abs(self.starting_cash or 0.0),
            abs(self.ending_cash or 0.0),
        )


def _backtest_scalar_checks(
    evidence: _BacktestEvidence,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for name, value, minimum, maximum in (
        ("starting_cash", evidence.starting_cash, 0.0, None),
        ("ending_cash", evidence.ending_cash, 0.0, None),
        ("realized_pnl", evidence.realized_pnl, None, None),
        ("total_fees", evidence.total_fees, 0.0, None),
        ("buy_hold_pnl", evidence.buy_hold_pnl, None, None),
        ("edge_vs_buy_hold", evidence.edge_vs_buy_hold, None, None),
        ("win_rate", evidence.win_rate, 0.0, 1.0),
        ("max_drawdown", evidence.max_drawdown, 0.0, 1.0),
        ("trades", evidence.trades, 0.0, None),
        ("closed_trades", evidence.closed_trades, 0.0, None),
        ("gross_exposure", evidence.gross_exposure, 0.0, None),
        ("max_exposure", evidence.max_exposure, 0.0, None),
        ("gross_profit", evidence.gross_profit, 0.0, None),
        ("gross_loss", evidence.gross_loss, 0.0, None),
        ("profit_factor", evidence.profit_factor, 0.0, 999.0),
        ("expectancy", evidence.expectancy, None, None),
        ("average_trade_return", evidence.average_trade_return, None, None),
        ("trade_return_stdev", evidence.trade_return_stdev, 0.0, None),
        (
            "max_consecutive_losses",
            evidence.max_consecutive_losses,
            0.0,
            None,
        ),
    ):
        if value is None:
            checks.append(
                _check(
                    "block",
                    "backtest accounting",
                    "missing or non-finite value",
                    path=name,
                )
            )
            continue
        if minimum is not None and value < minimum:
            checks.append(
                _check(
                    "block",
                    "backtest accounting",
                    "value below financial bound",
                    path=name,
                    metric=value,
                    limit=f">={minimum:g}",
                )
            )
        elif maximum is not None and value > maximum:
            checks.append(
                _check(
                    "block",
                    "backtest accounting",
                    "value above financial bound",
                    path=name,
                    metric=value,
                    limit=f"<={maximum:g}",
                )
            )
        else:
            checks.append(
                _check(
                    "ok",
                    "backtest accounting",
                    "finite bounded value",
                    path=name,
                    metric=value,
                )
            )
    return checks


def _backtest_liquidation_checks(
    evidence: _BacktestEvidence,
    *,
    reject_liquidation: bool,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    liquidation_status = "block" if reject_liquidation else "warn"
    if evidence.stopped_by_liquidation:
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "backtest stopped by liquidation",
                path="stopped_by_liquidation",
                metric="true",
                limit="false",
            )
        )
    else:
        checks.append(
            _check(
                "ok",
                "liquidation evidence",
                "not stopped by liquidation",
                path="stopped_by_liquidation",
            )
        )
    if evidence.liquidation_events is None:
        checks.append(
            _check(
                "block",
                "liquidation events",
                "missing or non-finite value",
                path="liquidation_events",
            )
        )
    elif evidence.liquidation_events < 0.0:
        checks.append(
            _check(
                "block",
                "liquidation events",
                "liquidation event count is negative",
                path="liquidation_events",
                metric=evidence.liquidation_events,
                limit=">=0",
            )
        )
    elif evidence.liquidation_events > 0.0:
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "backtest contains liquidation events",
                path="liquidation_events",
                metric=evidence.liquidation_events,
                limit=0,
            )
        )
    else:
        checks.append(
            _check(
                "ok",
                "liquidation evidence",
                "no liquidation events",
                path="liquidation_events",
                metric=0,
            )
        )
    if evidence.liquidation_loss is None:
        checks.append(
            _check(
                "block",
                "liquidation loss",
                "missing or non-finite value",
                path="liquidation_loss",
            )
        )
    elif evidence.liquidation_loss < 0.0:
        checks.append(
            _check(
                "block",
                "liquidation loss",
                "liquidation loss is negative",
                path="liquidation_loss",
                metric=evidence.liquidation_loss,
                limit=">=0",
            )
        )
    elif evidence.liquidation_loss > 0.0:
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "backtest contains liquidation loss",
                path="liquidation_loss",
                metric=evidence.liquidation_loss,
                limit=0,
            )
        )
    else:
        checks.append(
            _check(
                "ok",
                "liquidation evidence",
                "no liquidation loss",
                path="liquidation_loss",
                metric=0.0,
            )
        )
    return checks


def _backtest_cash_identity_checks(
    evidence: _BacktestEvidence,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if all(
        value is not None
        for value in (
            evidence.starting_cash,
            evidence.ending_cash,
            evidence.realized_pnl,
        )
    ):
        starting_cash = cast(float, evidence.starting_cash)
        ending_cash = cast(float, evidence.ending_cash)
        realized_pnl = cast(float, evidence.realized_pnl)
        expected_realized = ending_cash - starting_cash
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        realized_pnl,
                        expected_realized,
                        scale=evidence.cash_scale,
                    )
                    else "block"
                ),
                "backtest cash identity",
                "realized_pnl equals ending_cash - starting_cash",
                path="realized_pnl",
                metric=realized_pnl,
                limit=expected_realized,
            )
        )
    if all(
        value is not None
        for value in (
            evidence.realized_pnl,
            evidence.buy_hold_pnl,
            evidence.edge_vs_buy_hold,
        )
    ):
        realized_pnl = cast(float, evidence.realized_pnl)
        buy_hold_pnl = cast(float, evidence.buy_hold_pnl)
        edge_vs_buy_hold = cast(float, evidence.edge_vs_buy_hold)
        expected_edge = realized_pnl - buy_hold_pnl
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        edge_vs_buy_hold,
                        expected_edge,
                        scale=evidence.cash_scale,
                    )
                    else "block"
                ),
                "backtest edge identity",
                "edge_vs_buy_hold equals realized_pnl - buy_hold_pnl",
                path="edge_vs_buy_hold",
                metric=edge_vs_buy_hold,
                limit=expected_edge,
            )
        )
    return checks


@dataclass(frozen=True)
class _BacktestSequenceState:
    closed_count: int | None
    trade_count: int | None
    loss_streak_count: int | None
    trade_log: Sequence[object]


def _backtest_sequence_checks(
    evidence: _BacktestEvidence,
) -> tuple[list[FinancialSanityCheck], _BacktestSequenceState]:
    checks: list[FinancialSanityCheck] = []
    closed_count = _integer_count_check(
        checks,
        evidence.closed_trades,
        path="closed_trades",
        label="closed trade count",
    )
    trade_count = _integer_count_check(
        checks,
        evidence.trades,
        path="trades",
        label="trade count",
    )
    loss_streak_count = _integer_count_check(
        checks,
        evidence.max_consecutive_losses,
        path="max_consecutive_losses",
        label="loss streak count",
    )
    if closed_count is not None and trade_count is not None:
        checks.append(
            _check(
                "ok" if trade_count == closed_count else "block",
                "trade count identity",
                "trades equals closed_trades",
                path="trades",
                metric=trade_count,
                limit=closed_count,
            )
        )
    if evidence.gross_exposure is not None and evidence.max_exposure is not None:
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        evidence.gross_exposure,
                        evidence.max_exposure,
                        scale=max(
                            1.0,
                            evidence.gross_exposure,
                            evidence.max_exposure,
                        ),
                    )
                    else "block"
                ),
                "exposure identity",
                "gross_exposure equals max_exposure for single-position backtests",
                path="gross_exposure",
                metric=evidence.gross_exposure,
                limit=evidence.max_exposure,
            )
        )
    if isinstance(evidence.trade_log, (tuple, list)):
        trade_log: Sequence[object] = evidence.trade_log
        if closed_count is not None:
            checks.append(
                _check(
                    "ok" if len(trade_log) == closed_count else "block",
                    "trade log length",
                    f"trade_log={len(trade_log)} closed_trades={closed_count}",
                    path="trade_log",
                    metric=len(trade_log),
                    limit=closed_count,
                )
            )
    else:
        checks.append(
            _check(
                "block",
                "trade log",
                "trade_log is not a sequence",
                path="trade_log",
            )
        )
        trade_log = ()
    for label, values, raw in (
        ("trade_pnls", evidence.trade_pnls, evidence.raw_trade_pnls),
        ("trade_returns", evidence.trade_returns, evidence.raw_trade_returns),
    ):
        if values is None:
            checks.append(
                _check(
                    "block",
                    label,
                    "missing or non-finite sequence",
                    path=label,
                )
            )
        elif closed_count is not None:
            checks.append(
                _check(
                    "ok" if len(values) == closed_count else "block",
                    label,
                    f"{label}={len(values)} closed_trades={closed_count}",
                    path=label,
                    metric=len(values),
                    limit=closed_count,
                )
            )
        elif not isinstance(raw, (tuple, list)):
            checks.append(_check("block", label, "not a sequence", path=label))
    if evidence.trade_pnls is not None and evidence.realized_pnl is not None:
        pnl_sum = sum(evidence.trade_pnls)
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        pnl_sum,
                        evidence.realized_pnl,
                        scale=evidence.cash_scale,
                    )
                    else "block"
                ),
                "trade PnL identity",
                "sum(trade_pnls) equals realized_pnl",
                path="trade_pnls",
                metric=pnl_sum,
                limit=evidence.realized_pnl,
            )
        )
    return checks, _BacktestSequenceState(
        closed_count=closed_count,
        trade_count=trade_count,
        loss_streak_count=loss_streak_count,
        trade_log=trade_log,
    )


@dataclass(frozen=True)
class _TradeEntryResult:
    checks: tuple[FinancialSanityCheck, ...]
    fee_sum: float
    net_pnl: float | None
    return_pct: float | None


def _trade_market_field_checks(
    trade: Mapping[str, Any],
    *,
    index: int,
) -> tuple[list[FinancialSanityCheck], float | None]:
    checks: list[FinancialSanityCheck] = []
    opened_at = _finite(trade.get("opened_at"))
    closed_at = _finite(trade.get("closed_at"))
    side = _finite(trade.get("side"))
    gross_notional = _finite(trade.get("gross_notional"))
    entry_price = _finite(trade.get("entry_price"))
    exit_mark_price = _finite(trade.get("exit_mark_price"))
    return_pct = _finite(trade.get("return_pct"))
    if opened_at is None or closed_at is None:
        checks.append(
            _check(
                "block",
                "trade timestamp",
                "missing or non-finite timestamp",
                path=f"trade_log[{index}]",
            )
        )
    else:
        checks.append(
            _check(
                "ok" if opened_at <= closed_at else "block",
                "trade timestamp",
                "opened_at is not after closed_at",
                path=f"trade_log[{index}].opened_at",
                metric=opened_at,
                limit=f"<={closed_at:g}",
            )
        )
    checks.append(
        _check(
            "ok" if side in {-1.0, 1.0} else "block",
            "trade side",
            "side is long or short",
            path=f"trade_log[{index}].side",
            metric=side if side is not None else "missing",
            limit="-1|1",
        )
    )
    for name, value, low in (
        ("gross_notional", gross_notional, 0.0),
        ("entry_price", entry_price, 0.0),
        ("exit_mark_price", exit_mark_price, -1e-12),
    ):
        if value is None:
            checks.append(
                _check(
                    "block",
                    "trade notional/price",
                    "missing or non-finite value",
                    path=f"trade_log[{index}].{name}",
                )
            )
        elif value <= low:
            checks.append(
                _check(
                    "block",
                    "trade notional/price",
                    "non-positive value",
                    path=f"trade_log[{index}].{name}",
                    metric=value,
                    limit=f">{low:g}",
                )
            )
    if return_pct is None:
        checks.append(
            _check(
                "block",
                "trade return",
                "missing or non-finite value",
                path=f"trade_log[{index}].return_pct",
            )
        )
    return checks, return_pct


def _trade_exit_reason_checks(
    trade: Mapping[str, Any],
    *,
    index: int,
    liquidation_status: str,
) -> list[FinancialSanityCheck]:
    reason = str(trade.get("exit_reason") or "").strip()
    checks = [
        _check(
            "ok" if reason else "block",
            "trade exit reason",
            reason or "missing exit reason",
            path=f"trade_log[{index}].exit_reason",
        )
    ]
    if reason.lower() == "liquidation":
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "trade log contains liquidation exit",
                path=f"trade_log[{index}].exit_reason",
                metric=reason,
                limit="not liquidation",
            )
        )
    return checks


def _trade_accounting_field_checks(
    trade: Mapping[str, Any],
    *,
    index: int,
    cash_scale: float,
) -> tuple[list[FinancialSanityCheck], float, float | None]:
    checks: list[FinancialSanityCheck] = []
    realized = _finite(trade.get("realized_pnl"))
    net = _finite(trade.get("net_pnl"))
    entry_fee = _finite(trade.get("entry_fee"))
    exit_fee = _finite(trade.get("exit_fee"))
    for name, value in (
        ("realized_pnl", realized),
        ("net_pnl", net),
        ("entry_fee", entry_fee),
        ("exit_fee", exit_fee),
    ):
        if value is None:
            checks.append(
                _check(
                    "block",
                    "trade log numeric",
                    "missing or non-finite value",
                    path=f"trade_log[{index}].{name}",
                )
            )
        elif name.endswith("fee") and value < 0.0:
            checks.append(
                _check(
                    "block",
                    "trade fee",
                    "fee is negative",
                    path=f"trade_log[{index}].{name}",
                    metric=value,
                    limit=">=0",
                )
            )
    fee_sum = (
        entry_fee + exit_fee if entry_fee is not None and exit_fee is not None else 0.0
    )
    complete_net = all(
        value is not None for value in (realized, net, entry_fee, exit_fee)
    )
    if complete_net:
        checked_realized = cast(float, realized)
        checked_net = cast(float, net)
        checked_entry_fee = cast(float, entry_fee)
        checked_exit_fee = cast(float, exit_fee)
        expected_net = checked_realized - checked_entry_fee - checked_exit_fee
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(checked_net, expected_net, scale=cash_scale)
                    else "block"
                ),
                "trade net PnL identity",
                "net_pnl equals realized_pnl - entry_fee - exit_fee",
                path=f"trade_log[{index}].net_pnl",
                metric=checked_net,
                limit=expected_net,
            )
        )
    return checks, fee_sum, net if complete_net else None


def _trade_entry_checks(
    trade: object,
    *,
    index: int,
    liquidation_status: str,
    cash_scale: float,
) -> _TradeEntryResult:
    if not isinstance(trade, Mapping):
        return _TradeEntryResult(
            checks=(
                _check(
                    "block",
                    "trade log entry",
                    "trade entry is not a mapping",
                    path=f"trade_log[{index}]",
                ),
            ),
            fee_sum=0.0,
            net_pnl=None,
            return_pct=None,
        )
    checks, return_pct = _trade_market_field_checks(trade, index=index)
    checks.extend(
        _trade_exit_reason_checks(
            trade,
            index=index,
            liquidation_status=liquidation_status,
        )
    )
    accounting_checks, fee_sum, net_pnl = _trade_accounting_field_checks(
        trade,
        index=index,
        cash_scale=cash_scale,
    )
    checks.extend(accounting_checks)
    return _TradeEntryResult(
        checks=tuple(checks),
        fee_sum=fee_sum,
        net_pnl=net_pnl,
        return_pct=return_pct,
    )


@dataclass(frozen=True)
class _TradeLogSummary:
    fee_sum: float
    net_values: tuple[float, ...]
    return_values: tuple[float, ...]


def _backtest_trade_log_checks(
    state: _BacktestSequenceState,
    *,
    liquidation_status: str,
    cash_scale: float,
) -> tuple[list[FinancialSanityCheck], _TradeLogSummary]:
    checks: list[FinancialSanityCheck] = []
    fee_sum = 0.0
    net_values: list[float] = []
    return_values: list[float] = []
    for index, trade in enumerate(state.trade_log):
        entry = _trade_entry_checks(
            trade,
            index=index,
            liquidation_status=liquidation_status,
            cash_scale=cash_scale,
        )
        checks.extend(entry.checks)
        fee_sum += entry.fee_sum
        if entry.return_pct is not None:
            return_values.append(entry.return_pct)
        if entry.net_pnl is not None:
            net_values.append(entry.net_pnl)
    return checks, _TradeLogSummary(
        fee_sum=fee_sum,
        net_values=tuple(net_values),
        return_values=tuple(return_values),
    )


def _backtest_log_identity_checks(
    evidence: _BacktestEvidence,
    state: _BacktestSequenceState,
    summary: _TradeLogSummary,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    if evidence.trade_pnls is not None and len(summary.net_values) == len(
        evidence.trade_pnls
    ):
        for index, (pnl, net) in enumerate(
            zip(evidence.trade_pnls, summary.net_values, strict=True)
        ):
            checks.append(
                _check(
                    (
                        "ok"
                        if _approx_equal(pnl, net, scale=evidence.cash_scale)
                        else "block"
                    ),
                    "trade PnL log identity",
                    "trade_pnls entry equals trade_log net_pnl",
                    path=f"trade_pnls[{index}]",
                    metric=pnl,
                    limit=net,
                )
            )
    if evidence.trade_returns is not None and len(summary.return_values) == len(
        evidence.trade_returns
    ):
        for index, (stored, logged) in enumerate(
            zip(evidence.trade_returns, summary.return_values, strict=True)
        ):
            checks.append(
                _check(
                    ("ok" if _approx_equal(stored, logged, scale=1.0) else "block"),
                    "trade return log identity",
                    "trade_returns entry equals trade_log return_pct",
                    path=f"trade_returns[{index}]",
                    metric=stored,
                    limit=logged,
                )
            )
    if evidence.total_fees is not None:
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        summary.fee_sum,
                        evidence.total_fees,
                        scale=evidence.cash_scale,
                    )
                    else "block"
                ),
                "fee identity",
                "sum(entry_fee + exit_fee) equals total_fees",
                path="total_fees",
                metric=summary.fee_sum,
                limit=evidence.total_fees,
            )
        )
    if (
        state.closed_count is not None
        and state.closed_count > 0
        and evidence.win_rate is not None
        and len(summary.net_values) == state.closed_count
    ):
        expected_win_rate = (
            sum(1 for value in summary.net_values if value > 0.0) / state.closed_count
        )
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        evidence.win_rate,
                        expected_win_rate,
                        scale=1.0,
                    )
                    else "block"
                ),
                "win rate identity",
                "win_rate equals positive net-PnL trades divided by closed_trades",
                path="win_rate",
                metric=evidence.win_rate,
                limit=expected_win_rate,
            )
        )
    return checks


def _backtest_pnl_path_checks(
    evidence: _BacktestEvidence,
    state: _BacktestSequenceState,
) -> list[FinancialSanityCheck]:
    if evidence.trade_pnls is None:
        return []
    checks: list[FinancialSanityCheck] = []
    expected_gross_profit = sum(value for value in evidence.trade_pnls if value > 0.0)
    expected_gross_loss = abs(
        sum(value for value in evidence.trade_pnls if value < 0.0)
    )
    expected_profit_factor = _expected_profit_factor(
        expected_gross_profit,
        expected_gross_loss,
    )
    expected_expectancy = (
        sum(evidence.trade_pnls) / len(evidence.trade_pnls)
        if evidence.trade_pnls
        else 0.0
    )
    expected_loss_streak = _max_consecutive_losses(evidence.trade_pnls)
    for label, path, metric, expected in (
        (
            "gross profit identity",
            "gross_profit",
            evidence.gross_profit,
            expected_gross_profit,
        ),
        (
            "gross loss identity",
            "gross_loss",
            evidence.gross_loss,
            expected_gross_loss,
        ),
        (
            "profit factor identity",
            "profit_factor",
            evidence.profit_factor,
            expected_profit_factor,
        ),
        (
            "expectancy identity",
            "expectancy",
            evidence.expectancy,
            expected_expectancy,
        ),
    ):
        if metric is not None:
            checks.append(
                _check(
                    (
                        "ok"
                        if _approx_equal(
                            metric,
                            expected,
                            scale=evidence.cash_scale,
                        )
                        else "block"
                    ),
                    label,
                    f"{path} matches trade_pnls",
                    path=path,
                    metric=metric,
                    limit=expected,
                )
            )
    if state.loss_streak_count is not None:
        checks.append(
            _check(
                ("ok" if state.loss_streak_count == expected_loss_streak else "block"),
                "loss streak identity",
                "max_consecutive_losses matches trade_pnls",
                path="max_consecutive_losses",
                metric=state.loss_streak_count,
                limit=expected_loss_streak,
            )
        )
    return checks


def _backtest_return_path_checks(
    evidence: _BacktestEvidence,
) -> list[FinancialSanityCheck]:
    if evidence.trade_returns is None:
        return []
    checks: list[FinancialSanityCheck] = []
    expected_average_return = (
        sum(evidence.trade_returns) / len(evidence.trade_returns)
        if evidence.trade_returns
        else 0.0
    )
    expected_return_stdev = _sample_stdev(evidence.trade_returns)
    for label, path, metric, expected in (
        (
            "average return identity",
            "average_trade_return",
            evidence.average_trade_return,
            expected_average_return,
        ),
        (
            "return stdev identity",
            "trade_return_stdev",
            evidence.trade_return_stdev,
            expected_return_stdev,
        ),
    ):
        if metric is not None:
            checks.append(
                _check(
                    ("ok" if _approx_equal(metric, expected, scale=1.0) else "block"),
                    label,
                    f"{path} matches trade_returns",
                    path=path,
                    metric=metric,
                    limit=expected,
                )
            )
    return checks


@dataclass(frozen=True)
class _EquityPointResult:
    checks: tuple[FinancialSanityCheck, ...]
    timestamp: float | None
    peak: float | None
    drawdown: float | None
    equity: float | None
    side: float | None


def _equity_point_checks(
    point: object,
    *,
    index: int,
    previous_timestamp: float | None,
    curve_peak: float | None,
) -> _EquityPointResult:
    if not isinstance(point, Mapping):
        return _EquityPointResult(
            checks=(
                _check(
                    "block",
                    "equity curve point",
                    "point is not a mapping",
                    path=f"equity_curve[{index}]",
                ),
            ),
            timestamp=previous_timestamp,
            peak=curve_peak,
            drawdown=None,
            equity=None,
            side=None,
        )
    timestamp = _finite(point.get("timestamp"))
    equity = _finite(point.get("equity"))
    drawdown = _finite(point.get("drawdown"))
    side = _finite(point.get("position_side"))
    if any(value is None for value in (timestamp, equity, drawdown, side)):
        return _EquityPointResult(
            checks=(
                _check(
                    "block",
                    "equity curve point",
                    "missing or non-finite point value",
                    path=f"equity_curve[{index}]",
                ),
            ),
            timestamp=previous_timestamp,
            peak=curve_peak,
            drawdown=None,
            equity=None,
            side=None,
        )
    checked_timestamp = cast(float, timestamp)
    checked_equity = cast(float, equity)
    checked_drawdown = cast(float, drawdown)
    checked_side = cast(float, side)
    checks: list[FinancialSanityCheck] = []
    if previous_timestamp is not None and checked_timestamp < previous_timestamp:
        checks.append(
            _check(
                "block",
                "equity curve chronology",
                "timestamps must be non-decreasing",
                path=f"equity_curve[{index}].timestamp",
                metric=checked_timestamp,
                limit=f">={previous_timestamp:g}",
            )
        )
    checks.append(
        _check(
            "ok" if 0.0 <= checked_drawdown <= 1.0 else "block",
            "equity curve drawdown",
            "drawdown is normalized 0-1",
            path=f"equity_curve[{index}].drawdown",
            metric=checked_drawdown,
            limit="0-1",
        )
    )
    next_peak = (
        checked_equity if curve_peak is None else max(curve_peak, checked_equity)
    )
    expected_drawdown = (
        1.0
        if checked_equity <= 0.0 and next_peak > 0.0
        else ((next_peak - checked_equity) / next_peak if next_peak else 0.0)
    )
    checks.append(
        _check(
            (
                "ok"
                if _approx_equal(checked_drawdown, expected_drawdown, scale=1.0)
                else "block"
            ),
            "equity curve drawdown identity",
            "point drawdown matches running equity peak",
            path=f"equity_curve[{index}].drawdown",
            metric=checked_drawdown,
            limit=expected_drawdown,
        )
    )
    return _EquityPointResult(
        checks=tuple(checks),
        timestamp=checked_timestamp,
        peak=next_peak,
        drawdown=checked_drawdown,
        equity=checked_equity,
        side=checked_side,
    )


def _backtest_equity_curve_checks(
    evidence: _BacktestEvidence,
) -> list[FinancialSanityCheck]:
    if not isinstance(evidence.equity_curve, (tuple, list)):
        return [
            _check(
                "block",
                "equity curve",
                "equity_curve is not a sequence",
                path="equity_curve",
            )
        ]
    if not evidence.equity_curve:
        return []
    checks: list[FinancialSanityCheck] = []
    curve_peak: float | None = None
    curve_drawdowns: list[float] = []
    final_equity: float | None = None
    final_side: float | None = None
    previous_timestamp: float | None = None
    for index, point in enumerate(evidence.equity_curve):
        point_result = _equity_point_checks(
            point,
            index=index,
            previous_timestamp=previous_timestamp,
            curve_peak=curve_peak,
        )
        checks.extend(point_result.checks)
        previous_timestamp = point_result.timestamp
        curve_peak = point_result.peak
        if point_result.drawdown is not None:
            curve_drawdowns.append(point_result.drawdown)
            final_equity = point_result.equity
            final_side = point_result.side
    if curve_drawdowns and evidence.max_drawdown is not None:
        expected_max_drawdown = max(curve_drawdowns)
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        evidence.max_drawdown,
                        expected_max_drawdown,
                        scale=1.0,
                    )
                    else "block"
                ),
                "max drawdown identity",
                "max_drawdown matches equity_curve",
                path="max_drawdown",
                metric=evidence.max_drawdown,
                limit=expected_max_drawdown,
            )
        )
    if (
        final_equity is not None
        and final_side == 0.0
        and evidence.ending_cash is not None
    ):
        checks.append(
            _check(
                (
                    "ok"
                    if _approx_equal(
                        final_equity,
                        evidence.ending_cash,
                        scale=evidence.cash_scale,
                    )
                    else "block"
                ),
                "ending equity identity",
                "final flat equity equals ending_cash",
                path="equity_curve[-1].equity",
                metric=final_equity,
                limit=evidence.ending_cash,
            )
        )
    return checks


def build_backtest_financial_sanity_report(
    result: object,
    *,
    source: str = "backtest",
    reject_liquidation: bool = True,
) -> FinancialSanityReport:
    """Validate internal accounting consistency for a generated backtest result."""

    evidence = _BacktestEvidence.from_result(result)
    checks = _backtest_scalar_checks(evidence)
    checks.extend(
        _backtest_liquidation_checks(
            evidence,
            reject_liquidation=reject_liquidation,
        )
    )
    checks.extend(_backtest_cash_identity_checks(evidence))
    sequence_checks, sequence_state = _backtest_sequence_checks(evidence)
    checks.extend(sequence_checks)
    liquidation_status = "block" if reject_liquidation else "warn"
    trade_checks, trade_summary = _backtest_trade_log_checks(
        sequence_state,
        liquidation_status=liquidation_status,
        cash_scale=evidence.cash_scale,
    )
    checks.extend(trade_checks)
    checks.extend(
        _backtest_log_identity_checks(
            evidence,
            sequence_state,
            trade_summary,
        )
    )
    checks.extend(_backtest_pnl_path_checks(evidence, sequence_state))
    checks.extend(_backtest_return_path_checks(evidence))
    checks.extend(_backtest_equity_curve_checks(evidence))
    return FinancialSanityReport(tuple(checks), source=source)


def blocking_reasons(report: FinancialSanityReport) -> list[str]:
    return [
        f"{check.path or check.label}: {check.detail}"
        for check in report.checks
        if check.status == "block"
    ]

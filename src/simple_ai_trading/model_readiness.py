"""Promotion/readiness checks for models before live-style execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .financial_sanity import build_model_financial_sanity_report
from .microstructure_data import MICROSTRUCTURE_SCHEMA_VERSION
from .model import ModelLoadError, TrainedModel, load_model

_ACCELERATOR_BACKENDS = frozenset({"cuda", "rocm", "directml", "mps"})
_LIVE_DATA_SOURCES = frozenset({"sqlite_market_data"})
_MICROSTRUCTURE_SCHEMA = MICROSTRUCTURE_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelReadinessCheck:
    status: str
    label: str
    detail: str
    metric: float | int | str | None = None
    limit: float | int | str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelReadinessReport:
    checks: tuple[ModelReadinessCheck, ...]
    model_path: str | None = None

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


class ModelPromotionError(ModelLoadError):
    """Raised when a readable model lacks required promotion evidence."""


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    metric: float | int | str | None = None,
    limit: float | int | str | None = None,
) -> ModelReadinessCheck:
    return ModelReadinessCheck(status, label, detail, metric=metric, limit=limit)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_at_least(value: object, fallback: int, minimum: int) -> int:
    parsed = _finite(value)
    if parsed is None:
        return max(minimum, fallback)
    return max(minimum, int(parsed))


def _is_accelerated_backend(kind: object, device: object) -> bool:
    backend_kind = str(kind or "").strip().lower()
    backend_device = str(device or "").strip().lower()
    return backend_kind in _ACCELERATOR_BACKENDS and backend_device not in {"", "cpu"}


def _walk_forward_gate_passed(raw: object) -> bool:
    if not isinstance(raw, dict) or raw.get("passed") is not True:
        return False
    if raw.get("reason") not in (None, ""):
        return False
    try:
        fold_count = int(raw.get("fold_count", 0) or 0)
        accepted_folds = int(raw.get("accepted_folds", 0) or 0)
        worst_score = float(raw.get("worst_score", 0.0) or 0.0)
        worst_pnl = float(raw.get("worst_realized_pnl", 0.0) or 0.0)
        worst_drawdown = float(raw.get("worst_max_drawdown", 1.0) or 1.0)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        fold_count > 0
        and accepted_folds == fold_count
        and worst_score > 0.0
        and worst_pnl > 0.0
        and 0.0 <= worst_drawdown <= 1.0
    )


def _terminal_holdout_passed(raw: object) -> bool:
    if not isinstance(raw, dict):
        return False
    result = raw.get("result")
    fingerprint = str(raw.get("dataset_fingerprint") or "").lower()
    try:
        evaluation_count = int(raw.get("evaluation_count", 0) or 0)
        rows = int(raw.get("rows", 0) or 0)
        score = float(raw.get("score", 0.0) or 0.0)
        realized_pnl = float(result.get("realized_pnl", 0.0)) if isinstance(result, dict) else 0.0
        liquidation_events = int(result.get("liquidation_events", 0) or 0) if isinstance(result, dict) else 1
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        raw.get("schema_version") == "terminal-holdout-v1"
        and raw.get("passed") is True
        and raw.get("reason") in (None, "")
        and evaluation_count == 1
        and rows > 0
        and score > 0.0
        and realized_pnl > 0.0
        and liquidation_events == 0
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
        and isinstance(result, dict)
        and result.get("accepted") is True
        and result.get("stopped_by_liquidation") is False
    )


def _accelerator_check(
    model: TrainedModel,
    *,
    label: str,
    kind_attr: str,
    device_attr: str,
    reason_attr: str,
) -> ModelReadinessCheck:
    kind = str(getattr(model, kind_attr, "") or "").strip().lower()
    device = str(getattr(model, device_attr, "") or "").strip()
    reason = str(getattr(model, reason_attr, "") or "").strip()
    if _is_accelerated_backend(kind, device):
        return _check("ok", label, f"{kind} device={device}", metric=kind)
    detail = f"{kind or 'missing'} device={device or 'missing'}"
    if reason:
        detail = f"{detail}; {reason}"
    return _check("block", label, detail, metric=kind or "missing", limit="gpu")


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _live_data_evidence_check(
    execution_validation: Mapping[str, object] | None,
    *,
    expected_symbol: str | None,
    expected_market_type: str | None,
    expected_interval: str | None,
    min_live_data_years: float,
    min_live_coverage_ratio: float,
    max_live_gap_count: int,
) -> ModelReadinessCheck:
    if not isinstance(execution_validation, dict):
        return _check("block", "live data evidence", "missing execution validation")
    raw = execution_validation.get("data_coverage")
    if not isinstance(raw, dict) or not raw:
        return _check("block", "live data evidence", "missing data_coverage evidence")

    source = _string_or_none(raw.get("source_scope"))
    symbol = str(raw.get("symbol") or "").strip().upper()
    market_type = str(raw.get("market_type") or "").strip().lower()
    interval = str(raw.get("interval") or "").strip()
    integrity_status = str(raw.get("integrity_status") or "").strip().lower()
    expected_interval_ms = _finite(raw.get("expected_interval_ms"))
    rows_used = _finite(raw.get("rows_used"))
    candles_used = _finite(raw.get("candles_used"))
    coverage_ratio = _finite(raw.get("coverage_ratio"))
    gap_count = _finite(raw.get("gap_count"))
    years = _finite(raw.get("used_duration_years"))
    full_available_history_used = raw.get("full_available_history_used") is True
    warnings = tuple(str(item) for item in raw.get("integrity_warnings") or () if str(item))

    reasons: list[str] = []
    if source not in _LIVE_DATA_SOURCES:
        reasons.append(f"source={source or 'missing'}")
    if expected_symbol and symbol != expected_symbol.strip().upper():
        reasons.append(f"symbol={symbol or 'missing'}!={expected_symbol.strip().upper()}")
    elif not symbol:
        reasons.append("symbol=missing")
    if expected_market_type and market_type != expected_market_type.strip().lower():
        reasons.append(f"market={market_type or 'missing'}!={expected_market_type.strip().lower()}")
    elif not market_type:
        reasons.append("market=missing")
    required_interval = "1s"
    runtime_interval = str(expected_interval or "").strip()
    if runtime_interval and runtime_interval != required_interval:
        reasons.append(f"runtime_interval={runtime_interval}!={required_interval}")
    if interval != required_interval:
        reasons.append(f"interval={interval or 'missing'}!={required_interval}")
    if expected_interval_ms is None or expected_interval_ms != 1000.0:
        reasons.append(f"expected_interval_ms={expected_interval_ms if expected_interval_ms is not None else 'missing'}")
    if integrity_status != "ok":
        reasons.append(f"integrity_status={integrity_status or 'missing'}")
    if rows_used is None or rows_used <= 0.0:
        reasons.append(f"rows_used={rows_used if rows_used is not None else 'missing'}")
    if candles_used is None or candles_used <= 0.0:
        reasons.append(f"candles_used={candles_used if candles_used is not None else 'missing'}")
    coverage_floor = max(0.0, min(1.0, float(min_live_coverage_ratio)))
    if coverage_ratio is None or coverage_ratio < coverage_floor:
        reasons.append(
            f"coverage_ratio={coverage_ratio if coverage_ratio is not None else 'missing'}<{coverage_floor:.4f}"
        )
    gap_limit = max(0, int(max_live_gap_count))
    if gap_count is None or int(gap_count) > gap_limit:
        reasons.append(f"gap_count={gap_count if gap_count is not None else 'missing'}>{gap_limit}")
    min_years = max(0.0, float(min_live_data_years))
    if years is None or years < min_years:
        reasons.append(f"used_duration_years={years if years is not None else 'missing'}<{min_years:.2f}")
    if not full_available_history_used:
        reasons.append("full_available_history_used=false")
    hard_warnings = {
        "no_candles_used",
        "no_model_rows_used",
        "coverage_gaps_detected",
        "coverage_ratio_below_99_5_percent",
        "recent_api_limit_not_full_history",
    }
    warning_hits = sorted(set(warnings).intersection(hard_warnings))
    if warning_hits:
        reasons.append("integrity_warnings=" + ",".join(warning_hits))

    if reasons:
        return _check(
            "block",
            "live data evidence",
            "failed live data contract: " + "; ".join(reasons),
            metric=coverage_ratio if coverage_ratio is not None else "missing",
            limit=f">={coverage_floor:.4f}, gaps<={gap_limit}, years>={min_years:.2f}",
        )
    return _check(
        "ok",
        "live data evidence",
        f"{symbol} {market_type} {interval} sqlite coverage={coverage_ratio:.4f} years={years:.2f}",
        metric=coverage_ratio,
        limit=f">={coverage_floor:.4f}",
    )


def _microstructure_replay_evidence_check(
    execution_validation: Mapping[str, object] | None,
    *,
    expected_symbol: str | None,
    min_captured_seconds: float,
    min_span_days: float,
    min_unique_days: int,
    min_normalized_rows: int,
) -> ModelReadinessCheck:
    if not isinstance(execution_validation, dict):
        return _check("block", "microstructure replay", "missing execution validation")
    raw = execution_validation.get("microstructure_replay")
    if not isinstance(raw, dict) or not raw:
        return _check("block", "microstructure replay", "missing HftBacktest replay evidence")

    passed = raw.get("passed") is True
    strategy_replay_passed = raw.get("strategy_replay_passed") is True
    replay_smoke_passed = raw.get("replay_smoke_passed") is True
    artifact_hashes_verified = raw.get("artifact_hashes_verified") is True
    immutable_market_data = raw.get("immutable_market_data") is True
    engine = str(raw.get("engine") or "").strip().lower()
    engine_version = str(raw.get("engine_version") or "").strip()
    schema_version = str(raw.get("schema_version") or "").strip()
    symbol = str(raw.get("symbol") or "").strip().upper()
    queue_model = str(raw.get("queue_model") or "").strip()
    latency_model = str(raw.get("latency_model") or "").strip()
    captured_seconds = _finite(raw.get("captured_seconds"))
    span_days = _finite(raw.get("span_days"))
    unique_days = _finite(raw.get("unique_days"))
    normalized_rows = _finite(raw.get("normalized_rows"))
    sequence_gaps = _finite(raw.get("sequence_gap_count"))
    crossed_books = _finite(raw.get("crossed_book_count"))
    invalid_events = _finite(raw.get("invalid_event_count"))
    clock_samples = _finite(raw.get("clock_sync_samples"))

    seconds_floor = max(0.0, float(min_captured_seconds))
    span_floor = max(0.0, float(min_span_days))
    day_floor = max(1, int(min_unique_days))
    row_floor = max(1, int(min_normalized_rows))
    reasons: list[str] = []
    if not passed:
        reasons.append("passed=false")
    if not strategy_replay_passed:
        reasons.append("strategy_replay_passed=false")
    if not replay_smoke_passed:
        reasons.append("replay_smoke_passed=false")
    if engine != "hftbacktest" or not engine_version:
        reasons.append(f"engine={engine or 'missing'} version={engine_version or 'missing'}")
    if schema_version != _MICROSTRUCTURE_SCHEMA:
        reasons.append(f"schema={schema_version or 'missing'}!={_MICROSTRUCTURE_SCHEMA}")
    if expected_symbol and symbol != expected_symbol.strip().upper():
        reasons.append(f"symbol={symbol or 'missing'}!={expected_symbol.strip().upper()}")
    elif not symbol:
        reasons.append("symbol=missing")
    if not queue_model:
        reasons.append("queue_model=missing")
    if not latency_model:
        reasons.append("latency_model=missing")
    if not immutable_market_data:
        reasons.append("immutable_market_data=false")
    if not artifact_hashes_verified:
        reasons.append("artifact_hashes_verified=false")
    if captured_seconds is None or captured_seconds < seconds_floor:
        reasons.append(
            f"captured_seconds={captured_seconds if captured_seconds is not None else 'missing'}<{seconds_floor:.0f}"
        )
    if span_days is None or span_days < span_floor:
        reasons.append(f"span_days={span_days if span_days is not None else 'missing'}<{span_floor:.1f}")
    if unique_days is None or int(unique_days) < day_floor:
        reasons.append(f"unique_days={unique_days if unique_days is not None else 'missing'}<{day_floor}")
    if normalized_rows is None or int(normalized_rows) < row_floor:
        reasons.append(
            f"normalized_rows={normalized_rows if normalized_rows is not None else 'missing'}<{row_floor}"
        )
    if sequence_gaps is None or int(sequence_gaps) != 0:
        reasons.append(f"sequence_gap_count={sequence_gaps if sequence_gaps is not None else 'missing'}")
    if crossed_books is None or int(crossed_books) != 0:
        reasons.append(f"crossed_book_count={crossed_books if crossed_books is not None else 'missing'}")
    if invalid_events is None or int(invalid_events) != 0:
        reasons.append(f"invalid_event_count={invalid_events if invalid_events is not None else 'missing'}")
    if clock_samples is None or int(clock_samples) < 3:
        reasons.append(f"clock_sync_samples={clock_samples if clock_samples is not None else 'missing'}<3")

    if reasons:
        return _check(
            "block",
            "microstructure replay",
            "failed promotion-grade L2 contract: " + "; ".join(reasons),
            metric=captured_seconds if captured_seconds is not None else "missing",
            limit=f">={seconds_floor:.0f}s across >={day_floor} days and >={span_floor:.1f}d span",
        )
    return _check(
        "ok",
        "microstructure replay",
        (
            f"{symbol} hftbacktest={engine_version} rows={int(normalized_rows or 0)} "
            f"captured_days={float(captured_seconds or 0.0) / 86400.0:.1f} span_days={float(span_days or 0.0):.1f}"
        ),
        metric=captured_seconds,
        limit=f">={seconds_floor:.0f}",
    )


def build_model_readiness_report(
    model: TrainedModel,
    *,
    model_path: str | Path | None = None,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
    require_model_candidate_search: bool = False,
    min_model_candidates: int = 2,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    min_live_data_years: float = 1.0,
    min_live_coverage_ratio: float = 0.995,
    max_live_gap_count: int = 0,
    require_microstructure_evidence: bool = False,
    min_microstructure_captured_seconds: float = 20.0 * 86_400.0,
    min_microstructure_span_days: float = 365.0,
    min_microstructure_unique_days: int = 20,
    min_microstructure_normalized_rows: int = 1_000_000,
) -> ModelReadinessReport:
    checks: list[ModelReadinessCheck] = []
    selection_risk = getattr(model, "selection_risk", None)
    if require_selection_risk:
        if not isinstance(selection_risk, dict) or not selection_risk:
            checks.append(_check("block", "selection risk", "missing promotion evidence"))
        else:
            passed = selection_risk.get("passed") is True
            deflated_score = _finite(selection_risk.get("deflated_score"))
            selected_score = _finite(selection_risk.get("selected_score"))
            effective_trials = int(_finite(selection_risk.get("effective_trials")) or 0)
            terminal_holdout_passed = _terminal_holdout_passed(selection_risk.get("terminal_holdout"))
            if (
                passed
                and deflated_score is not None
                and deflated_score > 0.0
                and effective_trials > 0
                and terminal_holdout_passed
            ):
                checks.append(
                    _check(
                        "ok",
                        "selection risk",
                        f"passed deflated_score={deflated_score:+.4f} trials={effective_trials}",
                        metric=deflated_score,
                        limit=">0",
                    )
                )
            else:
                checks.append(
                    _check(
                        "block",
                        "selection risk",
                        (
                            "failed promotion evidence "
                            f"passed={passed} deflated_score={deflated_score} trials={effective_trials} "
                            f"terminal={terminal_holdout_passed}"
                        ),
                        metric=deflated_score if deflated_score is not None else "missing",
                        limit=">0",
                    )
                )
            if selected_score is None or selected_score <= 0.0:
                checks.append(
                    _check(
                        "block",
                        "selected score",
                        f"non-positive selected_score={selected_score}",
                        metric=selected_score if selected_score is not None else "missing",
                        limit=">0",
                    )
                )

    required_candidates = _int_at_least(min_model_candidates, 2, 2)
    candidate_count = _int_at_least(getattr(model, "model_candidate_count", 1), 1, 1)
    selected_candidate = str(getattr(model, "model_selected_candidate", "") or "").strip()
    selection_score = _finite(getattr(model, "model_selection_score", None))
    candidate_ok = (
        candidate_count >= required_candidates
        and selected_candidate
        and selected_candidate.lower() not in {"default", "single"}
        and selection_score is not None
        and selection_score > 0.0
    )
    if candidate_ok:
        checks.append(
            _check(
                "ok",
                "model candidate search",
                f"selected {selected_candidate} from {candidate_count} candidates score={selection_score:+.4f}",
                metric=selection_score,
                limit=f">={required_candidates}",
            )
        )
    else:
        reasons: list[str] = []
        if candidate_count < required_candidates:
            reasons.append(f"candidates={candidate_count}<{required_candidates}")
        if not selected_candidate or selected_candidate.lower() in {"default", "single"}:
            reasons.append(f"selected={selected_candidate or 'missing'}")
        if selection_score is None or selection_score <= 0.0:
            reasons.append(f"score={selection_score if selection_score is not None else 'missing'}")
        checks.append(
            _check(
                "block" if require_model_candidate_search else "warn",
                "model candidate search",
                "single/default candidate evidence only"
                if not reasons
                else "insufficient candidate search evidence: " + ", ".join(reasons),
                metric=candidate_count,
                limit=f">={required_candidates}",
            )
        )

    if require_accelerator_evidence:
        checks.append(
            _accelerator_check(
                model,
                label="training accelerator",
                kind_attr="training_backend_kind",
                device_attr="training_backend_device",
                reason_attr="training_backend_reason",
            )
        )
        probability_calibration_size = _int_at_least(
            getattr(model, "probability_calibration_size", 0),
            0,
            0,
        )
        if probability_calibration_size <= 0:
            checks.append(
                _check(
                    "block",
                    "probability calibration accelerator",
                    "missing probability calibration sample evidence",
                    metric=probability_calibration_size,
                    limit=">0",
                )
            )
        else:
            checks.append(
                _accelerator_check(
                    model,
                    label="probability calibration accelerator",
                    kind_attr="probability_calibration_backend_kind",
                    device_attr="probability_calibration_backend_device",
                    reason_attr="probability_calibration_backend_reason",
                )
            )

    execution_validation = getattr(model, "execution_validation", None)
    if require_execution_validation:
        if not isinstance(execution_validation, dict) or not execution_validation:
            checks.append(_check("block", "execution validation", "missing symbol-specific execution evidence"))
        else:
            passed = execution_validation.get("passed") is True
            stress = execution_validation.get("stress")
            temporal = execution_validation.get("temporal_robustness")
            portfolio = execution_validation.get("portfolio")
            walk_forward = execution_validation.get("walk_forward_gate")
            stress_passed = isinstance(stress, dict) and stress.get("accepted") is True
            temporal_passed = isinstance(temporal, dict) and temporal.get("accepted") is True
            portfolio_passed = isinstance(portfolio, dict) and portfolio.get("accepted") is True
            walk_forward_passed = _walk_forward_gate_passed(walk_forward)
            symbol = str(execution_validation.get("symbol") or "").strip().upper()
            if passed and stress_passed and temporal_passed and portfolio_passed and walk_forward_passed and symbol:
                checks.append(_check("ok", "execution validation", f"{symbol} walk-forward+stress+temporal+portfolio accepted"))
            else:
                checks.append(
                    _check(
                        "block",
                        "execution validation",
                        (
                            "failed symbol-specific execution evidence "
                            f"passed={passed} stress={stress_passed} temporal={temporal_passed} "
                            f"portfolio={portfolio_passed} walk_forward={walk_forward_passed} "
                            f"symbol={symbol or 'missing'}"
                        ),
                    )
                )

    if require_live_data_evidence:
        checks.append(
            _live_data_evidence_check(
                execution_validation if isinstance(execution_validation, dict) else None,
                expected_symbol=expected_symbol,
                expected_market_type=expected_market_type,
                expected_interval=expected_interval,
                min_live_data_years=min_live_data_years,
                min_live_coverage_ratio=min_live_coverage_ratio,
                max_live_gap_count=max_live_gap_count,
            )
        )

    if require_microstructure_evidence:
        checks.append(
            _microstructure_replay_evidence_check(
                execution_validation if isinstance(execution_validation, dict) else None,
                expected_symbol=expected_symbol,
                min_captured_seconds=min_microstructure_captured_seconds,
                min_span_days=min_microstructure_span_days,
                min_unique_days=min_microstructure_unique_days,
                min_normalized_rows=min_microstructure_normalized_rows,
            )
        )

    policy = getattr(model, "meta_label_policy", None)
    if isinstance(policy, dict) and policy.get("enabled") is True:
        checks.append(_check("ok", "meta-label policy", str(policy.get("mode") or "enabled")))
    else:
        checks.append(_check("warn", "meta-label policy", "not enabled; entries use primary signal only"))

    warnings = [str(value) for value in getattr(model, "quality_warnings", []) or [] if str(value)]
    if any("meta_label_policy_unavailable" == warning for warning in warnings):
        checks.append(_check("warn", "model quality warnings", "meta-label policy unavailable"))
    elif warnings:
        checks.append(_check("warn", "model quality warnings", "; ".join(warnings[:3])))
    else:
        checks.append(_check("ok", "model quality warnings", "none"))

    sanity = build_model_financial_sanity_report(model, source=str(model_path or "model"))
    for item in sanity.checks:
        if item.status == "ok":
            continue
        checks.append(
            _check(
                item.status,
                f"financial sanity: {item.label}",
                item.detail if not item.path else f"{item.path}: {item.detail}",
                metric=item.metric,
                limit=item.limit,
            )
        )

    return ModelReadinessReport(
        checks=tuple(checks),
        model_path=str(model_path) if model_path is not None else None,
    )


def assert_model_promoted(
    model: TrainedModel,
    *,
    model_path: str | Path | None = None,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
    require_model_candidate_search: bool = False,
    min_model_candidates: int = 2,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    min_live_data_years: float = 1.0,
    min_live_coverage_ratio: float = 0.995,
    max_live_gap_count: int = 0,
    require_microstructure_evidence: bool = False,
    min_microstructure_captured_seconds: float = 20.0 * 86_400.0,
    min_microstructure_span_days: float = 365.0,
    min_microstructure_unique_days: int = 20,
    min_microstructure_normalized_rows: int = 1_000_000,
) -> ModelReadinessReport:
    report = build_model_readiness_report(
        model,
        model_path=model_path,
        require_selection_risk=require_selection_risk,
        require_execution_validation=require_execution_validation,
        require_model_candidate_search=require_model_candidate_search,
        min_model_candidates=min_model_candidates,
        require_accelerator_evidence=require_accelerator_evidence,
        require_live_data_evidence=require_live_data_evidence,
        expected_symbol=expected_symbol,
        expected_market_type=expected_market_type,
        expected_interval=expected_interval,
        min_live_data_years=min_live_data_years,
        min_live_coverage_ratio=min_live_coverage_ratio,
        max_live_gap_count=max_live_gap_count,
        require_microstructure_evidence=require_microstructure_evidence,
        min_microstructure_captured_seconds=min_microstructure_captured_seconds,
        min_microstructure_span_days=min_microstructure_span_days,
        min_microstructure_unique_days=min_microstructure_unique_days,
        min_microstructure_normalized_rows=min_microstructure_normalized_rows,
    )
    if not report.allowed:
        reasons = "; ".join(f"{check.label}: {check.detail}" for check in report.checks if check.status == "block")
        raise ModelPromotionError(reasons or "model promotion evidence failed")
    return report


def load_model_readiness_report(
    model_path: str | Path,
    *,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
    require_model_candidate_search: bool = False,
    min_model_candidates: int = 2,
    require_accelerator_evidence: bool = False,
    require_live_data_evidence: bool = False,
    expected_symbol: str | None = None,
    expected_market_type: str | None = None,
    expected_interval: str | None = None,
    min_live_data_years: float = 1.0,
    min_live_coverage_ratio: float = 0.995,
    max_live_gap_count: int = 0,
    require_microstructure_evidence: bool = False,
    min_microstructure_captured_seconds: float = 20.0 * 86_400.0,
    min_microstructure_span_days: float = 365.0,
    min_microstructure_unique_days: int = 20,
    min_microstructure_normalized_rows: int = 1_000_000,
) -> ModelReadinessReport:
    path = Path(model_path)
    model = load_model(path, expected_feature_version=None, expected_feature_dim=None, expected_feature_signature=None)
    return build_model_readiness_report(
        model,
        model_path=path,
        require_selection_risk=require_selection_risk,
        require_execution_validation=require_execution_validation,
        require_model_candidate_search=require_model_candidate_search,
        min_model_candidates=min_model_candidates,
        require_accelerator_evidence=require_accelerator_evidence,
        require_live_data_evidence=require_live_data_evidence,
        expected_symbol=expected_symbol,
        expected_market_type=expected_market_type,
        expected_interval=expected_interval,
        min_live_data_years=min_live_data_years,
        min_live_coverage_ratio=min_live_coverage_ratio,
        max_live_gap_count=max_live_gap_count,
        require_microstructure_evidence=require_microstructure_evidence,
        min_microstructure_captured_seconds=min_microstructure_captured_seconds,
        min_microstructure_span_days=min_microstructure_span_days,
        min_microstructure_unique_days=min_microstructure_unique_days,
        min_microstructure_normalized_rows=min_microstructure_normalized_rows,
    )

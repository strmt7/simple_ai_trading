"""Financial sanity checks for model and model-lab artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .assets import MAX_AUTONOMOUS_LEVERAGE
from .model import TrainedModel

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
_AI_UPLIFT_DEFAULT_MIN_MODEL_PARAMETERS_B = 2.0
_AI_UPLIFT_DEFAULT_MIN_PAIRED_SAMPLES = 8
_AI_UPLIFT_DEFAULT_MAX_SIGN_TEST_P = 0.40
_AI_UPLIFT_DEFAULT_MIN_POSITIVE_DELTA_RATE = 0.55
_AI_UPLIFT_DEFAULT_MIN_MEAN_SAMPLE_DELTA = 0.0
_REQUIRED_DATA_COVERAGE_TRUTH_BASIS = (
    "prices_from_timestamped_closed_candles",
    "coverage_measured_from_candle_close_time",
    "execution_results_are_simulated_not_exchange_fills",
)
_BLOCKED_DATA_SOURCE_TOKENS = ("synthetic", "fake", "mock", "demo", "sample", "placeholder", "generated")


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
    return FinancialSanityCheck(status, label, detail, path=path, metric=metric, limit=limit)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


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


def _finite_sequence(values: Sequence[object], *, path: str, label: str) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    for index, value in enumerate(values):
        parsed = _finite(value)
        if parsed is None:
            checks.append(_check("block", label, "non-finite numeric parameter", path=f"{path}[{index}]"))
        elif abs(parsed) > 1e9:
            checks.append(_check("block", label, "implausibly large numeric parameter", path=f"{path}[{index}]", metric=parsed, limit="abs<=1e9"))
        elif abs(parsed) > 1e6:
            checks.append(_check("warn", label, "large numeric parameter requires review", path=f"{path}[{index}]", metric=parsed, limit="abs<=1e6 preferred"))
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
        return _check("block", label, "missing or non-finite value", path=path, metric="missing", limit=f"{low:g}-{high:g}")
    hard_low = low if hard_low is None else hard_low
    hard_high = high if hard_high is None else hard_high
    if parsed < hard_low or parsed > hard_high:
        return _check("block", label, "outside hard financial bounds", path=path, metric=parsed, limit=f"{hard_low:g}-{hard_high:g}")
    if parsed < low or parsed > high:
        return _check("warn", label, "outside preferred financial bounds", path=path, metric=parsed, limit=f"{low:g}-{high:g}")
    return _check("ok", label, "within financial bounds", path=path, metric=parsed, limit=f"{low:g}-{high:g}")


def _has_promotion_evidence(model: TrainedModel) -> bool:
    selection_risk = getattr(model, "selection_risk", None)
    execution_validation = getattr(model, "execution_validation", None)
    return (
        (isinstance(selection_risk, Mapping) and bool(selection_risk))
        or (isinstance(execution_validation, Mapping) and bool(execution_validation))
    )


def _probability_calibration_checks(model: TrainedModel) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    promoted = _has_promotion_evidence(model)
    calibration_size = _finite(getattr(model, "probability_calibration_size", 0))
    brier_before = _finite(getattr(model, "probability_brier_before", None))
    brier_after = _finite(getattr(model, "probability_brier_after", None))
    ece_before = _finite(getattr(model, "probability_ece_before", None))
    ece_after = _finite(getattr(model, "probability_ece_after", None))
    log_loss_before = _finite(getattr(model, "probability_log_loss_before", None))
    log_loss_after = _finite(getattr(model, "probability_log_loss_after", None))
    any_metric = any(
        value is not None
        for value in (brier_before, brier_after, ece_before, ece_after, log_loss_before, log_loss_after)
    )

    if promoted and (calibration_size is None or calibration_size <= 0):
        checks.append(
            _check(
                "block",
                "probability calibration evidence",
                "promoted model is missing calibration sample evidence",
                path="probability_calibration_size",
                metric=calibration_size if calibration_size is not None else "missing",
                limit=">0",
            )
        )
    elif calibration_size is not None and calibration_size > 0:
        checks.append(
            _check(
                "ok",
                "probability calibration evidence",
                f"rows={int(calibration_size)}",
                path="probability_calibration_size",
                metric=int(calibration_size),
                limit=">0",
            )
        )

    if promoted and brier_after is None:
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
    elif brier_after is not None:
        checks.append(
            _range_check(
                brier_after,
                path="probability_brier_after",
                label="probability Brier score",
                low=0.0,
                high=_PREFERRED_PROBABILITY_BRIER_MAX,
                hard_low=0.0,
                hard_high=_HARD_PROBABILITY_BRIER_MAX,
            )
        )

    if promoted and ece_after is None:
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
    elif ece_after is not None:
        checks.append(
            _range_check(
                ece_after,
                path="probability_ece_after",
                label="probability calibration error",
                low=0.0,
                high=_PREFERRED_PROBABILITY_ECE_MAX,
                hard_low=0.0,
                hard_high=_HARD_PROBABILITY_ECE_MAX,
            )
        )

    if brier_before is not None and brier_after is not None and brier_after > brier_before + 1e-9:
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability Brier score",
                "calibration worsened Brier score",
                path="probability_brier_after",
                metric=brier_after,
                limit=f"<={brier_before:g}",
            )
        )
    if log_loss_before is not None and log_loss_after is not None and log_loss_after > log_loss_before + 1e-9:
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability log loss",
                "calibration worsened log loss",
                path="probability_log_loss_after",
                metric=log_loss_after,
                limit=f"<={log_loss_before:g}",
            )
        )
    if any_metric and not promoted and brier_after is None and ece_after is None:
        checks.append(
            _check(
                "warn",
                "probability calibration evidence",
                "partial calibration metrics are present without calibrated Brier or ECE",
                path="probability_brier_after",
            )
        )
    if ece_before is not None and ece_after is not None and ece_after > ece_before + 1e-9:
        checks.append(
            _check(
                "block" if promoted else "warn",
                "probability calibration error",
                "calibration increased expected calibration error",
                path="probability_ece_after",
                metric=ece_after,
                limit=f"<={ece_before:g}",
            )
        )
    return checks


def build_model_financial_sanity_report(model: TrainedModel, *, source: str = "model") -> FinancialSanityReport:
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
    checks.extend(_finite_sequence([getattr(model, "bias", None)], path="bias", label="bias"))
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
            checks.append(_check("block", "data coverage", "execution validation contains failed coverage", path="execution_validation.data_coverage"))
    return FinancialSanityReport(tuple(checks), source=source)


def _accepted_outcomes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        return []
    return [item for item in outcomes if isinstance(item, Mapping) and item.get("accepted") is True]


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


def _selection_risk_checks(
    outcome: Mapping[str, Any],
    *,
    objectives: Sequence[str],
    prefix: str,
) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
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
    for objective in objectives:
        report = _selection_risk_report_for_objective(raw, str(objective))
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
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
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
                "ok" if effective_trials is not None and effective_trials >= 1.0 else "block",
                "selection risk",
                "accepted selection-risk effective trial count",
                path=f"{report_path}.effective_trials",
                metric=effective_trials if effective_trials is not None else "missing",
                limit=">=1",
            )
        )
        overfit = report.get("overfit_diagnostics")
        if not isinstance(overfit, Mapping):
            checks.append(
                _check(
                    "block",
                    "selection risk",
                    "accepted selection-risk report is missing overfit diagnostics",
                    path=f"{report_path}.overfit_diagnostics",
                    metric="missing",
                    limit="passed diagnostics",
                )
            )
            continue
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
            if probability is None or max_probability is None or probability > max_probability:
                checks.append(
                    _check(
                        "block",
                        "selection risk",
                        "accepted selection-risk PBO exceeds limit",
                        path=f"{report_path}.overfit_diagnostics.probability_backtest_overfit",
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


def _stress_validation_checks(payload: Mapping[str, Any], *, path: str) -> list[FinancialSanityCheck]:
    checks = [
        _positive_numeric_check(payload, key="scenario_count", path=path, label="stress validation"),
        _nonnegative_numeric_check(payload, key="worst_realized_pnl", path=path, label="stress validation"),
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


def _robustness_validation_checks(payload: Mapping[str, Any], *, path: str) -> list[FinancialSanityCheck]:
    checks = [
        _positive_numeric_check(payload, key="window_count", path=path, label="temporal robustness"),
        _positive_numeric_check(payload, key="accepted_windows", path=path, label="temporal robustness"),
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
    checks.append(_nonnegative_numeric_check(payload, key="worst_realized_pnl", path=path, label="temporal robustness"))
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


def _iter_market_edge_reports(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
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
            reports.append((f"objectives[{objective_index}].market_edge", objective_edge))
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
                    reports.append((f"objectives[{objective_index}].{collection_name}[{item_index}].result.market_edge", edge))
    return reports


def _market_edge_checks(payload: Mapping[str, Any], *, path: str) -> list[FinancialSanityCheck]:
    checks: list[FinancialSanityCheck] = []
    reports = _iter_market_edge_reports(payload)
    summary_accepted = payload.get("market_edge_accepted")
    if summary_accepted is False:
        checks.append(_check("block", "market edge", "summary reports failed market-edge evidence", path=f"{path}.market_edge_accepted"))
    if payload.get("accepted") is True and isinstance(payload.get("objectives"), list) and not reports:
        checks.append(_check("block", "market edge", "accepted validation report is missing market-edge evidence", path=path))
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
                limit=f">={min_edge_pct:g}" if min_edge_pct is not None else "positive audited edge",
            )
        )
        if net_edge_pct is None:
            checks.append(_check("block", "market edge pct", "missing or non-finite net edge", path=f"{full_path}.net_edge_pct"))
        liquidation_events = _finite(report.get("liquidation_events"))
        if accepted is True and liquidation_events is not None and liquidation_events > 0:
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
                        metric=downside_ratio if downside_ratio is not None else "missing",
                        limit=f">={min_downside_ratio:g}",
                    )
                )
    return checks


def _truth_basis_values(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _positive_count_check(value: object, *, path: str, label: str) -> FinancialSanityCheck:
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
        checks.append(_check("block", "data coverage", "coverage integrity failed", path=path))
    elif integrity_status in {"ok", "warn"}:
        checks.append(_check("ok", "data coverage", f"integrity_status={integrity_status}", path=f"{path}.integrity_status"))
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
    elif any(token in source_scope_lc for token in _BLOCKED_DATA_SOURCE_TOKENS) or "binance" not in source_scope_lc:
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
        checks.append(_check("ok", "data source", f"source_scope={source_scope}", path=f"{path}.source_scope"))

    truth_basis = _truth_basis_values(coverage.get("truth_basis"))
    missing_basis = [item for item in _REQUIRED_DATA_COVERAGE_TRUTH_BASIS if item not in truth_basis]
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
        checks.append(_check("ok", "data truth basis", "required truth basis present", path=f"{path}.truth_basis"))

    checks.append(_positive_count_check(coverage.get("candles_used"), path=f"{path}.candles_used", label="coverage candles"))
    checks.append(_positive_count_check(coverage.get("rows_used"), path=f"{path}.rows_used", label="coverage rows"))
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


def build_model_lab_financial_sanity_report(payload: Mapping[str, Any], *, source: str = "model_lab") -> FinancialSanityReport:
    checks: list[FinancialSanityCheck] = []
    portfolio = payload.get("portfolio_risk")
    accepted_outcomes = _accepted_outcomes(payload)
    top_level_symbols = _symbol_sequence(payload.get("accepted_symbols"))
    if accepted_outcomes and not isinstance(portfolio, Mapping):
        checks.append(
            _check(
                "block",
                "portfolio risk",
                "accepted outcomes require a portfolio-risk report",
                path="portfolio_risk",
                metric="missing",
                limit="accepted portfolio-risk report",
            )
        )
    elif accepted_outcomes and portfolio.get("accepted") is not True:
        checks.append(
            _check(
                "block",
                "portfolio risk",
                "accepted outcomes require accepted portfolio-risk evidence",
                path="portfolio_risk.accepted",
                metric=portfolio.get("accepted"),
                limit=True,
            )
        )
    if top_level_symbols:
        duplicates = _duplicate_symbols(top_level_symbols)
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
                    metric=",".join(top_level_symbols),
                    limit="matching accepted outcomes",
                )
            )
    if isinstance(portfolio, Mapping) and portfolio.get("accepted") is True:
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
        if portfolio_symbols is None or not portfolio_symbols:
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
        if top_level_symbols is None or not top_level_symbols:
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
            top_level_symbols = []
        if portfolio_symbols and top_level_symbols and set(portfolio_symbols) != set(top_level_symbols):
            checks.append(
                _check(
                    "block",
                    "portfolio symbols",
                    "portfolio symbols differ from top-level accepted symbols",
                    path="portfolio_risk.accepted_symbols",
                    metric=",".join(portfolio_symbols),
                    limit=",".join(top_level_symbols),
                )
            )
        if portfolio_symbols and outcome_symbols and set(portfolio_symbols) != set(outcome_symbols):
            checks.append(
                _check(
                    "block",
                    "portfolio symbols",
                    "portfolio symbols differ from accepted outcome symbols",
                    path="portfolio_risk.accepted_symbols",
                    metric=",".join(portfolio_symbols),
                    limit=",".join(outcome_symbols),
                )
            )
        if top_level_symbols and outcome_symbols and set(top_level_symbols) != set(outcome_symbols):
            checks.append(
                _check(
                    "block",
                    "accepted symbols",
                    "top-level accepted symbols differ from accepted outcome symbols",
                    path="accepted_symbols",
                    metric=",".join(top_level_symbols),
                    limit=",".join(outcome_symbols),
                )
            )
        accepted_symbols = portfolio_symbols or top_level_symbols or outcome_symbols
        accepted_symbol_limit = (
            float(len(accepted_symbols))
            if accepted_symbols
            else 100.0
        )
        for key in ("effective_symbol_count", "correlation_adjusted_effective_symbol_count"):
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
        for key in ("portfolio_cvar_95", "portfolio_max_drawdown", "deployed_weight", "max_pairwise_correlation", "max_cluster_weight"):
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
    for outcome_index, outcome in enumerate(accepted_outcomes):
        prefix = f"outcomes[{outcome_index}]"
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
            checks.append(_check("block", "objective scores", "missing accepted objective scores", path=f"{prefix}.objective_scores"))
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
                _selection_risk_checks(
                    outcome,
                    objectives=accepted_objectives,
                    prefix=prefix,
                )
            )
        checks.extend(_data_coverage_checks(outcome.get("data_coverage"), path=f"{prefix}.data_coverage"))
        stress = outcome.get("stress_validation")
        robustness = outcome.get("robustness_validation")
        for field_name, field_value in (("stress_validation", stress), ("robustness_validation", robustness)):
            if not isinstance(field_value, Mapping):
                checks.append(_check("block", field_name, "missing accepted validation report", path=f"{prefix}.{field_name}"))
                continue
            if field_value.get("accepted") is not True:
                checks.append(_check("block", field_name, "accepted outcome has failed validation", path=f"{prefix}.{field_name}.accepted"))
            if field_name == "stress_validation":
                checks.extend(_stress_validation_checks(field_value, path=f"{prefix}.{field_name}"))
            if field_name == "robustness_validation":
                checks.extend(_robustness_validation_checks(field_value, path=f"{prefix}.{field_name}"))
            checks.extend(_market_edge_checks(field_value, path=f"{prefix}.{field_name}"))
        ai_uplift = outcome.get("ai_uplift")
        if isinstance(ai_uplift, Mapping):
            ai_uplift_accepted = ai_uplift.get("accepted") is True
            reasons = ai_uplift.get("reasons")
            if ai_uplift_accepted and isinstance(reasons, list) and reasons:
                checks.append(
                    _check(
                        "block",
                        "AI uplift",
                        "accepted AI uplift contains rejection reasons",
                        path=f"{prefix}.ai_uplift.reasons",
                    )
                )
            if ai_uplift_accepted:
                policy = ai_uplift.get("policy")
                min_parameters_b = _AI_UPLIFT_DEFAULT_MIN_MODEL_PARAMETERS_B
                min_paired_samples = _AI_UPLIFT_DEFAULT_MIN_PAIRED_SAMPLES
                max_sign_test_p = _AI_UPLIFT_DEFAULT_MAX_SIGN_TEST_P
                min_positive_delta_rate = _AI_UPLIFT_DEFAULT_MIN_POSITIVE_DELTA_RATE
                min_mean_sample_delta = _AI_UPLIFT_DEFAULT_MIN_MEAN_SAMPLE_DELTA
                if isinstance(policy, Mapping):
                    parsed_min = _finite(policy.get("min_model_parameters_b"))
                    if parsed_min is not None:
                        min_parameters_b = max(0.0, parsed_min)
                    parsed_samples = _finite(policy.get("min_paired_samples"))
                    if parsed_samples is not None:
                        min_paired_samples = max(0, int(parsed_samples))
                    parsed_sign_p = _finite(policy.get("max_sign_test_p_value"))
                    if parsed_sign_p is not None:
                        max_sign_test_p = max(0.0, min(1.0, parsed_sign_p))
                    parsed_rate = _finite(policy.get("min_positive_delta_rate"))
                    if parsed_rate is not None:
                        min_positive_delta_rate = max(0.0, min(1.0, parsed_rate))
                    parsed_mean_delta = _finite(policy.get("min_mean_sample_delta"))
                    if parsed_mean_delta is not None:
                        min_mean_sample_delta = parsed_mean_delta
                model_parameters_b = _finite(ai_uplift.get("model_parameters_b"))
                if model_parameters_b is None or model_parameters_b < min_parameters_b:
                    checks.append(
                        _check(
                            "block",
                            "AI uplift evidence",
                            "accepted AI uplift is missing required model-size evidence",
                            path=f"{prefix}.ai_uplift.model_parameters_b",
                            metric=model_parameters_b if model_parameters_b is not None else "missing",
                            limit=f">={min_parameters_b:g}",
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
                statistical = ai_uplift.get("statistical_evidence")
                if not isinstance(statistical, Mapping):
                    checks.append(
                        _check(
                            "block",
                            "AI uplift statistical evidence",
                            "accepted AI uplift is missing paired holdout statistical evidence",
                            path=f"{prefix}.ai_uplift.statistical_evidence",
                            metric="missing",
                            limit="accepted paired-sample evidence",
                        )
                    )
                else:
                    if statistical.get("accepted") is not True:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift has failed paired-sample evidence",
                                path=f"{prefix}.ai_uplift.statistical_evidence.accepted",
                                metric=statistical.get("accepted"),
                                limit=True,
                            )
                        )
                    if statistical.get("paired_sample_length_mismatch") is True:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift has unpaired sample lengths",
                                path=f"{prefix}.ai_uplift.statistical_evidence.paired_sample_length_mismatch",
                                metric=True,
                                limit=False,
                            )
                        )
                    sample_count = _finite(statistical.get("sample_count"))
                    if sample_count is None or sample_count < min_paired_samples:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift has too few paired holdout samples",
                                path=f"{prefix}.ai_uplift.statistical_evidence.sample_count",
                                metric=sample_count if sample_count is not None else "missing",
                                limit=f">={min_paired_samples}",
                            )
                        )
                    sample_count_is_integer = (
                        sample_count is not None
                        and sample_count >= 0.0
                        and float(sample_count).is_integer()
                    )
                    if sample_count is not None and not sample_count_is_integer:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift sample count must be a nonnegative integer",
                                path=f"{prefix}.ai_uplift.statistical_evidence.sample_count",
                                metric=sample_count,
                                limit="nonnegative integer",
                            )
                        )
                    positive_count = _finite(statistical.get("positive_delta_count"))
                    positive_count_is_integer = (
                        positive_count is not None
                        and positive_count >= 0.0
                        and float(positive_count).is_integer()
                    )
                    if (
                        positive_count is None
                        or positive_count < 0.0
                        or (sample_count is not None and positive_count > sample_count)
                        or not positive_count_is_integer
                    ):
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift positive-delta count is inconsistent",
                                path=f"{prefix}.ai_uplift.statistical_evidence.positive_delta_count",
                                metric=positive_count if positive_count is not None else "missing",
                                limit="integer and 0<=positive_delta_count<=sample_count",
                            )
                        )
                    positive_rate = _finite(statistical.get("positive_delta_rate"))
                    if positive_rate is None or positive_rate < min_positive_delta_rate:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift positive-delta rate is too weak",
                                path=f"{prefix}.ai_uplift.statistical_evidence.positive_delta_rate",
                                metric=positive_rate if positive_rate is not None else "missing",
                                limit=f">={min_positive_delta_rate:g}",
                            )
                        )
                    if (
                        sample_count is not None
                        and sample_count > 0.0
                        and positive_count is not None
                        and sample_count_is_integer
                        and positive_count_is_integer
                        and positive_count <= sample_count
                        and positive_rate is not None
                    ):
                        expected_rate = positive_count / sample_count
                        if abs(positive_rate - expected_rate) > 1e-6:
                            checks.append(
                                _check(
                                    "block",
                                    "AI uplift statistical evidence",
                                    "accepted AI uplift positive-delta rate does not match counts",
                                    path=f"{prefix}.ai_uplift.statistical_evidence.positive_delta_rate",
                                    metric=positive_rate,
                                    limit=f"{expected_rate:g}",
                                )
                            )
                    sign_p = _finite(statistical.get("sign_test_p_value"))
                    if sign_p is None or sign_p > max_sign_test_p:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift sign test is too weak",
                                path=f"{prefix}.ai_uplift.statistical_evidence.sign_test_p_value",
                                metric=sign_p if sign_p is not None else "missing",
                                limit=f"<={max_sign_test_p:g}",
                            )
                        )
                    if (
                        sample_count is not None
                        and positive_count is not None
                        and sample_count_is_integer
                        and positive_count_is_integer
                        and positive_count <= sample_count
                        and sign_p is not None
                    ):
                        expected_sign_p = _binomial_upper_tail(int(sample_count), int(positive_count))
                        if abs(sign_p - expected_sign_p) > 1e-9:
                            checks.append(
                                _check(
                                    "block",
                                    "AI uplift statistical evidence",
                                    "accepted AI uplift sign-test p-value does not match counts",
                                    path=f"{prefix}.ai_uplift.statistical_evidence.sign_test_p_value",
                                    metric=sign_p,
                                    limit=f"{expected_sign_p:g}",
                                )
                            )
                    mean_delta = _finite(statistical.get("mean_delta"))
                    if mean_delta is None or mean_delta <= min_mean_sample_delta:
                        checks.append(
                            _check(
                                "block",
                                "AI uplift statistical evidence",
                                "accepted AI uplift mean paired delta is too weak",
                                path=f"{prefix}.ai_uplift.statistical_evidence.mean_delta",
                                metric=mean_delta if mean_delta is not None else "missing",
                                limit=f">{min_mean_sample_delta:g}",
                            )
                        )
            deltas = ai_uplift.get("deltas")
            if isinstance(deltas, Mapping):
                for key, value in deltas.items():
                    parsed = _finite(value)
                    if parsed is None:
                        checks.append(_check("block", "AI uplift delta", "non-finite delta", path=f"{prefix}.ai_uplift.deltas.{key}"))
                if ai_uplift_accepted:
                    for key in ("max_consecutive_losses",):
                        parsed = _finite(deltas.get(key))
                        if parsed is not None and parsed > 0.0:
                            checks.append(
                                _check(
                                    "block",
                                    "AI uplift tail risk",
                                    "accepted AI uplift worsens loss-streak risk",
                                    path=f"{prefix}.ai_uplift.deltas.{key}",
                                    metric=parsed,
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
            ai_metrics = ai_uplift.get("ai")
            if ai_uplift_accepted and isinstance(ai_metrics, Mapping):
                liquidations = _finite(ai_metrics.get("liquidation_events"))
                if liquidations is not None and liquidations > 0.0:
                    checks.append(
                        _check(
                            "block",
                            "AI uplift liquidation risk",
                            "accepted AI uplift contains liquidation events",
                            path=f"{prefix}.ai_uplift.ai.liquidation_events",
                            metric=liquidations,
                            limit=0,
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
        checks.append(_check("block", label, "negative count", path=path, metric=value, limit=">=0"))
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


def build_backtest_financial_sanity_report(
    result: object,
    *,
    source: str = "backtest",
    reject_liquidation: bool = True,
) -> FinancialSanityReport:
    """Validate internal accounting consistency for a generated backtest result."""

    checks: list[FinancialSanityCheck] = []
    starting_cash = _finite(getattr(result, "starting_cash", None))
    ending_cash = _finite(getattr(result, "ending_cash", None))
    realized_pnl = _finite(getattr(result, "realized_pnl", None))
    total_fees = _finite(getattr(result, "total_fees", None))
    buy_hold_pnl = _finite(getattr(result, "buy_hold_pnl", 0.0))
    edge_vs_buy_hold = _finite(getattr(result, "edge_vs_buy_hold", 0.0))
    win_rate = _finite(getattr(result, "win_rate", None))
    max_drawdown = _finite(getattr(result, "max_drawdown", None))
    trades = _finite(getattr(result, "trades", None))
    closed_trades = _finite(getattr(result, "closed_trades", None))
    gross_exposure = _finite(getattr(result, "gross_exposure", None))
    max_exposure = _finite(getattr(result, "max_exposure", None))
    gross_profit = _finite(getattr(result, "gross_profit", 0.0))
    gross_loss = _finite(getattr(result, "gross_loss", 0.0))
    profit_factor = _finite(getattr(result, "profit_factor", 0.0))
    expectancy = _finite(getattr(result, "expectancy", 0.0))
    average_trade_return = _finite(getattr(result, "average_trade_return", 0.0))
    trade_return_stdev = _finite(getattr(result, "trade_return_stdev", 0.0))
    max_consecutive_losses = _finite(getattr(result, "max_consecutive_losses", 0))
    stopped_by_liquidation = bool(getattr(result, "stopped_by_liquidation", False))
    liquidation_events = _finite(getattr(result, "liquidation_events", 0))
    liquidation_loss = _finite(getattr(result, "liquidation_loss", 0.0))
    equity_curve = getattr(result, "equity_curve", ())
    trade_log = getattr(result, "trade_log", ())
    trade_pnls = _numeric_sequence(getattr(result, "trade_pnls", ()))
    trade_returns = _numeric_sequence(getattr(result, "trade_returns", ()))

    for name, value, minimum, maximum in (
        ("starting_cash", starting_cash, 0.0, None),
        ("ending_cash", ending_cash, 0.0, None),
        ("realized_pnl", realized_pnl, None, None),
        ("total_fees", total_fees, 0.0, None),
        ("buy_hold_pnl", buy_hold_pnl, None, None),
        ("edge_vs_buy_hold", edge_vs_buy_hold, None, None),
        ("win_rate", win_rate, 0.0, 1.0),
        ("max_drawdown", max_drawdown, 0.0, 1.0),
        ("trades", trades, 0.0, None),
        ("closed_trades", closed_trades, 0.0, None),
        ("gross_exposure", gross_exposure, 0.0, None),
        ("max_exposure", max_exposure, 0.0, None),
        ("gross_profit", gross_profit, 0.0, None),
        ("gross_loss", gross_loss, 0.0, None),
        ("profit_factor", profit_factor, 0.0, 999.0),
        ("expectancy", expectancy, None, None),
        ("average_trade_return", average_trade_return, None, None),
        ("trade_return_stdev", trade_return_stdev, 0.0, None),
        ("max_consecutive_losses", max_consecutive_losses, 0.0, None),
    ):
        if value is None:
            checks.append(_check("block", "backtest accounting", "missing or non-finite value", path=name))
            continue
        if minimum is not None and value < minimum:
            checks.append(_check("block", "backtest accounting", "value below financial bound", path=name, metric=value, limit=f">={minimum:g}"))
        elif maximum is not None and value > maximum:
            checks.append(_check("block", "backtest accounting", "value above financial bound", path=name, metric=value, limit=f"<={maximum:g}"))
        else:
            checks.append(_check("ok", "backtest accounting", "finite bounded value", path=name, metric=value))

    liquidation_status = "block" if reject_liquidation else "warn"
    if stopped_by_liquidation:
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
        checks.append(_check("ok", "liquidation evidence", "not stopped by liquidation", path="stopped_by_liquidation"))
    if liquidation_events is None:
        checks.append(_check("block", "liquidation events", "missing or non-finite value", path="liquidation_events"))
    elif liquidation_events < 0.0:
        checks.append(
            _check(
                "block",
                "liquidation events",
                "liquidation event count is negative",
                path="liquidation_events",
                metric=liquidation_events,
                limit=">=0",
            )
        )
    elif liquidation_events > 0.0:
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "backtest contains liquidation events",
                path="liquidation_events",
                metric=liquidation_events,
                limit=0,
            )
        )
    else:
        checks.append(_check("ok", "liquidation evidence", "no liquidation events", path="liquidation_events", metric=0))
    if liquidation_loss is None:
        checks.append(_check("block", "liquidation loss", "missing or non-finite value", path="liquidation_loss"))
    elif liquidation_loss < 0.0:
        checks.append(
            _check(
                "block",
                "liquidation loss",
                "liquidation loss is negative",
                path="liquidation_loss",
                metric=liquidation_loss,
                limit=">=0",
            )
        )
    elif liquidation_loss > 0.0:
        checks.append(
            _check(
                liquidation_status,
                "liquidation evidence",
                "backtest contains liquidation loss",
                path="liquidation_loss",
                metric=liquidation_loss,
                limit=0,
            )
        )
    else:
        checks.append(_check("ok", "liquidation evidence", "no liquidation loss", path="liquidation_loss", metric=0.0))

    cash_scale = max(1.0, abs(starting_cash or 0.0), abs(ending_cash or 0.0))
    if starting_cash is not None and ending_cash is not None and realized_pnl is not None:
        expected_realized = ending_cash - starting_cash
        checks.append(
            _check(
                "ok" if _approx_equal(realized_pnl, expected_realized, scale=cash_scale) else "block",
                "backtest cash identity",
                "realized_pnl equals ending_cash - starting_cash",
                path="realized_pnl",
                metric=realized_pnl,
                limit=expected_realized,
            )
        )
    if realized_pnl is not None and buy_hold_pnl is not None and edge_vs_buy_hold is not None:
        expected_edge = realized_pnl - buy_hold_pnl
        checks.append(
            _check(
                "ok" if _approx_equal(edge_vs_buy_hold, expected_edge, scale=cash_scale) else "block",
                "backtest edge identity",
                "edge_vs_buy_hold equals realized_pnl - buy_hold_pnl",
                path="edge_vs_buy_hold",
                metric=edge_vs_buy_hold,
                limit=expected_edge,
            )
        )

    closed_count = _integer_count_check(checks, closed_trades, path="closed_trades", label="closed trade count")
    trade_count = _integer_count_check(checks, trades, path="trades", label="trade count")
    loss_streak_count = _integer_count_check(
        checks,
        max_consecutive_losses,
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
    if gross_exposure is not None and max_exposure is not None:
        checks.append(
            _check(
                "ok" if _approx_equal(gross_exposure, max_exposure, scale=max(1.0, gross_exposure, max_exposure)) else "block",
                "exposure identity",
                "gross_exposure equals max_exposure for single-position backtests",
                path="gross_exposure",
                metric=gross_exposure,
                limit=max_exposure,
            )
        )
    if isinstance(trade_log, (tuple, list)):
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
        checks.append(_check("block", "trade log", "trade_log is not a sequence", path="trade_log"))
        trade_log = ()

    for label, values in (("trade_pnls", trade_pnls), ("trade_returns", trade_returns)):
        raw = getattr(result, label, ())
        if values is None:
            checks.append(_check("block", label, "missing or non-finite sequence", path=label))
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

    if trade_pnls is not None and realized_pnl is not None:
        pnl_sum = sum(trade_pnls)
        checks.append(
            _check(
                "ok" if _approx_equal(pnl_sum, realized_pnl, scale=cash_scale) else "block",
                "trade PnL identity",
                "sum(trade_pnls) equals realized_pnl",
                path="trade_pnls",
                metric=pnl_sum,
                limit=realized_pnl,
            )
        )

    fee_sum = 0.0
    net_log_values: list[float] = []
    return_log_values: list[float] = []
    for index, trade in enumerate(trade_log):
        if not isinstance(trade, Mapping):
            checks.append(_check("block", "trade log entry", "trade entry is not a mapping", path=f"trade_log[{index}]"))
            continue
        opened_at = _finite(trade.get("opened_at"))
        closed_at = _finite(trade.get("closed_at"))
        side = _finite(trade.get("side"))
        gross_notional = _finite(trade.get("gross_notional"))
        entry_price = _finite(trade.get("entry_price"))
        exit_mark_price = _finite(trade.get("exit_mark_price"))
        return_pct = _finite(trade.get("return_pct"))
        if opened_at is None or closed_at is None:
            checks.append(_check("block", "trade timestamp", "missing or non-finite timestamp", path=f"trade_log[{index}]"))
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
                checks.append(_check("block", "trade notional/price", "missing or non-finite value", path=f"trade_log[{index}].{name}"))
            elif value <= low:
                checks.append(_check("block", "trade notional/price", "non-positive value", path=f"trade_log[{index}].{name}", metric=value, limit=f">{low:g}"))
        if return_pct is None:
            checks.append(_check("block", "trade return", "missing or non-finite value", path=f"trade_log[{index}].return_pct"))
        else:
            return_log_values.append(return_pct)
        reason = str(trade.get("exit_reason") or "").strip()
        checks.append(
            _check(
                "ok" if reason else "block",
                "trade exit reason",
                reason or "missing exit reason",
                path=f"trade_log[{index}].exit_reason",
            )
        )
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
        realized = _finite(trade.get("realized_pnl"))
        net = _finite(trade.get("net_pnl"))
        entry_fee = _finite(trade.get("entry_fee"))
        exit_fee = _finite(trade.get("exit_fee"))
        for name, value in (("realized_pnl", realized), ("net_pnl", net), ("entry_fee", entry_fee), ("exit_fee", exit_fee)):
            if value is None:
                checks.append(_check("block", "trade log numeric", "missing or non-finite value", path=f"trade_log[{index}].{name}"))
            elif name.endswith("fee") and value < 0.0:
                checks.append(_check("block", "trade fee", "fee is negative", path=f"trade_log[{index}].{name}", metric=value, limit=">=0"))
        if entry_fee is not None and exit_fee is not None:
            fee_sum += entry_fee + exit_fee
        if realized is not None and net is not None and entry_fee is not None and exit_fee is not None:
            expected_net = realized - entry_fee - exit_fee
            checks.append(
                _check(
                    "ok" if _approx_equal(net, expected_net, scale=cash_scale) else "block",
                    "trade net PnL identity",
                    "net_pnl equals realized_pnl - entry_fee - exit_fee",
                    path=f"trade_log[{index}].net_pnl",
                    metric=net,
                    limit=expected_net,
                )
            )
            net_log_values.append(net)

    if trade_pnls is not None and len(net_log_values) == len(trade_pnls):
        for index, (pnl, net) in enumerate(zip(trade_pnls, net_log_values, strict=True)):
            checks.append(
                _check(
                    "ok" if _approx_equal(pnl, net, scale=cash_scale) else "block",
                    "trade PnL log identity",
                    "trade_pnls entry equals trade_log net_pnl",
                    path=f"trade_pnls[{index}]",
                    metric=pnl,
                    limit=net,
                )
            )
    if trade_returns is not None and len(return_log_values) == len(trade_returns):
        for index, (stored, logged) in enumerate(zip(trade_returns, return_log_values, strict=True)):
            checks.append(
                _check(
                    "ok" if _approx_equal(stored, logged, scale=1.0) else "block",
                    "trade return log identity",
                    "trade_returns entry equals trade_log return_pct",
                    path=f"trade_returns[{index}]",
                    metric=stored,
                    limit=logged,
                )
            )
    if total_fees is not None and isinstance(trade_log, (tuple, list)):
        checks.append(
            _check(
                "ok" if _approx_equal(fee_sum, total_fees, scale=cash_scale) else "block",
                "fee identity",
                "sum(entry_fee + exit_fee) equals total_fees",
                path="total_fees",
                metric=fee_sum,
                limit=total_fees,
            )
        )
    if closed_count is not None and closed_count > 0 and win_rate is not None and len(net_log_values) == closed_count:
        expected_win_rate = sum(1 for value in net_log_values if value > 0.0) / closed_count
        checks.append(
            _check(
                "ok" if _approx_equal(win_rate, expected_win_rate, scale=1.0) else "block",
                "win rate identity",
                "win_rate equals positive net-PnL trades divided by closed_trades",
                path="win_rate",
                metric=win_rate,
                limit=expected_win_rate,
            )
        )
    if trade_pnls is not None:
        expected_gross_profit = sum(value for value in trade_pnls if value > 0.0)
        expected_gross_loss = abs(sum(value for value in trade_pnls if value < 0.0))
        expected_profit_factor = _expected_profit_factor(expected_gross_profit, expected_gross_loss)
        expected_expectancy = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0
        expected_loss_streak = _max_consecutive_losses(trade_pnls)
        for label, path, metric, expected in (
            ("gross profit identity", "gross_profit", gross_profit, expected_gross_profit),
            ("gross loss identity", "gross_loss", gross_loss, expected_gross_loss),
            ("profit factor identity", "profit_factor", profit_factor, expected_profit_factor),
            ("expectancy identity", "expectancy", expectancy, expected_expectancy),
        ):
            if metric is not None:
                checks.append(
                    _check(
                        "ok" if _approx_equal(metric, expected, scale=cash_scale) else "block",
                        label,
                        f"{path} matches trade_pnls",
                        path=path,
                        metric=metric,
                        limit=expected,
                    )
                )
        if loss_streak_count is not None:
            checks.append(
                _check(
                    "ok" if loss_streak_count == expected_loss_streak else "block",
                    "loss streak identity",
                    "max_consecutive_losses matches trade_pnls",
                    path="max_consecutive_losses",
                    metric=loss_streak_count,
                    limit=expected_loss_streak,
                )
            )
    if trade_returns is not None:
        expected_average_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        expected_return_stdev = _sample_stdev(trade_returns)
        for label, path, metric, expected in (
            ("average return identity", "average_trade_return", average_trade_return, expected_average_return),
            ("return stdev identity", "trade_return_stdev", trade_return_stdev, expected_return_stdev),
        ):
            if metric is not None:
                checks.append(
                    _check(
                        "ok" if _approx_equal(metric, expected, scale=1.0) else "block",
                        label,
                        f"{path} matches trade_returns",
                        path=path,
                        metric=metric,
                        limit=expected,
                    )
                )
    if not isinstance(equity_curve, (tuple, list)):
        checks.append(_check("block", "equity curve", "equity_curve is not a sequence", path="equity_curve"))
    elif equity_curve:
        curve_peak: float | None = None
        curve_drawdowns: list[float] = []
        final_equity: float | None = None
        final_side: float | None = None
        previous_timestamp: float | None = None
        for index, point in enumerate(equity_curve):
            if not isinstance(point, Mapping):
                checks.append(_check("block", "equity curve point", "point is not a mapping", path=f"equity_curve[{index}]"))
                continue
            timestamp = _finite(point.get("timestamp"))
            equity = _finite(point.get("equity"))
            drawdown = _finite(point.get("drawdown"))
            side = _finite(point.get("position_side"))
            if timestamp is None or equity is None or drawdown is None or side is None:
                checks.append(_check("block", "equity curve point", "missing or non-finite point value", path=f"equity_curve[{index}]"))
                continue
            if previous_timestamp is not None and timestamp < previous_timestamp:
                checks.append(_check("block", "equity curve chronology", "timestamps must be non-decreasing", path=f"equity_curve[{index}].timestamp", metric=timestamp, limit=f">={previous_timestamp:g}"))
            previous_timestamp = timestamp
            checks.append(
                _check(
                    "ok" if 0.0 <= drawdown <= 1.0 else "block",
                    "equity curve drawdown",
                    "drawdown is normalized 0-1",
                    path=f"equity_curve[{index}].drawdown",
                    metric=drawdown,
                    limit="0-1",
                )
            )
            curve_peak = equity if curve_peak is None else max(curve_peak, equity)
            expected_drawdown = 1.0 if equity <= 0.0 and curve_peak > 0.0 else ((curve_peak - equity) / curve_peak if curve_peak else 0.0)
            checks.append(
                _check(
                    "ok" if _approx_equal(drawdown, expected_drawdown, scale=1.0) else "block",
                    "equity curve drawdown identity",
                    "point drawdown matches running equity peak",
                    path=f"equity_curve[{index}].drawdown",
                    metric=drawdown,
                    limit=expected_drawdown,
                )
            )
            curve_drawdowns.append(drawdown)
            final_equity = equity
            final_side = side
        if curve_drawdowns and max_drawdown is not None:
            expected_max_drawdown = max(curve_drawdowns)
            checks.append(
                _check(
                    "ok" if _approx_equal(max_drawdown, expected_max_drawdown, scale=1.0) else "block",
                    "max drawdown identity",
                    "max_drawdown matches equity_curve",
                    path="max_drawdown",
                    metric=max_drawdown,
                    limit=expected_max_drawdown,
                )
            )
        if final_equity is not None and final_side == 0.0 and ending_cash is not None:
            checks.append(
                _check(
                    "ok" if _approx_equal(final_equity, ending_cash, scale=cash_scale) else "block",
                    "ending equity identity",
                    "final flat equity equals ending_cash",
                    path="equity_curve[-1].equity",
                    metric=final_equity,
                    limit=ending_cash,
                )
            )
    return FinancialSanityReport(tuple(checks), source=source)


def blocking_reasons(report: FinancialSanityReport) -> list[str]:
    return [
        f"{check.path or check.label}: {check.detail}"
        for check in report.checks
        if check.status == "block"
    ]

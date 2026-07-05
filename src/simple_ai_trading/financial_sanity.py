"""Financial sanity checks for model and model-lab artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .assets import MAX_AUTONOMOUS_LEVERAGE
from .model import TrainedModel


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


def build_model_lab_financial_sanity_report(payload: Mapping[str, Any], *, source: str = "model_lab") -> FinancialSanityReport:
    checks: list[FinancialSanityCheck] = []
    portfolio = payload.get("portfolio_risk")
    if isinstance(portfolio, Mapping) and portfolio.get("accepted") is True:
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
    for outcome_index, outcome in enumerate(_accepted_outcomes(payload)):
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
        if not isinstance(scores, Mapping) or not scores:
            checks.append(_check("block", "objective scores", "missing accepted objective scores", path=f"{prefix}.objective_scores"))
        else:
            for objective, value in scores.items():
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
        coverage = outcome.get("data_coverage")
        if isinstance(coverage, Mapping):
            if coverage.get("integrity_status") == "fail":
                checks.append(_check("block", "data coverage", "coverage integrity failed", path=f"{prefix}.data_coverage"))
            checks.append(
                _range_check(
                    coverage.get("coverage_ratio"),
                    path=f"{prefix}.data_coverage.coverage_ratio",
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
                    path=f"{prefix}.data_coverage.gap_count",
                    metric=gap_count if gap_count is not None else "missing",
                    limit=0,
                )
            )
        stress = outcome.get("stress_validation")
        robustness = outcome.get("robustness_validation")
        for field_name, field_value in (("stress_validation", stress), ("robustness_validation", robustness)):
            if not isinstance(field_value, Mapping):
                checks.append(_check("block", field_name, "missing accepted validation report", path=f"{prefix}.{field_name}"))
                continue
            if field_value.get("accepted") is not True:
                checks.append(_check("block", field_name, "accepted outcome has failed validation", path=f"{prefix}.{field_name}.accepted"))
            if field_name == "robustness_validation" and field_value.get("statistical_edge_accepted") is False:
                checks.append(_check("block", field_name, "accepted outcome failed statistical-edge evidence", path=f"{prefix}.{field_name}.statistical_edge_accepted"))
            checks.extend(_market_edge_checks(field_value, path=f"{prefix}.{field_name}"))
            drawdown_key = "worst_max_drawdown"
            if drawdown_key in field_value:
                checks.append(
                    _range_check(
                        field_value.get(drawdown_key),
                        path=f"{prefix}.{field_name}.{drawdown_key}",
                        label=drawdown_key,
                        low=0.0,
                        high=1.0,
                        hard_low=0.0,
                        hard_high=1.0,
                    )
                )
        ai_uplift = outcome.get("ai_uplift")
        if isinstance(ai_uplift, Mapping):
            deltas = ai_uplift.get("deltas")
            if isinstance(deltas, Mapping):
                for key, value in deltas.items():
                    parsed = _finite(value)
                    if parsed is None:
                        checks.append(_check("block", "AI uplift delta", "non-finite delta", path=f"{prefix}.ai_uplift.deltas.{key}"))
    return FinancialSanityReport(tuple(checks), source=source)


def blocking_reasons(report: FinancialSanityReport) -> list[str]:
    return [
        f"{check.path or check.label}: {check.detail}"
        for check in report.checks
        if check.status == "block"
    ]

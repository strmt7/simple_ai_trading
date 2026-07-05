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


def build_backtest_financial_sanity_report(result: object, *, source: str = "backtest") -> FinancialSanityReport:
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
    closed_trades = _finite(getattr(result, "closed_trades", None))
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
        ("closed_trades", closed_trades, 0.0, None),
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

    closed_count = int(closed_trades) if closed_trades is not None and closed_trades >= 0 else None
    if closed_count is not None and closed_trades is not None and abs(closed_trades - closed_count) > 1e-9:
        checks.append(_check("block", "closed trade count", "closed_trades is not an integer count", path="closed_trades", metric=closed_trades))
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
    for index, trade in enumerate(trade_log):
        if not isinstance(trade, Mapping):
            checks.append(_check("block", "trade log entry", "trade entry is not a mapping", path=f"trade_log[{index}]"))
            continue
        reason = str(trade.get("exit_reason") or "").strip()
        checks.append(
            _check(
                "ok" if reason else "block",
                "trade exit reason",
                reason or "missing exit reason",
                path=f"trade_log[{index}].exit_reason",
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
    return FinancialSanityReport(tuple(checks), source=source)


def blocking_reasons(report: FinancialSanityReport) -> list[str]:
    return [
        f"{check.path or check.label}: {check.detail}"
        for check in report.checks
        if check.status == "block"
    ]

"""Frozen policy calibration for fill-aware make/take action values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Sequence

import numpy as np

from .make_take_action_values import MakeTakeActionValueBatch
from .make_take_predictive_evaluation import (
    MakeTakePredictiveEvaluation,
    validate_make_take_predictive_evaluation,
)
from .make_take_replay import (
    MakeTakeFixedLedger,
    MakeTakeReplayMetrics,
    build_make_take_fixed_ledger,
    replay_make_take_fixed_ledger,
    validate_make_take_fixed_ledger,
)
from .make_take_targets import MakeTakeTargetBatch


MAKE_TAKE_POLICY_SELECTION_SCHEMA_VERSION = "make-take-policy-selection-v1"


@dataclass(frozen=True)
class MakeTakePolicySpec:
    coverage_quantiles: tuple[float, ...] = (0.0, 0.50, 0.65, 0.80)
    conditional_q20_floor_bps: float = 0.0
    minimum_calibration_closed_trades: int = 4
    minimum_positive_symbols: int = 2
    maximum_drawdown_bps: float = 100.0
    maximum_single_symbol_positive_pnl_share: float = 0.70
    minimum_profit_factor: float = 1.0
    drawdown_penalty: float = 1.0

    def __post_init__(self) -> None:
        numeric = (
            *self.coverage_quantiles,
            self.conditional_q20_floor_bps,
            self.maximum_drawdown_bps,
            self.maximum_single_symbol_positive_pnl_share,
            self.minimum_profit_factor,
            self.drawdown_penalty,
        )
        if (
            not self.coverage_quantiles
            or tuple(sorted(set(self.coverage_quantiles))) != self.coverage_quantiles
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in numeric
            )
            or any(not 0.0 <= value < 1.0 for value in self.coverage_quantiles)
            or self.conditional_q20_floor_bps != 0.0
            or isinstance(self.minimum_calibration_closed_trades, bool)
            or not isinstance(self.minimum_calibration_closed_trades, int)
            or self.minimum_calibration_closed_trades < 4
            or isinstance(self.minimum_positive_symbols, bool)
            or not isinstance(self.minimum_positive_symbols, int)
            or not 2 <= self.minimum_positive_symbols <= 3
            or not 0.0 < self.maximum_drawdown_bps <= 100.0
            or not 0.0 < self.maximum_single_symbol_positive_pnl_share <= 0.70
            or self.minimum_profit_factor < 1.0
            or self.drawdown_penalty < 0.0
        ):
            raise ValueError("make/take policy specification is invalid")


@dataclass(frozen=True)
class ActionValueQuintileDiagnostic:
    rows: int
    lower_threshold_bps: float
    upper_threshold_bps: float
    base_bottom_mean_net_bps: float
    base_top_mean_net_bps: float
    stress_bottom_mean_net_bps: float
    stress_top_mean_net_bps: float
    passed: bool


@dataclass(frozen=True)
class MakeTakePolicyCandidate:
    coverage_quantile: float
    expected_mean_threshold_bps: float
    ledger_sha256: str
    base_metrics: MakeTakeReplayMetrics
    stress_metrics: MakeTakeReplayMetrics
    utility_bps: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MakeTakePolicySelection:
    schema_version: str
    spec: MakeTakePolicySpec
    predictive_evaluation_sha256: str
    expected_days: tuple[int, int]
    action_value_quintile: ActionValueQuintileDiagnostic
    candidates: tuple[MakeTakePolicyCandidate, ...]
    accepted: bool
    selected_coverage_quantile: float | None
    selected_expected_mean_threshold_bps: float | None
    selected_ledger: MakeTakeFixedLedger | None
    rejection_reasons: tuple[str, ...]
    selection_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def evidence(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _selection_payload(selection: MakeTakePolicySelection) -> dict[str, object]:
    payload = asdict(selection)
    payload.pop("selection_sha256")
    return payload


def _profit_factor_passed(metrics: MakeTakeReplayMetrics, minimum: float) -> bool:
    if metrics.profit_factor is None:
        return metrics.closed_trades > 0 and metrics.total_net_bps > 0.0
    return metrics.profit_factor > minimum


def _metric_reasons(
    base: MakeTakeReplayMetrics,
    stress: MakeTakeReplayMetrics,
    *,
    expected_days: tuple[int, int],
    spec: MakeTakePolicySpec,
) -> list[str]:
    reasons: list[str] = []
    for metrics in (base, stress):
        prefix = metrics.scenario
        if metrics.closed_trades < spec.minimum_calibration_closed_trades:
            reasons.append(f"{prefix}_minimum_closed_trades_not_met")
        if metrics.total_net_bps <= 0.0:
            reasons.append(f"{prefix}_total_net_bps_not_positive")
        if not _profit_factor_passed(metrics, spec.minimum_profit_factor):
            reasons.append(f"{prefix}_profit_factor_not_above_one")
        if metrics.maximum_drawdown_bps > spec.maximum_drawdown_bps:
            reasons.append(f"{prefix}_drawdown_limit_exceeded")
        if metrics.positive_symbols < spec.minimum_positive_symbols:
            reasons.append(f"{prefix}_positive_symbol_support_insufficient")
        if (
            metrics.maximum_single_symbol_positive_pnl_share
            > spec.maximum_single_symbol_positive_pnl_share
        ):
            reasons.append(f"{prefix}_positive_pnl_concentration_exceeded")
        daily = dict(metrics.daily_net_bps)
        if tuple(sorted(daily)) != expected_days or any(
            daily[day] <= 0.0 for day in expected_days
        ):
            reasons.append(f"{prefix}_constituent_day_not_positive")
    return reasons


def _quintile_diagnostic(ledger: MakeTakeFixedLedger) -> ActionValueQuintileDiagnostic:
    scores = np.asarray(
        [order.expected_mean_bps for order in ledger.orders], dtype=np.float64
    )
    if scores.size < 5:
        return ActionValueQuintileDiagnostic(
            rows=int(scores.size),
            lower_threshold_bps=0.0,
            upper_threshold_bps=0.0,
            base_bottom_mean_net_bps=0.0,
            base_top_mean_net_bps=0.0,
            stress_bottom_mean_net_bps=0.0,
            stress_top_mean_net_bps=0.0,
            passed=False,
        )
    lower = float(np.quantile(scores, 0.20, method="lower"))
    upper = float(np.quantile(scores, 0.80, method="higher"))
    bottom = scores <= lower
    top = scores >= upper
    base = np.asarray(
        [order.base_realized_net_bps for order in ledger.orders], dtype=np.float64
    )
    stress = np.asarray(
        [order.stress_realized_net_bps for order in ledger.orders], dtype=np.float64
    )
    base_bottom = float(np.mean(base[bottom]))
    base_top = float(np.mean(base[top]))
    stress_bottom = float(np.mean(stress[bottom]))
    stress_top = float(np.mean(stress[top]))
    passed = bool(
        lower < upper
        and base_top > 0.0
        and stress_top > 0.0
        and base_top > base_bottom
        and stress_top > stress_bottom
    )
    return ActionValueQuintileDiagnostic(
        rows=int(scores.size),
        lower_threshold_bps=lower,
        upper_threshold_bps=upper,
        base_bottom_mean_net_bps=base_bottom,
        base_top_mean_net_bps=base_top,
        stress_bottom_mean_net_bps=stress_bottom,
        stress_top_mean_net_bps=stress_top,
        passed=passed,
    )


def validate_make_take_policy_selection(selection: MakeTakePolicySelection) -> None:
    selection.spec.__post_init__()
    if selection.selected_ledger is not None:
        validate_make_take_fixed_ledger(selection.selected_ledger)
    selected_candidates = tuple(candidate for candidate in selection.candidates if candidate.accepted)
    if (
        selection.schema_version != MAKE_TAKE_POLICY_SELECTION_SCHEMA_VERSION
        or not _is_sha256(selection.predictive_evaluation_sha256)
        or len(selection.expected_days) != 2
        or tuple(sorted(set(selection.expected_days))) != selection.expected_days
        or any(
            not math.isfinite(value)
            for value in (
                selection.action_value_quintile.lower_threshold_bps,
                selection.action_value_quintile.upper_threshold_bps,
                selection.action_value_quintile.base_bottom_mean_net_bps,
                selection.action_value_quintile.base_top_mean_net_bps,
                selection.action_value_quintile.stress_bottom_mean_net_bps,
                selection.action_value_quintile.stress_top_mean_net_bps,
            )
        )
        or tuple(candidate.coverage_quantile for candidate in selection.candidates)
        != selection.spec.coverage_quantiles
        or any(
            not math.isfinite(candidate.expected_mean_threshold_bps)
            or candidate.expected_mean_threshold_bps < 0.0
            or not _is_sha256(candidate.ledger_sha256)
            or not math.isfinite(candidate.utility_bps)
            or candidate.accepted is not (not candidate.rejection_reasons)
            for candidate in selection.candidates
        )
        or selection.accepted is not bool(selected_candidates)
        or (
            selection.accepted
            and (
                selection.selected_ledger is None
                or selection.selected_coverage_quantile is None
                or selection.selected_expected_mean_threshold_bps is None
                or not any(
                    candidate.accepted
                    and candidate.coverage_quantile
                    == selection.selected_coverage_quantile
                    and candidate.expected_mean_threshold_bps
                    == selection.selected_expected_mean_threshold_bps
                    and candidate.ledger_sha256 == selection.selected_ledger.ledger_sha256
                    for candidate in selection.candidates
                )
            )
        )
        or (
            not selection.accepted
            and (
                selection.selected_ledger is not None
                or selection.selected_coverage_quantile is not None
                or selection.selected_expected_mean_threshold_bps is not None
                or not selection.rejection_reasons
            )
        )
        or selection.trading_authority is not False
        or selection.execution_claim is not False
        or selection.profitability_claim is not False
        or selection.portfolio_claim is not False
        or selection.leverage_applied is not False
        or not _is_sha256(selection.selection_sha256)
        or selection.selection_sha256 != _sha256(_selection_payload(selection))
    ):
        raise ValueError("make/take policy selection is invalid")


def calibrate_make_take_policy(
    *,
    predictive_evaluation: MakeTakePredictiveEvaluation,
    action_values: Sequence[MakeTakeActionValueBatch],
    base_targets: Sequence[MakeTakeTargetBatch],
    stress_targets: Sequence[MakeTakeTargetBatch],
    expected_days: Sequence[int],
    spec: MakeTakePolicySpec = MakeTakePolicySpec(),
) -> MakeTakePolicySelection:
    """Choose one coverage only on two-day, paired base/stress evidence."""

    validate_make_take_predictive_evaluation(predictive_evaluation)
    days = tuple(int(value) for value in expected_days)
    if (
        predictive_evaluation.role != "policy_calibration"
        or not predictive_evaluation.predictive_gate_passed
        or len(days) != 2
        or tuple(sorted(set(days))) != days
    ):
        raise ValueError("make/take policy calibration evidence is ineligible")
    broad = build_make_take_fixed_ledger(
        action_values=action_values,
        base_targets=base_targets,
        stress_targets=stress_targets,
        expected_mean_threshold_bps=0.0,
        conditional_q20_floor_bps=spec.conditional_q20_floor_bps,
    )
    if (
        broad.fill_model_sha256 != predictive_evaluation.fill_model_sha256
        or broad.payoff_model_sha256 != predictive_evaluation.payoff_model_sha256
    ):
        raise ValueError("make/take policy model identities drifted")
    quintile = _quintile_diagnostic(broad)
    candidates: list[MakeTakePolicyCandidate] = []
    ranked: list[
        tuple[
            tuple[float, float, float, int],
            float,
            float,
            MakeTakeFixedLedger,
        ]
    ] = []
    scores = np.asarray(
        [order.expected_mean_bps for order in broad.orders], dtype=np.float64
    )
    if scores.size:
        for quantile in spec.coverage_quantiles:
            threshold = float(np.quantile(scores, quantile, method="higher"))
            if quantile == 0.0:
                threshold = max(0.0, float(np.nextafter(threshold, -np.inf)))
            ledger = build_make_take_fixed_ledger(
                action_values=action_values,
                base_targets=base_targets,
                stress_targets=stress_targets,
                expected_mean_threshold_bps=threshold,
                conditional_q20_floor_bps=spec.conditional_q20_floor_bps,
            )
            base_metrics = replay_make_take_fixed_ledger(
                ledger,
                scenario="base",
                expected_days=days,
            )
            stress_metrics = replay_make_take_fixed_ledger(
                ledger,
                scenario="stress",
                expected_days=days,
            )
            reasons = _metric_reasons(
                base_metrics,
                stress_metrics,
                expected_days=days,
                spec=spec,
            )
            if not quintile.passed:
                reasons.append("action_value_quintile_gate_failed")
            utility = float(
                stress_metrics.total_net_bps
                - spec.drawdown_penalty * stress_metrics.maximum_drawdown_bps
            )
            candidate = MakeTakePolicyCandidate(
                coverage_quantile=quantile,
                expected_mean_threshold_bps=threshold,
                ledger_sha256=ledger.ledger_sha256,
                base_metrics=base_metrics,
                stress_metrics=stress_metrics,
                utility_bps=utility,
                accepted=not reasons,
                rejection_reasons=tuple(reasons),
            )
            candidates.append(candidate)
            if candidate.accepted:
                ranked.append(
                    (
                        (
                            utility,
                            stress_metrics.total_net_bps,
                            base_metrics.total_net_bps,
                            stress_metrics.closed_trades,
                        ),
                        quantile,
                        threshold,
                        ledger,
                    )
                )
    accepted = bool(ranked)
    if accepted:
        _rank, selected_quantile, selected_threshold, selected_ledger = max(
            ranked, key=lambda value: value[0]
        )
        rejection_reasons: tuple[str, ...] = ()
    else:
        selected_quantile = None
        selected_threshold = None
        selected_ledger = None
        rejection_reasons = (
            "policy_calibration_has_no_candidate"
            if scores.size
            else "policy_calibration_has_no_eligible_orders",
        )
    provisional = MakeTakePolicySelection(
        schema_version=MAKE_TAKE_POLICY_SELECTION_SCHEMA_VERSION,
        spec=spec,
        predictive_evaluation_sha256=predictive_evaluation.report_sha256,
        expected_days=(days[0], days[1]),
        action_value_quintile=quintile,
        candidates=tuple(candidates),
        accepted=accepted,
        selected_coverage_quantile=selected_quantile,
        selected_expected_mean_threshold_bps=selected_threshold,
        selected_ledger=selected_ledger,
        rejection_reasons=rejection_reasons,
        selection_sha256="",
    )
    selection = MakeTakePolicySelection(
        **{
            **provisional.__dict__,
            "selection_sha256": _sha256(_selection_payload(provisional)),
        }
    )
    validate_make_take_policy_selection(selection)
    return selection


__all__ = [
    "MAKE_TAKE_POLICY_SELECTION_SCHEMA_VERSION",
    "ActionValueQuintileDiagnostic",
    "MakeTakePolicyCandidate",
    "MakeTakePolicySelection",
    "MakeTakePolicySpec",
    "calibrate_make_take_policy",
    "validate_make_take_policy_selection",
]

"""Terminal, fixed-policy economic screen for Round 57 make/take research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

from .make_take_action_values import MakeTakeActionValueBatch
from .make_take_policy import (
    MakeTakePolicySelection,
    validate_make_take_policy_selection,
)
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


MAKE_TAKE_ECONOMIC_EVALUATION_SCHEMA_VERSION = "make-take-economic-evaluation-v1"


@dataclass(frozen=True)
class MakeTakeEconomicGateSpec:
    minimum_closed_trades: int = 30
    minimum_positive_symbols: int = 2
    maximum_drawdown_bps: float = 100.0
    maximum_single_symbol_positive_pnl_share: float = 0.70
    minimum_profit_factor: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.minimum_closed_trades != 30
            or self.minimum_positive_symbols != 2
            or self.maximum_drawdown_bps != 100.0
            or self.maximum_single_symbol_positive_pnl_share != 0.70
            or self.minimum_profit_factor != 1.0
        ):
            raise ValueError("make/take economic gate specification is frozen")


@dataclass(frozen=True)
class MakeTakeEconomicEvaluation:
    schema_version: str
    gate_spec: MakeTakeEconomicGateSpec
    policy_selection_sha256: str
    predictive_evaluation_sha256: str
    expected_days: tuple[int, ...]
    evaluation_ledger: MakeTakeFixedLedger
    base_metrics: MakeTakeReplayMetrics
    stress_metrics: MakeTakeReplayMetrics
    economic_gate_passed: bool
    rejection_reasons: tuple[str, ...]
    evaluation_sha256: str
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


def _evaluation_payload(evaluation: MakeTakeEconomicEvaluation) -> dict[str, object]:
    payload = asdict(evaluation)
    payload.pop("evaluation_sha256")
    return payload


def _profit_factor_passed(metrics: MakeTakeReplayMetrics, minimum: float) -> bool:
    if metrics.profit_factor is None:
        return metrics.closed_trades > 0 and metrics.total_net_bps > 0.0
    return metrics.profit_factor > minimum


def _economic_reasons(
    base: MakeTakeReplayMetrics,
    stress: MakeTakeReplayMetrics,
    spec: MakeTakeEconomicGateSpec,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for metrics in (base, stress):
        prefix = metrics.scenario
        if metrics.closed_trades < spec.minimum_closed_trades:
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
    return tuple(reasons)


def validate_make_take_economic_evaluation(
    evaluation: MakeTakeEconomicEvaluation,
) -> None:
    evaluation.gate_spec.__post_init__()
    validate_make_take_fixed_ledger(evaluation.evaluation_ledger)
    days = tuple(evaluation.expected_days)
    if len(days) != 6 or tuple(sorted(set(days))) != days:
        raise ValueError("make/take economic evaluation days are invalid")
    base = replay_make_take_fixed_ledger(
        evaluation.evaluation_ledger,
        scenario="base",
        expected_days=days,
    )
    stress = replay_make_take_fixed_ledger(
        evaluation.evaluation_ledger,
        scenario="stress",
        expected_days=days,
    )
    reasons = _economic_reasons(base, stress, evaluation.gate_spec)
    if (
        evaluation.schema_version != MAKE_TAKE_ECONOMIC_EVALUATION_SCHEMA_VERSION
        or not _is_sha256(evaluation.policy_selection_sha256)
        or not _is_sha256(evaluation.predictive_evaluation_sha256)
        or evaluation.base_metrics != base
        or evaluation.stress_metrics != stress
        or evaluation.economic_gate_passed is not (not reasons)
        or evaluation.rejection_reasons != reasons
        or evaluation.trading_authority is not False
        or evaluation.execution_claim is not False
        or evaluation.profitability_claim is not False
        or evaluation.portfolio_claim is not False
        or evaluation.leverage_applied is not False
        or not _is_sha256(evaluation.evaluation_sha256)
        or evaluation.evaluation_sha256 != _sha256(_evaluation_payload(evaluation))
    ):
        raise ValueError("make/take economic evaluation is invalid")


def evaluate_make_take_policy(
    *,
    policy_selection: MakeTakePolicySelection,
    predictive_evaluation: MakeTakePredictiveEvaluation,
    action_values: Sequence[MakeTakeActionValueBatch],
    base_targets: Sequence[MakeTakeTargetBatch],
    stress_targets: Sequence[MakeTakeTargetBatch],
    expected_days: Sequence[int],
    gate_spec: MakeTakeEconomicGateSpec = MakeTakeEconomicGateSpec(),
) -> MakeTakeEconomicEvaluation:
    """Apply the unchanged calibration policy to the six-day evaluation role."""

    validate_make_take_policy_selection(policy_selection)
    validate_make_take_predictive_evaluation(predictive_evaluation)
    days = tuple(int(value) for value in expected_days)
    if (
        not policy_selection.accepted
        or policy_selection.selected_ledger is None
        or policy_selection.selected_expected_mean_threshold_bps is None
        or predictive_evaluation.role != "evaluation"
        or not predictive_evaluation.predictive_gate_passed
        or len(days) != 6
        or tuple(sorted(set(days))) != days
        or policy_selection.selected_ledger.fill_model_sha256
        != predictive_evaluation.fill_model_sha256
        or policy_selection.selected_ledger.payoff_model_sha256
        != predictive_evaluation.payoff_model_sha256
    ):
        raise ValueError("make/take terminal evaluation evidence is ineligible")
    ledger = build_make_take_fixed_ledger(
        action_values=action_values,
        base_targets=base_targets,
        stress_targets=stress_targets,
        expected_mean_threshold_bps=(
            policy_selection.selected_expected_mean_threshold_bps
        ),
        conditional_q20_floor_bps=policy_selection.spec.conditional_q20_floor_bps,
    )
    if (
        ledger.fill_model_sha256 != predictive_evaluation.fill_model_sha256
        or ledger.payoff_model_sha256 != predictive_evaluation.payoff_model_sha256
    ):
        raise ValueError("make/take evaluation model identities drifted")
    base = replay_make_take_fixed_ledger(
        ledger,
        scenario="base",
        expected_days=days,
    )
    stress = replay_make_take_fixed_ledger(
        ledger,
        scenario="stress",
        expected_days=days,
    )
    reasons = _economic_reasons(base, stress, gate_spec)
    provisional = MakeTakeEconomicEvaluation(
        schema_version=MAKE_TAKE_ECONOMIC_EVALUATION_SCHEMA_VERSION,
        gate_spec=gate_spec,
        policy_selection_sha256=policy_selection.selection_sha256,
        predictive_evaluation_sha256=predictive_evaluation.report_sha256,
        expected_days=days,
        evaluation_ledger=ledger,
        base_metrics=base,
        stress_metrics=stress,
        economic_gate_passed=not reasons,
        rejection_reasons=reasons,
        evaluation_sha256="",
    )
    evaluation = MakeTakeEconomicEvaluation(
        **{
            **provisional.__dict__,
            "evaluation_sha256": _sha256(_evaluation_payload(provisional)),
        }
    )
    validate_make_take_economic_evaluation(evaluation)
    return evaluation


__all__ = [
    "MAKE_TAKE_ECONOMIC_EVALUATION_SCHEMA_VERSION",
    "MakeTakeEconomicEvaluation",
    "MakeTakeEconomicGateSpec",
    "evaluate_make_take_policy",
    "validate_make_take_economic_evaluation",
]

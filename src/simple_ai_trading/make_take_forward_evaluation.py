"""Forward-use chronology guard around the preserved Round 57 evaluator.

This proves separation of supplied role batches and their recorded labels, not
raw-stream continuity, model-training independence or partial-fill economics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from numbers import Integral

import numpy as np

from .make_take_action_values import MakeTakeActionValueBatch
from .make_take_evaluation import MakeTakeEconomicEvaluation, evaluate_make_take_policy
from .make_take_policy import (
    MakeTakePolicySelection,
    validate_make_take_policy_selection,
)
from .make_take_predictive_evaluation import MakeTakePredictiveEvaluation
from .make_take_replay import _ordered_targets, _ordered_values
from .make_take_targets import MakeTakeTargetBatch

DAY_MS = 86_400_000


@dataclass(frozen=True)
class MakeTakeRoleInputs:
    days: tuple[int, ...]
    action_values: tuple[MakeTakeActionValueBatch, ...]
    base_targets: tuple[MakeTakeTargetBatch, ...]
    stress_targets: tuple[MakeTakeTargetBatch, ...]


@dataclass(frozen=True)
class ForwardMakeTakeEvaluation:
    calibration_days: tuple[int, ...]
    evaluation_days: tuple[int, ...]
    calibration_last_recorded_label_ms: int
    first_evaluation_decision_ms: int
    calibration_selection_sha256: str
    evaluation: MakeTakeEconomicEvaluation
    schema_version: str = field(
        default="causal-forward-make-take-evaluation-v1", init=False
    )
    qualified_edge: bool = field(default=False, init=False)

    def evidence(self) -> dict[str, object]:
        value = asdict(self)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return value | {
            "result_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest()
        }


def _role(role: MakeTakeRoleInputs, count: int) -> MakeTakeRoleInputs:
    days = role.days
    if (
        not isinstance(days, tuple)
        or len(days) != count
        or any(
            isinstance(d, bool) or not isinstance(d, Integral) or d < 0 for d in days
        )
        or tuple(int(d) for d in days)
        != tuple(range(int(days[0]), int(days[0]) + count))
    ):
        raise ValueError("exact consecutive integer role days required")
    days = tuple(int(d) for d in days)
    values = _ordered_values(role.action_values)
    base = _ordered_targets(role.base_targets, scenario="base")
    stress = _ordered_targets(role.stress_targets, scenario="stress")
    expected_paths = {str(day * DAY_MS) for day in days}
    for value, first, second in zip(values, base, stress, strict=True):
        times = np.asarray(value.decision_time_ms)
        if times.dtype.kind not in "iu" or not set((times // DAY_MS).tolist()) <= set(
            days
        ):
            raise ValueError("decision instants lie outside declared role days")
        for target in (first, second):
            if (
                target.source_dataset_sha256 != value.source_dataset_sha256
                or target.action_rows != value.rows
                or set(target.day_path_sha256) != expected_paths
                or np.asarray(target.terminal_time_ms).dtype.kind not in "iu"
                or not np.any(target.realized_valid)
            ):
                raise ValueError(
                    "role source identity or full day-path coverage is missing"
                )
    return MakeTakeRoleInputs(days, values, base, stress)


def evaluate_make_take_policy_forward(
    *,
    policy_selection: MakeTakePolicySelection,
    calibration: MakeTakeRoleInputs,
    evaluation: MakeTakeRoleInputs,
    predictive_evaluation: MakeTakePredictiveEvaluation,
) -> ForwardMakeTakeEvaluation:
    """Reject overlapping roles, missing day paths and labels reaching evaluation."""
    validate_make_take_policy_selection(policy_selection)
    if not policy_selection.accepted or policy_selection.selected_ledger is None:
        raise ValueError("accepted calibration selection required")
    calibration = _role(calibration, 2)
    evaluation = _role(evaluation, 6)
    ledger = policy_selection.selected_ledger
    if (
        calibration.days != policy_selection.expected_days
        or evaluation.days[0] <= calibration.days[-1]
        or tuple((x.symbol, x.batch_sha256) for x in calibration.action_values)
        != ledger.source_action_value_sha256_by_symbol
        or tuple((x.symbol, x.target_sha256) for x in calibration.base_targets)
        != ledger.base_target_sha256_by_symbol
        or tuple((x.symbol, x.target_sha256) for x in calibration.stress_targets)
        != ledger.stress_target_sha256_by_symbol
    ):
        raise ValueError(
            "calibration source binding or chronological role separation failed"
        )
    last_label = max(
        int(np.max(x.terminal_time_ms[x.realized_valid]))
        for x in (*calibration.base_targets, *calibration.stress_targets)
    )
    first_decision = min(
        int(x.decision_time_ms.min()) for x in evaluation.action_values
    )
    if last_label >= first_decision:
        raise ValueError("calibration label reaches evaluation decisions")
    result = evaluate_make_take_policy(
        policy_selection=policy_selection,
        predictive_evaluation=predictive_evaluation,
        action_values=evaluation.action_values,
        base_targets=evaluation.base_targets,
        stress_targets=evaluation.stress_targets,
        expected_days=evaluation.days,
    )
    return ForwardMakeTakeEvaluation(
        calibration.days,
        evaluation.days,
        last_label,
        first_decision,
        policy_selection.selection_sha256,
        result,
    )

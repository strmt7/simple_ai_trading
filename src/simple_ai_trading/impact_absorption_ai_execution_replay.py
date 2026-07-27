"""Exact delayed-entry L2 replay for paired Round 74 AI reviews."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Sequence

from .impact_absorption_ai_runtime import ROUND74_AI_RUNTIME_STATUSES
from .impact_absorption_ai_uplift import (
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
)
from .impact_absorption_event_action_policy import (
    Round74ActionInferenceContext,
    Round74ActionPolicySelection,
    Round74ActionTrace,
)
from .impact_absorption_event_dataset import Round74EventRunPartition
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_SIDES,
    Round74ReplayObservation,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS,
    Round74EventExecutionOverride,
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
    Round74EventTargetOutcome,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly


ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION = (
    "round-074-ai-execution-replay-plan-v1"
)
ROUND74_AI_EXECUTION_PRE_REPLAY_STATUSES = frozenset(
    {
        "replay_required",
        "runtime_veto",
        "ai_veto",
        "historical_review_expired",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 AI execution {label} digest differs")
    return selected


def _selected_trace(
    action_selection: Round74ActionPolicySelection,
) -> Round74ActionTrace:
    action_selection.validate()
    selected = [
        value
        for value in action_selection.evaluations
        if value.accepted
        and value.quantile == action_selection.selected_quantile
        and value.threshold_score == action_selection.selected_threshold_score
    ]
    if not action_selection.accepted or len(selected) != 1:
        raise ValueError("Round 74 AI execution lacks an accepted action policy")
    return selected[0].trace


@dataclass(frozen=True)
class Round74AIExecutionReplayInstruction:
    """One target-free AI decision scheduled for exact historical execution."""

    row_index: int
    run_id: str
    symbol: str
    anchor_index: int
    decision_monotonic_ns: int
    decision_wall_ns: int
    endpoint_frame_index: int
    endpoint_message_index: int
    sample_sha256: str
    feature_window_sha256: str
    feature_row_sha256: str
    side: int
    horizon_seconds: int
    source_review_sha256: str
    model_manifest_sha256: str
    runtime_status: str
    effective_review_latency_ns: int
    same_entry_latency_eligible: bool
    requested_size_multiplier_bps: int
    pre_replay_status: str
    partition_sha256: str
    action_selection_sha256: str
    schema_version: str = ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        digests = (
            self.sample_sha256,
            self.feature_window_sha256,
            self.feature_row_sha256,
            self.source_review_sha256,
            self.model_manifest_sha256,
            self.partition_sha256,
            self.action_selection_sha256,
        )
        if (
            self.schema_version != ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION
            or isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.symbol not in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
            or isinstance(self.anchor_index, bool)
            or not isinstance(self.anchor_index, int)
            or self.anchor_index < 0
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.decision_monotonic_ns,
                    self.endpoint_frame_index,
                    self.endpoint_message_index,
                )
            )
            or isinstance(self.decision_wall_ns, bool)
            or not isinstance(self.decision_wall_ns, int)
            or self.decision_wall_ns <= 0
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or self.side not in (-1, 1)
            or self.horizon_seconds not in (30, 300)
            or self.runtime_status not in ROUND74_AI_RUNTIME_STATUSES
            or isinstance(self.effective_review_latency_ns, bool)
            or not isinstance(self.effective_review_latency_ns, int)
            or self.effective_review_latency_ns < 0
            or not isinstance(self.same_entry_latency_eligible, bool)
            or isinstance(self.requested_size_multiplier_bps, bool)
            or not isinstance(self.requested_size_multiplier_bps, int)
            or not 0 <= self.requested_size_multiplier_bps <= 10_000
            or self.pre_replay_status not in ROUND74_AI_EXECUTION_PRE_REPLAY_STATUSES
        ):
            raise ValueError("Round 74 AI execution instruction differs")
        if self.pre_replay_status == "replay_required":
            if (
                self.runtime_status != "accepted"
                or self.requested_size_multiplier_bps <= 0
                or self.effective_review_latency_ns
                > ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
            ):
                raise ValueError("Round 74 AI replay-required instruction differs")
        elif self.pre_replay_status == "runtime_veto":
            if (
                self.runtime_status == "accepted"
                or self.requested_size_multiplier_bps != 0
            ):
                raise ValueError("Round 74 AI runtime-veto instruction differs")
        elif self.pre_replay_status == "ai_veto":
            if (
                self.runtime_status != "accepted"
                or self.requested_size_multiplier_bps != 0
            ):
                raise ValueError("Round 74 AI veto instruction differs")
        elif (
            self.runtime_status != "accepted"
            or self.requested_size_multiplier_bps <= 0
            or self.effective_review_latency_ns
            <= ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
        ):
            raise ValueError("Round 74 AI expired-review instruction differs")

    @property
    def instruction_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            **self.__dict__,
            "realized_target_accessed_during_instruction_build": False,
            "may_change_side_or_horizon": False,
            "baseline_fill_reused": False,
            "trading_authority": False,
        }
        if include_sha256:
            value["instruction_sha256"] = _canonical_sha256(value)
        return value


def build_round74_ai_execution_replay_instructions(
    action_selection: Round74ActionPolicySelection,
    *,
    contexts: Sequence[Round74ActionInferenceContext],
    reviews: Sequence[Round74AIPairedReviewEvidence],
) -> tuple[Round74AIExecutionReplayInstruction, ...]:
    """Bind selected actions to raw capture anchors without reading targets."""

    trace = _selected_trace(action_selection)
    selected_contexts = tuple(contexts)
    if not selected_contexts:
        raise ValueError("Round 74 AI execution context panel is empty")
    flattened: dict[int, tuple[Round74ActionInferenceContext, int]] = {}
    offset = 0
    partition_sha256: str | None = None
    for context in selected_contexts:
        context.validate()
        if partition_sha256 is None:
            partition_sha256 = context.partition_sha256
        elif context.partition_sha256 != partition_sha256:
            raise ValueError("Round 74 AI execution partition differs")
        for local_index in range(context.rows):
            flattened[offset + local_index] = (context, local_index)
        offset += context.rows
    assert partition_sha256 is not None
    review_rows = tuple(reviews)
    for review in review_rows:
        review.validate()
    if (
        len(review_rows) != trace.metrics.trades
        or tuple(review.row_index for review in review_rows) != trace.row_index
        or len({review.row_index for review in review_rows}) != len(review_rows)
    ):
        raise ValueError("Round 74 AI execution review coverage differs")
    instructions: list[Round74AIExecutionReplayInstruction] = []
    for trace_index, review in enumerate(review_rows):
        try:
            context, local_index = flattened[review.row_index]
        except KeyError as exc:
            raise ValueError("Round 74 AI execution row is absent") from exc
        if (
            context.feature_row_sha256[local_index]
            != trace.feature_row_sha256[trace_index]
            or review.feature_row_sha256 != trace.feature_row_sha256[trace_index]
            or context.run_id[local_index] != trace.run_id[trace_index]
            or review.run_id != trace.run_id[trace_index]
            or context.symbol[local_index] != trace.symbol[trace_index]
            or review.symbol != trace.symbol[trace_index]
            or review.side != trace.side[trace_index]
            or review.horizon_seconds != trace.horizon_seconds[trace_index]
            or review.pretest_policy_sha256 != action_selection.pretest_policy_sha256
            or review.probability_calibration_sha256
            != action_selection.probability_calibration_sha256
        ):
            raise ValueError("Round 74 AI execution action identity differs")
        if review.runtime_status != "accepted":
            status = "runtime_veto"
            multiplier = 0
        else:
            assert review.decision is not None
            multiplier = review.decision.size_multiplier_bps
            if multiplier == 0:
                status = "ai_veto"
            elif (
                review.effective_review_latency_ns
                > ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
            ):
                status = "historical_review_expired"
            else:
                status = "replay_required"
        instruction = Round74AIExecutionReplayInstruction(
            row_index=review.row_index,
            run_id=review.run_id,
            symbol=review.symbol,
            anchor_index=int(context.anchor_index[local_index]),
            decision_monotonic_ns=int(context.decision_monotonic_ns[local_index]),
            decision_wall_ns=int(context.decision_wall_ns[local_index]),
            endpoint_frame_index=int(context.endpoint_frame_index[local_index]),
            endpoint_message_index=int(context.endpoint_message_index[local_index]),
            sample_sha256=context.sample_sha256[local_index],
            feature_window_sha256=context.feature_window_sha256[local_index],
            feature_row_sha256=review.feature_row_sha256,
            side=review.side,
            horizon_seconds=review.horizon_seconds,
            source_review_sha256=review.review_sha256,
            model_manifest_sha256=review.model_manifest_sha256,
            runtime_status=review.runtime_status,
            effective_review_latency_ns=review.effective_review_latency_ns,
            same_entry_latency_eligible=review.same_entry_latency_eligible,
            requested_size_multiplier_bps=multiplier,
            pre_replay_status=status,
            partition_sha256=partition_sha256,
            action_selection_sha256=action_selection.selection_sha256,
        )
        instruction.validate()
        instructions.append(instruction)
    result = tuple(instructions)
    if len({value.instruction_sha256 for value in result}) != len(result):
        raise ValueError("Round 74 AI execution instructions are duplicated")
    return result


def _non_replay_evidence(
    instruction: Round74AIExecutionReplayInstruction,
    *,
    capture_report_sha256: str,
    target_spec_sha256: str,
) -> Round74AIExecutionReplayEvidence:
    status = instruction.pre_replay_status
    if status == "replay_required":
        raise ValueError("Round 74 AI required replay was not performed")
    result = Round74AIExecutionReplayEvidence(
        row_index=instruction.row_index,
        feature_row_sha256=instruction.feature_row_sha256,
        run_id=instruction.run_id,
        symbol=instruction.symbol,
        side=instruction.side,
        horizon_seconds=instruction.horizon_seconds,
        source_review_sha256=instruction.source_review_sha256,
        partition_sha256=instruction.partition_sha256,
        source_capture_report_sha256=capture_report_sha256,
        target_spec_sha256=target_spec_sha256,
        status=status,
        requested_size_multiplier_bps=(instruction.requested_size_multiplier_bps),
        applied_size_multiplier_bps=0,
        exact_l2_replay_performed=False,
        target_outcome_sha256=None,
        target_context_sha256=None,
        target_ineligible_reason="",
        requested_entry_monotonic_ns=None,
        actual_entry_monotonic_ns=None,
        actual_exit_monotonic_ns=None,
        capital_scaled_net_payoff_bps=0.0,
        capital_scaled_maximum_adverse_excursion_bps=0.0,
        adverse_selection=False,
    )
    result.validate()
    return result


def _outcome_sha256(outcome: Round74EventTargetOutcome) -> str:
    outcome.validate()
    return _canonical_sha256(outcome.as_dict())


def replay_round74_ai_execution_run(
    assembly: Round74SourceTargetAssembly,
    instructions: Sequence[Round74AIExecutionReplayInstruction],
    observations: Iterable[Round74ReplayObservation],
    *,
    capture_report_sha256: str,
) -> tuple[Round74AIExecutionReplayEvidence, ...]:
    """Replay one run once, using exact delayed L2 states for every AI action."""

    if not isinstance(assembly, Round74SourceTargetAssembly):
        raise TypeError("Round 74 AI execution assembly type differs")
    assembly.__post_init__()
    capture_report = _require_sha256(
        capture_report_sha256,
        "capture report",
    )
    rows = tuple(instructions)
    for row in rows:
        row.validate()
    if (
        not rows
        or len({row.row_index for row in rows}) != len(rows)
        or tuple(row.row_index for row in rows)
        != tuple(sorted(row.row_index for row in rows))
        or len({row.run_id for row in rows}) != 1
        or len({row.partition_sha256 for row in rows}) != 1
    ):
        raise ValueError("Round 74 AI execution run instruction panel differs")
    required = tuple(row for row in rows if row.pre_replay_status == "replay_required")
    target_spec_sha256 = assembly.spec.spec_sha256
    if not required:
        return tuple(
            _non_replay_evidence(
                row,
                capture_report_sha256=capture_report,
                target_spec_sha256=target_spec_sha256,
            )
            for row in rows
        )
    anchors = tuple(
        Round74EventTargetAnchor(
            symbol=row.symbol,
            anchor_index=row.anchor_index,
            decision_monotonic_ns=row.decision_monotonic_ns,
            decision_wall_ns=row.decision_wall_ns,
            endpoint_frame_index=row.endpoint_frame_index,
            endpoint_message_index=row.endpoint_message_index,
            feature_window_sha256=row.feature_window_sha256,
        )
        for row in required
    )
    overrides = tuple(
        Round74EventExecutionOverride(
            symbol=row.symbol,
            anchor_index=row.anchor_index,
            feature_window_sha256=row.feature_window_sha256,
            additional_entry_latency_ns=row.effective_review_latency_ns,
            quote_size_multiplier_bps=row.requested_size_multiplier_bps,
            source_review_sha256=row.source_review_sha256,
        )
        for row in required
    )
    engine = Round74EventTargetEngine(
        spec=assembly.spec,
        anchors=anchors,
        quantity_rules=assembly.quantity_rules_mapping(),
        execution_overrides=overrides,
    )
    for observation in observations:
        observation.validate()
        if (
            observation.depth_state is not None
            and not observation.depth_update_is_stale
        ):
            engine.observe_depth(
                received_monotonic_ns=observation.received_monotonic_ns,
                frame_index=observation.frame_index,
                message_index=observation.message_index,
                state=observation.depth_state,
            )
    outcomes = engine.finish()
    selected_outcomes = {
        (
            outcome.symbol,
            outcome.anchor_index,
            outcome.horizon_seconds,
            1 if outcome.side == ROUND74_EVENT_PAYOFF_SIDES[0] else -1,
        ): outcome
        for outcome in outcomes
    }
    required_keys = {
        (row.symbol, row.anchor_index, row.horizon_seconds, row.side)
        for row in required
    }
    if not required_keys.issubset(selected_outcomes):
        raise ValueError("Round 74 AI execution target coverage differs")
    result: list[Round74AIExecutionReplayEvidence] = []
    open_until: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.pre_replay_status != "replay_required":
            result.append(
                _non_replay_evidence(
                    row,
                    capture_report_sha256=capture_report,
                    target_spec_sha256=target_spec_sha256,
                )
            )
            continue
        outcome = selected_outcomes[
            (row.symbol, row.anchor_index, row.horizon_seconds, row.side)
        ]
        outcome_sha256 = _outcome_sha256(outcome)
        status = "target_ineligible"
        applied_multiplier = 0
        target_reason = outcome.ineligible_reason
        net_bps = 0.0
        adverse_excursion_bps = 0.0
        adverse_selection = False
        if outcome.eligible:
            assert outcome.actual_entry_monotonic_ns is not None
            assert outcome.actual_exit_monotonic_ns is not None
            key = (row.run_id, row.symbol)
            if outcome.actual_entry_monotonic_ns < open_until.get(key, -1):
                status = "delayed_overlap_veto"
                target_reason = ""
            else:
                status = "executed"
                target_reason = ""
                applied_multiplier = row.requested_size_multiplier_bps
                scale = applied_multiplier / 10_000.0
                assert outcome.net_payoff_bps is not None
                assert outcome.maximum_adverse_excursion_bps is not None
                assert outcome.adverse_selection is not None
                net_bps = float(outcome.net_payoff_bps) * scale
                adverse_excursion_bps = (
                    float(outcome.maximum_adverse_excursion_bps) * scale
                )
                adverse_selection = bool(outcome.adverse_selection)
                open_until[key] = outcome.actual_exit_monotonic_ns
        evidence = Round74AIExecutionReplayEvidence(
            row_index=row.row_index,
            feature_row_sha256=row.feature_row_sha256,
            run_id=row.run_id,
            symbol=row.symbol,
            side=row.side,
            horizon_seconds=row.horizon_seconds,
            source_review_sha256=row.source_review_sha256,
            partition_sha256=row.partition_sha256,
            source_capture_report_sha256=capture_report,
            target_spec_sha256=target_spec_sha256,
            status=status,
            requested_size_multiplier_bps=row.requested_size_multiplier_bps,
            applied_size_multiplier_bps=applied_multiplier,
            exact_l2_replay_performed=True,
            target_outcome_sha256=outcome_sha256,
            target_context_sha256=outcome.target_context_sha256,
            target_ineligible_reason=target_reason,
            requested_entry_monotonic_ns=(outcome.requested_entry_monotonic_ns),
            actual_entry_monotonic_ns=outcome.actual_entry_monotonic_ns,
            actual_exit_monotonic_ns=outcome.actual_exit_monotonic_ns,
            capital_scaled_net_payoff_bps=net_bps,
            capital_scaled_maximum_adverse_excursion_bps=(adverse_excursion_bps),
            adverse_selection=adverse_selection,
        )
        evidence.validate()
        result.append(evidence)
    replayed = tuple(result)
    if tuple(value.row_index for value in replayed) != tuple(
        row.row_index for row in rows
    ):
        raise ValueError("Round 74 AI execution replay order differs")
    return replayed


def replay_round74_ai_execution_store_run(
    store: object,
    *,
    partition: Round74EventRunPartition,
    run_id: str,
    assembly: Round74SourceTargetAssembly,
    instructions: Sequence[Round74AIExecutionReplayInstruction],
) -> tuple[Round74AIExecutionReplayEvidence, ...]:
    """Audit one persisted capture and replay it without duplicate storage."""

    from .impact_absorption_event_dataset import (
        validate_round74_capture_report_binding,
    )
    from .impact_absorption_event_sequence import (
        iter_round74_v10_event_observations,
    )
    from .impact_absorption_store import ImpactAbsorptionStore

    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 AI execution replay requires an event store")
    if not store.read_only:
        raise ValueError("Round 74 AI execution replay requires a read-only store")
    partition.validate()
    entry = partition.entry(str(run_id))
    report_row = (
        store.connect()
        .execute(
            """
            SELECT report_sha256
            FROM impact_capture_report
            WHERE run_id = ?
            """,
            [entry.run_id],
        )
        .fetchone()
    )
    if report_row is None:
        raise ValueError("Round 74 AI execution capture report is missing")
    validate_round74_capture_report_binding(
        entry,
        stored_capture_report_sha256=report_row[0],
    )
    rows = tuple(instructions)
    if any(
        row.run_id != entry.run_id or row.partition_sha256 != partition.partition_sha256
        for row in rows
    ):
        raise ValueError("Round 74 AI execution store identity differs")
    return replay_round74_ai_execution_run(
        assembly,
        rows,
        iter_round74_v10_event_observations(store, run_id=entry.run_id),
        capture_report_sha256=entry.capture_report_sha256,
    )


__all__ = [
    "ROUND74_AI_EXECUTION_PRE_REPLAY_STATUSES",
    "ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION",
    "Round74AIExecutionReplayInstruction",
    "build_round74_ai_execution_replay_instructions",
    "replay_round74_ai_execution_run",
    "replay_round74_ai_execution_store_run",
]

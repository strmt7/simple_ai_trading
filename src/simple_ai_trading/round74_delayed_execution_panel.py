"""Replay measured decision delay into exact Round 74 L2 policy economics.

Each capture stream is read once. Three independent target engines consume the
same observations in memory so conservative, regular, and aggressive policies
receive profile-specific delayed fills without repeated database scans.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionExecutionOutcomeRow,
    Round74ActionExecutionPanel,
    build_round74_action_inference_context,
)
from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventTrainingBatch,
    validate_round74_capture_report_binding,
)
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    Round74ReplayObservation,
    iter_round74_v10_event_observations,
)
from .impact_absorption_event_targets import (
    Round74EventExecutionOverride,
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .round74_online_decision_latency import (
    Round74OnlineDecisionLatencyEvidence,
)


ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION = (
    "round-074-delayed-execution-replay-v1"
)

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _module_sha256() -> str:
    payload = Path(__file__).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True)
class Round74DelayedExecutionRun:
    """One run's profile replay result before the six-run panel is assembled."""

    profile: str
    run_id: str
    partition_sha256: str
    target_batch_sha256: str
    target_assembly_sha256: str
    capture_report_sha256: str
    decision_latency_evidence_sha256: str
    additional_entry_latency_ns: int
    rows: tuple[Round74ActionExecutionOutcomeRow, ...]
    schema_version: str = ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION

    def validate(self) -> None:
        digests = (
            self.partition_sha256,
            self.target_batch_sha256,
            self.target_assembly_sha256,
            self.capture_report_sha256,
            self.decision_latency_evidence_sha256,
        )
        if (
            self.schema_version != ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION
            or self.profile not in ROUND74_ACTION_PROFILES
            or _RUN_ID.fullmatch(self.run_id) is None
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or isinstance(self.additional_entry_latency_ns, bool)
            or not isinstance(self.additional_entry_latency_ns, int)
            or self.additional_entry_latency_ns <= 0
            or not self.rows
        ):
            raise ValueError("Round 74 delayed execution run differs")
        for row in self.rows:
            row.validate()
        if (
            any(row.run_id != self.run_id for row in self.rows)
            or len({row.feature_row_sha256 for row in self.rows}) != len(self.rows)
        ):
            raise ValueError("Round 74 delayed execution run coverage differs")

    @property
    def replay_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "profile": self.profile,
                "run_id": self.run_id,
                "partition_sha256": self.partition_sha256,
                "target_batch_sha256": self.target_batch_sha256,
                "target_assembly_sha256": self.target_assembly_sha256,
                "capture_report_sha256": self.capture_report_sha256,
                "decision_latency_evidence_sha256": (
                    self.decision_latency_evidence_sha256
                ),
                "additional_entry_latency_ns": self.additional_entry_latency_ns,
                "row_sha256": [row.row_sha256 for row in self.rows],
            }
        )


def _profile_latency_ns(
    evidence: Round74OnlineDecisionLatencyEvidence,
) -> dict[str, int]:
    evidence.validate()
    return {
        row.profile: int(row.p99_upper_confidence_ns) for row in evidence.profiles
    }


def replay_round74_delayed_execution_run(
    batch: Round74EventTrainingBatch,
    assembly: Round74SourceTargetAssembly,
    latency_evidence: Round74OnlineDecisionLatencyEvidence,
    observations: Iterable[Round74ReplayObservation],
    *,
    capture_report_sha256: str,
) -> tuple[Round74DelayedExecutionRun, ...]:
    """Consume one audited observation stream once for all three profiles."""

    batch.validate()
    assembly.__post_init__()
    latency_evidence.validate()
    capture_report = str(capture_report_sha256)
    run_ids = set(batch.run_id)
    if (
        batch.role not in {"tuning", "test"}
        or len(run_ids) != 1
        or _SHA256.fullmatch(capture_report) is None
        or batch.scaler_sha256 != latency_evidence.scaler_sha256
    ):
        raise ValueError("Round 74 delayed execution run input differs")
    run_id = next(iter(run_ids))
    context = build_round74_action_inference_context(batch)
    latency_by_profile = _profile_latency_ns(latency_evidence)
    anchors = tuple(
        Round74EventTargetAnchor(
            symbol=batch.symbol[row_index],
            anchor_index=int(batch.anchor_index[row_index]),
            decision_monotonic_ns=int(batch.decision_monotonic_ns[row_index]),
            decision_wall_ns=int(batch.decision_wall_ns[row_index]),
            endpoint_frame_index=int(batch.endpoint_frame_index[row_index]),
            endpoint_message_index=int(batch.endpoint_message_index[row_index]),
            feature_window_sha256=batch.feature_window_sha256[row_index],
        )
        for row_index in range(batch.rows)
    )
    engines: dict[str, Round74EventTargetEngine] = {}
    for profile in ROUND74_ACTION_PROFILES:
        additional_latency = latency_by_profile[profile]
        source_sha256 = _canonical_sha256(
            {
                "schema_version": ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION,
                "decision_latency_evidence_sha256": latency_evidence.evidence_sha256,
                "profile": profile,
                "p99_upper_confidence_ns": additional_latency,
            }
        )
        overrides = tuple(
            Round74EventExecutionOverride(
                symbol=anchor.symbol,
                anchor_index=anchor.anchor_index,
                feature_window_sha256=anchor.feature_window_sha256,
                additional_entry_latency_ns=additional_latency,
                quote_size_multiplier_bps=10_000,
                source_review_sha256=source_sha256,
            )
            for anchor in anchors
        )
        engines[profile] = Round74EventTargetEngine(
            spec=assembly.spec,
            anchors=anchors,
            quantity_rules=assembly.quantity_rules_mapping(),
            execution_overrides=overrides,
        )
    for observation in observations:
        if not isinstance(observation, Round74ReplayObservation):
            raise TypeError("Round 74 delayed execution observation type differs")
        observation.validate()
        if (
            observation.depth_state is not None
            and not observation.depth_update_is_stale
        ):
            for engine in engines.values():
                engine.observe_depth(
                    received_monotonic_ns=observation.received_monotonic_ns,
                    frame_index=observation.frame_index,
                    message_index=observation.message_index,
                    state=observation.depth_state,
                )
    results: list[Round74DelayedExecutionRun] = []
    for profile in ROUND74_ACTION_PROFILES:
        outcomes = engines[profile].finish()
        by_key = {
            (
                outcome.symbol,
                outcome.anchor_index,
                outcome.horizon_seconds,
                outcome.side,
            ): outcome
            for outcome in outcomes
        }
        expected_count = (
            batch.rows
            * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES)
        )
        if (
            len(by_key) != expected_count
            or any(
                outcome.target_spec_sha256 != assembly.spec.spec_sha256
                for outcome in by_key.values()
            )
        ):
            raise ValueError("Round 74 delayed execution target coverage differs")
        rows = tuple(
            Round74ActionExecutionOutcomeRow(
                feature_row_sha256=context.feature_row_sha256[row_index],
                run_id=run_id,
                symbol=batch.symbol[row_index],
                anchor_index=int(batch.anchor_index[row_index]),
                feature_window_sha256=batch.feature_window_sha256[row_index],
                outcomes=tuple(
                    by_key[
                        (
                            batch.symbol[row_index],
                            int(batch.anchor_index[row_index]),
                            horizon,
                            side,
                        )
                    ]
                    for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
                    for side in ROUND74_EVENT_PAYOFF_SIDES
                ),
            )
            for row_index in range(batch.rows)
        )
        result = Round74DelayedExecutionRun(
            profile=profile,
            run_id=run_id,
            partition_sha256=batch.partition_sha256,
            target_batch_sha256=batch.batch_sha256,
            target_assembly_sha256=assembly.assembly_sha256,
            capture_report_sha256=capture_report,
            decision_latency_evidence_sha256=latency_evidence.evidence_sha256,
            additional_entry_latency_ns=latency_by_profile[profile],
            rows=rows,
        )
        result.validate()
        results.append(result)
    return tuple(results)


def build_round74_delayed_execution_panels(
    store: object,
    *,
    partition: Round74EventRunPartition,
    policy_selection_batches: Sequence[Round74EventTrainingBatch],
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    latency_evidence: Round74OnlineDecisionLatencyEvidence,
) -> tuple[Round74ActionExecutionPanel, ...]:
    """Read each policy-selection capture once and return three exact panels."""

    from .impact_absorption_store import ImpactAbsorptionStore

    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 delayed execution replay requires an event store")
    if not store.read_only:
        raise ValueError("Round 74 delayed execution replay requires a read-only store")
    partition.validate()
    latency_evidence.validate()
    batches = tuple(policy_selection_batches)
    assemblies = dict(target_assembly_by_run_id)
    run_ids = tuple(next(iter(set(batch.run_id))) for batch in batches)
    if (
        not batches
        or len(run_ids) != len(set(run_ids))
        or any(
            batch.role != "tuning"
            or len(set(batch.run_id)) != 1
            or batch.partition_sha256 != partition.partition_sha256
            for batch in batches
        )
        or set(assemblies) != set(run_ids)
    ):
        raise ValueError("Round 74 delayed execution panel input differs")
    runs_by_profile: dict[str, list[Round74DelayedExecutionRun]] = {
        profile: [] for profile in ROUND74_ACTION_PROFILES
    }
    reports: list[tuple[str, str]] = []
    connection = store.connect()
    for batch, run_id in zip(batches, run_ids, strict=True):
        entry = partition.entry(run_id)
        report_row = connection.execute(
            """
            SELECT report_sha256
            FROM impact_capture_report
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if report_row is None:
            raise ValueError("Round 74 delayed execution capture report is missing")
        validate_round74_capture_report_binding(
            entry,
            stored_capture_report_sha256=report_row[0],
        )
        reports.append((run_id, entry.capture_report_sha256))
        replayed = replay_round74_delayed_execution_run(
            batch,
            assemblies[run_id],
            latency_evidence,
            iter_round74_v10_event_observations(store, run_id=run_id),
            capture_report_sha256=entry.capture_report_sha256,
        )
        for result in replayed:
            runs_by_profile[result.profile].append(result)
    source_assemblies = tuple(
        (run_id, assemblies[run_id].assembly_sha256) for run_id in run_ids
    )
    module_sha256 = _module_sha256()
    panels = tuple(
        Round74ActionExecutionPanel(
            profile=profile,
            partition_sha256=partition.partition_sha256,
            decision_latency_evidence_sha256=latency_evidence.evidence_sha256,
            additional_entry_latency_ns=profile_runs[0].additional_entry_latency_ns,
            source_target_assembly_sha256=source_assemblies,
            source_capture_report_sha256=tuple(reports),
            execution_replay_module_sha256=module_sha256,
            rows=tuple(row for run in profile_runs for row in run.rows),
        )
        for profile in ROUND74_ACTION_PROFILES
        for profile_runs in (runs_by_profile[profile],)
    )
    for panel in panels:
        panel.validate()
    return panels


__all__ = [
    "ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION",
    "Round74DelayedExecutionRun",
    "build_round74_delayed_execution_panels",
    "replay_round74_delayed_execution_run",
]

"""Leak-resistant streaming datasets for Round 74 event models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from itertools import pairwise
import json
import math
import re
from typing import Iterator, Mapping, Sequence

import numpy as np

from .impact_absorption_event_sequence import (
    ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS,
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
    ROUND74_EVENT_SYMBOLS,
    Round74EventToken,
    Round74ReplayObservation,
)
from .impact_absorption_event_scaling import Round74EventFeatureScaler
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS,
    ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS,
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
    Round74EventTargetOutcome,
)


ROUND74_EVENT_DATASET_SCHEMA_VERSION = "round-074-event-dataset-v8"
ROUND74_EVENT_PARTITION_SCHEMA_VERSION = "round-074-run-partition-v4"
ROUND74_EVENT_PARTITION_ROLES = ("training", "tuning", "test")
ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS = (
    max(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS) * 1_000_000_000
    + 2 * ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
    + 2 * ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS
)
ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS = (
    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
)
ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS = (
    ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS
)

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLE_CODE = {role: index for index, role in enumerate(ROUND74_EVENT_PARTITION_ROLES)}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 {label} digest is invalid")
    return selected


@dataclass(frozen=True)
class Round74EventRunPartitionEntry:
    """One immutable capture run assigned wholly to one chronological role."""

    run_id: str
    role: str
    capture_report_sha256: str
    capture_start_wall_ns: int
    capture_end_wall_ns: int
    eligible_anchor_start_wall_ns: int
    eligible_anchor_end_wall_ns: int

    def validate(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("Round 74 partition run id is invalid")
        if self.role not in ROUND74_EVENT_PARTITION_ROLES:
            raise ValueError("Round 74 partition role is invalid")
        _require_sha256(self.capture_report_sha256, "capture report")
        times = (
            int(self.capture_start_wall_ns),
            int(self.capture_end_wall_ns),
            int(self.eligible_anchor_start_wall_ns),
            int(self.eligible_anchor_end_wall_ns),
        )
        if min(times) <= 0 or not (times[0] <= times[2] <= times[3] <= times[1]):
            raise ValueError("Round 74 partition interval is invalid")
        if times[3] + ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS > times[1]:
            raise ValueError("Round 74 partition leaves insufficient target coverage")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "run_id": self.run_id,
            "role": self.role,
            "capture_report_sha256": self.capture_report_sha256,
            "capture_start_wall_ns": self.capture_start_wall_ns,
            "capture_end_wall_ns": self.capture_end_wall_ns,
            "eligible_anchor_start_wall_ns": (self.eligible_anchor_start_wall_ns),
            "eligible_anchor_end_wall_ns": self.eligible_anchor_end_wall_ns,
        }


@dataclass(frozen=True)
class Round74EventRunPartition:
    """Whole-run chronological roles with explicit transition purges."""

    entries: tuple[Round74EventRunPartitionEntry, ...]
    cohort_plan_sha256: str
    purge_ns: int = ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    embargo_ns: int = ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
    schema_version: str = ROUND74_EVENT_PARTITION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != ROUND74_EVENT_PARTITION_SCHEMA_VERSION:
            raise ValueError("Round 74 partition schema differs")
        _require_sha256(self.cohort_plan_sha256, "cohort plan")
        if (
            int(self.purge_ns) < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
            or int(self.embargo_ns) < ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS
        ):
            raise ValueError("Round 74 partition purge or embargo is too short")
        if len(self.entries) < len(ROUND74_EVENT_PARTITION_ROLES):
            raise ValueError("Round 74 partition has too few runs")
        prior: Round74EventRunPartitionEntry | None = None
        run_ids: set[str] = set()
        observed_roles: set[str] = set()
        for entry in self.entries:
            entry.validate()
            if entry.run_id in run_ids:
                raise ValueError("Round 74 partition run is duplicated")
            run_ids.add(entry.run_id)
            observed_roles.add(entry.role)
            if prior is not None:
                if entry.capture_start_wall_ns < prior.capture_end_wall_ns:
                    raise ValueError("Round 74 partition capture runs overlap")
                if _ROLE_CODE[entry.role] < _ROLE_CODE[prior.role]:
                    raise ValueError("Round 74 partition role order regressed")
                if entry.role != prior.role and (
                    prior.eligible_anchor_end_wall_ns
                    > prior.capture_end_wall_ns - int(self.purge_ns)
                    or entry.eligible_anchor_start_wall_ns
                    < entry.capture_start_wall_ns + int(self.embargo_ns)
                ):
                    raise ValueError("Round 74 partition transition is not purged")
            prior = entry
        if observed_roles != set(ROUND74_EVENT_PARTITION_ROLES):
            raise ValueError("Round 74 partition roles are incomplete")

    @property
    def partition_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "cohort_plan_sha256": self.cohort_plan_sha256,
            "split_unit": "whole_capture_run",
            "random_row_split_permitted": False,
            "purge_ns": int(self.purge_ns),
            "embargo_ns": int(self.embargo_ns),
            "maximum_target_span_ns": (ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS),
            "entries": [entry.as_dict() for entry in self.entries],
        }
        if include_sha256:
            value["partition_sha256"] = _canonical_sha256(value)
        return value

    def entry(self, run_id: str) -> Round74EventRunPartitionEntry:
        self.validate()
        selected = [entry for entry in self.entries if entry.run_id == run_id]
        if len(selected) != 1:
            raise ValueError("Round 74 partition run is not unique")
        return selected[0]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74EventRunPartition:
        payload = dict(value)
        claimed = str(payload.pop("partition_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 partition payload digest differs")
        rows = payload.get("entries")
        if not isinstance(rows, list):
            raise ValueError("Round 74 partition entries payload differs")
        entries = tuple(
            Round74EventRunPartitionEntry(
                run_id=str(row["run_id"]),
                role=str(row["role"]),
                capture_report_sha256=str(row["capture_report_sha256"]),
                capture_start_wall_ns=int(row["capture_start_wall_ns"]),
                capture_end_wall_ns=int(row["capture_end_wall_ns"]),
                eligible_anchor_start_wall_ns=int(row["eligible_anchor_start_wall_ns"]),
                eligible_anchor_end_wall_ns=int(row["eligible_anchor_end_wall_ns"]),
            )
            for row in rows
            if isinstance(row, Mapping)
        )
        if len(entries) != len(rows):
            raise ValueError("Round 74 partition entry payload differs")
        selected = cls(
            entries=entries,
            cohort_plan_sha256=str(payload["cohort_plan_sha256"]),
            purge_ns=int(payload["purge_ns"]),
            embargo_ns=int(payload["embargo_ns"]),
            schema_version=str(payload["schema_version"]),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 partition static policy differs")
        selected.validate()
        return selected


def _feature_window_sha256(
    *,
    run_id: str,
    symbol: str,
    anchor_index: int,
    endpoint: Round74EventToken,
    feature_values: tuple[tuple[float, ...], ...],
) -> str:
    metadata = {
        "schema_version": ROUND74_EVENT_DATASET_SCHEMA_VERSION,
        "feature_names_sha256": ROUND74_EVENT_FEATURE_NAMES_SHA256,
        "run_id": run_id,
        "symbol": symbol,
        "anchor_index": int(anchor_index),
        "endpoint_frame_index": endpoint.frame_index,
        "endpoint_message_index": endpoint.message_index,
        "endpoint_received_monotonic_ns": endpoint.received_monotonic_ns,
        "endpoint_received_wall_ns": endpoint.received_wall_ns,
    }
    values = np.ascontiguousarray(feature_values, dtype="<f8")
    digest = hashlib.sha256(_canonical_json(metadata).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Round74LabeledEventWindow:
    """One causal feature window with a complete horizon/side target panel."""

    run_id: str
    role: str
    partition_sha256: str
    test_access_sha256: str
    symbol: str
    anchor_index: int
    decision_monotonic_ns: int
    decision_wall_ns: int
    endpoint_frame_index: int
    endpoint_message_index: int
    feature_window_sha256: str
    feature_values: tuple[tuple[float, ...], ...]
    outcomes: tuple[Round74EventTargetOutcome, ...]
    schema_version: str = ROUND74_EVENT_DATASET_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_EVENT_DATASET_SCHEMA_VERSION
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or self.symbol not in ROUND74_EVENT_SYMBOLS
            or int(self.anchor_index) < 0
            or min(
                int(self.decision_monotonic_ns),
                int(self.decision_wall_ns),
                int(self.endpoint_frame_index),
                int(self.endpoint_message_index),
            )
            < 0
        ):
            raise ValueError("Round 74 labeled window identity differs")
        _require_sha256(self.partition_sha256, "partition")
        _require_sha256(self.feature_window_sha256, "feature window")
        if self.role == "test":
            _require_sha256(self.test_access_sha256, "test access")
        elif self.test_access_sha256:
            raise ValueError("Round 74 development window has test access")
        if len(self.feature_values) != ROUND74_EVENT_SEQUENCE_LENGTH or any(
            len(row) != len(ROUND74_EVENT_FEATURE_NAMES)
            or not all(math.isfinite(float(value)) for value in row)
            for row in self.feature_values
        ):
            raise ValueError("Round 74 labeled window features differ")
        expected = {
            (horizon, side)
            for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
            for side in ROUND74_EVENT_PAYOFF_SIDES
        }
        if (
            len(self.outcomes) != len(expected)
            or {(row.horizon_seconds, row.side) for row in self.outcomes} != expected
        ):
            raise ValueError("Round 74 labeled window target panel differs")
        for outcome in self.outcomes:
            outcome.validate()
            if (
                outcome.symbol != self.symbol
                or outcome.anchor_index != self.anchor_index
                or outcome.feature_window_sha256 != self.feature_window_sha256
            ):
                raise ValueError("Round 74 labeled window outcome identity differs")
        contexts = {outcome.target_context_sha256 for outcome in self.outcomes}
        if len(contexts) != 1:
            raise ValueError("Round 74 labeled window target context differs")

    @property
    def sample_sha256(self) -> str:
        self.validate()
        metadata = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "role": self.role,
            "partition_sha256": self.partition_sha256,
            "test_access_sha256": self.test_access_sha256,
            "symbol": self.symbol,
            "anchor_index": self.anchor_index,
            "decision_monotonic_ns": self.decision_monotonic_ns,
            "decision_wall_ns": self.decision_wall_ns,
            "endpoint_frame_index": self.endpoint_frame_index,
            "endpoint_message_index": self.endpoint_message_index,
            "feature_window_sha256": self.feature_window_sha256,
            "outcome_sha256": [outcome.outcome_sha256 for outcome in self.outcomes],
        }
        values = np.ascontiguousarray(self.feature_values, dtype="<f8")
        digest = hashlib.sha256(_canonical_json(metadata).encode("ascii"))
        digest.update(memoryview(values).cast("B"))
        return digest.hexdigest()

    @property
    def eligible_action_count(self) -> int:
        return sum(outcome.eligible for outcome in self.outcomes)


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _update_array_digest(digest: object, value: np.ndarray) -> None:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(int(canonical.ndim).to_bytes(2, "little", signed=False))
    for size in canonical.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


@dataclass(frozen=True)
class Round74EventTrainingBatch:
    """Scaled model tensors with explicit eligibility and source identity."""

    role: str
    partition_sha256: str
    scaler_sha256: str
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    decision_monotonic_ns: np.ndarray
    decision_wall_ns: np.ndarray
    endpoint_frame_index: np.ndarray
    endpoint_message_index: np.ndarray
    anchor_index: np.ndarray
    sample_sha256: tuple[str, ...]
    feature_window_sha256: tuple[str, ...]
    target_context_sha256: tuple[str, ...]
    test_access_sha256: tuple[str, ...]
    feature_values: np.ndarray
    actual_entry_monotonic_ns: np.ndarray
    actual_exit_monotonic_ns: np.ndarray
    net_payoff_bps: np.ndarray
    maximum_adverse_excursion_bps: np.ndarray
    adverse_selection: np.ndarray
    regime_unpredictability: np.ndarray
    action_eligibility: np.ndarray
    regime_unpredictability_eligibility: np.ndarray
    schema_version: str = ROUND74_EVENT_DATASET_SCHEMA_VERSION

    @property
    def rows(self) -> int:
        return len(self.sample_sha256)

    def validate(self) -> None:
        action_shape = (
            self.rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        regime_shape = (
            self.rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        )
        action_arrays = (
            self.net_payoff_bps,
            self.maximum_adverse_excursion_bps,
            self.adverse_selection,
            self.action_eligibility,
        )
        action_timing_arrays = (
            self.actual_entry_monotonic_ns,
            self.actual_exit_monotonic_ns,
        )
        regime_arrays = (
            self.regime_unpredictability,
            self.regime_unpredictability_eligibility,
        )
        model_arrays = (
            self.feature_values,
            *action_arrays,
            *regime_arrays,
        )
        identity_arrays = (
            self.decision_monotonic_ns,
            self.decision_wall_ns,
            self.endpoint_frame_index,
            self.endpoint_message_index,
            self.anchor_index,
        )
        if (
            self.schema_version != ROUND74_EVENT_DATASET_SCHEMA_VERSION
            or self.role not in ROUND74_EVENT_PARTITION_ROLES
            or self.rows < 1
            or len(self.run_id) != self.rows
            or len(self.symbol) != self.rows
            or len(self.feature_window_sha256) != self.rows
            or len(self.target_context_sha256) != self.rows
            or len(self.test_access_sha256) != self.rows
            or any(_RUN_ID.fullmatch(value) is None for value in self.run_id)
            or any(value not in ROUND74_EVENT_SYMBOLS for value in self.symbol)
            or any(_SHA256.fullmatch(value) is None for value in self.sample_sha256)
            or any(
                _SHA256.fullmatch(value) is None for value in self.feature_window_sha256
            )
            or any(
                _SHA256.fullmatch(value) is None for value in self.target_context_sha256
            )
            or (
                self.role == "test"
                and (
                    len(set(self.test_access_sha256)) != 1
                    or any(
                        _SHA256.fullmatch(value) is None
                        for value in self.test_access_sha256
                    )
                )
            )
            or (self.role != "test" and any(self.test_access_sha256))
            or _SHA256.fullmatch(self.partition_sha256) is None
            or _SHA256.fullmatch(self.scaler_sha256) is None
            or self.feature_values.shape
            != (
                self.rows,
                ROUND74_EVENT_SEQUENCE_LENGTH,
                len(ROUND74_EVENT_FEATURE_NAMES),
            )
            or any(value.shape != (self.rows,) for value in identity_arrays)
            or any(value.dtype != np.int64 for value in identity_arrays)
            or any(value.flags.writeable for value in identity_arrays)
            or any(np.any(value < 0) for value in identity_arrays)
            or any(value.shape != action_shape for value in action_arrays)
            or any(
                value.shape != action_shape
                or value.dtype != np.int64
                or value.flags.writeable
                for value in action_timing_arrays
            )
            or any(value.shape != regime_shape for value in regime_arrays)
            or any(value.dtype != np.float32 for value in model_arrays)
            or any(value.flags.writeable for value in model_arrays)
            or not all(np.isfinite(value).all() for value in model_arrays)
        ):
            raise ValueError("Round 74 event training batch contract differs")
        order_keys = tuple(
            (
                int(self.decision_wall_ns[index]),
                self.run_id[index],
                int(self.decision_monotonic_ns[index]),
                int(self.endpoint_frame_index[index]),
                int(self.endpoint_message_index[index]),
                self.symbol[index],
                int(self.anchor_index[index]),
            )
            for index in range(self.rows)
        )
        if any(current <= prior for prior, current in pairwise(order_keys)):
            raise ValueError("Round 74 event training batch order regressed")
        action_mask = self.action_eligibility
        regime_mask = self.regime_unpredictability_eligibility
        if (
            np.any((action_mask != 0.0) & (action_mask != 1.0))
            or np.any((regime_mask != 0.0) & (regime_mask != 1.0))
            or float(action_mask.sum()) <= 0.0
            or float(regime_mask.sum()) <= 0.0
            or np.any(self.maximum_adverse_excursion_bps < 0.0)
            or np.any((self.adverse_selection < 0.0) | (self.adverse_selection > 1.0))
            or np.any(
                (self.regime_unpredictability < 0.0)
                | (self.regime_unpredictability > 1.0)
            )
            or any(
                np.any(value[action_mask == 0.0] != 0.0)
                for value in (
                    self.net_payoff_bps,
                    self.maximum_adverse_excursion_bps,
                    self.adverse_selection,
                )
            )
            or np.any(self.regime_unpredictability[regime_mask == 0.0] != 0.0)
            or any(
                np.any(value[action_mask == 0.0] != -1)
                for value in action_timing_arrays
            )
            or any(
                np.any(value[action_mask == 1.0] < 0) for value in action_timing_arrays
            )
            or np.any(
                self.actual_exit_monotonic_ns[action_mask == 1.0]
                < self.actual_entry_monotonic_ns[action_mask == 1.0]
            )
        ):
            raise ValueError("Round 74 event training batch targets differ")

    @property
    def batch_sha256(self) -> str:
        self.validate()
        identity = {
            "schema_version": self.schema_version,
            "role": self.role,
            "partition_sha256": self.partition_sha256,
            "scaler_sha256": self.scaler_sha256,
            "run_id": list(self.run_id),
            "symbol": list(self.symbol),
            "sample_sha256": list(self.sample_sha256),
            "feature_window_sha256": list(self.feature_window_sha256),
            "target_context_sha256": list(self.target_context_sha256),
            "test_access_sha256": list(self.test_access_sha256),
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
        for value in (
            self.decision_monotonic_ns,
            self.decision_wall_ns,
            self.endpoint_frame_index,
            self.endpoint_message_index,
            self.anchor_index,
            self.feature_values,
            self.actual_entry_monotonic_ns,
            self.actual_exit_monotonic_ns,
            self.net_payoff_bps,
            self.maximum_adverse_excursion_bps,
            self.adverse_selection,
            self.regime_unpredictability,
            self.action_eligibility,
            self.regime_unpredictability_eligibility,
        ):
            _update_array_digest(digest, value)
        return digest.hexdigest()


def build_round74_event_training_batch(
    samples: Sequence[Round74LabeledEventWindow],
    *,
    scaler: Round74EventFeatureScaler,
) -> Round74EventTrainingBatch:
    """Convert complete windows into one bounded, mask-aware model batch."""

    selected = tuple(samples)
    if not selected:
        raise ValueError("Round 74 event training batch is empty")
    for sample in selected:
        sample.validate()
    roles = {sample.role for sample in selected}
    partitions = {sample.partition_sha256 for sample in selected}
    if len(roles) != 1 or len(partitions) != 1:
        raise ValueError("Round 74 event batch crossed role or partition")
    raw_features = np.asarray(
        [sample.feature_values for sample in selected],
        dtype=np.float64,
    )
    feature_values = scaler.transform(raw_features)
    action_shape = (
        len(selected),
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        len(selected),
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    payoff = np.zeros(action_shape, dtype=np.float32)
    actual_entry = np.full(action_shape, -1, dtype=np.int64)
    actual_exit = np.full(action_shape, -1, dtype=np.int64)
    adverse_excursion = np.zeros(action_shape, dtype=np.float32)
    adverse_selection = np.zeros(action_shape, dtype=np.float32)
    unpredictability = np.zeros(regime_shape, dtype=np.float32)
    action_eligibility = np.zeros(action_shape, dtype=np.float32)
    unpredictability_eligibility = np.zeros(regime_shape, dtype=np.float32)
    contexts: list[str] = []
    for sample_index, sample in enumerate(selected):
        context_values = {outcome.target_context_sha256 for outcome in sample.outcomes}
        if len(context_values) != 1:
            raise ValueError("Round 74 sample target context differs")
        contexts.append(next(iter(context_values)))
        regime_values: dict[int, list[float]] = {
            horizon: [] for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
        }
        for outcome in sample.outcomes:
            if not outcome.eligible:
                continue
            horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(
                outcome.horizon_seconds
            )
            side_index = ROUND74_EVENT_PAYOFF_SIDES.index(outcome.side)
            assert (
                outcome.actual_entry_monotonic_ns is not None
                and outcome.actual_exit_monotonic_ns is not None
                and outcome.capital_scaled_net_payoff_bps is not None
                and outcome.capital_scaled_maximum_adverse_excursion_bps is not None
                and outcome.adverse_selection is not None
                and outcome.regime_unpredictability is not None
            )
            actual_entry[
                sample_index,
                horizon_index,
                side_index,
            ] = outcome.actual_entry_monotonic_ns
            actual_exit[
                sample_index,
                horizon_index,
                side_index,
            ] = outcome.actual_exit_monotonic_ns
            payoff[sample_index, horizon_index, side_index] = (
                outcome.capital_scaled_net_payoff_bps
            )
            adverse_excursion[sample_index, horizon_index, side_index] = (
                outcome.capital_scaled_maximum_adverse_excursion_bps
            )
            adverse_selection[sample_index, horizon_index, side_index] = float(
                outcome.adverse_selection
            )
            action_eligibility[sample_index, horizon_index, side_index] = 1.0
            regime_values[outcome.horizon_seconds].append(
                outcome.regime_unpredictability
            )
        for horizon_index, horizon in enumerate(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS):
            values = regime_values[horizon]
            if not values:
                continue
            if not all(
                math.isclose(
                    value,
                    values[0],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for value in values[1:]
            ):
                raise ValueError("Round 74 side-independent regime target differs")
            unpredictability[sample_index, horizon_index] = values[0]
            unpredictability_eligibility[sample_index, horizon_index] = 1.0
    if float(action_eligibility.sum()) <= 0.0:
        raise ValueError("Round 74 event batch has no eligible actions")
    batch = Round74EventTrainingBatch(
        role=next(iter(roles)),
        partition_sha256=next(iter(partitions)),
        scaler_sha256=scaler.scaler_sha256,
        run_id=tuple(sample.run_id for sample in selected),
        symbol=tuple(sample.symbol for sample in selected),
        decision_monotonic_ns=_readonly(
            np.asarray(
                [sample.decision_monotonic_ns for sample in selected],
                dtype=np.int64,
            )
        ),
        decision_wall_ns=_readonly(
            np.asarray(
                [sample.decision_wall_ns for sample in selected],
                dtype=np.int64,
            )
        ),
        endpoint_frame_index=_readonly(
            np.asarray(
                [sample.endpoint_frame_index for sample in selected],
                dtype=np.int64,
            )
        ),
        endpoint_message_index=_readonly(
            np.asarray(
                [sample.endpoint_message_index for sample in selected],
                dtype=np.int64,
            )
        ),
        anchor_index=_readonly(
            np.asarray(
                [sample.anchor_index for sample in selected],
                dtype=np.int64,
            )
        ),
        sample_sha256=tuple(sample.sample_sha256 for sample in selected),
        feature_window_sha256=tuple(
            sample.feature_window_sha256 for sample in selected
        ),
        target_context_sha256=tuple(contexts),
        test_access_sha256=tuple(sample.test_access_sha256 for sample in selected),
        feature_values=_readonly(feature_values),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(adverse_selection),
        regime_unpredictability=_readonly(unpredictability),
        action_eligibility=_readonly(action_eligibility),
        regime_unpredictability_eligibility=_readonly(unpredictability_eligibility),
    )
    batch.validate()
    return batch


@dataclass
class _PendingWindow:
    run_id: str
    role: str
    partition_sha256: str
    test_access_sha256: str
    symbol: str
    anchor_index: int
    decision_monotonic_ns: int
    decision_wall_ns: int
    endpoint_frame_index: int
    endpoint_message_index: int
    feature_window_sha256: str
    feature_values: tuple[tuple[float, ...], ...]
    outcomes: list[Round74EventTargetOutcome] = field(default_factory=list)

    def complete(self) -> Round74LabeledEventWindow:
        rows = tuple(
            sorted(
                self.outcomes,
                key=lambda outcome: (outcome.horizon_seconds, outcome.side),
            )
        )
        sample = Round74LabeledEventWindow(
            run_id=self.run_id,
            role=self.role,
            partition_sha256=self.partition_sha256,
            test_access_sha256=self.test_access_sha256,
            symbol=self.symbol,
            anchor_index=self.anchor_index,
            decision_monotonic_ns=self.decision_monotonic_ns,
            decision_wall_ns=self.decision_wall_ns,
            endpoint_frame_index=self.endpoint_frame_index,
            endpoint_message_index=self.endpoint_message_index,
            feature_window_sha256=self.feature_window_sha256,
            feature_values=self.feature_values,
            outcomes=rows,
        )
        sample.validate()
        return sample


class Round74EventDatasetAssembler:
    """Join feature windows and target panels in one bounded exact replay."""

    def __init__(
        self,
        *,
        partition: Round74EventRunPartition,
        run_id: str,
        target_engine: Round74EventTargetEngine,
        pretest_model_policy_sha256: str | None = None,
        test_unlock_sha256: str | None = None,
        maximum_window_gap_ns: int = (ROUND74_EVENT_DEFAULT_MAX_WINDOW_GAP_NS),
    ) -> None:
        partition.validate()
        self.partition = partition
        self.run = partition.entry(str(run_id))
        self.target_engine = target_engine
        if target_engine.anchor_count or target_engine.outcomes:
            raise ValueError("Round 74 assembler target engine is not empty")
        if int(maximum_window_gap_ns) < 1:
            raise ValueError("Round 74 assembler window gap is invalid")
        self.maximum_window_gap_ns = int(maximum_window_gap_ns)
        if self.run.role == "test":
            policy = _require_sha256(
                pretest_model_policy_sha256,
                "pretest model policy",
            )
            unlock = _require_sha256(test_unlock_sha256, "test unlock")
            self.test_access_sha256 = _canonical_sha256(
                {
                    "pretest_model_policy_sha256": policy,
                    "test_unlock_sha256": unlock,
                }
            )
        elif pretest_model_policy_sha256 is not None or test_unlock_sha256 is not None:
            raise ValueError(
                "Round 74 development assembly received test authorization"
            )
        else:
            self.test_access_sha256 = ""
        self._buffers = {
            symbol: deque(maxlen=ROUND74_EVENT_SEQUENCE_LENGTH)
            for symbol in ROUND74_EVENT_SYMBOLS
        }
        self._prior_token_ns: dict[str, int] = {}
        self._last_anchor_ns: dict[str, int] = {}
        self._next_anchor_index = {symbol: 0 for symbol in ROUND74_EVENT_SYMBOLS}
        self._pending: dict[str, _PendingWindow] = {}
        self._outcome_cursor = 0
        self._prior_observation_key: tuple[int, int, int] | None = None
        self._finished = False

    @property
    def pending_window_count(self) -> int:
        return len(self._pending)

    def _collect_completed(self) -> tuple[Round74LabeledEventWindow, ...]:
        completed: list[Round74LabeledEventWindow] = []
        new_outcomes = self.target_engine.outcomes[self._outcome_cursor :]
        self._outcome_cursor = len(self.target_engine.outcomes)
        for outcome in new_outcomes:
            pending = self._pending.get(outcome.feature_window_sha256)
            if pending is None:
                raise ValueError("Round 74 target outcome has no feature window")
            pending.outcomes.append(outcome)
            if len(pending.outcomes) > (
                len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
                * len(ROUND74_EVENT_PAYOFF_SIDES)
            ):
                raise ValueError("Round 74 target panel has duplicate outcomes")
            if len(pending.outcomes) == (
                len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
                * len(ROUND74_EVENT_PAYOFF_SIDES)
            ):
                completed.append(pending.complete())
                del self._pending[outcome.feature_window_sha256]
        completed.sort(
            key=lambda sample: (
                sample.decision_monotonic_ns,
                sample.symbol,
                sample.anchor_index,
            )
        )
        return tuple(completed)

    def _append_token(self, token: Round74EventToken) -> None:
        symbol = token.symbol
        prior = self._prior_token_ns.get(symbol)
        buffer = self._buffers[symbol]
        if (
            prior is not None
            and token.received_monotonic_ns - prior > self.maximum_window_gap_ns
        ):
            buffer.clear()
        self._prior_token_ns[symbol] = token.received_monotonic_ns
        buffer.append(token)
        if len(buffer) < ROUND74_EVENT_SEQUENCE_LENGTH:
            return
        if not (
            self.run.eligible_anchor_start_wall_ns
            <= token.received_wall_ns
            <= self.run.eligible_anchor_end_wall_ns
        ):
            return
        last_anchor = self._last_anchor_ns.get(symbol)
        spacing = self.target_engine.spec.minimum_anchor_spacing_ns
        if (
            last_anchor is not None
            and token.received_monotonic_ns - last_anchor < spacing
        ):
            return
        anchor_index = self._next_anchor_index[symbol]
        feature_values = tuple(item.feature_values for item in buffer)
        feature_sha = _feature_window_sha256(
            run_id=self.run.run_id,
            symbol=symbol,
            anchor_index=anchor_index,
            endpoint=token,
            feature_values=feature_values,
        )
        if feature_sha in self._pending:
            raise ValueError("Round 74 feature-window digest is duplicated")
        anchor = Round74EventTargetAnchor(
            symbol=symbol,
            anchor_index=anchor_index,
            decision_monotonic_ns=token.received_monotonic_ns,
            decision_wall_ns=token.received_wall_ns,
            endpoint_frame_index=token.frame_index,
            endpoint_message_index=token.message_index,
            feature_window_sha256=feature_sha,
        )
        self.target_engine.add_anchor(anchor)
        self._pending[feature_sha] = _PendingWindow(
            run_id=self.run.run_id,
            role=self.run.role,
            partition_sha256=self.partition.partition_sha256,
            test_access_sha256=self.test_access_sha256,
            symbol=symbol,
            anchor_index=anchor_index,
            decision_monotonic_ns=token.received_monotonic_ns,
            decision_wall_ns=token.received_wall_ns,
            endpoint_frame_index=token.frame_index,
            endpoint_message_index=token.message_index,
            feature_window_sha256=feature_sha,
            feature_values=feature_values,
        )
        self._next_anchor_index[symbol] += 1
        self._last_anchor_ns[symbol] = token.received_monotonic_ns

    def consume(
        self,
        observation: Round74ReplayObservation,
    ) -> tuple[Round74LabeledEventWindow, ...]:
        if self._finished:
            raise ValueError("Round 74 dataset assembler is already finished")
        observation.validate()
        order_key = (
            observation.received_monotonic_ns,
            observation.frame_index,
            observation.message_index,
        )
        if (
            self._prior_observation_key is not None
            and order_key <= self._prior_observation_key
        ):
            raise ValueError("Round 74 dataset observation order regressed")
        self._prior_observation_key = order_key
        if not (
            self.run.capture_start_wall_ns
            <= observation.received_wall_ns
            <= self.run.capture_end_wall_ns
        ):
            raise ValueError("Round 74 dataset observation is outside its run")
        if (
            observation.depth_state is not None
            and not observation.depth_update_is_stale
        ):
            self.target_engine.observe_depth(
                received_monotonic_ns=observation.received_monotonic_ns,
                frame_index=observation.frame_index,
                message_index=observation.message_index,
                state=observation.depth_state,
            )
        completed = list(self._collect_completed())
        if observation.token is not None:
            self._append_token(observation.token)
        completed.extend(self._collect_completed())
        completed.sort(
            key=lambda sample: (
                sample.decision_monotonic_ns,
                sample.symbol,
                sample.anchor_index,
            )
        )
        return tuple(completed)

    def finish(self) -> tuple[Round74LabeledEventWindow, ...]:
        if self._finished:
            return ()
        completed = list(self._collect_completed())
        self.target_engine.finish()
        completed.extend(self._collect_completed())
        if self._pending:
            raise ValueError("Round 74 dataset windows remain unresolved")
        self._finished = True
        completed.sort(
            key=lambda sample: (
                sample.decision_monotonic_ns,
                sample.symbol,
                sample.anchor_index,
            )
        )
        return tuple(completed)


def validate_round74_capture_report_binding(
    entry: Round74EventRunPartitionEntry,
    *,
    stored_capture_report_sha256: object,
) -> None:
    entry.validate()
    stored = _require_sha256(
        stored_capture_report_sha256,
        "stored capture report",
    )
    if stored != entry.capture_report_sha256:
        raise ValueError("Round 74 partition capture report differs")


def iter_round74_labeled_event_windows(
    store: object,
    *,
    partition: Round74EventRunPartition,
    run_id: str,
    target_engine: Round74EventTargetEngine,
    pretest_model_policy_sha256: str | None = None,
    test_unlock_sha256: str | None = None,
) -> Iterator[Round74LabeledEventWindow]:
    """Audit and join one hash-bound run without writing duplicate windows."""

    from .impact_absorption_event_sequence import (
        iter_round74_v10_event_observations,
    )
    from .impact_absorption_store import ImpactAbsorptionStore

    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 labeled replay requires an ImpactAbsorptionStore")
    if not store.read_only:
        raise ValueError("Round 74 labeled replay requires a read-only store")
    entry = partition.entry(str(run_id))
    report_row = (
        store.connect()
        .execute(
            """
        SELECT report_sha256 FROM impact_capture_report WHERE run_id = ?
        """,
            [entry.run_id],
        )
        .fetchone()
    )
    if report_row is None:
        raise ValueError("Round 74 partition capture report is missing")
    validate_round74_capture_report_binding(
        entry,
        stored_capture_report_sha256=report_row[0],
    )
    assembler = Round74EventDatasetAssembler(
        partition=partition,
        run_id=entry.run_id,
        target_engine=target_engine,
        pretest_model_policy_sha256=pretest_model_policy_sha256,
        test_unlock_sha256=test_unlock_sha256,
    )
    for observation in iter_round74_v10_event_observations(
        store,
        run_id=entry.run_id,
    ):
        yield from assembler.consume(observation)
    yield from assembler.finish()


__all__ = [
    "ROUND74_EVENT_DATASET_SCHEMA_VERSION",
    "ROUND74_EVENT_PARTITION_MAXIMUM_TARGET_SPAN_NS",
    "ROUND74_EVENT_PARTITION_MINIMUM_EMBARGO_NS",
    "ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS",
    "ROUND74_EVENT_PARTITION_ROLES",
    "ROUND74_EVENT_PARTITION_SCHEMA_VERSION",
    "Round74EventDatasetAssembler",
    "Round74EventTrainingBatch",
    "Round74EventRunPartition",
    "Round74EventRunPartitionEntry",
    "Round74LabeledEventWindow",
    "build_round74_event_training_batch",
    "iter_round74_labeled_event_windows",
    "validate_round74_capture_report_binding",
]

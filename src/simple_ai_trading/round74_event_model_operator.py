"""Bounded, read-only data preparation for the Round 74 event models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_ROLES,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    Round74EventTrainingBatch,
    build_round74_event_training_batch,
    iter_round74_labeled_event_windows,
    validate_round74_capture_report_binding,
)
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    Round74EventFeatureScaler,
    fit_round74_event_feature_scaler_stream,
)
from .impact_absorption_event_sequence import iter_round74_v10_event_observations
from .impact_absorption_store import ImpactAbsorptionStore
from .impact_absorption_target_assembly import Round74SourceTargetAssembly


ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION = "round-074-event-model-operator-v1"
ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS = 8_192


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


def _require_read_only_store(store: object) -> ImpactAbsorptionStore:
    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 model operator requires an ImpactAbsorptionStore")
    if not store.read_only:
        raise ValueError("Round 74 model operator requires a read-only store")
    return store


def _validate_capture_report(
    store: ImpactAbsorptionStore,
    *,
    partition: Round74EventRunPartition,
    run_id: str,
) -> None:
    entry = partition.entry(run_id)
    row = (
        store.connect()
        .execute(
            """
            SELECT report_sha256 FROM impact_capture_report WHERE run_id = ?
            """,
            [entry.run_id],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 74 model operator capture report is missing")
    validate_round74_capture_report_binding(
        entry,
        stored_capture_report_sha256=row[0],
    )


def iter_round74_training_feature_chunks(
    store: object,
    *,
    partition: Round74EventRunPartition,
    chunk_rows: int = ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS,
) -> Iterator[np.ndarray]:
    """Yield each raw training event once, without overlapping-window copies."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 2
    ):
        raise ValueError("Round 74 model operator feature chunk size differs")
    training_entries = tuple(
        entry for entry in partition.entries if entry.role == "training"
    )
    values: list[tuple[float, ...]] = []
    emitted_rows = 0
    for entry in training_entries:
        _validate_capture_report(
            selected_store,
            partition=partition,
            run_id=entry.run_id,
        )
        for observation in iter_round74_v10_event_observations(
            selected_store,
            run_id=entry.run_id,
        ):
            token = observation.token
            if token is None:
                continue
            if not (
                entry.capture_start_wall_ns
                <= token.received_wall_ns
                <= entry.capture_end_wall_ns
            ):
                raise ValueError(
                    "Round 74 model operator event is outside its capture run"
                )
            values.append(token.feature_values)
            if len(values) == chunk_rows:
                chunk = np.asarray(values, dtype=np.float64)
                emitted_rows += int(chunk.shape[0])
                yield chunk
                values.clear()
    if values:
        chunk = np.asarray(values, dtype=np.float64)
        emitted_rows += int(chunk.shape[0])
        yield chunk
    if emitted_rows < 2:
        raise ValueError("Round 74 model operator training events are insufficient")


def fit_round74_cohort_feature_scaler(
    store: object,
    *,
    partition: Round74EventRunPartition,
    chunk_rows: int = ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74EventFeatureScaler:
    """Fit the bounded scaler from training events and no later role."""

    return fit_round74_event_feature_scaler_stream(
        iter_round74_training_feature_chunks(
            store,
            partition=partition,
            chunk_rows=chunk_rows,
        ),
        partition_role="training",
        maximum_fit_rows=maximum_fit_rows,
    )


def _role_entries(
    partition: Round74EventRunPartition,
    role: str,
) -> tuple[Round74EventRunPartitionEntry, ...]:
    selected_role = str(role)
    if selected_role not in ROUND74_EVENT_PARTITION_ROLES:
        raise ValueError("Round 74 model operator role differs")
    return tuple(entry for entry in partition.entries if entry.role == selected_role)


def assemble_round74_role_batches(
    store: object,
    *,
    partition: Round74EventRunPartition,
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    pretest_model_policy_sha256: str | None = None,
    test_unlock_sha256: str | None = None,
) -> tuple[Round74EventTrainingBatch, ...]:
    """Replay and assemble exactly one in-memory batch per capture run."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    entries = _role_entries(partition, role)
    expected_run_ids = tuple(entry.run_id for entry in entries)
    assemblies = dict(target_assembly_by_run_id)
    if set(assemblies) != set(expected_run_ids) or any(
        not isinstance(assemblies[run_id], Round74SourceTargetAssembly)
        for run_id in expected_run_ids
    ):
        raise ValueError("Round 74 model operator target assembly panel differs")
    if role == "test":
        if pretest_model_policy_sha256 is None or test_unlock_sha256 is None:
            raise ValueError("Round 74 model operator test authorization is missing")
    elif pretest_model_policy_sha256 is not None or test_unlock_sha256 is not None:
        raise ValueError(
            "Round 74 model operator development role received test authorization"
        )
    batches: list[Round74EventTrainingBatch] = []
    for entry in entries:
        engine = assemblies[entry.run_id].create_engine(anchors=())
        samples = tuple(
            iter_round74_labeled_event_windows(
                selected_store,
                partition=partition,
                run_id=entry.run_id,
                target_engine=engine,
                pretest_model_policy_sha256=pretest_model_policy_sha256,
                test_unlock_sha256=test_unlock_sha256,
            )
        )
        if not samples:
            raise ValueError(
                f"Round 74 model operator run {entry.run_id} has no samples"
            )
        batch = build_round74_event_training_batch(samples, scaler=scaler)
        batch.validate()
        if batch.role != role or set(batch.run_id) != {entry.run_id}:
            raise ValueError("Round 74 model operator batch crossed capture runs")
        batches.append(batch)
    return tuple(batches)


@dataclass(frozen=True)
class Round74PreparedDevelopmentData:
    """Training/tuning batches and their training-only scaler identity."""

    scaler: Round74EventFeatureScaler
    training_batches: tuple[Round74EventTrainingBatch, ...]
    tuning_batches: tuple[Round74EventTrainingBatch, ...]
    schema_version: str = ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION

    def validate(self) -> None:
        batches = (*self.training_batches, *self.tuning_batches)
        if (
            self.schema_version != ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION
            or not isinstance(self.scaler, Round74EventFeatureScaler)
            or not self.training_batches
            or not self.tuning_batches
        ):
            raise ValueError("Round 74 prepared development data differs")
        for batch in batches:
            batch.validate()
        if (
            any(batch.role != "training" for batch in self.training_batches)
            or any(batch.role != "tuning" for batch in self.tuning_batches)
            or any(len(set(batch.run_id)) != 1 for batch in batches)
            or len({batch.partition_sha256 for batch in batches}) != 1
            or {batch.scaler_sha256 for batch in batches} != {self.scaler.scaler_sha256}
        ):
            raise ValueError("Round 74 prepared development data identity differs")
        run_ids = tuple(next(iter(set(batch.run_id))) for batch in batches)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Round 74 prepared development run is repeated")
        if int(self.training_batches[-1].decision_wall_ns[-1]) >= int(
            self.tuning_batches[0].decision_wall_ns[0]
        ):
            raise ValueError("Round 74 prepared development chronology differs")

    @property
    def preparation_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "scaler_sha256": self.scaler.scaler_sha256,
            "training_batch_sha256": [
                batch.batch_sha256 for batch in self.training_batches
            ],
            "tuning_batch_sha256": [
                batch.batch_sha256 for batch in self.tuning_batches
            ],
            "training_rows": sum(batch.rows for batch in self.training_batches),
            "tuning_rows": sum(batch.rows for batch in self.tuning_batches),
            "source_database_access": "read_only",
            "training_source_replay_passes": 2,
            "tuning_source_replay_passes": 1,
            "overlapping_windows_persisted": False,
            "test_role_accessed": False,
        }


def prepare_round74_development_data(
    store: object,
    *,
    partition: Round74EventRunPartition,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    chunk_rows: int = ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74PreparedDevelopmentData:
    """Fit the scaler, then build development batches without test access."""

    partition.validate()
    development_entries = tuple(
        entry for entry in partition.entries if entry.role != "test"
    )
    expected_run_ids = {entry.run_id for entry in development_entries}
    if set(target_assembly_by_run_id) != expected_run_ids:
        raise ValueError("Round 74 model operator development assembly panel differs")
    scaler = fit_round74_cohort_feature_scaler(
        store,
        partition=partition,
        chunk_rows=chunk_rows,
        maximum_fit_rows=maximum_fit_rows,
    )
    training_run_ids = {
        entry.run_id for entry in development_entries if entry.role == "training"
    }
    training = assemble_round74_role_batches(
        store,
        partition=partition,
        scaler=scaler,
        role="training",
        target_assembly_by_run_id={
            run_id: target_assembly_by_run_id[run_id] for run_id in training_run_ids
        },
    )
    tuning = assemble_round74_role_batches(
        store,
        partition=partition,
        scaler=scaler,
        role="tuning",
        target_assembly_by_run_id={
            entry.run_id: target_assembly_by_run_id[entry.run_id]
            for entry in development_entries
            if entry.role == "tuning"
        },
    )
    result = Round74PreparedDevelopmentData(
        scaler=scaler,
        training_batches=training,
        tuning_batches=tuning,
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS",
    "ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION",
    "Round74PreparedDevelopmentData",
    "assemble_round74_role_batches",
    "fit_round74_cohort_feature_scaler",
    "iter_round74_training_feature_chunks",
    "prepare_round74_development_data",
]

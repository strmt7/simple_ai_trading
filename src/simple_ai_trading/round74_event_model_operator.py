"""Bounded, read-only data preparation for the Round 74 event models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import heapq
import hashlib
import json
from typing import Protocol

import numpy as np

from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_ROLES,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74LabeledEventWindow,
    Round74MatchedEventWindowPair,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    Round74EventTrainingBatch,
    build_round74_event_training_batch,
    iter_round74_labeled_event_windows,
    iter_round74_matched_labeled_event_windows,
    validate_round74_capture_report_binding,
)
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    Round74EventFeatureScaler,
    fit_round74_event_feature_scaler_stream,
)
from .impact_absorption_event_sequence import iter_round74_v10_event_observations
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS, ImpactAbsorptionStore
from .impact_absorption_target_assembly import Round74SourceTargetAssembly

class Round74TuningSubpartitionProtocol(Protocol):
    """Fields shared by legacy and segmented target-blind tuning splits."""

    parent_partition_sha256: str
    model_selection_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    policy_selection_run_ids: tuple[str, ...]

    def validate(self) -> None: ...

    @property
    def subpartition_sha256(self) -> str: ...


ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION = "round-074-event-model-operator-v6"
ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS = 8_192
ROUND74_EVENT_MODEL_TEMPORAL_STRATA = 16
ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM = 16
ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL = (
    ROUND74_EVENT_MODEL_TEMPORAL_STRATA * ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM
)
ROUND74_EVENT_MODEL_WINDOWS_PER_RUN = (
    len(IMPACT_CAPTURE_SYMBOLS) * ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL
)
ROUND74_EVENT_MODEL_WINDOW_SELECTION_SCHEMA_VERSION = (
    "round-074-target-blind-window-selection-v1"
)
ROUND74_EVENT_MODEL_MATCHED_WINDOW_SELECTION_SCHEMA_VERSION = (
    "round-074-target-blind-matched-window-selection-v1"
)


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

    partition.validate()
    training_run_ids = tuple(
        entry.run_id for entry in partition.entries if entry.role == "training"
    )
    return fit_round74_event_feature_scaler_stream(
        iter_round74_training_feature_chunks(
            store,
            partition=partition,
            chunk_rows=chunk_rows,
        ),
        partition_role="training",
        maximum_fit_rows=maximum_fit_rows,
        fit_source_scope="training_partition_all_runs",
        fit_source_run_ids=training_run_ids,
        fit_source_partition_sha256=partition.partition_sha256,
    )


def _role_entries(
    partition: Round74EventRunPartition,
    role: str,
) -> tuple[Round74EventRunPartitionEntry, ...]:
    selected_role = str(role)
    if selected_role not in ROUND74_EVENT_PARTITION_ROLES:
        raise ValueError("Round 74 model operator role differs")
    return tuple(entry for entry in partition.entries if entry.role == selected_role)


def round74_representative_window_policy() -> dict[str, object]:
    """Return the frozen target-blind bounded sampling contract."""

    policy: dict[str, object] = {
        "schema_version": ROUND74_EVENT_MODEL_WINDOW_SELECTION_SCHEMA_VERSION,
        "symbols": list(IMPACT_CAPTURE_SYMBOLS),
        "temporal_strata": ROUND74_EVENT_MODEL_TEMPORAL_STRATA,
        "windows_per_symbol_stratum": (ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM),
        "windows_per_symbol": ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL,
        "windows_per_run": ROUND74_EVENT_MODEL_WINDOWS_PER_RUN,
        "stratum_axis": "eligible_anchor_wall_time",
        "ranking": (
            "ascending_sha256_of_schema_run_symbol_stratum_decision_and_"
            "feature_window_identity"
        ),
        "target_label_or_outcome_used_for_selection": False,
        "model_output_used_for_selection": False,
        "selected_windows_restored_to_chronological_order": True,
        "underfilled_symbol_stratum_policy": "reject_run",
    }
    policy["policy_sha256"] = _canonical_sha256(policy)
    return policy


def _window_stratum(
    entry: Round74EventRunPartitionEntry,
    decision_wall_ns: int,
) -> int:
    start = int(entry.eligible_anchor_start_wall_ns)
    end = int(entry.eligible_anchor_end_wall_ns)
    decision = int(decision_wall_ns)
    if not start <= decision <= end:
        raise ValueError("Round 74 selected window is outside the eligible interval")
    span = end - start + 1
    return min(
        ROUND74_EVENT_MODEL_TEMPORAL_STRATA - 1,
        (decision - start) * ROUND74_EVENT_MODEL_TEMPORAL_STRATA // span,
    )


def _target_blind_window_rank(
    sample: Round74LabeledEventWindow,
    *,
    stratum: int,
) -> int:
    metadata = {
        "schema_version": ROUND74_EVENT_MODEL_WINDOW_SELECTION_SCHEMA_VERSION,
        "run_id": sample.run_id,
        "symbol": sample.symbol,
        "stratum": int(stratum),
        "decision_wall_ns": int(sample.decision_wall_ns),
        "feature_window_sha256": sample.feature_window_sha256,
    }
    return int(_canonical_sha256(metadata), 16)


def select_round74_representative_event_windows(
    windows: Iterable[Round74LabeledEventWindow],
    *,
    entry: Round74EventRunPartitionEntry,
) -> tuple[Round74LabeledEventWindow, ...]:
    """Select exact symbol/time coverage without inspecting any target value."""

    entry.validate()
    selected: dict[
        tuple[str, int],
        list[tuple[int, str, int, Round74LabeledEventWindow]],
    ] = {
        (symbol, stratum): []
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for stratum in range(ROUND74_EVENT_MODEL_TEMPORAL_STRATA)
    }
    observed = 0
    for sample in windows:
        observed += 1
        if (
            sample.run_id != entry.run_id
            or sample.role != entry.role
            or sample.symbol not in IMPACT_CAPTURE_SYMBOLS
        ):
            raise ValueError("Round 74 selected window identity differs")
        stratum = _window_stratum(entry, sample.decision_wall_ns)
        rank = _target_blind_window_rank(sample, stratum=stratum)
        heap = selected[(sample.symbol, stratum)]
        item = (-rank, sample.feature_window_sha256, observed, sample)
        if len(heap) < ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if observed == 0:
        raise ValueError(f"Round 74 model operator run {entry.run_id} has no samples")
    underfilled = tuple(
        f"{symbol}:{stratum}:{len(heap)}"
        for (symbol, stratum), heap in selected.items()
        if len(heap) != ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM
    )
    if underfilled:
        raise ValueError(
            "Round 74 representative window coverage is incomplete: "
            + ",".join(underfilled)
        )
    output = tuple(item[3] for heap in selected.values() for item in heap)
    if len(output) != ROUND74_EVENT_MODEL_WINDOWS_PER_RUN:
        raise RuntimeError("Round 74 representative window count differs")
    ordered = tuple(
        sorted(
            output,
            key=lambda sample: (
                sample.decision_monotonic_ns,
                sample.symbol,
                sample.anchor_index,
                sample.feature_window_sha256,
            ),
        )
    )
    if len({sample.feature_window_sha256 for sample in ordered}) != len(ordered):
        raise ValueError("Round 74 representative window identity is duplicated")
    return ordered


def round74_matched_representative_window_policy() -> dict[str, object]:
    """Return the endpoint-only ranking contract for representation comparison."""

    policy: dict[str, object] = {
        "schema_version": (
            ROUND74_EVENT_MODEL_MATCHED_WINDOW_SELECTION_SCHEMA_VERSION
        ),
        "representations": list(ROUND74_EVENT_WINDOW_REPRESENTATIONS),
        "symbols": list(IMPACT_CAPTURE_SYMBOLS),
        "temporal_strata": ROUND74_EVENT_MODEL_TEMPORAL_STRATA,
        "windows_per_symbol_stratum": (ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM),
        "windows_per_run": ROUND74_EVENT_MODEL_WINDOWS_PER_RUN,
        "stratum_axis": "eligible_anchor_wall_time",
        "ranking": (
            "ascending_sha256_of_schema_run_symbol_stratum_and_shared_endpoint"
        ),
        "feature_value_or_representation_digest_used_for_ranking": False,
        "target_value_or_outcome_used_for_ranking": False,
        "target_panel_used_only_for_exact_pair_validation": True,
        "underfilled_symbol_stratum_policy": "reject_run",
    }
    policy["policy_sha256"] = _canonical_sha256(policy)
    return policy


def _target_blind_pair_rank(
    pair: Round74MatchedEventWindowPair,
    *,
    stratum: int,
) -> int:
    sample = pair.per_symbol
    metadata = {
        "schema_version": (
            ROUND74_EVENT_MODEL_MATCHED_WINDOW_SELECTION_SCHEMA_VERSION
        ),
        "run_id": sample.run_id,
        "symbol": sample.symbol,
        "stratum": int(stratum),
        "decision_monotonic_ns": int(sample.decision_monotonic_ns),
        "decision_wall_ns": int(sample.decision_wall_ns),
        "endpoint_frame_index": int(sample.endpoint_frame_index),
        "endpoint_message_index": int(sample.endpoint_message_index),
    }
    return int(_canonical_sha256(metadata), 16)


def select_round74_representative_matched_event_windows(
    pairs: Iterable[Round74MatchedEventWindowPair],
    *,
    entry: Round74EventRunPartitionEntry,
) -> tuple[Round74MatchedEventWindowPair, ...]:
    """Select one target-blind endpoint panel shared by both representations."""

    entry.validate()
    selected: dict[
        tuple[str, int],
        list[tuple[int, str, int, Round74MatchedEventWindowPair]],
    ] = {
        (symbol, stratum): []
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for stratum in range(ROUND74_EVENT_MODEL_TEMPORAL_STRATA)
    }
    observed = 0
    for pair in pairs:
        pair.validate()
        sample = pair.per_symbol
        observed += 1
        if (
            sample.run_id != entry.run_id
            or sample.role != entry.role
            or sample.symbol not in IMPACT_CAPTURE_SYMBOLS
        ):
            raise ValueError("Round 74 matched window identity differs")
        stratum = _window_stratum(entry, sample.decision_wall_ns)
        rank = _target_blind_pair_rank(pair, stratum=stratum)
        endpoint_sha256 = pair.endpoint_sha256
        heap = selected[(sample.symbol, stratum)]
        item = (-rank, endpoint_sha256, observed, pair)
        if len(heap) < ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if observed == 0:
        raise ValueError(f"Round 74 model operator run {entry.run_id} has no pairs")
    underfilled = tuple(
        f"{symbol}:{stratum}:{len(heap)}"
        for (symbol, stratum), heap in selected.items()
        if len(heap) != ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM
    )
    if underfilled:
        raise ValueError(
            "Round 74 matched representative coverage is incomplete: "
            + ",".join(underfilled)
        )
    output = tuple(item[3] for heap in selected.values() for item in heap)
    if len(output) != ROUND74_EVENT_MODEL_WINDOWS_PER_RUN:
        raise RuntimeError("Round 74 matched representative count differs")
    ordered = tuple(
        sorted(
            output,
            key=lambda pair: (
                pair.per_symbol.decision_monotonic_ns,
                pair.per_symbol.symbol,
                pair.per_symbol.anchor_index,
                pair.endpoint_sha256,
            ),
        )
    )
    if len({pair.endpoint_sha256 for pair in ordered}) != len(ordered):
        raise ValueError("Round 74 matched endpoint identity is duplicated")
    return ordered


def assemble_round74_role_batches(
    store: object,
    *,
    partition: Round74EventRunPartition,
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    pretest_model_policy_sha256: str | None = None,
    test_unlock_sha256: str | None = None,
    window_representation: str = "per_symbol",
) -> tuple[Round74EventTrainingBatch, ...]:
    """Replay and assemble exactly one in-memory batch per capture run."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    entries = _role_entries(partition, role)
    selected_representation = str(window_representation)
    if selected_representation not in ROUND74_EVENT_WINDOW_REPRESENTATIONS:
        raise ValueError("Round 74 model operator window representation differs")
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
        samples = select_round74_representative_event_windows(
            iter_round74_labeled_event_windows(
                selected_store,
                partition=partition,
                run_id=entry.run_id,
                target_engine=engine,
                pretest_model_policy_sha256=pretest_model_policy_sha256,
                test_unlock_sha256=test_unlock_sha256,
                window_representation=selected_representation,
            ),
            entry=entry,
        )
        if not samples:
            raise ValueError(
                f"Round 74 model operator run {entry.run_id} has no samples"
            )
        batch = build_round74_event_training_batch(samples, scaler=scaler)
        batch.validate()
        if (
            batch.role != role
            or set(batch.run_id) != {entry.run_id}
            or batch.window_representation != selected_representation
        ):
            raise ValueError("Round 74 model operator batch crossed capture runs")
        batches.append(batch)
    return tuple(batches)


def _matched_batch_endpoint_sha256(batch: Round74EventTrainingBatch) -> str:
    return _canonical_sha256(
        {
            "role": batch.role,
            "partition_sha256": batch.partition_sha256,
            "scaler_sha256": batch.scaler_sha256,
            "run_id": list(batch.run_id),
            "symbol": list(batch.symbol),
            "decision_monotonic_ns": batch.decision_monotonic_ns.tolist(),
            "decision_wall_ns": batch.decision_wall_ns.tolist(),
            "endpoint_frame_index": batch.endpoint_frame_index.tolist(),
            "endpoint_message_index": batch.endpoint_message_index.tolist(),
            "anchor_index": batch.anchor_index.tolist(),
            "target_context_sha256": list(batch.target_context_sha256),
            "test_access_sha256": list(batch.test_access_sha256),
        }
    )


def _matched_batches_differ(
    left: Round74EventTrainingBatch,
    right: Round74EventTrainingBatch,
) -> bool:
    scalar_or_tuple_fields = (
        "role",
        "partition_sha256",
        "scaler_sha256",
        "run_id",
        "symbol",
        "target_context_sha256",
        "test_access_sha256",
    )
    identity_arrays = (
        "decision_monotonic_ns",
        "decision_wall_ns",
        "endpoint_frame_index",
        "endpoint_message_index",
        "anchor_index",
    )
    target_arrays = (
        "actual_entry_monotonic_ns",
        "actual_exit_monotonic_ns",
        "net_payoff_bps",
        "maximum_adverse_excursion_bps",
        "adverse_selection",
        "regime_unpredictability",
        "action_eligibility",
        "regime_unpredictability_eligibility",
    )
    return (
        any(getattr(left, name) != getattr(right, name) for name in scalar_or_tuple_fields)
        or any(
            not np.array_equal(getattr(left, name), getattr(right, name))
            for name in identity_arrays
        )
        or any(
            not np.array_equal(
                getattr(left, name),
                getattr(right, name),
                equal_nan=True,
            )
            for name in target_arrays
        )
    )


@dataclass(frozen=True)
class Round74MatchedRepresentationRoleBatches:
    """Endpoint- and target-identical batches for one development role."""

    role: str
    per_symbol: tuple[Round74EventTrainingBatch, ...]
    global_cross_asset: tuple[Round74EventTrainingBatch, ...]
    schema_version: str = ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION
            or self.role not in {"training", "tuning"}
            or not self.per_symbol
            or len(self.per_symbol) != len(self.global_cross_asset)
        ):
            raise ValueError("Round 74 matched role batches differ")
        for left, right in zip(
            self.per_symbol,
            self.global_cross_asset,
            strict=True,
        ):
            left.validate()
            right.validate()
            if (
                left.role != self.role
                or left.window_representation != "per_symbol"
                or right.window_representation != "global_cross_asset"
                or left.rows != ROUND74_EVENT_MODEL_WINDOWS_PER_RUN
                or right.rows != ROUND74_EVENT_MODEL_WINDOWS_PER_RUN
                or left.batch_sha256 == right.batch_sha256
                or _matched_batches_differ(left, right)
            ):
                raise ValueError("Round 74 matched role batch identity differs")
        run_ids = tuple(batch.run_id[0] for batch in self.per_symbol)
        first_wall_ns = tuple(
            int(batch.decision_wall_ns[0]) for batch in self.per_symbol
        )
        if (
            len(run_ids) != len(set(run_ids))
            or any(
                current <= prior
                for prior, current in zip(first_wall_ns, first_wall_ns[1:])
            )
        ):
            raise ValueError("Round 74 matched role capture run order differs")

    @property
    def matched_role_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "representations": list(ROUND74_EVENT_WINDOW_REPRESENTATIONS),
            "per_symbol_batch_sha256": [
                batch.batch_sha256 for batch in self.per_symbol
            ],
            "global_cross_asset_batch_sha256": [
                batch.batch_sha256 for batch in self.global_cross_asset
            ],
            "endpoint_panel_sha256": [
                _matched_batch_endpoint_sha256(batch) for batch in self.per_symbol
            ],
            "rows": sum(batch.rows for batch in self.per_symbol),
            "source_replay_passes_per_run": 1,
            "target_value_or_outcome_used_for_sampling": False,
            "test_role_accessed": False,
            "representative_window_policy": (
                round74_matched_representative_window_policy()
            ),
        }


def assemble_round74_matched_representation_role_batches(
    store: object,
    *,
    partition: Round74EventRunPartition,
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
) -> Round74MatchedRepresentationRoleBatches:
    """Replay one development role once into endpoint-identical batches."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    if role not in {"training", "tuning"}:
        raise ValueError("Round 74 matched role must be development data")
    entries = _role_entries(partition, role)
    expected_run_ids = tuple(entry.run_id for entry in entries)
    assemblies = dict(target_assembly_by_run_id)
    if set(assemblies) != set(expected_run_ids) or any(
        not isinstance(assemblies[run_id], Round74SourceTargetAssembly)
        for run_id in expected_run_ids
    ):
        raise ValueError("Round 74 matched target assembly panel differs")
    batches: dict[str, list[Round74EventTrainingBatch]] = {
        representation: []
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    }
    for entry in entries:
        assembly = assemblies[entry.run_id]
        pairs = select_round74_representative_matched_event_windows(
            iter_round74_matched_labeled_event_windows(
                selected_store,
                partition=partition,
                run_id=entry.run_id,
                target_engines={
                    representation: assembly.create_engine(anchors=())
                    for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
                },
            ),
            entry=entry,
        )
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS:
            samples = tuple(
                getattr(pair, representation)
                for pair in pairs
            )
            batch = build_round74_event_training_batch(samples, scaler=scaler)
            batch.validate()
            batches[representation].append(batch)
    result = Round74MatchedRepresentationRoleBatches(
        role=role,
        per_symbol=tuple(batches["per_symbol"]),
        global_cross_asset=tuple(batches["global_cross_asset"]),
    )
    result.validate()
    return result


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
            or len({batch.window_representation for batch in batches}) != 1
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
            "window_representation": self.training_batches[
                0
            ].window_representation,
            "training_batch_sha256": [
                batch.batch_sha256 for batch in self.training_batches
            ],
            "tuning_batch_sha256": [
                batch.batch_sha256 for batch in self.tuning_batches
            ],
            "training_rows": sum(batch.rows for batch in self.training_batches),
            "tuning_rows": sum(batch.rows for batch in self.tuning_batches),
            "source_database_access": "read_only",
            "representative_window_policy": (round74_representative_window_policy()),
            "training_source_replay_passes": 2,
            "tuning_source_replay_passes": 1,
            "overlapping_windows_persisted": False,
            "test_role_accessed": False,
        }


@dataclass(frozen=True)
class Round74PreparedTuningRoles:
    """Disjoint tuning batches with model-selection reuse made impossible."""

    subpartition: Round74TuningSubpartitionProtocol
    model_selection_batches: tuple[Round74EventTrainingBatch, ...]
    calibration_batches: tuple[Round74EventTrainingBatch, ...]
    policy_selection_batches: tuple[Round74EventTrainingBatch, ...]
    ai_qualification_batches: tuple[Round74EventTrainingBatch, ...] = ()
    schema_version: str = ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION

    def validate(self) -> None:
        self.subpartition.validate()
        groups = (
            (
                self.model_selection_batches,
                self.subpartition.model_selection_run_ids,
            ),
            (self.calibration_batches, self.subpartition.calibration_run_ids),
            (
                self.policy_selection_batches,
                self.subpartition.policy_selection_run_ids,
            ),
            (
                self.ai_qualification_batches,
                tuple(
                    getattr(
                        self.subpartition,
                        "ai_qualification_run_ids",
                        (),
                    )
                ),
            ),
        )
        batches = tuple(batch for group, _run_ids in groups for batch in group)
        if self.schema_version != ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION:
            raise ValueError("Round 74 prepared tuning roles schema differs")
        for batch in batches:
            batch.validate()
        if any(
            tuple(next(iter(set(batch.run_id))) for batch in group) != run_ids
            for group, run_ids in groups
        ):
            raise ValueError("Round 74 prepared tuning role assignment differs")
        if (
            any(batch.role != "tuning" for batch in batches)
            or len({next(iter(set(batch.run_id))) for batch in batches}) != len(batches)
            or {batch.partition_sha256 for batch in batches}
            != {self.subpartition.parent_partition_sha256}
            or len({batch.scaler_sha256 for batch in batches}) != 1
            or len({batch.window_representation for batch in batches}) != 1
        ):
            raise ValueError("Round 74 prepared tuning role identity differs")

    @property
    def role_assignment_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "subpartition_sha256": self.subpartition.subpartition_sha256,
            "window_representation": self.model_selection_batches[
                0
            ].window_representation,
            "model_selection_batch_sha256": [
                batch.batch_sha256 for batch in self.model_selection_batches
            ],
            "calibration_batch_sha256": [
                batch.batch_sha256 for batch in self.calibration_batches
            ],
            "policy_selection_batch_sha256": [
                batch.batch_sha256 for batch in self.policy_selection_batches
            ],
            "ai_qualification_batch_sha256": [
                batch.batch_sha256 for batch in self.ai_qualification_batches
            ],
            "model_selection_run_count": len(self.model_selection_batches),
            "calibration_run_count": len(self.calibration_batches),
            "policy_selection_run_count": len(self.policy_selection_batches),
            "ai_qualification_run_count": len(self.ai_qualification_batches),
            "cross_role_run_reuse_permitted": False,
            "sealed_test_role_accessed": False,
        }


def split_round74_prepared_tuning_roles(
    prepared: Round74PreparedDevelopmentData,
    *,
    subpartition: Round74TuningSubpartitionProtocol,
) -> Round74PreparedTuningRoles:
    """Assign a validated development panel to its frozen tuning subroles."""

    prepared.validate()
    subpartition.validate()
    if {
        batch.partition_sha256
        for batch in (*prepared.training_batches, *prepared.tuning_batches)
    } != {subpartition.parent_partition_sha256}:
        raise ValueError("Round 74 prepared tuning parent partition differs")
    return split_round74_tuning_batch_roles(
        prepared.tuning_batches,
        subpartition=subpartition,
    )


def split_round74_tuning_batch_roles(
    tuning_batches: Iterable[Round74EventTrainingBatch],
    *,
    subpartition: Round74TuningSubpartitionProtocol,
) -> Round74PreparedTuningRoles:
    """Assign every frozen tuning batch without requiring a legacy preparation."""

    subpartition.validate()
    selected_batches = tuple(tuning_batches)
    for batch in selected_batches:
        batch.validate()
    if (
        not selected_batches
        or any(batch.role != "tuning" for batch in selected_batches)
        or any(len(set(batch.run_id)) != 1 for batch in selected_batches)
    ):
        raise ValueError("Round 74 prepared tuning batch panel differs")
    expected_run_ids = (
        *subpartition.model_selection_run_ids,
        *subpartition.calibration_run_ids,
        *subpartition.policy_selection_run_ids,
        *tuple(getattr(subpartition, "ai_qualification_run_ids", ())),
    )
    observed_run_ids = tuple(
        next(iter(set(batch.run_id))) for batch in selected_batches
    )
    if observed_run_ids != expected_run_ids:
        raise ValueError("Round 74 prepared tuning chronology differs")
    model_end = len(subpartition.model_selection_run_ids)
    calibration_end = model_end + len(subpartition.calibration_run_ids)
    policy_end = calibration_end + len(subpartition.policy_selection_run_ids)
    selected = Round74PreparedTuningRoles(
        subpartition=subpartition,
        model_selection_batches=selected_batches[:model_end],
        calibration_batches=selected_batches[model_end:calibration_end],
        policy_selection_batches=selected_batches[calibration_end:policy_end],
        ai_qualification_batches=selected_batches[policy_end:],
    )
    selected.validate()
    return selected


def prepare_round74_development_data(
    store: object,
    *,
    partition: Round74EventRunPartition,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    chunk_rows: int = ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    window_representation: str = "per_symbol",
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
        window_representation=window_representation,
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
        window_representation=window_representation,
    )
    result = Round74PreparedDevelopmentData(
        scaler=scaler,
        training_batches=training,
        tuning_batches=tuning,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74PreparedMatchedDevelopmentData:
    """One-scaler, one-replay development panel for both representations."""

    scaler: Round74EventFeatureScaler
    training: Round74MatchedRepresentationRoleBatches
    tuning: Round74MatchedRepresentationRoleBatches
    schema_version: str = ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION

    def validate(self) -> None:
        self.training.validate()
        self.tuning.validate()
        batches = (
            *self.training.per_symbol,
            *self.training.global_cross_asset,
            *self.tuning.per_symbol,
            *self.tuning.global_cross_asset,
        )
        if (
            self.schema_version != ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION
            or not isinstance(self.scaler, Round74EventFeatureScaler)
            or self.training.role != "training"
            or self.tuning.role != "tuning"
            or {batch.scaler_sha256 for batch in batches}
            != {self.scaler.scaler_sha256}
            or len({batch.partition_sha256 for batch in batches}) != 1
        ):
            raise ValueError("Round 74 prepared matched development data differs")
        if int(self.training.per_symbol[-1].decision_wall_ns[-1]) >= int(
            self.tuning.per_symbol[0].decision_wall_ns[0]
        ):
            raise ValueError("Round 74 prepared matched chronology differs")

    def representation(self, value: str) -> Round74PreparedDevelopmentData:
        self.validate()
        selected = str(value)
        if selected not in ROUND74_EVENT_WINDOW_REPRESENTATIONS:
            raise ValueError("Round 74 prepared matched representation differs")
        result = Round74PreparedDevelopmentData(
            scaler=self.scaler,
            training_batches=getattr(self.training, selected),
            tuning_batches=getattr(self.tuning, selected),
        )
        result.validate()
        return result

    @property
    def preparation_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "scaler_sha256": self.scaler.scaler_sha256,
            "training_matched_role_sha256": self.training.matched_role_sha256,
            "tuning_matched_role_sha256": self.tuning.matched_role_sha256,
            "representations": list(ROUND74_EVENT_WINDOW_REPRESENTATIONS),
            "training_source_replay_passes": 2,
            "tuning_source_replay_passes": 1,
            "matched_representation_replay_passes_per_run": 1,
            "overlapping_windows_persisted": False,
            "test_role_accessed": False,
        }


def prepare_round74_matched_development_data(
    store: object,
    *,
    partition: Round74EventRunPartition,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    chunk_rows: int = ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74PreparedMatchedDevelopmentData:
    """Fit one training-only scaler and replay both representations once."""

    partition.validate()
    development_entries = tuple(
        entry for entry in partition.entries if entry.role != "test"
    )
    expected_run_ids = {entry.run_id for entry in development_entries}
    if set(target_assembly_by_run_id) != expected_run_ids:
        raise ValueError("Round 74 matched development assembly panel differs")
    scaler = fit_round74_cohort_feature_scaler(
        store,
        partition=partition,
        chunk_rows=chunk_rows,
        maximum_fit_rows=maximum_fit_rows,
    )
    roles = {
        role: assemble_round74_matched_representation_role_batches(
            store,
            partition=partition,
            scaler=scaler,
            role=role,
            target_assembly_by_run_id={
                entry.run_id: target_assembly_by_run_id[entry.run_id]
                for entry in development_entries
                if entry.role == role
            },
        )
        for role in ("training", "tuning")
    }
    result = Round74PreparedMatchedDevelopmentData(
        scaler=scaler,
        training=roles["training"],
        tuning=roles["tuning"],
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_EVENT_MODEL_FEATURE_CHUNK_ROWS",
    "ROUND74_EVENT_MODEL_MATCHED_WINDOW_SELECTION_SCHEMA_VERSION",
    "ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION",
    "ROUND74_EVENT_MODEL_TEMPORAL_STRATA",
    "ROUND74_EVENT_MODEL_WINDOWS_PER_RUN",
    "ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL",
    "ROUND74_EVENT_MODEL_WINDOWS_PER_SYMBOL_STRATUM",
    "ROUND74_EVENT_MODEL_WINDOW_SELECTION_SCHEMA_VERSION",
    "Round74MatchedRepresentationRoleBatches",
    "Round74PreparedDevelopmentData",
    "Round74PreparedMatchedDevelopmentData",
    "Round74PreparedTuningRoles",
    "assemble_round74_matched_representation_role_batches",
    "assemble_round74_role_batches",
    "fit_round74_cohort_feature_scaler",
    "iter_round74_training_feature_chunks",
    "prepare_round74_matched_development_data",
    "prepare_round74_development_data",
    "round74_matched_representative_window_policy",
    "round74_representative_window_policy",
    "select_round74_representative_matched_event_windows",
    "select_round74_representative_event_windows",
    "split_round74_prepared_tuning_roles",
    "split_round74_tuning_batch_roles",
]

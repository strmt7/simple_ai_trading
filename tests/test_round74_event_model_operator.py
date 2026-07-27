from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    fit_round74_event_feature_scaler,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    Round74EventToken,
    Round74ReplayObservation,
)
from simple_ai_trading.impact_absorption_store import ImpactAbsorptionStore
from simple_ai_trading.impact_absorption_target_assembly import (
    Round74SourceTargetAssembly,
)
from simple_ai_trading import round74_event_model_operator as subject


WALL = 1_800_000_000_000_000_000
NS = 1_000_000_000


def _partition() -> Round74EventRunPartition:
    entries = []
    for index, role in enumerate(("training", "tuning", "test")):
        start = WALL + index * 2_000 * NS
        anchor_start = start + (0 if index == 0 else 311 * NS)
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=f"{index + 1:032x}",
                role=role,
                capture_report_sha256=f"{index + 1:064x}",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + 1_000 * NS,
                eligible_anchor_start_wall_ns=anchor_start,
                eligible_anchor_end_wall_ns=start + 600 * NS,
            )
        )
    result = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="d" * 64,
    )
    result.validate()
    return result


def _token(index: int, *, wall_ns: int) -> Round74EventToken:
    values = [0.0] * len(ROUND74_EVENT_FEATURE_NAMES)
    values[0] = 1.0
    values[5] = 1.0
    values[-1] = float(index)
    result = Round74EventToken(
        symbol="BTCUSDT",
        event_type="aggTrade",
        frame_index=index,
        message_index=0,
        received_monotonic_ns=(index + 1) * NS,
        received_wall_ns=wall_ns,
        exchange_event_time_ms=1_000 + index,
        source_sequence_number=index,
        feature_values=tuple(values),
    )
    result.validate()
    return result


def _observation(
    index: int,
    *,
    wall_ns: int,
    include_token: bool = True,
) -> Round74ReplayObservation:
    token = _token(index, wall_ns=wall_ns) if include_token else None
    result = Round74ReplayObservation(
        symbol="BTCUSDT",
        event_type="aggTrade",
        frame_index=index,
        message_index=0,
        received_monotonic_ns=(index + 1) * NS,
        received_wall_ns=wall_ns,
        token=token,
        depth_state=None,
        depth_update_is_stale=False,
    )
    result.validate()
    return result


def _store() -> ImpactAbsorptionStore:
    return ImpactAbsorptionStore("unused.duckdb", read_only=True)


class _CaptureReportCursor:
    def __init__(self, row: tuple[str] | None) -> None:
        self.row = row

    def execute(
        self,
        _query: str,
        _parameters: list[str],
    ) -> _CaptureReportCursor:
        return self

    def fetchone(self) -> tuple[str] | None:
        return self.row


def test_capture_report_is_reconciled_before_feature_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    entry = partition.entries[0]
    store = _store()
    cursor = _CaptureReportCursor((entry.capture_report_sha256,))
    monkeypatch.setattr(store, "connect", lambda: cursor)
    subject._validate_capture_report(
        store,
        partition=partition,
        run_id=entry.run_id,
    )

    cursor.row = None
    with pytest.raises(ValueError, match="report is missing"):
        subject._validate_capture_report(
            store,
            partition=partition,
            run_id=entry.run_id,
        )
    cursor.row = ("f" * 64,)
    with pytest.raises(ValueError, match="capture report differs"):
        subject._validate_capture_report(
            store,
            partition=partition,
            run_id=entry.run_id,
        )


def test_scaler_stream_reads_unique_training_events_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    observed_runs: list[str] = []
    monkeypatch.setattr(subject, "_validate_capture_report", lambda *_args, **_kw: None)

    def observations(
        _store: object,
        *,
        run_id: str,
    ) -> tuple[Round74ReplayObservation, ...]:
        observed_runs.append(run_id)
        entry = partition.entry(run_id)
        return tuple(
            _observation(index, wall_ns=entry.capture_start_wall_ns + index + 1)
            for index in range(5)
        )

    monkeypatch.setattr(subject, "iter_round74_v10_event_observations", observations)
    chunks = tuple(
        subject.iter_round74_training_feature_chunks(
            _store(),
            partition=partition,
            chunk_rows=3,
        )
    )

    assert observed_runs == [partition.entries[0].run_id]
    assert [chunk.shape[0] for chunk in chunks] == [3, 2]
    np.testing.assert_array_equal(
        np.concatenate(chunks)[:, -1],
        np.arange(5, dtype=np.float64),
    )
    scaler = subject.fit_round74_cohort_feature_scaler(
        _store(),
        partition=partition,
        chunk_rows=3,
        maximum_fit_rows=4,
    )
    assert scaler.fit_input_rows == 5
    assert scaler.fit_sample_rows == 4


def test_feature_stream_rejects_bad_store_chunk_and_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    entry = partition.entries[0]
    monkeypatch.setattr(subject, "_validate_capture_report", lambda *_args, **_kw: None)
    with pytest.raises(TypeError, match="ImpactAbsorptionStore"):
        tuple(
            subject.iter_round74_training_feature_chunks(
                object(),
                partition=partition,
            )
        )
    with pytest.raises(ValueError, match="chunk size"):
        tuple(
            subject.iter_round74_training_feature_chunks(
                _store(),
                partition=partition,
                chunk_rows=1,
            )
        )

    monkeypatch.setattr(
        subject,
        "iter_round74_v10_event_observations",
        lambda *_args, **_kwargs: (
            _observation(
                0,
                wall_ns=entry.capture_start_wall_ns + 1,
                include_token=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="events are insufficient"):
        tuple(
            subject.iter_round74_training_feature_chunks(
                _store(),
                partition=partition,
            )
        )
    monkeypatch.setattr(
        subject,
        "iter_round74_v10_event_observations",
        lambda *_args, **_kwargs: (
            _observation(0, wall_ns=entry.capture_end_wall_ns + 1),
        ),
    )
    with pytest.raises(ValueError, match="outside its capture run"):
        tuple(
            subject.iter_round74_training_feature_chunks(
                _store(),
                partition=partition,
            )
        )


@dataclass
class _Sample:
    run_id: str


@dataclass
class _Batch:
    role: str
    run_id: tuple[str, ...]
    scaler_sha256: str
    partition_sha256: str
    batch_sha256: str
    rows: int = 1
    decision_wall_ns: np.ndarray | None = None

    def validate(self) -> None:
        if len(set(self.run_id)) != 1:
            raise ValueError("batch crossed runs")
        if self.decision_wall_ns is None:
            self.decision_wall_ns = np.asarray([WALL], dtype=np.int64)


def test_role_assembly_is_one_run_per_batch_and_test_is_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    scaler_values = np.zeros((3, len(ROUND74_EVENT_FEATURE_NAMES)))
    scaler_values[:, 0] = 1.0
    scaler_values[:, 5] = 1.0
    scaler = fit_round74_event_feature_scaler(
        scaler_values,
        partition_role="training",
    )
    assembly = object.__new__(Round74SourceTargetAssembly)
    monkeypatch.setattr(
        Round74SourceTargetAssembly,
        "create_engine",
        lambda _self, *, anchors: object(),
    )
    authorizations: list[tuple[str | None, str | None]] = []

    def labeled(
        _store: object,
        *,
        run_id: str,
        pretest_model_policy_sha256: str | None,
        test_unlock_sha256: str | None,
        **_kwargs: object,
    ) -> tuple[_Sample, ...]:
        authorizations.append((pretest_model_policy_sha256, test_unlock_sha256))
        return (_Sample(run_id),)

    monkeypatch.setattr(subject, "iter_round74_labeled_event_windows", labeled)

    def batch_for(samples: tuple[_Sample, ...], *, scaler: object) -> _Batch:
        run_id = samples[0].run_id
        entry = partition.entry(run_id)
        return _Batch(
            role=entry.role,
            run_id=(run_id,),
            scaler_sha256=scaler.scaler_sha256,
            partition_sha256=partition.partition_sha256,
            batch_sha256=f"{int(run_id, 16):064x}",
            decision_wall_ns=np.asarray(
                [entry.eligible_anchor_start_wall_ns],
                dtype=np.int64,
            ),
        )

    monkeypatch.setattr(
        subject,
        "build_round74_event_training_batch",
        batch_for,
    )
    training_id = partition.entries[0].run_id
    training = subject.assemble_round74_role_batches(
        _store(),
        partition=partition,
        scaler=scaler,
        role="training",
        target_assembly_by_run_id={training_id: assembly},
    )
    assert [batch.run_id for batch in training] == [(training_id,)]
    assert authorizations == [(None, None)]

    test_id = partition.entries[2].run_id
    with pytest.raises(ValueError, match="authorization is missing"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="test",
            target_assembly_by_run_id={test_id: assembly},
        )
    test = subject.assemble_round74_role_batches(
        _store(),
        partition=partition,
        scaler=scaler,
        role="test",
        target_assembly_by_run_id={test_id: assembly},
        pretest_model_policy_sha256="a" * 64,
        test_unlock_sha256="b" * 64,
    )
    assert [batch.run_id for batch in test] == [(test_id,)]
    assert authorizations[-1] == ("a" * 64, "b" * 64)


def test_role_assembly_rejects_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    with pytest.raises(ValueError, match="read-only"):
        tuple(
            subject.iter_round74_training_feature_chunks(
                ImpactAbsorptionStore(":memory:"),
                partition=partition,
            )
        )
    assembly = object.__new__(Round74SourceTargetAssembly)
    scaler_values = np.zeros((3, len(ROUND74_EVENT_FEATURE_NAMES)))
    scaler_values[:, 0] = 1.0
    scaler_values[:, 5] = 1.0
    scaler = fit_round74_event_feature_scaler(
        scaler_values,
        partition_role="training",
    )
    training_id = partition.entries[0].run_id
    with pytest.raises(ValueError, match="role differs"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="validation",
            target_assembly_by_run_id={},
        )
    with pytest.raises(ValueError, match="assembly panel differs"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="training",
            target_assembly_by_run_id={},
        )
    with pytest.raises(ValueError, match="received test authorization"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="training",
            target_assembly_by_run_id={training_id: assembly},
            pretest_model_policy_sha256="a" * 64,
            test_unlock_sha256="b" * 64,
        )
    monkeypatch.setattr(
        Round74SourceTargetAssembly,
        "create_engine",
        lambda _self, *, anchors: object(),
    )
    monkeypatch.setattr(
        subject,
        "iter_round74_labeled_event_windows",
        lambda *_args, **_kwargs: (),
    )
    with pytest.raises(ValueError, match="has no samples"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="training",
            target_assembly_by_run_id={training_id: assembly},
        )
    monkeypatch.setattr(
        subject,
        "iter_round74_labeled_event_windows",
        lambda *_args, **_kwargs: (_Sample(training_id),),
    )
    monkeypatch.setattr(
        subject,
        "build_round74_event_training_batch",
        lambda *_args, **_kwargs: _Batch(
            role="tuning",
            run_id=(training_id,),
            scaler_sha256=scaler.scaler_sha256,
            partition_sha256=partition.partition_sha256,
            batch_sha256="a" * 64,
        ),
    )
    with pytest.raises(ValueError, match="crossed capture runs"):
        subject.assemble_round74_role_batches(
            _store(),
            partition=partition,
            scaler=scaler,
            role="training",
            target_assembly_by_run_id={training_id: assembly},
        )


def test_development_preparation_excludes_test_and_binds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    scaler_values = np.zeros((3, len(ROUND74_EVENT_FEATURE_NAMES)))
    scaler_values[:, 0] = 1.0
    scaler_values[:, 5] = 1.0
    scaler = fit_round74_event_feature_scaler(
        scaler_values,
        partition_role="training",
    )
    assembly = object.__new__(Round74SourceTargetAssembly)
    development_entries = partition.entries[:2]
    observed_roles: list[str] = []
    monkeypatch.setattr(
        subject,
        "fit_round74_cohort_feature_scaler",
        lambda *_args, **_kwargs: scaler,
    )

    def assemble(
        _store: object,
        *,
        role: str,
        target_assembly_by_run_id: dict[str, object],
        **_kwargs: object,
    ) -> tuple[_Batch, ...]:
        observed_roles.append(role)
        entry = next(item for item in development_entries if item.role == role)
        assert set(target_assembly_by_run_id) == {entry.run_id}
        return (
            _Batch(
                role=role,
                run_id=(entry.run_id,),
                scaler_sha256=scaler.scaler_sha256,
                partition_sha256=partition.partition_sha256,
                batch_sha256=f"{int(entry.run_id, 16):064x}",
                decision_wall_ns=np.asarray(
                    [entry.eligible_anchor_start_wall_ns],
                    dtype=np.int64,
                ),
            ),
        )

    monkeypatch.setattr(subject, "assemble_round74_role_batches", assemble)
    prepared = subject.prepare_round74_development_data(
        _store(),
        partition=partition,
        target_assembly_by_run_id={
            entry.run_id: assembly for entry in development_entries
        },
        chunk_rows=7,
        maximum_fit_rows=11,
    )

    assert observed_roles == ["training", "tuning"]
    assert prepared.as_dict()["test_role_accessed"] is False
    assert len(prepared.preparation_sha256) == 64
    assert prepared.as_dict()["training_rows"] == 1
    assert prepared.as_dict()["tuning_rows"] == 1
    with pytest.raises(ValueError, match="data differs"):
        replace(prepared, schema_version="wrong").validate()
    with pytest.raises(ValueError, match="identity differs"):
        replace(
            prepared,
            training_batches=(replace(prepared.training_batches[0], role="tuning"),),
        ).validate()
    with pytest.raises(ValueError, match="run is repeated"):
        replace(
            prepared,
            tuning_batches=(
                replace(
                    prepared.tuning_batches[0],
                    run_id=prepared.training_batches[0].run_id,
                ),
            ),
        ).validate()
    with pytest.raises(ValueError, match="chronology differs"):
        replace(
            prepared,
            tuning_batches=(
                replace(
                    prepared.tuning_batches[0],
                    decision_wall_ns=prepared.training_batches[
                        0
                    ].decision_wall_ns.copy(),
                ),
            ),
        ).validate()


def test_development_preparation_rejects_identity_drift() -> None:
    partition = _partition()
    assembly = object.__new__(Round74SourceTargetAssembly)
    with pytest.raises(ValueError, match="development assembly panel"):
        subject.prepare_round74_development_data(
            _store(),
            partition=partition,
            target_assembly_by_run_id={
                entry.run_id: assembly for entry in partition.entries
            },
        )

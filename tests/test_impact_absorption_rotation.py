from __future__ import annotations

import asyncio
import hashlib
import json
import time

import pytest

from simple_ai_trading.impact_absorption_rotation import (
    ROUND73_ROTATION_BATCH_TABLE,
    ROUND73_ROTATION_LEASE_TABLE,
    ROUND73_ROTATION_SCHEMA_VERSION,
    ROUND73_ROTATION_SEGMENT_TABLE,
    ROUND73_ROTATION_V1_CONTRACT_SHA256,
    ROUND73_ROTATION_V1_SCHEMA_VERSION,
    Round73CorpusRotationConfig,
    _acquire_lease,
    _release_lease,
    audit_round73_rotation_batch,
    run_round73_corpus_rotation,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
)


class _Attempt:
    def __init__(self, run_id: str, *, qualified: bool) -> None:
        self.run_id = run_id
        self.status = "completed" if qualified else "failed"
        self.capture_schema_version = IMPACT_CAPTURE_V9_SCHEMA_VERSION
        self.qualification_passed = qualified

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "capture_schema_version": self.capture_schema_version,
            "qualification_passed": self.qualification_passed,
        }


class _Supervisor:
    def __init__(self, run_id: str, *, qualified: bool = True) -> None:
        self.status = "completed" if qualified else "failed"
        self.capture_schema_version = IMPACT_CAPTURE_V9_SCHEMA_VERSION
        self.qualification_passed = qualified
        self.selected_run_id = run_id if qualified else ""
        self.attempt_count = 1
        self.reconnect_count = 0
        self.attempts = (_Attempt(run_id, qualified=qualified),)
        self.terminal_error = "" if qualified else "capture gate failed"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "capture_schema_version": self.capture_schema_version,
            "qualification_passed": self.qualification_passed,
            "selected_run_id": self.selected_run_id,
            "attempt_count": self.attempt_count,
            "reconnect_count": self.reconnect_count,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "terminal_error": self.terminal_error,
        }


class _Manifest:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.manifest_sha256 = hashlib.sha256(run_id.encode("ascii")).hexdigest()


class _Audit:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.manifest_sha256 = hashlib.sha256(run_id.encode("ascii")).hexdigest()
        self.passed = True


def _config(tmp_path, *, segments: int) -> Round73CorpusRotationConfig:
    return Round73CorpusRotationConfig(
        database=str(tmp_path / "rotation.duckdb"),
        segment_count=segments,
        memory_limit="1GB",
        database_threads=1,
    )


def test_rotation_defers_replay_until_every_capture_is_terminal(tmp_path) -> None:
    config = _config(tmp_path, segments=2)
    events: list[str] = []
    run_ids = iter(("a" * 32, "b" * 32))

    async def capture(capture_config):
        assert capture_config.duration_seconds == 3_600
        assert capture_config.maximum_reconnects == 0
        assert capture_config.schema_version == IMPACT_CAPTURE_V9_SCHEMA_VERSION
        run_id = next(run_ids)
        events.append(f"capture:{run_id}")
        return _Supervisor(run_id)

    def index(_database, *, run_id, **_kwargs):
        events.append(f"index:{run_id}")
        return _Manifest(run_id)

    def audit(_database, *, run_id, **_kwargs):
        events.append(f"audit:{run_id}")
        return _Audit(run_id)

    report = asyncio.run(
        run_round73_corpus_rotation(
            config,
            capture_function=capture,
            index_function=index,
            audit_function=audit,
        )
    )

    assert events == [
        f"capture:{'a' * 32}",
        f"capture:{'b' * 32}",
        f"index:{'a' * 32}",
        f"audit:{'a' * 32}",
        f"index:{'b' * 32}",
        f"audit:{'b' * 32}",
    ]
    assert report.status == "completed"
    assert report.qualified_capture_segment_count == 2
    assert report.indexed_segment_count == 2
    assert report.as_dict()["orders_submitted"] is False
    with ImpactAbsorptionStore(config.database, read_only=True) as store:
        connection = store.connect()
        batch = connection.execute(
            f"SELECT schema_version, status, report_json, report_sha256 "
            f"FROM {ROUND73_ROTATION_BATCH_TABLE} WHERE batch_id = ?",
            [report.batch_id],
        ).fetchone()
        segment_count = connection.execute(
            f"SELECT count(*) FROM {ROUND73_ROTATION_SEGMENT_TABLE} WHERE batch_id = ?",
            [report.batch_id],
        ).fetchone()[0]
        lease_count = connection.execute(
            f"SELECT count(*) FROM {ROUND73_ROTATION_LEASE_TABLE}"
        ).fetchone()[0]
    assert batch[0] == ROUND73_ROTATION_SCHEMA_VERSION
    assert batch[1] == "completed"
    assert hashlib.sha256(str(batch[2]).encode("ascii")).hexdigest() == batch[3]
    assert segment_count == 2
    assert lease_count == 0
    journal_audit = audit_round73_rotation_batch(
        config.database,
        batch_id=report.batch_id,
        memory_limit="1GB",
        threads=1,
    )
    assert journal_audit.passed is True
    assert journal_audit.segment_count == 2
    assert journal_audit.deeply_audited_manifest_count == 0


def test_rotation_stops_capture_but_indexes_prior_qualified_segment(tmp_path) -> None:
    config = _config(tmp_path, segments=3)
    events: list[str] = []
    capture_count = 0

    async def capture(_capture_config):
        nonlocal capture_count
        capture_count += 1
        events.append(f"capture:{capture_count}")
        return _Supervisor(
            ("a" if capture_count == 1 else "b") * 32,
            qualified=capture_count == 1,
        )

    def index(_database, *, run_id, **_kwargs):
        events.append(f"index:{run_id}")
        return _Manifest(run_id)

    def audit(_database, *, run_id, **_kwargs):
        events.append(f"audit:{run_id}")
        return _Audit(run_id)

    report = asyncio.run(
        run_round73_corpus_rotation(
            config,
            capture_function=capture,
            index_function=index,
            audit_function=audit,
        )
    )

    assert events == [
        "capture:1",
        "capture:2",
        f"index:{'a' * 32}",
        f"audit:{'a' * 32}",
    ]
    assert report.status == "failed"
    assert report.failed_phase == "capture"
    assert report.qualified_capture_segment_count == 1
    assert report.indexed_segment_count == 1
    assert len(report.segments) == 2


def test_rotation_keeps_coordinator_responsive_during_exact_replay(tmp_path) -> None:
    config = _config(tmp_path, segments=1)
    index_started = False

    async def capture(_capture_config):
        return _Supervisor("f" * 32)

    def index(_database, *, run_id, **_kwargs):
        nonlocal index_started
        index_started = True
        time.sleep(0.1)
        return _Manifest(run_id)

    def audit(_database, *, run_id, **_kwargs):
        return _Audit(run_id)

    async def scenario():
        task = asyncio.create_task(
            run_round73_corpus_rotation(
                config,
                capture_function=capture,
                index_function=index,
                audit_function=audit,
            )
        )
        while not index_started:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.01)
        assert task.done() is False
        return await task

    report = asyncio.run(scenario())

    assert report.status == "completed"


def test_rotation_recovers_qualified_unindexed_run_before_capture(tmp_path) -> None:
    config = _config(tmp_path, segments=0)
    run_id = "c" * 32
    with ImpactAbsorptionStore(config.database) as store:
        store.start_run(
            run_id=run_id,
            started_wall_ns=1,
            started_monotonic_ns=1,
            config={"mode": "qualification"},
            schema_version=IMPACT_CAPTURE_V9_SCHEMA_VERSION,
        )
        store.finish_run(run_id=run_id, status="completed", ended_wall_ns=2)
        store.record_report(
            run_id=run_id,
            report={
                "schema_version": IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
                "run_id": run_id,
                "qualification_passed": True,
            },
            recorded_at_wall_ns=3,
        )
    events: list[str] = []

    async def capture(_capture_config):
        raise AssertionError("recovery-only rotation must not capture")

    def index(_database, *, run_id, **_kwargs):
        events.append(f"index:{run_id}")
        return _Manifest(run_id)

    def audit(_database, *, run_id, **_kwargs):
        events.append(f"audit:{run_id}")
        return _Audit(run_id)

    report = asyncio.run(
        run_round73_corpus_rotation(
            config,
            capture_function=capture,
            index_function=index,
            audit_function=audit,
        )
    )

    assert events == [f"index:{run_id}", f"audit:{run_id}"]
    assert report.status == "completed"
    assert report.recovered_segment_count == 1
    assert report.qualified_capture_segment_count == 0
    assert report.indexed_segment_count == 1


def test_rotation_rejects_second_live_lease_before_capture(tmp_path) -> None:
    config = _config(tmp_path, segments=1)
    owner_id = "d" * 32
    _acquire_lease(config, owner_id=owner_id, now_wall_ns=time.time_ns())
    capture_called = False

    async def capture(_capture_config):
        nonlocal capture_called
        capture_called = True
        return _Supervisor("e" * 32)

    try:
        with pytest.raises(RuntimeError, match="owns the active lease"):
            asyncio.run(run_round73_corpus_rotation(config, capture_function=capture))
    finally:
        _release_lease(config, owner_id=owner_id)

    assert capture_called is False


def test_rotation_batch_audit_rejects_report_tampering(tmp_path) -> None:
    config = _config(tmp_path, segments=0)
    report = asyncio.run(run_round73_corpus_rotation(config))
    with ImpactAbsorptionStore(config.database) as store:
        store.connect().execute(
            f"UPDATE {ROUND73_ROTATION_BATCH_TABLE} SET report_json = '{{}}' "
            "WHERE batch_id = ?",
            [report.batch_id],
        )

    audit = audit_round73_rotation_batch(
        config.database,
        batch_id=report.batch_id,
        memory_limit="1GB",
        threads=1,
    )

    assert audit.passed is False
    assert any(error.startswith("journal:ValueError") for error in audit.errors)


def test_rotation_batch_audit_preserves_historical_v1_protocol(tmp_path) -> None:
    config = _config(tmp_path, segments=0)
    report = asyncio.run(run_round73_corpus_rotation(config))
    identity = report._identity()
    identity["schema_version"] = ROUND73_ROTATION_V1_SCHEMA_VERSION
    identity["contract_sha256"] = ROUND73_ROTATION_V1_CONTRACT_SHA256
    report_text = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    report_sha = hashlib.sha256(report_text.encode("ascii")).hexdigest()
    with ImpactAbsorptionStore(config.database) as store:
        store.connect().execute(
            f"""
            UPDATE {ROUND73_ROTATION_BATCH_TABLE}
            SET schema_version = ?, contract_sha256 = ?,
                report_json = ?, report_sha256 = ?
            WHERE batch_id = ?
            """,
            [
                ROUND73_ROTATION_V1_SCHEMA_VERSION,
                ROUND73_ROTATION_V1_CONTRACT_SHA256,
                report_text,
                report_sha,
                report.batch_id,
            ],
        )

    audit = audit_round73_rotation_batch(
        config.database,
        batch_id=report.batch_id,
        memory_limit="1GB",
        threads=1,
    )

    assert audit.passed is True
    assert audit.runner_schema_version == ROUND73_ROTATION_V1_SCHEMA_VERSION
    assert audit.runner_contract_sha256 == ROUND73_ROTATION_V1_CONTRACT_SHA256


@pytest.mark.parametrize("segment_count", [-1, 169, True, 1.5])
def test_rotation_rejects_unbounded_or_noninteger_segment_counts(
    tmp_path,
    segment_count,
) -> None:
    config = _config(tmp_path, segments=segment_count)

    with pytest.raises(ValueError, match="segment count"):
        config.validate()

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from threading import Event, Thread

import pytest

from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveError,
)
from simple_ai_trading.polymarket_runtime_control import (
    POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
    PolymarketRuntimeControl,
    PolymarketRuntimeControlService,
    PolymarketRuntimeLeaseInterlock,
)
from simple_ai_trading import polymarket_runtime_control as runtime_control_module


def test_runtime_control_enforces_one_lease_and_persistent_stop(tmp_path) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")

    assert control.snapshot().state == "stopped"
    lease = control.acquire(owner_process_id=101)
    running = control.snapshot()
    assert running.state == "running"
    assert running.lease_id == lease
    assert running.owner_process_id == 101
    assert control.heartbeat(lease) is True
    control.assert_opening_allowed(lease)

    with pytest.raises(PolymarketLiveBlocked, match="already active"):
        control.acquire(owner_process_id=202)

    requested = control.request_stop(reason="operator_stop")
    assert requested.state == "stop_requested"
    assert requested.stop_epoch == 1
    assert control.heartbeat(lease) is False
    with pytest.raises(PolymarketLiveBlocked, match="does not permit"):
        control.assert_opening_allowed(lease)
    with pytest.raises(PolymarketLiveBlocked, match="Stop completion"):
        control.acquire(owner_process_id=202)

    control.release(lease, reason="owned_exposure_closed")
    stopped = control.snapshot()
    assert stopped.state == "stopped"
    assert stopped.lease_id == ""
    assert stopped.stop_epoch == 1


def test_pause_is_durable_ordered_and_keeps_the_heartbeat_alive(tmp_path) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
    lease = control.acquire(owner_process_id=101)
    entered = Event()
    release = Event()
    paused = Event()

    def hold_submission() -> None:
        with control.submission_guard(lease):
            entered.set()
            assert release.wait(timeout=5)

    def request_pause() -> None:
        control.set_paused(True, reason="operator_pause")
        paused.set()

    holder = Thread(target=hold_submission)
    pauser = Thread(target=request_pause)
    holder.start()
    assert entered.wait(timeout=5)
    pauser.start()
    assert not paused.wait(timeout=0.1)
    release.set()
    holder.join(timeout=5)
    pauser.join(timeout=5)

    snapshot = control.snapshot()
    assert snapshot.paused is True
    assert snapshot.pause_epoch == 1
    assert control.heartbeat(lease) is True
    with pytest.raises(PolymarketLiveBlocked, match="does not permit"):
        control.assert_opening_allowed(lease)

    resumed = control.set_paused(False, reason="operator_resume")
    assert resumed.paused is False
    assert resumed.pause_epoch == 2
    control.assert_opening_allowed(lease)


def test_stop_is_ordered_after_an_inflight_submission_guard(tmp_path) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
    lease = control.acquire(owner_process_id=101)
    entered = Event()
    release = Event()
    stop_completed = Event()

    def hold_submission() -> None:
        with control.submission_guard(lease):
            entered.set()
            assert release.wait(timeout=5)

    def request_stop() -> None:
        control.request_stop(reason="operator_stop")
        stop_completed.set()

    holder = Thread(target=hold_submission)
    stopper = Thread(target=request_stop)
    holder.start()
    assert entered.wait(timeout=5)
    stopper.start()
    assert not stop_completed.wait(timeout=0.1)
    release.set()
    holder.join(timeout=5)
    stopper.join(timeout=5)

    assert not holder.is_alive()
    assert not stopper.is_alive()
    assert stop_completed.is_set()
    assert control.snapshot().state == "stop_requested"
    with pytest.raises(PolymarketLiveBlocked, match="does not permit"):
        with control.submission_guard(lease):
            pass


def test_owned_close_routines_are_cross_process_serialized(tmp_path) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
    first_entered = Event()
    first_release = Event()
    second_entered = Event()

    def first_close() -> None:
        with control.close_guard(timeout_seconds=5):
            first_entered.set()
            assert first_release.wait(timeout=5)

    def second_close() -> None:
        with control.close_guard(timeout_seconds=5):
            second_entered.set()

    first = Thread(target=first_close)
    second = Thread(target=second_close)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert not second_entered.wait(timeout=0.1)
    first_release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()


def test_stale_stop_requires_closed_exposure_and_expired_heartbeat(tmp_path) -> None:
    now = [1_800_000_000_000]
    control = PolymarketRuntimeControl(
        tmp_path / "ownership.sqlite3",
        maximum_heartbeat_age_ms=5_000,
        clock_ms=lambda: now[0],
    )
    lease = control.acquire(owner_process_id=101)
    control.request_stop(reason="operator_stop")

    with pytest.raises(PolymarketLiveBlocked, match="owned exposure"):
        control.complete_stale_stop(exposure_closed=False)
    assert control.complete_stale_stop(exposure_closed=True) is False

    now[0] += 5_001
    assert control.complete_stale_stop(exposure_closed=True) is True
    assert control.snapshot().state == "stopped"
    control.release(lease, reason="late_process_exit")
    assert control.snapshot().state == "stopped"


def test_ownerless_stop_stays_latched_until_exposure_is_proven_closed(
    tmp_path,
) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")

    requested = control.request_stop(reason="operator_stop")

    assert requested.state == "stop_requested"
    assert requested.lease_id == ""
    with pytest.raises(PolymarketLiveBlocked, match="Stop completion"):
        control.acquire(owner_process_id=101)
    with pytest.raises(PolymarketLiveBlocked, match="owned exposure"):
        control.complete_stale_stop(exposure_closed=False)
    assert control.complete_stale_stop(exposure_closed=True) is True
    assert control.snapshot().state == "stopped"


def test_runtime_control_detects_record_tampering(tmp_path) -> None:
    path = tmp_path / "ownership.sqlite3"
    control = PolymarketRuntimeControl(path)
    control.snapshot()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE polymarket_runtime_control
            SET owner_process_id = 999
            WHERE singleton = 1
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PolymarketLiveError, match="record differs"):
        control.snapshot()


def test_runtime_lease_interlock_delegates_both_checks(tmp_path) -> None:
    control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
    lease = control.acquire(owner_process_id=101)
    interlock = PolymarketRuntimeLeaseInterlock(control, lease)

    interlock.assert_opening_allowed()
    with interlock.submission_guard():
        interlock.assert_opening_allowed()

    control.request_stop(reason="operator_stop")
    with pytest.raises(PolymarketLiveBlocked, match="does not permit"):
        interlock.assert_opening_allowed()


def test_control_service_turns_external_stop_into_shutdown(tmp_path) -> None:
    async def scenario() -> None:
        control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
        lease = control.acquire(owner_process_id=101)
        service = PolymarketRuntimeControlService(
            control,
            lease_id=lease,
            heartbeat_interval_seconds=1,
        )
        services_stop = asyncio.Event()
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            service.run(services_stop, request_stop=shutdown.set)
        )
        await asyncio.sleep(0)
        control.request_stop(reason="operator_stop")
        await asyncio.wait_for(shutdown.wait(), timeout=2)
        await asyncio.wait_for(task, timeout=2)
        control.release(lease, reason="owned_exposure_closed")

    asyncio.run(scenario())


def test_control_service_applies_external_pause_and_resume(tmp_path) -> None:
    async def scenario() -> None:
        control = PolymarketRuntimeControl(tmp_path / "ownership.sqlite3")
        lease = control.acquire(owner_process_id=101)
        service = PolymarketRuntimeControlService(
            control,
            lease_id=lease,
            heartbeat_interval_seconds=1,
            stop_poll_interval_seconds=0.1,
        )
        services_stop = asyncio.Event()
        shutdown = asyncio.Event()
        paused = asyncio.Event()
        resumed = asyncio.Event()
        task = asyncio.create_task(
            service.run(
                services_stop,
                request_stop=shutdown.set,
                pause_runtime=paused.set,
                resume_runtime=resumed.set,
            )
        )
        await asyncio.sleep(0)
        control.set_paused(True, reason="operator_pause")
        await asyncio.wait_for(paused.wait(), timeout=2)
        control.set_paused(False, reason="operator_resume")
        await asyncio.wait_for(resumed.wait(), timeout=2)
        assert shutdown.is_set() is False
        services_stop.set()
        await asyncio.wait_for(task, timeout=2)
        control.release(lease, reason="test_complete")

    asyncio.run(scenario())


def _write_v1_runtime_control(path, legacy, *, record_sha256=None) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE polymarket_runtime_control (
                singleton INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL,
                state TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                owner_process_id INTEGER NOT NULL,
                started_at_ms INTEGER NOT NULL,
                heartbeat_at_ms INTEGER NOT NULL,
                stop_epoch INTEGER NOT NULL,
                stop_requested_at_ms INTEGER NOT NULL,
                stop_reason TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO polymarket_runtime_control VALUES "
            "(1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                *legacy.values(),
                record_sha256 or runtime_control_module._canonical_sha256(legacy),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_runtime_control_migrates_a_hash_valid_v1_record(tmp_path) -> None:
    path = tmp_path / "ownership.sqlite3"
    legacy = {
        "schema_version": "polymarket-runtime-control-v1",
        "state": "stopped",
        "lease_id": "",
        "owner_process_id": 0,
        "started_at_ms": 0,
        "heartbeat_at_ms": 0,
        "stop_epoch": 3,
        "stop_requested_at_ms": 123,
        "stop_reason": "legacy_stop_complete",
        "updated_at_ms": 456,
    }
    _write_v1_runtime_control(path, legacy)

    snapshot = PolymarketRuntimeControl(path).snapshot()

    assert snapshot.state == "stopped"
    assert snapshot.stop_epoch == 3
    assert snapshot.paused is False
    assert snapshot.pause_epoch == 0
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT schema_version, pause_reason FROM polymarket_runtime_control"
        ).fetchone()
    finally:
        connection.close()
    assert row == (POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION, "migrated_from_v1")


def test_runtime_control_rejects_tampered_v1_without_partial_migration(
    tmp_path,
) -> None:
    path = tmp_path / "ownership.sqlite3"
    legacy = {
        "schema_version": "polymarket-runtime-control-v1",
        "state": "running",
        "lease_id": "legacy-lease",
        "owner_process_id": 101,
        "started_at_ms": 100,
        "heartbeat_at_ms": 200,
        "stop_epoch": 0,
        "stop_requested_at_ms": 0,
        "stop_reason": "",
        "updated_at_ms": 200,
    }
    _write_v1_runtime_control(path, legacy, record_sha256="0" * 64)

    with pytest.raises(PolymarketLiveError, match="v1 record differs"):
        PolymarketRuntimeControl(path).snapshot()

    connection = sqlite3.connect(path)
    try:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(polymarket_runtime_control)"
            ).fetchall()
        }
    finally:
        connection.close()
    assert "paused" not in columns


def test_resume_rejects_stale_heartbeat_and_preserves_pause(tmp_path) -> None:
    now = [100_000]
    control = PolymarketRuntimeControl(
        tmp_path / "ownership.sqlite3",
        maximum_heartbeat_age_ms=5_000,
        clock_ms=lambda: now[0],
    )
    control.acquire(owner_process_id=101)
    control.set_paused(True, reason="operator_pause")
    now[0] += 5_001

    with pytest.raises(PolymarketLiveBlocked, match="heartbeat is stale"):
        control.set_paused(False, reason="operator_resume")

    snapshot = control.snapshot()
    assert snapshot.paused is True
    assert snapshot.pause_epoch == 1


@pytest.mark.parametrize("clock_delta", [5_001, -5_001])
@pytest.mark.parametrize("paused", [False, True])
def test_invalid_heartbeat_latches_stop_across_restart(
    tmp_path, clock_delta, paused
) -> None:
    now = [100_000]
    path = tmp_path / "ownership.sqlite3"
    control = PolymarketRuntimeControl(
        path, maximum_heartbeat_age_ms=5_000, clock_ms=lambda: now[0]
    )
    lease = control.acquire(owner_process_id=101)
    if paused:
        control.set_paused(True, reason="operator_pause")
    previous = control.snapshot()
    now[0] += clock_delta

    assert control.heartbeat(lease) is False
    stopped = control.snapshot()
    assert stopped.state == "stop_requested"
    assert stopped.paused is True
    assert stopped.stop_reason == "heartbeat_invalid_or_expired"
    assert stopped.heartbeat_at_ms == previous.heartbeat_at_ms
    assert stopped.stop_epoch == previous.stop_epoch + 1
    assert stopped.lease_id == lease

    # A clock correction or a fresh controller must not revive the old worker.
    now[0] = 100_001
    reopened = PolymarketRuntimeControl(
        path, maximum_heartbeat_age_ms=5_000, clock_ms=lambda: now[0]
    )
    assert reopened.heartbeat(lease) is False
    assert reopened.snapshot() == stopped
    with pytest.raises(PolymarketLiveBlocked):
        reopened.assert_opening_allowed(lease)
    with pytest.raises(PolymarketLiveBlocked):
        reopened.set_paused(False, reason="operator_resume")
    with pytest.raises(PolymarketLiveBlocked):
        reopened.acquire(owner_process_id=202)


def test_heartbeat_age_boundary_and_foreign_lease(tmp_path) -> None:
    now = [100_000]
    control = PolymarketRuntimeControl(
        tmp_path / "ownership.sqlite3",
        maximum_heartbeat_age_ms=5_000,
        clock_ms=lambda: now[0],
    )
    lease = control.acquire()
    initial = control.snapshot()
    now[0] += 5_000
    assert control.heartbeat("foreign") is False
    assert control.snapshot() == initial
    assert control.heartbeat(lease) is True
    control.assert_opening_allowed(lease)


def test_expired_control_service_requests_shutdown_without_renewal(tmp_path) -> None:
    async def scenario() -> None:
        now = [100_000]
        control = PolymarketRuntimeControl(
            tmp_path / "ownership.sqlite3",
            maximum_heartbeat_age_ms=5_000,
            clock_ms=lambda: now[0],
        )
        lease = control.acquire()
        now[0] += 5_001
        shutdown = asyncio.Event()
        service = PolymarketRuntimeControlService(control, lease_id=lease)
        await asyncio.wait_for(
            service.run(asyncio.Event(), request_stop=shutdown.set), timeout=2
        )
        assert shutdown.is_set()
        assert control.snapshot().state == "stop_requested"

    asyncio.run(scenario())


def test_expiry_is_ordered_after_inflight_submission(tmp_path) -> None:
    now = [100_000]
    control = PolymarketRuntimeControl(
        tmp_path / "ownership.sqlite3",
        maximum_heartbeat_age_ms=5_000,
        clock_ms=lambda: now[0],
    )
    lease = control.acquire()
    entered, release, renewed = Event(), Event(), Event()
    outcomes = []

    def hold_submission() -> None:
        with control.submission_guard(lease):
            entered.set()
            assert release.wait(timeout=5)

    def heartbeat() -> None:
        outcomes.append(control.heartbeat(lease))
        renewed.set()

    holder = Thread(target=hold_submission)
    renewer = Thread(target=heartbeat)
    holder.start()
    assert entered.wait(timeout=5)
    now[0] += 5_001
    try:
        renewer.start()
        assert not renewed.wait(timeout=0.1)
        assert control.snapshot().state == "running"
    finally:
        release.set()
        holder.join(timeout=5)
        renewer.join(timeout=5)
    assert outcomes == [False]
    assert not holder.is_alive() and not renewer.is_alive()
    assert control.snapshot().state == "stop_requested"


def test_expired_heartbeat_write_failure_never_reenables_open(
    tmp_path, monkeypatch
) -> None:
    now = [100_000]
    control = PolymarketRuntimeControl(
        tmp_path / "ownership.sqlite3",
        maximum_heartbeat_age_ms=5_000,
        clock_ms=lambda: now[0],
    )
    lease = control.acquire()
    initial = control.snapshot()
    now[0] += 5_001

    def disk_full(connection, payload) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    with monkeypatch.context() as patch:
        patch.setattr(control, "_write", disk_full)
        with pytest.raises(sqlite3.OperationalError, match="disk is full"):
            control.heartbeat(lease)
    assert control.snapshot() == initial
    with pytest.raises(PolymarketLiveBlocked):
        control.assert_opening_allowed(lease)
    assert control.heartbeat(lease) is False
    assert control.snapshot().state == "stop_requested"


def test_expiry_stop_is_visible_in_a_fresh_interpreter(tmp_path) -> None:
    now = [100_000]
    path = tmp_path / "ownership.sqlite3"
    control = PolymarketRuntimeControl(
        path, maximum_heartbeat_age_ms=5_000, clock_ms=lambda: now[0]
    )
    lease = control.acquire()
    now[0] += 5_001
    assert control.heartbeat(lease) is False
    # This is real cross-process persistence, not an OS reboot/power-loss test.
    script = (
        "import sys; "
        "from simple_ai_trading.polymarket_runtime_control import PolymarketRuntimeControl; "
        "control = PolymarketRuntimeControl(sys.argv[1]); "
        "snapshot = control.snapshot(); "
        "assert snapshot.state == 'stop_requested'; "
        "assert not control.heartbeat(snapshot.lease_id); "
        "print(snapshot.stop_reason)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == "heartbeat_invalid_or_expired"

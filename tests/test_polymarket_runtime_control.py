from __future__ import annotations

import asyncio
import sqlite3
from threading import Event, Thread

import pytest

from simple_ai_trading.polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveError,
)
from simple_ai_trading.polymarket_runtime_control import (
    PolymarketRuntimeControl,
    PolymarketRuntimeControlService,
    PolymarketRuntimeLeaseInterlock,
)


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

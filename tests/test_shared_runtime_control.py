"""Venue independence, durable Stop and shared dispatch ordering."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from queue import Queue
import sqlite3
import sys
from threading import Event, Thread
import time

import pytest

from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.polymarket_live import PolymarketLiveBlocked, PolymarketLiveError
from simple_ai_trading.polymarket_runtime_control import (
    PolymarketRuntimeControl,
    PolymarketRuntimeControlSnapshot,
)
from simple_ai_trading.runtime_control import (
    RuntimeControl,
    RuntimeControlBlocked,
    RuntimeControlError,
    RuntimeControlNamespace,
    RuntimeLeaseInterlock,
)
from tools.cocoindex_agent_search import run_captured_process


def _enroll_flat_control(control):
    """Provision this test's empty namespace and explicitly reconcile it flat."""
    assert not control.enrollment_path.exists()
    control.provision_storage()
    assert control.complete_stale_stop(exposure_closed=True)
    return control


@pytest.mark.parametrize("factory,blocked", [(BinanceRuntimeControl, RuntimeControlBlocked), (PolymarketRuntimeControl, PolymarketLiveBlocked)])
def test_empty_release_cannot_clear_ownerless_stop(tmp_path, factory, blocked):
    control = factory(tmp_path / "control.sqlite3")
    control.provision_storage()
    before = control.snapshot()
    with pytest.raises(blocked):
        control.release("", reason="unowned release")
    assert control.snapshot() == before


def test_owned_release_waits_for_the_current_submission_guard(tmp_path):
    control = _enroll_flat_control(BinanceRuntimeControl(tmp_path / "control.sqlite3"))
    lease = control.acquire()
    competing = BinanceRuntimeControl(control.ledger_path, interlock_timeout_seconds=1)
    outcomes = Queue()

    def release():
        try:
            competing.release(lease, reason="owned shutdown")
        except RuntimeControlError:
            outcomes.put("blocked")
        else:
            outcomes.put("released")

    with control.submission_guard(lease):
        thread = Thread(target=release)
        thread.start()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert outcomes.get_nowait() == "blocked"
        assert control.snapshot().state == "running"
    control.release(lease, reason="owned shutdown")
    assert control.snapshot().state == "stopped"


@pytest.mark.parametrize(
    ("factory", "blocked", "error"),
    [
        (BinanceRuntimeControl, RuntimeControlBlocked, RuntimeControlError),
        (PolymarketRuntimeControl, PolymarketLiveBlocked, PolymarketLiveError),
    ],
)
def test_stop_survives_controller_restart_and_missing_record_is_not_reset(
    tmp_path, factory, blocked, error
):
    path = tmp_path / "control.sqlite3"
    control = _enroll_flat_control(factory(path))
    lease = control.acquire()
    stopped = control.request_stop(reason="risk_limit")
    reopened = factory(path)
    assert reopened.snapshot() == stopped
    assert reopened.heartbeat(lease) is False
    with pytest.raises(blocked, match="does not permit"):
        reopened.assert_opening_allowed(lease)
    with pytest.raises(blocked, match="Stop completion"):
        reopened.acquire()
    with closing(sqlite3.connect(path)) as db:
        db.execute(f"DELETE FROM {control.namespace.table_name}")
        db.commit()
    with pytest.raises(error, match="record is missing"):
        factory(path).snapshot()


def test_venue_control_rows_and_error_types_remain_independent(tmp_path):
    path = tmp_path / "shared.sqlite3"
    poly = _enroll_flat_control(PolymarketRuntimeControl(path))
    binance = _enroll_flat_control(BinanceRuntimeControl(path))
    poly_lease = poly.acquire()
    binance_lease = binance.acquire()
    assert isinstance(poly.snapshot(), PolymarketRuntimeControlSnapshot)
    assert poly_lease != binance_lease
    original = binance.snapshot()
    poly.request_stop()
    assert binance.snapshot() == original
    binance.assert_opening_allowed(binance_lease)
    with pytest.raises(PolymarketLiveBlocked):
        poly.assert_opening_allowed(poly_lease)
    binance.set_paused(True, reason="operator_pause")
    with pytest.raises(RuntimeControlBlocked):
        binance.assert_opening_allowed(binance_lease)
    assert poly.snapshot().state == "stop_requested"
    # Same-database use conservatively shares an OS interlock, not authority.
    assert poly.lock_path == binance.lock_path


def test_binance_pause_and_expired_heartbeat_cannot_revive_old_lease(tmp_path):
    now = [100_000]
    path = tmp_path / "control.sqlite3"
    control = _enroll_flat_control(
        BinanceRuntimeControl(
            path, maximum_heartbeat_age_ms=5_000, clock_ms=lambda: now[0]
        )
    )
    lease = control.acquire()
    interlock = RuntimeLeaseInterlock(control, lease)
    with interlock.submission_guard():
        pass
    control.set_paused(True, reason="operator_pause")
    with pytest.raises(RuntimeControlBlocked):
        interlock.assert_opening_allowed()
    assert control.heartbeat(lease)
    now[0] += 5_001
    assert not control.heartbeat(lease)
    stopped = control.snapshot()
    now[0] = 100_001
    assert not control.heartbeat(lease)
    assert control.snapshot() == stopped
    with pytest.raises(RuntimeControlBlocked):
        control.set_paused(False, reason="resume")
    # Close serialization does not require permission to open new exposure.
    with control.close_guard(timeout_seconds=1):
        assert control.snapshot().state == "stop_requested"


@pytest.mark.parametrize("table", ["", "x; DROP TABLE runtime_control", "x-y", "X", 7])
def test_storage_namespace_rejects_unsafe_sql_identifiers(table):
    with pytest.raises(ValueError, match="table name"):
        RuntimeControlNamespace(table, "v1", "Test")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": ""},
        {"schema_version": 7},
        {"label": " "},
        {"label": None},
        {"legacy_schema_version": ""},
        {"legacy_schema_version": 1},
        {"legacy_schema_version": "v1"},
    ],
)
def test_storage_namespace_rejects_ambiguous_versions_and_labels(kwargs):
    values = {"table_name": "control", "schema_version": "v1", "label": "Test"}
    with pytest.raises(ValueError):
        RuntimeControlNamespace(**{**values, **kwargs})


def test_new_namespace_does_not_guess_legacy_migration(tmp_path):
    path = tmp_path / "control.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "CREATE TABLE binance_runtime_control (singleton INTEGER PRIMARY KEY)"
        )
        db.execute("INSERT INTO binance_runtime_control VALUES (1)")
        db.commit()
    with pytest.raises(RuntimeControlError, match="pause fields are missing"):
        BinanceRuntimeControl(path).provision_storage(expected_record_sha256="0" * 64)
    with closing(sqlite3.connect(path)) as db:
        assert (
            len(db.execute("PRAGMA table_info(binance_runtime_control)").fetchall())
            == 1
        )


def test_stop_orders_after_another_process_guard_and_blocks_later_dispatch(tmp_path):
    """Use a real contained process and real OS file locks, not a mocked mutex."""
    path = tmp_path / "control.sqlite3"
    control = _enroll_flat_control(BinanceRuntimeControl(path))
    lease = control.acquire()
    entered_path = tmp_path / "entered"
    release_path = tmp_path / "release"
    worker = """
from pathlib import Path
import sys, time
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
path, lease, entered, release = sys.argv[1:]
control = BinanceRuntimeControl(path)
with control.submission_guard(lease):
    Path(entered).write_text('entered')
    deadline = time.monotonic() + 8
    while not Path(release).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('test release did not arrive')
        time.sleep(0.01)
print('guard_released')
"""
    outcomes = Queue()
    stop_started, stop_finished = Event(), Event()

    def child():
        try:
            outcomes.put(
                run_captured_process(
                    [
                        str(Path(sys.executable).resolve()),
                        "-c",
                        worker,
                        str(path),
                        lease,
                        str(entered_path),
                        str(release_path),
                    ],
                    cwd=Path.cwd(),
                    env=None,
                    input_text="",
                    timeout=12,
                )
            )
        except BaseException as error:
            outcomes.put(error)

    def stop():
        stop_started.set()
        try:
            control.request_stop(reason="operator_stop")
        except BaseException as error:
            outcomes.put(error)
        finally:
            stop_finished.set()

    child_thread, stop_thread = Thread(target=child), Thread(target=stop)
    child_thread.start()
    try:
        deadline = time.monotonic() + 6
        while not entered_path.exists() and child_thread.is_alive():
            assert time.monotonic() < deadline, "child did not acquire guard"
            time.sleep(0.01)
        assert entered_path.exists(), "child failed before acquiring guard"
        stop_thread.start()
        assert stop_started.wait(timeout=1)
        assert not stop_finished.wait(timeout=0.15)
        assert control.snapshot().state == "running"
    finally:
        release_path.write_text("release")
        child_thread.join(timeout=15)
        if stop_thread.ident is not None:
            stop_thread.join(timeout=15)
    assert not child_thread.is_alive() and not stop_thread.is_alive()
    assert stop_finished.is_set()
    for outcome in list(outcomes.queue):
        if isinstance(outcome, BaseException):
            raise outcome
        assert outcome.returncode == 0, outcome.stderr
        assert outcome.stdout.strip() == "guard_released"
    assert control.snapshot().state == "stop_requested"
    with pytest.raises(RuntimeControlBlocked):
        with control.submission_guard(lease):
            pytest.fail("post-Stop dispatch must not enter")


def test_generic_control_uses_separate_default_namespace(tmp_path):
    control = _enroll_flat_control(RuntimeControl(tmp_path / "generic.sqlite3"))
    assert control.snapshot().state == "stopped"
    assert control.namespace.table_name == "runtime_control"

"""Independent storage enrollment must survive loss without creating authority."""

from __future__ import annotations

from contextlib import closing
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from simple_ai_trading import runtime_control as runtime
from simple_ai_trading import runtime_control_storage as storage
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.polymarket_runtime_control import PolymarketRuntimeControl


@pytest.fixture(params=[BinanceRuntimeControl, PolymarketRuntimeControl])
def control(tmp_path, request):
    return request.param(
        tmp_path / "control.sqlite3",
        enrollment_path=tmp_path / "policy" / "control.json",
    )


def enrolled(control):
    state = control.provision_storage()
    assert state.state == "stop_requested"
    assert state.paused and state.stop_epoch == 1
    assert control.complete_stale_stop(exposure_closed=True)
    return control


@pytest.mark.parametrize(
    "operation",
    [
        "snapshot",
        "acquire",
        "request_stop",
        "heartbeat",
        "submission_guard",
        "close_guard",
    ],
)
def test_normal_missing_storage_never_creates_any_authority(control, operation):
    with pytest.raises(control.error):
        if operation == "heartbeat":
            control.heartbeat("old-lease")
        elif operation == "submission_guard":
            with control.submission_guard("old-lease"):
                pytest.fail("missing enrollment allowed dispatch")
        elif operation == "close_guard":
            with control.close_guard(timeout_seconds=1):
                pytest.fail("missing close serialization was recreated")
        else:
            getattr(control, operation)()
    assert not control.ledger_path.exists()
    assert not control.lock_path.exists()
    assert not control.close_lock_path.exists()
    assert not control.enrollment_path.exists()


def test_explicit_new_store_requires_reconciliation_and_never_reenrolls(control):
    initial = control.provision_storage()
    with pytest.raises(control.blocked_error):
        control.acquire()
    with pytest.raises(control.blocked_error):
        control.complete_stale_stop(exposure_closed=False)
    assert control.snapshot() == initial
    assert control.complete_stale_stop(exposure_closed=True)
    lease = control.acquire()
    control.assert_opening_allowed(lease)
    with pytest.raises(control.error, match="already exists"):
        control.provision_storage()
    assert control.snapshot().lease_id == lease
    value = json.loads(control.enrollment_path.read_bytes())
    assert set(value["resources"]) == {"database", "submission_lock", "close_lock"}
    assert not any("path" in key for key in value)


@pytest.mark.parametrize(
    "operation", ["snapshot", "acquire", "request_stop", "heartbeat"]
)
def test_database_loss_preserves_stop_and_cannot_bootstrap_a_replacement(
    control, operation
):
    enrolled(control)
    lease = control.acquire()
    previous = control.request_stop(reason="risk_limit")
    saved = control.ledger_path.with_suffix(".retained")
    control.ledger_path.rename(saved)
    with pytest.raises(control.error):
        if operation == "heartbeat":
            control.heartbeat(lease)
        else:
            getattr(control, operation)()
    assert not control.ledger_path.exists()
    with control.close_guard(timeout_seconds=1):
        assert saved.exists()
    saved.rename(control.ledger_path)
    assert control.snapshot() == previous


@pytest.mark.parametrize("resource", ["ledger_path", "lock_path", "close_lock_path"])
def test_replaced_or_missing_resource_is_not_adopted(control, resource):
    enrolled(control)
    path = getattr(control, resource)
    saved = path.with_suffix(path.suffix + ".retained")
    path.rename(saved)
    with pytest.raises(control.error):
        control.snapshot()
    assert not path.exists()
    path.write_bytes(saved.read_bytes())
    with pytest.raises(control.error):
        control.snapshot()
    assert saved.exists()


def test_anchor_loss_does_not_recreate_missing_locks_even_during_provisioning(control):
    enrolled(control)
    control.enrollment_path.rename(control.enrollment_path.with_suffix(".retained"))
    control.lock_path.rename(control.lock_path.with_suffix(".retained-lock"))
    with pytest.raises(control.error):
        control.acquire()
    with pytest.raises(control.error, match="already enrolled"):
        control.provision_storage()
    assert not control.lock_path.exists()
    assert not control.enrollment_path.exists()


def test_database_and_anchor_loss_do_not_reset_normal_open(control):
    enrolled(control)
    control.enrollment_path.rename(control.enrollment_path.with_suffix(".retained"))
    control.ledger_path.rename(control.ledger_path.with_suffix(".retained"))
    with pytest.raises(control.error):
        control.acquire()
    assert not control.ledger_path.exists()
    assert not control.enrollment_path.exists()


def test_deleted_namespace_is_not_recreated_inside_enrolled_database(control):
    enrolled(control)
    with closing(sqlite3.connect(control.ledger_path)) as db:
        db.execute(f"DROP TABLE {control.namespace.table_name}")
        db.commit()
    with pytest.raises(control.error, match="namespace is missing"):
        control.snapshot()
    with closing(sqlite3.connect(control.ledger_path)) as db:
        assert (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE name=?",
                (control.namespace.table_name,),
            ).fetchone()
            is None
        )


def test_independent_enrollment_rejects_changed_database_store_marker(control):
    enrolled(control)
    with closing(sqlite3.connect(control.ledger_path)) as db:
        db.execute("UPDATE runtime_storage_enrollment SET store_id=?", ("0" * 32,))
        db.commit()
    with pytest.raises(control.error, match="enrollment differs"):
        control.snapshot()


def test_explicit_adoption_requires_exact_current_record_and_preserves_stop(control):
    # A retained schema/row without enrollment models a pre-enrollment store.
    control.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(control.ledger_path, isolation_level=None)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        control._bootstrap_control(db, require_reconciliation=True)
        db.execute("COMMIT")
    observed = control.inspect_storage()
    assert not observed["enrollment_exists"]
    with pytest.raises(control.error, match="exact existing"):
        control.provision_storage(expected_record_sha256="0" * 64)
    assert not control.enrollment_path.exists()
    adopted = control.provision_storage(
        expected_record_sha256=observed["control_record_sha256"]
    )
    assert adopted.state == "stop_requested" and adopted.stop_epoch == 1
    assert (
        control.inspect_storage()["control_record_sha256"]
        == observed["control_record_sha256"]
    )


def test_interrupted_commit_keeps_prepared_enrollment_until_exact_explicit_recovery(
    control, monkeypatch
):
    finish = storage.RuntimeStorageEnrollment.finish

    def interrupt(*_args):
        raise KeyboardInterrupt("interrupted after database commit")

    monkeypatch.setattr(storage.RuntimeStorageEnrollment, "finish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        control.provision_storage()
    pending = storage.RuntimeStorageEnrollment.read(
        control.enrollment_path, allow_prepared=True
    )
    observed = control.inspect_storage()
    with pytest.raises(control.error):
        control.acquire()
    with pytest.raises(control.error, match="identity differs"):
        control.recover_storage_enrollment(
            expected_store_id="0" * 32,
            expected_record_sha256=observed["control_record_sha256"],
        )
    monkeypatch.setattr(storage.RuntimeStorageEnrollment, "finish", finish)
    state = control.recover_storage_enrollment(
        expected_store_id=pending.store_id,
        expected_record_sha256=observed["control_record_sha256"],
    )
    assert state.state == "stop_requested" and state.stop_epoch == 1
    assert (
        control.inspect_storage()["control_record_sha256"]
        == observed["control_record_sha256"]
    )


def test_interrupted_bootstrap_rolls_back_and_never_invents_recoverable_state(
    control, monkeypatch
):
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("interrupted before database commit")

    monkeypatch.setattr(control, "_bootstrap_control", interrupt)
    with pytest.raises(KeyboardInterrupt):
        control.provision_storage()
    pending = storage.RuntimeStorageEnrollment.read(
        control.enrollment_path, allow_prepared=True
    )
    with pytest.raises(control.error):
        control.snapshot()
    with pytest.raises(control.error):
        control.recover_storage_enrollment(
            expected_store_id=pending.store_id, expected_record_sha256="0" * 64
        )
    with closing(sqlite3.connect(control.ledger_path)) as db:
        assert (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE name=?",
                (control.namespace.table_name,),
            ).fetchone()
            is None
        )


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "oversized", "digest", "boolean_identity", "unknown_field"],
)
def test_malformed_enrollment_cannot_authorize_access(control, corruption):
    enrolled(control)
    raw = control.enrollment_path.read_bytes()
    body = json.loads(raw)
    if corruption == "duplicate":
        raw = b'{"state":"ready",' + raw[1:]
    elif corruption == "oversized":
        raw = b" " * (storage.MAX_ENROLLMENT_BYTES + 1)
    else:
        if corruption == "digest":
            body["record_sha256"] = "0" * 64
        else:
            body.pop("record_sha256")
            if corruption == "boolean_identity":
                body["resources"]["database"][0] = True
            else:
                body["unknown"] = "value"
            body["record_sha256"] = hashlib.sha256(storage._canonical(body)).hexdigest()
        raw = json.dumps(body).encode()
    control.enrollment_path.write_bytes(raw)
    with pytest.raises(control.error):
        control.snapshot()


def test_unused_mutex_paths_are_not_retained_forever(tmp_path):
    gc.collect()
    baseline = len(runtime._LOCAL_LOCKS)
    locks = [
        runtime._CrossProcessFileLock(tmp_path / str(i), timeout_seconds=1)
        for i in range(128)
    ]
    assert len(runtime._LOCAL_LOCKS) == baseline + 128
    same = runtime._CrossProcessFileLock(tmp_path / "0", timeout_seconds=1)
    assert same._local is locks[0]._local
    del same, locks
    gc.collect()
    assert len(runtime._LOCAL_LOCKS) == baseline


def test_reentering_one_lock_instance_does_not_lose_its_cleanup(tmp_path):
    lock = runtime._CrossProcessFileLock(tmp_path / "lock", timeout_seconds=1)
    with lock:
        with pytest.raises(runtime.RuntimeControlBlocked, match="already entered"):
            with lock:
                pytest.fail("reentry discarded the original cleanup")
        assert lock._handle is not None and not lock._handle.closed
    assert lock._handle is None and lock._cleanup is None
    with runtime._CrossProcessFileLock(tmp_path / "lock", timeout_seconds=1):
        pass


@pytest.mark.parametrize("kind", ["json", "bytes"])
def test_atomic_storage_cancellation_removes_only_its_temporary_file(
    tmp_path, monkeypatch, kind
):
    from simple_ai_trading import storage as atomic

    target = tmp_path / "policy.json"
    target.write_bytes(b"retained policy")

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("cancelled atomic replacement")

    monkeypatch.setattr(atomic.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        if kind == "json":
            atomic.write_json_atomic(target, {"state": "stopped"})
        else:
            atomic.write_bytes_atomic(target, b"new policy")
    assert target.read_bytes() == b"retained policy"
    assert list(tmp_path.iterdir()) == [target]


def test_owned_close_reuses_the_custom_configured_control_without_opening_authority(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from simple_ai_trading import polymarket_live_stop as stop
    from simple_ai_trading.polymarket_live_runtime import PolymarketLiveRuntimeGuard
    from simple_ai_trading.polymarket_runtime_control import (
        PolymarketRuntimeLeaseInterlock,
    )

    control = PolymarketRuntimeControl(
        tmp_path / "data.sqlite3", enrollment_path=tmp_path / "policy.json"
    )
    control.provision_storage()
    guard = PolymarketLiveRuntimeGuard(
        opening_interlock=PolymarketRuntimeLeaseInterlock(control, "not-a-live-lease")
    )
    coordinator = SimpleNamespace(runtime_authority=guard)
    ledger = SimpleNamespace(path=control.ledger_path)
    monkeypatch.setattr(
        stop,
        "_stop_owned_polymarket_exposure_unlocked",
        lambda **_kwargs: ({"completed": True}, 0),
    )
    payload, status = stop.stop_owned_polymarket_exposure(
        coordinator=coordinator, ledger=ledger, timeout_seconds=1
    )
    assert payload["completed"] and status == 0
    assert control.snapshot().state == "stop_requested"


def test_control_cli_provisions_and_inspects_locally_without_credentials(
    tmp_path, monkeypatch, capsys
):
    from simple_ai_trading import entrypoint
    from simple_ai_trading import polymarket_live_cli as cli

    def forbidden():
        pytest.fail("local control setup constructed an authenticated venue")

    monkeypatch.setattr(cli, "_credentials_and_venue", forbidden)
    path, policy = tmp_path / "data.sqlite3", tmp_path / "policy.json"

    def arguments(action):
        return entrypoint._parse_args(
            [
                "polymarket-live",
                "--action",
                action,
                "--ledger",
                str(path),
                "--runtime-enrollment",
                str(policy),
                "--json",
            ]
        )

    assert cli.command_polymarket_live(arguments("enroll-control")) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["runtime_control"]["state"] == "stop_requested"
    assert cli.command_polymarket_live(arguments("inspect-control")) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["runtime_control"]["enrollment_exists"]
    assert observed["runtime_control"]["enrollment_state"] == "ready"
    assert observed["runtime_control"]["new_exposure_authorized"] is False
    control = PolymarketRuntimeControl(path, enrollment_path=policy)
    assert control.complete_stale_stop(exposure_closed=True)
    control.acquire()
    assert cli.command_polymarket_live(arguments("pause")) == 0
    assert control.snapshot().paused


def test_status_preserves_missing_control_as_unknown_without_creating_it(tmp_path):
    from simple_ai_trading.polymarket_live import PolymarketLiveOrderLedger
    from simple_ai_trading.polymarket_live_cli import _local_status

    path = tmp_path / "ownership.sqlite3"
    PolymarketLiveOrderLedger(path).records()
    control = PolymarketRuntimeControl(path)
    status = _local_status(path)
    assert status["runtime_state"] == "control_unavailable"
    assert status["runtime_control"] is None and not status["can_open"]
    assert not control.enrollment_path.exists() and not control.lock_path.exists()


def test_cli_inspection_supplies_exact_prepared_identity_for_local_recovery(
    tmp_path, monkeypatch, capsys
):
    from simple_ai_trading import entrypoint
    from simple_ai_trading import polymarket_live_cli as cli

    path = tmp_path / "control.sqlite3"
    control = PolymarketRuntimeControl(path)
    finish = storage.RuntimeStorageEnrollment.finish

    def interrupted(*_args):
        raise KeyboardInterrupt("interrupted after database commit")

    monkeypatch.setattr(storage.RuntimeStorageEnrollment, "finish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        control.provision_storage()
    monkeypatch.setattr(storage.RuntimeStorageEnrollment, "finish", finish)
    monkeypatch.setattr(
        cli,
        "_credentials_and_venue",
        lambda: pytest.fail("local enrollment recovery read credentials"),
    )
    inspect = entrypoint._parse_args(
        [
            "polymarket-live",
            "--action",
            "inspect-control",
            "--ledger",
            str(path),
            "--json",
        ]
    )
    assert cli.command_polymarket_live(inspect) == 0
    observed = json.loads(capsys.readouterr().out)["runtime_control"]
    assert observed["enrollment_state"] == "prepared"
    recover = entrypoint._parse_args(
        [
            "polymarket-live",
            "--action",
            "recover-control-enrollment",
            "--ledger",
            str(path),
            "--expected-control-store-id",
            observed["enrollment_store_id"],
            "--expected-control-record-sha256",
            observed["control_record_sha256"],
            "--json",
        ]
    )
    assert cli.command_polymarket_live(recover) == 0
    assert (
        json.loads(capsys.readouterr().out)["runtime_control"]["state"]
        == "stop_requested"
    )


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_actual_child_exit_preserves_uncertainty_and_allows_only_exact_committed_recovery(
    control, phase
):
    from tools.cocoindex_agent_search import run_captured_process

    worker = """
import os, sys
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.polymarket_runtime_control import PolymarketRuntimeControl
from simple_ai_trading.runtime_control_storage import RuntimeStorageEnrollment
kind, database, policy, phase = sys.argv[1:]
factory = BinanceRuntimeControl if kind == 'BinanceRuntimeControl' else PolymarketRuntimeControl
control = factory(database, enrollment_path=policy)
if phase == 'after_commit':
    RuntimeStorageEnrollment.finish = lambda *_args: os._exit(42)
else:
    original = control._bootstrap_control
    def stop_before_commit(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(42)
    control._bootstrap_control = stop_before_commit
control.provision_storage()
raise AssertionError('worker unexpectedly survived the selected exit stage')
"""
    result = run_captured_process(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            worker,
            type(control).__name__,
            str(control.ledger_path),
            str(control.enrollment_path),
            phase,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=None,
        input_text="",
        timeout=30,
    )
    assert result.returncode == 42, result.stderr
    pending = storage.RuntimeStorageEnrollment.read(
        control.enrollment_path, allow_prepared=True
    )
    with pytest.raises(control.error):
        control.acquire()
    if phase == "after_commit":
        digest = control.inspect_storage()["control_record_sha256"]
        state = control.recover_storage_enrollment(
            expected_store_id=pending.store_id, expected_record_sha256=digest
        )
        assert state.state == "stop_requested" and state.stop_epoch == 1
    else:
        with pytest.raises(control.error):
            control.recover_storage_enrollment(
                expected_store_id=pending.store_id, expected_record_sha256="0" * 64
            )
        assert (
            storage.RuntimeStorageEnrollment.read(
                control.enrollment_path, allow_prepared=True
            )
            == pending
        )


def test_post_lock_identity_failure_releases_the_actual_native_lock(
    control, monkeypatch
):
    enrolled(control)
    identity = storage.FileIdentity.at(control.lock_path)
    original = storage.FileIdentity.verify
    calls = 0

    def interrupted(self, path, handle=None):
        nonlocal calls
        if path == control.lock_path:
            calls += 1
            if calls == 3:
                raise FileNotFoundError("injected post-acquisition identity failure")
        return original(self, path, handle)

    with monkeypatch.context() as scoped:
        scoped.setattr(storage.FileIdentity, "verify", interrupted)
        with pytest.raises(
            control.blocked_error, match="interlock is missing or changed"
        ):
            with control._interlock():
                pytest.fail("changed identity admitted a guard")
    with control.lock_type(
        control.lock_path, timeout_seconds=1, expected_identity=identity
    ):
        pass


def test_database_disappearing_at_connect_is_not_created_by_sqlite(
    control, monkeypatch
):
    enrolled(control)
    original = runtime.sqlite3.connect
    saved = control.ledger_path.with_suffix(".retained")

    def connect(*args, **kwargs):
        if args[0] == control.ledger_path.as_uri() + "?mode=rw":
            control.ledger_path.rename(saved)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.sqlite3, "connect", connect)
    with pytest.raises(control.error, match="could not be opened"):
        control.snapshot()
    assert not control.ledger_path.exists() and saved.exists()


@pytest.mark.parametrize(
    "suffix", ["-wal", "-shm", "-journal", ".runtime.lock", ".runtime.close.lock", ""]
)
def test_enrollment_cannot_overwrite_database_or_sqlite_and_dispatch_sidecars(
    tmp_path, suffix
):
    path = tmp_path / "control.sqlite3"
    with pytest.raises(ValueError, match="independent file"):
        BinanceRuntimeControl(path, enrollment_path=path.with_name(path.name + suffix))
    assert list(tmp_path.iterdir()) == []


def test_noncreating_sqlite_uri_handles_unicode_and_reserved_url_characters(tmp_path):
    control = BinanceRuntimeControl(tmp_path / "δοκιμή # %.sqlite3")
    enrolled(control)
    lease = control.acquire()
    assert control.snapshot().lease_id == lease

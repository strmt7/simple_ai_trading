"""Offline execution-store loss and explicit enrollment lifecycle checks."""

from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.position_transactions import position_transaction
from simple_ai_trading.runtime_control_storage import RuntimeStorageEnrollment

SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-storage-key"
)


@pytest.fixture
def storage(tmp_path):
    control = BinanceRuntimeControl(tmp_path / "runtime.sqlite3")
    control.provision_storage()
    return BinanceExecutionStorage(
        BinanceOpenIntentJournal(tmp_path / "intents.sqlite3"), control
    )


def test_fresh_enrollment_preserves_stop_and_binds_existing_interlocks(storage):
    before = storage.control.snapshot()
    result = storage.provision(scope=SCOPE)
    assert result["enrollment_state"] == "ready"
    assert result["new_exposure_authorized"] is False
    assert storage.control.snapshot() == before
    storage.verify(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        storage.provision(scope=SCOPE)


def test_missing_authority_is_never_created_by_inspection_or_verification(storage):
    assert storage.inspect()["journal_exists"] is False
    with pytest.raises(OpenIntentError):
        storage.verify(scope=SCOPE)
    assert not storage.journal.path.exists()
    assert not storage.enrollment_path.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "replacement",
        "truncated",
        "uuid-missing",
        "uuid-changed",
        "marker-missing",
    ],
)
def test_execution_loss_rejects_without_repair(storage, damage):
    storage.provision(scope=SCOPE)
    if damage in {"missing", "replacement"}:
        original = storage.journal.path.read_bytes()
        storage.journal.path.rename(storage.journal.path.with_suffix(".retained"))
        if damage == "replacement":
            storage.journal.path.write_bytes(original)
    elif damage == "truncated":
        storage.journal.path.write_bytes(b"")
    elif damage == "marker-missing":
        storage.enrollment_path.unlink()
    else:
        with closing(sqlite3.connect(storage.journal.path)) as connection:
            connection.execute(
                "DELETE FROM runtime_storage_enrollment"
                if damage == "uuid-missing"
                else "UPDATE runtime_storage_enrollment SET store_id=?",
                () if damage == "uuid-missing" else ("b" * 32,),
            )
            connection.commit()
    existed = storage.journal.path.exists()
    before = storage.journal.path.read_bytes() if existed else None
    with pytest.raises(OpenIntentError):
        storage.verify(scope=SCOPE)
    assert storage.journal.path.exists() == existed
    assert (storage.journal.path.read_bytes() if existed else None) == before


def test_scope_or_controller_interlock_changes_reject(storage, tmp_path):
    storage.provision(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        storage.verify(
            scope=BinanceExecutionScope.from_api_key(
                BINANCE_SPOT_TESTNET, "spot", "other-key"
            )
        )
    other = BinanceRuntimeControl(tmp_path / "other.sqlite3")
    other.provision_storage()
    with pytest.raises(OpenIntentError):
        BinanceExecutionStorage(storage.journal, other, storage.enrollment_path).verify(
            scope=SCOPE
        )


def test_existing_journal_requires_exact_snapshot_adoption(storage):
    with position_transaction(storage.journal, write=True) as transaction:
        storage.journal._prepare_scope(transaction, SCOPE)
    snapshot = storage.inspect()["journal_snapshot_sha256"]
    with pytest.raises(OpenIntentError):
        storage.provision(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        storage.provision(scope=SCOPE, expected_snapshot_sha256="0" * 64)
    assert not storage.enrollment_path.exists()
    storage.provision(scope=SCOPE, expected_snapshot_sha256=snapshot)
    storage.verify(scope=SCOPE)


def test_lost_marker_cannot_reenroll_existing_uuid(storage):
    storage.provision(scope=SCOPE)
    storage.enrollment_path.unlink()
    digest = storage.inspect()["journal_snapshot_sha256"]
    with pytest.raises(OpenIntentError):
        storage.provision(scope=SCOPE, expected_snapshot_sha256=digest)


@pytest.mark.parametrize("committed", [True, False])
def test_interrupted_enrollment_needs_exact_committed_prepared_recovery(
    storage, monkeypatch, committed
):
    method = "finish" if committed else "prepare"
    original = getattr(RuntimeStorageEnrollment, method)

    def interrupt(record, path):
        if not committed:
            original(record, path)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(RuntimeStorageEnrollment, method, interrupt)
        with pytest.raises(KeyboardInterrupt):
            storage.provision(scope=SCOPE)
    inspected = storage.inspect()
    assert inspected["enrollment_state"] == "prepared"
    with pytest.raises(OpenIntentError):
        storage.verify(scope=SCOPE)
    arguments = dict(
        scope=SCOPE,
        expected_store_id=inspected["enrollment_store_id"],
        expected_record_sha256=inspected["enrollment_record_sha256"],
    )
    with pytest.raises(OpenIntentError):
        storage.recover(**{**arguments, "expected_record_sha256": "0" * 64})
    if committed:
        storage.recover(**arguments)
        storage.verify(scope=SCOPE)
    else:
        with pytest.raises(OpenIntentError):
            storage.recover(**arguments)
        assert storage.inspect()["enrollment_state"] == "prepared"
    assert storage.control.snapshot().state == "stop_requested"


def test_running_controller_cannot_provision_or_recover(storage):
    storage.control.complete_stale_stop(exposure_closed=True)
    storage.control.acquire()
    with pytest.raises(OpenIntentError):
        storage.provision(scope=SCOPE)
    assert not storage.journal.path.exists()


def test_duplicate_or_wrong_marker_fields_reject(storage):
    storage.provision(scope=SCOPE)
    record = storage._record()
    storage.enrollment_path.write_bytes(replace(record, namespace="other").encoded())
    with pytest.raises(OpenIntentError):
        storage.verify(scope=SCOPE)


def test_inspection_budget_prevents_unbounded_adoption_snapshot(storage, monkeypatch):
    storage.provision(scope=SCOPE)
    monkeypatch.setattr(
        "simple_ai_trading.binance_execution_storage._MAX_INSPECTION_BYTES", 1
    )
    with pytest.raises(OpenIntentError):
        storage.inspect()
    storage.verify(scope=SCOPE)


def test_explicit_marker_location_is_supported(storage, tmp_path):
    custom = BinanceExecutionStorage(
        storage.journal, storage.control, tmp_path / "execution.json"
    )
    custom.provision(scope=SCOPE)
    custom.verify(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        storage.verify(scope=SCOPE)


def test_adoption_preserves_unknown_operation(storage):
    from simple_ai_trading.binance_spot_roundtrip_intents import (
        SpotRoundtripJournal,
        SpotRoundtripRequest,
    )

    journal = SpotRoundtripJournal(storage.journal)
    request = SpotRoundtripRequest("a" * 29, "BTCUSDC", "buy-sell", "100", "0", "100")
    journal.prepare(request, scope=SCOPE)
    before = journal.pending(scope=SCOPE)
    digest = storage.inspect()["journal_snapshot_sha256"]
    storage.provision(scope=SCOPE, expected_snapshot_sha256=digest)
    assert journal.pending(scope=SCOPE) == before
    assert before[1]["phase"] == "FIRST_UNKNOWN"
    storage.verify(scope=SCOPE)

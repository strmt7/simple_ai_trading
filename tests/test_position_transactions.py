"""Active store crash recovery and writer serialization, without venue access."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from simple_ai_trading import position_transactions as transactions
from simple_ai_trading.positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    compute_stats,
)


def _position():
    return OpenPosition(
        id="retained",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
    )


def _trade():
    return ClosedTrade(
        id="retained",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        exit_price=101,
        leverage=1,
        opened_at_ms=1,
        closed_at_ms=2,
        realized_pnl=1,
        realized_pnl_pct=0.01,
    )


@pytest.mark.parametrize("completed_files", [0, 1, 2])
@pytest.mark.parametrize("partial", [False, True])
def test_interrupted_close_recovers_whole_pair_once(
    tmp_path, monkeypatch, completed_files, partial
):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    real_write = transactions.write_bytes_atomic
    calls = 0

    def interrupted(path, body):
        nonlocal calls
        if calls == completed_files:
            raise OSError("injected replacement failure")
        calls += 1
        real_write(path, body)
        if calls == 2 and completed_files == 2:
            raise OSError("injected failure after both replacements")

    with monkeypatch.context() as patch:
        patch.setattr(transactions, "write_bytes_atomic", interrupted)
        with pytest.raises(ValueError, match="storage is unavailable"):
            if partial:
                store.record_close_result(_position(), replace(_trade(), qty=0.5))
            else:
                store.record_close(_trade())
    # A fresh API reader must not see the intermediate on-disk pair.
    reopened = PositionsStore(tmp_path)
    opens, closed = reopened.load_snapshot(strict=True)
    assert [p.qty for p in opens] == ([0.5] if partial else [])
    assert [t.qty for t in closed] == ([0.5] if partial else [1])
    assert reopened.load_snapshot(strict=True) == (opens, closed)
    stats = compute_stats(reopened, mark_price=101)
    assert stats.closed_trades == 1
    assert stats.open_positions == int(partial)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute(
            "SELECT state FROM position_replacement"
        ).fetchone() == ("APPLIED",)


def test_actual_child_exit_between_replacements_recovers(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    script = """
import json, os, sys
from simple_ai_trading import position_transactions as module
from simple_ai_trading.positions import ClosedTrade, PositionsStore
original = module.write_bytes_atomic
def crash(path, body):
    original(path, body)
    os._exit(72)
module.write_bytes_atomic = crash
PositionsStore(sys.argv[1]).record_close(ClosedTrade(**json.loads(sys.argv[2])))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(asdict(_trade()))],
        timeout=20,
        check=False,
    )
    assert result.returncode == 72
    assert json.loads(store.open_path.read_bytes()) == []
    assert not store.ledger_path.exists()
    assert store.load_snapshot(strict=True) == ([], [_trade()])


def test_competing_store_instances_do_not_lose_new_positions(tmp_path):
    def add(index):
        PositionsStore(tmp_path).record_open(replace(_position(), id=f"owned-{index}"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(12)))
    assert {p.id for p in PositionsStore(tmp_path).load_open(strict=True)} == {
        f"owned-{index}" for index in range(12)
    }


@pytest.mark.parametrize("name", ["open_positions.json", "ledger.json"])
@pytest.mark.parametrize("body", [None, b"[]", b"{broken"])
def test_missing_or_changed_enrolled_file_is_not_silently_restored(
    tmp_path, name, body
):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    store.record_close_result(_position(), replace(_trade(), qty=0.5))
    path = tmp_path / name
    if body is None:
        path.unlink()
    else:
        path.write_bytes(body)
    with pytest.raises(ValueError, match="differs from committed"):
        store.load_snapshot()
    with pytest.raises(ValueError):
        store.record_open(replace(_position(), id="new"))
    assert store.open_integrity_errors() == ("position_transaction_unresolved",)
    assert (path.read_bytes() if path.exists() else None) == body


def test_conflicting_second_file_is_checked_before_recovery_writes_first(
    tmp_path, monkeypatch
):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    with monkeypatch.context() as patch:

        def fail(*args):
            raise OSError("injected before projection")

        patch.setattr(transactions, "write_bytes_atomic", fail)
        with pytest.raises(ValueError):
            store.record_close(_trade())
    store.ledger_path.write_bytes(b"foreign bytes")
    with pytest.raises(ValueError, match="conflicting files"):
        store.load_open()
    assert store.open_path.read_bytes() == original
    assert store.ledger_path.read_bytes() == b"foreign bytes"


def test_stale_partial_close_cannot_resurrect_a_previously_closed_lot(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    store.record_close(_trade())
    before = store.load_snapshot(strict=True)
    with pytest.raises(ValueError, match="changed before recording"):
        store.record_close_result(_position(), replace(_trade(), qty=0.5))
    assert store.load_snapshot(strict=True) == before


def test_first_read_does_not_create_storage(tmp_path):
    root = tmp_path / "absent"
    assert PositionsStore(root).load_snapshot() == ([], [])
    assert not root.exists()


def test_legacy_files_are_preserved_until_first_valid_mutation(tmp_path):
    store = PositionsStore(tmp_path)
    body = json.dumps([asdict(_position())], indent=4).encode()
    store.open_path.write_bytes(body)
    assert store.load_snapshot(strict=True) == ([_position()], [])
    assert store.open_path.read_bytes() == body
    assert not store.opening_intents.path.exists()
    store.record_open(replace(_position(), id="second"))
    assert len(store.load_open(strict=True)) == 2


@pytest.mark.parametrize("corruption", ["schema", "empty", "text_blob"])
def test_invalid_transaction_record_rejects_without_repair(tmp_path, corruption):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    with sqlite3.connect(store.opening_intents.path) as connection:
        if corruption == "schema":
            connection.execute(
                "ALTER TABLE position_replacement ADD COLUMN unknown TEXT"
            )
        elif corruption == "empty":
            connection.execute("DELETE FROM position_replacement")
        else:
            connection.execute("UPDATE position_replacement SET after_open='[]'")
    with pytest.raises(ValueError):
        store.load_open()
    assert store.open_path.read_bytes() == original


def test_locked_writer_fails_without_overwriting_files(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="storage is unavailable"):
            store.record_open(replace(_position(), id="new"))
    assert store.open_path.read_bytes() == original


def test_corrupt_pending_blob_cannot_overwrite_valid_projection(tmp_path, monkeypatch):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    with monkeypatch.context() as patch:

        def fail(*args):
            raise OSError("injected before projection")

        patch.setattr(transactions, "write_bytes_atomic", fail)
        with pytest.raises(ValueError):
            store.record_close(_trade())
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute(
            "UPDATE position_replacement SET after_open=?", (b"{broken",)
        )
    with pytest.raises(ValueError, match="checksum differs"):
        store.load_open()
    assert store.open_path.read_bytes() == original
    assert not store.ledger_path.exists()


def test_aborted_redo_commit_never_publishes_projection(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_replace BEFORE UPDATE ON position_replacement "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(ValueError, match="storage is unavailable"):
        store.record_close(_trade())
    assert store.open_path.read_bytes() == original
    assert not store.ledger_path.exists()
    assert store.load_snapshot(strict=True) == ([_position()], [])


@pytest.mark.parametrize("operation", ["open", "remove"])
def test_first_mutation_does_not_enroll_corrupt_closed_state(tmp_path, operation):
    store = PositionsStore(tmp_path)
    original = json.dumps([asdict(_position())]).encode()
    store.open_path.write_bytes(original)
    store.ledger_path.write_bytes(b"{broken")
    with pytest.raises(ValueError):
        if operation == "open":
            store.record_open(replace(_position(), id="new"))
        else:
            store.remove_open("retained")
    assert store.open_path.read_bytes() == original
    assert store.ledger_path.read_bytes() == b"{broken"


def test_unreadable_legacy_file_reports_integrity_rejection(tmp_path):
    store = PositionsStore(tmp_path)
    store.open_path.mkdir()
    assert store.open_integrity_errors() == ("position_transaction_unresolved",)


@pytest.mark.parametrize("failure", ["new_nonfinite", "legacy_corrupt"])
def test_rejected_first_mutation_does_not_poison_empty_database(tmp_path, failure):
    store = PositionsStore(tmp_path)
    if failure == "legacy_corrupt":
        store.ledger_path.write_bytes(b"{broken")
    with pytest.raises(ValueError):
        store.record_open(
            replace(_position(), qty=float("nan"))
            if failure == "new_nonfinite"
            else _position()
        )
    # Fixture-only repair of a never-enrolled legacy file, not automatic recovery.
    if failure == "legacy_corrupt":
        store.ledger_path.write_bytes(b"[]")
    store.record_open(_position())
    assert store.load_snapshot(strict=True) == ([_position()], [])

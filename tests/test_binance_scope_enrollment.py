"""Scope enrollment must not assign legacy capital to new credentials."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import sqlite3
from threading import Barrier

import pytest

from simple_ai_trading.binance_close_intents import prepare_close
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_owned_execution import submit_durable_open
from simple_ai_trading.position_transactions import position_transaction
from simple_ai_trading.positions import ClosedTrade, OpenPosition, PositionsStore


def _scope(key="offline-new-account"):
    return BinanceExecutionScope.from_api_key(BINANCE_SPOT_TESTNET, "spot", key)


def _position(**changes):
    return replace(
        OpenPosition(
            id="new",
            symbol="BTCUSDT",
            market_type="spot",
            side="LONG",
            qty=0.001,
            entry_price=50000.0,
            leverage=1.0,
            opened_at_ms=1,
            notional=50.0,
            dry_run=False,
            open_client_order_id="sait-o-new",
        ),
        **changes,
    )


def _closed(**changes):
    return replace(
        ClosedTrade(
            id="old",
            symbol="BTCUSDT",
            market_type="spot",
            side="LONG",
            qty=0.001,
            entry_price=50000.0,
            exit_price=51000.0,
            leverage=1.0,
            opened_at_ms=1,
            closed_at_ms=2,
            realized_pnl=1.0,
            realized_pnl_pct=0.02,
            dry_run=False,
        ),
        **changes,
    )


def _row_counts(store):
    with sqlite3.connect(store.opening_intents.path) as connection:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in ("execution_scope", "open_intent")
        )


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("classification", [False, None, "true", 1])
def test_legacy_live_or_unclassified_history_never_acquires_scope(
    tmp_path, closed, classification, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    row = asdict(_closed() if closed else _position(id="legacy"))
    if classification is None:
        row.pop("dry_run")
    else:
        row["dry_run"] = classification
    path = store.ledger_path if closed else store.open_path
    original = json.dumps([row], indent=4).encode()
    path.write_bytes(original)

    class Client:
        writes = 0

        def execution_scope(self):
            return _scope()

        def place_order(self, *args, **kwargs):
            self.writes += 1
            raise AssertionError("unscoped history reached native dispatch")

    client = Client()
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage

    authority = enrolled_binance_runtime(store, _scope(), execution=False, start=False)
    binding = BinanceExecutionStorage(store.opening_intents, authority.control)
    existing = binding.inspect()
    with pytest.raises(OpenIntentError, match="migration"):
        binding.provision(
            scope=_scope(),
            expected_snapshot_sha256=existing.get("journal_snapshot_sha256"),
        )
    with pytest.raises(OpenIntentError):
        submit_durable_open(client, _position(), store)
    assert client.writes == 0
    assert _row_counts(store) == (0, 0)
    assert path.read_bytes() == original


def test_store_created_unscoped_live_inventory_is_not_reassigned(tmp_path):
    store = PositionsStore(tmp_path)
    legacy = _position(id="legacy", open_client_order_id="sait-o-legacy")
    store.record_open(legacy)
    before = store.open_path.read_bytes()
    with pytest.raises(OpenIntentError, match="migration"):
        store.opening_intents.prepare(_position(), scope=_scope())
    with pytest.raises(OpenIntentError, match="scope"):
        prepare_close(
            store, legacy, client_id="sait-c-legacy", scope=_scope(), reduce_only=True
        )
    assert _row_counts(store) == (0, 0)
    assert store.open_path.read_bytes() == before


@pytest.mark.parametrize("populated", [False, True])
def test_explicit_paper_pair_enrolls_exact_bytes_without_replacement(
    tmp_path, populated
):
    store = PositionsStore(tmp_path)
    bodies = (
        json.dumps(
            [asdict(_position(dry_run=True))] if populated else [], indent=4
        ).encode(),
        json.dumps(
            [asdict(_closed(dry_run=True))] if populated else [], indent=3
        ).encode(),
    )
    store.open_path.write_bytes(bodies[0])
    store.ledger_path.write_bytes(bodies[1])
    store.opening_intents.prepare(_position(), scope=_scope())
    assert (store.open_path.read_bytes(), store.ledger_path.read_bytes()) == bodies
    assert _row_counts(store) == (1, 1)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
        assert connection.execute(
            "SELECT state, before_open, before_closed, after_open, after_closed FROM position_replacement"
        ).fetchone() == ("APPLIED", *bodies, *bodies)


@pytest.mark.parametrize(
    "body",
    [b"{broken", b"[{}]", b'[{"dry_run":true}]', b'[{"dry_run":true,"dry_run":false}]'],
)
def test_invalid_paper_history_does_not_enroll(tmp_path, body):
    store = PositionsStore(tmp_path)
    store.open_path.write_bytes(body)
    with pytest.raises(OpenIntentError):
        store.opening_intents.prepare(_position(), scope=_scope())
    assert _row_counts(store) == (0, 0)
    assert store.open_path.read_bytes() == body


@pytest.mark.parametrize("damage", ["scope_and_intents", "pair_table", "new_file"])
def test_enrolled_empty_namespace_never_becomes_fresh_again(tmp_path, damage):
    store = PositionsStore(tmp_path)
    store.opening_intents.prepare(_position(), scope=_scope())
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE open_intent SET state='RECORDED'")
        if damage == "scope_and_intents":
            connection.execute("DELETE FROM execution_scope")
            connection.execute("DELETE FROM open_intent")
        elif damage == "pair_table":
            connection.execute("DROP TABLE position_replacement")
    if damage == "new_file":
        store.open_path.write_bytes(b"[]")
    with pytest.raises(OpenIntentError):
        store.opening_intents.prepare(
            _position(id="two", open_client_order_id="sait-o-two"),
            scope=_scope("other"),
        )
    assert not store.ledger_path.exists()


def test_known_scope_projection_drift_rejects_before_native_write(tmp_path):
    store = PositionsStore(tmp_path)
    requested = _position()
    store.opening_intents.prepare(requested, scope=_scope())
    store.opening_intents.record_complete(
        requested,
        replace(requested, exchange_status="FILLED", open_exchange_order_id="1"),
        scope=_scope(),
    )
    store.ledger_path.write_bytes(b"[]")
    with pytest.raises(OpenIntentError) as failure:
        store.opening_intents.prepare(
            _position(id="two", open_client_order_id="sait-o-two"), scope=_scope()
        )
    assert "differs from committed" in str(failure.value.__cause__)
    assert _row_counts(store) == (1, 1)


@pytest.mark.parametrize("populated", [False, True])
def test_unscoped_auxiliary_history_cannot_be_adopted(tmp_path, populated):
    store = PositionsStore(tmp_path)
    with position_transaction(store.opening_intents, write=True) as transaction:
        transaction.connection.execute(
            'CREATE TABLE "unusual""execution" (identity TEXT)'
        )
        if populated:
            transaction.connection.execute(
                'INSERT INTO "unusual""execution" VALUES (\'old\')'
            )
    if populated:
        with pytest.raises(OpenIntentError, match="history"):
            store.opening_intents.prepare(_position(), scope=_scope())
        assert _row_counts(store) == (0, 0)
    else:
        store.opening_intents.prepare(_position(), scope=_scope())
        assert _row_counts(store) == (1, 1)


def test_cancellation_rolls_back_scope_version_pair_and_intent_together(
    tmp_path, monkeypatch
):
    store = PositionsStore(tmp_path)
    original = BinanceOpenIntentJournal._prepare_scope

    def interrupted(self, transaction, scope):
        original(self, transaction, scope)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(BinanceOpenIntentJournal, "_prepare_scope", interrupted)
        with pytest.raises(KeyboardInterrupt):
            store.opening_intents.prepare(_position(), scope=_scope())
    assert _row_counts(store) == (0, 0)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert (
            connection.execute("PRAGMA table_info(position_replacement)").fetchall()
            == []
        )
    assert not store.open_path.exists()
    assert not store.ledger_path.exists()
    store.opening_intents.prepare(_position(), scope=_scope("replacement"))


def test_competing_first_scopes_cannot_downgrade_or_share_enrollment(
    tmp_path, monkeypatch
):
    store = PositionsStore(tmp_path)
    barrier = Barrier(2)
    connect = sqlite3.connect

    def simultaneous_connect(*args, **kwargs):
        # Both _connect calls must observe an absent file before either creates
        # it, exercising the initializer that waits behind an enrolled writer.
        barrier.wait(timeout=10)
        return connect(*args, **kwargs)

    def attempt(index):
        scope = _scope(str(index))
        position = _position(id=str(index), open_client_order_id=f"sait-o-{index}")
        try:
            store.opening_intents.prepare(position, scope=scope)
            return scope, position
        except OpenIntentError:
            return None

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", simultaneous_connect)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, range(2)))
    winners = [outcome for outcome in outcomes if outcome is not None]
    assert len(winners) == 1
    scope, position = winners[0]
    assert store.opening_intents.pending_position(scope=scope) == position
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)
    assert _row_counts(store) == (1, 1)


def test_bound_v2_retains_recovery_and_owned_close_but_requires_opening_migration(
    tmp_path,
):
    store = PositionsStore(tmp_path)
    requested = _position()
    store.opening_intents.prepare(requested, scope=_scope())
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("PRAGMA user_version=2")
    assert store.opening_intents.pending_position(scope=_scope()) == requested
    recorded = replace(requested, exchange_status="FILLED", open_exchange_order_id="1")
    store.record_open(recorded)
    store.opening_intents.record_complete(requested, recorded, scope=_scope())
    with pytest.raises(OpenIntentError, match="migration"):
        store.opening_intents.prepare(
            _position(id="two", open_client_order_id="sait-o-two"), scope=_scope()
        )
    prepare_close(
        store, recorded, client_id="sait-c-owned", scope=_scope(), reduce_only=True
    )
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize("damage", ["read_only", "projection_changed"])
def test_unchanged_enrollment_requires_writer_and_matching_snapshot(tmp_path, damage):
    store = PositionsStore(tmp_path)
    with pytest.raises(ValueError, match="active writer|files changed"):
        with position_transaction(
            store.opening_intents, write=damage != "read_only"
        ) as transaction:
            if damage == "projection_changed":
                store.open_path.write_bytes(b"[]")
            transaction.enroll_unchanged()
    assert not store.ledger_path.exists()


def test_failed_unknown_insert_does_not_commit_an_enrollment(tmp_path):
    store = PositionsStore(tmp_path)
    with position_transaction(store.opening_intents, write=True) as transaction:
        transaction.connection.execute(
            "CREATE TRIGGER reject_intent BEFORE INSERT ON open_intent "
            "BEGIN SELECT RAISE(ABORT, 'injected insert failure'); END"
        )
    with pytest.raises(OpenIntentError):
        store.opening_intents.prepare(_position(), scope=_scope())
    assert _row_counts(store) == (0, 0)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert (
            connection.execute("PRAGMA table_info(position_replacement)").fetchall()
            == []
        )
    assert not store.open_path.exists()
    assert not store.ledger_path.exists()

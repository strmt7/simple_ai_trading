"""Closing obligations survive partial responses, crashes and competing callers."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.autonomous import _close_to_trade, _submit_durable_close_position
from simple_ai_trading.binance_close_intents import complete_close, prepare_close
from simple_ai_trading.binance_execution_scope import (
    BinanceExecutionScope,
    BINANCE_SPOT_TESTNET,
    BINANCE_FUTURES_TESTNET,
)
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _scope(product="spot", key="offline-placeholder"):
    return BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET if product == "spot" else BINANCE_FUTURES_TESTNET,
        product,
        key,
    )


def _stage(tmp_path, *, product="spot", side="LONG", identity="owned"):
    store = PositionsStore(tmp_path)
    requested = OpenPosition(
        id=identity,
        symbol="BTCUSDT",
        market_type=product,
        side=side,
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id=f"sait-o-{identity}",
    )
    scope = _scope(product)
    store.opening_intents.prepare(requested, scope=scope)
    position = replace(
        requested, exchange_status="FILLED", open_exchange_order_id="100"
    )
    store.record_open(position)
    store.opening_intents.record_complete(requested, position, scope=scope)
    return store, position


class _Client:
    def __init__(
        self, store, *, product="spot", changes=None, failure=None, lost=False
    ):
        self.store, self.product = store, product
        self.changes, self.failure, self.lost = changes or {}, failure, lost
        self.writes = self.queries = 0
        self.order = None

    def execution_scope(self):
        return _scope(self.product)

    def place_order(self, symbol, side, quantity, **kwargs):
        assert kwargs["expected_scope"] == self.execution_scope()
        assert (
            self.store.opening_intents.entry_block_reason()
            == "unresolved_closing_intents=1"
        )
        self.writes += 1
        if self.failure:
            raise self.failure
        self.order = {
            "symbol": symbol,
            "side": side,
            "clientOrderId": kwargs["client_order_id"],
            "orderId": 200,
            "status": "FILLED",
            "executedQty": str(quantity),
            "avgPrice": "101",
            "reduceOnly": kwargs["reduce_only"],
            "positionSide": "BOTH",
            **self.changes,
        }
        if self.lost:
            raise BinanceAPIError("injected response loss")
        return self.order

    def get_order(
        self, symbol, *, orig_client_order_id, expected_scope, expected_order_binding
    ):
        assert expected_scope == self.execution_scope()
        self.queries += 1
        if self.order is None:
            raise BinanceAPIError("injected query uncertainty")
        assert self.order["clientOrderId"] == orig_client_order_id
        return self.order


def _close(client, store, position):
    trade = _close_to_trade(position, 101, "test-close", clock=lambda: 2)
    return _submit_durable_close_position(
        client, position, trade, store, reduce_only=True
    )


@pytest.mark.parametrize(
    "product,side", [("spot", "LONG"), ("futures", "LONG"), ("futures", "SHORT")]
)
@pytest.mark.parametrize("lost", [False, True])
def test_full_close_commits_before_release_and_queries_only_exact_id(
    tmp_path, product, side, lost
):
    store, position = _stage(tmp_path, product=product, side=side)
    client = _Client(store, product=product, lost=lost)
    trade = _close(client, store, position)
    assert client.writes == 1 and client.queries == int(lost)
    assert store.load_snapshot(strict=True) == ([], [trade])
    assert store.opening_intents.entry_block_reason() is None
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("SELECT state FROM close_intent").fetchone() == (
            "RECORDED",
        )
    with pytest.raises(OpenIntentError, match="changed before submission"):
        _close(client, store, position)
    assert client.writes == 1


def test_partial_blocks_second_order_and_direct_new_entry_after_restart(tmp_path):
    store, position = _stage(tmp_path)
    client = _Client(
        store, changes={"status": "PARTIALLY_FILLED", "executedQty": "0.4"}
    )
    trade = _close(client, store, position)
    reopened = PositionsStore(tmp_path)
    remaining = reopened.load_open(strict=True)[0]
    assert remaining.qty == pytest.approx(0.6)
    assert reopened.load_ledger() == [trade]
    with pytest.raises(OpenIntentError, match="unresolved closing"):
        _close(client, reopened, remaining)
    with pytest.raises(OpenIntentError, match="blocks new exposure"):
        reopened.opening_intents.prepare(
            replace(position, id="new", open_client_order_id="sait-o-new"),
            scope=_scope(),
        )
    assert client.writes == 1
    assert reopened.load_ledger() == [trade]


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "ETHUSDT"},
        {"clientOrderId": "foreign"},
        {"origClientOrderId": "foreign"},
        {"side": "BUY"},
        {"side": None},
        {"status": "NEW"},
        {"status": "REJECTED"},
        {"orderId": ""},
        {"executedQty": "0.5"},
        {"executedQty": "2"},
        {"executedQty": "nan"},
        {"avgPrice": "0"},
    ],
)
def test_unresolved_or_foreign_ack_preserves_owned_lot_and_barrier(tmp_path, changes):
    store, position = _stage(tmp_path)
    client = _Client(store, changes=changes)
    with pytest.raises((OpenIntentError, BinanceAPIError)):
        _close(client, store, position)
    assert store.load_snapshot() == ([position], [])
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "failure", [BinanceAPIError("injected timeout"), KeyboardInterrupt()]
)
def test_unresolved_transport_and_interruption_never_resubmit(tmp_path, failure):
    store, position = _stage(tmp_path)
    client = _Client(store, failure=failure)
    with pytest.raises((BinanceAPIError, KeyboardInterrupt)):
        _close(client, store, position)
    with pytest.raises(OpenIntentError, match="unresolved closing"):
        _close(client, PositionsStore(tmp_path), position)
    assert client.writes == 1
    assert store.load_snapshot() == ([position], [])


def test_wrong_scope_cannot_adopt_or_close_existing_lot(tmp_path):
    store, position = _stage(tmp_path)
    with pytest.raises(OpenIntentError, match="different or missing execution scope"):
        prepare_close(
            store,
            position,
            client_id="sait-c-owned",
            scope=_scope(key="other-offline-placeholder"),
            reduce_only=True,
        )
    assert store.load_snapshot() == ([position], [])


def test_legacy_unbound_lot_cannot_inherit_current_account(tmp_path):
    store, position = _stage(tmp_path / "scoped")
    legacy = PositionsStore(tmp_path / "legacy")
    legacy.record_open(position)
    client = _Client(legacy)
    with pytest.raises(OpenIntentError, match="different or missing execution scope"):
        _close(client, legacy, position)
    assert client.writes == 0


def test_futures_close_requires_reduce_only_before_transmission(tmp_path):
    store, position = _stage(tmp_path, product="futures")
    with pytest.raises(OpenIntentError, match="reduction scope"):
        prepare_close(
            store,
            position,
            client_id="sait-c-owned",
            scope=_scope("futures"),
            reduce_only=False,
        )


def test_crash_before_accounting_retains_barrier(tmp_path, monkeypatch):
    store, position = _stage(tmp_path)
    client = _Client(store)

    def fail(*args):
        raise OSError("injected accounting failure")

    monkeypatch.setattr(store, "record_close_result", fail)
    with pytest.raises(OSError):
        _close(client, store, position)
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    assert store.load_snapshot() == ([position], [])


def test_crash_after_accounting_cannot_duplicate_order_or_ledger(tmp_path, monkeypatch):
    from simple_ai_trading import binance_close_intents

    store, position = _stage(tmp_path)
    client = _Client(store)

    def fail(*args, **kwargs):
        raise OSError("injected completion failure")

    monkeypatch.setattr(binance_close_intents, "complete_close", fail)
    with pytest.raises(OSError):
        _close(client, store, position)
    assert store.load_open() == [] and len(store.load_ledger()) == 1
    with pytest.raises(OpenIntentError):
        _close(client, store, position)
    assert client.writes == 1
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


def test_one_unresolved_lot_does_not_block_other_owned_close(tmp_path):
    store, first = _stage(tmp_path, identity="first")
    _, second = _stage(tmp_path, identity="second")
    prepare_close(
        store, first, client_id="sait-c-first", scope=_scope(), reduce_only=True
    )
    # This preflight concerns owned close admission, never new exposure.
    prepare_close(
        store, second, client_id="sait-c-second", scope=_scope(), reduce_only=True
    )
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=2"


def test_concurrent_callers_commit_only_one_close(tmp_path):
    store, position = _stage(tmp_path)
    client = _Client(store)

    def attempt(_):
        try:
            return _close(client, PositionsStore(tmp_path), position)
        except OpenIntentError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, range(2)))
    assert sum(value is not None for value in outcomes) == client.writes == 1
    assert store.load_open() == [] and len(store.load_ledger()) == 1


def test_actual_child_exit_at_submission_retains_unknown(tmp_path):
    store, position = _stage(tmp_path)
    script = """
import json, os, sys
from simple_ai_trading.autonomous import _close_to_trade, _submit_durable_close_position
from simple_ai_trading.binance_execution_scope import BinanceExecutionScope, BINANCE_SPOT_TESTNET
from simple_ai_trading.positions import OpenPosition, PositionsStore
class Client:
    def execution_scope(self):
        return BinanceExecutionScope.from_api_key(BINANCE_SPOT_TESTNET, 'spot', 'offline-placeholder')
    def place_order(self, *args, **kwargs):
        os._exit(73)
position = OpenPosition(**json.loads(sys.argv[2]))
_submit_durable_close_position(Client(), position, _close_to_trade(position, 101, 'child', clock=lambda: 2), PositionsStore(sys.argv[1]), reduce_only=True)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(asdict(position))],
        timeout=20,
        check=False,
    )
    assert child.returncode == 73
    assert (
        PositionsStore(tmp_path).opening_intents.entry_block_reason()
        == "unresolved_closing_intents=1"
    )
    client = _Client(store)
    with pytest.raises(OpenIntentError, match="unresolved closing"):
        _close(client, store, position)
    assert client.writes == 0


def test_completion_requires_unique_persisted_full_trade(tmp_path):
    store, position = _stage(tmp_path)
    client_id = "sait-c-owned"
    prepare_close(
        store, position, client_id=client_id, scope=_scope(), reduce_only=True
    )
    trade = replace(
        _close_to_trade(position, 101, "test", clock=lambda: 2),
        close_client_order_id=client_id,
        close_exchange_order_id="200",
        exchange_status="FILLED",
    )
    with pytest.raises(OpenIntentError, match="not uniquely persisted"):
        complete_close(
            store,
            position,
            trade,
            client_id=client_id,
            scope=_scope(),
            reduce_only=True,
        )
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "changes", [{"reduceOnly": False}, {"positionSide": "LONG"}, {"reduceOnly": "true"}]
)
def test_futures_ack_requires_one_way_reduction_evidence(tmp_path, changes):
    store, position = _stage(tmp_path, product="futures")
    client = _Client(store, product="futures", changes=changes)
    with pytest.raises(BinanceAPIError, match="identity or status"):
        _close(client, store, position)
    assert store.load_snapshot() == ([position], [])


def test_changed_lot_after_submission_is_not_silently_removed(tmp_path):
    store, position = _stage(tmp_path)

    class ConcurrentFillClient(_Client):
        def place_order(self, *args, **kwargs):
            order = super().place_order(*args, **kwargs)
            store.record_open(replace(position, qty=2, notional=200))
            return order

    client = ConcurrentFillClient(store)
    with pytest.raises(ValueError, match="changed before recording"):
        _close(client, store, position)
    assert store.load_open()[0].qty == 2
    assert store.load_ledger() == []
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "client_id,scope,reduce_only",
    [
        ("foreign", _scope(), True),
        ("sait-c-owned", None, True),
        ("sait-c-owned", _scope(), "true"),
    ],
)
def test_invalid_close_authority_is_rejected_before_a_journal_row(
    tmp_path, client_id, scope, reduce_only
):
    store, position = _stage(tmp_path)
    with pytest.raises(OpenIntentError):
        prepare_close(
            store, position, client_id=client_id, scope=scope, reduce_only=reduce_only
        )
    assert store.opening_intents.entry_block_reason() is None


def test_corrupt_closing_schema_blocks_new_entry_and_close(tmp_path):
    store, position = _stage(tmp_path)
    prepare_close(
        store, position, client_id="sait-c-owned", scope=_scope(), reduce_only=True
    )
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("ALTER TABLE close_intent ADD COLUMN unexpected TEXT")
    assert (
        store.opening_intents.entry_block_reason()
        == "opening_intent_journal_unreadable"
    )
    with pytest.raises(OpenIntentError, match="schema"):
        store.opening_intents.prepare(
            replace(position, id="new", open_client_order_id="sait-o-new"),
            scope=_scope(),
        )
    with pytest.raises(OpenIntentError, match="schema"):
        _close(_Client(store), store, position)

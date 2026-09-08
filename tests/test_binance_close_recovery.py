"""Terminal closing evidence retains native economics without replay or rearm."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.autonomous import _close_to_trade
from simple_ai_trading.binance_close_intents import prepare_close
from simple_ai_trading.binance_close_recovery import collect_closing_recovery
from simple_ai_trading.binance_execution_scope import (
    BINANCE_FUTURES_TESTNET,
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_terminal_fills import validate_terminal_fills
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _scope(product="spot", key="offline-placeholder"):
    return BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET if product == "spot" else BINANCE_FUTURES_TESTNET,
        product,
        key,
    )


def _stage(tmp_path, product="spot", side="LONG"):
    store = PositionsStore(tmp_path)
    requested = OpenPosition(
        id="recover-close",
        symbol="BTCUSDT",
        market_type=product,
        side=side,
        qty=0.002,
        entry_price=50000,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-recover-close",
    )
    scope = _scope(product)
    store.opening_intents.prepare(requested, scope=scope)
    position = replace(
        requested, exchange_status="FILLED", open_exchange_order_id="987"
    )
    store.record_open(position)
    store.opening_intents.record_complete(requested, position, scope=scope)
    client_id = "sait-c-recover-close"
    prepare_close(store, position, client_id=client_id, scope=scope, reduce_only=True)
    order = {
        "symbol": "BTCUSDT",
        "orderId": 123,
        "clientOrderId": client_id,
        "side": "SELL" if side == "LONG" else "BUY",
        "type": "MARKET",
        "status": "FILLED",
        "origQty": "0.002",
        "executedQty": "0.002",
        "time": 10,
        "updateTime": 20,
        "cummulativeQuoteQty" if product == "spot" else "cumQuote": "110",
    }
    trade = {
        "symbol": "BTCUSDT",
        "orderId": 123,
        "id": 0,
        "price": "55000",
        "qty": "0.002",
        "quoteQty": "110",
        "commission": "0.11",
        "commissionAsset": "USDT",
        "time": 15,
    }
    if product == "spot":
        trade.update(isBuyer=False, isMaker=False)
    else:
        order.update(positionSide="BOTH", reduceOnly=True)
        trade.update(
            side=order["side"],
            positionSide="BOTH",
            buyer=side == "SHORT",
            maker=False,
            realizedPnl="10" if side == "LONG" else "-10",
        )
    return store, position, client_id, order, [trade]


class _Client:
    def __init__(self, order, trades, *, product="spot", callback=None, scope=None):
        self.order, self.trades, self.product = order, trades, product
        self.callback, self.scope = callback, scope
        self.calls = []

    def execution_scope(self):
        return self.scope or _scope(self.product)

    def get_order(self, symbol, *, orig_client_order_id, expected_scope):
        assert expected_scope == self.execution_scope()
        self.calls.append(("order", symbol, orig_client_order_id))
        if self.callback:
            self.callback()
        return self.order

    def get_order_trades(self, symbol, *, order_id, expected_scope):
        assert expected_scope == self.execution_scope()
        self.calls.append(("trades", symbol, order_id))
        return self.trades

    def place_order(self, *args, **kwargs):
        pytest.fail("recovery must not submit orders")


@pytest.mark.parametrize(
    "product,side", [("spot", "LONG"), ("futures", "LONG"), ("futures", "SHORT")]
)
def test_exact_closing_evidence_is_retained_once_without_rearm(tmp_path, product, side):
    store, position, client_id, order, trades = _stage(tmp_path, product, side)
    client = _Client(order, trades, product=product)
    result = collect_closing_recovery(client, store, client_id=client_id)
    assert client.calls == [
        ("order", "BTCUSDT", client_id),
        ("trades", "BTCUSDT", "123"),
    ]
    assert result.execution.order.order_id == "123" != position.open_exchange_order_id
    assert result.execution.executed_quantity == "0.002"
    assert result.execution.quote_quantity == "110"
    assert result.execution.commissions == (("USDT", "0.11"),)
    assert result.futures_realized_pnl == (
        None if product == "spot" else "10" if side == "LONG" else "-10"
    )
    assert (
        collect_closing_recovery(client, PositionsStore(tmp_path), client_id=client_id)
        == result
    )
    assert len(client.calls) == 2
    assert store.load_snapshot() == ([position], [])
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM closing_recovery"
        ).fetchone() == (1,)


def test_partial_terminal_uses_original_intent_not_remaining_inventory(tmp_path):
    store, position, client_id, order, trades = _stage(tmp_path)
    first = replace(
        _close_to_trade(position, 55000, "partial", clock=lambda: 1),
        qty=0.001,
        close_client_order_id=client_id,
        close_exchange_order_id="123",
        exchange_status="PARTIALLY_FILLED",
    )
    store.record_close_result(position, first)
    order.update(status="CANCELED", executedQty="0.0015", cummulativeQuoteQty="82.5")
    trades[0].update(
        qty="0.001", quoteQty="55", commission="0.000001", commissionAsset="BTC"
    )
    trades.append(
        dict(
            trades[0],
            id=1,
            qty="0.0005",
            quoteQty="27.5",
            commission="-0.01",
            commissionAsset="USDT",
        )
    )
    before = store.load_snapshot()
    result = collect_closing_recovery(
        _Client(order, trades), store, client_id=client_id
    )
    assert result.execution.executed_quantity == "0.0015"
    assert result.execution.order.original_quantity == "0.002"
    assert result.execution.commissions == (("BTC", "0.000001"), ("USDT", "-0.01"))
    assert store.load_snapshot() == before  # No double application of the first .001.
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "status", ["CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"]
)
@pytest.mark.parametrize("product", ["spot", "futures"])
def test_zero_terminal_requires_no_trade_query_but_never_rearms(
    tmp_path, status, product
):
    store, position, client_id, order, _ = _stage(tmp_path, product)
    order.update(status=status, executedQty="0")
    order["cummulativeQuoteQty" if product == "spot" else "cumQuote"] = "0"
    client = _Client(order, [], product=product)
    result = collect_closing_recovery(client, store, client_id=client_id)
    assert result.execution.executed_quantity == "0" and result.execution.fills == ()
    assert result.futures_realized_pnl == (None if product == "spot" else "0")
    assert len(client.calls) == 1 and store.load_open() == [position]
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "NEW"},
        {"status": "PARTIALLY_FILLED"},
        {"clientOrderId": "foreign"},
        {"origClientOrderId": "foreign"},
        {"symbol": "ETHUSDT"},
        {"side": "BUY"},
        {"type": "LIMIT"},
        {"origQty": "0.003"},
        {"executedQty": "0.001"},
        {"executedQty": "NaN"},
        {"executedQty": True},
        {"orderId": True},
        {"orderId": "123 "},
        {"updateTime": 9},
        {"time": 0},
    ],
)
def test_invalid_terminal_order_stops_before_trade_access(tmp_path, change):
    store, position, client_id, order, trades = _stage(tmp_path)
    order.update(change)
    client = _Client(order, trades)
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id=client_id)
    assert len(client.calls) == 1 and store.load_snapshot() == ([position], [])


@pytest.mark.parametrize(
    "change",
    [
        {"isBuyer": True},
        {"isMaker": True},
        {"id": True},
        {"orderId": 987},
        {"qty": "0.001"},
        {"quoteQty": "109"},
        {"commission": "NaN"},
        {"commissionAsset": ""},
        {"price": "0"},
        {"time": 21},
    ],
)
def test_invalid_trade_evidence_never_creates_a_recovery_record(tmp_path, change):
    store, _, client_id, order, trades = _stage(tmp_path)
    trades[0].update(change)
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(_Client(order, trades), store, client_id=client_id)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert not connection.execute("PRAGMA table_info(closing_recovery)").fetchall()
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "target,change",
    [
        ("order", {"reduceOnly": False}),
        ("order", {"reduceOnly": "true"}),
        ("order", {"positionSide": "LONG"}),
        ("trade", {"buyer": True}),
        ("trade", {"positionSide": "SHORT"}),
        ("trade", {"realizedPnl": "NaN"}),
        ("trade", {"realizedPnl": 10}),
    ],
)
def test_futures_closing_semantics_cannot_be_inferred(tmp_path, target, change):
    store, _, client_id, order, trades = _stage(tmp_path, "futures")
    (order if target == "order" else trades[0]).update(change)
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(
            _Client(order, trades, product="futures"), store, client_id=client_id
        )


@pytest.mark.parametrize("kind", ["empty", "duplicate", "over_budget", "not_list"])
def test_fill_page_is_exact_bounded_and_complete(tmp_path, kind):
    store, _, client_id, order, trades = _stage(tmp_path)
    variants = {
        "empty": [],
        "duplicate": trades * 2,
        "over_budget": trades * 1001,
        "not_list": {},
    }
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(
            _Client(order, variants[kind]), store, client_id=client_id
        )


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_unvalidated_extra_fields_are_excluded_from_retention(tmp_path, product):
    store, _, client_id, order, trades = _stage(tmp_path, product)
    order["irrelevant"] = "unvalidated-extra-marker"
    trades[0]["irrelevant"] = "unvalidated-extra-marker"
    if product == "spot":
        order["reduceOnly"] = "unvalidated-extra-marker"
        trades[0]["realizedPnl"] = "unvalidated-extra-marker"
    else:
        order["cummulativeQuoteQty"] = "unvalidated-extra-marker"
        trades[0]["isBuyer"] = "unvalidated-extra-marker"
    collect_closing_recovery(
        _Client(order, trades, product=product), store, client_id=client_id
    )
    with sqlite3.connect(store.opening_intents.path) as connection:
        row = connection.execute(
            "SELECT order_json, trades_json, evidence_json FROM closing_recovery"
        ).fetchone()
    assert all("unvalidated-extra-marker" not in value for value in row)


@pytest.mark.parametrize(
    "column", ["request_json", "order_json", "trades_json", "evidence_json"]
)
def test_corrupt_cached_evidence_is_not_overwritten_or_refetched(tmp_path, column):
    store, _, client_id, order, trades = _stage(tmp_path)
    client = _Client(order, trades)
    collect_closing_recovery(client, store, client_id=client_id)
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute(f"UPDATE closing_recovery SET {column}='null'")
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id=client_id)
    assert len(client.calls) == 2


def test_absent_pending_and_foreign_identity_do_not_query_or_create(tmp_path):
    store = PositionsStore(tmp_path)
    client = _Client({}, [])
    assert collect_closing_recovery(client, store, client_id="sait-c-none") is None
    assert not store.opening_intents.path.exists()
    store, _, client_id, _, _ = _stage(tmp_path)
    assert collect_closing_recovery(client, store, client_id="sait-c-missing") is None
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id="sait-o-wrong")
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE close_intent SET state='RECORDED'")
    assert collect_closing_recovery(client, store, client_id=client_id) is None
    assert client.calls == []


def test_wrong_scope_or_malformed_intent_rejects_before_network(tmp_path):
    store, _, client_id, order, trades = _stage(tmp_path)
    client = _Client(order, trades, scope=_scope(key="different-offline-placeholder"))
    with pytest.raises(OpenIntentError, match="scope"):
        collect_closing_recovery(client, store, client_id=client_id)
    client.scope = None
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE close_intent SET request_json='{}'")
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id=client_id)
    assert client.calls == []


def test_obligation_change_during_query_is_not_committed(tmp_path):
    store, _, client_id, order, trades = _stage(tmp_path)

    def change():
        with sqlite3.connect(store.opening_intents.path) as connection:
            connection.execute("UPDATE close_intent SET state='RECORDED'")

    client = _Client(order, trades, callback=change)
    with pytest.raises(OpenIntentError, match="changed while"):
        collect_closing_recovery(client, store, client_id=client_id)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert not connection.execute("PRAGMA table_info(closing_recovery)").fetchall()


def test_failed_commit_preserves_unknown(tmp_path, monkeypatch):
    store, _, client_id, order, trades = _stage(tmp_path)
    original = BinanceOpenIntentJournal._connect

    def fail(self, **kwargs):
        if kwargs.get("write"):
            raise sqlite3.OperationalError("injected persistence failure")
        return original(self, **kwargs)

    monkeypatch.setattr(BinanceOpenIntentJournal, "_connect", fail)
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(_Client(order, trades), store, client_id=client_id)
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


def test_transport_failure_preserves_unknown(tmp_path):
    store, _, client_id, order, trades = _stage(tmp_path)

    def fail():
        raise BinanceAPIError("injected timeout")

    with pytest.raises(BinanceAPIError):
        collect_closing_recovery(
            _Client(order, trades, callback=fail), store, client_id=client_id
        )
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_terminal_observations_never_overwrite(tmp_path, conflict):
    store, _, client_id, order, trades = _stage(tmp_path)
    barrier = Barrier(2)

    def attempt(index):
        own_trades = [
            dict(trades[0], commission="0.12" if conflict and index else "0.11")
        ]
        try:
            return collect_closing_recovery(
                _Client(order, own_trades, callback=barrier.wait),
                store,
                client_id=client_id,
            )
        except OpenIntentError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(value is not None for value in results) == (1 if conflict else 2)
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM closing_recovery"
        ).fetchone() == (1,)
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


def test_closing_argument_never_weakens_default_opening_semantics(tmp_path):
    _, position, client_id, order, trades = _stage(tmp_path, "futures")
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, _scope("futures"), order, trades)
    for invalid in ("", "foreign", "sait-c-" + "x" * 30, True):
        with pytest.raises(OpenIntentError):
            validate_terminal_fills(
                position, _scope("futures"), order, trades, closing_client_id=invalid
            )
    assert (
        validate_terminal_fills(
            position, _scope("futures"), order, trades, closing_client_id=client_id
        ).executed_quantity
        == "0.002"
    )


def test_terminal_retention_survives_actual_child_exit(tmp_path):
    store, _, client_id, order, trades = _stage(tmp_path)
    script = """
import json, os, sys
from simple_ai_trading.binance_close_recovery import collect_closing_recovery
from simple_ai_trading.binance_execution_scope import BinanceExecutionScope, BINANCE_SPOT_TESTNET
from simple_ai_trading.positions import PositionsStore
class Client:
    def execution_scope(self):
        return BinanceExecutionScope.from_api_key(BINANCE_SPOT_TESTNET, 'spot', 'offline-placeholder')
    def get_order(self, *args, **kwargs): return json.loads(sys.argv[3])
    def get_order_trades(self, *args, **kwargs): return json.loads(sys.argv[4])
collect_closing_recovery(Client(), PositionsStore(sys.argv[1]), client_id=sys.argv[2])
os._exit(74)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            client_id,
            json.dumps(order),
            json.dumps(trades),
        ],
        timeout=20,
        check=False,
    )
    assert child.returncode == 74
    client = _Client({}, [])
    result = collect_closing_recovery(
        client, PositionsStore(tmp_path), client_id=client_id
    )
    assert result.execution.executed_quantity == "0.002" and client.calls == []
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "client_id",
    [True, "sait-c-", "sait-c-invalid!", "sait-c-a b", "sait-c-" + "a" * 30],
)
def test_invalid_closing_identity_is_shared_pre_io_boundary(tmp_path, client_id):
    store, position, _, order, trades = _stage(tmp_path)
    client = _Client(order, trades)
    with pytest.raises(OpenIntentError, match="client ID"):
        collect_closing_recovery(client, store, client_id=client_id)
    with pytest.raises(OpenIntentError, match="client ID"):
        prepare_close(
            store, position, client_id=client_id, scope=_scope(), reduce_only=True
        )
    assert client.calls == []


@pytest.mark.parametrize(
    "corruption", ["schema", "duplicate", "request_identity", "state"]
)
def test_corrupt_obligation_rejects_before_transport(tmp_path, corruption):
    store, _, client_id, order, trades = _stage(tmp_path)
    with sqlite3.connect(store.opening_intents.path) as connection:
        if corruption == "schema":
            connection.execute("ALTER TABLE close_intent ADD COLUMN extra TEXT")
        elif corruption == "request_identity":
            connection.execute("UPDATE close_intent SET position_id='foreign'")
        else:
            connection.execute("ALTER TABLE close_intent RENAME TO prior_close_intent")
            connection.execute(
                "CREATE TABLE close_intent AS SELECT * FROM prior_close_intent"
            )
            if corruption == "duplicate":
                connection.execute(
                    "INSERT INTO close_intent SELECT * FROM prior_close_intent"
                )
            else:
                connection.execute("UPDATE close_intent SET state='other'")
    client = _Client(order, trades)
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id=client_id)
    assert client.calls == []


@pytest.mark.parametrize("corruption", ["schema", "duplicate"])
def test_ambiguous_cached_storage_never_refetches(tmp_path, corruption):
    store, _, client_id, order, trades = _stage(tmp_path)
    client = _Client(order, trades)
    collect_closing_recovery(client, store, client_id=client_id)
    with sqlite3.connect(store.opening_intents.path) as connection:
        if corruption == "schema":
            connection.execute("ALTER TABLE closing_recovery ADD COLUMN extra TEXT")
        else:
            connection.execute(
                "ALTER TABLE closing_recovery RENAME TO prior_closing_recovery"
            )
            connection.execute(
                "CREATE TABLE closing_recovery AS SELECT * FROM prior_closing_recovery"
            )
            connection.execute(
                "INSERT INTO closing_recovery SELECT * FROM prior_closing_recovery"
            )
    with pytest.raises(OpenIntentError):
        collect_closing_recovery(client, store, client_id=client_id)
    assert len(client.calls) == 2


def test_futures_realized_pnl_is_exact_signed_sum_not_local_account_pnl(tmp_path):
    store, _, client_id, order, trades = _stage(tmp_path, "futures")
    trades[0].update(
        qty="0.001", quoteQty="55", realizedPnl="0.123456789012345678901234567891"
    )
    trades.append(
        dict(trades[0], id=1, realizedPnl="-0.123456789012345678901234567890")
    )
    result = collect_closing_recovery(
        _Client(order, trades, product="futures"), store, client_id=client_id
    )
    assert result.futures_realized_pnl == "0.000000000000000000000000000001"

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.binance_execution_scope import (
    BINANCE_FUTURES_TESTNET,
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_open_recovery import collect_opening_recovery
from simple_ai_trading.binance_terminal_fills import validate_terminal_fills
from simple_ai_trading.positions import OpenPosition


def _case(product="spot", side="LONG"):
    scope = BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET if product == "spot" else BINANCE_FUTURES_TESTNET,
        product,
        "offline-placeholder",
    )
    position = OpenPosition(
        id="recovery-one",
        symbol="BTCUSDT",
        market_type=product,
        side=side,
        qty=0.002,
        entry_price=50_000,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-recovery-one",
    )
    order = {
        "symbol": "BTCUSDT",
        "orderId": 123,
        "clientOrderId": position.open_client_order_id,
        "side": "BUY" if side == "LONG" else "SELL",
        "type": "MARKET",
        "status": "FILLED",
        "origQty": "0.002",
        "executedQty": "0.002",
        "time": 10,
        "updateTime": 20,
        "cummulativeQuoteQty" if product == "spot" else "cumQuote": "100",
    }
    trade = {
        "symbol": "BTCUSDT",
        "orderId": 123,
        "id": 0,
        "price": "50000",
        "qty": "0.002",
        "quoteQty": "100",
        "commission": "0.000002",
        "commissionAsset": "BTC",
        "time": 15,
    }
    if product == "spot":
        trade.update(isBuyer=side == "LONG", isMaker=False)
    else:
        order.update(positionSide="BOTH", reduceOnly=False)
        trade.update(
            side=order["side"],
            positionSide="BOTH",
            buyer=side == "LONG",
            maker=False,
            realizedPnl="0",
        )
    return position, scope, order, [trade]


@pytest.mark.parametrize(
    "product,side", [("spot", "LONG"), ("futures", "LONG"), ("futures", "SHORT")]
)
def test_terminal_fills_preserve_native_fees_without_rearming(product, side):
    position, scope, order, trades = _case(product, side)
    evidence = validate_terminal_fills(position, scope, order, trades)
    assert evidence.executed_quantity == "0.002"
    assert evidence.quote_quantity == "100"
    assert evidence.commissions == (("BTC", "0.000002"),)
    assert evidence.fills[0].trade_id == "0"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "NEW"},
        {"status": "PARTIALLY_FILLED"},
        {"clientOrderId": "foreign"},
        {"origClientOrderId": "foreign"},
        {"symbol": "ETHUSDT"},
        {"side": "SELL"},
        {"type": "LIMIT"},
        {"origQty": "0.003"},
        {"executedQty": "0.001"},
        {"executedQty": "NaN"},
        {"executedQty": True},
        {"orderId": True},
        {"orderId": "123 "},
        {"updateTime": 9},
        {"time": 0},
        {"status": "filled"},
        {"cummulativeQuoteQty": "99"},
        {"cummulativeQuoteQty": "-1"},
    ],
)
def test_order_identity_status_and_cumulative_evidence_reject(change):
    position, scope, order, trades = _case()
    order.update(change)
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "ETHUSDT"),
        ("orderId", 124),
        ("id", True),
        ("qty", "0.001"),
        ("quoteQty", "99"),
        ("price", "0"),
        ("commission", "NaN"),
        ("commissionAsset", ""),
        ("time", 21),
        ("time", 9),
        ("isBuyer", False),
        ("isBuyer", "true"),
        ("isMaker", 0),
        ("qty", 0.002),
    ],
)
def test_trade_evidence_rejects_missing_mismatched_or_lossy_fields(field, value):
    position, scope, order, trades = _case()
    trades[0][field] = value
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)
    trades[0].pop(field)
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)


@pytest.mark.parametrize(
    "status", ["CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"]
)
def test_exact_zero_fill_terminal_has_no_inventory_or_fees(status):
    position, scope, order, _ = _case()
    order.update(status=status, executedQty="0", cummulativeQuoteQty="0")
    evidence = validate_terminal_fills(position, scope, order, [])
    assert evidence.executed_quantity == "0" and evidence.fills == ()
    assert evidence.commissions == ()


def test_partial_terminal_and_multiple_fee_assets_use_exact_decimal_sums():
    position, scope, order, trades = _case()
    order.update(status="CANCELED", executedQty="0.001", cummulativeQuoteQty="50")
    trades[0].update(
        qty="0.0004", quoteQty="20", commission="0.01", commissionAsset="USDT"
    )
    trades.append(
        dict(
            trades[0],
            id=1,
            qty="0.0006",
            quoteQty="30",
            commission="-0.00001",
            commissionAsset="BNB",
        )
    )
    evidence = validate_terminal_fills(position, scope, order, trades)
    assert evidence.executed_quantity == "0.001"
    assert evidence.commissions == (("BNB", "-0.00001"), ("USDT", "0.01"))
    assert (
        validate_terminal_fills(position, scope, order, list(reversed(trades)))
        == evidence
    )


@pytest.mark.parametrize(
    "trades_kind", ["missing", "duplicate", "wrong_shape", "over_budget"]
)
def test_no_completeness_inference_from_a_short_or_invalid_trade_page(trades_kind):
    position, scope, order, trades = _case()
    trades = {
        "missing": [],
        "duplicate": trades * 2,
        "wrong_shape": {},
        "over_budget": trades * 1001,
    }[trades_kind]
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)


@pytest.mark.parametrize(
    "target,change",
    [
        ("order", {"positionSide": "LONG"}),
        ("order", {"reduceOnly": True}),
        ("trade", {"positionSide": "SHORT"}),
        ("trade", {"side": "SELL"}),
        ("trade", {"buyer": False}),
        ("trade", {"realizedPnl": "1"}),
    ],
)
def test_futures_opening_semantics_are_not_inferred(target, change):
    position, scope, order, trades = _case("futures")
    (order if target == "order" else trades[0]).update(change)
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)


class _Client:
    def __init__(self, scope, order, trades):
        self.scope, self.order, self.trades = scope, order, trades
        self.calls = []

    def execution_scope(self):
        return self.scope

    def get_order(self, symbol, **kwargs):
        self.calls.append(("order", symbol, kwargs))
        if isinstance(self.order, Exception):
            raise self.order
        return self.order

    def get_order_trades(self, symbol, **kwargs):
        self.calls.append(("trades", symbol, kwargs))
        return self.trades


def test_recovery_persists_once_across_restart_and_never_clears_unknown(tmp_path):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    client = _Client(scope, order, trades)
    evidence = collect_opening_recovery(client, journal)
    assert client.calls == [
        (
            "order",
            "BTCUSDT",
            {
                "orig_client_order_id": position.open_client_order_id,
                "expected_scope": scope,
            },
        ),
        ("trades", "BTCUSDT", {"order_id": "123", "expected_scope": scope}),
    ]
    assert (
        collect_opening_recovery(client, BinanceOpenIntentJournal(journal.path))
        == evidence
    )
    assert len(client.calls) == 2  # A retained terminal observation is not refetched.
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"
    assert journal.pending_position(scope=scope) == position
    with sqlite3.connect(journal.path) as connection:
        payload = json.loads(
            connection.execute("SELECT evidence_json FROM opening_recovery").fetchone()[
                0
            ]
        )
    assert payload["commissions"] == [["BTC", "0.000002"]]
    assert payload["executed_quantity"] == "0.002"


@pytest.mark.parametrize(
    "failure", ["not_found", "wrong_order", "missing_trades", "different_scope"]
)
def test_recovery_failure_never_modifies_intent_or_fetches_foreign_fills(
    tmp_path, failure
):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    if failure == "not_found":
        order = BinanceAPIError("order not found")
    elif failure == "wrong_order":
        order["clientOrderId"] = "foreign"
    elif failure == "missing_trades":
        trades = []
    else:
        scope = replace(scope, credential_fingerprint="a" * 64)
    client = _Client(scope, order, trades)
    with pytest.raises((OpenIntentError, BinanceAPIError)):
        collect_opening_recovery(client, journal)
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"
    assert len(client.calls) <= (2 if failure == "missing_trades" else 1)


@pytest.mark.parametrize(
    "product,endpoint",
    [("spot", "/api/v3/myTrades"), ("futures", "/fapi/v1/userTrades")],
)
def test_order_trade_api_is_read_only_bounded_and_scope_bound(
    monkeypatch, product, endpoint
):
    client = BinanceClient(
        api_key="offline-placeholder",
        api_secret="offline-placeholder",
        market_type=product,
    )
    scope = client.execution_scope()
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(client, "_request", request)
    assert (
        client.get_order_trades("BTCUSDT", order_id="123", expected_scope=scope) == []
    )
    assert calls == [
        (
            ("GET", endpoint, {"symbol": "BTCUSDT", "orderId": "123", "limit": 1000}),
            {"signed": True, "expected_scope": scope},
        )
    ]


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_unvalidated_extra_fields_are_not_persisted(tmp_path, product):
    position, scope, order, trades = _case(product)
    order["unrelated_account_metadata"] = "discard-this-field"
    trades[0]["isBuyer" if product == "futures" else "buyer"] = "discard-this-field"
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    client = _Client(scope, order, trades)
    evidence = collect_opening_recovery(client, journal)
    assert collect_opening_recovery(client, journal) == evidence
    with sqlite3.connect(journal.path) as connection:
        record = connection.execute(
            "SELECT order_json, trades_json FROM opening_recovery"
        ).fetchone()
    assert all("discard-this-field" not in value for value in record)


def test_absent_intent_does_not_create_storage_or_query_a_venue(tmp_path):
    _, scope, order, trades = _case()
    client = _Client(scope, order, trades)
    journal = BinanceOpenIntentJournal(tmp_path / "absent.sqlite")
    assert collect_opening_recovery(client, journal) is None
    assert not journal.path.exists() and not client.calls


def test_zero_execution_uses_only_exact_terminal_order_query(tmp_path):
    position, scope, order, _ = _case()
    order.update(status="CANCELED", executedQty="0", cummulativeQuoteQty="0")
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    client = _Client(scope, order, None)
    assert collect_opening_recovery(client, journal).fills == ()
    assert len(client.calls) == 1
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


@pytest.mark.parametrize(
    "column,changed",
    [
        ("request_json", "{}"),
        ("order_json", "{}"),
        ("trades_json", "[]"),
        ("evidence_json", "{}"),
        ("order_json", "not-json"),
    ],
)
def test_corrupt_retained_evidence_blocks_without_refetch(tmp_path, column, changed):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    client = _Client(scope, order, trades)
    collect_opening_recovery(client, journal)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(f"UPDATE opening_recovery SET {column}=?", (changed,))
    with pytest.raises(OpenIntentError):
        collect_opening_recovery(client, journal)
    assert len(client.calls) == 2
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_changed_obligation_during_network_access_cannot_commit(tmp_path):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    client = _Client(scope, order, trades)

    def change_while_querying(*args, **kwargs):
        with sqlite3.connect(journal.path) as connection:
            connection.execute("UPDATE open_intent SET request_json='{}'")
        return trades

    client.get_order_trades = change_while_querying
    with pytest.raises(OpenIntentError, match="changed"):
        collect_opening_recovery(client, journal)
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_failed_persistence_preserves_unknown_obligation(tmp_path, monkeypatch):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    original = BinanceOpenIntentJournal._connect

    def connect(instance, **kwargs):
        if kwargs.get("write"):
            raise OSError("simulated storage failure")
        return original(instance, **kwargs)

    monkeypatch.setattr(BinanceOpenIntentJournal, "_connect", connect)
    with pytest.raises(OpenIntentError, match="retained"):
        collect_opening_recovery(_Client(scope, order, trades), journal)
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_scope_rotation_rejects_trade_reads_before_transport(product):
    client = BinanceClient(
        "offline-placeholder", "offline-placeholder", market_type=product
    )
    scope = client.execution_scope()
    client.api_key = "rotated-offline-placeholder"
    client.session.request = Mock(side_effect=AssertionError("network forbidden"))
    try:
        with pytest.raises(BinanceAPIError, match="scope"):
            client.get_order_trades("BTCUSDT", order_id=123, expected_scope=scope)
        client.session.request.assert_not_called()
    finally:
        client.session.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "BTCUSDT "},
        {"order_id": True},
        {"order_id": -1},
        {"order_id": "01"},
        {"expected_scope": None},
    ],
)
def test_trade_query_rejects_invalid_inputs_before_transport(changes):
    client = BinanceClient("offline-placeholder", "offline-placeholder")
    client.session.request = Mock(side_effect=AssertionError("network forbidden"))
    args = {
        "symbol": "BTCUSDT",
        "order_id": 123,
        "expected_scope": client.execution_scope(),
    }
    args.update(changes)
    try:
        with pytest.raises(BinanceAPIError):
            client.get_order_trades(**args)
        client.session.request.assert_not_called()
    finally:
        client.session.close()


@pytest.mark.parametrize("scope_kind", ["missing", "wrong_product", "exchange_id"])
def test_terminal_binding_rejects_missing_scope_or_recorded_identity(scope_kind):
    position, scope, order, trades = _case()
    if scope_kind == "missing":
        scope = None
    elif scope_kind == "wrong_product":
        scope = _case("futures")[1]
    else:
        position = replace(position, open_exchange_order_id="124")
    with pytest.raises(OpenIntentError):
        validate_terminal_fills(position, scope, order, trades)


def test_price_quantity_disagreement_is_not_hidden_by_matching_totals():
    position, scope, order, trades = _case()
    trades[0]["price"] = "49999.999999"
    with pytest.raises(OpenIntentError, match="precision"):
        validate_terminal_fills(position, scope, order, trades)


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_recovery_does_not_overwrite_terminal_evidence(tmp_path, conflict):
    position, scope, order, trades = _case()
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    barrier = Barrier(2)

    def recover(index):
        client = _Client(
            scope,
            order,
            [
                dict(
                    trades[0],
                    commission="0.000003" if conflict and index else "0.000002",
                )
            ],
        )
        query = client.get_order

        def synchronized(*args, **kwargs):
            barrier.wait(timeout=5)
            return query(*args, **kwargs)

        client.get_order = synchronized
        try:
            return collect_opening_recovery(client, journal)
        except OpenIntentError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(recover, (0, 1)))
    assert sum(isinstance(outcome, OpenIntentError) for outcome in outcomes) == int(
        conflict
    )
    with sqlite3.connect(journal.path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM opening_recovery").fetchone()[0]
            == 1
        )
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_terminal_observation_survives_abrupt_process_exit(tmp_path):
    script = """
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_binance_open_recovery import _case, _Client
from simple_ai_trading.binance_open_intents import BinanceOpenIntentJournal
from simple_ai_trading.binance_open_recovery import collect_opening_recovery
position, scope, order, trades = _case()
journal = BinanceOpenIntentJournal(Path(sys.argv[1]))
journal.prepare(position, scope=scope)
collect_opening_recovery(_Client(scope, order, trades), journal)
os._exit(71)
"""
    path = tmp_path / "intents.sqlite"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 71, result.stderr.decode()
    _, scope, order, trades = _case()
    client = _Client(scope, order, trades)
    journal = BinanceOpenIntentJournal(path)
    assert collect_opening_recovery(client, journal).executed_quantity == "0.002"
    assert client.calls == []
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"

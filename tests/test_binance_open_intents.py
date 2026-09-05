from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_execution_scope import (
    BinanceExecutionScope,
    BINANCE_SPOT_TESTNET,
    BINANCE_FUTURES_TESTNET,
)
from simple_ai_trading.autonomous import _submit_durable_open_position
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _position(**changes) -> OpenPosition:
    return replace(
        OpenPosition(
            id="intent-one",
            symbol="BTCUSDT",
            market_type="spot",
            side="LONG",
            qty=0.001,
            entry_price=50_000.0,
            leverage=1.0,
            opened_at_ms=1,
            notional=50.0,
            dry_run=False,
            open_client_order_id="sait-o-intentone",
        ),
        **changes,
    )


def _scope(market_type="spot"):
    origin = BINANCE_SPOT_TESTNET if market_type == "spot" else BINANCE_FUTURES_TESTNET
    return BinanceExecutionScope.from_api_key(
        origin, market_type, "offline-placeholder"
    )


class _Client:
    def __init__(self, store, *, failure=None, status="FILLED", market_type="spot"):
        self.store = store
        self.failure = failure
        self.status = status
        self.writes = 0
        self.market_type = market_type

    def execution_scope(self):
        return _scope(self.market_type)

    def place_order(self, symbol, side, quantity, **kwargs):
        assert (
            self.store.opening_intents.entry_block_reason()
            == "unresolved_opening_intents=1"
        )
        self.writes += 1
        if self.failure:
            raise self.failure
        return {
            "symbol": symbol,
            "side": side,
            "executedQty": str(
                quantity / 2 if self.status == "PARTIALLY_FILLED" else quantity
            ),
            "avgPrice": "50000",
            "status": self.status,
            "orderId": "123",
            "clientOrderId": kwargs["client_order_id"],
        }

    def get_order(self, *args, **kwargs):
        raise BinanceAPIError("query remains inconclusive")


@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_intent_precedes_submission_and_releases_after_full_fill(tmp_path, market_type):
    store = PositionsStore(tmp_path)
    client = _Client(store, market_type=market_type)
    result = _submit_durable_open_position(
        client, _position(market_type=market_type), store
    )
    assert client.writes == 1
    assert store.load_open() == [result]
    assert store.opening_intents.entry_block_reason() is None
    with sqlite3.connect(store.opening_intents.path) as connection:
        row = connection.execute(
            "SELECT request_json, state FROM open_intent"
        ).fetchone()
    assert row[1] == "RECORDED"
    assert set(json.loads(row[0])) == {
        "position_id",
        "client_id",
        "symbol",
        "market_type",
        "side",
        "quantity",
        "position_template",
        "intent_version",
        "execution_scope",
    }
    with pytest.raises(OpenIntentError):
        _submit_durable_open_position(client, _position(market_type=market_type), store)
    assert client.writes == 1


@pytest.mark.parametrize("failure", [BinanceAPIError("timeout"), KeyboardInterrupt()])
def test_failed_or_interrupted_submission_survives_store_reopen(tmp_path, failure):
    store = PositionsStore(tmp_path)
    client = _Client(store, failure=failure)
    with pytest.raises((BinanceAPIError, KeyboardInterrupt)):
        _submit_durable_open_position(client, _position(), store)
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open() == []
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    )
    with pytest.raises(OpenIntentError, match="unresolved"):
        _submit_durable_open_position(
            client, _position(id="two", open_client_order_id="sait-o-two"), reopened
        )
    assert client.writes == 1


def test_ledger_write_failure_preserves_obligation(tmp_path, monkeypatch):
    store = PositionsStore(tmp_path)
    client = _Client(store)

    def fail_write(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(store, "record_open", fail_write)
    with pytest.raises(OSError):
        _submit_durable_open_position(client, _position(), store)
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    assert client.writes == 1


def test_partial_fill_is_recorded_without_releasing_pending_remainder(tmp_path):
    store = PositionsStore(tmp_path)
    result = _submit_durable_open_position(
        _Client(store, status="PARTIALLY_FILLED"), _position(), store
    )
    assert store.load_open() == [result]
    assert result.qty == 0.0005
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"


@pytest.mark.parametrize(
    "changes",
    [
        {"dry_run": True},
        {"owner": "foreign"},
        {"qty": float("nan")},
        {"qty": 0.0},
        {"qty": True},
        {"open_client_order_id": ""},
        {"open_client_order_id": "sait-o-" + "x" * 40},
        {"side": "UNKNOWN"},
        {"market_type": "other"},
        {"id": ""},
        {"symbol": ""},
    ],
)
def test_invalid_intents_never_reach_client_or_create_journal(tmp_path, changes):
    store = PositionsStore(tmp_path)
    client = _Client(store)
    with pytest.raises(OpenIntentError):
        _submit_durable_open_position(client, _position(**changes), store)
    assert client.writes == 0
    assert not store.opening_intents.path.exists()


@pytest.mark.parametrize("contents", [b"", b"not a database"])
def test_existing_invalid_database_is_not_reinitialized(tmp_path, contents):
    journal = BinanceOpenIntentJournal(tmp_path / "journal.sqlite3")
    journal.path.write_bytes(contents)
    assert journal.entry_block_reason() == "opening_intent_journal_unreadable"
    with pytest.raises(OpenIntentError):
        journal.prepare(_position(), scope=_scope())
    assert journal.path.read_bytes() == contents


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "wrong"},
        {"open_client_order_id": "sait-o-wrong"},
        {"symbol": "ETHUSDT"},
        {"side": "SHORT"},
        {"entry_price": float("inf")},
        {"exchange_status": "NEW"},
        {"open_exchange_order_id": ""},
        {"qty": 0.0005},
        {"qty": 0.002},
    ],
)
def test_invalid_acknowledgement_keeps_obligation(tmp_path, changes):
    journal = BinanceOpenIntentJournal(tmp_path / "journal.sqlite3")
    requested = _position()
    journal.prepare(requested, scope=_scope())
    recorded = replace(
        requested, exchange_status="FILLED", open_exchange_order_id="123"
    )
    with pytest.raises(OpenIntentError):
        journal.record_complete(requested, replace(recorded, **changes), scope=_scope())
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_process_death_during_submission_retains_durable_intent(tmp_path):
    code = """
import os, sys
from simple_ai_trading.autonomous import _submit_durable_open_position
from simple_ai_trading.positions import OpenPosition, PositionsStore
from simple_ai_trading.binance_execution_scope import BinanceExecutionScope, BINANCE_SPOT_TESTNET
class Client:
    def execution_scope(self):
        return BinanceExecutionScope.from_api_key(BINANCE_SPOT_TESTNET, 'spot', 'offline-placeholder')
    def place_order(self, *args, **kwargs):
        os._exit(71)
position = OpenPosition(id='crash', symbol='BTCUSDT', market_type='spot',
    side='LONG', qty=0.001, entry_price=50000., leverage=1., opened_at_ms=1,
    notional=50., dry_run=False, open_client_order_id='sait-o-crash')
_submit_durable_open_position(Client(), position, PositionsStore(sys.argv[1]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], timeout=15, capture_output=True
    )
    assert result.returncode == 71
    reopened = PositionsStore(tmp_path)
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    )
    assert reopened.load_open() == []


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "ETHUSDT"},
        {"side": "SELL"},
        {"status": "NEW"},
        {"status": None},
        {"clientOrderId": None},
        {"origClientOrderId": "sait-o-other"},
    ],
)
def test_raw_acknowledgement_must_bind_exact_intent(tmp_path, changes):
    store = PositionsStore(tmp_path)

    class Client(_Client):
        def place_order(self, *args, **kwargs):
            return {**super().place_order(*args, **kwargs), **changes}

    client = Client(store)
    with pytest.raises(BinanceAPIError, match="acknowledgement"):
        _submit_durable_open_position(client, _position(), store)
    assert client.writes == 1
    assert store.load_open() == []
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"


def test_interruption_after_position_write_keeps_intent_blocked(tmp_path, monkeypatch):
    store = PositionsStore(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(BinanceOpenIntentJournal, "record_complete", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _submit_durable_open_position(_Client(store), _position(), store)
    assert len(PositionsStore(tmp_path).load_open()) == 1
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"


def test_recorded_journal_does_not_mask_missing_position_ledger(tmp_path):
    store = PositionsStore(tmp_path)
    client = _Client(store)
    _submit_durable_open_position(client, _position(), store)
    store.open_path.unlink()
    assert (
        store.opening_intents.entry_block_reason(positions_present=False)
        == "recorded_openings_missing_position_ledger"
    )
    with pytest.raises(OpenIntentError, match="missing_position_ledger"):
        _submit_durable_open_position(
            client, _position(id="two", open_client_order_id="sait-o-two"), store
        )
    assert client.writes == 1


def test_failed_intent_storage_prevents_transmission(tmp_path):
    store = PositionsStore(tmp_path)
    store.opening_intents.path.mkdir()
    client = _Client(store)
    with pytest.raises(OpenIntentError):
        _submit_durable_open_position(client, _position(), store)
    assert client.writes == 0


def test_concurrent_preparations_admit_only_one_unresolved_intent(tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "journal.sqlite3")

    def prepare(number):
        try:
            journal.prepare(
                _position(id=f"p{number}", open_client_order_id=f"sait-o-p{number}"),
                scope=_scope(),
            )
            return True
        except OpenIntentError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(prepare, [1, 2]))
    assert results.count(True) == 1
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_pending_intent_prevents_autonomous_restart_even_with_flat_account(tmp_path):
    from simple_ai_trading.autonomous import AutonomousConfig, run_loop
    from simple_ai_trading.execution_lifecycle import build_execution_lifecycle_plan
    from simple_ai_trading.reconciliation import ReconciliationReport
    from simple_ai_trading.types import RuntimeConfig, StrategyConfig

    store = PositionsStore(tmp_path)
    store.opening_intents.prepare(_position(), scope=_scope())
    client = _Client(store)
    client.base_url = "https://testnet.binance.vision"
    config = AutonomousConfig(
        positions_root=tmp_path,
        control_path=tmp_path / "state.json",
        log_path=tmp_path / "worker.log",
        dry_run=False,
    )
    # Deliberately non-credential placeholders; every venue operation is stubbed.
    runtime = RuntimeConfig(
        symbol="BTCUSDT",
        market_type="spot",
        testnet=True,
        api_key="offline-placeholder",
        api_secret="offline-placeholder",
        managed_usdc=1000.0,
    )

    report = ReconciliationReport(
        ok=True,
        market_type="spot",
        symbols_checked=["BTCUSDT"],
        local_open_count=0,
        local_live_open_count=0,
        local_paper_open_count=0,
        exchange_exposure_count=0,
    )
    plan = build_execution_lifecycle_plan(
        runtime,
        StrategyConfig(),
        store,
        action="start",
        effective_dry_run=False,
        reconciliation=report,
        require_api_budget_headroom=False,
    )
    assert plan.can_close is True
    assert [check.label for check in plan.checks if check.status == "block"] == [
        "opening intents"
    ]
    with pytest.raises(RuntimeError, match="unsafe execution lifecycle"):
        run_loop(
            client, runtime, StrategyConfig(), config, reconcile_fn=lambda *_: report
        )
    assert client.writes == 0
    assert not config.control_path.exists()

"""Exact native movements and durable idempotence without accounts or rearming."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from decimal import localcontext
from threading import Barrier
from unittest.mock import Mock

import pytest

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
from simple_ai_trading.binance_opening_inventory import retain_opening_inventory
from simple_ai_trading.positions import OpenPosition


def _case(
    tmp_path,
    product="spot",
    side="LONG",
    fee_asset="BTC",
    fee="0.000002",
    *,
    collect=True,
    base="BTC",
    quote="USDT",
):
    scope = BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET if product == "spot" else BINANCE_FUTURES_TESTNET,
        product,
        "offline-placeholder",
    )
    position = OpenPosition(
        id="native-one",
        symbol=base + quote,
        market_type=product,
        side=side,
        qty=0.002,
        entry_price=50000,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-native-one",
    )
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(position, scope=scope)
    order = {
        "symbol": position.symbol,
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
        "symbol": position.symbol,
        "orderId": 123,
        "id": 0,
        "price": "50000",
        "qty": "0.002",
        "quoteQty": "100",
        "commission": fee,
        "commissionAsset": fee_asset,
        "time": 15,
    }
    instrument = {"symbol": position.symbol, "baseAsset": base, "quoteAsset": quote}
    if product == "spot":
        trade.update(isBuyer=True, isMaker=False)
    else:
        order.update(positionSide="BOTH", reduceOnly=False)
        trade.update(
            side=order["side"],
            positionSide="BOTH",
            buyer=side == "LONG",
            maker=False,
            realizedPnl="0",
        )
        instrument.update(
            contractType="PERPETUAL", marginAsset=quote, underlyingType="COIN"
        )
    client = Mock()
    client.execution_scope.return_value = scope
    client.get_order.return_value = order
    client.get_order_trades.return_value = [trade]
    if collect:
        collect_opening_recovery(client, journal)
    kwargs = {"scope": scope, "instrument": instrument, "instrument_scope": scope}
    return journal, kwargs, client


@pytest.mark.parametrize(
    "asset,fee,expected",
    [
        ("BTC", "0.000002", {"BTC": "0.001998", "USDT": "-100"}),
        ("USDT", "0.1", {"BTC": "0.002", "USDT": "-100.1"}),
        ("BNB", "0.00001", {"BTC": "0.002", "USDT": "-100", "BNB": "-0.00001"}),
        ("BNB", "-0.00001", {"BTC": "0.002", "USDT": "-100", "BNB": "0.00001"}),
        ("BTC", "0.003", {"BTC": "-0.001", "USDT": "-100"}),
    ],
)
def test_spot_native_fees_change_assets_without_quote_conversion(
    tmp_path, asset, fee, expected
):
    journal, kwargs, client = _case(tmp_path, fee_asset=asset, fee=fee)
    result = retain_opening_inventory(journal, **kwargs)
    assert dict(result.asset_deltas) == expected
    assert result.gross_executed_quantity == "0.002"
    assert result.derivative_position_delta == "0"
    assert (
        not result.inventory_applied
        and not result.rearmed
        and not result.account_balance_qualified
    )
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"
    assert len(client.mock_calls) == 3


@pytest.mark.parametrize("side,quantity", [("LONG", "0.002"), ("SHORT", "-0.002")])
@pytest.mark.parametrize("asset", ["BTC", "USDT", "BNB"])
def test_linear_futures_notional_is_not_a_spot_cash_purchase(
    tmp_path, side, quantity, asset
):
    journal, kwargs, _ = _case(tmp_path, "futures", side, asset, "0.1")
    result = retain_opening_inventory(journal, **kwargs)
    assert result.derivative_position_delta == quantity
    assert result.asset_deltas == ((asset, "-0.1"),)
    assert result.gross_quote_quantity == "100"


@pytest.mark.parametrize(
    "change",
    [
        {"baseAsset": "ETH"},
        {"quoteAsset": "BTC"},
        {"symbol": "ETHUSDT"},
        {"baseAsset": None},
        {"quoteAsset": []},
    ],
)
def test_wrong_instrument_identity_cannot_be_attached(tmp_path, change):
    journal, kwargs, _ = _case(tmp_path)
    kwargs["instrument"].update(change)
    with pytest.raises(OpenIntentError):
        retain_opening_inventory(journal, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contractType", "TRADIFI_PERPETUAL"),
        ("marginAsset", "BTC"),
        ("underlyingType", "HK_EQUITY"),
    ],
)
def test_quanto_inverse_and_unproved_units_reject(tmp_path, field, value):
    journal, kwargs, _ = _case(tmp_path, "futures")
    kwargs["instrument"][field] = value
    with pytest.raises(OpenIntentError, match="linear crypto"):
        retain_opening_inventory(journal, **kwargs)


def test_different_metadata_account_scope_is_rejected(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    kwargs["instrument_scope"] = replace(
        kwargs["scope"], credential_fingerprint="0" * 64
    )
    with pytest.raises(OpenIntentError, match="scope"):
        retain_opening_inventory(journal, **kwargs)


def test_source_fields_and_exact_movements_survive_restart_without_duplicates(tmp_path):
    journal, kwargs, client = _case(tmp_path)
    kwargs["instrument"]["irrelevant_field"] = "must-not-be-stored"
    first = retain_opening_inventory(journal, **kwargs)
    assert (
        retain_opening_inventory(BinanceOpenIntentJournal(journal.path), **kwargs)
        == first
    )
    with sqlite3.connect(journal.path) as connection:
        rows = connection.execute(
            "SELECT assets_json, movement_json FROM opening_inventory"
        ).fetchall()
    assert len(rows) == 1 and "must-not-be-stored" not in rows[0][0]
    assert json.loads(rows[0][1]) == json.loads(json.dumps(asdict(first)))
    assert client.get_order.call_count == client.get_order_trades.call_count == 1


def test_concurrent_projection_writers_commit_one_identical_observation(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    barrier = Barrier(4)

    def worker():
        barrier.wait(timeout=5)
        return retain_opening_inventory(journal, **kwargs)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: worker(), range(4)))
    assert results == [results[0]] * 4
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM opening_inventory"
        ).fetchone() == (1,)


@pytest.mark.parametrize("target", ["terminal", "movement", "schema"])
def test_inconsistent_retained_evidence_cannot_be_repaired_or_replaced(
    tmp_path, target
):
    journal, kwargs, _ = _case(tmp_path)
    retain_opening_inventory(journal, **kwargs)
    with sqlite3.connect(journal.path) as connection:
        if target == "terminal":
            connection.execute("UPDATE opening_recovery SET evidence_json='{}'")
        elif target == "movement":
            connection.execute("UPDATE opening_inventory SET movement_json='{}'")
        else:
            connection.execute("ALTER TABLE opening_inventory ADD COLUMN unknown TEXT")
    original = journal.path.read_bytes()
    with pytest.raises(OpenIntentError):
        retain_opening_inventory(journal, **kwargs)
    assert journal.path.read_bytes() == original
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_missing_terminal_evidence_does_not_create_inventory_table(tmp_path):
    journal, kwargs, _ = _case(tmp_path, collect=False)
    with pytest.raises(OpenIntentError, match="terminal fills"):
        retain_opening_inventory(journal, **kwargs)
    with sqlite3.connect(journal.path) as connection:
        assert (
            connection.execute("PRAGMA table_info(opening_inventory)").fetchall() == []
        )


def test_abrupt_child_exit_retains_exact_native_movements_and_unknown(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    script = """
import json,os,sys
from pathlib import Path
from simple_ai_trading.binance_open_intents import BinanceOpenIntentJournal
from simple_ai_trading.binance_execution_scope import BinanceExecutionScope
from simple_ai_trading.binance_opening_inventory import retain_opening_inventory
scope=BinanceExecutionScope(**json.loads(sys.argv[2]))
retain_opening_inventory(BinanceOpenIntentJournal(Path(sys.argv[1])), scope=scope, instrument_scope=scope, instrument={'symbol':'BTCUSDT','baseAsset':'BTC','quoteAsset':'USDT'})
os._exit(71)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(journal.path),
            json.dumps(asdict(kwargs["scope"])),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 71
    result = retain_opening_inventory(BinanceOpenIntentJournal(journal.path), **kwargs)
    assert dict(result.asset_deltas)["BTC"] == "0.001998"
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_exact_zero_fill_retains_no_fictitious_cash_or_inventory(tmp_path):
    journal, kwargs, client = _case(tmp_path, collect=False)
    client.get_order.return_value.update(
        status="REJECTED", executedQty="0", cummulativeQuoteQty="0"
    )
    collect_opening_recovery(client, journal)
    result = retain_opening_inventory(journal, **kwargs)
    assert result.asset_deltas == result.native_commissions == ()
    assert result.derivative_position_delta == result.gross_executed_quantity == "0"
    client.get_order_trades.assert_not_called()
    assert not result.rearmed


def test_partial_terminal_fill_has_exact_net_received_base_under_low_precision(
    tmp_path,
):
    journal, kwargs, client = _case(tmp_path, collect=False)
    client.get_order.return_value.update(
        status="CANCELED", executedQty="0.001", cummulativeQuoteQty="50"
    )
    client.get_order_trades.return_value[0].update(qty="0.001", quoteQty="50")
    collect_opening_recovery(client, journal)
    with localcontext() as context:
        context.prec = 3
        result = retain_opening_inventory(journal, **kwargs)
    assert dict(result.asset_deltas) == {"BTC": "0.000998", "USDT": "-50"}
    assert result.gross_executed_quantity == "0.001"


def test_absent_journal_is_not_created_by_projection(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    absent = BinanceOpenIntentJournal(tmp_path / "never-created.sqlite")
    assert retain_opening_inventory(absent, **kwargs) is None
    assert not absent.path.exists()


def test_pending_transition_is_rechecked_inside_writer_transaction(
    tmp_path, monkeypatch
):
    journal, kwargs, _ = _case(tmp_path)
    original = BinanceOpenIntentJournal.pending_position

    def changed(self, **options):
        position = original(self, **options)
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE open_intent SET state='RECORDED'")
        return position

    monkeypatch.setattr(BinanceOpenIntentJournal, "pending_position", changed)
    with pytest.raises(OpenIntentError, match="changed"):
        retain_opening_inventory(journal, **kwargs)
    with sqlite3.connect(journal.path) as connection:
        assert (
            connection.execute("PRAGMA table_info(opening_inventory)").fetchall() == []
        )


def test_sql_write_failure_leaves_unknown_and_no_new_inventory(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "CREATE TABLE opening_inventory (client_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, assets_json TEXT NOT NULL, movement_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TRIGGER reject_inventory BEFORE INSERT ON opening_inventory BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    with pytest.raises(OpenIntentError, match="could not be retained"):
        retain_opening_inventory(journal, **kwargs)
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM opening_inventory"
        ).fetchone() == (0,)
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_malformed_retained_order_never_becomes_zero_inventory(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE opening_recovery SET order_json='{broken'")
    with pytest.raises(OpenIntentError, match="could not be retained"):
        retain_opening_inventory(journal, **kwargs)


def test_nonmapping_instrument_is_not_inferred_from_symbol(tmp_path):
    journal, kwargs, _ = _case(tmp_path)
    kwargs["instrument"] = None
    with pytest.raises(OpenIntentError, match="explicit instrument"):
        retain_opening_inventory(journal, **kwargs)


@pytest.mark.parametrize("base", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("quote", ["USDT", "USDC"])
@pytest.mark.parametrize("product", ["spot", "futures"])
def test_explicit_major_asset_units_are_not_btc_or_usdt_only(
    tmp_path, base, quote, product
):
    journal, kwargs, _ = _case(
        tmp_path, product, fee_asset=base, base=base, quote=quote
    )
    result = retain_opening_inventory(journal, **kwargs)
    assert result.base_asset == base and result.quote_asset == quote
    if product == "spot":
        assert dict(result.asset_deltas) == {base: "0.001998", quote: "-100"}
    else:
        assert result.asset_deltas == ((base, "-0.000002"),)
        assert result.derivative_position_delta == "0.002"

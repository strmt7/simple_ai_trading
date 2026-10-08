"""Native closing units stay observational, parent-bound and restart-safe."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from decimal import localcontext
import json
import sqlite3
from threading import Barrier

import pytest

from simple_ai_trading.binance_close_recovery import collect_closing_recovery
from simple_ai_trading.binance_closing_inventory import retain_closing_inventory
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.positions import PositionsStore
from test_binance_close_recovery import _Client, _scope, _stage


def case(
    tmp_path, product="spot", side="LONG", *, asset="USDT", fee="0.11", collect=True
):
    store, position, client_id, order, trades = _stage(tmp_path, product, side)
    trades[0].update(commissionAsset=asset, commission=fee)
    client = _Client(order, trades, product=product)
    if collect:
        collect_closing_recovery(client, store, client_id=client_id)
    scope = _scope(product)
    instrument = {
        "symbol": "BTCUSDT",
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "ignored": "do-not-retain",
    }
    if product == "futures":
        instrument.update(
            contractType="PERPETUAL", marginAsset="USDT", underlyingType="COIN"
        )
    kwargs = {
        "client_id": client_id,
        "scope": scope,
        "instrument": instrument,
        "instrument_scope": scope,
    }
    return store, kwargs, client, position


@pytest.mark.parametrize(
    "asset,fee,expected",
    [
        ("USDT", "0.11", {"BTC": "-0.002", "USDT": "109.89"}),
        ("BTC", "0.000002", {"BTC": "-0.002002", "USDT": "110"}),
        ("BNB", "0.00001", {"BTC": "-0.002", "USDT": "110", "BNB": "-0.00001"}),
        ("BNB", "-0.00001", {"BTC": "-0.002", "USDT": "110", "BNB": "0.00001"}),
    ],
)
def test_spot_closing_principal_and_fee_assets(tmp_path, asset, fee, expected):
    store, kwargs, client, _ = case(tmp_path, asset=asset, fee=fee)
    result = retain_closing_inventory(store, **kwargs)
    assert dict(result.asset_deltas) == expected
    assert result.derivative_position_delta == "0"
    assert result.reported_realized_pnl is result.reported_realized_pnl_asset is None
    assert (
        not result.account_balance_qualified
        and not result.inventory_applied
        and not result.rearmed
    )
    assert len(client.calls) == 2
    assert len(store.load_open()) == 1 and store.load_ledger() == []


@pytest.mark.parametrize(
    "side,asset,fee,expected,delta",
    [
        ("LONG", "USDT", "0.11", {"USDT": "-0.11"}, "-0.002"),
        ("SHORT", "USDT", "0.11", {"USDT": "-0.11"}, "0.002"),
        ("LONG", "BNB", "0.00001", {"BNB": "-0.00001"}, "-0.002"),
        ("LONG", "USDT", "-0.11", {"USDT": "0.11"}, "-0.002"),
    ],
)
def test_futures_notional_is_not_principal_cash(
    tmp_path, side, asset, fee, expected, delta
):
    store, kwargs, _, _ = case(tmp_path, "futures", side, asset=asset, fee=fee)
    result = retain_closing_inventory(store, **kwargs)
    assert result.gross_quote_quantity == "110"
    assert dict(result.asset_deltas) == expected
    assert result.derivative_position_delta == delta
    assert result.reported_realized_pnl_asset == "USDT"
    assert result.reported_realized_pnl == ("10" if side == "LONG" else "-10")
    assert not result.financially_qualified
    assert (
        not result.local_lot_pnl_qualified
        and not result.inventory_applied
        and not result.rearmed
    )


def test_restart_idempotence_and_no_arbitrary_metadata_retention(tmp_path):
    store, kwargs, client, _ = case(tmp_path, "futures")
    first = retain_closing_inventory(store, **kwargs)
    assert retain_closing_inventory(PositionsStore(tmp_path), **kwargs) == first
    with sqlite3.connect(store.opening_intents.path) as connection:
        rows = connection.execute(
            "SELECT assets_json, movement_json FROM closing_inventory"
        ).fetchall()
        assert connection.execute("SELECT state FROM close_intent").fetchone() == (
            "UNKNOWN",
        )
    assert len(rows) == 1 and "do-not-retain" not in rows[0][0]
    assert json.loads(rows[0][1]) == json.loads(json.dumps(asdict(first)))
    assert len(client.calls) == 2


def test_concurrent_writers_retain_one_record(tmp_path):
    store, kwargs, _, _ = case(tmp_path)
    barrier = Barrier(4)

    def worker(_):
        barrier.wait(timeout=5)
        return retain_closing_inventory(store, **kwargs)

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(worker, range(4)))
    assert values == [values[0]] * 4
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM closing_inventory"
        ).fetchone() == (1,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("marginAsset", "USDC"),
        ("contractType", "CURRENT_QUARTER"),
        ("underlyingType", "STOCK"),
        ("baseAsset", "ETH"),
    ],
)
def test_instrument_unit_mismatch_rejects(tmp_path, field, value):
    store, kwargs, _, _ = case(tmp_path, "futures")
    kwargs["instrument"][field] = value
    with pytest.raises(OpenIntentError):
        retain_closing_inventory(store, **kwargs)


def test_scope_and_missing_evidence_reject_without_applying_inventory(tmp_path):
    store, kwargs, _, _ = case(tmp_path, collect=False)
    with pytest.raises(OpenIntentError):
        retain_closing_inventory(store, **kwargs)
    kwargs["instrument_scope"] = _scope("spot", key="other")
    with pytest.raises(OpenIntentError):
        retain_closing_inventory(store, **kwargs)
    assert len(store.load_open()) == 1 and store.load_ledger() == []


@pytest.mark.parametrize("target", ["terminal", "movement", "schema"])
def test_retained_data_tamper_rejects(tmp_path, target):
    store, kwargs, _, _ = case(tmp_path)
    retain_closing_inventory(store, **kwargs)
    with sqlite3.connect(store.opening_intents.path) as connection:
        if target == "terminal":
            connection.execute("UPDATE closing_recovery SET evidence_json='{}'")
        elif target == "movement":
            connection.execute("UPDATE closing_inventory SET movement_json='{}'")
        else:
            connection.execute("ALTER TABLE closing_inventory ADD COLUMN unknown TEXT")
    with pytest.raises(OpenIntentError):
        retain_closing_inventory(store, **kwargs)


def test_terminal_partial_close_never_erases_residual_position(tmp_path):
    store, kwargs, client, position = case(tmp_path, "futures", collect=False)
    client.order.update(status="CANCELED", executedQty="0.001", cumQuote="55")
    client.trades[0].update(
        qty="0.001", quoteQty="55", commission="0.055", realizedPnl="5"
    )
    collect_closing_recovery(client, store, client_id=kwargs["client_id"])
    result = retain_closing_inventory(store, **kwargs)
    assert result.gross_executed_quantity == "0.001"
    assert result.derivative_position_delta == "-0.001"
    assert dict(result.asset_deltas) == {"USDT": "-0.055"}
    assert result.reported_realized_pnl == "5"
    assert store.load_open() == [position] and store.load_ledger() == []
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute("SELECT state FROM close_intent").fetchone() == (
            "UNKNOWN",
        )


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_zero_fill_terminal_has_no_principal_fee_or_contract_delta(tmp_path, product):
    store, kwargs, client, position = case(tmp_path, product, collect=False)
    quote_key = "cummulativeQuoteQty" if product == "spot" else "cumQuote"
    client.order.update(status="EXPIRED", executedQty="0", **{quote_key: "0"})
    client.trades = []
    collect_closing_recovery(client, store, client_id=kwargs["client_id"])
    result = retain_closing_inventory(store, **kwargs)
    assert result.gross_executed_quantity == result.derivative_position_delta == "0"
    assert result.asset_deltas == result.native_commissions == ()
    assert store.load_open() == [position]


def test_decimal_context_does_not_round_native_units(tmp_path):
    store, kwargs, _, _ = case(
        tmp_path, "futures", fee="0.12345678901234567890123456789"
    )
    with localcontext() as context:
        context.prec = 2
        result = retain_closing_inventory(store, **kwargs)
    assert dict(result.asset_deltas) == {"USDT": "-0.12345678901234567890123456789"}


def test_no_journal_or_no_pending_close_returns_none_without_creation(tmp_path):
    store = PositionsStore(tmp_path)
    scope = _scope()
    assert (
        retain_closing_inventory(
            store,
            client_id="sait-c-none",
            scope=scope,
            instrument={},
            instrument_scope=scope,
        )
        is None
    )
    assert not store.opening_intents.path.exists()


def test_foreign_close_is_not_observed(tmp_path):
    store, kwargs, _, _ = case(tmp_path)
    kwargs["client_id"] = "sait-c-other"
    assert retain_closing_inventory(store, **kwargs) is None

"""Native entry inventory, original intent binding and restart compatibility."""

import json
import sqlite3
from dataclasses import asdict, replace
from decimal import Decimal, localcontext

import pytest

from simple_ai_trading.autonomous import (
    _apply_open_order,
    _close_to_trade,
    _submit_durable_close_position,
    _submit_durable_open_position,
)
from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_close_intents import _request as close_request
from simple_ai_trading.binance_execution_scope import (
    BinanceExecutionScope,
    BINANCE_SPOT_TESTNET,
)
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_spot_receipts import (
    exact_spot_wire_quantity,
    native_spot_entry_net,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore


def position(**changes):
    return replace(
        OpenPosition(
            id="native",
            symbol="BTCUSDC",
            market_type="spot",
            side="LONG",
            qty=1,
            entry_price=100,
            leverage=1,
            opened_at_ms=1,
            notional=100,
            dry_run=False,
            open_client_order_id="sait-o-native",
            entry_fees=0.1,
        ),
        **changes,
    )


def receipt(p, *, fee="0.001", fee_asset="BTC"):
    return {
        "symbol": p.symbol,
        "side": "BUY",
        "type": "MARKET",
        "status": "FILLED",
        "orderId": "123",
        "clientOrderId": p.open_client_order_id,
        "executedQty": str(p.qty),
        "origQty": str(p.qty),
        "fills": [
            {
                "qty": str(p.qty),
                "price": "100",
                "commission": fee,
                "commissionAsset": fee_asset,
            }
        ],
    }


class Client:
    def __init__(self, opening):
        self.opening = opening
        self.orders = []

    def execution_scope(self):
        return BinanceExecutionScope.from_api_key(
            BINANCE_SPOT_TESTNET, "spot", "offline-placeholder"
        )

    def place_order(self, symbol, side, quantity, **kwargs):
        self.orders.append((symbol, side, quantity))
        if side == "BUY":
            return self.opening
        return {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "status": "FILLED",
            "executedQty": str(quantity),
            "origQty": str(quantity),
            "avgPrice": "100",
            "orderId": "124",
            "clientOrderId": kwargs["client_order_id"],
        }


@pytest.mark.parametrize("base", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("quote", ["USDT", "USDC"])
@pytest.mark.parametrize("fee_kind", ["base", "quote", "BNB", "zero"])
def test_native_open_close_persists_only_received_inventory(
    tmp_path, base, quote, fee_kind
):
    p = position(symbol=base + quote)
    fee_asset = (
        base
        if fee_kind in {"base", "zero"}
        else quote
        if fee_kind == "quote"
        else "BNB"
    )
    client = Client(
        receipt(p, fee="0" if fee_kind == "zero" else "0.001", fee_asset=fee_asset)
    )
    store = PositionsStore(tmp_path)
    opened = _submit_durable_open_position(client, p, store)
    expected = 0.999 if fee_kind == "base" else 1.0
    assert opened.qty == expected
    assert opened.spot_gross_entry_quantity == "1"
    assert opened.spot_entry_base_commission == ("0.001" if fee_kind == "base" else "0")
    assert opened.entry_fees == 0.1  # Still modelled, not native cash PnL.
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open() == [opened]
    assert reopened.opening_intents.entry_block_reason() is None
    with sqlite3.connect(reopened.opening_intents.path) as connection:
        original = json.loads(
            connection.execute("SELECT request_json FROM open_intent").fetchone()[0]
        )
    assert original["quantity"] == 1
    trade = _close_to_trade(opened, 100, "native-check", clock=lambda: 2)
    closed = _submit_durable_close_position(
        client, opened, trade, reopened, reduce_only=False
    )
    assert client.orders == [(p.symbol, "BUY", 1), (p.symbol, "SELL", expected)]
    assert PositionsStore(tmp_path).load_open() == []
    assert PositionsStore(tmp_path).load_ledger() == [closed]
    assert closed.spot_entry_base_commission == opened.spot_entry_base_commission
    assert reopened.opening_intents.entry_block_reason() is None


@pytest.mark.parametrize(
    "failure", ["missing_fee", "partial", "sub_wire", "missing_status"]
)
def test_unproved_native_open_preserves_unknown_after_restart(tmp_path, failure):
    p = position()
    order = receipt(p)
    if failure == "missing_fee":
        del order["fills"][0]["commission"]
    elif failure == "partial":
        order["status"] = "PARTIALLY_FILLED"
    elif failure == "sub_wire":
        order["fills"][0]["commission"] = "0.000000001"
    else:
        del order["status"]
    client = Client(order)
    with pytest.raises((OpenIntentError, BinanceAPIError)):
        _submit_durable_open_position(client, p, PositionsStore(tmp_path))
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open() == []
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    )
    assert (
        reopened.opening_intents.pending_position(scope=client.execution_scope()) == p
    )
    assert len(client.orders) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {
            "spot_gross_entry_quantity": "1.0000000000001",
            "spot_entry_base_commission": "0.0010000000001",
        },
        {"spot_gross_entry_quantity": "0.999", "spot_entry_base_commission": "0"},
        {"qty": 0.998},
        {"exchange_status": "PARTIALLY_FILLED"},
    ],
)
def test_net_inventory_cannot_disguise_wrong_gross_intent(tmp_path, changes):
    p = position()
    recorded = _apply_open_order(p, receipt(p))
    with pytest.raises(OpenIntentError):
        PositionsStore(tmp_path).opening_intents.validate_result(
            p, replace(recorded, **changes)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"spot_entry_base_commission": ""},
        {"spot_entry_base_commission": "-0.001"},
        {"spot_entry_base_commission": "NaN"},
        {"spot_gross_entry_quantity": 1},
        {"spot_entry_base_commission": "1"},
        {"qty": 1.1},
        {"qty": 0},
        {"market_type": "futures"},
        {"side": "SHORT"},
        {"dry_run": True},
        {"symbol": "DOGEUSDC"},
        {"qty": 0.999000001},
    ],
)
def test_invalid_native_metadata_cannot_enter_or_replace_ledger(tmp_path, changes):
    opened = _apply_open_order(position(), receipt(position()))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    before = store.open_path.read_bytes()
    invalid = replace(opened, **changes)
    with pytest.raises(OpenIntentError):
        store.record_open(invalid)
    assert store.open_path.read_bytes() == before
    assert not store._valid_open_entry(asdict(invalid))
    assert store._open_entry_integrity_errors(0, asdict(invalid))


def test_partial_native_close_retains_one_satoshi_exactly(tmp_path):
    p = position(qty=2, notional=200)
    opened = _apply_open_order(p, receipt(p))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    trade = replace(
        _close_to_trade(opened, 100, "partial", clock=lambda: 2), qty=1.99899999
    )
    with localcontext() as context:
        context.prec = 3
        store.record_close_result(opened, trade)
    remaining = PositionsStore(tmp_path).load_open()[0]
    assert remaining.qty == 0.00000001
    assert native_spot_entry_net(remaining) == Decimal("1.999")
    assert store.load_ledger() == [trade]


@pytest.mark.parametrize(
    "changes",
    [
        {"spot_entry_base_commission": ""},
        {"qty": 1},
        {"spot_gross_entry_quantity": "2"},
    ],
)
def test_native_close_rejects_overfill_or_changed_entry_evidence(tmp_path, changes):
    opened = _apply_open_order(position(), receipt(position()))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    trade = replace(_close_to_trade(opened, 100, "invalid", clock=lambda: 2), **changes)
    with pytest.raises(ValueError):
        store.record_close_result(opened, trade)
    assert store.load_open() == [opened]
    assert store.load_ledger() == []


def test_empty_receipt_fields_preserve_legacy_pending_request_bytes(tmp_path):
    p = position()
    store = PositionsStore(tmp_path)
    scope = Client({}).execution_scope()
    payload = json.loads(store.opening_intents._request(p, scope))
    template = asdict(p)
    template.pop("spot_gross_entry_quantity")
    template.pop("spot_entry_base_commission")
    assert payload["position_template"] == template
    original = json.dumps(payload, sort_keys=True, allow_nan=False)
    store.opening_intents.prepare(p, scope=scope)
    reopened = PositionsStore(tmp_path)
    assert reopened.opening_intents.pending_position(scope=scope) == p
    assert reopened.opening_intents._request(p, scope) == original
    p = replace(p, open_exchange_order_id="123", exchange_status="FILLED")
    closing = json.loads(close_request(store, p, "sait-c-native", scope, False))
    assert set(closing["opening"]["position_template"]) == set(template)


@pytest.mark.parametrize(
    "value", [True, "NaN", "0", "-1", "0.000000001", "9007199254740993"]
)
def test_wire_quantity_rejects_unrepresentable_or_invalid_value(value):
    with pytest.raises(OpenIntentError):
        exact_spot_wire_quantity(value)


@pytest.mark.parametrize("changes", [{"dry_run": True}, {"market_type": "futures"}])
def test_paper_and_futures_do_not_claim_native_spot_receipts(changes):
    p = position(**changes)
    opened = _apply_open_order(p, {"executedQty": "1", "avgPrice": "100"})
    assert opened.qty == 1
    assert native_spot_entry_net(opened) is None


def test_native_close_unrepresentable_fill_keeps_lot_and_unknown(tmp_path):
    p = position()
    store = PositionsStore(tmp_path)
    client = Client(receipt(p))
    opened = _submit_durable_open_position(client, p, store)

    class PartialClient(Client):
        def place_order(self, *args, **kwargs):
            order = super().place_order(*args, **kwargs)
            order.update(executedQty="0.998999999", status="PARTIALLY_FILLED")
            return order

    closer = PartialClient({})
    with pytest.raises(OpenIntentError, match="wire"):
        _submit_durable_close_position(
            closer,
            opened,
            _close_to_trade(opened, 100, "partial", clock=lambda: 2),
            store,
            reduce_only=False,
        )
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open() == [opened]
    assert reopened.load_ledger() == []
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    )
    assert len(closer.orders) == 1


def test_native_open_interruption_preserves_net_position_and_pending_gross(
    tmp_path, monkeypatch
):
    from simple_ai_trading.binance_open_intents import BinanceOpenIntentJournal

    p = position()
    store = PositionsStore(tmp_path)
    client = Client(receipt(p))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(BinanceOpenIntentJournal, "record_complete", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _submit_durable_open_position(client, p, store)
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open()[0].qty == 0.999
    assert (
        reopened.opening_intents.pending_position(scope=client.execution_scope()).qty
        == 1
    )
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    )

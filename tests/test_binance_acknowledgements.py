"""Contradictory first acknowledgements cannot manufacture inventory or flatness."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from simple_ai_trading import cli
from simple_ai_trading.autonomous import (
    _close_to_trade,
    _submit_durable_close_position,
    _submit_durable_open_position,
)
from simple_ai_trading.binance_acknowledgements import acknowledged_fill
from simple_ai_trading.binance_execution_scope import (
    BinanceExecutionScope,
    BINANCE_SPOT_TESTNET,
)
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.config import RuntimeConfig
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _scope():
    return BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET, "spot", "offline-placeholder"
    )


def _position():
    return OpenPosition(
        id="ack-owned",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-ack-owned",
    )


class _Client:
    def __init__(self, changes):
        self.changes, self.calls = changes, []

    def execution_scope(self):
        return _scope()

    def place_order(self, symbol, side, quantity, **kwargs):
        self.calls.append("write")
        return {
            "symbol": symbol,
            "side": side,
            "status": "FILLED",
            "orderId": 200,
            "clientOrderId": kwargs["client_order_id"],
            "executedQty": str(quantity),
            "avgPrice": "101",
            **self.changes,
        }

    def get_order(self, *args, **kwargs):
        pytest.fail("contradictory evidence must not trigger an adaptive query")


@pytest.mark.parametrize("phase", ["opening", "closing"])
@pytest.mark.parametrize(
    "changes",
    [
        {"executedQty": "0.5", "fills": [{"qty": "1", "price": "101"}]},
        {"fills": [{"qty": "0.5", "price": "101"}]},
        {"cummulativeQuoteQty": "100", "fills": [{"qty": "1", "price": "101"}]},
        {"orderId": True},
        {"orderId": "not-an-id"},
        {"origQty": "2"},
        {"avgPrice": "NaN"},
        {"fills": []},
    ],
)
def test_active_open_and_close_preserve_unknown_on_contradiction(
    tmp_path, phase, changes
):
    store, position = PositionsStore(tmp_path), _position()
    if phase == "closing":
        store.opening_intents.prepare(position, scope=_scope())
        recorded = replace(
            position, exchange_status="FILLED", open_exchange_order_id="100"
        )
        store.record_open(recorded)
        store.opening_intents.record_complete(position, recorded, scope=_scope())
        position = recorded
    before = store.load_snapshot()
    client = _Client(changes)
    with pytest.raises(OpenIntentError):
        if phase == "opening":
            _submit_durable_open_position(client, position, store)
        else:
            _submit_durable_close_position(
                client,
                position,
                _close_to_trade(position, 101, "test", clock=lambda: 2),
                store,
                reduce_only=True,
            )
    assert store.load_snapshot() == before
    assert store.opening_intents.entry_block_reason() == f"unresolved_{phase}_intents=1"
    assert client.calls == ["write"]


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_cash_and_fill_prices_override_non_authoritative_average(product):
    quote_key = "cummulativeQuoteQty" if product == "spot" else "cumQuote"
    raw = {
        "executedQty": "1",
        quote_key: "101",
        "avgPrice": "999",
        "price": "888",
        "fills": [{"qty": "0.5", "price": "100"}, {"qty": "0.5", "price": "102"}],
    }
    result = acknowledged_fill(raw, market_type=product)
    assert (
        result.quantity,
        result.price,
        result.quote_quantity,
        result.price_basis,
    ) == ("1", "101", "101", "cumulative_quote")
    del raw[quote_key]
    assert acknowledged_fill(raw, market_type=product).price_basis == "fills"
    del raw["fills"]
    result = acknowledged_fill(raw, market_type=product)
    assert (
        result.price == "999"
        and result.quote_quantity is None
        and result.price_basis == "average_only"
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"fills": [{"qty": "1", "price": "100"}]},
        {"executedQty": True},
        {"executedQty": "NaN"},
        {"executedQty": "0", "cummulativeQuoteQty": "1"},
        {"executedQty": "1", "cummulativeQuoteQty": "0"},
        {"executedQty": "1", "cumBase": "100"},
        {"executedQty": "1", "cumQuote": "100"},
        {"executedQty": "1", "fills": {}},
        {"executedQty": "1", "fills": [None]},
        {"executedQty": "1", "fills": [{"qty": "0", "price": "100"}]},
        {
            "executedQty": "1",
            "fills": [{"qty": "1", "price": "100", "quoteQty": "101"}],
        },
        {"executedQty": "1", "fills": [{"qty": "1", "price": "100", "tradeId": True}]},
        {
            "executedQty": "1",
            "fills": [{"qty": "0.5", "price": "100", "tradeId": 1}] * 2,
        },
        {
            "executedQty": "1",
            "fills": [
                {
                    "qty": "1",
                    "price": "100",
                    "commission": "NaN",
                    "commissionAsset": "BTC",
                }
            ],
        },
        {
            "executedQty": "1",
            "fills": [{"qty": "1", "price": "100", "commission": "0.01"}],
        },
        {
            "executedQty": "1",
            "fills": [{"qty": "1", "price": "100", "commissionAsset": "BTC"}],
        },
        {
            "executedQty": "1",
            "fills": [
                {
                    "qty": "1",
                    "price": "100",
                    "commission": "0.01",
                    "commissionAsset": "btc",
                }
            ],
        },
        {"executedQty": "1", "fills": [{"qty": "1", "price": "100"}] * 1001},
    ],
)
def test_invalid_quantity_cash_and_fill_payloads_reject(raw):
    with pytest.raises(OpenIntentError):
        acknowledged_fill(raw, market_type="spot")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"executedQty": "0"},
        {"executedQty": "0", "fills": []},
        {"executedQty": "1", "price": "100"},
    ],
)
def test_no_execution_or_order_limit_price_never_fabricates_fill(raw):
    assert acknowledged_fill(raw, market_type="spot") is None


def test_cash_products_and_sums_do_not_round_at_one_hundred_digits():
    value = "999999999999999999999999999999.999999999999999999999999999999"
    result = acknowledged_fill(
        {"executedQty": value, "fills": [{"qty": value, "price": value}]},
        market_type="spot",
    )
    with localcontext() as context:
        context.prec = 150
        assert Decimal(result.quote_quantity) == Decimal(value) ** 2


def test_valid_native_fee_fields_do_not_claim_native_inventory_application():
    result = acknowledged_fill(
        {
            "executedQty": "1",
            "fills": [
                {
                    "qty": "1",
                    "price": "100",
                    "tradeId": 0,
                    "commission": "-0.01",
                    "commissionAsset": "USDT",
                }
            ],
        },
        market_type="spot",
    )
    assert result.quote_quantity == "100"  # Gross cash, not cash after fees.


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize("query", [False, True])
def test_cli_active_response_and_query_use_same_consistency_gate(product, query):
    class Client:
        def __init__(self):
            self.calls = 0

        def get_order(self, *args, **kwargs):
            self.calls += 1
            return contradictory

    contradictory = {"executedQty": "0.5", "fills": [{"qty": "1", "price": "101"}]}
    client = Client()
    with pytest.raises(OpenIntentError, match="quantities conflict"):
        cli._resolved_order_fill_details(
            client,
            RuntimeConfig(market_type=product),
            {"orderId": 200} if query else contradictory,
            fallback_qty=1,
            fallback_price=100,
            dry_run=False,
        )
    assert client.calls == int(query)


def test_cli_dry_run_preserves_explicit_simulation_fallback():
    assert cli._resolved_order_fill_details(
        object(), RuntimeConfig(), {}, fallback_qty=2, fallback_price=100, dry_run=True
    ) == (2, 100, 200, "order_response")


@pytest.mark.parametrize("raw,product", [(None, "spot"), ({}, "unknown")])
def test_invalid_product_or_payload_is_not_a_fill(raw, product):
    with pytest.raises(OpenIntentError):
        acknowledged_fill(raw, market_type=product)


def test_maximum_fill_count_sums_without_floating_point_loss():
    result = acknowledged_fill(
        {
            "executedQty": "1",
            "cummulativeQuoteQty": "101",
            "fills": [
                {"qty": "0.001", "price": "101", "tradeId": index}
                for index in range(1000)
            ],
        },
        market_type="spot",
    )
    assert (result.quantity, result.quote_quantity, result.price) == ("1", "101", "101")

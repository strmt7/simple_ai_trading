"""Real client recovery methods, explicit local authority, synthetic transport."""

from copy import deepcopy
from decimal import Decimal

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_owned_execution import (
    close_trade_template,
    submit_durable_close,
    submit_durable_open,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore


@pytest.fixture
def lost_spot_response(tmp_path, monkeypatch, enrolled_binance_runtime):
    clients = []

    def setup(phase):
        client = BinanceClient("offline-key", "offline-secret", max_retries=0)
        clients.append(client)
        scope = client.execution_scope()
        store = PositionsStore(tmp_path / phase)
        enrolled_binance_runtime(store, scope)
        state = {"lose": phase == "opening", "calls": [], "count": 0}

        def forbidden(*args, **kwargs):
            raise AssertionError("real network forbidden")

        monkeypatch.setattr(client.session, "request", forbidden)

        def transport(method, path, params, **kwargs):
            assert kwargs["expected_scope"] == scope
            state["calls"].append((method, path))
            if method == "GET":
                assert params["origClientOrderId"] == state["order"]["clientOrderId"]
                if state.get("rotate_scope"):
                    client.api_key = "another-offline-key"
                if "status" in state:
                    state["order"]["status"] = state["status"]
                return {
                    key: value
                    for key, value in state["order"].items()
                    if key != "fills"
                }
            assert (method, path) == ("POST", "/api/v3/order")
            state["count"] += 1
            quantity = params["quantity"]
            partial = state.get("partial") and params["side"] == "SELL"
            executed = str(Decimal(quantity) / 2) if partial else quantity
            quote = str(Decimal(executed) * 100)
            state["order"] = {
                "symbol": params["symbol"],
                "side": params["side"],
                "type": "MARKET",
                "status": "PARTIALLY_FILLED" if partial else "FILLED",
                "origQty": quantity,
                "executedQty": executed,
                "cummulativeQuoteQty": quote,
                "orderId": state["count"],
                "clientOrderId": params["newClientOrderId"],
                "fills": [
                    {
                        "tradeId": state["count"],
                        "qty": executed,
                        "price": "100",
                        "quoteQty": quote,
                        "commission": "0.001" if params["side"] == "BUY" else "0.01",
                        "commissionAsset": "BTC" if params["side"] == "BUY" else "USDT",
                    }
                ],
            }
            if state["lose"]:
                raise BinanceAPIError("synthetic lost acceptance response")
            return deepcopy(state["order"])

        def trades(method, path, params, **kwargs):
            assert (method, path) == ("GET", "/api/v3/myTrades")
            assert kwargs["expected_scope"] == scope
            assert params == {
                "symbol": "BTCUSDT",
                "orderId": str(state["order"]["orderId"]),
                "limit": 1000,
            }
            state["calls"].append((method, path))
            if "trade_error" in state:
                raise state["trade_error"]
            rows = [
                {
                    "symbol": "BTCUSDT",
                    "orderId": state["order"]["orderId"],
                    "id": row["tradeId"],
                    "isBuyer": state["order"]["side"] == "BUY",
                    **{key: value for key, value in row.items() if key != "tradeId"},
                }
                for row in state["order"]["fills"]
            ]
            return state.get("damage", lambda rows: rows)(rows)

        monkeypatch.setattr(client, "_request_dict", transport)
        monkeypatch.setattr(client, "_request_list", trades)
        requested = OpenPosition(
            id="receipt",
            symbol="BTCUSDT",
            market_type="spot",
            side="LONG",
            qty=1,
            entry_price=100,
            leverage=1,
            opened_at_ms=1,
            notional=100,
            dry_run=False,
            open_client_order_id="sait-o-receipt",
        )
        if phase == "closing":
            requested = submit_durable_open(client, requested, store).position
            state["calls"].clear()
            state["lose"] = True

        def execute():
            if phase == "opening":
                return submit_durable_open(client, requested, store)
            return submit_durable_close(
                client,
                requested,
                close_trade_template(requested, 100, "recovery"),
                store,
                reduce_only=False,
            )

        return store, requested, state, execute

    yield setup
    for client in clients:
        client.session.close()


@pytest.mark.parametrize("phase", ["opening", "closing"])
def test_missing_inline_receipt_is_completed_without_resubmission(
    lost_spot_response, phase
):
    store, requested, state, execute = lost_spot_response(phase)
    result = execute()
    assert state["calls"] == [
        ("POST", "/api/v3/order"),
        ("GET", "/api/v3/order"),
        ("GET", "/api/v3/myTrades"),
    ]
    assert store.opening_intents.entry_block_reason() is None
    if phase == "opening":
        assert result.position.qty == pytest.approx(0.999)
        assert store.load_open() == [result.position]
    else:
        assert result.trade.qty == requested.qty
        assert result.trade.realized_pnl == pytest.approx(-0.11)
        assert store.load_open() == [] and store.load_ledger() == [result.trade]


def damage_rows(rows, kind):
    if kind == "empty":
        return []
    if kind == "unbounded":
        return rows * 1001
    if kind == "not_list":
        return {"rows": rows}
    if kind == "duplicate":
        return rows + deepcopy(rows)
    if kind == "not_mapping":
        return [None]
    field, value = {
        "order": ("orderId", 999),
        "symbol": ("symbol", "ETHUSDT"),
        "side": ("isBuyer", not rows[0]["isBuyer"]),
        "integer_side": ("isBuyer", 1),
        "id": ("id", True),
        "quantity": ("qty", "2"),
        "cash": ("quoteQty", "99"),
        "commission": ("commission", None),
        "negative_fee": ("commission", "-1"),
        "fee_asset": ("commissionAsset", "btc"),
    }[kind]
    rows[0][field] = value
    return rows


@pytest.mark.parametrize("phase", ["opening", "closing"])
@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "unbounded",
        "not_list",
        "duplicate",
        "not_mapping",
        "order",
        "symbol",
        "side",
        "integer_side",
        "id",
        "quantity",
        "cash",
        "commission",
        "negative_fee",
        "fee_asset",
    ],
)
def test_incomplete_or_foreign_trade_page_keeps_owned_obligation(
    lost_spot_response, phase, kind
):
    store, requested, state, execute = lost_spot_response(phase)
    state["damage"] = lambda rows: damage_rows(rows, kind)
    with pytest.raises(OpenIntentError):
        execute()
    assert sum(method == "POST" for method, _ in state["calls"]) == 1
    assert store.opening_intents.entry_block_reason() == f"unresolved_{phase}_intents=1"
    assert store.load_open() == ([] if phase == "opening" else [requested])
    assert store.load_ledger() == []


@pytest.mark.parametrize("phase", ["opening", "closing"])
@pytest.mark.parametrize(
    "failure", [BinanceAPIError("synthetic trade query loss"), KeyboardInterrupt()]
)
def test_trade_query_loss_or_cancel_never_replays_or_releases(
    lost_spot_response, phase, failure
):
    store, requested, state, execute = lost_spot_response(phase)
    state["trade_error"] = failure
    with pytest.raises(type(failure)):
        execute()
    assert sum(method == "POST" for method, _ in state["calls"]) == 1
    assert store.opening_intents.entry_block_reason() == f"unresolved_{phase}_intents=1"
    assert store.load_open() == ([] if phase == "opening" else [requested])
    assert store.load_ledger() == []


@pytest.mark.parametrize("phase", ["opening", "closing"])
def test_scope_change_after_status_rejects_before_trade_query(
    lost_spot_response, phase
):
    store, requested, state, execute = lost_spot_response(phase)
    state["rotate_scope"] = True
    with pytest.raises(OpenIntentError, match="scope changed"):
        execute()
    assert state["calls"] == [("POST", "/api/v3/order"), ("GET", "/api/v3/order")]
    assert store.opening_intents.entry_block_reason() == f"unresolved_{phase}_intents=1"
    assert store.load_open() == ([] if phase == "opening" else [requested])


@pytest.mark.parametrize("phase", ["opening", "closing"])
def test_unusable_order_state_does_not_spend_another_trade_query(
    lost_spot_response, phase
):
    store, requested, state, execute = lost_spot_response(phase)
    state["status"] = "CANCELED"
    with pytest.raises(BinanceAPIError, match="status"):
        execute()
    assert state["calls"] == [("POST", "/api/v3/order"), ("GET", "/api/v3/order")]
    assert store.opening_intents.entry_block_reason() == f"unresolved_{phase}_intents=1"
    assert store.load_open() == ([] if phase == "opening" else [requested])


def test_partial_close_recovers_only_the_observed_fill_and_retains_barrier(
    lost_spot_response,
):
    store, requested, state, execute = lost_spot_response("closing")
    state["partial"] = True
    result = execute()
    assert result.trade.qty == pytest.approx(requested.qty / 2)
    assert store.load_open()[0].qty == pytest.approx(requested.qty / 2)
    assert store.load_ledger() == [result.trade]
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    assert sum(method == "POST" for method, _ in state["calls"]) == 1


def test_trades_advancing_after_partial_status_remain_unresolved(lost_spot_response):
    store, requested, state, execute = lost_spot_response("closing")
    state["partial"] = True

    def advance(rows):
        later = deepcopy(rows[0])
        later["id"] += 10
        return rows + [later]

    state["damage"] = advance
    with pytest.raises(OpenIntentError, match="cumulative"):
        execute()
    assert store.load_open() == [requested] and store.load_ledger() == []
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    assert sum(method == "POST" for method, _ in state["calls"]) == 1

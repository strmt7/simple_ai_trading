from copy import deepcopy
from decimal import Decimal as D
import time

import pytest

from simple_ai_trading.spot_testnet_evidence import Rules, grid, owned_trade_cash
from simple_ai_trading.spot_testnet_coverage import lifecycle_coverage
from tools.run_spot_testnet_execution_campaign import Campaign
from tools.spot_testnet_campaign_transport import Journal, SYMBOLS, Transport


def symbol_info(symbol):
    return {
        "symbol": symbol,
        "baseAsset": symbol[:-4],
        "quoteAsset": "USDT",
        "status": "TRADING",
        "orderTypes": ["LIMIT", "LIMIT_MAKER"],
        "allowedSelfTradePreventionModes": ["EXPIRE_TAKER"],
        "filters": [
            {
                "filterType": "LOT_SIZE",
                "stepSize": "0.001",
                "minQty": "0.001",
                "maxQty": "1000",
            },
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "10000"},
        ],
    }


def test_decimal_grid_and_published_filters():
    assert grid(D("1.13"), D("0.05")) == D("1.10")
    assert grid(D("1.13"), D("0.05"), up=True) == D("1.15")
    rules = Rules.parse(symbol_info("BTCUSDT"))
    rules.validate(D("0.100"), D("100.01"))
    for qty, price in [(D("0.1001"), D(100)), (D("0.001"), D(100)), (D("NaN"), D(100))]:
        with pytest.raises((ValueError, ArithmeticError)):
            rules.validate(qty, price)


def example_fill():
    order = {
        "symbol": "BTCUSDT",
        "orderId": 1,
        "side": "BUY",
        "status": "EXPIRED",
        "origQty": "2",
        "executedQty": "1",
        "cummulativeQuoteQty": "100",
    }
    trades = [
        {
            "symbol": "BTCUSDT",
            "orderId": 1,
            "id": 3,
            "isBuyer": True,
            "qty": "1",
            "quoteQty": "100",
            "commission": "0.01",
            "commissionAsset": "BTC",
        }
    ]
    return order, trades


def test_partial_fill_deducts_received_base_fee():
    order, trades = example_fill()
    result = owned_trade_cash(order, trades, "BTC")
    assert result["base_delta"] == "0.99"
    assert result["quote_delta"] == "-100"
    assert result["partial_fill"] is True


@pytest.mark.parametrize(
    "case", ["foreign", "duplicate", "missing", "nonterminal", "wrong_side"]
)
def test_unreconciled_fills_reject(case):
    order, trades = example_fill()
    if case == "foreign":
        trades[0]["orderId"] = 999
    elif case == "duplicate":
        trades += deepcopy(trades)
    elif case == "missing":
        trades = []
    elif case == "nonterminal":
        order["status"] = "PARTIALLY_FILLED"
    else:
        trades[0]["isBuyer"] = False
    with pytest.raises(ValueError):
        owned_trade_cash(order, trades, "BTC")


def transport(tmp_path, monkeypatch):
    for name in ("BINANCE_BASE_URL", "BINANCE_SPOT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    result = Transport(
        "fake-api-key",
        "fake-secret",
        Journal(tmp_path / "journal.jsonl", create=True),
        "probe",
    )
    result.client.last_request_info = {"status": 200}
    return result


def test_foreign_order_and_duplicate_submission_block_before_network(
    tmp_path, monkeypatch
):
    t = transport(tmp_path, monkeypatch)
    intent = {"symbol": "BTCUSDT", "side": "BUY", "newClientOrderId": "probe-00"}
    t.journal.add("order_intent", params=intent)
    with pytest.raises(ValueError, match="foreign"):
        t._check(
            "DELETE",
            "/api/v3/order",
            {"symbol": "BTCUSDT", "origClientOrderId": "probe-00", "orderId": 99},
        )
    t.journal.add("http_intent", method="POST", path="/api/v3/order", params=intent)
    with pytest.raises(ValueError, match="resubmit"):
        t._check("POST", "/api/v3/order", intent)
    with pytest.raises(ValueError):
        t._check("POST", "/sapi/v1/asset/transfer", {})
    t.client.session.close()


def test_hash_chained_journal_detects_tamper(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path, create=True)
    journal.add("order_intent", quantity="1")
    assert len(Journal(path).rows) == 1
    path.write_text(path.read_text().replace('"quantity": "1"', '"quantity": "2"'))
    with pytest.raises(ValueError, match="integrity"):
        Journal(path)


class Venue:
    """Stateful synthetic venue; intentionally cannot access the network."""

    def __init__(self, *, lose_first_ack=False, change_cancel_id=True):
        self.orders = {}
        self.trades = {}
        self.posts = 0
        self.lose_first_ack = lose_first_ack
        self.change_cancel_id = change_cancel_id

    def request(self, method, path, params, signed=False):
        if path == "/api/v3/time":
            return {"serverTime": time.time_ns() // 1_000_000}
        if path == "/api/v3/exchangeInfo":
            return {
                "symbols": [symbol_info(s) for s in SYMBOLS],
                "rateLimits": [
                    {
                        "rateLimitType": "REQUEST_WEIGHT",
                        "interval": "MINUTE",
                        "intervalNum": 1,
                        "limit": 6000,
                    }
                ],
            }
        if path == "/api/v3/account":
            return {"canTrade": True, "balances": [{"asset": "USDT", "free": "10000"}]}
        if path == "/api/v3/account/commission":
            return {
                "symbol": params["symbol"],
                **{
                    category: dict.fromkeys(("maker", "taker", "buyer", "seller"), "0")
                    for category in (
                        "standardCommission",
                        "specialCommission",
                        "taxCommission",
                    )
                },
            }
        if path == "/api/v3/ticker/bookTicker":
            return {"symbol": params["symbol"], "bidPrice": "100", "askPrice": "100.1"}
        if path == "/api/v3/openOrders":
            return [
                deepcopy(x)
                for x in self.orders.values()
                if x["symbol"] == params["symbol"] and x["status"] == "NEW"
            ]
        if path == "/api/v3/myTrades":
            return deepcopy(self.trades[params["orderId"]])
        assert path == "/api/v3/order" and signed
        if method == "POST":
            self.posts += 1
            oid = self.posts
            maker = params["type"] == "LIMIT_MAKER"
            execution = D("100.1") if params["side"] == "BUY" else D(100)
            qty = D(0) if maker else D(params["quantity"])
            order = {
                "symbol": params["symbol"],
                "side": params["side"],
                "clientOrderId": params["newClientOrderId"],
                "orderId": oid,
                "status": "NEW" if maker else "FILLED",
                "origQty": params["quantity"],
                "executedQty": str(qty),
                "cummulativeQuoteQty": str(qty * execution),
                "price": params["price"],
            }
            self.orders[oid] = order
            self.trades[oid] = (
                []
                if maker
                else [
                    {
                        "symbol": params["symbol"],
                        "orderId": oid,
                        "id": oid,
                        "isBuyer": params["side"] == "BUY",
                        "qty": str(qty),
                        "quoteQty": str(qty * execution),
                        "commission": "0",
                        "commissionAsset": "USDT",
                        "price": str(execution),
                    }
                ]
            )
            if self.lose_first_ack and oid == 1:
                raise TimeoutError("fake-secret must never reach a journal")
        else:
            oid = params.get("orderId")
            if oid is None:
                oid = next(
                    k
                    for k, v in self.orders.items()
                    if v["clientOrderId"] == params["origClientOrderId"]
                )
            order = self.orders[oid]
            if method == "DELETE":
                order["status"] = "CANCELED"
                if self.change_cancel_id:
                    order["clientOrderId"] = params["newClientOrderId"]
        return deepcopy(order)


@pytest.mark.parametrize("lose_ack", [False, True])
@pytest.mark.parametrize("change_cancel_id", [False, True])
def test_complete_three_asset_lifecycle_and_lost_ack_recovery(
    tmp_path, monkeypatch, lose_ack, change_cancel_id
):
    t = transport(tmp_path, monkeypatch)
    venue = Venue(lose_first_ack=lose_ack, change_cancel_id=change_cancel_id)
    monkeypatch.setattr(t.client, "_request", venue.request)
    campaign = Campaign(t, {"client_id_prefix": "probe"})
    result = campaign.execute()
    assert result["completed"] is True
    assert result["required_live_cases_passed"] is True
    assert result["owned_order_intents"] == venue.posts == 9
    assert all(D(x) == 0 for x in result["residual_base"].values())
    assert len(result["order_cash"]) == 9
    assert sum(len(x) for x in venue.trades.values()) == 6
    assert D(result["quote_cash_delta"]) < 0
    assert not result["profitability_evidence"]
    assert "fake-secret" not in t.journal.path.read_text()
    t.client.session.close()


@pytest.mark.parametrize(
    "damage", ["cancel_receipt", "cancel_identity", "cash", "open_order"]
)
def test_coverage_requires_exact_cancel_receipt_complete_cash_and_flat_orders(
    tmp_path, monkeypatch, damage
):
    t = transport(tmp_path, monkeypatch)
    monkeypatch.setattr(t.client, "_request", Venue().request)
    campaign = Campaign(t, {"client_id_prefix": "probe"})
    campaign.execute()
    rows = deepcopy(t.journal.rows)
    cash = deepcopy(campaign.cash)
    flag = "cancellation_observed"
    if damage == "cancel_receipt":
        next(
            x for x in rows if x["kind"] == "http_completed" and x["method"] == "DELETE"
        )["http_status"] = 500
    elif damage == "cancel_identity":
        next(x for x in rows if x["kind"] == "http_intent" and x["method"] == "DELETE")[
            "params"
        ]["orderId"] = 999
    elif damage == "cash":
        del cash["probe-00"]
        flag = "owned_flat"
    else:
        [
            x
            for x in rows
            if x["kind"] == "open_order_check" and x["symbol"] == "BTCUSDT"
        ][-1]["total"] = 1
        flag = "no_open_orders"
    assert lifecycle_coverage(rows, cash, SYMBOLS)["BTCUSDT"][flag] is False
    t.client.session.close()


def test_never_sell_foreign_inventory(tmp_path, monkeypatch):
    t = transport(tmp_path, monkeypatch)
    c = Campaign(t, {"client_id_prefix": "probe"})
    c.rules["BTCUSDT"] = Rules.parse(symbol_info("BTCUSDT"))
    with pytest.raises(ValueError, match="owned base"):
        c.submit("BTCUSDT", "SELL", D(1), D(100))
    assert not t.journal.intents()
    t.client.session.close()

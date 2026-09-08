"""Exact acquired-inventory limits and active CLI resale isolation."""

from copy import deepcopy
from decimal import Decimal
from argparse import Namespace

import pytest

from simple_ai_trading import cli
from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_spot_receipts import spot_buy_received_quantity
from simple_ai_trading.config import RuntimeConfig


def receipt(base="BTC", quote="USDC", fee="0.001", fee_asset=None):
    return {
        "symbol": base + quote,
        "side": "BUY",
        "type": "MARKET",
        "status": "FILLED",
        "executedQty": "1",
        "origQty": "1",
        "cummulativeQuoteQty": "100",
        "fills": [
            {
                "qty": "1",
                "price": "100",
                "commission": fee,
                "commissionAsset": fee_asset or base,
            }
        ],
    }


@pytest.mark.parametrize("base", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("quote", ["USDT", "USDC"])
@pytest.mark.parametrize(
    "fee_asset,expected", [(None, "0.999"), ("BNB", "1"), ("USDC", "1")]
)
def test_complete_native_fees_limit_only_received_base(
    base, quote, fee_asset, expected
):
    assert spot_buy_received_quantity(
        receipt(base, quote, fee_asset=fee_asset),
        symbol=base + quote,
        base_asset=base,
        quote_asset=quote,
    ) == Decimal(expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("side", "SELL"),
        ("status", "PARTIALLY_FILLED"),
        ("type", "LIMIT"),
        ("symbol", "ETHUSDC"),
        ("executedQty", "0.5"),
        ("fills", []),
        ("fills", None),
    ],
)
def test_nonterminal_or_inconsistent_receipt_cannot_authorize_resale(field, value):
    order = receipt()
    order[field] = value
    with pytest.raises(OpenIntentError):
        spot_buy_received_quantity(
            order, symbol="BTCUSDC", base_asset="BTC", quote_asset="USDC"
        )


@pytest.mark.parametrize("fee", ["-0.001", "NaN", "1", "1.1"])
def test_invalid_or_exhausting_fee_cannot_authorize_resale(fee):
    with pytest.raises(OpenIntentError):
        spot_buy_received_quantity(
            receipt(fee=fee), symbol="BTCUSDC", base_asset="BTC", quote_asset="USDC"
        )


@pytest.mark.parametrize("missing", ["commission", "commissionAsset"])
def test_missing_fee_is_not_assumed_zero(missing):
    order = receipt()
    del order["fills"][0][missing]
    with pytest.raises(OpenIntentError):
        spot_buy_received_quantity(
            order, symbol="BTCUSDC", base_asset="BTC", quote_asset="USDC"
        )


def test_multiple_fills_sum_base_fees_without_rounding():
    order = receipt()
    order["fills"] = [
        {
            "qty": "0.4",
            "price": "100",
            "commission": "0.0004",
            "commissionAsset": "BTC",
        },
        {
            "qty": "0.6",
            "price": "100",
            "commission": "0.0006",
            "commissionAsset": "BTC",
        },
    ]
    assert spot_buy_received_quantity(
        order, symbol="BTCUSDC", base_asset="BTC", quote_asset="USDC"
    ) == Decimal("0.999")


@pytest.mark.parametrize("status", ["CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"])
def test_terminal_partial_purchase_can_resell_its_proved_received_quantity(status):
    order = receipt()
    order.update(status=status, origQty="2")
    assert spot_buy_received_quantity(
        order, symbol="BTCUSDC", base_asset="BTC", quote_asset="USDC"
    ) == Decimal("0.999")


class Client:
    def normalize_quantity(self, symbol, quantity):
        return quantity, None


def second(client=None, *, available="1.999", first=None):
    return cli._roundtrip_second_quantity(
        client or Client(),
        "BTCUSDC",
        "USDC",
        "BTC",
        "SELL",
        1.0,
        {"balances": [{"asset": "BTC", "free": available}]},
        100.0,
        first_order=receipt() if first is None else first,
    )


def test_preexisting_wallet_inventory_cannot_cover_the_buy_fee():
    assert second() == 0.999
    assert second(available="0.4") == 0.4
    assert second(available="0") == 0


def test_wire_precision_floors_fractional_fee_residual():
    assert second(first=receipt(fee="0.000000001")) == 0.99999999


@pytest.mark.parametrize(
    "normalized", [1.0, float("nan"), float("inf"), -1.0, True, "0.5", 0.999000001]
)
def test_normalizer_cannot_enlarge_the_owned_sale_limit(normalized):
    class BadClient:
        def normalize_quantity(self, symbol, quantity):
            return normalized, None

    with pytest.raises(BinanceAPIError):
        second(BadClient())


def test_eight_decimal_wire_rounding_cannot_exceed_unrounded_receipt():
    class RoundedUpOnWire:
        def normalize_quantity(self, symbol, quantity):
            return 0.999999995, None

    with pytest.raises(BinanceAPIError):
        second(RoundedUpOnWire(), first=receipt(fee="0.000000005"))


def test_unsupported_pair_and_absent_payload_rejected():
    for order, base, quote in [
        (receipt(), "BNB", "USDC"),
        (receipt(), "BTC", "EUR"),
        (None, "BTC", "USDC"),
    ]:
        with pytest.raises(OpenIntentError):
            spot_buy_received_quantity(
                order, symbol=base + quote, base_asset=base, quote_asset=quote
            )


def test_complete_receipt_is_not_modified():
    order = receipt()
    before = deepcopy(order)
    second(first=order)
    assert order == before


@pytest.mark.parametrize("missing_fee", [False, True])
def test_active_command_preserves_preexisting_base_or_stops_before_resale(
    tmp_path, monkeypatch, capsys, missing_fee
):
    class RoundtripClient(Client):
        def __init__(self):
            self.base = Decimal("1")
            self.quote = Decimal("1000")
            self.orders = []

        def get_symbol_price(self, symbol):
            return 100.0, 1

        def get_account(self):
            return {
                "balances": [
                    {"asset": "BTC", "free": str(self.base)},
                    {"asset": "USDC", "free": str(self.quote)},
                ]
            }

        def place_order(self, symbol, side, quantity, **kwargs):
            assert kwargs["dry_run"] is False
            self.orders.append((side, quantity))
            if side == "BUY":
                self.base += Decimal("0.999")
                self.quote -= 100
                order = receipt()
                order["orderId"] = 1
                if missing_fee:
                    order["fills"][0].pop("commission")
                    order["fills"][0].pop("commissionAsset")
                return order
            self.base -= Decimal(str(quantity))
            self.quote += Decimal(str(quantity)) * 100
            return {"orderId": 2, "status": "FILLED", "executedQty": str(quantity)}

    client = RoundtripClient()
    artifacts = []
    monkeypatch.setattr(
        cli, "load_runtime", lambda: RuntimeConfig(market_type="spot", testnet=True)
    )
    monkeypatch.setattr(cli, "_has_api_credentials", lambda _: True)
    monkeypatch.setattr(cli, "_build_client", lambda _: client)
    monkeypatch.setattr(cli, "_ensure_runtime_symbol", lambda *_: None)
    monkeypatch.setattr(cli, "_roundtrip_quantity", lambda *_: (1.0, None, 100.0))
    monkeypatch.setattr(
        cli,
        "_persist_run_artifact",
        lambda _, directory, payload: (
            artifacts.append(payload) or tmp_path / "result.json"
        ),
    )
    status = cli.command_spot_roundtrip(
        Namespace(yes=True, mode="buy-sell", quantity=1.0)
    )
    capsys.readouterr()
    if missing_fee:
        assert status == 2
        assert client.orders == [("BUY", 1.0)]
        assert client.base == Decimal("1.999")
        assert artifacts[-1]["status"] == "partial_failed"
    else:
        assert status == 0
        assert client.orders == [("BUY", 1.0), ("SELL", 0.999)]
        assert client.base == Decimal("1")
        assert artifacts[-1]["quantity_second"] == 0.999

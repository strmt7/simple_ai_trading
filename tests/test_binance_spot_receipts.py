"""Exact acquired-inventory limits and active CLI resale isolation."""

from copy import deepcopy
from decimal import Decimal

import pytest

from simple_ai_trading import cli
from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_spot_receipts import spot_buy_received_quantity


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

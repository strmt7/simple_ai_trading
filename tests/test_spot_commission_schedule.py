"""Constructed fee data only; no documentation percentages become defaults."""

from decimal import localcontext
from fractions import Fraction
from unittest.mock import Mock

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.binance_spot_commissions import parse_spot_commission_schedule


def payload():
    return {
        "symbol": "BTCUSDC",
        "standardCommission": dict(
            maker="0.001", taker="0.002", buyer="0.003", seller="0.004"
        ),
        "taxCommission": dict(
            maker="0.0001", taker="0.0002", buyer="0.0003", seller="0.0004"
        ),
        "specialCommission": dict(
            maker="0.01", taker="0.02", buyer="0.03", seller="0.04"
        ),
        "discount": dict(
            enabledForAccount=True,
            enabledForSymbol=True,
            discountAsset="BNB",
            discount="0.75",
        ),
    }


def parse(value=None):
    return parse_spot_commission_schedule(
        payload() if value is None else value, symbol="BTCUSDC"
    )


@pytest.mark.parametrize(
    "side,role,total",
    [
        ("BUY", "taker", "0.0555"),
        ("SELL", "taker", "0.0666"),
        ("BUY", "maker", "0.0444"),
        ("SELL", "maker", "0.0555"),
    ],
)
def test_each_fill_combines_role_and_exact_side_for_all_three_components(
    side, role, total
):
    with localcontext() as context:
        context.prec = 2
        result = parse().equivalent_rate(side=side, role=role, apply_discount=False)
    assert result.total == Fraction(total)


@pytest.mark.parametrize(
    "multiplier,total",
    [("0.75", "0.05425"), ("0.25", "0.05175"), ("0", "0.0505"), ("1", "0.0555")],
)
def test_returned_multiplier_is_not_one_minus_discount_and_does_not_reduce_tax_or_special(
    multiplier, total
):
    value = payload()
    value["discount"]["discount"] = multiplier
    result = parse(value).equivalent_rate(side="BUY", role="taker", apply_discount=True)
    assert result.total == Fraction(total)
    assert result.tax == Fraction("0.0005") and result.special == Fraction("0.05")


@pytest.mark.parametrize(
    "account,symbol", [(False, False), (False, True), (True, False), (True, True)]
)
def test_both_flags_and_received_asset_fallback_remain_explicit(account, symbol):
    value = payload()
    value["discount"].update(enabledForAccount=account, enabledForSymbol=symbol)
    schedule = parse(value)
    eligible = account and symbol
    assert schedule.possible_fee_assets(side="BUY") == frozenset(
        {"BTC", "BNB"} if eligible else {"BTC"}
    )
    assert schedule.possible_fee_assets(side="SELL") == frozenset(
        {"USDC", "BNB"} if eligible else {"USDC"}
    )
    if not eligible:
        with pytest.raises(ValueError):
            schedule.equivalent_rate(side="BUY", role="taker", apply_discount=True)


def test_disabled_unknown_multiplier_and_asset_are_not_invented_or_used():
    value = payload()
    value["discount"] = dict(enabledForAccount=False, enabledForSymbol=True)
    schedule = parse(value)
    assert schedule.discount.asset is None and schedule.discount.multiplier is None
    assert schedule.possible_fee_assets(side="BUY") == frozenset({"BTC"})
    assert schedule.equivalent_rate(
        side="BUY", role="taker", apply_discount=False
    ).total == Fraction("0.0555")


@pytest.mark.parametrize(
    "component", ["standardCommission", "specialCommission", "taxCommission"]
)
@pytest.mark.parametrize("field", ["maker", "taker", "buyer", "seller"])
def test_no_missing_rate_is_defaulted_to_zero(component, field):
    value = payload()
    del value[component][field]
    with pytest.raises(ValueError):
        parse(value)


@pytest.mark.parametrize(
    "bad", [0.001, True, None, "-0.1", "NaN", "1e-3", "1.01", "1" * 31]
)
def test_native_rates_require_exact_supported_nonnegative_decimal_strings(bad):
    value = payload()
    value["standardCommission"]["taker"] = bad
    with pytest.raises(ValueError):
        parse(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabledForAccount", "false"),
        ("enabledForSymbol", 1),
        ("enabledForAccount", None),
        ("discountAsset", None),
        ("discountAsset", "bnb"),
        ("discount", None),
        ("discount", 0.75),
        ("discount", "1.1"),
    ],
)
def test_eligible_routing_cannot_use_truthiness_or_missing_metadata(field, value):
    data = payload()
    data["discount"][field] = value
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize(
    "changes", [{"symbol": "ETHUSDC"}, {"standardCommission": None}, {"discount": None}]
)
def test_foreign_or_incomplete_account_data_rejects(changes):
    with pytest.raises(ValueError):
        parse({**payload(), **changes})


def test_order_test_schema_is_not_mistaken_for_account_side_components():
    value = payload()
    for name in ("standardCommission", "taxCommission", "specialCommission"):
        component = value.pop(name)
        value[name + "ForOrder"] = {key: component[key] for key in ("maker", "taker")}
    with pytest.raises(ValueError):
        parse(value)


def test_exact_scoped_api_preserves_routing_and_does_not_duplicate_queries(monkeypatch):
    client = BinanceClient("offline-key", "offline-secret", market_type="spot")
    try:
        monkeypatch.setattr(
            client.session,
            "request",
            Mock(side_effect=AssertionError("network forbidden")),
        )
        transport = Mock(return_value=payload())
        monkeypatch.setattr(client, "_request_dict", transport)
        scope = client.execution_scope()
        schedule = client.get_spot_commission_schedule("btcusdc", expected_scope=scope)
        assert schedule.discount.multiplier == Fraction(3, 4)
        args, kwargs = transport.call_args
        assert args[:3] == ("GET", "/api/v3/account/commission", {"symbol": "BTCUSDC"})
        assert kwargs["expected_scope"] == scope and kwargs["signed"] is True
        assert transport.call_count == 1
    finally:
        client.session.close()


def test_new_scoped_reader_rejects_rotation_before_http(monkeypatch):
    client = BinanceClient("offline-key", "offline-secret", market_type="spot")
    try:
        scope = client.execution_scope()
        client.api_key = "rotated-offline-key"
        transport = Mock(side_effect=AssertionError("foreign scope must not query"))
        monkeypatch.setattr(client, "_request_dict", transport)
        with pytest.raises(BinanceAPIError):
            client.get_spot_commission_schedule("BTCUSDC", expected_scope=scope)
        transport.assert_not_called()
    finally:
        client.session.close()


@pytest.mark.parametrize("symbol", ["BNBUSDC", "BTCBUSD", "BTC/USDC", "BTC-USDC", ""])
def test_unqualified_symbol_never_becomes_a_signed_query(monkeypatch, symbol):
    client = BinanceClient("offline-key", "offline-secret", market_type="spot")
    try:
        transport = Mock(side_effect=AssertionError("invalid symbol must not query"))
        monkeypatch.setattr(client, "_request_dict", transport)
        with pytest.raises(BinanceAPIError):
            client.get_spot_commission_schedule(
                symbol, expected_scope=client.execution_scope()
            )
        transport.assert_not_called()
    finally:
        client.session.close()


@pytest.mark.parametrize(
    "side,role,discount",
    [
        ("HOLD", "taker", False),
        ("BUY", "market", False),
        ([], "taker", False),
        ("BUY", "taker", 1),
    ],
)
def test_invalid_fill_or_discount_selection_cannot_change_fee_meaning(
    side, role, discount
):
    with pytest.raises(ValueError):
        parse().equivalent_rate(side=side, role=role, apply_discount=discount)

"""Exact quote principals through the real client; all HTTP is forbidden."""

from decimal import localcontext
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.binance_order_responses import SpotQuoteOrderBinding


@pytest.fixture
def client(monkeypatch):
    instance = BinanceClient("offline-key", "offline-secret", market_type="spot")
    monkeypatch.setattr(
        instance.session,
        "request",
        Mock(side_effect=AssertionError("network forbidden")),
    )
    yield instance
    instance.session.close()


def receipt(quote="100", spent="100", **changes):
    return {
        "symbol": "BTCUSDC",
        "side": "BUY",
        "type": "MARKET",
        "orderId": 42,
        "clientOrderId": "sait-o-quote-bound",
        "origQuoteOrderQty": quote,
        "cummulativeQuoteQty": spent,
        "origQty": "0.1",
        "executedQty": "0.1",
        "status": "FILLED",
        **changes,
    }


def submit(client, **changes):
    arguments = {
        "symbol": "BTCUSDC",
        "quote_quantity": "100",
        "dry_run": False,
        "client_order_id": "sait-o-quote-bound",
        "expected_scope": client.execution_scope(),
    }
    arguments.update(changes)
    return client.place_spot_quote_buy(**arguments)


@pytest.mark.parametrize("derived_base", ["1", "0.1", "0.01"])
def test_price_changes_do_not_convert_quote_budget_back_to_fixed_base(
    client, monkeypatch, derived_base
):
    native = receipt(origQty=derived_base, executedQty=derived_base)
    transport = Mock(return_value=native)
    monkeypatch.setattr(client, "_request_dict", transport)
    assert submit(client) is native
    args, kwargs = transport.call_args
    assert args[:2] == ("POST", "/api/v3/order")
    assert args[2] == {
        "symbol": "BTCUSDC",
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": "100",
        "newClientOrderId": "sait-o-quote-bound",
        "newOrderRespType": "FULL",
    }
    assert kwargs["expected_scope"] == client.execution_scope()
    assert transport.call_count == 1


def test_long_decimal_quote_is_never_rounded_through_float(client, monkeypatch):
    amount = "12345678901234567890.12345678901234567890123456789"
    transport = Mock(return_value=receipt(amount, amount))
    monkeypatch.setattr(client, "_request_dict", transport)
    with localcontext() as context:
        context.prec = 2
        submit(client, quote_quantity=amount)
    assert transport.call_args.args[2]["quoteOrderQty"] == amount


@pytest.mark.parametrize(
    "changes",
    [
        {"origQuoteOrderQty": "99"},
        {"origQuoteOrderQty": "0"},
        {"origQuoteOrderQty": "100.000000000000000000000000000001"},
        {"origQuoteOrderQty": 100},
        {"origQuoteOrderQty": "1e2"},
        {"cummulativeQuoteQty": "100.000000000000000000000000000001"},
        {"cummulativeQuoteQty": "-1"},
        {"cummulativeQuoteQty": "NaN"},
        {"cummulativeQuoteQty": 100},
        {"cummulativeQuoteQty": None},
        {"symbol": "ETHUSDC"},
        {"side": "SELL"},
        {"type": "LIMIT"},
        {"clientOrderId": "foreign"},
        {"origClientOrderId": "foreign"},
        {"orderId": True},
    ],
)
def test_wrong_or_overspending_acknowledgment_cannot_be_substituted_or_retried(
    client, monkeypatch, changes
):
    transport = Mock(return_value=receipt(**changes))
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        submit(client)
    assert transport.call_count == 1


@pytest.mark.parametrize(
    "amount",
    [
        0,
        100.0,
        True,
        None,
        "0",
        "-1",
        "NaN",
        "1e2",
        " 100",
        "１００",
        "1" * 31,
        "1." + "1" * 31,
    ],
)
def test_invalid_or_inexact_request_never_reaches_transport(
    client, monkeypatch, amount
):
    transport = Mock(side_effect=AssertionError("invalid request must not dispatch"))
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        submit(client, quote_quantity=amount)
    transport.assert_not_called()


def test_changed_scope_rejects_before_dispatch(client, monkeypatch):
    scope = client.execution_scope()
    client.api_key = "rotated-offline-key"
    transport = Mock(side_effect=AssertionError("changed scope must not dispatch"))
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError, match="scope changed"):
        submit(client, expected_scope=scope)
    transport.assert_not_called()


def test_quote_path_cannot_become_futures_or_mainnet(client, monkeypatch):
    scope = client.execution_scope()
    transport = Mock(side_effect=AssertionError("forbidden product must not dispatch"))
    monkeypatch.setattr(client, "_request_dict", transport)
    client.market_type = "futures"
    with pytest.raises(BinanceAPIError, match="requires Spot"):
        client.place_spot_quote_buy(
            "BTCUSDC",
            "100",
            dry_run=False,
            client_order_id="sait-o-quote-bound",
            expected_scope=scope,
        )
    client.market_type = "spot"
    client.base_url = "https://api.binance.com"
    with pytest.raises(BinanceAPIError, match="disabled"):
        client.place_spot_quote_buy(
            "BTCUSDC",
            "100",
            dry_run=False,
            client_order_id="sait-o-quote-bound",
            expected_scope=scope,
        )
    transport.assert_not_called()


def test_dry_run_has_explicit_quote_units_and_no_http(client, monkeypatch):
    transport = Mock(side_effect=AssertionError("dry run must not dispatch"))
    monkeypatch.setattr(client, "_request_dict", transport)
    proposal = submit(client, dry_run=True)
    assert proposal["dryRun"] is True and proposal["quoteOrderQty"] == "100"
    assert "quantity" not in proposal
    transport.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "BNBUSDC"},
        {"symbol": "BTCBUSD"},
        {"expected_scope": None},
        {"expected_scope": True},
        {"dry_run": 1},
        {"client_order_id": ""},
        {"client_order_id": None},
    ],
)
def test_missing_authority_or_unsupported_asset_never_dispatches(
    client, monkeypatch, changes
):
    transport = Mock(side_effect=AssertionError("invalid authority must not dispatch"))
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        submit(client, **changes)
    transport.assert_not_called()


def test_quote_recovery_without_frozen_scope_never_queries(client, monkeypatch):
    binding = SpotQuoteOrderBinding("BTCUSDC", "100", "sait-o-quote-bound")
    transport = Mock(side_effect=AssertionError("unscoped recovery must not query"))
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError, match="explicit Spot scope"):
        client.get_order(
            "BTCUSDC",
            orig_client_order_id=binding.client_order_id,
            expected_order_binding=binding,
        )
    transport.assert_not_called()


@pytest.mark.parametrize("status", [200, 503])
def test_actual_signed_transport_keeps_quote_principal_and_never_retries_post(
    client, monkeypatch, status
):
    amount = "100.12345678901234567890123456789"
    response = Mock(status_code=status, headers={}, text="unavailable")
    response.json.return_value = receipt(amount, amount)
    wire = Mock(return_value=response)
    monkeypatch.setattr(client.session, "request", wire)
    monkeypatch.setattr(client, "_throttle", lambda: None)
    client.max_retries = 9
    if status == 200:
        submit(client, quote_quantity=amount)
    else:
        with pytest.raises(BinanceAPIError, match="503"):
            submit(client, quote_quantity=amount)
    assert wire.call_count == 1
    assert wire.call_args.args[0] == "POST"
    query = parse_qs(urlsplit(wire.call_args.args[1]).query)
    assert query["quoteOrderQty"] == [amount]
    assert query["newClientOrderId"] == ["sait-o-quote-bound"]
    assert "quantity" not in query
    assert wire.call_args.kwargs["params"] is None


def test_real_transport_rechecks_scope_after_high_level_admission(client, monkeypatch):
    native_request = client._request_dict

    def rotate_before_native_request(*args, **kwargs):
        client.api_key = "rotated-before-native-request"
        return native_request(*args, **kwargs)

    monkeypatch.setattr(client, "_request_dict", rotate_before_native_request)
    with pytest.raises(BinanceAPIError, match="differs from durable intent"):
        submit(client)
    client.session.request.assert_not_called()


@pytest.mark.parametrize(
    "spent,executed,status",
    [
        ("0", "0", "NEW"),
        ("49.99", "0.4999", "PARTIALLY_FILLED"),
        ("100", "1", "FILLED"),
    ],
)
def test_exact_quote_binding_survives_query_recovery_with_dynamic_base(
    client, monkeypatch, spent, executed, status
):
    binding = SpotQuoteOrderBinding("BTCUSDC", "100", "sait-o-quote-bound")
    native = receipt(spent=spent, status=status, origQty="1", executedQty=executed)
    transport = Mock(return_value=native)
    monkeypatch.setattr(client, "_request_dict", transport)
    assert (
        client.get_order(
            "BTCUSDC",
            order_id=42,
            orig_client_order_id=binding.client_order_id,
            expected_scope=client.execution_scope(),
            expected_order_binding=binding,
        )
        is native
    )
    assert transport.call_count == 1


def test_query_recovery_does_not_treat_orig_base_as_quote_evidence(client, monkeypatch):
    binding = SpotQuoteOrderBinding("BTCUSDC", "100", "sait-o-quote-bound")
    native = receipt(quote="99", origQty="100")
    transport = Mock(return_value=native)
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        client.get_order(
            "BTCUSDC",
            orig_client_order_id=binding.client_order_id,
            expected_scope=client.execution_scope(),
            expected_order_binding=binding,
        )
    assert transport.call_count == 1

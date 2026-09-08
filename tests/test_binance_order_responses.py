"""Request-bound response admission through the real client with offline transport."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from unittest.mock import Mock

import pytest

from simple_ai_trading import cli
from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.autonomous import (
    _close_to_trade,
    _submit_durable_close_position,
    _submit_durable_open_position,
)
from simple_ai_trading.binance_order_responses import (
    MarketOrderBinding,
    validate_queried_order,
)
from simple_ai_trading.config import RuntimeConfig, StrategyConfig
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _binding(product: str = "spot", *, reduce_only: bool = False) -> MarketOrderBinding:
    return MarketOrderBinding(
        "BTCUSDC",
        "SELL" if reduce_only else "BUY",
        "1.00000000",
        "sait-o-bound",
        product,
        reduce_only,
    )


def _response(binding: MarketOrderBinding) -> dict:
    value = {
        "symbol": binding.symbol,
        "side": binding.side,
        "type": "MARKET",
        "origQty": binding.quantity,
        "executedQty": binding.quantity,
        "orderId": 12,
        "clientOrderId": binding.client_order_id,
        "status": "FILLED",
        "avgPrice": "100",
        "positionSide": "BOTH",
        "reduceOnly": binding.reduce_only,
        "closePosition": False,
        "origType": "MARKET",
        "fills": [
            {
                "qty": binding.quantity,
                "price": "100",
                "commission": "0",
                "commissionAsset": "BTC",
            }
        ],
    }
    value["cummulativeQuoteQty" if binding.market_type == "spot" else "cumQuote"] = str(
        Decimal(binding.quantity) * 100
    )
    return value


def _client(product: str, monkeypatch) -> BinanceClient:
    client = BinanceClient(
        "offline-placeholder", "offline-placeholder", market_type=product
    )
    monkeypatch.setattr(
        client.session, "request", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(client, "set_leverage", Mock(return_value={"leverage": 1}))
    return client


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "ETHUSDT"},
        {"orderId": True},
        {"orderId": "01"},
        {"clientOrderId": "foreign"},
        {"origClientOrderId": "foreign"},
        {"side": "SELL"},
        {"type": "LIMIT"},
        {"origQty": "2"},
        {"origQty": 1},
        {"origQty": "NaN"},
        {"origQty": "1e0"},
        {"origQty": "1.000000000000000000000000000001"},
    ],
)
def test_post_rejects_mismatched_identity_and_exact_request_quantity(
    monkeypatch, product, changes
):
    client = _client(product, monkeypatch)
    response = {**_response(_binding(product)), **changes}
    transport = Mock(return_value=response)
    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        client.place_order(
            "BTCUSDC", "BUY", 1, dry_run=False, client_order_id="sait-o-bound"
        )
    assert transport.call_count == 1


@pytest.mark.parametrize("reduce_only", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"positionSide": "LONG"},
        {"positionSide": "SHORT"},
        {"reduceOnly": "false"},
        {"reduceOnly": 0},
        {"closePosition": True},
        {"closePosition": "false"},
        {"origType": "STOP_MARKET"},
    ],
)
def test_futures_market_mode_never_changes_meaning(monkeypatch, reduce_only, changes):
    client = _client("futures", monkeypatch)
    binding = _binding("futures", reduce_only=reduce_only)
    monkeypatch.setattr(
        client, "_request_dict", Mock(return_value={**_response(binding), **changes})
    )
    with pytest.raises(BinanceAPIError):
        client.place_order(
            binding.symbol,
            binding.side,
            1,
            dry_run=False,
            reduce_only=reduce_only,
            client_order_id=binding.client_order_id,
        )
    assert client.set_leverage.call_count == int(not reduce_only)


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize("selectors", ["order", "client", "both"])
def test_query_requires_every_supplied_selector(monkeypatch, product, selectors):
    client = _client(product, monkeypatch)
    response = _response(_binding(product))
    kwargs = {}
    if selectors != "client":
        kwargs["order_id"] = 12
    if selectors != "order":
        kwargs["orig_client_order_id"] = "sait-o-bound"
    transport = Mock(return_value=response)
    monkeypatch.setattr(client, "_request_dict", transport)
    assert client.get_order("BTCUSDC", **kwargs) == response
    for field, wrong in (
        ("symbol", "ETHUSDT"),
        ("orderId", 13),
        ("clientOrderId", "foreign"),
    ):
        if (
            field == "orderId"
            and selectors == "client"
            or field == "clientOrderId"
            and selectors == "order"
        ):
            continue
        transport.return_value = {**response, field: wrong}
        with pytest.raises(BinanceAPIError):
            client.get_order("BTCUSDC", **kwargs)


@pytest.mark.parametrize("bad", [True, 1.0, " 1", "01", "1.0", -1, "", 10**30])
def test_query_rejects_ambiguous_id_before_transport(monkeypatch, bad):
    client = _client("spot", monkeypatch)
    with pytest.raises(BinanceAPIError, match="exact integer"):
        client.get_order("BTCUSDC", order_id=bad)
    client.session.request.assert_not_called()


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_generated_identity_is_known_before_post_and_explicit_identity_preserved(
    monkeypatch, product
):
    client = _client(product, monkeypatch)
    sent = []

    def transport(method, path, params, **kwargs):
        sent.append(dict(params))
        binding = MarketOrderBinding(
            params["symbol"],
            params["side"],
            params["quantity"],
            params["newClientOrderId"],
            product,
            params.get("reduceOnly") == "true",
        )
        return _response(binding)

    monkeypatch.setattr(client, "_request_dict", transport)
    for identity in (None, None, "exact-user-id"):
        returned = client.place_order(
            "BTCUSDC", "BUY", 1.000000004, dry_run=False, client_order_id=identity
        )
        assert returned["origQty"] == "1.00000000"
        assert returned["clientOrderId"] == sent[-1]["newClientOrderId"]
    assert sent[0]["newClientOrderId"] != sent[1]["newClientOrderId"]
    assert len(sent[0]["newClientOrderId"]) == 36
    assert sent[-1]["newClientOrderId"] == "exact-user-id"


@pytest.mark.parametrize("quantity,reduce_only", [(1e-12, False), (1, "false"), (1, 0)])
def test_invalid_transmitted_quantity_or_boolean_never_posts(
    monkeypatch, quantity, reduce_only
):
    client = _client("futures", monkeypatch)
    with pytest.raises(BinanceAPIError):
        client.place_order(
            "BTCUSDC", "BUY", quantity, dry_run=False, reduce_only=reduce_only
        )
    client.set_leverage.assert_not_called()


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize(
    "changes", [{"side": "SELL"}, {"origQty": "2"}, {"type": "LIMIT"}]
)
def test_response_loss_query_cannot_bypass_submission_binding(
    monkeypatch, tmp_path, product, changes
):
    client = _client(product, monkeypatch)
    binding = _binding(product)
    position = OpenPosition(
        id="bound",
        symbol=binding.symbol,
        market_type=product,
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id=binding.client_order_id,
    )
    calls = []

    def transport(method, path, params, **kwargs):
        calls.append(method)
        if method == "POST":
            raise BinanceAPIError("injected lost response")
        return {**_response(binding), **changes}

    monkeypatch.setattr(client, "_request_dict", transport)
    store = PositionsStore(tmp_path)
    with pytest.raises(BinanceAPIError):
        _submit_durable_open_position(client, position, store)
    assert calls == ["POST", "GET"]
    assert store.load_open() == []
    assert (
        store.opening_intents.pending_position(scope=client.execution_scope())
        == position
    )


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_cli_pending_query_uses_first_validated_submission_semantics(
    monkeypatch, product
):
    client = _client(product, monkeypatch)
    binding = _binding(product)
    original = _response(binding)
    original.update(executedQty="0", status="NEW", avgPrice="0", fills=[])
    original["cummulativeQuoteQty" if product == "spot" else "cumQuote"] = "0"
    monkeypatch.setattr(
        client,
        "_request_dict",
        Mock(return_value={**_response(binding), "origQty": "2"}),
    )
    with pytest.raises(BinanceAPIError, match="quantity"):
        cli._resolved_order_fill_details(
            client,
            RuntimeConfig(symbol="BTCUSDC", market_type=product),
            original,
            fallback_qty=1,
            fallback_price=100,
            dry_run=False,
        )


@pytest.mark.parametrize("product", ["spot", "futures"])
def test_cli_response_loss_carries_original_side_and_quantity(monkeypatch, product):
    client = _client(product, monkeypatch)
    binding = _binding(product)

    def transport(method, path, params, **kwargs):
        if method == "POST":
            raise BinanceAPIError("injected lost response")
        return {**_response(binding), "side": "SELL"}

    monkeypatch.setattr(client, "_request_dict", transport)
    with pytest.raises(BinanceAPIError):
        cli._paper_or_live_order(
            client,
            RuntimeConfig(symbol="BTCUSDC", market_type=product),
            StrategyConfig(),
            side="BUY",
            size=1,
            dry_run=False,
            client_order_id=binding.client_order_id,
        )


@pytest.mark.parametrize(
    "changes",
    [{"symbol": "ETHUSDT"}, {"market_type": "futures"}, {"client_order_id": "foreign"}],
)
def test_query_context_mismatch_stops_before_access(monkeypatch, changes):
    client = _client("spot", monkeypatch)
    with pytest.raises(BinanceAPIError, match="binding"):
        client.get_order(
            "BTCUSDC",
            orig_client_order_id="sait-o-bound",
            expected_order_binding=replace(_binding(), **changes),
        )
    client.session.request.assert_not_called()


def test_nonmapping_response_rejects_and_zero_order_id_is_exact():
    with pytest.raises(ValueError):
        validate_queried_order(
            None, symbol="BTCUSDC", order_id="0", client_order_id=None
        )
    validate_queried_order(
        {"symbol": "BTCUSDC", "orderId": 0},
        symbol="BTCUSDC",
        order_id="0",
        client_order_id=None,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "bad"},
        {"side": "HOLD"},
        {"quantity": "0"},
        {"quantity": "NaN"},
        {"quantity": 1},
        {"market_type": "inverse"},
        {"client_order_id": " "},
        {"client_order_id": None},
        {"reduce_only": "false"},
    ],
)
def test_invalid_binding_cannot_be_constructed(changes):
    with pytest.raises(ValueError):
        replace(_binding(), **changes)


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize(
    "field", ["symbol", "side", "type", "origQty", "orderId", "clientOrderId"]
)
def test_missing_required_post_evidence_is_not_accepted(monkeypatch, product, field):
    client = _client(product, monkeypatch)
    response = _response(_binding(product))
    del response[field]
    monkeypatch.setattr(client, "_request_dict", Mock(return_value=response))
    with pytest.raises(BinanceAPIError):
        client.place_order(
            "BTCUSDC", "BUY", 1, dry_run=False, client_order_id="sait-o-bound"
        )


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize(
    "changes", [{"side": "BUY"}, {"origQty": "2"}, {"type": "LIMIT"}]
)
def test_close_response_loss_preserves_owned_lot_and_unknown(
    monkeypatch, tmp_path, product, changes
):
    client = _client(product, monkeypatch)
    binding = _binding(product)
    position = OpenPosition(
        id="bound",
        symbol=binding.symbol,
        market_type=product,
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id=binding.client_order_id,
    )
    monkeypatch.setattr(client, "_request_dict", Mock(return_value=_response(binding)))
    store = PositionsStore(tmp_path)
    recorded = _submit_durable_open_position(client, position, store)
    calls = []
    closing = None

    def transport(method, path, params, **kwargs):
        nonlocal closing
        calls.append(method)
        if method == "POST":
            closing = replace(
                binding,
                side="SELL",
                client_order_id=params["newClientOrderId"],
                reduce_only=product == "futures",
            )
            raise BinanceAPIError("injected lost close response")
        return {**_response(closing), **changes}

    monkeypatch.setattr(client, "_request_dict", transport)
    trade = _close_to_trade(recorded, 100, "binding-test", clock=lambda: 2)
    with pytest.raises(BinanceAPIError):
        _submit_durable_close_position(
            client, recorded, trade, store, reduce_only=product == "futures"
        )
    assert calls == ["POST", "GET"]
    assert store.load_open() == [recorded]
    assert store.load_ledger() == []
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


def test_missing_pending_response_semantics_never_query(monkeypatch):
    client = _client("spot", monkeypatch)
    with pytest.raises(BinanceAPIError, match="request semantics"):
        cli._resolved_order_fill_details(
            client,
            RuntimeConfig(),
            {"orderId": 12, "clientOrderId": "sait-o-bound", "origQty": "1"},
            fallback_qty=1,
            fallback_price=100,
            dry_run=False,
        )
    client.session.request.assert_not_called()


def test_binding_validation_errors_use_api_error_before_transmission(monkeypatch):
    client = _client("spot", monkeypatch)
    with pytest.raises(BinanceAPIError, match="binding"):
        client.place_order("BTC USDC", "BUY", 1, dry_run=False)
    client.session.request.assert_not_called()


def test_nonbinding_query_context_is_rejected_before_transmission(monkeypatch):
    client = _client("spot", monkeypatch)
    with pytest.raises(BinanceAPIError, match="binding"):
        client.get_order(
            "BTCUSDC", orig_client_order_id="sait-o-bound", expected_order_binding={}
        )
    client.session.request.assert_not_called()


@pytest.mark.parametrize("product", ["spot", "futures"])
@pytest.mark.parametrize("phase", ["opening", "closing"])
def test_exact_response_loss_recovery_still_completes_normally(
    monkeypatch, tmp_path, product, phase
):
    client = _client(product, monkeypatch)
    binding = _binding(product)
    position = OpenPosition(
        id="bound",
        symbol=binding.symbol,
        market_type=product,
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id=binding.client_order_id,
    )
    store = PositionsStore(tmp_path)
    if phase == "closing":
        monkeypatch.setattr(
            client, "_request_dict", Mock(return_value=_response(binding))
        )
        position = _submit_durable_open_position(client, position, store)
    calls = []
    queried = None

    def transport(method, path, params, **kwargs):
        nonlocal queried
        calls.append(method)
        if method == "POST":
            queried = replace(
                binding,
                side=params["side"],
                client_order_id=params["newClientOrderId"],
                reduce_only=params.get("reduceOnly") == "true",
            )
            raise BinanceAPIError("injected response loss")
        return _response(queried)

    monkeypatch.setattr(client, "_request_dict", transport)
    if phase == "opening":
        result = _submit_durable_open_position(client, position, store)
        assert store.load_open() == [result]
    else:
        trade = _close_to_trade(position, 100, "binding-test", clock=lambda: 2)
        result = _submit_durable_close_position(
            client, position, trade, store, reduce_only=product == "futures"
        )
        assert store.load_open() == [] and store.load_ledger() == [result]
    assert calls == ["POST", "GET"]
    assert store.opening_intents.entry_block_reason() is None

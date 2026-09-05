"""Ambiguous writes must reach reconciliation before another submission."""

from __future__ import annotations

from json import JSONDecodeError
from unittest.mock import Mock

import pytest
import requests

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.autonomous import _submit_open_position
from simple_ai_trading.positions import OpenPosition


def _response(failure: str) -> Mock | requests.Timeout:
    if failure == "transport_timeout":
        return requests.Timeout("response lost after simulated acceptance")
    response = Mock(status_code=200, headers={}, text="simulated response")
    response.json.return_value = {"orderId": 123, "status": "FILLED"}
    if failure == "server_error":
        response.status_code = 503
    elif failure == "rate_limit":
        response.status_code = 429
        response.headers = {"Retry-After": "1"}
    elif failure == "malformed_json":
        response.json.side_effect = JSONDecodeError("incomplete", "", 0)
    elif failure == "api_timeout":
        response.json.return_value = {"code": -1007, "msg": "execution unknown"}
    elif failure == "api_rate_limit":
        response.json.return_value = {"code": -1003, "msg": "rate limited"}
    return response


@pytest.mark.parametrize("method", ["POST", "post", "PUT", "DELETE"])
@pytest.mark.parametrize(
    "failure",
    [
        "transport_timeout",
        "server_error",
        "rate_limit",
        "malformed_json",
        "api_timeout",
        "api_rate_limit",
    ],
)
def test_ambiguous_write_never_replays_in_transport(
    monkeypatch, method, failure
) -> None:
    client = BinanceClient(api_key="", api_secret="", max_retries=4)
    request = Mock(side_effect=[_response(failure), _response("success")])
    sleep = Mock()
    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(client, "_throttle", Mock())
    monkeypatch.setattr("simple_ai_trading.api.time.sleep", sleep)

    # Signing is intentionally absent: test the shared transport with no secrets.
    with pytest.raises(BinanceAPIError):
        client._request(method, "/api/v3/order", {"newClientOrderId": "sait-o-123"})

    assert request.call_count == 1
    sleep.assert_not_called()
    assert client.last_request_info["attempts"] == 1
    assert client.last_request_info["retries"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        "transport_timeout",
        "server_error",
        "rate_limit",
        "malformed_json",
        "api_timeout",
        "api_rate_limit",
    ],
)
def test_existing_read_retry_budget_is_preserved(monkeypatch, failure) -> None:
    client = BinanceClient(api_key="", api_secret="", max_retries=1)
    request = Mock(side_effect=[_response(failure), _response("success")])
    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(client, "_throttle", Mock())
    monkeypatch.setattr("simple_ai_trading.api.time.sleep", Mock())

    result = client._request(
        "GET", "/api/v3/order", {"origClientOrderId": "sait-o-123"}
    )

    assert result == {"orderId": 123, "status": "FILLED"}
    assert request.call_count == 2
    assert client.last_request_info["attempts"] == 2
    assert client.last_request_info["retries"] == 1


@pytest.mark.parametrize("query_found", [True, False])
def test_autonomous_recovery_queries_before_any_resubmission(
    monkeypatch, query_found
) -> None:
    client = BinanceClient(api_key="", api_secret="", max_retries=4)
    identity = "sait-o-recovery123"
    position = OpenPosition(
        id="recovery123",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=0.001,
        entry_price=50_000.0,
        leverage=1.0,
        opened_at_ms=1,
        notional=50.0,
        dry_run=False,
        open_client_order_id=identity,
    )
    recovered = _response("success")
    recovered.json.return_value = (
        {
            "symbol": "BTCUSDT",
            "orderId": 123,
            "clientOrderId": identity,
            "status": "FILLED",
            "executedQty": "0.001",
            "cummulativeQuoteQty": "50",
        }
        if query_found
        else {"code": -2013, "msg": "order not found"}
    )
    request = Mock(side_effect=[_response("transport_timeout"), recovered])
    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(client, "_throttle", Mock())

    def unsigned_test_transport(method, path, params=None, *, signed=False, label=""):
        # Retain production submission, transport and recovery; omit only signing.
        return client._request(method, path, params, signed=False)

    monkeypatch.setattr(client, "_request_dict", unsigned_test_transport)
    if query_found:
        result = _submit_open_position(client, position)
        assert result.open_client_order_id == identity
        assert result.open_exchange_order_id == "123"
        assert result.exchange_status == "FILLED"
    else:
        with pytest.raises(BinanceAPIError, match="order not found"):
            _submit_open_position(client, position)
        assert position.open_exchange_order_id == ""

    assert [call.args[0] for call in request.call_args_list] == ["POST", "GET"]
    assert request.call_args_list[0].kwargs["params"]["newClientOrderId"] == identity
    assert request.call_args_list[1].kwargs["params"]["origClientOrderId"] == identity


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_requests_cannot_replay_or_forward_a_write_through_redirect(
    monkeypatch, status
) -> None:
    sent = []

    class RedirectAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            sent.append(request)
            response = requests.Response()
            response.request = request
            response.url = request.url
            response.status_code = status if len(sent) == 1 else 200
            response._content = b"{}"
            if len(sent) == 1:
                response.headers["Location"] = "https://redirect.invalid/order"
            return response

        def close(self) -> None:
            pass

    client = BinanceClient(api_key="", api_secret="", max_retries=4)
    client.session.mount("https://", RedirectAdapter())
    monkeypatch.setattr(client, "_throttle", Mock())
    try:
        with pytest.raises(BinanceAPIError):
            client._request("POST", "/api/v3/order", {"newClientOrderId": "sait-o-123"})
        assert len(sent) == 1
        assert sent[0].url.startswith("https://testnet.binance.vision/")
        assert client.last_request_info["attempts"] == 1
    finally:
        client.session.close()

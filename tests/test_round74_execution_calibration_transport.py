from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from typing import Mapping
from urllib.parse import urlsplit

import pytest
import requests

from simple_ai_trading.round74_execution_calibration_coordinator import (
    Round74OrderSubmissionRejected,
    Round74OrderSubmissionUnknown,
)
from simple_ai_trading.round74_execution_calibration_transport import (
    ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES,
    ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL,
    ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL,
    Round74BinanceTestnetExecutionTransport,
)


@dataclass(frozen=True)
class _Response:
    payload: object
    headers: Mapping[str, object]
    status_code: int = 200

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload, separators=(",", ":")).encode("ascii")

    def json(self) -> object:
        return self.payload


class _Session:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.order_status = 200
        self.order_payload: object = {"orderId": 1001}
        self.raise_order_timeout = False
        self.exchange_information_padding = ""

    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _Response:
        call = {"method": method, "url": url, **kwargs}
        self.calls.append(call)
        path = urlsplit(url).path
        params = dict(kwargs.get("params", {}))
        if path == "/fapi/v1/time":
            return _Response(
                {"serverTime": 1_800_000_000_000},
                {"X-MBX-USED-WEIGHT-1M": "1"},
            )
        if path == "/fapi/v1/listenKey" and method == "POST":
            return _Response({"listenKey": "test-listen-key"}, {})
        if path == "/fapi/v1/listenKey" and method == "DELETE":
            return _Response({}, {})
        if path == "/fapi/v2/positionRisk":
            return _Response(
                [
                    {
                        "symbol": params["symbol"],
                        "positionSide": "BOTH",
                        "positionAmt": "0",
                    }
                ],
                {"X-MBX-USED-WEIGHT-1M": "2"},
            )
        if path == "/fapi/v1/openOrders":
            return _Response([], {})
        if path == "/fapi/v1/exchangeInfo":
            return _Response(
                {
                    "padding": self.exchange_information_padding,
                    "rateLimits": [
                        {
                            "rateLimitType": "REQUEST_WEIGHT",
                            "interval": "MINUTE",
                            "intervalNum": 1,
                            "limit": 2400,
                        },
                        {
                            "rateLimitType": "ORDERS",
                            "interval": "MINUTE",
                            "intervalNum": 1,
                            "limit": 1200,
                        },
                    ],
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "pair": "BTCUSDT",
                            "contractType": "PERPETUAL",
                            "status": "TRADING",
                            "quoteAsset": "USDT",
                            "marginAsset": "USDT",
                            "filters": [
                                {
                                    "filterType": "MARKET_LOT_SIZE",
                                    "minQty": "0.001",
                                    "maxQty": "1000",
                                    "stepSize": "0.001",
                                },
                                {
                                    "filterType": "MIN_NOTIONAL",
                                    "notional": "5",
                                },
                            ],
                        }
                    ]
                },
                {},
            )
        if path == "/fapi/v1/premiumIndex":
            return _Response(
                {
                    "symbol": params["symbol"],
                    "markPrice": "100.00",
                },
                {},
            )
        if path == "/fapi/v1/depth":
            return _Response(
                {
                    "lastUpdateId": 42,
                    "bids": [["99.99", "10"]],
                    "asks": [["100.01", "10"]],
                },
                {},
            )
        if path == "/fapi/v1/order" and method == "POST":
            if self.raise_order_timeout:
                raise requests.Timeout("secret-bearing URL omitted")
            return _Response(
                self.order_payload,
                {"X-MBX-ORDER-COUNT-10S": "1"},
                status_code=self.order_status,
            )
        if path == "/fapi/v1/order" and method == "GET":
            if params["origClientOrderId"] == "sat-r74-cal-missing":
                return _Response(
                    {"code": -2013, "msg": "Order does not exist."},
                    {},
                    status_code=400,
                )
            return _Response(
                {
                    "symbol": params["symbol"],
                    "side": "BUY",
                    "clientOrderId": params["origClientOrderId"],
                    "reduceOnly": False,
                    "type": "MARKET",
                    "positionSide": "BOTH",
                    "status": "FILLED",
                    "orderId": 1001,
                    "executedQty": "1",
                    "avgPrice": "100.02",
                },
                {},
            )
        if path == "/fapi/v1/userTrades":
            return _Response(
                [
                    {
                        "id": 2001,
                        "orderId": params["orderId"],
                        "symbol": params["symbol"],
                        "side": "BUY",
                        "buyer": True,
                        "maker": False,
                        "price": "100.02",
                        "qty": "1",
                        "quoteQty": "100.02",
                    }
                ],
                {},
            )
        raise AssertionError(f"unexpected request: {method} {path}")

    def close(self) -> None:
        self.closed = True


class _WebSocket:
    def __init__(self, messages: list[object]) -> None:
        self.messages = list(messages)
        self.closed = False

    def recv(self, timeout: float | None = None) -> str:
        assert timeout is not None and timeout > 0
        if not self.messages:
            raise TimeoutError
        return json.dumps(self.messages.pop(0), separators=(",", ":"))

    def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason
        self.closed = True


def _terminal(client_order_id: str) -> dict[str, object]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_800_000_000_001,
        "T": 1_800_000_000_000,
        "o": {
            "s": "BTCUSDT",
            "i": 1001,
            "c": client_order_id,
            "S": "BUY",
            "ps": "BOTH",
            "o": "MARKET",
            "x": "TRADE",
            "X": "FILLED",
            "m": False,
            "R": False,
            "q": "1",
            "z": "1",
            "ap": "100.02",
        },
    }


def test_transport_uses_only_official_testnet_surfaces_and_signed_calls() -> None:
    session = _Session()
    websocket = _WebSocket([{"e": "ACCOUNT_UPDATE"}, _terminal("sat-r74-cal-test-i")])
    websocket_calls: list[tuple[str, dict[str, object]]] = []

    def connect(url: str, **kwargs: object) -> _WebSocket:
        websocket_calls.append((url, kwargs))
        return websocket

    transport = Round74BinanceTestnetExecutionTransport(
        api_key="ephemeral-key",
        api_secret="ephemeral-secret",
        session_factory=lambda: session,
        websocket_factory=connect,
    )
    with transport:
        assert transport.position("BTCUSDT")["positionAmt"] == "0"
        assert transport.open_orders("BTCUSDT") == ()
        assert (
            transport.exchange_information("BTCUSDT")["symbol_payload"][
                "status"
            ]
            == "TRADING"
        )
        assert transport.mark_price("BTCUSDT")["mark_price"] == "100.00"
        assert transport.book("BTCUSDT")["update_id"] == 42
        received_ns, ack = transport.submit_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1"),
            reduce_only=False,
            client_order_id="sat-r74-cal-test-i",
        )
        assert received_ns > 0
        assert ack == {"orderId": 1001}
        terminal = transport.wait_terminal_order_update(
            symbol="BTCUSDT",
            client_order_id="sat-r74-cal-test-i",
            timeout_seconds=2.0,
        )
        assert terminal is not None
        assert terminal[1]["o"]["X"] == "FILLED"
        assert transport.query_order(
            symbol="BTCUSDT",
            client_order_id="sat-r74-cal-test-i",
        )["status"] == "FILLED"
        assert len(transport.account_trades(symbol="BTCUSDT", order_id=1001)) == 1

    assert websocket.closed is True
    assert session.closed is True
    assert websocket_calls[0][0] == (
        f"{ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL}/ws/test-listen-key"
    )
    assert all(
        str(call["url"]).startswith(ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL)
        for call in session.calls
    )
    signed_calls = [
        call
        for call in session.calls
        if urlsplit(str(call["url"])).path
        in {
            "/fapi/v2/positionRisk",
            "/fapi/v1/openOrders",
            "/fapi/v1/order",
            "/fapi/v1/userTrades",
        }
    ]
    assert signed_calls
    assert all(call["headers"]["X-MBX-APIKEY"] == "ephemeral-key" for call in signed_calls)
    assert all(len(str(call["params"]["signature"])) == 64 for call in signed_calls)
    serialized_public_state = json.dumps(
        {
            "rate_limits": transport.last_rate_limit_headers,
            "websocket_url": websocket_calls[0][0].replace(
                "test-listen-key",
                "<redacted>",
            ),
        }
    )
    assert "ephemeral-key" not in serialized_public_state
    assert "ephemeral-secret" not in serialized_public_state
    assert transport.last_rate_limit_headers == {
        "x-mbx-order-count-10s": "1",
        "x-mbx-used-weight-1m": "2",
    }


def test_transport_allows_bounded_full_exchange_information_payload() -> None:
    session = _Session()
    session.exchange_information_padding = "x" * (
        ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES + 1
    )
    transport = Round74BinanceTestnetExecutionTransport(
        api_key="key",
        api_secret="secret",
        session_factory=lambda: session,
        websocket_factory=lambda _url, **_kwargs: _WebSocket([]),
    )

    result = transport.exchange_information("BTCUSDT")

    assert result["symbol"] == "BTCUSDT"
    transport.close()


def test_transport_classifies_order_timeout_as_unknown_without_retry() -> None:
    session = _Session()
    session.raise_order_timeout = True
    websocket = _WebSocket([])
    transport = Round74BinanceTestnetExecutionTransport(
        api_key="key",
        api_secret="secret",
        session_factory=lambda: session,
        websocket_factory=lambda _url, **_kwargs: websocket,
    )
    transport.open()

    with pytest.raises(Round74OrderSubmissionUnknown):
        transport.submit_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1"),
            reduce_only=False,
            client_order_id="sat-r74-cal-timeout-i",
        )

    order_posts = [
        call
        for call in session.calls
        if call["method"] == "POST"
        and urlsplit(str(call["url"])).path == "/fapi/v1/order"
    ]
    assert len(order_posts) == 1
    transport.close()


def test_transport_classifies_deterministic_order_rejection() -> None:
    session = _Session()
    session.order_status = 400
    session.order_payload = {"code": -2019, "msg": "Margin is insufficient."}
    transport = Round74BinanceTestnetExecutionTransport(
        api_key="key",
        api_secret="secret",
        session_factory=lambda: session,
        websocket_factory=lambda _url, **_kwargs: _WebSocket([]),
    )
    transport.open()

    with pytest.raises(Round74OrderSubmissionRejected):
        transport.submit_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1"),
            reduce_only=False,
            client_order_id="sat-r74-cal-reject-i",
        )
    transport.close()


def test_transport_query_not_found_is_explicit_none() -> None:
    session = _Session()
    transport = Round74BinanceTestnetExecutionTransport(
        api_key="key",
        api_secret="secret",
        session_factory=lambda: session,
        websocket_factory=lambda _url, **_kwargs: _WebSocket([]),
    )

    assert (
        transport.query_order(
            symbol="BTCUSDT",
            client_order_id="sat-r74-cal-missing",
        )
        is None
    )
    transport.close()


def test_transport_closes_listen_key_when_websocket_open_fails() -> None:
    session = _Session()

    def fail_connect(_url: str, **_kwargs: object) -> _WebSocket:
        raise OSError("connection failed")

    transport = Round74BinanceTestnetExecutionTransport(
        api_key="key",
        api_secret="secret",
        session_factory=lambda: session,
        websocket_factory=fail_connect,
    )

    with pytest.raises(RuntimeError, match="user stream connection failed"):
        transport.open()

    listen_methods = [
        call["method"]
        for call in session.calls
        if urlsplit(str(call["url"])).path == "/fapi/v1/listenKey"
    ]
    assert listen_methods == ["POST", "DELETE"]
    transport.close()

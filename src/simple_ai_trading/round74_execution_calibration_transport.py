"""Official Binance USD-M test-environment transport for Round 74 calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
import hashlib
import hmac
import json
import math
import time
from typing import Protocol
from urllib.parse import urlencode

import requests
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as websocket_connect

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .round74_execution_calibration_coordinator import (
    Round74OrderSubmissionRejected,
    Round74OrderSubmissionUnknown,
)


ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL = "https://demo-fapi.binance.com"
ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL = "wss://demo-fstream.binance.com"
ROUND74_EXECUTION_TRANSPORT_RECV_WINDOW_MS = 2_000
ROUND74_EXECUTION_TRANSPORT_MAXIMUM_TIMEOUT_SECONDS = 20.0
ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES = 256 * 1024
ROUND74_EXECUTION_TRANSPORT_EXCHANGE_INFO_MAXIMUM_BYTES = 4 * 1024 * 1024
ROUND74_EXECUTION_TRANSPORT_CLOCK_REFRESH_SECONDS = 60.0
ROUND74_EXECUTION_TRANSPORT_BOOK_LIMIT = 100
_USER_AGENT = "simple-ai-trading-round74-testnet-calibration/1"
_ORDER_UNKNOWN_HTTP_STATUSES = frozenset({408, 500, 502, 503, 504})
_TERMINAL_ORDER_STATUSES = frozenset(
    {"CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "FILLED", "REJECTED"}
)


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, object]

    def json(self) -> object: ...


class _Session(Protocol):
    def request(self, method: str, url: str, **kwargs: object) -> _Response: ...

    def close(self) -> None: ...


class _WebSocket(Protocol):
    def recv(self, timeout: float | None = None) -> str | bytes: ...

    def close(self, code: int = 1000, reason: str = "") -> None: ...


RequestSessionFactory = Callable[[], _Session]
WebSocketFactory = Callable[..., _WebSocket]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _positive_timeout(value: object) -> float:
    selected = float(value)
    if (
        not math.isfinite(selected)
        or selected < 1.0
        or selected > ROUND74_EXECUTION_TRANSPORT_MAXIMUM_TIMEOUT_SECONDS
    ):
        raise ValueError("Round 74 execution transport timeout differs")
    return selected


def _symbol(value: object) -> str:
    selected = str(value).strip().upper()
    if selected not in ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 execution transport symbol differs")
    return selected


def _json_payload(
    response: _Response,
    *,
    maximum_response_bytes: int,
) -> object:
    content = bytes(response.content)
    if (
        not content
        or len(content) > maximum_response_bytes
    ):
        raise RuntimeError("Round 74 execution response size differs")
    try:
        value = response.json()
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError):
        raise RuntimeError("Round 74 execution response JSON differs") from None


class Round74BinanceTestnetExecutionTransport:
    """No-retry REST and authoritative user-stream adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        timeout_seconds: float = 10.0,
        session_factory: RequestSessionFactory = requests.Session,
        websocket_factory: WebSocketFactory = websocket_connect,
    ) -> None:
        key = str(api_key)
        secret = str(api_secret)
        if (
            not key
            or not secret
            or any(character.isspace() for character in key)
            or any(character.isspace() for character in secret)
        ):
            raise ValueError("Round 74 execution transport credentials differ")
        self._api_key = key
        self._api_secret = secret.encode("utf-8")
        self._timeout_seconds = _positive_timeout(timeout_seconds)
        self._session = session_factory()
        self._websocket_factory = websocket_factory
        self._websocket: _WebSocket | None = None
        self._listen_key = ""
        self._clock_offset_ms = 0
        self._clock_refreshed_monotonic = float("-inf")
        self._terminal_cache: dict[
            tuple[str, str],
            tuple[int, Mapping[str, object]],
        ] = {}
        self.last_rate_limit_headers: dict[str, str] = {}

    def __enter__(self) -> Round74BinanceTestnetExecutionTransport:
        self.open()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        self.close()

    @staticmethod
    def _headers(*, api_key: str = "") -> dict[str, str]:
        selected = {"User-Agent": _USER_AGENT}
        if api_key:
            selected["X-MBX-APIKEY"] = api_key
        return selected

    def _record_rate_limits(self, headers: Mapping[str, object]) -> None:
        selected: dict[str, str] = {}
        for key, value in headers.items():
            normalized = str(key).strip().lower()
            if (
                normalized.startswith("x-mbx-used-weight-")
                or normalized.startswith("x-mbx-order-count-")
                or normalized == "retry-after"
            ):
                selected[normalized] = str(value)
        merged = dict(self.last_rate_limit_headers)
        if "retry-after" not in selected:
            merged.pop("retry-after", None)
        merged.update(selected)
        self.last_rate_limit_headers = dict(sorted(merged.items()))

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        signed: bool = False,
        api_key_only: bool = False,
        order_submission: bool = False,
        allow_order_not_found: bool = False,
        maximum_response_bytes: int = (
            ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES
        ),
    ) -> tuple[int, object, str]:
        if (
            isinstance(maximum_response_bytes, bool)
            or not isinstance(maximum_response_bytes, int)
            or maximum_response_bytes <= 0
            or maximum_response_bytes
            > ROUND74_EXECUTION_TRANSPORT_EXCHANGE_INFO_MAXIMUM_BYTES
        ):
            raise ValueError(
                "Round 74 execution response byte limit differs"
            )
        selected_params = dict(params or {})
        if signed:
            self._ensure_fresh_clock()
            selected_params.update(
                {
                    "recvWindow": ROUND74_EXECUTION_TRANSPORT_RECV_WINDOW_MS,
                    "timestamp": int(time.time() * 1000)
                    + self._clock_offset_ms,
                }
            )
            unsigned_query = urlencode(sorted(selected_params.items()))
            signature = hmac.new(
                self._api_secret,
                unsigned_query.encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            selected_params["signature"] = signature
        url = f"{ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL}{path}"
        started_ns = time.monotonic_ns()
        try:
            response = self._session.request(
                method,
                url,
                headers=self._headers(
                    api_key=self._api_key
                    if signed or api_key_only
                    else ""
                ),
                params=selected_params,
                timeout=self._timeout_seconds,
            )
        except requests.RequestException:
            if order_submission:
                raise Round74OrderSubmissionUnknown(
                    "order submission transport outcome is unknown"
                ) from None
            raise RuntimeError("Round 74 execution request failed") from None
        received_ns = time.monotonic_ns()
        if received_ns < started_ns:
            raise RuntimeError("Round 74 execution request clock differs")
        self._record_rate_limits(response.headers)
        payload = _json_payload(
            response,
            maximum_response_bytes=maximum_response_bytes,
        )
        if response.status_code != 200:
            code = payload.get("code") if isinstance(payload, Mapping) else None
            if allow_order_not_found and code == -2013:
                return received_ns, None, _canonical_sha256(payload)
            if order_submission:
                if response.status_code in _ORDER_UNKNOWN_HTTP_STATUSES:
                    raise Round74OrderSubmissionUnknown(
                        "order submission HTTP outcome is unknown"
                    )
                raise Round74OrderSubmissionRejected(
                    "order submission was authoritatively rejected"
                )
            raise RuntimeError("Round 74 execution response status differs")
        return received_ns, payload, hashlib.sha256(bytes(response.content)).hexdigest()

    def _ensure_fresh_clock(self) -> None:
        now = time.monotonic()
        if (
            now - self._clock_refreshed_monotonic
            < ROUND74_EXECUTION_TRANSPORT_CLOCK_REFRESH_SECONDS
        ):
            return
        local_before_ms = int(time.time() * 1000)
        _received_ns, payload, _payload_sha = self._request(
            "GET",
            "/fapi/v1/time",
        )
        local_after_ms = int(time.time() * 1000)
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"serverTime"}
            or isinstance(payload["serverTime"], bool)
            or not isinstance(payload["serverTime"], int)
            or local_after_ms < local_before_ms
        ):
            raise RuntimeError("Round 74 execution clock payload differs")
        midpoint_ms = local_before_ms + (local_after_ms - local_before_ms) // 2
        self._clock_offset_ms = int(payload["serverTime"]) - midpoint_ms
        self._clock_refreshed_monotonic = now

    def open(self) -> None:
        if self._websocket is not None or self._listen_key:
            raise RuntimeError("Round 74 execution user stream is already open")
        _received_ns, payload, _payload_sha = self._request(
            "POST",
            "/fapi/v1/listenKey",
            api_key_only=True,
        )
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"listenKey"}
            or not isinstance(payload["listenKey"], str)
            or not payload["listenKey"]
            or any(character.isspace() for character in payload["listenKey"])
        ):
            raise RuntimeError("Round 74 execution listen key differs")
        self._listen_key = str(payload["listenKey"])
        try:
            self._websocket = self._websocket_factory(
                f"{ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL}/ws/"
                f"{self._listen_key}",
                open_timeout=self._timeout_seconds,
                close_timeout=self._timeout_seconds,
                ping_interval=20,
                ping_timeout=20,
                max_size=ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES,
                max_queue=64,
                user_agent_header=_USER_AGENT,
            )
        except Exception:
            try:
                self._request(
                    "DELETE",
                    "/fapi/v1/listenKey",
                    api_key_only=True,
                )
            except RuntimeError:
                pass
            self._listen_key = ""
            raise RuntimeError(
                "Round 74 execution user stream connection failed"
            ) from None

    def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass
        if self._listen_key:
            try:
                self._request(
                    "DELETE",
                    "/fapi/v1/listenKey",
                    api_key_only=True,
                )
            except RuntimeError:
                pass
            self._listen_key = ""
        self._session.close()

    def position(self, symbol: str) -> Mapping[str, object]:
        selected_symbol = _symbol(symbol)
        _received_ns, payload, _payload_sha = self._request(
            "GET",
            "/fapi/v2/positionRisk",
            params={"symbol": selected_symbol},
            signed=True,
        )
        if not isinstance(payload, list):
            raise RuntimeError("Round 74 execution position root differs")
        selected = [
            row
            for row in payload
            if isinstance(row, Mapping)
            and row.get("symbol") == selected_symbol
            and row.get("positionSide") == "BOTH"
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Round 74 execution requires Binance one-way position mode"
            )
        return dict(selected[0])

    def open_orders(self, symbol: str) -> Sequence[Mapping[str, object]]:
        selected_symbol = _symbol(symbol)
        _received_ns, payload, _payload_sha = self._request(
            "GET",
            "/fapi/v1/openOrders",
            params={"symbol": selected_symbol},
            signed=True,
        )
        if not isinstance(payload, list) or any(
            not isinstance(row, Mapping)
            or row.get("symbol") != selected_symbol
            for row in payload
        ):
            raise RuntimeError("Round 74 execution open orders differ")
        return tuple(dict(row) for row in payload)

    def exchange_information(self, symbol: str) -> Mapping[str, object]:
        selected_symbol = _symbol(symbol)
        received_ns, payload, payload_sha = self._request(
            "GET",
            "/fapi/v1/exchangeInfo",
            maximum_response_bytes=(
                ROUND74_EXECUTION_TRANSPORT_EXCHANGE_INFO_MAXIMUM_BYTES
            ),
        )
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("symbols"), list)
            or not isinstance(payload.get("rateLimits"), list)
            or any(
                not isinstance(value, Mapping)
                for value in payload["rateLimits"]
            )
        ):
            raise RuntimeError(
                "Round 74 execution exchange information differs"
            )
        selected = [
            row
            for row in payload["symbols"]
            if isinstance(row, Mapping)
            and row.get("symbol") == selected_symbol
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Round 74 execution exchange symbol differs"
            )
        return {
            "schema_version": (
                "round-074-execution-exchange-information-v1"
            ),
            "symbol": selected_symbol,
            "received_monotonic_ns": received_ns,
            "symbol_payload": dict(selected[0]),
            "rate_limits": [
                dict(value) for value in payload["rateLimits"]
            ],
            "source_payload_sha256": payload_sha,
        }

    def mark_price(self, symbol: str) -> Mapping[str, object]:
        selected_symbol = _symbol(symbol)
        received_ns, payload, payload_sha = self._request(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": selected_symbol},
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("symbol") != selected_symbol
            or not isinstance(payload.get("markPrice"), str)
        ):
            raise RuntimeError("Round 74 execution mark price differs")
        return {
            "schema_version": "round-074-execution-mark-price-v1",
            "symbol": selected_symbol,
            "received_monotonic_ns": received_ns,
            "mark_price": payload["markPrice"],
            "source_payload_sha256": payload_sha,
        }

    def book(self, symbol: str) -> Mapping[str, object]:
        selected_symbol = _symbol(symbol)
        received_ns, payload, payload_sha = self._request(
            "GET",
            "/fapi/v1/depth",
            params={
                "symbol": selected_symbol,
                "limit": ROUND74_EXECUTION_TRANSPORT_BOOK_LIMIT,
            },
        )
        if (
            not isinstance(payload, Mapping)
            or isinstance(payload.get("lastUpdateId"), bool)
            or not isinstance(payload.get("lastUpdateId"), int)
            or not isinstance(payload.get("bids"), list)
            or not isinstance(payload.get("asks"), list)
        ):
            raise RuntimeError("Round 74 execution depth payload differs")
        return {
            "schema_version": "round-074-execution-book-state-v1",
            "symbol": selected_symbol,
            "update_id": int(payload["lastUpdateId"]),
            "received_monotonic_ns": received_ns,
            "bids": payload["bids"],
            "asks": payload["asks"],
            "source_payload_sha256": payload_sha,
        }

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> tuple[int, Mapping[str, object]]:
        selected_symbol = _symbol(symbol)
        selected_side = str(side).strip().upper()
        selected_quantity = Decimal(str(quantity))
        selected_client_id = str(client_order_id).strip()
        if (
            self._websocket is None
            or selected_side not in {"BUY", "SELL"}
            or not selected_quantity.is_finite()
            or selected_quantity <= 0
            or not selected_client_id.startswith("sat-r74-cal-")
            or len(selected_client_id) > 36
        ):
            raise ValueError("Round 74 execution order arguments differ")
        params: dict[str, object] = {
            "symbol": selected_symbol,
            "side": selected_side,
            "type": "MARKET",
            "positionSide": "BOTH",
            "quantity": format(selected_quantity, "f"),
            "newClientOrderId": selected_client_id,
            "newOrderRespType": "ACK",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        received_ns, payload, _payload_sha = self._request(
            "POST",
            "/fapi/v1/order",
            params=params,
            signed=True,
            order_submission=True,
        )
        if not isinstance(payload, Mapping):
            raise Round74OrderSubmissionUnknown(
                "order submission response differs"
            )
        return received_ns, dict(payload)

    def wait_terminal_order_update(
        self,
        *,
        symbol: str,
        client_order_id: str,
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, object]] | None:
        selected_symbol = _symbol(symbol)
        selected_client_id = str(client_order_id).strip()
        cached = self._terminal_cache.pop(
            (selected_symbol, selected_client_id),
            None,
        )
        if cached is not None:
            return cached
        websocket = self._websocket
        if websocket is None:
            return None
        deadline = time.monotonic() + _positive_timeout(timeout_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                message = websocket.recv(timeout=remaining)
            except (TimeoutError, ConnectionClosed):
                return None
            received_ns = time.monotonic_ns()
            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            try:
                payload = json.loads(str(message))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, Mapping):
                continue
            order = payload.get("o")
            if (
                payload.get("e") != "ORDER_TRADE_UPDATE"
                or not isinstance(order, Mapping)
                or not isinstance(order.get("s"), str)
                or not isinstance(order.get("c"), str)
                or order.get("X") not in _TERMINAL_ORDER_STATUSES
            ):
                continue
            key = (str(order["s"]), str(order["c"]))
            normalized = json.loads(_canonical_json(dict(payload)))
            selected = (received_ns, normalized)
            if key == (selected_symbol, selected_client_id):
                return selected
            if str(order["c"]).startswith("sat-r74-cal-"):
                self._terminal_cache[key] = selected

    def query_order(
        self,
        *,
        symbol: str,
        client_order_id: str,
    ) -> Mapping[str, object] | None:
        selected_symbol = _symbol(symbol)
        _received_ns, payload, _payload_sha = self._request(
            "GET",
            "/fapi/v1/order",
            params={
                "symbol": selected_symbol,
                "origClientOrderId": str(client_order_id).strip(),
            },
            signed=True,
            allow_order_not_found=True,
        )
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise RuntimeError("Round 74 execution query order root differs")
        return dict(payload)

    def account_trades(
        self,
        *,
        symbol: str,
        order_id: int,
    ) -> Sequence[Mapping[str, object]]:
        selected_symbol = _symbol(symbol)
        if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
            raise ValueError("Round 74 execution account trade order ID differs")
        _received_ns, payload, _payload_sha = self._request(
            "GET",
            "/fapi/v1/userTrades",
            params={
                "symbol": selected_symbol,
                "orderId": order_id,
                "limit": 1000,
            },
            signed=True,
        )
        if (
            not isinstance(payload, list)
            or not payload
            or any(
                not isinstance(row, Mapping)
                or row.get("symbol") != selected_symbol
                or row.get("orderId") != order_id
                for row in payload
            )
        ):
            raise RuntimeError("Round 74 execution account trades differ")
        return tuple(dict(row) for row in payload)


__all__ = [
    "ROUND74_EXECUTION_TRANSPORT_BOOK_LIMIT",
    "ROUND74_EXECUTION_TRANSPORT_EXCHANGE_INFO_MAXIMUM_BYTES",
    "ROUND74_EXECUTION_TRANSPORT_MAXIMUM_RESPONSE_BYTES",
    "ROUND74_EXECUTION_TRANSPORT_RECV_WINDOW_MS",
    "ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL",
    "ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL",
    "Round74BinanceTestnetExecutionTransport",
]

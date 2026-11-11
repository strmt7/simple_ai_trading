"""Bounded, exact-order authenticated reads for enrolled Binance close recovery."""

from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
import math
from pathlib import Path
import re
import sqlite3
from threading import Lock
from time import monotonic, time
from urllib.parse import urlencode

from .api import BinanceAPIError
from .binance_order_responses import validate_queried_order
from .binance_close_recovery import (
    ClosingRecoveryEvidence,
    _pending,
    collect_closing_recovery,
)
from .binance_close_intents import validate_closing_client_id
from .binance_execution_scope import BinanceExecutionScope, parse_execution_id
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_runtime_control import BinanceRuntimeControl
from .positions import PositionsStore


def _http_process(**configuration):
    try:
        from .polymarket_http_process import PolymarketHttpProcess
    except ImportError:
        raise OpenIntentError(
            "Bounded recovery requires the locked polymarket-live optional environment; "
            "provision it in the application's isolated environment before collection."
        ) from None
    return PolymarketHttpProcess(**configuration)


class BinanceRecoveryReadClient:
    """Query one bound order and at most one trade page, without replay or writes.

    The shared HTTP engine owns its hidden child and hard response deadline.
    Signing stays in this process; the child receives only the signed GET.
    This is process containment, not a credential-security sandbox.
    """

    def __init__(
        self,
        *,
        scope: BinanceExecutionScope,
        symbol: str,
        client_id: str,
        api_key: str,
        api_secret: str,
        timeout_seconds: float = 20.0,
    ):
        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("recovery reads require the original execution scope")
        if (
            not isinstance(symbol, str)
            or re.fullmatch(r"[A-Z0-9]{1,32}", symbol) is None
        ):
            raise OpenIntentError("recovery symbol is invalid")
        validate_closing_client_id(client_id)
        if (
            not isinstance(api_key, str)
            or re.fullmatch(r"[\x21-\x7e]{1,256}", api_key) is None
            or not isinstance(api_secret, str)
            or not 0 < len(api_secret) <= 1024
            or api_secret != api_secret.strip()
        ):
            raise OpenIntentError("recovery requires explicitly supplied credentials")
        if (
            BinanceExecutionScope.from_api_key(scope.origin, scope.market_type, api_key)
            != scope
        ):
            raise OpenIntentError(
                "recovery credentials differ from the original key scope"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 60
        ):
            raise OpenIntentError("recovery deadline must be between 1 and 60 seconds")
        prefix = "/api/v3" if scope.market_type == "spot" else "/fapi/v1"
        self._order_endpoint = scope.origin + prefix + "/order"
        self._trade_endpoint = (
            scope.origin
            + prefix
            + ("/myTrades" if scope.market_type == "spot" else "/userTrades")
        )
        self._scope, self._symbol, self._client_id = scope, symbol, client_id
        self._key, self._secret = api_key, api_secret.encode("utf-8")
        self._deadline = monotonic() + timeout_seconds
        self._order_read = self._trades_read = self._closed = False
        self._order_id: str | None = None
        self._reads = Lock()
        self._http = _http_process(
            base_url=scope.origin,
            read_only=True,
            allowed_endpoints=(self._order_endpoint, self._trade_endpoint),
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=2 * 1024 * 1024,
        )

    def execution_scope(self) -> BinanceExecutionScope:
        return self._scope

    def _selection(self, symbol, expected_scope):
        if self._closed or symbol != self._symbol or expected_scope != self._scope:
            raise OpenIntentError(
                "recovery read selection differs from its owned order"
            )

    def _get(self, endpoint, params):
        try:
            remaining = self._deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError
            # Sign after startup; no local clock correction or retry is inferred.
            self._http.start(timeout_seconds=min(5.0, remaining))
            if monotonic() >= self._deadline:
                raise TimeoutError
            params = {
                **params,
                "recvWindow": "5000",
                "timestamp": str(int(time() * 1000)),
            }
            params["signature"] = hmac.new(
                self._secret, urlencode(params).encode("ascii"), hashlib.sha256
            ).hexdigest()
            return self._http.request(
                "GET",
                endpoint,
                headers={"X-MBX-APIKEY": self._key},
                params=params,
                deadline=self._deadline,
            )
        except Exception:
            # Neither upstream error bodies nor signed query details are diagnostics.
            self.close()
            raise BinanceAPIError(
                "Exact-order recovery read failed or expired; no order was submitted."
            ) from None

    def get_order(self, symbol, *, orig_client_order_id, expected_scope):
        with self._reads:
            self._selection(symbol, expected_scope)
            if self._order_read or orig_client_order_id != self._client_id:
                raise OpenIntentError("recovery permits one exact owned order query")
            self._order_read = True
        result = self._get(
            self._order_endpoint,
            {
                "symbol": symbol,
                "origClientOrderId": self._client_id,
            },
        )
        try:
            if not isinstance(result, dict):
                raise ValueError
            validate_queried_order(
                result, symbol=symbol, order_id=None, client_order_id=self._client_id
            )
            self._order_id = parse_execution_id(result.get("orderId"))
            if self._order_id is None:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            self.close()
            raise OpenIntentError(
                "recovery order response identity is invalid"
            ) from None
        return result

    def get_order_trades(self, symbol, *, order_id, expected_scope):
        with self._reads:
            self._selection(symbol, expected_scope)
            identity = parse_execution_id(order_id)
            if (
                self._trades_read
                or self._order_id is None
                or identity != self._order_id
            ):
                raise OpenIntentError(
                    "recovery trades require the exact observed order"
                )
            self._trades_read = True
        result = self._get(
            self._trade_endpoint,
            {
                "symbol": symbol,
                "orderId": identity,
                "limit": "1000",
            },
        )
        if not isinstance(result, list) or len(result) > 1000:
            self.close()
            raise OpenIntentError("recovery trade page is invalid or oversized")
        return result

    def close(self):
        self._closed = True
        self._key, self._secret = "", b""
        try:
            self._http.close()
        except Exception:
            raise OpenIntentError(
                "Recovery worker cleanup failed; preserve Stop and inspect owned process state."
            ) from None


def closing_collection_sha256(
    store: PositionsStore, *, scope: BinanceExecutionScope, client_id: str
) -> str:
    """Inspect the exact unresolved request in one noncreating journal snapshot."""
    validate_closing_client_id(client_id)
    try:
        with closing(store.opening_intents._connect()) as connection:
            connection.execute("BEGIN")
            pending = _pending(connection, store, scope, client_id)
            if pending is None:
                raise OpenIntentError(
                    "an exact unresolved close is required for collection"
                )
            return hashlib.sha256(pending[1].encode("utf-8")).hexdigest()
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, OpenIntentError):
            raise
        raise OpenIntentError("closing collection request is unavailable") from None


def collect_stopped_closing_recovery(
    store: PositionsStore,
    *,
    scope: BinanceExecutionScope,
    client_id: str,
    expected_request_sha256: str,
    control: BinanceRuntimeControl,
    api_key: str,
    api_secret: str,
    execution_enrollment_path: Path | None = None,
    timeout_seconds: float = 20.0,
) -> ClosingRecoveryEvidence:
    """Collect under ownerless Stop; retain uncertainty, never apply or rearm.

    Calling this API is a separate authenticated read operation. It does not
    obtain credentials, establish account authority or authorize future calls.
    """
    if (
        not isinstance(control, BinanceRuntimeControl)
        or not isinstance(expected_request_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_request_sha256) is None
    ):
        raise OpenIntentError(
            "collection requires enrolled control and an exact request digest"
        )
    storage = BinanceExecutionStorage(
        store.opening_intents, control, execution_enrollment_path
    )
    with control.stopped_recovery_guard():
        storage.verify(scope=scope)
        if (
            closing_collection_sha256(store, scope=scope, client_id=client_id)
            != expected_request_sha256
        ):
            raise OpenIntentError(
                "closing request differs from the inspected collection digest"
            )
        with closing(store.opening_intents._connect()) as connection:
            connection.execute("BEGIN")
            pending = _pending(connection, store, scope, client_id)
            if (
                pending is None
                or hashlib.sha256(pending[1].encode("utf-8")).hexdigest()
                != expected_request_sha256
            ):
                raise OpenIntentError("closing request changed before collection")
            symbol = pending[0].symbol
        with closing(
            BinanceRecoveryReadClient(
                scope=scope,
                symbol=symbol,
                client_id=client_id,
                api_key=api_key,
                api_secret=api_secret,
                timeout_seconds=timeout_seconds,
            )
        ) as client:
            evidence = collect_closing_recovery(client, store, client_id=client_id)
        storage.verify(scope=scope)
        if evidence is None:
            raise OpenIntentError(
                "closing changed during collection; inspect the original journal"
            )
        return evidence

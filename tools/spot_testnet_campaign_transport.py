"""Sequential, ownership-restricted transport and durable testnet campaign journal."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from simple_ai_trading.api import BinanceClient
from simple_ai_trading.spot_testnet_evidence import decimal

HOST = "https://testnet.binance.vision"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Journal:
    def __init__(self, path: Path, *, create: bool = False):
        self.path = path
        if create:
            with path.open("x"):
                pass
        self.rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        previous = "0" * 64
        for row in self.rows:
            content = {k: v for k, v in row.items() if k != "sha256"}
            if row["previous"] != previous or digest(content) != row["sha256"]:
                raise ValueError("journal integrity failure")
            previous = row["sha256"]

    def add(self, kind: str, **fields: Any) -> None:
        row = {
            "kind": kind,
            "utc": datetime.now(timezone.utc).isoformat(),
            "previous": self.rows[-1]["sha256"] if self.rows else "0" * 64,
            **fields,
        }
        row["sha256"] = digest(row)
        with self.path.open("a", encoding="ascii", newline="\n") as out:
            out.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.rows.append(row)

    def intents(self) -> list[dict[str, Any]]:
        return [row["params"] for row in self.rows if row["kind"] == "order_intent"]


class Transport:
    def __init__(self, key: str, secret: str, journal: Journal, prefix: str):
        self.journal, self.prefix = journal, prefix
        self.client = BinanceClient(
            key,
            secret,
            testnet=True,
            market_type="spot",
            max_retries=0,
            max_calls_per_minute=60,
            timeout=15,
        )
        if self.client.base_url != HOST:
            raise ValueError("exact Spot-testnet host required")
        self.client.session.trust_env = False
        original = self.client.session.request
        self.client.session.request = lambda *args, **kwargs: original(
            *args, allow_redirects=False, **kwargs
        )
        self.weight_limit = 0
        self.used_weight = 0

    def _check(self, method: str, path: str, params: dict[str, Any]) -> None:
        allowed = {
            ("GET", "/api/v3/time"),
            ("GET", "/api/v3/exchangeInfo"),
            ("GET", "/api/v3/ticker/bookTicker"),
            ("GET", "/api/v3/account"),
            ("GET", "/api/v3/account/commission"),
            ("GET", "/api/v3/openOrders"),
            ("GET", "/api/v3/order"),
            ("GET", "/api/v3/myTrades"),
            ("POST", "/api/v3/order"),
            ("DELETE", "/api/v3/order"),
        }
        if (method, path) not in allowed or self.client.base_url != HOST:
            raise ValueError("endpoint outside campaign")
        if "symbol" in params and params["symbol"] not in SYMBOLS:
            raise ValueError("symbol outside campaign")
        if path == "/api/v3/exchangeInfo" and params != {
            "symbols": json.dumps(SYMBOLS, separators=(",", ":"))
        }:
            raise ValueError("population differs")
        if (
            path
            in {
                "/api/v3/openOrders",
                "/api/v3/myTrades",
                "/api/v3/order",
                "/api/v3/account/commission",
                "/api/v3/ticker/bookTicker",
            }
            and params.get("symbol") not in SYMBOLS
        ):
            raise ValueError("exact symbol required")
        intents = self.journal.intents()
        if path == "/api/v3/order":
            if method == "POST":
                if not any(params == x for x in intents):
                    raise ValueError("unjournaled order")
                if not str(params.get("newClientOrderId", "")).startswith(
                    self.prefix + "-"
                ):
                    raise ValueError("foreign campaign identity")
                if any(
                    row["kind"] == "http_intent"
                    and row["method"] == "POST"
                    and row["params"].get("newClientOrderId")
                    == params["newClientOrderId"]
                    for row in self.journal.rows
                ):
                    raise ValueError("never resubmit an ambiguous or completed order")
                if (
                    params.get("side") == "BUY"
                    and self.weight_limit
                    and self.used_weight >= self.weight_limit * 0.8
                ):
                    raise ValueError("insufficient shared IP headroom for new exposure")
            else:
                cid = params.get("origClientOrderId")
                owned_ids = {
                    row["orderId"]
                    for row in self.journal.rows
                    if row["kind"] == "order_identity"
                    and row["symbol"] == params["symbol"]
                }
                cid_owned = any(
                    x["symbol"] == params["symbol"] and x["newClientOrderId"] == cid
                    for x in intents
                )
                if (
                    (cid is not None and not cid_owned)
                    or ("orderId" in params and params["orderId"] not in owned_ids)
                    or (cid is None and "orderId" not in params)
                ):
                    raise ValueError("foreign order reference")
        if path == "/api/v3/myTrades":
            if not any(
                row["kind"] == "order_identity"
                and row["symbol"] == params["symbol"]
                and row["orderId"] == params.get("orderId")
                for row in self.journal.rows
            ):
                raise ValueError("foreign trade query")

    def call(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        self._check(method, path, params)
        if sum(row["kind"] == "http_intent" for row in self.journal.rows) >= 150:
            raise ValueError("request ceiling")
        public = path in {
            "/api/v3/time",
            "/api/v3/exchangeInfo",
            "/api/v3/ticker/bookTicker",
        }
        self.journal.add(
            "http_intent", method=method, path=path, params=params, signed=not public
        )
        try:
            response = self.client._request(method, path, params, signed=not public)
        except Exception as exc:
            info = self.client.last_request_info
            self.journal.add(
                "http_failure",
                method=method,
                path=path,
                error_type=type(exc).__name__,
                http_status=info.get("status"),
                retry_after=info.get("retry_after_seconds"),
            )
            raise RuntimeError(
                "request failed; reconcile exact owned identity"
            ) from None
        info = self.client.last_request_info
        headers = info.get("rate_limit_headers", {})
        used = max(
            (
                int(v)
                for k, v in headers.items()
                if k.lower() == "x-mbx-used-weight-1m" and str(v).isdigit()
            ),
            default=0,
        )
        self.journal.add(
            "http_completed",
            method=method,
            path=path,
            http_status=info.get("status"),
            used_weight_1m=used,
        )
        self.used_weight = used
        return response

    def order_view(
        self, order: dict[str, Any], intent: dict[str, Any]
    ) -> dict[str, Any]:
        cid = intent["newClientOrderId"]
        if (
            order.get("symbol") != intent["symbol"]
            or order.get("side") != intent["side"]
        ):
            raise ValueError("order identity mismatch")
        if order.get("clientOrderId") not in {cid, cid + "c"}:
            raise ValueError("foreign client identity")
        if type(order.get("orderId")) is not int or order["orderId"] < 0:
            raise ValueError("invalid order identity")
        if order.get("status") not in {
            "NEW",
            "PARTIALLY_FILLED",
            "FILLED",
            "CANCELED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
            "REJECTED",
            "PENDING_CANCEL",
        }:
            raise ValueError("unknown order status")
        view = {
            k: order[k]
            for k in ("symbol", "side", "clientOrderId", "orderId", "status")
        }
        for key in ("origQty", "executedQty", "cummulativeQuoteQty", "price"):
            view[key] = str(decimal(order[key]))
        if decimal(view["origQty"]) != decimal(intent["quantity"]):
            raise ValueError("order quantity differs")
        self.journal.add(
            "order_identity",
            symbol=view["symbol"],
            orderId=view["orderId"],
            original_client_id=cid,
        )
        self.journal.add("order_observation", order=view)
        return view

    def trade_view(
        self, rows: list[dict[str, Any]], order: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ValueError("incomplete trade response")
        out = []
        for row in rows:
            if (
                row.get("symbol") != order["symbol"]
                or row.get("orderId") != order["orderId"]
            ):
                raise ValueError("foreign trade response")
            if not re.fullmatch(r"[A-Z0-9]{1,12}", str(row.get("commissionAsset", ""))):
                raise ValueError("invalid fee asset")
            if type(row.get("id")) is not int or type(row.get("isBuyer")) is not bool:
                raise ValueError("invalid trade identity")
            view = {
                k: row[k]
                for k in ("symbol", "orderId", "id", "isBuyer", "commissionAsset")
            }
            for field in ("qty", "quoteQty", "commission", "price"):
                view[field] = str(decimal(row[field]))
            out.append(view)
        self.journal.add("owned_trades", orderId=order["orderId"], trades=out)
        return out

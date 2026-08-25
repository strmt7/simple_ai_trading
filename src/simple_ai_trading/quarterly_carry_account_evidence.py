"""GET-only account evidence for Binance quarterly cash-and-carry research."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import math
import time
from typing import Protocol
from urllib.parse import urlencode

import requests


SCHEMA_VERSION = "binance-quarterly-carry-account-evidence-v1"
SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
SPOT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
FUTURES_SYMBOLS = (
    "BTCUSDT_260925",
    "ETHUSDT_260925",
    "BTCUSDT_261225",
    "ETHUSDT_261225",
)
RECV_WINDOW_MS = 2_000
MAXIMUM_TIMEOUT_SECONDS = 20.0
MAXIMUM_RESPONSE_BYTES = 32 * 1024
MAXIMUM_CLOCK_RTT_NS = 2_000_000_000


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, object]

    def iter_content(self, chunk_size: int = ...) -> object: ...

    def close(self) -> None: ...


RequestFunction = Callable[..., _Response]
JournalFunction = Callable[[Mapping[str, object]], None]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _decimal_rate(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{label} must be a decimal rate")
    try:
        selected = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a decimal rate") from exc
    if not selected.is_finite() or selected < 0 or selected > 1:
        raise ValueError(f"{label} is outside the allowed rate range")
    return format(selected, "f")


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _rate_limit_headers(headers: Mapping[str, object]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key).strip().lower()
        if normalized.startswith("x-mbx-used-weight-"):
            cleaned = str(value).strip()
            if not cleaned or len(cleaned) > 64:
                raise ValueError("rate-limit response header differs")
            selected[normalized.replace("-", "_")] = cleaned
    return dict(sorted(selected.items()))


def _emit(journal: JournalFunction | None, event: Mapping[str, object]) -> None:
    if journal is not None:
        journal(json.loads(canonical_json(dict(event))))


@dataclass(frozen=True)
class RequestEvidence:
    request_id: str
    venue: str
    path: str
    symbol: str | None
    request_started_wall_ns: int
    received_wall_ns: int
    request_started_monotonic_ns: int
    received_monotonic_ns: int
    payload_sha256: str
    response_bytes: int
    rate_limit_headers: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        if (
            self.venue not in {"spot", "usdm_futures"}
            or not self.request_id
            or not self.path.startswith("/")
            or self.received_wall_ns < self.request_started_wall_ns
            or self.received_monotonic_ns < self.request_started_monotonic_ns
            or len(self.payload_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.payload_sha256
            )
            or not 0 < self.response_bytes <= MAXIMUM_RESPONSE_BYTES
        ):
            raise ValueError("request evidence differs")
        return {
            "request_id": self.request_id,
            "venue": self.venue,
            "method": "GET",
            "path": self.path,
            "symbol": self.symbol,
            "request_started_wall_ns": self.request_started_wall_ns,
            "received_wall_ns": self.received_wall_ns,
            "request_started_monotonic_ns": self.request_started_monotonic_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "payload_sha256": self.payload_sha256,
            "response_bytes": self.response_bytes,
            "rate_limit_headers": dict(self.rate_limit_headers),
            "retry_count": 0,
            "signed_query_persisted": False,
            "credential_material_persisted": False,
        }


@dataclass(frozen=True)
class QuarterlyCarryAccountEvidence:
    spot_commissions: tuple[tuple[str, Mapping[str, object]], ...]
    futures_commissions: tuple[tuple[str, Mapping[str, object]], ...]
    futures_account_configuration: Mapping[str, object]
    requests: tuple[RequestEvidence, ...]

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        if tuple(symbol for symbol, _ in self.spot_commissions) != SPOT_SYMBOLS:
            raise ValueError("spot commission symbol coverage differs")
        if tuple(symbol for symbol, _ in self.futures_commissions) != FUTURES_SYMBOLS:
            raise ValueError("futures commission symbol coverage differs")
        if len(self.requests) != 2 * (len(SPOT_SYMBOLS) + len(FUTURES_SYMBOLS) + 1):
            raise ValueError("request evidence coverage differs")
        value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "environment": "binance_mainnet_read_only_user_data",
            "spot_commissions": dict(self.spot_commissions),
            "futures_commissions": dict(self.futures_commissions),
            "futures_account_configuration": dict(self.futures_account_configuration),
            "request_evidence": [item.as_dict() for item in self.requests],
            "authority": {
                "read_only_account_evidence": True,
                "orders_submitted": False,
                "orders_modified": False,
                "orders_cancelled": False,
                "transfers_requested": False,
                "live_trading_authority": False,
                "profitability_claim": False,
            },
        }
        if include_sha256:
            value["result_sha256"] = canonical_sha256(value)
        return value


def _strict_json(content: bytes) -> object:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("response contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"response contains non-finite JSON value {value}")

    try:
        return json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not strict JSON") from exc


def _request_json(
    request: RequestFunction,
    *,
    request_id: str,
    venue: str,
    base_url: str,
    path: str,
    symbol: str | None,
    headers: Mapping[str, str],
    params: Mapping[str, object],
    timeout_seconds: float,
    journal: JournalFunction | None,
) -> tuple[object, RequestEvidence]:
    identity = {
        "request_id": request_id,
        "venue": venue,
        "method": "GET",
        "path": path,
        "symbol": symbol,
    }
    _emit(journal, {"phase": "reserved_before_request", **identity})
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        response = request(
            "GET",
            f"{base_url}{path}",
            headers=dict(headers),
            params=dict(params),
            timeout=timeout_seconds,
            stream=True,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        _emit(
            journal,
            {
                "phase": "terminal_failure",
                "failure_type": type(exc).__name__,
                **identity,
            },
        )
        raise RuntimeError("account-evidence request failed without retry") from None
    content_parts: list[bytes] = []
    captured_bytes = 0
    try:
        try:
            for chunk in response.iter_content(chunk_size=4_096):
                selected = bytes(chunk)
                if not selected:
                    continue
                remaining = MAXIMUM_RESPONSE_BYTES + 1 - captured_bytes
                content_parts.append(selected[:remaining])
                captured_bytes += min(len(selected), remaining)
                if captured_bytes > MAXIMUM_RESPONSE_BYTES:
                    break
        except requests.RequestException as exc:
            _emit(
                journal,
                {
                    "phase": "terminal_failure",
                    "failure_type": type(exc).__name__,
                    **identity,
                },
            )
            raise RuntimeError(
                "account-evidence response stream failed without retry"
            ) from None
    finally:
        response.close()
    received_monotonic_ns = time.monotonic_ns()
    received_wall_ns = time.time_ns()
    content = b"".join(content_parts)
    payload_sha256 = hashlib.sha256(content).hexdigest()
    _emit(
        journal,
        {
            "phase": "response_persisted_before_validation",
            "status_code": int(response.status_code),
            "captured_body_sha256": payload_sha256,
            "captured_response_bytes": len(content),
            "response_exceeded_limit": len(content) > MAXIMUM_RESPONSE_BYTES,
            **identity,
        },
    )
    if (
        response.status_code != 200
        or not content
        or len(content) > MAXIMUM_RESPONSE_BYTES
    ):
        _emit(
            journal,
            {
                "phase": "terminal_failure",
                "failure_type": "response_contract",
                "status_code": int(response.status_code),
                **identity,
            },
        )
        raise RuntimeError("account-evidence response differs")
    try:
        payload = _strict_json(content)
    except ValueError:
        _emit(
            journal,
            {
                "phase": "terminal_failure",
                "failure_type": "strict_json",
                **identity,
            },
        )
        raise RuntimeError("account-evidence JSON differs") from None
    evidence = RequestEvidence(
        request_id=request_id,
        venue=venue,
        path=path,
        symbol=symbol,
        request_started_wall_ns=started_wall_ns,
        received_wall_ns=received_wall_ns,
        request_started_monotonic_ns=started_monotonic_ns,
        received_monotonic_ns=received_monotonic_ns,
        payload_sha256=payload_sha256,
        response_bytes=len(content),
        rate_limit_headers=tuple(_rate_limit_headers(response.headers).items()),
    )
    return payload, evidence


def _signed_get(
    request: RequestFunction,
    *,
    sequence: int,
    venue: str,
    base_url: str,
    time_path: str,
    path: str,
    symbol: str | None,
    api_key: str,
    api_secret: str,
    timeout_seconds: float,
    journal: JournalFunction | None,
) -> tuple[object, tuple[RequestEvidence, RequestEvidence]]:
    user_agent = "simple-ai-trading-quarterly-carry-research/1"
    clock_payload, clock_evidence = _request_json(
        request,
        request_id=f"{sequence:02d}-clock",
        venue=venue,
        base_url=base_url,
        path=time_path,
        symbol=symbol,
        headers={"Accept": "application/json", "User-Agent": user_agent},
        params={},
        timeout_seconds=timeout_seconds,
        journal=journal,
    )
    clock = _mapping(clock_payload, label="clock response")
    if set(clock) != {"serverTime"}:
        raise ValueError("clock response fields differ")
    server_time = _integer(clock["serverTime"], label="serverTime")
    if (
        server_time <= 0
        or clock_evidence.received_monotonic_ns
        - clock_evidence.request_started_monotonic_ns
        > MAXIMUM_CLOCK_RTT_NS
    ):
        raise ValueError("clock response timing differs")
    unsigned: dict[str, object] = {
        "recvWindow": RECV_WINDOW_MS,
        "timestamp": server_time,
    }
    if symbol is not None:
        unsigned["symbol"] = symbol
    query = urlencode(sorted(unsigned.items()))
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    payload, signed_evidence = _request_json(
        request,
        request_id=f"{sequence:02d}-signed",
        venue=venue,
        base_url=base_url,
        path=path,
        symbol=symbol,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
            "X-MBX-APIKEY": api_key,
        },
        params={**unsigned, "signature": signature},
        timeout_seconds=timeout_seconds,
        journal=journal,
    )
    return payload, (clock_evidence, signed_evidence)


def _normalize_spot_commission(payload: object, *, symbol: str) -> dict[str, object]:
    selected = _mapping(payload, label="spot commission response")
    if str(selected.get("symbol", "")).upper() != symbol:
        raise ValueError("spot commission response symbol differs")
    result: dict[str, object] = {"symbol": symbol}
    for component_name in (
        "standardCommission",
        "specialCommission",
        "taxCommission",
    ):
        component = _mapping(selected.get(component_name), label=component_name)
        result[component_name] = {
            side: _decimal_rate(component.get(side), label=f"{component_name}.{side}")
            for side in ("maker", "taker", "buyer", "seller")
        }
    discount = _mapping(selected.get("discount"), label="discount")
    result["discount"] = {
        "enabledForAccount": _boolean(
            discount.get("enabledForAccount"), label="discount.enabledForAccount"
        ),
        "enabledForSymbol": _boolean(
            discount.get("enabledForSymbol"), label="discount.enabledForSymbol"
        ),
        "discountAsset": str(discount.get("discountAsset", "")),
        "discount": _decimal_rate(discount.get("discount"), label="discount.discount"),
        "applied_in_economics": False,
        "non_application_reason": "fee_asset_balance_and_actual_payment_asset_not_proved",
    }
    if not result["discount"]["discountAsset"]:  # type: ignore[index]
        raise ValueError("discount asset differs")
    return result


def _normalize_futures_commission(payload: object, *, symbol: str) -> dict[str, object]:
    selected = _mapping(payload, label="futures commission response")
    if str(selected.get("symbol", "")).upper() != symbol:
        raise ValueError("futures commission response symbol differs")
    return {
        "symbol": symbol,
        "makerCommissionRate": _decimal_rate(
            selected.get("makerCommissionRate"), label="makerCommissionRate"
        ),
        "takerCommissionRate": _decimal_rate(
            selected.get("takerCommissionRate"), label="takerCommissionRate"
        ),
        "rpiCommissionRate": _decimal_rate(
            selected.get("rpiCommissionRate"), label="rpiCommissionRate"
        ),
    }


def _normalize_account_configuration(payload: object) -> dict[str, object]:
    selected = _mapping(payload, label="futures account configuration")
    return {
        "feeTier": _integer(selected.get("feeTier"), label="feeTier"),
        "canTrade": _boolean(selected.get("canTrade"), label="canTrade"),
        "dualSidePosition": _boolean(
            selected.get("dualSidePosition"), label="dualSidePosition"
        ),
        "multiAssetsMargin": _boolean(
            selected.get("multiAssetsMargin"), label="multiAssetsMargin"
        ),
        "tradeGroupId": _integer(selected.get("tradeGroupId"), label="tradeGroupId"),
        "balances_persisted": False,
        "positions_persisted": False,
    }


def capture_quarterly_carry_account_evidence(
    *,
    api_key: str,
    api_secret: str,
    timeout_seconds: float = 10.0,
    request: RequestFunction = requests.request,
    journal: JournalFunction | None = None,
) -> QuarterlyCarryAccountEvidence:
    """Capture exact fee components and minimal futures configuration via GET."""

    key = str(api_key)
    secret = str(api_secret)
    timeout = float(timeout_seconds)
    if (
        not key
        or not secret
        or any(character.isspace() for character in key)
        or any(character.isspace() for character in secret)
        or not math.isfinite(timeout)
        or not 1.0 <= timeout <= MAXIMUM_TIMEOUT_SECONDS
    ):
        raise ValueError("account-evidence capture arguments differ")

    spot: list[tuple[str, Mapping[str, object]]] = []
    futures: list[tuple[str, Mapping[str, object]]] = []
    evidence: list[RequestEvidence] = []
    sequence = 0
    for symbol in SPOT_SYMBOLS:
        sequence += 1
        payload, request_evidence = _signed_get(
            request,
            sequence=sequence,
            venue="spot",
            base_url=SPOT_BASE_URL,
            time_path="/api/v3/time",
            path="/api/v3/account/commission",
            symbol=symbol,
            api_key=key,
            api_secret=secret,
            timeout_seconds=timeout,
            journal=journal,
        )
        evidence.extend(request_evidence)
        spot.append((symbol, _normalize_spot_commission(payload, symbol=symbol)))

    for symbol in FUTURES_SYMBOLS:
        sequence += 1
        payload, request_evidence = _signed_get(
            request,
            sequence=sequence,
            venue="usdm_futures",
            base_url=FUTURES_BASE_URL,
            time_path="/fapi/v1/time",
            path="/fapi/v1/commissionRate",
            symbol=symbol,
            api_key=key,
            api_secret=secret,
            timeout_seconds=timeout,
            journal=journal,
        )
        evidence.extend(request_evidence)
        futures.append((symbol, _normalize_futures_commission(payload, symbol=symbol)))

    sequence += 1
    payload, request_evidence = _signed_get(
        request,
        sequence=sequence,
        venue="usdm_futures",
        base_url=FUTURES_BASE_URL,
        time_path="/fapi/v1/time",
        path="/fapi/v1/accountConfig",
        symbol=None,
        api_key=key,
        api_secret=secret,
        timeout_seconds=timeout,
        journal=journal,
    )
    evidence.extend(request_evidence)
    result = QuarterlyCarryAccountEvidence(
        spot_commissions=tuple(spot),
        futures_commissions=tuple(futures),
        futures_account_configuration=_normalize_account_configuration(payload),
        requests=tuple(evidence),
    )
    result.as_dict()
    return result


__all__ = [
    "FUTURES_SYMBOLS",
    "SCHEMA_VERSION",
    "SPOT_SYMBOLS",
    "QuarterlyCarryAccountEvidence",
    "RequestEvidence",
    "canonical_sha256",
    "capture_quarterly_carry_account_evidence",
]

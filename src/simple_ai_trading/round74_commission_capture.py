"""Read-only mainnet commission capture for Round 74 target evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import time
from typing import Protocol
from urllib.parse import urlencode

import requests

from .impact_absorption_event_evidence import (
    Round74CommissionEvidenceBundle,
    build_round74_commission_evidence,
)
from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS


ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION = "round-074-commission-capture-v1"
ROUND74_COMMISSION_CAPTURE_BASE_URL = "https://fapi.binance.com"
ROUND74_COMMISSION_CAPTURE_ENVIRONMENT = "binance_usdm_mainnet"
ROUND74_COMMISSION_CAPTURE_RECV_WINDOW_MS = 2_000
ROUND74_COMMISSION_CAPTURE_MAXIMUM_TIMEOUT_SECONDS = 20.0
ROUND74_COMMISSION_CAPTURE_MAXIMUM_RESPONSE_BYTES = 16 * 1024
ROUND74_COMMISSION_CAPTURE_MAXIMUM_CLOCK_RTT_NS = 2_000_000_000


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, object]
    content: bytes

    def json(self) -> object: ...


RequestFunction = Callable[..., _Response]


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _rate_limit_headers(headers: Mapping[str, object]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key).strip().lower()
        if normalized.startswith("x-mbx-used-weight-"):
            selected[normalized.replace("-", "_")] = str(value)
    return dict(sorted(selected.items()))


@dataclass(frozen=True)
class Round74CommissionRequestEvidence:
    symbol: str
    clock_request_started_wall_ns: int
    clock_received_wall_ns: int
    clock_request_started_monotonic_ns: int
    clock_received_monotonic_ns: int
    exchange_time_ms: int
    clock_payload_sha256: str
    commission_request_started_wall_ns: int
    commission_received_wall_ns: int
    commission_request_started_monotonic_ns: int
    commission_received_monotonic_ns: int
    commission_payload_sha256: str
    response_bytes: int
    rate_limit_headers: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        wall_times = (
            self.clock_request_started_wall_ns,
            self.clock_received_wall_ns,
            self.commission_request_started_wall_ns,
            self.commission_received_wall_ns,
        )
        monotonic_times = (
            self.clock_request_started_monotonic_ns,
            self.clock_received_monotonic_ns,
            self.commission_request_started_monotonic_ns,
            self.commission_received_monotonic_ns,
        )
        if (
            self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (*wall_times, *monotonic_times)
            )
            or self.clock_received_wall_ns < self.clock_request_started_wall_ns
            or self.commission_received_wall_ns
            < self.commission_request_started_wall_ns
            or self.clock_received_monotonic_ns
            < self.clock_request_started_monotonic_ns
            or self.commission_received_monotonic_ns
            < self.commission_request_started_monotonic_ns
            or self.clock_received_monotonic_ns
            - self.clock_request_started_monotonic_ns
            > ROUND74_COMMISSION_CAPTURE_MAXIMUM_CLOCK_RTT_NS
            or self.commission_request_started_monotonic_ns
            < self.clock_received_monotonic_ns
            or isinstance(self.exchange_time_ms, bool)
            or not isinstance(self.exchange_time_ms, int)
            or self.exchange_time_ms <= 0
            or not _is_sha256(self.clock_payload_sha256)
            or not _is_sha256(self.commission_payload_sha256)
            or isinstance(self.response_bytes, bool)
            or not isinstance(self.response_bytes, int)
            or not 0
            < self.response_bytes
            <= ROUND74_COMMISSION_CAPTURE_MAXIMUM_RESPONSE_BYTES
            or len(dict(self.rate_limit_headers)) != len(self.rate_limit_headers)
            or any(
                not key.startswith("x_mbx_used_weight_") or not value
                for key, value in self.rate_limit_headers
            )
        ):
            raise ValueError("Round 74 commission request evidence differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "symbol": self.symbol,
            "clock": {
                "path": "/fapi/v1/time",
                "request_started_wall_ns": self.clock_request_started_wall_ns,
                "received_wall_ns": self.clock_received_wall_ns,
                "request_started_monotonic_ns": (
                    self.clock_request_started_monotonic_ns
                ),
                "received_monotonic_ns": self.clock_received_monotonic_ns,
                "exchange_time_ms": self.exchange_time_ms,
                "payload_sha256": self.clock_payload_sha256,
            },
            "commission": {
                "method": "GET",
                "path": "/fapi/v1/commissionRate",
                "security_type": "USER_DATA_SIGNED",
                "request_weight": 20,
                "request_started_wall_ns": (self.commission_request_started_wall_ns),
                "received_wall_ns": self.commission_received_wall_ns,
                "request_started_monotonic_ns": (
                    self.commission_request_started_monotonic_ns
                ),
                "received_monotonic_ns": (self.commission_received_monotonic_ns),
                "payload_sha256": self.commission_payload_sha256,
                "response_bytes": self.response_bytes,
                "rate_limit_headers": dict(self.rate_limit_headers),
                "retry_count": 0,
                "signed_query_persisted": False,
                "credential_material_persisted": False,
            },
        }


@dataclass(frozen=True)
class Round74CommissionCapture:
    payload_by_symbol: tuple[tuple[str, Mapping[str, object]], ...]
    request_evidence: tuple[Round74CommissionRequestEvidence, ...]
    bundle: Round74CommissionEvidenceBundle
    schema_version: str = ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION

    def validate(self) -> None:
        payloads = dict(self.payload_by_symbol)
        if (
            self.schema_version != ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION
            or tuple(payloads) != ROUND74_EVENT_TARGET_SYMBOLS
            or tuple(item.symbol for item in self.request_evidence)
            != ROUND74_EVENT_TARGET_SYMBOLS
            or self.bundle.as_mapping()
            != build_round74_commission_evidence(
                payload_by_symbol=payloads,
                environment=ROUND74_COMMISSION_CAPTURE_ENVIRONMENT,
                observed_wall_ns=max(
                    item.commission_received_wall_ns for item in self.request_evidence
                ),
            ).as_mapping()
        ):
            raise ValueError("Round 74 commission capture differs")
        for item in self.request_evidence:
            item.validate()
        rebuilt = build_round74_commission_evidence(
            payload_by_symbol=payloads,
            environment=ROUND74_COMMISSION_CAPTURE_ENVIRONMENT,
            observed_wall_ns=max(
                item.commission_received_wall_ns for item in self.request_evidence
            ),
        )
        if rebuilt.evidence.as_dict() != self.bundle.evidence.as_dict():
            raise ValueError("Round 74 commission capture evidence differs")

    @property
    def capture_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "environment": ROUND74_COMMISSION_CAPTURE_ENVIRONMENT,
            "base_url": ROUND74_COMMISSION_CAPTURE_BASE_URL,
            "payload_by_symbol": dict(self.payload_by_symbol),
            "request_evidence": [item.as_dict() for item in self.request_evidence],
            "taker_fee_bps_by_symbol": self.bundle.as_mapping(),
            "target_evidence": self.bundle.evidence.as_dict(),
            "authority": {
                "orders_submitted": False,
                "model_training": False,
                "financial_edge_tested": False,
                "profitability_claim": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        if include_sha256:
            value["capture_sha256"] = _canonical_sha256(value)
        return value


def _request_json(
    request: RequestFunction,
    *,
    path: str,
    headers: Mapping[str, str],
    params: Mapping[str, object],
    timeout_seconds: float,
) -> tuple[dict[str, object], int, int, int, int, int, dict[str, str]]:
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        response = request(
            "GET",
            f"{ROUND74_COMMISSION_CAPTURE_BASE_URL}{path}",
            headers=dict(headers),
            params=dict(params),
            timeout=timeout_seconds,
        )
    except requests.RequestException:
        raise RuntimeError("Round 74 commission evidence request failed") from None
    received_monotonic_ns = time.monotonic_ns()
    received_wall_ns = time.time_ns()
    content = bytes(response.content)
    if (
        response.status_code != 200
        or not content
        or len(content) > ROUND74_COMMISSION_CAPTURE_MAXIMUM_RESPONSE_BYTES
    ):
        raise RuntimeError("Round 74 commission evidence response differs")
    try:
        payload = response.json()
    except (TypeError, ValueError):
        raise RuntimeError("Round 74 commission evidence JSON differs") from None
    if not isinstance(payload, Mapping):
        raise RuntimeError("Round 74 commission evidence root differs")
    normalized = json.loads(_canonical_json(dict(payload)))
    return (
        normalized,
        started_wall_ns,
        received_wall_ns,
        started_monotonic_ns,
        received_monotonic_ns,
        len(content),
        _rate_limit_headers(response.headers),
    )


def capture_round74_mainnet_commission(
    *,
    api_key: str,
    api_secret: str,
    timeout_seconds: float = 10.0,
    request: RequestFunction = requests.request,
) -> Round74CommissionCapture:
    """Capture all three signed fee responses through a GET-only surface."""

    key = str(api_key)
    secret = str(api_secret)
    timeout = float(timeout_seconds)
    if (
        not key
        or not secret
        or any(character.isspace() for character in key)
        or any(character.isspace() for character in secret)
        or not math.isfinite(timeout)
        or not 1.0 <= timeout <= ROUND74_COMMISSION_CAPTURE_MAXIMUM_TIMEOUT_SECONDS
    ):
        raise ValueError("Round 74 commission capture arguments differ")
    headers = {
        "X-MBX-APIKEY": key,
        "User-Agent": "simple-ai-trading-round74-research/1",
    }
    payloads: list[tuple[str, Mapping[str, object]]] = []
    evidence: list[Round74CommissionRequestEvidence] = []
    for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
        (
            clock_payload,
            clock_started_wall_ns,
            clock_received_wall_ns,
            clock_started_monotonic_ns,
            clock_received_monotonic_ns,
            _clock_bytes,
            _clock_headers,
        ) = _request_json(
            request,
            path="/fapi/v1/time",
            headers={"User-Agent": headers["User-Agent"]},
            params={},
            timeout_seconds=timeout,
        )
        if set(clock_payload) != {"serverTime"}:
            raise RuntimeError("Round 74 commission clock payload differs")
        exchange_time = clock_payload["serverTime"]
        if isinstance(exchange_time, bool) or not isinstance(exchange_time, int):
            raise RuntimeError("Round 74 commission clock value differs")
        unsigned = {
            "recvWindow": ROUND74_COMMISSION_CAPTURE_RECV_WINDOW_MS,
            "symbol": symbol,
            "timestamp": exchange_time,
        }
        query = urlencode(sorted(unsigned.items()))
        signature = hmac.new(
            secret.encode("utf-8"),
            query.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        (
            payload,
            commission_started_wall_ns,
            commission_received_wall_ns,
            commission_started_monotonic_ns,
            commission_received_monotonic_ns,
            response_bytes,
            rate_headers,
        ) = _request_json(
            request,
            path="/fapi/v1/commissionRate",
            headers=headers,
            params={**unsigned, "signature": signature},
            timeout_seconds=timeout,
        )
        payloads.append((symbol, payload))
        evidence.append(
            Round74CommissionRequestEvidence(
                symbol=symbol,
                clock_request_started_wall_ns=clock_started_wall_ns,
                clock_received_wall_ns=clock_received_wall_ns,
                clock_request_started_monotonic_ns=clock_started_monotonic_ns,
                clock_received_monotonic_ns=clock_received_monotonic_ns,
                exchange_time_ms=exchange_time,
                clock_payload_sha256=_canonical_sha256(clock_payload),
                commission_request_started_wall_ns=commission_started_wall_ns,
                commission_received_wall_ns=commission_received_wall_ns,
                commission_request_started_monotonic_ns=(
                    commission_started_monotonic_ns
                ),
                commission_received_monotonic_ns=(commission_received_monotonic_ns),
                commission_payload_sha256=_canonical_sha256(payload),
                response_bytes=response_bytes,
                rate_limit_headers=tuple(rate_headers.items()),
            )
        )
    observed_wall_ns = max(item.commission_received_wall_ns for item in evidence)
    result = Round74CommissionCapture(
        payload_by_symbol=tuple(payloads),
        request_evidence=tuple(evidence),
        bundle=build_round74_commission_evidence(
            payload_by_symbol=dict(payloads),
            environment=ROUND74_COMMISSION_CAPTURE_ENVIRONMENT,
            observed_wall_ns=observed_wall_ns,
        ),
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_COMMISSION_CAPTURE_BASE_URL",
    "ROUND74_COMMISSION_CAPTURE_ENVIRONMENT",
    "ROUND74_COMMISSION_CAPTURE_SCHEMA_VERSION",
    "Round74CommissionCapture",
    "Round74CommissionRequestEvidence",
    "capture_round74_mainnet_commission",
]

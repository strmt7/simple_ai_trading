"""Bounded, target-blind Round 22 Polymarket historical-L2 ingestion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import re
import time
from typing import Protocol
from urllib.parse import urlparse

import requests

from .polymarket_historical_l2 import (
    HistoricalL2Window,
    PolymarketHistoricalL2Client,
)
from .polymarket_historical_screen import (
    HistoricalBtcMarket,
    parse_historical_btc_event,
)
from .polymarket_round22_pilot import (
    Round22ExpectedCondition,
    Round22PilotContract,
    Round22PilotStore,
    development_conditions,
    validate_round22_market_identity,
)


POLYMARKET_GAMMA_EVENT_BY_SLUG_URL = "https://gamma-api.polymarket.com/events/slug"
POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN = 48

_SLUG = re.compile(r"^btc-updown-5m-[0-9]{10}$")
_RETRYABLE_STATUS_CODES = frozenset((425, 429, 500, 502, 503, 504))
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "poly_address",
        "poly_signature",
        "poly_timestamp",
        "poly_nonce",
        "poly_api_key",
        "poly_passphrase",
    }
)
_EVENT_IDENTITY_FIELDS = (
    "id",
    "ticker",
    "slug",
    "closed",
)
_MARKET_IDENTITY_FIELDS = (
    "id",
    "conditionId",
    "slug",
    "question",
    "eventStartTime",
    "endDate",
    "active",
    "closed",
    "enableOrderBook",
    "acceptingOrders",
    "outcomes",
    "clobTokenIds",
    "resolutionSource",
    "orderPriceMinTickSize",
    "orderMinSize",
    "feesEnabled",
    "feeSchedule",
)
_MAXIMUM_RESPONSE_BYTES = 8 * 1024 * 1024
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    url: str


class _Session(Protocol):
    headers: Mapping[str, str]
    cookies: object

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> _Response: ...


class _HistoricalL2Source(Protocol):
    def fetch_closed_window(
        self,
        *,
        condition_id: str,
        asset_id: str,
        event_start_ms: int,
        event_end_ms: int,
        limit: int = 1_000,
    ) -> HistoricalL2Window: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 22 Gamma JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 Gamma JSON contains {value}")


def sanitize_round22_identity_event(value: object) -> dict[str, object]:
    """Drop resolution and price fields before the feature boundary is crossed."""

    if not isinstance(value, Mapping):
        raise ValueError("Round 22 Gamma event is not an object")
    event = dict(value)
    markets = event.get("markets")
    series = event.get("series")
    if (
        not isinstance(markets, list)
        or len(markets) != 1
        or not isinstance(markets[0], Mapping)
        or not isinstance(series, list)
    ):
        raise ValueError("Round 22 Gamma identity shape differs")
    sanitized_series: list[dict[str, object]] = []
    for item in series:
        if not isinstance(item, Mapping):
            raise ValueError("Round 22 Gamma series identity differs")
        sanitized_series.append({"id": item.get("id")})
    sanitized = {key: event.get(key) for key in _EVENT_IDENTITY_FIELDS}
    sanitized["series"] = sanitized_series
    sanitized["markets"] = [
        {key: markets[0].get(key) for key in _MARKET_IDENTITY_FIELDS}
    ]
    return sanitized


class Round22GammaIdentityClient:
    """Credential-free exact-slug identity client with target fields discarded."""

    def __init__(
        self,
        *,
        session: _Session | None = None,
        timeout_seconds: float = 20.0,
        minimum_request_interval_seconds: float = 0.2,
        maximum_attempts: int = 4,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = max(2.0, min(60.0, float(timeout_seconds)))
        self.minimum_request_interval_seconds = max(
            0.0,
            min(5.0, float(minimum_request_interval_seconds)),
        )
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 8:
            raise ValueError("Round 22 Gamma maximum attempts are outside the bound")
        self.maximum_attempts = maximum_attempts
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.monotonic = monotonic or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._last_request_at: float | None = None

    def _assert_public_session(self) -> None:
        headers = {str(name).strip().lower() for name in self.session.headers}
        if headers & _SENSITIVE_HEADERS:
            raise ValueError("Round 22 Gamma session contains authority headers")
        cookies = getattr(self.session, "cookies", None)
        if cookies is not None and len(cookies):
            raise ValueError("Round 22 Gamma session contains cookies")

    def _discard_response_cookies(self) -> None:
        cookies = getattr(self.session, "cookies", None)
        if cookies is None or not len(cookies):
            return
        clear = getattr(cookies, "clear", None)
        if not callable(clear):
            raise ValueError("Round 22 Gamma session retained cookies")
        clear()
        if len(cookies):
            raise ValueError("Round 22 Gamma session retained cookies")

    def _wait_for_rate_limit(self) -> None:
        now = float(self.monotonic())
        if self._last_request_at is not None:
            delay = self.minimum_request_interval_seconds - (
                now - self._last_request_at
            )
            if delay > 0:
                self.sleeper(delay)
                now = float(self.monotonic())
        self._last_request_at = now

    @staticmethod
    def _retry_after_seconds(response: _Response, attempt: int) -> float:
        fallback = min(8.0, 0.5 * (2**attempt))
        value = str(response.headers.get("Retry-After") or "").strip()
        if not value.isdigit():
            return fallback
        return max(fallback, min(30.0, float(value)))

    def _request(self, expected: Round22ExpectedCondition) -> object:
        self._assert_public_session()
        url = f"{POLYMARKET_GAMMA_EVENT_BY_SLUG_URL}/{expected.slug}"
        for attempt in range(self.maximum_attempts):
            self._wait_for_rate_limit()
            response = self.session.get(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "simple-ai-trading-round22-identity/0.1",
                },
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
            self._discard_response_cookies()
            status = int(response.status_code)
            if status == 200:
                parsed_url = urlparse(str(response.url))
                if (
                    parsed_url.scheme != "https"
                    or parsed_url.netloc.lower() != "gamma-api.polymarket.com"
                    or parsed_url.path != f"/events/slug/{expected.slug}"
                ):
                    raise ValueError("Round 22 Gamma response changed origin")
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if not content_type.startswith("application/json"):
                    raise ValueError("Round 22 Gamma response is not JSON")
                content = bytes(response.content)
                if not 2 <= len(content) <= _MAXIMUM_RESPONSE_BYTES:
                    raise ValueError(
                        "Round 22 Gamma response size is outside the bound"
                    )
                try:
                    return json.loads(
                        content.decode("utf-8"),
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_nonfinite,
                    )
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "Round 22 Gamma response is not strict JSON"
                    ) from exc
            if (
                status not in _RETRYABLE_STATUS_CODES
                or attempt + 1 == self.maximum_attempts
            ):
                raise ValueError(f"Round 22 Gamma request failed with HTTP {status}")
            self.sleeper(self._retry_after_seconds(response, attempt))
        raise AssertionError("unreachable Round 22 Gamma retry state")

    def fetch_identity(
        self,
        expected: Round22ExpectedCondition,
        *,
        contract: Round22PilotContract,
    ) -> HistoricalBtcMarket:
        if (
            _SLUG.fullmatch(expected.slug) is None
            or expected not in contract.conditions
        ):
            raise ValueError("Round 22 Gamma expected identity differs")
        sanitized = sanitize_round22_identity_event(self._request(expected))
        if sanitized.get("slug") != expected.slug:
            raise ValueError("Round 22 Gamma event slug differs")
        market = parse_historical_btc_event(
            sanitized,
            contract=contract.identity_parser_contract(),
            observed_at_ms=int(self.clock_ms()),
        )
        validate_round22_market_identity(market, contract=contract)
        return market


@dataclass(frozen=True, slots=True)
class Round22IngestionResult:
    requested_limit: int
    selection_role: str
    already_complete_count: int
    committed_count: int
    remaining_development_count: int
    committed_slugs: tuple[str, ...]
    target_accessed: bool = False
    binance_used: bool = False
    authentication_used: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False


def ingest_round22_development_conditions(
    *,
    store: Round22PilotStore,
    contract: Round22PilotContract,
    identity_client: Round22GammaIdentityClient,
    l2_client: _HistoricalL2Source | PolymarketHistoricalL2Client,
    maximum_conditions: int = 1,
    role: str | None = None,
    progress: ProgressCallback | None = None,
) -> Round22IngestionResult:
    """Ingest a bounded development batch; sealed identities remain unreachable."""

    if store.contract != contract:
        raise ValueError("Round 22 ingestion store contract differs")
    if (
        type(maximum_conditions) is not int
        or not 1 <= maximum_conditions <= POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN
    ):
        raise ValueError("Round 22 ingestion condition limit is outside the bound")
    selection_role = str(role or "all_development").strip().lower()
    if selection_role not in {
        "all_development",
        "train",
        "tune_calibration",
        "tune_selection",
    }:
        raise ValueError("Round 22 ingestion role is invalid")
    development = tuple(
        condition
        for condition in development_conditions(contract)
        if selection_role == "all_development" or condition.role == selection_role
    )
    completed = store.completed_slugs()
    pending = tuple(item for item in development if item.slug not in completed)
    selected = pending[:maximum_conditions]
    committed: list[str] = []
    for index, expected in enumerate(selected, start=1):
        if progress is not None:
            progress(
                "identity_fetch",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "role": expected.role,
                    "slug": expected.slug,
                },
            )
        market = identity_client.fetch_identity(expected, contract=contract)
        if progress is not None:
            progress("up_book_fetch", {"slug": expected.slug})
        up_window = l2_client.fetch_closed_window(
            condition_id=market.condition_id,
            asset_id=market.up_token_id,
            event_start_ms=market.event_start_ms,
            event_end_ms=market.end_ms,
        )
        if progress is not None:
            progress("down_book_fetch", {"slug": expected.slug})
        down_window = l2_client.fetch_closed_window(
            condition_id=market.condition_id,
            asset_id=market.down_token_id,
            event_start_ms=market.event_start_ms,
            event_end_ms=market.end_ms,
        )
        if not store.put_condition(
            market=market,
            up_window=up_window,
            down_window=down_window,
        ):
            raise ValueError("Round 22 pending condition became non-atomic")
        committed.append(expected.slug)
        if progress is not None:
            progress(
                "condition_committed",
                {
                    "batch_index": index,
                    "batch_size": len(selected),
                    "slug": expected.slug,
                },
            )
    remaining = len(pending) - len(committed)
    return Round22IngestionResult(
        requested_limit=maximum_conditions,
        selection_role=selection_role,
        already_complete_count=len(completed & {item.slug for item in development}),
        committed_count=len(committed),
        remaining_development_count=remaining,
        committed_slugs=tuple(committed),
    )


__all__ = [
    "POLYMARKET_GAMMA_EVENT_BY_SLUG_URL",
    "POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN",
    "Round22GammaIdentityClient",
    "Round22IngestionResult",
    "ingest_round22_development_conditions",
    "sanitize_round22_identity_event",
]

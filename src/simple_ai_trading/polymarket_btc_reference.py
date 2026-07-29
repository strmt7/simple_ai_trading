"""Causal BTC reference data for Polymarket five-minute research.

The opening and closing reference comes from the first-party endpoint used by
Polymarket's own market page. That endpoint is not a documented public API, so
every response is treated as a drift-sensitive observation and fails closed.
Chainlink RTDS is the only live price source accepted for the settlement target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


POLYMARKET_CRYPTO_REFERENCE_URL = (
    "https://polymarket.com/api/crypto/crypto-price"
)
POLYMARKET_BTC_REFERENCE_SCHEMA_VERSION = (
    "polymarket-btc-five-minute-reference-v1"
)
POLYMARKET_CHAINLINK_TICK_SCHEMA_VERSION = "polymarket-chainlink-btc-tick-v1"
_BTC_WINDOW_MS = 300_000
_MAX_RESPONSE_BYTES = 64 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket reference JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket reference JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _positive_time_ms(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive Unix timestamp")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive Unix timestamp")
    return parsed if parsed >= 10_000_000_000 else parsed * 1_000


def _utc_iso_seconds(timestamp_ms: int) -> str:
    if timestamp_ms % 1_000:
        raise ValueError("Polymarket reference window must align to a whole second")
    return (
        datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class PolymarketBtcReferenceWindow:
    event_start_ms: int
    end_ms: int
    open_price: Decimal
    close_price: Decimal | None
    observed_at_ms: int
    completed: bool
    incomplete: bool
    cached: bool
    source_payload_sha256: str

    def __post_init__(self) -> None:
        start = int(self.event_start_ms)
        end = int(self.end_ms)
        if start <= 0 or end - start != _BTC_WINDOW_MS or start % _BTC_WINDOW_MS:
            raise ValueError("Polymarket BTC reference window is not an exact five minutes")
        object.__setattr__(self, "event_start_ms", start)
        object.__setattr__(self, "end_ms", end)
        object.__setattr__(
            self,
            "open_price",
            _positive_decimal(self.open_price, name="open_price"),
        )
        if self.close_price is not None:
            object.__setattr__(
                self,
                "close_price",
                _positive_decimal(self.close_price, name="close_price"),
            )
        observed = _positive_time_ms(self.observed_at_ms, name="observed_at_ms")
        object.__setattr__(self, "observed_at_ms", observed)
        if type(self.completed) is not bool or type(self.incomplete) is not bool:
            raise ValueError("Polymarket BTC reference completion flags are invalid")
        if type(self.cached) is not bool:
            raise ValueError("Polymarket BTC reference cache flag is invalid")
        if self.completed == self.incomplete:
            raise ValueError("Polymarket BTC reference completion flags contradict")
        if self.completed != (self.close_price is not None):
            raise ValueError("Polymarket BTC reference close state contradicts")
        digest = str(self.source_payload_sha256 or "").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Polymarket BTC reference payload digest is invalid")
        object.__setattr__(self, "source_payload_sha256", digest)

    @property
    def winning_outcome(self) -> str | None:
        if self.close_price is None:
            return None
        return "Up" if self.close_price >= self.open_price else "Down"

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_BTC_REFERENCE_SCHEMA_VERSION,
            "event_start_ms": self.event_start_ms,
            "end_ms": self.end_ms,
            "open_price": format(self.open_price, "f"),
            "close_price": (
                None if self.close_price is None else format(self.close_price, "f")
            ),
            "observed_at_ms": self.observed_at_ms,
            "completed": self.completed,
            "incomplete": self.incomplete,
            "cached": self.cached,
            "winning_outcome": self.winning_outcome,
            "source_payload_sha256": self.source_payload_sha256,
        }


def parse_polymarket_btc_reference_window(
    payload: Mapping[str, object],
    *,
    event_start_ms: int,
    end_ms: int,
) -> PolymarketBtcReferenceWindow:
    raw = dict(payload)
    required = {
        "openPrice",
        "closePrice",
        "timestamp",
        "completed",
        "incomplete",
        "cached",
    }
    if set(raw) != required:
        raise ValueError("Polymarket BTC reference response schema drifted")
    completed = raw["completed"]
    incomplete = raw["incomplete"]
    cached = raw["cached"]
    if any(type(value) is not bool for value in (completed, incomplete, cached)):
        raise ValueError("Polymarket BTC reference flags are invalid")
    close_raw = raw["closePrice"]
    canonical = _canonical_json(raw)
    return PolymarketBtcReferenceWindow(
        event_start_ms=event_start_ms,
        end_ms=end_ms,
        open_price=_positive_decimal(raw["openPrice"], name="openPrice"),
        close_price=(
            None
            if close_raw is None
            else _positive_decimal(close_raw, name="closePrice")
        ),
        observed_at_ms=_positive_time_ms(raw["timestamp"], name="timestamp"),
        completed=completed,
        incomplete=incomplete,
        cached=cached,
        source_payload_sha256=hashlib.sha256(
            canonical.encode("ascii")
        ).hexdigest(),
    )


def _public_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "simple-ai-trading/0.1.0-beta.1 polymarket-reference"}
    )
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2),
    )
    return session


class OfficialPolymarketBtcReferenceClient:
    """Read the exact first-party BTC window reference without credentials."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.session = session or _public_session()
        self.timeout_seconds = max(1.0, min(30.0, float(timeout_seconds)))

    def window(
        self,
        *,
        event_start_ms: int,
        end_ms: int,
    ) -> PolymarketBtcReferenceWindow:
        start = int(event_start_ms)
        end = int(end_ms)
        if end - start != _BTC_WINDOW_MS or start % _BTC_WINDOW_MS:
            raise ValueError("Polymarket BTC reference request is not five-minute aligned")
        response = self.session.get(
            POLYMARKET_CRYPTO_REFERENCE_URL,
            params={
                "symbol": "BTC",
                "eventStartTime": _utc_iso_seconds(start),
                "variant": "fiveminute",
                "endDate": _utc_iso_seconds(end),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ValueError("Polymarket BTC reference response exceeded the size bound")
        try:
            decoded = response.content.decode("utf-8", errors="strict")
            payload = json.loads(
                decoded,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Polymarket BTC reference response is not strict JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket BTC reference response is not an object")
        return parse_polymarket_btc_reference_window(
            payload,
            event_start_ms=start,
            end_ms=end,
        )


@dataclass(frozen=True, slots=True)
class PolymarketChainlinkBtcTick:
    source_time_ms: int
    publisher_time_ms: int
    received_at_ms: int
    price: Decimal
    source_payload_sha256: str

    def __post_init__(self) -> None:
        source = _positive_time_ms(self.source_time_ms, name="source_time_ms")
        publisher = _positive_time_ms(
            self.publisher_time_ms,
            name="publisher_time_ms",
        )
        received = _positive_time_ms(self.received_at_ms, name="received_at_ms")
        if source > publisher + 250 or publisher > received + 5_000:
            raise ValueError("Chainlink RTDS tick chronology is invalid")
        object.__setattr__(self, "source_time_ms", source)
        object.__setattr__(self, "publisher_time_ms", publisher)
        object.__setattr__(self, "received_at_ms", received)
        object.__setattr__(
            self,
            "price",
            _positive_decimal(self.price, name="Chainlink BTC price"),
        )
        digest = str(self.source_payload_sha256 or "").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Chainlink RTDS payload digest is invalid")
        object.__setattr__(self, "source_payload_sha256", digest)

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_CHAINLINK_TICK_SCHEMA_VERSION,
            "source_time_ms": self.source_time_ms,
            "publisher_time_ms": self.publisher_time_ms,
            "received_at_ms": self.received_at_ms,
            "price": format(self.price, "f"),
            "source_payload_sha256": self.source_payload_sha256,
        }


def parse_polymarket_chainlink_btc_tick(
    payload: Mapping[str, object],
    *,
    received_at_ms: int,
) -> PolymarketChainlinkBtcTick:
    raw = dict(payload)
    required_envelope = {"topic", "type", "timestamp", "payload"}
    if not required_envelope.issubset(raw) or set(raw) - required_envelope != {
        "connection_id"
    } and set(raw) != required_envelope:
        raise ValueError("Chainlink RTDS envelope schema drifted")
    if raw["topic"] != "crypto_prices_chainlink" or raw["type"] != "update":
        raise ValueError("Chainlink RTDS envelope identity is invalid")
    if "connection_id" in raw and (
        not isinstance(raw["connection_id"], str)
        or not 1 <= len(raw["connection_id"]) <= 128
    ):
        raise ValueError("Chainlink RTDS connection identity is invalid")
    body = raw["payload"]
    required_body = {"symbol", "timestamp", "value"}
    if (
        not isinstance(body, Mapping)
        or not required_body.issubset(body)
        or set(body) - required_body not in (set(), {"full_accuracy_value"})
    ):
        raise ValueError("Chainlink RTDS BTC payload schema drifted")
    if str(body["symbol"]).strip().lower() != "btc/usd":
        raise ValueError("Chainlink RTDS tick is not BTC/USD")
    price = _positive_decimal(body["value"], name="Chainlink BTC price")
    if "full_accuracy_value" in body:
        exact = str(body["full_accuracy_value"] or "").strip()
        if not exact.isdigit() or len(exact) > 80:
            raise ValueError("Chainlink RTDS exact BTC value is invalid")
        exact_price = Decimal(exact) / Decimal("1000000000000000000")
        if float(exact_price) != float(price):
            raise ValueError("Chainlink RTDS exact and rounded BTC values differ")
        price = exact_price
    return PolymarketChainlinkBtcTick(
        source_time_ms=_positive_time_ms(body["timestamp"], name="source timestamp"),
        publisher_time_ms=_positive_time_ms(
            raw["timestamp"],
            name="publisher timestamp",
        ),
        received_at_ms=received_at_ms,
        price=price,
        source_payload_sha256=_sha256(raw),
    )


@dataclass(frozen=True, slots=True)
class PolymarketBtcEndpointEstimate:
    available: bool
    probability_up: float | None
    variance_rate_per_second: float | None
    log_distance_from_open: float | None
    remaining_seconds: float
    tick_count: int
    coverage_seconds: float
    reasons: tuple[str, ...]
    grants_execution_authority: bool = False

    def __post_init__(self) -> None:
        if self.grants_execution_authority:
            raise ValueError("structural BTC estimate cannot grant execution authority")
        if self.available:
            probability = float(self.probability_up)
            variance = float(self.variance_rate_per_second)
            distance = float(self.log_distance_from_open)
            if (
                not 0 < probability < 1
                or not math.isfinite(variance)
                or variance <= 0
                or not math.isfinite(distance)
                or self.reasons
            ):
                raise ValueError("available structural BTC estimate is invalid")
        elif self.probability_up is not None or not self.reasons:
            raise ValueError("unavailable structural BTC estimate is invalid")


class PolymarketBtcEndpointEstimator:
    """Estimate endpoint direction from causal Chainlink ticks only."""

    def __init__(
        self,
        *,
        lookback_seconds: float = 180.0,
        minimum_coverage_seconds: float = 30.0,
        minimum_return_count: int = 20,
        maximum_staleness_ms: int = 1_500,
        variance_floor_per_second: float = 1e-12,
        variance_ceiling_per_second: float = 1e-4,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.lookback_seconds = float(lookback_seconds)
        self.minimum_coverage_seconds = float(minimum_coverage_seconds)
        self.minimum_return_count = int(minimum_return_count)
        self.maximum_staleness_ms = int(maximum_staleness_ms)
        self.variance_floor_per_second = float(variance_floor_per_second)
        self.variance_ceiling_per_second = float(variance_ceiling_per_second)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        if not 30 <= self.lookback_seconds <= 300:
            raise ValueError("lookback_seconds must lie in [30, 300]")
        if not 1 <= self.minimum_coverage_seconds <= self.lookback_seconds:
            raise ValueError("minimum_coverage_seconds is invalid")
        if not 2 <= self.minimum_return_count <= 10_000:
            raise ValueError("minimum_return_count is invalid")
        if not 100 <= self.maximum_staleness_ms <= 10_000:
            raise ValueError("maximum_staleness_ms is invalid")
        if not (
            math.isfinite(self.variance_floor_per_second)
            and math.isfinite(self.variance_ceiling_per_second)
            and 0 < self.variance_floor_per_second
            < self.variance_ceiling_per_second
        ):
            raise ValueError("variance bounds are invalid")
        self._ticks: list[PolymarketChainlinkBtcTick] = []

    def observe(self, tick: PolymarketChainlinkBtcTick) -> bool:
        if not isinstance(tick, PolymarketChainlinkBtcTick):
            raise TypeError("tick must be PolymarketChainlinkBtcTick")
        if self._ticks:
            prior = self._ticks[-1]
            if tick.source_time_ms < prior.source_time_ms:
                raise ValueError("Chainlink RTDS source time regressed")
            if tick.source_time_ms == prior.source_time_ms:
                if tick.price != prior.price:
                    raise ValueError("Chainlink RTDS duplicate time contradicts")
                return False
            if tick.received_at_ms < prior.received_at_ms:
                raise ValueError("Chainlink RTDS receipt time regressed")
        self._ticks.append(tick)
        earliest = tick.source_time_ms - int(self.lookback_seconds * 1_000)
        first_retained = 0
        while (
            first_retained < len(self._ticks) - 1
            and self._ticks[first_retained + 1].source_time_ms < earliest
        ):
            first_retained += 1
        if first_retained:
            del self._ticks[:first_retained]
        return True

    @staticmethod
    def _unavailable(
        *,
        remaining_seconds: float,
        tick_count: int,
        coverage_seconds: float,
        reasons: Sequence[str],
    ) -> PolymarketBtcEndpointEstimate:
        return PolymarketBtcEndpointEstimate(
            available=False,
            probability_up=None,
            variance_rate_per_second=None,
            log_distance_from_open=None,
            remaining_seconds=remaining_seconds,
            tick_count=tick_count,
            coverage_seconds=coverage_seconds,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def estimate(
        self,
        reference: PolymarketBtcReferenceWindow,
        *,
        observed_at_ms: int | None = None,
    ) -> PolymarketBtcEndpointEstimate:
        if not isinstance(reference, PolymarketBtcReferenceWindow):
            raise TypeError("reference must be PolymarketBtcReferenceWindow")
        now = self._clock_ms() if observed_at_ms is None else int(observed_at_ms)
        remaining = max(0.0, (reference.end_ms - now) / 1_000)
        ticks = tuple(
            tick
            for tick in self._ticks
            if reference.event_start_ms <= tick.source_time_ms < reference.end_ms
            and tick.source_time_ms <= now + 250
        )
        coverage = (
            0.0
            if len(ticks) < 2
            else (ticks[-1].source_time_ms - ticks[0].source_time_ms) / 1_000
        )
        reasons: list[str] = []
        if reference.completed or now >= reference.end_ms or remaining <= 0:
            reasons.append("market_window_not_live")
        if not ticks:
            reasons.append("no_causal_chainlink_ticks")
        else:
            source_age = now - ticks[-1].source_time_ms
            if source_age < -250 or source_age > self.maximum_staleness_ms:
                reasons.append("chainlink_tick_stale_or_future")
        if len(ticks) - 1 < self.minimum_return_count:
            reasons.append("insufficient_return_count")
        if coverage < self.minimum_coverage_seconds:
            reasons.append("insufficient_time_coverage")
        if reasons:
            return self._unavailable(
                remaining_seconds=remaining,
                tick_count=len(ticks),
                coverage_seconds=coverage,
                reasons=reasons,
            )

        squared_returns = 0.0
        elapsed_seconds = 0.0
        for previous, current in zip(ticks, ticks[1:], strict=False):
            elapsed = (current.source_time_ms - previous.source_time_ms) / 1_000
            if elapsed <= 0:
                raise ValueError("Chainlink RTDS return interval is invalid")
            change = math.log(float(current.price / previous.price))
            squared_returns += change * change
            elapsed_seconds += elapsed
        variance_rate = squared_returns / elapsed_seconds
        if not math.isfinite(variance_rate) or variance_rate < 0:
            return self._unavailable(
                remaining_seconds=remaining,
                tick_count=len(ticks),
                coverage_seconds=coverage,
                reasons=("invalid_realized_variance",),
            )
        if variance_rate > self.variance_ceiling_per_second:
            return self._unavailable(
                remaining_seconds=remaining,
                tick_count=len(ticks),
                coverage_seconds=coverage,
                reasons=("realized_variance_above_safety_bound",),
            )
        conservative_variance = max(
            variance_rate,
            self.variance_floor_per_second,
        )
        distance = math.log(float(ticks[-1].price / reference.open_price))
        scale = math.sqrt(conservative_variance * remaining)
        z_score = distance / scale
        probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        probability = min(0.999, max(0.001, probability))
        return PolymarketBtcEndpointEstimate(
            available=True,
            probability_up=probability,
            variance_rate_per_second=conservative_variance,
            log_distance_from_open=distance,
            remaining_seconds=remaining,
            tick_count=len(ticks),
            coverage_seconds=coverage,
            reasons=(),
        )


__all__ = [
    "OfficialPolymarketBtcReferenceClient",
    "POLYMARKET_BTC_REFERENCE_SCHEMA_VERSION",
    "POLYMARKET_CHAINLINK_TICK_SCHEMA_VERSION",
    "POLYMARKET_CRYPTO_REFERENCE_URL",
    "PolymarketBtcEndpointEstimate",
    "PolymarketBtcEndpointEstimator",
    "PolymarketBtcReferenceWindow",
    "PolymarketChainlinkBtcTick",
    "parse_polymarket_btc_reference_window",
    "parse_polymarket_chainlink_btc_tick",
]

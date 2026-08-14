"""Exact causal Chainlink TWAP60 state for BTC five-minute research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
from typing import Mapping, Sequence

from .polymarket import (
    POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
    PolymarketFiveMinuteMarket,
)


POLYMARKET_TWAP60_TICK_SCHEMA_VERSION = "polymarket-chainlink-twap60-tick-v1"
POLYMARKET_TWAP60_FEATURE_SCHEMA_VERSION = "polymarket-twap60-causal-features-v1"
POLYMARKET_TWAP60_TOPIC = "crypto_prices_twap_sixty"
_E18 = Decimal(10**18)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketTwap60Tick:
    source_time_ms: int
    publisher_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    exact_e18: int
    source_payload_sha256: str

    def __post_init__(self) -> None:
        source = int(self.source_time_ms)
        publisher = int(self.publisher_time_ms)
        received = int(self.received_wall_ms)
        monotonic = int(self.received_monotonic_ns)
        exact = int(self.exact_e18)
        digest = str(self.source_payload_sha256 or "").strip().lower()
        if (
            source <= 0
            or source % 1_000 != 0
            or publisher < source
            or received <= 0
            or monotonic <= 0
            or exact <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Polymarket TWAP60 tick is invalid")
        object.__setattr__(self, "source_time_ms", source)
        object.__setattr__(self, "publisher_time_ms", publisher)
        object.__setattr__(self, "received_wall_ms", received)
        object.__setattr__(self, "received_monotonic_ns", monotonic)
        object.__setattr__(self, "exact_e18", exact)
        object.__setattr__(self, "source_payload_sha256", digest)

    @property
    def price(self) -> Decimal:
        return Decimal(self.exact_e18) / _E18

    @property
    def source_to_publisher_ms(self) -> int:
        return self.publisher_time_ms - self.source_time_ms

    @property
    def publisher_to_receipt_ms(self) -> int:
        return self.received_wall_ms - self.publisher_time_ms

    @property
    def source_to_receipt_ms(self) -> int:
        return self.received_wall_ms - self.source_time_ms

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_TWAP60_TICK_SCHEMA_VERSION,
            "source_time_ms": self.source_time_ms,
            "publisher_time_ms": self.publisher_time_ms,
            "received_wall_ms": self.received_wall_ms,
            "received_monotonic_ns": self.received_monotonic_ns,
            "exact_e18": str(self.exact_e18),
            "price": format(self.price, "f"),
            "source_to_publisher_ms": self.source_to_publisher_ms,
            "publisher_to_receipt_ms": self.publisher_to_receipt_ms,
            "source_to_receipt_ms": self.source_to_receipt_ms,
            "source_payload_sha256": self.source_payload_sha256,
        }


def parse_polymarket_twap60_tick(
    event: Mapping[str, object],
    *,
    received_wall_ms: int,
    received_monotonic_ns: int,
) -> PolymarketTwap60Tick:
    raw = dict(event)
    allowed_envelope = {"topic", "type", "timestamp", "payload", "connection_id"}
    if (
        not {"topic", "type", "timestamp", "payload"}.issubset(raw)
        or set(raw) - allowed_envelope
        or raw.get("topic") != POLYMARKET_TWAP60_TOPIC
        or raw.get("type") != "update"
    ):
        raise ValueError("Polymarket TWAP60 envelope differs")
    payload = raw.get("payload")
    required_payload = {
        "symbol",
        "timestamp",
        "value",
        "window_s",
        "full_accuracy_value",
    }
    if not isinstance(payload, Mapping) or set(payload) != required_payload:
        raise ValueError("Polymarket TWAP60 payload schema differs")
    source_time = payload.get("timestamp")
    publisher_time = raw.get("timestamp")
    exact_text = payload.get("full_accuracy_value")
    if (
        isinstance(source_time, bool)
        or not isinstance(source_time, int)
        or isinstance(publisher_time, bool)
        or not isinstance(publisher_time, int)
        or payload.get("symbol") != "btc/usd"
        or payload.get("window_s") != 60
        or not isinstance(exact_text, str)
        or not exact_text.isascii()
        or not exact_text.isdigit()
    ):
        raise ValueError("Polymarket TWAP60 payload differs")
    exact = int(exact_text)
    try:
        rounded = Decimal(str(payload.get("value")))
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("Polymarket TWAP60 rounded value differs") from exc
    exact_price = Decimal(exact) / _E18
    if not rounded.is_finite() or rounded <= 0 or float(rounded) != float(exact_price):
        raise ValueError("Polymarket TWAP60 exact and rounded values differ")
    return PolymarketTwap60Tick(
        source_time_ms=source_time,
        publisher_time_ms=publisher_time,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=received_monotonic_ns,
        exact_e18=exact,
        source_payload_sha256=_canonical_sha256(raw),
    )


@dataclass(frozen=True, slots=True)
class PolymarketTwap60Features:
    available: bool
    observed_wall_ms: int
    remaining_seconds: float
    tick_count: int
    coverage_seconds: float
    opening_exact_e18: int | None
    current_exact_e18: int | None
    log_distance_from_open: float | None
    realized_variance_rate_per_second: float | None
    path_efficiency: float | None
    current_source_age_ms: int | None
    current_source_to_publisher_ms: int | None
    current_publisher_to_receipt_ms: int | None
    reasons: tuple[str, ...]
    grants_execution_authority: bool = False

    def __post_init__(self) -> None:
        if self.grants_execution_authority:
            raise ValueError("TWAP60 features cannot grant execution authority")
        if self.available:
            finite = (
                self.log_distance_from_open,
                self.realized_variance_rate_per_second,
                self.path_efficiency,
            )
            if (
                self.reasons
                or self.opening_exact_e18 is None
                or self.current_exact_e18 is None
                or any(value is None or not math.isfinite(value) for value in finite)
                or self.realized_variance_rate_per_second is None
                or self.realized_variance_rate_per_second < 0
                or self.path_efficiency is None
                or not 0 <= self.path_efficiency <= 1
            ):
                raise ValueError("available TWAP60 features are invalid")
        elif not self.reasons:
            raise ValueError("unavailable TWAP60 features require reasons")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = POLYMARKET_TWAP60_FEATURE_SCHEMA_VERSION
        if self.opening_exact_e18 is not None:
            payload["opening_exact_e18"] = str(self.opening_exact_e18)
        if self.current_exact_e18 is not None:
            payload["current_exact_e18"] = str(self.current_exact_e18)
        return payload


class PolymarketTwap60FeatureState:
    """Maintain exact ticks and expose target-free causal settlement features."""

    def __init__(
        self,
        *,
        minimum_return_count: int = 10,
        minimum_coverage_seconds: float = 10.0,
        maximum_source_age_ms: int = 5_000,
    ) -> None:
        self.minimum_return_count = int(minimum_return_count)
        self.minimum_coverage_seconds = float(minimum_coverage_seconds)
        self.maximum_source_age_ms = int(maximum_source_age_ms)
        if (
            not 2 <= self.minimum_return_count <= 300
            or not 1 <= self.minimum_coverage_seconds <= 300
            or not 1_000 <= self.maximum_source_age_ms <= 30_000
        ):
            raise ValueError("TWAP60 feature-state controls are invalid")
        self._ticks: list[PolymarketTwap60Tick] = []
        self._tick_by_source_time: dict[int, PolymarketTwap60Tick] = {}

    def observe(self, tick: PolymarketTwap60Tick) -> bool:
        if not isinstance(tick, PolymarketTwap60Tick):
            raise TypeError("tick must be PolymarketTwap60Tick")
        existing = self._tick_by_source_time.get(tick.source_time_ms)
        if existing is not None:
            if tick.exact_e18 != existing.exact_e18:
                raise ValueError("Polymarket TWAP60 duplicate time contradicts")
            return False
        if self._ticks:
            previous = self._ticks[-1]
            if tick.received_monotonic_ns < previous.received_monotonic_ns:
                raise ValueError("Polymarket TWAP60 receipt clock regressed")
            if tick.source_time_ms < previous.source_time_ms:
                raise ValueError("Polymarket TWAP60 source time regressed")
        self._ticks.append(tick)
        self._tick_by_source_time[tick.source_time_ms] = tick
        earliest = tick.source_time_ms - 360_000
        while len(self._ticks) > 1 and self._ticks[1].source_time_ms < earliest:
            removed = self._ticks.pop(0)
            self._tick_by_source_time.pop(removed.source_time_ms, None)
        return True

    @staticmethod
    def _unavailable(
        *,
        now: int,
        remaining: float,
        ticks: Sequence[PolymarketTwap60Tick],
        coverage: float,
        opening: PolymarketTwap60Tick | None,
        reasons: Sequence[str],
    ) -> PolymarketTwap60Features:
        current = ticks[-1] if ticks else None
        return PolymarketTwap60Features(
            available=False,
            observed_wall_ms=now,
            remaining_seconds=remaining,
            tick_count=len(ticks),
            coverage_seconds=coverage,
            opening_exact_e18=None if opening is None else opening.exact_e18,
            current_exact_e18=None if current is None else current.exact_e18,
            log_distance_from_open=None,
            realized_variance_rate_per_second=None,
            path_efficiency=None,
            current_source_age_ms=(
                None if current is None else now - current.source_time_ms
            ),
            current_source_to_publisher_ms=(
                None if current is None else current.source_to_publisher_ms
            ),
            current_publisher_to_receipt_ms=(
                None if current is None else current.publisher_to_receipt_ms
            ),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def features(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        observed_wall_ms: int,
        observed_monotonic_ns: int,
    ) -> PolymarketTwap60Features:
        now = int(observed_wall_ms)
        monotonic = int(observed_monotonic_ns)
        remaining = max(0.0, (market.end_ms - now) / 1_000)
        if (
            market.asset != "BTC"
            or market.resolution_source.rstrip("/")
            != POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
        ):
            raise ValueError("TWAP60 features require a BTC TWAP60 market")
        ticks = tuple(
            tick
            for tick in self._ticks
            if market.event_start_ms <= tick.source_time_ms < market.end_ms
            and tick.received_wall_ms <= now
            and tick.received_monotonic_ns <= monotonic
        )
        opening = next(
            (tick for tick in ticks if tick.source_time_ms == market.event_start_ms),
            None,
        )
        coverage = (
            0.0
            if len(ticks) < 2
            else (ticks[-1].source_time_ms - ticks[0].source_time_ms) / 1_000
        )
        reasons: list[str] = []
        if not market.event_start_ms <= now < market.end_ms:
            reasons.append("market_window_not_live")
        if opening is None:
            reasons.append("exact_opening_twap_missing")
        if not ticks:
            reasons.append("no_causal_twap_ticks")
        else:
            source_age = now - ticks[-1].source_time_ms
            if source_age < 0 or source_age > self.maximum_source_age_ms:
                reasons.append("current_twap_stale_or_future")
        if len(ticks) - 1 < self.minimum_return_count:
            reasons.append("insufficient_return_count")
        if coverage < self.minimum_coverage_seconds:
            reasons.append("insufficient_time_coverage")
        if reasons:
            return self._unavailable(
                now=now,
                remaining=remaining,
                ticks=ticks,
                coverage=coverage,
                opening=opening,
                reasons=reasons,
            )
        assert opening is not None
        log_returns: list[float] = []
        elapsed_seconds = 0.0
        for previous, current in zip(ticks, ticks[1:], strict=False):
            elapsed = (current.source_time_ms - previous.source_time_ms) / 1_000
            if elapsed <= 0:
                raise ValueError("Polymarket TWAP60 interval is invalid")
            log_returns.append(math.log(current.exact_e18 / previous.exact_e18))
            elapsed_seconds += elapsed
        variance_rate = sum(value * value for value in log_returns) / elapsed_seconds
        absolute_path = sum(abs(value) for value in log_returns)
        net_path = math.log(ticks[-1].exact_e18 / opening.exact_e18)
        efficiency = 0.0 if absolute_path == 0 else min(1.0, abs(net_path) / absolute_path)
        current = ticks[-1]
        return PolymarketTwap60Features(
            available=True,
            observed_wall_ms=now,
            remaining_seconds=remaining,
            tick_count=len(ticks),
            coverage_seconds=coverage,
            opening_exact_e18=opening.exact_e18,
            current_exact_e18=current.exact_e18,
            log_distance_from_open=net_path,
            realized_variance_rate_per_second=variance_rate,
            path_efficiency=efficiency,
            current_source_age_ms=now - current.source_time_ms,
            current_source_to_publisher_ms=current.source_to_publisher_ms,
            current_publisher_to_receipt_ms=current.publisher_to_receipt_ms,
            reasons=(),
        )


__all__ = [
    "POLYMARKET_TWAP60_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_TWAP60_TICK_SCHEMA_VERSION",
    "POLYMARKET_TWAP60_TOPIC",
    "PolymarketTwap60FeatureState",
    "PolymarketTwap60Features",
    "PolymarketTwap60Tick",
    "parse_polymarket_twap60_tick",
]

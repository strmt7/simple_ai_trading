"""Read-only BTC price-discovery features for the independent Polymarket bot.

The contract accepts observations collected elsewhere. It has no exchange
client, credentials, balances, positions, or order methods and cannot authorize
new exposure. Signal quality may only preserve, reduce, or veto a separately
approved Polymarket action.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math


_ALLOWED_SOURCES = frozenset({"BINANCE_SPOT", "BINANCE_USD_M_FUTURES"})


def _finite_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _finite_nonnegative_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class BtcPriceDiscoveryTick:
    """One public, read-only BTC/USDT top-of-book observation."""

    source: str
    symbol: str
    event_time_ms: int
    received_at_ms: int
    sequence: int
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        source = str(self.source or "").strip().upper()
        if source not in _ALLOWED_SOURCES:
            raise ValueError("unsupported BTC price-discovery source")
        object.__setattr__(self, "source", source)
        symbol = str(self.symbol or "").strip().upper()
        if symbol != "BTCUSDT":
            raise ValueError("Polymarket price discovery is BTCUSDT-only")
        object.__setattr__(self, "symbol", symbol)
        event_time_ms = int(self.event_time_ms)
        received_at_ms = int(self.received_at_ms)
        sequence = int(self.sequence)
        if event_time_ms <= 0 or received_at_ms <= 0 or sequence < 0:
            raise ValueError("price-discovery chronology is invalid")
        object.__setattr__(self, "event_time_ms", event_time_ms)
        object.__setattr__(self, "received_at_ms", received_at_ms)
        object.__setattr__(self, "sequence", sequence)
        bid = _finite_decimal(self.bid, name="bid")
        ask = _finite_decimal(self.ask, name="ask")
        if ask <= bid:
            raise ValueError("price-discovery book is crossed or locked")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        return (self.ask - self.bid) / self.mid * Decimal("10000")


@dataclass(frozen=True, slots=True)
class PolymarketBtcReferenceFeatures:
    observed_at_ms: int
    spot_mid: Decimal
    futures_mid: Decimal
    spot_spread_bps: Decimal
    futures_spread_bps: Decimal
    futures_basis_bps: Decimal
    spot_log_return: float
    futures_log_return: float
    event_time_skew_ms: int
    receive_time_skew_ms: int


@dataclass(frozen=True, slots=True)
class PolymarketExternalSignalDecision:
    """A fail-closed advisory result that never grants trading authority."""

    action: str
    maximum_size_multiplier: Decimal
    reasons: tuple[str, ...]
    features: PolymarketBtcReferenceFeatures | None
    grants_execution_authority: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"preserve", "reduce", "abstain"}:
            raise ValueError("external signal action is invalid")
        multiplier = _finite_nonnegative_decimal(
            self.maximum_size_multiplier,
            name="maximum_size_multiplier",
        )
        if multiplier > 1:
            raise ValueError("external signal cannot increase position size")
        if self.action == "abstain" and multiplier != 0:
            raise ValueError("abstain must use a zero size multiplier")
        if self.grants_execution_authority:
            raise ValueError("external price discovery cannot grant execution authority")
        object.__setattr__(self, "maximum_size_multiplier", multiplier)


class PolymarketBtcPriceDiscoveryMonitor:
    """Build quality-gated exogenous features without exchange coupling."""

    def __init__(
        self,
        *,
        maximum_staleness_ms: int = 1_500,
        maximum_future_event_ms: int = 250,
        maximum_transport_latency_ms: int = 2_000,
        maximum_event_skew_ms: int = 1_000,
        maximum_receive_skew_ms: int = 1_000,
        reduction_spread_bps: Decimal = Decimal("5"),
    ) -> None:
        self.maximum_staleness_ms = int(maximum_staleness_ms)
        self.maximum_future_event_ms = int(maximum_future_event_ms)
        self.maximum_transport_latency_ms = int(maximum_transport_latency_ms)
        self.maximum_event_skew_ms = int(maximum_event_skew_ms)
        self.maximum_receive_skew_ms = int(maximum_receive_skew_ms)
        self.reduction_spread_bps = _finite_decimal(
            reduction_spread_bps,
            name="reduction_spread_bps",
        )
        limits = (
            self.maximum_staleness_ms,
            self.maximum_future_event_ms,
            self.maximum_transport_latency_ms,
            self.maximum_event_skew_ms,
            self.maximum_receive_skew_ms,
        )
        if any(value < 0 or value > 60_000 for value in limits):
            raise ValueError("price-discovery time limits must lie in [0, 60000]")
        self._last_ticks: dict[str, BtcPriceDiscoveryTick] = {}
        self._previous_mids: dict[str, Decimal] = {}

    @staticmethod
    def _abstain(*reasons: str) -> PolymarketExternalSignalDecision:
        return PolymarketExternalSignalDecision(
            action="abstain",
            maximum_size_multiplier=Decimal("0"),
            reasons=tuple(dict.fromkeys(reasons)),
            features=None,
        )

    def evaluate(
        self,
        *,
        spot: BtcPriceDiscoveryTick,
        futures: BtcPriceDiscoveryTick,
        observed_at_ms: int,
    ) -> PolymarketExternalSignalDecision:
        now = int(observed_at_ms)
        if now <= 0:
            raise ValueError("observed_at_ms must be positive")
        if spot.source != "BINANCE_SPOT" or futures.source != "BINANCE_USD_M_FUTURES":
            return self._abstain("source_pair_mismatch")
        reasons: list[str] = []
        for tick in (spot, futures):
            prior = self._last_ticks.get(tick.source)
            if prior is not None and (
                tick.sequence <= prior.sequence
                or tick.event_time_ms < prior.event_time_ms
                or tick.received_at_ms < prior.received_at_ms
            ):
                reasons.append(f"{tick.source.lower()}_sequence_regression")
            if now - tick.received_at_ms > self.maximum_staleness_ms:
                reasons.append(f"{tick.source.lower()}_stale")
            if tick.event_time_ms - now > self.maximum_future_event_ms:
                reasons.append(f"{tick.source.lower()}_future_event")
            transport_latency = tick.received_at_ms - tick.event_time_ms
            if (
                transport_latency < -self.maximum_future_event_ms
                or transport_latency > self.maximum_transport_latency_ms
            ):
                reasons.append(f"{tick.source.lower()}_transport_latency")
        event_skew = abs(spot.event_time_ms - futures.event_time_ms)
        receive_skew = abs(spot.received_at_ms - futures.received_at_ms)
        if event_skew > self.maximum_event_skew_ms:
            reasons.append("cross_feed_event_skew")
        if receive_skew > self.maximum_receive_skew_ms:
            reasons.append("cross_feed_receive_skew")
        if reasons:
            return self._abstain(*reasons)

        spot_previous = self._previous_mids.get(spot.source, spot.mid)
        futures_previous = self._previous_mids.get(futures.source, futures.mid)
        features = PolymarketBtcReferenceFeatures(
            observed_at_ms=now,
            spot_mid=spot.mid,
            futures_mid=futures.mid,
            spot_spread_bps=spot.spread_bps,
            futures_spread_bps=futures.spread_bps,
            futures_basis_bps=(futures.mid / spot.mid - 1) * Decimal("10000"),
            spot_log_return=math.log(float(spot.mid / spot_previous)),
            futures_log_return=math.log(float(futures.mid / futures_previous)),
            event_time_skew_ms=event_skew,
            receive_time_skew_ms=receive_skew,
        )
        self._last_ticks[spot.source] = spot
        self._last_ticks[futures.source] = futures
        self._previous_mids[spot.source] = spot.mid
        self._previous_mids[futures.source] = futures.mid
        wide_spread = (
            spot.spread_bps > self.reduction_spread_bps
            or futures.spread_bps > self.reduction_spread_bps
        )
        return PolymarketExternalSignalDecision(
            action="reduce" if wide_spread else "preserve",
            maximum_size_multiplier=Decimal("0.5") if wide_spread else Decimal("1"),
            reasons=("wide_reference_spread",) if wide_spread else (),
            features=features,
        )


__all__ = [
    "BtcPriceDiscoveryTick",
    "PolymarketBtcPriceDiscoveryMonitor",
    "PolymarketBtcReferenceFeatures",
    "PolymarketExternalSignalDecision",
]

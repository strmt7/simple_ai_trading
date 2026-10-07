"""Exact linear funding cash for supplied events, not complete account P&L."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import re
from typing import Sequence


USDM_SYMBOLS = frozenset(("BTCUSDT", "ETHUSDT", "SOLUSDT"))
MAX_HISTORY_ROWS = 1_000
MAX_HISTORY_BYTES = 5_000_000


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _decimal_fraction(value: object) -> Fraction:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value) is None
    ):
        raise ValueError("funding numeric field is not a finite decimal string")
    return Fraction(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("funding response contains duplicate object keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("funding response contains a nonfinite JSON constant")


@dataclass(frozen=True)
class LinearFundingSettlement:
    symbol: str
    payment_asset: str
    funding_time_ms: int
    rate: Fraction
    settlement_mark: Fraction
    source_body_sha256: str
    rate_type: str | None = None

    def __post_init__(self) -> None:
        if (
            self.symbol not in USDM_SYMBOLS
            or self.payment_asset != "USDT"
            or type(self.funding_time_ms) is not int
            or self.funding_time_ms < 0
            or not isinstance(self.rate, Fraction)
            or not isinstance(self.settlement_mark, Fraction)
            or self.settlement_mark <= 0
            or not _valid_sha256(self.source_body_sha256)
            or self.rate_type not in (None, "Regular")
        ):
            raise ValueError("linear funding settlement contract is invalid")


def parse_binance_usdm_funding_history(
    raw_body: bytes, *, expected_sha256: str, expected_symbol: str
) -> tuple[LinearFundingSettlement, ...]:
    """Parse hash-matched public funding rows without fetching or proving coverage.

    Only BTC/ETH/SOL USDT linear contracts are supported. Empty/truncated pages
    do not prove absence of payments; callers must reconcile the full expected
    event population and capture origin before admitting a cash-qualified label.
    A source digest proves byte identity, not exchange origin or authenticity.
    """
    if (
        not isinstance(raw_body, bytes)
        or len(raw_body) > MAX_HISTORY_BYTES
        or not _valid_sha256(expected_sha256)
        or hashlib.sha256(raw_body).hexdigest() != expected_sha256
        or expected_symbol not in USDM_SYMBOLS
    ):
        raise ValueError("funding response source binding is invalid")
    payload = json.loads(
        raw_body.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, list) or len(payload) > MAX_HISTORY_ROWS:
        raise ValueError("funding response must be a bounded row array")
    settlements: list[LinearFundingSettlement] = []
    required = {"symbol", "fundingTime", "fundingRate", "markPrice"}
    for row in payload:
        if (
            not isinstance(row, dict)
            or not required <= row.keys()
            or not row.keys() <= required | {"rateType"}
            or row["symbol"] != expected_symbol
        ):
            raise ValueError("funding row schema or symbol is invalid")
        settlement = LinearFundingSettlement(
            symbol=expected_symbol,
            payment_asset="USDT",
            funding_time_ms=row["fundingTime"],
            rate=_decimal_fraction(row["fundingRate"]),
            settlement_mark=_decimal_fraction(row["markPrice"]),
            source_body_sha256=expected_sha256,
            rate_type=row.get("rateType"),
        )
        if (
            settlements
            and settlement.funding_time_ms <= settlements[-1].funding_time_ms
        ):
            raise ValueError("funding events are duplicated or not strictly increasing")
        settlements.append(settlement)
    return tuple(settlements)


class FundingEntitlement(Enum):
    HELD = "held"
    NOT_HELD = "not_held"
    UNKNOWN = "unknown"


def reconcile_funding_event_population(
    settlements: Sequence[LinearFundingSettlement],
    expected_events: Sequence[tuple[int, Fraction]],
    *,
    expected_symbol: str,
) -> tuple[LinearFundingSettlement, ...]:
    """Require exact ordered time/rate coverage of an independently certified list.

    Missing, extra, repeated or conflicting events cannot qualify mark coverage.
    This checks alignment only; the caller must validate the expected population's
    source certificate, UTC boundaries and completeness independently.
    """
    if expected_symbol not in USDM_SYMBOLS or len(settlements) != len(expected_events):
        raise ValueError("funding mark population does not match expected events")
    previous_time: int | None = None
    for settlement, expected in zip(settlements, expected_events, strict=True):
        if (
            not isinstance(expected, tuple)
            or len(expected) != 2
            or type(expected[0]) is not int
            or expected[0] < 0
            or not isinstance(expected[1], Fraction)
            or not isinstance(settlement, LinearFundingSettlement)
            or settlement.symbol != expected_symbol
            or (settlement.funding_time_ms, settlement.rate) != expected
            or (previous_time is not None and expected[0] <= previous_time)
        ):
            raise ValueError(
                "funding mark population has unqualified or conflicting events"
            )
        previous_time = expected[0]
    return tuple(settlements)


@dataclass(frozen=True)
class FundingCashBounds:
    payment_asset: str
    cash_lower: Fraction
    cash_upper: Fraction
    entry_relative_lower_bps: Fraction
    entry_relative_upper_bps: Fraction
    uncertain_entitlement_events: int
    supplied_events: int
    source_body_sha256: tuple[str, ...]

    @property
    def cash_amount_exact(self) -> bool:
        return self.cash_lower == self.cash_upper


def funding_cash_bounds_for_events(
    settlements: Sequence[LinearFundingSettlement],
    entitlements: Sequence[FundingEntitlement],
    *,
    expected_symbol: str,
    signed_base_quantity: Fraction,
    entry_price: Fraction,
) -> FundingCashBounds:
    """Bound supplied-event funding cash without guessing settlement entitlement.

    Positive quantity is long; positive cash is received. Unknown entitlement
    encloses both zero and the payment, independently for each event. This is a
    conservative enclosure, not proof that its extreme is jointly attainable.
    It excludes fees, financing, FX, rounding and unprovided events. A timestamp
    comparison, supplied HELD flag or digest alone is not admission evidence.
    """
    if (
        expected_symbol not in USDM_SYMBOLS
        or len(settlements) != len(entitlements)
        or not isinstance(signed_base_quantity, Fraction)
        or signed_base_quantity == 0
        or not isinstance(entry_price, Fraction)
        or entry_price <= 0
    ):
        raise ValueError("funding cash position contract is invalid")
    lower = upper = Fraction(0)
    uncertain = 0
    previous_time: int | None = None
    sources: set[str] = set()
    for settlement, entitlement in zip(settlements, entitlements, strict=True):
        if (
            not isinstance(settlement, LinearFundingSettlement)
            or settlement.symbol != expected_symbol
            or not isinstance(entitlement, FundingEntitlement)
            or (
                previous_time is not None
                and settlement.funding_time_ms <= previous_time
            )
        ):
            raise ValueError("funding event population or entitlement is invalid")
        previous_time = settlement.funding_time_ms
        sources.add(settlement.source_body_sha256)
        payment = -signed_base_quantity * settlement.settlement_mark * settlement.rate
        if entitlement is FundingEntitlement.HELD:
            lower += payment
            upper += payment
        elif entitlement is FundingEntitlement.UNKNOWN:
            lower += min(Fraction(0), payment)
            upper += max(Fraction(0), payment)
            uncertain += 1
    entry_notional = abs(signed_base_quantity) * entry_price
    return FundingCashBounds(
        payment_asset="USDT",
        cash_lower=lower,
        cash_upper=upper,
        entry_relative_lower_bps=lower / entry_notional * 10_000,
        entry_relative_upper_bps=upper / entry_notional * 10_000,
        uncertain_entitlement_events=uncertain,
        supplied_events=len(settlements),
        source_body_sha256=tuple(sorted(sources)),
    )

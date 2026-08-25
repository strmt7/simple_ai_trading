"""Exact, target-free parity screens for plain-vanilla option chains."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from itertools import combinations
from math import gcd, lcm
from typing import Mapping, Sequence, cast


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


def _fraction(value: Decimal) -> Fraction:
    return Fraction(value)


def _fraction_lcm(values: Sequence[Fraction]) -> Fraction:
    return Fraction(
        reduce(lcm, (value.numerator for value in values)),
        reduce(gcd, (value.denominator for value in values)),
    )


@dataclass(frozen=True, slots=True)
class OptionContractQuote:
    """One option contract and its ticker discovery prices."""

    symbol: str
    underlying: str
    expiry_date_ms: int
    side: str
    strike: Decimal
    unit: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    step_size: Decimal
    bid_price: Decimal | None
    ask_price: Decimal | None

    def validated(self) -> "OptionContractQuote":
        symbol = str(self.symbol or "").strip().upper()
        underlying = str(self.underlying or "").strip().upper()
        side = str(self.side or "").strip().upper()
        if (
            not symbol
            or len(symbol) > 80
            or not symbol.replace("-", "").isalnum()
            or not underlying
            or not underlying.isalnum()
            or side not in {"CALL", "PUT"}
        ):
            raise ValueError("option contract identity is invalid")
        if (
            isinstance(self.expiry_date_ms, bool)
            or not isinstance(self.expiry_date_ms, int)
            or self.expiry_date_ms <= 0
        ):
            raise ValueError("option expiry date must be a positive integer")
        strike = _positive_decimal(self.strike, name="option strike")
        unit = _positive_decimal(self.unit, name="option unit")
        minimum = _positive_decimal(
            self.minimum_quantity,
            name="option minimum quantity",
        )
        maximum = _positive_decimal(
            self.maximum_quantity,
            name="option maximum quantity",
        )
        step = _positive_decimal(self.step_size, name="option step size")
        if minimum > maximum:
            raise ValueError("option minimum quantity exceeds maximum quantity")

        def optional_price(value: Decimal | None, *, name: str) -> Decimal | None:
            if value is None:
                return None
            return _positive_decimal(value, name=name)

        bid = optional_price(self.bid_price, name="option bid price")
        ask = optional_price(self.ask_price, name="option ask price")
        if bid is not None and ask is not None and bid > ask:
            raise ValueError("option ticker is crossed")
        return OptionContractQuote(
            symbol=symbol,
            underlying=underlying,
            expiry_date_ms=self.expiry_date_ms,
            side=side,
            strike=strike,
            unit=unit,
            minimum_quantity=minimum,
            maximum_quantity=maximum,
            step_size=step,
            bid_price=bid,
            ask_price=ask,
        )


@dataclass(frozen=True, slots=True)
class OptionBookLevel:
    """One displayed option price level."""

    price: Decimal
    quantity: Decimal

    def validated(self) -> "OptionBookLevel":
        return OptionBookLevel(
            price=_positive_decimal(self.price, name="option book price"),
            quantity=_positive_decimal(
                self.quantity,
                name="option book quantity",
            ),
        )


@dataclass(frozen=True, slots=True)
class OptionDepthQuote:
    """One timestamped displayed order book used for confirmation."""

    symbol: str
    event_time_ms: int
    bids: tuple[OptionBookLevel, ...]
    asks: tuple[OptionBookLevel, ...]

    def validated(self) -> "OptionDepthQuote":
        symbol = str(self.symbol or "").strip().upper()
        if not symbol or len(symbol) > 80 or not symbol.replace("-", "").isalnum():
            raise ValueError("option depth symbol is invalid")
        if (
            isinstance(self.event_time_ms, bool)
            or not isinstance(self.event_time_ms, int)
            or self.event_time_ms <= 0
        ):
            raise ValueError("option depth event time must be a positive integer")

        def levels(
            raw: Sequence[OptionBookLevel], *, descending: bool
        ) -> tuple[OptionBookLevel, ...]:
            parsed = tuple(level.validated() for level in raw)
            expected = tuple(
                sorted(parsed, key=lambda level: level.price, reverse=descending)
            )
            if parsed != expected:
                raise ValueError("option depth levels are not price-sorted")
            if len({level.price for level in parsed}) != len(parsed):
                raise ValueError("option depth prices are duplicated")
            return parsed

        bids = levels(self.bids, descending=True)
        asks = levels(self.asks, descending=False)
        if bids and asks and bids[0].price > asks[0].price:
            raise ValueError("option depth is crossed")
        return OptionDepthQuote(
            symbol=symbol,
            event_time_ms=self.event_time_ms,
            bids=bids,
            asks=asks,
        )


@dataclass(frozen=True, slots=True)
class OptionParityCandidate:
    """A gross-positive payoff-dominance portfolio found during discovery."""

    mechanism: str
    symbols: tuple[str, ...]
    roles: tuple[str, ...]
    strikes: tuple[Decimal, ...]
    integer_weights: tuple[int, ...]
    quantities: tuple[Decimal, ...]
    gross_credit_quote: Decimal


@dataclass(frozen=True, slots=True)
class OptionParityDiscovery:
    """Aggregate result for every supported vertical and convexity identity."""

    evaluated_vertical_count: int
    executable_vertical_count: int
    evaluated_convexity_count: int
    executable_convexity_count: int
    gross_positive_candidates: tuple[OptionParityCandidate, ...]


@dataclass(frozen=True, slots=True)
class OptionParityConfirmation:
    """Displayed-depth confirmation of one discovery candidate."""

    executable: bool
    gross_credit_quote: Decimal | None
    book_event_times_ms: tuple[int, ...]


def primitive_convexity_weights(
    lower_strike: Decimal,
    middle_strike: Decimal,
    upper_strike: Decimal,
) -> tuple[int, int, int]:
    """Return primitive long-lower, short-middle, long-upper quantities."""

    lower = _positive_decimal(lower_strike, name="lower strike")
    middle = _positive_decimal(middle_strike, name="middle strike")
    upper = _positive_decimal(upper_strike, name="upper strike")
    if not lower < middle < upper:
        raise ValueError("option strikes must be strictly increasing")
    first_gap = _fraction(middle - lower)
    second_gap = _fraction(upper - middle)
    denominator = lcm(first_gap.denominator, second_gap.denominator)
    lower_weight = second_gap.numerator * (denominator // second_gap.denominator)
    upper_weight = first_gap.numerator * (denominator // first_gap.denominator)
    divisor = gcd(lower_weight, upper_weight)
    lower_weight //= divisor
    upper_weight //= divisor
    return lower_weight, lower_weight + upper_weight, upper_weight


def minimum_ratio_quantities(
    integer_weights: Sequence[int],
    contracts: Sequence[OptionContractQuote],
) -> tuple[Decimal, ...]:
    """Find the smallest exact ratio satisfying every lot step and minimum."""

    weights = tuple(integer_weights)
    normalized = tuple(contract.validated() for contract in contracts)
    if (
        not weights
        or len(weights) != len(normalized)
        or any(isinstance(weight, bool) or weight <= 0 for weight in weights)
    ):
        raise ValueError("option ratio weights are invalid")
    base = _fraction_lcm(
        tuple(
            _fraction(contract.step_size) / weight
            for weight, contract in zip(weights, normalized, strict=True)
        )
    )
    multiplier = 1
    for weight, contract in zip(weights, normalized, strict=True):
        required = _fraction(contract.minimum_quantity) / (weight * base)
        multiplier = max(
            multiplier,
            (required.numerator + required.denominator - 1) // required.denominator,
        )
    scale = base * multiplier
    quantities = tuple(
        Decimal(quantity.numerator) / Decimal(quantity.denominator)
        for quantity in (weight * scale for weight in weights)
    )
    if any(
        quantity > contract.maximum_quantity
        for quantity, contract in zip(quantities, normalized, strict=True)
    ):
        raise ValueError("minimum option ratio exceeds a maximum quantity")
    return quantities


def _candidate(
    *,
    mechanism: str,
    contracts: tuple[OptionContractQuote, ...],
    roles: tuple[str, ...],
    weights: tuple[int, ...],
) -> OptionParityCandidate | None:
    quantities = minimum_ratio_quantities(weights, contracts)
    gross = Decimal("0")
    for contract, role, quantity in zip(contracts, roles, quantities, strict=True):
        price = cast(
            Decimal,
            contract.ask_price if role == "buy" else contract.bid_price,
        )
        gross += (-price if role == "buy" else price) * quantity
    if gross <= 0:
        return None
    return OptionParityCandidate(
        mechanism=mechanism,
        symbols=tuple(contract.symbol for contract in contracts),
        roles=roles,
        strikes=tuple(contract.strike for contract in contracts),
        integer_weights=weights,
        quantities=quantities,
        gross_credit_quote=gross,
    )


def discover_option_parity(
    contracts: Sequence[OptionContractQuote],
) -> OptionParityDiscovery:
    """Enumerate direction-independent vertical and convexity violations."""

    normalized = tuple(contract.validated() for contract in contracts)
    if not normalized or len({contract.symbol for contract in normalized}) != len(
        normalized
    ):
        raise ValueError("option contracts must be nonempty and unique")
    grouped: dict[tuple[str, int, str], list[OptionContractQuote]] = {}
    for contract in normalized:
        grouped.setdefault(
            (contract.underlying, contract.expiry_date_ms, contract.side), []
        ).append(contract)

    evaluated_vertical = 0
    executable_vertical = 0
    evaluated_convexity = 0
    executable_convexity = 0
    positive: list[OptionParityCandidate] = []
    for group in grouped.values():
        group.sort(key=lambda contract: contract.strike)
        if len({contract.strike for contract in group}) != len(group):
            raise ValueError("option strike is duplicated within a chain")
        if len({contract.unit for contract in group}) != 1:
            raise ValueError("option units differ within a chain")

        for lower, upper in combinations(group, 2):
            evaluated_vertical += 1
            if lower.side == "CALL":
                legs = (lower, upper)
            else:
                legs = (upper, lower)
            if legs[0].ask_price is None or legs[1].bid_price is None:
                continue
            executable_vertical += 1
            candidate = _candidate(
                mechanism="vertical_dominance",
                contracts=legs,
                roles=("buy", "sell"),
                weights=(1, 1),
            )
            if candidate is not None:
                positive.append(candidate)

        for lower, middle, upper in combinations(group, 3):
            evaluated_convexity += 1
            if (
                lower.ask_price is None
                or middle.bid_price is None
                or upper.ask_price is None
            ):
                continue
            executable_convexity += 1
            weights = primitive_convexity_weights(
                lower.strike,
                middle.strike,
                upper.strike,
            )
            candidate = _candidate(
                mechanism="strike_convexity",
                contracts=(lower, middle, upper),
                roles=("buy", "sell", "buy"),
                weights=weights,
            )
            if candidate is not None:
                positive.append(candidate)
    positive.sort(
        key=lambda candidate: (
            candidate.gross_credit_quote,
            candidate.symbols,
        ),
        reverse=True,
    )
    return OptionParityDiscovery(
        evaluated_vertical_count=evaluated_vertical,
        executable_vertical_count=executable_vertical,
        evaluated_convexity_count=evaluated_convexity,
        executable_convexity_count=executable_convexity,
        gross_positive_candidates=tuple(positive),
    )


def _walk_depth(levels: Sequence[OptionBookLevel], quantity: Decimal) -> Decimal | None:
    remaining = _positive_decimal(quantity, name="option fill quantity")
    quote = Decimal("0")
    for raw_level in levels:
        level = raw_level.validated()
        consumed = min(remaining, level.quantity)
        quote += consumed * level.price
        remaining -= consumed
        if remaining == 0:
            return quote
    return None


def confirm_option_candidate(
    candidate: OptionParityCandidate,
    depths: Mapping[str, OptionDepthQuote],
) -> OptionParityConfirmation:
    """Reprice a candidate against its exact displayed depth without fees."""

    if not candidate.symbols or not (
        len(candidate.symbols) == len(candidate.roles) == len(candidate.quantities)
    ):
        raise ValueError("option parity candidate shape is invalid")
    books: list[OptionDepthQuote] = []
    gross = Decimal("0")
    for symbol, role, quantity in zip(
        candidate.symbols,
        candidate.roles,
        candidate.quantities,
        strict=True,
    ):
        if role not in {"buy", "sell"}:
            raise ValueError("option parity candidate role is invalid")
        raw_book = depths.get(symbol)
        if raw_book is None:
            raise ValueError(f"option depth is absent for {symbol}")
        book = raw_book.validated()
        if book.symbol != symbol:
            raise ValueError("option depth symbol does not match candidate")
        books.append(book)
        fill = _walk_depth(book.asks if role == "buy" else book.bids, quantity)
        if fill is None:
            return OptionParityConfirmation(
                executable=False,
                gross_credit_quote=None,
                book_event_times_ms=tuple(item.event_time_ms for item in books),
            )
        gross += -fill if role == "buy" else fill
    return OptionParityConfirmation(
        executable=True,
        gross_credit_quote=gross,
        book_event_times_ms=tuple(book.event_time_ms for book in books),
    )


__all__ = [
    "OptionBookLevel",
    "OptionContractQuote",
    "OptionDepthQuote",
    "OptionParityCandidate",
    "OptionParityConfirmation",
    "OptionParityDiscovery",
    "confirm_option_candidate",
    "discover_option_parity",
    "minimum_ratio_quantities",
    "primitive_convexity_weights",
]

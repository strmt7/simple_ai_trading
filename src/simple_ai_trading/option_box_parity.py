"""Fixed-payoff box-spread screens for European unit-one option chains."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Mapping, Sequence, cast

from .option_parity import (
    OptionContractQuote,
    OptionDepthQuote,
    OptionParityCandidate,
    confirm_option_candidate,
    minimum_ratio_quantities,
)


MILLISECONDS_PER_YEAR = Decimal("31536000000")


@dataclass(frozen=True, slots=True)
class OptionBoxCandidate:
    """One minimum-lot box with a positive ticker-only expiry surplus."""

    kind: str
    underlying: str
    expiry_date_ms: int
    lower_strike: Decimal
    upper_strike: Decimal
    symbols: tuple[str, str, str, str]
    roles: tuple[str, str, str, str]
    quantity: Decimal
    fixed_expiry_cashflow_quote: Decimal
    initial_credit_quote: Decimal
    gross_expiry_profit_quote: Decimal
    annualized_simple_return: Decimal | None


@dataclass(frozen=True, slots=True)
class OptionBoxDiscovery:
    """Aggregate ticker result for fixed-payoff long and short boxes."""

    chain_count: int
    evaluated_strike_pair_count: int
    executable_long_box_count: int
    executable_short_box_count: int
    nominal_positive_long_boxes: tuple[OptionBoxCandidate, ...]
    strict_positive_short_boxes: tuple[OptionBoxCandidate, ...]

    @property
    def candidates(self) -> tuple[OptionBoxCandidate, ...]:
        return self.strict_positive_short_boxes + self.nominal_positive_long_boxes


@dataclass(frozen=True, slots=True)
class OptionBoxConfirmation:
    """Displayed-depth repricing of one fixed-payoff candidate."""

    executable: bool
    initial_credit_quote: Decimal | None
    gross_expiry_profit_quote: Decimal | None
    book_event_times_ms: tuple[int, ...]


def _candidate(
    *,
    kind: str,
    lower_call: OptionContractQuote,
    upper_call: OptionContractQuote,
    upper_put: OptionContractQuote,
    lower_put: OptionContractQuote,
    as_of_ms: int,
) -> OptionBoxCandidate | None:
    legs = (lower_call, upper_call, upper_put, lower_put)
    quantity = minimum_ratio_quantities((1, 1, 1, 1), legs)[0]
    fixed = (upper_call.strike - lower_call.strike) * quantity
    roles: tuple[str, str, str, str]
    if kind == "short":
        roles = ("sell", "buy", "sell", "buy")
    elif kind == "long":
        roles = ("buy", "sell", "buy", "sell")
    else:
        raise ValueError("option box kind is invalid")

    credit = Decimal("0")
    for contract, role in zip(legs, roles, strict=True):
        price = cast(
            Decimal,
            contract.bid_price if role == "sell" else contract.ask_price,
        )
        credit += (price if role == "sell" else -price) * quantity
    profit = credit - fixed if kind == "short" else credit + fixed
    if profit <= 0:
        return None

    annualized: Decimal | None = None
    if kind == "long":
        cost = -credit
        time_to_expiry_ms = Decimal(lower_call.expiry_date_ms - as_of_ms)
        if cost <= 0 or time_to_expiry_ms <= 0:
            raise ValueError("positive long box has invalid cost or expiry")
        annualized = profit / cost * MILLISECONDS_PER_YEAR / time_to_expiry_ms
    return OptionBoxCandidate(
        kind=kind,
        underlying=lower_call.underlying,
        expiry_date_ms=lower_call.expiry_date_ms,
        lower_strike=lower_call.strike,
        upper_strike=upper_call.strike,
        symbols=tuple(contract.symbol for contract in legs),
        roles=roles,
        quantity=quantity,
        fixed_expiry_cashflow_quote=fixed,
        initial_credit_quote=credit,
        gross_expiry_profit_quote=profit,
        annualized_simple_return=annualized,
    )


def discover_option_boxes(
    contracts: Sequence[OptionContractQuote],
    *,
    as_of_ms: int,
) -> OptionBoxDiscovery:
    """Find ticker-only long and strict short boxes without a market forecast."""

    if isinstance(as_of_ms, bool) or not isinstance(as_of_ms, int) or as_of_ms <= 0:
        raise ValueError("option box as-of time must be a positive integer")
    normalized = tuple(contract.validated() for contract in contracts)
    if not normalized or len({contract.symbol for contract in normalized}) != len(
        normalized
    ):
        raise ValueError("option box contracts must be nonempty and unique")
    if any(contract.unit != 1 for contract in normalized):
        raise ValueError("option box screen requires unit-one contracts")

    grouped: dict[tuple[str, int], dict[Decimal, dict[str, OptionContractQuote]]] = {}
    for contract in normalized:
        sides = grouped.setdefault(
            (contract.underlying, contract.expiry_date_ms), {}
        ).setdefault(contract.strike, {})
        if contract.side in sides:
            raise ValueError("option box strike side is duplicated")
        sides[contract.side] = contract

    evaluated = 0
    executable_long = 0
    executable_short = 0
    long_boxes: list[OptionBoxCandidate] = []
    short_boxes: list[OptionBoxCandidate] = []
    for (_underlying, expiry), strike_map in grouped.items():
        if expiry <= as_of_ms:
            continue
        strikes = sorted(
            strike
            for strike, sides in strike_map.items()
            if set(sides) == {"CALL", "PUT"}
        )
        for lower_strike, upper_strike in combinations(strikes, 2):
            evaluated += 1
            lower_call = strike_map[lower_strike]["CALL"]
            lower_put = strike_map[lower_strike]["PUT"]
            upper_call = strike_map[upper_strike]["CALL"]
            upper_put = strike_map[upper_strike]["PUT"]
            if all(
                price is not None
                for price in (
                    lower_call.ask_price,
                    upper_call.bid_price,
                    upper_put.ask_price,
                    lower_put.bid_price,
                )
            ):
                executable_long += 1
                candidate = _candidate(
                    kind="long",
                    lower_call=lower_call,
                    upper_call=upper_call,
                    upper_put=upper_put,
                    lower_put=lower_put,
                    as_of_ms=as_of_ms,
                )
                if candidate is not None:
                    long_boxes.append(candidate)
            if all(
                price is not None
                for price in (
                    lower_call.bid_price,
                    upper_call.ask_price,
                    upper_put.bid_price,
                    lower_put.ask_price,
                )
            ):
                executable_short += 1
                candidate = _candidate(
                    kind="short",
                    lower_call=lower_call,
                    upper_call=upper_call,
                    upper_put=upper_put,
                    lower_put=lower_put,
                    as_of_ms=as_of_ms,
                )
                if candidate is not None:
                    short_boxes.append(candidate)
    long_boxes.sort(
        key=lambda candidate: (
            candidate.annualized_simple_return or Decimal("0"),
            candidate.symbols,
        ),
        reverse=True,
    )
    short_boxes.sort(
        key=lambda candidate: (
            candidate.gross_expiry_profit_quote,
            candidate.symbols,
        ),
        reverse=True,
    )
    return OptionBoxDiscovery(
        chain_count=len(grouped),
        evaluated_strike_pair_count=evaluated,
        executable_long_box_count=executable_long,
        executable_short_box_count=executable_short,
        nominal_positive_long_boxes=tuple(long_boxes),
        strict_positive_short_boxes=tuple(short_boxes),
    )


def confirm_option_box(
    candidate: OptionBoxCandidate,
    depths: Mapping[str, OptionDepthQuote],
) -> OptionBoxConfirmation:
    """Reprice one box at displayed depth while retaining its fixed cashflow."""

    parity_candidate = OptionParityCandidate(
        mechanism=f"{candidate.kind}_box",
        symbols=candidate.symbols,
        roles=candidate.roles,
        strikes=(
            candidate.lower_strike,
            candidate.upper_strike,
            candidate.upper_strike,
            candidate.lower_strike,
        ),
        integer_weights=(1, 1, 1, 1),
        quantities=(candidate.quantity,) * 4,
        gross_credit_quote=candidate.initial_credit_quote,
    )
    confirmation = confirm_option_candidate(parity_candidate, depths)
    if not confirmation.executable or confirmation.gross_credit_quote is None:
        return OptionBoxConfirmation(
            executable=False,
            initial_credit_quote=None,
            gross_expiry_profit_quote=None,
            book_event_times_ms=confirmation.book_event_times_ms,
        )
    credit = confirmation.gross_credit_quote
    profit = (
        credit - candidate.fixed_expiry_cashflow_quote
        if candidate.kind == "short"
        else credit + candidate.fixed_expiry_cashflow_quote
    )
    return OptionBoxConfirmation(
        executable=True,
        initial_credit_quote=credit,
        gross_expiry_profit_quote=profit,
        book_event_times_ms=confirmation.book_event_times_ms,
    )


__all__ = [
    "OptionBoxCandidate",
    "OptionBoxConfirmation",
    "OptionBoxDiscovery",
    "confirm_option_box",
    "discover_option_boxes",
]

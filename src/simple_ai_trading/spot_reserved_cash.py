"""Supplied spot cash availability after open-order quote reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import re

from .spot_inventory_cash import SpotInventoryCashPath, cash_input_digest
from .spot_prefunding import SpotPrefunding, evaluate_spot_prefunding


@dataclass(frozen=True)
class SpotQuoteReservations:
    """Declared maximum locked quote while each fill-prefix cash stays constant.

    Entry zero covers the initial state through the first fill. Entry i covers
    the state after fill i through the next fill, or the declared end for the
    last entry. Every within-interval reservation peak must be included, even
    with no fill. These are maximum remaining locks, not original order totals
    added again after a partial fill. A cancel request is not release evidence.

    The source and spot-input digests bind a supplied declaration; they do not
    establish venue origin, coverage, timing or authoritative cancellation.
    Missing/unknown intervals must not be silently declared zero. Non-fill cash
    changes and delayed proceeds require another cash model.
    """

    spot_input_sha256: str
    maximum_locked_quote_by_prefix: tuple[Fraction, ...]
    source_body_sha256: str

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (self.spot_input_sha256, self.source_body_sha256)
            )
            or not isinstance(self.maximum_locked_quote_by_prefix, tuple)
            or not self.maximum_locked_quote_by_prefix
            or any(
                not isinstance(value, Fraction) or value < 0
                for value in self.maximum_locked_quote_by_prefix
            )
        ):
            raise ValueError("supplied quote reservation declaration is invalid")


@dataclass(frozen=True)
class SpotReservedCash:
    """Necessary available-cash constraint under explicit supplied lock maxima."""

    fill_only: SpotPrefunding
    available_quote_lower_bound_by_prefix: tuple[Fraction, ...]
    minimum_available_quote: Fraction
    binding_fill_prefix: int
    minimum_initial_quote_cash: Fraction
    additional_quote_cash_required: Fraction
    reservation_increment_over_fill_only: Fraction
    supplied_cash_covers_declared_locks_and_buffer: bool
    reservation_source_sha256: str
    input_sha256: str
    native_reservations_qualified: bool = field(default=False, init=False)
    continuous_solvency_qualified: bool = field(default=False, init=False)
    training_admitted: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


def evaluate_spot_reserved_cash(
    path: SpotInventoryCashPath,
    reservations: SpotQuoteReservations,
    *,
    initial_quote_cash: Fraction,
    required_quote_buffer: Fraction,
) -> SpotReservedCash:
    """Subtract remaining locks from wallet cash, never from portfolio wealth.

    With fill-prefix net cash D_i and supplied maximum remaining quote lock L_i,
    required initial cash is B + max_i(L_i - D_i). A lock consumes availability,
    not wealth: neither a release nor cancellation creates income. Immediate
    fill settlement and no other wallet debits/credits are explicit conditions.
    This does not reserve funds or authorize order submission in a live account.
    """
    if not isinstance(reservations, SpotQuoteReservations):
        raise ValueError("validated supplied quote reservations required")
    fill_only = evaluate_spot_prefunding(
        path,
        initial_quote_cash=initial_quote_cash,
        required_quote_buffer=required_quote_buffer,
    )
    if reservations.spot_input_sha256 != fill_only.spot_input_sha256:
        raise ValueError("quote reservations are bound to a different spot path")
    locks = reservations.maximum_locked_quote_by_prefix
    if len(locks) != len(path.trades) + 1:
        raise ValueError("quote reservations must cover every fill prefix")

    cash = initial_quote_cash
    available = [cash - locks[0]]
    for trade, locked in zip(path.trades, locks[1:], strict=True):
        cash += trade.quote_delta
        available.append(cash - locked)
    minimum = min(available)
    required = required_quote_buffer + initial_quote_cash - minimum
    if required < fill_only.minimum_initial_quote_cash:
        raise ValueError("reservation accounting reduced necessary fill-only cash")
    binding = cash_input_digest(
        {
            "schema": "supplied-spot-reserved-cash-v1",
            "fill_prefunding_sha256": fill_only.input_sha256,
            "spot_input_sha256": reservations.spot_input_sha256,
            "maximum_locked_quote_by_prefix": locks,
            "reservation_source_sha256": reservations.source_body_sha256,
        }
    )
    return SpotReservedCash(
        fill_only=fill_only,
        available_quote_lower_bound_by_prefix=tuple(available),
        minimum_available_quote=minimum,
        binding_fill_prefix=available.index(minimum),
        minimum_initial_quote_cash=required,
        additional_quote_cash_required=max(Fraction(0), required - initial_quote_cash),
        reservation_increment_over_fill_only=(
            required - fill_only.minimum_initial_quote_cash
        ),
        supplied_cash_covers_declared_locks_and_buffer=minimum >= required_quote_buffer,
        reservation_source_sha256=reservations.source_body_sha256,
        input_sha256=binding,
    )

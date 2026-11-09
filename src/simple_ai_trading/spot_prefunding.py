"""Necessary spot cash for supplied fills, separately from wealth and margin."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .hedged_inventory_cash import (
    HedgedInventoryCashPath,
    HedgedInventoryCashReplay,
    replay_hedged_inventory_cash,
)
from .spot_inventory_cash import (
    SpotInventoryCashPath,
    SpotInventoryCashReplay,
    cash_input_digest,
    replay_spot_inventory_cash,
)


SPOT_PREFUNDING_SCHEMA = "supplied-spot-prefunding-v1"


@dataclass(frozen=True)
class SpotPrefunding:
    """Exact fill-prefix cash under immediate settlement and no outside credit.

    No derivative mark, inventory appreciation or opportunity-cost adjustment
    supplies quote cash here. Unfilled-order locks, financing provenance, delayed
    proceeds, external debits and venue reserves require separate qualification.
    """

    symbol: str
    initial_quote_cash: Fraction
    required_quote_buffer: Fraction
    initial_marked_wealth: Fraction
    quote_cash_after_boundary: tuple[Fraction, ...]
    minimum_quote_cash: Fraction
    minimum_cash_execution_id: str | None
    minimum_initial_quote_cash: Fraction
    additional_quote_cash_required: Fraction
    supplied_cash_covers_every_prefix: bool
    spot_input_sha256: str
    input_sha256: str
    capital_source_qualified: bool = field(default=False, init=False)
    venue_margin_qualified: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


@dataclass(frozen=True)
class HedgedSpotFundingCheck:
    """A paired wealth replay and its necessary spot-only cash constraint."""

    cash_replay: HedgedInventoryCashReplay
    spot_funding: SpotPrefunding
    input_sha256: str
    perpetual_margin_assessed: bool = field(default=False, init=False)
    continuous_solvency_qualified: bool = field(default=False, init=False)
    training_admitted: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


def _cash_prefixes(
    path: SpotInventoryCashPath,
    replay: SpotInventoryCashReplay,
    initial_quote_cash: Fraction,
    required_quote_buffer: Fraction,
) -> SpotPrefunding:
    for value in (initial_quote_cash, required_quote_buffer):
        if not isinstance(value, Fraction) or value < 0:
            raise ValueError("quote cash and buffer must be nonnegative Fractions")
    cash = minimum = initial_quote_cash
    minimum_execution_id = None
    balances = []
    index = 0
    for boundary in range(len(path.boundary_time_ms)):
        while (
            index < len(path.trades) and path.trades[index].boundary_index == boundary
        ):
            trade = path.trades[index]
            cash += trade.quote_delta
            if cash < minimum:
                minimum, minimum_execution_id = cash, trade.execution_id
            index += 1
        balances.append(cash)
    if cash - initial_quote_cash != replay.recorded_quote_cash_change:
        raise ValueError("spot prefix cash does not conserve the existing cash replay")
    required = required_quote_buffer + initial_quote_cash - minimum
    binding = cash_input_digest(
        {
            "schema": SPOT_PREFUNDING_SCHEMA,
            "spot_input_sha256": replay.input_sha256,
            "initial_quote_cash": initial_quote_cash,
            "required_quote_buffer": required_quote_buffer,
        }
    )
    return SpotPrefunding(
        symbol=path.symbol,
        initial_quote_cash=initial_quote_cash,
        required_quote_buffer=required_quote_buffer,
        initial_marked_wealth=initial_quote_cash + replay.initial_marked_inventory,
        quote_cash_after_boundary=tuple(balances),
        minimum_quote_cash=minimum,
        minimum_cash_execution_id=minimum_execution_id,
        minimum_initial_quote_cash=required,
        additional_quote_cash_required=max(Fraction(0), required - initial_quote_cash),
        supplied_cash_covers_every_prefix=minimum >= required_quote_buffer,
        spot_input_sha256=replay.input_sha256,
        input_sha256=binding,
    )


def evaluate_spot_prefunding(
    path: SpotInventoryCashPath,
    *,
    initial_quote_cash: Fraction,
    required_quote_buffer: Fraction,
) -> SpotPrefunding:
    """Evaluate all ordered fill prefixes, including intermediate boundary fills.

    Cash equal to the supplied buffer is feasible under this explicit cash model;
    it does not prove strict safety or executable order reservation. Initial
    marked inventory remains wealth, while only supplied quote cash can pay buys.
    """
    replay = replay_spot_inventory_cash(path)
    return _cash_prefixes(path, replay, initial_quote_cash, required_quote_buffer)


def evaluate_hedged_spot_prefunding(
    path: HedgedInventoryCashPath,
    *,
    initial_spot_quote_cash: Fraction,
    required_spot_quote_buffer: Fraction,
) -> HedgedSpotFundingCheck:
    """Keep hedge wealth and spot funding separate; never assume cross-wallet credit.

    Financing/opportunity debits in the paired wealth model are not silently
    charged as spot wallet transfers. Their actual settlement wallet and clock,
    and every derivative collateral requirement, still need explicit evidence.
    """
    cash = replay_hedged_inventory_cash(path)
    spot = _cash_prefixes(
        path.spot, cash.spot, initial_spot_quote_cash, required_spot_quote_buffer
    )
    return HedgedSpotFundingCheck(
        cash,
        spot,
        cash_input_digest(
            {
                "paired_input_sha256": cash.input_sha256,
                "spot_funding_sha256": spot.input_sha256,
            }
        ),
    )

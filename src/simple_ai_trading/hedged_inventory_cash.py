"""Compose supplied spot and linear-perpetual cash without hiding orphan risk."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

from .funding_cash_inventory import (
    InventoryCashPath,
    InventoryCashReplay,
    replay_fixed_base_inventory,
)
from .spot_inventory_cash import (
    SpotInventoryCashPath,
    SpotInventoryCashReplay,
    _amount,
    _digest,
    cash_input_digest,
    replay_spot_inventory_cash,
)


@dataclass(frozen=True)
class HedgedInventoryCashPath:
    perpetual: InventoryCashPath
    spot: SpotInventoryCashPath
    financing_debit_quote: tuple[Fraction, ...]
    opportunity_debit_quote: tuple[Fraction, ...]
    other_debit_quote: tuple[Fraction, ...]
    cost_source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.perpetual, InventoryCashPath)
            or not isinstance(self.spot, SpotInventoryCashPath)
            or self.perpetual.symbol != self.spot.symbol
            or self.perpetual.boundary_time_ms != self.spot.boundary_time_ms
            or self.perpetual.initial_signed_base_quantity > 0
            or any(p > 0 for p in self.perpetual.desired_position)
            or not _digest(self.cost_source_sha256)
            or any(
                not isinstance(costs, tuple)
                or len(costs) != len(self.perpetual.desired_position)
                or any(not _amount(cost) for cost in costs)
                for costs in (
                    self.financing_debit_quote,
                    self.opportunity_debit_quote,
                    self.other_debit_quote,
                )
            )
        ):
            raise ValueError(
                "paired same-USDT long-spot/short-perpetual path is invalid"
            )


@dataclass(frozen=True)
class HedgeCashInterval:
    entry_time_ms: int
    exit_time_ms: int
    spot_base_quantity: Fraction
    perpetual_base_quantity: Fraction
    residual_base_quantity: Fraction
    spot_cash_change: Fraction
    external_debit_quote: Fraction
    net_cash_lower: Fraction
    net_cash_upper: Fraction


@dataclass(frozen=True)
class HedgedInventoryCashReplay:
    symbol: str
    payment_asset: str
    spot: SpotInventoryCashReplay
    perpetual: InventoryCashReplay
    intervals: tuple[HedgeCashInterval, ...]
    boundary_orphan_base_ranges: tuple[tuple[Fraction, Fraction], ...]
    total_external_debit_quote: Fraction
    total_net_cash_lower: Fraction
    total_net_cash_upper: Fraction
    base_matched_in_every_supplied_interval: bool
    input_sha256: str
    financially_qualified: bool = field(default=False, init=False)
    venue_margin_qualified: bool = field(default=False, init=False)
    training_admitted: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


def replay_hedged_inventory_cash(
    path: HedgedInventoryCashPath,
) -> HedgedInventoryCashReplay:
    """Retain both legs' units, funding bounds, costs and non-atomic boundaries.

    Prices, incremental spot fills and costs are supplied declarations, not
    authenticated executions. Perpetual execution costs remain the existing
    proportional model. Spot trade adjustments include actual declared gross
    cash, base/quote fees and deviation from the supplied reference once.
    An interval match is not atomic entry/exit, solvency or a native hedge.
    Opportunity debits must already compare equal initial wealth at one horizon;
    this function cannot infer the feasible alternative or collateral eligibility.
    """
    if not isinstance(path, HedgedInventoryCashPath):
        raise ValueError("validated paired inventory path required")
    spot = replay_spot_inventory_cash(path.spot)
    perpetual = replay_fixed_base_inventory(path.perpetual)
    intervals, orphan_ranges = [], []
    for index, row in enumerate(perpetual.intervals):
        spot_cash = (
            spot.price_cash_by_interval[index]
            + spot.execution_adjustment_by_boundary[index]
        )
        if index == len(perpetual.intervals) - 1:
            spot_cash += spot.execution_adjustment_by_boundary[-1]
        debit = (
            path.financing_debit_quote[index]
            + path.opportunity_debit_quote[index]
            + path.other_debit_quote[index]
        )
        quantity = spot.quantity_by_interval[index]
        intervals.append(
            HedgeCashInterval(
                row.entry_time_ms,
                row.exit_time_ms,
                quantity,
                row.signed_base_quantity,
                quantity + row.signed_base_quantity,
                spot_cash,
                debit,
                spot_cash + row.net_cash_lower - debit,
                spot_cash + row.net_cash_upper - debit,
            )
        )
    for index, (spot_low, spot_high) in enumerate(spot.quantity_range_by_boundary):
        before = (
            perpetual.intervals[index - 1].signed_base_quantity
            if index
            else path.perpetual.initial_signed_base_quantity
        )
        after = (
            perpetual.intervals[index].signed_base_quantity
            if index < len(perpetual.intervals)
            else Fraction(0)
        )
        orphan_ranges.append(
            (spot_low + min(before, after), spot_high + max(before, after))
        )
    debit = sum((row.external_debit_quote for row in intervals), Fraction(0))
    lower = sum((row.net_cash_lower for row in intervals), Fraction(0))
    upper = sum((row.net_cash_upper for row in intervals), Fraction(0))
    if (
        lower != spot.incremental_wealth_change + perpetual.total_net_cash_lower - debit
        or upper
        != spot.incremental_wealth_change + perpetual.total_net_cash_upper - debit
    ):
        raise ValueError("paired cash conservation failed")
    return HedgedInventoryCashReplay(
        path.perpetual.symbol,
        "USDT",
        spot,
        perpetual,
        tuple(intervals),
        tuple(orphan_ranges),
        debit,
        lower,
        upper,
        all(row.residual_base_quantity == 0 for row in intervals),
        cash_input_digest(asdict(path)),
    )

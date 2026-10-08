"""Exact conditional fixed-base inventory cash, not qualified account P&L."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
import re

from .funding_cash import (
    FundingEntitlement,
    LinearFundingSettlement,
    USDM_SYMBOLS,
    funding_cash_bounds_for_events,
    reconcile_funding_event_population,
)


@dataclass(frozen=True)
class InventoryCashPath:
    symbol: str
    boundary_time_ms: tuple[int, ...]
    boundary_price: tuple[Fraction, ...]
    desired_position: tuple[int, ...]
    opening_quote_notional: Fraction
    one_way_cost_fraction: Fraction
    price_source_sha256: str
    settlements: tuple[LinearFundingSettlement, ...]
    expected_events: tuple[tuple[int, Fraction], ...]
    coverage_start_ms: int
    coverage_end_exclusive_ms: int
    population_certificate_sha256: str

    def __post_init__(self) -> None:
        if (
            self.symbol not in USDM_SYMBOLS
            or any(
                not isinstance(value, tuple)
                for value in (
                    self.boundary_time_ms,
                    self.boundary_price,
                    self.desired_position,
                    self.settlements,
                    self.expected_events,
                )
            )
            or not self.desired_position
            or len(self.boundary_time_ms) != len(self.desired_position) + 1
            or len(self.boundary_price) != len(self.boundary_time_ms)
            or any(type(time) is not int or time < 0 for time in self.boundary_time_ms)
            or any(
                right <= left
                for left, right in zip(
                    self.boundary_time_ms[:-1], self.boundary_time_ms[1:], strict=True
                )
            )
            or any(
                not isinstance(price, Fraction) or price <= 0
                for price in self.boundary_price
            )
            or any(
                type(position) is not int or position not in (-1, 0, 1)
                for position in self.desired_position
            )
            or not isinstance(self.opening_quote_notional, Fraction)
            or self.opening_quote_notional <= 0
            or not isinstance(self.one_way_cost_fraction, Fraction)
            or self.one_way_cost_fraction < 0
            or type(self.coverage_start_ms) is not int
            or type(self.coverage_end_exclusive_ms) is not int
            or not 0 <= self.coverage_start_ms <= self.boundary_time_ms[0]
            or not self.boundary_time_ms[-1] < self.coverage_end_exclusive_ms < 2**63
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (
                    self.price_source_sha256,
                    self.population_certificate_sha256,
                )
            )
        ):
            raise ValueError("inventory cash path contract is invalid")
        reconcile_funding_event_population(
            self.settlements, self.expected_events, expected_symbol=self.symbol
        )
        if any(
            not self.coverage_start_ms
            <= event.funding_time_ms
            < self.coverage_end_exclusive_ms
            for event in self.settlements
        ):
            raise ValueError("inventory funding event is outside its certified scope")


@dataclass(frozen=True)
class InventoryFundingCash:
    funding_time_ms: int
    interval_index: int
    quantity_before: Fraction
    quantity_after: Fraction
    cash_lower: Fraction
    cash_upper: Fraction
    boundary_quantity_uncertain: bool
    source_body_sha256: str


@dataclass(frozen=True)
class InventoryCashInterval:
    entry_time_ms: int
    exit_time_ms: int
    signed_base_quantity: Fraction
    entry_quantity_change: Fraction
    entry_traded_quote: Fraction
    terminal_traded_quote: Fraction
    price_cash: Fraction
    funding_cash_lower: Fraction
    funding_cash_upper: Fraction
    execution_cost: Fraction
    net_cash_lower: Fraction
    net_cash_upper: Fraction


@dataclass(frozen=True)
class InventoryCashReplay:
    symbol: str
    payment_asset: str
    intervals: tuple[InventoryCashInterval, ...]
    funding_events: tuple[InventoryFundingCash, ...]
    total_price_cash: Fraction
    total_execution_cost: Fraction
    total_net_cash_lower: Fraction
    total_net_cash_upper: Fraction
    total_traded_quote: Fraction
    price_source_sha256: str
    population_certificate_sha256: str


def replay_fixed_base_inventory(path: InventoryCashPath) -> InventoryCashReplay:
    """Keep quantity until a direction change, charging every modeled trade.

    Opening/reversal quote size is explicit, not a leverage or capital gate.
    Native fills, fees, financing, margin and capture origin are not qualified.
    Funding at a modeled quantity-change boundary encloses old, new and flat
    inventory, once; continuous interior holdings use the supplied event law.
    """
    if not isinstance(path, InventoryCashPath):
        raise ValueError("inventory cash replay requires a validated path")
    quantities: list[Fraction] = []
    changes: list[Fraction] = []
    traded: list[Fraction] = []
    previous_position = 0
    previous_quantity = Fraction(0)
    for position, price in zip(
        path.desired_position, path.boundary_price[:-1], strict=True
    ):
        quantity = (
            previous_quantity
            if position == previous_position
            else position * path.opening_quote_notional / price
        )
        change = quantity - previous_quantity
        quantities.append(quantity)
        changes.append(change)
        traded.append(abs(change) * price)
        previous_position, previous_quantity = position, quantity
    lower = [Fraction(0)] * len(quantities)
    upper = [Fraction(0)] * len(quantities)
    funding: list[InventoryFundingCash] = []
    for event in path.settlements:
        time = event.funding_time_ms
        if time < path.boundary_time_ms[0] or time > path.boundary_time_ms[-1]:
            continue
        boundary = bisect_left(path.boundary_time_ms, time)
        at_boundary = (
            boundary < len(path.boundary_time_ms)
            and path.boundary_time_ms[boundary] == time
        )
        interval = min(
            len(quantities) - 1, bisect_right(path.boundary_time_ms, time) - 1
        )
        if at_boundary:
            before = quantities[boundary - 1] if boundary else Fraction(0)
            after = quantities[boundary] if boundary < len(quantities) else Fraction(0)
        else:
            before = after = quantities[interval]
        uncertain = before != after
        candidates = {before, after, Fraction(0)} if uncertain else {after}
        payments = [
            Fraction(0)
            if quantity == 0
            else funding_cash_bounds_for_events(
                (event,),
                (FundingEntitlement.HELD,),
                expected_symbol=path.symbol,
                signed_base_quantity=quantity,
                entry_price=Fraction(1),
            ).cash_lower
            for quantity in candidates
        ]
        cash_low, cash_high = min(payments), max(payments)
        lower[interval] += cash_low
        upper[interval] += cash_high
        funding.append(
            InventoryFundingCash(
                time,
                interval,
                before,
                after,
                cash_low,
                cash_high,
                uncertain,
                event.source_body_sha256,
            )
        )
    intervals: list[InventoryCashInterval] = []
    for index, quantity in enumerate(quantities):
        entry_price, exit_price = path.boundary_price[index : index + 2]
        terminal_traded = (
            abs(quantity) * exit_price if index == len(quantities) - 1 else Fraction(0)
        )
        cost = (traded[index] + terminal_traded) * path.one_way_cost_fraction
        price_cash = quantity * (exit_price - entry_price)
        intervals.append(
            InventoryCashInterval(
                path.boundary_time_ms[index],
                path.boundary_time_ms[index + 1],
                quantity,
                changes[index],
                traded[index],
                terminal_traded,
                price_cash,
                lower[index],
                upper[index],
                cost,
                price_cash + lower[index] - cost,
                price_cash + upper[index] - cost,
            )
        )
    return InventoryCashReplay(
        symbol=path.symbol,
        payment_asset="USDT",
        intervals=tuple(intervals),
        funding_events=tuple(funding),
        total_price_cash=sum((row.price_cash for row in intervals), Fraction(0)),
        total_execution_cost=sum(
            (row.execution_cost for row in intervals), Fraction(0)
        ),
        total_net_cash_lower=sum(
            (row.net_cash_lower for row in intervals), Fraction(0)
        ),
        total_net_cash_upper=sum(
            (row.net_cash_upper for row in intervals), Fraction(0)
        ),
        total_traded_quote=sum(
            (row.entry_traded_quote + row.terminal_traded_quote for row in intervals),
            Fraction(0),
        ),
        price_source_sha256=path.price_source_sha256,
        population_certificate_sha256=path.population_certificate_sha256,
    )

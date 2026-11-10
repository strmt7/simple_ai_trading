"""Exact necessary capital bounds for a supplied linear opportunity charge.

The caller supplies comparable cash units, costs, duration convention and a
minimum capital amount. No FX, funding entitlement, leverage availability,
margin sufficiency or market edge is inferred from a positive budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class CapitalBudgetBounds:
    valuation_label: str
    cash_after_reserved_costs: Fraction
    charge_per_unit_capital: Fraction
    strict_capital_upper_bound: Fraction
    minimum_capital: Fraction
    net_at_minimum_capital: Fraction
    necessary_condition_passed: bool
    native_capital_qualified: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


def evaluate_capital_budget(
    *,
    valuation_label: str,
    gross_cash: Fraction,
    reserved_cost_cash: Fraction,
    minimum_capital: Fraction,
    annual_capital_charge: Fraction,
    duration_ms: int,
    annual_charge_basis_ms: int,
) -> CapitalBudgetBounds:
    """Solve gross-costs-charge*capital > 0 under the supplied fixed law.

    A nonempty interval above the supplied minimum is necessary only. A negative
    upper bound means no nonnegative capital can clear this law. Equality fails
    the strict-positive condition. Negative gross cash remains an actual debit.
    The result does not authorize replacing unknown fees or collateral with zero.
    """
    if (
        not isinstance(valuation_label, str)
        or not valuation_label.strip()
        or len(valuation_label) > 160
    ):
        raise ValueError("explicit comparable-cash valuation label required")
    for value in (
        gross_cash,
        reserved_cost_cash,
        minimum_capital,
        annual_capital_charge,
    ):
        if not isinstance(value, Fraction):
            raise ValueError("exact Fraction capital inputs required")
    if reserved_cost_cash < 0 or minimum_capital < 0 or annual_capital_charge <= 0:
        raise ValueError("nonnegative cost/capital and positive annual charge required")
    if (
        type(duration_ms) is not int
        or type(annual_charge_basis_ms) is not int
        or duration_ms <= 0
        or annual_charge_basis_ms <= 0
    ):
        raise ValueError("positive integer duration and annual charge basis required")
    budget = gross_cash - reserved_cost_cash
    coefficient = annual_capital_charge * Fraction(duration_ms, annual_charge_basis_ms)
    net = budget - coefficient * minimum_capital
    return CapitalBudgetBounds(
        valuation_label,
        budget,
        coefficient,
        budget / coefficient,
        minimum_capital,
        net,
        net > 0,
    )

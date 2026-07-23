"""Frozen financial mechanics for Round 73 executable quote-path targets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import math
from typing import Literal, Sequence


ROUND73_TARGET_SCHEMA_VERSION = "round-073-executable-target-v1"
ROUND73_TARGET_CONTRACT_SHA256 = (
    "3c4c6f0f10505897f71930245ae3252eccfe1ccac90217e6c07151a2416e2253"
)
ROUND73_TARGET_ENTRY_DELAYS_MS = (500, 1_000)
ROUND73_TARGET_HORIZONS_MS = (1_000, 5_000, 15_000)
ROUND73_TARGET_REFERENCE_NOTIONALS = (100.0, 1_000.0, 5_000.0)
ROUND73_TARGET_SIDES = ("long", "short")
ROUND73_TARGET_LEVELS = 20
ROUND73_TARGET_MAX_STATE_LATENESS_NS = 250_000_000
ROUND73_TARGET_MINIMUM_CHARGE_BPS = 12.0

TargetSide = Literal["long", "short"]


def _positive_decimal(value: object, label: str) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be decimal") from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return selected


@dataclass(frozen=True)
class Round73MarketQuantityRules:
    symbol: str
    step_size: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    minimum_notional: Decimal

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        step_size: object,
        minimum_quantity: object,
        maximum_quantity: object,
        minimum_notional: object,
    ) -> Round73MarketQuantityRules:
        selected = str(symbol).strip().upper()
        if selected not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            raise ValueError("Round 73 target quantity rules symbol differs")
        step = _positive_decimal(step_size, "market quantity step")
        minimum = _positive_decimal(minimum_quantity, "minimum market quantity")
        maximum = _positive_decimal(maximum_quantity, "maximum market quantity")
        notional = _positive_decimal(minimum_notional, "minimum notional")
        if minimum > maximum:
            raise ValueError("minimum market quantity exceeds maximum")
        if (minimum / step).to_integral_value() != minimum / step:
            raise ValueError("minimum market quantity is not step aligned")
        return cls(
            symbol=selected,
            step_size=step,
            minimum_quantity=minimum,
            maximum_quantity=maximum,
            minimum_notional=notional,
        )

    def quantize_reference_quantity(
        self,
        *,
        reference_quote_notional: float,
        decision_mid: float,
    ) -> float | None:
        quote = _positive_decimal(reference_quote_notional, "reference notional")
        mid = _positive_decimal(decision_mid, "decision mid")
        raw_steps = (quote / mid / self.step_size).to_integral_value(
            rounding=ROUND_FLOOR
        )
        quantity = raw_steps * self.step_size
        if quantity < self.minimum_quantity or quantity > self.maximum_quantity:
            return None
        return float(quantity)

    def is_step_aligned(self, quantity: float) -> bool:
        try:
            selected = Decimal(str(quantity))
        except (InvalidOperation, ValueError):
            return False
        if not selected.is_finite() or selected <= 0:
            return False
        steps = selected / self.step_size
        return steps == steps.to_integral_value()


@dataclass(frozen=True)
class Round73BookWalk:
    requested_base_quantity: float
    filled_base_quantity: float
    quote_notional: float
    average_price: float
    worst_price: float
    level_count: int
    available_base_quantity: float
    available_quote_notional: float

    @property
    def capacity_ratio(self) -> float:
        return self.available_quote_notional / self.quote_notional


def walk_round73_book(
    levels: Sequence[tuple[float, float]],
    *,
    base_quantity: float,
    ascending_prices: bool,
) -> Round73BookWalk | None:
    """Walk a fixed visible ladder without inventing partial-fill completion."""

    requested = float(base_quantity)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("Round 73 target base quantity must be finite and positive")
    selected = tuple((float(price), float(quantity)) for price, quantity in levels)
    if len(selected) != ROUND73_TARGET_LEVELS:
        raise ValueError("Round 73 target walk requires exactly 20 visible levels")
    if any(
        not math.isfinite(price)
        or not math.isfinite(quantity)
        or price <= 0
        or quantity <= 0
        for price, quantity in selected
    ):
        raise ValueError("Round 73 target ladder values must be positive and finite")
    prices = tuple(price for price, _quantity in selected)
    if ascending_prices:
        if any(right <= left for left, right in zip(prices, prices[1:], strict=False)):
            raise ValueError("Round 73 ask ladder is not strictly ascending")
    elif any(
        right >= left for left, right in zip(prices, prices[1:], strict=False)
    ):
        raise ValueError("Round 73 bid ladder is not strictly descending")
    available_base = math.fsum(quantity for _price, quantity in selected)
    available_quote = math.fsum(price * quantity for price, quantity in selected)
    if available_base + max(1e-15, requested * 1e-12) < requested:
        return None
    remaining = requested
    fill_parts: list[float] = []
    quote_parts: list[float] = []
    worst_price = 0.0
    level_count = 0
    for price, quantity in selected:
        fill = min(remaining, quantity)
        if fill > 0:
            fill_parts.append(fill)
            quote_parts.append(price * fill)
            remaining -= fill
            worst_price = price
            level_count += 1
        if remaining <= max(1e-15, requested * 1e-12):
            remaining = 0.0
            break
    if remaining > 0:
        return None
    filled = math.fsum(fill_parts)
    quote = math.fsum(quote_parts)
    if not math.isclose(filled, requested, rel_tol=1e-12, abs_tol=1e-15):
        raise ArithmeticError("Round 73 target book walk quantity differs")
    if quote <= 0 or not math.isfinite(quote):
        raise ArithmeticError("Round 73 target book walk quote value differs")
    return Round73BookWalk(
        requested_base_quantity=requested,
        filled_base_quantity=filled,
        quote_notional=quote,
        average_price=quote / filled,
        worst_price=worst_price,
        level_count=level_count,
        available_base_quantity=available_base,
        available_quote_notional=available_quote,
    )


@dataclass(frozen=True)
class Round73TargetPayoff:
    gross_payoff_quote: float
    charge_quote: float
    net_payoff_quote: float
    net_payoff_bps: float
    positive_net_payoff: bool


def round73_target_payoff(
    *,
    side: TargetSide,
    entry_quote_notional: float,
    exit_quote_notional: float,
    charge_bps: float = ROUND73_TARGET_MINIMUM_CHARGE_BPS,
) -> Round73TargetPayoff:
    selected_side = str(side)
    if selected_side not in ROUND73_TARGET_SIDES:
        raise ValueError("Round 73 target side differs")
    entry = float(entry_quote_notional)
    exit_value = float(exit_quote_notional)
    charge_rate = float(charge_bps)
    if (
        not math.isfinite(entry)
        or not math.isfinite(exit_value)
        or not math.isfinite(charge_rate)
        or entry <= 0
        or exit_value <= 0
        or charge_rate < ROUND73_TARGET_MINIMUM_CHARGE_BPS
    ):
        raise ValueError("Round 73 target payoff input differs")
    gross = exit_value - entry if selected_side == "long" else entry - exit_value
    charge = charge_rate / 10_000.0 * ((entry + exit_value) / 2.0)
    net = gross - charge
    net_bps = net / entry * 10_000.0
    values = (gross, charge, net, net_bps)
    if not all(math.isfinite(value) for value in values) or charge < 0:
        raise ArithmeticError("Round 73 target payoff is nonfinite")
    return Round73TargetPayoff(
        gross_payoff_quote=gross,
        charge_quote=charge,
        net_payoff_quote=net,
        net_payoff_bps=net_bps,
        positive_net_payoff=net > 0.0,
    )


__all__ = [
    "ROUND73_TARGET_CONTRACT_SHA256",
    "ROUND73_TARGET_ENTRY_DELAYS_MS",
    "ROUND73_TARGET_HORIZONS_MS",
    "ROUND73_TARGET_LEVELS",
    "ROUND73_TARGET_MAX_STATE_LATENESS_NS",
    "ROUND73_TARGET_MINIMUM_CHARGE_BPS",
    "ROUND73_TARGET_REFERENCE_NOTIONALS",
    "ROUND73_TARGET_SCHEMA_VERSION",
    "ROUND73_TARGET_SIDES",
    "Round73BookWalk",
    "Round73MarketQuantityRules",
    "Round73TargetPayoff",
    "round73_target_payoff",
    "walk_round73_book",
]

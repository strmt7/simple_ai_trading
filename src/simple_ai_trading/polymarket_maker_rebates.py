"""Exact conditional Polymarket maker-rebate economics for filled orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .polymarket_liquidity_rewards import PairedBuyEconomics, paired_buy_economics


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class FilledMakerRebateEconomics:
    """Unrounded nominal rebate conditional on one maker order being filled."""

    quantity: Decimal
    price: Decimal
    taker_fee_rate: Decimal
    rebate_fraction: Decimal
    fee_equivalent: Decimal
    nominal_maker_rebate: Decimal


@dataclass(frozen=True, slots=True)
class PairedFilledMakerRebateEconomics:
    """Complete-set economics conditional on both physical maker bids filling."""

    settlement: PairedBuyEconomics
    up_fill: FilledMakerRebateEconomics
    down_fill: FilledMakerRebateEconomics
    nominal_total_maker_rebate: Decimal
    nominal_both_fill_profit_including_rebates: Decimal


def filled_maker_rebate_economics(
    *,
    quantity: Decimal,
    price: Decimal,
    taker_fee_rate: Decimal,
    rebate_fraction: Decimal,
) -> FilledMakerRebateEconomics:
    """Apply the documented fee-equivalent and rebate-pool algebra unrounded."""

    size = _decimal(quantity, name="maker fill quantity", positive=True)
    fill_price = _decimal(price, name="maker fill price", positive=True)
    fee_rate = _decimal(taker_fee_rate, name="taker fee rate", positive=True)
    fraction = _decimal(rebate_fraction, name="maker rebate fraction", positive=True)
    if fill_price >= 1:
        raise ValueError("maker fill price must be inside (0, 1)")
    if fee_rate > 1 or fraction > 1:
        raise ValueError("maker fee or rebate fraction exceeds one")
    fee_equivalent = size * fee_rate * fill_price * (Decimal("1") - fill_price)
    return FilledMakerRebateEconomics(
        quantity=size,
        price=fill_price,
        taker_fee_rate=fee_rate,
        rebate_fraction=fraction,
        fee_equivalent=fee_equivalent,
        nominal_maker_rebate=fee_equivalent * fraction,
    )


def paired_filled_maker_rebate_economics(
    *,
    up_price: Decimal,
    down_price: Decimal,
    quantity: Decimal,
    taker_fee_rate: Decimal,
    rebate_fraction: Decimal,
) -> PairedFilledMakerRebateEconomics:
    """Combine equal-size paired BUY settlement and conditional fill rebates."""

    settlement = paired_buy_economics(
        yes_price=up_price,
        no_price=down_price,
        quantity=quantity,
    )
    if settlement.combined_price >= 1:
        raise ValueError("paired maker bids must remain below one")
    up = filled_maker_rebate_economics(
        quantity=quantity,
        price=up_price,
        taker_fee_rate=taker_fee_rate,
        rebate_fraction=rebate_fraction,
    )
    down = filled_maker_rebate_economics(
        quantity=quantity,
        price=down_price,
        taker_fee_rate=taker_fee_rate,
        rebate_fraction=rebate_fraction,
    )
    total = up.nominal_maker_rebate + down.nominal_maker_rebate
    return PairedFilledMakerRebateEconomics(
        settlement=settlement,
        up_fill=up,
        down_fill=down,
        nominal_total_maker_rebate=total,
        nominal_both_fill_profit_including_rebates=(
            settlement.both_fill_gross_profit + total
        ),
    )


__all__ = [
    "FilledMakerRebateEconomics",
    "PairedFilledMakerRebateEconomics",
    "filled_maker_rebate_economics",
    "paired_filled_maker_rebate_economics",
]

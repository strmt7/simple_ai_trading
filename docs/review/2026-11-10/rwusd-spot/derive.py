"""Reproduce the constructed RWUSD exit comparison without network access."""

from fractions import Fraction
import json


def terminal_spot_cash(
    *,
    quantity: Fraction,
    bid: Fraction,
    quote_fee: Fraction,
    usdc_per_usdt: Fraction,
    usdc_growth: Fraction,
) -> Fraction:
    return quantity * bid * (1 - quote_fee) * usdc_per_usdt * usdc_growth


def constructed_exit_comparison() -> dict[str, str]:
    """Compare full exits of the same already-held units in terminal USDC."""
    quantity = Fraction(1000)
    redemption = Fraction("999.50")
    bid = Fraction("0.998")
    fee = Fraction("0.001")
    base = terminal_spot_cash(
        quantity=quantity,
        bid=bid,
        quote_fee=fee,
        usdc_per_usdt=Fraction(1),
        usdc_growth=Fraction(1),
    )
    stressed = terminal_spot_cash(
        quantity=quantity,
        bid=bid,
        quote_fee=fee,
        usdc_per_usdt=Fraction("1.002"),
        usdc_growth=Fraction("1.001"),
    )
    usdc_at_unit_bid = stressed / bid
    return {
        "net_redemption_cash_usdc": str(redemption),
        "spot_cash_at_par_without_time_value_usdc": str(base),
        "spot_cash_with_conversion_and_time_value_usdc": str(stressed),
        "redemption_advantage_at_par_without_time_value_usdc": str(redemption - base),
        "redemption_advantage_with_conversion_and_time_value_usdc": str(
            redemption - stressed
        ),
        "strict_break_even_spot_bid_usdt_per_rwusd": str(redemption / usdc_at_unit_bid),
    }


if __name__ == "__main__":
    print(json.dumps(constructed_exit_comparison(), sort_keys=True))

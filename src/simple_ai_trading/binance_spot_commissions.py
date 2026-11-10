"""Exact Spot commission components and possible payment assets.

Parsed account data is not a freshness, wallet-allocation or FX certificate.
The returned discount is a multiplier on standard commission only. Never infer
current rates from documentation examples or apply side rates a second time to
the distinct order-test response schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import re

from .assets import symbol_base_for_supported_quote
from .binance_terminal_fills import _decimal


def _rate(value: object) -> Fraction:
    try:
        rate = Fraction(_decimal(value))
    except ValueError:
        raise ValueError(
            "commission coefficient requires an exact nonnegative decimal string"
        ) from None
    if rate > 1:
        raise ValueError("unsupported commission coefficient above one")
    return rate


@dataclass(frozen=True, slots=True)
class SpotCommissionComponent:
    maker: Fraction
    taker: Fraction
    buyer: Fraction
    seller: Fraction

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Fraction) or not 0 <= value <= 1
            for value in (self.maker, self.taker, self.buyer, self.seller)
        ):
            raise ValueError(
                "commission components require exact nonnegative fractions"
            )

    def for_fill(self, *, side: str, role: str) -> Fraction:
        if side not in ("BUY", "SELL") or role not in ("maker", "taker"):
            raise ValueError("commission fill side or liquidity role is invalid")
        return getattr(self, role) + (self.buyer if side == "BUY" else self.seller)


@dataclass(frozen=True, slots=True)
class SpotCommissionDiscount:
    enabled_for_account: bool
    enabled_for_symbol: bool
    asset: str | None
    multiplier: Fraction | None

    def __post_init__(self) -> None:
        if (
            type(self.enabled_for_account) is not bool
            or type(self.enabled_for_symbol) is not bool
        ):
            raise ValueError("commission discount flags must be explicit booleans")
        if self.asset is not None and (
            not isinstance(self.asset, str)
            or re.fullmatch(r"[A-Z0-9]{1,32}", self.asset) is None
        ):
            raise ValueError("commission discount asset is invalid")
        if self.multiplier is not None and (
            not isinstance(self.multiplier, Fraction) or not 0 <= self.multiplier <= 1
        ):
            raise ValueError("commission discount multiplier is invalid")
        if self.eligible and (self.asset is None or self.multiplier is None):
            raise ValueError(
                "commission discount flags, asset or multiplier are invalid"
            )

    @property
    def eligible(self) -> bool:
        # Eligibility alone does not prove sufficient owned discount-asset funds.
        return self.enabled_for_account and self.enabled_for_symbol


@dataclass(frozen=True, slots=True)
class SpotCommissionBreakdown:
    standard: Fraction
    tax: Fraction
    special: Fraction

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Fraction) or value < 0
            for value in (self.standard, self.tax, self.special)
        ):
            raise ValueError("commission breakdown requires exact nonnegative amounts")

    @property
    def total(self) -> Fraction:
        return self.standard + self.tax + self.special


@dataclass(frozen=True, slots=True)
class SpotCommissionSchedule:
    symbol: str
    standard: SpotCommissionComponent
    tax: SpotCommissionComponent
    special: SpotCommissionComponent
    discount: SpotCommissionDiscount

    def __post_init__(self) -> None:
        if (
            not isinstance(self.symbol, str)
            or re.fullmatch(r"[A-Z0-9]{1,32}", self.symbol) is None
            or not all(
                isinstance(value, SpotCommissionComponent)
                for value in (self.standard, self.tax, self.special)
            )
            or not isinstance(self.discount, SpotCommissionDiscount)
        ):
            raise ValueError("commission schedule identity or components are invalid")

    def equivalent_rate(
        self, *, side: str, role: str, apply_discount: bool
    ) -> SpotCommissionBreakdown:
        """Return costs in received-asset-equivalent units, before native rounding.

        In the discount branch all components require conversion to the payment
        asset; this method does not compute that FX conversion or authorize its
        balance. Tax and special fees receive no multiplier. Ineligible flags
        cannot be overridden by selecting the discounted calculation.
        """
        if (
            type(apply_discount) is not bool
            or apply_discount
            and not self.discount.eligible
        ):
            raise ValueError(
                "discounted commission is not enabled for this account/symbol"
            )
        standard = self.standard.for_fill(side=side, role=role)
        multiplier = self.discount.multiplier if apply_discount else Fraction(1)
        if multiplier is None:
            raise ValueError("discount multiplier is unknown")
        return SpotCommissionBreakdown(
            standard * multiplier,
            self.tax.for_fill(side=side, role=role),
            self.special.for_fill(side=side, role=role),
        )

    def possible_fee_assets(self, *, side: str) -> frozenset[str]:
        """Include both eligible payment and received-asset fallback, not a guess."""
        base = symbol_base_for_supported_quote(self.symbol)
        if not base or side not in ("BUY", "SELL"):
            raise ValueError(
                "commission payment assets require a supported quote and side"
            )
        received = base if side == "BUY" else self.symbol[len(base) :]
        assets = {received}
        if self.discount.eligible:
            if self.discount.asset is None:
                raise ValueError("eligible discount payment asset is unknown")
            assets.add(self.discount.asset)
        return frozenset(assets)


def parse_spot_commission_schedule(
    payload: Mapping[str, object], *, symbol: str
) -> SpotCommissionSchedule:
    """Require rates and flags; preserve irrelevant disabled-route unknowns as None."""
    if not isinstance(payload, Mapping) or payload.get("symbol") != symbol:
        raise ValueError("commission response symbol differs from the request")

    def component(name: str) -> SpotCommissionComponent:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise ValueError("commission response lacks a required component")
        return SpotCommissionComponent(
            *(
                _rate(value.get(field))
                for field in ("maker", "taker", "buyer", "seller")
            )
        )

    discount = payload.get("discount")
    if not isinstance(discount, Mapping):
        raise ValueError("commission response lacks discount-routing evidence")
    return SpotCommissionSchedule(
        symbol,
        component("standardCommission"),
        component("taxCommission"),
        component("specialCommission"),
        SpotCommissionDiscount(
            discount.get("enabledForAccount"),
            discount.get("enabledForSymbol"),
            discount.get("discountAsset"),
            _rate(discount["discount"])
            if discount.get("discount") is not None
            else None,
        ),
    )

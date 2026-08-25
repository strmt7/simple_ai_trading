"""Exact binary implication bundles for structural prediction-market screens."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Sequence

from .paper_execution import BookLevel
from .polymarket_fees import PolymarketFeeModel


_THRESHOLD_PATTERNS = (
    (
        "above",
        re.compile(
            r"^(?:Will the price of )?(Bitcoin|Ethereum|Solana)(?: be)? above "
            r"\$?([0-9][0-9,]*(?:\.[0-9]+)?) (on .+)\?$",
            re.IGNORECASE,
        ),
    ),
    (
        "reach",
        re.compile(
            r"^Will (Bitcoin|Ethereum|Solana) (?:reach|hit) "
            r"\$?([0-9][0-9,]*(?:\.[0-9]+)?)(k)? ((?:by|in|on) .+)\?$",
            re.IGNORECASE,
        ),
    ),
    (
        "dip",
        re.compile(
            r"^Will (Bitcoin|Ethereum|Solana) dip to "
            r"\$?([0-9][0-9,]*(?:\.[0-9]+)?)(k)? ((?:by|in|on) .+)\?$",
            re.IGNORECASE,
        ),
    ),
)


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"{name} must be a {qualifier}decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class CryptoThresholdQuestion:
    """A narrowly parsed threshold question with a deterministic implication order."""

    kind: str
    asset: str
    threshold: Decimal
    window: str

    def validated(self) -> "CryptoThresholdQuestion":
        kind = str(self.kind or "").strip().lower()
        asset = str(self.asset or "").strip().upper()
        window = " ".join(str(self.window or "").strip().lower().split())
        threshold = _decimal(
            self.threshold,
            name="logical threshold",
            positive=True,
        )
        if kind not in {"above", "reach", "dip"}:
            raise ValueError("logical threshold kind is unsupported")
        if asset not in {"BTC", "ETH", "SOL"}:
            raise ValueError("logical threshold asset is unsupported")
        if not window or len(window) > 200:
            raise ValueError("logical threshold window is invalid")
        return CryptoThresholdQuestion(kind, asset, threshold, window)

    def stronger_than(self, other: "CryptoThresholdQuestion") -> bool:
        """Return whether this proposition strictly implies ``other``."""

        left = self.validated()
        right = other.validated()
        if (left.kind, left.asset, left.window) != (
            right.kind,
            right.asset,
            right.window,
        ):
            return False
        if left.kind == "dip":
            return left.threshold < right.threshold
        return left.threshold > right.threshold


def parse_crypto_threshold_question(question: str) -> CryptoThresholdQuestion | None:
    """Parse only explicit BTC/ETH/SOL threshold forms used by the screen.

    Ambiguous forms, including binary ``X or Y first`` markets, return ``None``.
    """

    text = str(question or "").strip()
    for kind, pattern in _THRESHOLD_PATTERNS:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        groups = match.groups()
        asset_name = groups[0].lower()
        asset = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}[asset_name]
        threshold = Decimal(groups[1].replace(",", ""))
        if len(groups) == 4 and groups[2]:
            threshold *= Decimal("1000")
        return CryptoThresholdQuestion(
            kind=kind,
            asset=asset,
            threshold=threshold,
            window=groups[-1],
        ).validated()
    return None


@dataclass(frozen=True, slots=True)
class LogicalBinaryOutcome:
    """Displayed asks for one binary proposition in an implication bundle."""

    label: str
    yes_asks: tuple[BookLevel, ...]
    no_asks: tuple[BookLevel, ...]
    fee_model: PolymarketFeeModel

    def validated(self) -> "LogicalBinaryOutcome":
        label = str(self.label or "").strip()
        if not label or len(label) > 500:
            raise ValueError("logical outcome label is invalid")

        def levels(raw: Sequence[BookLevel]) -> tuple[BookLevel, ...]:
            parsed = tuple(level.validated() for level in raw)
            if tuple(sorted(parsed, key=lambda level: level.price)) != parsed:
                raise ValueError("logical outcome asks are not price-sorted")
            if len({level.price for level in parsed}) != len(parsed):
                raise ValueError("logical outcome ask prices are duplicated")
            if any(level.price >= 1 for level in parsed):
                raise ValueError("logical outcome ask price must lie below one")
            return parsed

        return LogicalBinaryOutcome(
            label=label,
            yes_asks=levels(self.yes_asks),
            no_asks=levels(self.no_asks),
            fee_model=self.fee_model,
        )


@dataclass(frozen=True, slots=True)
class LogicalImplicationBundle:
    """Buy YES(weaker) plus NO(stronger), whose terminal floor is one share."""

    weaker_label: str
    stronger_label: str
    quantity: Decimal
    gross_cost_quote: Decimal
    taker_fees_quote: Decimal
    initial_outlay_quote: Decimal
    terminal_payout_floor_quote: Decimal
    net_quote: Decimal


def _walk_asks(
    levels: Sequence[BookLevel],
    *,
    quantity: Decimal,
    fee_model: PolymarketFeeModel,
) -> tuple[Decimal, Decimal] | None:
    remaining = quantity
    gross = Decimal("0")
    fee = Decimal("0")
    for level in levels:
        consumed = min(remaining, level.quantity)
        gross += consumed * level.price
        fee += fee_model(level.price, consumed, "taker")
        remaining -= consumed
        if remaining == 0:
            return gross, fee
    return None


def screen_logical_implication_bundle(
    weaker: LogicalBinaryOutcome,
    stronger: LogicalBinaryOutcome,
    *,
    quantity: Decimal,
) -> LogicalImplicationBundle | None:
    """Price the only long-only guaranteed bundle induced by ``stronger => weaker``."""

    requested = _decimal(quantity, name="logical bundle quantity", positive=True)
    weak = weaker.validated()
    strong = stronger.validated()
    if weak.label == strong.label:
        raise ValueError("logical implication labels must differ")
    yes_fill = _walk_asks(
        weak.yes_asks,
        quantity=requested,
        fee_model=weak.fee_model,
    )
    no_fill = _walk_asks(
        strong.no_asks,
        quantity=requested,
        fee_model=strong.fee_model,
    )
    if yes_fill is None or no_fill is None:
        return None
    gross = yes_fill[0] + no_fill[0]
    fees = yes_fill[1] + no_fill[1]
    outlay = gross + fees
    return LogicalImplicationBundle(
        weaker_label=weak.label,
        stronger_label=strong.label,
        quantity=requested,
        gross_cost_quote=gross,
        taker_fees_quote=fees,
        initial_outlay_quote=outlay,
        terminal_payout_floor_quote=requested,
        net_quote=requested - outlay,
    )


__all__ = [
    "CryptoThresholdQuestion",
    "LogicalBinaryOutcome",
    "LogicalImplicationBundle",
    "parse_crypto_threshold_question",
    "screen_logical_implication_bundle",
]

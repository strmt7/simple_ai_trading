"""Target-free payoff-parity screens for structural Polymarket mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import combinations
from itertools import permutations
from typing import Sequence

from .paper_execution import BookLevel
from .polymarket_fees import PolymarketFeeModel


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
class StructuralParityFill:
    """One exact displayed-depth fill, including the recorded taker fee curve."""

    gross_quote: Decimal
    taker_fee_quote: Decimal

    @property
    def buy_cost(self) -> Decimal:
        return self.gross_quote + self.taker_fee_quote

    @property
    def sell_value(self) -> Decimal:
        return self.gross_quote - self.taker_fee_quote


@dataclass(frozen=True, slots=True)
class NegativeRiskOutcome:
    """Displayed books for one binary question in a fixed negative-risk event."""

    label: str
    yes_bids: tuple[BookLevel, ...]
    yes_asks: tuple[BookLevel, ...]
    no_asks: tuple[BookLevel, ...]
    fee_model: PolymarketFeeModel

    def validated(self) -> "NegativeRiskOutcome":
        label = str(self.label or "").strip()
        if not label or len(label) > 500:
            raise ValueError("negative-risk outcome label is invalid")

        def levels(
            raw: Sequence[BookLevel], *, descending: bool
        ) -> tuple[BookLevel, ...]:
            parsed = tuple(level.validated() for level in raw)
            if (
                tuple(sorted(parsed, key=lambda level: level.price, reverse=descending))
                != parsed
            ):
                raise ValueError("negative-risk book levels are not price-sorted")
            if len({level.price for level in parsed}) != len(parsed):
                raise ValueError("negative-risk book prices are duplicated")
            if any(level.price >= 1 for level in parsed):
                raise ValueError("negative-risk book price must lie below one")
            return parsed

        yes_bids = levels(self.yes_bids, descending=True)
        yes_asks = levels(self.yes_asks, descending=False)
        no_asks = levels(self.no_asks, descending=False)
        if yes_bids and yes_asks and yes_bids[0].price >= yes_asks[0].price:
            raise ValueError("negative-risk YES book is crossed or locked")
        return NegativeRiskOutcome(
            label=label,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            no_asks=no_asks,
            fee_model=self.fee_model,
        )


@dataclass(frozen=True, slots=True)
class NegativeRiskParityPath:
    """An optimistic structural path before gas and multi-leg execution risk."""

    mechanism: str
    selected_no_outcomes: tuple[str, ...]
    net_quote: Decimal
    taker_fees_quote: Decimal
    initial_outlay_quote: Decimal


@dataclass(frozen=True, slots=True)
class NegativeRiskParityScreen:
    """Compact result for all payoff-equivalent paths at one displayed state."""

    quantity: Decimal
    evaluated_path_count: int
    executable_path_count: int
    profitable_path_count: int
    buy_all_yes_hold: NegativeRiskParityPath | None
    mint_all_yes_sell: NegativeRiskParityPath | None
    best_no_conversion: NegativeRiskParityPath | None

    @property
    def best_path(self) -> NegativeRiskParityPath | None:
        paths = tuple(
            path
            for path in (
                self.buy_all_yes_hold,
                self.mint_all_yes_sell,
                self.best_no_conversion,
            )
            if path is not None
        )
        return max(paths, key=lambda path: path.net_quote, default=None)


@dataclass(frozen=True, slots=True)
class SpotPairQuote:
    """One source-bound spot best bid and ask for a directed conversion graph."""

    symbol: str
    base_asset: str
    quote_asset: str
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal

    def validated(self) -> "SpotPairQuote":
        symbol = str(self.symbol or "").strip().upper()
        base = str(self.base_asset or "").strip().upper()
        quote = str(self.quote_asset or "").strip().upper()
        if (
            not symbol
            or len(symbol) > 40
            or not base
            or not quote
            or base == quote
            or not all(value.isalnum() for value in (symbol, base, quote))
        ):
            raise ValueError("spot pair identity is invalid")
        bid_price = _decimal(self.bid_price, name="spot bid price", positive=True)
        bid_quantity = _decimal(
            self.bid_quantity, name="spot bid quantity", positive=True
        )
        ask_price = _decimal(self.ask_price, name="spot ask price", positive=True)
        ask_quantity = _decimal(
            self.ask_quantity, name="spot ask quantity", positive=True
        )
        if bid_price >= ask_price:
            raise ValueError("spot pair is crossed or locked")
        return SpotPairQuote(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            bid_price=bid_price,
            bid_quantity=bid_quantity,
            ask_price=ask_price,
            ask_quantity=ask_quantity,
        )


@dataclass(frozen=True, slots=True)
class SpotTrianglePath:
    """Three immediately executable top-book conversions returning to start."""

    assets: tuple[str, str, str, str]
    symbols: tuple[str, str, str]
    gross_multiplier: Decimal
    after_fee_multiplier: Decimal
    optimistic_capacity_start: Decimal
    break_even_fee_bips_per_leg: Decimal

    @property
    def gross_net_bips(self) -> Decimal:
        return (self.gross_multiplier - Decimal("1")) * Decimal("10000")

    @property
    def after_fee_net_bips(self) -> Decimal:
        return (self.after_fee_multiplier - Decimal("1")) * Decimal("10000")


@dataclass(frozen=True, slots=True)
class SpotTriangleScreen:
    """Compact current-state result for every simple three-leg spot cycle."""

    taker_fee_bips_per_leg: Decimal
    evaluated_path_count: int
    gross_positive_path_count: int
    after_fee_positive_path_count: int
    best_gross_path: SpotTrianglePath | None
    best_after_fee_path: SpotTrianglePath | None


@dataclass(frozen=True, slots=True)
class _ConversionEdge:
    symbol: str
    source_asset: str
    target_asset: str
    gross_rate: Decimal
    maximum_source_quantity: Decimal


def screen_spot_triangles(
    quotes: Sequence[SpotPairQuote],
    *,
    start_assets: Sequence[str],
    taker_fee_bips_per_leg: Decimal,
) -> SpotTriangleScreen:
    """Enumerate simple three-leg spot cycles without assuming an account fee tier."""

    fee_bips = _decimal(
        taker_fee_bips_per_leg,
        name="spot taker fee bips",
    )
    if fee_bips < 0 or fee_bips >= Decimal("10000"):
        raise ValueError("spot taker fee bips is outside [0, 10000)")
    fee_multiplier = Decimal("1") - fee_bips / Decimal("10000")
    normalized = tuple(quote.validated() for quote in quotes)
    if not normalized or len({quote.symbol for quote in normalized}) != len(normalized):
        raise ValueError("spot pair symbols must be nonempty and unique")
    unordered_pairs = {
        frozenset((quote.base_asset, quote.quote_asset)) for quote in normalized
    }
    if len(unordered_pairs) != len(normalized):
        raise ValueError("spot asset pair is duplicated in another orientation")
    starts = tuple(str(asset or "").strip().upper() for asset in start_assets)
    if (
        not starts
        or len(set(starts)) != len(starts)
        or any(not asset or not asset.isalnum() for asset in starts)
    ):
        raise ValueError("spot triangle start assets are invalid")

    edges: dict[tuple[str, str], _ConversionEdge] = {}
    assets: set[str] = set()
    for quote in normalized:
        assets.update((quote.base_asset, quote.quote_asset))
        sell = _ConversionEdge(
            symbol=quote.symbol,
            source_asset=quote.base_asset,
            target_asset=quote.quote_asset,
            gross_rate=quote.bid_price,
            maximum_source_quantity=quote.bid_quantity,
        )
        buy = _ConversionEdge(
            symbol=quote.symbol,
            source_asset=quote.quote_asset,
            target_asset=quote.base_asset,
            gross_rate=Decimal("1") / quote.ask_price,
            maximum_source_quantity=quote.ask_price * quote.ask_quantity,
        )
        for edge in (sell, buy):
            key = (edge.source_asset, edge.target_asset)
            edges[key] = edge

    paths: list[SpotTrianglePath] = []
    for start in starts:
        if start not in assets:
            continue
        middle_assets = sorted(assets - {start})
        for first, second in permutations(middle_assets, 2):
            route_assets = (start, first, second, start)
            route_edges = tuple(
                edges.get((route_assets[index], route_assets[index + 1]))
                for index in range(3)
            )
            if any(edge is None for edge in route_edges):
                continue
            selected = tuple(edge for edge in route_edges if edge is not None)
            gross_multiplier = Decimal("1")
            capacity = Decimal("Infinity")
            for edge in selected:
                capacity = min(
                    capacity,
                    edge.maximum_source_quantity / gross_multiplier,
                )
                gross_multiplier *= edge.gross_rate
            after_fee_multiplier = gross_multiplier * fee_multiplier**3
            break_even = Decimal("0")
            if gross_multiplier > 1:
                break_even = (
                    Decimal("1")
                    - (Decimal("1") / gross_multiplier) ** (Decimal("1") / Decimal("3"))
                ) * Decimal("10000")
            paths.append(
                SpotTrianglePath(
                    assets=route_assets,
                    symbols=tuple(edge.symbol for edge in selected),
                    gross_multiplier=gross_multiplier,
                    after_fee_multiplier=after_fee_multiplier,
                    optimistic_capacity_start=capacity,
                    break_even_fee_bips_per_leg=break_even,
                )
            )
    return SpotTriangleScreen(
        taker_fee_bips_per_leg=fee_bips,
        evaluated_path_count=len(paths),
        gross_positive_path_count=sum(path.gross_multiplier > 1 for path in paths),
        after_fee_positive_path_count=sum(
            path.after_fee_multiplier > 1 for path in paths
        ),
        best_gross_path=max(
            paths, key=lambda path: path.gross_multiplier, default=None
        ),
        best_after_fee_path=max(
            paths, key=lambda path: path.after_fee_multiplier, default=None
        ),
    )


def walk_structural_parity_depth(
    levels: Sequence[BookLevel],
    *,
    quantity: Decimal,
    fee_model: PolymarketFeeModel,
) -> StructuralParityFill | None:
    """Walk already best-to-worst levels without inventing unavailable depth."""

    remaining = _decimal(quantity, name="structural parity quantity", positive=True)
    gross = Decimal("0")
    fee = Decimal("0")
    for raw_level in levels:
        level = raw_level.validated()
        if level.price >= 1:
            raise ValueError("structural parity price must lie below one")
        consumed = min(remaining, level.quantity)
        if consumed > 0:
            gross += consumed * level.price
            fee += fee_model(level.price, consumed, "taker")
            remaining -= consumed
        if remaining == 0:
            return StructuralParityFill(gross_quote=gross, taker_fee_quote=fee)
    return None


def screen_negative_risk_parity(
    outcomes: Sequence[NegativeRiskOutcome],
    *,
    quantity: Decimal,
    conversion_fee_bips: int,
) -> NegativeRiskParityScreen:
    """Evaluate exact fixed-event identities at displayed depth and recorded fees.

    The result is only an optimistic diagnostic. It deliberately excludes gas,
    order latency, atomicity, settlement delay, and adverse selection. A positive
    result therefore cannot by itself establish an executable or promoted edge.
    """

    requested_quantity = _decimal(
        quantity, name="negative-risk parity quantity", positive=True
    )
    fee_bips = int(conversion_fee_bips)
    if isinstance(conversion_fee_bips, bool) or fee_bips != conversion_fee_bips:
        raise ValueError("conversion fee bips must be an integer")
    if fee_bips != 0:
        raise ValueError(
            "nonzero negative-risk conversion fees require a separately verified model"
        )
    normalized = tuple(outcome.validated() for outcome in outcomes)
    if len(normalized) < 3 or len({item.label for item in normalized}) != len(
        normalized
    ):
        raise ValueError("negative-risk parity requires unique exhaustive outcomes")

    yes_buys = tuple(
        walk_structural_parity_depth(
            item.yes_asks,
            quantity=requested_quantity,
            fee_model=item.fee_model,
        )
        for item in normalized
    )
    yes_sells = tuple(
        walk_structural_parity_depth(
            item.yes_bids,
            quantity=requested_quantity,
            fee_model=item.fee_model,
        )
        for item in normalized
    )
    no_buys = tuple(
        walk_structural_parity_depth(
            item.no_asks,
            quantity=requested_quantity,
            fee_model=item.fee_model,
        )
        for item in normalized
    )

    buy_all_yes_hold: NegativeRiskParityPath | None = None
    if all(fill is not None for fill in yes_buys):
        fills = tuple(fill for fill in yes_buys if fill is not None)
        cost = sum((fill.buy_cost for fill in fills), Decimal("0"))
        buy_all_yes_hold = NegativeRiskParityPath(
            mechanism="buy_all_yes_hold_to_resolution",
            selected_no_outcomes=(),
            net_quote=requested_quantity - cost,
            taker_fees_quote=sum(
                (fill.taker_fee_quote for fill in fills), Decimal("0")
            ),
            initial_outlay_quote=cost,
        )

    mint_all_yes_sell: NegativeRiskParityPath | None = None
    if all(fill is not None for fill in yes_sells):
        fills = tuple(fill for fill in yes_sells if fill is not None)
        sale_value = sum((fill.sell_value for fill in fills), Decimal("0"))
        mint_all_yes_sell = NegativeRiskParityPath(
            mechanism="mint_all_yes_then_sell",
            selected_no_outcomes=tuple(item.label for item in normalized),
            net_quote=sale_value - requested_quantity,
            taker_fees_quote=sum(
                (fill.taker_fee_quote for fill in fills), Decimal("0")
            ),
            initial_outlay_quote=requested_quantity,
        )

    best_no_conversion: NegativeRiskParityPath | None = None
    executable_conversions = 0
    profitable_conversions = 0
    evaluated_conversions = 0
    outcome_count = len(normalized)
    for selected_count in range(1, outcome_count + 1):
        for selected_indices in combinations(range(outcome_count), selected_count):
            evaluated_conversions += 1
            selected = frozenset(selected_indices)
            selected_fills = tuple(no_buys[index] for index in selected_indices)
            complement_fills = tuple(
                yes_sells[index]
                for index in range(outcome_count)
                if index not in selected
            )
            if any(fill is None for fill in (*selected_fills, *complement_fills)):
                continue
            executable_conversions += 1
            no_fills = tuple(fill for fill in selected_fills if fill is not None)
            yes_fills = tuple(fill for fill in complement_fills if fill is not None)
            no_cost = sum((fill.buy_cost for fill in no_fills), Decimal("0"))
            sale_value = sum((fill.sell_value for fill in yes_fills), Decimal("0"))
            collateral = requested_quantity * Decimal(selected_count - 1)
            path = NegativeRiskParityPath(
                mechanism="buy_no_convert_sell_complement_yes",
                selected_no_outcomes=tuple(
                    normalized[index].label for index in selected_indices
                ),
                net_quote=collateral + sale_value - no_cost,
                taker_fees_quote=sum(
                    (fill.taker_fee_quote for fill in (*no_fills, *yes_fills)),
                    Decimal("0"),
                ),
                initial_outlay_quote=no_cost,
            )
            if path.net_quote > 0:
                profitable_conversions += 1
            if (
                best_no_conversion is None
                or path.net_quote > best_no_conversion.net_quote
            ):
                best_no_conversion = path

    direct_paths = tuple(
        path for path in (buy_all_yes_hold, mint_all_yes_sell) if path is not None
    )
    return NegativeRiskParityScreen(
        quantity=requested_quantity,
        evaluated_path_count=evaluated_conversions + 2,
        executable_path_count=executable_conversions + len(direct_paths),
        profitable_path_count=profitable_conversions
        + sum(path.net_quote > 0 for path in direct_paths),
        buy_all_yes_hold=buy_all_yes_hold,
        mint_all_yes_sell=mint_all_yes_sell,
        best_no_conversion=best_no_conversion,
    )


__all__ = [
    "NegativeRiskOutcome",
    "NegativeRiskParityPath",
    "NegativeRiskParityScreen",
    "SpotPairQuote",
    "SpotTrianglePath",
    "SpotTriangleScreen",
    "StructuralParityFill",
    "screen_negative_risk_parity",
    "screen_spot_triangles",
    "walk_structural_parity_depth",
]

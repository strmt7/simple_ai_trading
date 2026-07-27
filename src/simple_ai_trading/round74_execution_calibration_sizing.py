"""Exchange-rule and order-book sizing for Round 74 calibration pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import re

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION,
)


ROUND74_EXECUTION_SIZING_SCHEMA_VERSION = "round-074-execution-calibration-sizing-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _decimal(value: object, *, label: str, allow_zero: bool = False) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 74 execution {label} differs") from exc
    if not selected.is_finite():
        raise ValueError(f"Round 74 execution {label} differs")
    minimum_allowed = selected >= 0 if allow_zero else selected > 0
    if not minimum_allowed:
        raise ValueError(f"Round 74 execution {label} differs")
    return selected


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 74 execution {label} differs")
    return dict(value)


def _sha256(value: object, *, label: str) -> str:
    selected = str(value)
    if not _SHA256_PATTERN.fullmatch(selected):
        raise ValueError(f"Round 74 execution {label} differs")
    return selected


def _symbol(value: object) -> str:
    selected = str(value).strip().upper()
    if selected not in ROUND74_EVENT_TARGET_SYMBOLS:
        raise ValueError("Round 74 execution sizing symbol differs")
    return selected


def _side(value: object) -> str:
    selected = str(value).strip().upper()
    if selected not in {"BUY", "SELL"}:
        raise ValueError("Round 74 execution sizing side differs")
    return selected


def _filter(
    symbol_payload: Mapping[str, object],
    *,
    filter_type: str,
) -> dict[str, object]:
    filters = symbol_payload.get("filters")
    if not isinstance(filters, Sequence) or isinstance(filters, (str, bytes)):
        raise ValueError("Round 74 execution symbol filters differ")
    selected = [
        dict(value)
        for value in filters
        if isinstance(value, Mapping) and value.get("filterType") == filter_type
    ]
    if len(selected) != 1:
        raise ValueError(f"Round 74 execution {filter_type} filter differs")
    return selected[0]


def _levels(
    book: Mapping[str, object],
    *,
    side: str,
) -> tuple[tuple[Decimal, Decimal], ...]:
    key = "asks" if side == "BUY" else "bids"
    raw = book.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"Round 74 execution {key} differ")
    selected: list[tuple[Decimal, Decimal]] = []
    prior_price: Decimal | None = None
    for value in raw:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError(f"Round 74 execution {key} level differs")
        price = _decimal(value[0], label=f"{key} price")
        quantity = _decimal(value[1], label=f"{key} quantity")
        if prior_price is not None:
            ordered = price > prior_price if side == "BUY" else price < prior_price
            if not ordered:
                raise ValueError(f"Round 74 execution {key} ordering differs")
        selected.append((price, quantity))
        prior_price = price
    return tuple(selected)


def _quantity_within_quote_budget(
    levels: tuple[tuple[Decimal, Decimal], ...],
    *,
    quote_budget: Decimal,
) -> Decimal:
    remaining = quote_budget
    quantity = Decimal(0)
    for price, available_quantity in levels:
        available_quote = price * available_quantity
        if available_quote <= remaining:
            quantity += available_quantity
            remaining -= available_quote
            if remaining == 0:
                break
            continue
        quantity += remaining / price
        remaining = Decimal(0)
        break
    if remaining > 0:
        raise ValueError("Round 74 execution target exceeds captured book depth")
    return quantity


def _floor_to_lattice(
    value: Decimal,
    *,
    minimum: Decimal,
    maximum: Decimal,
    step: Decimal,
) -> Decimal:
    if value > maximum:
        raise ValueError("Round 74 execution target exceeds maximum market quantity")
    if value < minimum:
        raise ValueError("Round 74 execution target is below minimum quantity")
    steps = ((value - minimum) / step).to_integral_value(rounding=ROUND_FLOOR)
    selected = minimum + steps * step
    if selected < minimum or selected > maximum or (selected - minimum) % step != 0:
        raise ValueError("Round 74 execution quantity lattice differs")
    return selected


def _walk(
    levels: tuple[tuple[Decimal, Decimal], ...],
    *,
    quantity: Decimal,
) -> tuple[Decimal, Decimal]:
    remaining = quantity
    quote = Decimal(0)
    worst_price = levels[0][0]
    for price, available_quantity in levels:
        consumed = min(remaining, available_quantity)
        quote += consumed * price
        remaining -= consumed
        if consumed > 0:
            worst_price = price
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError(
            "Round 74 execution legal quantity exceeds captured book depth"
        )
    return quote, worst_price


@dataclass(frozen=True)
class Round74ExecutionSizingPlan:
    symbol: str
    entry_side: str
    target_quote_notional: Decimal
    quantity: Decimal
    reference_quote_notional: Decimal
    mark_price: Decimal
    best_price: Decimal
    expected_vwap: Decimal
    worst_price: Decimal
    expected_book_impact_bps: Decimal
    minimum_notional: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    quantity_step: Decimal
    exchange_information_sha256: str
    mark_price_sha256: str
    book_sha256: str
    book_update_id: int
    schema_version: str = ROUND74_EXECUTION_SIZING_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        if (
            self.schema_version != ROUND74_EXECUTION_SIZING_SCHEMA_VERSION
            or _symbol(self.symbol) != self.symbol
            or _side(self.entry_side) != self.entry_side
            or isinstance(self.book_update_id, bool)
            or not isinstance(self.book_update_id, int)
            or self.book_update_id < 0
        ):
            raise ValueError("Round 74 execution sizing plan differs")
        for label, value in (
            ("target quote notional", self.target_quote_notional),
            ("quantity", self.quantity),
            ("reference quote notional", self.reference_quote_notional),
            ("mark price", self.mark_price),
            ("best price", self.best_price),
            ("expected VWAP", self.expected_vwap),
            ("worst price", self.worst_price),
            ("minimum notional", self.minimum_notional),
            ("minimum quantity", self.minimum_quantity),
            ("maximum quantity", self.maximum_quantity),
            ("quantity step", self.quantity_step),
        ):
            _decimal(value, label=label)
        _decimal(
            self.expected_book_impact_bps,
            label="expected book impact",
            allow_zero=True,
        )
        for label, value in (
            ("exchange information hash", self.exchange_information_sha256),
            ("mark price hash", self.mark_price_sha256),
            ("book hash", self.book_sha256),
        ):
            _sha256(value, label=label)
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "entry_side": self.entry_side,
            "target_quote_notional": format(
                self.target_quote_notional,
                "f",
            ),
            "quantity": format(self.quantity, "f"),
            "reference_quote_notional": format(
                self.reference_quote_notional,
                "f",
            ),
            "mark_price": format(self.mark_price, "f"),
            "best_price": format(self.best_price, "f"),
            "expected_vwap": format(self.expected_vwap, "f"),
            "worst_price": format(self.worst_price, "f"),
            "expected_book_impact_bps": format(
                self.expected_book_impact_bps,
                "f",
            ),
            "minimum_notional": format(self.minimum_notional, "f"),
            "minimum_quantity": format(self.minimum_quantity, "f"),
            "maximum_quantity": format(self.maximum_quantity, "f"),
            "quantity_step": format(self.quantity_step, "f"),
            "exchange_information_sha256": (self.exchange_information_sha256),
            "mark_price_sha256": self.mark_price_sha256,
            "book_sha256": self.book_sha256,
            "book_update_id": self.book_update_id,
        }


def prepare_round74_execution_sizing(
    *,
    symbol: str,
    entry_side: str,
    target_quote_notional: Decimal,
    exchange_information: Mapping[str, object],
    mark_price: Mapping[str, object],
    book: Mapping[str, object],
) -> Round74ExecutionSizingPlan:
    """Derive a legal market quantity without exceeding the quote budget."""

    selected_symbol = _symbol(symbol)
    selected_side = _side(entry_side)
    target = _decimal(
        target_quote_notional,
        label="target quote notional",
    )
    exchange = _mapping(
        exchange_information,
        label="exchange information",
    )
    mark = _mapping(mark_price, label="mark price")
    selected_book = _mapping(book, label="book")
    if (
        exchange.get("schema_version") != "round-074-execution-exchange-information-v1"
        or exchange.get("symbol") != selected_symbol
        or mark.get("schema_version") != "round-074-execution-mark-price-v1"
        or mark.get("symbol") != selected_symbol
        or selected_book.get("schema_version") != "round-074-execution-book-state-v1"
        or selected_book.get("symbol") != selected_symbol
    ):
        raise ValueError("Round 74 execution sizing sources differ")
    symbol_payload = _mapping(
        exchange.get("symbol_payload"),
        label="exchange symbol",
    )
    if (
        symbol_payload.get("symbol") != selected_symbol
        or symbol_payload.get("pair") != selected_symbol
        or symbol_payload.get("contractType") != "PERPETUAL"
        or symbol_payload.get("status") != "TRADING"
        or symbol_payload.get("quoteAsset") != "USDT"
        or symbol_payload.get("marginAsset") != "USDT"
    ):
        raise ValueError("Round 74 execution contract eligibility differs")
    market_lot = _filter(symbol_payload, filter_type="MARKET_LOT_SIZE")
    minimum_notional_filter = _filter(
        symbol_payload,
        filter_type="MIN_NOTIONAL",
    )
    minimum_quantity = _decimal(
        market_lot.get("minQty"),
        label="minimum quantity",
    )
    maximum_quantity = _decimal(
        market_lot.get("maxQty"),
        label="maximum quantity",
    )
    quantity_step = _decimal(
        market_lot.get("stepSize"),
        label="quantity step",
    )
    minimum_notional = _decimal(
        minimum_notional_filter.get("notional"),
        label="minimum notional",
    )
    selected_mark_price = _decimal(
        mark.get("mark_price"),
        label="mark price",
    )
    levels = _levels(selected_book, side=selected_side)
    raw_quantity = _quantity_within_quote_budget(
        levels,
        quote_budget=target,
    )
    quantity = _floor_to_lattice(
        raw_quantity,
        minimum=minimum_quantity,
        maximum=maximum_quantity,
        step=quantity_step,
    )
    if quantity * selected_mark_price < minimum_notional:
        raise ValueError("Round 74 execution legal quantity is below minimum notional")
    reference_quote_notional, worst_price = _walk(
        levels,
        quantity=quantity,
    )
    if reference_quote_notional > target:
        raise ValueError("Round 74 execution sizing exceeds quote budget")
    relative_shortfall = (target - reference_quote_notional) / target
    if relative_shortfall > ROUND74_EXECUTION_CALIBRATION_NOTIONAL_TOLERANCE_FRACTION:
        raise ValueError(
            "Round 74 execution legal quantity exceeds calibration notional tolerance"
        )
    best_price = levels[0][0]
    expected_vwap = reference_quote_notional / quantity
    if selected_side == "BUY":
        expected_impact = (expected_vwap / best_price - 1) * Decimal(10_000)
    else:
        expected_impact = (1 - expected_vwap / best_price) * Decimal(10_000)
    if expected_impact < 0:
        raise ValueError("Round 74 execution book impact differs")
    update_id = selected_book.get("update_id")
    if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 0:
        raise ValueError("Round 74 execution book update ID differs")
    plan = Round74ExecutionSizingPlan(
        symbol=selected_symbol,
        entry_side=selected_side,
        target_quote_notional=target,
        quantity=quantity,
        reference_quote_notional=reference_quote_notional,
        mark_price=selected_mark_price,
        best_price=best_price,
        expected_vwap=expected_vwap,
        worst_price=worst_price,
        expected_book_impact_bps=expected_impact,
        minimum_notional=minimum_notional,
        minimum_quantity=minimum_quantity,
        maximum_quantity=maximum_quantity,
        quantity_step=quantity_step,
        exchange_information_sha256=_sha256(
            exchange.get("source_payload_sha256"),
            label="exchange information hash",
        ),
        mark_price_sha256=_sha256(
            mark.get("source_payload_sha256"),
            label="mark price hash",
        ),
        book_sha256=_sha256(
            selected_book.get("source_payload_sha256"),
            label="book hash",
        ),
        book_update_id=update_id,
    )
    plan.as_dict()
    return plan


__all__ = [
    "ROUND74_EXECUTION_SIZING_SCHEMA_VERSION",
    "Round74ExecutionSizingPlan",
    "prepare_round74_execution_sizing",
]

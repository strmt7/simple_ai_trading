"""Reconstruct causal Polymarket execution books from validated CLOB receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import re

from .paper_execution import BookLevel, PaperBookSnapshot
from .polymarket_redundant_union import PolymarketUnionEvent
from .polymarket_round21_core_features import validate_round21_union_event


_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_FULL_BOOK_KEYS = frozenset(
    {"event_type", "asset_id", "market", "timestamp", "hash", "bids", "asks"}
)
_FULL_BOOK_OPTIONAL_KEYS = frozenset({"tick_size", "last_trade_price"})
_PRICE_CHANGE_KEYS = frozenset(
    {"event_type", "market", "timestamp", "price_changes"}
)
_PRICE_CHANGE_ITEM_KEYS = frozenset(
    {"asset_id", "price", "size", "side", "hash", "best_bid", "best_ask"}
)
_BEST_BID_ASK_KEYS = frozenset(
    {
        "event_type",
        "market",
        "asset_id",
        "best_bid",
        "best_ask",
        "spread",
        "timestamp",
    }
)
_IGNORED_EVENT_TYPES = frozenset(
    {
        "last_trade_price",
        "new_market",
        "market_resolved",
        "tick_size_change",
    }
)


def _decimal(value: object, *, label: str, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Polymarket {label} is invalid")
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Polymarket {label} is invalid") from exc
    if not selected.is_finite() or selected < 0 or (not allow_zero and selected == 0):
        raise ValueError(f"Polymarket {label} is invalid")
    return selected


class _BookState:
    def __init__(self, token_id: str) -> None:
        self.token_id = token_id
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.valid = False

    @staticmethod
    def _levels(values: object, *, side: str) -> dict[Decimal, Decimal]:
        if not isinstance(values, list):
            raise ValueError(f"Polymarket {side} book levels are invalid")
        output: dict[Decimal, Decimal] = {}
        for item in values:
            if not isinstance(item, Mapping) or set(item) != {"price", "size"}:
                raise ValueError(f"Polymarket {side} book level schema drifted")
            price = _decimal(item["price"], label=f"{side} price")
            quantity = _decimal(item["size"], label=f"{side} quantity")
            if not Decimal("0") < price < Decimal("1") or price in output:
                raise ValueError(f"Polymarket {side} book level is invalid")
            output[price] = quantity
        return output

    def replace(self, payload: Mapping[str, object]) -> None:
        self.bids = self._levels(payload["bids"], side="bid")
        self.asks = self._levels(payload["asks"], side="ask")
        self.valid = self._valid_top()

    def _valid_top(self) -> bool:
        return bool(self.bids) and bool(self.asks) and max(self.bids) < min(self.asks)

    def apply(self, changes: Sequence[Mapping[str, object]]) -> None:
        if not self.valid:
            return
        for change in changes:
            side = str(change["side"] or "").upper()
            price = _decimal(change["price"], label="price-change price")
            quantity = _decimal(
                change["size"],
                label="price-change quantity",
                allow_zero=True,
            )
            if not Decimal("0") < price < Decimal("1") or side not in {"BUY", "SELL"}:
                raise ValueError("Polymarket price-change level is invalid")
            levels = self.bids if side == "BUY" else self.asks
            if quantity == 0:
                levels.pop(price, None)
            else:
                levels[price] = quantity
        self.valid = self._valid_top()
        if not self.valid:
            return
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        for change in changes:
            reported_bid = _decimal(
                change["best_bid"],
                label="reported best bid",
                allow_zero=True,
            )
            reported_ask = _decimal(
                change["best_ask"],
                label="reported best ask",
                allow_zero=True,
            )
            if (
                reported_bid != 0
                and reported_bid != best_bid
                or reported_ask != 0
                and reported_ask != best_ask
            ):
                self.valid = False
                return

    def snapshot(
        self,
        event: PolymarketUnionEvent,
        *,
        market_id: str,
    ) -> PaperBookSnapshot | None:
        if not self.valid or event.source_time_ms is None:
            return None
        return PaperBookSnapshot(
            venue="polymarket",
            market_id=market_id,
            asset_id=self.token_id,
            bids=tuple(
                BookLevel(price, self.bids[price])
                for price in sorted(self.bids, reverse=True)[:20]
            ),
            asks=tuple(
                BookLevel(price, self.asks[price])
                for price in sorted(self.asks)[:20]
            ),
            source_time_ms=event.source_time_ms,
            received_wall_ms=event.selected_received_wall_ms,
            received_monotonic_ns=event.selected_received_monotonic_ns,
            source_payload_sha256=event.event_sha256,
            connected=True,
            gap_free=True,
        )


def build_polymarket_execution_books(
    *,
    condition_id: str,
    up_token_id: str,
    down_token_id: str,
    union_events: Sequence[PolymarketUnionEvent],
    admitted_gap_free: bool,
) -> tuple[PaperBookSnapshot, ...]:
    """Rebuild exact top-20 books without using future receipts or model state."""

    condition = str(condition_id or "").strip().lower()
    tokens = tuple(str(value or "").strip() for value in (up_token_id, down_token_id))
    if (
        _CONDITION_ID.fullmatch(condition) is None
        or len(set(tokens)) != 2
        or any(_TOKEN_ID.fullmatch(token) is None for token in tokens)
        or admitted_gap_free is not True
    ):
        raise ValueError("Polymarket execution-book identity differs")
    states = {token: _BookState(token) for token in tokens}
    output: list[PaperBookSnapshot] = []
    identities: set[tuple[str, int, int]] = set()
    last_monotonic_ns = 0
    for event in union_events:
        payload = validate_round21_union_event(event)
        if str(payload.get("market") or "").strip().lower() != condition:
            raise ValueError("Polymarket execution-book market differs")
        if event.selected_received_monotonic_ns < last_monotonic_ns:
            raise ValueError("Polymarket execution-book receipt order regressed")
        last_monotonic_ns = event.selected_received_monotonic_ns
        updated_tokens: set[str] = set()
        if event.event_type == "book":
            keys = frozenset(payload)
            if (
                not _FULL_BOOK_KEYS.issubset(keys)
                or keys - _FULL_BOOK_KEYS - _FULL_BOOK_OPTIONAL_KEYS
            ):
                raise ValueError("Polymarket full-book schema drifted")
            token = str(payload["asset_id"] or "")
            state = states.get(token)
            if state is None:
                raise ValueError("Polymarket execution book has an unknown token")
            state.replace(payload)
            updated_tokens.add(token)
        elif event.event_type == "price_change":
            if set(payload) != _PRICE_CHANGE_KEYS:
                raise ValueError("Polymarket price-change schema drifted")
            raw_changes = payload["price_changes"]
            if not isinstance(raw_changes, list) or not raw_changes:
                raise ValueError("Polymarket price-change batch is invalid")
            grouped: dict[str, list[Mapping[str, object]]] = {}
            for change in raw_changes:
                if not isinstance(change, Mapping) or set(change) != _PRICE_CHANGE_ITEM_KEYS:
                    raise ValueError("Polymarket price-change item schema drifted")
                token = str(change["asset_id"] or "")
                if token not in states:
                    raise ValueError("Polymarket execution change has an unknown token")
                grouped.setdefault(token, []).append(change)
            for token, changes in grouped.items():
                states[token].apply(changes)
                updated_tokens.add(token)
        elif event.event_type == "best_bid_ask":
            if set(payload) != _BEST_BID_ASK_KEYS:
                raise ValueError("Polymarket best-bid-ask schema drifted")
        elif event.event_type not in _IGNORED_EVENT_TYPES:
            raise ValueError(f"Unsupported Polymarket CLOB event type: {event.event_type}")
        for token in sorted(updated_tokens):
            identity = (
                token,
                event.selected_received_wall_ms,
                event.selected_received_monotonic_ns,
            )
            if identity in identities:
                raise ValueError("Polymarket execution-book source identity differs")
            snapshot = states[token].snapshot(event, market_id=condition)
            if snapshot is None:
                continue
            identities.add(identity)
            output.append(snapshot.validated())
    if not output:
        raise ValueError("Polymarket execution-book population is empty")
    return tuple(
        sorted(
            output,
            key=lambda value: (
                value.received_wall_ms,
                value.received_monotonic_ns,
                value.asset_id,
                value.source_payload_sha256,
            ),
        )
    )


__all__ = ["build_polymarket_execution_books"]

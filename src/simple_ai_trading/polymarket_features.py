"""Leakage-safe feature rows from immutable Polymarket prospective evidence."""

from __future__ import annotations

from array import array
from bisect import bisect_right, insort
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence, TypeVar

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket_coverage import (
    PolymarketFeedCoverage,
    inspect_polymarket_feed_coverage,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
    PolymarketResolutionEvidence,
)


POLYMARKET_FEATURE_SCHEMA_VERSION = "polymarket-causal-feature-v5"
POLYMARKET_DATASET_SCHEMA_VERSION = "polymarket-causal-dataset-v5"
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
POLYMARKET_FEATURE_NAMES = (
    "elapsed_fraction",
    "remaining_seconds",
    "up_best_bid",
    "up_best_ask",
    "up_midpoint",
    "up_spread",
    "up_microprice",
    "up_top_imbalance",
    "up_bid_depth_3",
    "up_ask_depth_3",
    "down_best_bid",
    "down_best_ask",
    "down_midpoint",
    "down_spread",
    "down_microprice",
    "down_top_imbalance",
    "down_bid_depth_3",
    "down_ask_depth_3",
    "ask_pair_cost",
    "bid_pair_value",
    "up_book_age_ms",
    "down_book_age_ms",
    "chainlink_return_from_open_bps",
    "binance_distance_from_chainlink_open_bps",
    "binance_chainlink_basis_bps",
    "binance_best_bid",
    "binance_best_ask",
    "binance_spread_bps",
    "binance_top_imbalance",
    "binance_return_100ms_bps",
    "binance_return_250ms_bps",
    "binance_return_1000ms_bps",
    "binance_return_5000ms_bps",
    "binance_realized_volatility_100ms_bps",
    "binance_realized_volatility_1000ms_bps",
    "binance_realized_volatility_5000ms_bps",
    "binance_trade_imbalance_100ms",
    "binance_trade_imbalance_250ms",
    "binance_trade_imbalance_1000ms",
    "binance_trade_imbalance_5000ms",
    "log1p_binance_trade_quote_250ms",
    "log1p_binance_trade_quote_1000ms",
    "log1p_binance_trade_quote_5000ms",
    "direct_binance_age_ms",
    "chainlink_source_age_ms",
    "chainlink_arrival_age_ms",
    "chainlink_anchor_gap_ms",
    "log1p_market_liquidity_quote",
    "log1p_market_volume_quote",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _finite_decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    return float(_finite_decimal(value, name=name, positive=positive))


def _timestamp(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _ratio_imbalance(bid_quantity: float, ask_quantity: float) -> float:
    total = bid_quantity + ask_quantity
    return 0.0 if total <= 0.0 else (bid_quantity - ask_quantity) / total


def _log_basis_bps(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("price basis requires positive prices")
    return 10_000.0 * math.log(numerator / denominator)


@dataclass(frozen=True)
class PolymarketFeatureConfig:
    cadence_ms: int = 250
    warmup_ms: int = 5_000
    maximum_clob_age_ms: int = 2_000
    maximum_direct_binance_age_ms: int = 1_000
    maximum_chainlink_source_age_ms: int = 3_000
    maximum_chainlink_arrival_age_ms: int = 3_000
    maximum_chainlink_anchor_gap_ms: int = 2_000
    minimum_resolved_markets_per_asset: int = 30
    allow_segmented_gaps: bool = False

    def validated(self) -> "PolymarketFeatureConfig":
        bounds = {
            "cadence_ms": (50, 5_000),
            "warmup_ms": (0, 60_000),
            "maximum_clob_age_ms": (50, 10_000),
            "maximum_direct_binance_age_ms": (50, 10_000),
            "maximum_chainlink_source_age_ms": (250, 30_000),
            "maximum_chainlink_arrival_age_ms": (250, 30_000),
            "maximum_chainlink_anchor_gap_ms": (0, 10_000),
            "minimum_resolved_markets_per_asset": (1, 100_000),
        }
        for name, (minimum, maximum) in bounds.items():
            value = int(getattr(self, name))
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
        if not isinstance(self.allow_segmented_gaps, bool):
            raise ValueError("allow_segmented_gaps must be a boolean")
        return self

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allow_segmented_gaps"] = bool(self.allow_segmented_gaps)
        return payload


@dataclass(frozen=True)
class PolymarketFeatureRow:
    feature_id: str
    run_id: str
    condition_id: str
    market_id: str
    asset: str
    decision_event_id: str
    decision_received_wall_ms: int
    decision_received_monotonic_ns: int
    feature_values: tuple[float, ...]
    official_up: bool | None
    resolution_event_id: str
    input_provenance_sha256: str
    row_sha256: str

    def feature_map(self) -> dict[str, float]:
        return dict(zip(POLYMARKET_FEATURE_NAMES, self.feature_values, strict=True))

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_FEATURE_SCHEMA_VERSION,
            "feature_id": self.feature_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "market_id": self.market_id,
            "asset": self.asset,
            "decision_event_id": self.decision_event_id,
            "decision_received_wall_ms": self.decision_received_wall_ms,
            "decision_received_monotonic_ns": self.decision_received_monotonic_ns,
            "features": self.feature_map(),
            "official_up": self.official_up,
            "resolution_event_id": self.resolution_event_id,
            "input_provenance_sha256": self.input_provenance_sha256,
            "row_sha256": self.row_sha256,
        }

    def validated(self) -> "PolymarketFeatureRow":
        if (
            len(self.feature_id) != 64
            or len(self.row_sha256) != 64
            or len(self.input_provenance_sha256) != 64
            or not self.run_id
            or not self.condition_id
            or not self.market_id
            or self.asset not in _ASSETS
            or not self.decision_event_id
            or self.decision_received_wall_ms < 0
            or self.decision_received_monotonic_ns < 0
            or len(self.feature_values) != len(POLYMARKET_FEATURE_NAMES)
            or not all(math.isfinite(value) for value in self.feature_values)
            or not (self.official_up is None or isinstance(self.official_up, bool))
            or (self.official_up is None and bool(self.resolution_event_id))
            or (self.official_up is not None and not self.resolution_event_id)
            or self.row_sha256 != _canonical_sha256(_feature_row_payload(self))
        ):
            raise ValueError("Polymarket feature row identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketFeatureDataset:
    dataset_id: str
    run_id: str
    config: PolymarketFeatureConfig
    rows: tuple[PolymarketFeatureRow, ...]
    candidate_count: int
    skipped_counts: dict[str, int]
    labeled_market_counts: dict[str, int]
    shadow_errors: tuple[str, ...]
    training_errors: tuple[str, ...]
    replay_diagnostics: dict[str, object]
    coverage: dict[str, object]
    dataset_sha256: str

    @property
    def shadow_ready(self) -> bool:
        return not self.shadow_errors

    @property
    def training_ready(self) -> bool:
        return not self.training_errors

    @property
    def labeled_row_count(self) -> int:
        return sum(row.official_up is not None for row in self.rows)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_DATASET_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "config": self.config.asdict(),
            "feature_names": list(POLYMARKET_FEATURE_NAMES),
            "feature_count": len(POLYMARKET_FEATURE_NAMES),
            "candidate_count": self.candidate_count,
            "row_count": len(self.rows),
            "labeled_row_count": self.labeled_row_count,
            "labeled_market_counts": dict(self.labeled_market_counts),
            "skipped_counts": dict(self.skipped_counts),
            "shadow_ready": self.shadow_ready,
            "training_ready": self.training_ready,
            "shadow_errors": list(self.shadow_errors),
            "training_errors": list(self.training_errors),
            "replay_diagnostics": dict(self.replay_diagnostics),
            "coverage": dict(self.coverage),
            "dataset_sha256": self.dataset_sha256,
        }


@dataclass(frozen=True)
class PolymarketFeatureMaterialization:
    dataset_id: str
    status: str
    row_count: int
    labeled_row_count: int
    dataset_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _MarketSnapshotPoint:
    condition_id: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class _PricePoint:
    asset: str
    connection_id: str
    source_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    price: float
    event_id: str
    event_sha256: str


@dataclass(frozen=True, slots=True)
class _BinanceBookPoint:
    asset: str
    connection_id: str
    received_wall_ms: int
    received_monotonic_ns: int
    bid: float
    bid_quantity: float
    ask: float
    ask_quantity: float
    event_id: str
    event_sha256: str

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0


class _CompactBookTimeView(Sequence[int]):
    """Zero-copy receive-clock view over one compact book slice."""

    __slots__ = ("_books",)

    def __init__(self, books: _CompactBinanceBookView) -> None:
        self._books = books

    def __len__(self) -> int:
        return len(self._books)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        return self._books.received_monotonic_ns_at(index)


class _CompactBinanceBookView(Sequence[_BinanceBookPoint]):
    """Ordered zero-copy view over one connection's compact book events."""

    __slots__ = ("_books", "_connection_id", "_end", "_indices", "_start")
    is_received_ordered = True

    def __init__(
        self,
        books: _CompactBinanceBooks,
        *,
        connection_id: str,
        start: int = 0,
        end: int | None = None,
        indices: array[int] | None = None,
    ) -> None:
        self._books = books
        self._connection_id = connection_id
        self._start = int(start)
        self._end = len(books) if end is None else int(end)
        self._indices = indices
        if indices is None and not 0 <= self._start <= self._end <= len(books):
            raise ValueError("compact Binance book view bounds are invalid")

    @property
    def connection_id(self) -> str:
        return self._connection_id

    def __len__(self) -> int:
        return (
            len(self._indices) if self._indices is not None else self._end - self._start
        )

    def _absolute_index(self, index: int) -> int:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError("compact Binance book view index is outside bounds")
        return (
            int(self._indices[normalized])
            if self._indices is not None
            else self._start + normalized
        )

    def __getitem__(
        self, index: int | slice
    ) -> _BinanceBookPoint | tuple[_BinanceBookPoint, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        return self._books[self._absolute_index(index)]

    def received_monotonic_ns_at(self, index: int) -> int:
        return self._books.received_monotonic_ns_at(self._absolute_index(index))

    def event_id_at(self, index: int) -> str:
        return self._books.event_id_at(self._absolute_index(index))

    def event_sha256_at(self, index: int) -> str:
        return self._books.event_sha256_at(self._absolute_index(index))

    def midpoint_at(self, index: int) -> float:
        return self._books.midpoint_at(self._absolute_index(index))

    def times(self) -> _CompactBookTimeView:
        return _CompactBookTimeView(self)


class _CompactBinanceBooks(Sequence[_BinanceBookPoint]):
    """Memory-bounded immutable storage for high-rate Binance bookTicker rows."""

    __slots__ = (
        "_ask",
        "_ask_quantity",
        "_asset",
        "_bid",
        "_bid_quantity",
        "_connection_codes",
        "_connection_lookup",
        "_connections",
        "_event_ids",
        "_event_sha256",
        "_finished",
        "_pending",
        "_pending_ns",
        "_received_monotonic_ns",
        "_received_wall_ms",
    )
    is_received_ordered = True

    def __init__(self, asset: str) -> None:
        self._asset = str(asset)
        self._received_wall_ms = array("q")
        self._received_monotonic_ns = array("q")
        self._bid = array("d")
        self._bid_quantity = array("d")
        self._ask = array("d")
        self._ask_quantity = array("d")
        self._connection_codes = array("I")
        self._connections: list[str] = []
        self._connection_lookup: dict[str, int] = {}
        self._event_ids = bytearray()
        self._event_sha256 = bytearray()
        self._pending_ns: int | None = None
        self._pending: list[_BinanceBookPoint] = []
        self._finished = False

    def __len__(self) -> int:
        return len(self._received_monotonic_ns) + len(self._pending)

    @staticmethod
    def _digest_bytes(value: str, *, name: str) -> bytes:
        try:
            digest = bytes.fromhex(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} is not hexadecimal") from exc
        if len(digest) != 32:
            raise ValueError(f"{name} must be SHA-256")
        return digest

    def _connection_code(self, connection_id: str) -> int:
        normalized = str(connection_id)
        existing = self._connection_lookup.get(normalized)
        if existing is not None:
            return existing
        code = len(self._connections)
        if code >= 2**32:
            raise ValueError("too many Binance book connection segments")
        self._connection_lookup[normalized] = code
        self._connections.append(normalized)
        return code

    def _store(self, point: _BinanceBookPoint) -> None:
        if point.asset != self._asset:
            raise ValueError("compact Binance book asset differs")
        if (
            self._received_monotonic_ns
            and point.received_monotonic_ns < self._received_monotonic_ns[-1]
        ):
            raise ValueError("compact Binance book receive clock regressed")
        self._received_wall_ms.append(int(point.received_wall_ms))
        self._received_monotonic_ns.append(int(point.received_monotonic_ns))
        self._bid.append(float(point.bid))
        self._bid_quantity.append(float(point.bid_quantity))
        self._ask.append(float(point.ask))
        self._ask_quantity.append(float(point.ask_quantity))
        self._connection_codes.append(self._connection_code(point.connection_id))
        self._event_ids.extend(self._digest_bytes(point.event_id, name="event_id"))
        self._event_sha256.extend(
            self._digest_bytes(point.event_sha256, name="event_sha256")
        )

    def _flush_pending(self) -> None:
        for point in sorted(self._pending, key=_received_order_key):
            self._store(point)
        self._pending.clear()

    def append(self, point: _BinanceBookPoint) -> None:
        if self._finished:
            raise RuntimeError("compact Binance books are already finalized")
        received_ns = int(point.received_monotonic_ns)
        if self._pending_ns is None:
            self._pending_ns = received_ns
        elif received_ns < self._pending_ns:
            raise ValueError("compact Binance book receive clock regressed")
        elif received_ns > self._pending_ns:
            self._flush_pending()
            self._pending_ns = received_ns
        self._pending.append(point)

    def finish(self) -> _CompactBinanceBooks:
        if not self._finished:
            self._flush_pending()
            self._pending_ns = None
            self._finished = True
        return self

    def _normalized_index(self, index: int) -> int:
        if not self._finished:
            raise RuntimeError("compact Binance books are not finalized")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self._received_monotonic_ns)
        if not 0 <= normalized < len(self._received_monotonic_ns):
            raise IndexError("compact Binance book index is outside bounds")
        return normalized

    def _hex_at(self, values: bytearray, index: int) -> str:
        offset = index * 32
        return values[offset : offset + 32].hex()

    def __getitem__(
        self, index: int | slice
    ) -> _BinanceBookPoint | tuple[_BinanceBookPoint, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        position = self._normalized_index(index)
        return _BinanceBookPoint(
            asset=self._asset,
            connection_id=self.connection_id_at(position),
            received_wall_ms=int(self._received_wall_ms[position]),
            received_monotonic_ns=int(self._received_monotonic_ns[position]),
            bid=float(self._bid[position]),
            bid_quantity=float(self._bid_quantity[position]),
            ask=float(self._ask[position]),
            ask_quantity=float(self._ask_quantity[position]),
            event_id=self.event_id_at(position),
            event_sha256=self.event_sha256_at(position),
        )

    def received_monotonic_ns_at(self, index: int) -> int:
        return int(self._received_monotonic_ns[self._normalized_index(index)])

    def connection_id_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._connections[int(self._connection_codes[position])]

    def event_id_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._hex_at(self._event_ids, position)

    def event_sha256_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._hex_at(self._event_sha256, position)

    def midpoint_at(self, index: int) -> float:
        position = self._normalized_index(index)
        return (float(self._bid[position]) + float(self._ask[position])) / 2.0

    def connection_views(self) -> dict[str, _CompactBinanceBookView]:
        if not self._finished:
            raise RuntimeError("compact Binance books are not finalized")
        spans: dict[int, list[tuple[int, int]]] = {}
        start = 0
        while start < len(self._connection_codes):
            code = int(self._connection_codes[start])
            end = start + 1
            while (
                end < len(self._connection_codes)
                and int(self._connection_codes[end]) == code
            ):
                end += 1
            spans.setdefault(code, []).append((start, end))
            start = end
        views: dict[str, _CompactBinanceBookView] = {}
        for code, ranges in spans.items():
            connection_id = self._connections[code]
            if len(ranges) == 1:
                views[connection_id] = _CompactBinanceBookView(
                    self,
                    connection_id=connection_id,
                    start=ranges[0][0],
                    end=ranges[0][1],
                )
                continue
            indices = array("Q")
            for first, last in ranges:
                indices.extend(range(first, last))
            views[connection_id] = _CompactBinanceBookView(
                self,
                connection_id=connection_id,
                indices=indices,
            )
        return views


@dataclass(frozen=True, slots=True)
class _BinanceTradePoint:
    asset: str
    connection_id: str
    received_monotonic_ns: int
    signed_quote: Decimal
    gross_quote: Decimal
    event_id: str
    event_sha256: str


class _CompactTradeTimeView(Sequence[int]):
    """Zero-copy receive-clock view over one compact trade slice."""

    __slots__ = ("_trades",)

    def __init__(self, trades: _CompactBinanceTradeView) -> None:
        self._trades = trades

    def __len__(self) -> int:
        return len(self._trades)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        return self._trades.received_monotonic_ns_at(index)


class _CompactBinanceTradeView(Sequence[_BinanceTradePoint]):
    """Ordered zero-copy view over one connection's compact trade events."""

    __slots__ = ("_connection_id", "_end", "_indices", "_start", "_trades")
    is_received_ordered = True

    def __init__(
        self,
        trades: _CompactBinanceTrades,
        *,
        connection_id: str,
        start: int = 0,
        end: int | None = None,
        indices: array[int] | None = None,
    ) -> None:
        self._trades = trades
        self._connection_id = connection_id
        self._start = int(start)
        self._end = len(trades) if end is None else int(end)
        self._indices = indices
        if indices is None and not 0 <= self._start <= self._end <= len(trades):
            raise ValueError("compact Binance trade view bounds are invalid")

    @property
    def connection_id(self) -> str:
        return self._connection_id

    def __len__(self) -> int:
        return (
            len(self._indices) if self._indices is not None else self._end - self._start
        )

    def _absolute_index(self, index: int) -> int:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError("compact Binance trade view index is outside bounds")
        return (
            int(self._indices[normalized])
            if self._indices is not None
            else self._start + normalized
        )

    def __getitem__(
        self, index: int | slice
    ) -> _BinanceTradePoint | tuple[_BinanceTradePoint, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        return self._trades[self._absolute_index(index)]

    def received_monotonic_ns_at(self, index: int) -> int:
        return self._trades.received_monotonic_ns_at(self._absolute_index(index))

    def connection_id_at(self, index: int) -> str:
        return self._trades.connection_id_at(self._absolute_index(index))

    def event_id_at(self, index: int) -> str:
        return self._trades.event_id_at(self._absolute_index(index))

    def event_sha256_at(self, index: int) -> str:
        return self._trades.event_sha256_at(self._absolute_index(index))

    def signed_quote_at(self, index: int) -> Decimal:
        return self._trades.signed_quote_at(self._absolute_index(index))

    def gross_quote_at(self, index: int) -> Decimal:
        return self._trades.gross_quote_at(self._absolute_index(index))

    def times(self) -> _CompactTradeTimeView:
        return _CompactTradeTimeView(self)


class _CompactBinanceTrades(Sequence[_BinanceTradePoint]):
    """Lossless packed storage for high-rate Binance aggregate trades."""

    __slots__ = (
        "_asset",
        "_connection_codes",
        "_connection_lookup",
        "_connections",
        "_event_ids",
        "_event_sha256",
        "_finished",
        "_gross_quote_ends",
        "_gross_quote_text",
        "_negative",
        "_pending",
        "_pending_ns",
        "_received_monotonic_ns",
    )
    is_received_ordered = True

    def __init__(self, asset: str) -> None:
        self._asset = str(asset)
        self._received_monotonic_ns = array("q")
        self._connection_codes = array("I")
        self._connections: list[str] = []
        self._connection_lookup: dict[str, int] = {}
        self._event_ids = bytearray()
        self._event_sha256 = bytearray()
        self._gross_quote_text = bytearray()
        self._gross_quote_ends = array("Q")
        self._negative = bytearray()
        self._pending_ns: int | None = None
        self._pending: list[_BinanceTradePoint] = []
        self._finished = False

    def __len__(self) -> int:
        return len(self._received_monotonic_ns) + len(self._pending)

    def _connection_code(self, connection_id: str) -> int:
        normalized = str(connection_id)
        existing = self._connection_lookup.get(normalized)
        if existing is not None:
            return existing
        code = len(self._connections)
        if code >= 2**32:
            raise ValueError("too many Binance trade connection segments")
        self._connection_lookup[normalized] = code
        self._connections.append(normalized)
        return code

    def _store(self, point: _BinanceTradePoint) -> None:
        if point.asset != self._asset:
            raise ValueError("compact Binance trade asset differs")
        if point.gross_quote <= 0 or abs(point.signed_quote) != point.gross_quote:
            raise ValueError("compact Binance trade quote amounts are inconsistent")
        if (
            self._received_monotonic_ns
            and point.received_monotonic_ns < self._received_monotonic_ns[-1]
        ):
            raise ValueError("compact Binance trade receive clock regressed")
        quote_bytes = str(point.gross_quote).encode("ascii", errors="strict")
        self._received_monotonic_ns.append(int(point.received_monotonic_ns))
        self._connection_codes.append(self._connection_code(point.connection_id))
        self._event_ids.extend(
            _CompactBinanceBooks._digest_bytes(point.event_id, name="event_id")
        )
        self._event_sha256.extend(
            _CompactBinanceBooks._digest_bytes(
                point.event_sha256,
                name="event_sha256",
            )
        )
        self._gross_quote_text.extend(quote_bytes)
        self._gross_quote_ends.append(len(self._gross_quote_text))
        self._negative.append(point.signed_quote < 0)

    def _flush_pending(self) -> None:
        for point in sorted(self._pending, key=_received_order_key):
            self._store(point)
        self._pending.clear()

    def append(self, point: _BinanceTradePoint) -> None:
        if self._finished:
            raise RuntimeError("compact Binance trades are already finalized")
        received_ns = int(point.received_monotonic_ns)
        if self._pending_ns is None:
            self._pending_ns = received_ns
        elif received_ns < self._pending_ns:
            raise ValueError("compact Binance trade receive clock regressed")
        elif received_ns > self._pending_ns:
            self._flush_pending()
            self._pending_ns = received_ns
        self._pending.append(point)

    def finish(self) -> _CompactBinanceTrades:
        if not self._finished:
            self._flush_pending()
            self._pending_ns = None
            self._finished = True
        return self

    def _normalized_index(self, index: int) -> int:
        if not self._finished:
            raise RuntimeError("compact Binance trades are not finalized")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self._received_monotonic_ns)
        if not 0 <= normalized < len(self._received_monotonic_ns):
            raise IndexError("compact Binance trade index is outside bounds")
        return normalized

    @staticmethod
    def _hex_at(values: bytearray, index: int) -> str:
        offset = index * 32
        return values[offset : offset + 32].hex()

    def __getitem__(
        self, index: int | slice
    ) -> _BinanceTradePoint | tuple[_BinanceTradePoint, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        position = self._normalized_index(index)
        gross = self.gross_quote_at(position)
        return _BinanceTradePoint(
            asset=self._asset,
            connection_id=self.connection_id_at(position),
            received_monotonic_ns=int(self._received_monotonic_ns[position]),
            signed_quote=-gross if self._negative[position] else gross,
            gross_quote=gross,
            event_id=self.event_id_at(position),
            event_sha256=self.event_sha256_at(position),
        )

    def received_monotonic_ns_at(self, index: int) -> int:
        return int(self._received_monotonic_ns[self._normalized_index(index)])

    def connection_id_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._connections[int(self._connection_codes[position])]

    def event_id_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._hex_at(self._event_ids, position)

    def event_sha256_at(self, index: int) -> str:
        position = self._normalized_index(index)
        return self._hex_at(self._event_sha256, position)

    def gross_quote_at(self, index: int) -> Decimal:
        position = self._normalized_index(index)
        start = 0 if position == 0 else int(self._gross_quote_ends[position - 1])
        end = int(self._gross_quote_ends[position])
        return Decimal(self._gross_quote_text[start:end].decode("ascii"))

    def signed_quote_at(self, index: int) -> Decimal:
        position = self._normalized_index(index)
        gross = self.gross_quote_at(position)
        return -gross if self._negative[position] else gross

    def connection_views(self) -> dict[str, _CompactBinanceTradeView]:
        if not self._finished:
            raise RuntimeError("compact Binance trades are not finalized")
        spans: dict[int, list[tuple[int, int]]] = {}
        start = 0
        while start < len(self._connection_codes):
            code = int(self._connection_codes[start])
            end = start + 1
            while (
                end < len(self._connection_codes)
                and int(self._connection_codes[end]) == code
            ):
                end += 1
            spans.setdefault(code, []).append((start, end))
            start = end
        views: dict[str, _CompactBinanceTradeView] = {}
        for code, ranges in spans.items():
            connection_id = self._connections[code]
            if len(ranges) == 1:
                views[connection_id] = _CompactBinanceTradeView(
                    self,
                    connection_id=connection_id,
                    start=ranges[0][0],
                    end=ranges[0][1],
                )
                continue
            indices = array("Q")
            for first, last in ranges:
                indices.extend(range(first, last))
            views[connection_id] = _CompactBinanceTradeView(
                self,
                connection_id=connection_id,
                indices=indices,
            )
        return views


@dataclass(frozen=True)
class PolymarketFeatureSourceContext:
    """Audited cross-feed state shared by bounded CLOB reconstruction batches."""

    run_id: str
    config: PolymarketFeatureConfig
    coverage: PolymarketFeedCoverage
    chainlink: Mapping[str, tuple[_PricePoint, ...]]
    direct_books: Mapping[str, Sequence[_BinanceBookPoint]]
    direct_trades: Mapping[str, Sequence[_BinanceTradePoint]]
    market_snapshots: Mapping[str, _MarketSnapshotPoint]
    book_series: Mapping[str, Mapping[str, _BookSeries]]
    trade_series: Mapping[str, Mapping[str, _TradeSeries]]


_ReceivedPoint = TypeVar("_ReceivedPoint", _BinanceBookPoint, _BinanceTradePoint)


def _received_order_key(
    item: _BinanceBookPoint | _BinanceTradePoint,
) -> tuple[int, str]:
    return item.received_monotonic_ns, item.event_id


def _stable_received_order(
    points: Sequence[_ReceivedPoint],
) -> Sequence[_ReceivedPoint]:
    if getattr(points, "is_received_ordered", False) is True:
        return points
    ordered = points if isinstance(points, tuple) else tuple(points)
    if any(
        _received_order_key(previous) > _received_order_key(current)
        for previous, current in zip(ordered, ordered[1:])
    ):
        return tuple(sorted(ordered, key=_received_order_key))
    return ordered


def _book_point(points: Sequence[_BinanceBookPoint], index: int) -> _BinanceBookPoint:
    value = points[index]
    if not isinstance(value, _BinanceBookPoint):
        raise TypeError("Binance book sequence returned an invalid point")
    return value


def _book_received_monotonic_ns(points: Sequence[_BinanceBookPoint], index: int) -> int:
    accessor = getattr(points, "received_monotonic_ns_at", None)
    if callable(accessor):
        return int(accessor(index))
    return int(_book_point(points, index).received_monotonic_ns)


def _book_midpoint(points: Sequence[_BinanceBookPoint], index: int) -> float:
    accessor = getattr(points, "midpoint_at", None)
    if callable(accessor):
        return float(accessor(index))
    return _book_point(points, index).midpoint


def _book_event_id(points: Sequence[_BinanceBookPoint], index: int) -> str:
    accessor = getattr(points, "event_id_at", None)
    if callable(accessor):
        return str(accessor(index))
    return _book_point(points, index).event_id


def _book_event_sha256(points: Sequence[_BinanceBookPoint], index: int) -> str:
    accessor = getattr(points, "event_sha256_at", None)
    if callable(accessor):
        return str(accessor(index))
    return _book_point(points, index).event_sha256


def _trade_point(
    points: Sequence[_BinanceTradePoint], index: int
) -> _BinanceTradePoint:
    value = points[index]
    if not isinstance(value, _BinanceTradePoint):
        raise TypeError("Binance trade sequence returned an invalid point")
    return value


def _trade_received_monotonic_ns(
    points: Sequence[_BinanceTradePoint], index: int
) -> int:
    accessor = getattr(points, "received_monotonic_ns_at", None)
    if callable(accessor):
        return int(accessor(index))
    return int(_trade_point(points, index).received_monotonic_ns)


def _trade_connection_id(points: Sequence[_BinanceTradePoint], index: int) -> str:
    accessor = getattr(points, "connection_id_at", None)
    if callable(accessor):
        return str(accessor(index))
    return _trade_point(points, index).connection_id


def _trade_event_id(points: Sequence[_BinanceTradePoint], index: int) -> str:
    accessor = getattr(points, "event_id_at", None)
    if callable(accessor):
        return str(accessor(index))
    return _trade_point(points, index).event_id


def _trade_event_sha256(points: Sequence[_BinanceTradePoint], index: int) -> str:
    accessor = getattr(points, "event_sha256_at", None)
    if callable(accessor):
        return str(accessor(index))
    return _trade_point(points, index).event_sha256


def _trade_signed_quote(points: Sequence[_BinanceTradePoint], index: int) -> Decimal:
    accessor = getattr(points, "signed_quote_at", None)
    if callable(accessor):
        return Decimal(accessor(index))
    return _trade_point(points, index).signed_quote


def _trade_gross_quote(points: Sequence[_BinanceTradePoint], index: int) -> Decimal:
    accessor = getattr(points, "gross_quote_at", None)
    if callable(accessor):
        return Decimal(accessor(index))
    return _trade_point(points, index).gross_quote


class _FixedDigestSequence(Sequence[bytes]):
    """Packed SHA-256 sequence with the previous bytes-indexing contract."""

    __slots__ = ("_values",)

    def __init__(self, values: bytes | bytearray) -> None:
        if len(values) % 32:
            raise ValueError("packed SHA-256 sequence has a partial digest")
        self._values = values if isinstance(values, bytearray) else bytearray(values)

    def __len__(self) -> int:
        return len(self._values) // 32

    def __getitem__(self, index: int | slice) -> bytes | tuple[bytes, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError("packed SHA-256 index is outside bounds")
        offset = normalized * 32
        return bytes(self._values[offset : offset + 32])


class _ConnectionCursor:
    __slots__ = (
        "book_index",
        "book_points",
        "latest_connection_id",
        "latest_key",
        "trade_index",
        "trade_points",
    )

    def __init__(
        self,
        book_points: Sequence[_BinanceBookPoint],
        trade_points: Sequence[_BinanceTradePoint],
    ) -> None:
        self.book_points = _stable_received_order(book_points)
        self.trade_points = _stable_received_order(trade_points)
        self.book_index = 0
        self.trade_index = 0
        self.latest_connection_id = ""
        self.latest_key: tuple[int, str, int, int] | None = None

    def advance(self, received_monotonic_ns: int) -> str:
        previous_book_index = self.book_index
        while (
            self.book_index < len(self.book_points)
            and _book_received_monotonic_ns(self.book_points, self.book_index)
            <= received_monotonic_ns
        ):
            self.book_index += 1
        if self.book_index > previous_book_index:
            position = self.book_index - 1
            point = _book_point(self.book_points, position)
            key = (point.received_monotonic_ns, point.event_id, 0, position)
            if self.latest_key is None or key > self.latest_key:
                self.latest_key = key
                self.latest_connection_id = point.connection_id

        previous_trade_index = self.trade_index
        while (
            self.trade_index < len(self.trade_points)
            and _trade_received_monotonic_ns(self.trade_points, self.trade_index)
            <= received_monotonic_ns
        ):
            self.trade_index += 1
        if self.trade_index > previous_trade_index:
            position = self.trade_index - 1
            key = (
                _trade_received_monotonic_ns(self.trade_points, position),
                _trade_event_id(self.trade_points, position),
                1,
                position,
            )
            if self.latest_key is None or key > self.latest_key:
                self.latest_key = key
                self.latest_connection_id = _trade_connection_id(
                    self.trade_points,
                    position,
                )
        return self.latest_connection_id


class _PriceCursor:
    __slots__ = (
        "available",
        "available_by_connection",
        "index",
        "latest_received",
        "points",
    )

    def __init__(self, points: Sequence[_PricePoint]) -> None:
        self.points = tuple(
            sorted(
                points,
                key=lambda item: (
                    item.received_monotonic_ns,
                    item.source_time_ms,
                    item.event_id,
                ),
            )
        )
        self.index = 0
        self.available: list[tuple[int, int, str, _PricePoint]] = []
        self.available_by_connection: dict[
            str, list[tuple[int, int, str, _PricePoint]]
        ] = {}
        self.latest_received: _PricePoint | None = None

    def advance(self, received_monotonic_ns: int) -> None:
        while (
            self.index < len(self.points)
            and self.points[self.index].received_monotonic_ns <= received_monotonic_ns
        ):
            point = self.points[self.index]
            insort(
                self.available,
                (
                    point.source_time_ms,
                    point.received_monotonic_ns,
                    point.event_id,
                    point,
                ),
            )
            insort(
                self.available_by_connection.setdefault(point.connection_id, []),
                (
                    point.source_time_ms,
                    point.received_monotonic_ns,
                    point.event_id,
                    point,
                ),
            )
            self.latest_received = point
            self.index += 1

    def active_connection_id(self) -> str:
        return (
            "" if self.latest_received is None else self.latest_received.connection_id
        )

    def latest_at_or_before(
        self,
        source_time_ms: int,
        *,
        connection_id: str | None = None,
    ) -> _PricePoint | None:
        available = (
            self.available
            if connection_id is None
            else self.available_by_connection.get(str(connection_id), [])
        )
        index = bisect_right(
            available,
            int(source_time_ms),
            key=lambda item: item[0],
        )
        return None if index == 0 else available[index - 1][3]


class _BookCursor:
    __slots__ = ("index", "latest", "points")

    def __init__(self, points: Sequence[_BinanceBookPoint]) -> None:
        self.points = _stable_received_order(points)
        self.index = 0
        self.latest: _BinanceBookPoint | None = None

    def advance(self, received_monotonic_ns: int) -> _BinanceBookPoint | None:
        while (
            self.index < len(self.points)
            and _book_received_monotonic_ns(self.points, self.index)
            <= received_monotonic_ns
        ):
            self.index += 1
        if self.index:
            self.latest = _book_point(self.points, self.index - 1)
        return self.latest


class _BookSeries:
    __slots__ = (
        "connection_id",
        "points",
        "prefix_digests",
        "squared_returns",
        "times",
    )

    def __init__(self, points: Sequence[_BinanceBookPoint]) -> None:
        self.points = _stable_received_order(points)
        if isinstance(self.points, _CompactBinanceBookView):
            self.connection_id = self.points.connection_id
            self.times = self.points.times()
        else:
            connections = {item.connection_id for item in self.points}
            if len(connections) != 1:
                raise ValueError("Binance book series crossed connection segments")
            self.connection_id = next(iter(connections))
            self.times = tuple(item.received_monotonic_ns for item in self.points)
        previous_sha256 = _canonical_sha256(
            {
                "schema_version": "binance-book-causal-prefix-v2",
                "stream": "binance_spot",
                "connection_id": self.connection_id,
            }
        )
        prefix_digests = bytearray(bytes.fromhex(previous_sha256))
        for index in range(len(self.points)):
            previous_sha256 = _canonical_sha256(
                {
                    "schema_version": "binance-book-causal-prefix-v2",
                    "previous_sha256": previous_sha256,
                    "event_id": _book_event_id(self.points, index),
                    "event_sha256": _book_event_sha256(self.points, index),
                    "received_monotonic_ns": _book_received_monotonic_ns(
                        self.points, index
                    ),
                }
            )
            prefix_digests.extend(bytes.fromhex(previous_sha256))
        self.prefix_digests = _FixedDigestSequence(prefix_digests)
        squared_returns = array("d", [0.0])
        for index in range(1, len(self.points)):
            value = math.log(
                _book_midpoint(self.points, index)
                / _book_midpoint(self.points, index - 1)
            )
            squared_returns.append(value * value)
        self.squared_returns = squared_returns

    def _index_at_or_before(self, received_monotonic_ns: int) -> int:
        return bisect_right(self.times, int(received_monotonic_ns)) - 1

    def has_lookback(self, received_monotonic_ns: int, window_ms: int) -> bool:
        if not self.times:
            return False
        return self.times[0] <= (received_monotonic_ns - int(window_ms) * 1_000_000)

    def causal_prefix(self, received_monotonic_ns: int) -> tuple[int, str]:
        count = bisect_right(self.times, int(received_monotonic_ns))
        return count, self.prefix_digests[count].hex()

    def return_bps(self, received_monotonic_ns: int, window_ms: int) -> float:
        current = self._index_at_or_before(received_monotonic_ns)
        previous = self._index_at_or_before(
            received_monotonic_ns - int(window_ms) * 1_000_000
        )
        if current < 0 or previous < 0:
            return 0.0
        return _log_basis_bps(
            _book_midpoint(self.points, current),
            _book_midpoint(self.points, previous),
        )

    def realized_volatility_bps(
        self, received_monotonic_ns: int, window_ms: int
    ) -> float:
        current = self._index_at_or_before(received_monotonic_ns)
        anchor = self._index_at_or_before(
            received_monotonic_ns - int(window_ms) * 1_000_000
        )
        if current <= 0:
            return 0.0
        first_return = max(1, anchor + 1)
        variance = math.fsum(self.squared_returns[first_return : current + 1])
        return 10_000.0 * math.sqrt(max(0.0, variance))


class _TradeSeries:
    __slots__ = (
        "connection_id",
        "gross_prefix",
        "points",
        "prefix_digests",
        "signed_prefix",
        "times",
    )

    def __init__(self, points: Sequence[_BinanceTradePoint]) -> None:
        self.points = _stable_received_order(points)
        if isinstance(self.points, _CompactBinanceTradeView):
            self.connection_id = self.points.connection_id
            self.times = self.points.times()
        else:
            connections = {item.connection_id for item in self.points}
            if len(connections) != 1:
                raise ValueError("Binance trade series crossed connection segments")
            self.connection_id = next(iter(connections))
            self.times = array(
                "q",
                (item.received_monotonic_ns for item in self.points),
            )
        previous_sha256 = _canonical_sha256(
            {
                "schema_version": "binance-trade-causal-prefix-v2",
                "stream": "binance_spot",
                "connection_id": self.connection_id,
            }
        )
        prefix_digests = bytearray(bytes.fromhex(previous_sha256))
        signed_prefix = [Decimal(0)]
        gross_prefix = [Decimal(0)]
        for index in range(len(self.points)):
            signed_prefix.append(
                signed_prefix[-1] + _trade_signed_quote(self.points, index)
            )
            gross_prefix.append(
                gross_prefix[-1] + _trade_gross_quote(self.points, index)
            )
            previous_sha256 = _canonical_sha256(
                {
                    "schema_version": "binance-trade-causal-prefix-v2",
                    "previous_sha256": previous_sha256,
                    "event_id": _trade_event_id(self.points, index),
                    "event_sha256": _trade_event_sha256(self.points, index),
                    "received_monotonic_ns": _trade_received_monotonic_ns(
                        self.points,
                        index,
                    ),
                }
            )
            prefix_digests.extend(bytes.fromhex(previous_sha256))
        self.signed_prefix = tuple(signed_prefix)
        self.gross_prefix = tuple(gross_prefix)
        self.prefix_digests = _FixedDigestSequence(prefix_digests)

    def causal_prefix(self, received_monotonic_ns: int) -> tuple[int, str]:
        count = bisect_right(self.times, int(received_monotonic_ns))
        return count, self.prefix_digests[count].hex()

    def stats(self, received_monotonic_ns: int, window_ms: int) -> tuple[float, float]:
        high = bisect_right(self.times, int(received_monotonic_ns))
        low = bisect_right(
            self.times, received_monotonic_ns - int(window_ms) * 1_000_000
        )
        signed = self.signed_prefix[high] - self.signed_prefix[low]
        gross = self.gross_prefix[high] - self.gross_prefix[low]
        imbalance = 0.0 if gross <= 0 else float(signed / gross)
        return imbalance, float(gross)


def _parse_feed_points(
    store: PolymarketEvidenceStore,
    run_id: str,
    *,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[
    dict[str, tuple[_PricePoint, ...]],
    dict[str, Sequence[_BinanceBookPoint]],
    dict[str, Sequence[_BinanceTradePoint]],
]:
    chainlink: dict[str, list[_PricePoint]] = {key: [] for key in _ASSETS}
    direct_books = {key: _CompactBinanceBooks(key) for key in _ASSETS}
    direct_trades = {key: _CompactBinanceTrades(key) for key in _ASSETS}
    scan_started = time.monotonic()
    last_progress = scan_started
    parsed_count = 0

    def notify(*, force: bool = False) -> None:
        nonlocal last_progress
        if progress is None:
            return
        now = time.monotonic()
        if not force and now - last_progress < 30.0:
            return
        last_progress = now
        try:
            progress(
                "feature-source-scan",
                {
                    "elapsed_seconds": round(now - scan_started, 3),
                    "parsed_public_event_count": parsed_count,
                    "chainlink_point_count": sum(map(len, chainlink.values())),
                    "direct_book_point_count": sum(map(len, direct_books.values())),
                    "direct_trade_point_count": sum(map(len, direct_trades.values())),
                },
            )
        except Exception:
            return

    notify(force=True)
    for decoded in store.iter_public_events(
        run_id,
        streams=("binance_spot", "polymarket_rtds"),
        verified_source=True,
    ):
        parsed_count += 1
        if parsed_count % 4_096 == 0:
            notify()
        event_id = decoded.event_id
        event_sha256 = decoded.event_sha256
        stream = decoded.stream
        event_type = decoded.event_type
        received_wall_ms = decoded.received_wall_ms
        received_monotonic_ns = decoded.received_monotonic_ns
        connection_id = str(decoded.connection_id)
        asset = decoded.symbol.upper()
        if asset not in chainlink:
            raise ValueError(
                "prospective feature evidence contains an unsupported asset"
            )
        event = decoded.event
        normalized_stream = str(stream)
        normalized_type = str(event_type).lower()
        if normalized_stream == "binance_spot":
            payload = event.get("data")
            if not isinstance(payload, Mapping):
                raise ValueError("direct Binance feature payload is malformed")
            if normalized_type == "bookticker":
                bid = _finite_float(payload.get("b"), name="Binance bid", positive=True)
                ask = _finite_float(payload.get("a"), name="Binance ask", positive=True)
                bid_quantity = _finite_float(
                    payload.get("B"), name="Binance bid quantity", positive=True
                )
                ask_quantity = _finite_float(
                    payload.get("A"), name="Binance ask quantity", positive=True
                )
                if bid >= ask:
                    raise ValueError("direct Binance feature book is crossed")
                direct_books[asset].append(
                    _BinanceBookPoint(
                        asset=asset,
                        connection_id=connection_id,
                        received_wall_ms=int(received_wall_ms),
                        received_monotonic_ns=int(received_monotonic_ns),
                        bid=bid,
                        bid_quantity=bid_quantity,
                        ask=ask,
                        ask_quantity=ask_quantity,
                        event_id=str(event_id),
                        event_sha256=str(event_sha256),
                    )
                )
            elif normalized_type == "trade":
                price = _finite_decimal(
                    payload.get("p"), name="Binance trade price", positive=True
                )
                quantity = _finite_decimal(
                    payload.get("q"), name="Binance trade quantity", positive=True
                )
                buyer_is_maker = payload.get("m")
                if not isinstance(buyer_is_maker, bool):
                    raise ValueError("Binance aggressor-side flag is malformed")
                gross_quote = price * quantity
                direct_trades[asset].append(
                    _BinanceTradePoint(
                        asset=asset,
                        connection_id=connection_id,
                        received_monotonic_ns=int(received_monotonic_ns),
                        signed_quote=-gross_quote if buyer_is_maker else gross_quote,
                        gross_quote=gross_quote,
                        event_id=str(event_id),
                        event_sha256=str(event_sha256),
                    )
                )
            continue

        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("RTDS feature payload is malformed")
        raw_symbol = str(payload.get("symbol") or "")
        if not (
            normalized_type.startswith("crypto_prices_chainlink:") or "/" in raw_symbol
        ):
            continue
        message_type = str(event.get("type") or "").lower()
        raw_points: list[tuple[object, object, int]] = []
        if message_type == "subscribe":
            history = payload.get("data")
            if not isinstance(history, list) or not history:
                raise ValueError("RTDS feature history is empty or malformed")
            for index, item in enumerate(history):
                if not isinstance(item, Mapping):
                    raise ValueError("RTDS feature history row is malformed")
                raw_points.append((item.get("timestamp"), item.get("value"), index))
        elif message_type == "update":
            raw_points.append((payload.get("timestamp"), payload.get("value"), 0))
        for raw_time, raw_price, index in raw_points:
            chainlink[asset].append(
                _PricePoint(
                    asset=asset,
                    connection_id=connection_id,
                    source_time_ms=_timestamp(raw_time, name="RTDS source timestamp"),
                    received_wall_ms=int(received_wall_ms),
                    received_monotonic_ns=int(received_monotonic_ns),
                    price=_finite_float(
                        raw_price, name="RTDS source price", positive=True
                    ),
                    event_id=f"{event_id}:{index}",
                    event_sha256=str(event_sha256),
                )
            )
    notify(force=True)
    for books in direct_books.values():
        books.finish()
    for trades in direct_trades.values():
        trades.finish()
    return (
        {key: tuple(value) for key, value in chainlink.items()},
        direct_books,
        direct_trades,
    )


def _load_market_snapshot_points(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> dict[str, _MarketSnapshotPoint]:
    rows = (
        store.connect()
        .execute(
            """
        SELECT condition_id, observed_wall_ms, observed_monotonic_ns,
               snapshot_sha256
        FROM polymarket_market_snapshot
        WHERE run_id = ? ORDER BY condition_id
        """,
            [run_id],
        )
        .fetchall()
    )
    snapshots: dict[str, _MarketSnapshotPoint] = {}
    for condition_id, wall_ms, monotonic_ns, snapshot_sha256 in rows:
        condition = str(condition_id).lower()
        digest = str(snapshot_sha256)
        if not condition or len(digest) != 64:
            raise ValueError("Polymarket market snapshot identity is malformed")
        if condition in snapshots:
            raise ValueError("Polymarket market snapshot identity is duplicated")
        snapshots[condition] = _MarketSnapshotPoint(
            condition_id=condition,
            observed_wall_ms=_timestamp(wall_ms, name="market snapshot wall timestamp"),
            observed_monotonic_ns=_timestamp(
                monotonic_ns, name="market snapshot monotonic timestamp"
            ),
            snapshot_sha256=digest,
        )
    return snapshots


def load_polymarket_feature_source_context(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    config: PolymarketFeatureConfig | None = None,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> PolymarketFeatureSourceContext:
    """Parse immutable Binance/RTDS evidence once for bounded market batches."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("Polymarket feature source context requires a run ID")
    cfg = (config or PolymarketFeatureConfig()).validated()
    coverage = inspect_polymarket_feed_coverage(
        store,
        run_id=selected,
        minimum_resolved_markets_per_asset=cfg.minimum_resolved_markets_per_asset,
        allow_segmented_gaps=cfg.allow_segmented_gaps,
    )
    chainlink, direct_books, direct_trades = _parse_feed_points(
        store,
        selected,
        progress=progress,
    )
    book_series: dict[str, dict[str, _BookSeries]] = {}
    trade_series: dict[str, dict[str, _TradeSeries]] = {}
    for asset, points in direct_books.items():
        if progress is not None:
            progress(
                "feature-source-series",
                {
                    "asset": asset,
                    "kind": "book",
                    "point_count": len(points),
                    "status": "started",
                },
            )
        if isinstance(points, _CompactBinanceBooks):
            grouped_books: Mapping[str, Sequence[_BinanceBookPoint]] = (
                points.connection_views()
            )
        else:
            mutable_groups: dict[str, list[_BinanceBookPoint]] = {}
            for point in points:
                mutable_groups.setdefault(point.connection_id, []).append(point)
            grouped_books = mutable_groups
        book_series[asset] = {
            connection_id: _BookSeries(segment)
            for connection_id, segment in grouped_books.items()
        }
        if progress is not None:
            progress(
                "feature-source-series",
                {
                    "asset": asset,
                    "kind": "book",
                    "point_count": len(points),
                    "segment_count": len(grouped_books),
                    "status": "complete",
                },
            )
    for asset, points in direct_trades.items():
        if progress is not None:
            progress(
                "feature-source-series",
                {
                    "asset": asset,
                    "kind": "trade",
                    "point_count": len(points),
                    "status": "started",
                },
            )
        if isinstance(points, _CompactBinanceTrades):
            grouped_trades: Mapping[str, Sequence[_BinanceTradePoint]] = (
                points.connection_views()
            )
        else:
            mutable_groups: dict[str, list[_BinanceTradePoint]] = {}
            for point in points:
                mutable_groups.setdefault(point.connection_id, []).append(point)
            grouped_trades = mutable_groups
        trade_series[asset] = {
            connection_id: _TradeSeries(segment)
            for connection_id, segment in grouped_trades.items()
        }
        if progress is not None:
            progress(
                "feature-source-series",
                {
                    "asset": asset,
                    "kind": "trade",
                    "point_count": len(points),
                    "segment_count": len(grouped_trades),
                    "status": "complete",
                },
            )
    return PolymarketFeatureSourceContext(
        run_id=selected,
        config=cfg,
        coverage=coverage,
        chainlink=chainlink,
        direct_books=direct_books,
        direct_trades=direct_trades,
        market_snapshots=_load_market_snapshot_points(store, selected),
        book_series=book_series,
        trade_series=trade_series,
    )


def _clob_features(book: PolymarketRecordedBook) -> tuple[float, ...]:
    bids = book.snapshot.bids
    asks = book.snapshot.asks
    if not bids or not asks:
        raise ValueError("CLOB feature state requires executable bid and ask depth")
    best_bid = float(bids[0].price)
    best_ask = float(asks[0].price)
    bid_quantity = float(bids[0].quantity)
    ask_quantity = float(asks[0].quantity)
    if best_bid >= best_ask or min(bid_quantity, ask_quantity) <= 0.0:
        raise ValueError("CLOB feature state is crossed or empty")
    midpoint = (best_bid + best_ask) / 2.0
    microprice = (best_ask * bid_quantity + best_bid * ask_quantity) / (
        bid_quantity + ask_quantity
    )
    return (
        best_bid,
        best_ask,
        midpoint,
        best_ask - best_bid,
        microprice,
        _ratio_imbalance(bid_quantity, ask_quantity),
        sum(float(level.quantity) for level in bids[:3]),
        sum(float(level.quantity) for level in asks[:3]),
    )


def _skip(skipped: dict[str, int], reason: str) -> None:
    skipped[reason] = skipped.get(reason, 0) + 1


def _feature_row_payload(row: PolymarketFeatureRow) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_FEATURE_SCHEMA_VERSION,
        "feature_id": row.feature_id,
        "run_id": row.run_id,
        "condition_id": row.condition_id,
        "market_id": row.market_id,
        "asset": row.asset,
        "decision_event_id": row.decision_event_id,
        "decision_received_wall_ms": row.decision_received_wall_ms,
        "decision_received_monotonic_ns": row.decision_received_monotonic_ns,
        "feature_names": list(POLYMARKET_FEATURE_NAMES),
        "feature_values": [format(value, ".17g") for value in row.feature_values],
        "official_up": row.official_up,
        "resolution_event_id": row.resolution_event_id,
        "input_provenance_sha256": row.input_provenance_sha256,
    }


def polymarket_feature_row_sha256(row: PolymarketFeatureRow) -> str:
    """Return the canonical digest for one immutable feature row."""

    return _canonical_sha256(_feature_row_payload(row))


def build_polymarket_feature_dataset(
    store: PolymarketEvidenceStore,
    *,
    run_id: str | None = None,
    config: PolymarketFeatureConfig | None = None,
    condition_ids: Sequence[str] | None = None,
    source_context: PolymarketFeatureSourceContext | None = None,
) -> PolymarketFeatureDataset:
    """Build causal features; official outcomes are attached only as future labels."""

    cfg = (config or PolymarketFeatureConfig()).validated()
    replay = PolymarketEvidenceReplay.load(
        store,
        run_id=run_id,
        allow_segmented_gaps=cfg.allow_segmented_gaps,
        book_sample_interval_ms=cfg.cadence_ms,
        condition_ids=condition_ids,
    )
    selected = replay.run_id
    context = source_context or load_polymarket_feature_source_context(
        store,
        run_id=selected,
        config=cfg,
    )
    if context.run_id != selected or context.config.asdict() != cfg.asdict():
        raise ValueError("Polymarket feature source context contract differs")
    coverage = context.coverage
    chainlink = context.chainlink
    direct_books = context.direct_books
    direct_trades = context.direct_trades
    market_snapshots = context.market_snapshots
    chainlink_cursors = {
        asset: _PriceCursor(points) for asset, points in chainlink.items()
    }
    book_cursors = {
        asset: _BookCursor(points) for asset, points in direct_books.items()
    }
    direct_connection_cursors = {
        asset: _ConnectionCursor(direct_books[asset], direct_trades[asset])
        for asset in _ASSETS
    }
    book_series = context.book_series
    trade_series = context.trade_series
    resolution_by_condition: dict[str, PolymarketResolutionEvidence] = {
        item.condition_id: item for item in replay.resolutions
    }
    states: dict[str, dict[str, PolymarketRecordedBook]] = {}
    active_segments: dict[str, str] = {}
    last_emitted_ns: dict[str, int] = {}
    skipped: dict[str, int] = {}
    rows: list[PolymarketFeatureRow] = []
    candidate_count = 0

    for trigger in replay.books:
        market = trigger.market
        condition = market.condition_id
        if active_segments.get(condition) != trigger.segment_id:
            states[condition] = {}
            active_segments[condition] = trigger.segment_id
        states.setdefault(condition, {})[trigger.outcome] = trigger
        candidate_count += 1
        pair = states[condition]
        if "Up" not in pair or "Down" not in pair:
            _skip(skipped, "waiting_for_both_outcome_books")
            continue
        decision_wall_ms = trigger.received_wall_ms
        decision_ns = trigger.received_monotonic_ns
        market_snapshot = market_snapshots.get(condition)
        if market_snapshot is None:
            raise ValueError("replay market has no immutable discovery snapshot")
        if market_snapshot.observed_monotonic_ns > decision_ns:
            _skip(skipped, "future_market_snapshot")
            continue
        if decision_wall_ms < market.event_start_ms + cfg.warmup_ms:
            _skip(skipped, "before_market_warmup")
            continue
        if decision_wall_ms >= market.end_ms:
            _skip(skipped, "at_or_after_market_end")
            continue
        previous_emit = last_emitted_ns.get(condition)
        if (
            previous_emit is not None
            and decision_ns - previous_emit < cfg.cadence_ms * 1_000_000
        ):
            _skip(skipped, "cadence_throttle")
            continue

        up = pair["Up"]
        down = pair["Down"]
        if up.segment_id != trigger.segment_id or down.segment_id != trigger.segment_id:
            raise ValueError("Polymarket feature books crossed continuity segments")
        up_age_ms = (decision_ns - up.received_monotonic_ns) / 1_000_000.0
        down_age_ms = (decision_ns - down.received_monotonic_ns) / 1_000_000.0
        if (
            min(up_age_ms, down_age_ms) < 0.0
            or max(up_age_ms, down_age_ms) > cfg.maximum_clob_age_ms
        ):
            _skip(skipped, "stale_or_future_clob_pair")
            continue

        asset = market.asset
        chainlink_cursor = chainlink_cursors[asset]
        chainlink_cursor.advance(decision_ns)
        direct_book = book_cursors[asset].advance(decision_ns)
        active_direct_connection = direct_connection_cursors[asset].advance(decision_ns)
        active_chainlink_connection = chainlink_cursor.active_connection_id()
        chainlink_now = chainlink_cursor.latest_at_or_before(
            decision_wall_ms,
            connection_id=active_chainlink_connection,
        )
        if chainlink_now is None:
            _skip(skipped, "missing_chainlink_current_price")
            continue
        anchor = chainlink_cursor.latest_at_or_before(
            market.event_start_ms,
            connection_id=chainlink_now.connection_id,
        )
        if anchor is None:
            _skip(skipped, "missing_chainlink_open_anchor_in_active_segment")
            continue
        anchor_gap_ms = market.event_start_ms - anchor.source_time_ms
        if anchor_gap_ms < 0 or anchor_gap_ms > cfg.maximum_chainlink_anchor_gap_ms:
            _skip(skipped, "chainlink_open_anchor_gap")
            continue
        if direct_book is None:
            _skip(skipped, "missing_direct_binance_book")
            continue
        if direct_book.connection_id != active_direct_connection:
            _skip(skipped, "missing_direct_binance_book_in_active_segment")
            continue
        active_book_series = book_series[asset].get(direct_book.connection_id)
        active_trade_series = trade_series[asset].get(direct_book.connection_id)
        if active_book_series is None or active_trade_series is None:
            _skip(skipped, "missing_direct_binance_segment")
            continue

        direct_age_ms = (decision_ns - direct_book.received_monotonic_ns) / 1_000_000.0
        chainlink_arrival_age_ms = (
            decision_ns - chainlink_now.received_monotonic_ns
        ) / 1_000_000.0
        chainlink_source_age_ms = decision_wall_ms - chainlink_now.source_time_ms
        if direct_age_ms < 0.0 or direct_age_ms > cfg.maximum_direct_binance_age_ms:
            _skip(skipped, "stale_or_future_direct_binance_book")
            continue
        if (
            chainlink_source_age_ms < 0
            or chainlink_source_age_ms > cfg.maximum_chainlink_source_age_ms
            or chainlink_arrival_age_ms < 0.0
            or chainlink_arrival_age_ms > cfg.maximum_chainlink_arrival_age_ms
        ):
            _skip(skipped, "stale_or_future_chainlink_price")
            continue
        if not active_book_series.has_lookback(decision_ns, 5_000):
            _skip(skipped, "insufficient_direct_binance_lookback")
            continue

        try:
            up_values = _clob_features(up)
            down_values = _clob_features(down)
        except ValueError:
            _skip(skipped, "non_executable_clob_pair")
            continue
        market_duration_ms = market.end_ms - market.event_start_ms
        if market_duration_ms <= 0:
            raise ValueError("Polymarket feature market duration is invalid")
        elapsed_fraction = (
            decision_wall_ms - market.event_start_ms
        ) / market_duration_ms
        remaining_seconds = (market.end_ms - decision_wall_ms) / 1_000.0
        trade_100, _gross_100 = active_trade_series.stats(decision_ns, 100)
        trade_250, gross_250 = active_trade_series.stats(decision_ns, 250)
        trade_1000, gross_1000 = active_trade_series.stats(decision_ns, 1_000)
        trade_5000, gross_5000 = active_trade_series.stats(decision_ns, 5_000)
        book_prefix_count, book_prefix_sha256 = active_book_series.causal_prefix(
            decision_ns
        )
        trade_prefix_count, trade_prefix_sha256 = active_trade_series.causal_prefix(
            decision_ns
        )
        if book_prefix_count < 1:
            raise ValueError("direct Binance feature provenance is incomplete")
        values = (
            elapsed_fraction,
            remaining_seconds,
            *up_values,
            *down_values,
            up_values[1] + down_values[1],
            up_values[0] + down_values[0],
            up_age_ms,
            down_age_ms,
            _log_basis_bps(chainlink_now.price, anchor.price),
            _log_basis_bps(direct_book.midpoint, anchor.price),
            _log_basis_bps(direct_book.midpoint, chainlink_now.price),
            direct_book.bid,
            direct_book.ask,
            10_000.0 * (direct_book.ask - direct_book.bid) / direct_book.midpoint,
            _ratio_imbalance(direct_book.bid_quantity, direct_book.ask_quantity),
            active_book_series.return_bps(decision_ns, 100),
            active_book_series.return_bps(decision_ns, 250),
            active_book_series.return_bps(decision_ns, 1_000),
            active_book_series.return_bps(decision_ns, 5_000),
            active_book_series.realized_volatility_bps(decision_ns, 100),
            active_book_series.realized_volatility_bps(decision_ns, 1_000),
            active_book_series.realized_volatility_bps(decision_ns, 5_000),
            trade_100,
            trade_250,
            trade_1000,
            trade_5000,
            math.log1p(gross_250),
            math.log1p(gross_1000),
            math.log1p(gross_5000),
            direct_age_ms,
            float(chainlink_source_age_ms),
            chainlink_arrival_age_ms,
            float(anchor_gap_ms),
            math.log1p(float(market.liquidity_quote)),
            math.log1p(float(market.volume_quote)),
        )
        if len(values) != len(POLYMARKET_FEATURE_NAMES) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("Polymarket feature vector is non-finite or misaligned")
        resolution = resolution_by_condition.get(condition)
        official_up = None if resolution is None else resolution.winning_outcome == "Up"
        resolution_event_id = "" if resolution is None else resolution.event_id
        provenance = _canonical_sha256(
            {
                "schema_version": POLYMARKET_FEATURE_SCHEMA_VERSION,
                "market_snapshot": {
                    "condition_id": market_snapshot.condition_id,
                    "observed_wall_ms": market_snapshot.observed_wall_ms,
                    "observed_monotonic_ns": market_snapshot.observed_monotonic_ns,
                    "snapshot_sha256": market_snapshot.snapshot_sha256,
                },
                "up_book": {
                    "event_id": up.event_id,
                    "connection_id": up.connection_id,
                    "segment_id": up.segment_id,
                    "source_time_ms": up.snapshot.source_time_ms,
                    "received_wall_ms": up.received_wall_ms,
                    "received_monotonic_ns": up.received_monotonic_ns,
                    "state_provenance_sha256": up.snapshot.source_payload_sha256,
                },
                "down_book": {
                    "event_id": down.event_id,
                    "connection_id": down.connection_id,
                    "segment_id": down.segment_id,
                    "source_time_ms": down.snapshot.source_time_ms,
                    "received_wall_ms": down.received_wall_ms,
                    "received_monotonic_ns": down.received_monotonic_ns,
                    "state_provenance_sha256": down.snapshot.source_payload_sha256,
                },
                "direct_binance_latest_book": {
                    "event_id": direct_book.event_id,
                    "event_sha256": direct_book.event_sha256,
                    "connection_id": direct_book.connection_id,
                    "received_wall_ms": direct_book.received_wall_ms,
                    "received_monotonic_ns": direct_book.received_monotonic_ns,
                },
                "direct_binance_causal_prefix": {
                    "connection_id": active_book_series.connection_id,
                    "book_event_count": book_prefix_count,
                    "book_prefix_sha256": book_prefix_sha256,
                    "trade_event_count": trade_prefix_count,
                    "trade_prefix_sha256": trade_prefix_sha256,
                },
                "chainlink_anchor": {
                    "event_id": anchor.event_id,
                    "event_sha256": anchor.event_sha256,
                    "connection_id": anchor.connection_id,
                    "source_time_ms": anchor.source_time_ms,
                    "received_wall_ms": anchor.received_wall_ms,
                    "received_monotonic_ns": anchor.received_monotonic_ns,
                },
                "chainlink_current": {
                    "event_id": chainlink_now.event_id,
                    "event_sha256": chainlink_now.event_sha256,
                    "connection_id": chainlink_now.connection_id,
                    "source_time_ms": chainlink_now.source_time_ms,
                    "received_wall_ms": chainlink_now.received_wall_ms,
                    "received_monotonic_ns": chainlink_now.received_monotonic_ns,
                },
            }
        )
        feature_id = _canonical_sha256(
            {
                "run_id": selected,
                "condition_id": condition,
                "decision_event_id": trigger.event_id,
                "decision_received_monotonic_ns": decision_ns,
                "config": cfg.asdict(),
            }
        )
        row = PolymarketFeatureRow(
            feature_id=feature_id,
            run_id=selected,
            condition_id=condition,
            market_id=market.market_id,
            asset=asset,
            decision_event_id=trigger.event_id,
            decision_received_wall_ms=decision_wall_ms,
            decision_received_monotonic_ns=decision_ns,
            feature_values=tuple(values),
            official_up=official_up,
            resolution_event_id=resolution_event_id,
            input_provenance_sha256=provenance,
            row_sha256="",
        )
        rows.append(
            replace(row, row_sha256=_canonical_sha256(_feature_row_payload(row)))
        )
        last_emitted_ns[condition] = decision_ns

    labeled_conditions = {asset: set() for asset in _ASSETS}
    row_assets = {asset: 0 for asset in labeled_conditions}
    for row in rows:
        row_assets[row.asset] += 1
        if row.official_up is not None:
            labeled_conditions[row.asset].add(row.condition_id)
    labeled_market_counts = {
        asset: len(conditions) for asset, conditions in labeled_conditions.items()
    }
    shadow_errors = list(coverage.shadow_errors)
    for asset, count in row_assets.items():
        if count == 0:
            shadow_errors.append(f"no_causal_feature_rows:{asset}")
    training_errors = list(shadow_errors)
    for asset, count in labeled_market_counts.items():
        if count < cfg.minimum_resolved_markets_per_asset:
            training_errors.append(
                f"insufficient_featured_resolved_markets:{asset}:"
                f"{count}/{cfg.minimum_resolved_markets_per_asset}"
            )

    report_row = (
        store.connect()
        .execute(
            "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
            [selected],
        )
        .fetchone()
    )
    if report_row is None:
        raise ValueError("Polymarket feature run report is unavailable")
    dataset_payload = {
        "schema_version": POLYMARKET_DATASET_SCHEMA_VERSION,
        "run_id": selected,
        "run_report_sha256": str(report_row[0]),
        "config": cfg.asdict(),
        "feature_names": list(POLYMARKET_FEATURE_NAMES),
        "candidate_count": candidate_count,
        "skipped_counts": dict(sorted(skipped.items())),
        "labeled_market_counts": labeled_market_counts,
        "shadow_errors": sorted(set(shadow_errors)),
        "training_errors": sorted(set(training_errors)),
        "row_sha256": [row.row_sha256 for row in rows],
        "replay_diagnostics": replay.diagnostics.asdict(),
        "coverage": coverage.asdict(),
    }
    dataset_sha256 = _canonical_sha256(dataset_payload)
    return PolymarketFeatureDataset(
        dataset_id=dataset_sha256,
        run_id=selected,
        config=cfg,
        rows=tuple(rows),
        candidate_count=candidate_count,
        skipped_counts=dict(sorted(skipped.items())),
        labeled_market_counts=labeled_market_counts,
        shadow_errors=tuple(sorted(set(shadow_errors))),
        training_errors=tuple(sorted(set(training_errors))),
        replay_diagnostics=replay.diagnostics.asdict(),
        coverage=coverage.asdict(),
        dataset_sha256=dataset_sha256,
    )


def materialize_polymarket_feature_dataset(
    store: PolymarketEvidenceStore,
    dataset: PolymarketFeatureDataset,
) -> PolymarketFeatureMaterialization:
    """Persist one immutable derived dataset beside, never inside, raw evidence."""

    report_row = (
        store.connect()
        .execute(
            "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
            [dataset.run_id],
        )
        .fetchone()
    )
    if report_row is None:
        raise ValueError("Polymarket feature run report is unavailable")
    expected_dataset_payload = {
        "schema_version": POLYMARKET_DATASET_SCHEMA_VERSION,
        "run_id": dataset.run_id,
        "run_report_sha256": str(report_row[0]),
        "config": dataset.config.asdict(),
        "feature_names": list(POLYMARKET_FEATURE_NAMES),
        "candidate_count": dataset.candidate_count,
        "skipped_counts": dict(dataset.skipped_counts),
        "labeled_market_counts": dict(dataset.labeled_market_counts),
        "shadow_errors": list(dataset.shadow_errors),
        "training_errors": list(dataset.training_errors),
        "row_sha256": [row.row_sha256 for row in dataset.rows],
        "replay_diagnostics": dict(dataset.replay_diagnostics),
        "coverage": dict(dataset.coverage),
    }
    expected_dataset_sha256 = _canonical_sha256(expected_dataset_payload)
    if (
        dataset.dataset_id != dataset.dataset_sha256
        or dataset.dataset_sha256 != expected_dataset_sha256
    ):
        raise ValueError("Polymarket dataset identity does not match its manifest")
    for row in dataset.rows:
        if row.run_id != dataset.run_id:
            raise ValueError("Polymarket feature row belongs to another run")
        if row.row_sha256 != _canonical_sha256(_feature_row_payload(row)):
            raise ValueError("Polymarket feature row digest is invalid")
        if len(row.feature_values) != len(POLYMARKET_FEATURE_NAMES):
            raise ValueError("Polymarket feature row width is invalid")

    connection = store.connect()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_feature_dataset (
            dataset_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            feature_schema_version VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            config_json VARCHAR NOT NULL,
            feature_names_json VARCHAR NOT NULL,
            candidate_count UBIGINT NOT NULL,
            row_count UBIGINT NOT NULL,
            labeled_row_count UBIGINT NOT NULL,
            skipped_counts_json VARCHAR NOT NULL,
            labeled_market_counts_json VARCHAR NOT NULL,
            shadow_errors_json VARCHAR NOT NULL,
            training_errors_json VARCHAR NOT NULL,
            replay_diagnostics_json VARCHAR NOT NULL,
            coverage_json VARCHAR NOT NULL,
            dataset_sha256 VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS polymarket_feature_row (
            dataset_id VARCHAR NOT NULL,
            feature_id VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            market_id VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            decision_event_id VARCHAR NOT NULL,
            decision_received_wall_ms BIGINT NOT NULL,
            decision_received_monotonic_ns UBIGINT NOT NULL,
            feature_values_json VARCHAR NOT NULL,
            official_up BOOLEAN,
            resolution_event_id VARCHAR NOT NULL,
            input_provenance_sha256 VARCHAR NOT NULL,
            row_sha256 VARCHAR NOT NULL,
            PRIMARY KEY(dataset_id, feature_id)
        );
        """
    )
    manifest_values = [
        dataset.dataset_id,
        POLYMARKET_DATASET_SCHEMA_VERSION,
        POLYMARKET_FEATURE_SCHEMA_VERSION,
        dataset.run_id,
        _canonical_json(dataset.config.asdict()),
        _canonical_json(list(POLYMARKET_FEATURE_NAMES)),
        dataset.candidate_count,
        len(dataset.rows),
        dataset.labeled_row_count,
        _canonical_json(dataset.skipped_counts),
        _canonical_json(dataset.labeled_market_counts),
        _canonical_json(list(dataset.shadow_errors)),
        _canonical_json(list(dataset.training_errors)),
        _canonical_json(dataset.replay_diagnostics),
        _canonical_json(dataset.coverage),
        dataset.dataset_sha256,
    ]
    expected_stored_rows = sorted(
        (
            dataset.dataset_id,
            row.feature_id,
            row.run_id,
            row.condition_id,
            row.market_id,
            row.asset,
            row.decision_event_id,
            row.decision_received_wall_ms,
            row.decision_received_monotonic_ns,
            _canonical_json([format(value, ".17g") for value in row.feature_values]),
            row.official_up,
            row.resolution_event_id,
            row.input_provenance_sha256,
            row.row_sha256,
        )
        for row in dataset.rows
    )
    existing = connection.execute(
        """
        SELECT dataset_id, schema_version, feature_schema_version, run_id,
               config_json, feature_names_json, candidate_count, row_count,
               labeled_row_count, skipped_counts_json,
               labeled_market_counts_json, shadow_errors_json,
               training_errors_json, replay_diagnostics_json, coverage_json,
               dataset_sha256
        FROM polymarket_feature_dataset WHERE dataset_id = ?
        """,
        [dataset.dataset_id],
    ).fetchone()
    if existing is not None:
        if list(existing) != manifest_values:
            raise ValueError("stored Polymarket dataset manifest is inconsistent")
        stored_rows = connection.execute(
            """
            SELECT dataset_id, feature_id, run_id, condition_id, market_id,
                   asset, decision_event_id, decision_received_wall_ms,
                   decision_received_monotonic_ns, feature_values_json,
                   official_up, resolution_event_id, input_provenance_sha256,
                   row_sha256
            FROM polymarket_feature_row
            WHERE dataset_id = ? ORDER BY feature_id
            """,
            [dataset.dataset_id],
        ).fetchall()
        if [tuple(item) for item in stored_rows] != expected_stored_rows:
            raise ValueError("stored Polymarket feature rows are inconsistent")
        return PolymarketFeatureMaterialization(
            dataset_id=dataset.dataset_id,
            status="existing",
            row_count=len(dataset.rows),
            labeled_row_count=dataset.labeled_row_count,
            dataset_sha256=dataset.dataset_sha256,
        )

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT INTO polymarket_feature_dataset VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            manifest_values,
        )
        if expected_stored_rows:
            connection.executemany(
                """
                INSERT INTO polymarket_feature_row VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                expected_stored_rows,
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return PolymarketFeatureMaterialization(
        dataset_id=dataset.dataset_id,
        status="created",
        row_count=len(dataset.rows),
        labeled_row_count=dataset.labeled_row_count,
        dataset_sha256=dataset.dataset_sha256,
    )


__all__ = [
    "POLYMARKET_DATASET_SCHEMA_VERSION",
    "POLYMARKET_FEATURE_NAMES",
    "POLYMARKET_FEATURE_SCHEMA_VERSION",
    "PolymarketFeatureConfig",
    "PolymarketFeatureDataset",
    "PolymarketFeatureMaterialization",
    "PolymarketFeatureRow",
    "PolymarketFeatureSourceContext",
    "build_polymarket_feature_dataset",
    "load_polymarket_feature_source_context",
    "materialize_polymarket_feature_dataset",
    "polymarket_feature_row_sha256",
]

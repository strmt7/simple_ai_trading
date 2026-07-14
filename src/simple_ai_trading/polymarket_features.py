"""Leakage-safe feature rows from immutable Polymarket prospective evidence."""

from __future__ import annotations

from bisect import bisect_right, insort
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Mapping, Sequence

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket_coverage import inspect_polymarket_feed_coverage
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
    PolymarketResolutionEvidence,
)


POLYMARKET_FEATURE_SCHEMA_VERSION = "polymarket-causal-feature-v1"
POLYMARKET_DATASET_SCHEMA_VERSION = "polymarket-causal-dataset-v1"
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
    "rtds_binance_chainlink_basis_bps",
    "direct_rtds_binance_basis_bps",
    "binance_best_bid",
    "binance_best_ask",
    "binance_spread_bps",
    "binance_top_imbalance",
    "binance_return_250ms_bps",
    "binance_return_1000ms_bps",
    "binance_return_5000ms_bps",
    "binance_realized_volatility_1000ms_bps",
    "binance_realized_volatility_5000ms_bps",
    "binance_trade_imbalance_250ms",
    "binance_trade_imbalance_1000ms",
    "binance_trade_imbalance_5000ms",
    "log1p_binance_trade_quote_250ms",
    "log1p_binance_trade_quote_1000ms",
    "log1p_binance_trade_quote_5000ms",
    "direct_binance_age_ms",
    "chainlink_source_age_ms",
    "chainlink_arrival_age_ms",
    "rtds_binance_source_age_ms",
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
    maximum_rtds_binance_source_age_ms: int = 120_000
    maximum_chainlink_anchor_gap_ms: int = 2_000
    minimum_resolved_markets_per_asset: int = 30

    def validated(self) -> "PolymarketFeatureConfig":
        bounds = {
            "cadence_ms": (50, 5_000),
            "warmup_ms": (0, 60_000),
            "maximum_clob_age_ms": (50, 10_000),
            "maximum_direct_binance_age_ms": (50, 10_000),
            "maximum_chainlink_source_age_ms": (250, 30_000),
            "maximum_chainlink_arrival_age_ms": (250, 30_000),
            "maximum_rtds_binance_source_age_ms": (1_000, 300_000),
            "maximum_chainlink_anchor_gap_ms": (0, 10_000),
            "minimum_resolved_markets_per_asset": (1, 100_000),
        }
        for name, (minimum, maximum) in bounds.items():
            value = int(getattr(self, name))
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
        return self

    def asdict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


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


@dataclass(frozen=True)
class _MarketSnapshotPoint:
    condition_id: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    snapshot_sha256: str


@dataclass(frozen=True)
class _PricePoint:
    asset: str
    source_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    price: float
    event_id: str
    event_sha256: str


@dataclass(frozen=True)
class _BinanceBookPoint:
    asset: str
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


@dataclass(frozen=True)
class _BinanceTradePoint:
    asset: str
    received_monotonic_ns: int
    signed_quote: Decimal
    gross_quote: Decimal
    event_id: str
    event_sha256: str


class _PriceCursor:
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

    def advance(self, received_monotonic_ns: int) -> None:
        while (
            self.index < len(self.points)
            and self.points[self.index].received_monotonic_ns
            <= received_monotonic_ns
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
            self.index += 1

    def latest_at_or_before(self, source_time_ms: int) -> _PricePoint | None:
        index = bisect_right(
            self.available,
            int(source_time_ms),
            key=lambda item: item[0],
        )
        return None if index == 0 else self.available[index - 1][3]


class _BookCursor:
    def __init__(self, points: Sequence[_BinanceBookPoint]) -> None:
        self.points = tuple(
            sorted(
                points,
                key=lambda item: (item.received_monotonic_ns, item.event_id),
            )
        )
        self.index = 0
        self.latest: _BinanceBookPoint | None = None

    def advance(self, received_monotonic_ns: int) -> _BinanceBookPoint | None:
        while (
            self.index < len(self.points)
            and self.points[self.index].received_monotonic_ns
            <= received_monotonic_ns
        ):
            self.latest = self.points[self.index]
            self.index += 1
        return self.latest


class _BookSeries:
    def __init__(self, points: Sequence[_BinanceBookPoint]) -> None:
        self.points = tuple(
            sorted(
                points,
                key=lambda item: (item.received_monotonic_ns, item.event_id),
            )
        )
        self.times = tuple(item.received_monotonic_ns for item in self.points)
        prefix_digests = [
            _canonical_sha256(
                {
                    "schema_version": "binance-book-causal-prefix-v1",
                    "stream": "binance_spot",
                }
            )
        ]
        for item in self.points:
            prefix_digests.append(
                _canonical_sha256(
                    {
                        "schema_version": "binance-book-causal-prefix-v1",
                        "previous_sha256": prefix_digests[-1],
                        "event_id": item.event_id,
                        "event_sha256": item.event_sha256,
                        "received_monotonic_ns": item.received_monotonic_ns,
                    }
                )
            )
        self.prefix_digests = tuple(prefix_digests)
        squared_returns = [0.0]
        for previous, current in zip(self.points, self.points[1:]):
            value = math.log(current.midpoint / previous.midpoint)
            squared_returns.append(value * value)
        self.squared_returns = tuple(squared_returns)

    def _index_at_or_before(self, received_monotonic_ns: int) -> int:
        return bisect_right(self.times, int(received_monotonic_ns)) - 1

    def has_lookback(self, received_monotonic_ns: int, window_ms: int) -> bool:
        if not self.times:
            return False
        return self.times[0] <= (
            received_monotonic_ns - int(window_ms) * 1_000_000
        )

    def causal_prefix(self, received_monotonic_ns: int) -> tuple[int, str]:
        count = bisect_right(self.times, int(received_monotonic_ns))
        return count, self.prefix_digests[count]

    def return_bps(self, received_monotonic_ns: int, window_ms: int) -> float:
        current = self._index_at_or_before(received_monotonic_ns)
        previous = self._index_at_or_before(
            received_monotonic_ns - int(window_ms) * 1_000_000
        )
        if current < 0 or previous < 0:
            return 0.0
        return _log_basis_bps(
            self.points[current].midpoint,
            self.points[previous].midpoint,
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
        variance = math.fsum(
            self.squared_returns[first_return : current + 1]
        )
        return 10_000.0 * math.sqrt(max(0.0, variance))


class _TradeSeries:
    def __init__(self, points: Sequence[_BinanceTradePoint]) -> None:
        self.points = tuple(
            sorted(
                points,
                key=lambda item: (item.received_monotonic_ns, item.event_id),
            )
        )
        self.times = tuple(item.received_monotonic_ns for item in self.points)
        prefix_digests = [
            _canonical_sha256(
                {
                    "schema_version": "binance-trade-causal-prefix-v1",
                    "stream": "binance_spot",
                }
            )
        ]
        signed_prefix = [Decimal(0)]
        gross_prefix = [Decimal(0)]
        for item in self.points:
            signed_prefix.append(signed_prefix[-1] + item.signed_quote)
            gross_prefix.append(gross_prefix[-1] + item.gross_quote)
            prefix_digests.append(
                _canonical_sha256(
                    {
                        "schema_version": "binance-trade-causal-prefix-v1",
                        "previous_sha256": prefix_digests[-1],
                        "event_id": item.event_id,
                        "event_sha256": item.event_sha256,
                        "received_monotonic_ns": item.received_monotonic_ns,
                    }
                )
            )
        self.signed_prefix = tuple(signed_prefix)
        self.gross_prefix = tuple(gross_prefix)
        self.prefix_digests = tuple(prefix_digests)

    def causal_prefix(self, received_monotonic_ns: int) -> tuple[int, str]:
        count = bisect_right(self.times, int(received_monotonic_ns))
        return count, self.prefix_digests[count]

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
) -> tuple[
    dict[str, tuple[_PricePoint, ...]],
    dict[str, tuple[_PricePoint, ...]],
    dict[str, tuple[_BinanceBookPoint, ...]],
    dict[str, tuple[_BinanceTradePoint, ...]],
]:
    chainlink: dict[str, list[_PricePoint]] = {key: [] for key in _ASSETS}
    rtds_binance: dict[str, list[_PricePoint]] = {key: [] for key in _ASSETS}
    direct_books: dict[str, list[_BinanceBookPoint]] = {key: [] for key in _ASSETS}
    direct_trades: dict[str, list[_BinanceTradePoint]] = {key: [] for key in _ASSETS}
    rows = store.connect().execute(
        """
        SELECT e.event_id, e.event_sha256, e.stream, e.event_type, e.symbol,
               e.event_json, r.received_wall_ms, r.received_monotonic_ns
        FROM polymarket_public_event AS e
        JOIN polymarket_raw_message AS r ON r.message_id = e.message_id
        WHERE e.run_id = ? AND e.stream IN ('binance_spot', 'polymarket_rtds')
        ORDER BY r.received_monotonic_ns, e.event_id
        """,
        [run_id],
    ).fetchall()
    for (
        event_id,
        event_sha256,
        stream,
        event_type,
        symbol,
        event_json,
        received_wall_ms,
        received_monotonic_ns,
    ) in rows:
        asset = str(symbol).upper()
        if asset not in chainlink:
            raise ValueError("prospective feature evidence contains an unsupported asset")
        try:
            event = json.loads(str(event_json))
        except json.JSONDecodeError as exc:
            raise ValueError("prospective feature evidence contains invalid JSON") from exc
        if not isinstance(event, Mapping):
            raise ValueError("prospective feature event must be an object")
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
        target = (
            chainlink[asset]
            if normalized_type.startswith("crypto_prices_chainlink:")
            else rtds_binance[asset]
        )
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
            target.append(
                _PricePoint(
                    asset=asset,
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
    return (
        {key: tuple(value) for key, value in chainlink.items()},
        {key: tuple(value) for key, value in rtds_binance.items()},
        {key: tuple(value) for key, value in direct_books.items()},
        {key: tuple(value) for key, value in direct_trades.items()},
    )


def _load_market_snapshot_points(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> dict[str, _MarketSnapshotPoint]:
    rows = store.connect().execute(
        """
        SELECT condition_id, observed_wall_ms, observed_monotonic_ns,
               snapshot_sha256
        FROM polymarket_market_snapshot
        WHERE run_id = ? ORDER BY condition_id
        """,
        [run_id],
    ).fetchall()
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
    microprice = (
        best_ask * bid_quantity + best_bid * ask_quantity
    ) / (bid_quantity + ask_quantity)
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


def build_polymarket_feature_dataset(
    store: PolymarketEvidenceStore,
    *,
    run_id: str | None = None,
    config: PolymarketFeatureConfig | None = None,
) -> PolymarketFeatureDataset:
    """Build causal features; official outcomes are attached only as future labels."""

    cfg = (config or PolymarketFeatureConfig()).validated()
    replay = PolymarketEvidenceReplay.load(store, run_id=run_id)
    selected = replay.run_id
    coverage = inspect_polymarket_feed_coverage(
        store,
        run_id=selected,
        minimum_resolved_markets_per_asset=cfg.minimum_resolved_markets_per_asset,
    )
    chainlink, rtds_binance, direct_books, direct_trades = _parse_feed_points(
        store, selected
    )
    market_snapshots = _load_market_snapshot_points(store, selected)
    chainlink_cursors = {
        asset: _PriceCursor(points) for asset, points in chainlink.items()
    }
    rtds_cursors = {
        asset: _PriceCursor(points) for asset, points in rtds_binance.items()
    }
    book_cursors = {
        asset: _BookCursor(points) for asset, points in direct_books.items()
    }
    book_series = {
        asset: _BookSeries(points) for asset, points in direct_books.items()
    }
    trade_series = {
        asset: _TradeSeries(points) for asset, points in direct_trades.items()
    }
    resolution_by_condition: dict[str, PolymarketResolutionEvidence] = {
        item.condition_id: item for item in replay.resolutions
    }
    states: dict[str, dict[str, PolymarketRecordedBook]] = {}
    last_emitted_ns: dict[str, int] = {}
    skipped: dict[str, int] = {}
    rows: list[PolymarketFeatureRow] = []
    candidate_count = 0

    for trigger in replay.books:
        market = trigger.market
        condition = market.condition_id
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
        rtds_cursor = rtds_cursors[asset]
        chainlink_cursor.advance(decision_ns)
        rtds_cursor.advance(decision_ns)
        direct_book = book_cursors[asset].advance(decision_ns)
        anchor = chainlink_cursor.latest_at_or_before(market.event_start_ms)
        chainlink_now = chainlink_cursor.latest_at_or_before(decision_wall_ms)
        rtds_now = rtds_cursor.latest_at_or_before(decision_wall_ms)
        if anchor is None:
            _skip(skipped, "missing_chainlink_open_anchor")
            continue
        anchor_gap_ms = market.event_start_ms - anchor.source_time_ms
        if anchor_gap_ms < 0 or anchor_gap_ms > cfg.maximum_chainlink_anchor_gap_ms:
            _skip(skipped, "chainlink_open_anchor_gap")
            continue
        if chainlink_now is None:
            _skip(skipped, "missing_chainlink_current_price")
            continue
        if rtds_now is None:
            _skip(skipped, "missing_rtds_binance_price")
            continue
        if direct_book is None:
            _skip(skipped, "missing_direct_binance_book")
            continue

        direct_age_ms = (decision_ns - direct_book.received_monotonic_ns) / 1_000_000.0
        chainlink_arrival_age_ms = (
            decision_ns - chainlink_now.received_monotonic_ns
        ) / 1_000_000.0
        chainlink_source_age_ms = decision_wall_ms - chainlink_now.source_time_ms
        rtds_source_age_ms = decision_wall_ms - rtds_now.source_time_ms
        if (
            direct_age_ms < 0.0
            or direct_age_ms > cfg.maximum_direct_binance_age_ms
        ):
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
        if (
            rtds_source_age_ms < 0
            or rtds_source_age_ms > cfg.maximum_rtds_binance_source_age_ms
        ):
            _skip(skipped, "stale_or_future_rtds_binance_price")
            continue
        if not book_series[asset].has_lookback(decision_ns, 5_000):
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
        trade_250, gross_250 = trade_series[asset].stats(decision_ns, 250)
        trade_1000, gross_1000 = trade_series[asset].stats(decision_ns, 1_000)
        trade_5000, gross_5000 = trade_series[asset].stats(decision_ns, 5_000)
        book_prefix_count, book_prefix_sha256 = book_series[asset].causal_prefix(
            decision_ns
        )
        trade_prefix_count, trade_prefix_sha256 = trade_series[asset].causal_prefix(
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
            _log_basis_bps(rtds_now.price, chainlink_now.price),
            _log_basis_bps(direct_book.midpoint, rtds_now.price),
            direct_book.bid,
            direct_book.ask,
            10_000.0 * (direct_book.ask - direct_book.bid) / direct_book.midpoint,
            _ratio_imbalance(direct_book.bid_quantity, direct_book.ask_quantity),
            book_series[asset].return_bps(decision_ns, 250),
            book_series[asset].return_bps(decision_ns, 1_000),
            book_series[asset].return_bps(decision_ns, 5_000),
            book_series[asset].realized_volatility_bps(decision_ns, 1_000),
            book_series[asset].realized_volatility_bps(decision_ns, 5_000),
            trade_250,
            trade_1000,
            trade_5000,
            math.log1p(gross_250),
            math.log1p(gross_1000),
            math.log1p(gross_5000),
            direct_age_ms,
            float(chainlink_source_age_ms),
            chainlink_arrival_age_ms,
            float(rtds_source_age_ms),
            float(anchor_gap_ms),
            math.log1p(float(market.liquidity_quote)),
            math.log1p(float(market.volume_quote)),
        )
        if len(values) != len(POLYMARKET_FEATURE_NAMES) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("Polymarket feature vector is non-finite or misaligned")
        resolution = resolution_by_condition.get(condition)
        official_up = (
            None if resolution is None else resolution.winning_outcome == "Up"
        )
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
                    "source_time_ms": up.snapshot.source_time_ms,
                    "received_wall_ms": up.received_wall_ms,
                    "received_monotonic_ns": up.received_monotonic_ns,
                    "state_provenance_sha256": up.snapshot.source_payload_sha256,
                },
                "down_book": {
                    "event_id": down.event_id,
                    "source_time_ms": down.snapshot.source_time_ms,
                    "received_wall_ms": down.received_wall_ms,
                    "received_monotonic_ns": down.received_monotonic_ns,
                    "state_provenance_sha256": down.snapshot.source_payload_sha256,
                },
                "direct_binance_latest_book": {
                    "event_id": direct_book.event_id,
                    "event_sha256": direct_book.event_sha256,
                    "received_wall_ms": direct_book.received_wall_ms,
                    "received_monotonic_ns": direct_book.received_monotonic_ns,
                },
                "direct_binance_causal_prefix": {
                    "book_event_count": book_prefix_count,
                    "book_prefix_sha256": book_prefix_sha256,
                    "trade_event_count": trade_prefix_count,
                    "trade_prefix_sha256": trade_prefix_sha256,
                },
                "chainlink_anchor": {
                    "event_id": anchor.event_id,
                    "event_sha256": anchor.event_sha256,
                    "source_time_ms": anchor.source_time_ms,
                    "received_wall_ms": anchor.received_wall_ms,
                    "received_monotonic_ns": anchor.received_monotonic_ns,
                },
                "chainlink_current": {
                    "event_id": chainlink_now.event_id,
                    "event_sha256": chainlink_now.event_sha256,
                    "source_time_ms": chainlink_now.source_time_ms,
                    "received_wall_ms": chainlink_now.received_wall_ms,
                    "received_monotonic_ns": chainlink_now.received_monotonic_ns,
                },
                "rtds_binance_current": {
                    "event_id": rtds_now.event_id,
                    "event_sha256": rtds_now.event_sha256,
                    "source_time_ms": rtds_now.source_time_ms,
                    "received_wall_ms": rtds_now.received_wall_ms,
                    "received_monotonic_ns": rtds_now.received_monotonic_ns,
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

    report_row = store.connect().execute(
        "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
        [selected],
    ).fetchone()
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

    report_row = store.connect().execute(
        "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
        [dataset.run_id],
    ).fetchone()
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
    "build_polymarket_feature_dataset",
    "materialize_polymarket_feature_dataset",
]

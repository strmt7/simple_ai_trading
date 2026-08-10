"""Target-blind causal CLOB features for Polymarket BTC five-minute markets."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import re
from typing import Sequence

from .paper_execution import PaperBookSnapshot
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_DECISION_CADENCE_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION = (
    "polymarket-round25-causal-clob-features-v1"
)
POLYMARKET_ROUND25_CLOB_MAXIMUM_RECEIPT_AGE_MS = 500
POLYMARKET_ROUND25_CLOB_MAXIMUM_SOURCE_AGE_MS = 5_000
POLYMARKET_ROUND25_CLOB_WINDOWS_MS = (250, 1_000, 5_000, 15_000, 30_000, 60_000)
_BOOK_SNAPSHOT_METRICS = (
    "best_bid",
    "best_ask",
    "midpoint",
    "spread",
    "relative_spread",
    "microprice",
    "top_quantity_imbalance",
    "bid_depth_1",
    "ask_depth_1",
    "bid_depth_5",
    "ask_depth_5",
    "bid_depth_20",
    "ask_depth_20",
    "bid_slope_20_bps",
    "ask_slope_20_bps",
)
_BOOK_FLOW_METRICS = (
    "top_order_flow_imbalance",
    "mean_top_quantity_imbalance",
    "microprice_log_return",
    "level_quantity_flow_pressure",
    "log1p_gross_level_quantity_change",
    "book_update_count",
)
POLYMARKET_ROUND25_CLOB_FEATURE_NAMES = (
    *(f"clob.{outcome}_{metric}" for outcome in ("up", "down") for metric in _BOOK_SNAPSHOT_METRICS),
    "clob.normalized_market_prior_up",
    "clob.complement_buy_overround",
    "clob.complement_sell_underround",
    "clob.complement_midpoint_error",
    "clob.up_book_receipt_age_ms",
    "clob.down_book_receipt_age_ms",
    "clob.book_receipt_skew_ms",
    "clob.up_book_source_age_ms",
    "clob.down_book_source_age_ms",
    *(
        f"clob.{outcome}_{metric}_{window_ms}ms"
        for outcome in ("up", "down")
        for window_ms in POLYMARKET_ROUND25_CLOB_WINDOWS_MS
        for metric in _BOOK_FLOW_METRICS
    ),
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAX_LEVELS = 20
_MAXIMUM_LOOKBACK_MS = max(POLYMARKET_ROUND25_CLOB_WINDOWS_MS)
_COMPACT_THRESHOLD = 100_000
_COMPACT_MINIMUM_DROP = 20_000


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + _canonical_json(value).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class _BookPoint:
    source_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    source_payload_sha256: str

    @property
    def best_bid(self) -> tuple[float, float]:
        return self.bids[0]

    @property
    def best_ask(self) -> tuple[float, float]:
        return self.asks[0]

    @property
    def microprice(self) -> float:
        bid, bid_quantity = self.best_bid
        ask, ask_quantity = self.best_ask
        return (ask * bid_quantity + bid * ask_quantity) / (
            bid_quantity + ask_quantity
        )


@dataclass(frozen=True, slots=True)
class _BookFlowWindow:
    top_ofi: float
    mean_imbalance: float
    microprice_return: float
    level_pressure: float
    gross_level_change: float
    count: int


class _RollingBookSeries:
    def __init__(self) -> None:
        self.received_ms: list[int] = []
        self.top_ofi: list[float] = []
        self.top_gross: list[float] = []
        self.top_imbalance: list[float] = []
        self.log_microprice: list[float] = []
        self.level_pressure: list[float] = []
        self.level_gross: list[float] = []
        self.prefixes: list[list[float]] = [[0.0] for _ in range(5)]
        self.latest: _BookPoint | None = None

    @staticmethod
    def _top_flow(previous: _BookPoint, current: _BookPoint) -> tuple[float, float]:
        prior_bid, prior_bid_quantity = previous.best_bid
        bid, bid_quantity = current.best_bid
        prior_ask, prior_ask_quantity = previous.best_ask
        ask, ask_quantity = current.best_ask
        bid_flow = (
            bid_quantity
            if bid > prior_bid
            else (
                bid_quantity - prior_bid_quantity
                if bid == prior_bid
                else -prior_bid_quantity
            )
        )
        ask_flow = (
            ask_quantity
            if ask < prior_ask
            else (
                ask_quantity - prior_ask_quantity
                if ask == prior_ask
                else -prior_ask_quantity
            )
        )
        return bid_flow - ask_flow, abs(bid_flow) + abs(ask_flow)

    @staticmethod
    def _level_flow(previous: _BookPoint, current: _BookPoint) -> tuple[float, float]:
        previous_bids = dict(previous.bids)
        current_bids = dict(current.bids)
        previous_asks = dict(previous.asks)
        current_asks = dict(current.asks)
        bid_changes = tuple(
            current_bids.get(price, 0.0) - previous_bids.get(price, 0.0)
            for price in set(previous_bids) | set(current_bids)
        )
        ask_changes = tuple(
            current_asks.get(price, 0.0) - previous_asks.get(price, 0.0)
            for price in set(previous_asks) | set(current_asks)
        )
        return (
            math.fsum(bid_changes) - math.fsum(ask_changes),
            math.fsum(abs(value) for value in (*bid_changes, *ask_changes)),
        )

    def _rebuild_prefixes(self) -> None:
        self.prefixes = [[0.0] for _ in range(5)]
        rows = zip(
            self.top_ofi,
            self.top_gross,
            self.top_imbalance,
            self.level_pressure,
            self.level_gross,
            strict=True,
        )
        for row in rows:
            for index, value in enumerate(row):
                self.prefixes[index].append(self.prefixes[index][-1] + value)

    def _compact(self, latest_ms: int) -> None:
        if len(self.received_ms) < _COMPACT_THRESHOLD:
            return
        cutoff = bisect_left(
            self.received_ms,
            latest_ms - _MAXIMUM_LOOKBACK_MS - 1_000,
        )
        drop = max(0, cutoff - 1)
        if drop < _COMPACT_MINIMUM_DROP:
            return
        self.received_ms = self.received_ms[drop:]
        self.top_ofi = self.top_ofi[drop:]
        self.top_gross = self.top_gross[drop:]
        self.top_imbalance = self.top_imbalance[drop:]
        self.log_microprice = self.log_microprice[drop:]
        self.level_pressure = self.level_pressure[drop:]
        self.level_gross = self.level_gross[drop:]
        self._rebuild_prefixes()

    def append(self, point: _BookPoint) -> None:
        if self.received_ms and point.received_wall_ms < self.received_ms[-1]:
            raise ValueError("Round 25 CLOB outcome receipt time regressed")
        bid_quantity = point.best_bid[1]
        ask_quantity = point.best_ask[1]
        imbalance = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)
        if self.latest is None:
            top_signed = top_gross = level_signed = level_gross = 0.0
        else:
            top_signed, top_gross = self._top_flow(self.latest, point)
            level_signed, level_gross = self._level_flow(self.latest, point)
        self.received_ms.append(point.received_wall_ms)
        self.top_ofi.append(top_signed)
        self.top_gross.append(top_gross)
        self.top_imbalance.append(imbalance)
        self.log_microprice.append(math.log(point.microprice))
        self.level_pressure.append(level_signed)
        self.level_gross.append(level_gross)
        for prefix, value in zip(
            self.prefixes,
            (top_signed, top_gross, imbalance, level_signed, level_gross),
            strict=True,
        ):
            prefix.append(prefix[-1] + value)
        self.latest = point
        self._compact(point.received_wall_ms)

    def window(self, decision_ms: int, window_ms: int) -> _BookFlowWindow:
        window_start = decision_ms - window_ms
        left = bisect_left(self.received_ms, window_start)
        right = bisect_right(self.received_ms, decision_ms)
        count = right - left
        if count <= 0:
            return _BookFlowWindow(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        anchor = max(0, bisect_right(self.received_ms, window_start) - 1)
        top_signed, top_gross, imbalance, level_signed, level_gross = (
            prefix[right] - prefix[left] for prefix in self.prefixes
        )
        gross_tolerance = 1e-12 * max(1.0, top_gross, level_gross)
        if (
            top_gross < -gross_tolerance
            or level_gross < -gross_tolerance
            or abs(top_signed) > top_gross + gross_tolerance
            or abs(level_signed) > level_gross + gross_tolerance
            or abs(imbalance) > count + 1e-12 * max(1, count)
        ):
            raise RuntimeError("Round 25 CLOB flow accounting differs")
        top_gross = max(0.0, top_gross)
        level_gross = max(0.0, level_gross)
        top_signed = max(-top_gross, min(top_gross, top_signed))
        level_signed = max(-level_gross, min(level_gross, level_signed))
        imbalance = max(-float(count), min(float(count), imbalance))
        return _BookFlowWindow(
            top_ofi=0.0 if top_gross == 0.0 else top_signed / top_gross,
            mean_imbalance=imbalance / count,
            microprice_return=(
                0.0
                if anchor >= right - 1
                else self.log_microprice[right - 1] - self.log_microprice[anchor]
            ),
            level_pressure=(
                0.0 if level_gross == 0.0 else level_signed / level_gross
            ),
            gross_level_change=level_gross,
            count=count,
        )


def _book_point(snapshot: PaperBookSnapshot) -> _BookPoint:
    if (
        not isinstance(snapshot, PaperBookSnapshot)
        or type(snapshot.source_time_ms) is not int
        or type(snapshot.received_wall_ms) is not int
        or type(snapshot.received_monotonic_ns) is not int
        or snapshot.connected is not True
        or snapshot.gap_free is not True
    ):
        raise ValueError("Round 25 CLOB snapshot contract differs")
    validated = snapshot.validated()
    if (
        validated != snapshot
        or validated.venue != "polymarket"
        or not 1 <= len(validated.bids) <= _MAX_LEVELS
        or not 1 <= len(validated.asks) <= _MAX_LEVELS
        or any(
            not Decimal("0") < level.price < Decimal("1")
            for level in (*validated.bids, *validated.asks)
        )
        or validated.source_time_ms <= 0
        or validated.received_wall_ms <= 0
        or validated.received_monotonic_ns <= 0
    ):
        raise ValueError("Round 25 CLOB snapshot contract differs")
    return _BookPoint(
        source_time_ms=validated.source_time_ms,
        received_wall_ms=validated.received_wall_ms,
        received_monotonic_ns=validated.received_monotonic_ns,
        bids=tuple((float(level.price), float(level.quantity)) for level in validated.bids),
        asks=tuple((float(level.price), float(level.quantity)) for level in validated.asks),
        source_payload_sha256=validated.source_payload_sha256,
    )


def _snapshot_values(point: _BookPoint) -> tuple[float, ...]:
    bid, bid_quantity = point.best_bid
    ask, ask_quantity = point.best_ask
    midpoint = 0.5 * (bid + ask)
    spread = ask - bid

    def depth(levels: tuple[tuple[float, float], ...], count: int) -> float:
        return math.fsum(quantity for _price, quantity in levels[:count])

    bid_last = point.bids[min(19, len(point.bids) - 1)][0]
    ask_last = point.asks[min(19, len(point.asks) - 1)][0]
    return (
        bid,
        ask,
        midpoint,
        spread,
        spread / midpoint,
        point.microprice,
        (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity),
        bid_quantity,
        ask_quantity,
        depth(point.bids, 5),
        depth(point.asks, 5),
        depth(point.bids, 20),
        depth(point.asks, 20),
        (bid - bid_last) / bid * 10_000.0,
        (ask_last - ask) / ask * 10_000.0,
    )


@dataclass(frozen=True, slots=True)
class Round25ClobFeatureSnapshot:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    available: bool
    reasons: tuple[str, ...]
    market_prior_probability: float
    values: tuple[float, ...]
    source_chain_sha256: str
    maximum_receipt_ms: int
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or type(self.decision_time_ms) is not int
            or type(self.available) is not bool
            or not isinstance(self.reasons, tuple)
            or any(not isinstance(reason, str) or not reason for reason in self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
            or not isinstance(self.values, tuple)
            or len(self.values) != len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in self.values
            )
            or not isinstance(self.source_chain_sha256, str)
            or _SHA256.fullmatch(self.source_chain_sha256) is None
            or type(self.maximum_receipt_ms) is not int
            or self.model_design_sha256 != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.trading_authority is not False
        ):
            raise ValueError("Round 25 CLOB feature snapshot is invalid")
        if self.available:
            if (
                self.reasons
                or not 0.0 < self.market_prior_probability < 1.0
                or self.source_chain_sha256 == _EMPTY_SHA256
                or not 0 < self.maximum_receipt_ms <= self.decision_time_ms
            ):
                raise ValueError("Round 25 available CLOB snapshot differs")
        elif (
            not self.reasons
            or self.market_prior_probability != 0.5
            or any(self.values)
            or self.source_chain_sha256 != _EMPTY_SHA256
            or self.maximum_receipt_ms != 0
        ):
            raise ValueError("Round 25 unavailable CLOB snapshot differs")


class Round25ClobFeatureEngine:
    """Build efficient receipt-causal book features with no settlement target."""

    feature_schema_version = POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION
    feature_names = POLYMARKET_ROUND25_CLOB_FEATURE_NAMES
    credentials_used = False
    account_connected = False
    execution_connected = False
    trading_authority = False

    def __init__(
        self,
        *,
        condition_id: str,
        up_token_id: str,
        down_token_id: str,
        event_start_ms: int,
    ) -> None:
        if (
            not isinstance(condition_id, str)
            or not isinstance(up_token_id, str)
            or not isinstance(down_token_id, str)
            or type(event_start_ms) is not int
        ):
            raise ValueError("Round 25 CLOB condition identity is invalid")
        condition = condition_id.strip().lower()
        up = up_token_id.strip()
        down = down_token_id.strip()
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or _TOKEN_ID.fullmatch(up) is None
            or _TOKEN_ID.fullmatch(down) is None
            or up == down
            or event_start_ms <= 0
            or event_start_ms % POLYMARKET_ROUND25_CONDITION_DURATION_MS
        ):
            raise ValueError("Round 25 CLOB condition identity is invalid")
        self.condition_id = condition
        self.up_token_id = up
        self.down_token_id = down
        self.event_start_ms = event_start_ms
        self.event_end_ms = event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
        self._series = {up: _RollingBookSeries(), down: _RollingBookSeries()}
        self._last_token_monotonic_ns = {up: 0, down: 0}
        self._source_chain = _EMPTY_SHA256
        self._last_monotonic_ns = 0
        self._gap_detected = False

    def mark_stream_gap(self) -> None:
        self._gap_detected = True

    def ingest(self, snapshot: PaperBookSnapshot) -> None:
        point = _book_point(snapshot)
        if (
            snapshot.market_id != self.condition_id
            or snapshot.asset_id not in self._series
            or point.received_monotonic_ns < self._last_monotonic_ns
            or point.received_monotonic_ns
            <= self._last_token_monotonic_ns[snapshot.asset_id]
            or not self.event_start_ms <= point.source_time_ms <= self.event_end_ms
        ):
            raise ValueError("Round 25 CLOB snapshot identity or chronology differs")
        self._series[snapshot.asset_id].append(point)
        self._source_chain = _chain(
            self._source_chain,
            {
                "asset_id": snapshot.asset_id,
                "received_monotonic_ns": point.received_monotonic_ns,
                "received_wall_ms": point.received_wall_ms,
                "source_payload_sha256": point.source_payload_sha256,
                "source_time_ms": point.source_time_ms,
            },
        )
        self._last_token_monotonic_ns[snapshot.asset_id] = point.received_monotonic_ns
        self._last_monotonic_ns = point.received_monotonic_ns

    def _unavailable(
        self, *, decision_time_ms: int, reasons: Sequence[str]
    ) -> Round25ClobFeatureSnapshot:
        return Round25ClobFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision_time_ms,
            available=False,
            reasons=tuple(dict.fromkeys(reasons)),
            market_prior_probability=0.5,
            values=(0.0,) * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES),
            source_chain_sha256=_EMPTY_SHA256,
            maximum_receipt_ms=0,
        )

    def build(self, decision_time_ms: int) -> Round25ClobFeatureSnapshot:
        if type(decision_time_ms) is not int:
            raise ValueError("Round 25 CLOB decision time is invalid")
        decision = decision_time_ms
        if (
            not self.event_start_ms <= decision < self.event_end_ms
            or (decision - self.event_start_ms)
            % POLYMARKET_ROUND25_DECISION_CADENCE_MS
        ):
            raise ValueError("Round 25 CLOB decision time is invalid")
        if any(
            series.latest is not None and series.latest.received_wall_ms > decision
            for series in self._series.values()
        ):
            raise ValueError("Round 25 CLOB engine contains future receipts")
        reasons: list[str] = []
        if self._gap_detected:
            reasons.append("clob_stream_gap_detected")
        points: dict[str, _BookPoint] = {}
        for token, outcome in (
            (self.up_token_id, "up"),
            (self.down_token_id, "down"),
        ):
            point = self._series[token].latest
            if point is None:
                reasons.append(f"{outcome}_book_unavailable")
                continue
            if point.source_time_ms > decision:
                reasons.append(f"future_{outcome}_book_source_timestamp")
            if (
                decision - point.received_wall_ms
                > POLYMARKET_ROUND25_CLOB_MAXIMUM_RECEIPT_AGE_MS
            ):
                reasons.append(f"{outcome}_book_receipt_stale")
            if (
                decision - point.source_time_ms
                > POLYMARKET_ROUND25_CLOB_MAXIMUM_SOURCE_AGE_MS
            ):
                reasons.append(f"{outcome}_book_source_stale")
            points[outcome] = point
        if reasons:
            return self._unavailable(decision_time_ms=decision, reasons=reasons)
        if set(points) != {"up", "down"}:
            raise RuntimeError("Round 25 CLOB availability reconciliation failed")

        up_values = _snapshot_values(points["up"])
        down_values = _snapshot_values(points["down"])
        midpoint_sum = up_values[2] + down_values[2]
        market_prior = up_values[2] / midpoint_sum
        static = (
            *up_values,
            *down_values,
            market_prior,
            (up_values[1] + down_values[1]) - 1.0,
            1.0 - (up_values[0] + down_values[0]),
            midpoint_sum - 1.0,
            float(decision - points["up"].received_wall_ms),
            float(decision - points["down"].received_wall_ms),
            float(
                abs(
                    points["up"].received_wall_ms
                    - points["down"].received_wall_ms
                )
            ),
            float(decision - points["up"].source_time_ms),
            float(decision - points["down"].source_time_ms),
        )
        temporal: list[float] = []
        for token in (self.up_token_id, self.down_token_id):
            for window_ms in POLYMARKET_ROUND25_CLOB_WINDOWS_MS:
                window = self._series[token].window(decision, window_ms)
                temporal.extend(
                    (
                        window.top_ofi,
                        window.mean_imbalance,
                        window.microprice_return,
                        window.level_pressure,
                        math.log1p(window.gross_level_change),
                        float(window.count),
                    )
                )
        values = tuple((*static, *temporal))
        if len(values) != len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in values
        ):
            raise RuntimeError("Round 25 CLOB feature vector differs")
        return Round25ClobFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision,
            available=True,
            reasons=(),
            market_prior_probability=market_prior,
            values=tuple(float(value) for value in values),
            source_chain_sha256=_chain(
                self._source_chain,
                {
                    "condition_id": self.condition_id,
                    "decision_time_ms": decision,
                    "feature_schema_version": self.feature_schema_version,
                    "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
                },
            ),
            maximum_receipt_ms=max(
                points["up"].received_wall_ms,
                points["down"].received_wall_ms,
            ),
        )


__all__ = [
    "POLYMARKET_ROUND25_CLOB_FEATURE_NAMES",
    "POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_CLOB_MAXIMUM_RECEIPT_AGE_MS",
    "POLYMARKET_ROUND25_CLOB_MAXIMUM_SOURCE_AGE_MS",
    "POLYMARKET_ROUND25_CLOB_WINDOWS_MS",
    "Round25ClobFeatureEngine",
    "Round25ClobFeatureSnapshot",
]

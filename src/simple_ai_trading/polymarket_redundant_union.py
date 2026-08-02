"""Bounded, deterministic reconciliation of redundant Polymarket CLOB receipts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
import re
from typing import Any


POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION = "polymarket-redundant-union-v1"
POLYMARKET_REDUNDANT_UNION_AUDIT_SCHEMA_VERSION = (
    "polymarket-redundant-union-audit-v1"
)
POLYMARKET_REDUNDANT_CLOB_LANES = ("clob-a", "clob-b")
_CONNECTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,159}$")
_MAX_RAW_MESSAGE_BYTES = 512 * 1024
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_UNSIGNED_64 = (1 << 64) - 1


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Polymarket CLOB event contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket CLOB event contains {value}")


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


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _source_time_ms(event: Mapping[str, object]) -> int | None:
    value = event.get("timestamp")
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and value.isdigit():
        parsed = int(value)
    else:
        return None
    return parsed if 0 <= parsed <= _MAX_SIGNED_64 else None


@dataclass(frozen=True, slots=True)
class PolymarketClobLaneReceipt:
    """One exact WebSocket frame receipt from a named redundant CLOB lane."""

    lane_id: str
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    raw_text: str

    def validated(self) -> PolymarketClobLaneReceipt:
        raw_bytes = self.raw_text.encode("utf-8", errors="strict")
        if (
            self.lane_id not in POLYMARKET_REDUNDANT_CLOB_LANES
            or _CONNECTION_ID.fullmatch(self.connection_id) is None
            or not 1 <= self.sequence_number <= _MAX_UNSIGNED_64
            or not 0 <= self.received_wall_ms <= _MAX_SIGNED_64
            or not 0 <= self.received_monotonic_ns <= _MAX_UNSIGNED_64
            or len(raw_bytes) > _MAX_RAW_MESSAGE_BYTES
        ):
            raise ValueError("Polymarket redundant CLOB receipt is invalid")
        return self


@dataclass(frozen=True, slots=True)
class PolymarketUnionEvent:
    """One semantic venue event represented once across both receipt lanes."""

    union_sequence_number: int
    semantic_sha256: str
    semantic_occurrence_index: int
    event_type: str
    event_json: str
    source_time_ms: int | None
    selected_received_wall_ms: int
    selected_received_monotonic_ns: int
    selected_lane_id: str
    lane_receipts: tuple[PolymarketClobLaneReceipt, ...]
    event_sha256: str


@dataclass(frozen=True, slots=True)
class PolymarketRedundantUnionAudit:
    """Terminal, hash-bound accounting for one reconciled capture unit."""

    schema_version: str
    pairing_window_ms: int
    union_event_count: int
    shared_event_count: int
    single_lane_event_count: int
    lane_event_counts: dict[str, int]
    lane_coverage_fraction: dict[str, float]
    shared_fraction: float
    event_type_counts: dict[str, int]
    receipt_difference_ms: dict[str, float | None]
    maximum_pending_events_observed: int
    terminal_pending_event_count: int
    audit_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pairing_window_ms": self.pairing_window_ms,
            "union_event_count": self.union_event_count,
            "shared_event_count": self.shared_event_count,
            "single_lane_event_count": self.single_lane_event_count,
            "lane_event_counts": dict(self.lane_event_counts),
            "lane_coverage_fraction": dict(self.lane_coverage_fraction),
            "shared_fraction": self.shared_fraction,
            "event_type_counts": dict(self.event_type_counts),
            "receipt_difference_ms": dict(self.receipt_difference_ms),
            "maximum_pending_events_observed": self.maximum_pending_events_observed,
            "terminal_pending_event_count": self.terminal_pending_event_count,
            "audit_sha256": self.audit_sha256,
        }


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    semantic_sha256: str
    event_type: str
    event_json: str
    source_time_ms: int | None
    receipt: PolymarketClobLaneReceipt


@dataclass(order=True, slots=True)
class _ResolvedEvent:
    ordering_key: tuple[int, int, str, str]
    semantic_sha256: str = field(compare=False)
    event_type: str = field(compare=False)
    event_json: str = field(compare=False)
    source_time_ms: int | None = field(compare=False)
    receipts: tuple[PolymarketClobLaneReceipt, ...] = field(compare=False)


class PolymarketRedundantUnionBuilder:
    """Pair redundant receipts in a bounded window and emit one causal union."""

    def __init__(
        self,
        *,
        pairing_window_ms: int = 2_000,
        maximum_pending_events: int = 100_000,
    ) -> None:
        if not 100 <= pairing_window_ms <= 10_000:
            raise ValueError("pairing_window_ms must lie in [100, 10000]")
        if not 1_000 <= maximum_pending_events <= 1_000_000:
            raise ValueError(
                "maximum_pending_events must lie in [1000, 1000000]"
            )
        self.pairing_window_ms = int(pairing_window_ms)
        self.maximum_pending_events = int(maximum_pending_events)
        self._pairing_window_ns = self.pairing_window_ms * 1_000_000
        self._pending: dict[
            str, dict[str, deque[_PendingEvent]]
        ] = defaultdict(
            lambda: {
                lane: deque() for lane in POLYMARKET_REDUNDANT_CLOB_LANES
            }
        )
        self._resolved: list[_ResolvedEvent] = []
        self._last_sequence: dict[tuple[str, str], int] = {}
        self._last_lane_monotonic: dict[str, int] = {}
        self._last_global_monotonic = -1
        self._last_advanced_monotonic = -1
        self._emitted_occurrences: Counter[str] = Counter()
        self._lane_event_counts: Counter[str] = Counter()
        self._event_type_counts: Counter[str] = Counter()
        self._receipt_differences_ms: list[float] = []
        self._union_event_count = 0
        self._shared_event_count = 0
        self._maximum_pending_events_observed = 0
        self._pending_event_count = 0
        self._finished = False

    def _validate_receipt_order(
        self,
        receipt: PolymarketClobLaneReceipt,
    ) -> None:
        key = (receipt.lane_id, receipt.connection_id)
        previous_sequence = self._last_sequence.get(key)
        if previous_sequence is None:
            if receipt.sequence_number != 1:
                raise ValueError(
                    "Polymarket CLOB connection does not start at sequence one"
                )
        elif receipt.sequence_number != previous_sequence + 1:
            raise ValueError("Polymarket CLOB receipt sequence is not contiguous")
        previous_lane_time = self._last_lane_monotonic.get(receipt.lane_id, -1)
        if receipt.received_monotonic_ns < previous_lane_time:
            raise ValueError("Polymarket CLOB lane receipt time regressed")
        if receipt.received_monotonic_ns < self._last_global_monotonic:
            raise ValueError("Polymarket CLOB merged receipt order regressed")
        if receipt.received_monotonic_ns < self._last_advanced_monotonic:
            raise ValueError("Polymarket CLOB receipt precedes the union watermark")
        self._last_sequence[key] = receipt.sequence_number
        self._last_lane_monotonic[receipt.lane_id] = receipt.received_monotonic_ns
        self._last_global_monotonic = receipt.received_monotonic_ns

    def advance(
        self,
        received_monotonic_ns: int,
    ) -> tuple[PolymarketUnionEvent, ...]:
        """Advance causal time when another stream supplies the watermark."""

        if self._finished:
            raise RuntimeError("Polymarket redundant union is already finished")
        selected = received_monotonic_ns
        if (
            type(selected) is not int
            or selected <= 0
            or selected < self._last_global_monotonic
            or selected < self._last_advanced_monotonic
        ):
            raise ValueError("Polymarket redundant union watermark regressed")
        self._last_advanced_monotonic = selected
        watermark = selected - self._pairing_window_ns
        self._expire_unmatched(watermark)
        return self._drain_resolved(watermark)

    def _resolve(
        self,
        pending: _PendingEvent,
        peer: _PendingEvent | None = None,
    ) -> None:
        receipts = tuple(
            sorted(
                (
                    (pending.receipt,)
                    if peer is None
                    else (pending.receipt, peer.receipt)
                ),
                key=lambda value: (
                    value.received_monotonic_ns,
                    value.lane_id,
                    value.connection_id,
                    value.sequence_number,
                ),
            )
        )
        selected = receipts[0]
        if peer is not None:
            self._shared_event_count += 1
            self._receipt_differences_ms.append(
                abs(
                    pending.receipt.received_monotonic_ns
                    - peer.receipt.received_monotonic_ns
                )
                / 1_000_000
            )
        heapq.heappush(
            self._resolved,
            _ResolvedEvent(
                ordering_key=(
                    selected.received_monotonic_ns,
                    selected.received_wall_ms,
                    pending.semantic_sha256,
                    selected.lane_id,
                ),
                semantic_sha256=pending.semantic_sha256,
                event_type=pending.event_type,
                event_json=pending.event_json,
                source_time_ms=pending.source_time_ms,
                receipts=receipts,
            ),
        )

    def _expire_unmatched(self, watermark_ns: int) -> None:
        empty_digests: list[str] = []
        for digest, lanes in self._pending.items():
            for lane in POLYMARKET_REDUNDANT_CLOB_LANES:
                queue = lanes[lane]
                while (
                    queue
                    and queue[0].receipt.received_monotonic_ns <= watermark_ns
                ):
                    self._resolve(queue.popleft())
                    self._pending_event_count -= 1
            if not lanes["clob-a"] and not lanes["clob-b"]:
                empty_digests.append(digest)
        for digest in empty_digests:
            del self._pending[digest]

    def _drain_resolved(self, watermark_ns: int | None) -> tuple[PolymarketUnionEvent, ...]:
        output: list[PolymarketUnionEvent] = []
        while self._resolved and (
            watermark_ns is None
            or self._resolved[0].ordering_key[0] <= watermark_ns
        ):
            resolved = heapq.heappop(self._resolved)
            self._union_event_count += 1
            self._emitted_occurrences[resolved.semantic_sha256] += 1
            occurrence = self._emitted_occurrences[resolved.semantic_sha256]
            self._event_type_counts[resolved.event_type] += 1
            selected = resolved.receipts[0]
            identity: dict[str, Any] = {
                "schema_version": POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION,
                "union_sequence_number": self._union_event_count,
                "semantic_sha256": resolved.semantic_sha256,
                "semantic_occurrence_index": occurrence,
                "event_type": resolved.event_type,
                "source_time_ms": resolved.source_time_ms,
                "selected_received_wall_ms": selected.received_wall_ms,
                "selected_received_monotonic_ns": (
                    selected.received_monotonic_ns
                ),
                "selected_lane_id": selected.lane_id,
                "lane_receipts": [
                    {
                        "lane_id": receipt.lane_id,
                        "connection_id": receipt.connection_id,
                        "sequence_number": receipt.sequence_number,
                        "received_wall_ms": receipt.received_wall_ms,
                        "received_monotonic_ns": receipt.received_monotonic_ns,
                    }
                    for receipt in resolved.receipts
                ],
            }
            output.append(
                PolymarketUnionEvent(
                    union_sequence_number=self._union_event_count,
                    semantic_sha256=resolved.semantic_sha256,
                    semantic_occurrence_index=occurrence,
                    event_type=resolved.event_type,
                    event_json=resolved.event_json,
                    source_time_ms=resolved.source_time_ms,
                    selected_received_wall_ms=selected.received_wall_ms,
                    selected_received_monotonic_ns=(
                        selected.received_monotonic_ns
                    ),
                    selected_lane_id=selected.lane_id,
                    lane_receipts=resolved.receipts,
                    event_sha256=_canonical_sha256(identity),
                )
            )
        return tuple(output)

    def add(
        self,
        receipt: PolymarketClobLaneReceipt,
    ) -> tuple[PolymarketUnionEvent, ...]:
        """Validate one receipt and return causally ready union events."""

        if self._finished:
            raise RuntimeError("Polymarket redundant union is already finished")
        selected = receipt.validated()
        self._validate_receipt_order(selected)
        watermark = selected.received_monotonic_ns - self._pairing_window_ns
        self._expire_unmatched(watermark)
        if selected.raw_text == "PONG":
            return self._drain_resolved(watermark)
        try:
            decoded = json.loads(
                selected.raw_text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Polymarket CLOB frame is not strict JSON") from exc
        items = decoded if isinstance(decoded, list) else [decoded]
        if not items:
            raise ValueError("Polymarket CLOB frame contains no events")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("Polymarket CLOB event is not an object")
            event_json = _canonical_json(item)
            semantic_sha256 = hashlib.sha256(
                event_json.encode("ascii")
            ).hexdigest()
            pending = _PendingEvent(
                semantic_sha256=semantic_sha256,
                event_type=str(item.get("event_type") or "unknown"),
                event_json=event_json,
                source_time_ms=_source_time_ms(item),
                receipt=selected,
            )
            self._lane_event_counts[selected.lane_id] += 1
            peer_lane = (
                "clob-b" if selected.lane_id == "clob-a" else "clob-a"
            )
            peer_queue = self._pending[semantic_sha256][peer_lane]
            if peer_queue:
                peer = peer_queue.popleft()
                self._pending_event_count -= 1
                self._resolve(pending, peer)
            else:
                self._pending[semantic_sha256][selected.lane_id].append(pending)
                self._pending_event_count += 1
                self._maximum_pending_events_observed = max(
                    self._maximum_pending_events_observed,
                    self._pending_event_count,
                )
                if self._pending_event_count > self.maximum_pending_events:
                    raise MemoryError(
                        "Polymarket redundant union pending-event bound exceeded"
                    )
        return self._drain_resolved(watermark)

    def finish(
        self,
    ) -> tuple[tuple[PolymarketUnionEvent, ...], PolymarketRedundantUnionAudit]:
        """Flush unmatched events and return terminal events plus an exact audit."""

        if self._finished:
            raise RuntimeError("Polymarket redundant union is already finished")
        self._finished = True
        self._expire_unmatched(_MAX_UNSIGNED_64)
        events = self._drain_resolved(None)
        union_count = self._union_event_count
        coverage = {
            lane: (
                0.0
                if union_count == 0
                else self._lane_event_counts[lane] / union_count
            )
            for lane in POLYMARKET_REDUNDANT_CLOB_LANES
        }
        body: dict[str, object] = {
            "schema_version": POLYMARKET_REDUNDANT_UNION_AUDIT_SCHEMA_VERSION,
            "pairing_window_ms": self.pairing_window_ms,
            "union_event_count": union_count,
            "shared_event_count": self._shared_event_count,
            "single_lane_event_count": union_count - self._shared_event_count,
            "lane_event_counts": {
                lane: self._lane_event_counts[lane]
                for lane in POLYMARKET_REDUNDANT_CLOB_LANES
            },
            "lane_coverage_fraction": coverage,
            "shared_fraction": (
                0.0 if union_count == 0 else self._shared_event_count / union_count
            ),
            "event_type_counts": dict(sorted(self._event_type_counts.items())),
            "receipt_difference_ms": {
                "median": _quantile(self._receipt_differences_ms, 0.5),
                "p95": _quantile(self._receipt_differences_ms, 0.95),
                "maximum": (
                    None
                    if not self._receipt_differences_ms
                    else max(self._receipt_differences_ms)
                ),
            },
            "maximum_pending_events_observed": (
                self._maximum_pending_events_observed
            ),
            "terminal_pending_event_count": self._pending_event_count,
        }
        audit = PolymarketRedundantUnionAudit(
            **body,
            audit_sha256=_canonical_sha256(body),
        )
        return events, audit


__all__ = [
    "POLYMARKET_REDUNDANT_CLOB_LANES",
    "POLYMARKET_REDUNDANT_UNION_AUDIT_SCHEMA_VERSION",
    "POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION",
    "PolymarketClobLaneReceipt",
    "PolymarketRedundantUnionAudit",
    "PolymarketRedundantUnionBuilder",
    "PolymarketUnionEvent",
]

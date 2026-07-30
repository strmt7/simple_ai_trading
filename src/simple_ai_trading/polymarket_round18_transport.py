"""Preregistered redundant CLOB transport qualification for Polymarket."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from websockets.asyncio.client import connect

from .polymarket import PolymarketFiveMinuteMarket, PolymarketPublicClient


POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256 = (
    "ed2420fec365944784cb7b8c5b8e012f0fbdf4e329e376fb61e3880467599b4a"
)
POLYMARKET_ROUND18_TRANSPORT_RESULT_SCHEMA_VERSION = (
    "polymarket-round18-redundant-clob-transport-result-v1"
)
_CLOB_MARKET_WEBSOCKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_LANE_IDS = ("clob-a", "clob-b")
_SHA256 = frozenset("0123456789abcdef")


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


def _wall_ms() -> int:
    return time.time_ns() // 1_000_000


def load_round18_transport_contract(path: str | Path) -> dict[str, object]:
    """Load the exact preregistered qualification contract."""

    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= 1024 * 1024
    ):
        raise ValueError("Round 18 transport contract is unavailable")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 18 transport contract is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 18 transport contract is not an object")
    body = dict(value)
    claimed = str(body.pop("contract_sha256", "")).strip().lower()
    if (
        claimed != POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256
        or claimed != _canonical_sha256(body)
        or value.get("status") != "preregistered_before_qualification_feed_access"
        or value.get("authority")
        != {
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
        }
    ):
        raise ValueError("Round 18 transport contract differs")
    return dict(value)


def _contract_section(
    contract: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = contract.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 18 contract {name} is unavailable")
    return value


@dataclass(slots=True)
class RedundantClobLaneEvidence:
    """Bounded in-memory evidence for one independent WebSocket lane."""

    lane_id: str
    maximum_unique_event_digests: int
    event_counts: Counter[str] = field(default_factory=Counter)
    first_receipt_monotonic_ns: dict[str, int] = field(default_factory=dict)
    event_type_counts: Counter[str] = field(default_factory=Counter)
    frame_count: int = 0
    market_event_count: int = 0
    pong_count: int = 0
    json_parse_error_count: int = 0
    connection_count: int = 0
    transport_gap_count: int = 0
    connected_seconds: float = 0.0
    last_market_event_monotonic: float = 0.0
    memory_bound_exceeded: bool = False
    transport_gap_reasons: Counter[str] = field(default_factory=Counter)
    _connected_since: float | None = None

    def mark_connected(self, now: float) -> None:
        if self._connected_since is not None:
            raise RuntimeError("Round 18 CLOB lane is already connected")
        self.connection_count += 1
        self._connected_since = now

    def mark_disconnected(
        self,
        now: float,
        *,
        reason: str | None = None,
    ) -> None:
        if self._connected_since is not None:
            self.connected_seconds += max(0.0, now - self._connected_since)
            self._connected_since = None
        if reason:
            self.transport_gap_count += 1
            self.transport_gap_reasons[reason] += 1

    def finalize(self, now: float) -> None:
        self.mark_disconnected(now)

    def connected(self) -> bool:
        return self._connected_since is not None

    def record_frame(self, raw: str | bytes, now: float) -> None:
        self.frame_count += 1
        try:
            text = (
                raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
            )
        except UnicodeError:
            self.json_parse_error_count += 1
            return
        if text == "PONG":
            self.pong_count += 1
            return
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            self.json_parse_error_count += 1
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, Mapping):
                self.json_parse_error_count += 1
                continue
            try:
                digest = _canonical_sha256(item)
            except (TypeError, ValueError):
                self.json_parse_error_count += 1
                continue
            if (
                digest not in self.event_counts
                and len(self.event_counts) >= self.maximum_unique_event_digests
            ):
                self.memory_bound_exceeded = True
                return
            self.event_counts[digest] += 1
            self.first_receipt_monotonic_ns.setdefault(
                digest,
                time.monotonic_ns(),
            )
            self.event_type_counts[str(item.get("event_type") or "unknown")] += 1
            self.market_event_count += 1
            self.last_market_event_monotonic = now

    def summary(self, duration_seconds: float) -> dict[str, object]:
        duration = max(duration_seconds, 1e-9)
        return {
            "lane_id": self.lane_id,
            "connection_count": self.connection_count,
            "reconnect_count": max(0, self.connection_count - 1),
            "transport_gap_count": self.transport_gap_count,
            "transport_gap_reasons": dict(sorted(self.transport_gap_reasons.items())),
            "connected_seconds": self.connected_seconds,
            "connected_fraction": self.connected_seconds / duration,
            "frame_count": self.frame_count,
            "market_event_count": self.market_event_count,
            "unique_event_digest_count": len(self.event_counts),
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "pong_count": self.pong_count,
            "json_parse_error_count": self.json_parse_error_count,
            "memory_bound_exceeded": self.memory_bound_exceeded,
        }


@dataclass(slots=True)
class _MonitorEvidence:
    warmup_seconds: float
    monitor_interval_seconds: float
    simultaneous_unhealthy_seconds: float = 0.0
    observed_seconds_after_warmup: float = 0.0
    sample_count_after_warmup: int = 0

    def observe(
        self,
        lanes: Sequence[RedundantClobLaneEvidence],
        *,
        now: float,
        started: float,
        elapsed: float,
        fresh_event_seconds: float,
    ) -> None:
        if elapsed < self.warmup_seconds:
            return
        healthy = tuple(
            lane.connected()
            and lane.last_market_event_monotonic > 0.0
            and now - lane.last_market_event_monotonic <= fresh_event_seconds
            for lane in lanes
        )
        self.sample_count_after_warmup += 1
        self.observed_seconds_after_warmup = max(
            0.0,
            now - started - self.warmup_seconds,
        )
        if not any(healthy):
            self.simultaneous_unhealthy_seconds += self.monitor_interval_seconds


def _quantile(values: Sequence[float], probability: float) -> float | None:
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


def _market_identity(
    market: PolymarketFiveMinuteMarket,
) -> dict[str, object]:
    payload = {
        "asset": market.asset,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "event_start_ms": market.event_start_ms,
        "end_ms": market.end_ms,
        "token_ids": list(market.token_ids),
    }
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _counted_overlap(
    lanes: Sequence[RedundantClobLaneEvidence],
) -> dict[str, object]:
    if len(lanes) != 2:
        raise ValueError("Round 18 requires exactly two CLOB lanes")
    first, second = lanes
    union = first.event_counts | second.event_counts
    intersection = first.event_counts & second.event_counts
    union_count = sum(union.values())
    shared_count = sum(intersection.values())
    first_count = sum(first.event_counts.values())
    second_count = sum(second.event_counts.values())
    shared_digests = set(first.first_receipt_monotonic_ns).intersection(
        second.first_receipt_monotonic_ns
    )
    receipt_differences_ms = [
        abs(
            first.first_receipt_monotonic_ns[digest]
            - second.first_receipt_monotonic_ns[digest]
        )
        / 1_000_000
        for digest in shared_digests
    ]
    return {
        "union_counted_event_count": union_count,
        "shared_counted_event_count": shared_count,
        "counted_overlap_fraction": (
            0.0 if union_count == 0 else shared_count / union_count
        ),
        "lane_event_coverage_fraction": {
            first.lane_id: 0.0 if union_count == 0 else first_count / union_count,
            second.lane_id: 0.0 if union_count == 0 else second_count / union_count,
        },
        "union_unique_event_digest_count": len(union),
        "shared_unique_event_digest_count": len(shared_digests),
        "receipt_difference_ms": {
            "median": _quantile(receipt_differences_ms, 0.5),
            "p95": _quantile(receipt_differences_ms, 0.95),
            "maximum": (
                None if not receipt_differences_ms else max(receipt_differences_ms)
            ),
        },
    }


def evaluate_round18_transport_qualification(
    contract: Mapping[str, object],
    *,
    market_identities: Sequence[Mapping[str, object]],
    lanes: Sequence[RedundantClobLaneEvidence],
    started_at_ms: int,
    ended_at_ms: int,
    monitor: _MonitorEvidence,
    internal_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Build the self-hashed terminal qualification result."""

    selected = dict(contract)
    claimed_contract = str(selected.get("contract_sha256") or "").lower()
    if claimed_contract != POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256:
        raise ValueError("Round 18 result contract differs")
    qualification = _contract_section(selected, "qualification")
    duration = max(0.0, (int(ended_at_ms) - int(started_at_ms)) / 1000)
    lane_summaries = [lane.summary(duration) for lane in lanes]
    overlap = _counted_overlap(lanes)
    coverage = overlap["lane_event_coverage_fraction"]
    if not isinstance(coverage, Mapping):
        raise AssertionError("Round 18 lane coverage is unavailable")
    condition_min, condition_max = qualification["condition_count_range"]  # type: ignore[misc]
    token_min, token_max = qualification["token_count_range"]  # type: ignore[misc]
    condition_count = len(market_identities)
    token_ids = {
        str(token_id)
        for market in market_identities
        for token_id in market["token_ids"]  # type: ignore[index]
    }
    minimum_events = int(qualification["minimum_market_events_per_lane"])
    minimum_connected = float(qualification["minimum_connected_fraction_per_lane"])
    minimum_coverage = float(qualification["minimum_event_coverage_fraction_per_lane"])
    gates = {
        "duration_complete": (
            duration >= float(qualification["duration_seconds"]) * 0.99
        ),
        "condition_count_in_range": (
            int(condition_min) <= condition_count <= int(condition_max)
        ),
        "token_count_in_range": (int(token_min) <= len(token_ids) <= int(token_max)),
        "minimum_events_each_lane": all(
            int(summary["market_event_count"]) >= minimum_events
            for summary in lane_summaries
        ),
        "minimum_connected_fraction_each_lane": all(
            float(summary["connected_fraction"]) >= minimum_connected
            for summary in lane_summaries
        ),
        "minimum_event_coverage_each_lane": all(
            float(coverage[lane.lane_id]) >= minimum_coverage for lane in lanes
        ),
        "minimum_counted_overlap": (
            float(overlap["counted_overlap_fraction"])
            >= float(qualification["minimum_counted_overlap_fraction"])
        ),
        "simultaneous_unhealthy_within_limit": (
            monitor.simultaneous_unhealthy_seconds
            <= float(qualification["maximum_simultaneous_unhealthy_seconds"])
        ),
        "zero_json_parse_errors": (
            sum(int(summary["json_parse_error_count"]) for summary in lane_summaries)
            <= int(qualification["maximum_json_parse_errors"])
        ),
        "memory_bound_preserved": not any(
            bool(summary["memory_bound_exceeded"]) for summary in lane_summaries
        ),
        "zero_internal_errors": not internal_errors,
    }
    payload: dict[str, Any] = {
        "schema_version": POLYMARKET_ROUND18_TRANSPORT_RESULT_SCHEMA_VERSION,
        "contract_sha256": claimed_contract,
        "started_at_ms": int(started_at_ms),
        "ended_at_ms": int(ended_at_ms),
        "duration_seconds": duration,
        "market_identities": [dict(value) for value in market_identities],
        "market_identity_manifest_sha256": _canonical_sha256(
            [dict(value) for value in market_identities]
        ),
        "subscription_asset_ids_sha256": _canonical_sha256(sorted(token_ids)),
        "lanes": lane_summaries,
        "redundancy": overlap,
        "monitor": {
            "warmup_seconds": monitor.warmup_seconds,
            "monitor_interval_seconds": monitor.monitor_interval_seconds,
            "sample_count_after_warmup": monitor.sample_count_after_warmup,
            "observed_seconds_after_warmup": monitor.observed_seconds_after_warmup,
            "simultaneous_unhealthy_seconds": (monitor.simultaneous_unhealthy_seconds),
        },
        "internal_errors": list(internal_errors),
        "gates": gates,
        "qualified": all(gates.values()),
        "transport_disconnect_reclassified_as_complete": False,
        "venue_source_completeness_proven": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return validate_round18_transport_result(payload)


def validate_round18_transport_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate one terminal transport result and its non-authority boundary."""

    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).strip().lower()
    gates = payload.get("gates")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "started_at_ms",
        "ended_at_ms",
        "duration_seconds",
        "market_identities",
        "market_identity_manifest_sha256",
        "subscription_asset_ids_sha256",
        "lanes",
        "redundancy",
        "monitor",
        "internal_errors",
        "gates",
        "qualified",
        "transport_disconnect_reclassified_as_complete",
        "venue_source_completeness_proven",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
        "binance_credentials_used",
        "binance_execution_connected",
    }
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND18_TRANSPORT_RESULT_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256
        or not isinstance(gates, Mapping)
        or not gates
        or any(not isinstance(gate, bool) for gate in gates.values())
        or payload.get("qualified") is not all(gates.values())
        or any(
            payload.get(name) is not False
            for name in (
                "transport_disconnect_reclassified_as_complete",
                "venue_source_completeness_proven",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
                "binance_credentials_used",
                "binance_execution_connected",
            )
        )
        or not isinstance(payload.get("lanes"), list)
        or len(payload["lanes"]) != len(_LANE_IDS)  # type: ignore[arg-type]
        or not isinstance(payload.get("market_identities"), list)
        or not isinstance(payload.get("internal_errors"), list)
        or not isinstance(payload.get("started_at_ms"), int)
        or not isinstance(payload.get("ended_at_ms"), int)
        or int(payload["ended_at_ms"]) <= int(payload["started_at_ms"])
    ):
        raise ValueError("Round 18 terminal transport result differs")
    return {**payload, "result_sha256": claimed}


async def _lane_worker(
    lane: RedundantClobLaneEvidence,
    *,
    token_ids: Sequence[str],
    stop: asyncio.Event,
    ends_at: float,
    transport: Mapping[str, object],
) -> None:
    backoffs = tuple(float(value) for value in transport["reconnect_backoff_seconds"])  # type: ignore[arg-type]
    backoff_index = 0
    while not stop.is_set() and asyncio.get_running_loop().time() < ends_at:
        connected = False
        try:
            async with connect(
                str(transport["endpoint"]),
                open_timeout=10,
                close_timeout=3,
                ping_interval=transport["protocol_ping_interval"],
                ping_timeout=None,
                max_size=int(transport["maximum_message_bytes"]),
                max_queue=int(transport["maximum_websocket_queue_frames"]),
                compression=transport["compression"],
            ) as websocket:
                now = asyncio.get_running_loop().time()
                lane.mark_connected(now)
                connected = True
                await websocket.send(
                    _canonical_json(
                        {
                            "assets_ids": list(token_ids),
                            "type": "market",
                            "custom_feature_enabled": bool(
                                transport["custom_feature_enabled"]
                            ),
                        }
                    )
                )
                next_heartbeat = now + float(transport["heartbeat_interval_seconds"])
                backoff_index = 0
                while (
                    not stop.is_set()
                    and asyncio.get_running_loop().time() < ends_at
                    and not lane.memory_bound_exceeded
                ):
                    now = asyncio.get_running_loop().time()
                    if now >= next_heartbeat:
                        await websocket.send(str(transport["text_heartbeat"]))
                        next_heartbeat = now + float(
                            transport["heartbeat_interval_seconds"]
                        )
                    timeout = max(
                        0.05,
                        min(0.5, ends_at - now, next_heartbeat - now),
                    )
                    try:
                        raw = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=timeout,
                        )
                    except TimeoutError:
                        continue
                    lane.record_frame(raw, asyncio.get_running_loop().time())
                if lane.memory_bound_exceeded:
                    stop.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            now = asyncio.get_running_loop().time()
            reason = f"{exc.__class__.__name__}:{exc}"
            lane.mark_disconnected(now, reason=reason)
            connected = False
            if stop.is_set() or now >= ends_at:
                break
            delay = backoffs[min(backoff_index, len(backoffs) - 1)]
            backoff_index += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass
        finally:
            if connected:
                lane.mark_disconnected(asyncio.get_running_loop().time())


async def _run_probe(
    contract: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    *,
    progress: Callable[[Mapping[str, object]], None] | None,
) -> dict[str, object]:
    transport = _contract_section(contract, "transport")
    qualification = _contract_section(contract, "qualification")
    duration = float(qualification["duration_seconds"])
    warmup = float(qualification["warmup_seconds"])
    monitor_interval = float(qualification["monitor_interval_milliseconds"]) / 1000
    progress_interval = float(qualification["progress_interval_seconds"])
    fresh_event_seconds = float(qualification["fresh_market_event_seconds"])
    maximum_digests = int(qualification["maximum_unique_event_digests_per_lane"])
    token_ids = tuple(
        sorted({token_id for market in markets for token_id in market.token_ids})
    )
    lanes = tuple(
        RedundantClobLaneEvidence(lane_id, maximum_digests) for lane_id in _LANE_IDS
    )
    monitor = _MonitorEvidence(
        warmup_seconds=warmup,
        monitor_interval_seconds=monitor_interval,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    started_monotonic = loop.time()
    ends_at = started_monotonic + duration
    started_at_ms = _wall_ms()
    tasks = tuple(
        asyncio.create_task(
            _lane_worker(
                lane,
                token_ids=token_ids,
                stop=stop,
                ends_at=ends_at,
                transport=transport,
            )
        )
        for lane in lanes
    )
    next_progress = started_monotonic
    try:
        while not stop.is_set():
            now = loop.time()
            if now >= ends_at:
                break
            elapsed = now - started_monotonic
            monitor.observe(
                lanes,
                now=now,
                started=started_monotonic,
                elapsed=elapsed,
                fresh_event_seconds=fresh_event_seconds,
            )
            if progress is not None and now >= next_progress:
                progress(
                    {
                        "phase": "transport_probe",
                        "elapsed_seconds": round(elapsed, 3),
                        "duration_seconds": duration,
                        "lane_events": {
                            lane.lane_id: lane.market_event_count for lane in lanes
                        },
                        "lane_gaps": {
                            lane.lane_id: lane.transport_gap_count for lane in lanes
                        },
                        "simultaneous_unhealthy_seconds": round(
                            monitor.simultaneous_unhealthy_seconds,
                            3,
                        ),
                    }
                )
                next_progress = now + progress_interval
            await asyncio.sleep(min(monitor_interval, max(0.0, ends_at - loop.time())))
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        ended_monotonic = loop.time()
        for lane in lanes:
            lane.finalize(ended_monotonic)
    internal_errors = [
        f"{exc.__class__.__name__}:{exc}"
        for task in tasks
        if isinstance((exc := task.exception()), Exception)
    ]
    ended_at_ms = started_at_ms + round((ended_monotonic - started_monotonic) * 1000)
    return evaluate_round18_transport_qualification(
        contract,
        market_identities=[_market_identity(market) for market in markets],
        lanes=lanes,
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        monitor=monitor,
        internal_errors=internal_errors,
    )


def run_round18_transport_qualification(
    contract_path: str | Path,
    *,
    client: PolymarketPublicClient | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Discover BTC markets, then run the bounded dual-lane public probe."""

    contract = load_round18_transport_contract(contract_path)
    public = client or PolymarketPublicClient(timeout_seconds=10)
    now_ms = _wall_ms()
    markets = public.discover_five_minute_markets(
        now_ms=now_ms,
        include_next=True,
        require_all_assets=True,
        assets=("BTC",),
    )
    selected = tuple(
        market for market in markets if market.asset == "BTC" and market.end_ms > now_ms
    )
    qualification = _contract_section(contract, "qualification")
    condition_min, condition_max = qualification["condition_count_range"]  # type: ignore[misc]
    token_min, token_max = qualification["token_count_range"]  # type: ignore[misc]
    tokens = {token for market in selected for token in market.token_ids}
    if not int(condition_min) <= len(selected) <= int(condition_max) or not int(
        token_min
    ) <= len(tokens) <= int(token_max):
        raise ValueError("Round 18 discovered market scope differs")
    if progress is not None:
        progress(
            {
                "phase": "market_scope_frozen",
                "condition_count": len(selected),
                "token_count": len(tokens),
                "market_identity_manifest_sha256": _canonical_sha256(
                    [_market_identity(market) for market in selected]
                ),
            }
        )
    return asyncio.run(_run_probe(contract, selected, progress=progress))


__all__ = [
    "POLYMARKET_ROUND18_TRANSPORT_CONTRACT_SHA256",
    "POLYMARKET_ROUND18_TRANSPORT_RESULT_SCHEMA_VERSION",
    "RedundantClobLaneEvidence",
    "evaluate_round18_transport_qualification",
    "load_round18_transport_contract",
    "run_round18_transport_qualification",
    "validate_round18_transport_result",
]

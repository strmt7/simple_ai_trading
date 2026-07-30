"""Rotating redundant CLOB transport qualification for Polymarket."""

from __future__ import annotations

import asyncio
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
from .polymarket_round18_transport import RedundantClobLaneEvidence


POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256 = (
    "f412e449ba6716459444f07b9d18b98195e106d3322511b21ee78c7fe808ed5b"
)
POLYMARKET_ROUND19_TRANSPORT_RESULT_SCHEMA_VERSION = (
    "polymarket-round19-rotating-redundant-clob-result-v1"
)
_LANE_IDS = ("clob-a", "clob-b")


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


def load_round19_transport_contract(path: str | Path) -> dict[str, object]:
    """Load the exact preregistered rotating qualification contract."""

    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= 1024 * 1024
    ):
        raise ValueError("Round 19 transport contract is unavailable")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 19 transport contract is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 19 transport contract is not an object")
    body = dict(value)
    claimed = str(body.pop("contract_sha256", "")).strip().lower()
    authority = value.get("authority")
    if (
        claimed != POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256
        or claimed != _canonical_sha256(body)
        or value.get("status") != "preregistered_before_qualification_feed_access"
        or authority
        != {
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
        }
    ):
        raise ValueError("Round 19 transport contract differs")
    return dict(value)


def _section(
    contract: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    value = contract.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 19 contract {name} is unavailable")
    return value


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


@dataclass(slots=True)
class _RotatingRegistry:
    markets: dict[str, PolymarketFiveMinuteMarket] = field(default_factory=dict)
    identities: dict[str, dict[str, object]] = field(default_factory=dict)
    desired_token_ids: tuple[str, ...] = ()
    revision: int = 0
    discovery_count: int = 0
    discovery_error_count: int = 0
    discovery_error_reasons: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        markets: Sequence[PolymarketFiveMinuteMarket],
        *,
        now_ms: int,
    ) -> bool:
        selected = tuple(
            sorted(
                (
                    market
                    for market in markets
                    if market.asset == "BTC" and market.end_ms > now_ms
                ),
                key=lambda market: market.event_start_ms,
            )
        )
        if not selected:
            raise ValueError("Round 19 discovery returned no active BTC market")
        for market in selected:
            identity = _market_identity(market)
            prior = self.identities.get(market.condition_id)
            if prior is not None and prior != identity:
                raise ValueError("Round 19 market identity drifted")
            self.identities[market.condition_id] = identity
            self.markets[market.condition_id] = market
        desired = tuple(
            sorted({token_id for market in selected for token_id in market.token_ids})
        )
        self.discovery_count += 1
        if desired == self.desired_token_ids:
            return False
        self.desired_token_ids = desired
        self.revision += 1
        return True

    def record_error(self, exc: Exception) -> None:
        reason = f"{exc.__class__.__name__}:{exc}"
        self.discovery_error_count += 1
        self.discovery_error_reasons[reason] = (
            self.discovery_error_reasons.get(reason, 0) + 1
        )


@dataclass(slots=True)
class _RotatingLane:
    evidence: RedundantClobLaneEvidence
    applied_revision: int = 0
    subscribed_token_ids: tuple[str, ...] = ()
    subscription_update_count: int = 0

    def summary(self, duration_seconds: float) -> dict[str, object]:
        return {
            **self.evidence.summary(duration_seconds),
            "applied_revision": self.applied_revision,
            "subscription_update_count": self.subscription_update_count,
            "final_subscribed_token_count": len(self.subscribed_token_ids),
            "final_subscription_sha256": _canonical_sha256(
                list(self.subscribed_token_ids)
            ),
        }


@dataclass(slots=True)
class _Monitor:
    warmup_seconds: float
    interval_seconds: float
    simultaneous_unhealthy_seconds: float = 0.0
    observed_seconds_after_warmup: float = 0.0
    sample_count_after_warmup: int = 0

    def observe(
        self,
        lanes: Sequence[_RotatingLane],
        *,
        now: float,
        started: float,
        fresh_event_seconds: float,
    ) -> None:
        elapsed = now - started
        if elapsed < self.warmup_seconds:
            return
        healthy = tuple(
            lane.evidence.connected()
            and lane.evidence.last_market_event_monotonic > 0
            and now - lane.evidence.last_market_event_monotonic <= fresh_event_seconds
            for lane in lanes
        )
        self.sample_count_after_warmup += 1
        self.observed_seconds_after_warmup = max(
            0.0,
            elapsed - self.warmup_seconds,
        )
        if not any(healthy):
            self.simultaneous_unhealthy_seconds += self.interval_seconds


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


def _overlap(lanes: Sequence[_RotatingLane]) -> dict[str, object]:
    if len(lanes) != 2:
        raise ValueError("Round 19 requires exactly two CLOB lanes")
    first, second = (lane.evidence for lane in lanes)
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
            second.lane_id: (0.0 if union_count == 0 else second_count / union_count),
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


def evaluate_round19_transport_qualification(
    contract: Mapping[str, object],
    *,
    registry: _RotatingRegistry,
    lanes: Sequence[_RotatingLane],
    started_at_ms: int,
    ended_at_ms: int,
    monitor: _Monitor,
    internal_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Build and validate the terminal rotating qualification result."""

    claimed_contract = str(contract.get("contract_sha256") or "").lower()
    if claimed_contract != POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256:
        raise ValueError("Round 19 result contract differs")
    rotation = _section(contract, "rotation")
    qualification = _section(contract, "qualification")
    duration = max(0.0, (int(ended_at_ms) - int(started_at_ms)) / 1000)
    lane_summaries = [lane.summary(duration) for lane in lanes]
    overlap = _overlap(lanes)
    coverage = overlap["lane_event_coverage_fraction"]
    if not isinstance(coverage, Mapping):
        raise AssertionError("Round 19 lane coverage is unavailable")
    condition_min, condition_max = rotation["condition_count_range"]  # type: ignore[misc]
    token_min, token_max = rotation["token_count_range"]  # type: ignore[misc]
    identities = sorted(
        registry.identities.values(),
        key=lambda value: (int(value["event_start_ms"]), str(value["condition_id"])),
    )
    all_token_ids = {
        str(token_id)
        for market in identities
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
            int(condition_min) <= len(identities) <= int(condition_max)
        ),
        "token_count_in_range": (
            int(token_min) <= len(all_token_ids) <= int(token_max)
        ),
        "zero_discovery_errors": (
            registry.discovery_error_count <= int(rotation["maximum_discovery_errors"])
        ),
        "both_lanes_applied_final_revision": all(
            lane.applied_revision == registry.revision for lane in lanes
        ),
        "both_lanes_hold_final_desired_subscription": all(
            lane.subscribed_token_ids == registry.desired_token_ids for lane in lanes
        ),
        "minimum_events_each_lane": all(
            int(summary["market_event_count"]) >= minimum_events
            for summary in lane_summaries
        ),
        "minimum_connected_fraction_each_lane": all(
            float(summary["connected_fraction"]) >= minimum_connected
            for summary in lane_summaries
        ),
        "minimum_event_coverage_each_lane": all(
            float(coverage[lane.evidence.lane_id]) >= minimum_coverage for lane in lanes
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
        "schema_version": POLYMARKET_ROUND19_TRANSPORT_RESULT_SCHEMA_VERSION,
        "contract_sha256": claimed_contract,
        "parent_round18_result_sha256": contract["parent"][  # type: ignore[index]
            "round18_result_sha256"
        ],
        "started_at_ms": int(started_at_ms),
        "ended_at_ms": int(ended_at_ms),
        "duration_seconds": duration,
        "market_identities": identities,
        "market_identity_manifest_sha256": _canonical_sha256(identities),
        "all_discovered_asset_ids_sha256": _canonical_sha256(sorted(all_token_ids)),
        "final_desired_asset_ids_sha256": _canonical_sha256(
            list(registry.desired_token_ids)
        ),
        "rotation": {
            "discovery_count": registry.discovery_count,
            "discovery_error_count": registry.discovery_error_count,
            "discovery_error_reasons": dict(
                sorted(registry.discovery_error_reasons.items())
            ),
            "final_revision": registry.revision,
            "final_desired_token_count": len(registry.desired_token_ids),
        },
        "lanes": lane_summaries,
        "redundancy": overlap,
        "monitor": {
            "warmup_seconds": monitor.warmup_seconds,
            "monitor_interval_seconds": monitor.interval_seconds,
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
    return validate_round19_transport_result(payload)


def validate_round19_transport_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate one terminal rotating transport result."""

    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).strip().lower()
    gates = payload.get("gates")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "parent_round18_result_sha256",
        "started_at_ms",
        "ended_at_ms",
        "duration_seconds",
        "market_identities",
        "market_identity_manifest_sha256",
        "all_discovered_asset_ids_sha256",
        "final_desired_asset_ids_sha256",
        "rotation",
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
        != POLYMARKET_ROUND19_TRANSPORT_RESULT_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256
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
    ):
        raise ValueError("Round 19 terminal transport result differs")
    return {**payload, "result_sha256": claimed}


async def _apply_subscription(
    websocket: Any,
    lane: _RotatingLane,
    registry: _RotatingRegistry,
    *,
    custom_feature_enabled: bool,
) -> None:
    desired = set(registry.desired_token_ids)
    current = set(lane.subscribed_token_ids)
    additions = sorted(desired - current)
    removals = sorted(current - desired)
    if additions:
        await websocket.send(
            _canonical_json(
                {
                    "assets_ids": additions,
                    "operation": "subscribe",
                    "custom_feature_enabled": custom_feature_enabled,
                }
            )
        )
    if removals:
        await websocket.send(
            _canonical_json(
                {
                    "assets_ids": removals,
                    "operation": "unsubscribe",
                }
            )
        )
    lane.subscribed_token_ids = tuple(sorted(desired))
    lane.applied_revision = registry.revision
    lane.subscription_update_count += int(bool(additions or removals))


async def _lane_worker(
    lane: _RotatingLane,
    registry: _RotatingRegistry,
    *,
    stop: asyncio.Event,
    ends_at: float,
    transport: Mapping[str, object],
) -> None:
    backoffs = tuple(
        float(value)
        for value in transport["reconnect_backoff_seconds"]  # type: ignore[union-attr]
    )
    backoff_index = 0
    while not stop.is_set() and asyncio.get_running_loop().time() < ends_at:
        connected = False
        try:
            tokens = registry.desired_token_ids
            if not tokens:
                raise RuntimeError("Round 19 desired subscription is empty")
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
                lane.evidence.mark_connected(now)
                connected = True
                await websocket.send(
                    _canonical_json(
                        {
                            "assets_ids": list(tokens),
                            "type": "market",
                            "custom_feature_enabled": bool(
                                transport["custom_feature_enabled"]
                            ),
                        }
                    )
                )
                lane.subscribed_token_ids = tokens
                lane.applied_revision = registry.revision
                next_heartbeat = now + float(transport["heartbeat_interval_seconds"])
                backoff_index = 0
                while (
                    not stop.is_set()
                    and asyncio.get_running_loop().time() < ends_at
                    and not lane.evidence.memory_bound_exceeded
                ):
                    if lane.applied_revision != registry.revision:
                        await _apply_subscription(
                            websocket,
                            lane,
                            registry,
                            custom_feature_enabled=bool(
                                transport["custom_feature_enabled"]
                            ),
                        )
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
                    lane.evidence.record_frame(
                        raw,
                        asyncio.get_running_loop().time(),
                    )
                if lane.evidence.memory_bound_exceeded:
                    stop.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            now = asyncio.get_running_loop().time()
            lane.evidence.mark_disconnected(
                now,
                reason=f"{exc.__class__.__name__}:{exc}",
            )
            connected = False
            lane.subscribed_token_ids = ()
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
                lane.evidence.mark_disconnected(asyncio.get_running_loop().time())


async def _discovery_loop(
    client: PolymarketPublicClient,
    registry: _RotatingRegistry,
    *,
    stop: asyncio.Event,
    ends_at: float,
    interval_seconds: float,
) -> None:
    while not stop.is_set() and asyncio.get_running_loop().time() < ends_at:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            return
        except TimeoutError:
            pass
        if asyncio.get_running_loop().time() >= ends_at:
            return
        try:
            now_ms = _wall_ms()
            markets = await asyncio.to_thread(
                client.discover_five_minute_markets,
                now_ms=now_ms,
                include_next=True,
                require_all_assets=True,
                assets=("BTC",),
            )
            registry.update(markets, now_ms=now_ms)
        except Exception as exc:
            registry.record_error(exc)


async def _run_probe(
    contract: Mapping[str, object],
    client: PolymarketPublicClient,
    initial_markets: Sequence[PolymarketFiveMinuteMarket],
    *,
    progress: Callable[[Mapping[str, object]], None] | None,
) -> dict[str, object]:
    transport = _section(contract, "transport")
    rotation = _section(contract, "rotation")
    qualification = _section(contract, "qualification")
    duration = float(qualification["duration_seconds"])
    monitor_interval = float(qualification["monitor_interval_milliseconds"]) / 1000
    progress_interval = float(qualification["progress_interval_seconds"])
    fresh_event_seconds = float(qualification["fresh_market_event_seconds"])
    registry = _RotatingRegistry()
    now_ms = _wall_ms()
    registry.update(initial_markets, now_ms=now_ms)
    lanes = tuple(
        _RotatingLane(
            RedundantClobLaneEvidence(
                lane_id,
                int(qualification["maximum_unique_event_digests_per_lane"]),
            )
        )
        for lane_id in _LANE_IDS
    )
    monitor = _Monitor(
        warmup_seconds=float(qualification["warmup_seconds"]),
        interval_seconds=monitor_interval,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    started_monotonic = loop.time()
    ends_at = started_monotonic + duration
    started_at_ms = _wall_ms()
    tasks = (
        *(
            asyncio.create_task(
                _lane_worker(
                    lane,
                    registry,
                    stop=stop,
                    ends_at=ends_at,
                    transport=transport,
                )
            )
            for lane in lanes
        ),
        asyncio.create_task(
            _discovery_loop(
                client,
                registry,
                stop=stop,
                ends_at=ends_at,
                interval_seconds=float(rotation["gamma_discovery_interval_seconds"]),
            )
        ),
    )
    next_progress = started_monotonic
    try:
        while not stop.is_set():
            now = loop.time()
            if now >= ends_at:
                break
            monitor.observe(
                lanes,
                now=now,
                started=started_monotonic,
                fresh_event_seconds=fresh_event_seconds,
            )
            if progress is not None and now >= next_progress:
                progress(
                    {
                        "phase": "rotating_transport_probe",
                        "elapsed_seconds": round(now - started_monotonic, 3),
                        "duration_seconds": duration,
                        "discovered_condition_count": len(registry.identities),
                        "desired_revision": registry.revision,
                        "desired_token_count": len(registry.desired_token_ids),
                        "discovery_error_count": registry.discovery_error_count,
                        "lane_events": {
                            lane.evidence.lane_id: lane.evidence.market_event_count
                            for lane in lanes
                        },
                        "lane_gaps": {
                            lane.evidence.lane_id: lane.evidence.transport_gap_count
                            for lane in lanes
                        },
                        "lane_applied_revisions": {
                            lane.evidence.lane_id: lane.applied_revision
                            for lane in lanes
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
            lane.evidence.finalize(ended_monotonic)
    internal_errors = [
        f"{exc.__class__.__name__}:{exc}"
        for task in tasks
        if isinstance((exc := task.exception()), Exception)
    ]
    ended_at_ms = started_at_ms + round((ended_monotonic - started_monotonic) * 1000)
    return evaluate_round19_transport_qualification(
        contract,
        registry=registry,
        lanes=lanes,
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        monitor=monitor,
        internal_errors=internal_errors,
    )


def run_round19_transport_qualification(
    contract_path: str | Path,
    *,
    client: PolymarketPublicClient | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Discover initial BTC markets, then rotate a bounded dual-lane probe."""

    contract = load_round19_transport_contract(contract_path)
    public = client or PolymarketPublicClient(timeout_seconds=10)
    now_ms = _wall_ms()
    initial = public.discover_five_minute_markets(
        now_ms=now_ms,
        include_next=True,
        require_all_assets=True,
        assets=("BTC",),
    )
    selected = tuple(
        market for market in initial if market.asset == "BTC" and market.end_ms > now_ms
    )
    if not selected:
        raise ValueError("Round 19 initial market scope differs")
    if progress is not None:
        progress(
            {
                "phase": "initial_market_scope_frozen",
                "condition_count": len(selected),
                "token_count": len(
                    {token for market in selected for token in market.token_ids}
                ),
                "market_identity_manifest_sha256": _canonical_sha256(
                    [_market_identity(market) for market in selected]
                ),
            }
        )
    return asyncio.run(
        _run_probe(
            contract,
            public,
            selected,
            progress=progress,
        )
    )


__all__ = [
    "POLYMARKET_ROUND19_TRANSPORT_CONTRACT_SHA256",
    "POLYMARKET_ROUND19_TRANSPORT_RESULT_SCHEMA_VERSION",
    "evaluate_round19_transport_qualification",
    "load_round19_transport_contract",
    "run_round19_transport_qualification",
    "validate_round19_transport_result",
]

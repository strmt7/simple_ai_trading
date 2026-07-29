"""Isolated target-free BTC 15-minute Polymarket historical identities.

This module intentionally does not modify the hash-bound Round 14/15 screen
implementation. Binance remains an optional public, read-only feature source;
all market identity and eventual execution state belongs to Polymarket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .polymarket import PolymarketFeeSchedule
from .polymarket_historical_screen import (
    GAMMA_EVENTS_KEYSET_URL,
    HistoricalBtcMarket,
    HistoricalScreenContract,
    HistoricalScreenStore,
    HistoricalScreenTestGates,
    PolymarketHistoricalPublicClient,
    ProgressCallback,
    PublicPayload,
)


ROUND16_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round16-btc-15m-horizon-comparison-v1"
)
ROUND16_MARKET_SCHEMA_VERSION = "polymarket-round16-btc-15m-market-v1"
ROUND16_SERIES_ID = "10192"
ROUND16_DURATION_MS = 900_000
ROUND16_MARKETS_PER_DAY = 96
ROUND16_DECISION_OFFSETS_SECONDS = tuple(range(60, 841, 60))
ROUND16_RETURN_HORIZONS_SECONDS = (1, 5, 15, 30, 60, 120)
ROUND16_FLOW_WINDOWS_SECONDS = (1, 5, 15, 30, 60, 120)
ROUND16_RESOLUTION_SOURCE = "https://data.chain.link/streams/btc-usd"
_SLUG = re.compile(r"^btc-updown-15m-([0-9]{10})$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "id",
    "conditionId",
    "slug",
    "question",
    "eventStartTime",
    "endDate",
    "active",
    "closed",
    "enableOrderBook",
    "acceptingOrders",
    "outcomes",
    "clobTokenIds",
    "resolutionSource",
    "orderPriceMinTickSize",
    "orderMinSize",
    "feesEnabled",
    "feeSchedule",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 16 JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 16 JSON contains {value}")


def _decode_json(content: bytes, *, name: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc


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


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _decimal(
    value: object,
    *,
    name: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if (
        not parsed.is_finite()
        or (minimum is not None and parsed < minimum)
        or (maximum is not None and parsed > maximum)
    ):
        raise ValueError(f"{name} is outside its supported range")
    return parsed


def _json_array(value: object, *, name: str) -> list[object]:
    parsed = value
    if isinstance(value, str):
        parsed = _decode_json(value.encode("utf-8"), name=name)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be an array")
    return parsed


def _utc_ms(value: object, *, name: str) -> int:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset")
    return int(parsed.astimezone(UTC).timestamp() * 1_000)


def _day_bounds(day: str) -> tuple[int, int]:
    try:
        start = datetime.fromisoformat(day).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("Round 16 day is invalid") from exc
    if start.date().isoformat() != day:
        raise ValueError("Round 16 day is not canonical")
    return int(start.timestamp() * 1_000), int(
        (start + timedelta(days=1)).timestamp() * 1_000
    )


def _day_interval(value: object, *, name: str) -> tuple[str, ...]:
    payload = _mapping(value, name=name)
    first = str(payload.get("first_day") or "")
    last = str(payload.get("last_day") or "")
    first_ms, _ = _day_bounds(first)
    last_ms, _ = _day_bounds(last)
    if last_ms < first_ms:
        raise ValueError(f"{name} ends before it starts")
    count = (last_ms - first_ms) // 86_400_000 + 1
    if int(payload.get("day_count", 0)) != count:
        raise ValueError(f"{name} day count differs")
    first_day = datetime.fromtimestamp(first_ms / 1_000, tz=UTC)
    return tuple(
        (first_day + timedelta(days=index)).date().isoformat()
        for index in range(count)
    )


def _iso_seconds(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class Round16HistoricalContract:
    historical: HistoricalScreenContract
    duration_ms: int
    market_schema_version: str

    @property
    def contract_sha256(self) -> str:
        return self.historical.contract_sha256


def load_round16_historical_contract(
    path: str | Path,
) -> Round16HistoricalContract:
    selected = Path(path)
    if selected.is_symlink():
        raise ValueError("Round 16 contract cannot be a symlink")
    raw = _decode_json(selected.read_bytes(), name="Round 16 contract")
    payload = dict(_mapping(raw, name="Round 16 contract"))
    claimed = str(payload.pop("contract_sha256", "")).strip().lower()
    if _SHA256.fullmatch(claimed) is None or _canonical_sha256(payload) != claimed:
        raise ValueError("Round 16 contract SHA-256 differs")
    scope = _mapping(payload.get("scope"), name="Round 16 scope")
    source = _mapping(payload.get("source_contract"), name="Round 16 source")
    causal = _mapping(
        payload.get("causal_feature_contract"),
        name="Round 16 causal feature contract",
    )
    partition = _mapping(payload.get("partition"), name="Round 16 partition")
    gates = _mapping(payload.get("test_gates"), name="Round 16 test gates")
    authority = _mapping(scope.get("authority"), name="Round 16 authority")
    if (
        payload.get("schema_version") != ROUND16_CONTRACT_SCHEMA_VERSION
        or payload.get("status")
        != "preregistered_before_any_round16_identity_or_terminal_target_access"
        or scope.get("venue") != "polymarket"
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fifteenminute"
        or scope.get("development_only") is not True
        or dict(authority)
        != {
            "paper_trading": False,
            "live_trading": False,
            "promotion": False,
        }
        or payload.get("round16_identity_rows_consulted_before_freeze") is not False
        or payload.get("round16_terminal_targets_consulted_before_freeze") is not False
        or payload.get("round16_model_scores_consulted_before_freeze") is not False
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("Round 16 authority or preregistration boundary differs")
    eligible_days = _day_interval(
        source.get("eligible_interval"),
        name="Round 16 eligible interval",
    )
    train = _day_interval(
        partition.get("train_interval"),
        name="Round 16 train interval",
    )
    tune = _day_interval(
        partition.get("tune_interval"),
        name="Round 16 tune interval",
    )
    test = _day_interval(
        partition.get("test_interval"),
        name="Round 16 test interval",
    )
    decisions = tuple(
        int(value) for value in causal.get("decision_offsets_seconds", ())
    )
    returns = tuple(
        int(value) for value in causal.get("return_horizons_seconds", ())
    )
    windows = tuple(int(value) for value in causal.get("flow_windows_seconds", ()))
    inventory_sha = str(source.get("binance_inventory_sha256") or "").lower()
    slope = tuple(gates.get("calibration_slope_range", ()))
    if (
        tuple((*train, *tune, *test)) != eligible_days
        or len(eligible_days) != 154
        or eligible_days[0] != "2026-02-12"
        or eligible_days[-1] != "2026-07-15"
        or len(train) != 109
        or len(tune) != 30
        or len(test) != 15
        or str(source.get("polymarket_series_id")) != ROUND16_SERIES_ID
        or source.get("historical_polymarket_price_or_trade_features") is not False
        or source.get("binance_source")
        != "frozen_round15_btc_spot_perpetual_flow_1s"
        or source.get("binance_symbol") != "BTCUSDT"
        or source.get("shared_flow_database")
        != "data/polymarket-btc-flow-history-v1.duckdb"
        or source.get("duplicate_binance_storage_allowed") is not False
        or source.get("raw_binance_copy_allowed") is not False
        or _SHA256.fullmatch(inventory_sha) is None
        or int(source.get("required_flow_rows_per_day", 0)) != 86_400
        or int(source.get("required_market_count_per_day", 0))
        != ROUND16_MARKETS_PER_DAY
        or int(source.get("required_market_duration_ms", 0)) != ROUND16_DURATION_MS
        or source.get("required_resolution_source") != ROUND16_RESOLUTION_SOURCE
        or decisions != ROUND16_DECISION_OFFSETS_SECONDS
        or returns != ROUND16_RETURN_HORIZONS_SECONDS
        or windows != ROUND16_FLOW_WINDOWS_SECONDS
        or slope != ("0.85", "1.15")
        or int(gates.get("minimum_terminal_conditions", 0)) != 1_200
        or int(gates.get("minimum_outcomes_per_class", 0)) != 400
        or int(gates.get("minimum_decision_rows", 0)) != 10_000
        or int(gates.get("paired_condition_block_bootstrap_repetitions", 0))
        != 10_000
        or gates.get("expected_calibration_error_maximum") != "0.03"
    ):
        raise ValueError("Round 16 source, split, feature, or gate contract differs")
    roles = {
        **dict.fromkeys(train, "train"),
        **dict.fromkeys(tune, "tune"),
        **dict.fromkeys(test, "test"),
    }
    historical = HistoricalScreenContract(
        path=selected,
        contract_sha256=claimed,
        series_id=ROUND16_SERIES_ID,
        eligible_days=eligible_days,
        roles=roles,
        excluded_slugs=frozenset(),
        requested_page_limit=500,
        decision_offsets_seconds=decisions,
        return_horizons_seconds=returns,
        flow_windows_seconds=windows,
        source_inventory_sha256=inventory_sha,
        source_research_round=15,
        required_source_symbol_count=1,
        required_flow_rows_per_day=86_400,
        required_market_count_per_day=ROUND16_MARKETS_PER_DAY,
        test_gates=HistoricalScreenTestGates(
            minimum_terminal_conditions=1_200,
            minimum_outcomes_per_class=400,
            minimum_decision_rows=10_000,
            bootstrap_repetitions=10_000,
            calibration_slope_minimum=0.85,
            calibration_slope_maximum=1.15,
            expected_calibration_error_maximum=0.03,
        ),
    )
    return Round16HistoricalContract(
        historical=historical,
        duration_ms=ROUND16_DURATION_MS,
        market_schema_version=ROUND16_MARKET_SCHEMA_VERSION,
    )


def parse_round16_historical_btc_event(
    payload: Mapping[str, object],
    *,
    contract: Round16HistoricalContract,
    observed_at_ms: int,
) -> HistoricalBtcMarket:
    """Parse only target-free identity fields from one closed series event."""

    raw_event = dict(payload)
    event_id = str(raw_event.get("id") or "").strip()
    event_slug = str(raw_event.get("slug") or "").strip().lower()
    if (
        not event_id.isdigit()
        or len(event_id) > 20
        or _SLUG.fullmatch(event_slug) is None
        or str(raw_event.get("ticker") or "").strip().lower() != event_slug
        or raw_event.get("closed") is not True
    ):
        raise ValueError("Round 16 Gamma event identity is invalid")
    series = raw_event.get("series")
    if not isinstance(series, list) or ROUND16_SERIES_ID not in {
        str(item.get("id")) for item in series if isinstance(item, Mapping)
    }:
        raise ValueError("Round 16 Gamma event series differs")
    markets = raw_event.get("markets")
    if not isinstance(markets, list) or len(markets) != 1:
        raise ValueError("Round 16 Gamma event must contain one market")
    raw_market = markets[0]
    if not isinstance(raw_market, Mapping):
        raise ValueError("Round 16 Gamma market is malformed")
    raw = dict(raw_market)
    slug = str(raw.get("slug") or "").strip().lower()
    match = _SLUG.fullmatch(slug)
    if match is None or slug != event_slug:
        raise ValueError("Round 16 Gamma market slug differs")
    start = _utc_ms(raw.get("eventStartTime"), name="eventStartTime")
    end = _utc_ms(raw.get("endDate"), name="endDate")
    if (
        start != int(match.group(1)) * 1_000
        or end - start != contract.duration_ms
    ):
        raise ValueError("Round 16 Gamma market window differs")
    day = datetime.fromtimestamp(start / 1_000, tz=UTC).date().isoformat()
    role = contract.historical.role_for_day(day)
    if (
        raw.get("closed") is not True
        or raw.get("acceptingOrders") is not False
        or raw.get("enableOrderBook") is not True
        or str(raw.get("resolutionSource") or "").strip().lower().rstrip("/")
        != ROUND16_RESOLUTION_SOURCE
    ):
        raise ValueError("Round 16 Gamma market is not a closed BTC CLOB market")
    condition = str(raw.get("conditionId") or "").strip().lower()
    market_id = str(raw.get("id") or "").strip()
    if (
        _CONDITION_ID.fullmatch(condition) is None
        or not market_id.isdigit()
        or len(market_id) > 20
    ):
        raise ValueError("Round 16 Gamma market identifiers are invalid")
    outcomes = [
        str(value)
        for value in _json_array(raw.get("outcomes"), name="Round 16 outcomes")
    ]
    tokens = [
        str(value)
        for value in _json_array(
            raw.get("clobTokenIds"),
            name="Round 16 token IDs",
        )
    ]
    if (
        outcomes != ["Up", "Down"]
        or len(tokens) != 2
        or len(set(tokens)) != 2
        or any(_TOKEN_ID.fullmatch(value) is None for value in tokens)
    ):
        raise ValueError("Round 16 Gamma token mapping differs")
    tick = _decimal(
        raw.get("orderPriceMinTickSize"),
        name="Round 16 tick size",
        minimum=Decimal("0.0001"),
        maximum=Decimal("0.1"),
    )
    minimum_order = _decimal(
        raw.get("orderMinSize"),
        name="Round 16 minimum order size",
        minimum=Decimal("0.000001"),
    )
    fee = raw.get("feeSchedule")
    if raw.get("feesEnabled") is not True or not isinstance(fee, Mapping):
        raise ValueError("Round 16 Gamma fee schedule is unavailable")
    rate = _decimal(
        fee.get("rate"),
        name="Round 16 fee rate",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    rebate = _decimal(
        fee.get("rebateRate"),
        name="Round 16 rebate rate",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    exponent_value = _decimal(
        fee.get("exponent"),
        name="Round 16 fee exponent",
        minimum=Decimal("1"),
    )
    exponent = int(exponent_value)
    if (
        rate <= 0
        or fee.get("takerOnly") is not True
        or Decimal(exponent) != exponent_value
    ):
        raise ValueError("Round 16 Gamma fee schedule differs")
    question = str(raw.get("question") or "").strip()
    if not question or len(question) > 500:
        raise ValueError("Round 16 Gamma question is invalid")
    sanitized = {
        "schema_version": contract.market_schema_version,
        "event_id": event_id,
        "series_id": ROUND16_SERIES_ID,
        "market": {key: raw.get(key) for key in _IDENTITY_FIELDS},
        "role": role,
    }
    identity_json = _canonical_json(sanitized)
    observed = int(observed_at_ms)
    if observed <= end:
        raise ValueError("Round 16 identity was observed before terminal time")
    return HistoricalBtcMarket(
        event_id=event_id,
        market_id=market_id,
        condition_id=condition,
        slug=slug,
        question=question,
        event_start_ms=start,
        end_ms=end,
        role=role,
        up_token_id=tokens[0],
        down_token_id=tokens[1],
        tick_size=tick,
        minimum_order_size=minimum_order,
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=rate,
            exponent=exponent,
            taker_only=True,
            rebate_rate=rebate,
        ),
        excluded=False,
        exclusion_reason="",
        identity_payload_json=identity_json,
        identity_payload_sha256=hashlib.sha256(
            identity_json.encode("ascii")
        ).hexdigest(),
        source_payload_sha256=_canonical_sha256(raw),
        observed_at_ms=observed,
    )


class Round16HistoricalPublicClient(PolymarketHistoricalPublicClient):
    """Round 16 identity client with an exact 15-minute day boundary."""

    def events_page(
        self,
        *,
        contract: HistoricalScreenContract,
        day: str,
        after_cursor: str = "",
    ) -> PublicPayload:
        if contract.series_id != ROUND16_SERIES_ID:
            raise ValueError("Round 16 client received a different series")
        start, end = _day_bounds(day)
        params = [
            ("series_id", ROUND16_SERIES_ID),
            ("closed", "true"),
            ("end_date_min", _iso_seconds(start)),
            ("end_date_max", _iso_seconds(end + ROUND16_DURATION_MS)),
            ("order", "endDate"),
            ("ascending", "true"),
            ("limit", str(contract.requested_page_limit)),
        ]
        cursor = str(after_cursor or "").strip()
        if cursor:
            if len(cursor) > 4_096:
                raise ValueError("Round 16 Gamma cursor is too large")
            params.append(("after_cursor", cursor))
        raw = self._get(GAMMA_EVENTS_KEYSET_URL, params=params)
        return _sanitize_identity_page(raw)


def _sanitize_identity_page(page: PublicPayload) -> PublicPayload:
    """Create the only page shape allowed to cross the target-blind boundary."""

    value = page.value
    if not isinstance(value, Mapping):
        raise ValueError("Round 16 Gamma keyset response must be an object")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("Round 16 Gamma keyset events are malformed")
    sanitized_events: list[dict[str, object]] = []
    for event_value in events:
        if not isinstance(event_value, Mapping):
            raise ValueError("Round 16 Gamma keyset event is malformed")
        event = dict(event_value)
        markets = event.get("markets")
        if not isinstance(markets, list):
            raise ValueError("Round 16 Gamma event markets are malformed")
        sanitized_markets: list[dict[str, object]] = []
        for market_value in markets:
            if not isinstance(market_value, Mapping):
                raise ValueError("Round 16 Gamma market is malformed")
            sanitized_markets.append(
                {key: market_value.get(key) for key in _IDENTITY_FIELDS}
            )
        series = event.get("series")
        if not isinstance(series, list):
            raise ValueError("Round 16 Gamma event series are malformed")
        sanitized_events.append(
            {
                "id": event.get("id"),
                "ticker": event.get("ticker"),
                "slug": event.get("slug"),
                "closed": event.get("closed"),
                "series": [
                    {"id": item.get("id")}
                    for item in series
                    if isinstance(item, Mapping)
                ],
                "markets": sanitized_markets,
            }
        )
    sanitized: dict[str, object] = {
        "events": sanitized_events,
        "next_cursor": value.get("next_cursor"),
    }
    if "$schema" in value:
        sanitized["$schema"] = value["$schema"]
    canonical = _canonical_json(sanitized)
    return PublicPayload(
        value=sanitized,
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        observed_at_ms=page.observed_at_ms,
    )


def _page_events(page: PublicPayload) -> tuple[list[Mapping[str, object]], str]:
    value = page.value
    if not isinstance(value, Mapping) or set(value) - {
        "$schema",
        "events",
        "next_cursor",
    }:
        raise ValueError("Round 16 Gamma keyset response schema differs")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("Round 16 Gamma keyset events are malformed")
    parsed: list[Mapping[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("Round 16 Gamma keyset event is malformed")
        parsed.append(event)
    cursor = str(value.get("next_cursor") or "").strip()
    if cursor and len(cursor) > 4_096:
        raise ValueError("Round 16 Gamma keyset cursor is too large")
    return parsed, cursor


def collect_round16_market_identities(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    client: Round16HistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    """Collect immutable 15-minute identities without consulting outcomes."""

    if store.contract != contract.historical:
        raise ValueError("Round 16 store contract differs")
    if store.state != "initialized":
        raise ValueError("Round 16 identity collection has already closed")
    counts: dict[str, int] = {}
    seen_conditions: set[str] = set()
    seen_slugs: set[str] = set()
    completed_days = store.complete_identity_days()
    stored_by_day: dict[str, list[HistoricalBtcMarket]] = {
        day: [] for day in completed_days
    }
    for stored in store.markets(include_excluded=True):
        stored_day = (
            datetime.fromtimestamp(stored.event_start_ms / 1_000, tz=UTC)
            .date()
            .isoformat()
        )
        if stored_day in stored_by_day:
            stored_by_day[stored_day].append(stored)
    for day in contract.historical.eligible_days:
        if day in completed_days:
            reused = stored_by_day[day]
            for market in reused:
                if market.condition_id in seen_conditions or market.slug in seen_slugs:
                    raise ValueError("Round 16 identity checkpoint has duplicates")
                seen_conditions.add(market.condition_id)
                seen_slugs.add(market.slug)
            counts[day] = len(reused)
            if progress:
                progress(
                    "round16_identity_day_reused",
                    {"day": day, "day_admitted": len(reused)},
                )
            continue
        store.reset_incomplete_identity_day(day)
        start, end = _day_bounds(day)
        cursor = ""
        seen_cursors: set[str] = set()
        page_index = 0
        admitted = 0
        while True:
            page_index += 1
            page = client.events_page(
                contract=contract.historical,
                day=day,
                after_cursor=cursor,
            )
            events, next_cursor = _page_events(page)
            page_admitted = 0
            for event in events:
                event_slug = str(event.get("slug") or "").strip().lower()
                match = _SLUG.fullmatch(event_slug)
                if match is None:
                    raise ValueError("Round 16 Gamma event slug is invalid")
                if not start <= int(match.group(1)) * 1_000 < end:
                    continue
                market = parse_round16_historical_btc_event(
                    event,
                    contract=contract,
                    observed_at_ms=page.observed_at_ms,
                )
                if market.condition_id in seen_conditions or market.slug in seen_slugs:
                    raise ValueError("Round 16 Gamma returned a duplicate market")
                seen_conditions.add(market.condition_id)
                seen_slugs.add(market.slug)
                store.upsert_market(market)
                admitted += 1
                page_admitted += 1
            store.record_gamma_page(
                page=page,
                day=day,
                page_index=page_index,
                next_cursor=next_cursor,
                event_count=len(events),
                admitted_count=page_admitted,
            )
            if progress:
                progress(
                    "round16_identity_page",
                    {
                        "day": day,
                        "page_index": page_index,
                        "event_count": len(events),
                        "page_admitted": page_admitted,
                        "day_admitted": admitted,
                    },
                )
            if not next_cursor:
                break
            cursor_sha = hashlib.sha256(next_cursor.encode("utf-8")).hexdigest()
            if cursor_sha in seen_cursors:
                raise ValueError("Round 16 Gamma cursor repeated")
            seen_cursors.add(cursor_sha)
            cursor = next_cursor
            if page_index >= 20:
                raise ValueError("Round 16 day exceeded the page bound")
        counts[day] = admitted
    if any(value != ROUND16_MARKETS_PER_DAY for value in counts.values()):
        raise ValueError("Round 16 Gamma daily market count differs")
    store.transition("initialized", "identities_complete")
    return counts


__all__ = [
    "ROUND16_DECISION_OFFSETS_SECONDS",
    "ROUND16_DURATION_MS",
    "ROUND16_FLOW_WINDOWS_SECONDS",
    "ROUND16_MARKET_SCHEMA_VERSION",
    "ROUND16_MARKETS_PER_DAY",
    "ROUND16_RETURN_HORIZONS_SECONDS",
    "ROUND16_SERIES_ID",
    "Round16HistoricalContract",
    "Round16HistoricalPublicClient",
    "collect_round16_market_identities",
    "load_round16_historical_contract",
    "parse_round16_historical_btc_event",
]

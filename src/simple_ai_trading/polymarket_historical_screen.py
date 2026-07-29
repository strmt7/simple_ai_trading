"""Target-separated BTC Polymarket historical predictive screening.

This development-only lane has no wallet, account, order, position, or
execution boundary. Binance contributes public read-only BTC observations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .polymarket import PolymarketFeeSchedule, PolymarketFiveMinuteMarket
from .polymarket_resolution import validate_official_resolution


HISTORICAL_SCREEN_SCHEMA_VERSION = "polymarket-historical-screen-store-v2"
HISTORICAL_MARKET_SCHEMA_VERSION = "polymarket-historical-btc-5m-market-v2"
GAMMA_EVENTS_KEYSET_URL = "https://gamma-api.polymarket.com/events/keyset"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_MARKETS_URL = "https://clob.polymarket.com/markets"
_SERIES_ID = "10684"
_RESOLUTION_SOURCE = "https://data.chain.link/streams/btc-usd"
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SLUG = re.compile(r"^btc-updown-5m-([0-9]{10})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATE_ORDER = (
    "initialized",
    "identities_complete",
    "features_complete",
    "development_targets_complete",
    "pretest_complete",
    "targets_complete",
    "evaluated",
)
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
            raise ValueError("historical Polymarket JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"historical Polymarket JSON contains {value}")


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
        parsed = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset")
    return int(parsed.astimezone(UTC).timestamp() * 1_000)


def _day_bounds(day: str) -> tuple[int, int]:
    try:
        start = datetime.fromisoformat(day).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("historical screen day is invalid") from exc
    if start.date().isoformat() != day:
        raise ValueError("historical screen day is not canonical")
    return int(start.timestamp() * 1_000), int(
        (start + timedelta(days=1)).timestamp() * 1_000
    )


def _day_interval(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    first = str(value.get("first_day") or "")
    last = str(value.get("last_day") or "")
    first_start, _ = _day_bounds(first)
    last_start, _ = _day_bounds(last)
    if last_start < first_start:
        raise ValueError(f"{name} ends before it starts")
    count = (last_start - first_start) // 86_400_000 + 1
    if int(value.get("day_count", 0)) != count:
        raise ValueError(f"{name} day count differs")
    start = datetime.fromtimestamp(first_start / 1_000, tz=UTC)
    return tuple(
        (start + timedelta(days=index)).date().isoformat()
        for index in range(count)
    )


def _iso_seconds(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class HistoricalScreenTestGates:
    minimum_terminal_conditions: int
    minimum_outcomes_per_class: int
    minimum_decision_rows: int
    bootstrap_repetitions: int
    calibration_slope_minimum: float
    calibration_slope_maximum: float
    expected_calibration_error_maximum: float


@dataclass(frozen=True, slots=True)
class HistoricalScreenContract:
    path: Path
    contract_sha256: str
    series_id: str
    eligible_days: tuple[str, ...]
    roles: Mapping[str, str]
    excluded_slugs: frozenset[str]
    requested_page_limit: int
    decision_offsets_seconds: tuple[int, ...]
    return_horizons_seconds: tuple[int, ...]
    flow_windows_seconds: tuple[int, ...]
    source_inventory_sha256: str | None
    source_research_round: int | None
    required_source_symbol_count: int
    required_flow_rows_per_day: int
    required_market_count_per_day: int
    test_gates: HistoricalScreenTestGates

    def role_for_day(self, day: str) -> str:
        try:
            return self.roles[day]
        except KeyError as exc:
            raise ValueError(f"{day} is outside the historical screen") from exc


def load_historical_screen_contract(
    path: str | Path,
) -> HistoricalScreenContract:
    selected = Path(path)
    raw = _decode_json(selected.read_bytes(), name="historical screen contract")
    if not isinstance(raw, Mapping):
        raise ValueError("historical screen contract must be an object")
    payload = dict(raw)
    claimed = str(payload.pop("contract_sha256", "")).strip().lower()
    if _SHA256.fullmatch(claimed) is None or _canonical_sha256(payload) != claimed:
        raise ValueError("historical screen contract SHA-256 differs")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version not in {
        "polymarket-round14-btc-5m-historical-screen-v2",
        "polymarket-round15-btc-5m-historical-screen-v1",
    }:
        raise ValueError("historical screen contract schema differs")
    scope = payload.get("scope")
    source = payload.get("source_contract")
    target = payload.get("target_contract")
    causal = payload.get("causal_feature_contract")
    partition = payload.get("partition")
    test_gates_value = payload.get("test_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (scope, source, target, causal, partition, test_gates_value)
    ):
        raise ValueError("historical screen contract sections are missing")
    authority = scope.get("authority")
    expected_authority = {
        "paper_trading": False,
        "live_trading": False,
        "promotion": False,
    }
    if (
        scope.get("venue") != "polymarket"
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fiveminute"
        or scope.get("development_only") is not True
        or not isinstance(authority, Mapping)
        or dict(authority) != expected_authority
        or payload.get("profitability_claim") is not False
    ):
        raise ValueError("historical screen authority boundary differs")
    if schema_version == "polymarket-round14-btc-5m-historical-screen-v2":
        days = tuple(str(value) for value in source.get("eligible_days", ()))
        train = tuple(str(value) for value in partition.get("train_days", ()))
        tune = tuple(str(value) for value in partition.get("tune_days", ()))
        test = tuple(str(value) for value in partition.get("test_days", ()))
        if (
            days
            != ("2026-03-20", "2026-04-20", "2026-05-01", "2026-06-22")
            or train != days[:2]
            or tune != days[2:3]
            or test != days[3:]
        ):
            raise ValueError("historical screen day partition differs")
        source_inventory_sha256 = None
        source_research_round = None
        required_source_symbol_count = 3
    else:
        inventory_days = _day_interval(
            source.get("eligible_interval"),
            name="historical source interval",
        )
        excluded_days = tuple(
            str(value)
            for value in source.get("excluded_precontract_target_days", ())
        )
        excluded_day_set = frozenset(excluded_days)
        train_interval = _day_interval(
            partition.get("train_interval"),
            name="historical train interval",
        )
        tune_interval = _day_interval(
            partition.get("tune_interval"),
            name="historical tune interval",
        )
        test_interval = _day_interval(
            partition.get("test_interval"),
            name="historical test interval",
        )
        days = tuple(day for day in inventory_days if day not in excluded_day_set)
        train = tuple(day for day in train_interval if day not in excluded_day_set)
        tune = tuple(day for day in tune_interval if day not in excluded_day_set)
        test = tuple(day for day in test_interval if day not in excluded_day_set)
        source_inventory_sha256 = str(
            source.get("binance_inventory_sha256") or ""
        ).lower()
        source_research_round = int(source.get("binance_research_round", 0))
        required_source_symbol_count = int(
            source.get("required_source_symbol_count", 0)
        )
        model_day_counts = partition.get("model_day_counts_after_exclusion")
        if (
            tuple((*train, *tune, *test)) != days
            or inventory_days[0] != "2026-02-12"
            or inventory_days[-1] != "2026-07-15"
            or len(inventory_days) != 154
            or excluded_days
            != ("2026-03-20", "2026-04-20", "2026-05-01", "2026-06-22")
            or len(excluded_day_set) != len(excluded_days)
            or not excluded_day_set.issubset(inventory_days)
            or len(days) != 150
            or int(source.get("model_eligible_day_count", 0)) != len(days)
            or train[-1] != "2026-05-31"
            or tune[0] != "2026-06-01"
            or tune[-1] != "2026-06-30"
            or test[0] != "2026-07-01"
            or not isinstance(model_day_counts, Mapping)
            or dict(model_day_counts)
            != {"train": len(train), "tune": len(tune), "test": len(test)}
            or _SHA256.fullmatch(source_inventory_sha256) is None
            or source_research_round != 15
            or required_source_symbol_count != 1
        ):
            raise ValueError("historical screen day or source partition differs")
    if len(set(days)) != len(days):
        raise ValueError("historical screen days are not unique")
    for day in days:
        _day_bounds(day)
    roles = {
        **dict.fromkeys(train, "train"),
        **dict.fromkeys(tune, "tune"),
        **dict.fromkeys(test, "test"),
    }
    excluded = frozenset(
        str(value) for value in target.get("excluded_precontract_probes", ())
    )
    decisions = tuple(
        int(value) for value in causal.get("decision_offsets_seconds", ())
    )
    returns = tuple(int(value) for value in causal.get("return_horizons_seconds", ()))
    windows = tuple(int(value) for value in causal.get("flow_windows_seconds", ()))
    required_rows_per_symbol = int(
        source.get("required_flow_rows_per_symbol_day", 0)
    )
    required_market_count = int(source.get("required_market_count_per_day", 0))
    expected_binance_source = (
        "existing_audited_current_spot_perpetual_flow_1s"
        if schema_version == "polymarket-round14-btc-5m-historical-screen-v2"
        else "frozen_round15_btc_spot_perpetual_flow_1s"
    )
    slope_range = tuple(test_gates_value.get("calibration_slope_range", ()))
    if len(slope_range) != 2:
        raise ValueError("historical screen calibration slope range differs")
    test_gates = HistoricalScreenTestGates(
        minimum_terminal_conditions=int(
            test_gates_value.get("minimum_terminal_conditions", 0)
        ),
        minimum_outcomes_per_class=int(
            test_gates_value.get("minimum_outcomes_per_class", 0)
        ),
        minimum_decision_rows=int(
            test_gates_value.get("minimum_decision_rows", 0)
        ),
        bootstrap_repetitions=int(
            test_gates_value.get(
                "paired_condition_block_bootstrap_repetitions",
                0,
            )
        ),
        calibration_slope_minimum=float(
            _decimal(slope_range[0], name="calibration slope minimum")
        ),
        calibration_slope_maximum=float(
            _decimal(slope_range[1], name="calibration slope maximum")
        ),
        expected_calibration_error_maximum=float(
            _decimal(
                test_gates_value.get("expected_calibration_error_maximum"),
                name="expected calibration error maximum",
            )
        ),
    )
    expected_gate_numbers = (
        (250, 50, 1_500, 2_000, 0.75, 1.25, 0.05)
        if schema_version == "polymarket-round14-btc-5m-historical-screen-v2"
        else (3_500, 1_000, 20_000, 10_000, 0.85, 1.15, 0.03)
    )
    if (
        str(source.get("polymarket_series_id")) != _SERIES_ID
        or source.get("historical_polymarket_price_or_trade_features") is not False
        or source.get("binance_source") != expected_binance_source
        or source.get("binance_symbol") != "BTCUSDT"
        or source.get("raw_binance_copy_allowed") is not False
        or required_rows_per_symbol != 86_400
        or required_market_count != 288
        or decisions != (30, 60, 90, 120, 150, 180, 210, 240)
        or returns != (1, 5, 15, 30, 60)
        or windows != (1, 5, 15, 30)
        or not excluded
        or any(_SLUG.fullmatch(value) is None for value in excluded)
        or (
            test_gates.minimum_terminal_conditions,
            test_gates.minimum_outcomes_per_class,
            test_gates.minimum_decision_rows,
            test_gates.bootstrap_repetitions,
            test_gates.calibration_slope_minimum,
            test_gates.calibration_slope_maximum,
            test_gates.expected_calibration_error_maximum,
        )
        != expected_gate_numbers
        or any(
            test_gates_value.get(key) is not True
            for key in (
                "challenger_log_loss_skill_over_best_control_strictly_positive",
                "challenger_brier_skill_over_best_control_strictly_positive",
                "challenger_balanced_accuracy_not_lower_than_best_control",
                "paired_log_loss_improvement_lower_bound_strictly_positive",
            )
        )
    ):
        raise ValueError("historical screen source or feature contract differs")
    return HistoricalScreenContract(
        path=selected,
        contract_sha256=claimed,
        series_id=_SERIES_ID,
        eligible_days=days,
        roles=roles,
        excluded_slugs=excluded,
        requested_page_limit=500,
        decision_offsets_seconds=decisions,
        return_horizons_seconds=returns,
        flow_windows_seconds=windows,
        source_inventory_sha256=source_inventory_sha256,
        source_research_round=source_research_round,
        required_source_symbol_count=required_source_symbol_count,
        required_flow_rows_per_day=(
            required_rows_per_symbol * required_source_symbol_count
        ),
        required_market_count_per_day=required_market_count,
        test_gates=test_gates,
    )


@dataclass(frozen=True, slots=True)
class HistoricalBtcMarket:
    event_id: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    event_start_ms: int
    end_ms: int
    role: str
    up_token_id: str
    down_token_id: str
    tick_size: Decimal
    minimum_order_size: Decimal
    fee_schedule: PolymarketFeeSchedule
    excluded: bool
    exclusion_reason: str
    identity_payload_json: str
    identity_payload_sha256: str
    source_payload_sha256: str
    observed_at_ms: int

    @property
    def token_ids(self) -> tuple[str, str]:
        return self.up_token_id, self.down_token_id

    def official_market(
        self,
        gamma_payload: Mapping[str, object],
    ) -> PolymarketFiveMinuteMarket:
        canonical = _canonical_json(dict(gamma_payload))
        return PolymarketFiveMinuteMarket(
            asset="BTC",
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            question=self.question,
            event_start_ms=self.event_start_ms,
            end_ms=self.end_ms,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            tick_size=self.tick_size,
            minimum_order_size=self.minimum_order_size,
            fee_schedule=self.fee_schedule,
            liquidity_quote=Decimal("0"),
            volume_quote=Decimal("0"),
            resolution_source=_RESOLUTION_SOURCE,
            gamma_payload_sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
            gamma_payload_json=canonical,
        )


def parse_historical_btc_event(
    payload: Mapping[str, object],
    *,
    contract: HistoricalScreenContract,
    observed_at_ms: int,
) -> HistoricalBtcMarket:
    """Parse target-free identity from one closed Gamma series event."""

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
        raise ValueError("historical Gamma event identity is invalid")
    series = raw_event.get("series")
    if not isinstance(series, list) or contract.series_id not in {
        str(item.get("id")) for item in series if isinstance(item, Mapping)
    }:
        raise ValueError("historical Gamma event series differs")
    markets = raw_event.get("markets")
    if not isinstance(markets, list) or len(markets) != 1:
        raise ValueError("historical Gamma event must contain exactly one market")
    market_value = markets[0]
    if not isinstance(market_value, Mapping):
        raise ValueError("historical Gamma market is malformed")
    raw = dict(market_value)
    slug = str(raw.get("slug") or "").strip().lower()
    match = _SLUG.fullmatch(slug)
    if match is None or slug != event_slug:
        raise ValueError("historical Gamma market slug differs")
    start = _utc_ms(raw.get("eventStartTime"), name="eventStartTime")
    end = _utc_ms(raw.get("endDate"), name="endDate")
    if start != int(match.group(1)) * 1_000 or end - start != 300_000:
        raise ValueError("historical Gamma market window differs")
    day = datetime.fromtimestamp(start / 1_000, tz=UTC).date().isoformat()
    role = contract.role_for_day(day)
    if (
        raw.get("closed") is not True
        or raw.get("acceptingOrders") is not False
        or raw.get("enableOrderBook") is not True
        or str(raw.get("resolutionSource") or "").strip().lower().rstrip("/")
        != _RESOLUTION_SOURCE
    ):
        raise ValueError("historical Gamma market is not a closed BTC CLOB market")
    condition = str(raw.get("conditionId") or "").strip().lower()
    market_id = str(raw.get("id") or "").strip()
    if (
        _CONDITION_ID.fullmatch(condition) is None
        or not market_id.isdigit()
        or len(market_id) > 20
    ):
        raise ValueError("historical Gamma market identifiers are invalid")
    outcomes = [
        str(value)
        for value in _json_array(raw.get("outcomes"), name="historical outcomes")
    ]
    tokens = [
        str(value)
        for value in _json_array(
            raw.get("clobTokenIds"),
            name="historical token IDs",
        )
    ]
    if (
        outcomes != ["Up", "Down"]
        or len(tokens) != 2
        or len(set(tokens)) != 2
        or any(_TOKEN_ID.fullmatch(value) is None for value in tokens)
    ):
        raise ValueError("historical Gamma token mapping differs")
    tick = _decimal(
        raw.get("orderPriceMinTickSize"),
        name="historical tick size",
        minimum=Decimal("0.0001"),
        maximum=Decimal("0.1"),
    )
    minimum_order = _decimal(
        raw.get("orderMinSize"),
        name="historical minimum order size",
        minimum=Decimal("0.000001"),
    )
    fee = raw.get("feeSchedule")
    if raw.get("feesEnabled") is not True or not isinstance(fee, Mapping):
        raise ValueError("historical Gamma fee schedule is unavailable")
    rate = _decimal(
        fee.get("rate"),
        name="historical fee rate",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    rebate = _decimal(
        fee.get("rebateRate"),
        name="historical rebate rate",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    exponent_value = _decimal(
        fee.get("exponent"),
        name="historical fee exponent",
        minimum=Decimal("1"),
    )
    exponent = int(exponent_value)
    if (
        rate <= 0
        or fee.get("takerOnly") is not True
        or Decimal(exponent) != (exponent_value)
    ):
        raise ValueError("historical Gamma fee schedule differs")
    question = str(raw.get("question") or "").strip()
    if not question or len(question) > 500:
        raise ValueError("historical Gamma question is invalid")
    sanitized = {
        "schema_version": HISTORICAL_MARKET_SCHEMA_VERSION,
        "event_id": event_id,
        "series_id": contract.series_id,
        "market": {key: raw.get(key) for key in _IDENTITY_FIELDS},
        "role": role,
    }
    identity_json = _canonical_json(sanitized)
    excluded = slug in contract.excluded_slugs
    observed = int(observed_at_ms)
    if observed <= end:
        raise ValueError("historical Gamma identity was observed before terminal time")
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
        excluded=excluded,
        exclusion_reason="precontract_probe" if excluded else "",
        identity_payload_json=identity_json,
        identity_payload_sha256=hashlib.sha256(
            identity_json.encode("ascii")
        ).hexdigest(),
        source_payload_sha256=_canonical_sha256(raw),
        observed_at_ms=observed,
    )


class _OriginRateLimiter:
    def __init__(
        self,
        *,
        requests_per_second: float = 5.0,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        rate = float(requests_per_second)
        if not 0 < rate <= 5:
            raise ValueError("historical public request rate must lie in (0, 5]")
        self._minimum_interval = 1.0 / rate
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._last_by_origin: dict[str, float] = {}

    def wait(self, origin: str) -> None:
        now = self._monotonic()
        prior = self._last_by_origin.get(origin)
        if prior is not None:
            delay = self._minimum_interval - (now - prior)
            if delay > 0:
                self._sleeper(delay)
                now = self._monotonic()
        self._last_by_origin[origin] = now


def _public_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "simple-ai-trading/0.1.0-beta.1 historical-screen"}
    )
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4),
    )
    return session


@dataclass(frozen=True, slots=True)
class PublicPayload:
    value: object
    canonical_json: str
    sha256: str
    observed_at_ms: int


class PolymarketHistoricalPublicClient:
    """Bounded public-only client with no authentication or trading methods."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 15.0,
        maximum_response_bytes: int = 8 * 1024 * 1024,
        limiter: _OriginRateLimiter | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.session = session or _public_session()
        self.timeout_seconds = max(2.0, min(30.0, float(timeout_seconds)))
        self.maximum_response_bytes = max(
            1024,
            min(16 * 1024 * 1024, int(maximum_response_bytes)),
        )
        self.limiter = limiter or _OriginRateLimiter()
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def _get(
        self,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None = None,
    ) -> PublicPayload:
        self.limiter.wait(url.split("/", 3)[2].lower())
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        if len(response.content) > self.maximum_response_bytes:
            raise ValueError("historical public response exceeded the size bound")
        value = _decode_json(response.content, name="historical public response")
        canonical = _canonical_json(value)
        return PublicPayload(
            value=value,
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
            observed_at_ms=int(self.clock_ms()),
        )

    def events_page(
        self,
        *,
        contract: HistoricalScreenContract,
        day: str,
        after_cursor: str = "",
    ) -> PublicPayload:
        start, end = _day_bounds(day)
        params = [
            ("series_id", contract.series_id),
            ("closed", "true"),
            ("end_date_min", _iso_seconds(start)),
            ("end_date_max", _iso_seconds(end + 300_000)),
            ("order", "endDate"),
            ("ascending", "true"),
            ("limit", str(contract.requested_page_limit)),
        ]
        cursor = str(after_cursor or "").strip()
        if cursor:
            if len(cursor) > 4096:
                raise ValueError("historical Gamma cursor is too large")
            params.append(("after_cursor", cursor))
        return self._get(GAMMA_EVENTS_KEYSET_URL, params=params)

    def gamma_market(self, market_id: str) -> PublicPayload:
        selected = str(market_id or "").strip()
        if not selected.isdigit() or len(selected) > 20:
            raise ValueError("historical Gamma market id is invalid")
        return self._get(f"{GAMMA_MARKETS_URL}/{selected}")

    def clob_market(self, condition_id: str) -> PublicPayload:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("historical CLOB condition id is invalid")
        return self._get(f"{CLOB_MARKETS_URL}/{selected}")


class HistoricalScreenStore:
    """One compact DuckDB with feature and target schemas kept separate."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract: HistoricalScreenContract,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.contract = contract
        self.read_only = bool(read_only)
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.path), read_only=self.read_only)
        if not self.read_only:
            self._initialize()
        self._verify_manifest()

    def connect(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> HistoricalScreenStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE SCHEMA IF NOT EXISTS feature;
            CREATE SCHEMA IF NOT EXISTS target;
            CREATE TABLE IF NOT EXISTS feature.screen_manifest (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                state VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL,
                updated_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.market_identity (
                condition_id VARCHAR PRIMARY KEY,
                event_id VARCHAR NOT NULL,
                market_id VARCHAR NOT NULL,
                slug VARCHAR NOT NULL UNIQUE,
                event_start_ms BIGINT NOT NULL,
                end_ms BIGINT NOT NULL,
                role VARCHAR NOT NULL,
                up_token_id VARCHAR NOT NULL UNIQUE,
                down_token_id VARCHAR NOT NULL UNIQUE,
                tick_size VARCHAR NOT NULL,
                minimum_order_size VARCHAR NOT NULL,
                fee_rate VARCHAR NOT NULL,
                fee_exponent INTEGER NOT NULL,
                fee_rebate_rate VARCHAR NOT NULL,
                excluded BOOLEAN NOT NULL,
                exclusion_reason VARCHAR NOT NULL,
                identity_payload_json VARCHAR NOT NULL,
                identity_payload_sha256 VARCHAR NOT NULL,
                source_payload_sha256 VARCHAR NOT NULL,
                observed_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.gamma_page_receipt (
                page_sha256 VARCHAR PRIMARY KEY,
                day VARCHAR NOT NULL,
                page_index INTEGER NOT NULL,
                next_cursor_sha256 VARCHAR NOT NULL,
                event_count INTEGER NOT NULL,
                admitted_count INTEGER NOT NULL,
                observed_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.causal_row (
                condition_id VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                event_start_ms BIGINT NOT NULL,
                decision_time_ms BIGINT NOT NULL,
                decision_offset_seconds INTEGER NOT NULL,
                feature_values FLOAT[] NOT NULL,
                feature_vector_sha256 VARCHAR NOT NULL,
                row_sha256 VARCHAR NOT NULL,
                PRIMARY KEY (condition_id, decision_time_ms)
            );
            CREATE TABLE IF NOT EXISTS feature.dataset_manifest (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                feature_names_json VARCHAR NOT NULL,
                feature_names_sha256 VARCHAR NOT NULL,
                binance_source_manifest_json VARCHAR NOT NULL,
                binance_source_manifest_sha256 VARCHAR NOT NULL,
                market_identity_sha256 VARCHAR NOT NULL,
                materializer_sha256 VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                condition_count BIGINT NOT NULL,
                role_counts_json VARCHAR NOT NULL,
                row_chain_sha256 VARCHAR NOT NULL,
                dataset_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.pretest_manifest (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                dataset_sha256 VARCHAR NOT NULL,
                artifact_json VARCHAR NOT NULL,
                artifact_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS target.official_resolution (
                condition_id VARCHAR PRIMARY KEY,
                role VARCHAR NOT NULL,
                winning_token_id VARCHAR NOT NULL,
                winning_outcome VARCHAR NOT NULL,
                gamma_payload_json VARCHAR NOT NULL,
                gamma_payload_sha256 VARCHAR NOT NULL,
                clob_payload_json VARCHAR NOT NULL,
                clob_payload_sha256 VARCHAR NOT NULL,
                evidence_sha256 VARCHAR NOT NULL,
                observed_at_ms BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS target.evaluation_manifest (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                contract_sha256 VARCHAR NOT NULL,
                dataset_sha256 VARCHAR NOT NULL,
                pretest_artifact_sha256 VARCHAR NOT NULL,
                artifact_json VARCHAR NOT NULL,
                artifact_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            );
            """
        )
        now = time.time_ns() // 1_000_000
        self._connection.execute(
            """
            INSERT INTO feature.screen_manifest
            SELECT true, ?, ?, 'initialized', ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM feature.screen_manifest)
            """,
            [
                HISTORICAL_SCREEN_SCHEMA_VERSION,
                self.contract.contract_sha256,
                now,
                now,
            ],
        )

    def _verify_manifest(self) -> None:
        row = self._connection.execute(
            """
            SELECT schema_version, contract_sha256, state
            FROM feature.screen_manifest WHERE singleton
            """
        ).fetchone()
        if (
            row is None
            or str(row[0]) != HISTORICAL_SCREEN_SCHEMA_VERSION
            or str(row[1]) != self.contract.contract_sha256
            or str(row[2]) not in _STATE_ORDER
        ):
            raise ValueError("historical screen store manifest differs")

    @property
    def state(self) -> str:
        row = self._connection.execute(
            "SELECT state FROM feature.screen_manifest WHERE singleton"
        ).fetchone()
        if row is None:
            raise ValueError("historical screen store manifest is missing")
        return str(row[0])

    def transition(self, expected: str, new: str) -> None:
        if self.read_only:
            raise ValueError("historical screen store is read-only")
        if (
            expected not in _STATE_ORDER
            or new not in _STATE_ORDER
            or _STATE_ORDER.index(new) != _STATE_ORDER.index(expected) + 1
        ):
            raise ValueError("historical screen state transition is invalid")
        now = time.time_ns() // 1_000_000
        self._connection.execute("BEGIN TRANSACTION")
        try:
            changed = self._connection.execute(
                """
                UPDATE feature.screen_manifest
                SET state = ?, updated_at_ms = ?
                WHERE singleton AND state = ?
                RETURNING state
                """,
                [new, now, expected],
            ).fetchone()
            if changed is None:
                raise ValueError(
                    f"historical screen state is {self.state}, expected {expected}"
                )
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def upsert_market(self, market: HistoricalBtcMarket) -> None:
        if self.read_only or self.state != "initialized":
            raise ValueError("historical market identity phase is closed")
        self._connection.execute(
            """
            INSERT INTO feature.market_identity VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (condition_id) DO UPDATE SET
                event_id = excluded.event_id,
                market_id = excluded.market_id,
                slug = excluded.slug,
                event_start_ms = excluded.event_start_ms,
                end_ms = excluded.end_ms,
                role = excluded.role,
                up_token_id = excluded.up_token_id,
                down_token_id = excluded.down_token_id,
                tick_size = excluded.tick_size,
                minimum_order_size = excluded.minimum_order_size,
                fee_rate = excluded.fee_rate,
                fee_exponent = excluded.fee_exponent,
                fee_rebate_rate = excluded.fee_rebate_rate,
                excluded = excluded.excluded,
                exclusion_reason = excluded.exclusion_reason,
                identity_payload_json = excluded.identity_payload_json,
                identity_payload_sha256 = excluded.identity_payload_sha256,
                source_payload_sha256 = excluded.source_payload_sha256,
                observed_at_ms = excluded.observed_at_ms
            """,
            [
                market.condition_id,
                market.event_id,
                market.market_id,
                market.slug,
                market.event_start_ms,
                market.end_ms,
                market.role,
                market.up_token_id,
                market.down_token_id,
                format(market.tick_size, "f"),
                format(market.minimum_order_size, "f"),
                format(market.fee_schedule.rate, "f"),
                market.fee_schedule.exponent,
                format(market.fee_schedule.rebate_rate, "f"),
                market.excluded,
                market.exclusion_reason,
                market.identity_payload_json,
                market.identity_payload_sha256,
                market.source_payload_sha256,
                market.observed_at_ms,
            ],
        )

    def record_gamma_page(
        self,
        *,
        page: PublicPayload,
        day: str,
        page_index: int,
        next_cursor: str,
        event_count: int,
        admitted_count: int,
    ) -> None:
        cursor_sha = (
            hashlib.sha256(next_cursor.encode("utf-8")).hexdigest()
            if next_cursor
            else "0" * 64
        )
        self._connection.execute(
            """
            INSERT OR REPLACE INTO feature.gamma_page_receipt VALUES (
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                page.sha256,
                day,
                int(page_index),
                cursor_sha,
                int(event_count),
                int(admitted_count),
                page.observed_at_ms,
            ],
        )

    def complete_identity_days(self) -> Mapping[str, int]:
        if self.state != "initialized":
            raise ValueError("historical identity checkpoint phase is closed")
        terminal_days = {
            str(row[0])
            for row in self._connection.execute(
                """
                SELECT DISTINCT day
                FROM feature.gamma_page_receipt
                WHERE next_cursor_sha256 = ?
                """,
                ["0" * 64],
            ).fetchall()
        }
        counts = dict.fromkeys(self.contract.eligible_days, 0)
        for market in self.markets(include_excluded=True):
            day = (
                datetime.fromtimestamp(
                    market.event_start_ms / 1_000,
                    tz=UTC,
                )
                .date()
                .isoformat()
            )
            if day in counts:
                counts[day] += 1
        return {
            day: count
            for day, count in counts.items()
            if day in terminal_days
            and count == self.contract.required_market_count_per_day
        }

    def reset_incomplete_identity_day(self, day: str) -> None:
        if self.read_only or self.state != "initialized":
            raise ValueError("historical identity checkpoint phase is closed")
        start, end = _day_bounds(day)
        if day not in self.contract.eligible_days:
            raise ValueError("historical identity checkpoint day is outside scope")
        terminal = self._connection.execute(
            """
            SELECT count(*) FROM feature.gamma_page_receipt
            WHERE day = ? AND next_cursor_sha256 = ?
            """,
            [day, "0" * 64],
        ).fetchone()
        market_count = self._connection.execute(
            """
            SELECT count(*) FROM feature.market_identity
            WHERE event_start_ms >= ? AND event_start_ms < ?
            """,
            [start, end],
        ).fetchone()
        if (
            terminal is not None
            and int(terminal[0]) > 0
            and market_count is not None
            and int(market_count[0]) == self.contract.required_market_count_per_day
        ):
            raise ValueError("complete historical identity day cannot be reset")
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                """
                DELETE FROM feature.market_identity
                WHERE event_start_ms >= ? AND event_start_ms < ?
                """,
                [start, end],
            )
            self._connection.execute(
                "DELETE FROM feature.gamma_page_receipt WHERE day = ?",
                [day],
            )
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def markets(
        self,
        *,
        roles: Sequence[str] | None = None,
        include_excluded: bool = False,
    ) -> tuple[HistoricalBtcMarket, ...]:
        where = ["true" if include_excluded else "NOT excluded"]
        parameters: list[object] = []
        selected_roles = tuple(roles or ())
        if selected_roles:
            if any(role not in {"train", "tune", "test"} for role in selected_roles):
                raise ValueError("historical market role filter is invalid")
            where.append("role IN (" + ",".join("?" for _ in selected_roles) + ")")
            parameters.extend(selected_roles)
        rows = self._connection.execute(
            f"""
            SELECT event_id, market_id, condition_id, slug, event_start_ms,
                   end_ms, role, up_token_id, down_token_id, tick_size,
                   minimum_order_size, fee_rate, fee_exponent, fee_rebate_rate,
                   excluded, exclusion_reason, identity_payload_json,
                   identity_payload_sha256, source_payload_sha256, observed_at_ms
            FROM feature.market_identity
            WHERE {" AND ".join(where)}
            ORDER BY event_start_ms
            """,
            parameters,
        ).fetchall()
        output: list[HistoricalBtcMarket] = []
        for row in rows:
            identity = _decode_json(
                str(row[16]).encode("ascii"),
                name="stored historical identity",
            )
            market_value = (
                identity.get("market") if isinstance(identity, Mapping) else None
            )
            if not isinstance(market_value, Mapping):
                raise ValueError("stored historical identity is malformed")
            output.append(
                HistoricalBtcMarket(
                    event_id=str(row[0]),
                    market_id=str(row[1]),
                    condition_id=str(row[2]),
                    slug=str(row[3]),
                    question=str(market_value.get("question") or ""),
                    event_start_ms=int(row[4]),
                    end_ms=int(row[5]),
                    role=str(row[6]),
                    up_token_id=str(row[7]),
                    down_token_id=str(row[8]),
                    tick_size=Decimal(str(row[9])),
                    minimum_order_size=Decimal(str(row[10])),
                    fee_schedule=PolymarketFeeSchedule(
                        enabled=True,
                        rate=Decimal(str(row[11])),
                        exponent=int(row[12]),
                        taker_only=True,
                        rebate_rate=Decimal(str(row[13])),
                    ),
                    excluded=bool(row[14]),
                    exclusion_reason=str(row[15]),
                    identity_payload_json=str(row[16]),
                    identity_payload_sha256=str(row[17]),
                    source_payload_sha256=str(row[18]),
                    observed_at_ms=int(row[19]),
                )
            )
        return tuple(output)

    def resolved_conditions(self, *, roles: Sequence[str]) -> frozenset[str]:
        selected = tuple(roles)
        if not selected:
            return frozenset()
        placeholders = ",".join("?" for _ in selected)
        rows = self._connection.execute(
            f"""
            SELECT condition_id FROM target.official_resolution
            WHERE role IN ({placeholders})
            """,
            list(selected),
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def record_pretest_artifact(self, artifact: Mapping[str, object]) -> str:
        if self.read_only or self.state != "development_targets_complete":
            raise ValueError("historical pretest artifact phase is not authorized")
        test_count = int(
            self._connection.execute(
                """
                SELECT count(*) FROM target.official_resolution
                WHERE role = 'test'
                """
            ).fetchone()[0]
        )
        dataset = self._connection.execute(
            """
            SELECT dataset_sha256 FROM feature.dataset_manifest
            WHERE singleton
            """
        ).fetchone()
        value = dict(artifact)
        if (
            test_count != 0
            or dataset is None
            or value.get("schema_version") != "polymarket-historical-btc-pretest-v1"
            or value.get("contract_sha256") != self.contract.contract_sha256
            or value.get("dataset_sha256") != str(dataset[0])
        ):
            raise ValueError("historical pretest artifact binding differs")
        canonical = _canonical_json(value)
        artifact_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        now = time.time_ns() // 1_000_000
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                """
                INSERT INTO feature.pretest_manifest VALUES (
                    true, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    str(value["schema_version"]),
                    self.contract.contract_sha256,
                    str(dataset[0]),
                    canonical,
                    artifact_sha,
                    now,
                ],
            )
            changed = self._connection.execute(
                """
                UPDATE feature.screen_manifest
                SET state = 'pretest_complete', updated_at_ms = ?
                WHERE singleton AND state = 'development_targets_complete'
                RETURNING state
                """,
                [now],
            ).fetchone()
            if changed is None:
                raise ValueError("historical pretest state changed concurrently")
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return artifact_sha

    def pretest_artifact(self) -> tuple[Mapping[str, object], str]:
        row = self._connection.execute(
            """
            SELECT artifact_json, artifact_sha256
            FROM feature.pretest_manifest WHERE singleton
            """
        ).fetchone()
        if row is None:
            raise ValueError("historical pretest artifact is missing")
        value = _decode_json(
            str(row[0]).encode("ascii"),
            name="historical pretest artifact",
        )
        if (
            not isinstance(value, Mapping)
            or _canonical_json(dict(value)) != str(row[0])
            or hashlib.sha256(str(row[0]).encode("ascii")).hexdigest() != str(row[1])
        ):
            raise ValueError("historical pretest artifact integrity failed")
        return value, str(row[1])

    def record_evaluation_artifact(self, artifact: Mapping[str, object]) -> str:
        if self.read_only or self.state != "targets_complete":
            raise ValueError("historical evaluation phase is not authorized")
        pretest, pretest_sha = self.pretest_artifact()
        value = dict(artifact)
        if (
            value.get("schema_version") != "polymarket-historical-btc-evaluation-v1"
            or value.get("contract_sha256") != self.contract.contract_sha256
            or value.get("dataset_sha256") != pretest.get("dataset_sha256")
            or value.get("pretest_artifact_sha256") != pretest_sha
        ):
            raise ValueError("historical evaluation artifact binding differs")
        canonical = _canonical_json(value)
        artifact_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        now = time.time_ns() // 1_000_000
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                """
                INSERT INTO target.evaluation_manifest VALUES (
                    true, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    str(value["schema_version"]),
                    self.contract.contract_sha256,
                    str(value["dataset_sha256"]),
                    pretest_sha,
                    canonical,
                    artifact_sha,
                    now,
                ],
            )
            changed = self._connection.execute(
                """
                UPDATE feature.screen_manifest
                SET state = 'evaluated', updated_at_ms = ?
                WHERE singleton AND state = 'targets_complete'
                RETURNING state
                """,
                [now],
            ).fetchone()
            if changed is None:
                raise ValueError("historical evaluation state changed concurrently")
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return artifact_sha

    def record_resolution(
        self,
        *,
        market: HistoricalBtcMarket,
        gamma: PublicPayload,
        clob: PublicPayload,
    ) -> str:
        allowed = (
            market.role in {"train", "tune"} and self.state == "features_complete"
        ) or (market.role == "test" and self.state == "pretest_complete")
        if self.read_only or not allowed:
            raise ValueError("historical target role or phase is not authorized")
        if not isinstance(gamma.value, Mapping) or not isinstance(
            clob.value,
            Mapping,
        ):
            raise ValueError("historical official resolution payload is malformed")
        official = market.official_market(gamma.value)
        winner = validate_official_resolution(
            official,
            clob.value,
            gamma.value,
            observed_wall_ms=max(gamma.observed_at_ms, clob.observed_at_ms),
        )
        if winner is None:
            raise ValueError("historical official resolution is not terminal")
        evidence = {
            "schema_version": "polymarket-historical-resolution-v2",
            "contract_sha256": self.contract.contract_sha256,
            "condition_id": market.condition_id,
            "role": market.role,
            "winning_token_id": winner[0],
            "winning_outcome": winner[1],
            "gamma_payload_sha256": gamma.sha256,
            "clob_payload_sha256": clob.sha256,
        }
        evidence_sha = _canonical_sha256(evidence)
        self._connection.execute(
            """
            INSERT INTO target.official_resolution VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (condition_id) DO UPDATE SET
                role = excluded.role,
                winning_token_id = excluded.winning_token_id,
                winning_outcome = excluded.winning_outcome,
                gamma_payload_json = excluded.gamma_payload_json,
                gamma_payload_sha256 = excluded.gamma_payload_sha256,
                clob_payload_json = excluded.clob_payload_json,
                clob_payload_sha256 = excluded.clob_payload_sha256,
                evidence_sha256 = excluded.evidence_sha256,
                observed_at_ms = excluded.observed_at_ms
            """,
            [
                market.condition_id,
                market.role,
                winner[0],
                winner[1],
                gamma.canonical_json,
                gamma.sha256,
                clob.canonical_json,
                clob.sha256,
                evidence_sha,
                max(gamma.observed_at_ms, clob.observed_at_ms),
            ],
        )
        return winner[1]


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _page_events(page: PublicPayload) -> tuple[list[Mapping[str, object]], str]:
    value = page.value
    if not isinstance(value, Mapping) or set(value) - {
        "$schema",
        "events",
        "next_cursor",
    }:
        raise ValueError("historical Gamma keyset response schema differs")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("historical Gamma keyset events are malformed")
    parsed: list[Mapping[str, object]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("historical Gamma keyset event is malformed")
        parsed.append(event)
    cursor = str(value.get("next_cursor") or "").strip()
    if cursor and len(cursor) > 4096:
        raise ValueError("historical Gamma keyset cursor is too large")
    return parsed, cursor


def collect_historical_market_identities(
    store: HistoricalScreenStore,
    client: PolymarketHistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    if store.state != "initialized":
        raise ValueError("historical identity collection has already closed")
    counts: dict[str, int] = {}
    seen_conditions: set[str] = set()
    seen_slugs: set[str] = set()
    completed_days = store.complete_identity_days()
    stored_by_day: dict[str, list[HistoricalBtcMarket]] = {
        day: [] for day in completed_days
    }
    for stored in store.markets(include_excluded=True):
        stored_day = (
            datetime.fromtimestamp(
                stored.event_start_ms / 1_000,
                tz=UTC,
            )
            .date()
            .isoformat()
        )
        if stored_day in stored_by_day:
            stored_by_day[stored_day].append(stored)
    for day in store.contract.eligible_days:
        if day in completed_days:
            reused = stored_by_day[day]
            for market in reused:
                if (
                    market.condition_id in seen_conditions
                    or market.slug in seen_slugs
                ):
                    raise ValueError(
                        "historical identity checkpoint contains duplicates"
                    )
                seen_conditions.add(market.condition_id)
                seen_slugs.add(market.slug)
            counts[day] = len(reused)
            if progress:
                progress(
                    "historical_identity_day_reused",
                    {
                        "day": day,
                        "day_admitted": len(reused),
                    },
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
                contract=store.contract,
                day=day,
                after_cursor=cursor,
            )
            events, next_cursor = _page_events(page)
            page_admitted = 0
            for event in events:
                event_slug = str(event.get("slug") or "").strip().lower()
                match = _SLUG.fullmatch(event_slug)
                if match is None:
                    raise ValueError("historical Gamma event slug is invalid")
                if not start <= int(match.group(1)) * 1_000 < end:
                    continue
                market = parse_historical_btc_event(
                    event,
                    contract=store.contract,
                    observed_at_ms=page.observed_at_ms,
                )
                if market.condition_id in seen_conditions or market.slug in seen_slugs:
                    raise ValueError(
                        "historical Gamma keyset returned a duplicate market"
                    )
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
                    "historical_identity_page",
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
                raise ValueError("historical Gamma keyset cursor repeated")
            seen_cursors.add(cursor_sha)
            cursor = next_cursor
            if page_index >= 20:
                raise ValueError("historical Gamma day exceeded the page bound")
        counts[day] = admitted
    if any(
        value != store.contract.required_market_count_per_day
        for value in counts.values()
    ):
        raise ValueError("historical Gamma day market count differs")
    store.transition("initialized", "identities_complete")
    return counts


def _collect_targets(
    store: HistoricalScreenStore,
    client: PolymarketHistoricalPublicClient,
    *,
    roles: tuple[str, ...],
    expected_state: str,
    next_state: str,
    progress: ProgressCallback | None,
) -> Mapping[str, int]:
    if store.state != expected_state:
        raise ValueError(f"historical target collection requires {expected_state}")
    markets = store.markets(roles=roles)
    existing = store.resolved_conditions(roles=roles)
    counts = {"Up": 0, "Down": 0}
    if existing:
        placeholders = ",".join("?" for _ in roles)
        for winner, count in (
            store.connect()
            .execute(
                f"""
            SELECT winning_outcome, count(*)
            FROM target.official_resolution
            WHERE role IN ({placeholders})
            GROUP BY winning_outcome
            """,
                list(roles),
            )
            .fetchall()
        ):
            counts[str(winner)] = int(count)
    for index, market in enumerate(markets, start=1):
        if market.condition_id in existing:
            continue
        gamma = client.gamma_market(market.market_id)
        clob = client.clob_market(market.condition_id)
        winner = store.record_resolution(
            market=market,
            gamma=gamma,
            clob=clob,
        )
        counts[winner] += 1
        if progress:
            progress(
                "historical_target",
                {
                    "market_index": index,
                    "market_count": len(markets),
                    "event_start_ms": market.event_start_ms,
                    "role": market.role,
                    "up_count": counts["Up"],
                    "down_count": counts["Down"],
                },
            )
    if sum(counts.values()) != len(markets) or min(counts.values()) < 50:
        raise ValueError("historical target coverage is below the frozen gate")
    store.transition(expected_state, next_state)
    return counts


def collect_historical_development_targets(
    store: HistoricalScreenStore,
    client: PolymarketHistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    return _collect_targets(
        store,
        client,
        roles=("train", "tune"),
        expected_state="features_complete",
        next_state="development_targets_complete",
        progress=progress,
    )


def collect_historical_test_targets(
    store: HistoricalScreenStore,
    client: PolymarketHistoricalPublicClient,
    *,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    return _collect_targets(
        store,
        client,
        roles=("test",),
        expected_state="pretest_complete",
        next_state="targets_complete",
        progress=progress,
    )


__all__ = [
    "CLOB_MARKETS_URL",
    "GAMMA_EVENTS_KEYSET_URL",
    "GAMMA_MARKETS_URL",
    "HISTORICAL_MARKET_SCHEMA_VERSION",
    "HISTORICAL_SCREEN_SCHEMA_VERSION",
    "HistoricalBtcMarket",
    "HistoricalScreenContract",
    "HistoricalScreenTestGates",
    "HistoricalScreenStore",
    "PolymarketHistoricalPublicClient",
    "PublicPayload",
    "collect_historical_development_targets",
    "collect_historical_market_identities",
    "collect_historical_test_targets",
    "load_historical_screen_contract",
    "parse_historical_btc_event",
]

"""Receipt-time Polymarket core features for the Round 21 BTC study."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .polymarket_btc_reference import parse_polymarket_chainlink_btc_tick
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_redundant_union import (
    POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION,
    PolymarketUnionEvent,
)
from .polymarket_round21_binance_features import (
    POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    Round21OptionalBinanceFeatures,
)
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_CONDITION_DURATION_MS,
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS,
    Round21CausalFeatureRow,
    Round21FeatureSchema,
)


POLYMARKET_ROUND21_CORE_FEATURE_SCHEMA_VERSION = (
    "polymarket-round21-receipt-time-core-features-v1"
)
POLYMARKET_ROUND21_FEATURE_POLICY_SCHEMA_VERSION = (
    "polymarket-round21-causal-feature-policy-v1"
)
POLYMARKET_ROUND21_FEATURE_POLICY_SHA256 = (
    "a55408ebb99180cdfd21b443a924002ffd9cc1443b54ae1f882b49a38ae8ce70"
)
POLYMARKET_ROUND21_CORE_WINDOWS_MS = (
    250,
    1_000,
    5_000,
    15_000,
    30_000,
    60_000,
    120_000,
)
POLYMARKET_ROUND21_CORE_MAXIMUM_SOURCE_AGE_MS = 1_500
POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_RETURNS = 20
POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_COVERAGE_MS = 30_000
_CHAINLINK_METRICS = (
    "log_return",
    "realized_variance",
    "bipower_variation",
    "jump_fraction",
    "tick_count",
)
_BOOK_FLOW_METRICS = (
    "top_order_flow_imbalance",
    "mean_top_quantity_imbalance",
    "microprice_log_return",
    "level_quantity_flow_pressure",
    "log1p_gross_level_quantity_change",
    "book_update_count",
)
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
_STATIC_FEATURE_NAMES = (
    "core.elapsed_fraction",
    "core.remaining_seconds",
    "core.chainlink_log_distance_from_open",
    "core.chainlink_variance_rate_per_second",
    "core.chainlink_receipt_age_ms",
    "core.chainlink_return_count",
    "core.chainlink_coverage_seconds",
    "core.structural_probability_up",
    *(f"core.up_{metric}" for metric in _BOOK_SNAPSHOT_METRICS),
    *(f"core.down_{metric}" for metric in _BOOK_SNAPSHOT_METRICS),
    "core.normalized_market_prior_up",
    "core.structural_minus_market_prior",
    "core.complement_buy_overround",
    "core.complement_sell_underround",
    "core.book_receipt_skew_ms",
)
_TEMPORAL_FEATURE_NAMES = (
    *(
        f"core.chainlink_{metric}_{window}ms"
        for window in POLYMARKET_ROUND21_CORE_WINDOWS_MS
        for metric in _CHAINLINK_METRICS
    ),
    *(
        f"core.{outcome}_{metric}_{window}ms"
        for outcome in ("up", "down")
        for window in POLYMARKET_ROUND21_CORE_WINDOWS_MS
        for metric in _BOOK_FLOW_METRICS
    ),
)
POLYMARKET_ROUND21_CORE_FEATURE_NAMES = (
    *_STATIC_FEATURE_NAMES,
    *_TEMPORAL_FEATURE_NAMES,
)
POLYMARKET_ROUND21_FEATURE_SCHEMA = Round21FeatureSchema.create(
    core_names=POLYMARKET_ROUND21_CORE_FEATURE_NAMES,
    spot_names=POLYMARKET_ROUND21_SPOT_FEATURE_NAMES,
    usdm_names=POLYMARKET_ROUND21_USDM_FEATURE_NAMES,
    feature_policy_sha256=POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9:_-]{1,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_FULL_BOOK_KEYS = frozenset(
    {
        "event_type",
        "asset_id",
        "market",
        "timestamp",
        "hash",
        "bids",
        "asks",
    }
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
        "best_bid_ask",
    }
)
_COMPACT_THRESHOLD = 100_000
_COMPACT_MINIMUM_DROP = 20_000
_MAX_POLICY_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 core payload contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 core payload contains {value}")


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


def validate_round21_feature_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Reject rehashed changes to causal features or predictor independence."""

    policy = dict(value)
    claimed = str(policy.pop("design_sha256", "")).strip().lower()
    parents = policy.get("parents")
    clock = policy.get("clock_and_causality")
    chainlink = policy.get("chainlink_baseline")
    clob = policy.get("polymarket_clob")
    optional = policy.get("optional_binance")
    schema = policy.get("feature_schema")
    anti_leakage = policy.get("anti_leakage")
    authority = policy.get("authority")
    if (
        set(policy)
        != {
            "schema_version",
            "round",
            "status",
            "parents",
            "clock_and_causality",
            "chainlink_baseline",
            "polymarket_clob",
            "optional_binance",
            "feature_schema",
            "anti_leakage",
            "authority",
        }
        or claimed != POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
        or claimed != _canonical_sha256(policy)
        or policy.get("schema_version")
        != POLYMARKET_ROUND21_FEATURE_POLICY_SCHEMA_VERSION
        or policy.get("round") != 21
        or policy.get("status")
        != "preregistered_during_target_and_model_blind_capture"
        or parents
        != {
            "round21_contract_sha256": (
                "6aadbce31c175438c40c6a1204383d828fd78ddef93b280aa2f999f347669116"
            ),
            "round21_dataset_design_sha256": (
                POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
            ),
            "round21_sidecar_design_sha256": (
                "c802b13e169f868c7a37619669cdc957862a1cb58c6d3299c0aae63ff0d86d4a"
            ),
            "round20_campaign_plan_sha256": (
                "2c1d87577de566bd4934c9678bcbded5bf156b671a413b83fa6d463372db1d71"
            ),
        }
        or not isinstance(clock, Mapping)
        or clock.get("inference_clock") != "local_utc_receipt_time"
        or clock.get("source_timestamp_use") != "audit_only"
        or clock.get("future_receipts") != "rejected"
        or clock.get("forward_or_backward_fill") is not False
        or clock.get("cross_connection_gap_carry") is not False
        or not isinstance(chainlink, Mapping)
        or chainlink.get("mid_condition_reconnect") != "condition_ineligible"
        or chainlink.get("probability_bound")
        != "binary64_open_interval_only_no_economic_clipping"
        or not isinstance(clob, Mapping)
        or clob.get("crossed_empty_or_contradictory_book")
        != "invalid_until_fresh_full_book"
        or not isinstance(optional, Mapping)
        or optional.get("role")
        != "independent_credential_free_read_only_predictor"
        or optional.get("credentials") is not False
        or optional.get("account_access") is not False
        or optional.get("orders") is not False
        or optional.get("execution") is not False
        or optional.get("risk_or_stop_dependency") is not False
        or optional.get("optional_rows_may_change_core_admission") is not False
        or optional.get("optional_rows_may_expand_core_population") is not False
        or not isinstance(schema, Mapping)
        or schema
        != {
            "core_width": len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES),
            "spot_width": len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES),
            "usdm_width": len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES),
            "total_width": (
                len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES)
                + len(POLYMARKET_ROUND21_SPOT_FEATURE_NAMES)
                + len(POLYMARKET_ROUND21_USDM_FEATURE_NAMES)
            ),
            "core_names_sha256": POLYMARKET_ROUND21_FEATURE_SCHEMA.core_names_sha256,
            "spot_names_sha256": POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names_sha256,
            "usdm_names_sha256": POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names_sha256,
        }
        or not isinstance(anti_leakage, Mapping)
        or anti_leakage.get("future_books") != "rejected"
        or anti_leakage.get("future_reference_prices") != "rejected"
        or anti_leakage.get("outcomes_or_resolution") != "rejected"
        or anti_leakage.get("fees_fills_orders_or_pnl") != "rejected"
        or anti_leakage.get("test_role_access_during_development")
        != "rejected"
        or anti_leakage.get("feature_availability_may_use_target") is not False
        or authority
        != {
            "model_data_eligible": False,
            "model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 causal feature policy differs")
    return {**policy, "design_sha256": claimed}


def load_round21_feature_policy(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_POLICY_BYTES
    ):
        raise ValueError("Round 21 causal feature policy is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Round 21 causal feature policy is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 causal feature policy is not an object")
    return validate_round21_feature_policy(value)


def _decimal(
    value: object,
    *,
    name: str,
    allow_zero: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {name} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 21 {name} is invalid") from exc
    if (
        not parsed.is_finite()
        or parsed < 0
        or (not allow_zero and parsed == 0)
    ):
        raise ValueError(f"Round 21 {name} is invalid")
    return parsed


def _timestamp(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {name} is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Round 21 {name} is invalid") from exc
    if parsed <= 0 or str(parsed) != str(value):
        raise ValueError(f"Round 21 {name} is invalid")
    return parsed


def _chain(previous: str, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + _canonical_json(payload).encode("ascii")
    ).hexdigest()


def _validate_union_event(event: PolymarketUnionEvent) -> Mapping[str, object]:
    if not isinstance(event, PolymarketUnionEvent):
        raise TypeError("Round 21 CLOB input is not a redundant-union event")
    try:
        decoded = json.loads(
            event.event_json,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 union event is not strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 21 union event is not an object")
    canonical = _canonical_json(decoded)
    receipts = tuple(receipt.validated() for receipt in event.lane_receipts)
    ordered = tuple(
        sorted(
            receipts,
            key=lambda value: (
                value.received_monotonic_ns,
                value.lane_id,
                value.connection_id,
                value.sequence_number,
            ),
        )
    )
    if not ordered:
        raise ValueError("Round 21 union event has no lane receipt")
    selected = ordered[0]
    identity = {
        "schema_version": POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION,
        "union_sequence_number": event.union_sequence_number,
        "semantic_sha256": event.semantic_sha256,
        "semantic_occurrence_index": event.semantic_occurrence_index,
        "event_type": event.event_type,
        "source_time_ms": event.source_time_ms,
        "selected_received_wall_ms": event.selected_received_wall_ms,
        "selected_received_monotonic_ns": event.selected_received_monotonic_ns,
        "selected_lane_id": event.selected_lane_id,
        "lane_receipts": [
            {
                "lane_id": receipt.lane_id,
                "connection_id": receipt.connection_id,
                "sequence_number": receipt.sequence_number,
                "received_wall_ms": receipt.received_wall_ms,
                "received_monotonic_ns": receipt.received_monotonic_ns,
            }
            for receipt in ordered
        ],
    }
    source_time = decoded.get("timestamp")
    parsed_source = (
        int(source_time)
        if type(source_time) is int
        or isinstance(source_time, str)
        and source_time.isdigit()
        else None
    )
    if (
        int(event.union_sequence_number) <= 0
        or int(event.semantic_occurrence_index) <= 0
        or event.event_json != canonical
        or event.semantic_sha256
        != hashlib.sha256(canonical.encode("ascii")).hexdigest()
        or str(decoded.get("event_type") or "unknown") != event.event_type
        or event.source_time_ms != parsed_source
        or event.selected_received_wall_ms != selected.received_wall_ms
        or event.selected_received_monotonic_ns
        != selected.received_monotonic_ns
        or event.selected_lane_id != selected.lane_id
        or receipts != ordered
        or _SHA256.fullmatch(event.event_sha256) is None
        or event.event_sha256 != _canonical_sha256(identity)
    ):
        raise ValueError("Round 21 redundant-union event integrity differs")
    return decoded


def validate_round21_union_event(
    event: PolymarketUnionEvent,
) -> Mapping[str, object]:
    """Validate and decode one exact redundant-union event."""

    return _validate_union_event(event)


@dataclass(frozen=True, slots=True)
class _ChainlinkObservation:
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    price: float
    source_payload_sha256: str
    control: bool


def _parse_chainlink_record(record: CaptureFrameRecord) -> _ChainlinkObservation:
    if not isinstance(record, CaptureFrameRecord):
        raise TypeError("Round 21 Chainlink input is not a capture-frame record")
    connection = str(record.connection_id or "").strip().lower()
    if (
        record.stream != "polymarket_rtds"
        or _CONNECTION_ID.fullmatch(connection) is None
        or int(record.sequence_number) <= 0
        or int(record.received_wall_ms) <= 0
        or int(record.received_monotonic_ns) <= 0
    ):
        raise ValueError("Round 21 Chainlink capture metadata is invalid")
    raw = str(record.raw_text)
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if raw in {"PING", "PONG"}:
        return _ChainlinkObservation(
            connection_id=connection,
            sequence_number=int(record.sequence_number),
            received_wall_ms=int(record.received_wall_ms),
            received_monotonic_ns=int(record.received_monotonic_ns),
            price=0.0,
            source_payload_sha256=raw_sha,
            control=True,
        )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 Chainlink payload is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 21 Chainlink payload is not an object")
    tick = parse_polymarket_chainlink_btc_tick(
        payload,
        received_at_ms=int(record.received_wall_ms),
    )
    return _ChainlinkObservation(
        connection_id=connection,
        sequence_number=int(record.sequence_number),
        received_wall_ms=int(record.received_wall_ms),
        received_monotonic_ns=int(record.received_monotonic_ns),
        price=float(tick.price),
        source_payload_sha256=raw_sha,
        control=False,
    )


@dataclass(frozen=True, slots=True)
class _PriceWindow:
    log_return: float
    realized_variance: float
    bipower_variation: float
    jump_fraction: float
    count: int


class _RollingPriceSeries:
    def __init__(self) -> None:
        self.received_ms: list[int] = []
        self.log_price: list[float] = []
        self.squared_return: list[float] = []
        self.bipower_product: list[float] = []
        self.squared_prefix: list[float] = [0.0]
        self.bipower_prefix: list[float] = [0.0]

    def _rebuild_prefix(self) -> None:
        self.squared_prefix = [0.0]
        self.bipower_prefix = [0.0]
        for squared, bipower in zip(
            self.squared_return,
            self.bipower_product,
            strict=True,
        ):
            self.squared_prefix.append(self.squared_prefix[-1] + squared)
            self.bipower_prefix.append(self.bipower_prefix[-1] + bipower)

    def _compact(self, latest_ms: int) -> None:
        if len(self.received_ms) < _COMPACT_THRESHOLD:
            return
        cutoff = bisect_left(
            self.received_ms,
            latest_ms - POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS - 1_000,
        )
        drop = max(0, cutoff - 2)
        if drop < _COMPACT_MINIMUM_DROP:
            return
        self.received_ms = self.received_ms[drop:]
        self.log_price = self.log_price[drop:]
        returns = [
            0.0,
            *(
                self.log_price[index] - self.log_price[index - 1]
                for index in range(1, len(self.log_price))
            ),
        ]
        self.squared_return = [value * value for value in returns]
        self.bipower_product = [
            0.0,
            *(
                abs(returns[index - 1] * returns[index])
                for index in range(1, len(returns))
            ),
        ]
        self._rebuild_prefix()

    def append(self, received_ms: int, price: float) -> None:
        received = int(received_ms)
        if self.received_ms and received < self.received_ms[-1]:
            raise ValueError("Round 21 Chainlink receipt time regressed")
        log_price = math.log(float(price))
        previous_return = (
            0.0
            if len(self.log_price) < 2
            else self.log_price[-1] - self.log_price[-2]
        )
        current_return = (
            0.0 if not self.log_price else log_price - self.log_price[-1]
        )
        squared = current_return * current_return
        bipower = abs(previous_return * current_return)
        self.received_ms.append(received)
        self.log_price.append(log_price)
        self.squared_return.append(squared)
        self.bipower_product.append(bipower)
        self.squared_prefix.append(self.squared_prefix[-1] + squared)
        self.bipower_prefix.append(self.bipower_prefix[-1] + bipower)
        self._compact(received)

    def window(self, decision_ms: int, window_ms: int) -> _PriceWindow:
        left = bisect_left(self.received_ms, decision_ms - window_ms)
        right = bisect_right(self.received_ms, decision_ms)
        count = right - left
        if count <= 0:
            return _PriceWindow(0.0, 0.0, 0.0, 0.0, 0)
        variance = (
            0.0
            if count < 2
            else self.squared_prefix[right] - self.squared_prefix[left + 1]
        )
        bipower = (
            0.0
            if count < 3
            else math.pi
            / 2.0
            * (self.bipower_prefix[right] - self.bipower_prefix[left + 2])
        )
        tolerance = 1e-15
        if variance < -tolerance or bipower < -tolerance:
            raise RuntimeError("Round 21 Chainlink variance accounting differs")
        variance = max(0.0, variance)
        bipower = max(0.0, bipower)
        return _PriceWindow(
            log_return=(
                0.0
                if count < 2
                else self.log_price[right - 1] - self.log_price[left]
            ),
            realized_variance=variance,
            bipower_variation=bipower,
            jump_fraction=(
                0.0
                if variance <= 0.0
                else max(0.0, (variance - bipower) / variance)
            ),
            count=count,
        )

    def range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> tuple[list[int], list[float]]:
        left = bisect_left(self.received_ms, start_ms)
        right = bisect_right(self.received_ms, end_ms)
        return self.received_ms[left:right], self.log_price[left:right]

    @property
    def latest_received_ms(self) -> int | None:
        return None if not self.received_ms else self.received_ms[-1]


@dataclass(frozen=True, slots=True)
class _BookSnapshot:
    received_wall_ms: int
    received_monotonic_ns: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> tuple[float, float]:
        return self.bids[0]

    @property
    def best_ask(self) -> tuple[float, float]:
        return self.asks[0]

    @property
    def microprice(self) -> float:
        bid, bid_qty = self.best_bid
        ask, ask_qty = self.best_ask
        return (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)


class _BookState:
    def __init__(self, *, token_id: str) -> None:
        self.token_id = token_id
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.valid = False

    @staticmethod
    def _levels(
        values: object,
        *,
        side: str,
    ) -> dict[Decimal, Decimal]:
        if not isinstance(values, list):
            raise ValueError(f"Round 21 {side} book levels are invalid")
        output: dict[Decimal, Decimal] = {}
        for item in values:
            if not isinstance(item, Mapping) or set(item) != {"price", "size"}:
                raise ValueError(f"Round 21 {side} book level schema drifted")
            price = _decimal(item["price"], name=f"{side} price")
            quantity = _decimal(item["size"], name=f"{side} quantity")
            if not Decimal("0") < price < Decimal("1") or price in output:
                raise ValueError(f"Round 21 {side} book level is invalid")
            output[price] = quantity
        return output

    def replace(self, event: Mapping[str, object]) -> None:
        bids = self._levels(event["bids"], side="bid")
        asks = self._levels(event["asks"], side="ask")
        self.bids = bids
        self.asks = asks
        self.valid = self._valid_top()

    def _valid_top(self) -> bool:
        return (
            bool(self.bids)
            and bool(self.asks)
            and max(self.bids) < min(self.asks)
        )

    def apply(self, changes: Sequence[Mapping[str, object]]) -> None:
        if not self.valid:
            return
        for change in changes:
            side = str(change["side"] or "").upper()
            price = _decimal(change["price"], name="price-change price")
            quantity = _decimal(
                change["size"],
                name="price-change quantity",
                allow_zero=True,
            )
            if not Decimal("0") < price < Decimal("1") or side not in {"BUY", "SELL"}:
                raise ValueError("Round 21 price-change level is invalid")
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
                name="reported best bid",
                allow_zero=True,
            )
            reported_ask = _decimal(
                change["best_ask"],
                name="reported best ask",
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
        *,
        received_wall_ms: int,
        received_monotonic_ns: int,
    ) -> _BookSnapshot | None:
        if not self.valid:
            return None
        bids = tuple(
            (float(price), float(self.bids[price]))
            for price in sorted(self.bids, reverse=True)[:20]
        )
        asks = tuple(
            (float(price), float(self.asks[price]))
            for price in sorted(self.asks)[:20]
        )
        if not bids or not asks or bids[0][0] >= asks[0][0]:
            return None
        return _BookSnapshot(
            received_wall_ms=int(received_wall_ms),
            received_monotonic_ns=int(received_monotonic_ns),
            bids=bids,
            asks=asks,
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
        self.latest: _BookSnapshot | None = None

    @staticmethod
    def _top_flow(
        previous: _BookSnapshot,
        current: _BookSnapshot,
    ) -> tuple[float, float]:
        prior_bid, prior_bid_qty = previous.best_bid
        bid, bid_qty = current.best_bid
        prior_ask, prior_ask_qty = previous.best_ask
        ask, ask_qty = current.best_ask
        bid_flow = (
            bid_qty
            if bid > prior_bid
            else (
                bid_qty - prior_bid_qty
                if bid == prior_bid
                else -prior_bid_qty
            )
        )
        ask_flow = (
            ask_qty
            if ask < prior_ask
            else (
                ask_qty - prior_ask_qty
                if ask == prior_ask
                else -prior_ask_qty
            )
        )
        signed = bid_flow - ask_flow
        gross = abs(bid_flow) + abs(ask_flow)
        return signed, gross

    @staticmethod
    def _level_flow(
        previous: _BookSnapshot,
        current: _BookSnapshot,
    ) -> tuple[float, float]:
        prior_bids = dict(previous.bids)
        current_bids = dict(current.bids)
        prior_asks = dict(previous.asks)
        current_asks = dict(current.asks)
        bid_changes = (
            current_bids.get(price, 0.0) - prior_bids.get(price, 0.0)
            for price in set(prior_bids) | set(current_bids)
        )
        ask_changes = (
            current_asks.get(price, 0.0) - prior_asks.get(price, 0.0)
            for price in set(prior_asks) | set(current_asks)
        )
        bid_values = tuple(bid_changes)
        ask_values = tuple(ask_changes)
        signed = math.fsum(bid_values) - math.fsum(ask_values)
        gross = math.fsum(abs(value) for value in (*bid_values, *ask_values))
        return signed, gross

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
                self.prefixes[index].append(
                    self.prefixes[index][-1] + value
                )

    def _compact(self, latest_ms: int) -> None:
        if len(self.received_ms) < _COMPACT_THRESHOLD:
            return
        cutoff = bisect_left(
            self.received_ms,
            latest_ms - POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS - 1_000,
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

    def append(self, snapshot: _BookSnapshot) -> None:
        if self.received_ms and snapshot.received_wall_ms < self.received_ms[-1]:
            raise ValueError("Round 21 CLOB receipt time regressed")
        bid, bid_qty = snapshot.best_bid
        ask, ask_qty = snapshot.best_ask
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        if self.latest is None:
            top_signed = top_gross = level_signed = level_gross = 0.0
        else:
            top_signed, top_gross = self._top_flow(self.latest, snapshot)
            level_signed, level_gross = self._level_flow(self.latest, snapshot)
        self.received_ms.append(snapshot.received_wall_ms)
        self.top_ofi.append(top_signed)
        self.top_gross.append(top_gross)
        self.top_imbalance.append(imbalance)
        self.log_microprice.append(math.log(snapshot.microprice))
        self.level_pressure.append(level_signed)
        self.level_gross.append(level_gross)
        for prefix, value in zip(
            self.prefixes,
            (
                top_signed,
                top_gross,
                imbalance,
                level_signed,
                level_gross,
            ),
            strict=True,
        ):
            prefix.append(prefix[-1] + value)
        self.latest = snapshot
        self._compact(snapshot.received_wall_ms)

    def window(self, decision_ms: int, window_ms: int) -> _BookFlowWindow:
        left = bisect_left(self.received_ms, decision_ms - window_ms)
        right = bisect_right(self.received_ms, decision_ms)
        count = right - left
        if count <= 0:
            return _BookFlowWindow(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        top_signed, top_gross, imbalance, level_signed, level_gross = (
            prefix[right] - prefix[left] for prefix in self.prefixes
        )
        gross_scale = max(1.0, top_gross, level_gross)
        gross_tolerance = 1e-12 * gross_scale
        imbalance_tolerance = 1e-12 * max(1, count)
        if (
            top_gross < -gross_tolerance
            or level_gross < -gross_tolerance
            or abs(top_signed) > top_gross + gross_tolerance
            or abs(level_signed) > level_gross + gross_tolerance
            or abs(imbalance) > count + imbalance_tolerance
        ):
            raise RuntimeError("Round 21 CLOB flow accounting differs")
        top_gross = max(0.0, top_gross)
        level_gross = max(0.0, level_gross)
        if abs(top_signed) > top_gross:
            top_signed = math.copysign(top_gross, top_signed)
        if abs(level_signed) > level_gross:
            level_signed = math.copysign(level_gross, level_signed)
        if abs(imbalance) > count:
            imbalance = math.copysign(float(count), imbalance)
        return _BookFlowWindow(
            top_ofi=0.0 if top_gross == 0.0 else top_signed / top_gross,
            mean_imbalance=imbalance / count,
            microprice_return=(
                0.0
                if count < 2
                else self.log_microprice[right - 1]
                - self.log_microprice[left]
            ),
            level_pressure=(
                0.0 if level_gross == 0.0 else level_signed / level_gross
            ),
            gross_level_change=level_gross,
            count=count,
        )


def _snapshot_values(snapshot: _BookSnapshot) -> tuple[float, ...]:
    bid, bid_qty = snapshot.best_bid
    ask, ask_qty = snapshot.best_ask
    midpoint = 0.5 * (bid + ask)
    spread = ask - bid

    def depth(levels: tuple[tuple[float, float], ...], count: int) -> float:
        return math.fsum(quantity for _price, quantity in levels[:count])

    bid_last = snapshot.bids[min(19, len(snapshot.bids) - 1)][0]
    ask_last = snapshot.asks[min(19, len(snapshot.asks) - 1)][0]
    return (
        bid,
        ask,
        midpoint,
        spread,
        spread / midpoint,
        snapshot.microprice,
        (bid_qty - ask_qty) / (bid_qty + ask_qty),
        bid_qty,
        ask_qty,
        depth(snapshot.bids, 5),
        depth(snapshot.asks, 5),
        depth(snapshot.bids, 20),
        depth(snapshot.asks, 20),
        (bid - bid_last) / bid * 10_000.0,
        (ask_last - ask) / ask * 10_000.0,
    )


@dataclass(frozen=True, slots=True)
class Round21CoreFeatureSnapshot:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    available: bool
    reasons: tuple[str, ...]
    structural_probability: float
    market_prior_probability: float
    values: tuple[float, ...]
    source_chain_sha256: str
    maximum_receipt_ms: int
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms <= 0
            or self.decision_time_ms <= 0
            or type(self.available) is not bool
            or len(self.values) != len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or _SHA256.fullmatch(self.source_chain_sha256) is None
            or self.trading_authority
        ):
            raise ValueError("Round 21 core feature snapshot is invalid")
        if self.available:
            if (
                self.reasons
                or not 0.0 < self.structural_probability < 1.0
                or not 0.0 < self.market_prior_probability < 1.0
                or self.source_chain_sha256 == _EMPTY_SHA256
                or not 0 < self.maximum_receipt_ms <= self.decision_time_ms
            ):
                raise ValueError("Round 21 available core snapshot differs")
        elif (
            not self.reasons
            or self.structural_probability != 0.5
            or self.market_prior_probability != 0.5
            or any(self.values)
            or self.source_chain_sha256 != _EMPTY_SHA256
            or self.maximum_receipt_ms != 0
        ):
            raise ValueError("Round 21 unavailable core snapshot differs")


class Round21CoreFeatureEngine:
    """Build target-free core rows from one admitted condition segment."""

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
        condition = str(condition_id or "").strip().lower()
        up = str(up_token_id or "").strip()
        down = str(down_token_id or "").strip()
        start = int(event_start_ms)
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or _TOKEN_ID.fullmatch(up) is None
            or _TOKEN_ID.fullmatch(down) is None
            or up == down
            or start <= 0
            or start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
        ):
            raise ValueError("Round 21 core condition identity is invalid")
        self.condition_id = condition
        self.up_token_id = up
        self.down_token_id = down
        self.event_start_ms = start
        self.event_end_ms = start + POLYMARKET_ROUND21_CONDITION_DURATION_MS
        self._chainlink_connection_id = ""
        self._chainlink_sequence = 0
        self._chainlink_monotonic_ns = 0
        self._chainlink_wall_ms = 0
        self._chainlink_chain = _EMPTY_SHA256
        self._chainlink = _RollingPriceSeries()
        self._chainlink_gap_detected = False
        self._book_state = {
            up: _BookState(token_id=up),
            down: _BookState(token_id=down),
        }
        self._book_series = {
            up: _RollingBookSeries(),
            down: _RollingBookSeries(),
        }
        self._clob_chain = _EMPTY_SHA256
        self._last_union_monotonic_ns = 0

    def start_chainlink_epoch(
        self,
        connection_id: str,
        *,
        first_sequence_number: int = 1,
    ) -> None:
        connection = str(connection_id or "").strip().lower()
        first_sequence = int(first_sequence_number)
        if (
            _CONNECTION_ID.fullmatch(connection) is None
            or first_sequence <= 0
        ):
            raise ValueError("Round 21 Chainlink epoch identity is invalid")
        if self._chainlink_sequence:
            self._chainlink_gap_detected = True
        self._chainlink_connection_id = connection
        self._chainlink_sequence = first_sequence - 1
        self._chainlink_monotonic_ns = 0
        self._chainlink_wall_ms = 0
        self._chainlink_chain = _EMPTY_SHA256
        self._chainlink = _RollingPriceSeries()

    def ingest_chainlink_record(self, record: CaptureFrameRecord) -> None:
        observation = _parse_chainlink_record(record)
        if not self._chainlink_connection_id:
            self.start_chainlink_epoch(observation.connection_id)
        if (
            observation.connection_id != self._chainlink_connection_id
            or observation.sequence_number != self._chainlink_sequence + 1
            or observation.received_monotonic_ns
            <= self._chainlink_monotonic_ns
            or observation.received_wall_ms < self._chainlink_wall_ms
        ):
            raise ValueError("Round 21 Chainlink reconnect or chronology differs")
        self._chainlink_chain = _chain(
            self._chainlink_chain,
            {
                "connection_id": observation.connection_id,
                "sequence_number": observation.sequence_number,
                "received_wall_ms": observation.received_wall_ms,
                "received_monotonic_ns": observation.received_monotonic_ns,
                "source_payload_sha256": observation.source_payload_sha256,
                "control": observation.control,
            },
        )
        if not observation.control:
            self._chainlink.append(
                observation.received_wall_ms,
                observation.price,
            )
        self._chainlink_sequence = observation.sequence_number
        self._chainlink_monotonic_ns = observation.received_monotonic_ns
        self._chainlink_wall_ms = observation.received_wall_ms

    def ingest_union_event(self, event: PolymarketUnionEvent) -> None:
        payload = _validate_union_event(event)
        market = str(payload.get("market") or "").strip().lower()
        if market != self.condition_id:
            return
        if event.selected_received_monotonic_ns < self._last_union_monotonic_ns:
            raise ValueError("Round 21 union receipt order regressed")
        self._last_union_monotonic_ns = event.selected_received_monotonic_ns
        self._clob_chain = _chain(
            self._clob_chain,
            {
                "event_sha256": event.event_sha256,
                "selected_received_wall_ms": event.selected_received_wall_ms,
                "selected_received_monotonic_ns": (
                    event.selected_received_monotonic_ns
                ),
            },
        )
        event_type = event.event_type
        updated_tokens: set[str] = set()
        if event_type == "book":
            keys = frozenset(payload)
            if (
                not _FULL_BOOK_KEYS.issubset(keys)
                or keys - _FULL_BOOK_KEYS - _FULL_BOOK_OPTIONAL_KEYS
            ):
                raise ValueError("Round 21 full-book schema drifted")
            token = str(payload["asset_id"] or "")
            state = self._book_state.get(token)
            if state is None:
                raise ValueError("Round 21 full book has an unknown token")
            state.replace(payload)
            updated_tokens.add(token)
        elif event_type == "price_change":
            if set(payload) != _PRICE_CHANGE_KEYS:
                raise ValueError("Round 21 price-change schema drifted")
            raw_changes = payload["price_changes"]
            if not isinstance(raw_changes, list) or not raw_changes:
                raise ValueError("Round 21 price-change batch is invalid")
            grouped: dict[str, list[Mapping[str, object]]] = {}
            for change in raw_changes:
                if (
                    not isinstance(change, Mapping)
                    or set(change) != _PRICE_CHANGE_ITEM_KEYS
                ):
                    raise ValueError("Round 21 price-change item schema drifted")
                token = str(change["asset_id"] or "")
                if token not in self._book_state:
                    raise ValueError("Round 21 price change has an unknown token")
                grouped.setdefault(token, []).append(change)
            for token, changes in grouped.items():
                self._book_state[token].apply(changes)
                updated_tokens.add(token)
        elif event_type == "best_bid_ask":
            if set(payload) != _BEST_BID_ASK_KEYS:
                raise ValueError("Round 21 best-bid-ask schema drifted")
        elif event_type not in _IGNORED_EVENT_TYPES:
            raise ValueError(f"Round 21 unsupported CLOB event type: {event_type}")
        for token in updated_tokens:
            snapshot = self._book_state[token].snapshot(
                received_wall_ms=event.selected_received_wall_ms,
                received_monotonic_ns=event.selected_received_monotonic_ns,
            )
            if snapshot is not None:
                self._book_series[token].append(snapshot)

    def _unavailable(
        self,
        *,
        decision_ms: int,
        reasons: Sequence[str],
    ) -> Round21CoreFeatureSnapshot:
        return Round21CoreFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision_ms,
            available=False,
            reasons=tuple(dict.fromkeys(reasons)),
            structural_probability=0.5,
            market_prior_probability=0.5,
            values=(0.0,) * len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES),
            source_chain_sha256=_EMPTY_SHA256,
            maximum_receipt_ms=0,
        )

    def build(self, decision_time_ms: int) -> Round21CoreFeatureSnapshot:
        decision = int(decision_time_ms)
        if (
            not self.event_start_ms <= decision < self.event_end_ms
            or (decision - self.event_start_ms) % 250
        ):
            raise ValueError("Round 21 core decision time is invalid")
        if (
            self._chainlink_wall_ms > decision
            or any(
                series.latest is not None
                and series.latest.received_wall_ms > decision
                for series in self._book_series.values()
            )
        ):
            raise ValueError("Round 21 core engine contains future receipts")
        reasons: list[str] = []
        if self._chainlink_gap_detected:
            reasons.append("chainlink_connection_gap")
        latest_chainlink = self._chainlink.latest_received_ms
        if latest_chainlink is None:
            reasons.append("chainlink_unavailable")
        elif decision - latest_chainlink > (
            POLYMARKET_ROUND21_CORE_MAXIMUM_SOURCE_AGE_MS
        ):
            reasons.append("chainlink_stale")
        receipt_times, log_prices = self._chainlink.range(
            self.event_start_ms,
            decision,
        )
        if len(receipt_times) - 1 < (
            POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_RETURNS
        ):
            reasons.append("chainlink_return_count_below_minimum")
        coverage_ms = (
            0
            if len(receipt_times) < 2
            else receipt_times[-1] - receipt_times[0]
        )
        if coverage_ms < POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_COVERAGE_MS:
            reasons.append("chainlink_coverage_below_minimum")
        book_snapshots: dict[str, _BookSnapshot] = {}
        for token, name in (
            (self.up_token_id, "up"),
            (self.down_token_id, "down"),
        ):
            snapshot = self._book_series[token].latest
            if not self._book_state[token].valid:
                reasons.append(f"{name}_book_invalid")
            elif snapshot is None:
                reasons.append(f"{name}_book_unavailable")
            elif (
                decision - snapshot.received_wall_ms
                > POLYMARKET_ROUND21_CORE_MAXIMUM_SOURCE_AGE_MS
            ):
                reasons.append(f"{name}_book_stale")
            else:
                book_snapshots[name] = snapshot
        if reasons:
            return self._unavailable(decision_ms=decision, reasons=reasons)
        if len(log_prices) < 2 or set(book_snapshots) != {"up", "down"}:
            raise RuntimeError("Round 21 core availability reconciliation failed")
        squared = math.fsum(
            (current - previous) ** 2
            for previous, current in zip(log_prices, log_prices[1:], strict=False)
        )
        elapsed_seconds = coverage_ms / 1_000.0
        variance_rate = squared / elapsed_seconds
        if not math.isfinite(variance_rate) or variance_rate > 1e-4:
            return self._unavailable(
                decision_ms=decision,
                reasons=("chainlink_variance_outside_safety_bound",),
            )
        variance_rate = max(variance_rate, 1e-12)
        log_distance = log_prices[-1] - log_prices[0]
        remaining_seconds = (self.event_end_ms - decision) / 1_000.0
        scale = math.sqrt(variance_rate * remaining_seconds)
        z_score = log_distance / scale
        structural = min(
            math.nextafter(1.0, 0.0),
            max(
                math.nextafter(0.0, 1.0),
                0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))),
            ),
        )
        up_values = _snapshot_values(book_snapshots["up"])
        down_values = _snapshot_values(book_snapshots["down"])
        up_mid = up_values[2]
        down_mid = down_values[2]
        market_prior = up_mid / (up_mid + down_mid)
        static = (
            (decision - self.event_start_ms)
            / POLYMARKET_ROUND21_CONDITION_DURATION_MS,
            remaining_seconds,
            log_distance,
            variance_rate,
            float(decision - int(latest_chainlink)),
            float(len(log_prices) - 1),
            coverage_ms / 1_000.0,
            structural,
            *up_values,
            *down_values,
            market_prior,
            structural - market_prior,
            (up_values[1] + down_values[1]) - 1.0,
            1.0 - (up_values[0] + down_values[0]),
            float(
                abs(
                    book_snapshots["up"].received_wall_ms
                    - book_snapshots["down"].received_wall_ms
                )
            ),
        )
        temporal: list[float] = []
        for window_ms in POLYMARKET_ROUND21_CORE_WINDOWS_MS:
            item = self._chainlink.window(decision, window_ms)
            temporal.extend(
                (
                    item.log_return,
                    item.realized_variance,
                    item.bipower_variation,
                    item.jump_fraction,
                    float(item.count),
                )
            )
        for token in (self.up_token_id, self.down_token_id):
            for window_ms in POLYMARKET_ROUND21_CORE_WINDOWS_MS:
                item = self._book_series[token].window(decision, window_ms)
                temporal.extend(
                    (
                        item.top_ofi,
                        item.mean_imbalance,
                        item.microprice_return,
                        item.level_pressure,
                        math.log1p(item.gross_level_change),
                        float(item.count),
                    )
                )
        values = tuple((*static, *temporal))
        if len(values) != len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES):
            raise RuntimeError("Round 21 core feature width differs")
        core_chain = _canonical_sha256(
            {
                "condition_id": self.condition_id,
                "decision_time_ms": decision,
                "feature_policy_sha256": (
                    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
                ),
                "chainlink_source_chain_sha256": self._chainlink_chain,
                "clob_source_chain_sha256": self._clob_chain,
            }
        )
        maximum_receipt = max(
            int(latest_chainlink),
            book_snapshots["up"].received_wall_ms,
            book_snapshots["down"].received_wall_ms,
        )
        return Round21CoreFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision,
            available=True,
            reasons=(),
            structural_probability=structural,
            market_prior_probability=market_prior,
            values=values,
            source_chain_sha256=core_chain,
            maximum_receipt_ms=maximum_receipt,
        )


def join_round21_causal_features(
    core: Round21CoreFeatureSnapshot,
    optional: Round21OptionalBinanceFeatures,
) -> Round21CausalFeatureRow:
    """Join independent predictor values only after the core row is eligible."""

    if not isinstance(core, Round21CoreFeatureSnapshot) or not core.available:
        raise ValueError("Round 21 core feature row is unavailable")
    if not isinstance(optional, Round21OptionalBinanceFeatures):
        raise TypeError("Round 21 optional feature type differs")
    if optional.decision_time_ms != core.decision_time_ms:
        raise ValueError("Round 21 optional decision identity differs")
    return Round21CausalFeatureRow.create(
        condition_id=core.condition_id,
        event_start_ms=core.event_start_ms,
        decision_time_ms=core.decision_time_ms,
        structural_probability=core.structural_probability,
        market_prior_probability=core.market_prior_probability,
        core_values=core.values,
        spot_values=optional.spot_values,
        usdm_values=optional.usdm_values,
        spot_available=optional.spot_available,
        usdm_available=optional.usdm_available,
        feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
        core_source_chain_sha256=core.source_chain_sha256,
        spot_source_chain_sha256=optional.spot_source_chain_sha256,
        usdm_source_chain_sha256=optional.usdm_source_chain_sha256,
        core_maximum_receipt_ms=core.maximum_receipt_ms,
        spot_maximum_receipt_ms=optional.spot_maximum_receipt_ms,
        usdm_maximum_receipt_ms=optional.usdm_maximum_receipt_ms,
    )


__all__ = [
    "POLYMARKET_ROUND21_CORE_FEATURE_NAMES",
    "POLYMARKET_ROUND21_CORE_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_CORE_MAXIMUM_SOURCE_AGE_MS",
    "POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_COVERAGE_MS",
    "POLYMARKET_ROUND21_CORE_MINIMUM_CHAINLINK_RETURNS",
    "POLYMARKET_ROUND21_CORE_WINDOWS_MS",
    "POLYMARKET_ROUND21_FEATURE_SCHEMA",
    "POLYMARKET_ROUND21_FEATURE_POLICY_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_FEATURE_POLICY_SHA256",
    "Round21CoreFeatureEngine",
    "Round21CoreFeatureSnapshot",
    "join_round21_causal_features",
    "load_round21_feature_policy",
    "validate_round21_union_event",
    "validate_round21_feature_policy",
]

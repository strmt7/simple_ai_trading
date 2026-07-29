"""Target-free causal snapshot features for the Round 14 BTC 5m models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from .paper_execution import PaperBookSnapshot
from .polymarket_btc_reference import (
    PolymarketBtcEndpointEstimate,
    PolymarketBtcReferenceWindow,
    PolymarketChainlinkBtcTick,
)
from .polymarket_external_signal import PolymarketBtcReferenceFeatures


POLYMARKET_ROUND14_FEATURE_SCHEMA_VERSION = "polymarket-round14-causal-snapshot-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_BOOK_FEATURE_SUFFIXES = (
    "best_bid",
    "best_ask",
    "midpoint",
    "spread",
    "spread_bps",
    "microprice",
    "top_imbalance",
    "bid_depth_1",
    "ask_depth_1",
    "bid_depth_5",
    "ask_depth_5",
    "bid_depth_20",
    "ask_depth_20",
    "bid_slope_20_bps",
    "ask_slope_20_bps",
)
POLYMARKET_ROUND14_FEATURE_NAMES = (
    "elapsed_fraction",
    "remaining_seconds",
    "chainlink_log_distance_from_open",
    "chainlink_variance_rate_per_second",
    "chainlink_source_age_ms",
    "structural_probability_up",
    *(f"up_{name}" for name in _BOOK_FEATURE_SUFFIXES),
    *(f"down_{name}" for name in _BOOK_FEATURE_SUFFIXES),
    "normalized_market_prior_up",
    "structural_minus_market_prior",
    "ask_pair_cost",
    "bid_pair_value",
    "executable_parity_gap",
    "book_receipt_skew_ms",
    "external_available",
    "binance_spot_spread_bps",
    "binance_futures_spread_bps",
    "binance_futures_basis_bps",
    "binance_spot_log_return",
    "binance_futures_log_return",
    "binance_event_time_skew_ms",
    "binance_receive_time_skew_ms",
)
POLYMARKET_ROUND14_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        POLYMARKET_ROUND14_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _finite(value: object, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _book_features(book: PaperBookSnapshot) -> tuple[float, ...]:
    bids = book.bids
    asks = book.asks
    if not bids or not asks:
        raise ValueError("Round 14 CLOB snapshot requires both sides")
    best_bid = _finite(bids[0].price, name="best bid")
    best_ask = _finite(asks[0].price, name="best ask")
    bid_quantity = _finite(bids[0].quantity, name="best bid quantity")
    ask_quantity = _finite(asks[0].quantity, name="best ask quantity")
    if not 0 < best_bid < best_ask < 1 or bid_quantity <= 0 or ask_quantity <= 0:
        raise ValueError("Round 14 CLOB top of book is invalid")
    midpoint = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    microprice = (best_ask * bid_quantity + best_bid * ask_quantity) / (
        bid_quantity + ask_quantity
    )
    imbalance = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)

    def depth(levels: tuple[object, ...], count: int) -> float:
        return math.fsum(float(level.quantity) for level in levels[:count])

    bid_last = float(bids[min(19, len(bids) - 1)].price)
    ask_last = float(asks[min(19, len(asks) - 1)].price)
    return (
        best_bid,
        best_ask,
        midpoint,
        spread,
        spread / midpoint * 10_000,
        microprice,
        imbalance,
        bid_quantity,
        ask_quantity,
        depth(bids, 5),
        depth(asks, 5),
        depth(bids, 20),
        depth(asks, 20),
        (best_bid - bid_last) / best_bid * 10_000,
        (ask_last - best_ask) / best_ask * 10_000,
    )


@dataclass(frozen=True, slots=True)
class PolymarketRound14Snapshot:
    condition_id: str
    up_token_id: str
    down_token_id: str
    event_start_ms: int
    event_end_ms: int
    decision_time_ms: int
    reference: PolymarketBtcReferenceWindow
    chainlink_tick: PolymarketChainlinkBtcTick
    structural_estimate: PolymarketBtcEndpointEstimate
    up_book: PaperBookSnapshot
    down_book: PaperBookSnapshot
    external_features: PolymarketBtcReferenceFeatures | None = None
    maximum_source_age_ms: int = 1_500

    def __post_init__(self) -> None:
        condition = str(self.condition_id or "").strip().lower()
        up_token = str(self.up_token_id or "").strip()
        down_token = str(self.down_token_id or "").strip()
        if _CONDITION_ID.fullmatch(condition) is None:
            raise ValueError("Round 14 condition ID is invalid")
        if (
            _TOKEN_ID.fullmatch(up_token) is None
            or _TOKEN_ID.fullmatch(down_token) is None
            or up_token == down_token
        ):
            raise ValueError("Round 14 outcome token mapping is invalid")
        object.__setattr__(self, "condition_id", condition)
        object.__setattr__(self, "up_token_id", up_token)
        object.__setattr__(self, "down_token_id", down_token)
        start = int(self.event_start_ms)
        end = int(self.event_end_ms)
        decision = int(self.decision_time_ms)
        maximum_age = int(self.maximum_source_age_ms)
        if (
            start <= 0
            or start % 300_000
            or end - start != 300_000
            or not start <= decision < end
            or not 100 <= maximum_age <= 5_000
        ):
            raise ValueError("Round 14 snapshot chronology is invalid")
        object.__setattr__(self, "event_start_ms", start)
        object.__setattr__(self, "event_end_ms", end)
        object.__setattr__(self, "decision_time_ms", decision)
        object.__setattr__(self, "maximum_source_age_ms", maximum_age)


@dataclass(frozen=True, slots=True)
class PolymarketRound14FeatureRow:
    condition_id: str
    decision_time_ms: int
    feature_names_sha256: str
    input_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if _CONDITION_ID.fullmatch(self.condition_id) is None:
            raise ValueError("Round 14 feature condition ID is invalid")
        if int(self.decision_time_ms) <= 0:
            raise ValueError("Round 14 feature decision time is invalid")
        for name, digest in (
            ("feature names", self.feature_names_sha256),
            ("input", self.input_sha256),
        ):
            if _SHA256.fullmatch(str(digest or "").strip().lower()) is None:
                raise ValueError(f"Round 14 {name} hash is invalid")
        if len(self.values) != len(POLYMARKET_ROUND14_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in self.values
        ):
            raise ValueError("Round 14 feature vector is invalid")

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND14_FEATURE_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "feature_names_sha256": self.feature_names_sha256,
            "input_sha256": self.input_sha256,
            "values": list(self.values),
        }

    def value_map(self) -> dict[str, float]:
        return dict(zip(POLYMARKET_ROUND14_FEATURE_NAMES, self.values, strict=True))


def build_round14_snapshot_features(
    snapshot: PolymarketRound14Snapshot,
) -> PolymarketRound14FeatureRow:
    """Build one label-free feature row from data known by the decision time."""

    if not isinstance(snapshot, PolymarketRound14Snapshot):
        raise TypeError("snapshot must be PolymarketRound14Snapshot")
    reference = snapshot.reference
    tick = snapshot.chainlink_tick
    estimate = snapshot.structural_estimate
    if (
        reference.event_start_ms != snapshot.event_start_ms
        or reference.end_ms != snapshot.event_end_ms
        or reference.completed
        or reference.close_price is not None
        or reference.observed_at_ms > snapshot.decision_time_ms
    ):
        raise ValueError(
            "Round 14 feature input contains a target, future, or wrong reference"
        )
    if (
        tick.source_time_ms < snapshot.event_start_ms
        or tick.source_time_ms > snapshot.decision_time_ms + 250
        or tick.received_at_ms > snapshot.decision_time_ms
        or snapshot.decision_time_ms - tick.received_at_ms
        > snapshot.maximum_source_age_ms
    ):
        raise ValueError("Round 14 Chainlink input is stale, future, or out of scope")
    if (
        not estimate.available
        or estimate.probability_up is None
        or estimate.variance_rate_per_second is None
        or estimate.log_distance_from_open is None
    ):
        raise ValueError("Round 14 structural endpoint estimate is unavailable")
    expected_remaining = (snapshot.event_end_ms - snapshot.decision_time_ms) / 1_000
    if abs(estimate.remaining_seconds - expected_remaining) > 0.250001:
        raise ValueError("Round 14 structural estimate time differs")
    books = (
        (snapshot.up_book.validated(), snapshot.up_token_id),
        (snapshot.down_book.validated(), snapshot.down_token_id),
    )
    for book, token_id in books:
        if (
            book.venue != "polymarket"
            or book.market_id.lower() != snapshot.condition_id
            or book.asset_id != token_id
            or not book.connected
            or not book.gap_free
            or book.received_wall_ms > snapshot.decision_time_ms
            or snapshot.decision_time_ms - book.received_wall_ms
            > snapshot.maximum_source_age_ms
            or book.source_time_ms > snapshot.decision_time_ms + 250
        ):
            raise ValueError("Round 14 CLOB input identity or chronology differs")
    up_book, down_book = books[0][0], books[1][0]
    up_values = _book_features(up_book)
    down_values = _book_features(down_book)
    up_midpoint = up_values[2]
    down_midpoint = down_values[2]
    midpoint_sum = up_midpoint + down_midpoint
    if not 0 < midpoint_sum < 2:
        raise ValueError("Round 14 market prior cannot be normalized")
    market_prior_up = up_midpoint / midpoint_sum
    structural_probability = _finite(
        estimate.probability_up,
        name="structural probability",
    )
    if not 0 < structural_probability < 1:
        raise ValueError("Round 14 structural probability is invalid")
    external = snapshot.external_features
    if external is None:
        external_values = (0.0,) * 8
        external_hash_payload: object = None
    else:
        if (
            external.observed_at_ms > snapshot.decision_time_ms
            or snapshot.decision_time_ms - external.observed_at_ms
            > snapshot.maximum_source_age_ms
        ):
            raise ValueError("Round 14 external features are stale or future")
        external_values = (
            1.0,
            _finite(external.spot_spread_bps, name="spot spread"),
            _finite(external.futures_spread_bps, name="futures spread"),
            _finite(external.futures_basis_bps, name="futures basis"),
            _finite(external.spot_log_return, name="spot log return"),
            _finite(external.futures_log_return, name="futures log return"),
            float(external.event_time_skew_ms),
            float(external.receive_time_skew_ms),
        )
        external_hash_payload = {
            "observed_at_ms": external.observed_at_ms,
            "spot_mid": format(external.spot_mid, "f"),
            "futures_mid": format(external.futures_mid, "f"),
            "values": list(external_values),
        }
    elapsed = snapshot.decision_time_ms - snapshot.event_start_ms
    values = (
        elapsed / 300_000,
        expected_remaining,
        _finite(estimate.log_distance_from_open, name="Chainlink distance"),
        _finite(
            estimate.variance_rate_per_second,
            name="Chainlink variance",
        ),
        float(snapshot.decision_time_ms - tick.source_time_ms),
        structural_probability,
        *up_values,
        *down_values,
        market_prior_up,
        structural_probability - market_prior_up,
        up_values[1] + down_values[1],
        up_values[0] + down_values[0],
        1.0 - (up_values[1] + down_values[1]),
        float(abs(up_book.received_wall_ms - down_book.received_wall_ms)),
        *external_values,
    )
    input_payload = {
        "schema_version": POLYMARKET_ROUND14_FEATURE_SCHEMA_VERSION,
        "condition_id": snapshot.condition_id,
        "decision_time_ms": snapshot.decision_time_ms,
        "reference_payload_sha256": reference.source_payload_sha256,
        "chainlink_payload_sha256": tick.source_payload_sha256,
        "up_book_payload_sha256": up_book.source_payload_sha256,
        "down_book_payload_sha256": down_book.source_payload_sha256,
        "external": external_hash_payload,
        "feature_names_sha256": POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
        "values": list(values),
    }
    return PolymarketRound14FeatureRow(
        condition_id=snapshot.condition_id,
        decision_time_ms=snapshot.decision_time_ms,
        feature_names_sha256=POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
        input_sha256=_canonical_sha256(input_payload),
        values=tuple(values),
    )


__all__ = [
    "POLYMARKET_ROUND14_FEATURE_NAMES",
    "POLYMARKET_ROUND14_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND14_FEATURE_SCHEMA_VERSION",
    "PolymarketRound14FeatureRow",
    "PolymarketRound14Snapshot",
    "build_round14_snapshot_features",
]

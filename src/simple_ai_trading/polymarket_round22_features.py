"""Target-blind causal full-book features for the Polymarket Round 22 pilot."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .polymarket_historical_l2 import (
    HistoricalBookLevel,
    HistoricalBookSnapshot,
    HistoricalL2Window,
)
from .polymarket_round22_pilot import (
    POLYMARKET_ROUND22_PILOT_DESIGN_SHA256,
    POLYMARKET_ROUND22_SOURCE_QUALIFICATION_SHA256,
)


POLYMARKET_ROUND22_FEATURE_POLICY_SHA256 = (
    "9de2a470a7ef64dc802e5fc4df88b8acfd20ad929a8afb5fd2704a1b94e4725e"
)
POLYMARKET_ROUND22_FEATURE_SCHEMA_VERSION = (
    "polymarket-round22-causal-feature-schema-v1"
)
POLYMARKET_ROUND22_FEATURE_CADENCE_MS = 250
POLYMARKET_ROUND22_TABULAR_CADENCE_MS = 1_000
POLYMARKET_ROUND22_MAXIMUM_BOOK_AGE_MS = 1_000
POLYMARKET_ROUND22_SEQUENCE_LENGTH = 16
POLYMARKET_ROUND22_PRIOR_WINDOWS_MS = (1_000, 5_000, 15_000)
POLYMARKET_ROUND22_IMPACT_NOTIONALS = (5.0, 25.0, 100.0)

_POLICY_RELATIVE = (
    "docs/model-research/polymarket/round-022-causal-feature-policy-v1.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAXIMUM_POLICY_BYTES = 512 * 1024
_OUTCOME_STATIC_NAMES = (
    "best_bid",
    "best_ask",
    "midpoint",
    "spread",
    "microprice",
    "top_bid_size",
    "top_ask_size",
    "bid_depth_top20",
    "ask_depth_top20",
    "bid_level_count_top20",
    "ask_level_count_top20",
    "depth_imbalance_l1",
    "depth_imbalance_l5",
    "depth_imbalance_l10",
    "depth_imbalance_l20",
    "bid_log_depth_slope_top20",
    "bid_log_depth_convexity_top20",
    "ask_log_depth_slope_top20",
    "ask_log_depth_convexity_top20",
    *(
        name
        for notional in (5, 25, 100)
        for name in (
            f"buy_vwap_impact_bps_{notional}",
            f"buy_fill_ratio_{notional}",
            f"sell_vwap_impact_bps_{notional}",
            f"sell_fill_ratio_{notional}",
        )
    ),
)
_OUTCOME_FLOW_NAMES = (
    "bid_add_intensity_1s",
    "bid_cancel_intensity_1s",
    "ask_add_intensity_1s",
    "ask_cancel_intensity_1s",
    "book_update_count_1s",
)
_CROSS_FEATURE_NAMES = (
    "market_prior_up",
    "midpoint_sum_minus_one",
    "microprice_sum_minus_one",
    "buy_overround",
    "sell_underround",
    "up_book_age_ms",
    "down_book_age_ms",
    "absolute_book_age_skew_ms",
    "elapsed_fraction",
    "remaining_seconds",
    *(
        name
        for window in POLYMARKET_ROUND22_PRIOR_WINDOWS_MS
        for name in (
            f"prior_logit_return_{window}ms",
            f"prior_realized_logit_variation_{window}ms",
            f"prior_valid_grid_coverage_{window}ms",
        )
    ),
)
POLYMARKET_ROUND22_FEATURE_NAMES = (
    *(f"up.{name}" for name in (*_OUTCOME_STATIC_NAMES, *_OUTCOME_FLOW_NAMES)),
    *(f"down.{name}" for name in (*_OUTCOME_STATIC_NAMES, *_OUTCOME_FLOW_NAMES)),
    *_CROSS_FEATURE_NAMES,
)
POLYMARKET_ROUND22_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        POLYMARKET_ROUND22_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 22 feature policy contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 feature policy contains {value}")


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


def load_round22_feature_policy(repository: str | Path) -> dict[str, object]:
    path = Path(repository).resolve() / _POLICY_RELATIVE
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_POLICY_BYTES
    ):
        raise ValueError("Round 22 feature policy is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 22 feature policy is not strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 22 feature policy is not an object")
    policy = dict(decoded)
    claimed = str(policy.pop("policy_sha256", "")).strip().lower()
    parents = policy.get("parents")
    authority = policy.get("authority")
    anti_leakage = policy.get("anti_leakage")
    optional = policy.get("optional_binance_predictor")
    clock = policy.get("clock_and_grid")
    if (
        claimed != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
        or claimed != _canonical_sha256(policy)
        or policy.get("schema_version") != "polymarket-round22-causal-feature-policy-v1"
        or policy.get("status")
        != "frozen_before_feature_materialization_or_target_access"
        or parents
        != {
            "pilot_design_sha256": POLYMARKET_ROUND22_PILOT_DESIGN_SHA256,
            "source_qualification_sha256": (
                POLYMARKET_ROUND22_SOURCE_QUALIFICATION_SHA256
            ),
        }
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
        or not isinstance(anti_leakage, Mapping)
        or any(value is not False for value in anti_leakage.values())
        or not isinstance(optional, Mapping)
        or optional.get("absence_blocks_core") is not False
        or optional.get("credentials_allowed") is not False
        or optional.get("account_data_allowed") is not False
        or optional.get("execution_allowed") is not False
        or optional.get("risk_or_stop_authority") is not False
        or not isinstance(clock, Mapping)
        or clock.get("source_state_rule")
        != "latest_state_with_timestamp_strictly_before_decision"
        or clock.get("feature_cadence_ms") != POLYMARKET_ROUND22_FEATURE_CADENCE_MS
        or clock.get("tabular_anchor_cadence_ms")
        != POLYMARKET_ROUND22_TABULAR_CADENCE_MS
        or clock.get("maximum_book_age_ms") != POLYMARKET_ROUND22_MAXIMUM_BOOK_AGE_MS
        or clock.get("sequence_length") != POLYMARKET_ROUND22_SEQUENCE_LENGTH
    ):
        raise ValueError("Round 22 feature policy differs")
    return {**policy, "policy_sha256": claimed}


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Round 22 {name} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 22 {name} is invalid") from exc
    number = float(parsed)
    if not parsed.is_finite() or parsed <= 0 or not math.isfinite(number):
        raise ValueError(f"Round 22 {name} is invalid")
    return number


def _levels(
    values: Sequence[HistoricalBookLevel],
    *,
    side: str,
) -> tuple[tuple[float, float], ...]:
    output = tuple(
        (
            _positive_float(level.price, name=f"{side} price"),
            _positive_float(level.size, name=f"{side} size"),
        )
        for level in values[:20]
    )
    prices = [item[0] for item in output]
    if len(set(prices)) != len(prices):
        raise ValueError(f"Round 22 {side} prices contain duplicates")
    if side == "bid" and prices != sorted(prices, reverse=True):
        raise ValueError("Round 22 bid prices are not descending")
    if side == "ask" and prices != sorted(prices):
        raise ValueError("Round 22 ask prices are not ascending")
    return output


def _imbalance(
    bids: Sequence[tuple[float, float]],
    asks: Sequence[tuple[float, float]],
    levels: int,
) -> float:
    bid = math.fsum(value[1] for value in bids[:levels])
    ask = math.fsum(value[1] for value in asks[:levels])
    total = bid + ask
    return 0.0 if total <= 0 else (bid - ask) / total


def _depth_shape(
    values: Sequence[tuple[float, float]],
    *,
    tick_size: float,
    bid: bool,
) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    best = values[0][0]
    distances = np.asarray(
        [
            (best - price) / tick_size if bid else (price - best) / tick_size
            for price, _ in values
        ],
        dtype=np.float64,
    )
    maximum = float(np.max(distances))
    if not math.isfinite(maximum) or maximum <= 0:
        return 0.0, 0.0
    x = distances / maximum
    cumulative = np.cumsum(np.asarray([size for _, size in values], dtype=np.float64))
    y = np.log1p(cumulative)
    design = np.column_stack((np.ones_like(x), x, x * x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    slope = float(coefficients[1])
    convexity = float(coefficients[2]) if len(values) >= 3 else 0.0
    if not math.isfinite(slope) or not math.isfinite(convexity):
        raise ValueError("Round 22 depth-shape regression is nonfinite")
    return slope, convexity


def _walk_quote_notional(
    values: Sequence[tuple[float, float]],
    *,
    target_quote: float,
    buy: bool,
) -> tuple[float, float]:
    best = values[0][0]
    remaining = target_quote
    quote = 0.0
    shares = 0.0
    for price, available_shares in values:
        capacity = price * available_shares
        taken_quote = min(remaining, capacity)
        quote += taken_quote
        shares += taken_quote / price
        remaining -= taken_quote
        if remaining <= 1e-12:
            break
    fill_ratio = min(1.0, quote / target_quote)
    if shares <= 0:
        return 0.0, fill_ratio
    vwap = quote / shares
    impact = (vwap / best - 1.0) * 10_000.0 if buy else (1.0 - vwap / best) * 10_000.0
    return max(0.0, impact), fill_ratio


@dataclass(frozen=True, slots=True)
class _BookMetrics:
    values: tuple[float, ...]
    midpoint: float
    microprice: float
    best_bid: float
    best_ask: float


def _book_metrics(snapshot: HistoricalBookSnapshot) -> _BookMetrics:
    if snapshot.negative_risk or not snapshot.bids or not snapshot.asks:
        raise ValueError("Round 22 full book is empty or negative-risk")
    bids = _levels(snapshot.bids, side="bid")
    asks = _levels(snapshot.asks, side="ask")
    if bids[0][0] >= asks[0][0]:
        raise ValueError("Round 22 full book is crossed")
    tick_size = _positive_float(snapshot.tick_size, name="tick size")
    best_bid, top_bid_size = bids[0]
    best_ask, top_ask_size = asks[0]
    midpoint = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    microprice = (best_ask * top_bid_size + best_bid * top_ask_size) / (
        top_bid_size + top_ask_size
    )
    bid_depth = math.fsum(item[1] for item in bids)
    ask_depth = math.fsum(item[1] for item in asks)
    bid_slope, bid_convexity = _depth_shape(
        bids,
        tick_size=tick_size,
        bid=True,
    )
    ask_slope, ask_convexity = _depth_shape(
        asks,
        tick_size=tick_size,
        bid=False,
    )
    impact: list[float] = []
    for notional in POLYMARKET_ROUND22_IMPACT_NOTIONALS:
        buy_impact, buy_fill = _walk_quote_notional(
            asks,
            target_quote=notional,
            buy=True,
        )
        sell_impact, sell_fill = _walk_quote_notional(
            bids,
            target_quote=notional,
            buy=False,
        )
        impact.extend((buy_impact, buy_fill, sell_impact, sell_fill))
    values = (
        best_bid,
        best_ask,
        midpoint,
        spread,
        microprice,
        top_bid_size,
        top_ask_size,
        bid_depth,
        ask_depth,
        float(len(bids)),
        float(len(asks)),
        *(_imbalance(bids, asks, count) for count in (1, 5, 10, 20)),
        bid_slope,
        bid_convexity,
        ask_slope,
        ask_convexity,
        *impact,
    )
    if len(values) != len(_OUTCOME_STATIC_NAMES) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("Round 22 book feature vector differs")
    return _BookMetrics(
        values=values,
        midpoint=midpoint,
        microprice=microprice,
        best_bid=best_bid,
        best_ask=best_ask,
    )


def _book_state_reason(
    snapshot: HistoricalBookSnapshot,
    *,
    outcome: str,
) -> str:
    if snapshot.negative_risk:
        return f"{outcome}_book_negative_risk"
    if not snapshot.bids and not snapshot.asks:
        return f"{outcome}_book_empty_both_sides"
    if not snapshot.bids:
        return f"{outcome}_book_empty_bids"
    if not snapshot.asks:
        return f"{outcome}_book_empty_asks"
    return ""


def _side_change(
    previous: Sequence[HistoricalBookLevel],
    current: Sequence[HistoricalBookLevel],
) -> tuple[float, float]:
    prior = {
        level.price: _positive_float(level.size, name="prior level size")
        for level in previous[:20]
    }
    latest = {
        level.price: _positive_float(level.size, name="current level size")
        for level in current[:20]
    }
    additions = 0.0
    cancellations = 0.0
    for price in set(prior) | set(latest):
        change = latest.get(price, 0.0) - prior.get(price, 0.0)
        if change > 0:
            additions += change
        elif change < 0:
            cancellations -= change
    denominator = 0.5 * (math.fsum(prior.values()) + math.fsum(latest.values()))
    if denominator <= 0:
        return 0.0, 0.0
    return additions / denominator, cancellations / denominator


@dataclass(frozen=True, slots=True)
class _FlowSeries:
    timestamps: tuple[int, ...]
    prefixes: tuple[tuple[float, ...], ...]

    def window(self, decision_time_ms: int) -> tuple[float, ...]:
        left = bisect_left(self.timestamps, decision_time_ms - 1_000)
        right = bisect_left(self.timestamps, decision_time_ms)
        return tuple(
            self.prefixes[right][index] - self.prefixes[left][index]
            for index in range(len(_OUTCOME_FLOW_NAMES))
        )


def _flow_series(window: HistoricalL2Window) -> _FlowSeries:
    timestamps: list[int] = []
    prefixes: list[tuple[float, ...]] = [(0.0,) * len(_OUTCOME_FLOW_NAMES)]
    for previous, current in zip(
        window.snapshots,
        window.snapshots[1:],
        strict=False,
    ):
        bid_add, bid_cancel = _side_change(previous.bids, current.bids)
        ask_add, ask_cancel = _side_change(previous.asks, current.asks)
        values = (bid_add, bid_cancel, ask_add, ask_cancel, 1.0)
        prior = prefixes[-1]
        prefixes.append(tuple(left + right for left, right in zip(prior, values)))
        timestamps.append(current.timestamp_ms)
    return _FlowSeries(timestamps=tuple(timestamps), prefixes=tuple(prefixes))


@dataclass(frozen=True, slots=True)
class Round22CausalFeatureRow:
    condition_id: str
    decision_time_ms: int
    available: bool
    reasons: tuple[str, ...]
    sequence_complete: bool
    tabular_anchor: bool
    tabular_history_complete: bool
    values: tuple[float, ...]
    up_source_timestamp_ms: int
    down_source_timestamp_ms: int
    source_chain_sha256: str
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.decision_time_ms <= 0
            or type(self.available) is not bool
            or type(self.sequence_complete) is not bool
            or type(self.tabular_anchor) is not bool
            or type(self.tabular_history_complete) is not bool
            or len(self.values) != len(POLYMARKET_ROUND22_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.trading_authority
        ):
            raise ValueError("Round 22 causal feature row differs")
        if self.available:
            if (
                self.reasons
                or self.up_source_timestamp_ms <= 0
                or self.down_source_timestamp_ms <= 0
                or _SHA256.fullmatch(self.source_chain_sha256) is None
                or self.source_chain_sha256 == _EMPTY_SHA256
            ):
                raise ValueError("Round 22 available feature row differs")
        elif (
            not self.reasons
            or self.sequence_complete
            or self.tabular_history_complete
            or any(self.values)
            or self.up_source_timestamp_ms
            or self.down_source_timestamp_ms
            or self.source_chain_sha256 != _EMPTY_SHA256
        ):
            raise ValueError("Round 22 unavailable feature row differs")
        if self.tabular_history_complete and not self.tabular_anchor:
            raise ValueError("Round 22 tabular history is not on an anchor")


@dataclass(frozen=True, slots=True)
class Round22ConditionFeatures:
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    rows: tuple[Round22CausalFeatureRow, ...]
    feature_names_sha256: str
    policy_sha256: str
    target_accessed: bool = False
    binance_used: bool = False
    trading_authority: bool = False

    def __post_init__(self) -> None:
        expected_times = tuple(
            range(
                self.event_start_ms + POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
                self.event_end_ms,
                POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
            )
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_end_ms - self.event_start_ms != 300_000
            or len(self.rows) != 1_199
            or tuple(row.decision_time_ms for row in self.rows) != expected_times
            or any(row.condition_id != self.condition_id for row in self.rows)
            or self.feature_names_sha256 != POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
            or self.policy_sha256 != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
            or any((self.target_accessed, self.binance_used, self.trading_authority))
        ):
            raise ValueError("Round 22 condition feature population differs")


def _window_identity(
    up_window: HistoricalL2Window,
    down_window: HistoricalL2Window,
) -> None:
    if (
        up_window.condition_id != down_window.condition_id
        or _CONDITION_ID.fullmatch(up_window.condition_id) is None
        or up_window.asset_id == down_window.asset_id
        or up_window.event_start_ms != down_window.event_start_ms
        or up_window.event_end_ms != down_window.event_end_ms
        or up_window.event_end_ms - up_window.event_start_ms != 300_000
        or not up_window.snapshots
        or not down_window.snapshots
    ):
        raise ValueError("Round 22 paired book-window identity differs")
    for window in (up_window, down_window):
        timestamps = [snapshot.timestamp_ms for snapshot in window.snapshots]
        if (
            any(
                snapshot.condition_id != window.condition_id
                for snapshot in window.snapshots
            )
            or any(
                snapshot.asset_id != window.asset_id for snapshot in window.snapshots
            )
            or any(
                not window.event_start_ms <= timestamp < window.event_end_ms
                for timestamp in timestamps
            )
            or any(
                right <= left
                for left, right in zip(timestamps, timestamps[1:], strict=False)
            )
            or _SHA256.fullmatch(window.source_chain_sha256) is None
        ):
            raise ValueError("Round 22 book-window chronology differs")


def _unavailable_row(
    *,
    condition_id: str,
    decision_time_ms: int,
    reasons: Sequence[str],
    tabular_anchor: bool,
) -> Round22CausalFeatureRow:
    return Round22CausalFeatureRow(
        condition_id=condition_id,
        decision_time_ms=decision_time_ms,
        available=False,
        reasons=tuple(dict.fromkeys(reasons)),
        sequence_complete=False,
        tabular_anchor=tabular_anchor,
        tabular_history_complete=False,
        values=(0.0,) * len(POLYMARKET_ROUND22_FEATURE_NAMES),
        up_source_timestamp_ms=0,
        down_source_timestamp_ms=0,
        source_chain_sha256=_EMPTY_SHA256,
    )


def _prior_dynamics(
    times: Sequence[int],
    logits: Sequence[float],
    decision_time_ms: int,
) -> tuple[float, ...]:
    output: list[float] = []
    current = logits[-1]
    for window in POLYMARKET_ROUND22_PRIOR_WINDOWS_MS:
        cutoff = decision_time_ms - window
        anchor = bisect_right(times, cutoff) - 1
        if anchor < 0:
            anchor = 0
        within = bisect_left(times, cutoff)
        sequence = logits[anchor:]
        variation = math.sqrt(
            math.fsum(
                (right - left) ** 2
                for left, right in zip(sequence, sequence[1:], strict=False)
            )
        )
        expected = window // POLYMARKET_ROUND22_FEATURE_CADENCE_MS
        coverage = min(1.0, (len(times) - within) / expected)
        output.extend((current - logits[anchor], variation, coverage))
    return tuple(output)


def build_round22_condition_features(
    *,
    repository: str | Path,
    up_window: HistoricalL2Window,
    down_window: HistoricalL2Window,
) -> Round22ConditionFeatures:
    """Build one causal feature grid without consulting outcomes or Binance."""

    load_round22_feature_policy(repository)
    _window_identity(up_window, down_window)
    up_flow = _flow_series(up_window)
    down_flow = _flow_series(down_window)
    metric_cache: dict[tuple[str, int], _BookMetrics] = {}
    up_index = -1
    down_index = -1
    prior_times: list[int] = []
    prior_logits: list[float] = []
    rows: list[Round22CausalFeatureRow] = []
    start = up_window.event_start_ms
    end = up_window.event_end_ms
    for decision in range(
        start + POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
        end,
        POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
    ):
        while (
            up_index + 1 < len(up_window.snapshots)
            and up_window.snapshots[up_index + 1].timestamp_ms < decision
        ):
            up_index += 1
        while (
            down_index + 1 < len(down_window.snapshots)
            and down_window.snapshots[down_index + 1].timestamp_ms < decision
        ):
            down_index += 1
        offset = decision - start
        tabular_anchor = offset % POLYMARKET_ROUND22_TABULAR_CADENCE_MS == 0
        reasons: list[str] = []
        if up_index < 0:
            reasons.append("up_book_unavailable")
        if down_index < 0:
            reasons.append("down_book_unavailable")
        if reasons:
            rows.append(
                _unavailable_row(
                    condition_id=up_window.condition_id,
                    decision_time_ms=decision,
                    reasons=reasons,
                    tabular_anchor=tabular_anchor,
                )
            )
            continue
        up_snapshot = up_window.snapshots[up_index]
        down_snapshot = down_window.snapshots[down_index]
        up_age = decision - up_snapshot.timestamp_ms
        down_age = decision - down_snapshot.timestamp_ms
        if up_age > POLYMARKET_ROUND22_MAXIMUM_BOOK_AGE_MS:
            reasons.append("up_book_stale")
        if down_age > POLYMARKET_ROUND22_MAXIMUM_BOOK_AGE_MS:
            reasons.append("down_book_stale")
        for outcome, snapshot in (
            ("up", up_snapshot),
            ("down", down_snapshot),
        ):
            reason = _book_state_reason(snapshot, outcome=outcome)
            if reason:
                reasons.append(reason)
        if reasons:
            rows.append(
                _unavailable_row(
                    condition_id=up_window.condition_id,
                    decision_time_ms=decision,
                    reasons=reasons,
                    tabular_anchor=tabular_anchor,
                )
            )
            continue
        try:
            up_key = ("up", up_index)
            down_key = ("down", down_index)
            if up_key not in metric_cache:
                metric_cache[up_key] = _book_metrics(up_snapshot)
            if down_key not in metric_cache:
                metric_cache[down_key] = _book_metrics(down_snapshot)
            up_metrics = metric_cache[up_key]
            down_metrics = metric_cache[down_key]
        except ValueError:
            rows.append(
                _unavailable_row(
                    condition_id=up_window.condition_id,
                    decision_time_ms=decision,
                    reasons=("book_metric_invalid",),
                    tabular_anchor=tabular_anchor,
                )
            )
            continue
        denominator = up_metrics.microprice + down_metrics.microprice
        if not 0 < denominator < 2:
            raise ValueError("Round 22 normalized market prior denominator differs")
        prior = up_metrics.microprice / denominator
        if not 0 < prior < 1:
            raise ValueError("Round 22 normalized market prior differs")
        prior_times.append(decision)
        prior_logits.append(math.log(prior / (1.0 - prior)))
        dynamics = _prior_dynamics(prior_times, prior_logits, decision)
        cross = (
            prior,
            up_metrics.midpoint + down_metrics.midpoint - 1.0,
            up_metrics.microprice + down_metrics.microprice - 1.0,
            up_metrics.best_ask + down_metrics.best_ask - 1.0,
            1.0 - (up_metrics.best_bid + down_metrics.best_bid),
            float(up_age),
            float(down_age),
            float(abs(up_age - down_age)),
            offset / 300_000.0,
            (end - decision) / 1_000.0,
            *dynamics,
        )
        values = (
            *up_metrics.values,
            *up_flow.window(decision),
            *down_metrics.values,
            *down_flow.window(decision),
            *cross,
        )
        if len(values) != len(POLYMARKET_ROUND22_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("Round 22 causal feature width differs")
        eligible_clock = 4_000 <= offset <= 295_000
        last_sequence = rows[-(POLYMARKET_ROUND22_SEQUENCE_LENGTH - 1) :]
        sequence_complete = (
            eligible_clock
            and len(last_sequence) == POLYMARKET_ROUND22_SEQUENCE_LENGTH - 1
            and all(row.available for row in last_sequence)
        )
        coverage_values = dynamics[2::3]
        tabular_history_complete = (
            tabular_anchor
            and 15_000 <= offset <= 295_000
            and all(value >= 1.0 for value in coverage_values)
        )
        source_chain = _canonical_sha256(
            {
                "condition_id": up_window.condition_id,
                "decision_time_ms": decision,
                "down_snapshot_sha256": down_snapshot.source_payload_sha256,
                "down_source_chain_sha256": down_window.source_chain_sha256,
                "feature_names_sha256": POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
                "feature_policy_sha256": POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
                "up_snapshot_sha256": up_snapshot.source_payload_sha256,
                "up_source_chain_sha256": up_window.source_chain_sha256,
            }
        )
        rows.append(
            Round22CausalFeatureRow(
                condition_id=up_window.condition_id,
                decision_time_ms=decision,
                available=True,
                reasons=(),
                sequence_complete=sequence_complete,
                tabular_anchor=tabular_anchor,
                tabular_history_complete=tabular_history_complete,
                values=values,
                up_source_timestamp_ms=up_snapshot.timestamp_ms,
                down_source_timestamp_ms=down_snapshot.timestamp_ms,
                source_chain_sha256=source_chain,
            )
        )
    if len(rows) != 1_199:
        raise ValueError("Round 22 causal grid count differs")
    return Round22ConditionFeatures(
        condition_id=up_window.condition_id,
        event_start_ms=start,
        event_end_ms=end,
        rows=tuple(rows),
        feature_names_sha256=POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
        policy_sha256=POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
    )


__all__ = [
    "POLYMARKET_ROUND22_FEATURE_CADENCE_MS",
    "POLYMARKET_ROUND22_FEATURE_NAMES",
    "POLYMARKET_ROUND22_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND22_FEATURE_POLICY_SHA256",
    "POLYMARKET_ROUND22_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND22_IMPACT_NOTIONALS",
    "POLYMARKET_ROUND22_MAXIMUM_BOOK_AGE_MS",
    "POLYMARKET_ROUND22_PRIOR_WINDOWS_MS",
    "POLYMARKET_ROUND22_SEQUENCE_LENGTH",
    "POLYMARKET_ROUND22_TABULAR_CADENCE_MS",
    "Round22CausalFeatureRow",
    "Round22ConditionFeatures",
    "build_round22_condition_features",
    "load_round22_feature_policy",
]

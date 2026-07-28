"""Frozen state, order-flow, and clock groups for Round 74 event models."""

from __future__ import annotations

import hashlib
import json

from .impact_absorption_event_sequence import ROUND74_EVENT_FEATURE_NAMES


ROUND74_EVENT_FEATURE_VIEW_SCHEMA_VERSION = "round-074-feature-view-v3"
ROUND74_EVENT_FEATURE_VIEWS = (
    "market_state_clock_neutral",
    "clock_neutral",
    "market_state_with_clock",
    "full",
)
_ROUND74_EVENT_INTRADAY_CLOCK_FEATURE_NAMES = frozenset(
    {
        "utc_second_of_day_sine",
        "utc_second_of_day_cosine",
    }
)
ROUND74_EVENT_CLOCK_FEATURE_NAMES = tuple(
    name
    for name in ROUND74_EVENT_FEATURE_NAMES
    if name.startswith("exchange_clock_")
    or name in _ROUND74_EVENT_INTRADAY_CLOCK_FEATURE_NAMES
)
ROUND74_EVENT_CLOCK_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in ROUND74_EVENT_CLOCK_FEATURE_NAMES
)
ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        ROUND74_EVENT_CLOCK_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
_ROUND74_EVENT_ORDER_FLOW_EXACT_NAMES = frozenset(
    {
        "log1p_interarrival_us",
        "trade_signed_quote_scaled",
        "trade_absolute_quote_scaled",
        "trade_price_to_mid_bps",
        "liquidation_signed_quote_scaled",
        "bbo_bid_qty_change_scaled",
        "bbo_ask_qty_change_scaled",
        "log1p_ms_since_aggregate_trade",
        "log1p_ms_since_liquidation",
    }
)
_ROUND74_EVENT_ORDER_FLOW_PREFIXES = (
    "event_is_",
    "depth_signed_pressure_",
    "depth_absolute_flow_",
    "ewm_signed_trade_pressure_",
    "ewm_signed_depth_pressure_",
    "ewm_signed_liquidation_pressure_",
)
ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES = tuple(
    name
    for name in ROUND74_EVENT_FEATURE_NAMES
    if name in _ROUND74_EVENT_ORDER_FLOW_EXACT_NAMES
    or name.startswith(_ROUND74_EVENT_ORDER_FLOW_PREFIXES)
)
ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES
)
ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES = tuple(
    name
    for index, name in enumerate(ROUND74_EVENT_FEATURE_NAMES)
    if index not in ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES
    and index not in ROUND74_EVENT_CLOCK_FEATURE_INDICES
)
ROUND74_EVENT_MARKET_STATE_FEATURE_INDICES = tuple(
    ROUND74_EVENT_FEATURE_NAMES.index(name)
    for name in ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES
)
ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES = {
    "market_state_clock_neutral": tuple(
        sorted(
            {
                *ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,
                *ROUND74_EVENT_CLOCK_FEATURE_INDICES,
            }
        )
    ),
    "clock_neutral": ROUND74_EVENT_CLOCK_FEATURE_INDICES,
    "market_state_with_clock": ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,
    "full": (),
}

if (
    len(ROUND74_EVENT_CLOCK_FEATURE_NAMES) != 11
    or len(set(ROUND74_EVENT_CLOCK_FEATURE_INDICES)) != 11
    or len(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES) != 31
    or len(set(ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES)) != 31
    or len(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES) != 33
    or len(set(ROUND74_EVENT_MARKET_STATE_FEATURE_INDICES)) != 33
    or set(ROUND74_EVENT_CLOCK_FEATURE_INDICES)
    & set(ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES)
    or set(ROUND74_EVENT_MARKET_STATE_FEATURE_INDICES)
    & (
        set(ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES)
        | set(ROUND74_EVENT_CLOCK_FEATURE_INDICES)
    )
    or (
        set(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES)
        | set(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES)
        | set(ROUND74_EVENT_CLOCK_FEATURE_NAMES)
    )
    != set(ROUND74_EVENT_FEATURE_NAMES)
    or set(ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES)
    != set(ROUND74_EVENT_FEATURE_VIEWS)
):
    raise RuntimeError("Round 74 state-first feature panel differs")


__all__ = [
    "ROUND74_EVENT_CLOCK_FEATURE_INDICES",
    "ROUND74_EVENT_CLOCK_FEATURE_NAMES",
    "ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256",
    "ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES",
    "ROUND74_EVENT_FEATURE_VIEW_SCHEMA_VERSION",
    "ROUND74_EVENT_FEATURE_VIEWS",
    "ROUND74_EVENT_MARKET_STATE_FEATURE_INDICES",
    "ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES",
    "ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES_SHA256",
    "ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES",
    "ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES",
    "ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES_SHA256",
]

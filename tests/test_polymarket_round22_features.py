from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_historical_l2 import (
    HistoricalBookLevel,
    HistoricalBookSnapshot,
    HistoricalL2Window,
)
from simple_ai_trading.polymarket_round22_features import (
    POLYMARKET_ROUND22_FEATURE_NAMES,
    POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
    build_round22_condition_features,
    load_round22_feature_policy,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONDITION_ID = "0x" + ("a" * 64)
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40
EVENT_START_MS = 1_777_248_000_000
EVENT_END_MS = EVENT_START_MS + 300_000


def _levels(*, bid: bool) -> tuple[HistoricalBookLevel, ...]:
    prices = (
        [0.40 - index * 0.01 for index in range(20)]
        if bid
        else [0.60 + index * 0.01 for index in range(20)]
    )
    return tuple(
        HistoricalBookLevel(price=f"{price:.2f}", size="100") for price in prices
    )


def _window(asset_id: str) -> HistoricalL2Window:
    timestamps = [EVENT_START_MS + 100, EVENT_START_MS + 250]
    timestamps.extend(range(EVENT_START_MS + 1_250, EVENT_END_MS, 1_000))
    snapshots = tuple(
        HistoricalBookSnapshot(
            condition_id=CONDITION_ID,
            asset_id=asset_id,
            timestamp_ms=timestamp,
            book_hash="b" * 40,
            bids=_levels(bid=True),
            asks=_levels(bid=False),
            minimum_order_size="5",
            tick_size="0.01",
            negative_risk=False,
            last_trade_price="0.5",
            source_payload_sha256=hashlib.sha256(
                f"{asset_id}:{timestamp}".encode("ascii")
            ).hexdigest(),
        )
        for timestamp in timestamps
    )
    return HistoricalL2Window(
        condition_id=CONDITION_ID,
        asset_id=asset_id,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_END_MS,
        snapshots=snapshots,
        source_chain_sha256=hashlib.sha256(asset_id.encode("ascii")).hexdigest(),
    )


def test_round22_feature_policy_and_names_are_hash_bound_and_non_authoritative() -> (
    None
):
    policy = load_round22_feature_policy(REPOSITORY)

    assert policy["policy_sha256"] == POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
    assert not any(policy["authority"].values())
    assert not any(policy["anti_leakage"].values())
    assert policy["optional_binance_predictor"]["absence_blocks_core"] is False
    assert len(POLYMARKET_ROUND22_FEATURE_NAMES) == 91
    assert (
        POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
        == hashlib.sha256(
            json.dumps(
                POLYMARKET_ROUND22_FEATURE_NAMES,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
    )


def test_round22_feature_grid_is_strict_prior_complete_and_target_blind() -> None:
    result = build_round22_condition_features(
        repository=REPOSITORY,
        up_window=_window(UP_TOKEN_ID),
        down_window=_window(DOWN_TOKEN_ID),
    )
    names = {name: index for index, name in enumerate(POLYMARKET_ROUND22_FEATURE_NAMES)}
    first = result.rows[0]
    second = result.rows[1]

    assert len(result.rows) == 1_199
    assert all(row.available for row in result.rows)
    assert sum(row.tabular_anchor for row in result.rows) == 299
    assert sum(row.sequence_complete for row in result.rows) == 1_165
    assert sum(row.tabular_history_complete for row in result.rows) == 281
    assert first.decision_time_ms == EVENT_START_MS + 250
    assert first.up_source_timestamp_ms == EVENT_START_MS + 100
    assert second.decision_time_ms == EVENT_START_MS + 500
    assert second.up_source_timestamp_ms == EVENT_START_MS + 250
    assert first.values[names["up.best_bid"]] == pytest.approx(0.4)
    assert first.values[names["up.best_ask"]] == pytest.approx(0.6)
    assert first.values[names["market_prior_up"]] == pytest.approx(0.5)
    assert first.values[names["buy_overround"]] == pytest.approx(0.2)
    assert first.values[names["sell_underround"]] == pytest.approx(0.2)
    assert not any(
        (result.target_accessed, result.binance_used, result.trading_authority)
    )


def test_round22_feature_grid_marks_stale_books_unavailable_without_carrying_values() -> (
    None
):
    up = _window(UP_TOKEN_ID)
    down = _window(DOWN_TOKEN_ID)
    sparse_up = replace(up, snapshots=up.snapshots[:1])

    result = build_round22_condition_features(
        repository=REPOSITORY,
        up_window=sparse_up,
        down_window=down,
    )
    stale = result.rows[4]

    assert stale.decision_time_ms == EVENT_START_MS + 1_250
    assert not stale.available
    assert stale.reasons == ("up_book_stale",)
    assert not any(stale.values)
    assert stale.source_chain_sha256 == hashlib.sha256(b"").hexdigest()
    assert not stale.sequence_complete
    assert not stale.tabular_history_complete


def test_round22_feature_grid_rejects_cross_condition_or_nonfive_minute_windows() -> (
    None
):
    up = _window(UP_TOKEN_ID)
    down = _window(DOWN_TOKEN_ID)

    with pytest.raises(ValueError, match="paired book-window identity differs"):
        build_round22_condition_features(
            repository=REPOSITORY,
            up_window=up,
            down_window=replace(down, event_end_ms=EVENT_END_MS - 1),
        )

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.polymarket_historical_shadow import (
    BtcAggregateTradeObservation,
    PolymarketBtcFlowBuffer,
    PolymarketShadowDataUnavailable,
)
from simple_ai_trading.polymarket_round16_dataset import (
    build_round16_feature_row,
)
from simple_ai_trading.polymarket_round16_shadow import (
    PolymarketRound16LiveFeatureBuilder,
)


EVENT_START_MS = 1_800_000_000_000
DECISION_TIME_MS = EVENT_START_MS + 60_000


def _observation(
    market: str,
    *,
    second: int,
    aggregate_id: int,
) -> BtcAggregateTradeObservation:
    event_time_ms = EVENT_START_MS + second * 1_000 + 100
    return BtcAggregateTradeObservation(
        market=market,
        source=(
            "BINANCE_SPOT"
            if market == "spot"
            else "BINANCE_USD_M_FUTURES"
        ),
        symbol="BTCUSDT",
        event_time_ms=event_time_ms,
        received_at_ms=event_time_ms + 20,
        aggregate_trade_id=aggregate_id,
        first_trade_id=aggregate_id * 2,
        last_trade_id=aggregate_id * 2 + 1,
        price=(
            60_000.0 + second * 0.5
            if market == "spot"
            else 60_006.0 + second * 0.55
        ),
        quantity=0.01 + aggregate_id % 3 * 0.001,
        buyer_is_maker=aggregate_id % 2 == 0,
    )


def _flow_and_history() -> tuple[
    PolymarketBtcFlowBuffer,
    dict[str, np.ndarray],
]:
    flow = PolymarketBtcFlowBuffer(retention_seconds=300)
    day_start_ms = EVENT_START_MS - 300_000
    row_count = 600
    history: dict[str, np.ndarray] = {
        "second_ms": day_start_ms
        + np.arange(row_count, dtype=np.int64) * 1_000,
    }
    for market in ("spot", "perpetual"):
        history[f"{market}_close"] = np.full(
            row_count,
            np.nan,
            dtype=np.float64,
        )
        for name in (
            "quote_volume",
            "aggressive_buy_quote",
            "aggressive_sell_quote",
            "aggregate_count",
            "constituent_trade_count",
            "maximum_aggregate_quote",
            "squared_aggregate_quote_sum",
            "last_trade_age_seconds",
        ):
            history[f"{market}_{name}"] = np.zeros(
                row_count,
                dtype=np.float64,
            )
    for index, second in enumerate(range(-151, 61), start=1):
        for market, offset in (("spot", 0), ("perpetual", 10_000)):
            observation = _observation(
                market,
                second=second,
                aggregate_id=offset + index,
            )
            flow.ingest(observation)
            row = (observation.event_time_ms - day_start_ms) // 1_000
            quote = observation.quote_notional
            history[f"{market}_close"][row] = observation.price
            history[f"{market}_quote_volume"][row] += quote
            side = (
                "aggressive_sell_quote"
                if observation.buyer_is_maker
                else "aggressive_buy_quote"
            )
            history[f"{market}_{side}"][row] += quote
            history[f"{market}_aggregate_count"][row] += 1
            history[f"{market}_constituent_trade_count"][
                row
            ] += observation.constituent_trade_count
            history[f"{market}_maximum_aggregate_quote"][row] = quote
            history[f"{market}_squared_aggregate_quote_sum"][row] += (
                quote * quote
            )
    for market in ("spot", "perpetual"):
        close = history[f"{market}_close"]
        count = history[f"{market}_aggregate_count"]
        age = history[f"{market}_last_trade_age_seconds"]
        first_index = int(np.flatnonzero(np.isfinite(close))[0])
        close[:first_index] = close[first_index]
        last_close = np.nan
        last_age = np.iinfo(np.uint32).max
        for index in range(row_count):
            if count[index] > 0:
                last_close = close[index]
                last_age = 0
            elif np.isfinite(last_close):
                close[index] = last_close
                last_age += 1
            age[index] = last_age
    return flow, history


def test_round16_live_vector_is_bit_identical_to_historical_builder() -> None:
    flow, history = _flow_and_history()
    live = PolymarketRound16LiveFeatureBuilder(flow).feature_vector(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
    )
    historical = build_round16_feature_row(
        SimpleNamespace(
            event_start_ms=EVENT_START_MS,
            end_ms=EVENT_START_MS + 900_000,
            condition_id="0x" + "1" * 64,
            identity_payload_sha256="2" * 64,
            role="test",
        ),
        flow_start_ms=EVENT_START_MS - 300_000,
        decision_offset_seconds=60,
        flow=history,
    ).feature_values

    assert live.dtype == np.float32
    assert np.array_equal(live, historical)


def test_round16_live_builder_fails_closed_after_feed_epoch_reset() -> None:
    flow, _ = _flow_and_history()
    flow.reset_market("spot")
    builder = PolymarketRound16LiveFeatureBuilder(flow)

    with pytest.raises(PolymarketShadowDataUnavailable, match="spot_flow_missing"):
        builder.feature_vector(
            event_start_ms=EVENT_START_MS,
            decision_time_ms=DECISION_TIME_MS,
            observed_at_ms=DECISION_TIME_MS + 100,
        )


def test_causal_snapshot_is_a_copy_not_mutable_feed_state() -> None:
    flow, _ = _flow_and_history()
    first = flow.causal_flow_snapshot(
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
        second_count=150,
    )
    first["spot_close"][0] = 0
    second = flow.causal_flow_snapshot(
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
        second_count=150,
    )

    assert second["spot_close"][0] > 0

from __future__ import annotations

import asyncio
import json

import pytest

from simple_ai_trading.polymarket_historical_shadow import PolymarketBtcFlowBuffer
from simple_ai_trading.polymarket_historical_shadow_feed import (
    PolymarketHistoricalShadowFeed,
    _QueuedMessage,
    parse_binance_aggregate_trade,
)


NOW_MS = 1_782_086_700_500


def _spot() -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": NOW_MS - 20,
        "s": "BTCUSDT",
        "a": 123,
        "p": "60000.1",
        "q": "0.25",
        "f": 200,
        "l": 202,
        "T": NOW_MS - 25,
        "m": False,
        "M": True,
    }


def _futures() -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": NOW_MS - 15,
        "s": "BTCUSDT",
        "a": 456,
        "p": "60005.2",
        "q": "0.50",
        "nq": "0.45",
        "f": 800,
        "l": 803,
        "T": NOW_MS - 18,
        "m": True,
        "st": 1,
    }


def test_current_spot_and_usdm_wire_contracts_parse_exactly() -> None:
    spot = parse_binance_aggregate_trade(
        json.dumps(_spot()),
        market="spot",
        received_at_ms=NOW_MS,
    )
    futures = parse_binance_aggregate_trade(
        json.dumps(_futures()).encode(),
        market="perpetual",
        received_at_ms=NOW_MS,
    )
    assert spot.source == "BINANCE_SPOT"
    assert spot.constituent_trade_count == 3
    assert spot.quantity == 0.25
    assert futures.source == "BINANCE_USD_M_FUTURES"
    assert futures.constituent_trade_count == 4
    assert futures.quantity == 0.5


@pytest.mark.parametrize(
    ("payload", "market", "message"),
    [
        ({**_spot(), "unexpected": 1}, "spot", "schema drifted"),
        ({**_spot(), "s": "ETHUSDT"}, "spot", "identity differs"),
        ({**_futures(), "st": 2}, "perpetual", "not USD-M"),
        ({**_futures(), "nq": "0.51"}, "perpetual", "exceeds quantity"),
    ],
)
def test_schema_scope_and_quantity_drift_fail_closed(
    payload: dict[str, object],
    market: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_binance_aggregate_trade(
            json.dumps(payload),
            market=market,
            received_at_ms=NOW_MS,
        )


def test_futures_all_rpi_quantity_is_valid_public_telemetry() -> None:
    payload = {**_futures(), "nq": "0"}
    observation = parse_binance_aggregate_trade(
        json.dumps(payload),
        market="perpetual",
        received_at_ms=NOW_MS,
    )
    assert observation.quantity == 0.5


def test_duplicate_json_keys_are_rejected() -> None:
    raw = json.dumps(_spot()).replace(
        '"s": "BTCUSDT"',
        '"s": "BTCUSDT", "s": "ETHUSDT"',
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        parse_binance_aggregate_trade(
            raw,
            market="spot",
            received_at_ms=NOW_MS,
        )


def test_reset_market_requires_fresh_lookback() -> None:
    flow = PolymarketBtcFlowBuffer()
    observation = parse_binance_aggregate_trade(
        json.dumps(_spot()),
        market="spot",
        received_at_ms=NOW_MS,
    )
    flow.ingest(observation)
    assert flow.ingest(observation) is False
    flow.reset_market("spot")
    assert flow.ingest(observation) is True


def test_bounded_feed_consumes_both_epochs_without_authority() -> None:
    async def exercise() -> None:
        flow = PolymarketBtcFlowBuffer()
        feed = PolymarketHistoricalShadowFeed(flow=flow, queue_capacity=1_024)
        feed._epochs["spot"] = 1
        feed._epochs["perpetual"] = 1
        await feed._queue.put(
            _QueuedMessage("spot", 1, NOW_MS, json.dumps(_spot()))
        )
        await feed._queue.put(
            _QueuedMessage("perpetual", 1, NOW_MS, json.dumps(_futures()))
        )
        stop = asyncio.Event()
        consumer = asyncio.create_task(feed._consumer(stop))
        await asyncio.wait_for(feed._queue.join(), timeout=1.0)
        stop.set()
        await consumer
        health = feed.health()
        assert health.ingested_counts == {"spot": 1, "perpetual": 1}
        assert health.trading_authority is False

    asyncio.run(exercise())

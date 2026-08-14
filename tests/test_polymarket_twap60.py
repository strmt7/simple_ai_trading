from __future__ import annotations

from dataclasses import replace
import json

import pytest

from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_twap60 import (
    PolymarketTwap60FeatureState,
    parse_polymarket_twap60_tick,
)


_START_SECONDS = 1_784_058_600
_START_MS = _START_SECONDS * 1_000


def _market():
    return parse_polymarket_five_minute_market(
        {
            "id": "twap60-market",
            "question": "BTC Up or Down",
            "conditionId": "0x" + "7" * 64,
            "slug": f"btc-updown-5m-{_START_SECONDS}",
            "eventStartTime": "2026-07-14T19:50:00Z",
            "endDate": "2026-07-14T19:55:00Z",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "acceptingOrders": True,
            "clobTokenIds": json.dumps(["7" * 40, "7" * 39 + "1"]),
            "outcomes": '["Up", "Down"]',
            "orderPriceMinTickSize": 0.01,
            "orderMinSize": 5,
            "feesEnabled": True,
            "feeSchedule": {
                "exponent": 1,
                "rate": 0.07,
                "takerOnly": True,
                "rebateRate": 0.2,
            },
            "liquidityNum": 20_000,
            "volumeNum": 50_000,
            "resolutionSource": (
                "https://data.chain.link/streams/btc-usd-twap-60s-streams"
            ),
            "cryptoMarketConfigId": "btc-5m-twap-60",
            "cryptoMarketConfig": {
                "asset": "btc",
                "duration": "5m",
                "id": "btc-5m-twap-60",
                "twapEnabled": True,
                "twapLookbackSeconds": 60,
            },
        }
    )


def _tick(index: int, *, exact: int | None = None):
    source = _START_MS + index * 1_000
    value = (64_000 * 10**18 + index * 10**16) if exact is None else exact
    return parse_polymarket_twap60_tick(
        {
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "timestamp": source + 1_500,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": source,
                "value": value / 10**18,
                "window_s": 60,
                "full_accuracy_value": str(value),
            },
        },
        received_wall_ms=source + 3_000,
        received_monotonic_ns=(index + 1) * 1_000_000_000,
    )


def test_parse_twap60_tick_preserves_exact_value_and_delays() -> None:
    tick = _tick(0)

    assert tick.exact_e18 == 64_000 * 10**18
    assert tick.source_to_publisher_ms == 1_500
    assert tick.publisher_to_receipt_ms == 1_500
    assert tick.source_to_receipt_ms == 3_000
    assert tick.asdict()["price"] == "64000"


def test_parse_twap60_tick_rejects_schema_and_rounded_value_drift() -> None:
    event = {
        "topic": "crypto_prices_twap_sixty",
        "type": "update",
        "timestamp": _START_MS + 1_500,
        "payload": {
            "symbol": "btc/usd",
            "timestamp": _START_MS,
            "value": 1,
            "window_s": 60,
            "full_accuracy_value": str(64_000 * 10**18),
        },
    }

    with pytest.raises(ValueError, match="exact and rounded"):
        parse_polymarket_twap60_tick(
            event,
            received_wall_ms=_START_MS + 3_000,
            received_monotonic_ns=1,
        )
    event["unexpected"] = True
    with pytest.raises(ValueError, match="envelope differs"):
        parse_polymarket_twap60_tick(
            event,
            received_wall_ms=_START_MS + 3_000,
            received_monotonic_ns=1,
        )


def test_feature_state_is_causal_target_free_and_available_after_coverage() -> None:
    state = PolymarketTwap60FeatureState()
    for index in range(12):
        assert state.observe(_tick(index)) is True
    future = _tick(20)
    state.observe(future)

    features = state.features(
        _market(),
        observed_wall_ms=_START_MS + 15_000,
        observed_monotonic_ns=15_000_000_000,
    )

    assert features.available is True
    assert features.tick_count == 12
    assert features.opening_exact_e18 == 64_000 * 10**18
    assert features.current_exact_e18 == 64_000 * 10**18 + 11 * 10**16
    assert features.remaining_seconds == 285.0
    assert features.realized_variance_rate_per_second is not None
    assert features.realized_variance_rate_per_second > 0
    assert features.path_efficiency == pytest.approx(1.0)
    assert features.grants_execution_authority is False


def test_feature_state_fails_closed_when_opening_tick_is_missing() -> None:
    state = PolymarketTwap60FeatureState(minimum_return_count=2)
    for index in range(1, 5):
        state.observe(_tick(index))

    features = state.features(
        _market(),
        observed_wall_ms=_START_MS + 7_000,
        observed_monotonic_ns=7_000_000_000,
    )

    assert features.available is False
    assert "exact_opening_twap_missing" in features.reasons


def test_feature_state_deduplicates_exact_replay_and_rejects_conflict() -> None:
    state = PolymarketTwap60FeatureState()
    tick = _tick(0)

    assert state.observe(tick) is True
    assert state.observe(replace(tick, received_monotonic_ns=2)) is False
    with pytest.raises(ValueError, match="duplicate time contradicts"):
        state.observe(replace(tick, exact_e18=tick.exact_e18 + 1))

from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.price_discovery_dataset import build_price_discovery_day
from simple_ai_trading.price_discovery_spec import (
    FEATURE_BURN_IN_SECONDS,
    HORIZONS_SECONDS,
    layer_feature_names,
)
from simple_ai_trading.spot_perpetual_flow import FLOW_SYMBOLS, SECONDS_PER_DAY


PERIOD = "2024-01-02"


def _flow(price: np.ndarray, *, signed_share: float = 0.2):
    quote = price.copy()
    buy_share = (1.0 + signed_share) / 2.0
    sell_share = 1.0 - buy_share
    return {
        "open": price.copy(),
        "high": price * 1.0001,
        "low": price * 0.9999,
        "close": price.copy(),
        "base_volume": np.ones(SECONDS_PER_DAY, dtype=np.float64),
        "quote_volume": quote,
        "aggressive_buy_quote": quote * buy_share,
        "aggressive_sell_quote": quote * sell_share,
        "aggregate_count": np.ones(SECONDS_PER_DAY, dtype=np.uint32),
        "constituent_trade_count": np.ones(SECONDS_PER_DAY, dtype=np.uint32),
        "maximum_aggregate_quote": quote.copy(),
        "squared_aggregate_quote_sum": quote * quote,
        "last_trade_age_seconds": np.zeros(SECONDS_PER_DAY, dtype=np.uint32),
    }


def _streams():
    seconds = np.arange(SECONDS_PER_DAY, dtype=np.float64)
    output = {}
    for symbol_index, symbol in enumerate(FLOW_SYMBOLS, start=1):
        perpetual_rate = symbol_index * 1e-6
        perpetual = (100.0 + symbol_index) * np.exp(perpetual_rate * seconds)
        spot = perpetual / 1.001
        output[("futures", symbol)] = _flow(perpetual)
        output[("spot", symbol)] = _flow(spot)
    return output


def test_round72_day_features_match_known_causal_formulas() -> None:
    built = build_price_discovery_day(period=PERIOD, flow_by_stream=_streams())
    names = layer_feature_names("cross_asset")
    positions = {name: index for index, name in enumerate(names)}
    btc = built["BTCUSDT"]
    features = np.asarray(btc["features"])
    anchors = np.asarray(btc["anchor_second_ms"])
    primary = np.asarray(btc["primary_target_bps"])

    assert features.shape == (int(btc["candidate_anchors"]), 336)
    assert np.all(anchors % 30_000 == 29_000)
    assert (anchors[0] % 86_400_000) // 1_000 >= FEATURE_BURN_IN_SECONDS
    assert np.all(np.asarray(btc["primary_valid"]))
    assert np.all(np.asarray(btc["stress_valid"]))
    assert features[0, positions["perpetual_log_return_bps_30s"]] == pytest.approx(
        0.3, abs=1e-5
    )
    assert features[
        0, positions["perpetual_path_variation_bps_30s"]
    ] == pytest.approx(0.3, abs=1e-5)
    assert features[
        0, positions["perpetual_realized_volatility_bps_30s"]
    ] == pytest.approx(0.01 * np.sqrt(30), abs=1e-5)
    assert features[
        0, positions["perpetual_signed_quote_flow_30s"]
    ] == pytest.approx(0.2, abs=1e-6)
    assert features[
        0, positions["perpetual_zero_flow_fraction_30s"]
    ] == pytest.approx(0.0)
    assert features[
        0, positions["basis_change_bps_300s"]
    ] == pytest.approx(0.0, abs=1e-5)
    assert features[
        0, positions["cross_asset_perpetual_return_mean_bps_30s"]
    ] == pytest.approx(0.6, abs=1e-5)
    assert features[
        0, positions["cross_asset_leader_perpetual_return_bps_30s"]
    ] == pytest.approx(0.75, abs=1e-5)
    assert primary[0] == pytest.approx(
        np.asarray(HORIZONS_SECONDS, dtype=np.float64) * 0.01,
        abs=1e-8,
    )


def test_round72_age_and_exact_trade_target_gates_are_independent() -> None:
    streams = _streams()
    baseline = build_price_discovery_day(period=PERIOD, flow_by_stream=streams)
    first_anchor_index = int(
        (np.asarray(baseline["BTCUSDT"]["anchor_second_ms"])[0] % 86_400_000)
        // 1_000
    )
    streams[("spot", "BTCUSDT")]["last_trade_age_seconds"][first_anchor_index] = 3
    streams[("futures", "ETHUSDT")]["aggregate_count"][
        first_anchor_index + 2
    ] = 0
    streams[("futures", "ETHUSDT")]["base_volume"][first_anchor_index + 2] = 0.0
    streams[("futures", "ETHUSDT")]["quote_volume"][first_anchor_index + 2] = 0.0
    streams[("futures", "ETHUSDT")]["aggressive_buy_quote"][
        first_anchor_index + 2
    ] = 0.0
    streams[("futures", "ETHUSDT")]["aggressive_sell_quote"][
        first_anchor_index + 2
    ] = 0.0
    streams[("futures", "ETHUSDT")]["maximum_aggregate_quote"][
        first_anchor_index + 2
    ] = 0.0
    streams[("futures", "ETHUSDT")]["squared_aggregate_quote_sum"][
        first_anchor_index + 2
    ] = 0.0

    built = build_price_discovery_day(period=PERIOD, flow_by_stream=streams)

    assert int(built["BTCUSDT"]["age_eligible_anchors"]) == int(
        baseline["BTCUSDT"]["age_eligible_anchors"]
    ) - 1
    assert np.asarray(built["BTCUSDT"]["features"]).shape[0] == np.asarray(
        baseline["BTCUSDT"]["features"]
    ).shape[0] - 1
    assert not np.any(np.asarray(built["ETHUSDT"]["primary_valid"])[0])
    assert np.any(np.asarray(built["ETHUSDT"]["stress_valid"])[0])

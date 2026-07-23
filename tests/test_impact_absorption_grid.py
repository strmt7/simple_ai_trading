from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from simple_ai_trading.impact_absorption_grid import (
    ROUND73_GRID_BANDS,
    ROUND73_GRID_FEATURE_NAMES,
    ROUND73_GRID_FEATURE_NAMES_SHA256,
    ROUND73_GRID_STEP_NS,
    Round73CausalGridAccumulator,
    Round73L2State,
    Round73MarkState,
    Round73OpenInterestState,
    _CompensatedNonnegativeTotal,
    round73_grid_feature_invariant_errors,
)


BASE = 1_000_000_000_000
WALL = 1_800_000_000_000_000_000


def _flow(value: float = 0.0):
    return {
        side: {
            band: {
                "added_quote": value,
                "removed_quote": value,
            }
            for band in ROUND73_GRID_BANDS
        }
        for side in ("bid", "ask")
    }


def _l2(timestamp: int, *, width: int = 20) -> Round73L2State:
    return Round73L2State(
        received_monotonic_ns=timestamp,
        bid_prices=tuple(100.0 - index * 0.01 for index in range(width)),
        bid_quantities=tuple(1.0 + index * 0.01 for index in range(width)),
        ask_prices=tuple(100.1 + index * 0.01 for index in range(width)),
        ask_quantities=tuple(1.2 + index * 0.01 for index in range(width)),
        bid_depth_quote_5=1_000.0,
        ask_depth_quote_5=1_200.0,
        bid_depth_quote_10=2_100.0,
        ask_depth_quote_10=2_500.0,
        bid_depth_quote_20=4_500.0,
        ask_depth_quote_20=5_300.0,
        imbalance_5=-0.09,
        imbalance_10=-0.08,
        imbalance_20=-0.07,
        corrected_event_latency_ms=2.0,
    )


def _mark(timestamp: int, anchor_wall_ns: int) -> Round73MarkState:
    return Round73MarkState(
        received_monotonic_ns=timestamp,
        mark_price=100.06,
        index_price=100.04,
        funding_rate=0.0001,
        next_funding_time_ms=anchor_wall_ns // 1_000_000 + 3_600_000,
    )


def test_grid_feature_names_are_unique_and_hash_bound() -> None:
    canonical = json.dumps(
        ROUND73_GRID_FEATURE_NAMES,
        ensure_ascii=True,
        separators=(",", ":"),
    )

    assert len(ROUND73_GRID_FEATURE_NAMES) == len(set(ROUND73_GRID_FEATURE_NAMES))
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == (
        ROUND73_GRID_FEATURE_NAMES_SHA256
    )
    assert "w100ms_normalized_order_flow_imbalance_outside_20" not in (
        ROUND73_GRID_FEATURE_NAMES
    )


def test_compensated_nonnegative_total_resets_expired_terms_without_drift() -> None:
    total = _CompensatedNonnegativeTotal()
    for index in range(100_000):
        value = 6_000_000.0 + (index % 97) * 0.001
        total.add(value)
        total.remove(value)

    assert total.value == 0.0

    active: list[float] = []
    for index in range(10_000):
        value = 1_000_000_000.0 + (index % 31) * 0.01
        total.add(value)
        active.append(value)
        if index % 3 == 0:
            total.remove(active.pop(0))

    assert total.value == math.fsum(active)


def test_grid_emits_finite_causal_vector_with_exact_window_semantics() -> None:
    grid = Round73CausalGridAccumulator("BTCUSDT")
    anchor = BASE + 2_000_000_000
    grid.observe_open_interest(
        Round73OpenInterestState(BASE + 100_000_000, 1_000_000.0)
    )
    grid.observe_mark(_mark(BASE + 200_000_000, WALL))
    grid.observe_l2(state=_l2(BASE + 1_400_000_000), depth_band_flow=_flow())
    grid.observe_bbo(
        received_monotonic_ns=BASE + 1_450_000_000,
        bid=99.9,
        bid_qty=2.0,
        ask=100.0,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )
    grid.observe_bbo(
        received_monotonic_ns=BASE + 1_500_000_000,
        bid=100.0,
        bid_qty=2.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.5,
    )
    grid.observe_trade(
        received_monotonic_ns=BASE + 1_950_000_000,
        price=100.0,
        quantity=2.0,
        buyer_is_maker=False,
    )
    flow = _flow()
    for band in ROUND73_GRID_BANDS:
        flow["bid"][band]["added_quote"] = 10.0
        flow["bid"][band]["removed_quote"] = 2.0
        flow["ask"][band]["added_quote"] = 3.0
        flow["ask"][band]["removed_quote"] = 5.0
    grid.observe_l2(
        state=_l2(BASE + 1_960_000_000),
        depth_band_flow=flow,
    )

    result = grid.emit(anchor_monotonic_ns=anchor, anchor_wall_ns=WALL)

    assert result.valid is True
    assert result.invalid_reason_mask == 0
    assert result.source_max_received_monotonic_ns < anchor
    assert result.signed_aggressive_quote_1s == 200.0
    assert result.absolute_aggressive_quote_1s == 200.0
    assert result.trailing_median_absolute_aggressive_quote_60s is None
    assert result.feature_values is not None
    assert len(result.feature_values) == len(ROUND73_GRID_FEATURE_NAMES)
    assert all(math.isfinite(value) for value in result.feature_values)
    assert round73_grid_feature_invariant_errors(result.feature_values) == ()
    values = dict(zip(ROUND73_GRID_FEATURE_NAMES, result.feature_values, strict=True))
    assert values["w100ms_buy_aggressive_quote"] == 200.0
    assert values["w100ms_sell_aggressive_quote"] == 0.0
    assert values["w100ms_aggregate_trade_count"] == 1.0
    assert values["w100ms_buyer_taker_share"] == 1.0
    assert values["w100ms_bid_added_quote_levels_1_5"] == 10.0
    assert values["w100ms_ask_removed_quote_levels_1_5"] == 5.0
    assert values["w100ms_normalized_order_flow_imbalance_levels_1_5"] == (
        pytest.approx(10.0 / 2_200.0)
    )
    assert values["w100ms_bbo_update_count"] == 0.0
    assert values["w100ms_mean_spread_bps"] == pytest.approx(values["spread_bps"])
    expected_log_return = math.log(100.05) - math.log(99.95)
    assert values["w1000ms_mid_realized_variance"] == pytest.approx(
        expected_log_return * expected_log_return
    )
    assert result.as_dict()["target_constructed"] is False


def test_grid_feature_invariants_reject_impossible_financial_values() -> None:
    grid = Round73CausalGridAccumulator("BTCUSDT")
    anchor = BASE + 2_000_000_000
    grid.observe_open_interest(
        Round73OpenInterestState(BASE + 100_000_000, 1_000_000.0)
    )
    grid.observe_mark(_mark(BASE + 200_000_000, WALL))
    grid.observe_l2(state=_l2(BASE + 1_400_000_000), depth_band_flow=_flow())
    grid.observe_bbo(
        received_monotonic_ns=BASE + 1_500_000_000,
        bid=100.0,
        bid_qty=2.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )
    grid.observe_trade(
        received_monotonic_ns=BASE + 1_900_000_000,
        price=100.0,
        quantity=1.0,
        buyer_is_maker=False,
    )
    result = grid.emit(anchor_monotonic_ns=anchor, anchor_wall_ns=WALL)
    assert result.feature_values is not None
    values = list(result.feature_values)
    values[38] = 2.0

    assert "w100ms_buyer_share_domain" in round73_grid_feature_invariant_errors(
        values
    )


def test_grid_shock_ratio_uses_only_sixty_prior_complete_anchors() -> None:
    grid = Round73CausalGridAccumulator("ETHUSDT")
    last = None
    for index in range(61):
        anchor = BASE + (index + 1) * ROUND73_GRID_STEP_NS
        grid.observe_open_interest(
            Round73OpenInterestState(anchor - 500_000_000, 2_000_000.0)
        )
        grid.observe_mark(_mark(anchor - 400_000_000, WALL + index * 1_000_000_000))
        grid.observe_l2(
            state=_l2(anchor - 300_000_000),
            depth_band_flow=_flow(),
        )
        grid.observe_bbo(
            received_monotonic_ns=anchor - 200_000_000,
            bid=100.0,
            bid_qty=2.0,
            ask=100.1,
            ask_qty=1.0,
            corrected_event_latency_ms=1.0,
        )
        grid.observe_trade(
            received_monotonic_ns=anchor - 100_000_000,
            price=100.0,
            quantity=2.0 if index == 60 else 1.0,
            buyer_is_maker=False,
        )
        last = grid.emit(
            anchor_monotonic_ns=anchor,
            anchor_wall_ns=WALL + index * 1_000_000_000,
        )

    assert last is not None
    assert last.trailing_median_absolute_aggressive_quote_60s == 100.0
    assert last.shock_ratio == 2.0
    assert last.shock_direction == 1
    assert last.shock_direction_taker_share == 1.0


def test_grid_long_horizon_expiry_preserves_financial_invariants() -> None:
    grid = Round73CausalGridAccumulator("SOLUSDT")
    for index in range(3_600):
        anchor = BASE + (index + 1) * ROUND73_GRID_STEP_NS
        grid.observe_open_interest(
            Round73OpenInterestState(anchor - 500_000_000, 2_000_000.0)
        )
        grid.observe_mark(_mark(anchor - 400_000_000, WALL + index * 1_000_000_000))
        flow = _flow(float((index % 17) + 1) * 1_000_000.001)
        grid.observe_l2(
            state=_l2(anchor - 300_000_000),
            depth_band_flow=flow,
        )
        grid.observe_bbo(
            received_monotonic_ns=anchor - 200_000_000,
            bid=100.0,
            bid_qty=2.0,
            ask=100.1,
            ask_qty=1.0,
            corrected_event_latency_ms=1.0,
        )
        if index % 10 in {0, 1}:
            grid.observe_trade(
                received_monotonic_ns=anchor - 100_000_000,
                price=100.0 + (index % 7) * 0.01,
                quantity=60_000.0 + (index % 13) * 0.001,
                buyer_is_maker=index % 2 == 1,
            )
        result = grid.emit(
            anchor_monotonic_ns=anchor,
            anchor_wall_ns=WALL + index * 1_000_000_000,
        )

        assert result.valid is True
        assert result.absolute_aggressive_quote_1s >= 0.0
        assert 0.0 <= result.shock_direction_taker_share <= 1.0
        assert result.feature_values is not None
        assert round73_grid_feature_invariant_errors(result.feature_values) == ()


def test_grid_retains_invalid_anchor_without_feature_vector() -> None:
    grid = Round73CausalGridAccumulator("SOLUSDT")
    grid.observe_bbo(
        received_monotonic_ns=BASE,
        bid=100.0,
        bid_qty=1.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )

    result = grid.emit(
        anchor_monotonic_ns=BASE + 2_000_000_000,
        anchor_wall_ns=WALL,
    )

    assert result.valid is False
    assert result.feature_values is None
    assert "stale_bbo" in result.invalid_reasons
    assert "missing_l2" in result.invalid_reasons
    assert "missing_mark" in result.invalid_reasons
    assert "missing_open_interest" in result.invalid_reasons


def test_grid_rejects_future_receipt_and_noncontiguous_anchor() -> None:
    grid = Round73CausalGridAccumulator("BTCUSDT")
    grid.observe_bbo(
        received_monotonic_ns=BASE,
        bid=100.0,
        bid_qty=1.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )
    with pytest.raises(ValueError, match="strictly follow"):
        grid.emit(anchor_monotonic_ns=BASE, anchor_wall_ns=WALL)

    grid.emit(
        anchor_monotonic_ns=BASE + ROUND73_GRID_STEP_NS,
        anchor_wall_ns=WALL,
    )
    with pytest.raises(ValueError, match="contiguous"):
        grid.emit(
            anchor_monotonic_ns=BASE + 3 * ROUND73_GRID_STEP_NS,
            anchor_wall_ns=WALL + 2 * ROUND73_GRID_STEP_NS,
        )


def test_grid_incomplete_l2_is_invalid_not_an_exception() -> None:
    grid = Round73CausalGridAccumulator("BTCUSDT")
    grid.observe_open_interest(Round73OpenInterestState(BASE, 1_000_000.0))
    grid.observe_mark(_mark(BASE + 1, WALL))
    grid.observe_l2(state=_l2(BASE + 2, width=10), depth_band_flow=_flow())
    grid.observe_bbo(
        received_monotonic_ns=BASE + 3,
        bid=100.0,
        bid_qty=1.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )

    result = grid.emit(
        anchor_monotonic_ns=BASE + ROUND73_GRID_STEP_NS,
        anchor_wall_ns=WALL,
    )

    assert result.valid is False
    assert result.feature_values is None
    assert "incomplete_l2" in result.invalid_reasons


def test_grid_rejects_crossed_or_unordered_l2_geometry() -> None:
    grid = Round73CausalGridAccumulator("BTCUSDT")
    grid.observe_open_interest(Round73OpenInterestState(BASE, 1_000_000.0))
    grid.observe_mark(_mark(BASE + 1, WALL))
    state = _l2(BASE + 2)
    grid.observe_l2(
        state=replace(state, ask_prices=tuple(reversed(state.ask_prices))),
        depth_band_flow=_flow(),
    )
    grid.observe_bbo(
        received_monotonic_ns=BASE + 3,
        bid=100.0,
        bid_qty=1.0,
        ask=100.1,
        ask_qty=1.0,
        corrected_event_latency_ms=1.0,
    )

    result = grid.emit(
        anchor_monotonic_ns=BASE + ROUND73_GRID_STEP_NS,
        anchor_wall_ns=WALL,
    )

    assert result.valid is False
    assert result.feature_values is None
    assert "invalid_l2_geometry" in result.invalid_reasons

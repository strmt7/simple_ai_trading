from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.queue_censored_actions import (
    EXPONENTIAL_FLOW_HALF_LIVES_SECONDS,
    PASSIVE_FILL_BUCKETS_MS,
    PassiveFillRequest,
    build_chunked_queue_censored_inputs,
    build_exponential_flow_features,
    build_passive_fill_result,
)


def test_passive_fill_consumes_post_arrival_queue_and_own_quantity() -> None:
    result = build_passive_fill_result(
        arrival_time_ms=[1_000, 2_000],
        placement_price=[100.0, 100.0],
        queue_ahead_quantity=[2.0, 5.0],
        buyer_is_maker=True,
        order_notional_quote=100.0,
        trade_id=[10, 11, 12, 13, 14, 15],
        trade_time_ms=[1_000, 1_100, 1_200, 2_100, 2_200, 2_300],
        trade_price=[100.0] * 6,
        trade_quantity=[10.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        trade_buyer_is_maker=[True, True, True, True, False, True],
    )

    assert result.filled.tolist() == [True, False]
    assert result.fill_bucket.tolist() == [1, 0]
    assert result.fill_time_ms.tolist() == [1_200, -1]
    assert result.first_matching_trade_id.tolist() == [11, -1]
    assert result.completion_trade_id.tolist() == [12, -1]
    assert result.matching_trade_count.tolist() == [2, 0]
    assert result.printed_quantity_through_fill.tolist() == [3.0, 0.0]
    assert result.required_printed_quantity.tolist() == [3.0, 6.0]
    assert result.summary()["filled_rows"] == 1
    assert len(result.source_trade_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        result.filled[0] = False


def test_passive_fill_requires_exact_price_side_and_expiry() -> None:
    result = build_passive_fill_result(
        arrival_time_ms=[0, 0, 0],
        placement_price=[100.0, 101.0, 102.0],
        queue_ahead_quantity=[0.0, 0.0, 0.0],
        buyer_is_maker=False,
        order_notional_quote=100.0,
        trade_id=[1, 2, 3],
        trade_time_ms=[5_000, 10_000, PASSIVE_FILL_BUCKETS_MS[-1] + 1],
        trade_price=[100.0, 101.0, 102.0],
        trade_quantity=[1.0, 1.0, 1.0],
        trade_buyer_is_maker=[False, True, False],
    )

    assert result.filled.tolist() == [True, False, False]
    assert result.fill_bucket.tolist() == [1, 0, 0]


def test_exponential_flow_is_causal_directional_and_bounded() -> None:
    batch = build_exponential_flow_features(
        decision_time_ms=[1_000, 2_000, 3_000],
        trade_time_ms=[500, 1_500, 2_500, 9_000],
        trade_price=[100.0] * 4,
        trade_quantity=[1.0, 2.0, 4.0, 1_000_000.0],
        trade_buyer_is_maker=[False, True, False, False],
        observation_delay_ms=1_000,
    )

    assert batch.half_lives_seconds == EXPONENTIAL_FLOW_HALF_LIVES_SECONDS
    assert batch.features.shape == (3, 2 * len(EXPONENTIAL_FLOW_HALF_LIVES_SECONDS))
    assert np.all(batch.features[0] == 0.0)
    assert np.all(batch.features[1, 0::2] > 0.0)
    assert np.all(batch.features[1, 1::2] > 0.0)
    assert np.all(batch.features[2, 1::2] < batch.features[1, 1::2])
    assert np.all(np.abs(batch.features[:, 1::2]) <= 1.0)
    assert len(batch.source_trade_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        batch.features[0, 0] = 1.0


def test_empty_trade_windows_are_valid_observed_no_flow() -> None:
    fill = build_passive_fill_result(
        arrival_time_ms=[1_000],
        placement_price=[100.0],
        queue_ahead_quantity=[1.0],
        buyer_is_maker=True,
        order_notional_quote=100.0,
        trade_id=[],
        trade_time_ms=[],
        trade_price=[],
        trade_quantity=[],
        trade_buyer_is_maker=[],
    )
    flow = build_exponential_flow_features(
        decision_time_ms=[1_000, 2_000],
        trade_time_ms=[],
        trade_price=[],
        trade_quantity=[],
        trade_buyer_is_maker=[],
    )

    assert fill.filled.tolist() == [False]
    assert np.all(flow.features == 0.0)


def test_source_contract_rejects_lossy_time_and_side_coercion() -> None:
    common = {
        "arrival_time_ms": [1_000],
        "placement_price": [100.0],
        "queue_ahead_quantity": [1.0],
        "buyer_is_maker": True,
        "order_notional_quote": 100.0,
        "trade_id": [1],
        "trade_time_ms": [1_100],
        "trade_price": [100.0],
        "trade_quantity": [2.0],
        "trade_buyer_is_maker": [True],
    }
    with pytest.raises(ValueError, match="arrival times"):
        build_passive_fill_result(**{**common, "arrival_time_ms": [1_000.5]})
    with pytest.raises(ValueError, match="trade sides"):
        build_passive_fill_result(**{**common, "trade_buyer_is_maker": [1]})
    with pytest.raises(ValueError, match="execution contract"):
        build_passive_fill_result(**{**common, "order_notional_quote": "100"})
    with pytest.raises(ValueError, match="exponential-flow contract"):
        build_exponential_flow_features(
            decision_time_ms=[1_000],
            trade_time_ms=[],
            trade_price=[],
            trade_quantity=[],
            trade_buyer_is_maker=[],
            observation_delay_ms=999,
        )


def test_passive_fill_kernel_matches_independent_order_replay() -> None:
    generator = np.random.default_rng(5701)
    trade_rows = 600
    candidate_rows = 150
    trade_id = generator.permutation(np.arange(1, trade_rows + 1, dtype=np.int64))
    trade_time_ms = generator.integers(1, 60_000, size=trade_rows, dtype=np.int64)
    trade_price = generator.choice([100.0, 101.0, 102.0, 103.0], trade_rows)
    trade_quantity = generator.uniform(0.05, 2.0, size=trade_rows)
    trade_side = generator.integers(0, 2, size=trade_rows).astype(np.bool_)
    arrivals = np.sort(
        generator.integers(0, 45_000, size=candidate_rows, dtype=np.int64)
    )
    prices = generator.choice([100.0, 101.0, 102.0, 104.0], candidate_rows)
    queues = generator.uniform(0.0, 4.0, size=candidate_rows)
    result = build_passive_fill_result(
        arrival_time_ms=arrivals,
        placement_price=prices,
        queue_ahead_quantity=queues,
        buyer_is_maker=True,
        order_notional_quote=100.0,
        trade_id=trade_id,
        trade_time_ms=trade_time_ms,
        trade_price=trade_price,
        trade_quantity=trade_quantity,
        trade_buyer_is_maker=trade_side,
    )

    for row in range(candidate_rows):
        indexes = np.flatnonzero(
            trade_side
            & (trade_price == prices[row])
            & (trade_time_ms > arrivals[row])
            & (trade_time_ms <= arrivals[row] + PASSIVE_FILL_BUCKETS_MS[-1])
        )
        indexes = indexes[
            np.lexsort((trade_id[indexes], trade_time_ms[indexes]))
        ]
        cumulative = np.cumsum(trade_quantity[indexes])
        required = queues[row] + 100.0 / prices[row]
        completion = int(np.searchsorted(cumulative, required, side="left"))
        expected_fill = completion < indexes.size
        assert bool(result.filled[row]) is expected_fill
        if not expected_fill:
            continue
        completed_index = indexes[completion]
        delay = trade_time_ms[completed_index] - arrivals[row]
        expected_bucket = int(
            np.searchsorted(PASSIVE_FILL_BUCKETS_MS, delay, side="left") + 1
        )
        assert int(result.fill_bucket[row]) == expected_bucket
        assert int(result.fill_time_ms[row]) == int(trade_time_ms[completed_index])
        assert int(result.first_matching_trade_id[row]) == int(trade_id[indexes[0]])
        assert int(result.completion_trade_id[row]) == int(trade_id[completed_index])
        assert int(result.matching_trade_count[row]) == completion + 1
        assert result.printed_quantity_through_fill[row] == pytest.approx(
            cumulative[completion]
        )


def test_queue_censored_boundaries_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="arrival times"):
        build_passive_fill_result(
            arrival_time_ms=[2, 1],
            placement_price=[1.0, 1.0],
            queue_ahead_quantity=[0.0, 0.0],
            buyer_is_maker=True,
            order_notional_quote=1.0,
            trade_id=[1],
            trade_time_ms=[1],
            trade_price=[1.0],
            trade_quantity=[1.0],
            trade_buyer_is_maker=[True],
        )
    with pytest.raises(ValueError, match="exponential-flow contract"):
        build_exponential_flow_features(
            decision_time_ms=[1, 1],
            trade_time_ms=[0],
            trade_price=[1.0],
            trade_quantity=[1.0],
            trade_buyer_is_maker=[True],
        )


def test_chunked_inputs_match_monolithic_flow_and_fill_results() -> None:
    decisions = np.asarray([5_000, 10_000, 35_000, 40_000], dtype=np.int64)
    trade_id = np.arange(1, 13, dtype=np.int64)
    trade_time_ms = np.asarray(
        [
            1_000,
            6_000,
            7_000,
            11_000,
            12_000,
            29_000,
            31_000,
            36_000,
            37_000,
            41_000,
            42_000,
            59_000,
        ],
        dtype=np.int64,
    )
    trade_price = np.asarray(
        [100.0, 100.0, 101.0, 100.0, 101.0, 102.0] * 2,
        dtype=np.float64,
    )
    trade_quantity = np.asarray(
        [0.25, 2.0, 2.0, 2.0, 2.0, 0.5] * 2,
        dtype=np.float64,
    )
    trade_side = np.asarray(
        [True, True, False, True, False, True] * 2,
        dtype=np.bool_,
    )
    requests = (
        PassiveFillRequest(
            name="base_long",
            buyer_is_maker=True,
            arrival_time_ms=decisions + 750,
            placement_price=np.full(decisions.size, 100.0),
            queue_ahead_quantity=np.full(decisions.size, 0.5),
        ),
        PassiveFillRequest(
            name="base_short",
            buyer_is_maker=False,
            arrival_time_ms=decisions + 750,
            placement_price=np.full(decisions.size, 101.0),
            queue_ahead_quantity=np.full(decisions.size, 0.5),
        ),
        PassiveFillRequest(
            name="stress_long",
            buyer_is_maker=True,
            arrival_time_ms=decisions + 1_500,
            placement_price=np.full(decisions.size, 100.0),
            queue_ahead_quantity=np.full(decisions.size, 0.5),
        ),
        PassiveFillRequest(
            name="stress_short",
            buyer_is_maker=False,
            arrival_time_ms=decisions + 1_500,
            placement_price=np.full(decisions.size, 101.0),
            queue_ahead_quantity=np.full(decisions.size, 0.5),
        ),
    )
    loader_calls: list[tuple[int, int]] = []

    def load_chunk(start_ms: int, end_ms: int):
        loader_calls.append((start_ms, end_ms))
        selected = (trade_time_ms >= start_ms) & (trade_time_ms < end_ms)
        return {
            "trade_id": trade_id[selected],
            "trade_time_ms": trade_time_ms[selected],
            "trade_price": trade_price[selected],
            "trade_quantity": trade_quantity[selected],
            "trade_buyer_is_maker": trade_side[selected],
        }

    batch = build_chunked_queue_censored_inputs(
        decision_time_ms=decisions,
        fill_requests=requests,
        source_chunks=((0, 30_000), (30_000, 60_000)),
        load_trade_chunk=load_chunk,
        order_notional_quote=100.0,
    )
    monolithic_flow = build_exponential_flow_features(
        decision_time_ms=decisions,
        trade_time_ms=trade_time_ms,
        trade_price=trade_price,
        trade_quantity=trade_quantity,
        trade_buyer_is_maker=trade_side,
    )

    assert loader_calls == [(0, 30_000), (30_000, 60_000)]
    np.testing.assert_array_equal(batch.flow.features, monolithic_flow.features)
    for request in requests:
        expected = build_passive_fill_result(
            arrival_time_ms=request.arrival_time_ms,
            placement_price=request.placement_price,
            queue_ahead_quantity=request.queue_ahead_quantity,
            buyer_is_maker=request.buyer_is_maker,
            order_notional_quote=100.0,
            trade_id=trade_id,
            trade_time_ms=trade_time_ms,
            trade_price=trade_price,
            trade_quantity=trade_quantity,
            trade_buyer_is_maker=trade_side,
        )
        actual = batch.fill(request.name)
        for field in (
            "filled",
            "fill_bucket",
            "fill_time_ms",
            "first_matching_trade_id",
            "completion_trade_id",
            "matching_trade_count",
            "printed_quantity_through_fill",
        ):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_chunked_inputs_reject_fill_windows_crossing_chunk_boundaries() -> None:
    request = PassiveFillRequest(
        name="crossing_long",
        buyer_is_maker=True,
        arrival_time_ms=[20_000],
        placement_price=[100.0],
        queue_ahead_quantity=[0.0],
    )

    def load_empty(_start_ms: int, _end_ms: int):
        return {
            "trade_id": [],
            "trade_time_ms": [],
            "trade_price": [],
            "trade_quantity": [],
            "trade_buyer_is_maker": [],
        }

    with pytest.raises(ValueError, match="cross a chunk"):
        build_chunked_queue_censored_inputs(
            decision_time_ms=[19_000],
            fill_requests=[request],
            source_chunks=((0, 30_000), (30_000, 60_000)),
            load_trade_chunk=load_empty,
            order_notional_quote=100.0,
        )

from __future__ import annotations

import math

import numpy as np
import pytest

from simple_ai_trading.microstructure_features import MICROSTRUCTURE_FEATURE_NAMES
from simple_ai_trading.microstructure_live import (
    LateMicrostructureEventError,
    MicrostructureFeedIntegrityError,
    LiveMicrostructureSecondAggregator,
    StreamingMicrostructureCoordinator,
)
from simple_ai_trading.microstructure_model import MicrostructureActionPrediction


def _quote(second_ms: int, index: int, *, event_offset_ms: int = 100) -> dict[str, object]:
    mid = 50_000.0 + index * 0.2 + math.sin(index / 19.0)
    return {
        "e": "bookTicker",
        "E": second_ms + event_offset_ms,
        "T": second_ms + event_offset_ms - 8,
        "u": index + 1,
        "s": "BTCUSDT",
        "b": f"{mid - 0.05:.8f}",
        "B": "2.0",
        "a": f"{mid + 0.05:.8f}",
        "A": "2.5",
    }


def _trade(second_ms: int, index: int) -> dict[str, object]:
    return {
        "e": "trade",
        "E": second_ms + 150,
        "T": second_ms + 140,
        "t": index + 1,
        "s": "BTCUSDT",
        "p": f"{50_000.0 + index * 0.2:.8f}",
        "q": "0.02",
        "m": bool(index % 2),
    }


def _agg_trade(second_ms: int, index: int) -> dict[str, object]:
    first_trade_id = index * 3 + 1
    return {
        "e": "aggTrade",
        "E": second_ms + 150,
        "T": second_ms + 140,
        "a": index + 1,
        "f": first_trade_id,
        "l": first_trade_id + 2,
        "s": "BTCUSDT",
        "p": f"{50_000.0 + index * 0.2:.8f}",
        "q": "0.06",
        "m": bool(index % 2),
    }


def test_live_aggregator_orders_quotes_by_event_time_and_rejects_late_events() -> None:
    aggregator = LiveMicrostructureSecondAggregator("BTCUSDT", settlement_delay_ms=100)
    second_ms = 1_700_000_000_000
    late_close = _quote(second_ms, 2, event_offset_ms=700)
    early_open = _quote(second_ms, 1, event_offset_ms=100)
    aggregator.ingest(late_close)
    aggregator.ingest(early_open)
    aggregator.ingest(_trade(second_ms, 1))

    rows = aggregator.drain(second_ms + 1_100)

    assert len(rows) == 1
    row = rows[0]
    assert row.quote_updates == 2
    assert row.open_mid < row.close_mid
    assert row.trade_count == 1
    assert row.base_volume == pytest.approx(0.02)
    assert row.event_delay_p50_ms == pytest.approx(8.0)
    with pytest.raises(LateMicrostructureEventError, match="finalized second"):
        aggregator.ingest(_quote(second_ms, 3, event_offset_ms=900))
    assert aggregator.late_event_count == 1
    assert aggregator.integrity_reset_count == 1


def test_live_aggregator_deduplicates_exact_replay_and_rejects_conflict() -> None:
    aggregator = LiveMicrostructureSecondAggregator("BTCUSDT", settlement_delay_ms=100)
    second_ms = 1_700_000_000_000
    quote = _quote(second_ms, 1)
    trade = _trade(second_ms, 1)
    aggregator.ingest(quote)
    aggregator.ingest(quote)
    aggregator.ingest(trade)
    aggregator.ingest(trade)

    rows = aggregator.drain(second_ms + 1_100)

    assert len(rows) == 1
    assert rows[0].quote_updates == 1
    assert rows[0].trade_count == 1
    assert rows[0].base_volume == pytest.approx(0.02)
    assert aggregator.duplicate_event_count == 2

    conflicting = _quote(second_ms + 2_000, 3)
    aggregator.ingest(conflicting)
    with pytest.raises(MicrostructureFeedIntegrityError, match="conflicting duplicate"):
        aggregator.ingest({**conflicting, "B": "3.0"})
    assert aggregator.integrity_reset_count == 1


def test_live_aggregator_recovers_underlying_fill_count_from_aggregate_trade() -> None:
    aggregator = LiveMicrostructureSecondAggregator("BTCUSDT", settlement_delay_ms=100)
    second_ms = 1_700_000_000_000
    aggregator.ingest(_quote(second_ms, 1))
    aggregator.ingest(_agg_trade(second_ms, 1))

    rows = aggregator.drain(second_ms + 1_100)

    assert len(rows) == 1
    assert rows[0].trade_count == 3
    assert rows[0].base_volume == pytest.approx(0.06)


def test_live_aggregator_exposes_newest_unclosed_quote_for_execution_gates() -> None:
    aggregator = LiveMicrostructureSecondAggregator("BTCUSDT", settlement_delay_ms=100)
    second_ms = 1_700_000_000_000
    aggregator.ingest(_quote(second_ms, 1, event_offset_ms=900))
    assert aggregator.drain(second_ms + 1_100)
    next_quote = _quote(second_ms + 1_000, 2, event_offset_ms=50)

    aggregator.ingest(next_quote)
    current = aggregator.current_quote()

    assert current is not None
    assert current.event_time_ms == second_ms + 1_050
    assert current.update_id == 3
    assert current.bid == pytest.approx(float(next_quote["b"]))


class _Scorer:
    symbol = "BTCUSDT"
    decision_cadence_seconds = 5
    total_latency_ms = 500
    max_quote_age_ms = 1_000

    def score(self, features, **kwargs) -> MicrostructureActionPrediction:
        assert np.asarray(features).shape == (len(MICROSTRUCTURE_FEATURE_NAMES),)
        assert kwargs["decision_time_ms"] % 5_000 == 0
        assert kwargs["close_bid_qty"] > 0.0
        assert kwargs["quote_time_ms"] <= kwargs["observation_time_ms"]
        return MicrostructureActionPrediction(
            side="FLAT",
            long_expected_net_bps=0.0,
            short_expected_net_bps=0.0,
            long_profitable_probability=0.5,
            short_profitable_probability=0.5,
            minimum_predicted_edge_bps=1.0,
            minimum_profitable_probability=0.6,
            long_l1_participation=0.01,
            short_l1_participation=0.01,
            reason="test",
        )


class _FailingScorer(_Scorer):
    def score(self, features, **kwargs) -> MicrostructureActionPrediction:
        raise RuntimeError("model unavailable")


class _Clock:
    def __init__(self) -> None:
        self._values: list[int] = []
        self._fallback = 0

    def set_sequence(self, *values: int) -> None:
        self._values = list(values)

    def __call__(self) -> int:
        if self._values:
            return self._values.pop(0)
        return self._fallback


def _warm_coordinator(
    coordinator: StreamingMicrostructureCoordinator,
    *,
    count: int = 3_604,
) -> int:
    base_ms = 1_700_000_000_000
    for index in range(count):
        second_ms = base_ms + index * 1_000
        coordinator.ingest(_quote(second_ms, index))
        coordinator.ingest(_trade(second_ms, index))
        assert coordinator.evaluate_ready(
            exchange_now_ms=second_ms + 1_100,
            order_notional_quote=500.0,
        ) == ()
    return base_ms + (count - 1) * 1_000


def test_streaming_coordinator_preserves_warmup_cadence_and_latency_budget() -> None:
    coordinator = StreamingMicrostructureCoordinator(_Scorer(), settlement_delay_ms=100)
    base_ms = 1_700_000_000_000
    predictions = []
    for index in range(3_615):
        second_ms = base_ms + index * 1_000
        coordinator.ingest(_quote(second_ms, index))
        coordinator.ingest(_trade(second_ms, index))
        predictions.extend(
            coordinator.evaluate_ready(
                exchange_now_ms=second_ms + 1_100,
                order_notional_quote=500.0,
            )
        )

    assert predictions
    assert all(item.feature_row.decision_time_ms % 5_000 == 0 for item in predictions)
    assert all(0 < item.remaining_latency_budget_ms <= 400 for item in predictions)
    assert all(item.execution_quote.event_time_ms > 0 for item in predictions)
    assert coordinator.deadline_misses == 0
    assert coordinator.engine.gap_resets == 0


def test_streaming_coordinator_resets_on_invalid_feed_and_fails_closed_on_scorer_error() -> None:
    coordinator = StreamingMicrostructureCoordinator(_FailingScorer(), settlement_delay_ms=100)
    last_second_ms = _warm_coordinator(coordinator)
    assert coordinator.engine.ready
    next_second_ms = last_second_ms + 1_000
    coordinator.ingest(_quote(next_second_ms, 3_604))
    coordinator.ingest(_trade(next_second_ms, 3_604))
    predictions = coordinator.evaluate_ready(
        exchange_now_ms=next_second_ms + 1_100,
        order_notional_quote=500.0,
    )
    assert predictions == ()
    assert coordinator.inference_failures == 1
    assert coordinator.last_inference_error == "RuntimeError: model unavailable"

    with pytest.raises(MicrostructureFeedIntegrityError, match="maker flag"):
        coordinator.ingest({**_trade(next_second_ms + 1_000, 3_605), "m": "false"})
    assert coordinator.feed_integrity_resets == 1
    assert coordinator.engine.warmup_remaining_seconds == 3_601


def test_streaming_coordinator_rejects_prediction_that_finishes_after_deadline() -> None:
    clock = _Clock()
    coordinator = StreamingMicrostructureCoordinator(
        _Scorer(),
        settlement_delay_ms=100,
        monotonic_ns=clock,
    )
    last_second_ms = _warm_coordinator(coordinator)
    next_second_ms = last_second_ms + 1_000
    coordinator.ingest(_quote(next_second_ms, 3_604))
    coordinator.ingest(_trade(next_second_ms, 3_604))
    clock.set_sequence(0, 450_000_000)

    predictions = coordinator.evaluate_ready(
        exchange_now_ms=next_second_ms + 1_100,
        order_notional_quote=500.0,
    )

    assert predictions == ()
    assert coordinator.deadline_misses == 1
    assert coordinator.post_inference_deadline_misses == 1


def test_streaming_coordinator_validates_notional_before_consuming_ready_rows() -> None:
    coordinator = StreamingMicrostructureCoordinator(_Scorer(), settlement_delay_ms=100)
    last_second_ms = _warm_coordinator(coordinator)
    next_second_ms = last_second_ms + 1_000
    coordinator.ingest(_quote(next_second_ms, 3_604))
    coordinator.ingest(_trade(next_second_ms, 3_604))

    with pytest.raises(ValueError, match="finite and positive"):
        coordinator.evaluate_ready(
            exchange_now_ms=next_second_ms + 1_100,
            order_notional_quote=0.0,
        )
    assert coordinator.engine.ready

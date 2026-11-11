from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simple_ai_trading.polymarket_round21_live_features as live_module
from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_redundant_union import PolymarketClobLaneReceipt
from simple_ai_trading.polymarket_round21_prospective import (
    Round21ProspectivePrediction,
    Round21ProspectiveScorer,
)
from simple_ai_trading.polymarket_round21_live_features import (
    Round21LiveFeatureCoordinator,
    Round21PublicSourceGap,
)

from polymarket_round21_support import round21_replay_condition, sha


MARKET = round21_replay_condition().market


def _scorer(*, layer: str = "core") -> Mock:
    scorer = Mock(spec=Round21ProspectiveScorer)
    scorer.population_layer = layer
    return scorer


def _prediction(*, status: str = "observed") -> Mock:
    prediction = Mock(spec=Round21ProspectivePrediction)
    prediction.validated.return_value = prediction
    prediction.status = status
    prediction.reason = "" if status == "observed" else "optional_unavailable"
    prediction.condition_id = MARKET.condition_id
    prediction.event_start_ms = MARKET.event_start_ms
    prediction.decision_time_ms = MARKET.event_start_ms + 1_000
    prediction.prediction_sha256 = sha(f"prediction-{status}")
    prediction.asdict.return_value = {
        "prediction_sha256": prediction.prediction_sha256,
    }
    return prediction


class _Core:
    def __init__(self, **_kwargs) -> None:
        self.events = []
        self.records = []
        self.epochs = []
        self.snapshot = SimpleNamespace(
            available=True,
            reasons=(),
        )

    def ingest_union_event(self, event) -> None:
        self.events.append(event)

    def start_chainlink_epoch(self, connection, *, first_sequence_number=1) -> None:
        self.epochs.append((connection, first_sequence_number))

    def ingest_chainlink_record(self, record) -> None:
        self.records.append(record)

    def build(self, _decision):
        return self.snapshot


class _Optional:
    def __init__(self) -> None:
        self.resets = []
        self.records = []
        self.snapshot = SimpleNamespace(
            spot_available=False,
            usdm_available=False,
        )

    def reset_market(self, market, connection) -> None:
        self.resets.append((market, connection))

    def ingest_record(self, record) -> None:
        self.records.append(record)

    def build(self, _decision):
        return self.snapshot


class _Union:
    def __init__(self, **_kwargs) -> None:
        self.receipts = []
        self.watermarks = []

    def add(self, receipt):
        self.receipts.append(receipt)
        return ()

    def advance(self, watermark):
        self.watermarks.append(watermark)
        return ()


@pytest.fixture
def boundaries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(live_module, "Round21CoreFeatureEngine", _Core)
    monkeypatch.setattr(
        live_module,
        "Round21IndependentBinanceFeatureEngine",
        _Optional,
    )
    monkeypatch.setattr(live_module, "PolymarketRedundantUnionBuilder", _Union)
    monkeypatch.setattr(live_module, "join_round21_causal_features", Mock())


def test_core_gap_permanently_blocks_current_market_entry(boundaries) -> None:
    scorer = _scorer()
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)
    gap = Round21PublicSourceGap(
        stream="polymarket_rtds",
        connection_id="rtds:chainlink:btc:gap",
        observed_wall_ms=MARKET.event_start_ms + 500,
        observed_monotonic_ns=500,
        last_sequence_number=9,
        reason="connection_lost",
    )
    coordinator.record_gap(gap)

    result = coordinator.evaluate(
        decision_time_ms=MARKET.event_start_ms + 1_000,
        observed_at_ms=MARKET.event_start_ms + 1_010,
        observed_monotonic_ns=1_000,
    )

    assert result.status == "abstain"
    assert result.reasons == ("core_source_gap:polymarket_rtds",)
    assert result.core_gap_sha256 == (gap.gap_sha256,)
    assert result.core_source_healthy is False
    assert result.prediction is None
    assert scorer.evaluate.call_count == 0
    assert result.live_trading_authority is False


def test_optional_gap_does_not_block_core_model(boundaries, monkeypatch) -> None:
    scorer = _scorer(layer="core")
    prediction = _prediction()
    scorer.evaluate.return_value = prediction
    monkeypatch.setattr(live_module, "join_round21_causal_features", lambda *_: object())
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)
    coordinator.record_gap(
        Round21PublicSourceGap(
            stream="binance_spot",
            connection_id="binance:spot:gap",
            observed_wall_ms=MARKET.event_start_ms + 500,
            observed_monotonic_ns=500,
            last_sequence_number=4,
            reason="connection_lost",
        )
    )

    result = coordinator.evaluate(
        decision_time_ms=MARKET.event_start_ms + 1_000,
        observed_at_ms=MARKET.event_start_ms + 1_010,
        observed_monotonic_ns=1_000,
    )

    assert result.status == "observed"
    assert result.reasons == ()
    assert result.core_source_healthy is True
    assert result.optional_source_healthy is True
    assert result.prediction is prediction
    assert scorer.evaluate.call_count == 1
    assert coordinator.binance_execution_connected is False


def test_selected_optional_layer_abstains_without_affecting_core_health(
    boundaries,
    monkeypatch,
) -> None:
    scorer = _scorer(layer="core_spot")
    prediction = _prediction(status="abstain")
    scorer.evaluate.return_value = prediction
    monkeypatch.setattr(live_module, "join_round21_causal_features", lambda *_: object())
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)

    result = coordinator.evaluate(
        decision_time_ms=MARKET.event_start_ms + 1_000,
        observed_at_ms=MARKET.event_start_ms + 1_010,
        observed_monotonic_ns=1_000,
    )

    assert result.status == "abstain"
    assert result.reasons == ("optional_unavailable",)
    assert result.core_source_healthy is True
    assert result.optional_source_healthy is False
    assert result.prediction is prediction


def test_duplicate_decision_is_idempotent_and_regression_fails(boundaries) -> None:
    scorer = _scorer()
    prediction = _prediction()
    scorer.evaluate.return_value = prediction
    live_module.join_round21_causal_features.return_value = object()
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)

    first = coordinator.evaluate(
        decision_time_ms=MARKET.event_start_ms + 1_000,
        observed_at_ms=MARKET.event_start_ms + 1_010,
        observed_monotonic_ns=1_000,
    )
    duplicate = coordinator.evaluate(
        decision_time_ms=MARKET.event_start_ms + 1_000,
        observed_at_ms=MARKET.event_start_ms + 2_000,
        observed_monotonic_ns=2_000,
    )

    assert duplicate is first
    assert scorer.evaluate.call_count == 1
    with pytest.raises(ValueError, match="decision time regressed"):
        coordinator.evaluate(
            decision_time_ms=MARKET.event_start_ms + 750,
            observed_at_ms=MARKET.event_start_ms + 2_000,
            observed_monotonic_ns=2_000,
        )


@pytest.mark.parametrize(
    ("stream", "layer", "revocation"),
    [
        ("clob_market", "core", "core_source_gap:clob_market"),
        ("binance_spot", "core_spot", "optional_source_epoch_changed"),
        ("binance_spot", "core", None),
    ],
)
def test_feed_gap_is_processed_during_scoring(boundaries, stream, layer, revocation):
    entered, release = Event(), Event()
    scorer = _scorer(layer=layer)
    prediction = _prediction()
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)
    coordinator._optional.snapshot.spot_available = True
    gap = Round21PublicSourceGap(
        stream=stream,
        connection_id="fixture:gap",
        observed_wall_ms=MARKET.event_start_ms + 1_020,
        observed_monotonic_ns=1_020,
        last_sequence_number=1,
        reason="fixture_gap",
    )
    observation = {
        "decision_time_ms": MARKET.event_start_ms + 1_000,
        "observed_at_ms": MARKET.event_start_ms + 1_010,
        "observed_monotonic_ns": 1_000,
    }

    def score(*_args, **_kwargs):
        entered.set()
        assert release.wait(4)
        return prediction

    scorer.evaluate.side_effect = score
    with ThreadPoolExecutor(max_workers=2) as pool:
        scoring = pool.submit(coordinator.evaluate, **observation)
        try:
            assert entered.wait(2)
            pool.submit(coordinator.record_gap, gap).result(timeout=1)
            assert not scoring.done()
        finally:
            release.set()
        outcome = scoring.result(timeout=2)

    if revocation is None:
        assert outcome.status == "observed"
        assert coordinator.admits(outcome)
    else:
        assert outcome.status == "abstain" and outcome.prediction is None
        assert outcome.reasons == (revocation,)
        assert not coordinator.admits(outcome)
    assert coordinator.evaluate(**observation) is outcome
    assert scorer.evaluate.call_count == 1

    if layer == "core_spot":
        next_prediction = _prediction()
        next_prediction.decision_time_ms += 250
        scorer.evaluate.side_effect = None
        scorer.evaluate.return_value = next_prediction
        coordinator._optional.snapshot.spot_available = True
        next_outcome = coordinator.evaluate(
            decision_time_ms=next_prediction.decision_time_ms,
            observed_at_ms=next_prediction.decision_time_ms + 10,
            observed_monotonic_ns=1_300,
        )
        assert next_outcome.status == "observed"
        scorer.reset_history.assert_called_once()


def test_cached_forecast_cannot_survive_later_core_gap(boundaries):
    scorer = _scorer()
    scorer.evaluate.return_value = _prediction()
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)
    observation = {
        "decision_time_ms": MARKET.event_start_ms + 1_000,
        "observed_at_ms": MARKET.event_start_ms + 1_010,
        "observed_monotonic_ns": 1_000,
    }
    cached = coordinator.evaluate(**observation)
    assert coordinator.admits(cached)
    gap = Round21PublicSourceGap(
        stream="polymarket_rtds",
        connection_id="fixture:gap",
        observed_wall_ms=MARKET.event_start_ms + 1_025,
        observed_monotonic_ns=1_025,
        last_sequence_number=1,
        reason="fixture_gap",
    )
    coordinator.record_gap(gap)
    assert not coordinator.admits(cached)
    current = coordinator.evaluate(**observation)
    assert current.status == "abstain"
    assert current.prediction is None and current.core_gap_sha256 == (gap.gap_sha256,)
    scorer.evaluate.assert_called_once()


def test_gap_and_decision_evidence_reject_tampering(boundaries) -> None:
    with pytest.raises(ValueError, match="gap hash differs"):
        Round21PublicSourceGap(
            stream="clob_market",
            connection_id="clob-a:connection",
            observed_wall_ms=MARKET.event_start_ms,
            observed_monotonic_ns=1,
            last_sequence_number=1,
            reason="disconnect",
            gap_sha256="f" * 64,
        )


def test_ingress_serializes_core_epochs_and_resets_optional_state(boundaries) -> None:
    scorer = _scorer()
    coordinator = Round21LiveFeatureCoordinator(market=MARKET, scorer=scorer)
    clob = PolymarketClobLaneReceipt(
        lane_id="clob-a",
        connection_id="clob-a:connection",
        sequence_number=1,
        received_wall_ms=MARKET.event_start_ms + 100,
        received_monotonic_ns=100,
        raw_text="PONG",
    )
    coordinator.ingest_clob_receipt(clob)
    first = CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id="rtds:chainlink:btc:first",
        sequence_number=7,
        received_wall_ms=MARKET.event_start_ms + 200,
        received_monotonic_ns=200,
        raw_text="PING",
    )
    second = CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id=first.connection_id,
        sequence_number=8,
        received_wall_ms=MARKET.event_start_ms + 300,
        received_monotonic_ns=300,
        raw_text="PONG",
    )
    replacement = CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id="rtds:chainlink:btc:replacement",
        sequence_number=1,
        received_wall_ms=MARKET.event_start_ms + 400,
        received_monotonic_ns=400,
        raw_text="PING",
    )
    coordinator.ingest_chainlink_record(first)
    coordinator.ingest_chainlink_record(second)
    coordinator.ingest_chainlink_record(replacement)

    assert coordinator._union.receipts == [clob]
    assert coordinator._union.watermarks == [200, 300, 400]
    assert coordinator._core.epochs == [
        (first.connection_id, 7),
        (replacement.connection_id, 1),
    ]
    assert coordinator._core.records == [first, second, replacement]

    optional = CaptureFrameRecord(
        stream="binance_spot",
        connection_id="binance:spot:first",
        sequence_number=1,
        received_wall_ms=MARKET.event_start_ms + 500,
        received_monotonic_ns=500,
        raw_text="{}",
    )
    coordinator.ingest_optional_binance_record(optional)
    assert coordinator._optional.resets == [("spot", optional.connection_id)]
    prior_optional = coordinator._optional
    coordinator.record_gap(
        Round21PublicSourceGap(
            stream="binance_spot",
            connection_id=optional.connection_id,
            observed_wall_ms=MARKET.event_start_ms + 600,
            observed_monotonic_ns=600,
            last_sequence_number=1,
            reason="disconnect",
        )
    )
    assert coordinator._optional is not prior_optional
    assert coordinator._optional_connections == {}


def test_ingress_rejects_lane_mismatch_and_core_clock_regression(boundaries) -> None:
    coordinator = Round21LiveFeatureCoordinator(
        market=MARKET,
        scorer=_scorer(),
    )
    with pytest.raises(ValueError, match="lane and connection differ"):
        coordinator.ingest_clob_receipt(
            PolymarketClobLaneReceipt(
                lane_id="clob-a",
                connection_id="clob-b:connection",
                sequence_number=1,
                received_wall_ms=MARKET.event_start_ms + 100,
                received_monotonic_ns=100,
                raw_text="PONG",
            )
        )
    coordinator.ingest_clob_receipt(
        PolymarketClobLaneReceipt(
            lane_id="clob-a",
            connection_id="clob-a:connection",
            sequence_number=1,
            received_wall_ms=MARKET.event_start_ms + 200,
            received_monotonic_ns=200,
            raw_text="PONG",
        )
    )
    with pytest.raises(ValueError, match="receipt order regressed"):
        coordinator.ingest_chainlink_record(
            CaptureFrameRecord(
                stream="polymarket_rtds",
                connection_id="rtds:chainlink:btc:first",
                sequence_number=1,
                received_wall_ms=MARKET.event_start_ms + 100,
                received_monotonic_ns=100,
                raw_text="PING",
            )
        )
    with pytest.raises(ValueError, match="record metadata differs"):
        coordinator.ingest_chainlink_record(
            CaptureFrameRecord(
                stream="binance_spot",
                connection_id="binance:spot:first",
                sequence_number=1,
                received_wall_ms=MARKET.event_start_ms + 300,
                received_monotonic_ns=300,
                raw_text="{}",
            )
        )

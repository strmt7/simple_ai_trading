from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_historical_shadow import (
    PolymarketHistoricalShadowDecision,
)
from simple_ai_trading.polymarket_historical_shadow_opportunity import (
    PolymarketShadowMarketState,
    evaluate_shadow_settlement_opportunity,
)


EVENT_START_MS = 1_782_086_700_000
DECISION_TIME_MS = EVENT_START_MS + 120_000
OBSERVED_AT_MS = DECISION_TIME_MS + 100


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="123",
        condition_id="0x" + "1" * 64,
        slug=f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        question="Bitcoin Up or Down?",
        event_start_ms=EVENT_START_MS,
        end_ms=EVENT_START_MS + 300_000,
        up_token_id="2" * 64,
        down_token_id="3" * 64,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("1000000"),
        resolution_source=(
            "https://data.chain.link/streams/btc-usd"
        ),
        gamma_payload_sha256="4" * 64,
        gamma_payload_json="{}",
    )


def _book(
    market: PolymarketFiveMinuteMarket,
    *,
    token_id: str,
    ask: Decimal,
) -> PaperBookSnapshot:
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=token_id,
        bids=(BookLevel(ask - Decimal("0.02"), Decimal("100")),),
        asks=(
            BookLevel(ask, Decimal("3")),
            BookLevel(ask + Decimal("0.01"), Decimal("100")),
        ),
        source_time_ms=OBSERVED_AT_MS - 25,
        received_wall_ms=OBSERVED_AT_MS,
        received_monotonic_ns=100,
        source_payload_sha256="5" * 64,
    ).validated()


def _state() -> PolymarketShadowMarketState:
    market = _market()
    return PolymarketShadowMarketState(
        market=market,
        up_book=_book(market, token_id=market.up_token_id, ask=Decimal("0.55")),
        down_book=_book(
            market,
            token_id=market.down_token_id,
            ask=Decimal("0.48"),
        ),
        clob_market_info_sha256="6" * 64,
        general_order_delay_seconds=0,
        taker_order_delay_enabled=False,
        observed_at_ms=OBSERVED_AT_MS,
    )


def _prediction(probability: float = 0.70) -> PolymarketHistoricalShadowDecision:
    return PolymarketHistoricalShadowDecision(
        status="observed",
        reason="",
        event_start_ms=EVENT_START_MS,
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS,
        probability_up=probability,
        candidate_id="lgbm-depth2-leaves3",
        pretest_artifact_sha256="7" * 64,
        evaluation_artifact_sha256="8" * 64,
    )


def test_after_cost_depth_walk_selects_up_without_authority() -> None:
    result = evaluate_shadow_settlement_opportunity(
        _prediction(),
        _state(),
        minimum_edge_per_share=Decimal("0.01"),
    )
    assert result.status == "candidate"
    assert result.reason == ""
    assert result.selected_outcome == "Up"
    assert result.up_quote.filled_quantity == Decimal("5")
    assert result.up_quote.average_price == Decimal("0.554")
    assert result.up_quote.fee_quote > 0
    assert Decimal(result.expected_terminal_value_per_share) > Decimal("0.01")
    assert Decimal(result.maximum_loss_quote) == result.up_quote.total_cost_quote
    assert result.trading_authority is False
    assert result.proposal_authority is False
    assert result.execution_or_profitability_claim is False
    assert len(result.artifact_sha256) == 64


def test_no_after_cost_edge_abstains() -> None:
    result = evaluate_shadow_settlement_opportunity(
        _prediction(0.51),
        _state(),
        minimum_edge_per_share=Decimal("0.02"),
    )
    assert result.status == "abstain"
    assert "after_cost_edge_below_threshold" in result.reason


def test_stale_prediction_and_venue_delay_abstain() -> None:
    state = _state()
    stale_state = PolymarketShadowMarketState(
        market=state.market,
        up_book=state.up_book,
        down_book=state.down_book,
        clob_market_info_sha256=state.clob_market_info_sha256,
        general_order_delay_seconds=181,
        taker_order_delay_enabled=True,
        observed_at_ms=DECISION_TIME_MS + 6_000,
    )
    result = evaluate_shadow_settlement_opportunity(
        _prediction(),
        stale_state,
        maximum_prediction_age_ms=5_000,
    )
    assert result.status == "abstain"
    assert "prediction_stale" in result.reason
    assert "venue_order_delay_exceeds_remaining_time" in result.reason


def test_below_minimum_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="below the venue minimum"):
        evaluate_shadow_settlement_opportunity(
            _prediction(),
            _state(),
            quantity=Decimal("4.99"),
        )

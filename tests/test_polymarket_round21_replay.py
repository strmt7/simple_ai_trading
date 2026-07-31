from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round21_dataset import Round21OfficialOutcome
from simple_ai_trading.polymarket_round21_execution import (
    Round21MarketExecutionEvidence,
)
from simple_ai_trading.polymarket_round21_policy import Round21ProbabilityEnvelope
from simple_ai_trading.polymarket_round21_replay import (
    POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
    Round21ReplayCondition,
    replay_round21_economics,
    replay_round21_full_matrix,
)


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000
CONDITION_ID = "0x" + "7" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-economic-replay-design-v1.json"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.20"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=_sha("market"),
        gamma_payload_json="{}",
    )


def _evidence() -> Round21MarketExecutionEvidence:
    return Round21MarketExecutionEvidence.create(
        condition_id=CONDITION_ID,
        observed_wall_ms=DECISION_MS - 1_000,
        observed_monotonic_ns=(DECISION_MS - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=0,
        minimum_order_age_seconds=0,
        clob_info_sha256=_sha("clob"),
        up_fee_rate_sha256=_sha("up-fee"),
        down_fee_rate_sha256=_sha("down-fee"),
        snapshot_sha256=_sha("snapshot"),
    )


def _envelope() -> Round21ProbabilityEnvelope:
    return Round21ProbabilityEnvelope.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        probability_up=Decimal("0.80"),
        lower_up=Decimal("0.75"),
        upper_up=Decimal("0.85"),
        model_layer="core",
        source_model_artifact_sha256=_sha("model"),
        source_probability_batch_sha256=_sha("probability-batch"),
        feature_row_sha256=_sha("feature-row"),
    )


def _book(
    outcome: str,
    received_wall_ms: int,
    *,
    bid: str = "0.49",
    ask: str = "0.50",
    quantity: str = "1000",
    connected: bool = True,
    gap_free: bool = True,
) -> PaperBookSnapshot:
    token = UP_TOKEN if outcome == "Up" else DOWN_TOKEN
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token,
        bids=(BookLevel(Decimal(bid), Decimal(quantity)),),
        asks=(BookLevel(Decimal(ask), Decimal(quantity)),),
        source_time_ms=received_wall_ms - 10,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=received_wall_ms * 1_000_000,
        source_payload_sha256=_sha(
            f"{outcome}-{received_wall_ms}-{bid}-{ask}-{quantity}-"
            f"{connected}-{gap_free}"
        ),
        connected=connected,
        gap_free=gap_free,
    ).validated()


def _condition(
    *,
    resolved_up: bool = True,
    include_execution: bool = True,
    execution_ask: str = "0.50",
) -> Round21ReplayCondition:
    books = [
        _book("Up", DECISION_MS - 50),
        _book("Down", DECISION_MS - 50),
    ]
    if include_execution:
        for latency_offset in (500, 750, 1_250):
            books.extend(
                (
                    _book(
                        "Up",
                        DECISION_MS + latency_offset,
                        ask=execution_ask,
                    ),
                    _book("Down", DECISION_MS + latency_offset),
                )
            )
    return Round21ReplayCondition.create(
        market=_market(),
        market_evidence=_evidence(),
        envelopes=(_envelope(),),
        books=books,
        outcome=Round21OfficialOutcome.create(
            condition_id=CONDITION_ID,
            event_start_ms=START_MS,
            resolved_up=resolved_up,
            observed_at_ms=START_MS + 300_100,
            source="official-polymarket-resolution",
            source_payload_sha256=_sha(f"outcome-{resolved_up}"),
        ),
        source_manifest_sha256=_sha("source-manifest"),
        reconciliation_sha256=_sha("reconciliation"),
    )


def test_round21_economic_replay_design_is_canonical_and_independent() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")
    actual = hashlib.sha256(
        json.dumps(
            design,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert claimed == actual == POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256
    assert design["independence"]["execution_and_settlement_venue"] == "polymarket_only"
    assert design["independence"]["binance_execution"] is False
    assert not any(design["authority"].values())


def test_round21_replay_settles_exact_fills_without_claiming_qualification() -> None:
    replay = replay_round21_economics(
        (_condition(resolved_up=True),),
        scenario_name="primary",
    )

    assert replay.unknown_state_count == 0
    assert replay.risk_violation_count == 0
    assert replay.metrics.executed_action_count == 1
    assert replay.metrics.net_pnl_quote > 0
    assert replay.conditions[0].utility_quote == replay.metrics.net_pnl_quote
    assert replay.qualified is False
    assert "insufficient_resolved_conditions" in replay.qualification_reasons
    assert replay.profitability_claim is False
    assert replay.live_trading_authority is False
    with pytest.raises(ValueError, match="economic metrics differ"):
        replace(
            replay.metrics,
            net_pnl_quote=replay.metrics.net_pnl_quote + Decimal("1"),
        ).validated()
    with pytest.raises(ValueError, match="economic replay differs"):
        replace(replay, qualified=True).validated()


def test_round21_future_execution_book_cannot_change_the_action_decision() -> None:
    favorable = replay_round21_economics(
        (_condition(execution_ask="0.50"),),
        scenario_name="primary",
    )
    adverse = replay_round21_economics(
        (_condition(execution_ask="0.90"),),
        scenario_name="primary",
    )

    favorable_step = favorable.conditions[0].steps[0]
    adverse_step = adverse.conditions[0].steps[0]
    assert favorable_step.decision_sha256 == adverse_step.decision_sha256
    assert favorable_step.action == adverse_step.action == "buy_up"
    assert favorable_step.execution_state == "filled"
    assert adverse_step.execution_state == "known_no_fill"


def test_round21_missing_post_submit_book_rejects_the_scenario() -> None:
    replay = replay_round21_economics(
        (_condition(include_execution=False),),
        scenario_name="primary",
    )

    assert replay.unknown_state_count == 1
    assert replay.qualified is False
    assert replay.conditions == ()
    assert "unknown_post_submit_state" in replay.qualification_reasons


def test_round21_condition_identity_rejects_book_or_manifest_tampering() -> None:
    condition = _condition()

    tampered = replace(
        condition,
        source_manifest_sha256=_sha("other-manifest"),
    )
    with pytest.raises(ValueError, match="replay condition differs"):
        tampered.validated()


def test_round21_full_matrix_contains_all_independent_profile_scenarios() -> None:
    matrix = replay_round21_full_matrix((_condition(),))

    assert len(matrix) == 81
    assert len({(value.profile, value.scenario) for value in matrix}) == 81
    assert {value.profile for value in matrix} == {
        "conservative",
        "regular",
        "aggressive",
    }
    assert all(value.paper_trading_authority is False for value in matrix)

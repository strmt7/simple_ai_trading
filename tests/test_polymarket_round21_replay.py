from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round21_replay as replay_module
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round21_dataset import Round21OfficialOutcome
from simple_ai_trading.polymarket_round21_comparison import (
    POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256,
    compare_round21_optional_full_matrix,
    compare_round21_optional_replay_matrices,
)
from simple_ai_trading.polymarket_round21_execution import (
    Round21MarketExecutionEvidence,
)
from simple_ai_trading.polymarket_round21_policy import Round21ProbabilityEnvelope
from simple_ai_trading.polymarket_round21_replay import (
    POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
    Round21DirectionalPermission,
    Round21PairedEconomicMatrixAccumulator,
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
    / "round-021-economic-replay-design-v6.json"
)
COMPARISON_DESIGN_PATH = (
    DESIGN_PATH.parent / "round-021-matched-economic-comparison-design-v6.json"
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


def _envelope(*, layer: str = "core") -> Round21ProbabilityEnvelope:
    return Round21ProbabilityEnvelope.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        probability_up=Decimal("0.80"),
        lower_up=Decimal("0.75"),
        upper_up=Decimal("0.85"),
        model_layer=layer,
        source_model_artifact_sha256=_sha(f"model-{layer}"),
        source_probability_batch_sha256=_sha(f"probability-batch-{layer}"),
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
    monotonic_offset_ns: int = 0,
    condition_id: str = CONDITION_ID,
    up_token: str = UP_TOKEN,
    down_token: str = DOWN_TOKEN,
) -> PaperBookSnapshot:
    token = up_token if outcome == "Up" else down_token
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=condition_id,
        asset_id=token,
        bids=(BookLevel(Decimal(bid), Decimal(quantity)),),
        asks=(BookLevel(Decimal(ask), Decimal(quantity)),),
        source_time_ms=received_wall_ms - 10,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=(received_wall_ms * 1_000_000 + monotonic_offset_ns),
        source_payload_sha256=_sha(
            f"{outcome}-{received_wall_ms}-{bid}-{ask}-{quantity}-"
            f"{connected}-{gap_free}-{condition_id}"
            f"-{monotonic_offset_ns}"
        ),
        connected=connected,
        gap_free=gap_free,
    ).validated()


def test_round21_same_millisecond_books_follow_monotonic_receipt_order() -> None:
    earlier = _book("Up", DECISION_MS - 50, ask="0.50")
    later = _book(
        "Up",
        DECISION_MS - 50,
        ask="0.51",
        monotonic_offset_ns=1,
    )
    condition = Round21ReplayCondition.create(
        market=_market(),
        market_evidence=_evidence(),
        envelopes=(_envelope(),),
        books=(later, _book("Down", DECISION_MS - 50), earlier),
        outcome=Round21OfficialOutcome.create(
            condition_id=CONDITION_ID,
            event_start_ms=START_MS,
            resolved_up=True,
            observed_at_ms=START_MS + 300_100,
            source="official-polymarket-resolution",
            source_payload_sha256=_sha("same-ms-outcome"),
        ),
        source_manifest_sha256=_sha("source-manifest"),
        reconciliation_sha256=_sha("reconciliation"),
    )

    assert condition.books[0] == earlier
    assert condition.books[1] == _book("Down", DECISION_MS - 50)
    assert condition.books[2] == later
    assert (
        condition.creation_book(
            outcome="Up",
            decision_time_ms=DECISION_MS,
        )
        == later
    )


def _condition(
    *,
    resolved_up: bool = True,
    include_execution: bool = True,
    execution_ask: str = "0.50",
    layer: str = "core",
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
        envelopes=(_envelope(layer=layer),),
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


def _condition_at(slot: int, *, resolved_up: bool) -> Round21ReplayCondition:
    start_ms = START_MS + int(slot) * 300_000
    decision_ms = start_ms + 120_000
    digit = format(7 + int(slot), "x")
    condition_id = "0x" + digit * 64
    up_token = str(3 + int(slot)) * 40
    down_token = str(5 + int(slot)) * 40
    market = replace(
        _market(),
        market_id=str(12_345 + int(slot)),
        condition_id=condition_id,
        slug=f"btc-updown-5m-{start_ms // 1000}",
        event_start_ms=start_ms,
        end_ms=start_ms + 300_000,
        up_token_id=up_token,
        down_token_id=down_token,
        gamma_payload_sha256=_sha(f"market-{slot}"),
    )
    evidence = Round21MarketExecutionEvidence.create(
        condition_id=condition_id,
        observed_wall_ms=decision_ms - 1_000,
        observed_monotonic_ns=(decision_ms - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=0,
        minimum_order_age_seconds=0,
        clob_info_sha256=_sha(f"clob-{slot}"),
        up_fee_rate_sha256=_sha(f"up-fee-{slot}"),
        down_fee_rate_sha256=_sha(f"down-fee-{slot}"),
        snapshot_sha256=_sha(f"snapshot-{slot}"),
    )
    envelope = Round21ProbabilityEnvelope.create(
        condition_id=condition_id,
        decision_time_ms=decision_ms,
        probability_up=Decimal("0.80"),
        lower_up=Decimal("0.75"),
        upper_up=Decimal("0.85"),
        model_layer="core",
        source_model_artifact_sha256=_sha("model-core"),
        source_probability_batch_sha256=_sha(f"probability-batch-{slot}"),
        feature_row_sha256=_sha(f"feature-row-{slot}"),
    )
    books = []
    for offset in (-50, 500, 750, 1_250):
        books.extend(
            (
                _book(
                    "Up",
                    decision_ms + offset,
                    condition_id=condition_id,
                    up_token=up_token,
                    down_token=down_token,
                ),
                _book(
                    "Down",
                    decision_ms + offset,
                    condition_id=condition_id,
                    up_token=up_token,
                    down_token=down_token,
                ),
            )
        )
    return Round21ReplayCondition.create(
        market=market,
        market_evidence=evidence,
        envelopes=(envelope,),
        books=books,
        outcome=Round21OfficialOutcome.create(
            condition_id=condition_id,
            event_start_ms=start_ms,
            resolved_up=resolved_up,
            observed_at_ms=start_ms + 300_100,
            source="official-polymarket-resolution",
            source_payload_sha256=_sha(f"outcome-{slot}-{resolved_up}"),
        ),
        source_manifest_sha256=_sha("source-manifest"),
        reconciliation_sha256=_sha(f"reconciliation-{slot}"),
    )


def _rehash_replay(value):
    return replace(
        value,
        replay_sha256=replay_module._canonical_sha256(value.identity_payload()),
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

    comparison = json.loads(COMPARISON_DESIGN_PATH.read_text(encoding="utf-8"))
    comparison_claimed = comparison.pop("design_sha256")
    comparison_actual = hashlib.sha256(
        json.dumps(
            comparison,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert (
        comparison_claimed
        == comparison_actual
        == POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256
    )


def test_round21_replay_settles_exact_fills_without_claiming_qualification() -> None:
    condition = _condition(resolved_up=True)
    replay = replay_round21_economics(
        (condition,),
        scenario_name="primary",
    )
    reference = replay_module._replay_round21_economics_materialized_reference(
        (condition,),
        scenario_name="primary",
    )

    assert replay == reference
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


def test_round21_replay_rejects_rehashed_impossible_accounting() -> None:
    replay = replay_round21_economics(
        (_condition(resolved_up=True),),
        scenario_name="primary",
    )
    wrong_cash = replace(
        replay,
        final_cash_quote=replay.final_cash_quote + Decimal("1"),
        replay_sha256="",
    )
    with pytest.raises(ValueError, match="economic replay accounting differs"):
        _rehash_replay(wrong_cash).validated()

    wrong_metrics = replace(
        replay.metrics,
        calendar_day_count=replay.metrics.calendar_day_count + 1,
        metric_sha256="",
    )
    wrong_metrics = replace(
        wrong_metrics,
        metric_sha256=replay_module._canonical_sha256(wrong_metrics.identity_payload()),
    )
    wrong_calendar = replace(
        replay,
        metrics=wrong_metrics,
        replay_sha256="",
    )
    with pytest.raises(ValueError, match="economic replay accounting differs"):
        _rehash_replay(wrong_calendar).validated()

    duplicated_metrics = replay_module._metrics(
        utilities=(
            replay.conditions[0].utility_quote,
            replay.conditions[0].utility_quote,
        ),
        daily_values=(replay.conditions[0].utility_quote * 2,),
        executed_actions=replay.metrics.executed_action_count * 2,
        maximum_drawdown=replay.metrics.maximum_drawdown_fraction,
        realized_maximum_drawdown=(replay.metrics.realized_maximum_drawdown_fraction),
        bootstrap_identity="conservative:primary",
    )
    duplicated = replace(
        replay,
        final_cash_quote=(
            replay.initial_capital_quote + duplicated_metrics.net_pnl_quote
        ),
        metrics=duplicated_metrics,
        conditions=(replay.conditions[0], replay.conditions[0]),
        qualification_reasons=replay_module._round21_qualification_reasons(
            duplicated_metrics,
            profile=replay_module.round21_risk_profile("conservative"),
            unknown_state_count=0,
            risk_violation_count=0,
        ),
        replay_sha256="",
    )
    with pytest.raises(ValueError, match="economic replay accounting differs"):
        _rehash_replay(duplicated).validated()

    impossible_unknown_count = replace(
        replay,
        unknown_state_count=2,
        qualification_reasons=replay_module._round21_qualification_reasons(
            replay.metrics,
            profile=replay_module.round21_risk_profile("conservative"),
            unknown_state_count=2,
            risk_violation_count=0,
        ),
        replay_sha256="",
    )
    with pytest.raises(ValueError, match="economic replay differs"):
        _rehash_replay(impossible_unknown_count).validated()


def test_round21_settled_profit_becomes_the_next_drawdown_peak() -> None:
    replay = replay_round21_economics(
        (
            _condition_at(0, resolved_up=True),
            _condition_at(1, resolved_up=False),
        ),
        scenario_name="primary",
    )
    reference = replay_module._replay_round21_economics_materialized_reference(
        (
            _condition_at(0, resolved_up=True),
            _condition_at(1, resolved_up=False),
        ),
        scenario_name="primary",
    )
    first_settlement = replay.conditions[0].end_cash_quote
    second_entry_equity = replay.conditions[1].steps[0].conservative_equity_quote
    expected_drawdown = (
        first_settlement - second_entry_equity
    ) / replay.initial_capital_quote

    assert replay == reference
    assert first_settlement > replay.initial_capital_quote
    assert replay.metrics.maximum_drawdown_fraction == expected_drawdown


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


def test_round21_ai_permission_applies_only_after_measured_arrival() -> None:
    veto = Round21DirectionalPermission.create(
        condition_id=CONDITION_ID,
        effective_at_ms=DECISION_MS,
        directional_entry_allowed=False,
        source_evidence_sha256=_sha("ai-veto-evidence"),
    )
    late_veto = Round21DirectionalPermission.create(
        condition_id=CONDITION_ID,
        effective_at_ms=DECISION_MS + 1,
        directional_entry_allowed=False,
        source_evidence_sha256=_sha("late-ai-veto-evidence"),
    )

    blocked = replay_round21_economics(
        (_condition(),),
        scenario_name="primary",
        directional_permissions=(veto,),
    )
    not_yet_effective = replay_round21_economics(
        (_condition(),),
        scenario_name="primary",
        directional_permissions=(late_veto,),
    )

    assert blocked.conditions[0].steps[0].action == "abstain"
    assert blocked.metrics.executed_action_count == 0
    assert blocked.economic_gate_passed is False
    assert blocked.qualified is False
    assert "insufficient_executed_actions" in blocked.qualification_reasons
    assert "net_pnl_not_positive" in blocked.qualification_reasons
    assert not_yet_effective.conditions[0].steps[0].action == "buy_up"
    assert not_yet_effective.metrics.executed_action_count == 1
    assert (
        blocked.directional_permission_root_sha256
        != not_yet_effective.directional_permission_root_sha256
    )


def test_round21_condition_identity_rejects_book_or_manifest_tampering() -> None:
    condition = _condition()

    tampered = replace(
        condition,
        source_manifest_sha256=_sha("other-manifest"),
    )
    with pytest.raises(ValueError, match="replay condition differs"):
        tampered.validated()


def test_round21_full_matrix_contains_all_independent_profile_scenarios() -> None:
    iterations = 0

    def conditions():
        nonlocal iterations
        iterations += 1
        if iterations > 1:
            raise AssertionError("condition source was replayed more than once")
        yield _condition()

    matrix = replay_round21_full_matrix(conditions())

    assert iterations == 1
    assert len(matrix) == 81
    assert len({(value.profile, value.scenario) for value in matrix}) == 81
    assert {value.profile for value in matrix} == {
        "conservative",
        "regular",
        "aggressive",
    }
    assert all(value.paper_trading_authority is False for value in matrix)


def test_round21_paired_matrix_shares_one_source_pass_without_changing_results() -> (
    None
):
    condition = _condition()
    permission = Round21DirectionalPermission.create(
        condition_id=CONDITION_ID,
        effective_at_ms=DECISION_MS,
        directional_entry_allowed=False,
        source_evidence_sha256=_sha("paired-veto"),
    )
    iterations = 0

    def conditions():
        nonlocal iterations
        iterations += 1
        if iterations > 1:
            raise AssertionError("paired source was replayed more than once")
        yield condition

    pair = Round21PairedEconomicMatrixAccumulator(
        challenger_directional_permissions=(permission,),
    )
    for value in conditions():
        pair.observe(value)
    baseline, challenger = pair.finish()

    assert iterations == 1
    assert baseline == replay_round21_full_matrix((condition,))
    assert challenger == replay_round21_full_matrix(
        (condition,),
        directional_permissions=(permission,),
    )
    assert pair.condition_ids == (CONDITION_ID,)
    assert pair.matched_condition_sha256 == (condition.matched_population_sha256(),)
    assert pair.matched_decision_count == 1
    with pytest.raises(RuntimeError, match="already terminal"):
        pair.finish()


def test_round21_condition_audit_hashes_every_decision_with_bounded_retention() -> None:
    source = _condition(include_execution=False)
    envelopes = tuple(
        Round21ProbabilityEnvelope.create(
            condition_id=CONDITION_ID,
            decision_time_ms=DECISION_MS + 5_000 + index * 250,
            probability_up=Decimal("0.80"),
            lower_up=Decimal("0.75"),
            upper_up=Decimal("0.85"),
            model_layer="core",
            source_model_artifact_sha256=_sha("model-core"),
            source_probability_batch_sha256=_sha("probability-batch-core"),
            feature_row_sha256=_sha(f"feature-row-{index}"),
        )
        for index in range(100)
    )
    condition = Round21ReplayCondition.create(
        market=source.market,
        market_evidence=source.market_evidence,
        envelopes=envelopes,
        books=source.books,
        outcome=source.outcome,
        source_manifest_sha256=source.source_manifest_sha256,
        reconciliation_sha256=source.reconciliation_sha256,
    )

    replay = replay_round21_economics(
        (condition,),
        scenario_name="primary",
    )
    reference = replay_module._replay_round21_economics_materialized_reference(
        (condition,),
        scenario_name="primary",
    )
    result = replay.conditions[0]

    assert result.decision_count == 100
    assert len(result.steps) == 2
    assert all(step.action == "abstain" for step in result.steps)
    assert result.step_chain_sha256 != hashlib.sha256(b"").hexdigest()
    assert replay.metrics == reference.metrics
    assert result.utility_quote == reference.conditions[0].utility_quote
    assert result.executed_action_count == reference.conditions[0].executed_action_count
    assert tuple(step.action for step in result.steps) == (
        reference.conditions[0].steps[0].action,
        reference.conditions[0].steps[-1].action,
    )
    assert replay.validated() == replay


def test_round21_optional_comparison_uses_exact_matched_polymarket_paths() -> None:
    comparison = compare_round21_optional_full_matrix(
        baseline_conditions=(_condition(layer="core"),),
        challenger_conditions=(_condition(layer="core_spot"),),
    )

    assert comparison.challenger_layer == "core_spot"
    assert len(comparison.deltas) == 81
    assert comparison.all_replays_accepted is False
    assert comparison.optional_layer_selected is False
    assert all(
        "net_pnl_delta_not_positive" in value.reasons for value in comparison.deltas
    )
    assert comparison.live_trading_authority is False
    streamed = compare_round21_optional_replay_matrices(
        baseline_matrix=replay_round21_full_matrix((_condition(layer="core"),)),
        challenger_matrix=replay_round21_full_matrix((_condition(layer="core_spot"),)),
        challenger_layer="core_spot",
        matched_population_sha256=comparison.matched_population_sha256,
    )
    assert streamed == comparison
    with pytest.raises(
        ValueError,
        match="matched economic comparison differs",
    ):
        replace(comparison, optional_layer_selected=True).validated()

    with pytest.raises(ValueError, match="matched input population differs"):
        compare_round21_optional_full_matrix(
            baseline_conditions=(_condition(layer="core"),),
            challenger_conditions=(
                _condition(layer="core_spot", execution_ask="0.51"),
            ),
        )

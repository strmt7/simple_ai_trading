from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_ai_execution_replay import (
    Round74AIExecutionReplayInstruction,
    replay_round74_ai_execution_run,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    Round74ReplayObservation,
)
from simple_ai_trading.impact_absorption_event_targets import (
    Round74EventExecutionOverride,
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
    Round74EventTargetEvidence,
    Round74EventTargetSpec,
    round74_commission_evidence_claims,
    round74_event_action_payoff,
    round74_funding_schedule_evidence_claims,
    round74_latency_evidence_claims,
    round74_quantity_rules_evidence_claims,
    round74_slippage_evidence_claims,
)
from simple_ai_trading.impact_absorption_targets import (
    Round73MarketQuantityRules,
    walk_round73_book,
)
from simple_ai_trading.impact_absorption_target_assembly import (
    Round74SourceTargetAssembly,
)


NS = 1_000_000_000
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
EVIDENCE_WALL_NS = 1_784_000_000_000_000_000


def _evidence(
    kind: str,
    claims: object,
    *,
    payload_sha256: str,
) -> Round74EventTargetEvidence:
    return Round74EventTargetEvidence.create(
        kind=kind,
        environment="binance_usdm_mainnet",
        observed_wall_ns=EVIDENCE_WALL_NS,
        record_count=3,
        source_query_or_protocol_sha256=f"{len(kind):064x}",
        source_payload_sha256=payload_sha256,
        claims=claims,
    )


def _state(
    *,
    symbol: str = "BTCUSDT",
    mid: float = 100.0,
    update_id: int = 1,
    quantity: float = 10.0,
) -> L2BookState:
    bids = tuple((mid - 0.1 - index * 0.1, quantity) for index in range(20))
    asks = tuple((mid + 0.1 + index * 0.1, quantity) for index in range(20))
    bid_depths = tuple(
        sum(price * qty for price, qty in bids[:levels]) for levels in (5, 10, 20)
    )
    ask_depths = tuple(
        sum(price * qty for price, qty in asks[:levels]) for levels in (5, 10, 20)
    )
    imbalances = tuple(
        (bid - ask) / (bid + ask)
        for bid, ask in zip(bid_depths, ask_depths, strict=True)
    )
    return L2BookState(
        symbol=symbol,
        update_id=update_id,
        best_bid=bids[0][0],
        best_ask=asks[0][0],
        spread_bps=(asks[0][0] - bids[0][0]) / mid * 10_000.0,
        mid=mid,
        bid_levels=bids,
        ask_levels=asks,
        bid_depth_quote_5=bid_depths[0],
        ask_depth_quote_5=ask_depths[0],
        bid_depth_quote_10=bid_depths[1],
        ask_depth_quote_10=ask_depths[1],
        bid_depth_quote_20=bid_depths[2],
        ask_depth_quote_20=ask_depths[2],
        imbalance_5=imbalances[0],
        imbalance_10=imbalances[1],
        imbalance_20=imbalances[2],
    )


def _rules() -> dict[str, Round73MarketQuantityRules]:
    return {
        symbol: Round73MarketQuantityRules(
            symbol=symbol,
            step_size=Decimal("0.001"),
            minimum_quantity=Decimal("0.001"),
            maximum_quantity=Decimal("1000"),
            minimum_notional=Decimal("10"),
        )
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    }


def _spec(
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    latency_ns: int = 100_000_000,
    exit_latency_ns: int = 100_000_000,
    entry_latencies: dict[str, int] | None = None,
    exit_latencies: dict[str, int] | None = None,
    slippage_by_symbol: dict[str, float] | None = None,
    funding_boundaries: dict[str, tuple[int, ...]] | None = None,
    funding_intervals: dict[str, tuple[tuple[int, int], ...]] | None = None,
    funding_coverage: dict[str, tuple[int, int]] | None = None,
    funding_evidence_sha256: str = "e" * 64,
) -> Round74EventTargetSpec:
    funding = (
        funding_boundaries
        if funding_boundaries is not None
        else {symbol: () for symbol in SYMBOLS}
    )
    normalized_funding_intervals = (
        funding_intervals
        if funding_intervals is not None
        else {
            symbol: tuple((boundary, boundary) for boundary in boundaries)
            for symbol, boundaries in funding.items()
        }
    )
    coverage = (
        funding_coverage
        if funding_coverage is not None
        else {symbol: (0, 1_000_000_000_000) for symbol in SYMBOLS}
    )
    fees = {symbol: fee_bps for symbol in SYMBOLS}
    entries = (
        entry_latencies
        if entry_latencies is not None
        else {symbol: latency_ns for symbol in SYMBOLS}
    )
    exits = (
        exit_latencies
        if exit_latencies is not None
        else {symbol: exit_latency_ns for symbol in SYMBOLS}
    )
    slippage = (
        slippage_by_symbol
        if slippage_by_symbol is not None
        else {symbol: slippage_bps for symbol in SYMBOLS}
    )
    rules = _rules()
    reference_notional = 100.0
    return Round74EventTargetSpec.create(
        reference_quote_notional=reference_notional,
        decision_to_entry_latency_ns_by_symbol=entries,
        decision_to_exit_latency_ns_by_symbol=exits,
        taker_fee_bps_by_symbol=fees,
        funding_boundary_intervals_monotonic_ns=normalized_funding_intervals,
        funding_schedule_coverage_monotonic_ns=coverage,
        additional_slippage_bps_per_side_by_symbol=slippage,
        quantity_rules_evidence=_evidence(
            "quantity_rules",
            round74_quantity_rules_evidence_claims(rules),
            payload_sha256="f" * 64,
        ),
        commission_evidence=_evidence(
            "commission",
            round74_commission_evidence_claims(fees),
            payload_sha256="a" * 64,
        ),
        entry_exit_latency_evidence=_evidence(
            "entry_exit_latency",
            round74_latency_evidence_claims(
                decision_to_entry_latency_ns_by_symbol=entries,
                decision_to_exit_latency_ns_by_symbol=exits,
            ),
            payload_sha256="b" * 64,
        ),
        slippage_evidence=_evidence(
            "residual_slippage",
            round74_slippage_evidence_claims(
                reference_quote_notional=reference_notional,
                additional_slippage_bps_per_side_by_symbol=slippage,
            ),
            payload_sha256="d" * 64,
        ),
        funding_schedule_evidence=_evidence(
            "funding_schedule",
            round74_funding_schedule_evidence_claims(
                funding_boundary_intervals_monotonic_ns=(normalized_funding_intervals),
                funding_schedule_coverage_monotonic_ns=coverage,
            ),
            payload_sha256=funding_evidence_sha256,
        ),
    )


def _anchor(
    *,
    index: int = 0,
    decision_ns: int = NS,
    symbol: str = "BTCUSDT",
) -> Round74EventTargetAnchor:
    return Round74EventTargetAnchor(
        symbol=symbol,
        anchor_index=index,
        decision_monotonic_ns=decision_ns,
        decision_wall_ns=1_784_000_000_000_000_000 + decision_ns,
        endpoint_frame_index=1,
        endpoint_message_index=index,
        feature_window_sha256="c" * 64,
    )


def _complete_engine(
    engine: Round74EventTargetEngine,
    *,
    path_mids: tuple[float, ...] = (101.0, 102.0, 103.0, 104.0),
) -> tuple:
    engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    _observe_dense_path(
        engine,
        start_ns=1_100_000_000,
        checkpoints=tuple(
            (offset, mid)
            for offset, mid in zip(
                (
                    2_200_000_000,
                    6_200_000_000,
                    31_200_000_000,
                    301_200_000_000,
                ),
                path_mids,
                strict=True,
            )
        ),
    )
    return engine.finish()


def _selected_outcome(
    outcomes: tuple,
    *,
    horizon_seconds: int = 30,
    side: str = "long",
):
    return next(
        outcome
        for outcome in outcomes
        if outcome.horizon_seconds == horizon_seconds and outcome.side == side
    )


def _observe_dense_path(
    engine: Round74EventTargetEngine,
    *,
    start_ns: int,
    checkpoints: tuple[tuple[int, float], ...],
) -> None:
    prior_ns = int(start_ns)
    frame_index = 100
    for checkpoint_ns, mid in checkpoints:
        for received in range(
            prior_ns + 200_000_000,
            checkpoint_ns,
            200_000_000,
        ):
            engine.observe_depth(
                received_monotonic_ns=received,
                frame_index=frame_index,
                message_index=0,
                state=_state(mid=mid, update_id=frame_index),
            )
            frame_index += 1
        engine.observe_depth(
            received_monotonic_ns=checkpoint_ns,
            frame_index=frame_index,
            message_index=0,
            state=_state(mid=mid, update_id=frame_index),
        )
        frame_index += 1
        prior_ns = checkpoint_ns


def _observe_checkpoints_without_path_fill(
    engine: Round74EventTargetEngine,
    *,
    checkpoints: tuple[tuple[int, float, int], ...],
) -> None:
    for offset, mid, update_id in checkpoints:
        engine.observe_depth(
            received_monotonic_ns=offset,
            frame_index=update_id,
            message_index=0,
            state=_state(mid=mid, update_id=update_id),
        )


def test_round74_payoff_charges_both_actual_walked_legs() -> None:
    state = _state()
    entry = walk_round73_book(
        state.ask_levels,
        base_quantity=1.0,
        ascending_prices=True,
    )
    exit_walk = walk_round73_book(
        state.bid_levels,
        base_quantity=1.0,
        ascending_prices=False,
    )
    assert entry is not None and exit_walk is not None

    payoff = round74_event_action_payoff(
        side="long",
        entry_walk=entry,
        exit_walk=exit_walk,
        entry_mid=state.mid,
        exit_mid=state.mid,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
    )

    assert payoff.gross_payoff_quote == pytest.approx(
        exit_walk.quote_notional - entry.quote_notional
    )
    assert payoff.commission_quote == pytest.approx(
        5.0 / 10_000.0 * (entry.quote_notional + exit_walk.quote_notional)
    )
    assert payoff.additional_slippage_quote == pytest.approx(
        1.0 / 10_000.0 * (entry.quote_notional + exit_walk.quote_notional)
    )
    assert payoff.midpoint_payoff_quote == pytest.approx(0.0)
    assert payoff.book_walk_implementation_shortfall_quote == pytest.approx(
        -payoff.gross_payoff_quote
    )
    assert payoff.explicit_cost_quote == pytest.approx(
        payoff.commission_quote + payoff.additional_slippage_quote
    )
    assert payoff.total_implementation_shortfall_quote == pytest.approx(
        payoff.book_walk_implementation_shortfall_quote + payoff.explicit_cost_quote
    )
    assert payoff.net_payoff_quote == pytest.approx(
        payoff.midpoint_payoff_quote - payoff.total_implementation_shortfall_quote
    )
    assert payoff.net_payoff_quote < payoff.gross_payoff_quote


def test_execution_override_rewalks_delayed_book_with_quantized_size() -> None:
    anchor = _anchor()
    baseline = Round74EventTargetEngine(
        spec=_spec(),
        anchors=(anchor,),
        quantity_rules=_rules(),
    )
    delayed = Round74EventTargetEngine(
        spec=_spec(),
        anchors=(anchor,),
        quantity_rules=_rules(),
        execution_overrides=(
            Round74EventExecutionOverride(
                symbol=anchor.symbol,
                anchor_index=anchor.anchor_index,
                feature_window_sha256=anchor.feature_window_sha256,
                additional_entry_latency_ns=NS,
                quote_size_multiplier_bps=5_000,
                source_review_sha256="9" * 64,
            ),
        ),
    )

    baseline_outcome = _selected_outcome(_complete_engine(baseline))
    delayed_outcome = _selected_outcome(_complete_engine(delayed))

    assert baseline_outcome.requested_entry_monotonic_ns == 1_100_000_000
    assert baseline_outcome.actual_entry_monotonic_ns == 1_100_000_000
    assert baseline_outcome.base_quantity == pytest.approx(1.0)
    assert delayed_outcome.requested_entry_monotonic_ns == 2_100_000_000
    assert delayed_outcome.actual_entry_monotonic_ns == 2_100_000_000
    assert delayed_outcome.base_quantity == pytest.approx(0.5)
    assert delayed_outcome.entry_average_price != (baseline_outcome.entry_average_price)
    assert delayed_outcome.entry_quote_notional == pytest.approx(
        delayed_outcome.base_quantity * delayed_outcome.entry_average_price
    )
    assert delayed_outcome.reference_quote_notional == pytest.approx(100.0)
    assert delayed_outcome.capital_scaled_net_payoff_bps == pytest.approx(
        delayed_outcome.net_payoff_bps
        * delayed_outcome.entry_quote_notional
        / delayed_outcome.reference_quote_notional
    )
    assert (
        delayed_outcome.capital_scaled_maximum_adverse_excursion_bps
        == pytest.approx(
            delayed_outcome.maximum_adverse_excursion_bps
            * delayed_outcome.entry_quote_notional
            / delayed_outcome.reference_quote_notional
        )
    )
    assert delayed_outcome.target_context_sha256 != (
        baseline_outcome.target_context_sha256
    )


def test_execution_override_rejects_incomplete_or_unsafe_identity() -> None:
    anchor = _anchor()
    override = Round74EventExecutionOverride(
        symbol=anchor.symbol,
        anchor_index=anchor.anchor_index,
        feature_window_sha256=anchor.feature_window_sha256,
        additional_entry_latency_ns=NS,
        quote_size_multiplier_bps=5_000,
        source_review_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="coverage differs"):
        Round74EventTargetEngine(
            spec=_spec(),
            anchors=(),
            quantity_rules=_rules(),
            execution_overrides=(override,),
        )
    with pytest.raises(ValueError, match="execution override differs"):
        replace(override, quote_size_multiplier_bps=0).validate()
    with pytest.raises(ValueError, match="identity differs"):
        Round74EventTargetEngine(
            spec=_spec(),
            anchors=(replace(anchor, feature_window_sha256="8" * 64),),
            quantity_rules=_rules(),
            execution_overrides=(override,),
        )


def test_ai_execution_replay_uses_delayed_book_and_capital_scaled_size() -> None:
    anchor = _anchor()
    instruction = Round74AIExecutionReplayInstruction(
        row_index=0,
        run_id="1" * 32,
        symbol=anchor.symbol,
        anchor_index=anchor.anchor_index,
        decision_monotonic_ns=anchor.decision_monotonic_ns,
        decision_wall_ns=anchor.decision_wall_ns,
        endpoint_frame_index=anchor.endpoint_frame_index,
        endpoint_message_index=anchor.endpoint_message_index,
        sample_sha256="a" * 64,
        feature_window_sha256=anchor.feature_window_sha256,
        feature_row_sha256="b" * 64,
        side=1,
        horizon_seconds=30,
        source_review_sha256="9" * 64,
        model_manifest_sha256="8" * 64,
        runtime_status="accepted",
        effective_review_latency_ns=NS,
        action_latency_eligible=True,
        requested_size_multiplier_bps=5_000,
        pre_replay_status="replay_required",
        partition_sha256="7" * 64,
        action_selection_sha256="6" * 64,
    )
    instruction.validate()
    assembly = Round74SourceTargetAssembly(
        spec=_spec(),
        quantity_rules_by_symbol=tuple(_rules().items()),
    )
    observations: list[Round74ReplayObservation] = []
    for frame_index, received_ns in enumerate(
        range(900_000_000, 303_000_000_000, 200_000_000),
        start=1,
    ):
        mid = 100.0 if received_ns < 2_100_000_000 else 101.0
        observations.append(
            Round74ReplayObservation(
                symbol=anchor.symbol,
                event_type="depthUpdate",
                frame_index=frame_index,
                message_index=0,
                received_monotonic_ns=received_ns,
                received_wall_ns=anchor.decision_wall_ns + received_ns,
                token=None,
                depth_state=_state(
                    mid=mid,
                    update_id=frame_index,
                ),
                depth_update_is_stale=False,
            )
        )

    (evidence,) = replay_round74_ai_execution_run(
        assembly,
        (instruction,),
        observations,
        capture_report_sha256="5" * 64,
    )

    assert evidence.status == "executed"
    assert evidence.exact_l2_replay_performed
    assert evidence.requested_entry_monotonic_ns == 2_100_000_000
    assert evidence.actual_entry_monotonic_ns == 2_100_000_000
    assert evidence.applied_size_multiplier_bps == 5_000
    assert evidence.reference_quote_notional == pytest.approx(100.0)
    assert evidence.actual_entry_quote_notional is not None
    assert evidence.actual_deployed_capital_bps == pytest.approx(
        evidence.actual_entry_quote_notional
        / evidence.reference_quote_notional
        * 10_000.0
    )
    assert evidence.actual_deployed_capital_bps != pytest.approx(5_000.0)
    assert evidence.capital_scaled_net_payoff_bps == pytest.approx(
        evidence.position_net_payoff_bps
        * evidence.actual_entry_quote_notional
        / evidence.reference_quote_notional
    )
    assert evidence.target_outcome_sha256 is not None
    assert evidence.target_context_sha256 is not None
    assert evidence.replay_sha256


def test_round74_target_spec_uses_symbol_specific_latency_and_slippage() -> None:
    entry_latencies = {
        "BTCUSDT": 100_000_000,
        "ETHUSDT": 200_000_000,
        "SOLUSDT": 300_000_000,
    }
    exit_latencies = {
        "BTCUSDT": 400_000_000,
        "ETHUSDT": 500_000_000,
        "SOLUSDT": 600_000_000,
    }
    slippage = {
        "BTCUSDT": 0.25,
        "ETHUSDT": 0.50,
        "SOLUSDT": 1.00,
    }
    spec = _spec(
        entry_latencies=entry_latencies,
        exit_latencies=exit_latencies,
        slippage_by_symbol=slippage,
    )

    assert spec.entry_latency_ns("BTCUSDT") == 100_000_000
    assert spec.entry_latency_ns("ETHUSDT") == 200_000_000
    assert spec.exit_latency_ns("SOLUSDT") == 600_000_000
    assert spec.slippage_bps_per_side("BTCUSDT") == pytest.approx(0.25)
    assert spec.slippage_bps_per_side("SOLUSDT") == pytest.approx(1.00)
    payload = spec.as_dict()
    assert payload["execution_environment"] == "binance_usdm_mainnet"
    assert payload["latency_and_residual_slippage_are_symbol_specific"] is True
    assert "decision_to_entry_latency_ns" not in payload
    assert "additional_slippage_bps_per_side" not in payload
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    assert set(evidence) == {
        "quantity_rules",
        "commission",
        "entry_exit_latency",
        "residual_slippage",
        "funding_schedule",
    }

    engine = Round74EventTargetEngine(
        spec=spec,
        anchors=[
            _anchor(symbol=symbol, index=index) for index, symbol in enumerate(SYMBOLS)
        ],
        quantity_rules=_rules(),
    )
    outcomes = engine.finish()
    assert {
        symbol: {
            outcome.requested_entry_monotonic_ns
            for outcome in outcomes
            if outcome.symbol == symbol
        }
        for symbol in SYMBOLS
    } == {symbol: {NS + entry_latencies[symbol]} for symbol in SYMBOLS}


def test_round74_target_evidence_is_self_hashing_and_environment_consistent() -> None:
    spec = _spec()
    evidence_payload = spec.commission_evidence.as_dict()
    tampered = dict(evidence_payload)
    tampered["record_count"] = 4
    with pytest.raises(ValueError, match="evidence payload digest"):
        Round74EventTargetEvidence.from_dict(tampered)
    with pytest.raises(ValueError, match="evidence source differs"):
        replace(spec.commission_evidence, source_id="unverified")
    with pytest.raises(ValueError, match="record count differs"):
        replace(spec.commission_evidence, record_count=2)

    testnet_slippage = replace(
        spec.slippage_evidence,
        environment="binance_usdm_testnet",
    )
    with pytest.raises(ValueError, match="environments differ"):
        replace(spec, slippage_evidence=testnet_slippage)


def test_round74_target_engine_rejects_rules_not_bound_by_evidence() -> None:
    rules = _rules()
    rules["BTCUSDT"] = Round73MarketQuantityRules.create(
        symbol="BTCUSDT",
        step_size="0.01",
        minimum_quantity="0.01",
        maximum_quantity="1000",
        minimum_notional="10",
    )

    with pytest.raises(ValueError, match="quantity-rules claims differ"):
        Round74EventTargetEngine(
            spec=_spec(),
            anchors=[],
            quantity_rules=rules,
        )


def test_round74_target_engine_builds_complete_pathwise_panel() -> None:
    engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )

    outcomes = _complete_engine(engine)

    assert len(outcomes) == 8
    assert all(outcome.eligible for outcome in outcomes)
    assert len({outcome.outcome_sha256 for outcome in outcomes}) == 8
    assert {outcome.target_context_sha256 for outcome in outcomes} == {
        engine.target_context_sha256
    }
    one_second_long = next(
        outcome
        for outcome in outcomes
        if outcome.horizon_seconds == 1 and outcome.side == "long"
    )
    one_second_short = next(
        outcome
        for outcome in outcomes
        if outcome.horizon_seconds == 1 and outcome.side == "short"
    )
    assert one_second_long.net_payoff_bps > 0.0
    assert one_second_long.positive_net_payoff is True
    assert one_second_long.adverse_selection is False
    assert one_second_long.maximum_adverse_excursion_bps > 0.0
    assert one_second_long.regime_unpredictability == pytest.approx(0.0)
    assert one_second_long.gross_payoff_bps == pytest.approx(
        one_second_long.midpoint_payoff_bps
        - one_second_long.book_walk_implementation_shortfall_bps
    )
    assert one_second_long.total_implementation_shortfall_bps == pytest.approx(
        one_second_long.book_walk_implementation_shortfall_bps
        + one_second_long.explicit_cost_bps
    )
    assert one_second_long.net_payoff_bps == pytest.approx(
        one_second_long.midpoint_payoff_bps
        - one_second_long.total_implementation_shortfall_bps
    )
    assert "total_cost_quote" not in one_second_long.as_dict()
    corrupted = replace(
        one_second_long,
        explicit_cost_bps=one_second_long.explicit_cost_bps + 1.0,
    )
    with pytest.raises(ValueError, match="accounting differs"):
        corrupted.validate()
    assert one_second_short.net_payoff_bps < 0.0
    assert one_second_short.adverse_selection is True


def test_round74_target_engine_applies_separate_measured_exit_latency() -> None:
    fast = Round74EventTargetEngine(
        spec=_spec(exit_latency_ns=100_000_000),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    slow = Round74EventTargetEngine(
        spec=_spec(exit_latency_ns=300_000_000),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    observations = (
        (900_000_000, 100.0),
        (1_000_000_000, 100.0),
        (1_100_000_000, 100.0),
        (1_300_000_000, 100.0),
        (1_500_000_000, 100.0),
        (1_700_000_000, 100.0),
        (1_900_000_000, 100.0),
        (2_100_000_000, 100.0),
        (2_200_000_000, 101.0),
        (2_400_000_000, 95.0),
    )
    for frame_index, (received, mid) in enumerate(observations, start=1):
        for engine in (fast, slow):
            engine.observe_depth(
                received_monotonic_ns=received,
                frame_index=frame_index,
                message_index=0,
                state=_state(mid=mid, update_id=frame_index),
            )

    fast_one_second = next(
        outcome
        for outcome in fast.finish()
        if outcome.horizon_seconds == 1 and outcome.side == "long"
    )
    slow_one_second = next(
        outcome
        for outcome in slow.finish()
        if outcome.horizon_seconds == 1 and outcome.side == "long"
    )

    assert fast_one_second.eligible
    assert slow_one_second.eligible
    assert fast_one_second.actual_exit_monotonic_ns == 2_200_000_000
    assert slow_one_second.actual_exit_monotonic_ns == 2_400_000_000
    assert fast_one_second.net_payoff_bps > slow_one_second.net_payoff_bps


def test_round74_target_engine_accepts_only_causal_streamed_anchors() -> None:
    engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[],
        quantity_rules=_rules(),
    )
    engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    engine.add_anchor(
        Round74EventTargetAnchor(
            symbol="BTCUSDT",
            anchor_index=0,
            decision_monotonic_ns=NS,
            decision_wall_ns=1_784_000_000_000_000_000 + NS,
            endpoint_frame_index=2,
            endpoint_message_index=0,
            feature_window_sha256="c" * 64,
        )
    )
    with pytest.raises(ValueError, match="duplicated"):
        engine.add_anchor(
            Round74EventTargetAnchor(
                symbol="BTCUSDT",
                anchor_index=0,
                decision_monotonic_ns=2 * NS,
                decision_wall_ns=1_784_000_000_000_000_000 + 2 * NS,
                endpoint_frame_index=3,
                endpoint_message_index=0,
                feature_window_sha256="d" * 64,
            )
        )
    engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    _observe_dense_path(
        engine,
        start_ns=1_100_000_000,
        checkpoints=(
            (2_200_000_000, 101.0),
            (6_200_000_000, 102.0),
            (31_200_000_000, 103.0),
            (301_200_000_000, 104.0),
        ),
    )

    outcomes = engine.finish()

    assert len(outcomes) == 8
    assert all(outcome.eligible for outcome in outcomes)
    assert engine.finish() == outcomes
    with pytest.raises(ValueError, match="already finished"):
        engine.add_anchor(_anchor(index=1, decision_ns=400 * NS))

    late_engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[],
        quantity_rules=_rules(),
    )
    late_engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=1),
    )
    with pytest.raises(ValueError, match="added after"):
        late_engine.add_anchor(
            Round74EventTargetAnchor(
                symbol="BTCUSDT",
                anchor_index=0,
                decision_monotonic_ns=900_000_000,
                decision_wall_ns=1_784_000_000_900_000_000,
                endpoint_frame_index=1,
                endpoint_message_index=0,
                feature_window_sha256="c" * 64,
            )
        )


def test_round74_path_risk_uses_intervening_books_not_only_endpoint() -> None:
    engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    _observe_dense_path(
        engine,
        start_ns=1_100_000_000,
        checkpoints=(
            (1_600_000_000, 98.0),
            (2_200_000_000, 101.0),
            (6_200_000_000, 101.0),
            (31_200_000_000, 101.0),
            (301_200_000_000, 101.0),
        ),
    )
    outcomes = engine.finish()

    result = next(
        outcome
        for outcome in outcomes
        if outcome.horizon_seconds == 1 and outcome.side == "long"
    )
    assert result.net_payoff_bps > 0.0
    assert result.maximum_adverse_excursion_bps > 100.0
    assert result.regime_unpredictability > 0.5


def test_round74_target_engine_censors_funding_crossing_and_capacity() -> None:
    funding_engine = Round74EventTargetEngine(
        spec=_spec(
            funding_boundaries={
                "BTCUSDT": (3 * NS,),
                "ETHUSDT": (),
                "SOLUSDT": (),
            }
        ),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    funding_engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    funding_engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    funding_engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    _observe_dense_path(
        funding_engine,
        start_ns=1_100_000_000,
        checkpoints=((2_200_000_000, 101.0),),
    )
    funding = funding_engine.finish()

    assert sum(outcome.eligible for outcome in funding) == 2
    assert {
        outcome.ineligible_reason for outcome in funding if not outcome.eligible
    } == {"funding_boundary"}

    capacity_engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    capacity_engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1, quantity=0.001),
    )
    capacity_engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2, quantity=0.001),
    )
    capacity_engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3, quantity=0.001),
    )
    capacity = capacity_engine.finish()
    assert funding_engine.target_context_sha256 != capacity_engine.target_context_sha256
    assert not any(outcome.eligible for outcome in capacity)
    assert {outcome.ineligible_reason for outcome in capacity} == {"entry_capacity"}


@pytest.mark.parametrize(
    (
        "funding_boundaries",
        "funding_intervals",
        "funding_coverage",
        "expected_reason",
    ),
    (
        (
            {
                "BTCUSDT": (2_250_000_000,),
                "ETHUSDT": (),
                "SOLUSDT": (),
            },
            None,
            None,
            "funding_boundary",
        ),
        (
            {symbol: () for symbol in SYMBOLS},
            None,
            {
                "BTCUSDT": (0, 2_250_000_000),
                "ETHUSDT": (0, 1_000_000_000_000),
                "SOLUSDT": (0, 1_000_000_000_000),
            },
            "funding_coverage",
        ),
        (
            {symbol: () for symbol in SYMBOLS},
            {
                "BTCUSDT": ((2_250_000_000, 2_350_000_000),),
                "ETHUSDT": (),
                "SOLUSDT": (),
            },
            None,
            "funding_boundary",
        ),
    ),
)
def test_round74_target_engine_rechecks_funding_at_actual_exit(
    funding_boundaries: dict[str, tuple[int, ...]],
    funding_intervals: dict[str, tuple[tuple[int, int], ...]] | None,
    funding_coverage: dict[str, tuple[int, int]] | None,
    expected_reason: str,
) -> None:
    engine = Round74EventTargetEngine(
        spec=_spec(
            funding_boundaries=funding_boundaries,
            funding_intervals=funding_intervals,
            funding_coverage=funding_coverage,
        ),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    _observe_dense_path(
        engine,
        start_ns=1_100_000_000,
        checkpoints=((2_300_000_000, 101.0),),
    )

    outcomes = engine.finish()

    assert not any(outcome.eligible for outcome in outcomes)
    assert {outcome.ineligible_reason for outcome in outcomes} == {expected_reason}
    one_second = [outcome for outcome in outcomes if outcome.horizon_seconds == 1]
    assert {outcome.requested_exit_monotonic_ns for outcome in one_second} == {
        2_200_000_000
    }
    assert {outcome.actual_exit_monotonic_ns for outcome in one_second} == {
        2_300_000_000
    }


def test_round74_target_engine_censors_late_state_and_rejects_bad_order() -> None:
    late_engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    late_engine.observe_depth(
        received_monotonic_ns=900_000_000,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    late_engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=2,
        message_index=0,
        state=_state(update_id=2),
    )
    late_engine.observe_depth(
        received_monotonic_ns=1_400_000_000,
        frame_index=3,
        message_index=0,
        state=_state(update_id=3),
    )
    late = late_engine.finish()
    assert {outcome.ineligible_reason for outcome in late} == {"entry_state_late"}

    order_engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[],
        quantity_rules=_rules(),
    )
    order_engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=1,
        message_index=0,
        state=_state(update_id=1),
    )
    with pytest.raises(ValueError, match="global receipt order regressed"):
        order_engine.observe_depth(
            received_monotonic_ns=NS - 1,
            frame_index=2,
            message_index=0,
            state=_state(symbol="ETHUSDT", update_id=1),
        )


def test_round74_target_engine_excludes_later_tied_timestamp_messages() -> None:
    anchor = Round74EventTargetAnchor(
        symbol="BTCUSDT",
        anchor_index=0,
        decision_monotonic_ns=NS,
        decision_wall_ns=1_784_000_000_000_000_000 + NS,
        endpoint_frame_index=10,
        endpoint_message_index=5,
        feature_window_sha256="c" * 64,
    )
    engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[anchor],
        quantity_rules=_rules(),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=10,
        message_index=4,
        state=_state(mid=100.0, update_id=1),
    )
    engine.observe_depth(
        received_monotonic_ns=NS,
        frame_index=10,
        message_index=6,
        state=_state(mid=200.0, update_id=2),
    )
    engine.observe_depth(
        received_monotonic_ns=1_100_000_000,
        frame_index=11,
        message_index=0,
        state=_state(mid=200.0, update_id=3),
    )
    _observe_dense_path(
        engine,
        start_ns=1_100_000_000,
        checkpoints=(
            (2_200_000_000, 201.0),
            (6_200_000_000, 202.0),
            (31_200_000_000, 203.0),
            (301_200_000_000, 204.0),
        ),
    )

    outcomes = engine.finish()

    assert len(outcomes) == 8
    assert all(outcome.eligible for outcome in outcomes)
    assert {outcome.base_quantity for outcome in outcomes} == {1.0}
    assert {
        (
            outcome.actual_entry_monotonic_ns,
            outcome.actual_entry_frame_index,
            outcome.actual_entry_message_index,
        )
        for outcome in outcomes
    } == {(1_100_000_000, 11, 0)}


def test_round74_target_engine_censors_missing_path_state() -> None:
    engine = Round74EventTargetEngine(
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
    )
    _observe_checkpoints_without_path_fill(
        engine,
        checkpoints=(
            (900_000_000, 100.0, 1),
            (NS, 100.0, 2),
            (1_100_000_000, 100.0, 3),
            (1_400_000_000, 101.0, 4),
        ),
    )

    outcomes = engine.finish()

    assert not any(outcome.eligible for outcome in outcomes)
    assert {outcome.ineligible_reason for outcome in outcomes} == {"path_state_gap"}


def test_round74_target_spec_rejects_unverified_or_oversampled_inputs() -> None:
    spec_payload = _spec().as_dict()
    restored = Round74EventTargetSpec.from_dict(spec_payload)
    assert restored.spec_sha256 == spec_payload["spec_sha256"]
    assert spec_payload["funding_boundary_intervals_monotonic_ns"] == {
        "BTCUSDT": [],
        "ETHUSDT": [],
        "SOLUSDT": [],
    }
    assert spec_payload["funding_schedule_coverage_monotonic_ns"] == {
        symbol: [0, 1_000_000_000_000] for symbol in SYMBOLS
    }
    assert spec_payload["funding_schedule_is_mandatory_and_hash_bound"] is True
    assert (
        _spec(
            funding_boundaries={
                "BTCUSDT": (3 * NS,),
                "ETHUSDT": (),
                "SOLUSDT": (),
            }
        ).spec_sha256
        != restored.spec_sha256
    )
    tampered = dict(spec_payload)
    tampered["reference_quote_notional"] = 101.0
    with pytest.raises(ValueError, match="payload digest"):
        Round74EventTargetSpec.from_dict(tampered)

    with pytest.raises(ValueError, match="evidence claims differ"):
        replace(
            _spec(),
            decision_to_entry_latency_ns_by_symbol=(
                ("BTCUSDT", 200_000_000),
                ("ETHUSDT", 100_000_000),
                ("SOLUSDT", 100_000_000),
            ),
        )
    with pytest.raises(ValueError, match="oversampled"):
        Round74EventTargetEngine(
            spec=_spec(),
            anchors=[
                _anchor(index=0, decision_ns=NS),
                _anchor(index=1, decision_ns=NS + 500_000_000),
            ],
            quantity_rules=_rules(),
        )
    with pytest.raises(ValueError, match="slippage is too large"):
        _spec(slippage_bps=1_001.0)

    invalid_rules = _rules()
    invalid_rules["BTCUSDT"] = Round73MarketQuantityRules(
        symbol="BTCUSDT",
        step_size=Decimal("0.001"),
        minimum_quantity=Decimal("2"),
        maximum_quantity=Decimal("1"),
        minimum_notional=Decimal("10"),
    )
    with pytest.raises(ValueError, match="minimum market quantity exceeds"):
        Round74EventTargetEngine(
            spec=_spec(),
            anchors=[],
            quantity_rules=invalid_rules,
        )
    with pytest.raises(ValueError, match="funding schedule differs"):
        _spec(
            funding_boundaries={
                "BTCUSDT": (3 * NS, 3 * NS),
                "ETHUSDT": (),
                "SOLUSDT": (),
            }
        )
    with pytest.raises(ValueError, match="funding schedule differs"):
        _spec(funding_boundaries={"BTCUSDT": (3 * NS,)})
    with pytest.raises(ValueError, match="funding schedule differs"):
        _spec(
            funding_intervals={
                "BTCUSDT": ((2 * NS, 3 * NS), (3 * NS, 4 * NS)),
                "ETHUSDT": (),
                "SOLUSDT": (),
            }
        )
    with pytest.raises(ValueError, match="funding coverage differs"):
        _spec(
            funding_boundaries={
                "BTCUSDT": (3 * NS,),
                "ETHUSDT": (),
                "SOLUSDT": (),
            },
            funding_coverage={
                "BTCUSDT": (0, 2 * NS),
                "ETHUSDT": (0, 4 * NS),
                "SOLUSDT": (0, 4 * NS),
            },
        )
    with pytest.raises(ValueError, match="funding coverage differs"):
        _spec(funding_coverage={"BTCUSDT": (0, 4 * NS)})
    with pytest.raises(ValueError, match="evidence source payload"):
        _spec(funding_evidence_sha256="unverified")

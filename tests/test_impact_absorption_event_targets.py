from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_event_targets import (
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
    Round74EventTargetSpec,
    round74_event_action_payoff,
)
from simple_ai_trading.impact_absorption_targets import (
    Round73MarketQuantityRules,
    walk_round73_book,
)


NS = 1_000_000_000


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
        sum(price * qty for price, qty in bids[:levels])
        for levels in (5, 10, 20)
    )
    ask_depths = tuple(
        sum(price * qty for price, qty in asks[:levels])
        for levels in (5, 10, 20)
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
) -> Round74EventTargetSpec:
    return Round74EventTargetSpec.create(
        reference_quote_notional=100.0,
        decision_to_entry_latency_ns=latency_ns,
        taker_fee_bps_by_symbol={
            symbol: fee_bps for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        },
        additional_slippage_bps_per_side=slippage_bps,
        commission_evidence_sha256="a" * 64,
        latency_evidence_sha256="b" * 64,
        slippage_evidence_sha256="d" * 64,
    )


def _anchor(
    *,
    index: int = 0,
    decision_ns: int = NS,
) -> Round74EventTargetAnchor:
    return Round74EventTargetAnchor(
        symbol="BTCUSDT",
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
                    2_100_000_000,
                    6_100_000_000,
                    31_100_000_000,
                    301_100_000_000,
                ),
                path_mids,
                strict=True,
            )
        ),
    )
    return engine.finish()


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
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
    )

    assert payoff.gross_payoff_quote == pytest.approx(
        exit_walk.quote_notional - entry.quote_notional
    )
    assert payoff.commission_quote == pytest.approx(
        5.0 / 10_000.0
        * (entry.quote_notional + exit_walk.quote_notional)
    )
    assert payoff.additional_slippage_quote == pytest.approx(
        1.0 / 10_000.0
        * (entry.quote_notional + exit_walk.quote_notional)
    )
    assert payoff.net_payoff_quote < payoff.gross_payoff_quote


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
    assert one_second_short.net_payoff_bps < 0.0
    assert one_second_short.adverse_selection is True


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
            (2_100_000_000, 101.0),
            (6_100_000_000, 101.0),
            (31_100_000_000, 101.0),
            (301_100_000_000, 101.0),
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
        spec=_spec(),
        anchors=[_anchor()],
        quantity_rules=_rules(),
        funding_boundaries_monotonic_ns={"BTCUSDT": (3 * NS,)},
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
        checkpoints=((2_100_000_000, 101.0),),
    )
    funding = funding_engine.finish()

    assert sum(outcome.eligible for outcome in funding) == 2
    assert {
        outcome.ineligible_reason
        for outcome in funding
        if not outcome.eligible
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
    assert (
        funding_engine.target_context_sha256
        != capacity_engine.target_context_sha256
    )
    assert not any(outcome.eligible for outcome in capacity)
    assert {
        outcome.ineligible_reason for outcome in capacity
    } == {"entry_capacity"}


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
    assert {outcome.ineligible_reason for outcome in late} == {
        "entry_state_late"
    }

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
            (2_100_000_000, 201.0),
            (6_100_000_000, 202.0),
            (31_100_000_000, 203.0),
            (301_100_000_000, 204.0),
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
    assert {outcome.ineligible_reason for outcome in outcomes} == {
        "path_state_gap"
    }


def test_round74_target_spec_rejects_unverified_or_oversampled_inputs() -> None:
    spec_payload = _spec().as_dict()
    restored = Round74EventTargetSpec.from_dict(spec_payload)
    assert restored.spec_sha256 == spec_payload["spec_sha256"]
    tampered = dict(spec_payload)
    tampered["reference_quote_notional"] = 101.0
    with pytest.raises(ValueError, match="payload digest"):
        Round74EventTargetSpec.from_dict(tampered)

    with pytest.raises(ValueError, match="latency evidence"):
        Round74EventTargetSpec.create(
            reference_quote_notional=100.0,
            decision_to_entry_latency_ns=100_000_000,
            taker_fee_bps_by_symbol={
                symbol: 5.0
                for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
            },
            additional_slippage_bps_per_side=1.0,
            commission_evidence_sha256="a" * 64,
            latency_evidence_sha256="unverified",
            slippage_evidence_sha256="d" * 64,
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

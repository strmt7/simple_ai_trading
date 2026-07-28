from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventDatasetAssembler,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    build_round74_event_training_batch,
    iter_round74_labeled_event_windows,
    validate_round74_capture_report_binding,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    fit_round74_event_feature_scaler,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    Round74EventToken,
    Round74ReplayObservation,
)
from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    Round74EventTargetEngine,
    Round74EventTargetEvidence,
    Round74EventTargetSpec,
    round74_commission_evidence_claims,
    round74_funding_schedule_evidence_claims,
    round74_latency_evidence_claims,
    round74_quantity_rules_evidence_claims,
    round74_slippage_evidence_claims,
)
from simple_ai_trading.impact_absorption_targets import (
    Round73MarketQuantityRules,
)
from simple_ai_trading.impact_absorption_store import ImpactAbsorptionStore


NS = 1_000_000_000
WALL = 1_800_000_000_000_000_000


def _state(*, update_id: int, mid: float = 100.0) -> L2BookState:
    bids = tuple((mid - 0.1 - index * 0.1, 10.0) for index in range(20))
    asks = tuple((mid + 0.1 + index * 0.1, 10.0) for index in range(20))
    bid_depths = tuple(
        sum(price * quantity for price, quantity in bids[:levels])
        for levels in (5, 10, 20)
    )
    ask_depths = tuple(
        sum(price * quantity for price, quantity in asks[:levels])
        for levels in (5, 10, 20)
    )
    imbalances = tuple(
        (bid - ask) / (bid + ask)
        for bid, ask in zip(bid_depths, ask_depths, strict=True)
    )
    return L2BookState(
        symbol="BTCUSDT",
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


def _engine() -> Round74EventTargetEngine:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    entry_latencies = {symbol: 100_000_000 for symbol in symbols}
    exit_latencies = {symbol: 100_000_000 for symbol in symbols}
    fees = {symbol: 5.0 for symbol in symbols}
    funding_intervals = {symbol: () for symbol in symbols}
    funding_coverage = {
        symbol: (0, 1_000_000_000_000) for symbol in symbols
    }
    slippage = {symbol: 1.0 for symbol in symbols}
    rules = _rules()

    def evidence(
        kind: str,
        claims: object,
        payload_sha256: str,
    ) -> Round74EventTargetEvidence:
        return Round74EventTargetEvidence.create(
            kind=kind,
            environment="binance_usdm_mainnet",
            observed_wall_ns=WALL,
            record_count=3,
            source_query_or_protocol_sha256=f"{len(kind):064x}",
            source_payload_sha256=payload_sha256,
            claims=claims,
        )

    spec = Round74EventTargetSpec.create(
        reference_quote_notional=100.0,
        decision_to_entry_latency_ns_by_symbol=entry_latencies,
        decision_to_exit_latency_ns_by_symbol=exit_latencies,
        taker_fee_bps_by_symbol=fees,
        funding_boundary_intervals_monotonic_ns=funding_intervals,
        funding_schedule_coverage_monotonic_ns=funding_coverage,
        additional_slippage_bps_per_side_by_symbol=slippage,
        quantity_rules_evidence=evidence(
            "quantity_rules",
            round74_quantity_rules_evidence_claims(rules),
            "e" * 64,
        ),
        commission_evidence=evidence(
            "commission",
            round74_commission_evidence_claims(fees),
            "a" * 64,
        ),
        entry_exit_latency_evidence=evidence(
            "entry_exit_latency",
            round74_latency_evidence_claims(
                decision_to_entry_latency_ns_by_symbol=entry_latencies,
                decision_to_exit_latency_ns_by_symbol=exit_latencies,
            ),
            "b" * 64,
        ),
        slippage_evidence=evidence(
            "residual_slippage",
            round74_slippage_evidence_claims(
                reference_quote_notional=100.0,
                additional_slippage_bps_per_side_by_symbol=slippage,
            ),
            "c" * 64,
        ),
        funding_schedule_evidence=evidence(
            "funding_schedule",
            round74_funding_schedule_evidence_claims(
                funding_boundary_intervals_monotonic_ns=funding_intervals,
                funding_schedule_coverage_monotonic_ns=funding_coverage,
            ),
            "d" * 64,
        ),
    )
    return Round74EventTargetEngine(
        spec=spec,
        anchors=[],
        quantity_rules=rules,
    )


def _partition() -> Round74EventRunPartition:
    entries = []
    for index, role in enumerate(("training", "tuning", "test")):
        start = WALL + index * 1_200 * NS
        anchor = start + (
            12_700_000_000 if index == 0 else 310_500_000_000
        )
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=f"{index + 1:032x}",
                role=role,
                capture_report_sha256=f"{index + 1:064x}",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + 1_000 * NS,
                eligible_anchor_start_wall_ns=anchor,
                eligible_anchor_end_wall_ns=(
                    anchor if index == 0 else start + 600 * NS
                ),
            )
        )
    return Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="d" * 64,
    )


def _token(*, index: int, monotonic_ns: int) -> Round74EventToken:
    features = [0.0] * len(ROUND74_EVENT_FEATURE_NAMES)
    features[0] = 1.0
    features[5] = 1.0
    token = Round74EventToken(
        symbol="BTCUSDT",
        event_type="depthUpdate",
        frame_index=index,
        message_index=0,
        received_monotonic_ns=monotonic_ns,
        received_wall_ns=WALL + monotonic_ns,
        exchange_event_time_ms=1_000 + index,
        source_sequence_number=index,
        feature_values=tuple(features),
    )
    token.validate()
    return token


def _observation(
    *,
    index: int,
    monotonic_ns: int,
    token: bool,
    stale: bool = False,
) -> Round74ReplayObservation:
    selected_token = _token(index=index, monotonic_ns=monotonic_ns) if token else None
    observation = Round74ReplayObservation(
        symbol="BTCUSDT",
        event_type="depthUpdate",
        frame_index=index,
        message_index=0,
        received_monotonic_ns=monotonic_ns,
        received_wall_ns=WALL + monotonic_ns,
        token=selected_token,
        depth_state=_state(
            update_id=index + 1,
            mid=100.0 + monotonic_ns / NS * 0.001,
        ),
        depth_update_is_stale=stale,
    )
    observation.validate()
    return observation


def test_round74_partition_is_whole_run_chronological_and_hash_bound() -> None:
    partition = _partition()

    partition.validate()
    restored = Round74EventRunPartition.from_dict(partition.as_dict())

    assert restored.partition_sha256 == partition.partition_sha256
    assert partition.as_dict()["split_unit"] == "whole_capture_run"
    assert partition.as_dict()["random_row_split_permitted"] is False
    assert (
        partition.as_dict()["embargo_axis"]
        == "elapsed_wall_time_after_prior_audited_usable_end"
    )

    tampered = partition.as_dict()
    tampered["purge_ns"] = 1
    with pytest.raises(ValueError, match="payload digest"):
        Round74EventRunPartition.from_dict(tampered)

    entries = list(partition.entries)
    entries[1] = replace(
        entries[1],
        eligible_anchor_start_wall_ns=(entries[1].capture_start_wall_ns + 100 * NS),
    )
    with pytest.raises(ValueError, match="transition is not purged"):
        Round74EventRunPartition(
            entries=tuple(entries),
            cohort_plan_sha256=partition.cohort_plan_sha256,
        ).validate()

    validate_round74_capture_report_binding(
        partition.entries[0],
        stored_capture_report_sha256=partition.entries[0].capture_report_sha256,
    )
    with pytest.raises(ValueError, match="capture report differs"):
        validate_round74_capture_report_binding(
            partition.entries[0],
            stored_capture_report_sha256="f" * 64,
        )
    with pytest.raises(TypeError, match="ImpactAbsorptionStore"):
        list(
            iter_round74_labeled_event_windows(
                object(),
                partition=partition,
                run_id=partition.entries[0].run_id,
                target_engine=_engine(),
            )
        )
    with pytest.raises(ValueError, match="read-only store"):
        list(
            iter_round74_labeled_event_windows(
                ImpactAbsorptionStore(":memory:"),
                partition=partition,
                run_id=partition.entries[0].run_id,
                target_engine=_engine(),
            )
        )


def test_round74_streaming_assembler_builds_complete_bounded_panel() -> None:
    assembler = Round74EventDatasetAssembler(
        partition=_partition(),
        run_id=f"{1:032x}",
        target_engine=_engine(),
    )
    completed = []
    for index in range(128):
        completed.extend(
            assembler.consume(
                _observation(
                    index=index,
                    monotonic_ns=index * 100_000_000,
                    token=True,
                )
            )
        )
    assert assembler.pending_window_count == 1
    for index, monotonic_ns in enumerate(
        range(12_800_000_000, 313_000_000_001, 200_000_000),
        start=128,
    ):
        completed.extend(
            assembler.consume(
                _observation(
                    index=index,
                    monotonic_ns=monotonic_ns,
                    token=False,
                )
            )
        )
    completed.extend(assembler.finish())

    assert len(completed) == 1
    sample = completed[0]
    assert sample.role == "training"
    assert sample.eligible_action_count == 8
    assert len(sample.outcomes) == 8
    assert sample.sample_sha256 == sample.sample_sha256
    assert assembler.pending_window_count == 0
    scaler = fit_round74_event_feature_scaler(
        np.asarray(sample.feature_values, dtype=np.float64),
        partition_role="training",
    )
    batch = build_round74_event_training_batch([sample], scaler=scaler)
    assert batch.feature_values.shape == (
        1,
        128,
        len(ROUND74_EVENT_FEATURE_NAMES),
    )
    assert batch.run_id == (sample.run_id,)
    assert batch.symbol == (sample.symbol,)
    assert batch.decision_wall_ns.tolist() == [sample.decision_wall_ns]
    assert batch.decision_monotonic_ns.tolist() == [sample.decision_monotonic_ns]
    assert batch.endpoint_frame_index.tolist() == [sample.endpoint_frame_index]
    assert batch.endpoint_message_index.tolist() == [sample.endpoint_message_index]
    assert batch.anchor_index.tolist() == [sample.anchor_index]
    assert batch.test_access_sha256 == ("",)
    assert float(batch.action_eligibility.sum()) == 8.0
    selected = next(outcome for outcome in sample.outcomes if outcome.eligible)
    horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(
        selected.horizon_seconds
    )
    side_index = ROUND74_EVENT_PAYOFF_SIDES.index(selected.side)
    assert batch.net_payoff_bps[0, horizon_index, side_index] == pytest.approx(
        selected.capital_scaled_net_payoff_bps
    )
    assert batch.maximum_adverse_excursion_bps[
        0,
        horizon_index,
        side_index,
    ] == pytest.approx(selected.capital_scaled_maximum_adverse_excursion_bps)
    assert selected.capital_scaled_net_payoff_bps != pytest.approx(
        selected.net_payoff_bps
    )
    assert len(batch.batch_sha256) == 64
    assert not batch.feature_values.flags.writeable


def test_round74_streaming_assembler_never_uses_stale_depth_as_state() -> None:
    assembler = Round74EventDatasetAssembler(
        partition=_partition(),
        run_id=f"{1:032x}",
        target_engine=_engine(),
    )
    for index in range(128):
        assert (
            assembler.consume(
                _observation(
                    index=index,
                    monotonic_ns=index * 100_000_000,
                    token=True,
                    stale=True,
                )
            )
            == ()
        )

    completed = list(
        assembler.consume(
            _observation(
                index=128,
                monotonic_ns=12_800_000_000,
                token=False,
            )
        )
    )
    completed.extend(assembler.finish())

    assert len(completed) == 1
    assert completed[0].eligible_action_count == 0
    assert {outcome.ineligible_reason for outcome in completed[0].outcomes} == {
        "decision_state_missing"
    }
    scaler = fit_round74_event_feature_scaler(
        np.asarray(completed[0].feature_values, dtype=np.float64),
        partition_role="training",
    )
    with pytest.raises(ValueError, match="no eligible actions"):
        build_round74_event_training_batch(completed, scaler=scaler)


def test_round74_test_access_and_exact_order_fail_closed() -> None:
    partition = _partition()
    with pytest.raises(ValueError, match="pretest model policy"):
        Round74EventDatasetAssembler(
            partition=partition,
            run_id=f"{3:032x}",
            target_engine=_engine(),
        )
    test_assembler = Round74EventDatasetAssembler(
        partition=partition,
        run_id=f"{3:032x}",
        target_engine=_engine(),
        pretest_model_policy_sha256="d" * 64,
        test_unlock_sha256="e" * 64,
    )
    assert len(test_assembler.test_access_sha256) == 64
    with pytest.raises(ValueError, match="test authorization"):
        Round74EventDatasetAssembler(
            partition=partition,
            run_id=f"{1:032x}",
            target_engine=_engine(),
            pretest_model_policy_sha256="d" * 64,
            test_unlock_sha256="e" * 64,
        )

    observation = _observation(index=0, monotonic_ns=2_400 * NS, token=False)
    test_assembler.consume(observation)
    with pytest.raises(ValueError, match="order regressed"):
        test_assembler.consume(observation)

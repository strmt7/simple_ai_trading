from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_DEFAULT_PROFILE,
    ROUND74_ACTION_POSITION_CAPITAL_FRACTION,
    Round74ActionExecutionOutcomeRow,
    Round74ActionExecutionPanel,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    round74_action_profile,
    select_round74_action_policy,
    select_round74_action_policy_batches,
    simulate_round74_action_trace,
    simulate_round74_action_trace_batches,
    _eligible_target_score_threshold,
    _equal_run_score_threshold,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION,
    ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION,
    Round74ProbabilityCalibration,
    Round74RiskQuantileCalibration,
    Round74TemperatureFit,
    Round74TuningSubpartition,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_model import (
    Round74EventModelOutput,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
    ROUND74_EVENT_SYMBOLS,
)
from simple_ai_trading.impact_absorption_event_targets import (
    Round74EventTargetOutcome,
)
from simple_ai_trading.round74_segmented_model_operator import (
    Round74SegmentedTuningSubpartition,
)


POLICY_SHA256 = "1" * 64
PARTITION_SHA256 = "2" * 64
SCALER_SHA256 = "3" * 64


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _subpartition() -> Round74TuningSubpartition:
    run_ids = tuple(f"{index:032x}" for index in range(1, 25))
    result = Round74TuningSubpartition(
        parent_partition_sha256=PARTITION_SHA256,
        model_selection_run_ids=run_ids[:12],
        calibration_run_ids=run_ids[12:18],
        policy_selection_run_ids=run_ids[18:],
    )
    result.validate()
    return result


def _segmented_subpartition() -> Round74SegmentedTuningSubpartition:
    run_ids = tuple(f"{index:032x}" for index in range(100, 193))
    model_ordinals = tuple(
        ordinal for ordinal in range(514, 557) if ordinal not in {520, 530, 545}
    )
    calibration_ordinals = tuple(
        ordinal for ordinal in range(557, 579) if ordinal not in {565, 566, 567}
    )
    policy_ordinals = tuple(
        ordinal for ordinal in range(579, 600) if ordinal not in {589, 590}
    )
    result = Round74SegmentedTuningSubpartition(
        parent_partition_sha256=PARTITION_SHA256,
        cohort_plan_sha256="a" * 64,
        model_selection_run_ids=run_ids[:40],
        calibration_run_ids=run_ids[40:59],
        policy_selection_run_ids=run_ids[59:78],
        ai_qualification_run_ids=run_ids[78:],
        model_selection_slot_ordinals=model_ordinals,
        calibration_slot_ordinals=calibration_ordinals,
        policy_selection_slot_ordinals=policy_ordinals,
        ai_qualification_slot_ordinals=tuple(range(600, 615)),
        model_selection_eligible_anchor_ns=(900_000_000_000,) * 40,
        calibration_eligible_anchor_ns=(900_000_000_000,) * 19,
        policy_selection_eligible_anchor_ns=(900_000_000_000,) * 19,
        ai_qualification_eligible_anchor_ns=(900_000_000_000,) * 15,
    )
    result.validate()
    return result


def _fit(calibration_runs: int = 6) -> Round74TemperatureFit:
    return Round74TemperatureFit(
        temperature=1.0,
        eligible_observations=10,
        positive_observations=5,
        calibration_runs=calibration_runs,
        minimum_run_observations=1,
        maximum_run_observations=2,
        uncalibrated_run_balanced_nll=0.5,
        calibrated_run_balanced_nll=0.5,
        uncalibrated_nll=0.5,
        calibrated_nll=0.5,
        uncalibrated_brier=0.2,
        calibrated_brier=0.2,
        uncalibrated_ece=0.1,
        calibrated_ece=0.1,
    )


def _calibration(
    subpartition: Round74TuningSubpartition
    | Round74SegmentedTuningSubpartition
    | None = None,
) -> Round74ProbabilityCalibration:
    selected = subpartition or _subpartition()
    calibration_runs = len(selected.calibration_run_ids)
    fit = _fit(calibration_runs)
    return Round74ProbabilityCalibration(
        pretest_policy_sha256=POLICY_SHA256,
        tuning_subpartition_sha256=selected.subpartition_sha256,
        calibration_source_sha256="4" * 64,
        calibration_data_sha256="5" * 64,
        calibration_run_ids=selected.calibration_run_ids,
        calibration_row_run_ids_sha256="6" * 64,
        positive_payoff=fit,
        adverse_selection=fit,
        regime_unpredictability=fit,
        backend_kind="cpu",
        backend_device="test",
        optimization_population=(
            "eligible_target"
            if isinstance(selected, Round74SegmentedTuningSubpartition)
            else "capture_run"
        ),
        schema_version=ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION,
    )


def _risk_quantile_calibration() -> Round74RiskQuantileCalibration:
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    lower_offsets = tuple(
        tuple((100.0, 100.0) for _side in range(sides)) for _horizon in range(horizons)
    )
    coverage_before = tuple(
        tuple((0.5, 0.5) for _side in range(sides)) for _horizon in range(horizons)
    )
    coverage_after = tuple(
        tuple((1.0, 1.0) for _side in range(sides)) for _horizon in range(horizons)
    )
    return Round74RiskQuantileCalibration(
        payoff_lower_offsets_bps=lower_offsets,
        mae_upper_offsets_bps=tuple(
            tuple(100.0 for _side in range(sides)) for _horizon in range(horizons)
        ),
        eligible_observations=tuple(
            tuple(10 for _side in range(sides)) for _horizon in range(horizons)
        ),
        payoff_lower_empirical_coverage_before=coverage_before,
        payoff_lower_empirical_coverage_after=coverage_after,
        mae_upper_empirical_coverage_before=tuple(
            tuple(0.5 for _side in range(sides)) for _horizon in range(horizons)
        ),
        mae_upper_empirical_coverage_after=tuple(
            tuple(1.0 for _side in range(sides)) for _horizon in range(horizons)
        ),
        calibration_runs=6,
        optimization_population="capture_run",
    )


def _batch(
    *,
    payoff_sign: float = 1.0,
    role: str = "tuning",
    subpartition: (
        Round74TuningSubpartition | Round74SegmentedTuningSubpartition | None
    ) = None,
) -> Round74EventTrainingBatch:
    selected = subpartition or _subpartition()
    runs = selected.policy_selection_run_ids
    rows_per_run = len(ROUND74_EVENT_SYMBOLS) * 2
    rows = len(runs) * rows_per_run
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    run_id: list[str] = []
    symbols: list[str] = []
    monotonic: list[int] = []
    wall: list[int] = []
    for run_index, run in enumerate(runs):
        for local_index in range(rows_per_run):
            run_id.append(run)
            symbols.append(
                ROUND74_EVENT_SYMBOLS[local_index % len(ROUND74_EVENT_SYMBOLS)]
            )
            monotonic.append(1_000_000_000 + local_index * 1_000_000_000)
            wall.append(
                1_800_000_000_000_000_000
                + run_index * 100_000_000_000
                + local_index * 1_000_000_000
            )
    feature_values = np.zeros(
        (
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    for index, symbol in enumerate(symbols):
        feature_values[
            index,
            :,
            ROUND74_EVENT_FEATURE_NAMES.index(f"symbol_is_{symbol.lower()}"),
        ] = 1.0
    entry = np.empty(action_shape, dtype=np.int64)
    exit_value = np.empty(action_shape, dtype=np.int64)
    for row_index, value in enumerate(monotonic):
        entry[row_index, :, :] = value + 100_000_000
        exit_value[row_index, :, :] = value + 30_100_000_000
    payoff = np.full(action_shape, 2.0 * payoff_sign, dtype=np.float32)
    mae = np.full(action_shape, 1.0, dtype=np.float32)
    adverse = np.zeros(action_shape, dtype=np.float32)
    eligibility = np.ones(action_shape, dtype=np.float32)
    unpredictability = np.zeros(regime_shape, dtype=np.float32)
    regime_eligibility = np.ones(regime_shape, dtype=np.float32)
    result = Round74EventTrainingBatch(
        role=role,
        partition_sha256=PARTITION_SHA256,
        scaler_sha256=SCALER_SHA256,
        run_id=tuple(run_id),
        symbol=tuple(symbols),
        decision_monotonic_ns=_readonly(np.asarray(monotonic, dtype=np.int64)),
        decision_wall_ns=_readonly(np.asarray(wall, dtype=np.int64)),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.arange(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{1000 + index:064x}" for index in range(rows)),
        feature_window_sha256=tuple(f"{2000 + index:064x}" for index in range(rows)),
        target_context_sha256=tuple("6" * 64 for _ in range(rows)),
        test_access_sha256=tuple("" for _ in range(rows)),
        feature_values=_readonly(feature_values),
        actual_entry_monotonic_ns=_readonly(entry),
        actual_exit_monotonic_ns=_readonly(exit_value),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(mae),
        adverse_selection=_readonly(adverse),
        regime_unpredictability=_readonly(unpredictability),
        action_eligibility=_readonly(eligibility),
        regime_unpredictability_eligibility=_readonly(regime_eligibility),
    )
    result.validate()
    return result


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _output(rows: int) -> Round74EventModelOutput:
    horizons = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    sides = len(ROUND74_EVENT_PAYOFF_SIDES)
    quantiles = len(ROUND74_EVENT_PAYOFF_QUANTILES)
    payoff = torch.full(
        (rows, horizons, sides, quantiles),
        -5.0,
        dtype=torch.float32,
    )
    mae = torch.ones_like(payoff)
    positive = torch.full(
        (rows, horizons, sides),
        _logit(0.20),
        dtype=torch.float32,
    )
    adverse = torch.full(
        (rows, horizons, sides),
        _logit(0.80),
        dtype=torch.float32,
    )
    regime = torch.full(
        (rows, horizons),
        _logit(0.80),
        dtype=torch.float32,
    )
    for row_index in range(rows):
        local_index = row_index % (len(ROUND74_EVENT_SYMBOLS) * 2)
        median = 8.0 + local_index * 0.5
        for horizon in (30, 300):
            horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon)
            payoff[
                row_index,
                horizon_index,
                0,
                :,
            ] = torch.tensor((2.0, 4.0, median, median + 2.0, median + 4.0))
            mae[
                row_index,
                horizon_index,
                0,
                :,
            ] = torch.tensor((0.2, 0.4, 0.8, 1.2, 2.0))
            positive[row_index, horizon_index, 0] = _logit(0.80)
            adverse[row_index, horizon_index, 0] = _logit(0.10)
            regime[row_index, horizon_index] = _logit(0.10)
        payoff[row_index, 3, 0, :] -= 1.0
    mae[:, :2, 0, :] = mae[:, 2:3, 0, :]
    result = Round74EventModelOutput(
        payoff_quantiles_bps=payoff,
        maximum_adverse_excursion_quantiles_bps=mae,
        positive_payoff_logits=positive,
        adverse_selection_logits=adverse,
        regime_unpredictability_logits=regime,
    )
    result.validate(rows)
    return result


def _candidates(
    batch: Round74EventTrainingBatch,
    *,
    profile: str = "conservative",
    calibration: Round74ProbabilityCalibration | None = None,
):
    return derive_round74_action_candidates(
        _output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration or _calibration(),
        pretest_policy_sha256=POLICY_SHA256,
        profile=profile,
    )


def _slice_batch(
    batch: Round74EventTrainingBatch,
    start: int,
    stop: int,
) -> Round74EventTrainingBatch:
    result = replace(
        batch,
        run_id=batch.run_id[start:stop],
        symbol=batch.symbol[start:stop],
        decision_monotonic_ns=batch.decision_monotonic_ns[start:stop],
        decision_wall_ns=batch.decision_wall_ns[start:stop],
        endpoint_frame_index=batch.endpoint_frame_index[start:stop],
        endpoint_message_index=batch.endpoint_message_index[start:stop],
        anchor_index=batch.anchor_index[start:stop],
        sample_sha256=batch.sample_sha256[start:stop],
        feature_window_sha256=batch.feature_window_sha256[start:stop],
        target_context_sha256=batch.target_context_sha256[start:stop],
        test_access_sha256=batch.test_access_sha256[start:stop],
        feature_values=batch.feature_values[start:stop],
        actual_entry_monotonic_ns=batch.actual_entry_monotonic_ns[start:stop],
        actual_exit_monotonic_ns=batch.actual_exit_monotonic_ns[start:stop],
        net_payoff_bps=batch.net_payoff_bps[start:stop],
        maximum_adverse_excursion_bps=(batch.maximum_adverse_excursion_bps[start:stop]),
        adverse_selection=batch.adverse_selection[start:stop],
        regime_unpredictability=batch.regime_unpredictability[start:stop],
        action_eligibility=batch.action_eligibility[start:stop],
        regime_unpredictability_eligibility=(
            batch.regime_unpredictability_eligibility[start:stop]
        ),
    )
    result.validate()
    return result


def _run_batch_panel(
    batch: Round74EventTrainingBatch,
) -> tuple[Round74EventTrainingBatch, ...]:
    rows_per_run = len(ROUND74_EVENT_SYMBOLS) * 2
    return tuple(
        _slice_batch(batch, start, start + rows_per_run)
        for start in range(0, batch.rows, rows_per_run)
    )


def _delayed_outcome(
    batch: Round74EventTrainingBatch,
    row_index: int,
    *,
    horizon_seconds: int,
    side: str,
) -> Round74EventTargetOutcome:
    entry = int(batch.decision_monotonic_ns[row_index]) + 900_000_000
    exit_value = entry + horizon_seconds * 1_000_000_000
    result = Round74EventTargetOutcome(
        symbol=batch.symbol[row_index],
        anchor_index=int(batch.anchor_index[row_index]),
        horizon_seconds=horizon_seconds,
        side=side,
        eligible=True,
        ineligible_reason="",
        requested_entry_monotonic_ns=entry - 100_000_000,
        actual_entry_monotonic_ns=entry,
        actual_entry_frame_index=10_000 + row_index,
        actual_entry_message_index=0,
        requested_exit_monotonic_ns=exit_value - 100_000_000,
        actual_exit_monotonic_ns=exit_value,
        actual_exit_frame_index=20_000 + row_index,
        actual_exit_message_index=0,
        base_quantity=1.0,
        reference_quote_notional=10_000.0,
        entry_quote_notional=10_000.0,
        entry_average_price=10_000.0,
        exit_average_price=10_000.0,
        midpoint_payoff_bps=1.0,
        book_walk_implementation_shortfall_quote=1.0,
        book_walk_implementation_shortfall_bps=1.0,
        gross_payoff_bps=0.0,
        commission_quote=1.0,
        commission_bps=1.0,
        additional_slippage_quote=1.0,
        additional_slippage_bps=1.0,
        explicit_cost_quote=2.0,
        explicit_cost_bps=2.0,
        total_implementation_shortfall_quote=3.0,
        total_implementation_shortfall_bps=3.0,
        net_payoff_bps=-2.0,
        capital_scaled_net_payoff_bps=-2.0,
        positive_net_payoff=False,
        maximum_adverse_excursion_bps=2.0,
        capital_scaled_maximum_adverse_excursion_bps=2.0,
        maximum_favorable_excursion_bps=1.0,
        adverse_selection=True,
        regime_unpredictability=0.5,
        maximum_spread_bps=1.0,
        minimum_exit_side_capacity_ratio=2.0,
        entry_update_id=100 + row_index,
        exit_update_id=200 + row_index,
        target_spec_sha256="7" * 64,
        target_context_sha256="8" * 64,
        feature_window_sha256=batch.feature_window_sha256[row_index],
    )
    result.validate()
    return result


def _execution_panel(
    batches: tuple[Round74EventTrainingBatch, ...],
    candidates: tuple,
) -> Round74ActionExecutionPanel:
    rows = tuple(
        Round74ActionExecutionOutcomeRow(
            feature_row_sha256=candidate.feature_row_sha256[row_index],
            run_id=batch.run_id[row_index],
            symbol=batch.symbol[row_index],
            anchor_index=int(batch.anchor_index[row_index]),
            feature_window_sha256=batch.feature_window_sha256[row_index],
            outcomes=tuple(
                _delayed_outcome(
                    batch,
                    row_index,
                    horizon_seconds=horizon,
                    side=side,
                )
                for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
                for side in ROUND74_EVENT_PAYOFF_SIDES
            ),
        )
        for batch, candidate in zip(batches, candidates, strict=True)
        for row_index in range(batch.rows)
    )
    run_ids = _subpartition().policy_selection_run_ids
    result = Round74ActionExecutionPanel(
        profile="conservative",
        partition_sha256=PARTITION_SHA256,
        decision_latency_evidence_sha256="9" * 64,
        additional_entry_latency_ns=800_000_000,
        source_target_assembly_sha256=tuple((run_id, "a" * 64) for run_id in run_ids),
        source_capture_report_sha256=tuple((run_id, "b" * 64) for run_id in run_ids),
        execution_replay_module_sha256=(
            hashlib.sha256(
                (
                    Path(__file__).parents[1]
                    / "src"
                    / "simple_ai_trading"
                    / "round74_delayed_execution_panel.py"
                )
                .read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"\r", b"\n")
            ).hexdigest()
        ),
        rows=rows,
    )
    result.validate()
    return result


def test_default_profile_is_conservative_and_profiles_relax_monotonically() -> None:
    assert ROUND74_ACTION_DEFAULT_PROFILE == "conservative"
    conservative = round74_action_profile()
    regular = round74_action_profile("regular")
    aggressive = round74_action_profile("aggressive")

    assert conservative.profile == "conservative"
    assert (
        conservative.minimum_positive_probability
        > regular.minimum_positive_probability
        > aggressive.minimum_positive_probability
    )
    assert (
        conservative.maximum_adverse_probability
        < regular.maximum_adverse_probability
        < aggressive.maximum_adverse_probability
    )
    assert (
        conservative.objective_drawdown_penalty
        > regular.objective_drawdown_penalty
        > aggressive.objective_drawdown_penalty
    )


def test_threshold_grid_is_invariant_to_busy_run_row_duplication() -> None:
    runs = _subpartition().policy_selection_run_ids
    base = {
        runs[0]: (100.0,),
        runs[1]: (1.0,),
        runs[2]: (2.0,),
        runs[3]: (3.0,),
        runs[4]: (4.0,),
        runs[5]: (5.0,),
    }
    duplicated = {
        **base,
        runs[0]: (100.0,) * 100,
    }

    expected = _equal_run_score_threshold(
        base,
        quantile=0.5,
        expected_run_ids=runs,
    )
    observed = _equal_run_score_threshold(
        duplicated,
        quantile=0.5,
        expected_run_ids=runs,
    )

    assert expected == 3.5
    assert observed == expected


def test_eligible_target_threshold_weights_duration_normalized_rows() -> None:
    runs = _subpartition().policy_selection_run_ids
    base = {
        runs[0]: (100.0,),
        runs[1]: (1.0,),
        runs[2]: (2.0,),
        runs[3]: (3.0,),
        runs[4]: (4.0,),
        runs[5]: (5.0,),
    }
    duplicated = {
        **base,
        runs[0]: (100.0,) * 100,
    }

    expected = _eligible_target_score_threshold(
        base,
        quantile=0.5,
        expected_run_ids=runs,
    )
    observed = _eligible_target_score_threshold(
        duplicated,
        quantile=0.5,
        expected_run_ids=runs,
    )

    assert expected == 3.5
    assert observed == 100.0


def test_candidate_derivation_is_target_free_and_prefers_shorter_tie() -> None:
    positive_batch = _batch(payoff_sign=1.0)
    negative_batch = _batch(payoff_sign=-1.0)
    context = build_round74_action_inference_context(positive_batch)

    positive = _candidates(positive_batch)
    negative = _candidates(negative_batch)

    assert np.shares_memory(context.feature_values, positive_batch.feature_values)
    assert context.window_representation == "per_symbol"
    assert np.shares_memory(
        context.decision_wall_ns,
        positive_batch.decision_wall_ns,
    )
    assert positive.candidate_sha256 == negative.candidate_sha256
    assert positive.context_sha256 == negative.context_sha256
    assert positive.eligible.all()
    assert set(positive.horizon_seconds) == {30}
    assert set(positive.side) == {1}
    assert positive.trading_authority is False
    assert positive.leverage_applied is False

    global_batch = replace(
        positive_batch,
        window_representation="global_cross_asset",
    )
    global_batch.validate()
    global_context = build_round74_action_inference_context(global_batch)
    assert global_context.window_representation == "global_cross_asset"
    assert global_context.context_sha256 != context.context_sha256


def test_risk_tail_calibration_can_fail_closed_before_candidate_selection() -> None:
    batch = _batch(payoff_sign=1.0)
    calibration = replace(
        _calibration(),
        risk_quantiles=_risk_quantile_calibration(),
        schema_version=ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION,
    )

    candidates = derive_round74_action_candidates(
        _output(batch.rows),
        build_round74_action_inference_context(batch),
        calibration,
        pretest_policy_sha256=POLICY_SHA256,
    )

    assert not bool(candidates.eligible.any())
    assert not bool(candidates.quality_score.any())


def test_candidate_derivation_rejects_calibration_from_other_policy() -> None:
    batch = _batch()
    calibration = replace(
        _calibration(),
        pretest_policy_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="calibration policy differs"):
        derive_round74_action_candidates(
            _output(batch.rows),
            build_round74_action_inference_context(batch),
            calibration,
            pretest_policy_sha256=POLICY_SHA256,
        )


def test_exact_trace_rejects_same_symbol_overlap_but_resets_each_run() -> None:
    batch = _batch()
    candidates = _candidates(batch)

    trace = simulate_round74_action_trace(
        batch,
        candidates,
        threshold_score=0.0,
        expected_run_ids=_subpartition().policy_selection_run_ids,
    )

    assert trace.metrics.trades == 18
    assert trace.skipped_same_symbol_overlap == 18
    assert trace.position_capital_fraction == pytest.approx(1.0 / 3.0)
    assert trace.as_dict()["maximum_concurrent_gross_capital_fraction"] == (
        pytest.approx(1.0)
    )
    assert trace.metrics.total_net_bps == pytest.approx(sum(trace.net_payoff_bps))
    assert max(trace.net_payoff_bps) == pytest.approx(
        ROUND74_ACTION_POSITION_CAPITAL_FRACTION * float(np.max(batch.net_payoff_bps))
    )
    with pytest.raises(ValueError, match="metrics reconciliation differs"):
        replace(
            trace,
            metrics=replace(
                trace.metrics,
                total_net_bps=trace.metrics.total_net_bps + 1.0,
            ),
        ).validate()
    tampered = trace.as_dict()
    tampered["position_capital_fraction"] = 1.0
    with pytest.raises(ValueError, match="action trace differs"):
        type(trace).from_dict(tampered)
    assert trace.metrics.active_runs == 6
    assert trace.metrics.distinct_symbols == 3
    assert trace.metrics.maximum_symbol_trade_share == pytest.approx(1.0 / 3.0)
    assert all(
        exit_value >= entry
        for entry, exit_value in zip(
            trace.entry_monotonic_ns,
            trace.exit_monotonic_ns,
            strict=True,
        )
    )


def test_policy_selection_accepts_only_profitable_diversified_tuning_trace() -> None:
    batch = _batch(payoff_sign=1.0)
    selection = select_round74_action_policy(
        batch,
        _candidates(batch),
        _subpartition(),
    )

    assert selection.accepted
    assert selection.selected_quantile == 0.50
    assert selection.selected_threshold_score is not None
    assert selection.sealed_test_accessed is False
    assert selection.profitability_claim is False
    assert len(selection.selection_sha256) == 64
    accepted = [value for value in selection.evaluations if value.accepted]
    assert accepted
    assert all(value.trace.metrics.total_net_bps > 0.0 for value in accepted)
    for evaluation in selection.evaluations:
        metrics = evaluation.trace.metrics
        expected_objective = (
            metrics.mean_run_net_bps
            - round74_action_profile().objective_drawdown_penalty
            * metrics.maximum_drawdown_bps
            - round74_action_profile().objective_adverse_excursion_penalty
            * metrics.mean_run_maximum_adverse_excursion_bps
        )
        assert evaluation.objective_bps == pytest.approx(expected_objective)
        assert metrics.mean_run_net_bps == pytest.approx(
            metrics.total_net_bps / len(_subpartition().policy_selection_run_ids)
        )


def test_batch_panel_selection_matches_single_batch_without_feature_copy() -> None:
    batch = _batch(payoff_sign=1.0)
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value) for value in batches)

    panel_trace = simulate_round74_action_trace_batches(
        batches,
        candidates,
        threshold_score=0.0,
        expected_run_ids=_subpartition().policy_selection_run_ids,
    )
    single_trace = simulate_round74_action_trace(
        batch,
        _candidates(batch),
        threshold_score=0.0,
        expected_run_ids=_subpartition().policy_selection_run_ids,
    )
    panel_selection = select_round74_action_policy_batches(
        batches,
        candidates,
        _subpartition(),
    )
    single_selection = select_round74_action_policy(
        batch,
        _candidates(batch),
        _subpartition(),
    )

    assert panel_trace.as_dict() == single_trace.as_dict()
    assert panel_selection.selected_threshold_score == (
        single_selection.selected_threshold_score
    )
    assert panel_selection.evaluations == single_selection.evaluations
    assert len(panel_selection.target_batch_sha256) == 6
    assert len(panel_selection.candidate_sha256) == 6
    assert all(
        np.shares_memory(value.feature_values, batch.feature_values)
        for value in batches
    )


def test_segmented_policy_selection_uses_eligible_target_objective() -> None:
    batch = _batch(payoff_sign=1.0)
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value) for value in batches)

    selection = select_round74_action_policy_batches(
        batches,
        candidates,
        _subpartition(),
        optimization_population="eligible_target",
    )

    assert selection.optimization_population == "eligible_target"
    assert type(selection).from_dict(selection.as_dict()) == selection
    for evaluation in selection.evaluations:
        metrics = evaluation.trace.metrics
        expected_objective = (
            metrics.total_net_bps
            - round74_action_profile().objective_drawdown_penalty
            * metrics.maximum_drawdown_bps
            - round74_action_profile().objective_adverse_excursion_penalty
            * metrics.mean_run_maximum_adverse_excursion_bps
            * len(_subpartition().policy_selection_run_ids)
        )
        assert evaluation.objective_bps == pytest.approx(expected_objective)
        assert evaluation.objective_semantics == (
            "total_net_bps_minus_worst_drawdown_and_total_mae_penalties"
        )


def test_segmented_policy_selection_uses_every_frozen_policy_segment() -> None:
    subpartition = _segmented_subpartition()
    calibration = _calibration(subpartition)
    batch = _batch(payoff_sign=1.0, subpartition=subpartition)
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value, calibration=calibration) for value in batches)

    trace = simulate_round74_action_trace_batches(
        batches,
        candidates,
        threshold_score=0.0,
        expected_run_ids=subpartition.policy_selection_run_ids,
    )
    selection = select_round74_action_policy_batches(
        batches,
        candidates,
        subpartition,
        optimization_population="eligible_target",
    )

    assert len(batches) == 19
    assert trace.expected_run_ids == subpartition.policy_selection_run_ids
    assert trace.metrics.active_runs == 19
    assert selection.accepted
    assert all(
        evaluation.trace.expected_run_ids == subpartition.policy_selection_run_ids
        for evaluation in selection.evaluations
    )


def test_batch_panel_rejects_non_chronological_or_duplicate_panels() -> None:
    batch = _batch()
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value) for value in batches)

    with pytest.raises(ValueError, match="replay identity differs"):
        simulate_round74_action_trace_batches(
            (batches[1], batches[0], *batches[2:]),
            (candidates[1], candidates[0], *candidates[2:]),
            threshold_score=0.0,
            expected_run_ids=_subpartition().policy_selection_run_ids,
        )
    with pytest.raises(ValueError, match="replay identity differs"):
        simulate_round74_action_trace_batches(
            (*batches, batches[-1]),
            (*candidates, candidates[-1]),
            threshold_score=0.0,
            expected_run_ids=_subpartition().policy_selection_run_ids,
        )


def test_policy_selection_rejects_negative_after_cost_outcomes() -> None:
    batch = _batch(payoff_sign=-1.0)
    selection = select_round74_action_policy(
        batch,
        _candidates(batch),
        _subpartition(),
    )

    assert not selection.accepted
    assert selection.selected_quantile is None
    assert selection.rejection_reasons == ("no_policy_threshold_passed_risk_gates",)
    assert all(
        "positive_after_cost_payoff_not_met" in value.rejection_reasons
        for value in selection.evaluations
    )


def test_policy_selection_uses_profile_delayed_l2_economics_not_baseline() -> None:
    batch = _batch(payoff_sign=1.0)
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value) for value in batches)
    delayed = _execution_panel(batches, candidates)

    baseline = select_round74_action_policy_batches(
        batches,
        candidates,
        _subpartition(),
    )
    selected = select_round74_action_policy_batches(
        batches,
        candidates,
        _subpartition(),
        execution_panel=delayed,
    )

    assert baseline.accepted
    assert not selected.accepted
    assert selected.execution_outcome_panel_sha256 == delayed.panel_sha256
    assert all(
        evaluation.trace.metrics.total_net_bps < 0.0
        for evaluation in selected.evaluations
    )
    assert all(
        value >= 1_900_000_000
        for evaluation in selected.evaluations
        for value in evaluation.trace.entry_monotonic_ns
    )


def test_profile_delayed_l2_panel_rejects_missing_feature_row() -> None:
    batch = _batch()
    batches = _run_batch_panel(batch)
    candidates = tuple(_candidates(value) for value in batches)
    delayed = _execution_panel(batches, candidates)

    with pytest.raises(ValueError, match="panel binding differs"):
        select_round74_action_policy_batches(
            batches,
            candidates,
            _subpartition(),
            execution_panel=replace(delayed, rows=delayed.rows[:-1]),
        )


def test_policy_selection_rejects_future_censored_selected_action() -> None:
    batch = _batch(payoff_sign=1.0)
    horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(30)
    row_index = len(ROUND74_EVENT_SYMBOLS) * 2 - 1
    eligibility = np.array(batch.action_eligibility, copy=True)
    entry = np.array(batch.actual_entry_monotonic_ns, copy=True)
    exit_value = np.array(batch.actual_exit_monotonic_ns, copy=True)
    payoff = np.array(batch.net_payoff_bps, copy=True)
    mae = np.array(batch.maximum_adverse_excursion_bps, copy=True)
    adverse = np.array(batch.adverse_selection, copy=True)
    eligibility[row_index, horizon_index, 0] = 0.0
    entry[row_index, horizon_index, 0] = -1
    exit_value[row_index, horizon_index, 0] = -1
    payoff[row_index, horizon_index, 0] = 0.0
    mae[row_index, horizon_index, 0] = 0.0
    adverse[row_index, horizon_index, 0] = 0.0
    censored = replace(
        batch,
        action_eligibility=_readonly(eligibility),
        actual_entry_monotonic_ns=_readonly(entry),
        actual_exit_monotonic_ns=_readonly(exit_value),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(mae),
        adverse_selection=_readonly(adverse),
    )
    censored.validate()

    selection = select_round74_action_policy(
        censored,
        _candidates(censored),
        _subpartition(),
    )

    assert not selection.accepted
    assert all(
        evaluation.trace.skipped_target_ineligible == 1
        and "selected_action_target_coverage_incomplete" in evaluation.rejection_reasons
        for evaluation in selection.evaluations
    )


def test_policy_selection_rejects_non_policy_tuning_role() -> None:
    batch = _batch()
    candidates = _candidates(batch)
    wrong_runs = replace(
        _subpartition(),
        policy_selection_run_ids=tuple(f"{index:032x}" for index in range(25, 31)),
    )

    with pytest.raises(ValueError, match="data role differs"):
        select_round74_action_policy(batch, candidates, wrong_runs)

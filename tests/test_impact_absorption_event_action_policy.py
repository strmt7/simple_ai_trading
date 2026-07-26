from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_DEFAULT_PROFILE,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    round74_action_profile,
    select_round74_action_policy,
    simulate_round74_action_trace,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
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


def _fit() -> Round74TemperatureFit:
    return Round74TemperatureFit(
        temperature=1.0,
        eligible_observations=10,
        positive_observations=5,
        uncalibrated_nll=0.5,
        calibrated_nll=0.5,
        uncalibrated_brier=0.2,
        calibrated_brier=0.2,
        uncalibrated_ece=0.1,
        calibrated_ece=0.1,
    )


def _calibration() -> Round74ProbabilityCalibration:
    fit = _fit()
    return Round74ProbabilityCalibration(
        pretest_policy_sha256=POLICY_SHA256,
        tuning_subpartition_sha256=_subpartition().subpartition_sha256,
        calibration_source_sha256="4" * 64,
        calibration_data_sha256="5" * 64,
        positive_payoff=fit,
        adverse_selection=fit,
        regime_unpredictability=fit,
        backend_kind="cpu",
        backend_device="test",
    )


def _batch(
    *,
    payoff_sign: float = 1.0,
    role: str = "tuning",
) -> Round74EventTrainingBatch:
    runs = _subpartition().policy_selection_run_ids
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
        target_context_sha256=tuple("6" * 64 for _ in range(rows)),
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
):
    return derive_round74_action_candidates(
        _output(batch.rows),
        build_round74_action_inference_context(batch),
        _calibration(),
        pretest_policy_sha256=POLICY_SHA256,
        profile=profile,
    )


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


def test_candidate_derivation_is_target_free_and_prefers_shorter_tie() -> None:
    positive_batch = _batch(payoff_sign=1.0)
    negative_batch = _batch(payoff_sign=-1.0)

    positive = _candidates(positive_batch)
    negative = _candidates(negative_batch)

    assert positive.candidate_sha256 == negative.candidate_sha256
    assert positive.context_sha256 == negative.context_sha256
    assert positive.eligible.all()
    assert set(positive.horizon_seconds) == {30}
    assert set(positive.side) == {1}
    assert positive.trading_authority is False
    assert positive.leverage_applied is False


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


def test_policy_selection_rejects_non_policy_tuning_role() -> None:
    batch = _batch()
    candidates = _candidates(batch)
    wrong_runs = replace(
        _subpartition(),
        policy_selection_run_ids=tuple(f"{index:032x}" for index in range(25, 31)),
    )

    with pytest.raises(ValueError, match="data role differs"):
        select_round74_action_policy(batch, candidates, wrong_runs)

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import simple_ai_trading.microstructure_action_policy as policy
from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_action_policy import (
    ActionPolicySpec,
    ActionScoreBatch,
    barrier_trace_gate_reasons,
    derive_action_scores,
    select_barrier_threshold,
    simulate_barrier_action_trace,
)
from simple_ai_trading.microstructure_barriers import (
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
)
from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)


def _dataset(rows: int = 20) -> MicrostructureDataset:
    times = 1_000_000 + np.arange(rows, dtype=np.int64) * 1_000_000
    ones = np.ones(rows, dtype=np.float64)
    return MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=900,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=5,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=times,
        long_exit_time_ms=times + 900_750,
        short_exit_time_ms=times + 900_750,
        features=np.zeros((rows, len(MICROSTRUCTURE_FEATURE_NAMES)), dtype=np.float32),
        long_net_bps=4.0 * ones,
        short_net_bps=-4.0 * ones,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=100.1 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=100.1 * ones,
        entry_bid_qty=10.0 * ones,
        entry_ask_qty=10.0 * ones,
        fixed_exit_bid_qty=10.0 * ones,
        fixed_exit_ask_qty=10.0 * ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )


def _targets(dataset: MicrostructureDataset) -> AdaptiveBarrierTargets:
    rows = dataset.rows
    exits = dataset.decision_time_ms + dataset.total_latency_ms + 900_000
    outcomes = np.zeros(rows, dtype=np.int8)
    return AdaptiveBarrierTargets(
        schema_version=ADAPTIVE_BARRIER_SCHEMA_VERSION,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        spec=AdaptiveBarrierSpec(
            horizon_seconds=900,
            volatility_feature_name="realized_volatility_300s_bps",
            stop_volatility_multiple=1.0,
            take_volatility_multiple=1.5,
            minimum_stop_bps=18.0,
            maximum_stop_bps=60.0,
            minimum_take_bps=27.0,
            maximum_take_bps=90.0,
            base_protection_delay_ms=250,
            stress_protection_delay_ms=750,
            trigger_execution_slippage_bps=1.0,
        ),
        source_indexes=np.arange(rows, dtype=np.int64),
        valid=np.ones(rows, dtype=bool),
        stop_barrier_bps=np.full(rows, 18.0),
        take_barrier_bps=np.full(rows, 27.0),
        base_long_net_bps=np.full(rows, 4.0),
        base_short_net_bps=np.full(rows, -4.0),
        base_long_exit_time_ms=exits.copy(),
        base_short_exit_time_ms=exits.copy(),
        base_long_outcome=outcomes.copy(),
        base_short_outcome=outcomes.copy(),
        stress_long_net_bps=np.full(rows, 2.0),
        stress_short_net_bps=np.full(rows, -6.0),
        stress_long_exit_time_ms=exits.copy(),
        stress_short_exit_time_ms=exits.copy(),
        stress_long_outcome=outcomes.copy(),
        stress_short_outcome=outcomes.copy(),
    )


def _ensemble(rows: int = 3) -> ActionValueEnsembleBatch:
    endpoints = np.arange(rows, dtype=np.int64)
    long_mean = np.asarray([10.0, 2.0, 8.0])[:rows]
    short_mean = np.asarray([5.0, 10.0, 7.0])[:rows]
    ones = np.ones(rows, dtype=np.float64)
    probability = np.asarray([0.8, 0.8, 0.4])[:rows]
    return ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=long_mean,
        short_mean_bps=short_mean,
        long_epistemic_std_bps=ones,
        short_epistemic_std_bps=ones,
        long_profitable_probability=probability,
        short_profitable_probability=probability,
        long_lower_bps=np.full(rows, -5.0),
        short_lower_bps=np.full(rows, -5.0),
        long_upper_bps=long_mean + 5.0,
        short_upper_bps=short_mean + 5.0,
        long_positive_member_ratio=np.ones(rows),
        short_positive_member_ratio=np.ones(rows),
        member_count=3,
    )


def _spec(**overrides: object) -> ActionPolicySpec:
    values: dict[str, object] = {
        "profile": "conservative",
        "epistemic_penalty": 1.0,
        "minimum_profitable_probability": 0.6,
        "minimum_member_agreement": 2.0 / 3.0,
        "maximum_epistemic_std_bps": 5.0,
        "minimum_lower_bound_bps": -20.0,
    }
    values.update(overrides)
    return ActionPolicySpec(**values)  # type: ignore[arg-type]


def _score(dataset: MicrostructureDataset, *, active: bool = True) -> ActionScoreBatch:
    strength = (
        np.linspace(1.0, 20.0, dataset.rows) if active else np.zeros(dataset.rows)
    )
    side = (
        np.ones(dataset.rows, dtype=np.int8)
        if active
        else np.zeros(dataset.rows, dtype=np.int8)
    )
    return ActionScoreBatch(
        endpoint_indexes=np.arange(dataset.rows, dtype=np.int64),
        side=side,
        strength_bps=strength,
        eligible=side != 0,
        profile="conservative",
    )


def _gates(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "minimum_trades": 5,
        "minimum_total_net_bps": 0.0,
        "maximum_drawdown_bps": 20.0,
        "minimum_positive_day_ratio": 1.0,
        "minimum_worst_trade_bps": -20.0,
        "minimum_profit_factor": 1.0,
    }
    values.update(overrides)
    return values


def test_derive_action_scores_uses_all_selective_gates_and_best_side() -> None:
    score = derive_action_scores(_ensemble(), _spec())

    np.testing.assert_array_equal(score.side, [1, -1, 0])
    np.testing.assert_allclose(score.strength_bps, [9.0, 9.0, 0.0])
    np.testing.assert_array_equal(score.eligible, [True, True, False])
    assert score.rows == 3
    assert score.trading_authority is False


def test_simulation_uses_scenario_specific_targets_and_nonoverlap() -> None:
    dataset = _dataset()
    targets = _targets(dataset)
    score = _score(dataset)

    base = simulate_barrier_action_trace(
        dataset, targets, score, scenario="base", strength_threshold_bps=0.0
    )
    stress = simulate_barrier_action_trace(
        dataset, targets, score, scenario="stress", strength_threshold_bps=0.0
    )

    assert base.metrics.trades == dataset.rows
    assert base.metrics.total_net_bps == 4.0 * dataset.rows
    assert stress.metrics.total_net_bps == 2.0 * dataset.rows
    assert base.source_endpoint_indexes == tuple(range(dataset.rows))
    assert all(
        exit_time > entry
        for exit_time, entry in zip(base.exit_times_ms, base.timestamps_ms, strict=True)
    )
    assert base.trading_authority is False


def test_threshold_selection_uses_stress_gates_and_can_abstain() -> None:
    dataset = _dataset()
    targets = _targets(dataset)
    days = (int(dataset.decision_time_ms[0]) // policy._DAY_MS,)

    selected = select_barrier_threshold(
        dataset,
        targets,
        _score(dataset),
        quantiles=(0.1, 0.5),
        expected_days=days,
        gates=_gates(),
        drawdown_penalty=0.5,
    )
    assert selected.accepted is True
    assert selected.quantile == 0.1
    assert selected.stress_trace.metrics.total_net_bps > 0.0
    assert selected.asdict()["trading_authority"] is False

    targets.stress_long_net_bps[:] = -2.0
    rejected = select_barrier_threshold(
        dataset,
        targets,
        _score(dataset),
        quantiles=(0.1, 0.5),
        expected_days=days,
        gates=_gates(),
        drawdown_penalty=0.5,
    )
    assert rejected.accepted is False
    assert rejected.stress_trace.metrics.trades == 0
    assert rejected.rejection_reasons == (
        "no_calibration_threshold_passed_stress_gates",
    )


def test_threshold_selection_abstains_without_eligible_scores() -> None:
    dataset = _dataset()
    selected = select_barrier_threshold(
        dataset,
        _targets(dataset),
        _score(dataset, active=False),
        quantiles=(0.5,),
        expected_days=(int(dataset.decision_time_ms[0]) // policy._DAY_MS,),
        gates=_gates(),
        drawdown_penalty=0.5,
    )

    assert selected.accepted is False
    assert selected.candidates == ()
    assert selected.rejection_reasons == ("calibration_has_no_eligible_scores",)


def test_trace_gates_report_each_financial_failure() -> None:
    dataset = _dataset()
    targets = _targets(dataset)
    targets.stress_long_net_bps[:] = -2.0
    trace = simulate_barrier_action_trace(
        dataset,
        targets,
        _score(dataset),
        scenario="stress",
        strength_threshold_bps=15.0,
    )
    reasons = barrier_trace_gate_reasons(
        trace,
        expected_days=(int(dataset.decision_time_ms[0]) // policy._DAY_MS,),
        gates=_gates(
            minimum_trades=10,
            maximum_drawdown_bps=1.0,
            minimum_worst_trade_bps=-1.0,
            minimum_profit_factor=2.0,
        ),
    )

    assert set(reasons) == {
        "minimum_trades_not_met",
        "total_net_gate_failed",
        "drawdown_gate_failed",
        "positive_day_ratio_gate_failed",
        "worst_trade_gate_failed",
        "profit_factor_gate_failed",
    }


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"profile": "reckless"}, "unsupported"),
        ({"epistemic_penalty": float("nan")}, "must be finite"),
        ({"minimum_profitable_probability": 0.4}, "outside bounds"),
    ],
)
def test_policy_spec_rejects_invalid_contracts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_ensemble_and_score_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="ensemble contract"):
        derive_action_scores(replace(_ensemble(), member_count=1), _spec())
    dataset = _dataset()
    with pytest.raises(ValueError, match="score contract"):
        simulate_barrier_action_trace(
            dataset,
            _targets(dataset),
            replace(_score(dataset), strength_bps=np.full(dataset.rows, np.nan)),
            scenario="base",
            strength_threshold_bps=0.0,
        )
    with pytest.raises(ValueError, match="scenario is unsupported"):
        simulate_barrier_action_trace(
            dataset,
            _targets(dataset),
            _score(dataset),
            scenario="future",
            strength_threshold_bps=0.0,
        )


def test_target_day_gate_and_threshold_contracts_fail_closed() -> None:
    dataset = _dataset()
    targets = _targets(dataset)
    score = _score(dataset)
    truncated_fields = {
        name: getattr(targets, name)[:-1]
        for name in (
            "source_indexes",
            "valid",
            "stop_barrier_bps",
            "take_barrier_bps",
            "base_long_net_bps",
            "base_short_net_bps",
            "base_long_exit_time_ms",
            "base_short_exit_time_ms",
            "base_long_outcome",
            "base_short_outcome",
            "stress_long_net_bps",
            "stress_short_net_bps",
            "stress_long_exit_time_ms",
            "stress_short_exit_time_ms",
            "stress_long_outcome",
            "stress_short_outcome",
        )
    }
    with pytest.raises(ValueError, match="absent from barrier"):
        simulate_barrier_action_trace(
            dataset,
            replace(targets, **truncated_fields),
            score,
            scenario="base",
            strength_threshold_bps=0.0,
        )
    invalid_row_targets = _targets(dataset)
    invalid_row_targets.valid[0] = False
    for values in (
        invalid_row_targets.base_long_net_bps,
        invalid_row_targets.base_short_net_bps,
        invalid_row_targets.stress_long_net_bps,
        invalid_row_targets.stress_short_net_bps,
    ):
        values[0] = np.nan
    for values in (
        invalid_row_targets.base_long_exit_time_ms,
        invalid_row_targets.base_short_exit_time_ms,
        invalid_row_targets.stress_long_exit_time_ms,
        invalid_row_targets.stress_short_exit_time_ms,
        invalid_row_targets.base_long_outcome,
        invalid_row_targets.base_short_outcome,
        invalid_row_targets.stress_long_outcome,
        invalid_row_targets.stress_short_outcome,
    ):
        values[0] = -1
    with pytest.raises(ValueError, match="not a valid barrier"):
        simulate_barrier_action_trace(
            dataset,
            invalid_row_targets,
            score,
            scenario="base",
            strength_threshold_bps=0.0,
        )
    trace = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="base",
        strength_threshold_bps=0.0,
    )
    with pytest.raises(ValueError, match="risk controls are incomplete"):
        barrier_trace_gate_reasons(
            trace,
            expected_days=(0,),
            gates={},
        )
    with pytest.raises(ValueError, match="risk controls are invalid"):
        barrier_trace_gate_reasons(
            trace,
            expected_days=(0,),
            gates=_gates(minimum_trades=0),
        )
    with pytest.raises(ValueError, match="expected days are invalid"):
        barrier_trace_gate_reasons(trace, expected_days=(), gates=_gates())
    with pytest.raises(ValueError, match="outside expected days"):
        barrier_trace_gate_reasons(trace, expected_days=(1,), gates=_gates())
    with pytest.raises(ValueError, match="threshold policy is invalid"):
        select_barrier_threshold(
            dataset,
            targets,
            score,
            quantiles=(1.0,),
            expected_days=(0,),
            gates=_gates(),
            drawdown_penalty=0.5,
        )

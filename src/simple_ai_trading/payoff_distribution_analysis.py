"""Proper-score and paired-ledger analysis for payoff distributions."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .categorical_payoff_lightgbm import (
    CategoricalPayoffDataset,
    CategoricalPayoffPredictionBatch,
    TrainedCategoricalPayoffModel,
    multiclass_log_loss,
    ranked_probability_score,
)
from .direct_payoff_lightgbm import (
    DirectPayoffPredictionBatch,
    TrainedDirectPayoffModel,
)
from .microstructure_action_policy import (
    ActionScoreBatch,
    BarrierActionTrace,
    simulate_barrier_action_trace,
)
from .microstructure_barriers import AdaptiveBarrierTargets
from .microstructure_features import MicrostructureDataset
from .microstructure_model import _trading_metrics


_DAY_MS = 86_400_000
_SIDES = ("long", "short")


def finite_spearman(actual: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or len(left) < 3
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.ptp(left) <= 0.0
        or np.ptp(right) <= 0.0
    ):
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _calibration_error(
    probability: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    predicted = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(outcome, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or predicted.ndim != 1
        or len(predicted) < bins
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
        or not np.all(np.isin(observed, (0.0, 1.0)))
    ):
        raise ValueError("profitable-probability calibration inputs are invalid")
    groups = np.array_split(np.argsort(predicted, kind="stable"), bins)
    rows: list[dict[str, float | int]] = []
    for index, selected in enumerate(groups, start=1):
        mean_prediction = float(np.mean(predicted[selected]))
        event_rate = float(np.mean(observed[selected]))
        rows.append(
            {
                "bin": index,
                "rows": len(selected),
                "mean_prediction": mean_prediction,
                "event_rate": event_rate,
                "absolute_error": abs(mean_prediction - event_rate),
            }
        )
    return max(float(row["absolute_error"]) for row in rows), rows


def _categorical_side_metrics(
    *,
    side: str,
    model: TrainedCategoricalPayoffModel,
    dataset: CategoricalPayoffDataset,
    indexes: np.ndarray,
    prediction: CategoricalPayoffPredictionBatch,
) -> dict[str, object]:
    selected = np.asarray(indexes, dtype=np.int64)
    actual = np.asarray(
        dataset.long_net_bps if side == "long" else dataset.short_net_bps,
        dtype=np.float64,
    )[selected]
    stop = np.asarray(dataset.stop_width_bps, dtype=np.float64)[selected]
    risk = actual / stop
    probabilities = np.asarray(
        prediction.long_probabilities
        if side == "long"
        else prediction.short_probabilities,
        dtype=np.float64,
    )
    action = prediction.action_values
    expected = np.asarray(
        action.long_mean_bps if side == "long" else action.short_mean_bps,
        dtype=np.float64,
    )
    profitable_probability = np.asarray(
        action.long_profitable_probability
        if side == "long"
        else action.short_profitable_probability,
        dtype=np.float64,
    )
    lower = np.asarray(
        action.long_lower_bps if side == "long" else action.short_lower_bps,
        dtype=np.float64,
    )
    upper = np.asarray(
        action.long_upper_bps if side == "long" else action.short_upper_bps,
        dtype=np.float64,
    )
    edges = np.asarray(model.bin_edges_risk_units[side], dtype=np.float64)
    labels = np.searchsorted(edges, risk, side="right").astype(np.int64)
    train_support = np.asarray(model.class_support[side]["train"], dtype=np.float64)
    baseline_probability = train_support / np.sum(train_support)
    baseline_matrix = np.broadcast_to(baseline_probability, probabilities.shape)
    log_loss = multiclass_log_loss(probabilities, labels)
    baseline_log_loss = multiclass_log_loss(baseline_matrix, labels)
    rps = ranked_probability_score(probabilities, labels)
    baseline_rps = ranked_probability_score(baseline_matrix, labels)
    representatives = np.asarray(
        model.bin_representatives_risk_units[side], dtype=np.float64
    )
    train_risk_mean = float(baseline_probability @ representatives)
    baseline_expected = train_risk_mean * stop
    mse = float(np.mean((expected - actual) ** 2))
    baseline_mse = float(np.mean((baseline_expected - actual) ** 2))
    outcome = (actual > 0.0).astype(np.float64)
    train_profitable_probability = float(
        baseline_probability
        @ np.asarray(model.bin_positive_rates[side], dtype=np.float64)
    )
    brier = float(np.mean((profitable_probability - outcome) ** 2))
    baseline_brier = float(np.mean((train_profitable_probability - outcome) ** 2))
    maximum_calibration_error, calibration_bins = _calibration_error(
        profitable_probability,
        outcome,
    )
    day_ids = dataset.decision_time_ms[selected] // _DAY_MS
    daily: list[dict[str, float | int]] = []
    for day_id in np.unique(day_ids):
        mask = day_ids == day_id
        day_brier = float(np.mean((profitable_probability[mask] - outcome[mask]) ** 2))
        day_baseline = float(
            np.mean((train_profitable_probability - outcome[mask]) ** 2)
        )
        daily.append(
            {
                "utc_day_id": int(day_id),
                "rows": int(np.sum(mask)),
                "brier_score": day_brier,
                "baseline_brier_score": day_baseline,
                "brier_skill": 1.0 - day_brier / max(day_baseline, 1e-15),
            }
        )
    return {
        "rows": len(selected),
        "multinomial_log_loss": log_loss,
        "baseline_multinomial_log_loss": baseline_log_loss,
        "multinomial_log_loss_skill": 1.0 - log_loss / max(baseline_log_loss, 1e-15),
        "ranked_probability_score": rps,
        "baseline_ranked_probability_score": baseline_rps,
        "ranked_probability_skill": 1.0 - rps / max(baseline_rps, 1e-15),
        "expected_payoff_mse_bps2": mse,
        "baseline_expected_payoff_mse_bps2": baseline_mse,
        "expected_payoff_mse_skill": 1.0 - mse / max(baseline_mse, 1e-15),
        "expected_payoff_spearman": finite_spearman(actual, expected),
        "profitable_probability_brier": brier,
        "baseline_profitable_probability_brier": baseline_brier,
        "profitable_probability_brier_skill": 1.0 - brier / max(baseline_brier, 1e-15),
        "maximum_10_bin_calibration_error": maximum_calibration_error,
        "calibration_bins": calibration_bins,
        "days_with_positive_brier_skill": sum(
            float(row["brier_skill"]) > 0.0 for row in daily
        ),
        "daily_brier_skill": daily,
        "lower_10pct_empirical_coverage": float(np.mean(actual <= lower)),
        "upper_90pct_empirical_coverage": float(np.mean(actual <= upper)),
        "actual_mean_net_bps": float(np.mean(actual)),
        "predicted_mean_net_bps": float(np.mean(expected)),
    }


def categorical_forecast_metrics(
    model: TrainedCategoricalPayoffModel,
    dataset: CategoricalPayoffDataset,
    indexes: np.ndarray,
    prediction: CategoricalPayoffPredictionBatch,
) -> dict[str, object]:
    selected = np.asarray(indexes, dtype=np.int64)
    if prediction.action_values.rows != len(selected):
        raise ValueError("categorical forecast rows differ from evaluation indexes")
    return {
        side: _categorical_side_metrics(
            side=side,
            model=model,
            dataset=dataset,
            indexes=selected,
            prediction=prediction,
        )
        for side in _SIDES
    }


def direct_forecast_metrics(
    model: TrainedDirectPayoffModel,
    dataset: CategoricalPayoffDataset,
    indexes: np.ndarray,
    prediction: DirectPayoffPredictionBatch,
) -> dict[str, object]:
    selected = np.asarray(indexes, dtype=np.int64)
    if prediction.rows != len(selected):
        raise ValueError("direct forecast rows differ from evaluation indexes")
    output: dict[str, object] = {}
    for side in _SIDES:
        actual = np.asarray(
            dataset.long_net_bps if side == "long" else dataset.short_net_bps,
            dtype=np.float64,
        )[selected]
        predicted = np.asarray(
            prediction.long_mean_bps if side == "long" else prediction.short_mean_bps,
            dtype=np.float64,
        )
        baseline = float(model.training_target_mean_bps[side])
        mse = float(np.mean((predicted - actual) ** 2))
        baseline_mse = float(np.mean((baseline - actual) ** 2))
        output[side] = {
            "rows": len(selected),
            "expected_payoff_mse_bps2": mse,
            "baseline_expected_payoff_mse_bps2": baseline_mse,
            "expected_payoff_mse_skill": 1.0 - mse / max(baseline_mse, 1e-15),
            "expected_payoff_spearman": finite_spearman(actual, predicted),
            "actual_mean_net_bps": float(np.mean(actual)),
            "predicted_mean_net_bps": float(np.mean(predicted)),
        }
    return output


def pairwise_seed_spearman(
    predictions: Sequence[np.ndarray],
) -> dict[str, object]:
    if len(predictions) < 2:
        raise ValueError("pairwise seed agreement requires at least two predictions")
    arrays = [np.asarray(value, dtype=np.float64) for value in predictions]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("pairwise seed prediction shapes differ")
    rows: list[dict[str, float | int]] = []
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            rows.append(
                {
                    "left_seed_index": left,
                    "right_seed_index": right,
                    "spearman": finite_spearman(arrays[left], arrays[right]),
                }
            )
    return {
        "minimum_spearman": min(float(row["spearman"]) for row in rows),
        "pairs": rows,
    }


def ensemble_action_score(
    endpoint_indexes: np.ndarray,
    *,
    long_means: Sequence[np.ndarray],
    short_means: Sequence[np.ndarray],
    long_probabilities: Sequence[np.ndarray] | None,
    short_probabilities: Sequence[np.ndarray] | None,
) -> ActionScoreBatch:
    endpoints = np.asarray(endpoint_indexes, dtype=np.int64)
    long_stack = np.stack(long_means).astype(np.float64)
    short_stack = np.stack(short_means).astype(np.float64)
    if (
        endpoints.ndim != 1
        or len(endpoints) == 0
        or np.any(np.diff(endpoints) <= 0)
        or long_stack.shape != (len(long_means), len(endpoints))
        or short_stack.shape != (len(short_means), len(endpoints))
        or len(long_means) < 2
        or len(short_means) != len(long_means)
        or not np.all(np.isfinite(long_stack))
        or not np.all(np.isfinite(short_stack))
    ):
        raise ValueError("payoff ensemble means are invalid")
    long_eligible = np.all(long_stack > 0.0, axis=0)
    short_eligible = np.all(short_stack > 0.0, axis=0)
    if (long_probabilities is None) != (short_probabilities is None):
        raise ValueError("payoff ensemble probability sides differ")
    if long_probabilities is not None and short_probabilities is not None:
        long_probability = np.stack(long_probabilities).astype(np.float64)
        short_probability = np.stack(short_probabilities).astype(np.float64)
        if (
            long_probability.shape != long_stack.shape
            or short_probability.shape != short_stack.shape
            or np.any(long_probability < 0.0)
            or np.any(long_probability > 1.0)
            or np.any(short_probability < 0.0)
            or np.any(short_probability > 1.0)
        ):
            raise ValueError("payoff ensemble probabilities are invalid")
        long_eligible &= np.mean(long_probability, axis=0) > 0.50
        short_eligible &= np.mean(short_probability, axis=0) > 0.50
    long_worst = np.min(long_stack, axis=0)
    short_worst = np.min(short_stack, axis=0)
    choose_long = long_eligible & (~short_eligible | (long_worst > short_worst))
    choose_short = short_eligible & (~long_eligible | (short_worst > long_worst))
    side = np.zeros(len(endpoints), dtype=np.int8)
    strength = np.zeros(len(endpoints), dtype=np.float64)
    side[choose_long] = 1
    side[choose_short] = -1
    strength[choose_long] = long_worst[choose_long]
    strength[choose_short] = short_worst[choose_short]
    return ActionScoreBatch(
        endpoint_indexes=endpoints,
        side=side,
        strength_bps=strength,
        eligible=side != 0,
        profile="conservative",
    )


def base_and_paired_stress_traces(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    score: ActionScoreBatch,
    *,
    extra_stress_slippage_bps_per_side: float = 2.0,
) -> tuple[BarrierActionTrace, BarrierActionTrace, int]:
    base = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="base",
        strength_threshold_bps=0.0,
    )
    endpoints = np.asarray(base.source_endpoint_indexes, dtype=np.int64)
    if len(endpoints) == 0:
        empty = BarrierActionTrace(
            scenario="stress",
            metrics=_trading_metrics((), (), ()),
            net_bps=(),
            sides=(),
            timestamps_ms=(),
            exit_times_ms=(),
            source_endpoint_indexes=(),
        )
        return base, empty, 0
    positions = np.searchsorted(targets.source_indexes, endpoints)
    if (
        np.any(positions >= targets.rows)
        or not np.array_equal(targets.source_indexes[positions], endpoints)
        or not np.all(targets.valid[positions])
    ):
        raise ValueError("paired stress ledger endpoints are invalid")
    sides = np.asarray(base.sides, dtype=np.int8)
    stress_long = np.asarray(targets.stress_long_net_bps[positions], dtype=np.float64)
    stress_short = np.asarray(targets.stress_short_net_bps[positions], dtype=np.float64)
    stress_long_exit = np.asarray(
        targets.stress_long_exit_time_ms[positions], dtype=np.int64
    )
    stress_short_exit = np.asarray(
        targets.stress_short_exit_time_ms[positions], dtype=np.int64
    )
    stress_pnl = np.where(sides == 1, stress_long, stress_short)
    stress_pnl -= 2.0 * float(extra_stress_slippage_bps_per_side)
    stress_exits = np.where(sides == 1, stress_long_exit, stress_short_exit)
    timestamps = np.asarray(base.timestamps_ms, dtype=np.int64)
    overlap_violations = int(np.sum(timestamps[1:] < stress_exits[:-1]))
    stress = BarrierActionTrace(
        scenario="stress",
        metrics=_trading_metrics(stress_pnl, sides, timestamps),
        net_bps=tuple(float(value) for value in stress_pnl),
        sides=tuple(int(value) for value in sides),
        timestamps_ms=tuple(int(value) for value in timestamps),
        exit_times_ms=tuple(int(value) for value in stress_exits),
        source_endpoint_indexes=tuple(int(value) for value in endpoints),
    )
    return base, stress, overlap_violations


def portfolio_trace_metrics(
    traces: Mapping[str, BarrierActionTrace],
    *,
    symbol_weight: float,
) -> dict[str, object]:
    if not traces or not math.isfinite(symbol_weight) or symbol_weight <= 0.0:
        raise ValueError("portfolio trace inputs are invalid")
    rows: list[tuple[int, float, int, str]] = []
    for symbol, trace in traces.items():
        rows.extend(
            (int(exit_time), float(pnl) * symbol_weight, int(side), symbol)
            for exit_time, pnl, side in zip(
                trace.exit_times_ms,
                trace.net_bps,
                trace.sides,
                strict=True,
            )
        )
    rows.sort(key=lambda value: (value[0], value[3]))
    timestamps = tuple(value[0] for value in rows)
    pnls = tuple(value[1] for value in rows)
    sides = tuple(value[2] for value in rows)
    metrics = asdict(_trading_metrics(pnls, sides, timestamps))
    pnl_by_exit: dict[int, float] = {}
    for timestamp, pnl in zip(timestamps, pnls, strict=True):
        pnl_by_exit[timestamp] = pnl_by_exit.get(timestamp, 0.0) + pnl
    grouped_cumulative = np.cumsum(tuple(pnl_by_exit.values()), dtype=np.float64)
    grouped_peak = np.maximum.accumulate(np.concatenate(([0.0], grouped_cumulative)))[
        :-1
    ]
    metrics["max_drawdown_bps"] = float(
        np.max(grouped_peak - grouped_cumulative, initial=0.0)
    )
    daily: dict[int, float] = {}
    symbol_pnl: dict[str, float] = {symbol: 0.0 for symbol in traces}
    for timestamp, pnl, _side, symbol in rows:
        day = timestamp // _DAY_MS
        daily[day] = daily.get(day, 0.0) + pnl
        symbol_pnl[symbol] += pnl
    positive_total = sum(max(0.0, value) for value in symbol_pnl.values())
    positive_shares = {
        symbol: (max(0.0, value) / positive_total if positive_total > 0.0 else 0.0)
        for symbol, value in symbol_pnl.items()
    }
    return {
        "metrics": metrics,
        "daily_net_bps": [
            {"utc_day_id": day, "net_bps": daily[day]} for day in sorted(daily)
        ],
        "symbol_net_bps": symbol_pnl,
        "positive_pnl_share": positive_shares,
        "maximum_single_symbol_positive_pnl_share": max(
            positive_shares.values(), default=0.0
        ),
    }


__all__ = [
    "base_and_paired_stress_traces",
    "categorical_forecast_metrics",
    "direct_forecast_metrics",
    "ensemble_action_score",
    "finite_spearman",
    "pairwise_seed_spearman",
    "portfolio_trace_metrics",
]

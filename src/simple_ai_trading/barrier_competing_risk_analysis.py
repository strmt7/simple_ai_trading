"""Truthful forecast and fixed-policy analysis for Round 50."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import itertools
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .barrier_competing_risk_tcn_model import (
    EVENT_MINUTES,
    SEEDS,
    SIDES,
    TIMEOUT_CLASS,
    BarrierCompetingRiskForecastBundle,
)
from .barrier_payoff_data import (
    STOP_EVENT,
    TAKE_PROFIT_EVENT,
    TIMEOUT_EVENT,
    BarrierPayoffDataset,
)
from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .minute_logistic_mixture_tcn_model import MinuteTemporalDataset


STRESS_EXECUTION_CHARGE_BPS = 16.0
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_DAYS = 7
FAMILYWISE_LOWER_QUANTILE = 0.0125
EVENT_GROUP_NAMES = ("stop_loss", "timeout", "take_profit")


def _finite_spearman(actual: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if left.size < 2 or left.shape != right.shape:
        return 0.0
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _month(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).strftime("%Y-%m")


def _date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).date().isoformat()


def _binary_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    prediction = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, prediction.size)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (prediction >= edges[index]) & (prediction <= edges[index + 1])
        else:
            selected = (prediction >= edges[index]) & (prediction < edges[index + 1])
        if np.any(selected):
            value += (
                float(np.sum(selected))
                / total
                * abs(
                    float(np.mean(prediction[selected]))
                    - float(np.mean(target[selected]))
                )
            )
    return value


def _event_group_labels(events: np.ndarray) -> np.ndarray:
    labels = np.full(events.shape, -1, dtype=np.int64)
    labels[events == STOP_EVENT] = 0
    labels[events == TIMEOUT_EVENT] = 1
    labels[events == TAKE_PROFIT_EVENT] = 2
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("Round 50 event group labels are invalid")
    return labels


def _pairwise_seed_spearman(
    predictions: np.ndarray,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    minimum = 1.0
    for left, right in itertools.combinations(range(predictions.shape[0]), 2):
        for side_index, side in enumerate(("short", "long")):
            value = _finite_spearman(
                predictions[left, ..., side_index],
                predictions[right, ..., side_index],
            )
            rows.append(
                {
                    "seed_left": SEEDS[left],
                    "seed_right": SEEDS[right],
                    "side": side,
                    "spearman": value,
                }
            )
            minimum = min(minimum, value)
    return rows, minimum


def candidate_diagnostics(
    bundle: BarrierCompetingRiskForecastBundle,
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
    event_classes: np.ndarray,
) -> dict[str, object]:
    indices = bundle.global_indices
    evaluation_local = barrier.role_masks["viability"][indices]
    if not np.any(evaluation_local):
        raise ValueError("Round 50 evaluation role is empty")
    evaluation_indices = indices[evaluation_local]
    exact_target = barrier.net_payoff_bps[evaluation_indices].astype(np.float64)
    exact_events = barrier.event_code[evaluation_indices]
    exact_classes = event_classes[evaluation_indices]
    group_labels = _event_group_labels(exact_events)
    ensemble_true_probability = np.mean(
        bundle.seed_event_true_probabilities[:, evaluation_local], axis=0
    ).astype(np.float64)
    ensemble_groups = np.mean(
        bundle.seed_event_group_probabilities[:, evaluation_local], axis=0
    ).astype(np.float64)
    ensemble_action = np.mean(
        bundle.seed_action_values_bps[:, evaluation_local], axis=0
    ).astype(np.float64)
    training_event = bundle.target_baselines.event_class_probability
    training_group = np.stack(
        (
            np.sum(training_event[..., :EVENT_MINUTES], axis=-1),
            training_event[..., TIMEOUT_CLASS],
            np.sum(training_event[..., EVENT_MINUTES : 2 * EVENT_MINUTES], axis=-1),
        ),
        axis=-1,
    )
    baseline_action = barrier.stop_bps[evaluation_indices][
        ..., None
    ] * bundle.target_baselines.direct_mean_risk_units.reshape(
        1, len(SYMBOLS), len(SIDES)
    )
    side_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    symbol_rows: list[dict[str, object]] = []
    months = np.asarray(
        [_month(int(temporal.timestamps_ms[index])) for index in evaluation_indices]
    )
    for side_index, side in enumerate(("short", "long")):
        labels = exact_classes[..., side_index]
        probability = ensemble_true_probability[..., side_index]
        baseline_true = np.empty_like(probability)
        for symbol_index in range(len(SYMBOLS)):
            baseline_true[:, symbol_index] = training_event[
                symbol_index, side_index, labels[:, symbol_index]
            ]
        event_log_loss = float(-np.mean(np.log(np.clip(probability, 1e-12, 1.0))))
        baseline_event_log_loss = float(
            -np.mean(np.log(np.clip(baseline_true, 1e-12, 1.0)))
        )
        event_log_loss_skill = (
            baseline_event_log_loss - event_log_loss
        ) / baseline_event_log_loss
        observed_groups = np.eye(3, dtype=np.float64)[group_labels[..., side_index]]
        predicted_groups = ensemble_groups[..., side_index, :]
        baseline_groups = np.broadcast_to(
            training_group[:, side_index, :].reshape(1, len(SYMBOLS), 3),
            predicted_groups.shape,
        )
        group_brier = float(
            np.mean(np.sum((predicted_groups - observed_groups) ** 2, axis=-1))
        )
        baseline_group_brier = float(
            np.mean(np.sum((baseline_groups - observed_groups) ** 2, axis=-1))
        )
        group_brier_skill = (baseline_group_brier - group_brier) / baseline_group_brier
        group_ece = {
            name: _binary_ece(predicted_groups[..., group], observed_groups[..., group])
            for group, name in enumerate(EVENT_GROUP_NAMES)
        }
        actual = exact_target[..., side_index]
        predicted = ensemble_action[..., side_index]
        baseline = baseline_action[..., side_index]
        mse = float(np.mean((predicted - actual) ** 2))
        baseline_mse = float(np.mean((baseline - actual) ** 2))
        mse_skill = (baseline_mse - mse) / baseline_mse
        spearman = _finite_spearman(actual, predicted)
        monthly_positive_event_skill = 0
        monthly_positive_spearman = 0
        for month in sorted(set(months.tolist())):
            selected = months == month
            month_probability = probability[selected]
            month_baseline_probability = baseline_true[selected]
            month_event_loss = float(
                -np.mean(np.log(np.clip(month_probability, 1e-12, 1.0)))
            )
            month_baseline_loss = float(
                -np.mean(np.log(np.clip(month_baseline_probability, 1e-12, 1.0)))
            )
            month_event_skill = (
                month_baseline_loss - month_event_loss
            ) / month_baseline_loss
            month_spearman = _finite_spearman(actual[selected], predicted[selected])
            monthly_positive_event_skill += int(month_event_skill > 0.0)
            monthly_positive_spearman += int(month_spearman > 0.0)
            monthly_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "month": month,
                    "side": side,
                    "rows": int(np.sum(selected) * len(SYMBOLS)),
                    "event_log_loss_skill": month_event_skill,
                    "expected_payoff_spearman": month_spearman,
                }
            )
        positive_symbol_spearman = 0
        for symbol_index, symbol in enumerate(SYMBOLS):
            symbol_spearman = _finite_spearman(
                actual[:, symbol_index], predicted[:, symbol_index]
            )
            positive_symbol_spearman += int(symbol_spearman > 0.0)
            symbol_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "symbol": symbol,
                    "side": side,
                    "rows": int(actual.shape[0]),
                    "expected_payoff_spearman": symbol_spearman,
                    "expected_payoff_mse": float(
                        np.mean(
                            (predicted[:, symbol_index] - actual[:, symbol_index]) ** 2
                        )
                    ),
                }
            )
        side_rows.append(
            {
                "candidate_id": bundle.candidate_id,
                "side": side,
                "rows": int(actual.size),
                "event_log_loss": event_log_loss,
                "baseline_event_log_loss": baseline_event_log_loss,
                "event_log_loss_skill": event_log_loss_skill,
                "event_group_brier": group_brier,
                "baseline_event_group_brier": baseline_group_brier,
                "event_group_brier_skill": group_brier_skill,
                "event_group_ece": group_ece,
                "maximum_event_group_ece": max(group_ece.values()),
                "expected_payoff_mse": mse,
                "baseline_expected_payoff_mse": baseline_mse,
                "expected_payoff_mse_skill": mse_skill,
                "expected_payoff_spearman": spearman,
                "months_with_positive_event_log_loss_skill": monthly_positive_event_skill,
                "months_with_positive_expected_payoff_spearman": monthly_positive_spearman,
                "symbols_with_positive_expected_payoff_spearman": positive_symbol_spearman,
            }
        )
    seed_rows, minimum_seed_spearman = _pairwise_seed_spearman(
        bundle.seed_action_values_bps[:, evaluation_local]
    )
    reasons: list[str] = []
    for row in side_rows:
        side = row["side"]
        if float(row["event_log_loss_skill"]) <= 0.0:
            reasons.append(f"{side}_event_log_loss_skill_not_positive")
        if float(row["event_group_brier_skill"]) <= 0.0:
            reasons.append(f"{side}_event_group_brier_skill_not_positive")
        if float(row["maximum_event_group_ece"]) > 0.05:
            reasons.append(f"{side}_event_group_ece_above_0.05")
        if int(row["months_with_positive_event_log_loss_skill"]) < 4:
            reasons.append(f"{side}_event_month_breadth_below_4")
        if float(row["expected_payoff_mse_skill"]) <= 0.0:
            reasons.append(f"{side}_expected_payoff_mse_skill_not_positive")
        if float(row["expected_payoff_spearman"]) < 0.03:
            reasons.append(f"{side}_expected_payoff_spearman_below_0.03")
        if int(row["symbols_with_positive_expected_payoff_spearman"]) < 3:
            reasons.append(f"{side}_positive_symbol_spearman_below_3")
        if int(row["months_with_positive_expected_payoff_spearman"]) < 4:
            reasons.append(f"{side}_positive_month_spearman_below_4")
    if minimum_seed_spearman < 0.5:
        reasons.append("minimum_pairwise_seed_expected_payoff_spearman_below_0.5")
    return {
        "candidate_id": bundle.candidate_id,
        "evaluation_rows": int(evaluation_indices.size * len(SYMBOLS) * len(SIDES)),
        "sides": side_rows,
        "monthly": monthly_rows,
        "symbols": symbol_rows,
        "seed_stability": seed_rows,
        "minimum_pairwise_seed_expected_payoff_spearman": minimum_seed_spearman,
        "quality_gate": {"passed": not reasons, "reasons": reasons},
    }


def mechanism_ablation_gate(
    diagnostics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    control = diagnostics["direct_barrier_mean_tcn"]
    treatment = diagnostics["competing_risk_barrier_tcn"]
    control_sides = {row["side"]: row for row in control["sides"]}  # type: ignore[index]
    treatment_sides = {row["side"]: row for row in treatment["sides"]}  # type: ignore[index]
    spearman_improvements: list[float] = []
    relative_mse_changes: list[float] = []
    for side in ("short", "long"):
        control_row = control_sides[side]
        treatment_row = treatment_sides[side]
        spearman_improvements.append(
            float(treatment_row["expected_payoff_spearman"])
            - float(control_row["expected_payoff_spearman"])
        )
        control_mse = float(control_row["expected_payoff_mse"])
        relative_mse_changes.append(
            (float(treatment_row["expected_payoff_mse"]) - control_mse) / control_mse
        )
    average_improvement = float(np.mean(spearman_improvements))
    maximum_relative_mse_degradation = float(max(relative_mse_changes))
    reasons: list[str] = []
    if average_improvement < 0.005:
        reasons.append("average_spearman_improvement_below_0.005")
    if maximum_relative_mse_degradation > 0.01:
        reasons.append("relative_mse_degradation_above_0.01")
    if treatment["quality_gate"]["passed"] is not True:  # type: ignore[index]
        reasons.append("competing_risk_quality_gate_failed")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "average_expected_payoff_spearman_improvement": average_improvement,
        "side_spearman_improvements": dict(
            zip(("short", "long"), spearman_improvements, strict=True)
        ),
        "maximum_relative_expected_payoff_mse_degradation": (
            maximum_relative_mse_degradation
        ),
    }


def select_fixed_policy_trades(
    bundle: BarrierCompetingRiskForecastBundle,
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
) -> list[dict[str, object]]:
    indices = bundle.global_indices
    evaluation_local = np.flatnonzero(barrier.role_masks["viability"][indices])
    trades: list[dict[str, object]] = []
    extra_stress_cost = (
        STRESS_EXECUTION_CHARGE_BPS
        - barrier.specification.round_trip_execution_charge_bps
    )
    for symbol_index, symbol in enumerate(SYMBOLS):
        available_after_ms = -1
        for local_index in evaluation_local:
            global_index = int(indices[local_index])
            decision_time_ms = int(temporal.timestamps_ms[global_index])
            if decision_time_ms < available_after_ms:
                continue
            seed_values = bundle.seed_action_values_bps[
                :, local_index, symbol_index
            ].astype(np.float64)
            worst = np.min(seed_values, axis=0)
            eligible = worst > 0.0
            if not np.any(eligible):
                continue
            if eligible[0] and eligible[1]:
                if worst[0] == worst[1]:
                    continue
                side_index = int(np.argmax(worst))
            else:
                side_index = int(np.flatnonzero(eligible)[0])
            event_minute = int(
                barrier.event_minute[global_index, symbol_index, side_index]
            )
            entry_time_ms = decision_time_ms + MINUTE_MS
            exit_time_ms = entry_time_ms + event_minute * MINUTE_MS
            available_after_ms = exit_time_ms
            base_payoff = float(
                barrier.net_payoff_bps[global_index, symbol_index, side_index]
            )
            trades.append(
                {
                    "trade_id": (
                        f"{bundle.candidate_id}:{symbol}:{decision_time_ms}:"
                        f"{SIDES[side_index]}"
                    ),
                    "candidate_id": bundle.candidate_id,
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "decision_index": global_index,
                    "decision_time_ms": decision_time_ms,
                    "entry_time_ms": entry_time_ms,
                    "exit_time_ms": exit_time_ms,
                    "side": SIDES[side_index],
                    "side_name": ("short", "long")[side_index],
                    "event_code": int(
                        barrier.event_code[global_index, symbol_index, side_index]
                    ),
                    "event_name": (
                        "stop_loss",
                        "timeout",
                        "take_profit",
                    )[int(barrier.event_code[global_index, symbol_index, side_index])],
                    "holding_minutes": event_minute,
                    "stop_bps": float(barrier.stop_bps[global_index, symbol_index]),
                    "take_profit_bps": float(
                        barrier.take_profit_bps[global_index, symbol_index]
                    ),
                    "worst_seed_expected_payoff_bps": float(worst[side_index]),
                    "mean_seed_expected_payoff_bps": float(
                        np.mean(seed_values[:, side_index])
                    ),
                    "base_net_payoff_bps": base_payoff,
                    "stress_net_payoff_bps": base_payoff - extra_stress_cost,
                }
            )
    return sorted(trades, key=lambda item: (item["exit_time_ms"], item["trade_id"]))


def _date_grid(start_ms: int, end_ms: int) -> list[str]:
    start = datetime.fromtimestamp(start_ms / 1000.0, tz=UTC).date()
    end = datetime.fromtimestamp(end_ms / 1000.0, tz=UTC).date()
    days = (end - start).days + 1
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def _maximum_drawdown(daily_returns: np.ndarray) -> float:
    equity = 1.0 + np.cumsum(daily_returns, dtype=np.float64)
    curve = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(curve)
    return float(np.max((peak - curve) / peak))


def _circular_block_bootstrap(
    daily_returns: np.ndarray, *, seed: int
) -> dict[str, float]:
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.size == 0:
        return {"lower_return_fraction": 0.0, "median_return_fraction": 0.0}
    generator = np.random.default_rng(seed)
    blocks = math.ceil(values.size / BOOTSTRAP_BLOCK_DAYS)
    totals = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    offsets = np.arange(BOOTSTRAP_BLOCK_DAYS, dtype=np.int64)
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = generator.integers(0, values.size, size=blocks)
        selected = (starts[:, None] + offsets[None, :]) % values.size
        totals[sample] = float(np.sum(values[selected.reshape(-1)[: values.size]]))
    return {
        "lower_return_fraction": float(np.quantile(totals, FAMILYWISE_LOWER_QUANTILE)),
        "median_return_fraction": float(np.median(totals)),
    }


def replay_fixed_trades(
    trades: Sequence[Mapping[str, object]],
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
    *,
    candidate_index: int,
) -> dict[str, object]:
    evaluation_indices = np.flatnonzero(barrier.role_masks["viability"])
    dates = _date_grid(
        int(temporal.timestamps_ms[evaluation_indices[0]]),
        int(temporal.timestamps_ms[evaluation_indices[-1]]),
    )
    scenarios: dict[str, object] = {}
    for scenario_index, (scenario, payoff_field, cost_bps) in enumerate(
        (
            (
                "base",
                "base_net_payoff_bps",
                barrier.specification.round_trip_execution_charge_bps,
            ),
            ("stress", "stress_net_payoff_bps", STRESS_EXECUTION_CHARGE_BPS),
        )
    ):
        daily = {date: 0.0 for date in dates}
        pnl_values: list[float] = []
        symbol_pnl = {symbol: 0.0 for symbol in SYMBOLS}
        symbol_trades = {symbol: 0 for symbol in SYMBOLS}
        for trade in trades:
            payoff_bps = float(trade[payoff_field])
            return_fraction = payoff_bps / 10_000.0 * SLEEVE_FRACTION
            exit_date = _date(int(trade["exit_time_ms"]))
            daily[exit_date] += return_fraction
            pnl_values.append(return_fraction)
            symbol = str(trade["symbol"])
            symbol_pnl[symbol] += return_fraction
            symbol_trades[symbol] += 1
        daily_values = np.asarray([daily[date] for date in dates], dtype=np.float64)
        positive = sum(value for value in pnl_values if value > 0.0)
        negative = -sum(value for value in pnl_values if value < 0.0)
        total_return = float(np.sum(daily_values))
        bootstrap = _circular_block_bootstrap(
            daily_values,
            seed=5050 + 10 * candidate_index + scenario_index,
        )
        scenarios[scenario] = {
            "execution_charge_bps": float(cost_bps),
            "closed_trades": len(trades),
            "active_days": int(np.count_nonzero(daily_values)),
            "total_return_fraction": total_return,
            "maximum_drawdown_fraction": _maximum_drawdown(daily_values),
            "profit_factor": (
                float(positive / negative)
                if negative > 0.0
                else (math.inf if positive > 0.0 else 0.0)
            ),
            "win_rate": (
                float(np.mean(np.asarray(pnl_values) > 0.0)) if pnl_values else 0.0
            ),
            "mean_net_payoff_bps": (
                float(np.mean([float(item[payoff_field]) for item in trades]))
                if trades
                else 0.0
            ),
            "bootstrap": bootstrap,
            "symbols_with_positive_net_pnl": sum(
                value > 0.0 for value in symbol_pnl.values()
            ),
            "single_symbol_positive_pnl_share": (
                max(max(value, 0.0) for value in symbol_pnl.values())
                / sum(max(value, 0.0) for value in symbol_pnl.values())
                if any(value > 0.0 for value in symbol_pnl.values())
                else 1.0
            ),
            "symbol_net_return_fraction": symbol_pnl,
            "symbol_closed_trades": symbol_trades,
            "daily": [
                {
                    "date": date,
                    "return_fraction": daily[date],
                    "equity_fraction": 1.0 + float(np.sum(daily_values[: index + 1])),
                }
                for index, date in enumerate(dates)
            ],
        }
    return {"trades": list(trades), "scenarios": scenarios}


def economic_gate(
    replay: Mapping[str, object],
    *,
    quality_gate_passed: bool,
    mechanism_gate_passed: bool,
    candidate_id: str,
) -> dict[str, object]:
    reasons: list[str] = []
    if not quality_gate_passed:
        reasons.append("quality_gate_failed")
    if candidate_id == "competing_risk_barrier_tcn" and not mechanism_gate_passed:
        reasons.append("mechanism_gate_failed")
    scenarios = replay["scenarios"]
    for name in ("base", "stress"):
        metrics = scenarios[name]
        if float(metrics["total_return_fraction"]) <= 0.0:
            reasons.append(f"{name}_return_not_positive")
        if float(metrics["profit_factor"]) <= 1.0:
            reasons.append(f"{name}_profit_factor_not_above_one")
        if float(metrics["bootstrap"]["lower_return_fraction"]) <= 0.0:
            reasons.append(f"{name}_bootstrap_lower_bound_not_positive")
    base = scenarios["base"]
    if int(base["closed_trades"]) < 100:
        reasons.append("closed_trades_below_100")
    if int(base["active_days"]) < 30:
        reasons.append("active_days_below_30")
    if int(base["symbols_with_positive_net_pnl"]) < 2:
        reasons.append("positive_symbols_below_2")
    if float(base["single_symbol_positive_pnl_share"]) > 0.7:
        reasons.append("single_symbol_positive_pnl_share_above_0.7")
    if float(base["maximum_drawdown_fraction"]) > 0.08:
        reasons.append("unlevered_drawdown_above_0.08")
    return {"passed": not reasons, "reasons": reasons}


def leverage_sensitivity(
    replay: Mapping[str, object], economic_gate_result: Mapping[str, object]
) -> dict[str, object]:
    if economic_gate_result.get("passed") is not True:
        return {
            "run": False,
            "reason": "unlevered_candidate_failed_mandatory_gates",
            "leverage": [],
        }
    base_daily = np.asarray(
        [row["return_fraction"] for row in replay["scenarios"]["base"]["daily"]],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    for leverage in (5.0, 10.0, 15.0):
        leveraged = leverage * base_daily
        rows.append(
            {
                "leverage": leverage,
                "total_return_fraction": float(np.sum(leveraged)),
                "maximum_drawdown_fraction": _maximum_drawdown(leveraged),
                "liquidation_model_complete": False,
                "eligible_for_trading_authority": False,
            }
        )
    return {
        "run": True,
        "reason": "return_and_drawdown_sensitivity_only; historical exchange-tier liquidation reconstruction remains incomplete",
        "leverage": rows,
    }


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "FAMILYWISE_LOWER_QUANTILE",
    "STRESS_EXECUTION_CHARGE_BPS",
    "candidate_diagnostics",
    "economic_gate",
    "leverage_sensitivity",
    "mechanism_ablation_gate",
    "replay_fixed_trades",
    "select_fixed_policy_trades",
]

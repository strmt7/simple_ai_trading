from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from simple_ai_trading.microstructure_features import (
    apply_path_aware_lifecycle_targets,
    build_executable_microstructure_dataset,
)
from simple_ai_trading.microstructure_model import (
    _apply_platt_scaling,
    _backend_parameters,
    _fit_platt_scaling,
    _minimum_evaluation_trades,
    _performance_confidence,
    _purged_split,
    _purged_tuning_subsplit,
    _risk_parameters,
    _risk_utility,
    _SimulationTrace,
    _simulate_non_overlapping_trace,
    _train_booster,
    _trading_metrics,
    load_microstructure_model_artifact,
)
from simple_ai_trading.microstructure_warehouse import MicrostructureWarehouse
from simple_ai_trading.storage import write_json_atomic
try:
    from tools.run_action_value_discovery import (
        _canonical_sha256,
        _date_bounds,
        discovery_candidates,
        load_discovery_design,
    )
except ModuleNotFoundError:
    from run_action_value_discovery import (
        _canonical_sha256,
        _date_bounds,
        discovery_candidates,
        load_discovery_design,
    )


_TOP_COUNTS = (20, 50, 100, 250, 500, 1_000)
_DAY_MS = 86_400_000
_CUSUM_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_thresholds(scores: np.ndarray, risk_level: str) -> tuple[float, ...]:
    values = np.asarray(scores, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("score threshold search has no finite candidates")
    quantiles = {
        "conservative": (0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999),
        "regular": (0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998),
        "aggressive": (0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99),
    }[risk_level]
    return tuple(sorted({float(np.quantile(finite, value)) for value in quantiles}))


def _strongest_score(
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    long_eligible: np.ndarray,
    short_eligible: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        np.where(long_eligible, long_scores, -np.inf),
        np.where(short_eligible, short_scores, -np.inf),
    )


def _simulate_score_threshold(
    *,
    dataset,
    indexes: np.ndarray,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    threshold: float,
):
    ones = np.ones(len(indexes), dtype=np.float64)
    return _simulate_non_overlapping_trace(
        timestamps=dataset.decision_time_ms[indexes],
        long_exit_times=dataset.long_exit_time_ms[indexes],
        short_exit_times=dataset.short_exit_time_ms[indexes],
        long_targets=dataset.long_net_bps[indexes],
        short_targets=dataset.short_net_bps[indexes],
        long_edge=long_scores,
        short_edge=short_scores,
        long_probability=ones,
        short_probability=ones,
        edge_threshold=threshold,
        probability_threshold=0.0,
        long_eligible=dataset.long_liquidity_eligible[indexes],
        short_eligible=dataset.short_liquidity_eligible[indexes],
    )


def _select_score_threshold(
    *,
    dataset,
    indexes: np.ndarray,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    risk_level: str,
) -> dict[str, object]:
    long_values = np.asarray(long_scores, dtype=np.float64)
    short_values = np.asarray(short_scores, dtype=np.float64)
    long_eligible = np.asarray(dataset.long_liquidity_eligible[indexes], dtype=bool)
    short_eligible = np.asarray(dataset.short_liquidity_eligible[indexes], dtype=bool)
    if long_values.shape != short_values.shape or long_values.shape != indexes.shape:
        raise ValueError("score threshold arrays are inconsistent")
    strongest = _strongest_score(
        long_values,
        short_values,
        long_eligible,
        short_eligible,
    )
    thresholds = _score_thresholds(strongest, risk_level)
    minimum_trades = _minimum_evaluation_trades(dataset.decision_time_ms[indexes])
    best_trace = None
    best_threshold = None
    best_utility = -np.inf
    eligible_policies = 0
    evaluations: list[dict[str, object]] = []
    for threshold in thresholds:
        trace = _simulate_score_threshold(
            dataset=dataset,
            indexes=indexes,
            long_scores=long_values,
            short_scores=short_values,
            threshold=threshold,
        )
        utility = _risk_utility(trace.metrics, risk_level)
        qualifies = trace.metrics.trades >= minimum_trades
        eligible_policies += int(qualifies)
        evaluations.append(
            {
                "threshold": threshold,
                "utility_bps": utility,
                "metrics": asdict(trace.metrics),
                "meets_trade_minimum": qualifies,
            }
        )
        if qualifies and utility > best_utility:
            best_trace = trace
            best_threshold = threshold
            best_utility = utility
    accepted = best_trace is not None and best_utility > 0.0
    return {
        "accepted": accepted,
        "threshold": float(best_threshold) if accepted else None,
        "minimum_trades": minimum_trades,
        "evaluated_thresholds": len(thresholds),
        "policies_meeting_trade_minimum": eligible_policies,
        "best_observed_threshold": (
            float(best_threshold) if best_threshold is not None else None
        ),
        "best_observed_utility_bps": (
            float(best_utility) if np.isfinite(best_utility) else None
        ),
        "best_observed_metrics": (
            asdict(best_trace.metrics) if best_trace is not None else None
        ),
        "evaluations": evaluations,
    }


def _top_score_diagnostic(
    *,
    dataset,
    indexes: np.ndarray,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
) -> list[dict[str, object]]:
    long_eligible = np.asarray(dataset.long_liquidity_eligible[indexes], dtype=bool)
    short_eligible = np.asarray(dataset.short_liquidity_eligible[indexes], dtype=bool)
    strongest = _strongest_score(
        long_scores,
        short_scores,
        long_eligible,
        short_eligible,
    )
    choose_long = long_eligible & (~short_eligible | (long_scores >= short_scores))
    actual = np.where(
        choose_long,
        dataset.long_net_bps[indexes],
        dataset.short_net_bps[indexes],
    )
    usable = np.flatnonzero(np.isfinite(strongest))
    order = usable[np.argsort(strongest[usable], kind="stable")[::-1]]
    output: list[dict[str, object]] = []
    for requested in _TOP_COUNTS:
        count = min(requested, len(order))
        if count <= 0:
            continue
        selected = order[:count]
        output.append(
            {
                "requested_rows": requested,
                "rows": count,
                "mean_score": float(np.mean(strongest[selected])),
                "mean_actual_net_bps": float(np.mean(actual[selected])),
                "actual_total_net_bps": float(np.sum(actual[selected])),
                "actual_profitable_ratio": float(np.mean(actual[selected] > 0.0)),
            }
        )
    return output


def _daily_top_score_diagnostic(
    *,
    dataset,
    indexes: np.ndarray,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
) -> list[dict[str, object]]:
    day_ids = dataset.decision_time_ms[indexes] // _DAY_MS
    output: list[dict[str, object]] = []
    for day_id in np.unique(day_ids):
        mask = day_ids == day_id
        day_indexes = indexes[mask]
        top = _top_score_diagnostic(
            dataset=dataset,
            indexes=day_indexes,
            long_scores=np.asarray(long_scores)[mask],
            short_scores=np.asarray(short_scores)[mask],
        )
        output.append(
            {
                "utc_day_id": int(day_id),
                "decision_rows": int(np.sum(mask)),
                "top_score_rows": top,
            }
        )
    return output


def _walk_forward_score_policy(
    *,
    dataset,
    policy_indexes: np.ndarray,
    selection_indexes: np.ndarray,
    policy_long_scores: np.ndarray,
    policy_short_scores: np.ndarray,
    selection_long_scores: np.ndarray,
    selection_short_scores: np.ndarray,
    risk_level: str,
    lookback_days: int | None,
) -> tuple[_SimulationTrace, list[dict[str, object]]]:
    combined_indexes = np.concatenate((policy_indexes, selection_indexes))
    combined_long = np.concatenate((policy_long_scores, selection_long_scores))
    combined_short = np.concatenate((policy_short_scores, selection_short_scores))
    selection_days = np.unique(dataset.decision_time_ms[selection_indexes] // _DAY_MS)
    pnls: list[float] = []
    sides: list[int] = []
    timestamps: list[int] = []
    next_available_ms = -1
    daily: list[dict[str, object]] = []
    for day_id in selection_days:
        day_start = int(day_id * _DAY_MS)
        day_end = day_start + _DAY_MS
        history_mask = (
            (dataset.decision_time_ms[combined_indexes] < day_start)
            & (dataset.long_exit_time_ms[combined_indexes] < day_start)
            & (dataset.short_exit_time_ms[combined_indexes] < day_start)
        )
        if lookback_days is not None:
            history_mask &= dataset.decision_time_ms[combined_indexes] >= (
                day_start - lookback_days * _DAY_MS
            )
        history_indexes = combined_indexes[history_mask]
        policy = _select_score_threshold(
            dataset=dataset,
            indexes=history_indexes,
            long_scores=combined_long[history_mask],
            short_scores=combined_short[history_mask],
            risk_level=risk_level,
        )
        day_mask = (
            (dataset.decision_time_ms[selection_indexes] >= day_start)
            & (dataset.decision_time_ms[selection_indexes] < day_end)
            & (dataset.decision_time_ms[selection_indexes] >= next_available_ms)
        )
        day_indexes = selection_indexes[day_mask]
        if policy["accepted"] and len(day_indexes):
            trace = _simulate_score_threshold(
                dataset=dataset,
                indexes=day_indexes,
                long_scores=np.asarray(selection_long_scores)[day_mask],
                short_scores=np.asarray(selection_short_scores)[day_mask],
                threshold=float(policy["threshold"]),
            )
            pnls.extend(trace.pnls)
            sides.extend(trace.sides)
            timestamps.extend(trace.timestamps)
            if trace.timestamps:
                final_timestamp = int(trace.timestamps[-1])
                final_position = int(
                    np.searchsorted(dataset.decision_time_ms, final_timestamp)
                )
                next_available_ms = int(
                    dataset.long_exit_time_ms[final_position]
                    if trace.sides[-1] == 1
                    else dataset.short_exit_time_ms[final_position]
                )
        else:
            trace = _SimulationTrace(
                metrics=_trading_metrics([], [], []),
                pnls=(),
                sides=(),
                timestamps=(),
            )
        daily.append(
            {
                "utc_day_id": int(day_id),
                "history_rows": int(len(history_indexes)),
                "lookback_days": lookback_days,
                "policy": policy,
                "day_metrics": asdict(trace.metrics),
            }
        )
    return (
        _SimulationTrace(
            metrics=_trading_metrics(pnls, sides, timestamps),
            pnls=tuple(pnls),
            sides=tuple(sides),
            timestamps=tuple(timestamps),
        ),
        daily,
    )


def _reference_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(np.asarray(reference, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size < 256:
        raise ValueError("rank ensemble reference has insufficient finite rows")
    return np.searchsorted(finite, values, side="right").astype(np.float64) / finite.size


def _causal_cusum_events(
    dataset,
    *,
    volatility_multiplier: float,
    minimum_threshold_bps: float = 1.0,
) -> np.ndarray:
    multiplier = float(volatility_multiplier)
    floor = float(minimum_threshold_bps)
    if multiplier <= 0.0 or floor <= 0.0:
        raise ValueError("CUSUM event thresholds must be positive")
    try:
        return_index = dataset.feature_names.index("return_5s_bps")
        volatility_index = dataset.feature_names.index(
            "realized_volatility_60s_bps"
        )
    except ValueError as exc:
        raise ValueError("CUSUM event features are missing") from exc
    returns = np.asarray(dataset.features[:, return_index], dtype=np.float64)
    volatility = np.asarray(
        dataset.features[:, volatility_index],
        dtype=np.float64,
    )
    thresholds = np.maximum(
        floor,
        multiplier * np.maximum(volatility, 0.0) * np.sqrt(60.0),
    )
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(thresholds)):
        raise ValueError("CUSUM event inputs are non-finite")
    days = dataset.decision_time_ms // _DAY_MS
    events = np.zeros(dataset.rows, dtype=bool)
    positive = 0.0
    negative = 0.0
    prior_day = int(days[0]) if dataset.rows else 0
    for index, value in enumerate(returns):
        day = int(days[index])
        if day != prior_day:
            positive = 0.0
            negative = 0.0
            prior_day = day
        positive = max(0.0, positive + float(value))
        negative = min(0.0, negative + float(value))
        threshold = float(thresholds[index])
        if positive >= threshold or negative <= -threshold:
            events[index] = True
            positive = 0.0
            negative = 0.0
    return events


def _average_label_uniqueness(
    dataset,
    indexes: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    selected = np.asarray(indexes, dtype=np.int64)
    if side not in {"long", "short"} or selected.ndim != 1 or selected.size == 0:
        raise ValueError("label uniqueness inputs are invalid")
    exits = (
        dataset.long_exit_time_ms[selected]
        if side == "long"
        else dataset.short_exit_time_ms[selected]
    )
    end_positions = np.searchsorted(
        dataset.decision_time_ms,
        exits,
        side="right",
    ) - 1
    end_positions = np.maximum(selected, np.minimum(end_positions, dataset.rows - 1))
    difference = np.zeros(dataset.rows + 1, dtype=np.int32)
    np.add.at(difference, selected, 1)
    np.add.at(difference, end_positions + 1, -1)
    concurrency = np.cumsum(difference[:-1], dtype=np.int64)
    inverse = np.divide(
        1.0,
        concurrency,
        out=np.zeros(dataset.rows, dtype=np.float64),
        where=concurrency > 0,
    )
    prefix = np.concatenate(([0.0], np.cumsum(inverse, dtype=np.float64)))
    durations = end_positions - selected + 1
    uniqueness = (prefix[end_positions + 1] - prefix[selected]) / durations
    if not np.all(np.isfinite(uniqueness)) or np.any(uniqueness <= 0.0):
        raise ValueError("label uniqueness calculation failed")
    return (uniqueness / np.mean(uniqueness)).astype(np.float32)


def _train_event_models(
    dataset,
    splits: Mapping[str, np.ndarray],
    *,
    event_mask: np.ndarray,
    risk_level: str,
    compute_backend: str,
    seed: int,
) -> tuple[
    dict[str, lgb.Booster],
    dict[str, int],
    dict[str, tuple[float, float]],
    dict[str, object],
]:
    x = np.asarray(dataset.features, dtype=np.float32)
    train = np.asarray(splits["train"], dtype=np.int64)
    split_evidence = _purged_split(dataset)[1]
    early_stop, calibration = _purged_tuning_subsplit(
        dataset.decision_time_ms,
        splits["tuning"],
        purge_ms=split_evidence.purge_ms,
    )
    backend, kind, device = _backend_parameters(compute_backend, seed)
    base_parameters = {**backend, **_risk_parameters(risk_level, len(train))}
    models: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    calibrations: dict[str, tuple[float, float]] = {}
    evidence: dict[str, object] = {}
    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    eligibility = {
        "long": np.asarray(dataset.long_liquidity_eligible, dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible, dtype=bool),
    }
    specifications: Sequence[tuple[str, str, str, Mapping[str, object]]] = (
        ("mean", "regression", "l2", {}),
        ("q10", "quantile", "quantile", {"alpha": 0.10}),
        ("q90", "quantile", "quantile", {"alpha": 0.90}),
    )
    for side, target in targets.items():
        side_mask = eligibility[side] & event_mask
        role_indexes = {
            "train": train[side_mask[train]],
            "early_stop": early_stop[side_mask[early_stop]],
            "calibration": calibration[side_mask[calibration]],
        }
        role_weights = {
            role: _average_label_uniqueness(dataset, indexes, side=side)
            for role, indexes in role_indexes.items()
        }
        labels = {
            role: (target[indexes] > 0.0).astype(np.float32)
            for role, indexes in role_indexes.items()
        }
        if any(
            min(int(np.sum(value == 0.0)), int(np.sum(value == 1.0))) < 64
            for value in labels.values()
        ):
            raise ValueError(f"event model {side} class support is insufficient")
        probability, probability_iteration = _train_booster(
            x_train=x[role_indexes["train"]],
            y_train=labels["train"],
            x_tuning=x[role_indexes["early_stop"]],
            y_tuning=labels["early_stop"],
            objective="binary",
            metric="binary_logloss",
            parameters=base_parameters,
            train_weights=role_weights["train"],
            tuning_weights=role_weights["early_stop"],
        )
        probability_name = f"{side}_probability"
        models[probability_name] = probability
        iterations[probability_name] = probability_iteration
        raw_calibration = probability.predict(
            x[role_indexes["calibration"]],
            num_iteration=probability_iteration,
        )
        calibrations[side] = _fit_platt_scaling(
            raw_calibration,
            labels["calibration"],
        )
        for label, objective, metric, extra in specifications:
            model, iteration = _train_booster(
                x_train=x[role_indexes["train"]],
                y_train=np.asarray(
                    target[role_indexes["train"]],
                    dtype=np.float32,
                ),
                x_tuning=x[role_indexes["early_stop"]],
                y_tuning=np.asarray(
                    target[role_indexes["early_stop"]],
                    dtype=np.float32,
                ),
                objective=objective,
                metric=metric,
                parameters={**base_parameters, **extra},
                train_weights=role_weights["train"],
                tuning_weights=role_weights["early_stop"],
            )
            name = f"{side}_{label}"
            models[name] = model
            iterations[name] = iteration
        evidence[side] = {
            role: {
                "rows": int(len(indexes)),
                "profitable_rows": int(np.sum(labels[role] == 1.0)),
                "non_profitable_rows": int(np.sum(labels[role] == 0.0)),
                "uniqueness_min": float(np.min(role_weights[role])),
                "uniqueness_mean": float(np.mean(role_weights[role])),
                "uniqueness_max": float(np.max(role_weights[role])),
            }
            for role, indexes in role_indexes.items()
        }
    evidence.update(
        {
            "backend_kind": kind,
            "backend_device": device,
            "model_sha256": _canonical_sha256(
                {
                    name: model.model_to_string(num_iteration=iterations[name])
                    for name, model in sorted(models.items())
                }
            ),
        }
    )
    return models, iterations, calibrations, evidence


def _event_model_scores(
    *,
    features: np.ndarray,
    models: Mapping[str, lgb.Booster],
    iterations: Mapping[str, int],
    calibrations: Mapping[str, tuple[float, float]],
    risk_level: str,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {
        "event_profitability_probability": {},
        "event_direct_mean": {},
        "event_upper_quantile": {},
        "event_lower_quantile": {},
        "event_downside_adjusted_mean": {},
    }
    penalty = {"conservative": 1.0, "regular": 0.75, "aggressive": 0.5}[
        risk_level
    ]
    for side in ("long", "short"):
        probability_name = f"{side}_probability"
        raw_probability = models[probability_name].predict(
            features,
            num_iteration=iterations[probability_name],
        )
        output["event_profitability_probability"][side] = _apply_platt_scaling(
            raw_probability,
            calibrations[side],
        )
        values = {}
        for label in ("mean", "q10", "q90"):
            name = f"{side}_{label}"
            values[label] = models[name].predict(
                features,
                num_iteration=iterations[name],
            )
        output["event_direct_mean"][side] = values["mean"]
        output["event_upper_quantile"][side] = values["q90"]
        output["event_lower_quantile"][side] = values["q10"]
        downside = np.maximum(0.0, values["mean"] - values["q10"])
        output["event_downside_adjusted_mean"][side] = (
            values["mean"] - penalty * downside
        )
    return output


def _compact_policy(policy: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in policy.items()
        if key != "evaluations"
    }


def _requested_top_row(
    rows: Sequence[Mapping[str, object]],
    requested: int = 100,
) -> dict[str, object] | None:
    return next(
        (dict(value) for value in rows if int(value["requested_rows"]) == requested),
        None,
    )


def summarize_ablation(
    payload: Mapping[str, object],
    *,
    source_file_sha256: str,
) -> dict[str, object]:
    raw_outcomes = payload.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise ValueError("ablation outcomes are invalid")
    methods: list[dict[str, object]] = []
    static_accepted = 0
    adaptive_trading = 0
    cusum_accepted = 0
    for raw in raw_outcomes:
        if not isinstance(raw, Mapping):
            raise ValueError("ablation method result is invalid")
        policy = raw["policy"]
        adaptive = raw["adaptive_selection"]
        event_filters = raw["cusum_event_filters"]
        assert isinstance(policy, Mapping)
        assert isinstance(adaptive, Mapping)
        assert isinstance(event_filters, Mapping)
        static_accepted += int(policy["accepted"] is True)
        compact_adaptive: dict[str, object] = {}
        for lookback, value in adaptive.items():
            assert isinstance(value, Mapping)
            daily = value["daily"]
            metrics = value["metrics"]
            assert isinstance(daily, list)
            assert isinstance(metrics, Mapping)
            accepted_days = sum(
                int(
                    isinstance(day, Mapping)
                    and isinstance(day.get("policy"), Mapping)
                    and day["policy"].get("accepted") is True
                )
                for day in daily
            )
            adaptive_trading += int(int(metrics["trades"]) > 0)
            compact_adaptive[str(lookback)] = {
                "accepted_days": accepted_days,
                "metrics": dict(metrics),
                "confidence": dict(value["confidence"]),
            }
        compact_filters: dict[str, object] = {}
        for multiplier, value in event_filters.items():
            assert isinstance(value, Mapping)
            filter_policy = value["policy"]
            assert isinstance(filter_policy, Mapping)
            cusum_accepted += int(filter_policy["accepted"] is True)
            compact_filters[str(multiplier)] = {
                "policy_event_rows": int(value["policy_event_rows"]),
                "selection_event_rows": int(value["selection_event_rows"]),
                "policy": _compact_policy(filter_policy),
                "selection_metrics": dict(value["selection_metrics"]),
                "selection_confidence": dict(value["selection_confidence"]),
            }
        methods.append(
            {
                "method": str(raw["method"]),
                "policy": _compact_policy(policy),
                "selection_metrics": dict(raw["selection_metrics"]),
                "selection_confidence": dict(raw["selection_confidence"]),
                "policy_top_100": _requested_top_row(raw["policy_top_score_rows"]),
                "selection_top_100": _requested_top_row(
                    raw["selection_top_score_rows"]
                ),
                "adaptive_selection": compact_adaptive,
                "cusum_event_filters": compact_filters,
            }
        )
    summary: dict[str, object] = {
        "schema_version": "action-value-score-ablation-summary-v1",
        "artifact_class": payload["artifact_class"],
        "source_ablation_sha256": payload["ablation_sha256"],
        "source_ablation_file_sha256": source_file_sha256,
        "design_sha256": payload["design_sha256"],
        "source_report_sha256": payload["source_report_sha256"],
        "source_artifact_sha256": payload["source_artifact_sha256"],
        "corpus_certificate_sha256": payload["corpus_certificate_sha256"],
        "candidate": dict(payload["candidate"]),
        "dataset_rows": int(payload["dataset_rows"]),
        "direct_models": dict(payload["direct_models"]),
        "event_models": dict(payload["event_models"]),
        "selection_is_consumed": payload["selection_is_consumed"],
        "terminal_holdout_accessed": payload["terminal_holdout_accessed"],
        "trading_authority": payload["trading_authority"],
        "profitability_claim": payload["profitability_claim"],
        "conclusion": {
            "method_count": len(methods),
            "static_policy_accepted_count": static_accepted,
            "adaptive_trading_configuration_count": adaptive_trading,
            "cusum_filter_policy_accepted_count": cusum_accepted,
            "status": "rejected",
        },
        "methods": methods,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    return summary


def _train_direct_models(
    dataset,
    splits: Mapping[str, np.ndarray],
    *,
    risk_level: str,
    compute_backend: str,
    seed: int,
) -> tuple[dict[str, lgb.Booster], dict[str, int], dict[str, str]]:
    x = np.asarray(dataset.features, dtype=np.float32)
    train = np.asarray(splits["train"], dtype=np.int64)
    early_stop, _calibration = _purged_tuning_subsplit(
        dataset.decision_time_ms,
        splits["tuning"],
        purge_ms=int(_purged_split(dataset)[1].purge_ms),
    )
    backend, kind, device = _backend_parameters(compute_backend, seed)
    base_parameters = {**backend, **_risk_parameters(risk_level, len(train))}
    models: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    eligibility = {
        "long": np.asarray(dataset.long_liquidity_eligible, dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible, dtype=bool),
    }
    specifications: Sequence[tuple[str, str, str, Mapping[str, object]]] = (
        ("mean", "regression", "l2", {}),
        ("q10", "quantile", "quantile", {"alpha": 0.10}),
        ("q90", "quantile", "quantile", {"alpha": 0.90}),
    )
    for side, target in targets.items():
        train_indexes = train[eligibility[side][train]]
        tuning_indexes = early_stop[eligibility[side][early_stop]]
        for label, objective, metric, extra in specifications:
            model, iteration = _train_booster(
                x_train=x[train_indexes],
                y_train=np.asarray(target[train_indexes], dtype=np.float32),
                x_tuning=x[tuning_indexes],
                y_tuning=np.asarray(target[tuning_indexes], dtype=np.float32),
                objective=objective,
                metric=metric,
                parameters={**base_parameters, **extra},
                train_weights=np.ones(len(train_indexes), dtype=np.float32),
                tuning_weights=np.ones(len(tuning_indexes), dtype=np.float32),
            )
            name = f"{side}_{label}"
            models[name] = model
            iterations[name] = iteration
    identity = {
        "backend_kind": kind,
        "backend_device": device,
        "model_sha256": _canonical_sha256(
            {
                name: model.model_to_string(num_iteration=iterations[name])
                for name, model in sorted(models.items())
            }
        ),
    }
    return models, iterations, identity


def _artifact_scores(artifact, features: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    models = {
        name: lgb.Booster(model_str=value)
        for name, value in artifact.model_strings.items()
    }
    output: dict[str, dict[str, np.ndarray]] = {
        "hurdle_ev": {},
        "profitability_probability": {},
    }
    for side in ("long", "short"):
        probability_name = f"{side}_probability"
        raw_probability = models[probability_name].predict(
            features,
            num_iteration=artifact.best_iterations[probability_name],
        )
        probability = _apply_platt_scaling(
            raw_probability,
            artifact.probability_calibration[side],
        )
        win_name = f"{side}_win_magnitude"
        loss_name = f"{side}_loss_magnitude"
        win = np.maximum(
            0.0,
            models[win_name].predict(
                features,
                num_iteration=artifact.best_iterations[win_name],
            ),
        )
        loss = np.maximum(
            0.0,
            models[loss_name].predict(
                features,
                num_iteration=artifact.best_iterations[loss_name],
            ),
        )
        output["hurdle_ev"][side] = probability * win - (1.0 - probability) * loss
        output["profitability_probability"][side] = probability
    return output


def _segment_scores(
    *,
    dataset,
    indexes: np.ndarray,
    artifact,
    direct_models: Mapping[str, lgb.Booster],
    direct_iterations: Mapping[str, int],
) -> dict[str, dict[str, np.ndarray]]:
    features = np.asarray(dataset.features[indexes], dtype=np.float32)
    methods = _artifact_scores(artifact, features)
    methods.update(
        {
            "direct_mean": {},
            "upper_quantile": {},
            "lower_quantile": {},
            "downside_adjusted_mean": {},
        }
    )
    penalty = {"conservative": 1.0, "regular": 0.75, "aggressive": 0.5}[
        artifact.risk_level
    ]
    for side in ("long", "short"):
        predictions = {}
        for label in ("mean", "q10", "q90"):
            name = f"{side}_{label}"
            predictions[label] = direct_models[name].predict(
                features,
                num_iteration=direct_iterations[name],
            )
        methods["direct_mean"][side] = predictions["mean"]
        methods["upper_quantile"][side] = predictions["q90"]
        methods["lower_quantile"][side] = predictions["q10"]
        downside = np.maximum(0.0, predictions["mean"] - predictions["q10"])
        methods["downside_adjusted_mean"][side] = predictions["mean"] - penalty * downside
    return methods


def run_ablation(
    *,
    design_path: Path,
    evidence_root: Path,
    warehouse_path: Path,
    cache_root: Path,
    candidate_id: str,
    output_path: Path,
    memory_limit: str,
    threads: int,
    compute_backend: str,
    summary_output_path: Path | None = None,
) -> dict[str, object]:
    status_path = output_path.with_suffix(output_path.suffix + ".status.json")

    def progress(phase: str) -> None:
        print(f"ablate-action-value-scores {phase}", flush=True)
        write_json_atomic(
            status_path,
            {
                "schema_version": "action-value-score-ablation-status-v1",
                "candidate_id": candidate_id,
                "phase": phase,
            },
            indent=2,
            sort_keys=True,
        )

    progress("validate-design")
    design = load_discovery_design(design_path, require_current=True)
    candidate = next(
        (
            value
            for value in discovery_candidates(design)
            if value["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"unknown discovery candidate: {candidate_id}")
    artifact_path = evidence_root / f"{candidate_id}.json"
    report_path = evidence_root / "report.json"
    artifact = load_microstructure_model_artifact(artifact_path)
    if (
        artifact.status != "rejected"
        or artifact.risk_level != candidate["risk_level"]
        or artifact.horizon_seconds != candidate["horizon_seconds"]
        or artifact.terminal_evaluated_at is not None
    ):
        raise ValueError("ablation source artifact contract is invalid")
    data = design["data"]
    execution = design["execution"]
    training = design["training"]
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(training, Mapping)
    start_ms, end_ms = _date_bounds(design)
    progress("certify-corpus")
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        certificate = warehouse.require_corpus_certificate(
            str(data["symbol"]),
            required_data_types=tuple(data["required_data_types"]),
            required_start_ms=start_ms,
            required_end_ms=end_ms,
            require_full_history_inventory=True,
        )
        progress("build-dataset")
        base = build_executable_microstructure_dataset(
            warehouse,
            symbol=str(data["symbol"]),
            horizon_seconds=int(candidate["horizon_seconds"]),
            total_latency_ms=int(execution["total_latency_ms"]),
            taker_fee_bps=float(execution["taker_fee_bps_per_side"]),
            additional_slippage_bps_per_side=float(
                execution["additional_slippage_bps_per_side"]
            ),
            max_quote_age_ms=int(execution["max_quote_age_ms"]),
            reference_order_notional_quote=float(
                execution["reference_order_notional_quote"]
            ),
            max_l1_participation=1.0,
            decision_cadence_seconds=int(execution["decision_cadence_seconds"]),
            start_ms=start_ms,
            end_ms=end_ms,
        )
        limit = float(candidate["max_l1_participation"])
        dataset = replace(
            base,
            max_l1_participation=limit,
            long_liquidity_eligible=np.asarray(base.long_l1_participation) <= limit,
            short_liquidity_eligible=np.asarray(base.short_l1_participation) <= limit,
        )
        progress("build-lifecycle-targets")
        dataset, path_evidence = apply_path_aware_lifecycle_targets(
            warehouse,
            dataset,
            stop_loss_bps=float(candidate["stop_loss_bps"]),
            take_profit_bps=float(candidate["take_profit_bps"]),
            trigger_execution_slippage_bps=float(
                execution["trigger_execution_slippage_bps"]
            ),
        )
    splits, split_evidence = _purged_split(dataset)
    _early_stop, calibration = _purged_tuning_subsplit(
        dataset.decision_time_ms,
        splits["tuning"],
        purge_ms=split_evidence.purge_ms,
    )
    progress("train-direct-models")
    direct_models, direct_iterations, direct_identity = _train_direct_models(
        dataset,
        splits,
        risk_level=str(candidate["risk_level"]),
        compute_backend=compute_backend,
        seed=int(training["seed"]),
    )
    event_masks = {
        str(multiplier): _causal_cusum_events(
            dataset,
            volatility_multiplier=multiplier,
        )
        for multiplier in _CUSUM_MULTIPLIERS
    }
    event_training_mask = event_masks["1.0"]
    progress("train-event-models")
    (
        event_models,
        event_iterations,
        event_calibrations,
        event_identity,
    ) = _train_event_models(
        dataset,
        splits,
        event_mask=event_training_mask,
        risk_level=str(candidate["risk_level"]),
        compute_backend=compute_backend,
        seed=int(training["seed"]),
    )
    segment_indexes = {
        "calibration": calibration,
        "policy": splits["policy"],
        "selection": splits["selection"],
    }
    progress("reconstruct-scores")
    scores = {
        name: _segment_scores(
            dataset=dataset,
            indexes=indexes,
            artifact=artifact,
            direct_models=direct_models,
            direct_iterations=direct_iterations,
        )
        for name, indexes in segment_indexes.items()
    }
    for segment, indexes in segment_indexes.items():
        scores[segment].update(
            _event_model_scores(
                features=np.asarray(dataset.features[indexes], dtype=np.float32),
                models=event_models,
                iterations=event_iterations,
                calibrations=event_calibrations,
                risk_level=str(candidate["risk_level"]),
            )
        )
    for segment in segment_indexes:
        scores[segment]["rank_ensemble"] = {}
        for side in ("long", "short"):
            components = (
                "profitability_probability",
                "direct_mean",
                "upper_quantile",
                "lower_quantile",
            )
            reference = scores["calibration"]
            scores[segment]["rank_ensemble"][side] = np.mean(
                np.vstack(
                    [
                        _reference_percentile(
                            reference[method][side],
                            scores[segment][method][side],
                        )
                        for method in components
                    ]
                ),
                axis=0,
            )
        scores[segment]["event_rank_ensemble"] = {}
        for side in ("long", "short"):
            components = (
                "event_profitability_probability",
                "event_direct_mean",
                "event_upper_quantile",
                "event_lower_quantile",
            )
            reference_mask = (
                event_training_mask[calibration]
                & (
                    np.asarray(dataset.long_liquidity_eligible[calibration], dtype=bool)
                    if side == "long"
                    else np.asarray(
                        dataset.short_liquidity_eligible[calibration],
                        dtype=bool,
                    )
                )
            )
            scores[segment]["event_rank_ensemble"][side] = np.mean(
                np.vstack(
                    [
                        _reference_percentile(
                            scores["calibration"][method][side][reference_mask],
                            scores[segment][method][side],
                        )
                        for method in components
                    ]
                ),
                axis=0,
            )
    method_names = tuple(scores["policy"])
    event_dataset = replace(
        dataset,
        long_liquidity_eligible=(
            np.asarray(dataset.long_liquidity_eligible, dtype=bool)
            & event_training_mask
        ),
        short_liquidity_eligible=(
            np.asarray(dataset.short_liquidity_eligible, dtype=bool)
            & event_training_mask
        ),
    )
    outcomes: list[dict[str, object]] = []
    progress("evaluate-policies")
    for method in method_names:
        evaluation_dataset = event_dataset if method.startswith("event_") else dataset
        policy_scores = scores["policy"][method]
        policy = _select_score_threshold(
            dataset=evaluation_dataset,
            indexes=splits["policy"],
            long_scores=policy_scores["long"],
            short_scores=policy_scores["short"],
            risk_level=str(candidate["risk_level"]),
        )
        selection_scores = scores["selection"][method]
        if policy["accepted"]:
            selection_trace = _simulate_score_threshold(
                dataset=evaluation_dataset,
                indexes=splits["selection"],
                long_scores=selection_scores["long"],
                short_scores=selection_scores["short"],
                threshold=float(policy["threshold"]),
            )
        else:
            selection_trace = _simulate_score_threshold(
                dataset=evaluation_dataset,
                indexes=np.asarray(splits["selection"][:0], dtype=np.int64),
                long_scores=np.asarray([], dtype=np.float64),
                short_scores=np.asarray([], dtype=np.float64),
                threshold=np.inf,
            )
        confidence = _performance_confidence(
            selection_trace,
            dataset.decision_time_ms[splits["selection"]],
        )
        adaptive: dict[str, object] = {}
        for lookback in (1, 2, 4, None):
            adaptive_trace, daily = _walk_forward_score_policy(
                dataset=evaluation_dataset,
                policy_indexes=splits["policy"],
                selection_indexes=splits["selection"],
                policy_long_scores=policy_scores["long"],
                policy_short_scores=policy_scores["short"],
                selection_long_scores=selection_scores["long"],
                selection_short_scores=selection_scores["short"],
                risk_level=str(candidate["risk_level"]),
                lookback_days=lookback,
            )
            adaptive_confidence = _performance_confidence(
                adaptive_trace,
                dataset.decision_time_ms[splits["selection"]],
            )
            adaptive["all" if lookback is None else str(lookback)] = {
                "metrics": asdict(adaptive_trace.metrics),
                "confidence": asdict(adaptive_confidence),
                "daily": daily,
            }
        event_filters: dict[str, object] = {}
        if not method.startswith("event_"):
            for multiplier, event_mask in event_masks.items():
                filtered_dataset = replace(
                    dataset,
                    long_liquidity_eligible=(
                        np.asarray(dataset.long_liquidity_eligible, dtype=bool)
                        & event_mask
                    ),
                    short_liquidity_eligible=(
                        np.asarray(dataset.short_liquidity_eligible, dtype=bool)
                        & event_mask
                    ),
                )
                event_policy = _select_score_threshold(
                    dataset=filtered_dataset,
                    indexes=splits["policy"],
                    long_scores=policy_scores["long"],
                    short_scores=policy_scores["short"],
                    risk_level=str(candidate["risk_level"]),
                )
                if event_policy["accepted"]:
                    event_trace = _simulate_score_threshold(
                        dataset=filtered_dataset,
                        indexes=splits["selection"],
                        long_scores=selection_scores["long"],
                        short_scores=selection_scores["short"],
                        threshold=float(event_policy["threshold"]),
                    )
                else:
                    event_trace = _SimulationTrace(
                        metrics=_trading_metrics([], [], []),
                        pnls=(),
                        sides=(),
                        timestamps=(),
                    )
                event_filters[multiplier] = {
                    "policy_event_rows": int(
                        np.sum(event_mask[splits["policy"]])
                    ),
                    "selection_event_rows": int(
                        np.sum(event_mask[splits["selection"]])
                    ),
                    "policy": event_policy,
                    "selection_metrics": asdict(event_trace.metrics),
                    "selection_confidence": asdict(
                        _performance_confidence(
                            event_trace,
                            dataset.decision_time_ms[splits["selection"]],
                        )
                    ),
                }
        outcomes.append(
            {
                "method": method,
                "policy": policy,
                "selection_metrics": asdict(selection_trace.metrics),
                "selection_confidence": asdict(confidence),
                "policy_top_score_rows": _top_score_diagnostic(
                    dataset=evaluation_dataset,
                    indexes=splits["policy"],
                    long_scores=policy_scores["long"],
                    short_scores=policy_scores["short"],
                ),
                "selection_top_score_rows": _top_score_diagnostic(
                    dataset=evaluation_dataset,
                    indexes=splits["selection"],
                    long_scores=selection_scores["long"],
                    short_scores=selection_scores["short"],
                ),
                "policy_daily_top_score_rows": _daily_top_score_diagnostic(
                    dataset=evaluation_dataset,
                    indexes=splits["policy"],
                    long_scores=policy_scores["long"],
                    short_scores=policy_scores["short"],
                ),
                "selection_daily_top_score_rows": _daily_top_score_diagnostic(
                    dataset=evaluation_dataset,
                    indexes=splits["selection"],
                    long_scores=selection_scores["long"],
                    short_scores=selection_scores["short"],
                ),
                "adaptive_selection": adaptive,
                "cusum_event_filters": event_filters,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "action-value-score-ablation-v1",
        "artifact_class": "consumed_selection_research_no_trading_authority",
        "design_sha256": design["design_sha256"],
        "source_report_sha256": _file_sha256(report_path),
        "source_artifact_sha256": _file_sha256(artifact_path),
        "corpus_certificate_sha256": certificate["certificate_sha256"],
        "candidate": dict(candidate),
        "dataset_rows": dataset.rows,
        "path_target_evidence": asdict(path_evidence),
        "split": asdict(split_evidence),
        "direct_models": {
            **direct_identity,
            "best_iterations": direct_iterations,
        },
        "event_models": {
            **event_identity,
            "best_iterations": event_iterations,
            "probability_calibration": {
                side: list(values)
                for side, values in event_calibrations.items()
            },
            "cusum_volatility_multiplier": 1.0,
            "cusum_minimum_threshold_bps": 1.0,
        },
        "selection_is_consumed": True,
        "terminal_holdout_accessed": False,
        "trading_authority": False,
        "profitability_claim": False,
        "outcomes": outcomes,
    }
    payload["ablation_sha256"] = _canonical_sha256(payload)
    write_json_atomic(output_path, payload, indent=2, sort_keys=True)
    if summary_output_path is not None:
        summary = summarize_ablation(
            payload,
            source_file_sha256=_file_sha256(output_path),
        )
        write_json_atomic(summary_output_path, summary, indent=2, sort_keys=True)
    progress("complete")
    return payload


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare action-value scores on consumed, non-terminal evidence",
    )
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compute-backend", default="directml")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    payload = run_ablation(
        design_path=args.design,
        evidence_root=args.evidence_root,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        candidate_id=args.candidate,
        output_path=args.output,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
        summary_output_path=args.summary_output,
    )
    print(
        "ablate-action-value-scores: "
        f"candidate={args.candidate} methods={len(payload['outcomes'])} "
        f"sha256={payload['ablation_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

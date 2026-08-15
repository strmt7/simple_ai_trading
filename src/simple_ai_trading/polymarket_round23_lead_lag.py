"""Frozen Round 23 exploratory Binance-to-Polymarket lead-lag diagnostic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .polymarket_round22_feature_store import Round22FeatureStore
from .polymarket_round22_features import (
    POLYMARKET_ROUND22_FEATURE_NAMES,
    POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
)
from .polymarket_round22_pilot import Round22PilotStore
from .polymarket_round23_binance import (
    POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256,
    audit_round23_binance_archives,
)


POLYMARKET_ROUND23_LEAD_LAG_SPEC_RELATIVE = (
    "docs/model-research/polymarket/round-023-lead-lag-model-spec-v1.json"
)
POLYMARKET_ROUND23_LEAD_LAG_SPEC_SHA256 = (
    "03e4fae64f34f835cac338dee111701c73419929e97a22939e12037eba8dfec9"
)
POLYMARKET_ROUND23_LEAD_LAG_RESULT_SCHEMA_VERSION = (
    "polymarket-round23-lead-lag-result-v1"
)
_MAXIMUM_ARTIFACT_BYTES = 2 * 1024 * 1024
_INGESTION_RESULT_RELATIVE = (
    "docs/model-research/polymarket/round-023-binance-ingestion-result-v1.json"
)
_INGESTION_RESULT_SHA256 = (
    "477d9982d2348e7cbb5eb3c66283597ebdf0687467bb34c4e8acae97e5a90276"
)
_QUALIFICATION_RELATIVE = (
    "docs/model-research/polymarket/round-023-source-qualification-v2-2026-08-03.json"
)
_QUALIFICATION_SHA256 = (
    "3cf328dfee5d653690eba79ec0b9c7de818006ea6391506a0033d05ec9aa240d"
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 23 lead-lag JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 23 lead-lag JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_ARTIFACT_BYTES
    ):
        raise ValueError(f"Round 23 {name} is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 23 {name} is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Round 23 {name} is not an object")
    return dict(decoded)


def load_round23_lead_lag_spec(repository: str | Path) -> dict[str, object]:
    root = Path(repository).resolve()
    spec = _load_object(
        root / POLYMARKET_ROUND23_LEAD_LAG_SPEC_RELATIVE,
        name="lead-lag model specification",
    )
    claimed = str(spec.pop("specification_sha256", "")).strip().lower()
    parents = spec.get("parents")
    authority = spec.get("authority")
    if (
        claimed != POLYMARKET_ROUND23_LEAD_LAG_SPEC_SHA256
        or claimed != _canonical_sha256(spec)
        or spec.get("schema_version") != "polymarket-round23-lead-lag-model-spec-v1"
        or spec.get("status")
        != "frozen_before_binance_polymarket_relation_future_target_construction_or_metric_computation"
        or not isinstance(parents, Mapping)
        or parents.get("binance_source_contract_sha256")
        != POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256
        or parents.get("binance_ingestion_result_sha256") != _INGESTION_RESULT_SHA256
        or parents.get("binance_source_qualification_sha256") != _QUALIFICATION_SHA256
        or parents.get("round22_feature_names_sha256")
        != POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
        or parents.get("round22_feature_policy_sha256")
        != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise ValueError("Round 23 lead-lag model specification differs")
    for relative, expected, name in (
        (_INGESTION_RESULT_RELATIVE, _INGESTION_RESULT_SHA256, "ingestion result"),
        (_QUALIFICATION_RELATIVE, _QUALIFICATION_SHA256, "qualification result"),
    ):
        if _canonical_sha256(_load_object(root / relative, name=name)) != expected:
            raise ValueError(f"Round 23 {name} differs")
    return {**spec, "specification_sha256": claimed}


@dataclass(frozen=True, slots=True)
class _BinanceBar:
    close: float
    quote_volume: float
    signed_quote_volume: float


@dataclass(frozen=True, slots=True)
class _Partition:
    role: str
    baseline: NDArray[np.float64]
    candidate: NDArray[np.float64]
    target: NDArray[np.float64]
    conditions: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class _RidgeModel:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    penalty: float

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        standardized = np.clip((features - self.mean) / self.scale, -5.0, 5.0)
        design = np.column_stack((np.ones(features.shape[0]), standardized))
        return np.asarray(design @ self.coefficients, dtype=np.float64)


def _logit(probability: float, *, lower: float, upper: float) -> float:
    selected = min(upper, max(lower, float(probability)))
    return math.log(selected / (1.0 - selected))


def _condition_weights(conditions: NDArray[np.str_]) -> NDArray[np.float64]:
    counts = Counter(str(value) for value in conditions)
    if not counts:
        raise ValueError("Round 23 lead-lag partition is empty")
    return np.asarray(
        [1.0 / (len(counts) * counts[str(value)]) for value in conditions],
        dtype=np.float64,
    )


def _fit_ridge(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    conditions: NDArray[np.str_],
    *,
    penalty: float,
) -> _RidgeModel:
    weights = _condition_weights(conditions)
    mean = np.sum(features * weights[:, None], axis=0)
    variance = np.sum(((features - mean) ** 2) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    standardized = np.clip((features - mean) / scale, -5.0, 5.0)
    design = np.column_stack((np.ones(features.shape[0]), standardized))
    weighted_design = design * weights[:, None]
    system = design.T @ weighted_design
    system[1:, 1:] += np.eye(features.shape[1]) * penalty
    coefficients = np.linalg.solve(system, design.T @ (weights * target))
    return _RidgeModel(
        mean=mean,
        scale=scale,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        penalty=penalty,
    )


def _condition_metric(
    values: NDArray[np.float64],
    conditions: NDArray[np.str_],
) -> float:
    return float(
        np.mean(
            [
                float(np.mean(values[conditions == condition]))
                for condition in np.unique(conditions)
            ]
        )
    )


def _loco_scores(
    partition: _Partition,
    features: NDArray[np.float64],
    penalties: Sequence[float],
) -> dict[float, float]:
    output: dict[float, float] = {}
    for penalty in penalties:
        condition_errors: list[float] = []
        for condition in np.unique(partition.conditions):
            held_out = partition.conditions == condition
            model = _fit_ridge(
                features[~held_out],
                partition.target[~held_out],
                partition.conditions[~held_out],
                penalty=penalty,
            )
            error = partition.target[held_out] - model.predict(features[held_out])
            condition_errors.append(float(np.mean(error**2)))
        output[float(penalty)] = float(np.mean(condition_errors))
    return output


def _select_penalty(scores: Mapping[float, float]) -> float:
    return min(scores, key=lambda penalty: (scores[penalty], -penalty))


def _select_scale(
    partition: _Partition,
    predictions: NDArray[np.float64],
    scales: Sequence[float],
) -> tuple[float, dict[float, float]]:
    scores = {
        float(scale): _condition_metric(
            (partition.target - predictions * scale) ** 2,
            partition.conditions,
        )
        for scale in scales
    }
    return min(scores, key=lambda scale: (scores[scale], scale)), scores


def _flow_share(bars: Sequence[_BinanceBar]) -> float:
    denominator = sum(item.quote_volume for item in bars)
    if denominator <= 0.0:
        return 0.0
    return float(sum(item.signed_quote_volume for item in bars) / denominator)


def _binance_feature_vector(
    spot: Sequence[_BinanceBar],
    futures: Sequence[_BinanceBar],
) -> tuple[float, ...]:
    if len(spot) != 16 or len(futures) != 16:
        raise ValueError("Round 23 Binance feature history differs")

    def returns(values: Sequence[_BinanceBar]) -> tuple[float, float, float]:
        return tuple(
            10_000.0 * math.log(values[-1].close / values[-1 - horizon].close)
            for horizon in (1, 5, 15)
        )

    spot_returns = returns(spot)
    futures_returns = returns(futures)
    basis = 10_000.0 * math.log(futures[-1].close / spot[-1].close)
    prior_basis = 10_000.0 * math.log(futures[-6].close / spot[-6].close)
    output = (
        *spot_returns,
        *futures_returns,
        _flow_share(spot[-1:]),
        _flow_share(spot[-5:]),
        _flow_share(spot[-15:]),
        _flow_share(futures[-1:]),
        _flow_share(futures[-5:]),
        _flow_share(futures[-15:]),
        basis,
        basis - prior_basis,
    )
    if len(output) != 14 or any(not math.isfinite(value) for value in output):
        raise ValueError("Round 23 Binance feature vector differs")
    return output


def _load_binance_bars(
    store: Round22PilotStore,
) -> dict[tuple[str, str, int], _BinanceBar]:
    audits = audit_round23_binance_archives(store)
    if len(audits) != 6 or any(item.downloaded for item in audits):
        raise ValueError("Round 23 Binance archive audit differs")
    rows = store.connection.execute(
        """
        SELECT source, archive_date, timestamp_ms, close_price,
               quote_volume, signed_quote_volume
        FROM exploratory.round23_binance_second
        ORDER BY source, archive_date, timestamp_ms
        """
    ).fetchall()
    if len(rows) != 21_600:
        raise ValueError("Round 23 Binance retained population differs")
    output: dict[tuple[str, str, int], _BinanceBar] = {}
    for row in rows:
        identity = (str(row[0]), str(row[1]), int(row[2]))
        if identity in output:
            raise ValueError("Round 23 Binance retained identity is duplicated")
        output[identity] = _BinanceBar(
            close=float(row[3]),
            quote_volume=float(row[4]),
            signed_quote_volume=float(row[5]),
        )
    return output


def _dataset(
    store: Round22PilotStore,
    spec: Mapping[str, object],
) -> tuple[dict[str, _Partition], dict[str, object]]:
    name_indices = {
        name: index for index, name in enumerate(POLYMARKET_ROUND22_FEATURE_NAMES)
    }
    model_spec = spec["model"]
    data_selection = spec["data_selection"]
    assert isinstance(model_spec, Mapping)
    assert isinstance(data_selection, Mapping)
    baseline_names = tuple(str(name) for name in model_spec["baseline_features"])
    ordinary_names = tuple(
        name for name in baseline_names if name != "derived.market_prior_logit"
    )
    if (
        baseline_names[0] != "derived.market_prior_logit"
        or len(baseline_names) != 16
        or any(name not in name_indices for name in ordinary_names)
    ):
        raise ValueError("Round 23 baseline feature contract differs")
    lower, upper = (float(value) for value in data_selection["probability_clip"])
    bars = _load_binance_bars(store)
    feature_store = Round22FeatureStore(store)
    markets = store.connection.execute(
        """
        SELECT condition_id, role, event_start_ms, event_end_ms
        FROM feature.market_identity
        WHERE role IN ('train', 'tune_calibration', 'tune_selection')
        ORDER BY event_start_ms, condition_id
        """
    ).fetchall()
    role_counts = Counter(str(row[1]) for row in markets)
    if len(markets) != 36 or role_counts != {
        "train": 12,
        "tune_calibration": 12,
        "tune_selection": 12,
    }:
        raise ValueError("Round 23 Polymarket cohort differs")
    grouped: dict[str, dict[str, list[object]]] = {}
    chain = "0" * 64
    for condition_id_raw, role_raw, start_raw, end_raw in markets:
        condition_id = str(condition_id_raw)
        role = str(role_raw)
        start_ms = int(start_raw)
        end_ms = int(end_raw)
        date = datetime.fromtimestamp(start_ms / 1_000, tz=UTC).date().isoformat()
        store.audit_condition(condition_id)
        rows = feature_store.load_condition_rows(condition_id)
        by_time = {row.decision_time_ms: row for row in rows if row.tabular_anchor}
        bucket = grouped.setdefault(
            role,
            {"baseline": [], "candidate": [], "conditions": [], "target": []},
        )
        for current in rows:
            if (
                not current.tabular_anchor
                or not current.available
                or not current.tabular_history_complete
            ):
                continue
            future = by_time.get(current.decision_time_ms + 1_000)
            if (
                future is None
                or not future.available
                or future.decision_time_ms >= end_ms
            ):
                continue
            latest_bar_ms = current.decision_time_ms - 1_000
            history_ms = tuple(
                latest_bar_ms - offset * 1_000 for offset in range(15, -1, -1)
            )
            try:
                spot = tuple(
                    bars[("spot_1s", date, timestamp)] for timestamp in history_ms
                )
                futures = tuple(
                    bars[("futures_aggTrades", date, timestamp)]
                    for timestamp in history_ms
                )
            except KeyError:
                continue
            prior = float(current.values[name_indices["market_prior_up"]])
            future_prior = float(future.values[name_indices["market_prior_up"]])
            baseline = (
                _logit(prior, lower=lower, upper=upper),
                *(float(current.values[name_indices[name]]) for name in ordinary_names),
            )
            candidate_additions = _binance_feature_vector(spot, futures)
            target = _logit(future_prior, lower=lower, upper=upper) - baseline[0]
            body = {
                "baseline": baseline,
                "binance": candidate_additions,
                "condition_id": condition_id,
                "decision_time_ms": current.decision_time_ms,
                "role": role,
                "target": target,
            }
            chain = hashlib.sha256(
                bytes.fromhex(chain) + bytes.fromhex(_canonical_sha256(body))
            ).hexdigest()
            bucket["baseline"].append(baseline)
            bucket["candidate"].append((*baseline, *candidate_additions))
            bucket["conditions"].append(condition_id)
            bucket["target"].append(target)
    partitions: dict[str, _Partition] = {}
    counts: dict[str, object] = {}
    for role in ("train", "tune_calibration", "tune_selection"):
        bucket = grouped.get(role)
        if bucket is None:
            raise ValueError("Round 23 lead-lag partition is unavailable")
        partition = _Partition(
            role=role,
            baseline=np.asarray(bucket["baseline"], dtype=np.float64),
            candidate=np.asarray(bucket["candidate"], dtype=np.float64),
            target=np.asarray(bucket["target"], dtype=np.float64),
            conditions=np.asarray(bucket["conditions"], dtype=np.str_),
        )
        if (
            partition.baseline.ndim != 2
            or partition.baseline.shape[1] != 16
            or partition.candidate.shape[1] != 30
            or partition.target.shape[0] != partition.baseline.shape[0]
            or set(np.unique(partition.conditions)) == set()
            or not np.all(np.isfinite(partition.candidate))
            or not np.all(np.isfinite(partition.target))
        ):
            raise ValueError("Round 23 lead-lag partition differs")
        partitions[role] = partition
        counts[role] = {
            "condition_count": int(np.unique(partition.conditions).size),
            "row_count": int(partition.target.size),
            "target_changed_row_count": int(np.sum(np.abs(partition.target) >= 1e-6)),
        }
    return partitions, {
        "dataset_chain_sha256": chain,
        "partitions": counts,
        "target_source": "future_polymarket_market_prior_only",
        "official_resolution_accessed": False,
    }


def _metrics(
    partition: _Partition,
    predictions: NDArray[np.float64],
    *,
    changed_minimum: float,
) -> dict[str, float]:
    errors = partition.target - predictions
    weights = _condition_weights(partition.conditions)
    target_mean = float(np.sum(partition.target * weights))
    prediction_mean = float(np.sum(predictions * weights))
    covariance = float(
        np.sum(
            weights * (partition.target - target_mean) * (predictions - prediction_mean)
        )
    )
    target_variance = float(np.sum(weights * (partition.target - target_mean) ** 2))
    prediction_variance = float(np.sum(weights * (predictions - prediction_mean) ** 2))
    correlation = (
        covariance / math.sqrt(target_variance * prediction_variance)
        if target_variance > 0.0 and prediction_variance > 0.0
        else 0.0
    )
    changed = np.abs(partition.target) >= changed_minimum
    changed_conditions = np.unique(partition.conditions[changed])
    sign_accuracy = (
        float(
            np.mean(
                [
                    np.mean(
                        np.sign(
                            predictions[changed & (partition.conditions == condition)]
                        )
                        == np.sign(
                            partition.target[
                                changed & (partition.conditions == condition)
                            ]
                        )
                    )
                    for condition in changed_conditions
                ]
            )
        )
        if changed_conditions.size
        else 0.0
    )
    return {
        "changed_row_direction_accuracy": sign_accuracy,
        "condition_equal_mae": _condition_metric(np.abs(errors), partition.conditions),
        "condition_equal_mse": _condition_metric(errors**2, partition.conditions),
        "pearson_correlation": correlation,
    }


def _rounded(value: float) -> float:
    selected = round(float(value), 12)
    if not math.isfinite(selected):
        raise ValueError("Round 23 result contains a non-finite metric")
    return selected


def run_round23_lead_lag_diagnostic(
    store: Round22PilotStore,
) -> dict[str, object]:
    if not store.read_only:
        raise ValueError("Round 23 lead-lag diagnostic requires a read-only store")
    spec = load_round23_lead_lag_spec(store.contract.repository)
    partitions, dataset = _dataset(store, spec)
    model_spec = spec["model"]
    evaluation = spec["evaluation"]
    assert isinstance(model_spec, Mapping)
    assert isinstance(evaluation, Mapping)
    penalties = tuple(float(value) for value in model_spec["l2_penalty_grid"])
    train = partitions["train"]
    calibration = partitions["tune_calibration"]
    selection = partitions["tune_selection"]
    baseline_loco = _loco_scores(train, train.baseline, penalties)
    candidate_loco = _loco_scores(train, train.candidate, penalties)
    baseline_penalty = _select_penalty(baseline_loco)
    candidate_penalty = _select_penalty(candidate_loco)
    baseline_model = _fit_ridge(
        train.baseline,
        train.target,
        train.conditions,
        penalty=baseline_penalty,
    )
    candidate_model = _fit_ridge(
        train.candidate,
        train.target,
        train.conditions,
        penalty=candidate_penalty,
    )
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    baseline_scale, baseline_scale_scores = _select_scale(
        calibration,
        baseline_model.predict(calibration.baseline),
        scales,
    )
    candidate_scale, candidate_scale_scores = _select_scale(
        calibration,
        candidate_model.predict(calibration.candidate),
        scales,
    )
    changed_minimum = float(evaluation["changed_target_absolute_logit_minimum"])
    calibration_baseline = baseline_model.predict(calibration.baseline) * baseline_scale
    calibration_candidate = (
        candidate_model.predict(calibration.candidate) * candidate_scale
    )
    selection_baseline = baseline_model.predict(selection.baseline) * baseline_scale
    selection_candidate = candidate_model.predict(selection.candidate) * candidate_scale
    calibration_metrics = {
        "baseline": _metrics(
            calibration,
            calibration_baseline,
            changed_minimum=changed_minimum,
        ),
        "candidate": _metrics(
            calibration,
            calibration_candidate,
            changed_minimum=changed_minimum,
        ),
    }
    selection_metrics = {
        "baseline": _metrics(
            selection,
            selection_baseline,
            changed_minimum=changed_minimum,
        ),
        "candidate": _metrics(
            selection,
            selection_candidate,
            changed_minimum=changed_minimum,
        ),
    }
    unique_conditions = np.unique(selection.conditions)
    condition_improvements = np.asarray(
        [
            np.mean(
                (
                    selection.target[selection.conditions == condition]
                    - selection_baseline[selection.conditions == condition]
                )
                ** 2
            )
            - np.mean(
                (
                    selection.target[selection.conditions == condition]
                    - selection_candidate[selection.conditions == condition]
                )
                ** 2
            )
            for condition in unique_conditions
        ],
        dtype=np.float64,
    )
    draws = int(evaluation["bootstrap_draws"])
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    bootstrap = np.mean(
        condition_improvements[
            rng.integers(
                0,
                condition_improvements.size,
                size=(draws, condition_improvements.size),
            )
        ],
        axis=1,
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    improvement_probability = float(np.mean(bootstrap > 0.0))
    calibration_improvement = (
        calibration_metrics["baseline"]["condition_equal_mse"]
        - calibration_metrics["candidate"]["condition_equal_mse"]
    )
    selection_improvement = (
        selection_metrics["baseline"]["condition_equal_mse"]
        - selection_metrics["candidate"]["condition_equal_mse"]
    )
    sign_improvement = (
        selection_metrics["candidate"]["changed_row_direction_accuracy"]
        - selection_metrics["baseline"]["changed_row_direction_accuracy"]
    )
    positive_fraction = float(np.mean(condition_improvements > 0.0))
    gate = evaluation["mechanism_gate"]
    assert isinstance(gate, Mapping)
    passed = bool(
        calibration_improvement > 0.0
        and selection_improvement > 0.0
        and lower > 0.0
        and improvement_probability
        >= float(gate["selection_improvement_probability_minimum"])
        and selection_metrics["candidate"]["changed_row_direction_accuracy"]
        >= float(gate["candidate_changed_row_sign_accuracy_minimum"])
        and sign_improvement
        >= float(gate["candidate_sign_accuracy_improvement_minimum"])
        and positive_fraction >= float(gate["positive_condition_fraction_minimum"])
    )
    candidate_names = tuple(str(value) for value in spec["candidate_features"])
    standardized_coefficients = candidate_model.coefficients[-len(candidate_names) :]
    result: dict[str, object] = {
        "authority": {
            "ai_edge_claim": False,
            "economic_backtest": False,
            "live_trading": False,
            "model_promotion": False,
            "paper_trading": False,
            "profitability_claim": False,
        },
        "bootstrap": {
            "improvement_probability": _rounded(improvement_probability),
            "mean_mse_improvement": _rounded(float(np.mean(bootstrap))),
            "mse_improvement_95_interval": [_rounded(lower), _rounded(upper)],
        },
        "calibration": {
            "baseline_metrics": {
                key: _rounded(value)
                for key, value in calibration_metrics["baseline"].items()
            },
            "baseline_scale": baseline_scale,
            "baseline_scale_mse": {
                str(key): _rounded(value)
                for key, value in baseline_scale_scores.items()
            },
            "candidate_metrics": {
                key: _rounded(value)
                for key, value in calibration_metrics["candidate"].items()
            },
            "candidate_scale": candidate_scale,
            "candidate_scale_mse": {
                str(key): _rounded(value)
                for key, value in candidate_scale_scores.items()
            },
            "mse_improvement": _rounded(calibration_improvement),
        },
        "conclusion": (
            "exploratory_lead_lag_mechanism_passed_requires_fresh_receipt_time_holdout"
            if passed
            else "exploratory_lead_lag_mechanism_falsified"
        ),
        "dataset": dataset,
        "mechanism_gate_passed": passed,
        "model": {
            "baseline_loco_mse": {
                str(key): _rounded(value) for key, value in baseline_loco.items()
            },
            "baseline_selected_penalty": baseline_penalty,
            "candidate_binance_standardized_coefficients": {
                name: _rounded(value)
                for name, value in zip(
                    candidate_names,
                    standardized_coefficients,
                    strict=True,
                )
            },
            "candidate_loco_mse": {
                str(key): _rounded(value) for key, value in candidate_loco.items()
            },
            "candidate_selected_penalty": candidate_penalty,
        },
        "parents": {
            "model_specification_sha256": spec["specification_sha256"],
            "source_contract_sha256": POLYMARKET_ROUND23_BINANCE_SOURCE_SHA256,
        },
        "schema_version": POLYMARKET_ROUND23_LEAD_LAG_RESULT_SCHEMA_VERSION,
        "selection": {
            "baseline_metrics": {
                key: _rounded(value)
                for key, value in selection_metrics["baseline"].items()
            },
            "candidate_metrics": {
                key: _rounded(value)
                for key, value in selection_metrics["candidate"].items()
            },
            "changed_row_sign_accuracy_improvement": _rounded(sign_improvement),
            "condition_mse_improvements": [
                {
                    "condition_id": str(condition),
                    "mse_improvement": _rounded(value),
                }
                for condition, value in zip(
                    unique_conditions,
                    condition_improvements,
                    strict=True,
                )
            ],
            "mse_improvement": _rounded(selection_improvement),
            "positive_condition_fraction": _rounded(positive_fraction),
            "status": "exploratory_previously_consumed_partition",
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "POLYMARKET_ROUND23_LEAD_LAG_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND23_LEAD_LAG_SPEC_RELATIVE",
    "POLYMARKET_ROUND23_LEAD_LAG_SPEC_SHA256",
    "load_round23_lead_lag_spec",
    "run_round23_lead_lag_diagnostic",
]

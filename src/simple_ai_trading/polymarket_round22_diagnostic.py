"""Deterministic, condition-clustered Round 22 diagnostic model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from .polymarket_round22_feature_store import Round22FeatureStore
from .polymarket_round22_features import POLYMARKET_ROUND22_FEATURE_NAMES
from .polymarket_round22_pilot import Round22PilotStore
from . import polymarket_round22_targets as target_gate


POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_RELATIVE = (
    "docs/model-research/polymarket/round-022-diagnostic-model-spec-v1.json"
)
POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_SHA256 = (
    "0d457942b0d84ecc703d882f66a1a560234f65c4a99e97a0841cd50faa152107"
)
POLYMARKET_ROUND22_DIAGNOSTIC_RESULT_SCHEMA_VERSION = (
    "polymarket-round22-diagnostic-result-v1"
)
_MAXIMUM_SPEC_BYTES = 1 * 1024 * 1024
_EPSILON = 0.005


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 22 diagnostic JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 diagnostic JSON contains {value}")


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


def load_round22_diagnostic_model_spec(repository: str | Path) -> dict[str, object]:
    path = (
        Path(repository).resolve() / POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_RELATIVE
    )
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_SPEC_BYTES
    ):
        raise ValueError("Round 22 diagnostic model specification is unavailable")
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 22 diagnostic model specification is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 22 diagnostic model specification is not an object")
    spec = dict(decoded)
    claimed = str(spec.pop("specification_sha256", "")).strip().lower()
    data_selection = spec.get("data_selection")
    model = spec.get("model")
    if (
        claimed != POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_SHA256
        or claimed != _canonical_sha256(spec)
        or spec.get("schema_version") != "polymarket-round22-diagnostic-model-spec-v1"
        or spec.get("status")
        != "frozen_after_aggregate_target_count_before_condition_label_linkage"
        or not isinstance(data_selection, Mapping)
        or not isinstance(model, Mapping)
        or data_selection.get("sealed_test_access") is not False
        or model.get("candidate_count") != 1
    ):
        raise ValueError("Round 22 diagnostic model specification differs")
    return {**spec, "specification_sha256": claimed}


@dataclass(frozen=True, slots=True)
class _Partition:
    role: str
    features: NDArray[np.float64]
    priors: NDArray[np.float64]
    labels: NDArray[np.float64]
    condition_ids: NDArray[np.object_]
    elapsed_seconds: NDArray[np.float64]

    @property
    def condition_count(self) -> int:
        return len(set(str(value) for value in self.condition_ids))


@dataclass(frozen=True, slots=True)
class _Standardizer:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip((values - self.mean) / self.scale, -5.0, 5.0)


def _condition_weights(condition_ids: NDArray[np.object_]) -> NDArray[np.float64]:
    selected = [str(value) for value in condition_ids]
    unique = sorted(set(selected))
    if not unique:
        raise ValueError("Round 22 diagnostic partition is empty")
    counts = {condition_id: selected.count(condition_id) for condition_id in unique}
    return np.asarray(
        [1.0 / (len(unique) * counts[value]) for value in selected],
        dtype=np.float64,
    )


def _fit_standardizer(partition: _Partition) -> _Standardizer:
    weights = _condition_weights(partition.condition_ids)
    mean = np.sum(partition.features * weights[:, None], axis=0)
    variance = np.sum(
        np.square(partition.features - mean) * weights[:, None],
        axis=0,
    )
    scale = np.sqrt(np.maximum(variance, 1e-16))
    scale[scale < 1e-8] = 1.0
    return _Standardizer(mean=mean, scale=scale)


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _fit_residual(
    partition: _Partition,
    *,
    standardizer: _Standardizer,
    penalty: float,
) -> NDArray[np.float64]:
    features = standardizer.transform(partition.features)
    offset = _logit(partition.priors)
    weights = _condition_weights(partition.condition_ids)

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = offset + theta[0] + features @ theta[1:]
        probabilities = expit(logits)
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, logits) - partition.labels * logits))
            + 0.5 * penalty * np.dot(theta[1:], theta[1:])
        )
        error = weights * (probabilities - partition.labels)
        gradient = np.concatenate(
            (
                np.asarray([np.sum(error)], dtype=np.float64),
                features.T @ error + penalty * theta[1:],
            )
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(partition.features.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2_000},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Round 22 residual fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _subset(partition: _Partition, mask: NDArray[np.bool_]) -> _Partition:
    return _Partition(
        role=partition.role,
        features=partition.features[mask],
        priors=partition.priors[mask],
        labels=partition.labels[mask],
        condition_ids=partition.condition_ids[mask],
        elapsed_seconds=partition.elapsed_seconds[mask],
    )


def _raw_probabilities(
    partition: _Partition,
    *,
    standardizer: _Standardizer,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    residual = theta[0] + standardizer.transform(partition.features) @ theta[1:]
    return expit(_logit(partition.priors) + residual), residual


def _weighted_log_loss(
    partition: _Partition,
    probabilities: NDArray[np.float64],
) -> float:
    weights = _condition_weights(partition.condition_ids)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.sum(
            weights
            * (
                partition.labels * np.log(clipped)
                + (1.0 - partition.labels) * np.log1p(-clipped)
            )
        )
    )


def _select_penalty(
    partition: _Partition,
    penalties: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    conditions = sorted(set(str(value) for value in partition.condition_ids))
    scores: dict[str, float] = {}
    for penalty in penalties:
        predictions = np.zeros_like(partition.priors)
        for held_out in conditions:
            holdout = partition.condition_ids == held_out
            fit_partition = _subset(partition, ~holdout)
            standardizer = _fit_standardizer(fit_partition)
            theta = _fit_residual(
                fit_partition,
                standardizer=standardizer,
                penalty=penalty,
            )
            predictions[holdout] = _raw_probabilities(
                _subset(partition, holdout),
                standardizer=standardizer,
                theta=theta,
            )[0]
        scores[str(penalty)] = _weighted_log_loss(partition, predictions)
    selected = min(
        sorted(penalties, reverse=True),
        key=lambda value: round(scores[str(value)], 12),
    )
    return selected, scores


def _fit_calibration(
    partition: _Partition,
    residual: NDArray[np.float64],
) -> tuple[float, float]:
    weights = _condition_weights(partition.condition_ids)
    offset = _logit(partition.priors)

    def objective(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = offset + values[0] + values[1] * residual
        probabilities = expit(logits)
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, logits) - partition.labels * logits))
        )
        error = weights * (probabilities - partition.labels)
        gradient = np.asarray(
            [np.sum(error), np.sum(error * residual)],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-2.0, 2.0), (0.0, 1.0)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1_000},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Round 22 calibration fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _calibration_parameters(
    partition: _Partition,
    probabilities: NDArray[np.float64],
) -> tuple[float, float]:
    predictor = _logit(probabilities)
    weights = _condition_weights(partition.condition_ids)

    def objective(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = values[0] + values[1] * predictor
        fitted = expit(logits)
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, logits) - partition.labels * logits))
        )
        error = weights * (fitted - partition.labels)
        return loss, np.asarray(
            [np.sum(error), np.sum(error * predictor)],
            dtype=np.float64,
        )

    result = minimize(
        objective,
        np.asarray([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1_000},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Round 22 calibration diagnostic failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _condition_metric_deltas(
    partition: _Partition,
    model_probabilities: NDArray[np.float64],
    *,
    metric: str,
) -> NDArray[np.float64]:
    output: list[float] = []
    for condition_id in sorted(set(str(value) for value in partition.condition_ids)):
        mask = partition.condition_ids == condition_id
        labels = partition.labels[mask]
        baseline = np.clip(partition.priors[mask], 1e-12, 1.0 - 1e-12)
        model = np.clip(model_probabilities[mask], 1e-12, 1.0 - 1e-12)
        if metric == "brier":
            baseline_loss = np.mean(np.square(baseline - labels))
            model_loss = np.mean(np.square(model - labels))
        else:
            baseline_loss = -np.mean(
                labels * np.log(baseline) + (1.0 - labels) * np.log1p(-baseline)
            )
            model_loss = -np.mean(
                labels * np.log(model) + (1.0 - labels) * np.log1p(-model)
            )
        output.append(float(baseline_loss - model_loss))
    return np.asarray(output, dtype=np.float64)


def _bootstrap_report(
    deltas: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float]:
    sample_indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    means = np.mean(deltas[sample_indices], axis=1)
    return {
        "improvement_mean": float(np.mean(deltas)),
        "improvement_probability": float(np.mean(means > 0.0)),
        "improvement_ci_2_5": float(np.quantile(means, 0.025)),
        "improvement_ci_97_5": float(np.quantile(means, 0.975)),
    }


def _metric_report(
    partition: _Partition,
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    weights = _condition_weights(partition.condition_ids)
    baseline = np.clip(partition.priors, 1e-12, 1.0 - 1e-12)
    model = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    labels = partition.labels
    calibration_intercept, calibration_slope = _calibration_parameters(
        partition,
        model,
    )
    return {
        "baseline": {
            "brier_score": float(np.sum(weights * np.square(baseline - labels))),
            "log_loss": _weighted_log_loss(partition, baseline),
            "threshold_accuracy": float(
                np.sum(weights * ((baseline >= 0.5) == labels))
            ),
        },
        "condition_count": partition.condition_count,
        "model": {
            "brier_score": float(np.sum(weights * np.square(model - labels))),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "log_loss": _weighted_log_loss(partition, model),
            "threshold_accuracy": float(np.sum(weights * ((model >= 0.5) == labels))),
        },
        "row_count": len(labels),
    }


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, tuple):
        return [_rounded(item) for item in value]
    if isinstance(value, (np.floating, float)):
        selected = float(value)
        if not math.isfinite(selected):
            raise ValueError("Round 22 diagnostic result is non-finite")
        return round(selected, 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _load_partitions(
    store: Round22PilotStore,
    spec: Mapping[str, object],
) -> tuple[dict[str, _Partition], dict[str, dict[str, int]]]:
    preregistration = target_gate.load_round22_diagnostic_preregistration(
        store.contract.repository
    )
    claim = target_gate._claim(store)  # noqa: SLF001
    parents = spec["parents"]
    assert isinstance(parents, Mapping)
    if (
        str(claim[8]) != "complete"
        or str(claim[9]) != parents["target_access_claim_sha256"]
        or str(claim[6]) != parents["target_implementation_manifest_sha256"]
        or preregistration["preregistration_sha256"]
        != parents["diagnostic_preregistration_sha256"]
    ):
        raise ValueError("Round 22 diagnostic target claim differs")
    data_selection = spec["data_selection"]
    assert isinstance(data_selection, Mapping)
    selected_names = tuple(str(name) for name in data_selection["residual_features"])
    name_indices = {
        name: index for index, name in enumerate(POLYMARKET_ROUND22_FEATURE_NAMES)
    }
    if len(selected_names) != len(set(selected_names)) or any(
        name not in name_indices for name in selected_names
    ):
        raise ValueError("Round 22 diagnostic residual feature names differ")
    selected_indices = tuple(name_indices[name] for name in selected_names)
    prior_index = name_indices["market_prior_up"]
    feature_store = Round22FeatureStore(store)
    grouped: dict[str, dict[str, list[object]]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    population = preregistration["population"]
    assert isinstance(population, Mapping)
    conditions = population["conditions"]
    assert isinstance(conditions, list)
    claim_sha = str(claim[9])
    for item in conditions:
        assert isinstance(item, Mapping)
        condition_id = str(item["condition_id"])
        role = str(item["role"])
        outcome = target_gate._audit_target(  # noqa: SLF001
            store,
            condition_id=condition_id,
            claim_sha256=claim_sha,
        )
        audit, rows = feature_store.audit_condition_rows(condition_id)
        if (
            audit["manifest_sha256"] != item["feature_manifest_sha256"]
            or audit["target_row_count"] != 36
        ):
            raise ValueError("Round 22 diagnostic audited feature differs")
        market = store.market(condition_id)
        selected_rows = tuple(
            row for row in rows if row.available and row.tabular_history_complete
        )
        if not selected_rows:
            raise ValueError("Round 22 diagnostic condition has no eligible rows")
        bucket = grouped.setdefault(
            role,
            {
                "conditions": [],
                "elapsed": [],
                "features": [],
                "labels": [],
                "priors": [],
            },
        )
        label_counts.setdefault(role, {"Down": 0, "Up": 0})[outcome] += 1
        label = 1.0 if outcome == "Up" else 0.0
        for row in selected_rows:
            values = row.values
            bucket["conditions"].append(condition_id)
            bucket["elapsed"].append(
                (row.decision_time_ms - market.event_start_ms) / 1_000
            )
            bucket["features"].append([values[index] for index in selected_indices])
            bucket["labels"].append(label)
            bucket["priors"].append(values[prior_index])
    partitions: dict[str, _Partition] = {}
    for role in ("train", "tune_calibration", "tune_selection"):
        bucket = grouped.get(role)
        if bucket is None:
            raise ValueError("Round 22 diagnostic partition is unavailable")
        partition = _Partition(
            role=role,
            features=np.asarray(bucket["features"], dtype=np.float64),
            priors=np.asarray(bucket["priors"], dtype=np.float64),
            labels=np.asarray(bucket["labels"], dtype=np.float64),
            condition_ids=np.asarray(bucket["conditions"], dtype=object),
            elapsed_seconds=np.asarray(bucket["elapsed"], dtype=np.float64),
        )
        if partition.condition_count != 12:
            raise ValueError("Round 22 diagnostic partition condition count differs")
        partitions[role] = partition
    return partitions, label_counts


def run_round22_diagnostic(store: Round22PilotStore) -> dict[str, object]:
    if not store.read_only:
        raise ValueError("Round 22 diagnostic requires a read-only store")
    spec = load_round22_diagnostic_model_spec(store.contract.repository)
    partitions, label_counts = _load_partitions(store, spec)
    train = partitions["train"]
    calibration = partitions["tune_calibration"]
    selection = partitions["tune_selection"]
    model_spec = spec["model"]
    evaluation = spec["evaluation"]
    assert isinstance(model_spec, Mapping)
    assert isinstance(evaluation, Mapping)
    penalties = tuple(float(value) for value in model_spec["l2_penalty_grid"])
    selected_penalty, cross_validation = _select_penalty(train, penalties)
    standardizer = _fit_standardizer(train)
    theta = _fit_residual(
        train,
        standardizer=standardizer,
        penalty=selected_penalty,
    )
    calibration_raw, calibration_residual = _raw_probabilities(
        calibration,
        standardizer=standardizer,
        theta=theta,
    )
    calibration_intercept, calibration_scale = _fit_calibration(
        calibration,
        calibration_residual,
    )
    calibration_final = expit(
        _logit(calibration.priors)
        + calibration_intercept
        + calibration_scale * calibration_residual
    )
    selection_raw, selection_residual = _raw_probabilities(
        selection,
        standardizer=standardizer,
        theta=theta,
    )
    selection_final = expit(
        _logit(selection.priors)
        + calibration_intercept
        + calibration_scale * selection_residual
    )
    selection_report = _metric_report(selection, selection_final)
    draws = int(evaluation["bootstrap_draws"])
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    brier_bootstrap = _bootstrap_report(
        _condition_metric_deltas(selection, selection_final, metric="brier"),
        rng=rng,
        draws=draws,
    )
    log_bootstrap = _bootstrap_report(
        _condition_metric_deltas(selection, selection_final, metric="log_loss"),
        rng=rng,
        draws=draws,
    )
    baseline = selection_report["baseline"]
    model = selection_report["model"]
    assert isinstance(baseline, Mapping)
    assert isinstance(model, Mapping)
    gate = evaluation["diagnostic_pass"]
    assert isinstance(gate, Mapping)
    passed = bool(
        model["brier_score"] < baseline["brier_score"]
        and model["log_loss"] < baseline["log_loss"]
        and brier_bootstrap["improvement_probability"]
        >= float(gate["brier_improvement_probability_minimum"])
        and log_bootstrap["improvement_probability"]
        >= float(gate["log_loss_improvement_probability_minimum"])
    )
    horizon_reports: list[dict[str, object]] = []
    for lower, upper in evaluation["horizon_buckets_elapsed_seconds"]:
        mask = (selection.elapsed_seconds >= float(lower)) & (
            selection.elapsed_seconds < float(upper)
        )
        selected = _subset(selection, mask)
        report = _metric_report(selected, selection_final[mask])
        horizon_reports.append({"elapsed_seconds": [int(lower), int(upper)], **report})
    residual_names = spec["data_selection"]["residual_features"]
    result_body: dict[str, object] = {
        "authority": {
            "ai_edge_claim": False,
            "economic_backtest": False,
            "live_trading": False,
            "model_promotion": False,
            "paper_trading": False,
            "profitability_claim": False,
        },
        "calibration_partition": {
            "calibrated": _metric_report(calibration, calibration_final),
            "intercept": calibration_intercept,
            "raw": _metric_report(calibration, calibration_raw),
            "residual_scale": calibration_scale,
        },
        "conclusion": (
            "bounded_predictive_signal_requires_wider_backfill_and_economic_test"
            if passed
            else "diagnostic_candidate_falsified"
        ),
        "diagnostic_pass": passed,
        "fit": {
            "coefficients_standardized": [
                {"feature": str(name), "value": float(value)}
                for name, value in zip(residual_names, theta[1:], strict=True)
            ],
            "cross_validation_log_loss_by_penalty": cross_validation,
            "intercept": float(theta[0]),
            "selected_l2_penalty": selected_penalty,
        },
        "implementation": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "label_counts": label_counts,
        "partitions": {
            role: {
                "condition_count": partition.condition_count,
                "row_count": len(partition.labels),
            }
            for role, partition in partitions.items()
        },
        "parents": {
            "model_specification_sha256": spec["specification_sha256"],
            **dict(spec["parents"]),
        },
        "schema_version": POLYMARKET_ROUND22_DIAGNOSTIC_RESULT_SCHEMA_VERSION,
        "selection": {
            "bootstrap": {
                "brier": brier_bootstrap,
                "draws": draws,
                "log_loss": log_bootstrap,
                "seed": int(evaluation["bootstrap_seed"]),
            },
            "by_elapsed_horizon": horizon_reports,
            "calibrated": selection_report,
            "raw": _metric_report(selection, selection_raw),
        },
        "status": "diagnostic_only_not_economic_or_promotable",
    }
    rounded = _rounded(result_body)
    assert isinstance(rounded, dict)
    return {**rounded, "result_sha256": _canonical_sha256(rounded)}


__all__ = [
    "POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_RELATIVE",
    "POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_SHA256",
    "POLYMARKET_ROUND22_DIAGNOSTIC_RESULT_SCHEMA_VERSION",
    "load_round22_diagnostic_model_spec",
    "run_round22_diagnostic",
]

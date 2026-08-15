"""Condition-weighted model comparison for the Round 27 BTC campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Protocol, Sequence

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    Round27FeatureRow,
)


POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION = "polymarket-round27-offset-model-v1"
POLYMARKET_ROUND27_METRICS_SCHEMA_VERSION = "polymarket-round27-probability-metrics-v1"
POLYMARKET_ROUND27_L2_PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
POLYMARKET_ROUND27_CORRECTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
POLYMARKET_ROUND27_ROLE_NAMES = (
    "train",
    "calibration",
    "selection",
    "sealed",
    "purged",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-7, max(1e-7, float(value)))


def _logit(value: float) -> float:
    selected = _clip_probability(value)
    return math.log(selected / (1.0 - selected))


def _sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    selected = np.asarray(value, dtype=np.float64)
    output = np.empty_like(selected)
    positive = selected >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-selected[positive]))
    exponential = np.exp(selected[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return np.clip(output, 1e-7, 1.0 - 1e-7)


@dataclass(frozen=True, slots=True)
class Round27RoleInterval:
    role: str
    slot_id: str
    start_ms: int
    end_ms: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Round27RoleInterval":
        item = cls(
            role=str(value.get("role") or ""),
            slot_id=str(value.get("slot_id") or ""),
            start_ms=int(value.get("start_ms") or 0),
            end_ms=int(value.get("end_ms") or 0),
        )
        if (
            item.role not in POLYMARKET_ROUND27_ROLE_NAMES
            or not item.slot_id.startswith("stage1-")
            or item.start_ms <= 0
            or item.end_ms <= item.start_ms
            or item.start_ms % 300_000
            or item.end_ms % 300_000
        ):
            raise ValueError("Round 27 role interval differs")
        return item


@dataclass(frozen=True, slots=True)
class Round27ModelSample:
    slot_id: str
    role: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    values: tuple[float, ...]
    target_up: int
    condition_weight: float
    feature_row_sha256: str

    def validated(self) -> "Round27ModelSample":
        if (
            not self.slot_id.startswith("stage1-")
            or self.role not in POLYMARKET_ROUND27_ROLE_NAMES
            or self.role == "purged"
            or not self.condition_id.startswith("0x")
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or not 0.0 < self.market_prior_probability < 1.0
            or len(self.values) != len(POLYMARKET_ROUND27_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.values)
            or self.target_up not in {0, 1}
            or not math.isfinite(self.condition_weight)
            or self.condition_weight <= 0.0
            or len(self.feature_row_sha256) != 64
        ):
            raise ValueError("Round 27 model sample differs")
        return self


def build_round27_model_samples(
    *,
    rows_by_slot: Mapping[str, Sequence[Round27FeatureRow]],
    outcomes_up: Mapping[str, int],
    role_intervals: Sequence[Mapping[str, object]],
) -> tuple[Round27ModelSample, ...]:
    """Join official outcomes only after the target-blind contract is frozen."""

    intervals = tuple(Round27RoleInterval.from_mapping(item) for item in role_intervals)
    if not intervals or len(set(intervals)) != len(intervals):
        raise ValueError("Round 27 role intervals differ")
    ordered = tuple(sorted(intervals, key=lambda item: (item.start_ms, item.end_ms)))
    if any(
        current.start_ms < previous.end_ms
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("Round 27 role intervals overlap")
    rows: list[tuple[str, str, Round27FeatureRow]] = []
    keys: set[tuple[str, int]] = set()
    for slot_id, raw_rows in sorted(rows_by_slot.items()):
        for raw in raw_rows:
            row = raw.validated()
            key = (row.condition_id, row.decision_time_ms)
            if key in keys:
                raise ValueError("Round 27 feature row is duplicated across slots")
            keys.add(key)
            matches = tuple(
                item
                for item in intervals
                if item.slot_id == slot_id
                and item.start_ms <= row.event_start_ms < item.end_ms
            )
            if len(matches) != 1:
                raise ValueError("Round 27 feature row role is ambiguous")
            if matches[0].role != "purged":
                rows.append((slot_id, matches[0].role, row))
    condition_counts: dict[str, int] = {}
    condition_roles: dict[str, str] = {}
    for _slot, role, row in rows:
        prior = condition_roles.setdefault(row.condition_id, role)
        if prior != role:
            raise ValueError("Round 27 condition crosses roles")
        condition_counts[row.condition_id] = (
            condition_counts.get(row.condition_id, 0) + 1
        )
    samples: list[Round27ModelSample] = []
    for slot_id, role, row in rows:
        target = outcomes_up.get(row.condition_id)
        if type(target) is not int or target not in {0, 1}:
            raise ValueError("Round 27 official outcome population differs")
        samples.append(
            Round27ModelSample(
                slot_id=slot_id,
                role=role,
                condition_id=row.condition_id,
                event_start_ms=row.event_start_ms,
                decision_time_ms=row.decision_time_ms,
                market_prior_probability=row.market_prior_probability,
                values=row.values,
                target_up=target,
                condition_weight=1.0 / condition_counts[row.condition_id],
                feature_row_sha256=row.row_sha256,
            ).validated()
        )
    if not samples:
        raise ValueError("Round 27 labeled sample population is empty")
    for condition_id, count in condition_counts.items():
        weight = math.fsum(
            sample.condition_weight
            for sample in samples
            if sample.condition_id == condition_id
        )
        if count <= 0 or not math.isclose(weight, 1.0, abs_tol=1e-12):
            raise RuntimeError("Round 27 condition weights differ")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class Round27Partition:
    role: str
    samples: tuple[Round27ModelSample, ...]
    features: NDArray[np.float64]
    offsets: NDArray[np.float64]
    targets: NDArray[np.float64]
    weights: NDArray[np.float64]
    conditions: NDArray[np.str_]

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Round27ModelSample],
        *,
        role: str,
    ) -> "Round27Partition":
        selected = tuple(
            sample.validated() for sample in samples if sample.role == role
        )
        if (
            not selected
            or role not in POLYMARKET_ROUND27_ROLE_NAMES
            or role == "purged"
        ):
            raise ValueError("Round 27 partition is empty")
        features = np.asarray([sample.values for sample in selected], dtype=np.float64)
        offsets = np.asarray(
            [_logit(sample.market_prior_probability) for sample in selected],
            dtype=np.float64,
        )
        targets = np.asarray(
            [sample.target_up for sample in selected], dtype=np.float64
        )
        weights = np.asarray(
            [sample.condition_weight for sample in selected], dtype=np.float64
        )
        conditions = np.asarray(
            [sample.condition_id for sample in selected], dtype=np.str_
        )
        if (
            features.shape != (len(selected), len(POLYMARKET_ROUND27_FEATURE_NAMES))
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(offsets))
            or not np.all(np.isfinite(weights))
            or set(np.unique(targets)) - {0.0, 1.0}
        ):
            raise ValueError("Round 27 partition matrix differs")
        return cls(role, selected, features, offsets, targets, weights, conditions)


@dataclass(frozen=True, slots=True)
class Round27ProbabilityMetrics:
    condition_count: int
    row_count: int
    log_loss: float
    brier_score: float
    balanced_accuracy: float
    expected_calibration_error: float

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_METRICS_SCHEMA_VERSION,
            "condition_count": self.condition_count,
            "row_count": self.row_count,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "balanced_accuracy": self.balanced_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
        }


def round27_probability_metrics(
    partition: Round27Partition,
    predictions: Sequence[float] | NDArray[np.float64],
) -> Round27ProbabilityMetrics:
    probability = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if probability.shape != partition.targets.shape or not np.all(
        np.isfinite(probability)
    ):
        raise ValueError("Round 27 probability population differs")
    weight = partition.weights / np.sum(partition.weights)
    losses = -(
        partition.targets * np.log(probability)
        + (1.0 - partition.targets) * np.log1p(-probability)
    )
    brier = (probability - partition.targets) ** 2
    prediction_class = probability >= 0.5
    recalls: list[float] = []
    for target in (0.0, 1.0):
        selected = partition.targets == target
        denominator = np.sum(weight[selected])
        if denominator <= 0.0:
            raise ValueError("Round 27 balanced-accuracy class is absent")
        recalls.append(
            float(
                np.sum(weight[selected] * (prediction_class[selected] == bool(target)))
                / denominator
            )
        )
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        bucket_weight = float(np.sum(weight[selected]))
        if bucket_weight <= 0.0:
            continue
        confidence = float(
            np.sum(weight[selected] * probability[selected]) / bucket_weight
        )
        frequency = float(
            np.sum(weight[selected] * partition.targets[selected]) / bucket_weight
        )
        ece += bucket_weight * abs(confidence - frequency)
    return Round27ProbabilityMetrics(
        condition_count=int(np.unique(partition.conditions).size),
        row_count=int(partition.targets.size),
        log_loss=float(np.sum(weight * losses)),
        brier_score=float(np.sum(weight * brier)),
        balanced_accuracy=float(math.fsum(recalls) / 2.0),
        expected_calibration_error=ece,
    )


class Round27ProbabilityModel(Protocol):
    model_name: str

    def predict(
        self, features: NDArray[np.float64], offsets: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def asdict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class Round27L2OffsetModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    penalty: float
    correction_scale: float
    model_sha256: str
    model_name: str = "l2_offset_logistic"

    def predict(
        self, features: NDArray[np.float64], offsets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        selected = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            selected.ndim != 2
            or selected.shape[1] != mean.size
            or coefficients.size != mean.size + 1
        ):
            raise ValueError("Round 27 L2 inference width differs")
        standardized = np.clip((selected - mean) / scale, -12.0, 12.0)
        correction = coefficients[0] + standardized @ coefficients[1:]
        return _sigmoid(
            np.asarray(offsets, dtype=np.float64) + self.correction_scale * correction
        )

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "penalty": self.penalty,
            "correction_scale": self.correction_scale,
            "model_sha256": self.model_sha256,
        }


def fit_round27_l2_offset(
    partition: Round27Partition,
    *,
    penalty: float,
    correction_scale: float = 1.0,
) -> Round27L2OffsetModel:
    selected_penalty = float(penalty)
    selected_correction = float(correction_scale)
    if (
        selected_penalty <= 0.0
        or selected_correction not in POLYMARKET_ROUND27_CORRECTION_SCALES
    ):
        raise ValueError("Round 27 L2 controls differ")
    weights = partition.weights / np.sum(partition.weights)
    mean = np.average(partition.features, axis=0, weights=weights)
    variance = np.average((partition.features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = np.clip((partition.features - mean) / scale, -12.0, 12.0)
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    regularization = np.ones(design.shape[1], dtype=np.float64)
    regularization[0] = 0.0
    for _iteration in range(75):
        probability = _sigmoid(partition.offsets + design @ coefficients)
        curvature = weights * np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = design.T @ (weights * (partition.targets - probability))
        gradient -= selected_penalty * regularization * coefficients
        hessian = (design.T * curvature) @ design
        hessian += np.diag(selected_penalty * regularization + 1e-10)
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Round 27 L2 system is singular") from exc
        coefficients += delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Round 27 L2 coefficients are non-finite")
    body = {
        "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
        "model_name": "l2_offset_logistic",
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "penalty": selected_penalty,
        "correction_scale": selected_correction,
    }
    return Round27L2OffsetModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        penalty=selected_penalty,
        correction_scale=selected_correction,
        model_sha256=_canonical_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class Round27LightGbmOffsetModel:
    model_text: str
    correction_scale: float
    backend_kind: str
    backend_device: str
    model_sha256: str
    model_name: str = "shallow_lightgbm_offset"

    def predict(
        self, features: NDArray[np.float64], offsets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        booster = lgb.Booster(model_str=self.model_text)
        raw = np.asarray(
            booster.predict(np.asarray(features, dtype=np.float64)), dtype=np.float64
        )
        correction = np.asarray(
            [_logit(value) for value in raw], dtype=np.float64
        ) - np.asarray(offsets, dtype=np.float64)
        return _sigmoid(
            np.asarray(offsets, dtype=np.float64) + self.correction_scale * correction
        )

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "correction_scale": self.correction_scale,
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "model_text_sha256": hashlib.sha256(
                self.model_text.encode("utf-8")
            ).hexdigest(),
            "model_sha256": self.model_sha256,
        }


def fit_round27_lightgbm_offset(
    partition: Round27Partition,
    *,
    compute_backend: str = "auto",
    seed: int = 2701,
    correction_scale: float = 1.0,
) -> Round27LightGbmOffsetModel:
    selected_correction = float(correction_scale)
    if selected_correction not in POLYMARKET_ROUND27_CORRECTION_SCALES:
        raise ValueError("Round 27 LightGBM correction scale differs")
    backend, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
        pin_opencl_device=False,
    )
    parameters: dict[str, object] = {
        **backend,
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": max(20, int(partition.targets.size // 200)),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 5.0,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(
        partition.features,
        label=partition.targets,
        weight=partition.weights,
        feature_name=list(POLYMARKET_ROUND27_FEATURE_NAMES),
        free_raw_data=False,
    )
    booster = lgb.train(parameters, dataset, num_boost_round=120)
    model_text = booster.model_to_string()
    body = {
        "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
        "model_name": "shallow_lightgbm_offset",
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "correction_scale": selected_correction,
        "backend_kind": kind,
        "backend_device": device,
        "model_text_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
    }
    return Round27LightGbmOffsetModel(
        model_text=model_text,
        correction_scale=selected_correction,
        backend_kind=kind,
        backend_device=device,
        model_sha256=_canonical_sha256(body),
    )


def _condition_fold(condition_id: str, fold_count: int) -> int:
    digest = hashlib.sha256(condition_id.encode("ascii")).hexdigest()
    return int(digest[:16], 16) % fold_count


def select_round27_l2_penalty(
    partition: Round27Partition,
    *,
    fold_count: int = 5,
) -> tuple[float, dict[str, float]]:
    conditions = tuple(sorted(set(str(value) for value in partition.conditions)))
    folds = {
        condition: _condition_fold(condition, fold_count) for condition in conditions
    }
    if set(folds.values()) != set(range(fold_count)):
        raise ValueError("Round 27 condition folds are incomplete")
    scores: dict[str, float] = {}
    for penalty in POLYMARKET_ROUND27_L2_PENALTIES:
        fold_losses: list[float] = []
        for fold in range(fold_count):
            held_out = np.asarray(
                [folds[str(value)] == fold for value in partition.conditions]
            )
            train_samples = tuple(
                sample
                for sample, held in zip(partition.samples, held_out, strict=True)
                if not held
            )
            held_samples = tuple(
                sample
                for sample, held in zip(partition.samples, held_out, strict=True)
                if held
            )
            train = Round27Partition.from_samples(train_samples, role=partition.role)
            held = Round27Partition.from_samples(held_samples, role=partition.role)
            model = fit_round27_l2_offset(train, penalty=penalty)
            fold_losses.append(
                round27_probability_metrics(
                    held, model.predict(held.features, held.offsets)
                ).log_loss
            )
        scores[str(penalty)] = math.fsum(fold_losses) / len(fold_losses)
    selected = min(
        POLYMARKET_ROUND27_L2_PENALTIES, key=lambda value: (scores[str(value)], value)
    )
    return selected, scores


def select_round27_correction_scale(
    model: Round27ProbabilityModel,
    partition: Round27Partition,
) -> tuple[float, dict[str, float]]:
    if isinstance(model, Round27L2OffsetModel):
        factory = lambda scale: Round27L2OffsetModel(  # noqa: E731
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            penalty=model.penalty,
            correction_scale=scale,
            model_sha256=model.model_sha256,
        )
    elif isinstance(model, Round27LightGbmOffsetModel):
        factory = lambda scale: Round27LightGbmOffsetModel(  # noqa: E731
            model_text=model.model_text,
            correction_scale=scale,
            backend_kind=model.backend_kind,
            backend_device=model.backend_device,
            model_sha256=model.model_sha256,
        )
    else:
        raise TypeError("Round 27 probability model type differs")
    scores: dict[str, float] = {}
    for scale in POLYMARKET_ROUND27_CORRECTION_SCALES:
        candidate = factory(scale)
        scores[str(scale)] = round27_probability_metrics(
            partition,
            candidate.predict(partition.features, partition.offsets),
        ).log_loss
    return min(
        POLYMARKET_ROUND27_CORRECTION_SCALES,
        key=lambda value: (scores[str(value)], value),
    ), scores


def paired_round27_condition_bootstrap(
    partition: Round27Partition,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    draws: int = 5_000,
    seed: int = 2702,
) -> dict[str, float]:
    baseline_probability = np.clip(
        np.asarray(baseline, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    candidate_probability = np.clip(
        np.asarray(candidate, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    if (
        baseline_probability.shape != partition.targets.shape
        or candidate_probability.shape != partition.targets.shape
    ):
        raise ValueError("Round 27 bootstrap prediction population differs")
    condition_values: list[float] = []
    for condition in sorted(set(str(value) for value in partition.conditions)):
        selected = partition.conditions == condition
        target = partition.targets[selected]
        baseline_loss = -(
            target * np.log(baseline_probability[selected])
            + (1.0 - target) * np.log1p(-baseline_probability[selected])
        )
        candidate_loss = -(
            target * np.log(candidate_probability[selected])
            + (1.0 - target) * np.log1p(-candidate_probability[selected])
        )
        condition_values.append(float(np.mean(candidate_loss - baseline_loss)))
    values = np.asarray(condition_values, dtype=np.float64)
    if values.size < 20 or draws < 1_000:
        raise ValueError("Round 27 bootstrap population is insufficient")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        samples[start : start + count] = np.mean(values[indices], axis=1)
    return {
        "mean_candidate_minus_prior_log_loss": float(np.mean(values)),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "condition_count": int(values.size),
        "draw_count": draws,
    }


__all__ = [
    "POLYMARKET_ROUND27_CORRECTION_SCALES",
    "POLYMARKET_ROUND27_L2_PENALTIES",
    "POLYMARKET_ROUND27_METRICS_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION",
    "Round27L2OffsetModel",
    "Round27LightGbmOffsetModel",
    "Round27ModelSample",
    "Round27Partition",
    "Round27ProbabilityMetrics",
    "build_round27_model_samples",
    "fit_round27_l2_offset",
    "fit_round27_lightgbm_offset",
    "paired_round27_condition_bootstrap",
    "round27_probability_metrics",
    "select_round27_correction_scale",
    "select_round27_l2_penalty",
]

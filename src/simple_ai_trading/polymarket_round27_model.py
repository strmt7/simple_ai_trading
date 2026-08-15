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


POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION = "polymarket-round27-offset-model-v2"
POLYMARKET_ROUND27_METRICS_SCHEMA_VERSION = "polymarket-round27-probability-metrics-v1"
POLYMARKET_ROUND27_L2_PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
POLYMARKET_ROUND27_CORRECTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
POLYMARKET_ROUND27_BOOTSTRAP_EXPECTED_BLOCK_LENGTHS = (1, 4, 12)
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
        correction = np.asarray(
            booster.predict(
                np.asarray(features, dtype=np.float64),
                raw_score=True,
            ),
            dtype=np.float64,
        )
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
            "model_text": self.model_text,
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
        init_score=partition.offsets,
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


def scale_round27_probability_model(
    model: Round27L2OffsetModel | Round27LightGbmOffsetModel,
    correction_scale: float,
) -> Round27L2OffsetModel | Round27LightGbmOffsetModel:
    """Return the same fitted correction with a newly bound calibration scale."""

    selected_scale = float(correction_scale)
    if selected_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES:
        raise ValueError("Round 27 correction scale differs")
    if isinstance(model, Round27L2OffsetModel):
        body = {
            "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
            "model_name": model.model_name,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "mean": list(model.mean),
            "scale": list(model.scale),
            "coefficients": list(model.coefficients),
            "penalty": model.penalty,
            "correction_scale": selected_scale,
        }
        return Round27L2OffsetModel(
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            penalty=model.penalty,
            correction_scale=selected_scale,
            model_sha256=_canonical_sha256(body),
        )
    if isinstance(model, Round27LightGbmOffsetModel):
        model_text_sha256 = hashlib.sha256(
            model.model_text.encode("utf-8")
        ).hexdigest()
        body = {
            "schema_version": POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION,
            "model_name": model.model_name,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "correction_scale": selected_scale,
            "backend_kind": model.backend_kind,
            "backend_device": model.backend_device,
            "model_text_sha256": model_text_sha256,
        }
        return Round27LightGbmOffsetModel(
            model_text=model.model_text,
            correction_scale=selected_scale,
            backend_kind=model.backend_kind,
            backend_device=model.backend_device,
            model_sha256=_canonical_sha256(body),
        )
    raise TypeError("Round 27 probability model type differs")


def round27_model_from_payload(
    value: Mapping[str, object],
) -> Round27L2OffsetModel | Round27LightGbmOffsetModel:
    """Reconstruct only an exact, self-contained frozen model payload."""

    payload = dict(value)
    model_name = payload.get("model_name")
    if (
        payload.get("schema_version") != POLYMARKET_ROUND27_MODEL_SCHEMA_VERSION
        or payload.get("feature_names_sha256")
        != POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
    ):
        raise ValueError("Round 27 persisted model schema differs")
    if model_name == "l2_offset_logistic":
        expected = {
            "schema_version",
            "model_name",
            "feature_names_sha256",
            "mean",
            "scale",
            "coefficients",
            "penalty",
            "correction_scale",
            "model_sha256",
        }
        try:
            mean = tuple(float(item) for item in payload["mean"])  # type: ignore[union-attr]
            scale = tuple(float(item) for item in payload["scale"])  # type: ignore[union-attr]
            coefficients = tuple(
                float(item) for item in payload["coefficients"]  # type: ignore[union-attr]
            )
            penalty = float(payload["penalty"])
            correction_scale = float(payload["correction_scale"])
            claimed = str(payload["model_sha256"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 27 persisted L2 model differs") from exc
        feature_count = len(POLYMARKET_ROUND27_FEATURE_NAMES)
        body = dict(payload)
        body.pop("model_sha256", None)
        if (
            set(payload) != expected
            or len(mean) != feature_count
            or len(scale) != feature_count
            or len(coefficients) != feature_count + 1
            or any(not math.isfinite(item) for item in (*mean, *scale, *coefficients))
            or any(item <= 0.0 for item in scale)
            or not math.isfinite(penalty)
            or penalty <= 0.0
            or correction_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES
            or claimed != _canonical_sha256(body)
        ):
            raise ValueError("Round 27 persisted L2 model differs")
        return Round27L2OffsetModel(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            penalty=penalty,
            correction_scale=correction_scale,
            model_sha256=claimed,
        )
    if model_name == "shallow_lightgbm_offset":
        expected = {
            "schema_version",
            "model_name",
            "feature_names_sha256",
            "correction_scale",
            "backend_kind",
            "backend_device",
            "model_text",
            "model_text_sha256",
            "model_sha256",
        }
        model_text = payload.get("model_text")
        model_text_sha256 = str(payload.get("model_text_sha256") or "")
        claimed = str(payload.get("model_sha256") or "")
        backend_kind = str(payload.get("backend_kind") or "")
        backend_device = str(payload.get("backend_device") or "")
        try:
            correction_scale = float(payload["correction_scale"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 27 persisted LightGBM model differs") from exc
        body = dict(payload)
        body.pop("model_sha256", None)
        body.pop("model_text", None)
        if (
            set(payload) != expected
            or not isinstance(model_text, str)
            or not model_text
            or hashlib.sha256(model_text.encode("utf-8")).hexdigest()
            != model_text_sha256
            or not backend_kind
            or not backend_device
            or correction_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES
            or claimed != _canonical_sha256(body)
        ):
            raise ValueError("Round 27 persisted LightGBM model differs")
        try:
            booster = lgb.Booster(model_str=model_text)
        except (TypeError, ValueError, lgb.basic.LightGBMError) as exc:
            raise ValueError("Round 27 persisted LightGBM model is invalid") from exc
        if booster.num_feature() != len(POLYMARKET_ROUND27_FEATURE_NAMES):
            raise ValueError("Round 27 persisted LightGBM width differs")
        return Round27LightGbmOffsetModel(
            model_text=model_text,
            correction_scale=correction_scale,
            backend_kind=backend_kind,
            backend_device=backend_device,
            model_sha256=claimed,
        )
    raise ValueError("Round 27 persisted model family differs")


def _walk_forward_condition_folds(
    partition: Round27Partition,
    *,
    fold_count: int,
    embargo_ms: int = 600_000,
) -> tuple[tuple[Round27Partition, Round27Partition], ...]:
    """Build expanding condition-level folds without training on the future."""

    if fold_count < 2 or embargo_ms < 0 or embargo_ms % 300_000:
        raise ValueError("Round 27 walk-forward controls differ")
    event_start_by_condition: dict[str, int] = {}
    for sample in partition.samples:
        prior = event_start_by_condition.setdefault(
            sample.condition_id,
            sample.event_start_ms,
        )
        if prior != sample.event_start_ms:
            raise ValueError("Round 27 condition start time differs")
    ordered_conditions = tuple(
        condition
        for condition, _event_start_ms in sorted(
            event_start_by_condition.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    block_count = fold_count + 1
    if len(ordered_conditions) < block_count:
        raise ValueError("Round 27 walk-forward population is insufficient")
    minimum_size, larger_blocks = divmod(len(ordered_conditions), block_count)
    block_sizes = tuple(
        minimum_size + (1 if index < larger_blocks else 0)
        for index in range(block_count)
    )
    blocks: list[tuple[str, ...]] = []
    cursor = 0
    for size in block_sizes:
        blocks.append(ordered_conditions[cursor : cursor + size])
        cursor += size
    folds: list[tuple[Round27Partition, Round27Partition]] = []
    for validation_index in range(1, block_count):
        validation_conditions = frozenset(blocks[validation_index])
        validation_start_ms = min(
            event_start_by_condition[condition]
            for condition in validation_conditions
        )
        preceding_conditions = {
            condition
            for block in blocks[:validation_index]
            for condition in block
            if event_start_by_condition[condition] + 300_000 + embargo_ms
            <= validation_start_ms
        }
        train_samples = tuple(
            sample
            for sample in partition.samples
            if sample.condition_id in preceding_conditions
        )
        validation_samples = tuple(
            sample
            for sample in partition.samples
            if sample.condition_id in validation_conditions
        )
        if not train_samples or not validation_samples:
            raise ValueError("Round 27 embargoed walk-forward fold is empty")
        train = Round27Partition.from_samples(train_samples, role=partition.role)
        validation = Round27Partition.from_samples(
            validation_samples,
            role=partition.role,
        )
        if (
            max(sample.event_start_ms + 300_000 for sample in train.samples)
            + embargo_ms
            > min(sample.event_start_ms for sample in validation.samples)
        ):
            raise RuntimeError("Round 27 walk-forward embargo differs")
        folds.append((train, validation))
    if len(folds) != fold_count:
        raise RuntimeError("Round 27 walk-forward fold count differs")
    return tuple(folds)


def select_round27_l2_penalty(
    partition: Round27Partition,
    *,
    fold_count: int = 5,
) -> tuple[float, dict[str, float]]:
    folds = _walk_forward_condition_folds(
        partition,
        fold_count=fold_count,
    )
    scores: dict[str, float] = {}
    for penalty in POLYMARKET_ROUND27_L2_PENALTIES:
        weighted_losses: list[float] = []
        validation_condition_count = 0
        for train, validation in folds:
            model = fit_round27_l2_offset(train, penalty=penalty)
            probability = model.predict(validation.features, validation.offsets)
            metrics = round27_probability_metrics(validation, probability)
            weighted_losses.append(
                metrics.log_loss * metrics.condition_count
            )
            validation_condition_count += metrics.condition_count
        scores[str(penalty)] = (
            math.fsum(weighted_losses) / validation_condition_count
        )
    selected = min(
        POLYMARKET_ROUND27_L2_PENALTIES, key=lambda value: (scores[str(value)], value)
    )
    return selected, scores


def select_round27_correction_scale(
    model: Round27ProbabilityModel,
    partition: Round27Partition,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for scale in POLYMARKET_ROUND27_CORRECTION_SCALES:
        candidate = scale_round27_probability_model(model, scale)
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
) -> dict[str, object]:
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
    event_start_by_condition: dict[str, int] = {}
    for sample in partition.samples:
        prior = event_start_by_condition.setdefault(
            sample.condition_id,
            sample.event_start_ms,
        )
        if prior != sample.event_start_ms:
            raise ValueError("Round 27 bootstrap condition time differs")
    ordered_conditions = tuple(
        condition
        for condition, _event_start_ms in sorted(
            event_start_by_condition.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    condition_log_loss_values: list[float] = []
    condition_brier_values: list[float] = []
    for condition in ordered_conditions:
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
        condition_log_loss_values.append(
            float(np.mean(candidate_loss - baseline_loss))
        )
        baseline_brier = (baseline_probability[selected] - target) ** 2
        candidate_brier = (candidate_probability[selected] - target) ** 2
        condition_brier_values.append(
            float(np.mean(candidate_brier - baseline_brier))
        )
    log_loss_values = np.asarray(condition_log_loss_values, dtype=np.float64)
    brier_values = np.asarray(condition_brier_values, dtype=np.float64)
    if log_loss_values.size < 20 or draws < 1_000:
        raise ValueError("Round 27 bootstrap population is insufficient")
    log_loss_interval = round27_stationary_bootstrap_mean_interval(
        log_loss_values,
        draws=draws,
        seed=seed,
    )
    brier_interval = round27_stationary_bootstrap_mean_interval(
        brier_values,
        draws=draws,
        seed=seed + 97_409,
    )
    return {
        "mean_candidate_minus_prior_log_loss": float(np.mean(log_loss_values)),
        "mean_candidate_minus_prior_brier_score": float(np.mean(brier_values)),
        "condition_count": int(log_loss_values.size),
        "log_loss": log_loss_interval,
        "brier_score": brier_interval,
        **log_loss_interval,
    }


def round27_corrected_politis_white_block_length(
    values: Sequence[float] | NDArray[np.float64],
) -> int:
    """Estimate a conservative integer stationary-bootstrap block length."""

    selected = np.asarray(values, dtype=np.float64)
    if (
        selected.ndim != 1
        or selected.size < 20
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 27 automatic block population differs")
    observation_count = int(selected.size)
    centered = selected - float(np.mean(selected))
    if float(centered @ centered) <= np.finfo(np.float64).tiny:
        return 1
    maximum_candidate = int(
        math.ceil(min(3.0 * math.sqrt(observation_count), observation_count / 3.0))
    )
    consecutive_lags = max(5, int(math.log10(observation_count)))
    maximum_lag = min(
        observation_count - 2,
        int(math.ceil(math.sqrt(observation_count))) + consecutive_lags,
    )
    significance_band = 2.0 * math.sqrt(
        math.log10(observation_count) / observation_count
    )
    autocovariance = np.zeros(maximum_lag + 1, dtype=np.float64)
    absolute_autocorrelation = np.full(
        maximum_lag + 1,
        np.inf,
        dtype=np.float64,
    )
    first_insignificant_lag: int | None = None
    for lag in range(maximum_lag + 1):
        cross_product = float(centered[lag:] @ centered[: observation_count - lag])
        autocovariance[lag] = cross_product / observation_count
        leading = centered[lag + 1 :]
        trailing = centered[: -(lag + 1)]
        denominator = math.sqrt(float(leading @ leading) * float(trailing @ trailing))
        if denominator > np.finfo(np.float64).tiny:
            absolute_autocorrelation[lag] = abs(cross_product) / denominator
        if (
            lag >= consecutive_lags
            and first_insignificant_lag is None
            and np.all(
                absolute_autocorrelation[
                    lag - consecutive_lags : lag
                ]
                < significance_band
            )
        ):
            first_insignificant_lag = lag - consecutive_lags
    truncation_lag = (
        maximum_lag
        if first_insignificant_lag is None
        else min(2 * max(first_insignificant_lag, 1), maximum_lag)
    )
    weighted_lag_covariance = 0.0
    long_run_covariance = float(autocovariance[0])
    for lag in range(1, truncation_lag + 1):
        ratio = lag / truncation_lag
        flat_top_weight = 1.0 if ratio <= 0.5 else 2.0 * (1.0 - ratio)
        weighted_lag_covariance += (
            2.0 * flat_top_weight * lag * float(autocovariance[lag])
        )
        long_run_covariance += (
            2.0 * flat_top_weight * float(autocovariance[lag])
        )
    variance_constant = 2.0 * long_run_covariance**2
    if (
        variance_constant <= np.finfo(np.float64).tiny
        or not math.isfinite(variance_constant)
    ):
        return 1
    estimate = (
        (2.0 * weighted_lag_covariance**2 / variance_constant)
        ** (1.0 / 3.0)
        * observation_count ** (1.0 / 3.0)
    )
    if not math.isfinite(estimate) or estimate <= 1.0:
        return 1
    return min(maximum_candidate, max(1, int(math.ceil(estimate))))


def round27_stationary_bootstrap_mean_interval(
    values: Sequence[float] | NDArray[np.float64],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Return a conservative envelope over chronological stationary bootstraps."""

    selected = np.asarray(values, dtype=np.float64)
    if (
        selected.ndim != 1
        or selected.size < 20
        or draws < 1_000
        or not np.all(np.isfinite(selected))
    ):
        raise ValueError("Round 27 stationary bootstrap population differs")
    maximum_block = max(1, selected.size // 4)
    fixed_block_lengths = tuple(
        length
        for length in POLYMARKET_ROUND27_BOOTSTRAP_EXPECTED_BLOCK_LENGTHS
        if length <= maximum_block
    )
    automatic_block_length = min(
        maximum_block,
        round27_corrected_politis_white_block_length(selected),
    )
    block_lengths = tuple(
        sorted({*fixed_block_lengths, automatic_block_length})
    )
    intervals: list[dict[str, object]] = []
    for expected_block_length in block_lengths:
        rng = np.random.default_rng(seed + 1_000_003 * expected_block_length)
        means = np.empty(draws, dtype=np.float64)
        restart_probability = 1.0 / expected_block_length
        for start in range(0, draws, 500):
            count = min(500, draws - start)
            indices = np.empty((count, selected.size), dtype=np.int64)
            indices[:, 0] = rng.integers(0, selected.size, size=count)
            for offset in range(1, selected.size):
                restart = rng.random(count) < restart_probability
                replacement = rng.integers(0, selected.size, size=count)
                indices[:, offset] = np.where(
                    restart,
                    replacement,
                    (indices[:, offset - 1] + 1) % selected.size,
                )
            means[start : start + count] = np.mean(selected[indices], axis=1)
        intervals.append(
            {
                "expected_block_length_conditions": expected_block_length,
                "ci95_lower": float(np.quantile(means, 0.025)),
                "ci95_upper": float(np.quantile(means, 0.975)),
            }
        )
    return {
        "method": "stationary_bootstrap_block_length_sensitivity_envelope",
        "draw_count": draws,
        "draw_count_per_block_length": draws,
        "effective_draw_count": draws * len(intervals),
        "automatic_block_length_method": (
            "corrected_politis_white_2004_2009_ceiling_capped_at_population_quarter"
        ),
        "automatic_expected_block_length_conditions": automatic_block_length,
        "fixed_expected_block_lengths_conditions": list(fixed_block_lengths),
        "expected_block_lengths_conditions": list(block_lengths),
        "block_intervals": intervals,
        "ci95_lower": min(float(item["ci95_lower"]) for item in intervals),
        "ci95_upper": max(float(item["ci95_upper"]) for item in intervals),
    }


__all__ = [
    "POLYMARKET_ROUND27_BOOTSTRAP_EXPECTED_BLOCK_LENGTHS",
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
    "round27_corrected_politis_white_block_length",
    "round27_model_from_payload",
    "round27_probability_metrics",
    "round27_stationary_bootstrap_mean_interval",
    "scale_round27_probability_model",
    "select_round27_correction_scale",
    "select_round27_l2_penalty",
]

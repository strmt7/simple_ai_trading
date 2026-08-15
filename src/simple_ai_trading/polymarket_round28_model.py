"""Matched offset-model ablation for the Round 28 Binance BBO overlay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, Protocol

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
)
from .polymarket_round27_model import (
    POLYMARKET_ROUND27_CORRECTION_SCALES,
    POLYMARKET_ROUND27_ROLE_NAMES,
    Round27RoleInterval,
)
from .polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
    POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
    Round28FeatureRow,
)


POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION = "polymarket-round28-offset-model-v1"
POLYMARKET_ROUND28_METRICS_SCHEMA_VERSION = "polymarket-round28-probability-metrics-v1"
POLYMARKET_ROUND28_MATCHED_ABLATION_SCHEMA_VERSION = (
    "polymarket-round28-matched-bbo-ablation-v1"
)
POLYMARKET_ROUND28_L2_PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
Round28FeatureView = Literal["round27_base", "round28_bbo_augmented"]
_FEATURE_VIEWS: dict[Round28FeatureView, tuple[tuple[str, ...], str]] = {
    "round27_base": (
        POLYMARKET_ROUND27_FEATURE_NAMES,
        POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    ),
    "round28_bbo_augmented": (
        POLYMARKET_ROUND28_FEATURE_NAMES,
        POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
    ),
}
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
}


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


def _feature_contract(view: Round28FeatureView) -> tuple[tuple[str, ...], str]:
    try:
        return _FEATURE_VIEWS[view]
    except KeyError as exc:
        raise ValueError("Round 28 feature view differs") from exc


@dataclass(frozen=True, slots=True)
class Round28ModelSample:
    slot_id: str
    role: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    base_values: tuple[float, ...]
    augmented_values: tuple[float, ...]
    target_up: int
    condition_weight: float
    feature_row_sha256: str

    def validated(self) -> "Round28ModelSample":
        if (
            not self.slot_id.startswith("stage1-")
            or self.role not in POLYMARKET_ROUND27_ROLE_NAMES
            or self.role == "purged"
            or not self.condition_id.startswith("0x")
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or not 0.0 < self.market_prior_probability < 1.0
            or len(self.base_values) != len(POLYMARKET_ROUND27_FEATURE_NAMES)
            or len(self.augmented_values) != len(POLYMARKET_ROUND28_FEATURE_NAMES)
            or self.augmented_values[: len(self.base_values)] != self.base_values
            or any(
                not math.isfinite(value)
                for value in (*self.base_values, *self.augmented_values)
            )
            or self.target_up not in {0, 1}
            or not math.isfinite(self.condition_weight)
            or self.condition_weight <= 0.0
            or len(self.feature_row_sha256) != 64
        ):
            raise ValueError("Round 28 model sample differs")
        return self


def build_round28_model_samples(
    *,
    rows_by_slot: Mapping[str, Sequence[Round28FeatureRow]],
    outcomes_up: Mapping[str, int],
    role_intervals: Sequence[Mapping[str, object]],
) -> tuple[Round28ModelSample, ...]:
    """Join outcomes only after the target-blind overlay is terminal and frozen."""

    intervals = tuple(Round27RoleInterval.from_mapping(item) for item in role_intervals)
    if not intervals or len(set(intervals)) != len(intervals):
        raise ValueError("Round 28 role intervals differ")
    ordered = tuple(sorted(intervals, key=lambda item: (item.start_ms, item.end_ms)))
    if any(
        current.start_ms < previous.end_ms
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("Round 28 role intervals overlap")
    selected_rows: list[tuple[str, str, Round28FeatureRow]] = []
    keys: set[tuple[str, int]] = set()
    for slot_id, raw_rows in sorted(rows_by_slot.items()):
        for raw_row in raw_rows:
            row = raw_row.validated()
            key = (row.condition_id, row.decision_time_ms)
            if key in keys:
                raise ValueError("Round 28 feature row is duplicated across slots")
            keys.add(key)
            matches = tuple(
                interval
                for interval in intervals
                if interval.slot_id == slot_id
                and interval.start_ms <= row.event_start_ms < interval.end_ms
            )
            if len(matches) != 1:
                raise ValueError("Round 28 feature row role is ambiguous")
            if matches[0].role != "purged":
                selected_rows.append((slot_id, matches[0].role, row))
    condition_counts: dict[str, int] = {}
    condition_roles: dict[str, str] = {}
    for _slot_id, role, row in selected_rows:
        prior_role = condition_roles.setdefault(row.condition_id, role)
        if prior_role != role:
            raise ValueError("Round 28 condition crosses roles")
        condition_counts[row.condition_id] = (
            condition_counts.get(row.condition_id, 0) + 1
        )
    samples: list[Round28ModelSample] = []
    base_width = len(POLYMARKET_ROUND27_FEATURE_NAMES)
    for slot_id, role, row in selected_rows:
        target = outcomes_up.get(row.condition_id)
        if type(target) is not int or target not in {0, 1}:
            raise ValueError("Round 28 official outcome population differs")
        samples.append(
            Round28ModelSample(
                slot_id=slot_id,
                role=role,
                condition_id=row.condition_id,
                event_start_ms=row.event_start_ms,
                decision_time_ms=row.decision_time_ms,
                market_prior_probability=row.market_prior_probability,
                base_values=row.values[:base_width],
                augmented_values=row.values,
                target_up=target,
                condition_weight=1.0 / condition_counts[row.condition_id],
                feature_row_sha256=row.row_sha256,
            ).validated()
        )
    if not samples:
        raise ValueError("Round 28 labeled sample population is empty")
    for condition_id in condition_counts:
        total = math.fsum(
            sample.condition_weight
            for sample in samples
            if sample.condition_id == condition_id
        )
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise RuntimeError("Round 28 condition weights differ")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class Round28Partition:
    role: str
    samples: tuple[Round28ModelSample, ...]
    base_features: NDArray[np.float64]
    augmented_features: NDArray[np.float64]
    offsets: NDArray[np.float64]
    targets: NDArray[np.float64]
    weights: NDArray[np.float64]
    conditions: NDArray[np.str_]

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Round28ModelSample],
        *,
        role: str,
    ) -> "Round28Partition":
        selected = tuple(
            sample.validated() for sample in samples if sample.role == role
        )
        if (
            not selected
            or role not in POLYMARKET_ROUND27_ROLE_NAMES
            or role == "purged"
        ):
            raise ValueError("Round 28 partition is empty")
        base = np.asarray([sample.base_values for sample in selected], dtype=np.float64)
        augmented = np.asarray(
            [sample.augmented_values for sample in selected], dtype=np.float64
        )
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
            base.shape != (len(selected), len(POLYMARKET_ROUND27_FEATURE_NAMES))
            or augmented.shape != (len(selected), len(POLYMARKET_ROUND28_FEATURE_NAMES))
            or not np.array_equal(augmented[:, : base.shape[1]], base)
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(augmented))
            or not np.all(np.isfinite(offsets))
            or not np.all(np.isfinite(weights))
            or set(np.unique(targets)) - {0.0, 1.0}
        ):
            raise ValueError("Round 28 partition matrix differs")
        return cls(
            role,
            selected,
            base,
            augmented,
            offsets,
            targets,
            weights,
            conditions,
        )

    def features(self, view: Round28FeatureView) -> NDArray[np.float64]:
        _feature_contract(view)
        return self.base_features if view == "round27_base" else self.augmented_features


def _weighted_roc_auc(
    targets: NDArray[np.float64],
    probability: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    positive_total = float(np.sum(weights[targets == 1.0]))
    negative_total = float(np.sum(weights[targets == 0.0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("Round 28 ROC-AUC class is absent")
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_targets = targets[order]
    sorted_weights = weights[order]
    concordant = 0.0
    cumulative_negative = 0.0
    index = 0
    while index < sorted_probability.size:
        right = index + 1
        while (
            right < sorted_probability.size
            and sorted_probability[right] == sorted_probability[index]
        ):
            right += 1
        group_targets = sorted_targets[index:right]
        group_weights = sorted_weights[index:right]
        positive_weight = float(np.sum(group_weights[group_targets == 1.0]))
        negative_weight = float(np.sum(group_weights[group_targets == 0.0]))
        concordant += positive_weight * (cumulative_negative + 0.5 * negative_weight)
        cumulative_negative += negative_weight
        index = right
    return concordant / (positive_total * negative_total)


@dataclass(frozen=True, slots=True)
class Round28ProbabilityMetrics:
    condition_count: int
    row_count: int
    log_loss: float
    brier_score: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    expected_calibration_error: float

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_METRICS_SCHEMA_VERSION,
            "condition_count": self.condition_count,
            "row_count": self.row_count,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "expected_calibration_error": self.expected_calibration_error,
        }


def round28_probability_metrics(
    partition: Round28Partition,
    predictions: Sequence[float] | NDArray[np.float64],
) -> Round28ProbabilityMetrics:
    probability = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if probability.shape != partition.targets.shape or not np.all(
        np.isfinite(probability)
    ):
        raise ValueError("Round 28 probability population differs")
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
        denominator = float(np.sum(weight[selected]))
        if denominator <= 0.0:
            raise ValueError("Round 28 balanced-accuracy class is absent")
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
    return Round28ProbabilityMetrics(
        condition_count=int(np.unique(partition.conditions).size),
        row_count=int(partition.targets.size),
        log_loss=float(np.sum(weight * losses)),
        brier_score=float(np.sum(weight * brier)),
        accuracy=float(np.sum(weight * (prediction_class == partition.targets))),
        balanced_accuracy=float(math.fsum(recalls) / 2.0),
        roc_auc=_weighted_roc_auc(partition.targets, probability, weight),
        expected_calibration_error=ece,
    )


class Round28ProbabilityModel(Protocol):
    model_name: str
    feature_view: Round28FeatureView

    def predict(
        self,
        features: NDArray[np.float64],
        offsets: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def asdict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class Round28L2OffsetModel:
    feature_view: Round28FeatureView
    feature_names_sha256: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    penalty: float
    correction_scale: float
    model_sha256: str
    model_name: str = "l2_offset_logistic"

    def validated(self) -> "Round28L2OffsetModel":
        feature_names, feature_hash = _feature_contract(self.feature_view)
        body = {
            "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_view": self.feature_view,
            "feature_names_sha256": self.feature_names_sha256,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "penalty": self.penalty,
            "correction_scale": self.correction_scale,
        }
        if (
            self.model_name != "l2_offset_logistic"
            or self.feature_names_sha256 != feature_hash
            or len(self.mean) != len(feature_names)
            or len(self.scale) != len(feature_names)
            or len(self.coefficients) != len(feature_names) + 1
            or any(
                not math.isfinite(value)
                for value in (*self.mean, *self.scale, *self.coefficients)
            )
            or any(value <= 0.0 for value in self.scale)
            or self.penalty not in POLYMARKET_ROUND28_L2_PENALTIES
            or self.correction_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES
            or self.model_sha256 != _canonical_sha256(body)
        ):
            raise ValueError("Round 28 L2 model differs")
        return self

    def predict(
        self,
        features: NDArray[np.float64],
        offsets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.validated()
        selected = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            selected.ndim != 2
            or selected.shape[1] != mean.size
            or coefficients.size != mean.size + 1
        ):
            raise ValueError("Round 28 L2 inference width differs")
        standardized = np.clip((selected - mean) / scale, -12.0, 12.0)
        correction = coefficients[0] + standardized @ coefficients[1:]
        return _sigmoid(
            np.asarray(offsets, dtype=np.float64) + self.correction_scale * correction
        )

    def asdict(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_view": self.feature_view,
            "feature_names_sha256": self.feature_names_sha256,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "penalty": self.penalty,
            "correction_scale": self.correction_scale,
            "model_sha256": self.model_sha256,
        }


def fit_round28_l2_offset(
    partition: Round28Partition,
    *,
    feature_view: Round28FeatureView,
    penalty: float,
    correction_scale: float = 1.0,
) -> Round28L2OffsetModel:
    feature_names, feature_hash = _feature_contract(feature_view)
    selected_penalty = float(penalty)
    selected_correction = float(correction_scale)
    if (
        selected_penalty not in POLYMARKET_ROUND28_L2_PENALTIES
        or selected_correction not in POLYMARKET_ROUND27_CORRECTION_SCALES
    ):
        raise ValueError("Round 28 L2 controls differ")
    features = partition.features(feature_view)
    weights = partition.weights / np.sum(partition.weights)
    mean = np.average(features, axis=0, weights=weights)
    variance = np.average((features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = np.clip((features - mean) / scale, -12.0, 12.0)
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
            raise ValueError("Round 28 L2 system is singular") from exc
        coefficients += delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Round 28 L2 coefficients are non-finite")
    body = {
        "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
        "model_name": "l2_offset_logistic",
        "feature_view": feature_view,
        "feature_names_sha256": feature_hash,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "penalty": selected_penalty,
        "correction_scale": selected_correction,
    }
    if len(mean) != len(feature_names):
        raise RuntimeError("Round 28 L2 feature contract differs")
    return Round28L2OffsetModel(
        feature_view=feature_view,
        feature_names_sha256=feature_hash,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        penalty=selected_penalty,
        correction_scale=selected_correction,
        model_sha256=_canonical_sha256(body),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round28LightGbmOffsetModel:
    feature_view: Round28FeatureView
    feature_names_sha256: str
    model_text: str
    correction_scale: float
    backend_kind: str
    backend_device: str
    model_sha256: str
    model_name: str = "shallow_lightgbm_offset"

    def validated(self) -> "Round28LightGbmOffsetModel":
        feature_names, feature_hash = _feature_contract(self.feature_view)
        model_text_sha256 = hashlib.sha256(self.model_text.encode("utf-8")).hexdigest()
        body = {
            "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_view": self.feature_view,
            "feature_names_sha256": self.feature_names_sha256,
            "correction_scale": self.correction_scale,
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "model_text_sha256": model_text_sha256,
        }
        try:
            booster = lgb.Booster(model_str=self.model_text)
        except Exception as exc:
            raise ValueError("Round 28 LightGBM model text differs") from exc
        if (
            self.model_name != "shallow_lightgbm_offset"
            or self.feature_names_sha256 != feature_hash
            or not self.model_text
            or not self.backend_kind
            or not self.backend_device
            or self.correction_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES
            or booster.num_feature() != len(feature_names)
            or self.model_sha256 != _canonical_sha256(body)
        ):
            raise ValueError("Round 28 LightGBM model differs")
        return self

    def predict(
        self,
        features: NDArray[np.float64],
        offsets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.validated()
        booster = lgb.Booster(model_str=self.model_text)
        if booster.num_feature() != len(_feature_contract(self.feature_view)[0]):
            raise ValueError("Round 28 LightGBM inference width differs")
        correction = np.asarray(
            booster.predict(np.asarray(features, dtype=np.float64), raw_score=True),
            dtype=np.float64,
        )
        return _sigmoid(
            np.asarray(offsets, dtype=np.float64) + self.correction_scale * correction
        )

    def asdict(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
            "model_name": self.model_name,
            "feature_view": self.feature_view,
            "feature_names_sha256": self.feature_names_sha256,
            "correction_scale": self.correction_scale,
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "model_text": self.model_text,
            "model_text_sha256": hashlib.sha256(
                self.model_text.encode("utf-8")
            ).hexdigest(),
            "model_sha256": self.model_sha256,
        }


def fit_round28_lightgbm_offset(
    partition: Round28Partition,
    *,
    feature_view: Round28FeatureView,
    compute_backend: str = "auto",
    seed: int = 2801,
    correction_scale: float = 1.0,
) -> Round28LightGbmOffsetModel:
    feature_names, feature_hash = _feature_contract(feature_view)
    selected_correction = float(correction_scale)
    if selected_correction not in POLYMARKET_ROUND27_CORRECTION_SCALES:
        raise ValueError("Round 28 LightGBM correction scale differs")
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
        partition.features(feature_view),
        label=partition.targets,
        weight=partition.weights,
        init_score=partition.offsets,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    booster = lgb.train(parameters, dataset, num_boost_round=120)
    model_text = booster.model_to_string()
    body = {
        "schema_version": POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION,
        "model_name": "shallow_lightgbm_offset",
        "feature_view": feature_view,
        "feature_names_sha256": feature_hash,
        "correction_scale": selected_correction,
        "backend_kind": kind,
        "backend_device": device,
        "model_text_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
    }
    return Round28LightGbmOffsetModel(
        feature_view=feature_view,
        feature_names_sha256=feature_hash,
        model_text=model_text,
        correction_scale=selected_correction,
        backend_kind=kind,
        backend_device=device,
        model_sha256=_canonical_sha256(body),
    ).validated()


def round28_model_from_payload(
    value: Mapping[str, object],
) -> Round28L2OffsetModel | Round28LightGbmOffsetModel:
    """Reconstruct an exact persisted model and reject any identity drift."""

    payload = dict(value)
    model_name = payload.get("model_name")
    if payload.get("schema_version") != POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION:
        raise ValueError("Round 28 persisted model schema differs")
    if model_name == "l2_offset_logistic":
        expected = {
            "schema_version",
            "model_name",
            "feature_view",
            "feature_names_sha256",
            "mean",
            "scale",
            "coefficients",
            "penalty",
            "correction_scale",
            "model_sha256",
        }
        try:
            model = Round28L2OffsetModel(
                feature_view=str(payload["feature_view"]),  # type: ignore[arg-type]
                feature_names_sha256=str(payload["feature_names_sha256"]),
                mean=tuple(float(item) for item in payload["mean"]),  # type: ignore[union-attr]
                scale=tuple(float(item) for item in payload["scale"]),  # type: ignore[union-attr]
                coefficients=tuple(
                    float(item)
                    for item in payload["coefficients"]  # type: ignore[union-attr]
                ),
                penalty=float(payload["penalty"]),
                correction_scale=float(payload["correction_scale"]),
                model_sha256=str(payload["model_sha256"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 28 persisted L2 model differs") from exc
        if set(payload) != expected:
            raise ValueError("Round 28 persisted L2 model fields differ")
        return model.validated()
    if model_name == "shallow_lightgbm_offset":
        expected = {
            "schema_version",
            "model_name",
            "feature_view",
            "feature_names_sha256",
            "correction_scale",
            "backend_kind",
            "backend_device",
            "model_text",
            "model_text_sha256",
            "model_sha256",
        }
        try:
            model_text = str(payload["model_text"])
            model = Round28LightGbmOffsetModel(
                feature_view=str(payload["feature_view"]),  # type: ignore[arg-type]
                feature_names_sha256=str(payload["feature_names_sha256"]),
                model_text=model_text,
                correction_scale=float(payload["correction_scale"]),
                backend_kind=str(payload["backend_kind"]),
                backend_device=str(payload["backend_device"]),
                model_sha256=str(payload["model_sha256"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 28 persisted LightGBM model differs") from exc
        if (
            set(payload) != expected
            or payload.get("model_text_sha256")
            != hashlib.sha256(model_text.encode("utf-8")).hexdigest()
        ):
            raise ValueError("Round 28 persisted LightGBM model fields differ")
        return model.validated()
    raise ValueError("Round 28 persisted model name differs")


def round28_matched_ablation_report(
    partition: Round28Partition,
    *,
    base_model: Round28ProbabilityModel,
    augmented_model: Round28ProbabilityModel,
) -> dict[str, object]:
    """Compare both views on the exact same rows; this is not an edge claim."""

    if (
        base_model.feature_view != "round27_base"
        or augmented_model.feature_view != "round28_bbo_augmented"
    ):
        raise ValueError("Round 28 matched model views differ")
    base_probability = base_model.predict(
        partition.base_features,
        partition.offsets,
    )
    augmented_probability = augmented_model.predict(
        partition.augmented_features,
        partition.offsets,
    )
    base_metrics = round28_probability_metrics(partition, base_probability)
    augmented_metrics = round28_probability_metrics(partition, augmented_probability)
    identity_chain = hashlib.sha256(
        "\n".join(sample.feature_row_sha256 for sample in partition.samples).encode(
            "ascii"
        )
    ).hexdigest()
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_MATCHED_ABLATION_SCHEMA_VERSION,
        "role": partition.role,
        "matched_sample_identity_sha256": identity_chain,
        "condition_count": base_metrics.condition_count,
        "row_count": base_metrics.row_count,
        "base_model": base_model.asdict(),
        "augmented_model": augmented_model.asdict(),
        "base_metrics": base_metrics.asdict(),
        "augmented_metrics": augmented_metrics.asdict(),
        "augmented_minus_base": {
            "log_loss": augmented_metrics.log_loss - base_metrics.log_loss,
            "brier_score": augmented_metrics.brier_score - base_metrics.brier_score,
            "accuracy": augmented_metrics.accuracy - base_metrics.accuracy,
            "balanced_accuracy": augmented_metrics.balanced_accuracy
            - base_metrics.balanced_accuracy,
            "roc_auc": augmented_metrics.roc_auc - base_metrics.roc_auc,
            "expected_calibration_error": (
                augmented_metrics.expected_calibration_error
                - base_metrics.expected_calibration_error
            ),
        },
        "same_rows_targets_weights_offsets": True,
        "after_cost_economic_evaluation_required": True,
        "statistical_uplift_gate_required": True,
        **_AUTHORITY,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "POLYMARKET_ROUND28_L2_PENALTIES",
    "POLYMARKET_ROUND28_MATCHED_ABLATION_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_METRICS_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_MODEL_SCHEMA_VERSION",
    "Round28FeatureView",
    "Round28L2OffsetModel",
    "Round28LightGbmOffsetModel",
    "Round28ModelSample",
    "Round28Partition",
    "Round28ProbabilityMetrics",
    "Round28ProbabilityModel",
    "build_round28_model_samples",
    "fit_round28_l2_offset",
    "fit_round28_lightgbm_offset",
    "round28_matched_ablation_report",
    "round28_model_from_payload",
    "round28_probability_metrics",
]

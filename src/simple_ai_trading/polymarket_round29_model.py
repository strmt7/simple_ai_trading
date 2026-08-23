"""Matched L2 offset models for Polymarket Round 29 settlement features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

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
)
from .polymarket_round28_model import POLYMARKET_ROUND28_L2_PENALTIES
from .polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_BASE_FEATURE_NAMES,
    POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES,
    POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256,
    Round29FeatureRow,
)


POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION = "polymarket-round29-offset-model-v1"
POLYMARKET_ROUND29_METRICS_SCHEMA_VERSION = "polymarket-round29-probability-metrics-v1"
POLYMARKET_ROUND29_MATCHED_ABLATION_SCHEMA_VERSION = (
    "polymarket-round29-matched-settlement-ablation-v1"
)
POLYMARKET_ROUND29_L2_PENALTIES = POLYMARKET_ROUND28_L2_PENALTIES
Round29PairName = Literal["diagnostic", "primary"]
Round29FeatureView = Literal[
    "round27_base",
    "round29_settlement_augmented",
    "round28_bbo_augmented",
    "round29_bbo_settlement_augmented",
]
_FEATURE_VIEWS: dict[Round29FeatureView, tuple[tuple[str, ...], str]] = {
    "round27_base": (
        POLYMARKET_ROUND27_FEATURE_NAMES,
        POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    ),
    "round29_settlement_augmented": (
        POLYMARKET_ROUND29_BASE_FEATURE_NAMES,
        POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256,
    ),
    "round28_bbo_augmented": (
        POLYMARKET_ROUND28_FEATURE_NAMES,
        POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
    ),
    "round29_bbo_settlement_augmented": (
        POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES,
        POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256,
    ),
}
_PAIR_VIEWS: dict[Round29PairName, tuple[Round29FeatureView, Round29FeatureView]] = {
    "diagnostic": ("round27_base", "round29_settlement_augmented"),
    "primary": (
        "round28_bbo_augmented",
        "round29_bbo_settlement_augmented",
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


def _feature_contract(
    view: Round29FeatureView,
) -> tuple[tuple[str, ...], str]:
    try:
        return _FEATURE_VIEWS[view]
    except KeyError as exc:
        raise ValueError("Round 29 feature view differs") from exc


def _pair_views(
    pair_name: Round29PairName,
) -> tuple[Round29FeatureView, Round29FeatureView]:
    try:
        return _PAIR_VIEWS[pair_name]
    except KeyError as exc:
        raise ValueError("Round 29 model pair differs") from exc


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
class Round29ModelSample:
    slot_id: str
    role: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    diagnostic_base_values: tuple[float, ...]
    diagnostic_augmented_values: tuple[float, ...]
    primary_base_values: tuple[float, ...]
    primary_augmented_values: tuple[float, ...]
    target_up: int
    condition_weight: float
    diagnostic_feature_row_sha256: str
    primary_feature_row_sha256: str

    def validated(self) -> "Round29ModelSample":
        settlement_values = self.diagnostic_augmented_values[
            len(self.diagnostic_base_values) :
        ]
        if (
            not self.slot_id.startswith("stage1-")
            or self.role not in POLYMARKET_ROUND27_ROLE_NAMES
            or self.role == "purged"
            or not self.condition_id.startswith("0x")
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or not 0.0 < self.market_prior_probability < 1.0
            or len(self.diagnostic_base_values) != len(POLYMARKET_ROUND27_FEATURE_NAMES)
            or len(self.diagnostic_augmented_values)
            != len(POLYMARKET_ROUND29_BASE_FEATURE_NAMES)
            or len(self.primary_base_values) != len(POLYMARKET_ROUND28_FEATURE_NAMES)
            or len(self.primary_augmented_values)
            != len(POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES)
            or self.diagnostic_augmented_values[: len(self.diagnostic_base_values)]
            != self.diagnostic_base_values
            or self.primary_base_values[: len(self.diagnostic_base_values)]
            != self.diagnostic_base_values
            or self.primary_augmented_values[: len(self.primary_base_values)]
            != self.primary_base_values
            or self.primary_augmented_values[len(self.primary_base_values) :]
            != settlement_values
            or any(
                not math.isfinite(value)
                for value in (
                    *self.diagnostic_augmented_values,
                    *self.primary_augmented_values,
                )
            )
            or self.target_up not in {0, 1}
            or not math.isfinite(self.condition_weight)
            or self.condition_weight <= 0.0
            or len(self.diagnostic_feature_row_sha256) != 64
            or len(self.primary_feature_row_sha256) != 64
        ):
            raise ValueError("Round 29 model sample differs")
        return self


def _matched_feature_pair(
    diagnostic: Round29FeatureRow,
    primary: Round29FeatureRow,
) -> tuple[Round29FeatureRow, Round29FeatureRow]:
    selected_diagnostic = diagnostic.validated()
    selected_primary = primary.validated()
    identity_diagnostic = (
        selected_diagnostic.run_id,
        selected_diagnostic.condition_id,
        selected_diagnostic.event_start_ms,
        selected_diagnostic.decision_time_ms,
        selected_diagnostic.market_prior_probability,
        selected_diagnostic.base_row_sha256,
        selected_diagnostic.settlement_overlay_row_sha256,
    )
    identity_primary = (
        selected_primary.run_id,
        selected_primary.condition_id,
        selected_primary.event_start_ms,
        selected_primary.decision_time_ms,
        selected_primary.market_prior_probability,
        selected_primary.base_row_sha256,
        selected_primary.settlement_overlay_row_sha256,
    )
    if (
        selected_diagnostic.feature_view != "round29_settlement_augmented"
        or selected_primary.feature_view != "round29_bbo_settlement_augmented"
        or identity_diagnostic != identity_primary
        or selected_primary.maximum_receipt_wall_ms
        < selected_diagnostic.maximum_receipt_wall_ms
        or selected_primary.values[: len(POLYMARKET_ROUND28_FEATURE_NAMES)][
            : len(POLYMARKET_ROUND27_FEATURE_NAMES)
        ]
        != selected_diagnostic.values[: len(POLYMARKET_ROUND27_FEATURE_NAMES)]
        or selected_primary.values[len(POLYMARKET_ROUND28_FEATURE_NAMES) :]
        != selected_diagnostic.values[len(POLYMARKET_ROUND27_FEATURE_NAMES) :]
    ):
        raise ValueError("Round 29 matched feature pair differs")
    return selected_diagnostic, selected_primary


def build_round29_model_samples(
    *,
    rows_by_slot: Mapping[
        str,
        Sequence[tuple[Round29FeatureRow, Round29FeatureRow]],
    ],
    outcomes_up: Mapping[str, int],
    role_intervals: Sequence[Mapping[str, object]],
) -> tuple[Round29ModelSample, ...]:
    """Join outcomes only after both target-blind views are terminal and frozen."""

    intervals = tuple(Round27RoleInterval.from_mapping(item) for item in role_intervals)
    if not intervals or len(set(intervals)) != len(intervals):
        raise ValueError("Round 29 role intervals differ")
    ordered = tuple(sorted(intervals, key=lambda item: (item.start_ms, item.end_ms)))
    if any(
        current.start_ms < previous.end_ms
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("Round 29 role intervals overlap")
    selected_rows: list[tuple[str, str, Round29FeatureRow, Round29FeatureRow]] = []
    keys: set[tuple[str, int]] = set()
    for slot_id, raw_pairs in sorted(rows_by_slot.items()):
        for diagnostic, primary in raw_pairs:
            selected_diagnostic, selected_primary = _matched_feature_pair(
                diagnostic,
                primary,
            )
            key = (
                selected_diagnostic.condition_id,
                selected_diagnostic.decision_time_ms,
            )
            if key in keys:
                raise ValueError("Round 29 feature row is duplicated across slots")
            keys.add(key)
            matches = tuple(
                interval
                for interval in intervals
                if interval.slot_id == slot_id
                and interval.start_ms
                <= selected_diagnostic.event_start_ms
                < interval.end_ms
            )
            if len(matches) != 1:
                raise ValueError("Round 29 feature row role is ambiguous")
            if matches[0].role != "purged":
                selected_rows.append(
                    (slot_id, matches[0].role, selected_diagnostic, selected_primary)
                )
    condition_counts: dict[str, int] = {}
    condition_roles: dict[str, str] = {}
    for _slot_id, role, diagnostic, _primary in selected_rows:
        prior_role = condition_roles.setdefault(diagnostic.condition_id, role)
        if prior_role != role:
            raise ValueError("Round 29 condition crosses roles")
        condition_counts[diagnostic.condition_id] = (
            condition_counts.get(diagnostic.condition_id, 0) + 1
        )
    samples: list[Round29ModelSample] = []
    round27_width = len(POLYMARKET_ROUND27_FEATURE_NAMES)
    round28_width = len(POLYMARKET_ROUND28_FEATURE_NAMES)
    for slot_id, role, diagnostic, primary in selected_rows:
        target = outcomes_up.get(diagnostic.condition_id)
        if type(target) is not int or target not in {0, 1}:
            raise ValueError("Round 29 official outcome population differs")
        samples.append(
            Round29ModelSample(
                slot_id=slot_id,
                role=role,
                condition_id=diagnostic.condition_id,
                event_start_ms=diagnostic.event_start_ms,
                decision_time_ms=diagnostic.decision_time_ms,
                market_prior_probability=diagnostic.market_prior_probability,
                diagnostic_base_values=diagnostic.values[:round27_width],
                diagnostic_augmented_values=diagnostic.values,
                primary_base_values=primary.values[:round28_width],
                primary_augmented_values=primary.values,
                target_up=target,
                condition_weight=1.0 / condition_counts[diagnostic.condition_id],
                diagnostic_feature_row_sha256=diagnostic.row_sha256,
                primary_feature_row_sha256=primary.row_sha256,
            ).validated()
        )
    if not samples:
        raise ValueError("Round 29 labeled sample population is empty")
    for condition_id in condition_counts:
        total = math.fsum(
            sample.condition_weight
            for sample in samples
            if sample.condition_id == condition_id
        )
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise RuntimeError("Round 29 condition weights differ")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class Round29Partition:
    role: str
    samples: tuple[Round29ModelSample, ...]
    diagnostic_base_features: NDArray[np.float64]
    diagnostic_augmented_features: NDArray[np.float64]
    primary_base_features: NDArray[np.float64]
    primary_augmented_features: NDArray[np.float64]
    offsets: NDArray[np.float64]
    targets: NDArray[np.float64]
    weights: NDArray[np.float64]
    conditions: NDArray[np.str_]

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Round29ModelSample],
        *,
        role: str,
    ) -> "Round29Partition":
        selected = tuple(
            sample.validated() for sample in samples if sample.role == role
        )
        if (
            not selected
            or role not in POLYMARKET_ROUND27_ROLE_NAMES
            or role == "purged"
        ):
            raise ValueError("Round 29 partition is empty")
        diagnostic_base = np.asarray(
            [sample.diagnostic_base_values for sample in selected],
            dtype=np.float64,
        )
        diagnostic_augmented = np.asarray(
            [sample.diagnostic_augmented_values for sample in selected],
            dtype=np.float64,
        )
        primary_base = np.asarray(
            [sample.primary_base_values for sample in selected],
            dtype=np.float64,
        )
        primary_augmented = np.asarray(
            [sample.primary_augmented_values for sample in selected],
            dtype=np.float64,
        )
        offsets = np.asarray(
            [_logit(sample.market_prior_probability) for sample in selected],
            dtype=np.float64,
        )
        targets = np.asarray(
            [sample.target_up for sample in selected],
            dtype=np.float64,
        )
        weights = np.asarray(
            [sample.condition_weight for sample in selected],
            dtype=np.float64,
        )
        conditions = np.asarray(
            [sample.condition_id for sample in selected],
            dtype=np.str_,
        )
        matrices = (
            diagnostic_base,
            diagnostic_augmented,
            primary_base,
            primary_augmented,
        )
        expected_widths = tuple(
            len(_feature_contract(view)[0]) for view in _FEATURE_VIEWS
        )
        if (
            tuple(matrix.shape for matrix in matrices)
            != tuple((len(selected), width) for width in expected_widths)
            or not np.array_equal(
                diagnostic_augmented[:, : diagnostic_base.shape[1]],
                diagnostic_base,
            )
            or not np.array_equal(
                primary_base[:, : diagnostic_base.shape[1]],
                diagnostic_base,
            )
            or not np.array_equal(
                primary_augmented[:, : primary_base.shape[1]],
                primary_base,
            )
            or not np.array_equal(
                diagnostic_augmented[:, diagnostic_base.shape[1] :],
                primary_augmented[:, primary_base.shape[1] :],
            )
            or any(not np.all(np.isfinite(matrix)) for matrix in matrices)
            or not np.all(np.isfinite(offsets))
            or not np.all(np.isfinite(weights))
            or set(np.unique(targets)) - {0.0, 1.0}
        ):
            raise ValueError("Round 29 partition matrix differs")
        return cls(
            role,
            selected,
            diagnostic_base,
            diagnostic_augmented,
            primary_base,
            primary_augmented,
            offsets,
            targets,
            weights,
            conditions,
        )

    def features(self, view: Round29FeatureView) -> NDArray[np.float64]:
        _feature_contract(view)
        return {
            "round27_base": self.diagnostic_base_features,
            "round29_settlement_augmented": self.diagnostic_augmented_features,
            "round28_bbo_augmented": self.primary_base_features,
            "round29_bbo_settlement_augmented": self.primary_augmented_features,
        }[view]


def _weighted_roc_auc(
    targets: NDArray[np.float64],
    probability: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    positive_total = float(np.sum(weights[targets == 1.0]))
    negative_total = float(np.sum(weights[targets == 0.0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("Round 29 ROC-AUC class is absent")
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
class Round29ProbabilityMetrics:
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
            "schema_version": POLYMARKET_ROUND29_METRICS_SCHEMA_VERSION,
            "condition_count": self.condition_count,
            "row_count": self.row_count,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "expected_calibration_error": self.expected_calibration_error,
        }


def round29_probability_metrics(
    partition: Round29Partition,
    predictions: Sequence[float] | NDArray[np.float64],
) -> Round29ProbabilityMetrics:
    probability = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if probability.shape != partition.targets.shape or not np.all(
        np.isfinite(probability)
    ):
        raise ValueError("Round 29 probability population differs")
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
            raise ValueError("Round 29 balanced-accuracy class is absent")
        recalls.append(
            float(
                np.sum(weight[selected] * (prediction_class[selected] == bool(target)))
                / denominator
            )
        )
    ece = 0.0
    calibration_bins = np.minimum((probability * 10.0).astype(np.int64), 9)
    for bin_index in range(10):
        selected = calibration_bins == bin_index
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
    return Round29ProbabilityMetrics(
        condition_count=int(np.unique(partition.conditions).size),
        row_count=int(partition.targets.size),
        log_loss=float(np.sum(weight * losses)),
        brier_score=float(np.sum(weight * brier)),
        accuracy=float(np.sum(weight * (prediction_class == partition.targets))),
        balanced_accuracy=float(math.fsum(recalls) / 2.0),
        roc_auc=_weighted_roc_auc(partition.targets, probability, weight),
        expected_calibration_error=ece,
    )


class Round29ProbabilityModel(Protocol):
    model_name: str
    feature_view: Round29FeatureView
    model_sha256: str

    def predict(
        self,
        features: NDArray[np.float64],
        offsets: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def asdict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class Round29L2OffsetModel:
    feature_view: Round29FeatureView
    feature_names_sha256: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    penalty: float
    correction_scale: float
    model_sha256: str
    model_name: str = "l2_offset_logistic"

    def validated(self) -> "Round29L2OffsetModel":
        feature_names, feature_hash = _feature_contract(self.feature_view)
        body = {
            "schema_version": POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION,
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
            or self.penalty not in POLYMARKET_ROUND29_L2_PENALTIES
            or self.correction_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES
            or self.model_sha256 != _canonical_sha256(body)
        ):
            raise ValueError("Round 29 L2 model differs")
        return self

    def predict(
        self,
        features: NDArray[np.float64],
        offsets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.validated()
        selected = np.asarray(features, dtype=np.float64)
        selected_offsets = np.asarray(offsets, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            selected.ndim != 2
            or selected.shape[1] != mean.size
            or selected.shape[0] != selected_offsets.size
            or coefficients.size != mean.size + 1
            or not np.all(np.isfinite(selected))
            or not np.all(np.isfinite(selected_offsets))
        ):
            raise ValueError("Round 29 L2 inference population differs")
        standardized = np.clip((selected - mean) / scale, -12.0, 12.0)
        correction = coefficients[0] + standardized @ coefficients[1:]
        return _sigmoid(selected_offsets + self.correction_scale * correction)

    def asdict(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION,
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


def fit_round29_l2_offset(
    partition: Round29Partition,
    *,
    feature_view: Round29FeatureView,
    penalty: float,
    correction_scale: float = 1.0,
) -> Round29L2OffsetModel:
    feature_names, feature_hash = _feature_contract(feature_view)
    selected_penalty = float(penalty)
    selected_correction = float(correction_scale)
    if (
        selected_penalty not in POLYMARKET_ROUND29_L2_PENALTIES
        or selected_correction not in POLYMARKET_ROUND27_CORRECTION_SCALES
    ):
        raise ValueError("Round 29 L2 controls differ")
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
            raise ValueError("Round 29 L2 system is singular") from exc
        coefficients += delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Round 29 L2 coefficients are non-finite")
    body = {
        "schema_version": POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION,
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
        raise RuntimeError("Round 29 L2 feature contract differs")
    return Round29L2OffsetModel(
        feature_view=feature_view,
        feature_names_sha256=feature_hash,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        penalty=selected_penalty,
        correction_scale=selected_correction,
        model_sha256=_canonical_sha256(body),
    ).validated()


def round29_model_from_payload(
    value: Mapping[str, object],
) -> Round29L2OffsetModel:
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
        model = Round29L2OffsetModel(
            feature_view=str(value["feature_view"]),  # type: ignore[arg-type]
            feature_names_sha256=str(value["feature_names_sha256"]),
            mean=tuple(float(item) for item in value["mean"]),  # type: ignore[arg-type]
            scale=tuple(float(item) for item in value["scale"]),  # type: ignore[arg-type]
            coefficients=tuple(
                float(item)
                for item in value["coefficients"]  # type: ignore[arg-type]
            ),
            penalty=float(value["penalty"]),
            correction_scale=float(value["correction_scale"]),
            model_sha256=str(value["model_sha256"]),
            model_name=str(value["model_name"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 29 persisted L2 model differs") from exc
    if set(value) != expected or value.get("schema_version") != (
        POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("Round 29 persisted L2 model fields differ")
    return model.validated()


def round29_matched_ablation_report(
    partition: Round29Partition,
    *,
    pair_name: Round29PairName,
    base_model: Round29ProbabilityModel,
    augmented_model: Round29ProbabilityModel,
) -> dict[str, object]:
    """Compare one preregistered pair on exactly matched rows."""

    base_view, augmented_view = _pair_views(pair_name)
    if (
        base_model.feature_view != base_view
        or augmented_model.feature_view != augmented_view
        or base_model.model_name != augmented_model.model_name
    ):
        raise ValueError("Round 29 matched model views differ")
    base_probability = base_model.predict(
        partition.features(base_view),
        partition.offsets,
    )
    augmented_probability = augmented_model.predict(
        partition.features(augmented_view),
        partition.offsets,
    )
    base_metrics = round29_probability_metrics(partition, base_probability)
    augmented_metrics = round29_probability_metrics(partition, augmented_probability)
    identity_chain = hashlib.sha256(
        "\n".join(
            f"{sample.diagnostic_feature_row_sha256}:"
            f"{sample.primary_feature_row_sha256}"
            for sample in partition.samples
        ).encode("ascii")
    ).hexdigest()
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_MATCHED_ABLATION_SCHEMA_VERSION,
        "pair_name": pair_name,
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
            "balanced_accuracy": (
                augmented_metrics.balanced_accuracy - base_metrics.balanced_accuracy
            ),
            "roc_auc": augmented_metrics.roc_auc - base_metrics.roc_auc,
            "expected_calibration_error": (
                augmented_metrics.expected_calibration_error
                - base_metrics.expected_calibration_error
            ),
        },
        "same_rows_targets_weights_offsets": True,
        "promotion_controlling_pair": pair_name == "primary",
        "after_cost_economic_evaluation_required": pair_name == "primary",
        "statistical_uplift_gate_required": True,
        **_AUTHORITY,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "POLYMARKET_ROUND29_L2_PENALTIES",
    "POLYMARKET_ROUND29_MATCHED_ABLATION_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_METRICS_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_MODEL_SCHEMA_VERSION",
    "Round29FeatureView",
    "Round29L2OffsetModel",
    "Round29ModelSample",
    "Round29PairName",
    "Round29Partition",
    "Round29ProbabilityMetrics",
    "Round29ProbabilityModel",
    "build_round29_model_samples",
    "fit_round29_l2_offset",
    "round29_matched_ablation_report",
    "round29_model_from_payload",
    "round29_probability_metrics",
]

"""Pure-stdlib training and inference utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Iterable, List, Sequence, Tuple

from .compute import BackendInfo, resolve_backend
from .features import FEATURE_VERSION, ModelRow, feature_dimension as _feature_dimension
from .storage import write_json_atomic
from .strategy_overrides import StrategyOverrideValue, clean_strategy_overrides


def feature_dimension(enabled_features: Sequence[str] | None = None) -> int:
    return _feature_dimension(enabled_features)


def _clamp(x: float, low: float, high: float) -> float:
    if not math.isfinite(x):
        return low
    return low if x < low else (high if x > high else x)


class ModelLoadError(ValueError):
    """Raised when a model artifact is invalid or incompatible."""


class ModelFeatureMismatchError(ModelLoadError):
    """Raised when feature metadata diverges between model and current runtime."""


@dataclass
class EnsembleMember:
    weights: List[float]
    bias: float
    feature_means: List[float]
    feature_stds: List[float]
    seed: int = 7
    epochs: int = 0
    best_epoch: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None


@dataclass
class HybridPrototype:
    features: List[float]
    label: int
    timestamp: int = 0
    close: float = 0.0


@dataclass
class HybridExpert:
    name: str
    kind: str
    weight: float
    prototypes: List[HybridPrototype] = field(default_factory=list)
    k: int = 21
    bandwidth: float = 1.0
    alpha: float = 1.0
    feature_count: int = 13
    notes: str = ""


@dataclass
class TrainedModel:
    weights: List[float]
    bias: float
    feature_dim: int
    epochs: int
    feature_means: List[float]
    feature_stds: List[float]
    feature_version: str = FEATURE_VERSION
    feature_signature: str | None = None
    learning_rate: float = 0.05
    l2_penalty: float = 1e-4
    seed: int = 7
    class_weight_pos: float = 1.0
    class_weight_neg: float = 1.0
    decision_threshold: float | None = None
    long_decision_threshold: float | None = None
    short_decision_threshold: float | None = None
    calibration_size: int = 0
    validation_size: int = 0
    training_cutoff_timestamp: int | None = None
    best_epoch: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    quality_score: float | None = None
    quality_warnings: List[str] = field(default_factory=list)
    probability_temperature: float = 1.0
    probability_inverted: bool = False
    probability_calibration_size: int = 0
    probability_log_loss_before: float | None = None
    probability_log_loss_after: float | None = None
    probability_brier_before: float | None = None
    probability_brier_after: float | None = None
    probability_ece_before: float | None = None
    probability_ece_after: float | None = None
    probability_calibration_backend_requested: str = "cpu"
    probability_calibration_backend_kind: str = "cpu"
    probability_calibration_backend_device: str = "cpu"
    probability_calibration_backend_reason: str = ""
    threshold_source: str | None = None
    threshold_calibration_score: float | None = None
    threshold_calibration_pnl: float | None = None
    threshold_calibration_trades: int = 0
    threshold_diagnostic_best_threshold: float | None = None
    threshold_diagnostic_best_score: float | None = None
    threshold_diagnostic_best_pnl: float | None = None
    threshold_diagnostic_best_trades: int = 0
    threshold_diagnostic_best_long_threshold: float | None = None
    threshold_diagnostic_best_short_threshold: float | None = None
    strategy_overrides: dict[str, StrategyOverrideValue] = field(default_factory=dict)
    ensemble_members: List[EnsembleMember] = field(default_factory=list)
    training_backend_requested: str = "cpu"
    training_backend_kind: str = "cpu"
    training_backend_device: str = "cpu"
    training_backend_vendor: str = "Python stdlib"
    training_backend_reason: str = ""
    model_family: str = "advanced_logistic"
    model_candidate_count: int = 1
    model_selected_candidate: str = "default"
    model_selection_score: float | None = None
    round_candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    hybrid_base_weight: float = 1.0
    hybrid_experts: List[HybridExpert] = field(default_factory=list)
    meta_label_policy: dict[str, Any] = field(default_factory=dict)
    selection_risk: dict[str, Any] = field(default_factory=dict)
    execution_validation: dict[str, Any] = field(default_factory=dict)

    def _normalize(self, features: Tuple[float, ...]) -> Tuple[float, ...]:
        if len(features) != self.feature_dim:
            raise ValueError("Feature dimension does not match this model")
        if not self.feature_stds:
            return features
        return tuple(
            (x - mean_) / std_ if std_ != 0 else (x - mean_)
            for x, mean_, std_ in zip(features, self.feature_means, self.feature_stds, strict=True)
        )

    def _linear_score(self, features: Tuple[float, ...]) -> float:
        score = self.bias
        for weight, value in zip(self.weights, self._normalize(features), strict=True):
            score += weight * value
        return max(-50.0, min(50.0, score))

    def _member_probability(self, member: EnsembleMember, features: Tuple[float, ...]) -> float:
        score = _linear_score_with_stats(
            weights=member.weights,
            bias=member.bias,
            means=member.feature_means,
            stds=member.feature_stds,
            features=features,
            feature_dim=self.feature_dim,
        )
        return _sigmoid(_temperature_scaled_score(score, self.probability_temperature))

    def _base_probability(self, features: Tuple[float, ...]) -> float:
        if self.ensemble_members:
            probabilities = [self._member_probability(member, features) for member in self.ensemble_members]
            return sum(probabilities) / len(probabilities)
        score = _temperature_scaled_score(self._linear_score(features), self.probability_temperature)
        return _sigmoid(score)

    def _normalized_for_expert(self, features: Tuple[float, ...]) -> list[float]:
        return list(self._normalize(features))

    def _lorentzian_probability(self, expert: HybridExpert, features: Tuple[float, ...]) -> float | None:
        if not expert.prototypes:
            return None
        normalized = self._normalized_for_expert(features)
        distances: list[tuple[float, int]] = []
        for prototype in expert.prototypes:
            if len(prototype.features) != self.feature_dim:
                continue
            distance = 0.0
            for left, right in zip(normalized, prototype.features, strict=True):
                distance += math.log1p(abs(left - right))
            distances.append((distance / max(1, self.feature_dim), int(prototype.label)))
        if not distances:
            return None
        distances.sort(key=lambda item: item[0])
        neighbors = distances[: max(1, int(expert.k))]
        weighted_positive = 0.0
        total = 0.0
        for distance, label in neighbors:
            weight = 1.0 / max(1e-9, distance + 1e-6)
            weighted_positive += weight * float(1 if label else 0)
            total += weight
        return _clamp(weighted_positive / total if total else 0.5, 0.0, 1.0)

    def _kernel_probability(self, expert: HybridExpert, features: Tuple[float, ...]) -> float | None:
        if not expert.prototypes:
            return None
        normalized = self._normalized_for_expert(features)
        bandwidth = max(1e-6, float(expert.bandwidth))
        alpha = max(1e-6, float(expert.alpha))
        weighted_positive = 0.0
        total = 0.0
        for prototype in expert.prototypes:
            if len(prototype.features) != self.feature_dim:
                continue
            squared = 0.0
            for left, right in zip(normalized, prototype.features, strict=True):
                delta = left - right
                squared += delta * delta
            scaled = squared / max(1, self.feature_dim)
            weight = (1.0 + scaled / (2.0 * alpha * bandwidth * bandwidth)) ** (-alpha)
            weighted_positive += weight * float(1 if prototype.label else 0)
            total += weight
        return _clamp(weighted_positive / total if total else 0.5, 0.0, 1.0)

    def _technical_probability(self, expert: HybridExpert, features: Tuple[float, ...]) -> float | None:
        if not features:
            return None
        values = list(features[: max(1, min(int(expert.feature_count), len(features)))])
        while len(values) < 13:
            values.append(0.0)
        momentum_1, momentum_3, momentum_10, momentum_20 = values[0], values[1], values[2], values[3]
        ema_spread = values[4]
        rsi = _clamp(values[5], 0.0, 1.0)
        ema_gap = values[6]
        relative_atr = abs(values[7])
        volatility_20 = abs(values[8])
        volume_ratio = values[9]
        trend_acceleration = values[10]
        gap_to_vwap = values[11]
        volume_trend = values[12]

        trend = (
            0.24 * math.tanh(momentum_20 * 80.0)
            + 0.20 * math.tanh(momentum_10 * 100.0)
            + 0.14 * math.tanh(momentum_3 * 140.0)
            - 0.16 * math.tanh(ema_spread * 90.0)
            + 0.10 * math.tanh(trend_acceleration * 240.0)
            + 0.06 * math.tanh(volume_trend * 4.0)
        )
        mean_reversion = (
            0.18 * math.tanh((0.38 - rsi) * 5.0)
            - 0.14 * math.tanh(gap_to_vwap * 150.0)
            - 0.08 * math.tanh(momentum_1 * 180.0)
        )
        breakout = (
            0.10 * math.tanh(volume_ratio * 2.5)
            + 0.10 * math.tanh((relative_atr + volatility_20) * 80.0)
            + 0.08 * math.tanh((momentum_10 + momentum_20) * 80.0)
            - 0.04 * math.tanh(abs(ema_gap) * 150.0)
        )
        score = trend + mean_reversion + breakout
        return _clamp(_sigmoid(score * 2.2), 0.0, 1.0)

    def _expert_probability(self, expert: HybridExpert, features: Tuple[float, ...]) -> float | None:
        if expert.kind == "lorentzian_knn":
            return self._lorentzian_probability(expert, features)
        if expert.kind == "rational_quadratic_kernel":
            return self._kernel_probability(expert, features)
        if expert.kind == "technical_confluence":
            return self._technical_probability(expert, features)
        return None

    def predict_proba(self, features: Tuple[float, ...]) -> float:
        base_probability = self._base_probability(features)
        if not self.hybrid_experts:
            return _clamp(1.0 - base_probability if self.probability_inverted else base_probability, 0.0, 1.0)
        base_weight = max(0.0, float(self.hybrid_base_weight))
        weighted = base_probability * base_weight
        total = base_weight
        for expert in self.hybrid_experts:
            expert_weight = max(0.0, float(expert.weight))
            if expert_weight <= 0.0:
                continue
            probability = self._expert_probability(expert, features)
            if probability is None:
                continue
            weighted += expert_weight * probability
            total += expert_weight
        probability = weighted / total if total > 0.0 else base_probability
        return _clamp(1.0 - probability if self.probability_inverted else probability, 0.0, 1.0)

    def predict(self, features: Tuple[float, ...], threshold: float) -> int:
        threshold = _clamp(threshold, 0.0, 1.0)
        return int(self.predict_proba(features) >= threshold)


def _linear_score_with_stats(
    *,
    weights: Sequence[float],
    bias: float,
    means: Sequence[float],
    stds: Sequence[float],
    features: Tuple[float, ...],
    feature_dim: int,
) -> float:
    if len(features) != feature_dim:
        raise ValueError("Feature dimension does not match this model")
    score = float(bias)
    for weight, value, mean_, std_ in zip(weights, features, means, stds, strict=True):
        normalized = (value - mean_) / std_ if std_ != 0 else (value - mean_)
        score += weight * normalized
    return max(-50.0, min(50.0, score))


def _collect_feature_stats(rows: Iterable[ModelRow]) -> tuple[List[float], List[float]]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No rows to collect statistics")
    dim = len(rows_list[0].features)
    means = [0.0] * dim
    stds = [1.0] * dim

    columns = list(zip(*[r.features for r in rows_list], strict=True))
    for i, col in enumerate(columns):
        m = mean(col)
        s = pstdev(col)
        means[i] = float(m)
        stds[i] = float(s if s and abs(s) > 1e-12 else 1.0)
    return means, stds


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class ProbabilityStats:
    minimum: float
    maximum: float
    mean: float
    std: float

    def asdict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelQualityReport:
    quality_score: float
    status: str
    warnings: List[str]
    train_accuracy: float
    validation_accuracy: float
    validation_f1: float
    validation_majority_baseline: float
    train_validation_gap: float
    validation_rows: int
    validation_positive_rate: float
    probability_stats: ProbabilityStats

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["probability_stats"] = self.probability_stats.asdict()
        return payload


@dataclass(frozen=True)
class ProbabilityCalibrationReport:
    status: str
    warnings: List[str]
    rows: int
    temperature: float
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    expected_calibration_error_before: float
    expected_calibration_error_after: float
    improved: bool
    calibration_backend_requested: str = "cpu"
    calibration_backend_kind: str = "cpu"
    calibration_backend_device: str = "cpu"
    calibration_backend_reason: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDriftReport:
    status: str
    warnings: List[str]
    rows: int
    feature_dim: int
    max_abs_z: float
    mean_abs_z: float
    outlier_fraction: float
    warn_threshold: float
    fail_threshold: float
    mean_warn_threshold: float
    mean_fail_threshold: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalValidationSplit:
    train_rows: List[ModelRow]
    calibration_rows: List[ModelRow]
    validation_rows: List[ModelRow]


def _safe_division(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_model_rows(
    rows: List[ModelRow],
    *,
    label: str = "training rows",
    expected_feature_dim: int | None = None,
) -> int:
    """Validate feature rows before fitting or scoring.

    The training code intentionally accepts light duck-typed row objects in
    tests and integrations, so this guard focuses on the contract actually used
    by the model: fixed finite feature vectors plus binary labels.
    """

    if not rows:
        raise ValueError(f"No {label} available")
    try:
        feature_dim = len(rows[0].features)
    except (AttributeError, TypeError):
        raise ValueError(f"{label} row 0 is missing features") from None
    if feature_dim == 0:
        raise ValueError("Rows must contain at least one feature")
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"{label} feature dimension mismatch: row 0 has {feature_dim}, expected {expected_feature_dim}"
        )

    for index, row in enumerate(rows):
        try:
            features = tuple(row.features)
        except (AttributeError, TypeError):
            raise ValueError(f"{label} row {index} is missing features") from None
        if len(features) != feature_dim:
            raise ValueError(
                f"{label} row {index} feature dimension mismatch: "
                f"got {len(features)}, expected {feature_dim}"
            )
        for value_index, value in enumerate(features):
            try:
                feature_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{label} row {index} feature {value_index} is not numeric") from None
            if not math.isfinite(feature_value):
                raise ValueError(f"{label} row {index} feature {value_index} is not finite")
        try:
            row_label = int(row.label)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f"{label} row {index} label is not binary") from None
        if row_label not in {0, 1}:
            raise ValueError(f"{label} row {index} label is not binary")
    return feature_dim


def _normalize_rows(rows: List[ModelRow], means: List[float], stds: List[float]) -> List[Tuple[float, ...]]:
    return [tuple((x - m) / s for x, m, s in zip(r.features, means, stds, strict=True)) for r in rows]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _class_weights(rows: List[ModelRow]) -> tuple[float, float]:
    positives = sum(1 for row in rows if row.label == 1)
    negatives = len(rows) - positives
    if positives == 0:
        return 1.0, 1.0
    if negatives == 0:
        return 1.0, 1.0
    total = len(rows)
    pos_weight = float(negatives) / float(total)
    neg_weight = float(positives) / float(total)
    return pos_weight, neg_weight


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / denom


def _confusion(rows: List[ModelRow], model: TrainedModel, threshold: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = model.predict(row.features, threshold)
        label = row.label
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _log_loss(
    rows: List[ModelRow],
    weights: List[float],
    bias: float,
    means: List[float],
    stds: List[float],
) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        score = bias
        for weight, value, mean_, std_ in zip(weights, row.features, means, stds, strict=True):
            normalized = (value - mean_) / std_ if std_ != 0 else (value - mean_)
            score += weight * normalized
        probability = _clamp(_sigmoid(score), 1e-12, 1.0 - 1e-12)
        if int(row.label) == 1:
            total -= math.log(probability)
        else:
            total -= math.log(1.0 - probability)
    return total / len(rows)


def _coerce_temperature(temperature: Any) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 1e-6:
        return 1.0
    return value


def _temperature_scaled_score(score: float, temperature: object) -> float:
    value = _coerce_temperature(temperature)
    return max(-50.0, min(50.0, score / value))


def _member_probability_with_temperature(
    model: TrainedModel,
    member: EnsembleMember,
    features: Tuple[float, ...],
    *,
    temperature: float,
) -> float:
    score = _linear_score_with_stats(
        weights=member.weights,
        bias=member.bias,
        means=member.feature_means,
        stds=member.feature_stds,
        features=features,
        feature_dim=model.feature_dim,
    )
    return _sigmoid(_temperature_scaled_score(score, temperature))


def _model_probability(model: TrainedModel, features: Tuple[float, ...], *, temperature: float | None = None) -> float:
    chosen_temperature = model.probability_temperature if temperature is None else temperature
    if model.ensemble_members:
        probabilities = [
            _member_probability_with_temperature(model, member, features, temperature=chosen_temperature)
            for member in model.ensemble_members
        ]
        return sum(probabilities) / len(probabilities)
    return _sigmoid(_temperature_scaled_score(model._linear_score(features), chosen_temperature))


def _model_log_loss(rows: List[ModelRow], model: TrainedModel, *, temperature: float | None = None) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probability = _clamp(_model_probability(model, row.features, temperature=temperature), 1e-12, 1.0 - 1e-12)
        if int(row.label) == 1:
            total -= math.log(probability)
        else:
            total -= math.log(1.0 - probability)
    return total / len(rows)


def _brier_score(rows: List[ModelRow], model: TrainedModel, *, temperature: float | None = None) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probability = _model_probability(model, row.features, temperature=temperature)
        total += (probability - float(int(row.label))) ** 2
    return total / len(rows)


def _expected_calibration_error(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    temperature: float | None = None,
    bins: int = 5,
) -> float:
    if not rows:
        return 0.0
    bins = max(1, int(bins))
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row in rows:
        probability = _clamp(_model_probability(model, row.features, temperature=temperature), 0.0, 1.0)
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, int(row.label)))
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_probability = mean(probability for probability, _label in bucket)
        positive_rate = mean(label for _probability, label in bucket)
        error += (len(bucket) / len(rows)) * abs(avg_probability - positive_rate)
    return float(error)


def assess_probability_calibration(rows: List[ModelRow], model: TrainedModel) -> ProbabilityCalibrationReport:
    """Measure current probability reliability without changing the model."""

    if not rows:
        return ProbabilityCalibrationReport(
            status="fail",
            warnings=["no rows available for probability calibration report"],
            rows=0,
            temperature=_coerce_temperature(getattr(model, "probability_temperature", 1.0)),
            log_loss_before=0.0,
            log_loss_after=0.0,
            brier_before=0.0,
            brier_after=0.0,
            expected_calibration_error_before=0.0,
            expected_calibration_error_after=0.0,
            improved=False,
            calibration_backend_requested="cpu",
            calibration_backend_kind="cpu",
            calibration_backend_device="cpu",
            calibration_backend_reason="no calibration rows",
        )
    validate_model_rows(rows, label="probability calibration rows", expected_feature_dim=model.feature_dim)
    temperature = _coerce_temperature(getattr(model, "probability_temperature", 1.0))
    log_loss_before = _model_log_loss(rows, model, temperature=1.0)
    log_loss_after = _model_log_loss(rows, model, temperature=temperature)
    brier_before = _brier_score(rows, model, temperature=1.0)
    brier_after = _brier_score(rows, model, temperature=temperature)
    ece_before = _expected_calibration_error(rows, model, temperature=1.0)
    ece_after = _expected_calibration_error(rows, model, temperature=temperature)
    improved = (
        abs(temperature - 1.0) > 1e-12
        and (
            log_loss_after < log_loss_before - 1e-6
            or brier_after < brier_before - 1e-6
        )
    )
    warnings: List[str] = []
    if len(rows) < 20:
        warnings.append("probability calibration sample has fewer than 20 rows")
    if rows and _positive_rate(rows) in {0.0, 1.0}:
        warnings.append("probability calibration labels contain only one class")
    return ProbabilityCalibrationReport(
        status="warn" if warnings else "ok",
        warnings=warnings,
        rows=len(rows),
        temperature=temperature,
        log_loss_before=float(log_loss_before),
        log_loss_after=float(log_loss_after),
        brier_before=float(brier_before),
        brier_after=float(brier_after),
        expected_calibration_error_before=float(ece_before),
        expected_calibration_error_after=float(ece_after),
        improved=bool(improved),
        calibration_backend_requested="cpu",
        calibration_backend_kind="cpu",
        calibration_backend_device="cpu",
        calibration_backend_reason="stdlib probability calibration assessment",
    )


def _temperature_scan_torch(  # pragma: no cover - exercised by host GPU smoke verification
    rows: List[ModelRow],
    model: TrainedModel,
    candidates: Sequence[float],
    *,
    backend: BackendInfo,
    batch_size: int,
) -> tuple[float, float, float]:
    import torch  # type: ignore

    if not candidates:
        raise ValueError("No temperature candidates to scan")
    device = _torch_device_for_backend(backend)
    batch = max(1, int(batch_size or 8192))
    temperatures = torch.tensor([float(value) for value in candidates], dtype=torch.float32, device=device)
    log_loss_sums = torch.zeros((len(candidates),), dtype=torch.float32, device=device)
    brier_sums = torch.zeros((len(candidates),), dtype=torch.float32, device=device)

    members = list(model.ensemble_members)
    model_specs = []
    if members:
        for member in members:
            model_specs.append((member.weights, member.bias, member.feature_means, member.feature_stds))
    else:
        model_specs.append((model.weights, model.bias, model.feature_means, model.feature_stds))

    model_tensors = []
    for weights, bias, means, stds in model_specs:
        weight_t = torch.tensor(list(weights), dtype=torch.float32, device=device)
        bias_t = torch.tensor(float(bias), dtype=torch.float32, device=device)
        mean_t = torch.tensor(list(means), dtype=torch.float32, device=device)
        std_t = torch.tensor(list(stds), dtype=torch.float32, device=device)
        std_t = torch.where(torch.abs(std_t) > 0.0, std_t, torch.ones_like(std_t))
        model_tensors.append((weight_t, bias_t, mean_t, std_t))

    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        features = torch.tensor([row.features for row in chunk], dtype=torch.float32, device=device)
        labels = torch.tensor(
            [float(1 if int(row.label) else 0) for row in chunk],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1)
        probabilities = None
        for weight_t, bias_t, mean_t, std_t in model_tensors:
            normalized = (features - mean_t) / std_t
            logits = normalized.matmul(weight_t.reshape(-1, 1)).reshape(-1) + bias_t
            logits_by_temperature = torch.clamp(
                logits.reshape(-1, 1) / temperatures.reshape(1, -1),
                min=-50.0,
                max=50.0,
            )
            member_probabilities = torch.sigmoid(logits_by_temperature)
            probabilities = (
                member_probabilities
                if probabilities is None
                else probabilities + member_probabilities
            )
        if probabilities is None:  # pragma: no cover - model_tensors is always populated above
            raise RuntimeError("No probabilities were produced for temperature calibration")
        if len(model_tensors) > 1:
            probabilities = probabilities / float(len(model_tensors))
        probabilities = torch.clamp(probabilities, min=1e-6, max=1.0 - 1e-6)
        log_loss_sums = log_loss_sums + (
            -(labels * torch.log(probabilities) + (1.0 - labels) * torch.log(1.0 - probabilities))
        ).sum(dim=0)
        brier_sums = brier_sums + ((probabilities - labels) * (probabilities - labels)).sum(dim=0)

    row_count = max(1, len(rows))
    log_losses = [float(value) for value in (log_loss_sums / float(row_count)).detach().cpu().tolist()]
    briers = [float(value) for value in (brier_sums / float(row_count)).detach().cpu().tolist()]
    best_index = min(range(len(candidates)), key=lambda index: (log_losses[index], briers[index]))
    return float(candidates[best_index]), float(log_losses[best_index]), float(briers[best_index])


def calibrate_probability_temperature(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    min_temperature: float = 0.5,
    max_temperature: float = 5.0,
    steps: int = 46,
    compute_backend: str | None = None,
    batch_size: int = 8192,
) -> ProbabilityCalibrationReport:
    """Fit a one-parameter temperature calibrator on held-out rows."""

    base = assess_probability_calibration(rows, model)
    if base.status == "fail":
        return base
    warnings = list(base.warnings)
    if _positive_rate(rows) in {0.0, 1.0}:
        return ProbabilityCalibrationReport(
            status="fail",
            warnings=warnings,
            rows=base.rows,
            temperature=base.temperature,
            log_loss_before=base.log_loss_before,
            log_loss_after=base.log_loss_after,
            brier_before=base.brier_before,
            brier_after=base.brier_after,
            expected_calibration_error_before=base.expected_calibration_error_before,
            expected_calibration_error_after=base.expected_calibration_error_after,
            improved=False,
            calibration_backend_requested=base.calibration_backend_requested,
            calibration_backend_kind=base.calibration_backend_kind,
            calibration_backend_device=base.calibration_backend_device,
            calibration_backend_reason=base.calibration_backend_reason,
        )

    low = max(0.05, float(min_temperature))
    high = max(low, float(max_temperature))
    steps = max(2, int(steps))
    candidates = [low + (high - low) * i / (steps - 1) for i in range(steps)]
    current = _coerce_temperature(getattr(model, "probability_temperature", 1.0))
    if current not in candidates:
        candidates.append(current)
    best_temperature = current
    best_log_loss = base.log_loss_before
    best_brier = base.brier_before
    best_ece = base.expected_calibration_error_before
    backend = resolve_backend(effective_training_backend_name(compute_backend))
    if backend.kind != "cpu":
        try:
            best_temperature, best_log_loss, best_brier = _temperature_scan_torch(
                rows,
                model,
                candidates,
                backend=backend,
                batch_size=batch_size,
            )
            best_ece = _expected_calibration_error(rows, model, temperature=best_temperature)
        except Exception as exc:
            backend = _fallback_backend(
                backend,
                f"{backend.kind} temperature calibration failed ({exc.__class__.__name__}); fell back to CPU",
            )

    if backend.kind == "cpu":
        for temperature in candidates:
            log_loss = _model_log_loss(rows, model, temperature=temperature)
            brier = _brier_score(rows, model, temperature=temperature)
            if log_loss < best_log_loss - 1e-12 or (abs(log_loss - best_log_loss) <= 1e-12 and brier < best_brier):
                best_temperature = float(temperature)
                best_log_loss = float(log_loss)
                best_brier = float(brier)
                best_ece = _expected_calibration_error(rows, model, temperature=temperature)

    improved = best_log_loss < base.log_loss_before - 1e-6 or best_brier < base.brier_before - 1e-6
    if not improved:
        warnings.append("temperature calibration did not improve held-out probability loss")
        best_temperature = current
        best_log_loss = base.log_loss_before
        best_brier = base.brier_before
        best_ece = base.expected_calibration_error_before
        if backend.kind != "cpu":
            best_ece = _expected_calibration_error(rows, model, temperature=best_temperature)
    return ProbabilityCalibrationReport(
        status="warn" if warnings else "ok",
        warnings=warnings,
        rows=base.rows,
        temperature=float(best_temperature),
        log_loss_before=base.log_loss_before,
        log_loss_after=float(best_log_loss),
        brier_before=base.brier_before,
        brier_after=float(best_brier),
        expected_calibration_error_before=base.expected_calibration_error_before,
        expected_calibration_error_after=float(best_ece),
        improved=bool(improved),
        calibration_backend_requested=backend.requested,
        calibration_backend_kind=backend.kind,
        calibration_backend_device=backend.device,
        calibration_backend_reason=backend.reason,
    )


def _positive_rate(rows: List[ModelRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.label) == 1) / len(rows)


def _probability_stats(rows: List[ModelRow], model: TrainedModel) -> ProbabilityStats:
    if not rows:
        return ProbabilityStats(0.0, 0.0, 0.0, 0.0)
    probabilities = [model.predict_proba(row.features) for row in rows]
    return ProbabilityStats(
        minimum=min(probabilities),
        maximum=max(probabilities),
        mean=mean(probabilities),
        std=pstdev(probabilities) if len(probabilities) > 1 else 0.0,
    )


def _majority_baseline(rows: List[ModelRow]) -> float:
    if not rows:
        return 0.0
    positives = sum(1 for row in rows if int(row.label) == 1)
    negatives = len(rows) - positives
    return max(positives, negatives) / len(rows)


def build_model_quality_report(
    train_rows: List[ModelRow],
    validation_rows: List[ModelRow],
    model: TrainedModel,
    threshold: float,
) -> ModelQualityReport:
    """Summarize whether a fitted model is credible enough for operator review."""

    if train_rows:
        validate_model_rows(train_rows, label="quality train rows", expected_feature_dim=model.feature_dim)
    if validation_rows:
        validate_model_rows(validation_rows, label="quality validation rows", expected_feature_dim=model.feature_dim)

    train_report = evaluate_classification(train_rows, model, threshold=threshold) if train_rows else None
    validation_report = evaluate_classification(validation_rows, model, threshold=threshold)
    baseline = _majority_baseline(validation_rows)
    probability_stats = _probability_stats(validation_rows, model)
    validation_positive_rate = _positive_rate(validation_rows)
    train_accuracy = float(train_report.accuracy) if train_report is not None else 0.0
    validation_accuracy = float(validation_report.accuracy)
    gap = max(0.0, train_accuracy - validation_accuracy)
    warnings: List[str] = []
    penalty = 0.0

    if len(validation_rows) < 20:
        warnings.append("validation sample has fewer than 20 rows")
        penalty += 0.20
    if validation_rows and validation_positive_rate in {0.0, 1.0}:
        warnings.append("validation labels contain only one class")
        penalty += 0.20
    if validation_rows and validation_accuracy + 0.02 < baseline:
        warnings.append("validation accuracy is below majority-class baseline")
        penalty += 0.25
    if gap > 0.25:
        warnings.append("train/validation accuracy gap suggests overfitting")
        penalty += min(0.25, gap - 0.25)
    if validation_rows and probability_stats.std < 0.01:
        warnings.append("validation probabilities are nearly constant")
        penalty += 0.15
    if validation_rows and validation_report.f1 == 0.0 and 0.0 < validation_positive_rate < 1.0:
        warnings.append("validation F1 is zero despite mixed labels")
        penalty += 0.20

    quality_score = _clamp(1.0 - penalty, 0.0, 1.0)
    if quality_score >= 0.75 and not warnings:
        status = "ok"
    elif quality_score >= 0.45:
        status = "warn"
    else:
        status = "fail"
    return ModelQualityReport(
        quality_score=quality_score,
        status=status,
        warnings=warnings,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        validation_f1=float(validation_report.f1),
        validation_majority_baseline=float(baseline),
        train_validation_gap=float(gap),
        validation_rows=len(validation_rows),
        validation_positive_rate=float(validation_positive_rate),
        probability_stats=probability_stats,
    )


def feature_drift_report(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    warn_z: float = 4.0,
    fail_z: float = 8.0,
    mean_warn_z: float | None = None,
    mean_fail_z: float | None = None,
    outlier_warn_fraction: float = 0.10,
    outlier_fail_fraction: float = 0.25,
) -> FeatureDriftReport:
    """Compare feature rows against the model's fitted normalization stats."""

    warn_z = max(0.1, float(warn_z))
    fail_z = max(warn_z, float(fail_z))
    mean_warn_threshold = max(
        0.1,
        float(mean_warn_z if mean_warn_z is not None else warn_z * 0.60),
    )
    mean_fail_threshold = max(
        mean_warn_threshold,
        float(mean_fail_z if mean_fail_z is not None else fail_z * 0.50),
    )
    outlier_warn_fraction = _clamp(float(outlier_warn_fraction), 0.0, 1.0)
    outlier_fail_fraction = _clamp(float(outlier_fail_fraction), outlier_warn_fraction, 1.0)
    feature_dim = int(getattr(model, "feature_dim", 0) or 0)
    warnings: List[str] = []
    if not rows:
        return FeatureDriftReport(
            status="fail",
            warnings=["no feature rows available for drift check"],
            rows=0,
            feature_dim=feature_dim,
            max_abs_z=0.0,
            mean_abs_z=0.0,
            outlier_fraction=0.0,
            warn_threshold=warn_z,
            fail_threshold=fail_z,
            mean_warn_threshold=mean_warn_threshold,
            mean_fail_threshold=mean_fail_threshold,
        )

    validate_model_rows(rows, label="drift rows", expected_feature_dim=feature_dim)
    means = list(getattr(model, "feature_means", []) or [])
    stds = list(getattr(model, "feature_stds", []) or [])
    if len(means) != feature_dim or len(stds) != feature_dim:
        return FeatureDriftReport(
            status="fail",
            warnings=["model feature statistics are incomplete"],
            rows=len(rows),
            feature_dim=feature_dim,
            max_abs_z=0.0,
            mean_abs_z=0.0,
            outlier_fraction=1.0,
            warn_threshold=warn_z,
            fail_threshold=fail_z,
            mean_warn_threshold=mean_warn_threshold,
            mean_fail_threshold=mean_fail_threshold,
        )

    z_values: List[float] = []
    for row in rows:
        for value, mean_, std_ in zip(row.features, means, stds, strict=True):
            denominator = float(std_) if abs(float(std_)) > 1e-12 else 1.0
            z_values.append(abs((float(value) - float(mean_)) / denominator))
    max_abs_z = max(z_values) if z_values else 0.0
    mean_abs_z = mean(z_values) if z_values else 0.0
    outliers = sum(1 for value in z_values if value >= warn_z)
    outlier_fraction = outliers / len(z_values) if z_values else 0.0

    status = "ok"
    catastrophic_multiplier = 1.5 if len(rows) == 1 else 3.0
    catastrophic_outlier = max_abs_z >= fail_z * catastrophic_multiplier
    broad_hard_outliers = max_abs_z >= fail_z and outlier_fraction >= outlier_warn_fraction
    if catastrophic_outlier or broad_hard_outliers:
        warnings.append("feature drift exceeds hard threshold")
        status = "fail"
    elif max_abs_z >= fail_z:
        warnings.append("isolated feature drift exceeds hard threshold")
        status = "warn"
    elif max_abs_z >= warn_z:
        warnings.append("feature drift exceeds warning threshold")
        status = "warn"
    if outliers > 0 and outlier_fraction >= outlier_fail_fraction:
        warnings.append("too many feature values are out-of-distribution")
        status = "fail"
    elif outliers > 0 and outlier_fraction >= outlier_warn_fraction:
        warnings.append("elevated out-of-distribution feature fraction")
        if status != "fail":
            status = "warn"
    if mean_abs_z >= mean_fail_threshold:
        warnings.append("broad feature drift exceeds mean hard threshold")
        status = "fail"
    elif mean_abs_z >= mean_warn_threshold:
        warnings.append("broad feature drift exceeds mean warning threshold")
        if status != "fail":
            status = "warn"

    return FeatureDriftReport(
        status=status,
        warnings=warnings,
        rows=len(rows),
        feature_dim=feature_dim,
        max_abs_z=float(max_abs_z),
        mean_abs_z=float(mean_abs_z),
        outlier_fraction=float(outlier_fraction),
        warn_threshold=warn_z,
        fail_threshold=fail_z,
        mean_warn_threshold=mean_warn_threshold,
        mean_fail_threshold=mean_fail_threshold,
    )


def confidence_adjusted_probability(probability: float, beta: float | None) -> float:
    """Shrink a probability toward neutral confidence without changing order."""

    try:
        value = float(probability)
    except (TypeError, ValueError):
        value = 0.5
    if not math.isfinite(value):
        value = 0.5
    value = _clamp(value, 0.0, 1.0)
    if beta is None:
        return value
    try:
        shrink = float(beta)
    except (TypeError, ValueError):
        shrink = 1.0
    if not math.isfinite(shrink):
        shrink = 1.0
    shrink = _clamp(shrink, 0.0, 1.0)
    return 0.5 + (value - 0.5) * shrink


def model_decision_threshold(model: TrainedModel, fallback: float) -> float:
    threshold = getattr(model, "decision_threshold", None)
    if threshold is None:
        threshold = fallback
    try:
        parsed = float(threshold)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = 0.5
    return _clamp(parsed, 0.0, 1.0)


def _optional_decision_threshold(value: object, *, low: float, high: float) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return _clamp(parsed, low, high)


def model_direction_thresholds(
    model: TrainedModel,
    fallback: float,
    *,
    market_type: str = "spot",
) -> tuple[float | None, float | None]:
    """Return long and short probability cutoffs for the current market.

    Futures models may persist asymmetric side thresholds. If neither side
    threshold is present, the legacy symmetric threshold is used: long when
    probability >= T and short when probability <= 1 - T. If either side
    threshold is present, missing sides are disabled so calibration can choose
    long-only or short-only behavior without relaxing the opposite side.
    """

    base = model_decision_threshold(model, fallback)
    if str(market_type).lower() != "futures":
        long_threshold = _optional_decision_threshold(
            getattr(model, "long_decision_threshold", None),
            low=0.0,
            high=1.0,
        )
        return (long_threshold if long_threshold is not None else base), None

    long_raw = getattr(model, "long_decision_threshold", None)
    short_raw = getattr(model, "short_decision_threshold", None)
    if long_raw is None and short_raw is None:
        long_threshold = max(0.5, base)
        return long_threshold, min(0.5, 1.0 - long_threshold)
    return (
        _optional_decision_threshold(long_raw, low=0.5, high=1.0),
        _optional_decision_threshold(short_raw, low=0.0, high=0.5),
    )


def temporal_validation_split(
    rows: List[ModelRow],
    *,
    calibration_ratio: float = 0.15,
    validation_ratio: float = 0.15,
) -> TemporalValidationSplit:
    """Chronologically split rows so calibration and reported validation differ."""

    ordered = sorted(rows, key=lambda row: row.timestamp)
    total = len(ordered)
    if total < 3:
        return TemporalValidationSplit(ordered, [], [])

    calibration_ratio = _clamp(float(calibration_ratio), 0.0, 0.8)
    validation_ratio = _clamp(float(validation_ratio), 0.0, 0.8)
    calibration_size = max(1, int(total * calibration_ratio)) if calibration_ratio > 0.0 else 0
    validation_size = max(1, int(total * validation_ratio)) if validation_ratio > 0.0 else 0
    while calibration_size + validation_size >= total and (calibration_size > 0 or validation_size > 0):
        if validation_size >= calibration_size and validation_size > 0:
            validation_size -= 1
        else:
            calibration_size -= 1

    train_end = total - calibration_size - validation_size
    calibration_end = total - validation_size
    return TemporalValidationSplit(
        train_rows=ordered[:train_end],
        calibration_rows=ordered[train_end:calibration_end],
        validation_rows=ordered[calibration_end:],
    )


def calibrate_threshold(rows: List[ModelRow], model: TrainedModel, *, start: float = 0.1, end: float = 0.9,
                       steps: int = 17) -> float:
    """Pick a threshold that balances precision and recall for the current model."""
    if not rows:
        return 0.5
    if steps <= 1:
        return _clamp(0.5, 0.0, 1.0)

    best_threshold = 0.5
    best_f1 = -1.0
    if start < 0.0:
        start = 0.0
    if end > 1.0:
        end = 1.0
    if end <= start:
        end = min(1.0, start + 0.01)

    for i in range(steps):
        threshold = start + (end - start) * i / (steps - 1)
        tp, fp, _, fn = _confusion(rows, model, threshold)
        score = _f1(tp, fp, fn)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def _with_backend_metadata(model: TrainedModel, backend: BackendInfo) -> TrainedModel:
    model.training_backend_requested = backend.requested
    model.training_backend_kind = backend.kind
    model.training_backend_device = backend.device
    model.training_backend_vendor = backend.vendor
    model.training_backend_reason = backend.reason
    return model


def _fallback_backend(requested: BackendInfo, reason: str) -> BackendInfo:
    return BackendInfo(
        requested=requested.requested,
        kind="cpu",
        device="cpu",
        vendor="Python stdlib",
        reason=reason[:240],
    )


def effective_training_backend_name(compute_backend: str | None) -> str:
    """Return the backend name used by training/scoring when callers omit it.

    Omitted values mean GPU-first auto-probing. CPU is used only when explicitly
    requested or when every GPU probe fails, and the resolved metadata records
    that fallback.
    """

    requested = str(compute_backend or "").strip().lower()
    return requested or "auto"


def _torch_device_for_backend(backend: BackendInfo):  # pragma: no cover - optional GPU runtime
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


def _maybe_promote_averaged_params(
    rows: List[ModelRow],
    validation_rows: List[ModelRow],
    weights: List[float],
    bias: float,
    means: List[float],
    stds: List[float],
    averaged_weights: List[float],
    averaged_bias: float,
    averaged_count: int,
    *,
    min_delta: float,
) -> tuple[List[float], float]:
    if averaged_count <= 0:
        return weights, bias
    candidate_weights = [value / averaged_count for value in averaged_weights]
    candidate_bias = averaged_bias / averaged_count
    selector_rows = validation_rows if validation_rows else rows
    current_loss = _log_loss(selector_rows, weights, bias, means, stds)
    candidate_loss = _log_loss(selector_rows, candidate_weights, candidate_bias, means, stds)
    if candidate_loss < current_loss - float(min_delta):
        return candidate_weights, candidate_bias
    return weights, bias


def _train_torch(  # pragma: no cover - exercised by GPU smoke verification, not CI
    rows: List[ModelRow],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    l2_penalty: float,
    feature_signature: str | None,
    validation_rows: List[ModelRow],
    early_stopping_rounds: int | None,
    min_delta: float,
    batch_size: int,
    backend: BackendInfo,
) -> TrainedModel:
    import torch  # type: ignore

    feature_dim = len(rows[0].features)
    device = _torch_device_for_backend(backend)
    x_train = torch.tensor([row.features for row in rows], dtype=torch.float32, device=device)
    means_t = x_train.mean(dim=0)
    centered_train = x_train - means_t
    stds_t = torch.sqrt(torch.mean(centered_train * centered_train, dim=0))
    stds_t = torch.where(torch.abs(stds_t) > 1e-6, stds_t, torch.ones_like(stds_t))
    x_train = centered_train / stds_t
    means = [float(value) for value in means_t.detach().cpu().tolist()]
    stds = [float(value) for value in stds_t.detach().cpu().tolist()]
    y_train = torch.tensor([float(row.label) for row in rows], dtype=torch.float32, device=device)
    if validation_rows:
        x_validation_raw = torch.tensor([row.features for row in validation_rows], dtype=torch.float32, device=device)
        x_validation = (x_validation_raw - means_t) / stds_t
        y_validation = torch.tensor([float(row.label) for row in validation_rows], dtype=torch.float32, device=device)
    else:
        x_validation = None
        y_validation = None

    rng = random.Random(seed)  # nosec B311
    initial_weights = [rng.uniform(-0.05, 0.05) for _ in range(feature_dim)]
    weights = torch.tensor(initial_weights, dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([weights, bias], lr=float(learning_rate))
    class_weight_pos, class_weight_neg = _class_weights(rows)
    if class_weight_pos <= 0.0 or class_weight_neg <= 0.0:
        class_weight_pos = 1.0
        class_weight_neg = 1.0

    best_weights_t = weights.detach().clone()
    best_bias_t = bias.detach().clone()
    best_epoch: int | None = None
    best_validation_loss = float("inf")
    rounds_without_improvement = 0
    patience = int(early_stopping_rounds or 0)
    batch = max(1, min(int(batch_size or len(rows)), len(rows)))
    pos_weight = torch.tensor(float(class_weight_pos), dtype=torch.float32, device=device)
    neg_weight = torch.tensor(float(class_weight_neg), dtype=torch.float32, device=device)
    averaged_weights_t = torch.zeros_like(weights)
    averaged_bias_t = torch.zeros_like(bias)
    averaged_count = 0
    average_start_epoch = max(1, int(epochs) // 2)

    def tensor_log_loss(x_values, y_values, weight_values, bias_value):
        logits = x_values.matmul(weight_values.reshape(-1, 1)).reshape(-1) + bias_value
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        per_row_loss = torch.clamp(logits, min=0.0) - logits * y_values + torch.log1p(torch.exp(-torch.abs(logits)))
        return per_row_loss.mean()

    for epoch_index in range(1, int(epochs) + 1):
        for start in range(0, len(rows), batch):
            end = min(start + batch, len(rows))
            xb = x_train[start:end]
            yb = y_train[start:end]
            optimizer.zero_grad()
            logits = xb.matmul(weights.reshape(-1, 1)).reshape(-1) + bias
            # Stable BCE-with-logits written from tensor primitives supported by
            # CUDA, ROCm, DirectML, and MPS.
            per_row_loss = torch.clamp(logits, min=0.0) - logits * yb + torch.log1p(torch.exp(-torch.abs(logits)))
            sample_weights = torch.where(yb > 0.5, pos_weight, neg_weight)
            loss = (per_row_loss * sample_weights).mean() + 0.5 * float(l2_penalty) * torch.sum(weights * weights)
            loss.backward()
            optimizer.step()
        current_weights_t = weights.detach()
        current_bias_t = bias.detach()
        if epoch_index >= average_start_epoch:
            averaged_weights_t = averaged_weights_t + current_weights_t
            averaged_bias_t = averaged_bias_t + current_bias_t
            averaged_count += 1
        if x_validation is not None and y_validation is not None:
            current_loss = float(
                tensor_log_loss(x_validation, y_validation, current_weights_t, current_bias_t)
                .detach()
                .cpu()
                .item()
            )
            if current_loss < best_validation_loss - float(min_delta):
                best_validation_loss = current_loss
                best_weights_t = current_weights_t.clone()
                best_bias_t = current_bias_t.clone()
                best_epoch = epoch_index
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if patience > 0 and rounds_without_improvement >= patience:
                    break

    if x_validation is not None and best_epoch is not None:
        final_weights_t = best_weights_t
        final_bias_t = best_bias_t
    else:
        final_weights_t = weights.detach().clone()
        final_bias_t = bias.detach().clone()
    if averaged_count > 0:
        candidate_weights_t = averaged_weights_t / float(averaged_count)
        candidate_bias_t = averaged_bias_t / float(averaged_count)
        selector_x = x_validation if x_validation is not None else x_train
        selector_y = y_validation if y_validation is not None else y_train
        current_selector_loss = tensor_log_loss(selector_x, selector_y, final_weights_t, final_bias_t)
        candidate_selector_loss = tensor_log_loss(selector_x, selector_y, candidate_weights_t, candidate_bias_t)
        if float(candidate_selector_loss.detach().cpu().item()) < float(current_selector_loss.detach().cpu().item()) - float(min_delta):
            final_weights_t = candidate_weights_t.detach().clone()
            final_bias_t = candidate_bias_t.detach().clone()

    final_weights = [float(value) for value in final_weights_t.detach().cpu().tolist()]
    final_bias = float(final_bias_t.detach().cpu().item())
    training_loss = float(tensor_log_loss(x_train, y_train, final_weights_t, final_bias_t).detach().cpu().item())
    validation_loss = (
        float(tensor_log_loss(x_validation, y_validation, final_weights_t, final_bias_t).detach().cpu().item())
        if x_validation is not None and y_validation is not None
        else None
    )
    return TrainedModel(
        weights=final_weights,
        bias=final_bias,
        feature_dim=feature_dim,
        epochs=epochs,
        feature_means=means,
        feature_stds=stds,
        feature_signature=feature_signature,
        learning_rate=float(learning_rate),
        l2_penalty=float(l2_penalty),
        seed=int(seed),
        class_weight_pos=float(class_weight_pos),
        class_weight_neg=float(class_weight_neg),
        best_epoch=best_epoch,
        training_loss=float(training_loss),
        validation_loss=float(validation_loss) if validation_loss is not None else None,
    )


def train(rows: List[ModelRow], *, epochs: int = 200, learning_rate: float = 0.05,
          seed: int = 7, l2_penalty: float = 1e-4,
          feature_signature: str | None = None,
          validation_rows: List[ModelRow] | None = None,
          early_stopping_rounds: int | None = None,
          min_delta: float = 1e-6,
          compute_backend: str | None = None,
          batch_size: int = 8192) -> TrainedModel:
    feature_dim = validate_model_rows(rows)
    validation_rows = list(validation_rows or [])
    if validation_rows:
        validate_model_rows(
            validation_rows,
            label="validation rows",
            expected_feature_dim=feature_dim,
        )

    backend = resolve_backend(effective_training_backend_name(compute_backend))
    if backend.kind != "cpu":
        try:
            return _with_backend_metadata(
                _train_torch(
                    rows,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    seed=seed,
                    l2_penalty=l2_penalty,
                    feature_signature=feature_signature,
                    validation_rows=validation_rows,
                    early_stopping_rounds=early_stopping_rounds,
                    min_delta=min_delta,
                    batch_size=batch_size,
                    backend=backend,
                ),
                backend,
            )
        except Exception as exc:
            backend = _fallback_backend(
                backend,
                f"{backend.kind} training failed ({exc.__class__.__name__}); fell back to CPU",
            )

    means, stds = _collect_feature_stats(rows)
    normalized = _normalize_rows(rows, means, stds)

    rng = random.Random(seed)  # nosec B311
    weights = [rng.uniform(-0.05, 0.05) for _ in range(feature_dim)]
    bias = 0.0
    class_weight_pos, class_weight_neg = _class_weights(rows)
    if class_weight_pos <= 0.0 or class_weight_neg <= 0.0:
        class_weight_pos = 1.0
        class_weight_neg = 1.0

    indices = list(range(len(rows)))
    best_weights = list(weights)
    best_bias = bias
    best_epoch: int | None = None
    best_validation_loss = float("inf")
    rounds_without_improvement = 0
    patience = int(early_stopping_rounds or 0)
    averaged_weights = [0.0] * feature_dim
    averaged_bias = 0.0
    averaged_count = 0
    average_start_epoch = max(1, int(epochs) // 2)

    for epoch_index in range(1, int(epochs) + 1):
        rng.shuffle(indices)
        for idx in indices:
            row = rows[idx]
            x = normalized[idx]
            y = row.label
            score = bias + sum(w * xi for w, xi in zip(weights, x, strict=True))
            pred = _sigmoid(score)
            weight = class_weight_pos if y == 1 else class_weight_neg
            error = (pred - y) * weight
            for i, xi in enumerate(x):
                # L2 penalty and signed-gradient update
                grad = error * xi + l2_penalty * weights[i]
                weights[i] -= learning_rate * grad
            bias -= learning_rate * error
        if epoch_index >= average_start_epoch:
            averaged_weights = [left + right for left, right in zip(averaged_weights, weights, strict=True)]
            averaged_bias += bias
            averaged_count += 1
        if validation_rows:
            current_loss = _log_loss(validation_rows, weights, bias, means, stds)
            if current_loss < best_validation_loss - float(min_delta):
                best_validation_loss = current_loss
                best_weights = list(weights)
                best_bias = bias
                best_epoch = epoch_index
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if patience > 0 and rounds_without_improvement >= patience:
                    break

    if validation_rows and best_epoch is not None:
        weights = best_weights
        bias = best_bias
    weights, bias = _maybe_promote_averaged_params(
        rows,
        validation_rows,
        weights,
        bias,
        means,
        stds,
        averaged_weights,
        averaged_bias,
        averaged_count,
        min_delta=min_delta,
    )

    training_loss = _log_loss(rows, weights, bias, means, stds)
    validation_loss = _log_loss(validation_rows, weights, bias, means, stds) if validation_rows else None

    return _with_backend_metadata(TrainedModel(
        weights=weights,
        bias=bias,
        feature_dim=feature_dim,
        epochs=epochs,
        feature_means=means,
        feature_stds=stds,
        feature_signature=feature_signature,
        learning_rate=float(learning_rate),
        l2_penalty=float(l2_penalty),
        seed=int(seed),
        class_weight_pos=float(class_weight_pos),
        class_weight_neg=float(class_weight_neg),
        best_epoch=best_epoch,
        training_loss=float(training_loss),
        validation_loss=float(validation_loss) if validation_loss is not None else None,
    ), backend)


def evaluate(rows: List[ModelRow], model: TrainedModel, threshold: float = 0.5) -> float:
    if not rows:
        return 0.0
    correct = 0
    threshold = _clamp(threshold, 0.0, 1.0)
    for row in rows:
        pred = model.predict(row.features, threshold)
        if pred == row.label:
            correct += 1
    return correct / len(rows)


def evaluate_confusion(rows: List[ModelRow], model: TrainedModel, threshold: float = 0.5) -> tuple[int, int, int, int]:
    return _confusion(rows, model, threshold)


def evaluate_classification(
    rows: List[ModelRow],
    model: TrainedModel,
    threshold: float = 0.5,
) -> ClassificationReport:
    if not rows:
        return ClassificationReport(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            threshold=_clamp(threshold, 0.0, 1.0),
            true_positive=0,
            false_positive=0,
            true_negative=0,
            false_negative=0,
        )
    tp, fp, tn, fn = _confusion(rows, model, threshold)
    total = tp + fp + tn + fn
    precision = _safe_division(tp, tp + fp)
    recall = _safe_division(tp, tp + fn)
    f1 = _f1(tp, fp, fn)
    accuracy = (tp + tn) / total
    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=_clamp(threshold, 0.0, 1.0),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def walk_forward_report(
    rows: List[ModelRow],
    *,
    train_window: int = 300,
    test_window: int = 60,
    step: int = 20,
    epochs: int = 80,
    calibrate: bool = False,
    learning_rate: float = 0.05,
    l2_penalty: float = 1e-4,
    compute_backend: str | None = None,
    batch_size: int = 8192,
) -> dict[str, object]:
    if len(rows) <= train_window + test_window:
        raise ValueError("Not enough rows for walk-forward evaluation")
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError("train_window, test_window, and step must be positive")

    scores: List[float] = []
    thresholds: List[float] = []
    calibration_sizes: List[int] = []
    for start in range(0, len(rows) - train_window - test_window + 1, step):
        train_rows = rows[start : start + train_window]
        test_rows = rows[start + train_window : start + train_window + test_window]
        fit_rows = train_rows
        calibration_rows: List[ModelRow] = []
        if calibrate and len(train_rows) >= 10:
            calibration_size = max(1, int(len(train_rows) * 0.2))
            fit_rows = train_rows[:-calibration_size]
            calibration_rows = train_rows[-calibration_size:]
        model = train(
            fit_rows,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        threshold = 0.5
        if calibration_rows:
            threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)
        score = evaluate(test_rows, model, threshold=threshold)
        scores.append(score)
        thresholds.append(threshold)
        calibration_sizes.append(len(calibration_rows))

    return {
        "folds": len(scores),
        "average_score": mean(scores) if scores else 0.0,
        "scores": scores,
        "thresholds": thresholds,
        "calibration_sizes": calibration_sizes,
        "train_window": train_window,
        "test_window": test_window,
        "step": step,
        "learning_rate": float(learning_rate),
        "l2_penalty": float(l2_penalty),
        "compute_backend": compute_backend or "default",
        "batch_size": int(batch_size),
    }


def serialize_model(model: TrainedModel, path) -> None:
    write_json_atomic(path, asdict(model), indent=2)


def ensemble_member_from_model(model: TrainedModel) -> EnsembleMember:
    return EnsembleMember(
        weights=list(model.weights),
        bias=float(model.bias),
        feature_means=list(model.feature_means),
        feature_stds=list(model.feature_stds),
        seed=int(model.seed),
        epochs=int(model.epochs),
        best_epoch=model.best_epoch,
        training_loss=model.training_loss,
        validation_loss=model.validation_loss,
    )


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("not finite")
    return value


def _load_ensemble_members(raw: Any, feature_dim: int) -> list[EnsembleMember]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ModelLoadError("Model payload ensemble_members must be an array")
    members: list[EnsembleMember] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ModelLoadError(f"Model payload ensemble member {index} is not an object")
        try:
            weights = [float(value) for value in entry["weights"]]
            means = [float(value) for value in entry["feature_means"]]
            stds = [float(value) for value in entry["feature_stds"]]
            bias = float(entry["bias"])
            seed = int(entry.get("seed", 7))
            epochs = int(entry.get("epochs", 0))
            best_epoch = (
                int(entry["best_epoch"])
                if entry.get("best_epoch") is not None
                else None
            )
            training_loss = _optional_float(entry.get("training_loss"))
            validation_loss = _optional_float(entry.get("validation_loss"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"Model payload ensemble member {index} is invalid") from exc
        if len(weights) != feature_dim or len(means) != feature_dim or len(stds) != feature_dim:
            raise ModelLoadError(f"Model payload ensemble member {index} dimension mismatch")
        members.append(EnsembleMember(
            weights=weights,
            bias=bias,
            feature_means=means,
            feature_stds=stds,
            seed=seed,
            epochs=epochs,
            best_epoch=best_epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
        ))
    return members


def _load_hybrid_prototypes(raw: Any, feature_dim: int) -> list[HybridPrototype]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ModelLoadError("Model payload hybrid expert prototypes must be an array")
    prototypes: list[HybridPrototype] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ModelLoadError(f"Model payload hybrid prototype {index} is not an object")
        try:
            features = [float(value) for value in entry["features"]]
            label = int(entry.get("label", 0))
            timestamp = int(entry.get("timestamp", 0) or 0)
            close = float(entry.get("close", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"Model payload hybrid prototype {index} is invalid") from exc
        if len(features) != feature_dim:
            raise ModelLoadError(f"Model payload hybrid prototype {index} dimension mismatch")
        prototypes.append(HybridPrototype(
            features=features,
            label=1 if label else 0,
            timestamp=timestamp,
            close=close,
        ))
    return prototypes


def _load_hybrid_experts(raw: Any, feature_dim: int) -> list[HybridExpert]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ModelLoadError("Model payload hybrid_experts must be an array")
    experts: list[HybridExpert] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ModelLoadError(f"Model payload hybrid expert {index} is not an object")
        try:
            experts.append(HybridExpert(
                name=str(entry.get("name", f"expert_{index}") or f"expert_{index}"),
                kind=str(entry["kind"]),
                weight=max(0.0, float(entry.get("weight", 0.0) or 0.0)),
                prototypes=_load_hybrid_prototypes(entry.get("prototypes", []), feature_dim),
                k=max(1, int(entry.get("k", 21) or 21)),
                bandwidth=max(1e-6, float(entry.get("bandwidth", 1.0) or 1.0)),
                alpha=max(1e-6, float(entry.get("alpha", 1.0) or 1.0)),
                feature_count=max(1, int(entry.get("feature_count", feature_dim) or feature_dim)),
                notes=str(entry.get("notes", "") or ""),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"Model payload hybrid expert {index} is invalid") from exc
    return experts


def load_model(
    path,
    *,
    expected_feature_version: str | None = FEATURE_VERSION,
    expected_feature_dim: int | None = None,
    expected_feature_signature: str | None = None,
) -> TrainedModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_version = payload.get("feature_version")
    if not isinstance(model_version, str) or not model_version:
        raise ModelLoadError("Model metadata is missing `feature_version`; please retrain the model")

    if expected_feature_version is not None and model_version != expected_feature_version:
        raise ModelFeatureMismatchError(
            f"Feature version mismatch: model={model_version} runtime={expected_feature_version}"
        )
    payload_signature = payload.get("feature_signature")
    if expected_feature_signature is not None:
        if payload_signature is None:
            raise ModelFeatureMismatchError("Model metadata is missing `feature_signature`; please retrain the model")
        if str(payload_signature) != expected_feature_signature:
            raise ModelFeatureMismatchError(
                f"Feature signature mismatch: model={payload_signature} runtime={expected_feature_signature}"
            )

    dim = int(payload["feature_dim"])
    means = payload.get("feature_means")
    stds = payload.get("feature_stds")
    if means is None:
        raise ModelLoadError("Model payload missing feature_means")
    if stds is None:
        raise ModelLoadError("Model payload missing feature_stds")
    if not isinstance(means, list) or not isinstance(stds, list):
        raise ModelLoadError("Model payload feature stats must be arrays")

    if len(means) != dim:
        raise ModelLoadError("Model payload feature_dim does not match feature_means length")
    if len(stds) != dim:
        raise ModelLoadError("Model payload feature_dim does not match feature_stds length")

    if expected_feature_dim is not None and dim != expected_feature_dim:
        raise ModelFeatureMismatchError(
            f"Feature dimension mismatch: model={dim} expected={expected_feature_dim}"
        )

    weights = payload.get("weights")
    if not isinstance(weights, list):
        raise ModelLoadError("Model payload missing weights")
    if len(weights) != dim:
        raise ModelLoadError("Model payload weights length does not match feature_dim")

    return TrainedModel(
        weights=list(float(w) for w in weights),
        bias=float(payload["bias"]),
        feature_dim=dim,
        epochs=int(payload["epochs"]),
        feature_version=str(model_version),
        feature_signature=str(payload_signature) if payload_signature is not None else None,
        feature_means=list(float(x) for x in means),
        feature_stds=list(float(x) for x in stds),
        learning_rate=float(payload.get("learning_rate", 0.05)),
        l2_penalty=float(payload.get("l2_penalty", 1e-4)),
        seed=int(payload.get("seed", 7)),
        class_weight_pos=float(payload.get("class_weight_pos", 1.0)),
        class_weight_neg=float(payload.get("class_weight_neg", 1.0)),
        decision_threshold=(
            float(payload["decision_threshold"])
            if payload.get("decision_threshold") is not None
            else None
        ),
        long_decision_threshold=(
            float(payload["long_decision_threshold"])
            if payload.get("long_decision_threshold") is not None
            else None
        ),
        short_decision_threshold=(
            float(payload["short_decision_threshold"])
            if payload.get("short_decision_threshold") is not None
            else None
        ),
        calibration_size=int(payload.get("calibration_size", 0)),
        validation_size=int(payload.get("validation_size", 0)),
        training_cutoff_timestamp=(
            int(payload["training_cutoff_timestamp"])
            if payload.get("training_cutoff_timestamp") is not None
            else None
        ),
        best_epoch=(
            int(payload["best_epoch"])
            if payload.get("best_epoch") is not None
            else None
        ),
        training_loss=(
            float(payload["training_loss"])
            if payload.get("training_loss") is not None
            else None
        ),
        validation_loss=(
            float(payload["validation_loss"])
            if payload.get("validation_loss") is not None
            else None
        ),
        quality_score=(
            float(payload["quality_score"])
            if payload.get("quality_score") is not None
            else None
        ),
        quality_warnings=[
            str(value)
            for value in payload.get("quality_warnings", [])
            if isinstance(value, str)
        ],
        probability_temperature=float(payload.get("probability_temperature", 1.0)),
        probability_inverted=payload.get("probability_inverted") is True,
        probability_calibration_size=int(payload.get("probability_calibration_size", 0)),
        probability_log_loss_before=(
            float(payload["probability_log_loss_before"])
            if payload.get("probability_log_loss_before") is not None
            else None
        ),
        probability_log_loss_after=(
            float(payload["probability_log_loss_after"])
            if payload.get("probability_log_loss_after") is not None
            else None
        ),
        probability_brier_before=(
            float(payload["probability_brier_before"])
            if payload.get("probability_brier_before") is not None
            else None
        ),
        probability_brier_after=(
            float(payload["probability_brier_after"])
            if payload.get("probability_brier_after") is not None
            else None
        ),
        probability_ece_before=(
            float(payload["probability_ece_before"])
            if payload.get("probability_ece_before") is not None
            else None
        ),
        probability_ece_after=(
            float(payload["probability_ece_after"])
            if payload.get("probability_ece_after") is not None
            else None
        ),
        probability_calibration_backend_requested=str(
            payload.get("probability_calibration_backend_requested", "cpu") or "cpu"
        ),
        probability_calibration_backend_kind=str(
            payload.get("probability_calibration_backend_kind", "cpu") or "cpu"
        ),
        probability_calibration_backend_device=str(
            payload.get("probability_calibration_backend_device", "cpu") or "cpu"
        ),
        probability_calibration_backend_reason=str(
            payload.get("probability_calibration_backend_reason", "") or ""
        ),
        threshold_source=(
            str(payload["threshold_source"])
            if payload.get("threshold_source") is not None
            else None
        ),
        threshold_calibration_score=(
            float(payload["threshold_calibration_score"])
            if payload.get("threshold_calibration_score") is not None
            else None
        ),
        threshold_calibration_pnl=(
            float(payload["threshold_calibration_pnl"])
            if payload.get("threshold_calibration_pnl") is not None
            else None
        ),
        threshold_calibration_trades=int(payload.get("threshold_calibration_trades", 0) or 0),
        threshold_diagnostic_best_threshold=(
            float(payload["threshold_diagnostic_best_threshold"])
            if payload.get("threshold_diagnostic_best_threshold") is not None
            else None
        ),
        threshold_diagnostic_best_score=(
            float(payload["threshold_diagnostic_best_score"])
            if payload.get("threshold_diagnostic_best_score") is not None
            else None
        ),
        threshold_diagnostic_best_pnl=(
            float(payload["threshold_diagnostic_best_pnl"])
            if payload.get("threshold_diagnostic_best_pnl") is not None
            else None
        ),
        threshold_diagnostic_best_trades=int(payload.get("threshold_diagnostic_best_trades", 0) or 0),
        threshold_diagnostic_best_long_threshold=(
            float(payload["threshold_diagnostic_best_long_threshold"])
            if payload.get("threshold_diagnostic_best_long_threshold") is not None
            else None
        ),
        threshold_diagnostic_best_short_threshold=(
            float(payload["threshold_diagnostic_best_short_threshold"])
            if payload.get("threshold_diagnostic_best_short_threshold") is not None
            else None
        ),
        strategy_overrides=clean_strategy_overrides(payload.get("strategy_overrides", {})),
        ensemble_members=_load_ensemble_members(payload.get("ensemble_members"), dim),
        training_backend_requested=str(payload.get("training_backend_requested", "cpu") or "cpu"),
        training_backend_kind=str(payload.get("training_backend_kind", "cpu") or "cpu"),
        training_backend_device=str(payload.get("training_backend_device", "cpu") or "cpu"),
        training_backend_vendor=str(payload.get("training_backend_vendor", "Python stdlib") or "Python stdlib"),
        training_backend_reason=str(payload.get("training_backend_reason", "") or ""),
        model_family=str(payload.get("model_family", "advanced_logistic") or "advanced_logistic"),
        model_candidate_count=max(1, int(payload.get("model_candidate_count", 1) or 1)),
        model_selected_candidate=str(payload.get("model_selected_candidate", "default") or "default"),
        model_selection_score=(
            float(payload["model_selection_score"])
            if payload.get("model_selection_score") is not None
            else None
        ),
        round_candidate_diagnostics=[
            dict(item)
            for item in payload.get("round_candidate_diagnostics", [])
            if isinstance(item, dict)
        ],
        hybrid_base_weight=max(0.0, float(payload.get("hybrid_base_weight", 1.0) or 1.0)),
        hybrid_experts=_load_hybrid_experts(payload.get("hybrid_experts", []), dim),
        meta_label_policy=(
            dict(payload["meta_label_policy"])
            if isinstance(payload.get("meta_label_policy"), dict)
            else {}
        ),
        selection_risk=(
            dict(payload["selection_risk"])
            if isinstance(payload.get("selection_risk"), dict)
            else {}
        ),
        execution_validation=(
            dict(payload["execution_validation"])
            if isinstance(payload.get("execution_validation"), dict)
            else {}
        ),
    )

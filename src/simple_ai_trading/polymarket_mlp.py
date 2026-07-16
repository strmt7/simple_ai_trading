"""Frozen Round 9 nonlinear challenger for causal Polymarket actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import random
import time
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.special import expit, ndtr

from .compute import BackendInfo, resolve_backend
from .polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES
from .polymarket_fit_claim import (
    PolymarketFitClaim,
    begin_polymarket_fit_claim,
    complete_polymarket_fit_claim,
    fail_polymarket_fit_claim,
)
from .polymarket_ridge import (
    POLYMARKET_RIDGE_CONTRACT_SHA256,
    POLYMARKET_RIDGE_THRESHOLD_GRID,
    PolymarketPolicyEvaluation,
    PolymarketPolicyMetrics,
    PolymarketRidgeDataset,
    PolymarketRidgeReport,
    PolymarketRidgeSplit,
    evaluate_polymarket_policy,
    polymarket_selected_policy_tables,
    split_polymarket_ridge_dataset,
)
from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_MLP_CONTRACT_SHA256 = (
    "a5d87f65036e4a6c71835ce549668d81767b2ba16bd227ea2319c24b0880f7a2"
)
POLYMARKET_MLP_MODEL_SCHEMA_VERSION = "polymarket-round9-causal-mlp-model-v2"
POLYMARKET_MLP_REPORT_SCHEMA_VERSION = "polymarket-round9-causal-mlp-report-v3"
POLYMARKET_MLP_SEEDS = (4701, 4702, 4703)
POLYMARKET_MLP_BATCH_SIZE = 4096
POLYMARKET_MLP_MAX_EPOCHS = 200
POLYMARKET_MLP_PATIENCE = 20
POLYMARKET_MLP_MIN_DELTA = 0.000001
POLYMARKET_MLP_BOOTSTRAP_SAMPLES = 2000
POLYMARKET_MLP_MIN_TEST_GROUPS = 30
POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE = 0.00001
_FEATURE_COUNT = len(POLYMARKET_ACTION_FEATURE_NAMES)
_MEMBER_PARAMETER_COUNT = 4673
ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _float_text(values: Sequence[float]) -> list[str]:
    return [format(float(value), ".17g") for value in values]


def _binary_log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    if predicted.shape != labels.shape or predicted.size == 0:
        raise ValueError("Polymarket MLP log loss requires aligned nonempty values")
    clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )


@dataclass(frozen=True)
class PolymarketMLPBackendEvidence:
    requested: str
    kind: str
    device: str
    vendor: str
    fallback_reason: str
    torch_version: str
    torch_directml_version: str
    preflight_objective: float
    preflight_parameter_delta: float
    preflight_seconds: float
    training_seconds: float
    canonical_replay_max_probability_drift: float | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def identity_payload(self) -> dict[str, object]:
        payload = self.asdict()
        payload.pop("preflight_seconds")
        payload.pop("training_seconds")
        return payload

    def validated(self) -> PolymarketMLPBackendEvidence:
        values = (
            self.preflight_objective,
            self.preflight_parameter_delta,
            self.preflight_seconds,
            self.training_seconds,
        )
        allowed = {"cpu", "cuda", "rocm", "directml", "mps"}
        if (
            self.requested not in {"auto", *allowed}
            or self.kind not in allowed
            or (self.requested != "auto" and self.requested != self.kind)
            or not self.device
            or not self.vendor
            or not self.torch_version
            or (self.kind == "directml" and not self.torch_directml_version)
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
            or self.preflight_parameter_delta <= 0.0
            or self.training_seconds <= 0.0
            or self.canonical_replay_max_probability_drift is None
            or not math.isfinite(self.canonical_replay_max_probability_drift)
            or not 0.0
            <= self.canonical_replay_max_probability_drift
            <= POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
        ):
            raise ValueError("Polymarket MLP backend evidence is invalid")
        return self


@dataclass(frozen=True)
class PolymarketMLPEpoch:
    seed: int
    epoch: int
    training_loss: float
    validation_log_loss: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolymarketMLPMember:
    seed: int
    best_epoch: int
    epochs_ran: int
    hidden1_weight: tuple[float, ...]
    hidden1_bias: tuple[float, ...]
    hidden2_weight: tuple[float, ...]
    hidden2_bias: tuple[float, ...]
    output_weight: tuple[float, ...]
    output_bias: float
    trace: tuple[PolymarketMLPEpoch, ...]
    member_sha256: str

    def identity_payload(self) -> dict[str, object]:
        trace_payload = [item.asdict() for item in self.trace]
        return {
            "schema_version": POLYMARKET_MLP_MODEL_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_MLP_CONTRACT_SHA256,
            "architecture": [39, 64, 32, 1],
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "hidden1_weight": _float_text(self.hidden1_weight),
            "hidden1_bias": _float_text(self.hidden1_bias),
            "hidden2_weight": _float_text(self.hidden2_weight),
            "hidden2_bias": _float_text(self.hidden2_bias),
            "output_weight": _float_text(self.output_weight),
            "output_bias": format(self.output_bias, ".17g"),
            "trace_sha256": _sha256(trace_payload),
        }

    def validated(self) -> PolymarketMLPMember:
        values = (
            *self.hidden1_weight,
            *self.hidden1_bias,
            *self.hidden2_weight,
            *self.hidden2_bias,
            *self.output_weight,
            self.output_bias,
        )
        best_loss = math.inf
        expected_best_epoch = 0
        for item in self.trace:
            if item.validation_log_loss < best_loss - POLYMARKET_MLP_MIN_DELTA:
                best_loss = item.validation_log_loss
                expected_best_epoch = item.epoch
        if (
            self.seed not in POLYMARKET_MLP_SEEDS
            or not 1 <= self.best_epoch <= self.epochs_ran <= POLYMARKET_MLP_MAX_EPOCHS
            or len(self.hidden1_weight) != 64 * _FEATURE_COUNT
            or len(self.hidden1_bias) != 64
            or len(self.hidden2_weight) != 32 * 64
            or len(self.hidden2_bias) != 32
            or len(self.output_weight) != 32
            or len(values) != _MEMBER_PARAMETER_COUNT
            or not all(math.isfinite(value) for value in values)
            or len(self.trace) != self.epochs_ran
            or tuple(item.epoch for item in self.trace)
            != tuple(range(1, self.epochs_ran + 1))
            or any(
                item.seed != self.seed
                or not math.isfinite(item.training_loss)
                or not math.isfinite(item.validation_log_loss)
                or item.training_loss < 0.0
                or item.validation_log_loss < 0.0
                for item in self.trace
            )
            or self.best_epoch != expected_best_epoch
            or not _is_sha256(self.member_sha256)
            or self.member_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket MLP member is invalid")
        return self

    def predict_standardized(self, values: np.ndarray) -> np.ndarray:
        self.validated()
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != _FEATURE_COUNT:
            raise ValueError("Polymarket MLP prediction matrix is invalid")
        hidden1 = matrix @ np.asarray(self.hidden1_weight, dtype=np.float64).reshape(
            64,
            _FEATURE_COUNT,
        ).T + np.asarray(self.hidden1_bias, dtype=np.float64)
        hidden1 = hidden1 * ndtr(hidden1)
        hidden2 = hidden1 @ np.asarray(self.hidden2_weight, dtype=np.float64).reshape(
            32, 64
        ).T + np.asarray(self.hidden2_bias, dtype=np.float64)
        hidden2 = hidden2 * ndtr(hidden2)
        logits = (
            hidden2 @ np.asarray(self.output_weight, dtype=np.float64)
            + self.output_bias
        )
        probability = expit(logits)
        if not np.all(np.isfinite(probability)):
            raise ValueError("Polymarket MLP probabilities are non-finite")
        return np.asarray(probability, dtype=np.float64)


@dataclass(frozen=True)
class PolymarketMLPEnsemble:
    dataset_sha256: str
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    members: tuple[PolymarketMLPMember, ...]
    backend: PolymarketMLPBackendEvidence
    reproducibility_max_probability_drift: float
    ensemble_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_MLP_MODEL_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_MLP_CONTRACT_SHA256,
            "dataset_sha256": self.dataset_sha256,
            "feature_names": list(POLYMARKET_ACTION_FEATURE_NAMES),
            "feature_mean": _float_text(self.feature_mean),
            "feature_scale": _float_text(self.feature_scale),
            "member_sha256": [item.member_sha256 for item in self.members],
            "backend": self.backend.identity_payload(),
            "reproducibility_max_probability_drift": format(
                self.reproducibility_max_probability_drift,
                ".17g",
            ),
        }

    def validated(self) -> PolymarketMLPEnsemble:
        for member in self.members:
            member.validated()
        self.backend.validated()
        if (
            not _is_sha256(self.dataset_sha256)
            or len(self.feature_mean) != _FEATURE_COUNT
            or len(self.feature_scale) != _FEATURE_COUNT
            or not all(math.isfinite(value) for value in self.feature_mean)
            or not all(
                math.isfinite(value) and value > 0.0 for value in self.feature_scale
            )
            or tuple(item.seed for item in self.members) != POLYMARKET_MLP_SEEDS
            or not math.isfinite(self.reproducibility_max_probability_drift)
            or not 0.0
            <= self.reproducibility_max_probability_drift
            <= POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
            or not _is_sha256(self.ensemble_sha256)
            or self.ensemble_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket MLP ensemble is invalid")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.validated()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != _FEATURE_COUNT:
            raise ValueError("Polymarket MLP feature matrix is invalid")
        standardized = (
            matrix - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_scale, dtype=np.float64)
        probability = np.mean(
            np.stack(
                [member.predict_standardized(standardized) for member in self.members]
            ),
            axis=0,
        )
        if not np.all(np.isfinite(probability)):
            raise ValueError("Polymarket MLP ensemble probabilities are non-finite")
        return np.asarray(probability, dtype=np.float64)


@dataclass(frozen=True)
class PolymarketMLPBootstrap:
    sample_count: int
    block_length: int
    resamples: int
    mean_delta: float
    lower_95: float
    upper_95: float
    positive_mean_probability: float
    values_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def validated(self) -> PolymarketMLPBootstrap:
        expected_block = (
            max(
                1,
                min(self.sample_count, int(round(math.sqrt(self.sample_count)))),
            )
            if self.sample_count > 0
            else 0
        )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
            or isinstance(self.block_length, bool)
            or not isinstance(self.block_length, int)
            or self.block_length != expected_block
            or self.resamples != POLYMARKET_MLP_BOOTSTRAP_SAMPLES
            or not all(
                math.isfinite(value)
                for value in (
                    self.mean_delta,
                    self.lower_95,
                    self.upper_95,
                    self.positive_mean_probability,
                )
            )
            or self.lower_95 > self.upper_95
            or not 0.0 <= self.positive_mean_probability <= 1.0
            or not _is_sha256(self.values_sha256)
        ):
            raise ValueError("Polymarket MLP bootstrap evidence is invalid")
        return self


@dataclass(frozen=True)
class PolymarketMLPReport:
    dataset_sha256: str
    parent_ridge_report_sha256: str
    split: PolymarketRidgeSplit
    ensemble: PolymarketMLPEnsemble
    validation_log_loss: float
    ridge_validation_log_loss: float
    validation_log_loss_uplift: PolymarketMLPBootstrap
    validation_trials: tuple[PolymarketPolicyMetrics, ...]
    validation_stress_utility_uplift_quote: float | None
    validation_admission_reasons: tuple[str, ...]
    selected_policy: str
    selected_threshold: float | None
    test_evaluated: bool
    test_log_loss: float | None
    test_metrics: PolymarketPolicyMetrics | None
    test_utility_uplift: PolymarketMLPBootstrap | None
    test_gate_reasons: tuple[str, ...]
    development_passed: bool
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_MLP_REPORT_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_MLP_CONTRACT_SHA256,
            "parent_ridge_contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
            "dataset_sha256": self.dataset_sha256,
            "parent_ridge_report_sha256": self.parent_ridge_report_sha256,
            "split": self.split.asdict(),
            "ensemble_sha256": self.ensemble.ensemble_sha256,
            "validation_log_loss": self.validation_log_loss,
            "ridge_validation_log_loss": self.ridge_validation_log_loss,
            "validation_log_loss_uplift": self.validation_log_loss_uplift.asdict(),
            "validation_trials": [item.asdict() for item in self.validation_trials],
            "validation_stress_utility_uplift_quote": (
                self.validation_stress_utility_uplift_quote
            ),
            "validation_admission_reasons": list(self.validation_admission_reasons),
            "selected_policy": self.selected_policy,
            "selected_threshold": self.selected_threshold,
            "test_evaluated": self.test_evaluated,
            "test_log_loss": self.test_log_loss,
            "test_metrics": (
                None if self.test_metrics is None else self.test_metrics.asdict()
            ),
            "test_utility_uplift": (
                None
                if self.test_utility_uplift is None
                else self.test_utility_uplift.asdict()
            ),
            "test_gate_reasons": list(self.test_gate_reasons),
            "development_passed": self.development_passed,
            "test_evaluations": int(self.test_evaluated),
            "foundation_ai_authorized": False,
            "profitability_claim": False,
            "trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> PolymarketMLPReport:
        self.split.validated()
        self.ensemble.validated()
        self.validation_log_loss_uplift.validated()
        for metrics in self.validation_trials:
            metrics.validated(require_asset_profit=False)
        if self.test_metrics is not None:
            self.test_metrics.validated(require_asset_profit=True)
        if self.test_utility_uplift is not None:
            self.test_utility_uplift.validated()
        passing_trials = [item for item in self.validation_trials if item.gate_passed]
        selected_validation = (
            max(
                passing_trials,
                key=lambda item: (
                    item.wilson_lower_bound_95,
                    float(item.threshold or 0.0),
                ),
            )
            if passing_trials
            else None
        )
        expected_admission_reasons: list[str] = []
        if selected_validation is None:
            expected_admission_reasons.append("no_validation_threshold_passed")
        if self.validation_log_loss_uplift.lower_95 <= 0.0:
            expected_admission_reasons.append(
                "validation_log_loss_uplift_lower_not_positive"
            )
        if (
            self.validation_stress_utility_uplift_quote is None
            or self.validation_stress_utility_uplift_quote <= 0.0
        ):
            expected_admission_reasons.append(
                "validation_stress_utility_not_above_ridge"
            )
        frozen_admission_reasons = tuple(sorted(set(expected_admission_reasons)))
        admitted = not frozen_admission_reasons
        expected_policy = "causal_mlp" if admitted else "no_trade"
        expected_threshold = (
            selected_validation.threshold
            if admitted and selected_validation is not None
            else None
        )
        test_reasons_are_canonical = tuple(
            sorted(set(self.test_gate_reasons))
        ) == self.test_gate_reasons and all(
            isinstance(value, str) and value for value in self.test_gate_reasons
        )
        if (
            not _is_sha256(self.dataset_sha256)
            or self.dataset_sha256 != self.ensemble.dataset_sha256
            or not _is_sha256(self.parent_ridge_report_sha256)
            or tuple(item.threshold for item in self.validation_trials)
            != POLYMARKET_RIDGE_THRESHOLD_GRID
            or self.selected_policy not in {"causal_mlp", "no_trade"}
            or (self.selected_policy == "no_trade") != (self.selected_threshold is None)
            or self.validation_admission_reasons != frozen_admission_reasons
            or self.selected_policy != expected_policy
            or self.selected_threshold != expected_threshold
            or self.test_evaluated != (not self.validation_admission_reasons)
            or self.test_evaluated != (self.selected_policy == "causal_mlp")
            or self.test_evaluated
            != (
                self.test_log_loss is not None
                and self.test_metrics is not None
                and self.test_utility_uplift is not None
            )
            or self.development_passed
            != (self.test_evaluated and not self.test_gate_reasons)
            or not math.isfinite(self.validation_log_loss)
            or self.validation_log_loss < 0.0
            or not math.isfinite(self.ridge_validation_log_loss)
            or self.ridge_validation_log_loss < 0.0
            or self.validation_log_loss_uplift.sample_count
            != len(self.split.validation_groups)
            or (
                self.validation_stress_utility_uplift_quote is not None
                and not math.isfinite(self.validation_stress_utility_uplift_quote)
            )
            or (selected_validation is None)
            != (self.validation_stress_utility_uplift_quote is None)
            or tuple(sorted(set(self.validation_admission_reasons)))
            != self.validation_admission_reasons
            or not test_reasons_are_canonical
            or (
                not self.test_evaluated
                and (
                    self.test_log_loss is not None
                    or self.test_metrics is not None
                    or self.test_utility_uplift is not None
                    or self.test_gate_reasons
                )
            )
            or (
                self.test_evaluated
                and (
                    self.test_log_loss is None
                    or not math.isfinite(self.test_log_loss)
                    or self.test_log_loss < 0.0
                    or self.test_metrics is None
                    or self.test_metrics.threshold != self.selected_threshold
                    or self.test_utility_uplift is None
                    or self.test_utility_uplift.sample_count
                    != len(self.split.test_groups)
                    or not set(self.test_metrics.gate_reasons).issubset(
                        self.test_gate_reasons
                    )
                )
            )
            or not _is_sha256(self.report_sha256)
            or self.report_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket MLP report is invalid")
        return self


@dataclass(frozen=True)
class PolymarketMLPMaterialization:
    report_sha256: str
    status: str
    validation_prediction_count: int
    test_prediction_count: int
    selected_validation_action_count: int
    selected_test_action_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def begin_polymarket_mlp_fit(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
) -> PolymarketFitClaim:
    """Claim one ridge parent before nonlinear test access."""

    dataset.validated()
    parent.validated()
    if parent.dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("Polymarket MLP claim dataset is inconsistent")
    return begin_polymarket_fit_claim(
        store,
        experiment="round9_mlp",
        parent_sha256=parent.report_sha256,
        contract_sha256=POLYMARKET_MLP_CONTRACT_SHA256,
        dataset_sha256=dataset.dataset_sha256,
        report_table="polymarket_mlp_report",
        report_parent_column="parent_ridge_report_sha256",
    )


def complete_polymarket_mlp_fit(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
) -> None:
    """Bind a materialized nonlinear report to its fit claim."""

    dataset.validated()
    parent.validated()
    report.validated()
    if (
        parent.dataset_sha256 != dataset.dataset_sha256
        or report.dataset_sha256 != dataset.dataset_sha256
        or report.parent_ridge_report_sha256 != parent.report_sha256
    ):
        raise ValueError("Polymarket MLP completion identity is inconsistent")
    complete_polymarket_fit_claim(
        store,
        experiment="round9_mlp",
        parent_sha256=parent.report_sha256,
        contract_sha256=POLYMARKET_MLP_CONTRACT_SHA256,
        dataset_sha256=dataset.dataset_sha256,
        report_table="polymarket_mlp_report",
        report_parent_column="parent_ridge_report_sha256",
        report_sha256=report.report_sha256,
    )


def fail_polymarket_mlp_fit(
    store: PolymarketEvidenceStore,
    parent: PolymarketRidgeReport,
    error: BaseException,
) -> None:
    """Persist a nonlinear fit failure so test cannot be silently reopened."""

    fail_polymarket_fit_claim(
        store,
        experiment="round9_mlp",
        parent_sha256=parent.report_sha256,
        error=error,
    )


def _partition_indices(
    dataset: PolymarketRidgeDataset,
    groups: Sequence[int],
) -> np.ndarray:
    selected = set(groups)
    return np.asarray(
        [
            index
            for index, item in enumerate(dataset.observations)
            if item.event_start_ms in selected and item.classifier_eligible
        ],
        dtype=np.int64,
    )


def _matrix(
    dataset: PolymarketRidgeDataset,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [dataset.observations[int(index)].feature_values for index in indices],
        dtype=np.float64,
    )
    labels = np.asarray(
        [dataset.observations[int(index)].positive_complete for index in indices],
        dtype=np.float64,
    )
    if (
        features.ndim != 2
        or features.shape[1] != _FEATURE_COUNT
        or labels.shape != (features.shape[0],)
        or not np.all(np.isfinite(features))
        or set(np.unique(labels)) - {0.0, 1.0}
    ):
        raise ValueError("Polymarket MLP matrix is invalid")
    return features, labels


def _validate_partition_label_breadth(
    dataset: PolymarketRidgeDataset,
    *,
    name: str,
    groups: Sequence[int],
) -> None:
    indices = _partition_indices(dataset, groups)
    positive = sum(
        dataset.observations[int(index)].positive_complete for index in indices
    )
    negative = len(indices) - positive
    if positive < 100 or negative < 100:
        raise ValueError(
            f"Polymarket MLP {name} label breadth is insufficient:"
            f"positive={positive}/100 negative={negative}/100"
        )


def _validate_development_breadth(
    dataset: PolymarketRidgeDataset,
    split: PolymarketRidgeSplit,
) -> None:
    if len(dataset.group_starts_ms) < 60:
        raise ValueError(
            f"insufficient synchronized groups:{len(dataset.group_starts_ms)}/60"
        )
    for name, groups in (
        ("train", split.train_groups),
        ("validation", split.validation_groups),
    ):
        _validate_partition_label_breadth(dataset, name=name, groups=groups)


def _condition_weights(
    dataset: PolymarketRidgeDataset,
    indices: np.ndarray,
) -> np.ndarray:
    counts: dict[str, int] = {}
    for index in indices:
        condition = dataset.observations[int(index)].condition_id
        counts[condition] = counts.get(condition, 0) + 1
    weights = np.asarray(
        [
            1.0 / counts[dataset.observations[int(index)].condition_id]
            for index in indices
        ],
        dtype=np.float64,
    )
    weights /= float(np.mean(weights))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Polymarket MLP condition weights are invalid")
    return weights


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _torch_runtime(
    requested_backend: str,
) -> tuple[Any, object, BackendInfo]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError(
            "Polymarket MLP requires the optional torch runtime"
        ) from exc
    requested = requested_backend.strip().lower()
    backend = resolve_backend(requested)
    if requested != "auto" and backend.kind != requested:
        detail = f": {backend.reason}" if backend.reason else ""
        raise RuntimeError(
            f"requested compute backend {requested} resolved to {backend.kind}{detail}"
        )
    if backend.kind == "directml":
        try:
            import torch_directml
        except Exception as exc:  # pragma: no cover - environmental race
            raise RuntimeError("resolved DirectML backend is unavailable") from exc
        device = torch_directml.device()
    else:
        device = torch.device(backend.device)
    return torch, device, backend


def _new_torch_model(torch: Any) -> Any:
    class FixedCausalMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden1 = torch.nn.Linear(_FEATURE_COUNT, 64)
            self.hidden2 = torch.nn.Linear(64, 32)
            self.output = torch.nn.Linear(32, 1)

        def forward(self, values: Any) -> Any:
            values = torch.nn.functional.gelu(self.hidden1(values))
            values = torch.nn.functional.gelu(self.hidden2(values))
            return self.output(values).squeeze(-1)

    return FixedCausalMLP()


def _binary_logit_losses(torch: Any, logits: Any, labels: Any) -> Any:
    if logits.shape != labels.shape:
        raise ValueError("Polymarket MLP binary logit shapes are invalid")
    positive = torch.maximum(logits, torch.zeros_like(logits))
    return positive - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    fallback: list[str] = []
    for value in messages:
        normalized = value.casefold()
        if (
            "dml backend" in normalized
            and "cpu" in normalized
            and ("fall back" in normalized or "fallback" in normalized)
        ):
            fallback.append(value)
    return fallback


class _ExplicitAdamW:
    """Non-foreach AdamW avoids unsupported DirectML fused update operators."""

    def __init__(
        self,
        torch: Any,
        parameters: Sequence[Any],
        *,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
    ) -> None:
        self.torch = torch
        self.parameters = tuple(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.iteration = 0
        self.first_moment = [torch.zeros_like(value) for value in self.parameters]
        self.second_moment = [torch.zeros_like(value) for value in self.parameters]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def step(self) -> None:
        self.iteration += 1
        first_correction = 1.0 - 0.9**self.iteration
        second_correction = 1.0 - 0.999**self.iteration
        step_size = self.learning_rate / first_correction
        decay = 1.0 - self.learning_rate * self.weight_decay
        with self.torch.no_grad():
            for parameter, first, second in zip(
                self.parameters,
                self.first_moment,
                self.second_moment,
                strict=True,
            ):
                gradient = parameter.grad
                if gradient is None:
                    continue
                parameter.mul_(decay)
                first.mul_(0.9).add_(gradient, alpha=0.1)
                second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                denominator = second.sqrt() / math.sqrt(second_correction)
                denominator.add_(1e-8)
                parameter.addcdiv_(first, denominator, value=-step_size)


def _preflight(
    torch: Any,
    device: object,
    backend: BackendInfo,
) -> PolymarketMLPBackendEvidence:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.manual_seed(POLYMARKET_MLP_SEEDS[0])
        model = _new_torch_model(torch).to(device)
        optimizer = _ExplicitAdamW(
            torch,
            tuple(model.parameters()),
        )
        values = (
            torch.linspace(-1.0, 1.0, 8 * _FEATURE_COUNT)
            .reshape(
                8,
                _FEATURE_COUNT,
            )
            .to(device)
        )
        labels = torch.tensor([0.0, 1.0] * 4, dtype=torch.float32).to(device)
        before = model.hidden1.weight.detach().cpu().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        objective = torch.mean(_binary_logit_losses(torch, logits, labels))
        objective.backward()
        optimizer.step()
        parameter_delta = float(
            torch.max(torch.abs(model.hidden1.weight.detach().cpu() - before))
        )
        objective_value = float(objective.detach().cpu())
    fallback = _fallback_messages([str(item.message) for item in caught])
    if fallback:
        raise RuntimeError(f"Polymarket MLP preflight used CPU fallback: {fallback}")
    elapsed = time.perf_counter() - started
    if (
        not math.isfinite(objective_value)
        or not math.isfinite(parameter_delta)
        or parameter_delta <= 0.0
    ):
        raise RuntimeError("Polymarket MLP device preflight failed")
    return PolymarketMLPBackendEvidence(
        requested=backend.requested,
        kind=backend.kind,
        device=str(device),
        vendor=backend.vendor,
        fallback_reason=backend.reason,
        torch_version=str(torch.__version__),
        torch_directml_version=_package_version("torch-directml"),
        preflight_objective=objective_value,
        preflight_parameter_delta=parameter_delta,
        preflight_seconds=elapsed,
        training_seconds=0.0,
        canonical_replay_max_probability_drift=None,
    )


def _predict_torch(
    torch: Any,
    model: Any,
    device: object,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(
            device
        )
        probability = torch.sigmoid(model(values)).detach().cpu().numpy()
    result = np.asarray(probability, dtype=np.float64)
    if result.shape != (features.shape[0],) or not np.all(np.isfinite(result)):
        raise RuntimeError("Polymarket MLP torch prediction is invalid")
    return result


def _extract_member(
    model: Any,
    *,
    seed: int,
    best_epoch: int,
    trace: Sequence[PolymarketMLPEpoch],
) -> PolymarketMLPMember:
    def values(tensor: Any) -> tuple[float, ...]:
        array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        return tuple(float(value) for value in array.reshape(-1))

    provisional = PolymarketMLPMember(
        seed=seed,
        best_epoch=best_epoch,
        epochs_ran=len(trace),
        hidden1_weight=values(model.hidden1.weight),
        hidden1_bias=values(model.hidden1.bias),
        hidden2_weight=values(model.hidden2.weight),
        hidden2_bias=values(model.hidden2.bias),
        output_weight=values(model.output.weight),
        output_bias=float(model.output.bias.detach().cpu().item()),
        trace=tuple(trace),
        member_sha256="",
    )
    return replace(
        provisional,
        member_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _fit_member(
    torch: Any,
    device: object,
    *,
    seed: int,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    training_weights: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    progress: ProgressCallback | None = None,
    run_kind: str = "ensemble",
) -> tuple[PolymarketMLPMember, float]:
    torch.manual_seed(seed)
    model = _new_torch_model(torch).to(device)
    optimizer = _ExplicitAdamW(torch, tuple(model.parameters()))
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    trace: list[PolymarketMLPEpoch] = []
    last_batch_heartbeat = time.perf_counter()
    for epoch in range(1, POLYMARKET_MLP_MAX_EPOCHS + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(
            training_features.shape[0]
        )
        total_loss = 0.0
        total_rows = 0
        for offset in range(0, order.size, POLYMARKET_MLP_BATCH_SIZE):
            selected = order[offset : offset + POLYMARKET_MLP_BATCH_SIZE]
            features = torch.from_numpy(
                np.ascontiguousarray(training_features[selected], dtype=np.float32)
            ).to(device)
            labels = torch.from_numpy(
                np.ascontiguousarray(training_labels[selected], dtype=np.float32)
            ).to(device)
            weights = torch.from_numpy(
                np.ascontiguousarray(training_weights[selected], dtype=np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            losses = _binary_logit_losses(torch, logits, labels)
            loss = torch.mean(losses * weights)
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError("Polymarket MLP training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            optimizer.step()
            rows = int(selected.size)
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
            heartbeat = time.perf_counter()
            if progress is not None and heartbeat - last_batch_heartbeat >= 30.0:
                progress(
                    "polymarket_mlp_batch",
                    {
                        "run_kind": run_kind,
                        "seed": seed,
                        "epoch": epoch,
                        "rows_complete": min(offset + rows, order.size),
                        "rows_total": int(order.size),
                    },
                )
                last_batch_heartbeat = heartbeat
        validation_probability = _predict_torch(
            torch,
            model,
            device,
            validation_features,
        )
        validation_loss = _binary_log_loss(
            validation_probability,
            validation_labels,
        )
        trace.append(
            PolymarketMLPEpoch(
                seed=seed,
                epoch=epoch,
                training_loss=total_loss / total_rows,
                validation_log_loss=validation_loss,
            )
        )
        if validation_loss < best_loss - POLYMARKET_MLP_MIN_DELTA:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress is not None and (
            epoch == 1 or epoch % 5 == 0 or stale_epochs >= POLYMARKET_MLP_PATIENCE
        ):
            progress(
                "polymarket_mlp_epoch",
                {
                    "run_kind": run_kind,
                    "seed": seed,
                    "epoch": epoch,
                    "training_loss": trace[-1].training_loss,
                    "validation_log_loss": validation_loss,
                    "best_validation_log_loss": best_loss,
                    "stale_epochs": stale_epochs,
                },
            )
        if stale_epochs >= POLYMARKET_MLP_PATIENCE:
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Polymarket MLP produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    torch_probability = _predict_torch(
        torch,
        model,
        device,
        validation_features,
    )
    member = _extract_member(
        model,
        seed=seed,
        best_epoch=best_epoch,
        trace=trace,
    )
    canonical_probability = member.predict_standardized(validation_features)
    canonical_replay_drift = float(
        np.max(np.abs(torch_probability - canonical_probability))
    )
    if (
        not math.isfinite(canonical_replay_drift)
        or canonical_replay_drift > POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
    ):
        raise RuntimeError(
            "Polymarket MLP canonical replay drift exceeds tolerance:"
            f"{canonical_replay_drift:.9g}"
        )
    return member, canonical_replay_drift


def _bootstrap(
    values: Sequence[float],
    *,
    seed_material: str,
) -> PolymarketMLPBootstrap:
    observations = tuple(float(value) for value in values)
    if not observations or not all(math.isfinite(value) for value in observations):
        raise ValueError("Polymarket MLP bootstrap values are invalid")
    count = len(observations)
    block = max(1, min(count, int(round(math.sqrt(count)))))
    maximum_start = max(0, count - block)
    seed = int(hashlib.sha256(seed_material.encode("ascii")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(POLYMARKET_MLP_BOOTSTRAP_SAMPLES):
        sample: list[float] = []
        while len(sample) < count:
            start = rng.randint(0, maximum_start) if maximum_start else 0
            sample.extend(observations[start : start + block])
        means.append(math.fsum(sample[:count]) / count)
    lower, upper = np.quantile(np.asarray(means), [0.025, 0.975])
    return PolymarketMLPBootstrap(
        sample_count=count,
        block_length=block,
        resamples=POLYMARKET_MLP_BOOTSTRAP_SAMPLES,
        mean_delta=math.fsum(observations) / count,
        lower_95=float(lower),
        upper_95=float(upper),
        positive_mean_probability=sum(value > 0.0 for value in means) / len(means),
        values_sha256=_sha256(_float_text(observations)),
    )


def _group_log_loss_uplift(
    dataset: PolymarketRidgeDataset,
    indices: np.ndarray,
    ridge_probability: np.ndarray,
    mlp_probability: np.ndarray,
    groups: Sequence[int],
) -> tuple[float, ...]:
    values: list[float] = []
    for group in groups:
        selected = np.asarray(
            [
                offset
                for offset, index in enumerate(indices)
                if dataset.observations[int(index)].event_start_ms == group
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [
                dataset.observations[int(indices[offset])].positive_complete
                for offset in selected
            ],
            dtype=np.float64,
        )
        values.append(
            _binary_log_loss(ridge_probability[selected], labels)
            - _binary_log_loss(mlp_probability[selected], labels)
        )
    return tuple(values)


def _condition_utility(
    dataset: PolymarketRidgeDataset,
    evaluation: PolymarketPolicyEvaluation,
    groups: Sequence[int],
) -> dict[str, float]:
    allowed = set(groups)
    result = {
        item.condition_id: 0.0
        for item in dataset.observations
        if item.event_start_ms in allowed
    }
    for index in evaluation.selected_indices:
        item = dataset.observations[index]
        result[item.condition_id] += item.stress_utility_quote
    return result


def _group_utility_uplift(
    dataset: PolymarketRidgeDataset,
    groups: Sequence[int],
    baseline: Mapping[str, float],
    challenger: Mapping[str, float],
) -> tuple[float, ...]:
    values: list[float] = []
    for group in groups:
        conditions = sorted(
            {
                item.condition_id
                for item in dataset.observations
                if item.event_start_ms == group
            }
        )
        values.append(
            math.fsum(
                float(challenger[condition]) - float(baseline[condition])
                for condition in conditions
            )
        )
    return tuple(values)


def fit_and_evaluate_polymarket_mlp(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    *,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> PolymarketMLPReport:
    """Fit the preregistered nonlinear challenger and open test only if admitted."""

    dataset.validated()
    parent.validated()
    expected_split = split_polymarket_ridge_dataset(dataset)
    if (
        parent.dataset_sha256 != dataset.dataset_sha256
        or parent.split != expected_split
        or not parent.development_passed
    ):
        raise ValueError("Polymarket MLP parent ridge authority is insufficient")
    _validate_development_breadth(dataset, expected_split)
    if len(expected_split.test_groups) < POLYMARKET_MLP_MIN_TEST_GROUPS:
        raise ValueError(
            "insufficient untouched test groups:"
            f"{len(expected_split.test_groups)}/{POLYMARKET_MLP_MIN_TEST_GROUPS}"
        )
    train_indices = _partition_indices(dataset, expected_split.train_groups)
    validation_indices = _partition_indices(
        dataset,
        expected_split.validation_groups,
    )
    training_x, training_y = _matrix(dataset, train_indices)
    validation_x, validation_y = _matrix(dataset, validation_indices)
    mean = np.mean(training_x, axis=0, dtype=np.float64)
    scale = np.std(training_x, axis=0, dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized_training = (training_x - mean) / scale
    standardized_validation = (validation_x - mean) / scale
    weights = _condition_weights(dataset, train_indices)
    torch, device, backend = _torch_runtime(compute_backend)
    training_started = time.perf_counter()
    backend_evidence = _preflight(torch, device, backend)
    if progress is not None:
        progress(
            "polymarket_mlp_preflight",
            {
                "backend": backend_evidence.kind,
                "device": backend_evidence.device,
                "fallback_reason": backend_evidence.fallback_reason,
                "training_rows": int(training_x.shape[0]),
                "validation_rows": int(validation_x.shape[0]),
            },
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted_members: list[PolymarketMLPMember] = []
        canonical_replay_drifts: list[float] = []
        for seed in POLYMARKET_MLP_SEEDS:
            if progress is not None:
                progress("polymarket_mlp_seed", {"seed": seed, "status": "started"})
            member, canonical_replay_drift = _fit_member(
                torch,
                device,
                seed=seed,
                training_features=standardized_training,
                training_labels=training_y,
                training_weights=weights,
                validation_features=standardized_validation,
                validation_labels=validation_y,
                progress=progress,
            )
            fitted_members.append(member)
            canonical_replay_drifts.append(canonical_replay_drift)
            if progress is not None:
                progress(
                    "polymarket_mlp_seed",
                    {
                        "seed": seed,
                        "status": "complete",
                        "best_epoch": member.best_epoch,
                        "epochs_ran": member.epochs_ran,
                        "canonical_replay_max_probability_drift": (
                            canonical_replay_drift
                        ),
                    },
                )
        members = tuple(fitted_members)
        if progress is not None:
            progress(
                "polymarket_mlp_reproducibility",
                {"seed": POLYMARKET_MLP_SEEDS[0], "status": "started"},
            )
        repeated, repeated_replay_drift = _fit_member(
            torch,
            device,
            seed=POLYMARKET_MLP_SEEDS[0],
            training_features=standardized_training,
            training_labels=training_y,
            training_weights=weights,
            validation_features=standardized_validation,
            validation_labels=validation_y,
            progress=progress,
            run_kind="reproducibility",
        )
        canonical_replay_drifts.append(repeated_replay_drift)
    fallback = _fallback_messages([str(item.message) for item in caught])
    if fallback:
        raise RuntimeError(f"Polymarket MLP training used CPU fallback: {fallback}")
    backend_evidence = replace(
        backend_evidence,
        training_seconds=time.perf_counter() - training_started,
        canonical_replay_max_probability_drift=max(canonical_replay_drifts),
    )
    first_probability = members[0].predict_standardized(standardized_validation)
    repeated_probability = repeated.predict_standardized(standardized_validation)
    reproducibility_drift = float(
        np.max(np.abs(first_probability - repeated_probability))
    )
    if reproducibility_drift > POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE:
        raise ValueError(
            "Polymarket MLP same-seed probability drift exceeds tolerance:"
            f"{reproducibility_drift:.9g}"
        )
    if progress is not None:
        progress(
            "polymarket_mlp_reproducibility",
            {
                "seed": POLYMARKET_MLP_SEEDS[0],
                "status": "complete",
                "maximum_probability_drift": reproducibility_drift,
            },
        )
    provisional_ensemble = PolymarketMLPEnsemble(
        dataset_sha256=dataset.dataset_sha256,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        members=members,
        backend=backend_evidence,
        reproducibility_max_probability_drift=reproducibility_drift,
        ensemble_sha256="",
    )
    ensemble = replace(
        provisional_ensemble,
        ensemble_sha256=_sha256(provisional_ensemble.identity_payload()),
    ).validated()
    validation_probability = ensemble.predict(validation_x)
    validation_loss = _binary_log_loss(validation_probability, validation_y)
    ridge_validation_probability = parent.selected_model.predict(validation_x)
    ridge_validation_loss = _binary_log_loss(
        ridge_validation_probability,
        validation_y,
    )
    if not math.isclose(
        ridge_validation_loss,
        parent.selected_validation_log_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Polymarket MLP parent validation replay differs")
    validation_uplift = _bootstrap(
        _group_log_loss_uplift(
            dataset,
            validation_indices,
            ridge_validation_probability,
            validation_probability,
            expected_split.validation_groups,
        ),
        seed_material=dataset.dataset_sha256 + "validation-log-loss",
    )
    validation_evaluations = tuple(
        evaluate_polymarket_policy(
            dataset,
            validation_indices,
            validation_probability,
            threshold,
            require_asset_profit=False,
        )
        for threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
    )
    passed = [item for item in validation_evaluations if item.metrics.gate_passed]
    selected_validation = (
        max(
            passed,
            key=lambda item: (
                item.metrics.wilson_lower_bound_95,
                float(item.metrics.threshold or 0.0),
            ),
        )
        if passed
        else None
    )
    ridge_validation_evaluation = evaluate_polymarket_policy(
        dataset,
        validation_indices,
        ridge_validation_probability,
        parent.selected_threshold,
        require_asset_profit=False,
    )
    expected_ridge_validation = next(
        (
            item
            for item in parent.validation_trials
            if item.threshold == parent.selected_threshold
        ),
        None,
    )
    if (
        expected_ridge_validation is None
        or ridge_validation_evaluation.metrics.asdict()
        != expected_ridge_validation.asdict()
    ):
        raise ValueError("Polymarket MLP parent-policy validation simulation differs")
    validation_utility_uplift = (
        None
        if selected_validation is None
        else (
            selected_validation.metrics.aggregate_stress_utility_quote
            - ridge_validation_evaluation.metrics.aggregate_stress_utility_quote
        )
    )
    admission_reasons: list[str] = []
    if selected_validation is None:
        admission_reasons.append("no_validation_threshold_passed")
    if validation_uplift.lower_95 <= 0.0:
        admission_reasons.append("validation_log_loss_uplift_lower_not_positive")
    if validation_utility_uplift is None or validation_utility_uplift <= 0.0:
        admission_reasons.append("validation_stress_utility_not_above_ridge")
    frozen_admission_reasons = tuple(sorted(set(admission_reasons)))
    admitted = not frozen_admission_reasons
    if progress is not None:
        progress(
            "polymarket_mlp_validation",
            {
                "admitted_to_test": admitted,
                "admission_reasons": list(frozen_admission_reasons),
                "validation_log_loss": validation_loss,
                "ridge_validation_log_loss": ridge_validation_loss,
                "log_loss_uplift_lower_95": validation_uplift.lower_95,
                "stress_utility_uplift_quote": validation_utility_uplift,
                "untouched_test_group_count": len(expected_split.test_groups),
            },
        )
    if admitted:
        if progress is not None:
            progress("polymarket_mlp_test", {"status": "started"})
        if selected_validation is None:
            raise RuntimeError("Polymarket MLP admission state is inconsistent")
        selected_policy = "causal_mlp"
        selected_threshold = selected_validation.metrics.threshold
    else:
        selected_policy = "no_trade"
        selected_threshold = None
    test_loss: float | None = None
    test_metrics: PolymarketPolicyMetrics | None = None
    test_uplift: PolymarketMLPBootstrap | None = None
    test_reasons: list[str] = []
    if admitted:
        _validate_partition_label_breadth(
            dataset,
            name="test",
            groups=expected_split.test_groups,
        )
        test_indices = _partition_indices(dataset, expected_split.test_groups)
        test_x, test_y = _matrix(dataset, test_indices)
        test_probability = ensemble.predict(test_x)
        test_loss = _binary_log_loss(test_probability, test_y)
        test_evaluation = evaluate_polymarket_policy(
            dataset,
            test_indices,
            test_probability,
            selected_threshold,
            require_asset_profit=True,
        )
        ridge_test_probability = parent.selected_model.predict(test_x)
        ridge_test_evaluation = evaluate_polymarket_policy(
            dataset,
            test_indices,
            ridge_test_probability,
            parent.selected_threshold,
            require_asset_profit=True,
        )
        if ridge_test_evaluation.metrics.asdict() != parent.test_metrics.asdict():
            raise ValueError("Polymarket MLP parent test replay differs")
        baseline_utility = _condition_utility(
            dataset,
            ridge_test_evaluation,
            expected_split.test_groups,
        )
        challenger_utility = _condition_utility(
            dataset,
            test_evaluation,
            expected_split.test_groups,
        )
        test_uplift = _bootstrap(
            _group_utility_uplift(
                dataset,
                expected_split.test_groups,
                baseline_utility,
                challenger_utility,
            ),
            seed_material=dataset.dataset_sha256 + "test-utility",
        )
        test_metrics = test_evaluation.metrics
        if not test_metrics.gate_passed:
            test_reasons.extend(test_metrics.gate_reasons)
        if test_loss >= parent.test_log_loss:
            test_reasons.append("test_log_loss_not_below_ridge")
        if (
            test_metrics.aggregate_stress_utility_quote
            <= parent.test_metrics.aggregate_stress_utility_quote
        ):
            test_reasons.append("test_stress_utility_not_above_ridge")
        if (
            test_metrics.maximum_realized_drawdown_quote
            > parent.test_metrics.maximum_realized_drawdown_quote
        ):
            test_reasons.append("test_realized_drawdown_above_ridge")
        for asset in sorted(parent.test_metrics.pnl_by_asset):
            if (
                test_metrics.pnl_by_asset[asset]
                < parent.test_metrics.pnl_by_asset[asset]
            ):
                test_reasons.append(f"test_asset_utility_below_ridge:{asset}")
        if test_uplift.lower_95 <= 0.0:
            test_reasons.append("test_utility_bootstrap_lower_not_positive")
        if progress is not None:
            progress(
                "polymarket_mlp_test",
                {
                    "status": "complete",
                    "test_log_loss": test_loss,
                    "test_stress_utility_quote": (
                        test_metrics.aggregate_stress_utility_quote
                    ),
                    "gate_reason_count": len(test_reasons),
                },
            )
    provisional = PolymarketMLPReport(
        dataset_sha256=dataset.dataset_sha256,
        parent_ridge_report_sha256=parent.report_sha256,
        split=expected_split,
        ensemble=ensemble,
        validation_log_loss=validation_loss,
        ridge_validation_log_loss=ridge_validation_loss,
        validation_log_loss_uplift=validation_uplift,
        validation_trials=tuple(item.metrics for item in validation_evaluations),
        validation_stress_utility_uplift_quote=validation_utility_uplift,
        validation_admission_reasons=frozen_admission_reasons,
        selected_policy=selected_policy,
        selected_threshold=selected_threshold,
        test_evaluated=admitted,
        test_log_loss=test_loss,
        test_metrics=test_metrics,
        test_utility_uplift=test_uplift,
        test_gate_reasons=tuple(sorted(set(test_reasons))),
        development_passed=admitted and not test_reasons,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _prediction_rows(
    dataset: PolymarketRidgeDataset,
    *,
    report_sha256: str,
    partition: str,
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> list[tuple[object, ...]]:
    if probabilities.shape != indices.shape:
        raise ValueError("Polymarket MLP persisted predictions are misaligned")
    return [
        (
            report_sha256,
            partition,
            sequence,
            int(index),
            dataset.observations[int(index)].action_feature_sha256,
            dataset.observations[int(index)].condition_id,
            dataset.observations[int(index)].event_start_ms,
            dataset.observations[int(index)].decision_received_monotonic_ns,
            float(probability),
            dataset.observations[int(index)].positive_complete,
            dataset.observations[int(index)].category,
            format(
                dataset.observations[int(index)].stress_utility_quote,
                ".17g",
            ),
        )
        for sequence, (index, probability) in enumerate(
            zip(indices, probabilities, strict=True)
        )
    ]


def materialize_polymarket_mlp_report(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
) -> PolymarketMLPMaterialization:
    """Persist the nonlinear model and every reconstructable result atomically."""

    dataset.validated()
    parent.validated()
    report.validated()
    if (
        report.dataset_sha256 != dataset.dataset_sha256
        or report.parent_ridge_report_sha256 != parent.report_sha256
        or parent.dataset_sha256 != dataset.dataset_sha256
        or report.split != parent.split
    ):
        raise ValueError("Polymarket MLP materialization identity differs")
    validation_indices = _partition_indices(dataset, report.split.validation_groups)
    validation_x, _validation_y = _matrix(dataset, validation_indices)
    validation_probability = report.ensemble.predict(validation_x)
    validation_evaluation, validation_actions, validation_equity, validation_markets = (
        polymarket_selected_policy_tables(
            dataset,
            report_sha256=report.report_sha256,
            partition="validation",
            indices=validation_indices,
            probabilities=validation_probability,
            threshold=report.selected_threshold,
            require_asset_profit=False,
        )
    )
    if report.selected_threshold is not None:
        expected_validation = next(
            item
            for item in report.validation_trials
            if item.threshold == report.selected_threshold
        )
        if validation_evaluation.metrics.asdict() != expected_validation.asdict():
            raise ValueError("Polymarket MLP validation replay differs from report")
    validation_predictions = _prediction_rows(
        dataset,
        report_sha256=report.report_sha256,
        partition="validation",
        indices=validation_indices,
        probabilities=validation_probability,
    )
    test_predictions: list[tuple[object, ...]] = []
    test_actions: list[tuple[object, ...]] = []
    test_equity: list[tuple[object, ...]] = []
    test_markets: list[tuple[object, ...]] = []
    if report.test_evaluated:
        test_indices = _partition_indices(dataset, report.split.test_groups)
        test_x, _test_y = _matrix(dataset, test_indices)
        test_probability = report.ensemble.predict(test_x)
        test_evaluation, test_actions, test_equity, test_markets = (
            polymarket_selected_policy_tables(
                dataset,
                report_sha256=report.report_sha256,
                partition="test",
                indices=test_indices,
                probabilities=test_probability,
                threshold=report.selected_threshold,
                require_asset_profit=True,
            )
        )
        if (
            report.test_metrics is None
            or test_evaluation.metrics.asdict() != report.test_metrics.asdict()
        ):
            raise ValueError("Polymarket MLP test replay differs from report")
        test_predictions = _prediction_rows(
            dataset,
            report_sha256=report.report_sha256,
            partition="test",
            indices=test_indices,
            probabilities=test_probability,
        )
    member_rows = [
        (
            report.report_sha256,
            member.seed,
            member.member_sha256,
            member.best_epoch,
            member.epochs_ran,
            _canonical_json(
                {
                    **member.identity_payload(),
                    "member_sha256": member.member_sha256,
                }
            ),
        )
        for member in report.ensemble.members
    ]
    trace_rows = [
        (
            report.report_sha256,
            item.seed,
            item.epoch,
            item.training_loss,
            item.validation_log_loss,
        )
        for member in report.ensemble.members
        for item in member.trace
    ]
    prediction_rows = validation_predictions + test_predictions
    selected_rows = validation_actions + test_actions
    equity_rows = validation_equity + test_equity
    market_rows = validation_markets + test_markets
    report_row = (
        report.report_sha256,
        POLYMARKET_MLP_REPORT_SCHEMA_VERSION,
        POLYMARKET_MLP_CONTRACT_SHA256,
        dataset.dataset_sha256,
        parent.report_sha256,
        report.ensemble.ensemble_sha256,
        report.selected_policy,
        report.selected_threshold,
        report.test_evaluated,
        report.development_passed,
        _canonical_json(report.asdict()),
        _canonical_json(
            {
                **report.ensemble.identity_payload(),
                "ensemble_sha256": report.ensemble.ensemble_sha256,
            }
        ),
    )
    runtime_json = _canonical_json(report.ensemble.backend.asdict())
    runtime_row = (
        report.report_sha256,
        _sha256(report.ensemble.backend.asdict()),
        runtime_json,
    )
    connection = store.connect()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_mlp_report (
            report_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            dataset_sha256 VARCHAR NOT NULL,
            parent_ridge_report_sha256 VARCHAR NOT NULL,
            ensemble_sha256 VARCHAR NOT NULL,
            selected_policy VARCHAR NOT NULL,
            selected_threshold DOUBLE,
            test_evaluated BOOLEAN NOT NULL,
            development_passed BOOLEAN NOT NULL,
            report_json VARCHAR NOT NULL,
            ensemble_json VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_member (
            report_sha256 VARCHAR NOT NULL,
            seed INTEGER NOT NULL,
            member_sha256 VARCHAR NOT NULL,
            best_epoch INTEGER NOT NULL,
            epochs_ran INTEGER NOT NULL,
            model_json VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, seed)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_runtime_evidence (
            report_sha256 VARCHAR NOT NULL,
            runtime_sha256 VARCHAR NOT NULL,
            backend_json VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, runtime_sha256)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_epoch (
            report_sha256 VARCHAR NOT NULL,
            seed INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            training_loss DOUBLE NOT NULL,
            validation_log_loss DOUBLE NOT NULL,
            PRIMARY KEY(report_sha256, seed, epoch)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_prediction (
            report_sha256 VARCHAR NOT NULL,
            partition VARCHAR NOT NULL,
            sequence UBIGINT NOT NULL,
            dataset_observation_index UBIGINT NOT NULL,
            action_feature_sha256 VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            decision_received_monotonic_ns UBIGINT NOT NULL,
            probability DOUBLE NOT NULL,
            positive_complete BOOLEAN NOT NULL,
            category VARCHAR NOT NULL,
            stress_utility_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, sequence)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_selected_action (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            sequence UBIGINT NOT NULL, action_feature_sha256 VARCHAR NOT NULL,
            action_label_sha256 VARCHAR NOT NULL, condition_id VARCHAR NOT NULL,
            asset VARCHAR NOT NULL, outcome VARCHAR NOT NULL,
            event_start_ms BIGINT NOT NULL,
            decision_received_monotonic_ns UBIGINT NOT NULL,
            release_monotonic_ns UBIGINT NOT NULL, probability DOUBLE NOT NULL,
            category VARCHAR NOT NULL, positive_complete BOOLEAN NOT NULL,
            condition_blocked BOOLEAN NOT NULL,
            stress_utility_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, sequence)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_equity (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            sequence UBIGINT NOT NULL, release_monotonic_ns UBIGINT NOT NULL,
            action_feature_sha256 VARCHAR NOT NULL, pnl_quote VARCHAR NOT NULL,
            equity_quote VARCHAR NOT NULL, drawdown_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, sequence)
        );
        CREATE TABLE IF NOT EXISTS polymarket_mlp_market_pnl (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL, asset VARCHAR NOT NULL,
            attempt_count UBIGINT NOT NULL, completed_trade_count UBIGINT NOT NULL,
            pnl_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, condition_id)
        );
        """
    )
    tables = (
        ("polymarket_mlp_member", member_rows, "seed", lambda row: row[1]),
        (
            "polymarket_mlp_epoch",
            trace_rows,
            "seed, epoch",
            lambda row: (row[1], row[2]),
        ),
        (
            "polymarket_mlp_prediction",
            prediction_rows,
            "partition, sequence",
            lambda row: (row[1], row[2]),
        ),
        (
            "polymarket_mlp_selected_action",
            selected_rows,
            "partition, sequence",
            lambda row: (row[1], row[2]),
        ),
        (
            "polymarket_mlp_equity",
            equity_rows,
            "partition, sequence",
            lambda row: (row[1], row[2]),
        ),
        (
            "polymarket_mlp_market_pnl",
            market_rows,
            "partition, condition_id",
            lambda row: (row[1], row[2]),
        ),
    )
    existing = connection.execute(
        "SELECT * FROM polymarket_mlp_report WHERE report_sha256 = ?",
        [report.report_sha256],
    ).fetchone()
    if existing is not None:
        if tuple(existing) != report_row:
            raise ValueError("stored Polymarket MLP report is inconsistent")
        for table, expected, ordering, sort_key in tables:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE report_sha256 = ? ORDER BY {ordering}",
                [report.report_sha256],
            ).fetchall()
            if [tuple(row) for row in rows] != sorted(expected, key=sort_key):
                raise ValueError(f"stored {table} rows are inconsistent")
        stored_runtime = connection.execute(
            """
            SELECT backend_json FROM polymarket_mlp_runtime_evidence
            WHERE report_sha256 = ? AND runtime_sha256 = ?
            """,
            runtime_row[:2],
        ).fetchone()
        if stored_runtime is None:
            connection.execute(
                "INSERT INTO polymarket_mlp_runtime_evidence VALUES (?, ?, ?)",
                runtime_row,
            )
        elif str(stored_runtime[0]) != runtime_json:
            raise ValueError("stored Polymarket MLP runtime evidence is inconsistent")
        status = "existing"
    else:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "INSERT INTO polymarket_mlp_report VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                report_row,
            )
            connection.execute(
                "INSERT INTO polymarket_mlp_runtime_evidence VALUES (?, ?, ?)",
                runtime_row,
            )
            for table, rows, _ordering, _sort_key in tables:
                if rows:
                    placeholders = ", ".join("?" for _ in rows[0])
                    connection.executemany(
                        f"INSERT INTO {table} VALUES ({placeholders})",
                        rows,
                    )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        status = "created"
    return PolymarketMLPMaterialization(
        report_sha256=report.report_sha256,
        status=status,
        validation_prediction_count=len(validation_predictions),
        test_prediction_count=len(test_predictions),
        selected_validation_action_count=len(validation_actions),
        selected_test_action_count=len(test_actions),
    )


__all__ = [
    "POLYMARKET_MLP_BATCH_SIZE",
    "POLYMARKET_MLP_CONTRACT_SHA256",
    "POLYMARKET_MLP_MODEL_SCHEMA_VERSION",
    "POLYMARKET_MLP_MIN_TEST_GROUPS",
    "POLYMARKET_MLP_REPORT_SCHEMA_VERSION",
    "POLYMARKET_MLP_SEEDS",
    "PolymarketMLPBackendEvidence",
    "PolymarketMLPBootstrap",
    "PolymarketMLPEnsemble",
    "PolymarketMLPEpoch",
    "PolymarketMLPMaterialization",
    "PolymarketMLPMember",
    "PolymarketMLPReport",
    "begin_polymarket_mlp_fit",
    "complete_polymarket_mlp_fit",
    "fail_polymarket_mlp_fit",
    "fit_and_evaluate_polymarket_mlp",
    "materialize_polymarket_mlp_report",
]

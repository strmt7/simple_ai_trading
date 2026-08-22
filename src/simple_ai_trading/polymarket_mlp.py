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

from .compute import (
    BackendInfo,
    require_backend,
    resolve_backend,
    torch_device_for_backend,
)
from .polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES
from .polymarket_fit_claim import (
    PolymarketFitClaim,
    complete_polymarket_fit_claim,
    consume_polymarket_fit_claim,
    fail_polymarket_fit_claim,
)
from .polymarket_model_contracts import POLYMARKET_ROUND9_MLP_CONTRACT_SHA256
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


POLYMARKET_MLP_CONTRACT_SHA256 = POLYMARKET_ROUND9_MLP_CONTRACT_SHA256
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
        if not _backend_selection_is_valid(self):
            raise ValueError("Polymarket MLP backend evidence is invalid")
        if not _backend_identity_is_valid(self):
            raise ValueError("Polymarket MLP backend evidence is invalid")
        if not _backend_runtime_is_valid(self):
            raise ValueError("Polymarket MLP backend evidence is invalid")
        return self


def _backend_selection_is_valid(evidence: PolymarketMLPBackendEvidence) -> bool:
    allowed = {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
    return (
        evidence.requested in {"auto", *allowed}
        and evidence.kind in allowed
        and evidence.requested in ("auto", evidence.kind)
    )


def _backend_identity_is_valid(evidence: PolymarketMLPBackendEvidence) -> bool:
    return bool(
        evidence.device
        and evidence.vendor
        and evidence.torch_version
        and (evidence.kind != "directml" or evidence.torch_directml_version)
    )


def _backend_runtime_is_valid(evidence: PolymarketMLPBackendEvidence) -> bool:
    values = (
        evidence.preflight_objective,
        evidence.preflight_parameter_delta,
        evidence.preflight_seconds,
        evidence.training_seconds,
    )
    replay_drift = evidence.canonical_replay_max_probability_drift
    return (
        all(math.isfinite(value) and value >= 0.0 for value in values)
        and evidence.preflight_parameter_delta > 0.0
        and evidence.training_seconds > 0.0
        and replay_drift is not None
        and math.isfinite(replay_drift)
        and 0.0 <= replay_drift <= POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
    )


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
        values = _member_parameter_values(self)
        expected_best_epoch = _member_expected_best_epoch(self.trace)
        if not _member_header_is_valid(self):
            raise ValueError("Polymarket MLP member is invalid")
        if not _member_parameters_are_valid(self, values):
            raise ValueError("Polymarket MLP member is invalid")
        if not _member_trace_is_valid(self, expected_best_epoch):
            raise ValueError("Polymarket MLP member is invalid")
        if not _member_hash_is_valid(self):
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


def _member_parameter_values(member: PolymarketMLPMember) -> tuple[float, ...]:
    return (
        *member.hidden1_weight,
        *member.hidden1_bias,
        *member.hidden2_weight,
        *member.hidden2_bias,
        *member.output_weight,
        member.output_bias,
    )


def _member_expected_best_epoch(trace: Sequence[PolymarketMLPEpoch]) -> int:
    best_loss = math.inf
    expected_best_epoch = 0
    for item in trace:
        if item.validation_log_loss < best_loss - POLYMARKET_MLP_MIN_DELTA:
            best_loss = item.validation_log_loss
            expected_best_epoch = item.epoch
    return expected_best_epoch


def _member_header_is_valid(member: PolymarketMLPMember) -> bool:
    return (
        member.seed in POLYMARKET_MLP_SEEDS
        and 1 <= member.best_epoch <= member.epochs_ran <= POLYMARKET_MLP_MAX_EPOCHS
    )


def _member_parameters_are_valid(
    member: PolymarketMLPMember,
    values: Sequence[float],
) -> bool:
    return (
        len(member.hidden1_weight) == 64 * _FEATURE_COUNT
        and len(member.hidden1_bias) == 64
        and len(member.hidden2_weight) == 32 * 64
        and len(member.hidden2_bias) == 32
        and len(member.output_weight) == 32
        and len(values) == _MEMBER_PARAMETER_COUNT
        and all(math.isfinite(value) for value in values)
    )


def _member_trace_is_valid(
    member: PolymarketMLPMember,
    expected_best_epoch: int,
) -> bool:
    expected_epochs = tuple(range(1, member.epochs_ran + 1))
    return (
        len(member.trace) == member.epochs_ran
        and tuple(item.epoch for item in member.trace) == expected_epochs
        and all(
            item.seed == member.seed
            and math.isfinite(item.training_loss)
            and math.isfinite(item.validation_log_loss)
            and item.training_loss >= 0.0
            and item.validation_log_loss >= 0.0
            for item in member.trace
        )
        and member.best_epoch == expected_best_epoch
    )


def _member_hash_is_valid(member: PolymarketMLPMember) -> bool:
    return _is_sha256(member.member_sha256) and member.member_sha256 == _sha256(
        member.identity_payload()
    )


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
        if not _ensemble_features_are_valid(self):
            raise ValueError("Polymarket MLP ensemble is invalid")
        if not _ensemble_members_are_valid(self):
            raise ValueError("Polymarket MLP ensemble is invalid")
        if not _ensemble_reproducibility_is_valid(self):
            raise ValueError("Polymarket MLP ensemble is invalid")
        if not _ensemble_hash_is_valid(self):
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


def _ensemble_features_are_valid(ensemble: PolymarketMLPEnsemble) -> bool:
    return (
        _is_sha256(ensemble.dataset_sha256)
        and len(ensemble.feature_mean) == _FEATURE_COUNT
        and len(ensemble.feature_scale) == _FEATURE_COUNT
        and all(math.isfinite(value) for value in ensemble.feature_mean)
        and all(
            math.isfinite(value) and value > 0.0 for value in ensemble.feature_scale
        )
    )


def _ensemble_members_are_valid(ensemble: PolymarketMLPEnsemble) -> bool:
    return tuple(item.seed for item in ensemble.members) == POLYMARKET_MLP_SEEDS


def _ensemble_reproducibility_is_valid(ensemble: PolymarketMLPEnsemble) -> bool:
    drift = ensemble.reproducibility_max_probability_drift
    return (
        math.isfinite(drift)
        and 0.0 <= drift <= POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
    )


def _ensemble_hash_is_valid(ensemble: PolymarketMLPEnsemble) -> bool:
    return _is_sha256(ensemble.ensemble_sha256) and ensemble.ensemble_sha256 == _sha256(
        ensemble.identity_payload()
    )


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
class _PolymarketMLPReportExpectation:
    selected_validation: PolymarketPolicyMetrics | None
    admission_reasons: tuple[str, ...]
    selected_policy: str
    selected_threshold: float | None


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
        _validate_mlp_report_children(self)
        expectation = _mlp_report_expectation(self)
        test_reasons_are_canonical = _canonical_report_reasons(self.test_gate_reasons)
        if (
            not _mlp_report_identity_is_valid(self)
            or not _mlp_report_selection_is_valid(self, expectation)
            or not _mlp_report_validation_evidence_is_valid(self, expectation)
            or not test_reasons_are_canonical
            or not _mlp_report_test_evidence_is_valid(self)
            or not _mlp_report_hash_is_valid(self)
        ):
            raise ValueError("Polymarket MLP report is invalid")
        return self


def _validate_mlp_report_children(report: PolymarketMLPReport) -> None:
    report.split.validated()
    report.ensemble.validated()
    report.validation_log_loss_uplift.validated()
    for metrics in report.validation_trials:
        metrics.validated(require_asset_profit=False)
    if report.test_metrics is not None:
        report.test_metrics.validated(require_asset_profit=True)
    if report.test_utility_uplift is not None:
        report.test_utility_uplift.validated()


def _best_passing_validation_trial(
    trials: Sequence[PolymarketPolicyMetrics],
) -> PolymarketPolicyMetrics | None:
    passing_trials = [item for item in trials if item.gate_passed]
    if not passing_trials:
        return None
    return max(
        passing_trials,
        key=lambda item: (
            item.wilson_lower_bound_95,
            float(item.threshold or 0.0),
        ),
    )


def _mlp_report_expectation(
    report: PolymarketMLPReport,
) -> _PolymarketMLPReportExpectation:
    selected_validation = _best_passing_validation_trial(report.validation_trials)
    admission_reasons: list[str] = []
    if selected_validation is None:
        admission_reasons.append("no_validation_threshold_passed")
    if report.validation_log_loss_uplift.lower_95 <= 0.0:
        admission_reasons.append("validation_log_loss_uplift_lower_not_positive")
    if (
        report.validation_stress_utility_uplift_quote is None
        or report.validation_stress_utility_uplift_quote <= 0.0
    ):
        admission_reasons.append("validation_stress_utility_not_above_ridge")
    frozen_reasons = tuple(sorted(set(admission_reasons)))
    admitted = not frozen_reasons
    return _PolymarketMLPReportExpectation(
        selected_validation=selected_validation,
        admission_reasons=frozen_reasons,
        selected_policy="causal_mlp" if admitted else "no_trade",
        selected_threshold=(
            selected_validation.threshold
            if admitted and selected_validation is not None
            else None
        ),
    )


def _canonical_report_reasons(reasons: tuple[str, ...]) -> bool:
    return tuple(sorted(set(reasons))) == reasons and all(
        isinstance(value, str) and value for value in reasons
    )


def _mlp_report_identity_is_valid(report: PolymarketMLPReport) -> bool:
    return (
        _is_sha256(report.dataset_sha256)
        and report.dataset_sha256 == report.ensemble.dataset_sha256
        and _is_sha256(report.parent_ridge_report_sha256)
        and tuple(item.threshold for item in report.validation_trials)
        == POLYMARKET_RIDGE_THRESHOLD_GRID
    )


def _mlp_report_selection_is_valid(
    report: PolymarketMLPReport,
    expectation: _PolymarketMLPReportExpectation,
) -> bool:
    test_payload_is_complete = (
        report.test_log_loss is not None
        and report.test_metrics is not None
        and report.test_utility_uplift is not None
    )
    return (
        report.selected_policy in {"causal_mlp", "no_trade"}
        and (report.selected_policy == "no_trade")
        == (report.selected_threshold is None)
        and report.validation_admission_reasons == expectation.admission_reasons
        and report.selected_policy == expectation.selected_policy
        and report.selected_threshold == expectation.selected_threshold
        and report.test_evaluated == (not report.validation_admission_reasons)
        and report.test_evaluated == (report.selected_policy == "causal_mlp")
        and report.test_evaluated == test_payload_is_complete
        and report.development_passed
        == (report.test_evaluated and not report.test_gate_reasons)
    )


def _mlp_report_validation_evidence_is_valid(
    report: PolymarketMLPReport,
    expectation: _PolymarketMLPReportExpectation,
) -> bool:
    stress_uplift = report.validation_stress_utility_uplift_quote
    return (
        math.isfinite(report.validation_log_loss)
        and report.validation_log_loss >= 0.0
        and math.isfinite(report.ridge_validation_log_loss)
        and report.ridge_validation_log_loss >= 0.0
        and report.validation_log_loss_uplift.sample_count
        == len(report.split.validation_groups)
        and (stress_uplift is None or math.isfinite(stress_uplift))
        and (expectation.selected_validation is None) == (stress_uplift is None)
        and _canonical_report_reasons(report.validation_admission_reasons)
    )


def _mlp_report_test_evidence_is_valid(report: PolymarketMLPReport) -> bool:
    if not report.test_evaluated:
        return (
            report.test_log_loss is None
            and report.test_metrics is None
            and report.test_utility_uplift is None
            and not report.test_gate_reasons
        )
    test_log_loss = report.test_log_loss
    test_metrics = report.test_metrics
    test_utility_uplift = report.test_utility_uplift
    return (
        test_log_loss is not None
        and math.isfinite(test_log_loss)
        and test_log_loss >= 0.0
        and test_metrics is not None
        and test_metrics.threshold == report.selected_threshold
        and test_utility_uplift is not None
        and test_utility_uplift.sample_count == len(report.split.test_groups)
        and set(test_metrics.gate_reasons).issubset(report.test_gate_reasons)
    )


def _mlp_report_hash_is_valid(report: PolymarketMLPReport) -> bool:
    return _is_sha256(report.report_sha256) and report.report_sha256 == _sha256(
        report.identity_payload()
    )


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
    return consume_polymarket_fit_claim(
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
    backend = require_backend(resolve_backend(requested))
    device = (
        torch_device_for_backend(backend)
        if backend.kind == "directml"
        else torch.device(backend.device)
    )
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


@dataclass(frozen=True)
class _PolymarketMLPFitContext:
    torch: Any
    device: object
    model: Any
    optimizer: _ExplicitAdamW
    seed: int
    training_features: np.ndarray
    training_labels: np.ndarray
    training_weights: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    progress: ProgressCallback | None
    run_kind: str


@dataclass
class _PolymarketMLPFitState:
    best_loss: float
    best_epoch: int
    best_state: dict[str, Any] | None
    stale_epochs: int
    trace: list[PolymarketMLPEpoch]
    last_batch_heartbeat: float


def _train_member_batch(
    context: _PolymarketMLPFitContext,
    selected: np.ndarray,
) -> float:
    torch = context.torch
    features = torch.from_numpy(
        np.ascontiguousarray(context.training_features[selected], dtype=np.float32)
    ).to(context.device)
    labels = torch.from_numpy(
        np.ascontiguousarray(context.training_labels[selected], dtype=np.float32)
    ).to(context.device)
    weights = torch.from_numpy(
        np.ascontiguousarray(context.training_weights[selected], dtype=np.float32)
    ).to(context.device)
    context.optimizer.zero_grad(set_to_none=True)
    logits = context.model(features)
    losses = _binary_logit_losses(torch, logits, labels)
    loss = torch.mean(losses * weights)
    if not bool(torch.isfinite(loss).detach().cpu().item()):
        raise RuntimeError("Polymarket MLP training loss is non-finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        context.model.parameters(),
        1.0,
        foreach=False,
    )
    context.optimizer.step()
    return float(loss.detach().cpu())


def _emit_member_batch_heartbeat(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
    *,
    epoch: int,
    offset: int,
    rows: int,
    order_size: int,
    heartbeat: float,
) -> None:
    if context.progress is None or heartbeat - state.last_batch_heartbeat < 30.0:
        return
    context.progress(
        "polymarket_mlp_batch",
        {
            "run_kind": context.run_kind,
            "seed": context.seed,
            "epoch": epoch,
            "rows_complete": min(offset + rows, order_size),
            "rows_total": order_size,
        },
    )
    state.last_batch_heartbeat = heartbeat


def _train_member_epoch(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
    *,
    epoch: int,
) -> float:
    context.model.train()
    order = np.random.default_rng(context.seed + epoch).permutation(
        context.training_features.shape[0]
    )
    total_loss = 0.0
    total_rows = 0
    for offset in range(0, order.size, POLYMARKET_MLP_BATCH_SIZE):
        selected = order[offset : offset + POLYMARKET_MLP_BATCH_SIZE]
        loss = _train_member_batch(context, selected)
        rows = int(selected.size)
        total_loss += loss * rows
        total_rows += rows
        _emit_member_batch_heartbeat(
            context,
            state,
            epoch=epoch,
            offset=offset,
            rows=rows,
            order_size=int(order.size),
            heartbeat=time.perf_counter(),
        )
    return total_loss / total_rows


def _update_member_checkpoint(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
    *,
    epoch: int,
    validation_loss: float,
) -> None:
    if validation_loss < state.best_loss - POLYMARKET_MLP_MIN_DELTA:
        state.best_loss = validation_loss
        state.best_epoch = epoch
        state.best_state = {
            name: value.detach().cpu().clone()
            for name, value in context.model.state_dict().items()
        }
        state.stale_epochs = 0
    else:
        state.stale_epochs += 1


def _emit_member_epoch_progress(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
    *,
    epoch: int,
    validation_loss: float,
) -> None:
    if context.progress is None or not (
        epoch == 1 or epoch % 5 == 0 or state.stale_epochs >= POLYMARKET_MLP_PATIENCE
    ):
        return
    context.progress(
        "polymarket_mlp_epoch",
        {
            "run_kind": context.run_kind,
            "seed": context.seed,
            "epoch": epoch,
            "training_loss": state.trace[-1].training_loss,
            "validation_log_loss": validation_loss,
            "best_validation_log_loss": state.best_loss,
            "stale_epochs": state.stale_epochs,
        },
    )


def _evaluate_member_epoch(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
    *,
    epoch: int,
    training_loss: float,
) -> bool:
    validation_probability = _predict_torch(
        context.torch,
        context.model,
        context.device,
        context.validation_features,
    )
    validation_loss = _binary_log_loss(
        validation_probability,
        context.validation_labels,
    )
    state.trace.append(
        PolymarketMLPEpoch(
            seed=context.seed,
            epoch=epoch,
            training_loss=training_loss,
            validation_log_loss=validation_loss,
        )
    )
    _update_member_checkpoint(
        context,
        state,
        epoch=epoch,
        validation_loss=validation_loss,
    )
    _emit_member_epoch_progress(
        context,
        state,
        epoch=epoch,
        validation_loss=validation_loss,
    )
    return state.stale_epochs >= POLYMARKET_MLP_PATIENCE


def _finalize_member_fit(
    context: _PolymarketMLPFitContext,
    state: _PolymarketMLPFitState,
) -> tuple[PolymarketMLPMember, float]:
    if state.best_state is None or state.best_epoch <= 0:
        raise RuntimeError("Polymarket MLP produced no finite validation checkpoint")
    context.model.load_state_dict(state.best_state)
    torch_probability = _predict_torch(
        context.torch,
        context.model,
        context.device,
        context.validation_features,
    )
    member = _extract_member(
        context.model,
        seed=context.seed,
        best_epoch=state.best_epoch,
        trace=state.trace,
    )
    canonical_probability = member.predict_standardized(context.validation_features)
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
    context = _PolymarketMLPFitContext(
        torch=torch,
        device=device,
        model=model,
        optimizer=optimizer,
        seed=seed,
        training_features=training_features,
        training_labels=training_labels,
        training_weights=training_weights,
        validation_features=validation_features,
        validation_labels=validation_labels,
        progress=progress,
        run_kind=run_kind,
    )
    state = _PolymarketMLPFitState(
        best_loss=math.inf,
        best_epoch=0,
        best_state=None,
        stale_epochs=0,
        trace=[],
        last_batch_heartbeat=time.perf_counter(),
    )
    for epoch in range(1, POLYMARKET_MLP_MAX_EPOCHS + 1):
        training_loss = _train_member_epoch(context, state, epoch=epoch)
        if _evaluate_member_epoch(
            context,
            state,
            epoch=epoch,
            training_loss=training_loss,
        ):
            break
    return _finalize_member_fit(context, state)


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
    quantiles = np.asarray(
        np.quantile(np.asarray(means), [0.025, 0.975]),
        dtype=np.float64,
    ).reshape(-1)
    if quantiles.size != 2 or not np.all(np.isfinite(quantiles)):
        raise RuntimeError("Polymarket MLP bootstrap quantiles are invalid")
    return PolymarketMLPBootstrap(
        sample_count=count,
        block_length=block,
        resamples=POLYMARKET_MLP_BOOTSTRAP_SAMPLES,
        mean_delta=math.fsum(observations) / count,
        lower_95=float(quantiles[0]),
        upper_95=float(quantiles[1]),
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


@dataclass(frozen=True)
class _PolymarketMLPPreparedData:
    split: PolymarketRidgeSplit
    validation_indices: np.ndarray
    training_x: np.ndarray
    training_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    standardized_training: np.ndarray
    standardized_validation: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _PolymarketMLPValidationResult:
    validation_loss: float
    ridge_validation_loss: float
    validation_uplift: PolymarketMLPBootstrap
    evaluations: tuple[PolymarketPolicyEvaluation, ...]
    selected: PolymarketPolicyEvaluation | None
    utility_uplift: float | None
    admission_reasons: tuple[str, ...]
    admitted: bool


@dataclass(frozen=True)
class _PolymarketMLPTestResult:
    selected_policy: str
    selected_threshold: float | None
    test_loss: float | None
    test_metrics: PolymarketPolicyMetrics | None
    test_uplift: PolymarketMLPBootstrap | None
    test_reasons: tuple[str, ...]


def _prepare_mlp_fit_data(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
) -> _PolymarketMLPPreparedData:
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
    mean = np.asarray(np.mean(training_x, axis=0, dtype=np.float64), dtype=np.float64)
    scale = np.asarray(
        np.std(training_x, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized_training = (training_x - mean) / scale
    standardized_validation = (validation_x - mean) / scale
    weights = _condition_weights(dataset, train_indices)
    return _PolymarketMLPPreparedData(
        split=expected_split,
        validation_indices=validation_indices,
        training_x=training_x,
        training_y=training_y,
        validation_x=validation_x,
        validation_y=validation_y,
        mean=mean,
        scale=scale,
        standardized_training=standardized_training,
        standardized_validation=standardized_validation,
        weights=weights,
    )


def _emit_mlp_preflight_progress(
    progress: ProgressCallback | None,
    evidence: PolymarketMLPBackendEvidence,
    prepared: _PolymarketMLPPreparedData,
) -> None:
    if progress is None:
        return
    progress(
        "polymarket_mlp_preflight",
        {
            "backend": evidence.kind,
            "device": evidence.device,
            "fallback_reason": evidence.fallback_reason,
            "training_rows": int(prepared.training_x.shape[0]),
            "validation_rows": int(prepared.validation_x.shape[0]),
        },
    )


def _fit_mlp_seed_members(
    torch: Any,
    device: object,
    prepared: _PolymarketMLPPreparedData,
    progress: ProgressCallback | None,
) -> tuple[tuple[PolymarketMLPMember, ...], list[float]]:
    fitted_members: list[PolymarketMLPMember] = []
    canonical_replay_drifts: list[float] = []
    for seed in POLYMARKET_MLP_SEEDS:
        if progress is not None:
            progress("polymarket_mlp_seed", {"seed": seed, "status": "started"})
        member, canonical_replay_drift = _fit_member(
            torch,
            device,
            seed=seed,
            training_features=prepared.standardized_training,
            training_labels=prepared.training_y,
            training_weights=prepared.weights,
            validation_features=prepared.standardized_validation,
            validation_labels=prepared.validation_y,
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
                    "canonical_replay_max_probability_drift": (canonical_replay_drift),
                },
            )
    return tuple(fitted_members), canonical_replay_drifts


def _fit_mlp_reproducibility_member(
    torch: Any,
    device: object,
    prepared: _PolymarketMLPPreparedData,
    progress: ProgressCallback | None,
) -> tuple[PolymarketMLPMember, float]:
    if progress is not None:
        progress(
            "polymarket_mlp_reproducibility",
            {"seed": POLYMARKET_MLP_SEEDS[0], "status": "started"},
        )
    return _fit_member(
        torch,
        device,
        seed=POLYMARKET_MLP_SEEDS[0],
        training_features=prepared.standardized_training,
        training_labels=prepared.training_y,
        training_weights=prepared.weights,
        validation_features=prepared.standardized_validation,
        validation_labels=prepared.validation_y,
        progress=progress,
        run_kind="reproducibility",
    )


def _reproducibility_drift(
    members: Sequence[PolymarketMLPMember],
    repeated: PolymarketMLPMember,
    validation_features: np.ndarray,
) -> float:
    first_probability = members[0].predict_standardized(validation_features)
    repeated_probability = repeated.predict_standardized(validation_features)
    drift = float(np.max(np.abs(first_probability - repeated_probability)))
    if drift > POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE:
        raise ValueError(
            f"Polymarket MLP same-seed probability drift exceeds tolerance:{drift:.9g}"
        )
    return drift


def _emit_mlp_reproducibility_progress(
    progress: ProgressCallback | None,
    drift: float,
) -> None:
    if progress is None:
        return
    progress(
        "polymarket_mlp_reproducibility",
        {
            "seed": POLYMARKET_MLP_SEEDS[0],
            "status": "complete",
            "maximum_probability_drift": drift,
        },
    )


def _fit_mlp_ensemble(
    dataset: PolymarketRidgeDataset,
    prepared: _PolymarketMLPPreparedData,
    *,
    compute_backend: str,
    progress: ProgressCallback | None,
) -> PolymarketMLPEnsemble:
    torch, device, backend = _torch_runtime(compute_backend)
    training_started = time.perf_counter()
    backend_evidence = _preflight(torch, device, backend)
    _emit_mlp_preflight_progress(progress, backend_evidence, prepared)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        members, canonical_replay_drifts = _fit_mlp_seed_members(
            torch,
            device,
            prepared,
            progress,
        )
        repeated, repeated_replay_drift = _fit_mlp_reproducibility_member(
            torch,
            device,
            prepared,
            progress,
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
    reproducibility_drift = _reproducibility_drift(
        members,
        repeated,
        prepared.standardized_validation,
    )
    _emit_mlp_reproducibility_progress(progress, reproducibility_drift)
    provisional_ensemble = PolymarketMLPEnsemble(
        dataset_sha256=dataset.dataset_sha256,
        feature_mean=tuple(float(value) for value in prepared.mean),
        feature_scale=tuple(float(value) for value in prepared.scale),
        members=members,
        backend=backend_evidence,
        reproducibility_max_probability_drift=reproducibility_drift,
        ensemble_sha256="",
    )
    return replace(
        provisional_ensemble,
        ensemble_sha256=_sha256(provisional_ensemble.identity_payload()),
    ).validated()


def _require_parent_validation_replay(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    prepared: _PolymarketMLPPreparedData,
    ridge_validation_probability: np.ndarray,
) -> PolymarketPolicyEvaluation:
    ridge_validation_evaluation = evaluate_polymarket_policy(
        dataset,
        prepared.validation_indices,
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
    return ridge_validation_evaluation


def _select_mlp_validation(
    evaluations: Sequence[PolymarketPolicyEvaluation],
) -> PolymarketPolicyEvaluation | None:
    passed = [item for item in evaluations if item.metrics.gate_passed]
    if not passed:
        return None
    return max(
        passed,
        key=lambda item: (
            item.metrics.wilson_lower_bound_95,
            float(item.metrics.threshold or 0.0),
        ),
    )


def _mlp_admission_reasons(
    selected: PolymarketPolicyEvaluation | None,
    validation_uplift: PolymarketMLPBootstrap,
    utility_uplift: float | None,
) -> tuple[str, ...]:
    admission_reasons: list[str] = []
    if selected is None:
        admission_reasons.append("no_validation_threshold_passed")
    if validation_uplift.lower_95 <= 0.0:
        admission_reasons.append("validation_log_loss_uplift_lower_not_positive")
    if utility_uplift is None or utility_uplift <= 0.0:
        admission_reasons.append("validation_stress_utility_not_above_ridge")
    return tuple(sorted(set(admission_reasons)))


def _emit_mlp_validation_progress(
    progress: ProgressCallback | None,
    result: _PolymarketMLPValidationResult,
    split: PolymarketRidgeSplit,
) -> None:
    if progress is None:
        return
    progress(
        "polymarket_mlp_validation",
        {
            "admitted_to_test": result.admitted,
            "admission_reasons": list(result.admission_reasons),
            "validation_log_loss": result.validation_loss,
            "ridge_validation_log_loss": result.ridge_validation_loss,
            "log_loss_uplift_lower_95": result.validation_uplift.lower_95,
            "stress_utility_uplift_quote": result.utility_uplift,
            "untouched_test_group_count": len(split.test_groups),
        },
    )


def _evaluate_mlp_validation(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    prepared: _PolymarketMLPPreparedData,
    ensemble: PolymarketMLPEnsemble,
    progress: ProgressCallback | None,
) -> _PolymarketMLPValidationResult:
    validation_probability = ensemble.predict(prepared.validation_x)
    validation_loss = _binary_log_loss(
        validation_probability,
        prepared.validation_y,
    )
    ridge_validation_probability = parent.selected_model.predict(prepared.validation_x)
    ridge_validation_loss = _binary_log_loss(
        ridge_validation_probability,
        prepared.validation_y,
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
            prepared.validation_indices,
            ridge_validation_probability,
            validation_probability,
            prepared.split.validation_groups,
        ),
        seed_material=dataset.dataset_sha256 + "validation-log-loss",
    )
    validation_evaluations = tuple(
        evaluate_polymarket_policy(
            dataset,
            prepared.validation_indices,
            validation_probability,
            threshold,
            require_asset_profit=False,
        )
        for threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
    )
    selected_validation = _select_mlp_validation(validation_evaluations)
    ridge_validation_evaluation = _require_parent_validation_replay(
        dataset,
        parent,
        prepared,
        ridge_validation_probability,
    )
    validation_utility_uplift = (
        None
        if selected_validation is None
        else (
            selected_validation.metrics.aggregate_stress_utility_quote
            - ridge_validation_evaluation.metrics.aggregate_stress_utility_quote
        )
    )
    frozen_admission_reasons = _mlp_admission_reasons(
        selected_validation,
        validation_uplift,
        validation_utility_uplift,
    )
    result = _PolymarketMLPValidationResult(
        validation_loss=validation_loss,
        ridge_validation_loss=ridge_validation_loss,
        validation_uplift=validation_uplift,
        evaluations=validation_evaluations,
        selected=selected_validation,
        utility_uplift=validation_utility_uplift,
        admission_reasons=frozen_admission_reasons,
        admitted=not frozen_admission_reasons,
    )
    _emit_mlp_validation_progress(progress, result, prepared.split)
    return result


def _mlp_test_reasons(
    parent: PolymarketRidgeReport,
    test_loss: float,
    test_metrics: PolymarketPolicyMetrics,
    test_uplift: PolymarketMLPBootstrap,
) -> tuple[str, ...]:
    test_reasons: list[str] = []
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
        if test_metrics.pnl_by_asset[asset] < parent.test_metrics.pnl_by_asset[asset]:
            test_reasons.append(f"test_asset_utility_below_ridge:{asset}")
    if test_uplift.lower_95 <= 0.0:
        test_reasons.append("test_utility_bootstrap_lower_not_positive")
    return tuple(test_reasons)


def _emit_mlp_test_progress(
    progress: ProgressCallback | None,
    *,
    status: str,
    test_loss: float | None = None,
    test_metrics: PolymarketPolicyMetrics | None = None,
    reason_count: int = 0,
) -> None:
    if progress is None:
        return
    payload: dict[str, object] = {"status": status}
    if status == "complete" and test_metrics is not None:
        payload.update(
            {
                "test_log_loss": test_loss,
                "test_stress_utility_quote": (
                    test_metrics.aggregate_stress_utility_quote
                ),
                "gate_reason_count": reason_count,
            }
        )
    progress("polymarket_mlp_test", payload)


def _evaluate_admitted_mlp_test(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    prepared: _PolymarketMLPPreparedData,
    ensemble: PolymarketMLPEnsemble,
    selected_threshold: float | None,
    progress: ProgressCallback | None,
) -> _PolymarketMLPTestResult:
    _validate_partition_label_breadth(
        dataset,
        name="test",
        groups=prepared.split.test_groups,
    )
    test_indices = _partition_indices(dataset, prepared.split.test_groups)
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
        prepared.split.test_groups,
    )
    challenger_utility = _condition_utility(
        dataset,
        test_evaluation,
        prepared.split.test_groups,
    )
    test_uplift = _bootstrap(
        _group_utility_uplift(
            dataset,
            prepared.split.test_groups,
            baseline_utility,
            challenger_utility,
        ),
        seed_material=dataset.dataset_sha256 + "test-utility",
    )
    test_metrics = test_evaluation.metrics
    test_reasons = _mlp_test_reasons(
        parent,
        test_loss,
        test_metrics,
        test_uplift,
    )
    _emit_mlp_test_progress(
        progress,
        status="complete",
        test_loss=test_loss,
        test_metrics=test_metrics,
        reason_count=len(test_reasons),
    )
    return _PolymarketMLPTestResult(
        selected_policy="causal_mlp",
        selected_threshold=selected_threshold,
        test_loss=test_loss,
        test_metrics=test_metrics,
        test_uplift=test_uplift,
        test_reasons=test_reasons,
    )


def _evaluate_mlp_test_if_admitted(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    prepared: _PolymarketMLPPreparedData,
    ensemble: PolymarketMLPEnsemble,
    validation: _PolymarketMLPValidationResult,
    progress: ProgressCallback | None,
) -> _PolymarketMLPTestResult:
    if not validation.admitted:
        return _PolymarketMLPTestResult(
            selected_policy="no_trade",
            selected_threshold=None,
            test_loss=None,
            test_metrics=None,
            test_uplift=None,
            test_reasons=(),
        )
    _emit_mlp_test_progress(progress, status="started")
    if validation.selected is None:
        raise RuntimeError("Polymarket MLP admission state is inconsistent")
    return _evaluate_admitted_mlp_test(
        dataset,
        parent,
        prepared,
        ensemble,
        validation.selected.metrics.threshold,
        progress,
    )


def _build_mlp_report(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    prepared: _PolymarketMLPPreparedData,
    ensemble: PolymarketMLPEnsemble,
    validation: _PolymarketMLPValidationResult,
    test: _PolymarketMLPTestResult,
) -> PolymarketMLPReport:
    provisional = PolymarketMLPReport(
        dataset_sha256=dataset.dataset_sha256,
        parent_ridge_report_sha256=parent.report_sha256,
        split=prepared.split,
        ensemble=ensemble,
        validation_log_loss=validation.validation_loss,
        ridge_validation_log_loss=validation.ridge_validation_loss,
        validation_log_loss_uplift=validation.validation_uplift,
        validation_trials=tuple(item.metrics for item in validation.evaluations),
        validation_stress_utility_uplift_quote=validation.utility_uplift,
        validation_admission_reasons=validation.admission_reasons,
        selected_policy=test.selected_policy,
        selected_threshold=test.selected_threshold,
        test_evaluated=validation.admitted,
        test_log_loss=test.test_loss,
        test_metrics=test.test_metrics,
        test_utility_uplift=test.test_uplift,
        test_gate_reasons=tuple(sorted(set(test.test_reasons))),
        development_passed=validation.admitted and not test.test_reasons,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def fit_and_evaluate_polymarket_mlp(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    *,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> PolymarketMLPReport:
    """Fit the preregistered nonlinear challenger and open test only if admitted."""

    prepared = _prepare_mlp_fit_data(dataset, parent)
    ensemble = _fit_mlp_ensemble(
        dataset,
        prepared,
        compute_backend=compute_backend,
        progress=progress,
    )
    validation = _evaluate_mlp_validation(
        dataset,
        parent,
        prepared,
        ensemble,
        progress,
    )
    test = _evaluate_mlp_test_if_admitted(
        dataset,
        parent,
        prepared,
        ensemble,
        validation,
        progress,
    )
    return _build_mlp_report(
        dataset,
        parent,
        prepared,
        ensemble,
        validation,
        test,
    )


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


_MLP_MATERIALIZATION_SQL: Mapping[str, tuple[str, str]] = {
    "polymarket_mlp_member": (
        "SELECT * FROM polymarket_mlp_member WHERE report_sha256 = ? ORDER BY seed",
        "INSERT INTO polymarket_mlp_member VALUES (?, ?, ?, ?, ?, ?)",
    ),
    "polymarket_mlp_epoch": (
        "SELECT * FROM polymarket_mlp_epoch "
        "WHERE report_sha256 = ? ORDER BY seed, epoch",
        "INSERT INTO polymarket_mlp_epoch VALUES (?, ?, ?, ?, ?)",
    ),
    "polymarket_mlp_prediction": (
        "SELECT * FROM polymarket_mlp_prediction "
        "WHERE report_sha256 = ? ORDER BY partition, sequence",
        "INSERT INTO polymarket_mlp_prediction "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ),
    "polymarket_mlp_selected_action": (
        "SELECT * FROM polymarket_mlp_selected_action "
        "WHERE report_sha256 = ? ORDER BY partition, sequence",
        "INSERT INTO polymarket_mlp_selected_action "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    ),
    "polymarket_mlp_equity": (
        "SELECT * FROM polymarket_mlp_equity "
        "WHERE report_sha256 = ? ORDER BY partition, sequence",
        "INSERT INTO polymarket_mlp_equity VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    ),
    "polymarket_mlp_market_pnl": (
        "SELECT * FROM polymarket_mlp_market_pnl "
        "WHERE report_sha256 = ? ORDER BY partition, condition_id",
        "INSERT INTO polymarket_mlp_market_pnl VALUES (?, ?, ?, ?, ?, ?, ?)",
    ),
}


def _selected_validation_metrics(
    report: PolymarketMLPReport,
) -> PolymarketPolicyMetrics | None:
    threshold = report.selected_threshold
    if threshold is None:
        return None
    matches = tuple(
        item for item in report.validation_trials if item.threshold == threshold
    )
    if len(matches) != 1:
        raise ValueError(
            "Polymarket MLP selected validation threshold is missing or duplicated"
        )
    return matches[0]


_MLP_MATERIALIZATION_SCHEMA_SQL = """
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

_MLPRow = tuple[object, ...]
_MLPSortKey = Callable[[_MLPRow], Any]
_MLPTableRows = tuple[str, list[_MLPRow], _MLPSortKey]


@dataclass(frozen=True)
class _MLPPartitionRows:
    predictions: list[_MLPRow]
    actions: list[_MLPRow]
    equity: list[_MLPRow]
    markets: list[_MLPRow]


@dataclass(frozen=True)
class _MLPPersistenceRows:
    report: _MLPRow
    runtime: _MLPRow
    runtime_json: str
    tables: tuple[_MLPTableRows, ...]


def _validated_mlp_materialization_identity(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
) -> None:
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


def _replay_mlp_partition(
    dataset: PolymarketRidgeDataset,
    report: PolymarketMLPReport,
    *,
    partition: str,
    groups: Sequence[int],
    require_asset_profit: bool,
) -> tuple[PolymarketPolicyEvaluation, _MLPPartitionRows]:
    indices = _partition_indices(dataset, groups)
    features, _labels = _matrix(dataset, indices)
    probabilities = report.ensemble.predict(features)
    evaluation, actions, equity, markets = polymarket_selected_policy_tables(
        dataset,
        report_sha256=report.report_sha256,
        partition=partition,
        indices=indices,
        probabilities=probabilities,
        threshold=report.selected_threshold,
        require_asset_profit=require_asset_profit,
    )
    return evaluation, _MLPPartitionRows(
        predictions=_prediction_rows(
            dataset,
            report_sha256=report.report_sha256,
            partition=partition,
            indices=indices,
            probabilities=probabilities,
        ),
        actions=actions,
        equity=equity,
        markets=markets,
    )


def _replay_mlp_validation(
    dataset: PolymarketRidgeDataset,
    report: PolymarketMLPReport,
) -> _MLPPartitionRows:
    evaluation, rows = _replay_mlp_partition(
        dataset,
        report,
        partition="validation",
        groups=report.split.validation_groups,
        require_asset_profit=False,
    )
    expected = _selected_validation_metrics(report)
    if expected is not None and evaluation.metrics.asdict() != expected.asdict():
        raise ValueError("Polymarket MLP validation replay differs from report")
    return rows


def _replay_mlp_test(
    dataset: PolymarketRidgeDataset,
    report: PolymarketMLPReport,
) -> _MLPPartitionRows:
    if not report.test_evaluated:
        return _MLPPartitionRows([], [], [], [])
    evaluation, rows = _replay_mlp_partition(
        dataset,
        report,
        partition="test",
        groups=report.split.test_groups,
        require_asset_profit=True,
    )
    if (
        report.test_metrics is None
        or evaluation.metrics.asdict() != report.test_metrics.asdict()
    ):
        raise ValueError("Polymarket MLP test replay differs from report")
    return rows


def _mlp_member_rows(report: PolymarketMLPReport) -> list[_MLPRow]:
    return [
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


def _mlp_trace_rows(report: PolymarketMLPReport) -> list[_MLPRow]:
    return [
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


def _mlp_report_row(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
) -> _MLPRow:
    return (
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


def _mlp_persistence_rows(
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
    validation: _MLPPartitionRows,
    test: _MLPPartitionRows,
) -> _MLPPersistenceRows:
    runtime_json = _canonical_json(report.ensemble.backend.asdict())
    runtime_row = (
        report.report_sha256,
        _sha256(report.ensemble.backend.asdict()),
        runtime_json,
    )
    return _MLPPersistenceRows(
        report=_mlp_report_row(dataset, parent, report),
        runtime=runtime_row,
        runtime_json=runtime_json,
        tables=(
            ("polymarket_mlp_member", _mlp_member_rows(report), lambda row: row[1]),
            (
                "polymarket_mlp_epoch",
                _mlp_trace_rows(report),
                lambda row: (row[1], row[2]),
            ),
            (
                "polymarket_mlp_prediction",
                validation.predictions + test.predictions,
                lambda row: (row[1], row[2]),
            ),
            (
                "polymarket_mlp_selected_action",
                validation.actions + test.actions,
                lambda row: (row[1], row[2]),
            ),
            (
                "polymarket_mlp_equity",
                validation.equity + test.equity,
                lambda row: (row[1], row[2]),
            ),
            (
                "polymarket_mlp_market_pnl",
                validation.markets + test.markets,
                lambda row: (row[1], row[2]),
            ),
        ),
    )


def _ensure_mlp_materialization_schema(connection: Any) -> None:
    connection.execute(_MLP_MATERIALIZATION_SCHEMA_SQL)


def _validate_existing_mlp_materialization(
    connection: Any,
    report_sha256: str,
    rows: _MLPPersistenceRows,
) -> bool:
    existing = connection.execute(
        "SELECT * FROM polymarket_mlp_report WHERE report_sha256 = ?",
        [report_sha256],
    ).fetchone()
    if existing is None:
        return False
    if tuple(existing) != rows.report:
        raise ValueError("stored Polymarket MLP report is inconsistent")
    for table, expected, sort_key in rows.tables:
        select_sql, _insert_sql = _MLP_MATERIALIZATION_SQL[table]
        stored = connection.execute(select_sql, [report_sha256]).fetchall()
        if [tuple(row) for row in stored] != sorted(expected, key=sort_key):
            raise ValueError(f"stored {table} rows are inconsistent")
    stored_runtime = connection.execute(
        """
        SELECT backend_json FROM polymarket_mlp_runtime_evidence
        WHERE report_sha256 = ? AND runtime_sha256 = ?
        """,
        rows.runtime[:2],
    ).fetchone()
    if stored_runtime is None:
        connection.execute(
            "INSERT INTO polymarket_mlp_runtime_evidence VALUES (?, ?, ?)",
            rows.runtime,
        )
    elif str(stored_runtime[0]) != rows.runtime_json:
        raise ValueError("stored Polymarket MLP runtime evidence is inconsistent")
    return True


def _insert_mlp_materialization(connection: Any, rows: _MLPPersistenceRows) -> None:
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "INSERT INTO polymarket_mlp_report VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows.report,
        )
        connection.execute(
            "INSERT INTO polymarket_mlp_runtime_evidence VALUES (?, ?, ?)",
            rows.runtime,
        )
        for table, table_rows, _sort_key in rows.tables:
            if table_rows:
                _select_sql, insert_sql = _MLP_MATERIALIZATION_SQL[table]
                connection.executemany(insert_sql, table_rows)
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def materialize_polymarket_mlp_report(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    parent: PolymarketRidgeReport,
    report: PolymarketMLPReport,
) -> PolymarketMLPMaterialization:
    """Persist the nonlinear model and every reconstructable result atomically."""

    _validated_mlp_materialization_identity(dataset, parent, report)
    validation = _replay_mlp_validation(dataset, report)
    test = _replay_mlp_test(dataset, report)
    rows = _mlp_persistence_rows(dataset, parent, report, validation, test)
    connection = store.connect()
    _ensure_mlp_materialization_schema(connection)
    if _validate_existing_mlp_materialization(
        connection,
        report.report_sha256,
        rows,
    ):
        status = "existing"
    else:
        _insert_mlp_materialization(connection, rows)
        status = "created"
    return PolymarketMLPMaterialization(
        report_sha256=report.report_sha256,
        status=status,
        validation_prediction_count=len(validation.predictions),
        test_prediction_count=len(test.predictions),
        selected_validation_action_count=len(validation.actions),
        selected_test_action_count=len(test.actions),
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

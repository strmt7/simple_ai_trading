"""Frozen Round 9 ridge-logit selection and sequential policy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket_action_value import (
    POLYMARKET_ACTION_FEATURE_NAMES,
    POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
    PolymarketActionFeature,
    PolymarketActionLabel,
    PolymarketActionValueConfig,
    PolymarketActionValueDataset,
)
from .polymarket_action_pipeline import (
    POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION,
    polymarket_action_pipeline_implementation_sha256,
)
from .polymarket_fit_claim import (
    POLYMARKET_FIT_CLAIM_SCHEMA_VERSION,
    PolymarketFitClaim,
    complete_polymarket_fit_claim,
    consume_polymarket_fit_claim,
    discard_polymarket_fit_reservation,
    fail_polymarket_fit_claim,
    reserve_polymarket_fit_claim,
)
from .polymarket_model_contracts import (
    POLYMARKET_ROUND9_MLP_CONTRACT_SHA256,
    POLYMARKET_ROUND9_RIDGE_CONTRACT_SHA256,
)
from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_RIDGE_CONTRACT_SHA256 = POLYMARKET_ROUND9_RIDGE_CONTRACT_SHA256
POLYMARKET_RIDGE_MODEL_SCHEMA_VERSION = "polymarket-round9-ridge-model-v1"
POLYMARKET_RIDGE_REPORT_SCHEMA_VERSION = "polymarket-round9-ridge-report-v1"
POLYMARKET_RIDGE_FIT_CLAIM_SCHEMA_VERSION = POLYMARKET_FIT_CLAIM_SCHEMA_VERSION
POLYMARKET_RIDGE_L2_GRID = (0.01, 0.1, 1.0)
POLYMARKET_RIDGE_THRESHOLD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_TRAINING_BLOCKING_ENTRY_TERMINALS = frozenset(
    {
        "entry_confirmation_enters_excluded_close_window",
        "missing_entry_execution_book",
    }
)
_FEATURE_COUNT = len(POLYMARKET_ACTION_FEATURE_NAMES)
PolymarketRidgeFitClaim = PolymarketFitClaim


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


def _ridge_dataset_sha256(
    *,
    pipeline_report_sha256: str,
    eligibility_sha256: str,
    action_feature_sha256: Iterable[str],
    action_label_sha256: Iterable[str],
    source_feature_row_sha256: Iterable[str],
) -> str:
    """Hash the existing canonical dataset object without one large JSON copy."""

    payload: dict[str, str | Iterable[str]] = {
        "schema_version": "polymarket-round9-ridge-dataset-v1",
        "contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
        "pipeline_report_sha256": pipeline_report_sha256,
        "eligibility_sha256": eligibility_sha256,
        "action_feature_sha256": action_feature_sha256,
        "action_label_sha256": action_label_sha256,
        "source_feature_row_sha256": source_feature_row_sha256,
    }
    digest = hashlib.sha256()
    digest.update(b"{")
    for field_index, name in enumerate(sorted(payload)):
        if field_index:
            digest.update(b",")
        digest.update(_canonical_json(name).encode("ascii"))
        digest.update(b":")
        value = payload[name]
        if isinstance(value, str):
            digest.update(_canonical_json(value).encode("ascii"))
            continue
        digest.update(b"[")
        for value_index, item in enumerate(value):
            if value_index:
                digest.update(b",")
            digest.update(_canonical_json(item).encode("ascii"))
        digest.update(b"]")
    digest.update(b"}")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _float_text(values: Sequence[float]) -> list[str]:
    return [format(float(value), ".17g") for value in values]


def _binary_log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    labels = np.asarray(target, dtype=np.float64)
    if clipped.shape != labels.shape or clipped.size == 0:
        raise ValueError("binary log loss requires aligned nonempty vectors")
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )


@dataclass(frozen=True)
class PolymarketRidgeObservation:
    action_feature_sha256: str
    action_label_sha256: str
    source_feature_row_sha256: str
    condition_id: str
    asset: str
    outcome: str
    event_start_ms: int
    decision_received_monotonic_ns: int
    release_monotonic_ns: int
    feature_values: tuple[float, ...]
    official_up: bool | None
    classifier_eligible: bool
    positive_complete: bool
    category: str
    condition_blocked: bool
    stress_utility_quote: float

    def validated(self) -> PolymarketRidgeObservation:
        if (
            not _is_sha256(self.action_feature_sha256)
            or not _is_sha256(self.action_label_sha256)
            or not _is_sha256(self.source_feature_row_sha256)
            or not self.condition_id
            or self.asset not in _ASSETS
            or self.outcome not in {"Up", "Down"}
            or self.event_start_ms <= 0
            or self.decision_received_monotonic_ns < 0
            or self.release_monotonic_ns < self.decision_received_monotonic_ns
            or len(self.feature_values) != _FEATURE_COUNT
            or not all(math.isfinite(value) for value in self.feature_values)
            or not (self.official_up is None or isinstance(self.official_up, bool))
            or self.category
            not in {
                "action_unavailable",
                "entry_no_fill",
                "filled_entry_failed_exit",
                "successful_round_trip",
            }
            or not math.isfinite(self.stress_utility_quote)
            or (self.classifier_eligible == (self.category == "action_unavailable"))
            or (self.positive_complete and self.category != "successful_round_trip")
            or (self.condition_blocked != (self.category == "filled_entry_failed_exit"))
        ):
            raise ValueError("Polymarket ridge observation is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRidgeDataset:
    pipeline_report_sha256: str
    eligibility_sha256: str
    observations: tuple[PolymarketRidgeObservation, ...]
    dataset_sha256: str

    @property
    def group_starts_ms(self) -> tuple[int, ...]:
        return tuple(sorted({item.event_start_ms for item in self.observations}))

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round9-ridge-dataset-v1",
            "contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
            "pipeline_report_sha256": self.pipeline_report_sha256,
            "eligibility_sha256": self.eligibility_sha256,
            "action_feature_sha256": [
                item.action_feature_sha256 for item in self.observations
            ],
            "action_label_sha256": [
                item.action_label_sha256 for item in self.observations
            ],
            "source_feature_row_sha256": [
                item.source_feature_row_sha256 for item in self.observations
            ],
        }

    def validated(self) -> PolymarketRidgeDataset:
        for observation in self.observations:
            observation.validated()
        if (
            not _is_sha256(self.pipeline_report_sha256)
            or not _is_sha256(self.eligibility_sha256)
            or not self.observations
            or len({item.action_feature_sha256 for item in self.observations})
            != len(self.observations)
            or not _is_sha256(self.dataset_sha256)
            or self.dataset_sha256
            != _ridge_dataset_sha256(
                pipeline_report_sha256=self.pipeline_report_sha256,
                eligibility_sha256=self.eligibility_sha256,
                action_feature_sha256=(
                    item.action_feature_sha256 for item in self.observations
                ),
                action_label_sha256=(
                    item.action_label_sha256 for item in self.observations
                ),
                source_feature_row_sha256=(
                    item.source_feature_row_sha256 for item in self.observations
                ),
            )
        ):
            raise ValueError("Polymarket ridge dataset is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRidgeDatasetIdentity:
    """Opaque dataset identity that is safe to inspect before test access."""

    pipeline_report_sha256: str
    eligibility_sha256: str
    observation_count: int
    dataset_sha256: str

    def validated(self) -> PolymarketRidgeDatasetIdentity:
        if (
            not _is_sha256(self.pipeline_report_sha256)
            or not _is_sha256(self.eligibility_sha256)
            or isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count <= 0
            or not _is_sha256(self.dataset_sha256)
        ):
            raise ValueError("Polymarket ridge dataset identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRidgeSplit:
    train_groups: tuple[int, ...]
    validation_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    purged_groups: tuple[int, ...]

    def asdict(self) -> dict[str, object]:
        return {key: list(value) for key, value in asdict(self).items()}

    def validated(self) -> PolymarketRidgeSplit:
        partitions = (
            self.train_groups,
            self.validation_groups,
            self.test_groups,
            self.purged_groups,
        )
        values = tuple(value for partition in partitions for value in partition)
        total_groups = len(values)
        validation_count = math.floor(0.2 * total_groups)
        test_count = math.floor(0.2 * total_groups)
        train_count = total_groups - validation_count - test_count - 2
        chronological = (
            *self.train_groups,
            self.purged_groups[0] if len(self.purged_groups) == 2 else -1,
            *self.validation_groups,
            self.purged_groups[1] if len(self.purged_groups) == 2 else -1,
            *self.test_groups,
        )
        if (
            total_groups < 30
            or len(self.purged_groups) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            )
            or len(set(values)) != total_groups
            or chronological != tuple(sorted(chronological))
            or len(self.train_groups) != train_count
            or len(self.validation_groups) != validation_count
            or len(self.test_groups) != test_count
            or min(train_count, validation_count, test_count) <= 0
        ):
            raise ValueError("Polymarket ridge split is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRidgeModel:
    l2: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    optimizer_iterations: int
    optimizer_objective: float
    model_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_RIDGE_MODEL_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
            "feature_names": list(POLYMARKET_ACTION_FEATURE_NAMES),
            "l2": format(self.l2, ".17g"),
            "feature_mean": _float_text(self.feature_mean),
            "feature_scale": _float_text(self.feature_scale),
            "coefficients": _float_text(self.coefficients),
            "intercept": format(self.intercept, ".17g"),
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_objective": format(self.optimizer_objective, ".17g"),
        }

    def validated(self) -> PolymarketRidgeModel:
        vectors = (self.feature_mean, self.feature_scale, self.coefficients)
        if (
            self.l2 not in POLYMARKET_RIDGE_L2_GRID
            or any(len(value) != _FEATURE_COUNT for value in vectors)
            or not all(math.isfinite(item) for value in vectors for item in value)
            or not all(item > 0.0 for item in self.feature_scale)
            or not math.isfinite(self.intercept)
            or self.optimizer_iterations < 0
            or not math.isfinite(self.optimizer_objective)
            or not _is_sha256(self.model_sha256)
            or self.model_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket ridge model is invalid")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.validated()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != _FEATURE_COUNT:
            raise ValueError("Polymarket ridge prediction matrix is invalid")
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_scale, dtype=np.float64)
        logits = standardized @ np.asarray(self.coefficients) + self.intercept
        probabilities = expit(logits)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Polymarket ridge probabilities are non-finite")
        return np.asarray(probabilities, dtype=np.float64)


@dataclass(frozen=True)
class PolymarketRidgeCandidate:
    l2: float
    validation_log_loss: float
    model_sha256: str
    optimizer_iterations: int
    optimizer_objective: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)

    def validated(self) -> PolymarketRidgeCandidate:
        if (
            self.l2 not in POLYMARKET_RIDGE_L2_GRID
            or not math.isfinite(self.validation_log_loss)
            or self.validation_log_loss < 0.0
            or not _is_sha256(self.model_sha256)
            or isinstance(self.optimizer_iterations, bool)
            or not isinstance(self.optimizer_iterations, int)
            or self.optimizer_iterations < 0
            or not math.isfinite(self.optimizer_objective)
            or self.optimizer_objective < 0.0
        ):
            raise ValueError("Polymarket ridge candidate is invalid")
        return self


@dataclass(frozen=True)
class PolymarketPolicyMetrics:
    threshold: float | None
    attempt_count: int
    completed_trade_count: int
    completed_by_asset: dict[str, int]
    positive_complete_count: int
    failed_exit_count: int
    aggregate_stress_utility_quote: float
    pnl_by_asset: dict[str, float]
    median_market_pnl_quote: float
    maximum_realized_drawdown_quote: float
    positive_complete_precision: float
    wilson_lower_bound_95: float
    selected_action_sha256: str
    gate_passed: bool
    gate_reasons: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["completed_by_asset"] = dict(sorted(self.completed_by_asset.items()))
        payload["pnl_by_asset"] = dict(sorted(self.pnl_by_asset.items()))
        payload["gate_reasons"] = list(self.gate_reasons)
        return payload

    def validated(self, *, require_asset_profit: bool) -> PolymarketPolicyMetrics:
        count_values = (
            self.attempt_count,
            self.completed_trade_count,
            self.positive_complete_count,
            self.failed_exit_count,
            *self.completed_by_asset.values(),
        )
        expected_precision = (
            self.positive_complete_count / self.attempt_count
            if self.attempt_count
            else 0.0
        )
        reasons: list[str] = []
        if self.completed_trade_count < 30:
            reasons.append(f"completed_trade_count:{self.completed_trade_count}/30")
        for asset in _ASSETS:
            completed = self.completed_by_asset.get(asset, -1)
            if completed < 5:
                reasons.append(f"completed_trade_count:{asset}:{completed}/5")
        if self.aggregate_stress_utility_quote <= 0.0:
            reasons.append("aggregate_stress_utility_not_positive")
        if self.median_market_pnl_quote <= 0.0:
            reasons.append("median_market_pnl_not_positive")
        if self.failed_exit_count:
            reasons.append(f"filled_entry_failed_exit_count:{self.failed_exit_count}")
        if require_asset_profit:
            for asset in _ASSETS:
                if self.pnl_by_asset.get(asset, 0.0) <= 0.0:
                    reasons.append(f"asset_pnl_not_positive:{asset}")
        expected_reasons = tuple(sorted(reasons))
        if (
            self.threshold is not None
            and self.threshold not in POLYMARKET_RIDGE_THRESHOLD_GRID
            or set(self.completed_by_asset) != set(_ASSETS)
            or set(self.pnl_by_asset) != set(_ASSETS)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in count_values
            )
            or sum(self.completed_by_asset.values()) != self.completed_trade_count
            or self.positive_complete_count > self.completed_trade_count
            or self.completed_trade_count + self.failed_exit_count > self.attempt_count
            or not all(
                math.isfinite(value)
                for value in (
                    self.aggregate_stress_utility_quote,
                    *self.pnl_by_asset.values(),
                    self.median_market_pnl_quote,
                    self.maximum_realized_drawdown_quote,
                    self.positive_complete_precision,
                    self.wilson_lower_bound_95,
                )
            )
            or self.maximum_realized_drawdown_quote < 0.0
            or not math.isclose(
                self.aggregate_stress_utility_quote,
                math.fsum(self.pnl_by_asset.values()),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.positive_complete_precision,
                expected_precision,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                self.wilson_lower_bound_95,
                _wilson_lower_bound(
                    self.positive_complete_count,
                    self.attempt_count,
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not _is_sha256(self.selected_action_sha256)
            or not isinstance(self.gate_passed, bool)
            or self.gate_reasons != expected_reasons
            or self.gate_passed != (not expected_reasons)
        ):
            raise ValueError("Polymarket policy metrics are invalid")
        return self


@dataclass(frozen=True)
class PolymarketPolicyEvaluation:
    metrics: PolymarketPolicyMetrics
    selected_indices: tuple[int, ...]


@dataclass(frozen=True)
class PolymarketRidgeReport:
    dataset_sha256: str
    split: PolymarketRidgeSplit
    candidates: tuple[PolymarketRidgeCandidate, ...]
    selected_model: PolymarketRidgeModel
    prevalence_validation_log_loss: float
    selected_validation_log_loss: float
    validation_trials: tuple[PolymarketPolicyMetrics, ...]
    selected_policy: str
    selected_threshold: float | None
    test_log_loss: float
    test_metrics: PolymarketPolicyMetrics
    neural_challenger_authorized: bool
    development_passed: bool
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_RIDGE_REPORT_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_RIDGE_CONTRACT_SHA256,
            "dataset_sha256": self.dataset_sha256,
            "split": self.split.asdict(),
            "candidates": [item.asdict() for item in self.candidates],
            "selected_model": self.selected_model.identity_payload(),
            "selected_model_sha256": self.selected_model.model_sha256,
            "prevalence_validation_log_loss": self.prevalence_validation_log_loss,
            "selected_validation_log_loss": self.selected_validation_log_loss,
            "validation_trials": [item.asdict() for item in self.validation_trials],
            "selected_policy": self.selected_policy,
            "selected_threshold": self.selected_threshold,
            "test_log_loss": self.test_log_loss,
            "test_metrics": self.test_metrics.asdict(),
            "neural_challenger_authorized": self.neural_challenger_authorized,
            "development_passed": self.development_passed,
            "test_evaluations": 1,
            "profitability_claim": False,
            "trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> PolymarketRidgeReport:
        self.split.validated()
        self.selected_model.validated()
        for candidate in self.candidates:
            candidate.validated()
        for metrics in self.validation_trials:
            metrics.validated(require_asset_profit=False)
        self.test_metrics.validated(require_asset_profit=True)
        selected_candidate = min(
            self.candidates,
            key=lambda item: (item.validation_log_loss, -item.l2),
        )
        passing_trials = [item for item in self.validation_trials if item.gate_passed]
        expected_trial = (
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
        expected_policy = "ridge_logit" if expected_trial is not None else "no_trade"
        expected_threshold = (
            None if expected_trial is None else expected_trial.threshold
        )
        if (
            not _is_sha256(self.dataset_sha256)
            or len(self.candidates) != len(POLYMARKET_RIDGE_L2_GRID)
            or tuple(item.l2 for item in self.candidates) != POLYMARKET_RIDGE_L2_GRID
            or len(self.validation_trials) != len(POLYMARKET_RIDGE_THRESHOLD_GRID)
            or tuple(item.threshold for item in self.validation_trials)
            != POLYMARKET_RIDGE_THRESHOLD_GRID
            or self.selected_policy not in {"ridge_logit", "no_trade"}
            or (self.selected_policy == "no_trade") != (self.selected_threshold is None)
            or self.selected_policy != expected_policy
            or self.selected_threshold != expected_threshold
            or self.test_metrics.threshold != self.selected_threshold
            or self.selected_model.l2 != selected_candidate.l2
            or self.selected_model.model_sha256 != selected_candidate.model_sha256
            or self.selected_validation_log_loss
            != selected_candidate.validation_log_loss
            or self.neural_challenger_authorized
            or self.development_passed
            != (self.selected_policy == "ridge_logit" and self.test_metrics.gate_passed)
            or not math.isfinite(self.prevalence_validation_log_loss)
            or self.prevalence_validation_log_loss < 0.0
            or not math.isfinite(self.selected_validation_log_loss)
            or self.selected_validation_log_loss < 0.0
            or not math.isfinite(self.test_log_loss)
            or self.test_log_loss < 0.0
            or not _is_sha256(self.report_sha256)
            or self.report_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket ridge report is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRidgeMaterialization:
    report_sha256: str
    status: str
    selected_validation_action_count: int
    selected_test_action_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class PolymarketRidgeFitAlreadyComplete(ValueError):
    """Signal that a signed report exists without reopening clear test labels."""

    def __init__(self, report_sha256: str) -> None:
        self.report_sha256 = str(report_sha256)
        super().__init__(
            f"Polymarket ridge is already complete:report_sha256={self.report_sha256}"
        )


def begin_polymarket_ridge_fit(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
) -> PolymarketFitClaim:
    """Claim one parent dataset before any test evaluation."""

    dataset.validated()
    return consume_polymarket_fit_claim(
        store,
        experiment="round9_ridge",
        parent_sha256=dataset.pipeline_report_sha256,
        contract_sha256=POLYMARKET_RIDGE_CONTRACT_SHA256,
        dataset_sha256=dataset.dataset_sha256,
        report_table="polymarket_ridge_report",
        report_parent_column="pipeline_report_sha256",
    )


def complete_polymarket_ridge_fit(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    report: PolymarketRidgeReport,
) -> None:
    """Bind a materialized report to its durable fit claim."""

    dataset.validated()
    report.validated()
    if report.dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("Polymarket ridge completion dataset is inconsistent")
    complete_polymarket_fit_claim(
        store,
        experiment="round9_ridge",
        parent_sha256=dataset.pipeline_report_sha256,
        contract_sha256=POLYMARKET_RIDGE_CONTRACT_SHA256,
        dataset_sha256=dataset.dataset_sha256,
        report_table="polymarket_ridge_report",
        report_parent_column="pipeline_report_sha256",
        report_sha256=report.report_sha256,
    )


def fail_polymarket_ridge_fit(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    error: BaseException,
) -> None:
    """Persist a failed claim so a retry cannot silently reopen test."""

    fail_polymarket_fit_claim(
        store,
        experiment="round9_ridge",
        parent_sha256=dataset.pipeline_report_sha256,
        error=error,
    )


def build_polymarket_ridge_dataset(
    *,
    pipeline_report_sha256: str,
    eligibility_sha256: str,
    observations: Sequence[PolymarketRidgeObservation],
) -> PolymarketRidgeDataset:
    """Bind already validated observations to immutable pipeline evidence."""

    provisional = PolymarketRidgeDataset(
        pipeline_report_sha256=str(pipeline_report_sha256),
        eligibility_sha256=str(eligibility_sha256),
        observations=tuple(observations),
        dataset_sha256="",
    )
    return replace(
        provisional,
        dataset_sha256=_ridge_dataset_sha256(
            pipeline_report_sha256=provisional.pipeline_report_sha256,
            eligibility_sha256=provisional.eligibility_sha256,
            action_feature_sha256=(
                item.action_feature_sha256 for item in provisional.observations
            ),
            action_label_sha256=(
                item.action_label_sha256 for item in provisional.observations
            ),
            source_feature_row_sha256=(
                item.source_feature_row_sha256 for item in provisional.observations
            ),
        ),
    ).validated()


def _validated_stored_report(
    raw_json: object,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    try:
        payload = json.loads(str(raw_json))
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored {label} report JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"stored {label} report must be an object")
    claimed = str(payload.pop("report_sha256", ""))
    if claimed != expected_sha256 or _sha256(payload) != expected_sha256:
        raise ValueError(f"stored {label} report digest is invalid")
    return payload


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _training_blocking_entry_terminal_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("Polymarket ridge terminal counts are invalid")
    counts: dict[str, int] = {}
    for reason, count in value.items():
        if (
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("Polymarket ridge terminal counts are invalid")
        counts[reason] = count
    return {
        reason: counts.get(reason, 0)
        for reason in sorted(_TRAINING_BLOCKING_ENTRY_TERMINALS)
        if counts.get(reason, 0) > 0
    }


@dataclass(frozen=True)
class _PolymarketRidgePipelineAuthority:
    report_sha256: str
    report: dict[str, object]
    run_id: str
    run_report_sha256: str
    eligibility_sha256: str
    action_dataset_sha256: tuple[str, ...]


def _load_polymarket_ridge_pipeline_authority(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
) -> _PolymarketRidgePipelineAuthority:
    selected_report = str(pipeline_report_sha256 or "").strip()
    if not _is_sha256(selected_report):
        raise ValueError("Polymarket ridge pipeline report digest is invalid")
    connection = store.connect()
    pipeline_row = connection.execute(
        """
        SELECT report_json, schema_version, contract_sha256, run_id,
               run_report_sha256, eligibility_sha256,
               action_dataset_sha256_json, implementation_sha256
        FROM polymarket_action_value_pipeline WHERE report_sha256 = ?
        """,
        [selected_report],
    ).fetchone()
    if pipeline_row is None:
        raise ValueError("unknown Polymarket action pipeline report")
    pipeline = _validated_stored_report(
        pipeline_row[0],
        expected_sha256=selected_report,
        label="Polymarket action pipeline",
    )
    run_id = str(pipeline_row[3])
    run_report_sha256 = str(pipeline_row[4])
    eligibility_sha256 = str(pipeline_row[5])
    implementation_sha256 = str(pipeline_row[7] or "")
    if (
        str(pipeline_row[1]) != POLYMARKET_ACTION_PIPELINE_SCHEMA_VERSION
        or str(pipeline_row[2]) != POLYMARKET_ACTION_VALUE_CONTRACT_SHA256
        or str(pipeline.get("run_id")) != run_id
        or str(pipeline.get("run_report_sha256")) != run_report_sha256
        or str(pipeline.get("eligibility_sha256")) != eligibility_sha256
        or str(pipeline.get("implementation_sha256")) != implementation_sha256
        or implementation_sha256 != polymarket_action_pipeline_implementation_sha256()
        or not _is_sha256(eligibility_sha256)
    ):
        raise ValueError("Polymarket ridge pipeline authority is invalid")
    continuity_row = connection.execute(
        """
        SELECT report_json, run_id, run_report_sha256
        FROM polymarket_continuity_eligibility_report
        WHERE report_sha256 = ?
        """,
        [eligibility_sha256],
    ).fetchone()
    if continuity_row is None:
        raise ValueError("Polymarket ridge continuity authority is unavailable")
    continuity = _validated_stored_report(
        continuity_row[0],
        expected_sha256=eligibility_sha256,
        label="Polymarket continuity",
    )
    if (
        str(continuity_row[1]) != run_id
        or str(continuity_row[2]) != run_report_sha256
        or not bool(continuity.get("confirmation_eligible"))
        or bool(continuity.get("outcomes_consulted"))
        or bool(continuity.get("labels_consulted"))
        or bool(continuity.get("model_scores_consulted"))
    ):
        raise ValueError("Polymarket ridge continuity authority is insufficient")
    try:
        action_dataset_sha256 = tuple(
            str(value) for value in json.loads(str(pipeline_row[6]))
        )
        report_action_datasets = tuple(
            str(batch["action_dataset_sha256"]) for batch in pipeline["batches"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored Polymarket action dataset list is invalid") from exc
    if (
        not action_dataset_sha256
        or len(set(action_dataset_sha256)) != len(action_dataset_sha256)
        or any(not _is_sha256(value) for value in action_dataset_sha256)
        or action_dataset_sha256 != report_action_datasets
    ):
        raise ValueError("Polymarket ridge action dataset selection is invalid")
    return _PolymarketRidgePipelineAuthority(
        report_sha256=selected_report,
        report=pipeline,
        run_id=run_id,
        run_report_sha256=run_report_sha256,
        eligibility_sha256=eligibility_sha256,
        action_dataset_sha256=action_dataset_sha256,
    )


def load_polymarket_ridge_dataset_identity(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
) -> PolymarketRidgeDatasetIdentity:
    """Hash opaque row identities without reading labels, outcomes, or utilities."""

    authority = _load_polymarket_ridge_pipeline_authority(
        store,
        pipeline_report_sha256=pipeline_report_sha256,
    )
    selected_report = authority.report_sha256
    eligibility_sha256 = authority.eligibility_sha256
    action_dataset_sha256 = authority.action_dataset_sha256
    connection = store.connect()
    connection.execute("DROP TABLE IF EXISTS ridge_identity_action_dataset")
    connection.execute(
        """
        CREATE TEMP TABLE ridge_identity_action_dataset (
            dataset_sha256 VARCHAR PRIMARY KEY
        )
        """
    )
    connection.executemany(
        "INSERT INTO ridge_identity_action_dataset VALUES (?)",
        [(value,) for value in action_dataset_sha256],
    )
    cursor = connection.execute(
        """
        SELECT a.action_feature_sha256, a.action_label_sha256, f.row_sha256
        FROM polymarket_action_value_row AS a
        JOIN ridge_identity_action_dataset AS s USING (dataset_sha256)
        JOIN polymarket_action_value_dataset AS d USING (dataset_sha256)
        JOIN polymarket_feature_row AS f
          ON f.dataset_id = d.source_feature_dataset_sha256
         AND f.feature_id = a.source_feature_id
        JOIN polymarket_market_snapshot AS m
          ON m.run_id = d.source_run_id AND m.condition_id = a.condition_id
        ORDER BY m.event_start_ms, a.decision_received_monotonic_ns,
                 a.condition_id, a.outcome, a.action_feature_sha256
        """
    )
    feature_hashes: list[str] = []
    label_hashes: list[str] = []
    source_hashes: list[str] = []
    for batch in iter(lambda: cursor.fetchmany(16_384), []):
        for feature_hash, label_hash, source_hash in batch:
            values = tuple(map(str, (feature_hash, label_hash, source_hash)))
            if any(not _is_sha256(value) for value in values):
                raise ValueError("Polymarket ridge opaque row identity is invalid")
            feature_hashes.append(values[0])
            label_hashes.append(values[1])
            source_hashes.append(values[2])
    if not feature_hashes or len(set(feature_hashes)) != len(feature_hashes):
        raise ValueError("Polymarket ridge opaque row identities are invalid")
    dataset_sha256 = _ridge_dataset_sha256(
        pipeline_report_sha256=selected_report,
        eligibility_sha256=eligibility_sha256,
        action_feature_sha256=feature_hashes,
        action_label_sha256=label_hashes,
        source_feature_row_sha256=source_hashes,
    )
    return PolymarketRidgeDatasetIdentity(
        pipeline_report_sha256=selected_report,
        eligibility_sha256=eligibility_sha256,
        observation_count=len(feature_hashes),
        dataset_sha256=dataset_sha256,
    ).validated()


def _load_polymarket_ridge_dataset_after_claim(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
) -> PolymarketRidgeDataset:
    """Reconstruct clear rows after the durable test-access claim exists."""

    authority = _load_polymarket_ridge_pipeline_authority(
        store,
        pipeline_report_sha256=pipeline_report_sha256,
    )
    selected_report = authority.report_sha256
    pipeline = authority.report
    run_id = authority.run_id
    eligibility_sha256 = authority.eligibility_sha256
    action_dataset_sha256 = authority.action_dataset_sha256
    connection = store.connect()
    unresolved = _training_blocking_entry_terminal_counts(
        pipeline.get("terminal_reason_counts")
    )
    if unresolved:
        details = ",".join(
            f"{reason}:{count}" for reason, count in sorted(unresolved.items())
        )
        raise ValueError(f"unproven post-submission entry state:{details}")
    connection.execute("DROP TABLE IF EXISTS ridge_selected_action_dataset")
    connection.execute(
        "CREATE TEMP TABLE ridge_selected_action_dataset (dataset_sha256 VARCHAR PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO ridge_selected_action_dataset VALUES (?)",
        [(value,) for value in action_dataset_sha256],
    )
    manifest_rows = connection.execute(
        """
        SELECT d.dataset_sha256, d.source_feature_dataset_sha256,
               d.source_run_id, d.config_json, d.category_counts_json,
               d.terminal_reason_counts_json
        FROM polymarket_action_value_dataset AS d
        JOIN ridge_selected_action_dataset AS s USING (dataset_sha256)
        """
    ).fetchall()
    manifests = {str(row[0]): tuple(row[1:]) for row in manifest_rows}
    if set(manifests) != set(action_dataset_sha256):
        raise ValueError("Polymarket ridge action manifests are incomplete")
    action_rows = connection.execute(
        """
        SELECT a.dataset_sha256, a.action_index,
               a.action_feature_sha256, a.action_label_sha256,
               a.source_feature_id, a.source_input_provenance_sha256,
               a.source_label_free_sha256, a.condition_id, a.market_id,
               a.asset, a.outcome, a.token_id, a.decision_event_id,
               a.decision_received_wall_ms, a.decision_received_monotonic_ns,
               a.feature_values_json, a.terminal_reason, a.category,
               a.classifier_eligible, a.positive_complete,
               a.condition_blocked, a.entry_filled, a.exit_filled,
               a.stress_utility_quote, a.entry_cost_quote,
               a.exit_proceeds_quote, a.net_quote,
               a.creation_book_event_id, a.entry_book_event_id,
               a.exit_decision_book_event_id, a.exit_book_event_id,
               a.entry_execution_parameter_sha256,
               a.exit_execution_parameter_sha256,
               a.execution_evidence_sha256, f.row_sha256, f.official_up,
               m.event_start_ms
        FROM polymarket_action_value_row AS a
        JOIN ridge_selected_action_dataset AS s USING (dataset_sha256)
        JOIN polymarket_action_value_dataset AS d USING (dataset_sha256)
        JOIN polymarket_feature_row AS f
          ON f.dataset_id = d.source_feature_dataset_sha256
         AND f.feature_id = a.source_feature_id
        JOIN polymarket_market_snapshot AS m
          ON m.run_id = d.source_run_id AND m.condition_id = a.condition_id
        ORDER BY a.dataset_sha256, a.action_index
        """
    ).fetchall()
    integrity = store.resume_integrity_errors(run_id)
    if integrity:
        raise ValueError(
            "Polymarket ridge source integrity failed: " + "; ".join(integrity)
        )
    needed_event_ids = {
        str(row[index])
        for row in action_rows
        for index in (27, 28, 29, 30)
        if str(row[index])
    }
    selected_conditions = tuple(sorted({str(row[7]) for row in action_rows}))
    event_time: dict[str, int] = {}
    if needed_event_ids:
        for decoded in store.iter_public_events(
            run_id,
            streams=("clob_market", "clob_rest_book"),
            condition_ids=selected_conditions,
            verified_source=True,
        ):
            if decoded.event_id not in needed_event_ids:
                continue
            observed = event_time.get(decoded.event_id)
            if observed is not None and observed != decoded.received_monotonic_ns:
                raise ValueError(
                    "Polymarket ridge execution event receipt time differs"
                )
            event_time[decoded.event_id] = decoded.received_monotonic_ns
            if len(event_time) == len(needed_event_ids):
                break
    rows_by_dataset: dict[str, list[tuple[object, ...]]] = {
        value: [] for value in action_dataset_sha256
    }
    for row in action_rows:
        rows_by_dataset[str(row[0])].append(tuple(row))
    observations: list[PolymarketRidgeObservation] = []
    for dataset_sha256 in action_dataset_sha256:
        manifest = manifests[dataset_sha256]
        source_feature_dataset_sha256 = str(manifest[0])
        source_run_id = str(manifest[1])
        if source_run_id != run_id:
            raise ValueError("Polymarket ridge action run identity differs")
        try:
            config = PolymarketActionValueConfig(
                **json.loads(str(manifest[2]))
            ).validated()
            category_counts = {
                str(key): int(value)
                for key, value in json.loads(str(manifest[3])).items()
            }
            terminal_counts = {
                str(key): int(value)
                for key, value in json.loads(str(manifest[4])).items()
            }
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Polymarket ridge action manifest JSON is invalid"
            ) from exc
        features: list[PolymarketActionFeature] = []
        labels: list[PolymarketActionLabel] = []
        source_rows: list[tuple[str, bool | None, int, int]] = []
        stored_rows = rows_by_dataset[dataset_sha256]
        if [int(row[1]) for row in stored_rows] != list(range(len(stored_rows))):
            raise ValueError("Polymarket ridge action indexes are discontinuous")
        for row in stored_rows:
            try:
                values = tuple(float(value) for value in json.loads(str(row[15])))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("Polymarket ridge feature JSON is invalid") from exc
            feature = PolymarketActionFeature(
                action_feature_id=str(row[2]),
                source_run_id=source_run_id,
                source_feature_id=str(row[4]),
                source_input_provenance_sha256=str(row[5]),
                source_label_free_sha256=str(row[6]),
                condition_id=str(row[7]),
                market_id=str(row[8]),
                asset=str(row[9]),
                outcome=str(row[10]),
                token_id=str(row[11]),
                decision_event_id=str(row[12]),
                decision_received_wall_ms=int(row[13]),
                decision_received_monotonic_ns=int(row[14]),
                feature_values=values,
                action_feature_sha256=str(row[2]),
            ).validated()
            label = PolymarketActionLabel(
                action_label_id=str(row[3]),
                action_feature_sha256=str(row[2]),
                terminal_reason=str(row[16]),
                category=str(row[17]),
                classifier_eligible=bool(row[18]),
                positive_complete=bool(row[19]),
                condition_blocked=bool(row[20]),
                entry_filled=bool(row[21]),
                exit_filled=bool(row[22]),
                stress_utility_quote=Decimal(str(row[23])),
                entry_cost_quote=_optional_decimal(row[24]),
                exit_proceeds_quote=_optional_decimal(row[25]),
                net_quote=_optional_decimal(row[26]),
                creation_book_event_id=str(row[27]),
                entry_book_event_id=str(row[28]),
                exit_decision_book_event_id=str(row[29]),
                exit_book_event_id=str(row[30]),
                entry_execution_parameter_sha256=str(row[31]),
                exit_execution_parameter_sha256=str(row[32]),
                execution_evidence_sha256=str(row[33]),
                action_label_sha256=str(row[3]),
            ).validated()
            known_times = [feature.decision_received_monotonic_ns]
            for event_id in (str(row[27]), str(row[28]), str(row[29]), str(row[30])):
                if event_id:
                    observed = event_time.get(event_id)
                    if observed is None:
                        raise ValueError(
                            "Polymarket ridge execution event time is missing"
                        )
                    known_times.append(observed)
            if label.category == "successful_round_trip" and not str(row[30]):
                raise ValueError(
                    "Polymarket ridge completed action has no exit receipt"
                )
            if label.category == "entry_no_fill" and not str(row[28]):
                raise ValueError("Polymarket ridge no-fill action has no receipt")
            features.append(feature)
            labels.append(label)
            official_up = None if row[35] is None else bool(row[35])
            source_rows.append(
                (str(row[34]), official_up, int(row[36]), max(known_times))
            )
        action_dataset = PolymarketActionValueDataset(
            source_feature_dataset_sha256=source_feature_dataset_sha256,
            source_run_id=source_run_id,
            config=config,
            features=tuple(features),
            labels=tuple(labels),
            category_counts=category_counts,
            terminal_reason_counts=terminal_counts,
            dataset_sha256=dataset_sha256,
        ).validated()
        for feature, label, source in zip(
            action_dataset.features,
            action_dataset.labels,
            source_rows,
            strict=True,
        ):
            observations.append(
                PolymarketRidgeObservation(
                    action_feature_sha256=feature.action_feature_sha256,
                    action_label_sha256=label.action_label_sha256,
                    source_feature_row_sha256=source[0],
                    condition_id=feature.condition_id,
                    asset=feature.asset,
                    outcome=feature.outcome,
                    event_start_ms=source[2],
                    decision_received_monotonic_ns=(
                        feature.decision_received_monotonic_ns
                    ),
                    release_monotonic_ns=source[3],
                    feature_values=feature.feature_values,
                    official_up=source[1],
                    classifier_eligible=label.classifier_eligible,
                    positive_complete=label.positive_complete,
                    category=label.category,
                    condition_blocked=label.condition_blocked,
                    stress_utility_quote=float(label.stress_utility_quote),
                ).validated()
            )
    observations.sort(
        key=lambda item: (
            item.event_start_ms,
            item.decision_received_monotonic_ns,
            item.condition_id,
            item.outcome,
            item.action_feature_sha256,
        )
    )
    return build_polymarket_ridge_dataset(
        pipeline_report_sha256=selected_report,
        eligibility_sha256=eligibility_sha256,
        observations=observations,
    )


def load_polymarket_ridge_dataset(
    store: PolymarketEvidenceStore,
    *,
    pipeline_report_sha256: str,
) -> PolymarketRidgeDataset:
    """Claim opaque evidence, then reconstruct and validate clear action labels."""

    identity = load_polymarket_ridge_dataset_identity(
        store,
        pipeline_report_sha256=pipeline_report_sha256,
    )
    claim = reserve_polymarket_fit_claim(
        store,
        experiment="round9_ridge",
        parent_sha256=identity.pipeline_report_sha256,
        contract_sha256=POLYMARKET_RIDGE_CONTRACT_SHA256,
        dataset_sha256=identity.dataset_sha256,
        report_table="polymarket_ridge_report",
        report_parent_column="pipeline_report_sha256",
    )
    if claim.status == "existing":
        raise PolymarketRidgeFitAlreadyComplete(claim.report_sha256)
    try:
        dataset = _load_polymarket_ridge_dataset_after_claim(
            store,
            pipeline_report_sha256=identity.pipeline_report_sha256,
        )
        if (
            dataset.dataset_sha256 != identity.dataset_sha256
            or len(dataset.observations) != identity.observation_count
            or dataset.eligibility_sha256 != identity.eligibility_sha256
        ):
            raise ValueError("Polymarket ridge clear dataset identity differs")
    except Exception as exc:
        if claim.status == "claimed":
            discard_polymarket_fit_reservation(
                store,
                experiment="round9_ridge",
                parent_sha256=identity.pipeline_report_sha256,
                contract_sha256=POLYMARKET_RIDGE_CONTRACT_SHA256,
                dataset_sha256=identity.dataset_sha256,
                report_table="polymarket_ridge_report",
                report_parent_column="pipeline_report_sha256",
            )
            fail_polymarket_fit_claim(
                store,
                experiment="round9_ridge",
                parent_sha256=identity.pipeline_report_sha256,
                error=exc,
            )
        raise
    return dataset


def _ridge_metrics_from_payload(raw: object) -> PolymarketPolicyMetrics:
    if not isinstance(raw, Mapping):
        raise ValueError("stored Polymarket ridge metrics are invalid")

    def count(name: str) -> int:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{name} must be a nonnegative integer")
        return value

    def number(name: str) -> float:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise TypeError(f"{name} must be finite")
        return result

    def count_map(name: str) -> dict[str, int]:
        value = raw[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be an object")
        result: dict[str, int] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
            ):
                raise TypeError(f"{name} contains an invalid count")
            result[key] = item
        return result

    def number_map(name: str) -> dict[str, float]:
        value = raw[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be an object")
        result: dict[str, float] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise TypeError(f"{name} contains an invalid value")
            result[key] = float(item)
        return result

    try:
        threshold_raw = raw["threshold"]
        if threshold_raw is not None and (
            isinstance(threshold_raw, bool)
            or not isinstance(threshold_raw, (int, float))
        ):
            raise TypeError("threshold must be numeric or null")
        gate_passed = raw["gate_passed"]
        gate_reasons = raw["gate_reasons"]
        if not isinstance(gate_passed, bool) or not isinstance(
            gate_reasons,
            (list, tuple),
        ):
            raise TypeError("gate fields are invalid")
        if any(not isinstance(value, str) for value in gate_reasons):
            raise TypeError("gate reasons must be strings")
        selected_action_sha256 = raw["selected_action_sha256"]
        if not isinstance(selected_action_sha256, str):
            raise TypeError("selected action digest must be a string")
        return PolymarketPolicyMetrics(
            threshold=None if threshold_raw is None else float(threshold_raw),
            attempt_count=count("attempt_count"),
            completed_trade_count=count("completed_trade_count"),
            completed_by_asset=count_map("completed_by_asset"),
            positive_complete_count=count("positive_complete_count"),
            failed_exit_count=count("failed_exit_count"),
            aggregate_stress_utility_quote=number("aggregate_stress_utility_quote"),
            pnl_by_asset=number_map("pnl_by_asset"),
            median_market_pnl_quote=number("median_market_pnl_quote"),
            maximum_realized_drawdown_quote=number("maximum_realized_drawdown_quote"),
            positive_complete_precision=number("positive_complete_precision"),
            wilson_lower_bound_95=number("wilson_lower_bound_95"),
            selected_action_sha256=selected_action_sha256,
            gate_passed=gate_passed,
            gate_reasons=tuple(gate_reasons),
        )
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("stored Polymarket ridge metrics are invalid") from exc


def load_polymarket_ridge_report(
    store: PolymarketEvidenceStore,
    *,
    report_sha256: str,
) -> PolymarketRidgeReport:
    """Reconstruct and validate one persisted ridge report and model."""

    selected = str(report_sha256 or "").strip()
    if not _is_sha256(selected):
        raise ValueError("Polymarket ridge report digest is invalid")
    row = (
        store.connect()
        .execute(
            """
        SELECT schema_version, contract_sha256, dataset_sha256,
               pipeline_report_sha256, eligibility_sha256,
               selected_model_sha256, selected_policy, selected_threshold,
               development_passed, model_json, report_json
        FROM polymarket_ridge_report WHERE report_sha256 = ?
        """,
            [selected],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("unknown Polymarket ridge report")
    report_payload = _validated_stored_report(
        row[10],
        expected_sha256=selected,
        label="Polymarket ridge",
    )
    try:
        model_payload = json.loads(str(row[9]))
        if not isinstance(model_payload, dict):
            raise TypeError("model payload must be an object")
        model_sha256 = str(model_payload.pop("model_sha256"))
        model = PolymarketRidgeModel(
            l2=float(model_payload["l2"]),
            feature_mean=tuple(float(value) for value in model_payload["feature_mean"]),
            feature_scale=tuple(
                float(value) for value in model_payload["feature_scale"]
            ),
            coefficients=tuple(float(value) for value in model_payload["coefficients"]),
            intercept=float(model_payload["intercept"]),
            optimizer_iterations=int(model_payload["optimizer_iterations"]),
            optimizer_objective=float(model_payload["optimizer_objective"]),
            model_sha256=model_sha256,
        ).validated()
        split_payload = dict(report_payload["split"])
        split = PolymarketRidgeSplit(
            train_groups=tuple(int(value) for value in split_payload["train_groups"]),
            validation_groups=tuple(
                int(value) for value in split_payload["validation_groups"]
            ),
            test_groups=tuple(int(value) for value in split_payload["test_groups"]),
            purged_groups=tuple(int(value) for value in split_payload["purged_groups"]),
        )
        candidates = tuple(
            PolymarketRidgeCandidate(
                l2=float(value["l2"]),
                validation_log_loss=float(value["validation_log_loss"]),
                model_sha256=str(value["model_sha256"]),
                optimizer_iterations=int(value["optimizer_iterations"]),
                optimizer_objective=float(value["optimizer_objective"]),
            )
            for value in report_payload["candidates"]
        )
        report = PolymarketRidgeReport(
            dataset_sha256=str(report_payload["dataset_sha256"]),
            split=split,
            candidates=candidates,
            selected_model=model,
            prevalence_validation_log_loss=float(
                report_payload["prevalence_validation_log_loss"]
            ),
            selected_validation_log_loss=float(
                report_payload["selected_validation_log_loss"]
            ),
            validation_trials=tuple(
                _ridge_metrics_from_payload(value)
                for value in report_payload["validation_trials"]
            ),
            selected_policy=str(report_payload["selected_policy"]),
            selected_threshold=(
                None
                if report_payload["selected_threshold"] is None
                else float(report_payload["selected_threshold"])
            ),
            test_log_loss=float(report_payload["test_log_loss"]),
            test_metrics=_ridge_metrics_from_payload(report_payload["test_metrics"]),
            neural_challenger_authorized=bool(
                report_payload["neural_challenger_authorized"]
            ),
            development_passed=bool(report_payload["development_passed"]),
            report_sha256=selected,
        ).validated()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stored Polymarket ridge report structure is invalid") from exc
    if (
        str(row[0]) != POLYMARKET_RIDGE_REPORT_SCHEMA_VERSION
        or str(row[1]) != POLYMARKET_RIDGE_CONTRACT_SHA256
        or str(row[2]) != report.dataset_sha256
        or not _is_sha256(row[3])
        or not _is_sha256(row[4])
        or str(row[5]) != report.selected_model.model_sha256
        or str(row[6]) != report.selected_policy
        or row[7] != report.selected_threshold
        or bool(row[8]) != report.development_passed
        or model_payload != model.identity_payload()
        or report_payload.get("selected_model") != model.identity_payload()
        or str(report_payload.get("selected_model_sha256")) != model.model_sha256
        or int(report_payload.get("test_evaluations", 0)) != 1
        or bool(report_payload.get("profitability_claim"))
        or bool(report_payload.get("trading_authority"))
    ):
        raise ValueError("stored Polymarket ridge report columns are inconsistent")
    return report


def load_polymarket_ridge_evidence(
    store: PolymarketEvidenceStore,
    *,
    report_sha256: str,
) -> tuple[PolymarketRidgeDataset, PolymarketRidgeReport]:
    """Load and replay-check the complete parent evidence for a challenger."""

    selected = str(report_sha256 or "").strip()
    if not _is_sha256(selected):
        raise ValueError("Polymarket ridge report digest is invalid")
    row = (
        store.connect()
        .execute(
            """
        SELECT pipeline_report_sha256, eligibility_sha256,
               dataset_sha256, contract_sha256
        FROM polymarket_ridge_report
        WHERE report_sha256 = ?
        """,
            [selected],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("unknown Polymarket ridge report")
    dataset_sha256 = str(row[2])
    if (
        not _is_sha256(row[0])
        or not _is_sha256(row[1])
        or not _is_sha256(dataset_sha256)
        or str(row[3]) != POLYMARKET_RIDGE_CONTRACT_SHA256
    ):
        raise ValueError("Polymarket ridge parent identity is invalid")
    claim = reserve_polymarket_fit_claim(
        store,
        experiment="round9_mlp",
        parent_sha256=selected,
        contract_sha256=POLYMARKET_ROUND9_MLP_CONTRACT_SHA256,
        dataset_sha256=dataset_sha256,
        report_table="polymarket_mlp_report",
        report_parent_column="parent_ridge_report_sha256",
    )
    if claim.status == "existing":
        raise ValueError(
            f"Polymarket MLP is already complete:report_sha256={claim.report_sha256}"
        )
    try:
        dataset = _load_polymarket_ridge_dataset_after_claim(
            store,
            pipeline_report_sha256=str(row[0]),
        )
        report = load_polymarket_ridge_report(store, report_sha256=selected)
        if (
            report.dataset_sha256 != dataset.dataset_sha256
            or dataset.dataset_sha256 != dataset_sha256
            or dataset.pipeline_report_sha256 != str(row[0])
            or dataset.eligibility_sha256 != str(row[1])
        ):
            raise ValueError("Polymarket ridge report dataset is inconsistent")
        materialization = materialize_polymarket_ridge_report(store, dataset, report)
        if materialization.status != "existing":
            raise ValueError("Polymarket ridge parent was not previously materialized")
    except Exception as exc:
        if claim.status == "claimed":
            discard_polymarket_fit_reservation(
                store,
                experiment="round9_mlp",
                parent_sha256=selected,
                contract_sha256=POLYMARKET_ROUND9_MLP_CONTRACT_SHA256,
                dataset_sha256=dataset_sha256,
                report_table="polymarket_mlp_report",
                report_parent_column="parent_ridge_report_sha256",
            )
            fail_polymarket_fit_claim(
                store,
                experiment="round9_mlp",
                parent_sha256=selected,
                error=exc,
            )
        raise
    return dataset, report


def load_polymarket_ridge_materialization(
    store: PolymarketEvidenceStore,
    report: PolymarketRidgeReport,
) -> PolymarketRidgeMaterialization:
    """Read an existing materialization without reconstructing clear labels."""

    report.validated()
    rows = (
        store.connect()
        .execute(
            """
        SELECT partition, count(*)
        FROM polymarket_ridge_selected_action
        WHERE report_sha256 = ?
        GROUP BY partition
        """,
            [report.report_sha256],
        )
        .fetchall()
    )
    counts = {str(partition): int(count) for partition, count in rows}
    selected_validation = next(
        (
            item.attempt_count
            for item in report.validation_trials
            if item.threshold == report.selected_threshold
        ),
        0,
    )
    expected = {
        "validation": selected_validation,
        "test": report.test_metrics.attempt_count,
    }
    observed = {partition: counts.get(partition, 0) for partition in expected}
    if observed != expected or any(partition not in expected for partition in counts):
        raise ValueError("stored Polymarket ridge selected-action counts are invalid")
    return PolymarketRidgeMaterialization(
        report_sha256=report.report_sha256,
        status="existing",
        selected_validation_action_count=observed["validation"],
        selected_test_action_count=observed["test"],
    )


def split_polymarket_ridge_dataset(
    dataset: PolymarketRidgeDataset,
) -> PolymarketRidgeSplit:
    """Freeze chronological train/purge/validation/purge/test groups."""

    dataset.validated()
    groups = dataset.group_starts_ms
    if len(groups) < 30:
        raise ValueError(f"insufficient synchronized groups:{len(groups)}/30")
    validation_count = math.floor(0.2 * len(groups))
    test_count = math.floor(0.2 * len(groups))
    train_count = len(groups) - validation_count - test_count - 2
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Polymarket ridge split has an empty partition")
    train = groups[:train_count]
    first_purge = groups[train_count]
    validation_start = train_count + 1
    validation = groups[validation_start : validation_start + validation_count]
    second_purge = groups[validation_start + validation_count]
    test = groups[validation_start + validation_count + 1 :]
    split = PolymarketRidgeSplit(
        train_groups=train,
        validation_groups=validation,
        test_groups=test,
        purged_groups=(first_purge, second_purge),
    )
    for name, partition in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        allowed = set(partition)
        for asset in _ASSETS:
            outcomes = {
                item.official_up
                for item in dataset.observations
                if item.event_start_ms in allowed and item.asset == asset
            }
            if outcomes != {False, True}:
                raise ValueError(
                    f"Polymarket ridge {name} outcome breadth is incomplete:{asset}"
                )
    return split


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
    if features.ndim != 2 or features.shape[1] != _FEATURE_COUNT:
        raise ValueError("Polymarket ridge feature matrix is invalid")
    if not np.all(np.isfinite(features)) or set(np.unique(labels)) - {0.0, 1.0}:
        raise ValueError("Polymarket ridge matrix contains invalid values")
    return features, labels


def fit_polymarket_ridge_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
) -> PolymarketRidgeModel:
    """Fit one unweighted ridge logistic model under the frozen solver contract."""

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if (
        l2 not in POLYMARKET_RIDGE_L2_GRID
        or values.ndim != 2
        or values.shape[1] != _FEATURE_COUNT
        or target.shape != (values.shape[0],)
        or values.shape[0] == 0
        or set(np.unique(target)) != {0.0, 1.0}
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Polymarket ridge fit inputs are invalid")
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = (values - mean) / scale

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        intercept = theta[-1]
        logits = standardized @ coefficients + intercept
        loss = float(
            np.mean(np.logaddexp(0.0, logits) - target * logits)
            + 0.5 * l2 * np.dot(coefficients, coefficients)
        )
        residual = expit(logits) - target
        gradient = np.empty_like(theta)
        gradient[:-1] = standardized.T @ residual / target.size + l2 * coefficients
        gradient[-1] = float(np.mean(residual))
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(_FEATURE_COUNT + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(f"Polymarket ridge optimizer failed:{result.message}")
    provisional = PolymarketRidgeModel(
        l2=float(l2),
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in result.x[:-1]),
        intercept=float(result.x[-1]),
        optimizer_iterations=int(result.nit),
        optimizer_objective=float(result.fun),
        model_sha256="",
    )
    return replace(
        provisional,
        model_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _wilson_lower_bound(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = probability + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (center - margin) / denominator)


def _evaluate_policy(
    dataset: PolymarketRidgeDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None,
    *,
    require_asset_profit: bool,
) -> PolymarketPolicyEvaluation:
    if probabilities.shape != indices.shape:
        raise ValueError("Polymarket policy probabilities are misaligned")
    by_decision: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for index, probability in zip(indices, probabilities, strict=True):
        item = dataset.observations[int(index)]
        by_decision.setdefault(
            (item.condition_id, item.decision_received_monotonic_ns), []
        ).append((int(index), float(probability)))
    release_by_condition: dict[str, int] = {}
    blocked: set[str] = set()
    selected: list[int] = []
    completed_by_asset = {asset: 0 for asset in _ASSETS}
    pnl_by_asset = {asset: 0.0 for asset in _ASSETS}
    pnl_by_market: dict[str, float] = {}
    realized_events: list[tuple[int, int, float]] = []
    positive_count = 0
    failed_count = 0
    completed_count = 0
    for (condition_id, decision_ns), candidates in sorted(
        by_decision.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if threshold is None or condition_id in blocked:
            continue
        if decision_ns < release_by_condition.get(condition_id, 0):
            continue
        passing = [item for item in candidates if item[1] >= threshold]
        if not passing:
            continue
        best_probability = max(item[1] for item in passing)
        winners = [item for item in passing if item[1] == best_probability]
        if len(winners) != 1:
            continue
        selected_index = winners[0][0]
        observation = dataset.observations[selected_index]
        selected.append(selected_index)
        utility = observation.stress_utility_quote
        pnl_by_asset[observation.asset] += utility
        pnl_by_market[condition_id] = pnl_by_market.get(condition_id, 0.0) + utility
        realized_events.append(
            (observation.release_monotonic_ns, selected_index, utility)
        )
        positive_count += int(observation.positive_complete)
        if observation.category == "successful_round_trip":
            completed_count += 1
            completed_by_asset[observation.asset] += 1
            release_by_condition[condition_id] = observation.release_monotonic_ns
        elif observation.category == "entry_no_fill":
            release_by_condition[condition_id] = observation.release_monotonic_ns
        elif observation.category == "filled_entry_failed_exit":
            failed_count += 1
            blocked.add(condition_id)
    selected_hashes = [
        dataset.observations[index].action_feature_sha256 for index in selected
    ]
    attempts = len(selected)
    aggregate = math.fsum(pnl_by_asset.values())
    market_values = tuple(pnl_by_market.values())
    median_market = float(np.median(market_values)) if market_values else 0.0
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for _release, _index, value in sorted(realized_events):
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    precision = positive_count / attempts if attempts else 0.0
    reasons: list[str] = []
    if completed_count < 30:
        reasons.append(f"completed_trade_count:{completed_count}/30")
    for asset in _ASSETS:
        if completed_by_asset[asset] < 5:
            reasons.append(
                f"completed_trade_count:{asset}:{completed_by_asset[asset]}/5"
            )
    if aggregate <= 0.0:
        reasons.append("aggregate_stress_utility_not_positive")
    if median_market <= 0.0:
        reasons.append("median_market_pnl_not_positive")
    if failed_count:
        reasons.append(f"filled_entry_failed_exit_count:{failed_count}")
    if require_asset_profit:
        for asset in _ASSETS:
            if pnl_by_asset[asset] <= 0.0:
                reasons.append(f"asset_pnl_not_positive:{asset}")
    metrics = PolymarketPolicyMetrics(
        threshold=threshold,
        attempt_count=attempts,
        completed_trade_count=completed_count,
        completed_by_asset=completed_by_asset,
        positive_complete_count=positive_count,
        failed_exit_count=failed_count,
        aggregate_stress_utility_quote=aggregate,
        pnl_by_asset=pnl_by_asset,
        median_market_pnl_quote=median_market,
        maximum_realized_drawdown_quote=maximum_drawdown,
        positive_complete_precision=precision,
        wilson_lower_bound_95=_wilson_lower_bound(positive_count, attempts),
        selected_action_sha256=_sha256(selected_hashes),
        gate_passed=not reasons,
        gate_reasons=tuple(sorted(reasons)),
    )
    return PolymarketPolicyEvaluation(
        metrics=metrics,
        selected_indices=tuple(selected),
    )


def evaluate_polymarket_policy(
    dataset: PolymarketRidgeDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None,
    *,
    require_asset_profit: bool,
) -> PolymarketPolicyEvaluation:
    """Apply the shared causal one-condition action policy."""

    return _evaluate_policy(
        dataset,
        indices,
        probabilities,
        threshold,
        require_asset_profit=require_asset_profit,
    )


def fit_and_evaluate_polymarket_ridge(
    dataset: PolymarketRidgeDataset,
) -> PolymarketRidgeReport:
    """Select L2/threshold on validation and evaluate untouched test once."""

    dataset.validated()
    split = split_polymarket_ridge_dataset(dataset)
    train_indices = _partition_indices(dataset, split.train_groups)
    validation_indices = _partition_indices(dataset, split.validation_groups)
    test_indices = _partition_indices(dataset, split.test_groups)
    train_x, train_y = _matrix(dataset, train_indices)
    validation_x, validation_y = _matrix(dataset, validation_indices)
    test_x, test_y = _matrix(dataset, test_indices)
    candidates: list[PolymarketRidgeCandidate] = []
    models: dict[float, PolymarketRidgeModel] = {}
    for l2 in POLYMARKET_RIDGE_L2_GRID:
        model = fit_polymarket_ridge_model(train_x, train_y, l2=l2)
        loss = _binary_log_loss(model.predict(validation_x), validation_y)
        models[l2] = model
        candidates.append(
            PolymarketRidgeCandidate(
                l2=l2,
                validation_log_loss=loss,
                model_sha256=model.model_sha256,
                optimizer_iterations=model.optimizer_iterations,
                optimizer_objective=model.optimizer_objective,
            )
        )
    selected_candidate = min(
        candidates,
        key=lambda item: (item.validation_log_loss, -item.l2),
    )
    selected_model = models[selected_candidate.l2]
    validation_probability = selected_model.predict(validation_x)
    prevalence = float(np.mean(train_y))
    prevalence_loss = _binary_log_loss(
        np.full(validation_y.shape, prevalence, dtype=np.float64),
        validation_y,
    )
    validation_evaluations = tuple(
        _evaluate_policy(
            dataset,
            validation_indices,
            validation_probability,
            threshold,
            require_asset_profit=False,
        )
        for threshold in POLYMARKET_RIDGE_THRESHOLD_GRID
    )
    passed = [item for item in validation_evaluations if item.metrics.gate_passed]
    if passed:
        selected_evaluation = max(
            passed,
            key=lambda item: (
                item.metrics.wilson_lower_bound_95,
                float(item.metrics.threshold or 0.0),
            ),
        )
        selected_policy = "ridge_logit"
        selected_threshold = selected_evaluation.metrics.threshold
    else:
        selected_policy = "no_trade"
        selected_threshold = None
    test_probability = selected_model.predict(test_x)
    test_log_loss = _binary_log_loss(test_probability, test_y)
    test_evaluation = _evaluate_policy(
        dataset,
        test_indices,
        test_probability,
        selected_threshold,
        require_asset_profit=True,
    )
    provisional = PolymarketRidgeReport(
        dataset_sha256=dataset.dataset_sha256,
        split=split,
        candidates=tuple(candidates),
        selected_model=selected_model,
        prevalence_validation_log_loss=prevalence_loss,
        selected_validation_log_loss=selected_candidate.validation_log_loss,
        validation_trials=tuple(item.metrics for item in validation_evaluations),
        selected_policy=selected_policy,
        selected_threshold=selected_threshold,
        test_log_loss=test_log_loss,
        test_metrics=test_evaluation.metrics,
        neural_challenger_authorized=False,
        development_passed=(
            selected_policy == "ridge_logit" and test_evaluation.metrics.gate_passed
        ),
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def polymarket_selected_policy_tables(
    dataset: PolymarketRidgeDataset,
    *,
    report_sha256: str,
    partition: str,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None,
    require_asset_profit: bool,
) -> tuple[
    PolymarketPolicyEvaluation,
    list[tuple[object, ...]],
    list[tuple[object, ...]],
    list[tuple[object, ...]],
]:
    evaluation = _evaluate_policy(
        dataset,
        indices,
        probabilities,
        threshold,
        require_asset_profit=require_asset_profit,
    )
    probability_by_index = {
        int(index): float(probability)
        for index, probability in zip(indices, probabilities, strict=True)
    }
    selected_rows: list[tuple[object, ...]] = []
    for sequence, index in enumerate(evaluation.selected_indices):
        item = dataset.observations[index]
        selected_rows.append(
            (
                report_sha256,
                partition,
                sequence,
                item.action_feature_sha256,
                item.action_label_sha256,
                item.condition_id,
                item.asset,
                item.outcome,
                item.event_start_ms,
                item.decision_received_monotonic_ns,
                item.release_monotonic_ns,
                probability_by_index[index],
                item.category,
                item.positive_complete,
                item.condition_blocked,
                format(item.stress_utility_quote, ".17g"),
            )
        )
    equity = 0.0
    peak = 0.0
    equity_rows: list[tuple[object, ...]] = []
    for sequence, row in enumerate(
        sorted(
            selected_rows,
            key=lambda value: (int(value[10]), str(value[3])),
        )
    ):
        pnl = float(row[15])
        equity += pnl
        peak = max(peak, equity)
        equity_rows.append(
            (
                report_sha256,
                partition,
                sequence,
                int(row[10]),
                str(row[3]),
                format(pnl, ".17g"),
                format(equity, ".17g"),
                format(peak - equity, ".17g"),
            )
        )
    market_values: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for row in selected_rows:
        market_values.setdefault((str(row[5]), str(row[6])), []).append(
            (float(row[15]), str(row[12]))
        )
    market_rows = [
        (
            report_sha256,
            partition,
            condition_id,
            asset,
            len(values),
            sum(category == "successful_round_trip" for _pnl, category in values),
            format(math.fsum(pnl for pnl, _category in values), ".17g"),
        )
        for (condition_id, asset), values in sorted(market_values.items())
    ]
    return evaluation, selected_rows, equity_rows, market_rows


def materialize_polymarket_ridge_report(
    store: PolymarketEvidenceStore,
    dataset: PolymarketRidgeDataset,
    report: PolymarketRidgeReport,
) -> PolymarketRidgeMaterialization:
    """Persist model, trials, selected actions, and realized equity atomically."""

    dataset.validated()
    report.validated()
    if report.dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("Polymarket ridge report belongs to another dataset")
    validation_indices = _partition_indices(dataset, report.split.validation_groups)
    test_indices = _partition_indices(dataset, report.split.test_groups)
    validation_x, _validation_y = _matrix(dataset, validation_indices)
    test_x, _test_y = _matrix(dataset, test_indices)
    validation_probability = report.selected_model.predict(validation_x)
    test_probability = report.selected_model.predict(test_x)
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
    if test_evaluation.metrics.asdict() != report.test_metrics.asdict():
        raise ValueError("Polymarket ridge test replay differs from report")
    if report.selected_threshold is not None:
        expected_validation = next(
            item
            for item in report.validation_trials
            if item.threshold == report.selected_threshold
        )
        if validation_evaluation.metrics.asdict() != expected_validation.asdict():
            raise ValueError("Polymarket ridge validation replay differs from report")
    connection = store.connect()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_ridge_report (
            report_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            dataset_sha256 VARCHAR NOT NULL,
            pipeline_report_sha256 VARCHAR NOT NULL,
            eligibility_sha256 VARCHAR NOT NULL,
            selected_model_sha256 VARCHAR NOT NULL,
            selected_policy VARCHAR NOT NULL,
            selected_threshold DOUBLE,
            development_passed BOOLEAN NOT NULL,
            model_json VARCHAR NOT NULL,
            report_json VARCHAR NOT NULL
        );
        CREATE TABLE IF NOT EXISTS polymarket_ridge_candidate (
            report_sha256 VARCHAR NOT NULL, l2 DOUBLE NOT NULL,
            validation_log_loss DOUBLE NOT NULL, model_sha256 VARCHAR NOT NULL,
            optimizer_iterations INTEGER NOT NULL,
            optimizer_objective DOUBLE NOT NULL,
            PRIMARY KEY(report_sha256, l2)
        );
        CREATE TABLE IF NOT EXISTS polymarket_ridge_threshold_trial (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            trial_key VARCHAR NOT NULL, threshold DOUBLE,
            metrics_json VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, trial_key)
        );
        CREATE TABLE IF NOT EXISTS polymarket_ridge_split_group (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            ordinal INTEGER NOT NULL, event_start_ms BIGINT NOT NULL,
            PRIMARY KEY(report_sha256, partition, ordinal)
        );
        CREATE TABLE IF NOT EXISTS polymarket_ridge_selected_action (
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
        CREATE TABLE IF NOT EXISTS polymarket_ridge_equity (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            sequence UBIGINT NOT NULL, release_monotonic_ns UBIGINT NOT NULL,
            action_feature_sha256 VARCHAR NOT NULL, pnl_quote VARCHAR NOT NULL,
            equity_quote VARCHAR NOT NULL, drawdown_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, sequence)
        );
        CREATE TABLE IF NOT EXISTS polymarket_ridge_market_pnl (
            report_sha256 VARCHAR NOT NULL, partition VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL, asset VARCHAR NOT NULL,
            attempt_count UBIGINT NOT NULL, completed_trade_count UBIGINT NOT NULL,
            pnl_quote VARCHAR NOT NULL,
            PRIMARY KEY(report_sha256, partition, condition_id)
        );
        """
    )
    model_json = _canonical_json(
        {
            **report.selected_model.identity_payload(),
            "model_sha256": report.selected_model.model_sha256,
        }
    )
    report_row = (
        report.report_sha256,
        POLYMARKET_RIDGE_REPORT_SCHEMA_VERSION,
        POLYMARKET_RIDGE_CONTRACT_SHA256,
        dataset.dataset_sha256,
        dataset.pipeline_report_sha256,
        dataset.eligibility_sha256,
        report.selected_model.model_sha256,
        report.selected_policy,
        report.selected_threshold,
        report.development_passed,
        model_json,
        _canonical_json(report.asdict()),
    )
    candidate_rows = [
        (
            report.report_sha256,
            item.l2,
            item.validation_log_loss,
            item.model_sha256,
            item.optimizer_iterations,
            item.optimizer_objective,
        )
        for item in report.candidates
    ]
    threshold_rows = [
        (
            report.report_sha256,
            "validation",
            format(float(item.threshold), ".1f"),
            item.threshold,
            _canonical_json(item.asdict()),
        )
        for item in report.validation_trials
    ] + [
        (
            report.report_sha256,
            "test",
            "selected",
            report.selected_threshold,
            _canonical_json(report.test_metrics.asdict()),
        )
    ]
    split_rows: list[tuple[object, ...]] = []
    for partition, groups in (
        ("train", report.split.train_groups),
        ("purged", report.split.purged_groups),
        ("validation", report.split.validation_groups),
        ("test", report.split.test_groups),
    ):
        split_rows.extend(
            (report.report_sha256, partition, ordinal, value)
            for ordinal, value in enumerate(groups)
        )
    selected_rows = validation_actions + test_actions
    equity_rows = validation_equity + test_equity
    market_rows = validation_markets + test_markets
    existing = connection.execute(
        "SELECT * FROM polymarket_ridge_report WHERE report_sha256 = ?",
        [report.report_sha256],
    ).fetchone()
    tables = (
        ("polymarket_ridge_candidate", candidate_rows, "l2"),
        ("polymarket_ridge_threshold_trial", threshold_rows, "partition, trial_key"),
        ("polymarket_ridge_split_group", split_rows, "partition, ordinal"),
        ("polymarket_ridge_selected_action", selected_rows, "partition, sequence"),
        ("polymarket_ridge_equity", equity_rows, "partition, sequence"),
        ("polymarket_ridge_market_pnl", market_rows, "partition, condition_id"),
    )
    if existing is not None:
        if tuple(existing) != report_row:
            raise ValueError("stored Polymarket ridge report is inconsistent")
        for table, expected, ordering in tables:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE report_sha256 = ? ORDER BY {ordering}",
                [report.report_sha256],
            ).fetchall()
            if [tuple(row) for row in rows] != sorted(
                expected,
                key=lambda row: tuple(
                    row[index] for index in range(1, min(3, len(row)))
                ),
            ):
                raise ValueError(f"stored {table} rows are inconsistent")
        status = "existing"
    else:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "INSERT INTO polymarket_ridge_report VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                report_row,
            )
            for table, rows, _ordering in tables:
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
    return PolymarketRidgeMaterialization(
        report_sha256=report.report_sha256,
        status=status,
        selected_validation_action_count=len(validation_actions),
        selected_test_action_count=len(test_actions),
    )


__all__ = [
    "POLYMARKET_RIDGE_CONTRACT_SHA256",
    "POLYMARKET_RIDGE_FIT_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_RIDGE_L2_GRID",
    "POLYMARKET_RIDGE_MODEL_SCHEMA_VERSION",
    "POLYMARKET_RIDGE_REPORT_SCHEMA_VERSION",
    "POLYMARKET_RIDGE_THRESHOLD_GRID",
    "PolymarketPolicyMetrics",
    "PolymarketPolicyEvaluation",
    "PolymarketRidgeFitAlreadyComplete",
    "PolymarketRidgeCandidate",
    "PolymarketRidgeDataset",
    "PolymarketRidgeDatasetIdentity",
    "PolymarketRidgeFitClaim",
    "PolymarketRidgeModel",
    "PolymarketRidgeMaterialization",
    "PolymarketRidgeObservation",
    "PolymarketRidgeReport",
    "PolymarketRidgeSplit",
    "build_polymarket_ridge_dataset",
    "begin_polymarket_ridge_fit",
    "complete_polymarket_ridge_fit",
    "evaluate_polymarket_policy",
    "fit_and_evaluate_polymarket_ridge",
    "fail_polymarket_ridge_fit",
    "fit_polymarket_ridge_model",
    "load_polymarket_ridge_dataset",
    "load_polymarket_ridge_dataset_identity",
    "load_polymarket_ridge_evidence",
    "load_polymarket_ridge_materialization",
    "load_polymarket_ridge_report",
    "materialize_polymarket_ridge_report",
    "polymarket_selected_policy_tables",
    "split_polymarket_ridge_dataset",
]

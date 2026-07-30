"""Frozen one-use held-out evaluation for Polymarket Round 17."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .polymarket import PolymarketPublicClient
from .polymarket_round14_contract import load_round14_contract
from .polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
    Round17CampaignTestAccess,
    iter_round17_campaign_test_conditions,
    materialize_round17_campaign_test_index,
)
from .polymarket_round17_development_operator import (
    validate_round17_development_result,
)
from .polymarket_round17_cohort import (
    Round17CohortManifest,
    Round17CohortPlan,
    Round17ConditionLabel,
    load_round17_cohort_plan,
)
from .polymarket_round17_dataset import PolymarketRound17ConditionDataset
from .polymarket_round17_economic import evaluate_round17_economic_holdout
from .polymarket_round17_model import (
    Round17CandidateInferenceSession,
    Round17DevelopmentPanel,
    predict_round17_candidate,
    score_round17_predictions,
)
from .polymarket_round17_one_use import (
    POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256,
    POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE,
    POLYMARKET_ROUND17_TEST_START_MS,
    Round17OneUseClaimStore,
    Round17TestAccessClaim,
    stage_round17_one_use_claim,
)
from .polymarket_round17_outcomes import (
    build_round17_calibrated_decision_probability,
    materialize_round17_condition_economic_outcomes,
)
from .polymarket_round17_resolution import (
    Round17TestResolutionAcquisition,
    acquire_round17_test_resolutions,
    round17_test_resolution_acquisition_from_mapping,
)
from .polymarket_round17_uncertainty import (
    Round17CalibratedEnvelope,
    apply_round17_probability_calibration_rows,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND17_ENDPOINT_HOLDOUT_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-endpoint-holdout-v1"
)
POLYMARKET_ROUND17_FINAL_RESULT_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-one-use-result-v1"
)
POLYMARKET_ROUND17_TEST_TARGET_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-test-target-manifest-v1"
)
POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS = (
    "round17-market-prior-raw-control",
    "round17-chainlink-structural-control",
    "round17-market-prior-calibrated-control",
)
POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SEED = 17_017
POLYMARKET_ROUND17_ENDPOINT_LOWER_QUANTILE = 0.025
POLYMARKET_ROUND17_ENDPOINT_MINIMUM_CONDITIONS = 1_800
POLYMARKET_ROUND17_ENDPOINT_MINIMUM_CALENDAR_DAYS = 7
POLYMARKET_ROUND17_ENDPOINT_MINIMUM_NON_TIED_FINAL_PREDICTIONS = 300
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DAY_MS = 86_400_000
_PROBABILITY_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class Round17TestTargetManifest:
    plan_sha256: str
    claim_sha256: str
    test_access_sha256: str
    cohort_manifest_sha256: str
    labels: tuple[Round17ConditionLabel, ...]
    test_dataset_sha256: str
    target_manifest_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_TEST_TARGET_MANIFEST_SCHEMA_VERSION,
            "evaluation_contract_sha256": (
                POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
            ),
            "plan_sha256": self.plan_sha256,
            "claim_sha256": self.claim_sha256,
            "test_access_sha256": self.test_access_sha256,
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "labels": [
                {**item.identity_payload(), "label_sha256": item.label_sha256}
                for item in self.labels
            ],
            "test_dataset_sha256": self.test_dataset_sha256,
            "test_features_accessed": True,
            "test_targets_accessed": True,
            "model_scores_accessed": False,
            "execution_scores_accessed": False,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "target_manifest_sha256": self.target_manifest_sha256,
        }

    def validated(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
    ) -> Round17TestTargetManifest:
        selected_cohort = cohort.validated(plan)
        references = {item.condition_id: item for item in selected_cohort.conditions}
        for label in self.labels:
            label.validated()
        expected_dataset = _canonical_sha256(
            {
                "schema_version": "polymarket-round17-test-dataset-v1",
                "claim_sha256": self.claim_sha256,
                "test_access_sha256": self.test_access_sha256,
                "cohort_dataset_sha256": selected_cohort.cohort_dataset_sha256,
                "labels": [
                    [item.condition_id, item.label_sha256] for item in self.labels
                ],
            }
        )
        if (
            self.plan_sha256 != plan.plan_sha256
            or _SHA256.fullmatch(self.claim_sha256) is None
            or _SHA256.fullmatch(self.test_access_sha256) is None
            or any(item.role != "test" for item in selected_cohort.conditions)
            or self.cohort_manifest_sha256 != selected_cohort.manifest_sha256
            or self.labels
            != tuple(
                sorted(
                    self.labels,
                    key=lambda item: (item.event_start_ms, item.condition_id),
                )
            )
            or len({item.condition_id for item in self.labels}) != len(self.labels)
            or set(references) != {item.condition_id for item in self.labels}
            or any(
                label.source_run_id != references[label.condition_id].source_run_id
                or label.event_start_ms
                != references[label.condition_id].event_start_ms
                for label in self.labels
            )
            or self.test_dataset_sha256 != expected_dataset
            or self.target_manifest_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 test target manifest integrity differs")
        return self


def build_round17_test_target_manifest(
    plan: Round17CohortPlan,
    cohort: Round17CohortManifest,
    labels: tuple[Round17ConditionLabel, ...],
    *,
    claim_sha256: str,
    test_access_sha256: str,
) -> Round17TestTargetManifest:
    selected_cohort = cohort.validated(plan)
    selected_labels = tuple(
        sorted(
            (item.validated() for item in labels),
            key=lambda item: (item.event_start_ms, item.condition_id),
        )
    )
    claim = str(claim_sha256 or "").strip().lower()
    access = str(test_access_sha256 or "").strip().lower()
    dataset_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-round17-test-dataset-v1",
            "claim_sha256": claim,
            "test_access_sha256": access,
            "cohort_dataset_sha256": selected_cohort.cohort_dataset_sha256,
            "labels": [
                [item.condition_id, item.label_sha256] for item in selected_labels
            ],
        }
    )
    provisional = Round17TestTargetManifest(
        plan_sha256=plan.plan_sha256,
        claim_sha256=claim,
        test_access_sha256=access,
        cohort_manifest_sha256=selected_cohort.manifest_sha256,
        labels=selected_labels,
        test_dataset_sha256=dataset_sha256,
    )
    return replace(
        provisional,
        target_manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan, selected_cohort)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _condition_groups(
    condition_ids: np.ndarray,
) -> tuple[tuple[str, int, int], ...]:
    selected = np.asarray(condition_ids, dtype=object)
    if selected.ndim != 1 or len(selected) < 1:
        raise ValueError("Round 17 endpoint condition identities differ")
    boundaries = np.flatnonzero(selected[1:] != selected[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    ends = np.concatenate(
        (boundaries, np.asarray([len(selected)], dtype=np.int64))
    )
    groups = tuple(
        (str(selected[start]), int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    )
    if len({condition_id for condition_id, _start, _end in groups}) != len(groups):
        raise ValueError("Round 17 endpoint conditions are not contiguous")
    return groups


def _condition_vectors(
    panel: Round17DevelopmentPanel,
    predictions: np.ndarray,
) -> dict[str, object]:
    probability = np.clip(
        np.asarray(predictions, dtype=np.float64),
        _PROBABILITY_FLOOR,
        1.0 - _PROBABILITY_FLOOR,
    )
    if probability.shape != panel.labels.shape or not np.all(np.isfinite(probability)):
        raise ValueError("Round 17 endpoint predictions differ")
    groups = _condition_groups(panel.condition_ids)
    log_loss = np.empty(len(groups), dtype=np.float64)
    brier = np.empty(len(groups), dtype=np.float64)
    final_probability = np.empty(len(groups), dtype=np.float64)
    final_label = np.empty(len(groups), dtype=np.float64)
    conditions: list[str] = []
    for index, (condition_id, start, end) in enumerate(groups):
        labels = panel.labels[start:end]
        values = probability[start:end]
        log_loss[index] = float(
            np.mean(
                -(
                    labels * np.log(values)
                    + (1.0 - labels) * np.log1p(-values)
                )
            )
        )
        brier[index] = float(np.mean(np.square(values - labels)))
        final_probability[index] = values[-1]
        final_label[index] = labels[-1]
        conditions.append(condition_id)
    return {
        "condition_ids": tuple(conditions),
        "log_loss": log_loss,
        "brier": brier,
        "final_probability": final_probability,
        "final_label": final_label,
    }


def _weighted_prediction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    condition_losses: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(labels, dtype=np.float64)
    probability = np.clip(
        np.asarray(predictions, dtype=np.float64),
        _PROBABILITY_FLOOR,
        1.0 - _PROBABILITY_FLOOR,
    )
    selected_weights = np.asarray(weights, dtype=np.float64)
    if (
        target.shape != probability.shape
        or target.shape != selected_weights.shape
        or not np.all(np.isin(target, (0.0, 1.0)))
        or not np.all(np.isfinite(probability))
        or not np.all(np.isfinite(selected_weights))
        or np.any(selected_weights <= 0.0)
        or not math.isclose(
            float(np.sum(selected_weights)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Round 17 weighted endpoint metrics differ")
    linear_input = logit(probability)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * linear_input
        predicted = expit(linear)
        residual = predicted - target
        return (
            float(
                np.sum(
                    selected_weights
                    * (np.logaddexp(0.0, linear) - target * linear)
                )
            ),
            np.asarray(
                [
                    np.sum(selected_weights * residual),
                    np.sum(selected_weights * residual * linear_input),
                ],
                dtype=np.float64,
            ),
        )

    fitted = minimize(
        objective,
        np.asarray([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-20.0, 20.0), (-20.0, 20.0)),
        options={"maxiter": 256, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError("Round 17 endpoint calibration metric fit failed")
    decisions = probability >= 0.5
    positive = target >= 0.5
    negative = ~positive
    true_positive_rate = (
        0.0
        if not np.any(positive)
        else float(
            np.sum(selected_weights[positive] * decisions[positive])
            / np.sum(selected_weights[positive])
        )
    )
    true_negative_rate = (
        0.0
        if not np.any(negative)
        else float(
            np.sum(selected_weights[negative] * (~decisions[negative]))
            / np.sum(selected_weights[negative])
        )
    )
    true_positive = float(np.sum(selected_weights * decisions * positive))
    true_negative = float(np.sum(selected_weights * (~decisions) * negative))
    false_positive = float(np.sum(selected_weights * decisions * negative))
    false_negative = float(np.sum(selected_weights * (~decisions) * positive))
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        if not np.any(selected):
            continue
        bin_weight = float(np.sum(selected_weights[selected]))
        predicted_mean = float(
            np.sum(selected_weights[selected] * probability[selected]) / bin_weight
        )
        observed_mean = float(
            np.sum(selected_weights[selected] * target[selected]) / bin_weight
        )
        expected_calibration_error += bin_weight * abs(
            predicted_mean - observed_mean
        )
    losses = np.asarray(condition_losses, dtype=np.float64)
    return {
        "condition_count": len(losses),
        "row_count": len(target),
        "condition_balanced_log_loss": float(np.mean(losses)),
        "condition_log_loss_standard_error": (
            0.0
            if len(losses) < 2
            else float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        ),
        "condition_balanced_brier": float(
            np.sum(selected_weights * np.square(probability - target))
        ),
        "calibration_intercept": float(fitted.x[0]),
        "calibration_slope": float(fitted.x[1]),
        "expected_calibration_error": expected_calibration_error,
        "condition_weighted_balanced_accuracy": (
            true_positive_rate + true_negative_rate
        )
        / 2.0,
        "condition_weighted_matthews_correlation": (
            0.0
            if denominator <= 0.0
            else (true_positive * true_negative - false_positive * false_negative)
            / denominator
        ),
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.float64) >= 0.5
    decision = np.asarray(predictions, dtype=np.float64) >= 0.5
    positive = target
    negative = ~target
    if not np.any(positive) or not np.any(negative):
        return 0.0
    return float(
        (
            np.mean(decision[positive], dtype=np.float64)
            + np.mean(~decision[negative], dtype=np.float64)
        )
        / 2.0
    )


def _bootstrap_endpoint(
    candidate: Mapping[str, object],
    controls: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    condition_ids = candidate["condition_ids"]
    candidate_labels = np.asarray(candidate["final_label"], dtype=np.float64)
    candidate_probability = np.asarray(
        candidate["final_probability"],
        dtype=np.float64,
    )
    condition_count = len(candidate_labels)
    if not isinstance(condition_ids, tuple) or condition_count < 1:
        raise ValueError("Round 17 endpoint bootstrap panel differs")
    generator = np.random.default_rng(POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SEED)
    candidate_balanced_accuracy = np.empty(
        POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )
    improvement = {
        control_id: {
            "log_loss": np.empty(
                POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES,
                dtype=np.float64,
            ),
            "brier": np.empty(
                POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES,
                dtype=np.float64,
            ),
            "balanced_accuracy": np.empty(
                POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES,
                dtype=np.float64,
            ),
        }
        for control_id in controls
    }
    candidate_log_loss = np.asarray(candidate["log_loss"], dtype=np.float64)
    candidate_brier = np.asarray(candidate["brier"], dtype=np.float64)
    for sample_index in range(POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES):
        selected = generator.integers(0, condition_count, size=condition_count)
        selected_labels = candidate_labels[selected]
        selected_candidate_probability = candidate_probability[selected]
        candidate_balanced_accuracy[sample_index] = _balanced_accuracy(
            selected_labels,
            selected_candidate_probability,
        )
        for control_id, control in controls.items():
            if control["condition_ids"] != condition_ids:
                raise ValueError("Round 17 endpoint control panel differs")
            improvement[control_id]["log_loss"][sample_index] = float(
                np.mean(
                    np.asarray(control["log_loss"], dtype=np.float64)[selected]
                    - candidate_log_loss[selected]
                )
            )
            improvement[control_id]["brier"][sample_index] = float(
                np.mean(
                    np.asarray(control["brier"], dtype=np.float64)[selected]
                    - candidate_brier[selected]
                )
            )
            improvement[control_id]["balanced_accuracy"][sample_index] = (
                _balanced_accuracy(
                    selected_labels,
                    np.asarray(
                        control["final_probability"],
                        dtype=np.float64,
                    )[selected],
                )
                * -1.0
                + candidate_balanced_accuracy[sample_index]
            )
    quantile = POLYMARKET_ROUND17_ENDPOINT_LOWER_QUANTILE
    return {
        "unit": "condition",
        "samples": POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES,
        "seed": POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SEED,
        "lower_quantile": quantile,
        "candidate_balanced_accuracy_lower_95": float(
            np.quantile(candidate_balanced_accuracy, quantile, method="linear")
        ),
        "control_improvement_lower_95": {
            control_id: {
                metric: float(np.quantile(values, quantile, method="linear"))
                for metric, values in metrics.items()
            }
            for control_id, metrics in improvement.items()
        },
    }


def _validated_endpoint_parent(
    development_result: Mapping[str, object],
    claim: Round17TestAccessClaim,
    test_access_sha256: str,
) -> tuple[dict[str, object], Mapping[str, object]]:
    result = validate_round17_development_result(development_result)
    selected_claim = claim.validated()
    access = str(test_access_sha256 or "").strip().lower()
    artifacts = result["artifacts"]
    parents = result["parents"]
    if (
        result["status"] != "development_accepted"
        or result["result_sha256"] != selected_claim.development_result_sha256
        or not isinstance(artifacts, Mapping)
        or not isinstance(parents, Mapping)
        or _SHA256.fullmatch(access) is None
        or parents["model_pretest_sha256"] != selected_claim.model_pretest_sha256
        or parents["probability_calibration_sha256"]
        != selected_claim.probability_calibration_sha256
        or parents["economic_pretest_sha256"]
        != selected_claim.economic_pretest_sha256
    ):
        raise ValueError("Round 17 endpoint holdout parent differs")
    return result, artifacts


def _endpoint_models(
    artifacts: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, Mapping[str, object]],
    Mapping[str, object],
]:
    model_pretest = artifacts.get("model_pretest")
    if not isinstance(model_pretest, Mapping):
        raise ValueError("Round 17 endpoint model artifact differs")
    selected_candidate = model_pretest.get("selected_candidate")
    control_records = model_pretest.get("controls")
    if (
        not isinstance(selected_candidate, Mapping)
        or not isinstance(control_records, list)
        or any(not isinstance(item, Mapping) for item in control_records)
    ):
        raise ValueError("Round 17 endpoint models differ")
    control_models = {
        str(item["candidate_id"]): item["model"]
        for item in control_records
        if isinstance(item, Mapping)
    }
    if (
        tuple(control_models) != POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
        or any(not isinstance(model, Mapping) for model in control_models.values())
    ):
        raise ValueError("Round 17 endpoint controls differ")
    return selected_candidate, control_models, model_pretest


def _build_endpoint_holdout_payload(
    *,
    result: Mapping[str, object],
    model_pretest: Mapping[str, object],
    claim: Round17TestAccessClaim,
    access: str,
    test_dataset_sha256: str,
    test_target_manifest_sha256: str,
    condition_ids: tuple[str, ...],
    event_starts: np.ndarray,
    labels: np.ndarray,
    row_count: int,
    candidate_vectors: Mapping[str, object],
    control_vectors: Mapping[str, Mapping[str, object]],
    candidate_metrics: Mapping[str, object],
    control_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    calendar_days = len(set(int(value) // _DAY_MS for value in event_starts))
    bootstrap = _bootstrap_endpoint(candidate_vectors, control_vectors)
    strongest_control_id = str(model_pretest.get("strongest_control_id") or "")
    lower = bootstrap["control_improvement_lower_95"]
    if (
        strongest_control_id not in control_metrics
        or not isinstance(lower, Mapping)
    ):
        raise ValueError("Round 17 strongest endpoint control differs")
    non_tied = int(
        np.count_nonzero(
            np.abs(
                np.asarray(
                    candidate_vectors["final_probability"],
                    dtype=np.float64,
                )
                - 0.5
            )
            > 1e-12
        )
    )
    gates = {
        "minimum_resolved_conditions": (
            len(condition_ids) >= POLYMARKET_ROUND17_ENDPOINT_MINIMUM_CONDITIONS
        ),
        "minimum_calendar_days": (
            calendar_days >= POLYMARKET_ROUND17_ENDPOINT_MINIMUM_CALENDAR_DAYS
        ),
        "both_outcomes": set(labels.tolist()) == {0.0, 1.0},
        "minimum_non_tied_final_condition_predictions": (
            non_tied
            >= POLYMARKET_ROUND17_ENDPOINT_MINIMUM_NON_TIED_FINAL_PREDICTIONS
        ),
        "log_loss_strictly_below_every_control": all(
            float(candidate_metrics["condition_balanced_log_loss"])
            < float(metrics["condition_balanced_log_loss"])
            for metrics in control_metrics.values()
        ),
        "brier_strictly_below_every_control": all(
            float(candidate_metrics["condition_balanced_brier"])
            < float(metrics["condition_balanced_brier"])
            for metrics in control_metrics.values()
        ),
        "paired_log_loss_improvement_lower_95_positive_for_every_control": all(
            isinstance(lower[control_id], Mapping)
            and float(lower[control_id]["log_loss"]) > 0.0
            for control_id in POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
        ),
        "paired_brier_improvement_lower_95_positive_for_every_control": all(
            isinstance(lower[control_id], Mapping)
            and float(lower[control_id]["brier"]) > 0.0
            for control_id in POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
        ),
        "balanced_accuracy_lower_95_above_random": (
            float(bootstrap["candidate_balanced_accuracy_lower_95"]) > 0.5
        ),
        "balanced_accuracy_difference_lower_95_not_below_strongest_control": (
            isinstance(lower[strongest_control_id], Mapping)
            and float(lower[strongest_control_id]["balanced_accuracy"]) >= 0.0
        ),
    }
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND17_ENDPOINT_HOLDOUT_SCHEMA_VERSION,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
        ),
        "claim_sha256": claim.claim_sha256,
        "test_access_sha256": access,
        "development_result_sha256": result["result_sha256"],
        "model_pretest_sha256": model_pretest["pretest_sha256"],
        "test_dataset_sha256": test_dataset_sha256,
        "test_target_manifest_sha256": test_target_manifest_sha256,
        "condition_count": len(condition_ids),
        "row_count": int(row_count),
        "calendar_day_count": calendar_days,
        "up_condition_count": int(np.count_nonzero(labels == 1.0)),
        "down_condition_count": int(np.count_nonzero(labels == 0.0)),
        "non_tied_final_condition_prediction_count": non_tied,
        "first_event_start_ms": int(np.min(event_starts)),
        "last_event_start_ms": int(np.max(event_starts)),
        "selected_candidate_id": model_pretest["selected_candidate_id"],
        "strongest_control_id": strongest_control_id,
        "control_ids": list(POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS),
        "candidate_metrics": dict(candidate_metrics),
        "control_metrics": {
            control_id: dict(metrics)
            for control_id, metrics in control_metrics.items()
        },
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "endpoint_accepted": all(gates.values()),
        "test_features_accessed": True,
        "test_targets_accessed": True,
        "model_refit": False,
        "calibration_refit": False,
        "automatic_promotion": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in candidate_metrics.values()
    ):
        raise ValueError("Round 17 endpoint metrics are non-finite")
    payload["endpoint_holdout_sha256"] = _canonical_sha256(payload)
    return payload


def evaluate_round17_endpoint_holdout(
    panel: Round17DevelopmentPanel,
    *,
    development_result: Mapping[str, object],
    claim: Round17TestAccessClaim,
    test_access_sha256: str,
) -> dict[str, object]:
    """Evaluate the immutable candidate once without test-time fitting."""

    result, artifacts = _validated_endpoint_parent(
        development_result,
        claim,
        test_access_sha256,
    )
    selected_panel = panel.validate()
    model_pretest = artifacts["model_pretest"]
    access = str(test_access_sha256 or "").strip().lower()
    if selected_panel.role != "test" or not isinstance(model_pretest, Mapping):
        raise ValueError("Round 17 endpoint holdout panel differs")
    partition = model_pretest.get("dataset_and_partition")
    roles = partition.get("roles") if isinstance(partition, Mapping) else None
    if not isinstance(roles, Mapping):
        raise ValueError("Round 17 endpoint development partition differs")
    development_conditions = {
        str(condition_id)
        for role in roles.values()
        if isinstance(role, Mapping)
        for condition_id in role["condition_ids"]  # type: ignore[index]
    }
    groups = _condition_groups(selected_panel.condition_ids)
    condition_ids = tuple(condition_id for condition_id, _start, _end in groups)
    event_starts = np.asarray(
        [selected_panel.event_start_ms[start] for _condition, start, _end in groups],
        dtype=np.int64,
    )
    labels = np.asarray(
        [selected_panel.labels[end - 1] for _condition, _start, end in groups],
        dtype=np.float64,
    )
    if (
        development_conditions.intersection(condition_ids)
        or int(np.min(event_starts)) < POLYMARKET_ROUND17_TEST_START_MS
        or int(np.max(event_starts)) >= POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE
    ):
        raise ValueError("Round 17 endpoint holdout support differs")
    selected_candidate, control_models, model_pretest = _endpoint_models(artifacts)
    candidate_prediction = predict_round17_candidate(
        selected_candidate,
        selected_panel,
    )
    control_predictions = {
        control_id: predict_round17_candidate(control, selected_panel)
        for control_id, control in control_models.items()
    }
    candidate_vectors = _condition_vectors(selected_panel, candidate_prediction)
    control_vectors = {
        control_id: _condition_vectors(selected_panel, prediction)
        for control_id, prediction in control_predictions.items()
    }
    candidate_metrics = score_round17_predictions(
        selected_panel,
        candidate_prediction,
    )
    control_metrics = {
        control_id: score_round17_predictions(selected_panel, prediction)
        for control_id, prediction in control_predictions.items()
    }
    return _build_endpoint_holdout_payload(
        result=result,
        model_pretest=model_pretest,
        claim=claim,
        access=access,
        test_dataset_sha256=selected_panel.dataset_sha256,
        test_target_manifest_sha256=selected_panel.target_manifest_sha256,
        condition_ids=condition_ids,
        event_starts=event_starts,
        labels=labels,
        row_count=len(selected_panel.labels),
        candidate_vectors=candidate_vectors,
        control_vectors=control_vectors,
        candidate_metrics=candidate_metrics,
        control_metrics=control_metrics,
    )


class Round17EndpointHoldoutAccumulator:
    """Score test conditions incrementally without retaining feature matrices."""

    def __init__(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
        target_manifest: Round17TestTargetManifest,
        *,
        development_result: Mapping[str, object],
        claim: Round17TestAccessClaim,
        test_access_sha256: str,
    ) -> None:
        self.result, self.artifacts = _validated_endpoint_parent(
            development_result,
            claim,
            test_access_sha256,
        )
        self.claim = claim.validated()
        self.access = str(test_access_sha256 or "").strip().lower()
        self.cohort = cohort.validated(plan)
        self.target = target_manifest.validated(plan, self.cohort)
        if (
            self.target.claim_sha256 != self.claim.claim_sha256
            or self.target.test_access_sha256 != self.access
        ):
            raise ValueError("Round 17 streaming endpoint target parent differs")
        selected_candidate, controls, self.model_pretest = _endpoint_models(
            self.artifacts
        )
        partition = self.model_pretest.get("dataset_and_partition")
        roles = partition.get("roles") if isinstance(partition, Mapping) else None
        if not isinstance(roles, Mapping):
            raise ValueError("Round 17 streaming endpoint partition differs")
        development_conditions = {
            str(condition_id)
            for role in roles.values()
            if isinstance(role, Mapping)
            for condition_id in role["condition_ids"]  # type: ignore[index]
        }
        self.references = self.cohort.conditions
        if (
            not self.references
            or any(item.role != "test" for item in self.references)
            or development_conditions.intersection(
                item.condition_id for item in self.references
            )
        ):
            raise ValueError("Round 17 streaming endpoint conditions differ")
        self.label_by_condition = {
            item.condition_id: item for item in self.target.labels
        }
        self.models: dict[str, Mapping[str, object]] = {
            "candidate": selected_candidate,
            **dict(controls),
        }
        self.sessions = {
            model_id: Round17CandidateInferenceSession(model)
            for model_id, model in self.models.items()
        }
        self.prediction_batches: dict[str, list[np.ndarray]] = {
            model_id: [] for model_id in self.models
        }
        self.row_counts: list[int] = []
        self.consumed = 0
        self.finished = False

    def append(
        self,
        dataset: PolymarketRound17ConditionDataset,
        *,
        calibrated: tuple[Round17CalibratedEnvelope, ...] | None = None,
    ) -> None:
        if self.finished:
            raise RuntimeError("Round 17 streaming endpoint accumulator is finished")
        if self.consumed >= len(self.references):
            raise ValueError("Round 17 streaming endpoint has extra conditions")
        source = dataset.validated()
        reference = self.references[self.consumed]
        label = self.label_by_condition.get(reference.condition_id)
        if (
            label is None
            or source.condition_id != reference.condition_id
            or source.run_id != reference.source_run_id
            or source.event_start_ms != reference.event_start_ms
            or source.event_end_ms != reference.event_end_ms
            or source.admission_sha256 != reference.admission_sha256
            or source.dataset_sha256 != reference.condition_dataset_sha256
            or len(source.rows) != reference.feature_row_count
            or label.source_run_id != reference.source_run_id
            or label.event_start_ms != reference.event_start_ms
        ):
            raise ValueError("Round 17 streaming endpoint evidence differs")
        calibrated_rows = None if calibrated is None else tuple(calibrated)
        if calibrated_rows is not None and (
            len(calibrated_rows) != len(source.rows)
            or any(
                item.source_role != "test"
                or item.test_access_sha256 != self.access
                or item.dataset_sha256 != self.target.test_dataset_sha256
                or item.model_pretest_sha256 != self.claim.model_pretest_sha256
                or item.condition_id != source.condition_id
                or item.decision_time_ms != row.decision_time_ms
                or item.feature_input_sha256 != row.input_sha256
                or item.feature_values_sha256 != row.values_sha256
                for item, row in zip(calibrated_rows, source.rows, strict=True)
            )
        ):
            raise ValueError("Round 17 streaming calibration evidence differs")
        for model_id, session in self.sessions.items():
            prediction = (
                np.asarray(
                    [item.raw_probability_up for item in calibrated_rows],
                    dtype=np.float64,
                )
                if model_id == "candidate" and calibrated_rows is not None
                else np.asarray(
                    session.predict_rows(source.rows),
                    dtype=np.float64,
                )
            )
            if (
                prediction.shape != (len(source.rows),)
                or not np.all(np.isfinite(prediction))
                or np.any((prediction <= 0.0) | (prediction >= 1.0))
            ):
                raise ValueError("Round 17 streaming endpoint prediction differs")
            self.prediction_batches[model_id].append(prediction)
        self.row_counts.append(len(source.rows))
        self.consumed += 1

    def finish(self) -> dict[str, object]:
        if self.finished:
            raise RuntimeError("Round 17 streaming endpoint accumulator is finished")
        if self.consumed != len(self.references):
            raise ValueError("Round 17 streaming endpoint conditions are incomplete")
        self.finished = True
        condition_count = len(self.references)
        condition_ids = tuple(item.condition_id for item in self.references)
        event_starts = np.asarray(
            [item.event_start_ms for item in self.references],
            dtype=np.int64,
        )
        condition_labels = np.asarray(
            [
                self.label_by_condition[item.condition_id].target_up
                for item in self.references
            ],
            dtype=np.float64,
        )
        row_labels = np.concatenate(
            [
                np.full(count, label, dtype=np.float64)
                for count, label in zip(
                    self.row_counts,
                    condition_labels,
                    strict=True,
                )
            ]
        )
        row_weights = np.concatenate(
            [
                np.full(
                    count,
                    1.0 / (condition_count * count),
                    dtype=np.float64,
                )
                for count in self.row_counts
            ]
        )
        vectors: dict[str, dict[str, object]] = {}
        metrics: dict[str, dict[str, float | int]] = {}
        for model_id, batches in self.prediction_batches.items():
            condition_log_loss = np.asarray(
                [
                    float(
                        np.mean(
                            -(
                                label * np.log(batch)
                                + (1.0 - label) * np.log1p(-batch)
                            )
                        )
                    )
                    for batch, label in zip(
                        batches,
                        condition_labels,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            condition_brier = np.asarray(
                [
                    float(np.mean(np.square(batch - label)))
                    for batch, label in zip(
                        batches,
                        condition_labels,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            prediction = np.concatenate(batches)
            vectors[model_id] = {
                "condition_ids": condition_ids,
                "log_loss": condition_log_loss,
                "brier": condition_brier,
                "final_probability": np.asarray(
                    [batch[-1] for batch in batches],
                    dtype=np.float64,
                ),
                "final_label": condition_labels,
            }
            metrics[model_id] = _weighted_prediction_metrics(
                row_labels,
                prediction,
                row_weights,
                condition_log_loss,
            )
        candidate_vectors = vectors.pop("candidate")
        candidate_metrics = metrics.pop("candidate")
        return _build_endpoint_holdout_payload(
            result=self.result,
            model_pretest=self.model_pretest,
            claim=self.claim,
            access=self.access,
            test_dataset_sha256=self.target.test_dataset_sha256,
            test_target_manifest_sha256=self.target.target_manifest_sha256,
            condition_ids=condition_ids,
            event_starts=event_starts,
            labels=condition_labels,
            row_count=len(row_labels),
            candidate_vectors=candidate_vectors,
            control_vectors=vectors,
            candidate_metrics=candidate_metrics,
            control_metrics=metrics,
        )


def build_round17_one_use_result(
    *,
    claim: Round17TestAccessClaim,
    test_access_sha256: str,
    test_index_sha256: str,
    test_resolution_acquisition_sha256: str,
    test_target_manifest_sha256: str,
    endpoint_holdout: Mapping[str, object],
    economic_holdout: Mapping[str, object],
) -> dict[str, object]:
    """Bind predictive and economic gates into the terminal one-use result."""

    selected_claim = claim.validated()
    endpoint = dict(endpoint_holdout)
    economic = dict(economic_holdout)
    endpoint_claimed = str(endpoint.pop("endpoint_holdout_sha256", "")).lower()
    economic_claimed = str(economic.pop("economic_holdout_sha256", "")).lower()
    access = str(test_access_sha256 or "").strip().lower()
    test_index = str(test_index_sha256 or "").strip().lower()
    resolution = str(test_resolution_acquisition_sha256 or "").strip().lower()
    target = str(test_target_manifest_sha256 or "").strip().lower()
    if (
        _SHA256.fullmatch(access) is None
        or _SHA256.fullmatch(test_index) is None
        or _SHA256.fullmatch(resolution) is None
        or _SHA256.fullmatch(target) is None
        or endpoint_claimed != _canonical_sha256(endpoint)
        or economic_claimed != _canonical_sha256(economic)
        or endpoint.get("claim_sha256") != selected_claim.claim_sha256
        or endpoint.get("test_access_sha256") != access
        or endpoint.get("test_target_manifest_sha256") != target
        or economic.get("test_access_sha256") != access
    ):
        raise ValueError("Round 17 one-use result parents differ")
    accepted = bool(
        endpoint.get("endpoint_accepted") is True
        and economic.get("economic_accepted") is True
    )
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND17_FINAL_RESULT_SCHEMA_VERSION,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
        ),
        "claim_sha256": selected_claim.claim_sha256,
        "test_access_sha256": access,
        "test_index_sha256": test_index,
        "test_resolution_acquisition_sha256": resolution,
        "test_target_manifest_sha256": target,
        "endpoint_holdout_sha256": endpoint_claimed,
        "economic_holdout_sha256": economic_claimed,
        "endpoint_holdout": {
            **endpoint,
            "endpoint_holdout_sha256": endpoint_claimed,
        },
        "economic_holdout": {
            **economic,
            "economic_holdout_sha256": economic_claimed,
        },
        "status": "heldout_accepted" if accepted else "heldout_rejected",
        "heldout_accepted": accepted,
        "test_access_consumed": True,
        "test_features_accessed": True,
        "test_targets_accessed": True,
        "test_execution_accessed": True,
        "return_to_development": False,
        "automatic_promotion": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 evaluation JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 evaluation JSON contains {value}")


def _load_json_object(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= maximum_bytes
    ):
        raise ValueError(f"{label} is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(value)


def _write_durable_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    write_bytes_atomic(path, payload)


@dataclass(frozen=True, slots=True)
class Round17OneUseEvaluationConfig:
    repository: Path
    repository_commit_sha: str
    contract_path: Path
    development_result_path: Path
    risk_contract_path: Path
    claim_store_path: Path
    resolution_checkpoint_path: Path
    output_path: Path
    campaign: Round17CampaignOperatorConfig

    def validated(self) -> Round17OneUseEvaluationConfig:
        repository = self.repository.resolve()
        files = (
            self.contract_path,
            self.development_result_path,
            self.risk_contract_path,
        )
        outputs = (
            self.claim_store_path.resolve(),
            self.resolution_checkpoint_path.resolve(),
            self.output_path.resolve(),
        )
        capture_database = self.campaign.database_path.resolve()
        capture_state = self.campaign.state_root.resolve()
        if (
            not repository.is_dir()
            or repository.is_symlink()
            or _SHA256.fullmatch(str(self.repository_commit_sha).lower()) is None
            and not re.fullmatch(
                r"^[0-9a-f]{40}$",
                str(self.repository_commit_sha).lower(),
            )
            or any(path.is_symlink() or not path.is_file() for path in files)
            or len(set(outputs)) != len(outputs)
            or any(
                output == capture_database
                or output == capture_state
                or capture_state in output.parents
                or output in capture_state.parents
                for output in outputs
            )
        ):
            raise ValueError("Round 17 one-use evaluation configuration is invalid")
        self.campaign.validate()
        return self


def _emit(
    progress: Callable[[Mapping[str, object]], None] | None,
    phase: str,
    **details: object,
) -> None:
    if progress is not None:
        progress({"phase": phase, **details})


def run_round17_one_use_evaluation(
    config: Round17OneUseEvaluationConfig,
    *,
    client: PolymarketPublicClient | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run or resume the exact one-use test claim with no execution authority."""

    selected = config.validated()
    if selected.claim_store_path.is_file():
        with Round17OneUseClaimStore(selected.claim_store_path) as store:
            persisted = store.snapshot()
            if persisted["status"] == "completed":
                result = persisted["result"]
                if not isinstance(result, Mapping):
                    raise ValueError(
                        "Round 17 completed one-use result is unavailable"
                    )
                _write_durable_json(selected.output_path, result)
                return dict(result)
            if persisted["status"] == "failed":
                raise RuntimeError("Round 17 one-use evaluation is terminally failed")
    development_result = validate_round17_development_result(
        _load_json_object(
            selected.development_result_path,
            label="Round 17 development result",
        )
    )
    claim = stage_round17_one_use_claim(
        store_path=selected.claim_store_path,
        repository=selected.repository,
        repository_commit_sha=selected.repository_commit_sha,
        contract_path=selected.contract_path,
        development_result=development_result,
        campaign=selected.campaign,
    )
    with Round17OneUseClaimStore(selected.claim_store_path) as store:
        snapshot = store.snapshot()
        if snapshot["status"] == "completed":
            result = snapshot["result"]
            if not isinstance(result, Mapping):
                raise ValueError("Round 17 completed one-use result is unavailable")
            _write_durable_json(selected.output_path, result)
            return dict(result)
        if snapshot["status"] == "failed":
            raise RuntimeError("Round 17 one-use evaluation is terminally failed")
        access_sha256 = store.consume_test_access(claim)
    access = Round17CampaignTestAccess(
        claim_sha256=claim.claim_sha256,
        test_access_sha256=access_sha256,
        readiness_sha256=claim.campaign_readiness_sha256,
    )
    try:
        _emit(progress, "test_index_start")
        test_index = materialize_round17_campaign_test_index(
            selected.campaign,
            access,
            progress=progress,
        )
        _emit(
            progress,
            "test_index_ready",
            condition_count=len(test_index.cohort_manifest.conditions),
            test_index_sha256=test_index.index_sha256,
        )
        existing_resolution: Round17TestResolutionAcquisition | None = None
        cohort_plan = load_round17_cohort_plan(selected.campaign.cohort_plan_path)
        if selected.resolution_checkpoint_path.is_file():
            existing_resolution = (
                round17_test_resolution_acquisition_from_mapping(
                    _load_json_object(
                        selected.resolution_checkpoint_path,
                        label="Round 17 test resolution checkpoint",
                    ),
                    plan=cohort_plan,
                    cohort=test_index.cohort_manifest,
                    markets=test_index.market_mapping(),
                )
            )
        acquisition = acquire_round17_test_resolutions(
            cohort_plan,
            test_index.cohort_manifest,
            test_index.market_mapping(),
            claim_sha256=claim.claim_sha256,
            test_access_sha256=access_sha256,
            existing=existing_resolution,
            client=client,
            progress=progress,
            checkpoint=lambda value: _write_durable_json(
                selected.resolution_checkpoint_path,
                value.asdict(),
            ),
        )
        if not acquisition.complete:
            with Round17OneUseClaimStore(selected.claim_store_path) as store:
                store.mark_resolution_pending(
                    claim,
                    pending_condition_count=len(acquisition.pending_condition_ids),
                )
            return {
                "schema_version": POLYMARKET_ROUND17_FINAL_RESULT_SCHEMA_VERSION,
                "status": "resolution_pending",
                "claim_sha256": claim.claim_sha256,
                "test_access_sha256": access_sha256,
                "test_index_sha256": test_index.index_sha256,
                "test_resolution_acquisition_sha256": (
                    acquisition.acquisition_sha256
                ),
                "pending_condition_count": len(acquisition.pending_condition_ids),
                "return_to_development": False,
                "automatic_promotion": False,
                "profitability_claim": False,
                "paper_trading_authority": False,
                "live_trading_authority": False,
            }
        labels = acquisition.labels(
            cohort_plan,
            test_index.cohort_manifest,
            test_index.market_mapping(),
        )
        target = build_round17_test_target_manifest(
            cohort_plan,
            test_index.cohort_manifest,
            labels,
            claim_sha256=claim.claim_sha256,
            test_access_sha256=access_sha256,
        )
        artifacts = development_result["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise ValueError("Round 17 development artifacts differ")
        model_pretest = artifacts["model_pretest"]
        calibration = artifacts["probability_calibration"]
        economic_pretest = artifacts["economic_pretest"]
        if not all(
            isinstance(item, Mapping)
            for item in (model_pretest, calibration, economic_pretest)
        ):
            raise ValueError("Round 17 development model parents differ")
        selected_policies_raw = economic_pretest["selected_by_profile"]  # type: ignore[index]
        if not isinstance(selected_policies_raw, Mapping):
            raise ValueError("Round 17 selected economic policies differ")
        selected_policies = {
            profile: (
                str(policy["path"]),
                Decimal(str(policy["minimum_edge_quote_per_share"])),
            )
            for profile, policy in selected_policies_raw.items()
            if isinstance(policy, Mapping)
        }
        program = load_round14_contract(selected.risk_contract_path)
        risk_capital = Decimal(str(development_result["risk_capital_quote"]))
        accumulator = Round17EndpointHoldoutAccumulator(
            cohort_plan,
            test_index.cohort_manifest,
            target,
            development_result=development_result,
            claim=claim,
            test_access_sha256=access_sha256,
        )
        candidate_session = accumulator.sessions["candidate"]
        resolution_by_condition = {
            item.condition_id: item for item in acquisition.observations
        }
        economic_outcomes = []
        for count, materialized in enumerate(
            iter_round17_campaign_test_conditions(selected.campaign, access),
            start=1,
        ):
            calibrated = apply_round17_probability_calibration_rows(
                calibration,  # type: ignore[arg-type]
                model_pretest,  # type: ignore[arg-type]
                materialized.dataset.rows,
                dataset_sha256=target.test_dataset_sha256,
                event_start_ms=materialized.market.event_start_ms,
                source_role="test",
                test_access_sha256=access_sha256,
                inference_session=candidate_session,
            )
            probabilities = tuple(
                build_round17_calibrated_decision_probability(row, envelope)
                for row, envelope in zip(
                    materialized.dataset.rows,
                    calibrated,
                    strict=True,
                )
            )
            economic_outcomes.extend(
                materialize_round17_condition_economic_outcomes(
                    market=materialized.market,
                    dataset=materialized.dataset,
                    predictions=probabilities,
                    books=materialized.books,
                    resolution=resolution_by_condition[
                        materialized.market.condition_id
                    ].resolution_evidence(),
                    program=program,
                    risk_capital_quote=risk_capital,
                    source_partition="test",
                    selected_policy_by_profile=selected_policies,
                )
            )
            accumulator.append(materialized.dataset, calibrated=calibrated)
            if count == 1 or count % 25 == 0:
                _emit(
                    progress,
                    "test_replay",
                    completed_conditions=count,
                    total_conditions=len(test_index.cohort_manifest.conditions),
                    last_event_start_ms=materialized.market.event_start_ms,
                )
        endpoint = accumulator.finish()
        economic = evaluate_round17_economic_holdout(
            economic_outcomes,
            program,
            economic_pretest=economic_pretest,  # type: ignore[arg-type]
            model_pretest=model_pretest,  # type: ignore[arg-type]
            probability_calibration=calibration,  # type: ignore[arg-type]
            test_access_sha256=access_sha256,
        )
        result = build_round17_one_use_result(
            claim=claim,
            test_access_sha256=access_sha256,
            test_index_sha256=test_index.index_sha256,
            test_resolution_acquisition_sha256=acquisition.acquisition_sha256,
            test_target_manifest_sha256=target.target_manifest_sha256,
            endpoint_holdout=endpoint,
            economic_holdout=economic,
        )
        with Round17OneUseClaimStore(selected.claim_store_path) as store:
            completed = store.complete(claim, result)
        _write_durable_json(selected.output_path, completed)
        _emit(
            progress,
            "one_use_complete",
            status=completed["status"],
            result_sha256=completed["result_sha256"],
        )
        return completed
    except (AssertionError, ArithmeticError, KeyError, TypeError, ValueError):
        with Round17OneUseClaimStore(selected.claim_store_path) as store:
            store.fail(claim, reason="deterministic_evaluation_integrity_failure")
        raise


__all__ = [
    "POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SAMPLES",
    "POLYMARKET_ROUND17_ENDPOINT_BOOTSTRAP_SEED",
    "POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS",
    "POLYMARKET_ROUND17_ENDPOINT_HOLDOUT_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_FINAL_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_TEST_TARGET_MANIFEST_SCHEMA_VERSION",
    "Round17EndpointHoldoutAccumulator",
    "Round17OneUseEvaluationConfig",
    "Round17TestTargetManifest",
    "build_round17_test_target_manifest",
    "build_round17_one_use_result",
    "evaluate_round17_endpoint_holdout",
    "run_round17_one_use_evaluation",
]

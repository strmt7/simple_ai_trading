"""Chronological matched selection for Polymarket Round 29."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from .polymarket_round27_model import (
    POLYMARKET_ROUND27_CORRECTION_SCALES,
    round27_stationary_bootstrap_mean_interval,
)
from .polymarket_round28_contract_binding import (
    validate_loaded_round27_model_contract,
)
from .polymarket_round29_model import (
    POLYMARKET_ROUND29_L2_PENALTIES,
    POLYMARKET_ROUND29_MATCHED_ABLATION_SCHEMA_VERSION,
    Round29FeatureView,
    Round29L2OffsetModel,
    Round29ModelSample,
    Round29PairName,
    Round29Partition,
    fit_round29_l2_offset,
    round29_matched_ablation_report,
    round29_model_from_payload,
    round29_probability_metrics,
)
from .polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256,
)


POLYMARKET_ROUND29_BOOTSTRAP_SCHEMA_VERSION = (
    "polymarket-round29-paired-condition-bootstrap-v1"
)
POLYMARKET_ROUND29_PAIR_SELECTION_SCHEMA_VERSION = (
    "polymarket-round29-pair-selection-v1"
)
POLYMARKET_ROUND29_SELECTION_SCHEMA_VERSION = "polymarket-round29-selection-v1"
POLYMARKET_ROUND29_PREVALIDATION_EMBARGO_MS = 600_000
POLYMARKET_ROUND29_VALIDATION_FOLDS = 5
_VIEWS: tuple[Round29FeatureView, ...] = (
    "round27_base",
    "round29_settlement_augmented",
    "round28_bbo_augmented",
    "round29_bbo_settlement_augmented",
)
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
    "orders_submitted": False,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def _validated_claim(
    value: Mapping[str, object],
    *,
    hash_field: str,
    label: str,
) -> dict[str, object]:
    body = dict(value)
    claimed = str(body.pop(hash_field, "")).lower()
    if len(claimed) != 64 or claimed != _canonical_sha256(body):
        raise ValueError(f"{label} hash differs")
    return {**body, hash_field: claimed}


def _sha256(value: object, *, label: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 29 {label} SHA-256 differs")
    return selected


def _prior_probability(partition: Round29Partition) -> NDArray[np.float64]:
    offsets = np.asarray(partition.offsets, dtype=np.float64)
    output = np.empty_like(offsets)
    positive = offsets >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-offsets[positive]))
    exponential = np.exp(offsets[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return np.clip(output, 1e-7, 1.0 - 1e-7)


def _validated_probability(
    partition: Round29Partition,
    values: Sequence[float] | NDArray[np.float64],
    *,
    label: str,
) -> NDArray[np.float64]:
    selected = np.asarray(values, dtype=np.float64)
    if (
        selected.shape != partition.targets.shape
        or not np.all(np.isfinite(selected))
        or np.any((selected <= 0.0) | (selected >= 1.0))
    ):
        raise ValueError(f"Round 29 {label} probability population differs")
    return selected


def _weighted_log_loss(
    partition: Round29Partition,
    probability: Sequence[float] | NDArray[np.float64],
) -> float:
    selected = _validated_probability(partition, probability, label="log-loss")
    weights = partition.weights / np.sum(partition.weights)
    losses = -(
        partition.targets * np.log(selected)
        + (1.0 - partition.targets) * np.log1p(-selected)
    )
    return float(np.sum(weights * losses))


def scale_round29_probability_model(
    model: Round29L2OffsetModel,
    correction_scale: float,
) -> Round29L2OffsetModel:
    selected_scale = float(correction_scale)
    if selected_scale not in POLYMARKET_ROUND27_CORRECTION_SCALES:
        raise ValueError("Round 29 correction scale differs")
    payload = model.asdict()
    payload["correction_scale"] = selected_scale
    body = dict(payload)
    body.pop("model_sha256")
    payload["model_sha256"] = _canonical_sha256(body)
    return round29_model_from_payload(payload)


def round29_chronological_condition_folds(
    partition: Round29Partition,
    *,
    fold_count: int = POLYMARKET_ROUND29_VALIDATION_FOLDS,
    embargo_ms: int = POLYMARKET_ROUND29_PREVALIDATION_EMBARGO_MS,
) -> tuple[tuple[Round29Partition, Round29Partition], ...]:
    """Build expanding whole-condition folds for training-time selection."""

    if fold_count < 2 or embargo_ms < 0 or embargo_ms % 300_000:
        raise ValueError("Round 29 chronological validation controls differ")
    event_start_by_condition: dict[str, int] = {}
    for sample in partition.samples:
        prior = event_start_by_condition.setdefault(
            sample.condition_id,
            sample.event_start_ms,
        )
        if prior != sample.event_start_ms:
            raise ValueError("Round 29 condition start time differs")
    ordered_conditions = tuple(
        condition
        for condition, _event_start_ms in sorted(
            event_start_by_condition.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    block_count = fold_count + 1
    if len(ordered_conditions) < block_count:
        raise ValueError("Round 29 chronological population is insufficient")
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
    folds: list[tuple[Round29Partition, Round29Partition]] = []
    for validation_index in range(1, block_count):
        validation_conditions = frozenset(blocks[validation_index])
        validation_start_ms = min(
            event_start_by_condition[condition] for condition in validation_conditions
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
            raise ValueError("Round 29 embargoed chronological fold is empty")
        train = Round29Partition.from_samples(train_samples, role=partition.role)
        validation = Round29Partition.from_samples(
            validation_samples,
            role=partition.role,
        )
        if max(
            sample.event_start_ms + 300_000 for sample in train.samples
        ) + embargo_ms > min(sample.event_start_ms for sample in validation.samples):
            raise RuntimeError("Round 29 chronological embargo differs")
        folds.append((train, validation))
    if len(folds) != fold_count:
        raise RuntimeError("Round 29 chronological fold count differs")
    return tuple(folds)


def select_round29_l2_penalty(
    partition: Round29Partition,
    *,
    feature_view: Round29FeatureView,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    folds = round29_chronological_condition_folds(partition)
    scores: dict[str, float] = {}
    for penalty_index, penalty in enumerate(POLYMARKET_ROUND29_L2_PENALTIES, start=1):
        weighted_losses: list[float] = []
        condition_count = 0
        for train, validation in folds:
            model = fit_round29_l2_offset(
                train,
                feature_view=feature_view,
                penalty=penalty,
            )
            loss = _weighted_log_loss(
                validation,
                model.predict(validation.features(feature_view), validation.offsets),
            )
            validation_conditions = int(np.unique(validation.conditions).size)
            weighted_losses.append(loss * validation_conditions)
            condition_count += validation_conditions
        scores[str(penalty)] = math.fsum(weighted_losses) / condition_count
        if progress is not None:
            progress(
                "l2_penalty_evaluated",
                {
                    "feature_view": feature_view,
                    "penalty": penalty,
                    "penalty_index": penalty_index,
                    "penalty_count": len(POLYMARKET_ROUND29_L2_PENALTIES),
                    "chronological_log_loss": scores[str(penalty)],
                },
            )
    selected = min(
        POLYMARKET_ROUND29_L2_PENALTIES,
        key=lambda value: (scores[str(value)], value),
    )
    return selected, scores


def select_round29_correction_scale(
    model: Round29L2OffsetModel,
    partition: Round29Partition,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for scale in POLYMARKET_ROUND27_CORRECTION_SCALES:
        candidate = scale_round29_probability_model(model, scale)
        scores[str(scale)] = _weighted_log_loss(
            partition,
            candidate.predict(
                partition.features(candidate.feature_view),
                partition.offsets,
            ),
        )
    return (
        min(
            POLYMARKET_ROUND27_CORRECTION_SCALES,
            key=lambda value: (scores[str(value)], value),
        ),
        scores,
    )


def paired_round29_condition_bootstrap(
    partition: Round29Partition,
    baseline: Sequence[float] | NDArray[np.float64],
    candidate: Sequence[float] | NDArray[np.float64],
    *,
    draws: int = 5_000,
    seed: int = 2_902,
) -> dict[str, object]:
    baseline_probability = _validated_probability(
        partition,
        baseline,
        label="bootstrap baseline",
    )
    candidate_probability = _validated_probability(
        partition,
        candidate,
        label="bootstrap candidate",
    )
    event_start_by_condition: dict[str, int] = {}
    for sample in partition.samples:
        prior = event_start_by_condition.setdefault(
            sample.condition_id,
            sample.event_start_ms,
        )
        if prior != sample.event_start_ms:
            raise ValueError("Round 29 bootstrap condition time differs")
    ordered_conditions = tuple(
        condition
        for condition, _event_start_ms in sorted(
            event_start_by_condition.items(),
            key=lambda item: (item[1], item[0]),
        )
    )
    log_loss_values: list[float] = []
    brier_values: list[float] = []
    for condition in ordered_conditions:
        selected = partition.conditions == condition
        target = partition.targets[selected]
        weights = partition.weights[selected]
        weights = weights / np.sum(weights)
        baseline_loss = -(
            target * np.log(baseline_probability[selected])
            + (1.0 - target) * np.log1p(-baseline_probability[selected])
        )
        candidate_loss = -(
            target * np.log(candidate_probability[selected])
            + (1.0 - target) * np.log1p(-candidate_probability[selected])
        )
        log_loss_values.append(
            float(np.sum(weights * (candidate_loss - baseline_loss)))
        )
        baseline_brier = (baseline_probability[selected] - target) ** 2
        candidate_brier = (candidate_probability[selected] - target) ** 2
        brier_values.append(float(np.sum(weights * (candidate_brier - baseline_brier))))
    log_loss = np.asarray(log_loss_values, dtype=np.float64)
    brier = np.asarray(brier_values, dtype=np.float64)
    if log_loss.size < 20 or draws < 1_000:
        raise ValueError("Round 29 bootstrap population is insufficient")
    log_loss_interval = round27_stationary_bootstrap_mean_interval(
        log_loss,
        draws=draws,
        seed=seed,
    )
    brier_interval = round27_stationary_bootstrap_mean_interval(
        brier,
        draws=draws,
        seed=seed + 97_409,
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_BOOTSTRAP_SCHEMA_VERSION,
        "condition_count": int(log_loss.size),
        "draws": draws,
        "seed": seed,
        "mean_candidate_minus_baseline_log_loss": float(np.mean(log_loss)),
        "mean_candidate_minus_baseline_brier_score": float(np.mean(brier)),
        "log_loss": log_loss_interval,
        "brier_score": brier_interval,
    }
    body["bootstrap_sha256"] = _canonical_sha256(body)
    return body


def round29_pair_selection_report(
    partition: Round29Partition,
    *,
    pair_name: Round29PairName,
    base_model: Round29L2OffsetModel,
    augmented_model: Round29L2OffsetModel,
    prediction_evaluation: Mapping[str, object],
    training_detail: Mapping[str, object],
) -> dict[str, object]:
    matched = round29_matched_ablation_report(
        partition,
        pair_name=pair_name,
        base_model=base_model,
        augmented_model=augmented_model,
    )
    base_view, augmented_view = _PAIR_VIEWS[pair_name]
    base_probability = base_model.predict(
        partition.features(base_view),
        partition.offsets,
    )
    augmented_probability = augmented_model.predict(
        partition.features(augmented_view),
        partition.offsets,
    )
    prior_probability = _prior_probability(partition)
    prior_metrics = round29_probability_metrics(partition, prior_probability).asdict()
    base_metrics = matched["base_metrics"]
    augmented_metrics = matched["augmented_metrics"]
    if not isinstance(base_metrics, Mapping) or not isinstance(
        augmented_metrics,
        Mapping,
    ):
        raise RuntimeError("Round 29 matched metrics differ")
    draws = prediction_evaluation.get("bootstrap_draws")
    if type(draws) is not int:
        raise ValueError("Round 29 bootstrap contract differs")
    uplift = paired_round29_condition_bootstrap(
        partition,
        base_probability,
        augmented_probability,
        draws=draws,
        seed=2_902 if pair_name == "diagnostic" else 2_929,
    )
    versus_prior = paired_round29_condition_bootstrap(
        partition,
        prior_probability,
        augmented_probability,
        draws=draws,
        seed=2_999 if pair_name == "diagnostic" else 2_989,
    )
    try:
        checks = {
            "augmented_log_loss_better_than_base": float(augmented_metrics["log_loss"])
            < float(base_metrics["log_loss"]),
            "augmented_brier_better_than_base": float(augmented_metrics["brier_score"])
            < float(base_metrics["brier_score"]),
            "paired_log_loss_uplift_confidence_gate_met": float(
                uplift["log_loss"]["ci95_upper"]  # type: ignore[index]
            )
            < 0.0,
            "paired_brier_uplift_confidence_gate_met": float(
                uplift["brier_score"]["ci95_upper"]  # type: ignore[index]
            )
            < 0.0,
            "augmented_log_loss_better_than_market_prior": float(
                augmented_metrics["log_loss"]
            )
            < float(prior_metrics["log_loss"]),
            "augmented_brier_better_than_market_prior": float(
                augmented_metrics["brier_score"]
            )
            < float(prior_metrics["brier_score"]),
            "augmented_prior_log_loss_confidence_gate_met": float(
                versus_prior["log_loss"]["ci95_upper"]  # type: ignore[index]
            )
            < 0.0,
            "augmented_prior_brier_confidence_gate_met": float(
                versus_prior["brier_score"]["ci95_upper"]  # type: ignore[index]
            )
            < 0.0,
            "balanced_accuracy_floor_met": float(augmented_metrics["balanced_accuracy"])
            >= float(prediction_evaluation["balanced_accuracy_floor"]),
            "calibration_not_materially_worse_than_prior": float(
                augmented_metrics["expected_calibration_error"]
            )
            <= float(prior_metrics["expected_calibration_error"])
            + float(prediction_evaluation["calibration_ece_maximum_degradation"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 29 prediction evaluation contract differs") from exc
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_PAIR_SELECTION_SCHEMA_VERSION,
        "pair_name": pair_name,
        "model_family": base_model.model_name,
        "training": dict(training_detail),
        "matched_ablation": matched,
        "market_prior_metrics": prior_metrics,
        "paired_augmented_minus_base": uplift,
        "paired_augmented_minus_market_prior": versus_prior,
        "gate_checks": checks,
        "probability_gate_passed": all(checks.values()),
        "promotion_controlling_pair": pair_name == "primary",
        "economic_gate_required": pair_name == "primary",
        **_AUTHORITY,
    }
    body["pair_report_sha256"] = _canonical_sha256(body)
    return body


@dataclass(frozen=True, slots=True)
class Round29SelectedPair:
    model_family: str
    base_model: Round29L2OffsetModel
    augmented_model: Round29L2OffsetModel


def _validate_selection_inputs(
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    selected_contract = validate_loaded_round27_model_contract(contract)
    selected_preregistration = _validated_claim(
        preregistration,
        hash_field="preregistration_sha256",
        label="Round 29 preregistration",
    )
    feature = selected_preregistration.get("feature_contract")
    lineage = selected_preregistration.get("data_lineage")
    if (
        selected_preregistration.get("schema_version")
        != "polymarket-round29-settlement-state-matched-ablation-preregistration-v1"
        or not isinstance(feature, Mapping)
        or feature.get("settlement_augmented_feature_names_sha256")
        != POLYMARKET_ROUND29_BASE_FEATURE_NAMES_SHA256
        or feature.get("combined_feature_names_sha256")
        != POLYMARKET_ROUND29_COMBINED_FEATURE_NAMES_SHA256
        or not isinstance(lineage, Mapping)
        or lineage.get("round27_model_contract_sha256")
        != selected_contract["contract_sha256"]
    ):
        raise ValueError("Round 29 selection lineage differs")
    return selected_contract, selected_preregistration


def _minimum_conditions(
    partition: Round29Partition,
    minimum_population: Mapping[str, object],
) -> None:
    minimum = minimum_population.get(f"{partition.role}_conditions")
    condition_count = int(np.unique(partition.conditions).size)
    if type(minimum) is not int or condition_count < int(minimum):
        raise ValueError(f"Round 29 {partition.role} population is insufficient")


def _validate_partition_counts(
    value: object,
    *,
    minimum_population: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "train",
        "calibration",
        "selection",
    }:
        raise ValueError("Round 29 selection partition counts differ")
    for role, raw_counts in value.items():
        minimum = minimum_population.get(f"{role}_conditions")
        if (
            not isinstance(raw_counts, Mapping)
            or set(raw_counts) != {"condition_count", "row_count"}
            or type(minimum) is not int
            or type(raw_counts.get("condition_count")) is not int
            or type(raw_counts.get("row_count")) is not int
            or int(raw_counts["condition_count"]) < int(minimum)
            or int(raw_counts["row_count"]) < int(raw_counts["condition_count"])
        ):
            raise ValueError("Round 29 selection partition counts differ")


def _fit_view(
    *,
    train: Round29Partition,
    calibration: Round29Partition,
    view: Round29FeatureView,
    progress: Callable[[str, Mapping[str, object]], None] | None,
) -> tuple[Round29L2OffsetModel, dict[str, object]]:
    penalty, penalty_scores = select_round29_l2_penalty(
        train,
        feature_view=view,
        progress=progress,
    )
    unscaled = fit_round29_l2_offset(
        train,
        feature_view=view,
        penalty=penalty,
    )
    correction_scale, scale_scores = select_round29_correction_scale(
        unscaled,
        calibration,
    )
    model = scale_round29_probability_model(unscaled, correction_scale)
    detail: dict[str, object] = {
        "selected_l2_penalty": penalty,
        "chronological_log_loss": penalty_scores,
        "selected_correction_scale": correction_scale,
        "calibration_scale_log_loss": scale_scores,
    }
    return model, detail


def run_round29_matched_selection(
    *,
    samples: Sequence[Round29ModelSample],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest_sha256: str,
    claim_writer: Callable[[Mapping[str, object]], str],
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, object], Round29SelectedPair | None]:
    """Fit both pairs and persist the primary decision before economics."""

    selected_contract, selected_preregistration = _validate_selection_inputs(
        contract=contract,
        preregistration=preregistration,
    )
    minimum = selected_contract.get("minimum_population")
    evaluation = selected_contract.get("prediction_evaluation")
    if not isinstance(minimum, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("Round 29 inherited selection contract differs")
    train = Round29Partition.from_samples(samples, role="train")
    calibration = Round29Partition.from_samples(samples, role="calibration")
    selection = Round29Partition.from_samples(samples, role="selection")
    for partition in (train, calibration, selection):
        _minimum_conditions(partition, minimum)
    models: dict[Round29FeatureView, Round29L2OffsetModel] = {}
    details: dict[str, object] = {
        "pre_validation_embargo_ms": POLYMARKET_ROUND29_PREVALIDATION_EMBARGO_MS,
        "validation_fold_count": POLYMARKET_ROUND29_VALIDATION_FOLDS,
        "views": {},
    }
    for view in _VIEWS:
        model, detail = _fit_view(
            train=train,
            calibration=calibration,
            view=view,
            progress=progress,
        )
        models[view] = model
        details["views"][view] = detail  # type: ignore[index]
    reports = [
        round29_pair_selection_report(
            selection,
            pair_name=pair_name,
            base_model=models[base_view],
            augmented_model=models[augmented_view],
            prediction_evaluation=evaluation,
            training_detail=details,
        )
        for pair_name, (base_view, augmented_view) in _PAIR_VIEWS.items()
    ]
    primary = next(report for report in reports if report["pair_name"] == "primary")
    selected_pair = (
        Round29SelectedPair(
            model_family="l2_offset_logistic",
            base_model=models["round28_bbo_augmented"],
            augmented_model=models["round29_bbo_settlement_augmented"],
        )
        if primary["probability_gate_passed"] is True
        else None
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_SELECTION_SCHEMA_VERSION,
        "selection_input_manifest_sha256": _sha256(
            selection_input_manifest_sha256,
            label="selection input manifest",
        ),
        "round27_model_contract_sha256": selected_contract["contract_sha256"],
        "round27_model_implementation_amendment_sha256": selected_contract[
            "model_implementation_amendment_sha256"
        ],
        "round29_preregistration_sha256": selected_preregistration[
            "preregistration_sha256"
        ],
        "status": (
            "primary_probability_candidate_selected"
            if selected_pair is not None
            else "no_primary_candidate_passed_probability_gates"
        ),
        "selected_model_family": (
            selected_pair.model_family if selected_pair is not None else None
        ),
        "candidate_pairs": reports,
        "partition_counts": {
            partition.role: {
                "condition_count": int(np.unique(partition.conditions).size),
                "row_count": int(partition.targets.size),
            }
            for partition in (train, calibration, selection)
        },
        "diagnostic_pair_can_promote": False,
        "sealed_partition_accessed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["claim_sha256"] = _canonical_sha256(body)
    if claim_writer(body) != body["claim_sha256"]:
        raise ValueError("Round 29 selection claim writer differs")
    return body, selected_pair


def _validated_pair_report(value: Mapping[str, object]) -> dict[str, object]:
    report = _validated_claim(
        value,
        hash_field="pair_report_sha256",
        label="Round 29 pair report",
    )
    matched = report.get("matched_ablation")
    uplift = report.get("paired_augmented_minus_base")
    versus_prior = report.get("paired_augmented_minus_market_prior")
    checks = report.get("gate_checks")
    if (
        report.get("schema_version") != POLYMARKET_ROUND29_PAIR_SELECTION_SCHEMA_VERSION
        or report.get("pair_name") not in _PAIR_VIEWS
        or not isinstance(matched, Mapping)
        or not isinstance(uplift, Mapping)
        or not isinstance(versus_prior, Mapping)
        or not isinstance(checks, Mapping)
        or not checks
        or any(type(item) is not bool for item in checks.values())
        or report.get("probability_gate_passed") is not all(checks.values())
        or report.get("promotion_controlling_pair")
        is not (report.get("pair_name") == "primary")
        or report.get("economic_gate_required")
        is not (report.get("pair_name") == "primary")
        or any(report.get(key) is not expected for key, expected in _AUTHORITY.items())
    ):
        raise ValueError("Round 29 pair report differs")
    selected_matched = _validated_claim(
        matched,
        hash_field="report_sha256",
        label="Round 29 matched ablation report",
    )
    selected_uplift = _validated_claim(
        uplift,
        hash_field="bootstrap_sha256",
        label="Round 29 pair uplift bootstrap",
    )
    selected_prior = _validated_claim(
        versus_prior,
        hash_field="bootstrap_sha256",
        label="Round 29 prior bootstrap",
    )
    pair_name = report["pair_name"]
    base_view, augmented_view = _PAIR_VIEWS[pair_name]  # type: ignore[index]
    base_model = selected_matched.get("base_model")
    augmented_model = selected_matched.get("augmented_model")
    if (
        selected_matched.get("schema_version")
        != POLYMARKET_ROUND29_MATCHED_ABLATION_SCHEMA_VERSION
        or selected_matched.get("pair_name") != pair_name
        or selected_uplift.get("schema_version")
        != POLYMARKET_ROUND29_BOOTSTRAP_SCHEMA_VERSION
        or selected_prior.get("schema_version")
        != POLYMARKET_ROUND29_BOOTSTRAP_SCHEMA_VERSION
        or not isinstance(base_model, Mapping)
        or not isinstance(augmented_model, Mapping)
        or base_model.get("feature_view") != base_view
        or augmented_model.get("feature_view") != augmented_view
    ):
        raise ValueError("Round 29 pair evidence schema differs")
    return report


def load_round29_selected_pair(
    claim: Mapping[str, object],
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest_sha256: str,
) -> Round29SelectedPair | None:
    selected_contract, selected_preregistration = _validate_selection_inputs(
        contract=contract,
        preregistration=preregistration,
    )
    selected_claim = _validated_claim(
        claim,
        hash_field="claim_sha256",
        label="Round 29 selection claim",
    )
    reports = selected_claim.get("candidate_pairs")
    minimum = selected_contract.get("minimum_population")
    expected_input_manifest_sha256 = _sha256(
        selection_input_manifest_sha256,
        label="selection input manifest",
    )
    if (
        selected_claim.get("schema_version")
        != POLYMARKET_ROUND29_SELECTION_SCHEMA_VERSION
        or selected_claim.get("selection_input_manifest_sha256")
        != expected_input_manifest_sha256
        or selected_claim.get("round27_model_contract_sha256")
        != selected_contract["contract_sha256"]
        or selected_claim.get("round27_model_implementation_amendment_sha256")
        != selected_contract["model_implementation_amendment_sha256"]
        or selected_claim.get("round29_preregistration_sha256")
        != selected_preregistration["preregistration_sha256"]
        or not isinstance(reports, list)
        or not isinstance(minimum, Mapping)
        or selected_claim.get("diagnostic_pair_can_promote") is not False
        or selected_claim.get("sealed_partition_accessed") is not False
        or selected_claim.get("economic_metrics_computed") is not False
        or selected_claim.get("ai_assist_evaluated") is not False
        or any(
            selected_claim.get(key) is not value for key, value in _AUTHORITY.items()
        )
    ):
        raise ValueError("Round 29 selection claim differs")
    _validate_partition_counts(
        selected_claim.get("partition_counts"),
        minimum_population=minimum,
    )
    validated_reports = tuple(
        _validated_pair_report(report)
        for report in reports
        if isinstance(report, Mapping)
    )
    if len(validated_reports) != 2 or {
        report["pair_name"] for report in validated_reports
    } != {"diagnostic", "primary"}:
        raise ValueError("Round 29 candidate pair population differs")
    primary = next(
        report for report in validated_reports if report["pair_name"] == "primary"
    )
    if primary["probability_gate_passed"] is not True:
        if (
            selected_claim.get("status")
            != "no_primary_candidate_passed_probability_gates"
            or selected_claim.get("selected_model_family") is not None
        ):
            raise ValueError("Round 29 empty selection claim differs")
        return None
    if (
        selected_claim.get("status") != "primary_probability_candidate_selected"
        or selected_claim.get("selected_model_family") != "l2_offset_logistic"
    ):
        raise ValueError("Round 29 selected candidate differs")
    matched = primary.get("matched_ablation")
    if not isinstance(matched, Mapping):
        raise ValueError("Round 29 selected ablation differs")
    base = matched.get("base_model")
    augmented = matched.get("augmented_model")
    if not isinstance(base, Mapping) or not isinstance(augmented, Mapping):
        raise ValueError("Round 29 selected model payload differs")
    base_model = round29_model_from_payload(base)
    augmented_model = round29_model_from_payload(augmented)
    if (
        base_model.feature_view != "round28_bbo_augmented"
        or augmented_model.feature_view != "round29_bbo_settlement_augmented"
    ):
        raise ValueError("Round 29 selected model identity differs")
    return Round29SelectedPair("l2_offset_logistic", base_model, augmented_model)


__all__ = [
    "POLYMARKET_ROUND29_BOOTSTRAP_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_PAIR_SELECTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_PREVALIDATION_EMBARGO_MS",
    "POLYMARKET_ROUND29_SELECTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_VALIDATION_FOLDS",
    "Round29SelectedPair",
    "load_round29_selected_pair",
    "paired_round29_condition_bootstrap",
    "round29_chronological_condition_folds",
    "round29_pair_selection_report",
    "run_round29_matched_selection",
    "scale_round29_probability_model",
    "select_round29_correction_scale",
    "select_round29_l2_penalty",
]

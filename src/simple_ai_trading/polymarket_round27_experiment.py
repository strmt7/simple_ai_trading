"""Selection-then-sealed coordinator for the Round 27 model experiment."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Callable, Mapping, Sequence

import numpy as np

from .polymarket_round27_model import (
    Round27L2OffsetModel,
    Round27LightGbmOffsetModel,
    Round27ModelSample,
    Round27Partition,
    Round27ProbabilityModel,
    fit_round27_l2_offset,
    fit_round27_lightgbm_offset,
    paired_round27_condition_bootstrap,
    round27_probability_metrics,
    select_round27_correction_scale,
    select_round27_l2_penalty,
)


POLYMARKET_ROUND27_SELECTION_SCHEMA_VERSION = (
    "polymarket-round27-development-selection-v1"
)
POLYMARKET_ROUND27_SEALED_SCHEMA_VERSION = "polymarket-round27-sealed-evaluation-v1"


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


def _prior(partition: Round27Partition) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-partition.offsets))


def _scaled_model(
    model: Round27ProbabilityModel,
    correction_scale: float,
) -> Round27ProbabilityModel:
    if isinstance(model, Round27L2OffsetModel):
        return replace(model, correction_scale=float(correction_scale))
    if isinstance(model, Round27LightGbmOffsetModel):
        return replace(model, correction_scale=float(correction_scale))
    raise TypeError("Round 27 selected model type differs")


def _minimum_conditions(
    partition: Round27Partition,
    minimum_population: Mapping[str, object],
) -> None:
    field = f"{partition.role}_conditions"
    minimum = minimum_population.get(field)
    condition_count = int(np.unique(partition.conditions).size)
    if type(minimum) is not int or condition_count < int(minimum):
        raise ValueError(f"Round 27 {partition.role} population is insufficient")


def _candidate_gate(
    *,
    prior_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    bootstrap: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, bool]:
    evaluation = contract.get("prediction_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Round 27 prediction evaluation contract differs")
    checks = {
        "log_loss_better_than_market_prior": float(candidate_metrics["log_loss"])
        < float(prior_metrics["log_loss"]),
        "brier_better_than_market_prior": float(candidate_metrics["brier_score"])
        < float(prior_metrics["brier_score"]),
        "balanced_accuracy_floor_met": float(candidate_metrics["balanced_accuracy"])
        >= float(evaluation["balanced_accuracy_floor"]),
        "calibration_not_materially_worse": float(
            candidate_metrics["expected_calibration_error"]
        )
        <= float(prior_metrics["expected_calibration_error"])
        + float(evaluation["calibration_ece_maximum_degradation"]),
        "paired_log_loss_confidence_gate_met": float(bootstrap["ci95_upper"]) < 0.0,
    }
    return checks


def _candidate_report(
    *,
    model: Round27ProbabilityModel,
    partition: Round27Partition,
    prior_prediction: np.ndarray,
    contract: Mapping[str, object],
    calibration_scale_scores: Mapping[str, float],
    training_detail: Mapping[str, object],
) -> tuple[dict[str, object], np.ndarray]:
    prediction = model.predict(partition.features, partition.offsets)
    prior_metrics = round27_probability_metrics(partition, prior_prediction).asdict()
    candidate_metrics = round27_probability_metrics(partition, prediction).asdict()
    bootstrap = paired_round27_condition_bootstrap(
        partition,
        prior_prediction,
        prediction,
        draws=int(contract["prediction_evaluation"]["bootstrap_draws"]),
    )
    checks = _candidate_gate(
        prior_metrics=prior_metrics,
        candidate_metrics=candidate_metrics,
        bootstrap=bootstrap,
        contract=contract,
    )
    return (
        {
            "model_name": model.model_name,
            "model": model.asdict(),
            "training": dict(training_detail),
            "calibration_scale_scores": dict(calibration_scale_scores),
            "market_prior_metrics": prior_metrics,
            "candidate_metrics": candidate_metrics,
            "paired_condition_bootstrap": bootstrap,
            "gate_checks": checks,
            "passed": all(checks.values()),
        },
        prediction,
    )


def run_round27_development_selection(
    *,
    samples: Sequence[Round27ModelSample],
    contract: Mapping[str, object],
    claim_writer: Callable[[Mapping[str, object]], str],
    compute_backend: str = "auto",
) -> tuple[dict[str, object], Round27ProbabilityModel | None]:
    """Select a model and persist its claim before sealed data is accepted."""

    minimum = contract.get("minimum_population")
    prediction_evaluation = contract.get("prediction_evaluation")
    if (
        not isinstance(minimum, Mapping)
        or not isinstance(prediction_evaluation, Mapping)
        or contract.get("contract_sha256") is None
    ):
        raise ValueError("Round 27 selection contract differs")
    train = Round27Partition.from_samples(samples, role="train")
    calibration = Round27Partition.from_samples(samples, role="calibration")
    selection = Round27Partition.from_samples(samples, role="selection")
    for partition in (train, calibration, selection):
        _minimum_conditions(partition, minimum)
    prior_prediction = _prior(selection)

    penalty, penalty_scores = select_round27_l2_penalty(train)
    l2_unscaled = fit_round27_l2_offset(train, penalty=penalty)
    l2_scale, l2_scale_scores = select_round27_correction_scale(
        l2_unscaled,
        calibration,
    )
    l2 = _scaled_model(l2_unscaled, l2_scale)
    l2_report, _l2_prediction = _candidate_report(
        model=l2,
        partition=selection,
        prior_prediction=prior_prediction,
        contract=contract,
        calibration_scale_scores=l2_scale_scores,
        training_detail={
            "condition_group_cv_log_loss": penalty_scores,
            "selected_l2_penalty": penalty,
        },
    )

    tree_unscaled = fit_round27_lightgbm_offset(
        train,
        compute_backend=compute_backend,
    )
    tree_scale, tree_scale_scores = select_round27_correction_scale(
        tree_unscaled,
        calibration,
    )
    tree = _scaled_model(tree_unscaled, tree_scale)
    tree_report, _tree_prediction = _candidate_report(
        model=tree,
        partition=selection,
        prior_prediction=prior_prediction,
        contract=contract,
        calibration_scale_scores=tree_scale_scores,
        training_detail={
            "backend_kind": tree.backend_kind,
            "backend_device": tree.backend_device,
            "fixed_hyperparameters": True,
        },
    )

    reports = (l2_report, tree_report)
    passed = tuple(item for item in reports if item["passed"] is True)
    selected_name = (
        "market_prior"
        if not passed
        else min(
            passed,
            key=lambda item: (
                float(item["candidate_metrics"]["log_loss"]),
                str(item["model_name"]),
            ),
        )["model_name"]
    )
    selected_model: Round27ProbabilityModel | None = {
        "l2_offset_logistic": l2,
        "shallow_lightgbm_offset": tree,
    }.get(str(selected_name))
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_SELECTION_SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "status": (
            "learned_candidate_selected"
            if selected_model is not None
            else "no_learned_candidate_beat_market_prior"
        ),
        "selected_model_name": selected_name,
        "candidates": list(reports),
        "partition_counts": {
            partition.role: {
                "condition_count": int(np.unique(partition.conditions).size),
                "row_count": int(partition.targets.size),
            }
            for partition in (train, calibration, selection)
        },
        "sealed_partition_accessed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    body["claim_sha256"] = _canonical_sha256(body)
    written = claim_writer(body)
    if written != body["claim_sha256"]:
        raise ValueError("Round 27 selection claim writer differs")
    return body, selected_model


def run_round27_sealed_evaluation(
    *,
    samples: Sequence[Round27ModelSample],
    contract: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selected_model: Round27ProbabilityModel | None,
) -> dict[str, object]:
    """Evaluate exactly the frozen selection on the untouched sealed role."""

    claim = dict(selection_claim)
    claimed_sha256 = str(claim.pop("claim_sha256", ""))
    if (
        claimed_sha256 != _canonical_sha256(claim)
        or claim.get("schema_version") != POLYMARKET_ROUND27_SELECTION_SCHEMA_VERSION
        or claim.get("contract_sha256") != contract.get("contract_sha256")
        or claim.get("sealed_partition_accessed") is not False
        or claim.get("selected_model_name")
        != ("market_prior" if selected_model is None else selected_model.model_name)
    ):
        raise ValueError("Round 27 sealed selection claim differs")
    minimum = contract.get("minimum_population")
    if not isinstance(minimum, Mapping):
        raise ValueError("Round 27 sealed contract differs")
    sealed = Round27Partition.from_samples(samples, role="sealed")
    _minimum_conditions(sealed, minimum)
    prior_prediction = _prior(sealed)
    selected_prediction = (
        prior_prediction
        if selected_model is None
        else selected_model.predict(sealed.features, sealed.offsets)
    )
    prior_metrics = round27_probability_metrics(sealed, prior_prediction).asdict()
    selected_metrics = round27_probability_metrics(
        sealed,
        selected_prediction,
    ).asdict()
    bootstrap = paired_round27_condition_bootstrap(
        sealed,
        prior_prediction,
        selected_prediction,
        draws=int(contract["prediction_evaluation"]["bootstrap_draws"]),
    )
    checks = _candidate_gate(
        prior_metrics=prior_metrics,
        candidate_metrics=selected_metrics,
        bootstrap=bootstrap,
        contract=contract,
    )
    if selected_model is None:
        checks = {name: False for name in checks}
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_SEALED_SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "selection_claim_sha256": claimed_sha256,
        "selected_model_name": claim["selected_model_name"],
        "market_prior_metrics": prior_metrics,
        "selected_model_metrics": selected_metrics,
        "paired_condition_bootstrap": bootstrap,
        "gate_checks": checks,
        "prediction_edge_gate_passed": selected_model is not None
        and all(checks.values()),
        "economic_edge_gate_evaluated": False,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "POLYMARKET_ROUND27_SEALED_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_SELECTION_SCHEMA_VERSION",
    "run_round27_development_selection",
    "run_round27_sealed_evaluation",
]

"""Cheap paired development screen for the Round 21 probability basis."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
import simple_ai_trading.polymarket_round21_model as _model
from .polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS,
    Round21DevelopmentPanel,
    round21_development_dataset_identity,
    validate_round21_development_partitions,
)


POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-probability-basis-ablation-design-v1"
)
POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256 = (
    "a54391db7652e0d17d7bae65dd2706ff2bae5c9fbfd99d5d2a9a1e501e0c642a"
)
POLYMARKET_ROUND21_BASIS_ABLATION_PARENT_MODEL_DESIGN_SHA256 = (
    "3b90f4fb0940dccccd5dd673c27257882531685a2c7a1181659924cadc4ffab0"
)
POLYMARKET_ROUND21_BASIS_ABLATION_RESULT_SCHEMA_VERSION = (
    "polymarket-round21-probability-basis-ablation-result-v1"
)
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/round-021-probability-basis-ablation-design-v1.json"
)
_ARMS = ("baseline", "challenger")
_L2_GRID = (0.01, 0.1, 1.0)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESULT_KEYS = {
    "schema_version",
    "design_sha256",
    "model_design_sha256",
    "source_evidence",
    "dataset_and_partition",
    "arms",
    "paired_improvement",
    "basis_accepted",
    "next_action",
    "development_targets_accessed",
    "sealed_test_features_accessed",
    "sealed_test_targets_accessed",
    "economic_evaluation_completed",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
}
_SOURCE_EVIDENCE_KEYS = {
    "publication_manifest_sha256",
    "terminal_transport_manifest_sha256",
}
_DATASET_IDENTITY_KEYS = {
    "role",
    "row_count",
    "condition_count",
    "first_event_start_ms",
    "last_event_start_ms",
    "dataset_sha256",
    "target_manifest_sha256",
    "dataset_design_sha256",
}
_ARM_KEYS = {
    "arm",
    "include_basis",
    "feature_count",
    "feature_names_sha256",
    "regularization_candidates",
    "selected_l2",
    "selected_model",
    "paired_evaluation_metrics",
    "prediction_sha256",
}
_MODEL_KEYS = {
    "candidate_id",
    "family",
    "layer",
    "population_layer",
    "feature_layer",
    "feature_names_sha256",
    "l2",
    "transform",
    "intercept",
    "coefficient",
    "calibration",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {label} is invalid")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Round 21 {label} is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Round 21 {label} is invalid")
    return parsed


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 basis-ablation JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 basis-ablation JSON contains {value}")


def validate_round21_probability_basis_ablation_design(
    value: Mapping[str, object],
) -> dict[str, object]:
    design = dict(value)
    claimed = str(design.pop("design_sha256", "")).strip().lower()
    parents = design.get("parents")
    population = design.get("population")
    arms = design.get("arms")
    program = design.get("candidate_program")
    gate = design.get("paired_gate")
    authority = design.get("authority")
    if (
        claimed != POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256
        or claimed != _canonical_sha256(design)
        or design.get("schema_version")
        != POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SCHEMA_VERSION
        or design.get("round") != 21
        or design.get("status") != "preregistered_during_target_and_model_blind_capture"
        or not isinstance(parents, Mapping)
        or parents.get("round21_contract_sha256") != POLYMARKET_ROUND21_CONTRACT_SHA256
        or parents.get("round21_dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or parents.get("round21_model_design_v6_sha256")
        != POLYMARKET_ROUND21_BASIS_ABLATION_PARENT_MODEL_DESIGN_SHA256
        or not isinstance(population, Mapping)
        or population.get("feature_layer") != "core"
        or population.get("sealed_test_features_or_targets") is not False
        or population.get("minimum_conditions_per_stage")
        != POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
        or not isinstance(arms, Mapping)
        or arms.get("basis_identity") != "model.market_prior_minus_structural_log_odds"
        or arms.get("all_other_inputs_and_rows_identical") is not True
        or not isinstance(program, Mapping)
        or program.get("family") != "logistic_residual"
        or program.get("l2_grid") != ["0.01", "0.1", "1"]
        or program.get("target_access_before_design") is not False
        or program.get("bounded_candidate_fits") != 6
        or not isinstance(gate, Mapping)
        or gate.get("metrics")
        != ["condition_equal_log_loss", "condition_equal_brier_score"]
        or gate.get("inconclusive_is_accepted") is not False
        or gate.get("post_result_threshold_change") is not False
        or not isinstance(authority, Mapping)
        or any(item is not False for item in authority.values())
    ):
        raise ValueError("Round 21 probability-basis ablation design differs")
    return {**design, "design_sha256": claimed}


def load_round21_probability_basis_ablation_design(
    repository: str | Path,
) -> dict[str, object]:
    path = Path(repository).resolve() / _DESIGN_RELATIVE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("Round 21 probability-basis ablation design is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Round 21 probability-basis ablation design is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 probability-basis ablation design is invalid")
    return validate_round21_probability_basis_ablation_design(value)


def _arm_matrix(
    panel: Round21DevelopmentPanel,
    *,
    include_basis: bool,
) -> np.ndarray:
    matrix = _model._layer_matrix(panel, "core")
    selected = matrix if include_basis else matrix[:, :-1]
    if selected.shape[1] != panel.core_features.shape[1] + int(include_basis):
        raise ValueError("Round 21 probability-basis ablation width differs")
    return np.asarray(selected, dtype=np.float32, order="C")


def _arm_feature_sha256(
    panel: Round21DevelopmentPanel,
    *,
    include_basis: bool,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": (POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SCHEMA_VERSION),
            "core_feature_names_sha256": panel.core_feature_names_sha256,
            "model_basis_features": (
                ["model.market_prior_minus_structural_log_odds"]
                if include_basis
                else []
            ),
        }
    )


def _prediction_sha256(value: np.ndarray) -> str:
    selected = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(selected.tobytes(order="C")).hexdigest()


def _fit_arm(
    *,
    name: str,
    include_basis: bool,
    train: Round21DevelopmentPanel,
    calibration: Round21DevelopmentPanel,
    selection: Round21DevelopmentPanel,
) -> tuple[dict[str, object], np.ndarray]:
    stop_indices, platt_indices = _model._split_calibration_indices(
        calibration,
        "core",
    )
    train_indices = _model._selected_indices(train, "core")
    selection_indices = _model._selected_indices(selection, "core")
    for stage, panel, indices in (
        ("train", train, train_indices),
        ("regularization_selection", calibration, stop_indices),
        ("platt_calibration", calibration, platt_indices),
        ("paired_evaluation", selection, selection_indices),
    ):
        labels = panel.labels[indices]
        if _model._condition_count(
            panel.condition_ids[indices]
        ) < POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS or set(
            labels.tolist()
        ) != {0.0, 1.0}:
            raise ValueError(f"Round 21 probability-basis {stage} differs")

    train_matrix = _arm_matrix(train, include_basis=include_basis)[train_indices]
    stop_matrix = _arm_matrix(calibration, include_basis=include_basis)[stop_indices]
    platt_matrix = _arm_matrix(calibration, include_basis=include_basis)[platt_indices]
    selection_matrix = _arm_matrix(selection, include_basis=include_basis)[
        selection_indices
    ]
    train_weights = _model._condition_weights(train.condition_ids[train_indices])
    transform = _model._fit_transform(train_matrix, train_weights)
    feature_sha = _arm_feature_sha256(train, include_basis=include_basis)
    if any(
        _arm_feature_sha256(panel, include_basis=include_basis) != feature_sha
        for panel in (calibration, selection)
    ):
        raise ValueError("Round 21 probability-basis feature identity differs")

    candidates: list[tuple[dict[str, object], dict[str, float | int]]] = []
    for l2 in _L2_GRID:
        model = _model._fit_logistic_residual(
            train_matrix,
            train.labels[train_indices],
            train.structural_probability[train_indices],
            train_weights,
            transform,
            l2=l2,
            population_layer="core",
            feature_layer="core",
            feature_names_sha256=feature_sha,
            candidate_namespace=f"basis-ablation-{name}",
        )
        stop_prediction = _model._raw_prediction(
            model,
            stop_matrix,
            calibration.structural_probability[stop_indices],
            calibration.condition_ids[stop_indices],
            calibration.decision_time_ms[stop_indices],
        )
        candidates.append(
            (
                model,
                _model._metrics(
                    calibration.condition_ids[stop_indices],
                    calibration.labels[stop_indices],
                    stop_prediction,
                ),
            )
        )
    selected_model, _selected_stop_metrics = min(
        candidates,
        key=lambda value: (
            _finite_float(
                value[1]["condition_equal_log_loss"], label="condition log loss"
            ),
            -_finite_float(value[0]["l2"], label="regularization"),
            str(value[0]["candidate_id"]),
        ),
    )
    raw_calibration = _model._raw_prediction(
        selected_model,
        platt_matrix,
        calibration.structural_probability[platt_indices],
        calibration.condition_ids[platt_indices],
        calibration.decision_time_ms[platt_indices],
    )
    selected_model["calibration"] = _model._fit_platt(
        calibration.labels[platt_indices],
        raw_calibration,
        calibration.condition_ids[platt_indices],
    )
    prediction = _model._apply_platt(
        _model._raw_prediction(
            selected_model,
            selection_matrix,
            selection.structural_probability[selection_indices],
            selection.condition_ids[selection_indices],
            selection.decision_time_ms[selection_indices],
        ),
        selected_model["calibration"],  # type: ignore[arg-type]
    )
    return (
        {
            "arm": name,
            "include_basis": include_basis,
            "feature_count": int(train_matrix.shape[1]),
            "feature_names_sha256": feature_sha,
            "regularization_candidates": [
                {
                    "candidate_id": model["candidate_id"],
                    "l2": format(
                        _finite_float(model["l2"], label="regularization"), "g"
                    ),
                    "metrics": metrics,
                }
                for model, metrics in candidates
            ],
            "selected_l2": format(
                _finite_float(selected_model["l2"], label="regularization"), "g"
            ),
            "selected_model": deepcopy(selected_model),
            "paired_evaluation_metrics": _model._metrics(
                selection.condition_ids[selection_indices],
                selection.labels[selection_indices],
                prediction,
            ),
            "prediction_sha256": _prediction_sha256(prediction),
        },
        prediction,
    )


def evaluate_round21_probability_basis_ablation(
    *,
    train: Round21DevelopmentPanel,
    tune_calibration: Round21DevelopmentPanel,
    tune_selection: Round21DevelopmentPanel,
    publication_manifest_sha256: str,
    terminal_transport_manifest_sha256: str,
) -> dict[str, object]:
    """Evaluate one nested, target-bearing development ablation only."""

    train, calibration, selection = validate_round21_development_partitions(
        train=train,
        tune_calibration=tune_calibration,
        tune_selection=tune_selection,
    )
    source_evidence = {
        "publication_manifest_sha256": str(publication_manifest_sha256).lower(),
        "terminal_transport_manifest_sha256": str(
            terminal_transport_manifest_sha256
        ).lower(),
    }
    if any(
        _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
        for value in source_evidence.values()
    ):
        raise ValueError("Round 21 probability-basis source evidence differs")
    baseline, baseline_prediction = _fit_arm(
        name="baseline",
        include_basis=False,
        train=train,
        calibration=calibration,
        selection=selection,
    )
    challenger, challenger_prediction = _fit_arm(
        name="challenger",
        include_basis=True,
        train=train,
        calibration=calibration,
        selection=selection,
    )
    paired = {
        metric: _model.round21_paired_predictive_improvement(
            selection.condition_ids,
            selection.labels,
            baseline_prediction,
            challenger_prediction,
            metric=metric,
            seed_offset=700 + index,
        )
        for index, metric in enumerate(("log_loss", "brier"))
    }
    accepted = all(float(paired[metric]["lower_95"]) > 0.0 for metric in paired)
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_BASIS_ABLATION_RESULT_SCHEMA_VERSION,
        "design_sha256": POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256,
        "model_design_sha256": (
            POLYMARKET_ROUND21_BASIS_ABLATION_PARENT_MODEL_DESIGN_SHA256
        ),
        "source_evidence": source_evidence,
        "dataset_and_partition": {
            "train": round21_development_dataset_identity(train),
            "tune_calibration": round21_development_dataset_identity(calibration),
            "tune_selection": round21_development_dataset_identity(selection),
        },
        "arms": {
            "baseline": baseline,
            "challenger": challenger,
        },
        "paired_improvement": paired,
        "basis_accepted": accepted,
        "next_action": (
            "retain_basis_and_run_full_candidate_and_economic_development_pipeline"
            if accepted
            else "reject_basis_and_supersede_model_design_v6_before_full_fit"
        ),
        "development_targets_accessed": True,
        "sealed_test_features_accessed": False,
        "sealed_test_targets_accessed": False,
        "economic_evaluation_completed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return validate_round21_probability_basis_ablation_result(payload)


def _valid_metrics(value: object, *, condition_count: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "condition_count",
        "condition_equal_log_loss",
        "condition_equal_brier_score",
        "log_loss_standard_error",
    }:
        return False
    try:
        metrics = tuple(float(value[key]) for key in value if key != "condition_count")
    except (TypeError, ValueError):
        return False
    return value.get("condition_count") == condition_count and all(
        math.isfinite(item) and item >= 0.0 for item in metrics
    )


def _valid_paired(value: object, *, condition_count: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "condition_count",
        "mean",
        "lower_95",
        "upper_95",
    }:
        return False
    try:
        mean = float(value["mean"])
        lower = float(value["lower_95"])
        upper = float(value["upper_95"])
    except (TypeError, ValueError):
        return False
    return bool(
        value.get("condition_count") == condition_count
        and all(math.isfinite(item) for item in (mean, lower, upper))
        and lower <= mean <= upper
    )


def _positive_paired_lower_bounds(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        for metric in ("log_loss", "brier"):
            metric_value = value.get(metric)
            if (
                not isinstance(metric_value, Mapping)
                or _finite_float(
                    metric_value.get("lower_95"), label=f"{metric} lower bound"
                )
                <= 0.0
            ):
                return False
    except ValueError:
        return False
    return True


def _valid_dataset_identity(value: object, *, role: str) -> bool:
    if not isinstance(value, Mapping) or set(value) != _DATASET_IDENTITY_KEYS:
        return False
    if any(
        not isinstance(value.get(key), int) or isinstance(value.get(key), bool)
        for key in (
            "row_count",
            "condition_count",
            "first_event_start_ms",
            "last_event_start_ms",
        )
    ):
        return False
    try:
        row_count = int(value["row_count"])
        condition_count = int(value["condition_count"])
        first_event_start_ms = int(value["first_event_start_ms"])
        last_event_start_ms = int(value["last_event_start_ms"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    minimum_conditions = POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
    if role == "tune_calibration":
        minimum_conditions *= 2
    return bool(
        value.get("role") == role
        and row_count >= condition_count >= minimum_conditions
        and 0 <= first_event_start_ms <= last_event_start_ms
        and value.get("dataset_design_sha256")
        == POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        and all(
            _SHA256.fullmatch(str(value.get(key) or "")) is not None
            and str(value.get(key)) != _EMPTY_SHA256
            for key in ("dataset_sha256", "target_manifest_sha256")
        )
    )


def _regularization_condition_count(value: object) -> int:
    if (
        not isinstance(value, list)
        or not value
        or not isinstance(value[0], Mapping)
        or not isinstance(value[0].get("metrics"), Mapping)
    ):
        return 0
    count = value[0]["metrics"].get("condition_count", 0)
    return (
        count
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        else 0
    )


def _valid_regularization_candidate(
    value: object,
    *,
    arm_name: str,
    expected_l2: str,
    condition_count: int,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "candidate_id",
        "l2",
        "metrics",
    }:
        return False
    return bool(
        value.get("candidate_id")
        == f"basis-ablation-{arm_name}-logistic-l2-{expected_l2}"
        and value.get("l2") == expected_l2
        and _valid_metrics(value.get("metrics"), condition_count=condition_count)
    )


def _valid_logistic_model(
    value: object,
    *,
    arm_name: str,
    feature_count: int,
    feature_names_sha256: str,
    selected_l2: str,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != _MODEL_KEYS:
        return False
    transform = value.get("transform")
    calibration = value.get("calibration")
    try:
        l2 = float(value["l2"])
        intercept = float(value["intercept"])
        coefficient = np.asarray(value["coefficient"], dtype=np.float64)
        calibration_intercept = float(calibration["intercept"])  # type: ignore[index]
        calibration_slope = float(calibration["slope"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    candidate_id = f"basis-ablation-{arm_name}-logistic-l2-{selected_l2}"
    return bool(
        value.get("candidate_id") == candidate_id
        and value.get("family") == "logistic_residual"
        and value.get("layer") == "core"
        and value.get("population_layer") == "core"
        and value.get("feature_layer") == "core"
        and value.get("feature_names_sha256") == feature_names_sha256
        and format(l2, "g") == selected_l2
        and math.isfinite(intercept)
        and coefficient.shape == (feature_count,)
        and np.all(np.isfinite(coefficient))
        and _model._valid_transform(transform)
        and isinstance(transform, Mapping)
        and len(transform["lower"]) == feature_count  # type: ignore[arg-type]
        and isinstance(calibration, Mapping)
        and set(calibration) == {"intercept", "slope"}
        and math.isfinite(calibration_intercept)
        and math.isfinite(calibration_slope)
        and calibration_slope >= 0.0
    )


def validate_round21_probability_basis_ablation_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).strip().lower()
    dataset = payload.get("dataset_and_partition")
    source_evidence = payload.get("source_evidence")
    arms = payload.get("arms")
    paired = payload.get("paired_improvement")
    if (
        set(payload) != _RESULT_KEYS
        or not isinstance(source_evidence, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(arms, Mapping)
    ):
        raise ValueError("Round 21 probability-basis ablation result differs")
    selection_identity = dataset.get("tune_selection")
    try:
        condition_count = int(selection_identity["condition_count"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Round 21 probability-basis ablation result differs")
    valid_paired = bool(
        isinstance(paired, Mapping)
        and set(paired) == {"log_loss", "brier"}
        and all(
            _valid_paired(paired[metric], condition_count=condition_count)
            for metric in ("log_loss", "brier")
        )
    )
    expected_accepted = bool(valid_paired and _positive_paired_lower_bounds(paired))
    valid_arms = set(arms) == set(_ARMS)
    arm_feature_counts: dict[str, int] = {}
    if valid_arms:
        for index, arm_name in enumerate(_ARMS):
            arm = arms[arm_name]
            include_basis = bool(index)
            if not isinstance(arm, Mapping):
                valid_arms = False
                break
            candidates = arm.get("regularization_candidates")
            model = arm.get("selected_model")
            metrics = arm.get("paired_evaluation_metrics")
            feature_count = arm.get("feature_count")
            regularization_condition_count = _regularization_condition_count(candidates)
            feature_names_sha256 = str(arm.get("feature_names_sha256") or "")
            selected_l2 = str(arm.get("selected_l2") or "")
            if (
                set(arm) != _ARM_KEYS
                or arm.get("arm") != arm_name
                or arm.get("include_basis") is not include_basis
                or not isinstance(feature_count, int)
                or isinstance(feature_count, bool)
                or feature_count < 1
                or not isinstance(candidates, list)
                or len(candidates) != 3
                or not all(
                    _valid_regularization_candidate(
                        item,
                        arm_name=arm_name,
                        expected_l2=expected_l2,
                        condition_count=regularization_condition_count,
                    )
                    for item, expected_l2 in zip(
                        candidates,
                        ("0.01", "0.1", "1"),
                        strict=True,
                    )
                )
                or regularization_condition_count
                < POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
                or selected_l2 not in {"0.01", "0.1", "1"}
                or not _valid_logistic_model(
                    model,
                    arm_name=arm_name,
                    feature_count=feature_count,
                    feature_names_sha256=feature_names_sha256,
                    selected_l2=selected_l2,
                )
                or not _valid_metrics(metrics, condition_count=condition_count)
                or _SHA256.fullmatch(feature_names_sha256) is None
                or feature_names_sha256 == _EMPTY_SHA256
                or _SHA256.fullmatch(str(arm.get("prediction_sha256") or "")) is None
                or str(arm.get("prediction_sha256")) == _EMPTY_SHA256
            ):
                valid_arms = False
                break
            arm_feature_counts[arm_name] = feature_count
    valid_arms = bool(
        valid_arms
        and arm_feature_counts.get("challenger", 0)
        == arm_feature_counts.get("baseline", 0) + 1
        and str(arms["baseline"].get("feature_names_sha256"))
        != str(arms["challenger"].get("feature_names_sha256"))
    )
    expected_next = (
        "retain_basis_and_run_full_candidate_and_economic_development_pipeline"
        if expected_accepted
        else "reject_basis_and_supersede_model_design_v6_before_full_fit"
    )
    if (
        claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or claimed == _EMPTY_SHA256
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_BASIS_ABLATION_RESULT_SCHEMA_VERSION
        or payload.get("design_sha256")
        != POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256
        or payload.get("model_design_sha256")
        != POLYMARKET_ROUND21_BASIS_ABLATION_PARENT_MODEL_DESIGN_SHA256
        or set(source_evidence) != _SOURCE_EVIDENCE_KEYS
        or any(
            _SHA256.fullmatch(str(source_evidence.get(key) or "")) is None
            or str(source_evidence.get(key)) == _EMPTY_SHA256
            for key in _SOURCE_EVIDENCE_KEYS
        )
        or set(dataset) != {"train", "tune_calibration", "tune_selection"}
        or not all(
            _valid_dataset_identity(dataset.get(role), role=role)
            for role in ("train", "tune_calibration", "tune_selection")
        )
        or condition_count < POLYMARKET_ROUND21_MINIMUM_DEVELOPMENT_CONDITIONS
        or not valid_arms
        or not valid_paired
        or payload.get("basis_accepted") is not expected_accepted
        or payload.get("next_action") != expected_next
        or payload.get("development_targets_accessed") is not True
        or any(
            payload.get(key) is not False
            for key in (
                "sealed_test_features_accessed",
                "sealed_test_targets_accessed",
                "economic_evaluation_completed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 probability-basis ablation result differs")
    return {**payload, "result_sha256": claimed}


def load_round21_probability_basis_ablation_result(
    path: str | Path,
) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("Round 21 probability-basis ablation result is unavailable")
    selected = candidate.resolve()
    if not selected.is_file() or selected.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Round 21 probability-basis ablation result is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Round 21 probability-basis ablation result is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 probability-basis ablation result is invalid")
    return validate_round21_probability_basis_ablation_result(value)


__all__ = [
    "POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_BASIS_ABLATION_DESIGN_SHA256",
    "POLYMARKET_ROUND21_BASIS_ABLATION_PARENT_MODEL_DESIGN_SHA256",
    "POLYMARKET_ROUND21_BASIS_ABLATION_RESULT_SCHEMA_VERSION",
    "evaluate_round21_probability_basis_ablation",
    "load_round21_probability_basis_ablation_design",
    "load_round21_probability_basis_ablation_result",
    "validate_round21_probability_basis_ablation_design",
    "validate_round21_probability_basis_ablation_result",
]

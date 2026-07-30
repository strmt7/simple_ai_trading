"""Condition-block calibrated probability envelopes for Round 17."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping

import numpy as np

from .polymarket_round17_execution import Round17ProbabilityEnvelope
from .polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    PolymarketRound17FeatureRow,
)
from .polymarket_round17_model import (
    Round17CandidateInferenceSession,
    Round17DevelopmentPanel,
    predict_round17_candidate,
    validate_round17_pretest_artifact,
)


POLYMARKET_ROUND17_UNCERTAINTY_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-probability-calibration-v1"
)
POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT = 50
POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SAMPLES = 500
POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SEED = 117_017
POLYMARKET_ROUND17_CALIBRATION_LOWER_QUANTILE = 0.05
POLYMARKET_ROUND17_CALIBRATION_UPPER_QUANTILE = 0.95
POLYMARKET_ROUND17_CALIBRATION_MINIMUM_CONDITIONS_PER_BIN = 30
POLYMARKET_ROUND17_CALIBRATION_MINIMUM_TOTAL_CONDITIONS = 100
_PROBABILITY_FLOOR = 1e-6
_EMBARGO_MS = 3_600_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COHORT_PLAN_SHA256 = "37fede4da0d6c504bce7cb763b9bd49032e0252a8cede045f29f05acff67fc00"


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


def _grid() -> np.ndarray:
    return (
        np.arange(POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT, dtype=np.float64) + 0.5
    ) / POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT


def _bin_index(probabilities: np.ndarray) -> np.ndarray:
    selected = np.asarray(probabilities, dtype=np.float64)
    return np.minimum(
        (np.clip(selected, 0.0, 1.0) * POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT).astype(
            np.int64
        ),
        POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT - 1,
    )


def _condition_bin_statistics(
    panel: Round17DevelopmentPanel,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    conditions = tuple(str(value) for value in np.unique(panel.condition_ids))
    weights = np.zeros(
        (len(conditions), POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT),
        dtype=np.float64,
    )
    positives = np.zeros_like(weights)
    bins = _bin_index(predictions)
    for condition_index, condition in enumerate(conditions):
        rows = np.flatnonzero(panel.condition_ids == condition)
        row_weight = 1.0 / len(rows)
        for row in rows:
            index = int(bins[row])
            weights[condition_index, index] += row_weight
            positives[condition_index, index] += row_weight * panel.labels[row]
    return weights, positives, conditions


def _isotonic_grid(
    weights: np.ndarray,
    positives: np.ndarray,
) -> np.ndarray:
    grid = _grid()
    active = np.flatnonzero(weights > 0)
    if len(active) == 0:
        raise ValueError("Round 17 probability calibration has no supported bins")
    blocks: list[list[float]] = []
    for index in active:
        weight = float(weights[index])
        blocks.append(
            [
                float(index),
                float(index),
                weight,
                float(positives[index]),
            ]
        )
        while (
            len(blocks) >= 2
            and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                [
                    left[0],
                    right[1],
                    left[2] + right[2],
                    left[3] + right[3],
                ]
            )
    x: list[float] = []
    y: list[float] = []
    for start, end, weight, positive in blocks:
        value = positive / weight
        x.append((grid[int(start)] + grid[int(end)]) / 2.0)
        y.append(value)
    calibrated = np.interp(grid, np.asarray(x), np.asarray(y))
    return np.clip(calibrated, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)


def fit_round17_probability_calibration(
    panel: Round17DevelopmentPanel,
    model_pretest: Mapping[str, object],
) -> dict[str, object]:
    """Fit a test-blind condition-block isotonic bootstrap envelope."""

    selected = panel.validate()
    parent = validate_round17_pretest_artifact(model_pretest)
    if selected.role != "tune_uncertainty":
        raise ValueError("Round 17 uncertainty requires tune_uncertainty input")
    if parent["development_accepted"] is not True:
        raise ValueError("Round 17 model pretest did not pass development gates")
    partition = parent["dataset_and_partition"]
    if not isinstance(partition, Mapping):
        raise ValueError("Round 17 model partition identity differs")
    roles = partition["roles"]
    if not isinstance(roles, Mapping):
        raise ValueError("Round 17 model partition roles differ")
    tune_selection = roles["tune_selection"]
    if not isinstance(tune_selection, Mapping):
        raise ValueError("Round 17 model selection boundary differs")
    parent_conditions = {
        str(condition_id)
        for role in roles.values()
        if isinstance(role, Mapping)
        for condition_id in role["condition_ids"]  # type: ignore[index]
    }
    conditions = tuple(str(value) for value in np.unique(selected.condition_ids))
    first_event_start_ms = int(np.min(selected.event_start_ms))
    last_selection_start_ms = int(tune_selection["last_event_start_ms"])
    embargo_ms = first_event_start_ms - (last_selection_start_ms + 300_000)
    if (
        selected.dataset_sha256 != partition["dataset_sha256"]
        or selected.target_manifest_sha256 != partition["target_manifest_sha256"]
        or selected.cohort_plan_sha256 != partition["cohort_plan_sha256"]
        or parent_conditions.intersection(conditions)
        or embargo_ms < _EMBARGO_MS
    ):
        raise ValueError("Round 17 uncertainty partition leaks model development")
    candidate = parent["selected_candidate"]
    if not isinstance(candidate, Mapping):
        raise ValueError("Round 17 selected model identity differs")
    prediction = predict_round17_candidate(candidate, selected)
    weights, positives, calibrated_conditions = _condition_bin_statistics(
        selected,
        prediction,
    )
    if (
        calibrated_conditions != conditions
        or len(conditions) < POLYMARKET_ROUND17_CALIBRATION_MINIMUM_TOTAL_CONDITIONS
        or len(np.unique(selected.labels)) != 2
    ):
        raise ValueError("Round 17 uncertainty panel lacks condition or class support")
    total_weights = np.sum(weights, axis=0)
    total_positives = np.sum(positives, axis=0)
    point = _isotonic_grid(total_weights, total_positives)
    condition_support = np.sum(weights > 0, axis=0).astype(np.int64)

    generator = np.random.default_rng(POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SEED)
    bootstrap = np.empty(
        (
            POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SAMPLES,
            POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT,
        ),
        dtype=np.float64,
    )
    for sample_index in range(POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SAMPLES):
        multiplicity = generator.multinomial(
            len(conditions),
            np.full(len(conditions), 1.0 / len(conditions)),
        )
        sample_weights = multiplicity @ weights
        sample_positives = multiplicity @ positives
        bootstrap[sample_index] = _isotonic_grid(
            sample_weights,
            sample_positives,
        )
    lower = np.quantile(
        bootstrap,
        POLYMARKET_ROUND17_CALIBRATION_LOWER_QUANTILE,
        axis=0,
        method="linear",
    )
    upper = np.quantile(
        bootstrap,
        POLYMARKET_ROUND17_CALIBRATION_UPPER_QUANTILE,
        axis=0,
        method="linear",
    )
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    raw_bins = _bin_index(prediction)
    calibrated_prediction = point[raw_bins]
    condition_equal_weights = np.zeros(len(selected.labels), dtype=np.float64)
    for condition in conditions:
        rows = np.flatnonzero(selected.condition_ids == condition)
        condition_equal_weights[rows] = 1.0 / len(rows)
    condition_equal_weights /= np.sum(condition_equal_weights)
    brier = float(
        np.sum(
            condition_equal_weights * np.square(calibrated_prediction - selected.labels)
        )
    )
    probability = np.clip(
        calibrated_prediction,
        _PROBABILITY_FLOOR,
        1.0 - _PROBABILITY_FLOOR,
    )
    log_loss = float(
        -np.sum(
            condition_equal_weights
            * (
                selected.labels * np.log(probability)
                + (1.0 - selected.labels) * np.log1p(-probability)
            )
        )
    )
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND17_UNCERTAINTY_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "model_pretest_sha256": str(parent["pretest_sha256"]),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "model_development_accepted": True,
        "dataset_sha256": selected.dataset_sha256,
        "target_manifest_sha256": selected.target_manifest_sha256,
        "cohort_plan_sha256": selected.cohort_plan_sha256,
        "source_role": selected.role,
        "row_count": len(selected.labels),
        "condition_count": len(conditions),
        "condition_ids": list(conditions),
        "condition_ids_sha256": _canonical_sha256(list(conditions)),
        "first_event_start_ms": first_event_start_ms,
        "last_event_start_ms": int(np.max(selected.event_start_ms)),
        "partition_parent": {
            "source_role": "tune_selection",
            "last_event_start_ms": last_selection_start_ms,
            "embargo_ms": embargo_ms,
        },
        "bin_count": POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT,
        "grid_probability_up": _grid().tolist(),
        "calibrated_probability_up": point.tolist(),
        "lower_probability_up": lower.tolist(),
        "upper_probability_up": upper.tolist(),
        "condition_support_by_bin": condition_support.tolist(),
        "minimum_conditions_per_bin": (
            POLYMARKET_ROUND17_CALIBRATION_MINIMUM_CONDITIONS_PER_BIN
        ),
        "bootstrap": {
            "unit": "condition",
            "samples": POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SAMPLES,
            "seed": POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SEED,
            "lower_quantile": (POLYMARKET_ROUND17_CALIBRATION_LOWER_QUANTILE),
            "upper_quantile": (POLYMARKET_ROUND17_CALIBRATION_UPPER_QUANTILE),
        },
        "development_metrics": {
            "condition_equal_weighted_log_loss": log_loss,
            "condition_equal_weighted_brier": brier,
        },
        "unsupported_bin_action": "abstain",
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["calibration_sha256"] = _canonical_sha256(payload)
    return payload


def validate_round17_probability_calibration(
    value: Mapping[str, object],
    *,
    model_pretest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("calibration_sha256", "")).strip().lower()
    try:
        grid = np.asarray(payload.get("grid_probability_up"), dtype=np.float64)
        point = np.asarray(
            payload.get("calibrated_probability_up"),
            dtype=np.float64,
        )
        lower = np.asarray(payload.get("lower_probability_up"), dtype=np.float64)
        upper = np.asarray(payload.get("upper_probability_up"), dtype=np.float64)
        support = np.asarray(
            payload.get("condition_support_by_bin"),
            dtype=np.int64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 17 probability calibration integrity differs") from exc
    bootstrap = payload.get("bootstrap")
    partition_parent = payload.get("partition_parent")
    condition_ids = payload.get("condition_ids")
    condition_count = payload.get("condition_count")
    row_count = payload.get("row_count")
    first_event_start_ms = payload.get("first_event_start_ms")
    last_event_start_ms = payload.get("last_event_start_ms")
    parent_last_event_start_ms = (
        partition_parent.get("last_event_start_ms")
        if isinstance(partition_parent, Mapping)
        else None
    )
    embargo_ms = (
        partition_parent.get("embargo_ms")
        if isinstance(partition_parent, Mapping)
        else None
    )
    model_pretest_sha256 = str(payload.get("model_pretest_sha256") or "")
    target_manifest_sha256 = str(payload.get("target_manifest_sha256") or "")
    metrics = payload.get("development_metrics")
    if (
        claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND17_UNCERTAINTY_SCHEMA_VERSION
        or payload.get("contract_sha256") != POLYMARKET_ROUND17_CONTRACT_SHA256
        or payload.get("source_role") != "tune_uncertainty"
        or payload.get("model_development_accepted") is not True
        or len(model_pretest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in model_pretest_sha256
        )
        or not str(payload.get("candidate_id") or "")
        or payload.get("cohort_plan_sha256") != _COHORT_PLAN_SHA256
        or len(target_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in target_manifest_sha256
        )
        or not isinstance(condition_ids, list)
        or not condition_ids
        or condition_ids != sorted(condition_ids)
        or len(condition_ids) != len(set(condition_ids))
        or any(
            not isinstance(condition_id, str)
            or not condition_id.startswith("0x")
            or len(condition_id) != 66
            or any(
                character not in "0123456789abcdef" for character in condition_id[2:]
            )
            for condition_id in condition_ids
        )
        or condition_count != len(condition_ids)
        or payload.get("condition_ids_sha256") != _canonical_sha256(condition_ids)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < len(condition_ids)
        or isinstance(first_event_start_ms, bool)
        or not isinstance(first_event_start_ms, int)
        or isinstance(last_event_start_ms, bool)
        or not isinstance(last_event_start_ms, int)
        or first_event_start_ms <= 0
        or first_event_start_ms % 300_000
        or last_event_start_ms < first_event_start_ms
        or last_event_start_ms % 300_000
        or not isinstance(partition_parent, Mapping)
        or partition_parent.get("source_role") != "tune_selection"
        or isinstance(parent_last_event_start_ms, bool)
        or not isinstance(parent_last_event_start_ms, int)
        or parent_last_event_start_ms <= 0
        or parent_last_event_start_ms % 300_000
        or isinstance(embargo_ms, bool)
        or not isinstance(embargo_ms, int)
        or embargo_ms != first_event_start_ms - (parent_last_event_start_ms + 300_000)
        or embargo_ms < _EMBARGO_MS
        or payload.get("bin_count") != POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT
        or any(
            array.shape != (POLYMARKET_ROUND17_CALIBRATION_BIN_COUNT,)
            for array in (grid, point, lower, upper, support)
        )
        or not all(np.all(np.isfinite(array)) for array in (grid, point, lower, upper))
        or not np.array_equal(grid, _grid())
        or np.any(np.diff(grid) <= 0)
        or np.any(np.diff(point) < -1e-12)
        or np.any(np.diff(lower) < -1e-12)
        or np.any(np.diff(upper) < -1e-12)
        or np.any(lower <= 0)
        or np.any(upper >= 1)
        or np.any(lower > point)
        or np.any(point > upper)
        or np.any(support < 0)
        or np.any(support > len(condition_ids))
        or payload.get("minimum_conditions_per_bin")
        != POLYMARKET_ROUND17_CALIBRATION_MINIMUM_CONDITIONS_PER_BIN
        or not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "condition_equal_weighted_log_loss",
            "condition_equal_weighted_brier",
        }
        or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or float(metric) < 0.0
            for metric in metrics.values()
        )
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("unit") != "condition"
        or bootstrap.get("samples") != POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SAMPLES
        or bootstrap.get("seed") != POLYMARKET_ROUND17_CALIBRATION_BOOTSTRAP_SEED
        or payload.get("unsupported_bin_action") != "abstain"
        or any(
            payload.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 17 probability calibration integrity differs")
    if model_pretest is not None:
        parent = validate_round17_pretest_artifact(model_pretest)
        parent_partition = parent["dataset_and_partition"]
        parent_roles = (
            parent_partition["roles"] if isinstance(parent_partition, Mapping) else None
        )
        selection = (
            parent_roles["tune_selection"]
            if isinstance(parent_roles, Mapping)
            else None
        )
        if (
            parent["pretest_sha256"] != model_pretest_sha256
            or parent["development_accepted"] is not True
            or parent["selected_candidate_id"] != payload["candidate_id"]
            or not isinstance(parent_partition, Mapping)
            or parent_partition["dataset_sha256"] != payload["dataset_sha256"]
            or parent_partition["target_manifest_sha256"] != target_manifest_sha256
            or parent_partition["cohort_plan_sha256"] != _COHORT_PLAN_SHA256
            or not isinstance(selection, Mapping)
            or selection["last_event_start_ms"] != parent_last_event_start_ms
            or any(
                set(role["condition_ids"]).intersection(condition_ids)
                for role in parent_roles.values()  # type: ignore[union-attr]
            )
        ):
            raise ValueError("Round 17 probability calibration parent differs")
    return {**payload, "calibration_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round17CalibratedEnvelope:
    envelope: Round17ProbabilityEnvelope
    raw_probability_up: float
    support_condition_count: int
    supported: bool
    calibration_sha256: str
    model_pretest_sha256: str
    dataset_sha256: str
    source_role: str
    event_start_ms: int
    condition_id: str
    decision_time_ms: int
    feature_input_sha256: str
    feature_values_sha256: str
    test_access_sha256: str | None = None


def apply_round17_probability_calibration(
    calibration: Mapping[str, object],
    model_pretest: Mapping[str, object],
    row: PolymarketRound17FeatureRow,
    *,
    dataset_sha256: str,
    event_start_ms: int,
) -> Round17CalibratedEnvelope:
    """Apply the frozen calibration band to one exact feature-row prediction."""

    return apply_round17_probability_calibration_rows(
        calibration,
        model_pretest,
        (row,),
        dataset_sha256=dataset_sha256,
        event_start_ms=event_start_ms,
    )[0]


def apply_round17_probability_calibration_rows(
    calibration: Mapping[str, object],
    model_pretest: Mapping[str, object],
    rows: Sequence[PolymarketRound17FeatureRow],
    *,
    dataset_sha256: str,
    event_start_ms: int,
    source_role: str = "tune_economic",
    test_access_sha256: str | None = None,
    inference_session: Round17CandidateInferenceSession | None = None,
) -> tuple[Round17CalibratedEnvelope, ...]:
    """Apply one validated model and calibration to an exact condition batch."""

    session = Round17ProbabilityCalibrationSession(
        calibration,
        model_pretest,
        dataset_sha256=dataset_sha256,
        source_role=source_role,
        test_access_sha256=test_access_sha256,
        inference_session=inference_session,
    )
    return session.apply_rows(rows, event_start_ms=event_start_ms)


class Round17ProbabilityCalibrationSession:
    """Validate immutable calibration parents once, then stream exact conditions."""

    def __init__(
        self,
        calibration: Mapping[str, object],
        model_pretest: Mapping[str, object],
        *,
        dataset_sha256: str,
        source_role: str = "tune_economic",
        test_access_sha256: str | None = None,
        inference_session: Round17CandidateInferenceSession | None = None,
    ) -> None:
        self.artifact = validate_round17_probability_calibration(
            calibration,
            model_pretest=model_pretest,
        )
        self.parent = validate_round17_pretest_artifact(model_pretest)
        self.dataset_sha256 = str(dataset_sha256)
        self.source_role = str(source_role or "").strip()
        self.test_access_sha256 = (
            None
            if test_access_sha256 is None
            else str(test_access_sha256).strip().lower()
        )
        source_binding_valid = (
            self.source_role == "tune_economic"
            and self.dataset_sha256 == self.artifact["dataset_sha256"]
            and self.test_access_sha256 is None
        ) or (
            self.source_role == "test"
            and _SHA256.fullmatch(self.dataset_sha256) is not None
            and self.dataset_sha256 != self.artifact["dataset_sha256"]
            and self.test_access_sha256 is not None
            and _SHA256.fullmatch(self.test_access_sha256) is not None
        )
        candidate = self.parent["selected_candidate"]
        if not source_binding_valid or not isinstance(candidate, Mapping):
            raise ValueError("Round 17 calibration session parent differs")
        if inference_session is not None and (
            not isinstance(inference_session, Round17CandidateInferenceSession)
            or inference_session.candidate_sha256 != _canonical_sha256(candidate)
        ):
            raise ValueError("Round 17 inference session parent differs")
        self.inference_session = (
            Round17CandidateInferenceSession(candidate)
            if inference_session is None
            else inference_session
        )
        self.calibration_condition_ids = frozenset(self.artifact["condition_ids"])
        self.last_calibration_event_start_ms = int(
            self.artifact["last_event_start_ms"]
        )
        self.grid = np.asarray(
            self.artifact["grid_probability_up"],
            dtype=np.float64,
        )
        self.point_values = np.asarray(
            self.artifact["calibrated_probability_up"],
            dtype=np.float64,
        )
        self.lower_values = np.asarray(
            self.artifact["lower_probability_up"],
            dtype=np.float64,
        )
        self.upper_values = np.asarray(
            self.artifact["upper_probability_up"],
            dtype=np.float64,
        )
        self.support_values = np.asarray(
            self.artifact["condition_support_by_bin"],
            dtype=np.int64,
        )
        for values in (
            self.grid,
            self.point_values,
            self.lower_values,
            self.upper_values,
            self.support_values,
        ):
            values.setflags(write=False)

    def apply_rows(
        self,
        rows: Sequence[PolymarketRound17FeatureRow],
        *,
        event_start_ms: int,
    ) -> tuple[Round17CalibratedEnvelope, ...]:
        event_start = int(event_start_ms)
        selected_rows = tuple(rows)
        if not selected_rows or any(
            not isinstance(row, PolymarketRound17FeatureRow)
            for row in selected_rows
        ):
            raise ValueError("Round 17 economic inference partition differs")
        condition_ids = {row.condition_id for row in selected_rows}
        if (
            len(condition_ids) != 1
            or event_start <= 0
            or event_start % 300_000
            or any(
                not event_start <= row.decision_time_ms < event_start + 300_000
                for row in selected_rows
            )
            or next(iter(condition_ids)) in self.calibration_condition_ids
            or event_start
            - (self.last_calibration_event_start_ms + 300_000)
            < _EMBARGO_MS
        ):
            raise ValueError("Round 17 economic inference partition differs")
        raw_predictions = self.inference_session.predict_rows(selected_rows)
        if np.any(~np.isfinite(raw_predictions)) or np.any(
            (raw_predictions <= 0.0) | (raw_predictions >= 1.0)
        ):
            raise ValueError("Round 17 raw probability differs")
        output: list[Round17CalibratedEnvelope] = []
        for row, raw_value in zip(selected_rows, raw_predictions, strict=True):
            raw = float(raw_value)
            right = min(
                len(self.grid) - 1,
                int(np.searchsorted(self.grid, raw, side="left")),
            )
            left = max(
                0,
                right - 1 if self.grid[right] > raw else right,
            )
            support = int(
                min(self.support_values[left], self.support_values[right])
            )
            supported = (
                support
                >= POLYMARKET_ROUND17_CALIBRATION_MINIMUM_CONDITIONS_PER_BIN
            )
            point = float(np.interp(raw, self.grid, self.point_values))
            if supported:
                lower = min(
                    point,
                    float(np.interp(raw, self.grid, self.lower_values)),
                )
                upper = max(
                    point,
                    float(np.interp(raw, self.grid, self.upper_values)),
                )
            else:
                lower = _PROBABILITY_FLOOR
                upper = 1.0 - _PROBABILITY_FLOOR
            point = min(upper, max(lower, point))
            evidence = {
                "schema_version": "polymarket-round17-calibrated-envelope-v1",
                "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
                "calibration_sha256": self.artifact["calibration_sha256"],
                "model_pretest_sha256": str(self.parent["pretest_sha256"]),
                "dataset_sha256": self.dataset_sha256,
                "source_role": self.source_role,
                "test_access_sha256": self.test_access_sha256,
                "event_start_ms": event_start,
                "condition_id": row.condition_id,
                "decision_time_ms": row.decision_time_ms,
                "feature_input_sha256": row.input_sha256,
                "feature_values_sha256": row.values_sha256,
                "raw_probability_up": format(raw, ".17g"),
                "calibrated_probability_up": format(point, ".17g"),
                "lower_probability_up": format(lower, ".17g"),
                "upper_probability_up": format(upper, ".17g"),
                "support_condition_count": support,
                "supported": supported,
                "unsupported_action": None if supported else "abstain",
            }
            digest = _canonical_sha256(evidence)
            envelope = Round17ProbabilityEnvelope(
                probability_up=point,
                lower_up=lower,
                upper_up=upper,
                evidence_sha256=digest,
            ).validated()
            output.append(
                Round17CalibratedEnvelope(
                    envelope=envelope,
                    raw_probability_up=raw,
                    support_condition_count=support,
                    supported=supported,
                    calibration_sha256=str(self.artifact["calibration_sha256"]),
                    model_pretest_sha256=str(self.parent["pretest_sha256"]),
                    dataset_sha256=self.dataset_sha256,
                    source_role=self.source_role,
                    event_start_ms=event_start,
                    condition_id=row.condition_id,
                    decision_time_ms=row.decision_time_ms,
                    feature_input_sha256=row.input_sha256,
                    feature_values_sha256=row.values_sha256,
                    test_access_sha256=self.test_access_sha256,
                )
            )
        return tuple(output)


__all__ = [
    "POLYMARKET_ROUND17_UNCERTAINTY_SCHEMA_VERSION",
    "Round17CalibratedEnvelope",
    "Round17ProbabilityCalibrationSession",
    "apply_round17_probability_calibration",
    "apply_round17_probability_calibration_rows",
    "fit_round17_probability_calibration",
    "validate_round17_probability_calibration",
]

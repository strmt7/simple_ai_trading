"""Target-isolated model diagnostic for salvaged Round 25 evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .polymarket_fees import PolymarketFeeModel
from .polymarket_round25_controls import (
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    fit_round25_feature_transform,
    round25_logit,
    transform_round25_features,
)
from .polymarket_round25_forensic_partition import (
    validate_round25_forensic_partition_manifest,
)
from .polymarket_round25_forensic_resolution import (
    POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION,
    Round25ForensicResolutionTarget,
    load_round25_forensic_resolution_targets,
    validate_round25_forensic_selection_freeze,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    Round25JointFeatureSnapshot,
)
from .polymarket_round25_joint_store import load_round25_joint_endpoint_inputs
from .polymarket_round25_lightgbm import POLYMARKET_ROUND25_LIGHTGBM_CONFIGS


POLYMARKET_ROUND25_FORENSIC_MODEL_FIT_SCHEMA_VERSION = (
    "polymarket-round25-v2-forensic-model-fit-v1"
)
POLYMARKET_ROUND25_FORENSIC_PREDICTION_SCHEMA_VERSION = (
    "polymarket-round25-v2-forensic-selection-predictions-v1"
)
POLYMARKET_ROUND25_FORENSIC_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-v2-forensic-diagnostic-result-v1"
)
_CANDIDATES = (
    "market-prior-v1",
    "phase-isotonic-market-prior-v1",
    "l2-logistic-residual-v1",
    "lightgbm-residual-depth3-v1",
    "lightgbm-residual-depth5-v1",
)
_L2_GRID = (0.01, 0.1, 1.0)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_LOG_LOSS_CLIP = 1e-12
_ENDPOINTS_PER_CONDITION = 16
_UP_ASK = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES.index("clob.up_best_ask")
_UP_ASK_DEPTH = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES.index("clob.up_ask_depth_1")
_DOWN_ASK = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES.index("clob.down_best_ask")
_DOWN_ASK_DEPTH = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES.index(
    "clob.down_ask_depth_1"
)


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


def _hash_chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _write_once(path: str | Path, value: Mapping[str, object]) -> Path:
    target = Path(path)
    encoded = (_canonical_json(value) + "\n").encode("ascii")
    if target.exists():
        if target.read_bytes() == encoded:
            return target
        raise FileExistsError("Round 25 forensic model artifact already differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Round 25 forensic model temporary path differs")
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    return target


@dataclass(frozen=True, slots=True)
class _Rows:
    role: str
    condition_ids: tuple[str, ...]
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    features: np.ndarray
    prior: np.ndarray
    labels: np.ndarray | None
    source_sha256: tuple[str, ...]

    def validated(self, *, require_labels: bool) -> _Rows:
        count = len(self.condition_ids)
        expected_shape = (count, len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES))
        if (
            self.role not in {"train", "calibration", "selection"}
            or count == 0
            or count % _ENDPOINTS_PER_CONDITION
            or len(set(zip(self.condition_ids, self.decision_time_ms, strict=True)))
            != count
            or self.event_start_ms.shape != (count,)
            or self.decision_time_ms.shape != (count,)
            or self.features.shape != expected_shape
            or self.prior.shape != (count,)
            or not np.all(np.isfinite(self.features))
            or not np.all(np.isfinite(self.prior))
            or not np.all((self.prior > 0.0) & (self.prior < 1.0))
            or len(self.source_sha256) != count
            or any(_SHA256.fullmatch(value) is None for value in self.source_sha256)
            or any(_CONDITION_ID.fullmatch(value) is None for value in self.condition_ids)
        ):
            raise ValueError("Round 25 forensic model rows differ")
        if require_labels:
            if (
                self.labels is None
                or self.labels.shape != (count,)
                or not np.all((self.labels == 0.0) | (self.labels == 1.0))
                or len(np.unique(self.labels)) != 2
            ):
                raise ValueError("Round 25 forensic target classes are insufficient")
        elif self.labels is not None:
            raise ValueError("Round 25 forensic selection rows exposed targets")
        return self


def _partition_roles(partition: Mapping[str, object]) -> dict[str, str]:
    return {
        str(row["condition_id"]): str(row["role"])
        for row in partition["conditions"]
    }


def _rows(
    snapshots: Sequence[Round25JointFeatureSnapshot],
    *,
    role: str,
    roles: Mapping[str, str],
    targets: Mapping[str, Round25ForensicResolutionTarget] | None,
) -> _Rows:
    selected = tuple(
        row for row in snapshots if roles.get(row.condition_id) == role
    )
    grouped = Counter(row.condition_id for row in selected)
    if not selected or any(value != _ENDPOINTS_PER_CONDITION for value in grouped.values()):
        raise ValueError("Round 25 forensic endpoint support differs")
    ordered = tuple(
        sorted(selected, key=lambda row: (row.event_start_ms, row.condition_id, row.decision_time_ms))
    )
    if targets is not None and set(grouped) != set(targets):
        raise ValueError("Round 25 forensic feature and target populations differ")
    labels = None
    if targets is not None:
        labels = np.asarray(
            [1.0 if targets[row.condition_id].target_up else 0.0 for row in ordered],
            dtype=np.float64,
        )
    return _Rows(
        role=role,
        condition_ids=tuple(row.condition_id for row in ordered),
        event_start_ms=np.asarray([row.event_start_ms for row in ordered], dtype=np.int64),
        decision_time_ms=np.asarray([row.decision_time_ms for row in ordered], dtype=np.int64),
        features=np.asarray([row.values for row in ordered], dtype=np.float64),
        prior=np.asarray([row.market_prior_probability for row in ordered], dtype=np.float64),
        labels=labels,
        source_sha256=tuple(row.source_chain_sha256 for row in ordered),
    ).validated(require_labels=targets is not None)


def _phase(rows: _Rows) -> np.ndarray:
    offsets = rows.decision_time_ms - rows.event_start_ms
    if not np.all((offsets >= 0) & (offsets < 300_000)):
        raise ValueError("Round 25 forensic endpoint phase differs")
    return np.minimum(3, offsets * 4 // 300_000).astype(np.int8)


def _weighted_isotonic(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    order = np.argsort(x, kind="stable")
    ordered_x = x[order]
    ordered_y = y[order]
    unique_x, first = np.unique(ordered_x, return_index=True)
    weight = np.add.reduceat(np.ones_like(ordered_y), first)
    positive = np.add.reduceat(ordered_y, first)
    blocks: list[list[float | int]] = []
    for index, (block_weight, block_positive) in enumerate(
        zip(weight, positive, strict=True)
    ):
        blocks.append([index, index, float(block_weight), float(block_positive)])
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            if float(left[3]) / float(left[2]) <= float(right[3]) / float(right[2]):
                break
            blocks[-2:] = [[
                int(left[0]),
                int(right[1]),
                float(left[2]) + float(right[2]),
                float(left[3]) + float(right[3]),
            ]]
    thresholds_x: list[float] = []
    thresholds_y: list[float] = []
    for start, end, block_weight, block_positive in blocks:
        fitted = float(block_positive) / float(block_weight)
        thresholds_x.append(float(unique_x[int(start)]))
        thresholds_y.append(fitted)
        if int(start) != int(end):
            thresholds_x.append(float(unique_x[int(end)]))
            thresholds_y.append(fitted)
    return tuple(thresholds_x), tuple(thresholds_y)


def _fit_isotonic(train: _Rows) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    phases = _phase(train)
    assert train.labels is not None
    output = []
    for value in range(4):
        selected = phases == value
        if np.sum(selected) < 2 or len(np.unique(train.labels[selected])) != 2:
            raise ValueError("Round 25 forensic isotonic phase lacks both classes")
        output.append(_weighted_isotonic(train.prior[selected], train.labels[selected]))
    return tuple(output)


def _predict_isotonic(
    rows: _Rows,
    model: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
) -> np.ndarray:
    output = np.empty(len(rows.prior), dtype=np.float64)
    phases = _phase(rows)
    for value in range(4):
        selected = phases == value
        x, y = model[value]
        output[selected] = np.interp(
            rows.prior[selected],
            np.asarray(x),
            np.asarray(y),
            left=y[0],
            right=y[-1],
        )
    return output


def _fit_logistic(
    train: _Rows,
    normalized: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    assert train.labels is not None
    offset = round25_logit(train.prior)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficients = parameters[1:]
        raw = intercept + normalized @ coefficients
        scaled = raw / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
        derivative = 1.0 - np.tanh(scaled) ** 2
        linear = offset + POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * np.tanh(scaled)
        probability = expit(linear)
        residual = (probability - train.labels) / _ENDPOINTS_PER_CONDITION
        loss = float(
            np.sum(
                (np.logaddexp(0.0, linear) - train.labels * linear)
                / _ENDPOINTS_PER_CONDITION
            )
            + 0.5 * l2 * float(coefficients @ coefficients)
        )
        chain = residual * derivative
        gradient = np.concatenate(
            (
                np.asarray([np.sum(chain)], dtype=np.float64),
                normalized.T @ chain + l2 * coefficients,
            )
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(normalized.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 512, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Round 25 forensic logistic fit failed")
    return float(result.x[0]), np.asarray(result.x[1:], dtype=np.float64)


def _predict_logistic(
    rows: _Rows,
    normalized: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    raw = intercept + normalized @ coefficients
    bounded = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * np.tanh(
        raw / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    )
    return expit(round25_logit(rows.prior) + bounded)


def _metrics(rows: _Rows, probability: np.ndarray) -> dict[str, float]:
    if (
        rows.labels is None
        or probability.shape != rows.labels.shape
        or not np.all(np.isfinite(probability))
        or not np.all((probability >= 0.0) & (probability <= 1.0))
    ):
        raise ValueError("Round 25 forensic metric population differs")
    clipped = np.clip(probability, _LOG_LOSS_CLIP, 1.0 - _LOG_LOSS_CLIP)
    loss = -(
        rows.labels * np.log(clipped)
        + (1.0 - rows.labels) * np.log1p(-clipped)
    ).reshape(-1, _ENDPOINTS_PER_CONDITION)
    brier = ((probability - rows.labels) ** 2).reshape(
        -1, _ENDPOINTS_PER_CONDITION
    )
    predicted = probability >= 0.5
    positive = rows.labels == 1.0
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        raise ValueError("Round 25 forensic metric targets lack both classes")
    order = np.argsort(probability, kind="stable")
    bins = np.array_split(order, 10)
    ece = sum(
        len(indices)
        / len(probability)
        * abs(float(np.mean(probability[indices]) - np.mean(rows.labels[indices])))
        for indices in bins
        if len(indices)
    )
    sorted_probability = probability[order]
    sorted_labels = rows.labels[order]
    cumulative_negative = 0.0
    concordance = 0.0
    offset = 0
    while offset < len(probability):
        end = offset + 1
        while end < len(probability) and sorted_probability[end] == sorted_probability[offset]:
            end += 1
        group = sorted_labels[offset:end]
        group_positive = float(np.sum(group == 1.0))
        group_negative = float(np.sum(group == 0.0))
        concordance += group_positive * (
            cumulative_negative + 0.5 * group_negative
        )
        cumulative_negative += group_negative
        offset = end
    return {
        "balanced_accuracy": 0.5
        * (float(np.mean(predicted[positive])) + float(np.mean(~predicted[negative]))),
        "condition_equal_brier_score": float(np.mean(brier)),
        "condition_equal_log_loss": float(np.mean(loss)),
        "direction_accuracy": float(np.mean(predicted == positive)),
        "expected_calibration_error": float(ece),
        "roc_auc": concordance / (float(np.sum(positive)) * float(np.sum(negative))),
    }


def _model_payload(
    *,
    feature_store_manifest_sha256: str,
    partition_sha256: str,
    fit_claim_sha256: str,
    train: _Rows,
    calibration: _Rows,
    models: Mapping[str, object],
    metrics: Mapping[str, object],
    selected_candidate_id: str,
) -> dict[str, object]:
    body = {
        "calibration_condition_count": len(set(calibration.condition_ids)),
        "candidate_metrics": metrics,
        "candidates": models,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        ),
        "feature_store_manifest_sha256": feature_store_manifest_sha256,
        "fit_resolution_claim_sha256": fit_claim_sha256,
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "partition_sha256": partition_sha256,
        "profitability_claim": False,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_MODEL_FIT_SCHEMA_VERSION,
        "selected_candidate_id": selected_candidate_id,
        "selection_targets_accessed": False,
        "train_condition_count": len(set(train.condition_ids)),
    }
    return {**body, "model_fit_sha256": _canonical_sha256(body)}


def fit_and_freeze_round25_forensic_models(
    *,
    feature_database: str | Path,
    partition_manifest: Mapping[str, object],
    fit_resolution_database: str | Path,
    created_at_ms: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit only on train, nominate on calibration, then freeze selection actions."""

    feature_manifest, endpoints = load_round25_joint_endpoint_inputs(feature_database)
    partition = validate_round25_forensic_partition_manifest(
        partition_manifest,
        expected_feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
    )
    fit_claim, targets = load_round25_forensic_resolution_targets(
        fit_resolution_database
    )
    if (
        fit_claim["stage"] != "fit"
        or fit_claim["feature_store_manifest_sha256"] != feature_manifest["manifest_sha256"]
        or fit_claim["partition_sha256"] != partition["partition_sha256"]
    ):
        raise ValueError("Round 25 forensic fit target binding differs")
    roles = _partition_roles(partition)
    all_endpoints = tuple(
        row for values in endpoints.values() for row in values
    )
    target_maps = {
        role: {item.condition_id: item for item in targets if item.role == role}
        for role in ("train", "calibration")
    }
    train = _rows(
        all_endpoints,
        role="train",
        roles=roles,
        targets=target_maps["train"],
    )
    calibration = _rows(
        all_endpoints,
        role="calibration",
        roles=roles,
        targets=target_maps["calibration"],
    )
    selection = _rows(
        all_endpoints,
        role="selection",
        roles=roles,
        targets=None,
    )
    center, scale = fit_round25_feature_transform(train.features)
    train_normalized = transform_round25_features(train.features, center, scale)
    calibration_normalized = transform_round25_features(
        calibration.features, center, scale
    )
    selection_normalized = transform_round25_features(selection.features, center, scale)
    probabilities: dict[str, np.ndarray] = {"market-prior-v1": calibration.prior}
    selection_probabilities: dict[str, np.ndarray] = {
        "market-prior-v1": selection.prior
    }
    models: dict[str, object] = {
        "market-prior-v1": {"status": "fitted_control"}
    }
    try:
        isotonic = _fit_isotonic(train)
        probabilities["phase-isotonic-market-prior-v1"] = _predict_isotonic(
            calibration, isotonic
        )
        selection_probabilities["phase-isotonic-market-prior-v1"] = (
            _predict_isotonic(selection, isotonic)
        )
        models["phase-isotonic-market-prior-v1"] = {
            "fit_role": "train",
            "phase_thresholds": [
                {"x": list(x), "y": list(y)} for x, y in isotonic
            ],
            "status": "fitted",
        }
    except ValueError as exc:
        models["phase-isotonic-market-prior-v1"] = {
            "reason": str(exc),
            "status": "ineligible",
        }
    logistic_models: dict[float, tuple[float, np.ndarray]] = {}
    logistic_metrics: dict[str, object] = {}
    for l2 in _L2_GRID:
        intercept, coefficients = _fit_logistic(
            train,
            train_normalized,
            l2=l2,
        )
        logistic_models[l2] = (intercept, coefficients)
        probability = _predict_logistic(
            calibration,
            calibration_normalized,
            intercept,
            coefficients,
        )
        logistic_metrics[str(l2)] = _metrics(calibration, probability)
    selected_l2 = min(
        _L2_GRID,
        key=lambda value: (
            logistic_metrics[str(value)]["condition_equal_log_loss"],
            logistic_metrics[str(value)]["condition_equal_brier_score"],
            value,
        ),
    )
    intercept, coefficients = logistic_models[selected_l2]
    probabilities["l2-logistic-residual-v1"] = _predict_logistic(
        calibration,
        calibration_normalized,
        intercept,
        coefficients,
    )
    selection_probabilities["l2-logistic-residual-v1"] = _predict_logistic(
        selection,
        selection_normalized,
        intercept,
        coefficients,
    )
    models["l2-logistic-residual-v1"] = {
        "calibration_grid": logistic_metrics,
        "center": center.tolist(),
        "coefficients": coefficients.tolist(),
        "fit_role": "train",
        "intercept": intercept,
        "l2": selected_l2,
        "scale": scale.tolist(),
        "status": "fitted",
    }
    train_conditions = len(set(train.condition_ids))
    train_rows = len(train.condition_ids)
    for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS:
        required_conditions = 2 * config.minimum_conditions_per_leaf
        required_rows = 2 * config.minimum_rows_per_leaf
        models[config.candidate_id] = {
            "observed_train_conditions": train_conditions,
            "observed_train_rows": train_rows,
            "required_train_conditions": required_conditions,
            "required_train_rows": required_rows,
            "status": "ineligible_insufficient_independent_support",
        }
    candidate_metrics = {
        candidate_id: _metrics(calibration, probability)
        for candidate_id, probability in probabilities.items()
    }
    prior_metrics = candidate_metrics["market-prior-v1"]
    admissible = [
        candidate_id
        for candidate_id in probabilities
        if candidate_id != "market-prior-v1"
        and candidate_metrics[candidate_id]["condition_equal_log_loss"]
        < prior_metrics["condition_equal_log_loss"]
        and candidate_metrics[candidate_id]["condition_equal_brier_score"]
        < prior_metrics["condition_equal_brier_score"]
    ]
    selected_candidate_id = min(
        admissible,
        key=lambda candidate_id: (
            candidate_metrics[candidate_id]["condition_equal_log_loss"],
            candidate_metrics[candidate_id]["condition_equal_brier_score"],
            _CANDIDATES.index(candidate_id),
        ),
        default="market-prior-v1",
    )
    model_fit = _model_payload(
        feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
        partition_sha256=partition["partition_sha256"],
        fit_claim_sha256=fit_claim["claim_sha256"],
        train=train,
        calibration=calibration,
        models=models,
        metrics=candidate_metrics,
        selected_candidate_id=selected_candidate_id,
    )
    selected_probability = selection_probabilities[selected_candidate_id]
    prediction_rows = []
    chain = hashlib.sha256(b"").hexdigest()
    for index, probability in enumerate(selected_probability):
        row = {
            "condition_id": selection.condition_ids[index],
            "decision_time_ms": int(selection.decision_time_ms[index]),
            "event_start_ms": int(selection.event_start_ms[index]),
            "feature_source_sha256": selection.source_sha256[index],
            "market_prior_probability_up": float(selection.prior[index]),
            "probability_up": float(probability),
        }
        prediction_rows.append(row)
        chain = _hash_chain(chain, row)
    policy_rows = _freeze_trade_policy(selection, selected_probability)
    trade_policy_sha256 = _canonical_sha256(policy_rows)
    freeze_body = {
        "condition_count": len(set(selection.condition_ids)),
        "created_at_ms": created_at_ms,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        ),
        "feature_store_manifest_sha256": feature_manifest["manifest_sha256"],
        "partition_sha256": partition["partition_sha256"],
        "prediction_population_sha256": chain,
        "profitability_claim": False,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION,
        "selected_candidate_id": selected_candidate_id,
        "selection_predictions_frozen": True,
        "trade_policy_sha256": trade_policy_sha256,
    }
    access_freeze = {
        **freeze_body,
        "freeze_sha256": _canonical_sha256(freeze_body),
    }
    validate_round25_forensic_selection_freeze(
        access_freeze,
        partition_manifest=partition,
    )
    prediction_body = {
        "access_freeze": access_freeze,
        "candidate_selection_rule": "strictly_improve_both_calibration_primary_metrics_else_market_prior",
        "model_fit_sha256": model_fit["model_fit_sha256"],
        "prediction_rows": prediction_rows,
        "profitability_claim": False,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_PREDICTION_SCHEMA_VERSION,
        "selection_targets_accessed": False,
        "trade_policy": policy_rows,
    }
    prediction = {
        **prediction_body,
        "prediction_artifact_sha256": _canonical_sha256(prediction_body),
    }
    return validate_round25_forensic_model_fit(model_fit), (
        validate_round25_forensic_prediction_artifact(prediction)
    )


def _freeze_trade_policy(rows: _Rows, probability: np.ndarray) -> list[dict[str, object]]:
    fee_model = PolymarketFeeModel(
        enabled=True,
        rate=Decimal("0.07"),
        exponent=1,
        taker_only=False,
    )
    quantity = Decimal("5")
    output: list[dict[str, object]] = []
    for condition_id in dict.fromkeys(rows.condition_ids):
        indices = [
            index for index, value in enumerate(rows.condition_ids) if value == condition_id
        ]
        candidates: list[tuple[float, int, str, float]] = []
        for index in indices:
            up_price = min(0.99, float(rows.features[index, _UP_ASK]) + 0.01)
            down_price = min(0.99, float(rows.features[index, _DOWN_ASK]) + 0.01)
            if rows.features[index, _UP_ASK_DEPTH] >= 5.0:
                up_fee = fee_model(Decimal(str(up_price)), quantity, "taker")
                candidates.append(
                    (
                        float(probability[index]) - up_price - float(up_fee / quantity),
                        index,
                        "Up",
                        up_price,
                    )
                )
            if rows.features[index, _DOWN_ASK_DEPTH] >= 5.0:
                down_fee = fee_model(Decimal(str(down_price)), quantity, "taker")
                candidates.append(
                    (
                        1.0
                        - float(probability[index])
                        - down_price
                        - float(down_fee / quantity),
                        index,
                        "Down",
                        down_price,
                    )
                )
        eligible = [item for item in candidates if item[0] > 0.01]
        if not eligible:
            output.append({"condition_id": condition_id, "action": "abstain"})
            continue
        edge, index, outcome, price = max(
            eligible,
            key=lambda item: (item[0], -int(rows.decision_time_ms[item[1]]), item[2]),
        )
        output.append(
            {
                "action": "buy",
                "condition_id": condition_id,
                "decision_time_ms": int(rows.decision_time_ms[index]),
                "expected_edge_per_share": edge,
                "outcome": outcome,
                "quantity_shares": 5,
                "stressed_entry_price": price,
            }
        )
    return output


def write_round25_forensic_model_artifacts(
    *,
    model_fit_path: str | Path,
    prediction_path: str | Path,
    model_fit: Mapping[str, object],
    prediction: Mapping[str, object],
) -> tuple[Path, Path]:
    return _write_once(
        model_fit_path,
        validate_round25_forensic_model_fit(model_fit),
    ), _write_once(
        prediction_path,
        validate_round25_forensic_prediction_artifact(prediction),
    )


def validate_round25_forensic_model_fit(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Round 25 forensic model-fit artifact type differs")
    body = dict(value)
    claimed = str(body.pop("model_fit_sha256", "")).strip().lower()
    expected = {
        "calibration_condition_count",
        "candidate_metrics",
        "candidates",
        "evaluation_contract_sha256",
        "feature_store_manifest_sha256",
        "fit_resolution_claim_sha256",
        "live_trading_authority",
        "paper_trading_authority",
        "partition_sha256",
        "profitability_claim",
        "schema_version",
        "selected_candidate_id",
        "selection_targets_accessed",
        "train_condition_count",
    }
    candidates = body.get("candidates")
    metrics = body.get("candidate_metrics")
    selected = body.get("selected_candidate_id")
    metric_names = {
        "balanced_accuracy",
        "condition_equal_brier_score",
        "condition_equal_log_loss",
        "direction_accuracy",
        "expected_calibration_error",
        "roc_auc",
    }
    if (
        set(body) != expected
        or claimed != _canonical_sha256(body)
        or _SHA256.fullmatch(claimed) is None
        or body.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_MODEL_FIT_SCHEMA_VERSION
        or body.get("evaluation_contract_sha256")
        != POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        or any(
            _SHA256.fullmatch(str(body.get(field) or "")) is None
            for field in (
                "feature_store_manifest_sha256",
                "fit_resolution_claim_sha256",
                "partition_sha256",
            )
        )
        or type(body.get("train_condition_count")) is not int
        or body["train_condition_count"] < 1
        or type(body.get("calibration_condition_count")) is not int
        or body["calibration_condition_count"] < 1
        or not isinstance(candidates, Mapping)
        or set(candidates) != set(_CANDIDATES)
        or not isinstance(metrics, Mapping)
        or not {"market-prior-v1", str(selected)} <= set(metrics)
        or not set(metrics) <= set(_CANDIDATES)
        or selected not in _CANDIDATES
        or any(
            not isinstance(metric, Mapping)
            or set(metric) != metric_names
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                for item in metric.values()
            )
            for metric in metrics.values()
        )
        or any(
            body.get(field) is not False
            for field in (
                "live_trading_authority",
                "paper_trading_authority",
                "profitability_claim",
                "selection_targets_accessed",
            )
        )
    ):
        raise ValueError("Round 25 forensic model-fit artifact differs")
    prior = metrics["market-prior-v1"]
    admissible = [
        candidate_id
        for candidate_id, candidate_metric in metrics.items()
        if candidate_id != "market-prior-v1"
        and candidate_metric["condition_equal_log_loss"]
        < prior["condition_equal_log_loss"]
        and candidate_metric["condition_equal_brier_score"]
        < prior["condition_equal_brier_score"]
    ]
    expected_selected = min(
        admissible,
        key=lambda candidate_id: (
            metrics[candidate_id]["condition_equal_log_loss"],
            metrics[candidate_id]["condition_equal_brier_score"],
            _CANDIDATES.index(candidate_id),
        ),
        default="market-prior-v1",
    )
    if (
        selected != expected_selected
        or not isinstance(candidates[selected], Mapping)
        or candidates[selected].get("status") not in {"fitted", "fitted_control"}
    ):
        raise ValueError("Round 25 forensic candidate nomination differs")
    return {**body, "model_fit_sha256": claimed}


def validate_round25_forensic_prediction_artifact(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Round 25 forensic prediction artifact type differs")
    body = dict(value)
    claimed = str(body.pop("prediction_artifact_sha256", "")).strip().lower()
    expected = {
        "access_freeze",
        "candidate_selection_rule",
        "model_fit_sha256",
        "prediction_rows",
        "profitability_claim",
        "schema_version",
        "selection_targets_accessed",
        "trade_policy",
    }
    access = body.get("access_freeze")
    rows = body.get("prediction_rows")
    trades = body.get("trade_policy")
    if (
        set(body) != expected
        or claimed != _canonical_sha256(body)
        or _SHA256.fullmatch(claimed) is None
        or body.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_PREDICTION_SCHEMA_VERSION
        or body.get("candidate_selection_rule")
        != "strictly_improve_both_calibration_primary_metrics_else_market_prior"
        or _SHA256.fullmatch(str(body.get("model_fit_sha256") or "")) is None
        or body.get("profitability_claim") is not False
        or body.get("selection_targets_accessed") is not False
        or not isinstance(access, Mapping)
        or not isinstance(rows, list)
        or not isinstance(trades, list)
        or access.get("selected_candidate_id") not in _CANDIDATES
        or type(access.get("condition_count")) is not int
        or len(rows) != access["condition_count"] * _ENDPOINTS_PER_CONDITION
        or len(trades) != access["condition_count"]
    ):
        raise ValueError("Round 25 forensic prediction artifact differs")
    chain = hashlib.sha256(b"").hexdigest()
    previous: tuple[int, str, int] | None = None
    counts: Counter[str] = Counter()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "condition_id",
                "decision_time_ms",
                "event_start_ms",
                "feature_source_sha256",
                "market_prior_probability_up",
                "probability_up",
            }
            or _CONDITION_ID.fullmatch(str(row.get("condition_id") or "")) is None
            or type(row.get("event_start_ms")) is not int
            or type(row.get("decision_time_ms")) is not int
            or not row["event_start_ms"]
            <= row["decision_time_ms"]
            < row["event_start_ms"] + 300_000
            or _SHA256.fullmatch(str(row.get("feature_source_sha256") or "")) is None
            or isinstance(row.get("market_prior_probability_up"), bool)
            or not isinstance(row.get("market_prior_probability_up"), (int, float))
            or not 0.0 < float(row["market_prior_probability_up"]) < 1.0
            or isinstance(row.get("probability_up"), bool)
            or not isinstance(row.get("probability_up"), (int, float))
            or not math.isfinite(float(row["probability_up"]))
            or not 0.0 <= float(row["probability_up"]) <= 1.0
        ):
            raise ValueError("Round 25 forensic prediction row differs")
        identity = (
            int(row["event_start_ms"]),
            str(row["condition_id"]),
            int(row["decision_time_ms"]),
        )
        if previous is not None and identity <= previous:
            raise ValueError("Round 25 forensic prediction chronology differs")
        previous = identity
        counts[str(row["condition_id"])] += 1
        chain = _hash_chain(chain, row)
    policy_conditions: set[str] = set()
    for trade in trades:
        if not isinstance(trade, Mapping):
            raise ValueError("Round 25 forensic trade policy row differs")
        condition_id = str(trade.get("condition_id") or "")
        action = trade.get("action")
        if (
            _CONDITION_ID.fullmatch(condition_id) is None
            or condition_id in policy_conditions
            or action not in {"abstain", "buy"}
            or (
                action == "abstain"
                and set(trade) != {"action", "condition_id"}
            )
            or (
                action == "buy"
                and (
                    set(trade)
                    != {
                        "action",
                        "condition_id",
                        "decision_time_ms",
                        "expected_edge_per_share",
                        "outcome",
                        "quantity_shares",
                        "stressed_entry_price",
                    }
                    or trade.get("outcome") not in {"Up", "Down"}
                    or trade.get("quantity_shares") != 5
                    or type(trade.get("decision_time_ms")) is not int
                    or not isinstance(trade.get("expected_edge_per_share"), (int, float))
                    or float(trade["expected_edge_per_share"]) <= 0.01
                    or not isinstance(trade.get("stressed_entry_price"), (int, float))
                    or not 0.0 < float(trade["stressed_entry_price"]) <= 0.99
                )
            )
        ):
            raise ValueError("Round 25 forensic trade policy row differs")
        policy_conditions.add(condition_id)
    if (
        any(count != _ENDPOINTS_PER_CONDITION for count in counts.values())
        or set(counts) != policy_conditions
        or chain != access.get("prediction_population_sha256")
        or _canonical_sha256(trades) != access.get("trade_policy_sha256")
    ):
        raise ValueError("Round 25 forensic frozen prediction population differs")
    return {**body, "prediction_artifact_sha256": claimed}


def evaluate_round25_forensic_selection(
    *,
    prediction: Mapping[str, object],
    selection_resolution_database: str | Path,
    created_at_ms: int,
) -> dict[str, object]:
    """Consume sealed selection targets once and report predictive and exact-fee PnL."""

    selected_prediction = validate_round25_forensic_prediction_artifact(prediction)
    claimed = selected_prediction["prediction_artifact_sha256"]
    body = dict(selected_prediction)
    body.pop("prediction_artifact_sha256")
    access = body.get("access_freeze")
    if not isinstance(access, Mapping):
        raise ValueError("Round 25 forensic prediction freeze is unavailable")
    claim, targets = load_round25_forensic_resolution_targets(
        selection_resolution_database
    )
    if (
        claim["stage"] != "selection"
        or claim["selection_freeze_sha256"] != access.get("freeze_sha256")
    ):
        raise ValueError("Round 25 forensic selection target binding differs")
    rows = body.get("prediction_rows")
    trades = body.get("trade_policy")
    if not isinstance(rows, list) or not isinstance(trades, list):
        raise ValueError("Round 25 forensic selection prediction payload differs")
    chain = hashlib.sha256(b"").hexdigest()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "condition_id",
                "decision_time_ms",
                "event_start_ms",
                "feature_source_sha256",
                "market_prior_probability_up",
                "probability_up",
            }
        ):
            raise ValueError("Round 25 forensic selection prediction row differs")
        chain = _hash_chain(chain, row)
    if (
        chain != access.get("prediction_population_sha256")
        or _canonical_sha256(trades) != access.get("trade_policy_sha256")
    ):
        raise ValueError("Round 25 forensic frozen prediction population differs")
    target_map = {item.condition_id: item for item in targets}
    if (
        len(target_map) != len(targets)
        or Counter(str(row["condition_id"]) for row in rows)
        != Counter({condition_id: 16 for condition_id in target_map})
        or len(trades) != len(targets)
        or {str(trade.get("condition_id")) for trade in trades} != set(target_map)
    ):
        raise ValueError("Round 25 forensic selection population differs")
    labels = np.asarray(
        [1.0 if target_map[str(row["condition_id"])].target_up else 0.0 for row in rows],
        dtype=np.float64,
    )
    probability = np.asarray([float(row["probability_up"]) for row in rows])
    prior = np.asarray(
        [float(row["market_prior_probability_up"]) for row in rows],
        dtype=np.float64,
    )
    condition_count = len(targets)
    pseudo_rows = _Rows(
        role="selection",
        condition_ids=tuple(str(row["condition_id"]) for row in rows),
        event_start_ms=np.asarray(
            [int(row["event_start_ms"]) for row in rows], dtype=np.int64
        ),
        decision_time_ms=np.asarray(
            [int(row["decision_time_ms"]) for row in rows], dtype=np.int64
        ),
        features=np.zeros(
            (len(rows), len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES))
        ),
        prior=prior,
        labels=labels,
        source_sha256=tuple(str(row["feature_source_sha256"]) for row in rows),
    ).validated(require_labels=True)
    selected_metrics = _metrics(pseudo_rows, probability)
    prior_metrics = _metrics(pseudo_rows, prior)
    predictive_gate = (
        selected_metrics["condition_equal_log_loss"]
        < prior_metrics["condition_equal_log_loss"]
        and selected_metrics["condition_equal_brier_score"]
        < prior_metrics["condition_equal_brier_score"]
    )
    fee_model = PolymarketFeeModel(
        enabled=True,
        rate=Decimal("0.07"),
        exponent=1,
        taker_only=False,
    )
    pnl: list[float] = []
    committed = 0.0
    trade_results: list[dict[str, object]] = []
    cumulative_pnl = 0.0
    for trade in trades:
        if trade.get("action") != "buy":
            continue
        target = target_map[str(trade["condition_id"])]
        price = Decimal(str(trade["stressed_entry_price"]))
        quantity = Decimal("5")
        fee = fee_model(price, quantity, "taker")
        won = target.target_up is (trade["outcome"] == "Up")
        value = quantity if won else Decimal("0")
        cost = quantity * price + fee
        trade_pnl = float(value - cost)
        pnl.append(trade_pnl)
        committed += float(cost)
        cumulative_pnl += trade_pnl
        trade_results.append(
            {
                "condition_id": target.condition_id,
                "entry_cost_quote": float(cost),
                "event_start_ms": target.event_start_ms,
                "fee_quote": float(fee),
                "net_pnl_quote": trade_pnl,
                "outcome": str(trade["outcome"]),
                "resolved_up": target.target_up,
                "cumulative_net_pnl_quote": cumulative_pnl,
            }
        )
    gains = sum(value for value in pnl if value > 0.0)
    losses = -sum(value for value in pnl if value < 0.0)
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    net_profit = sum(pnl)
    profit_factor = None if losses == 0.0 else gains / losses
    profit_factor_gate = (losses == 0.0 and gains > 0.0) or (
        profit_factor is not None and profit_factor > 1.0
    )
    economic_gate = (
        predictive_gate
        and len(pnl) >= 3
        and net_profit > 0.0
        and profit_factor_gate
        and maximum_drawdown <= max((-value for value in pnl if value < 0.0), default=0.0)
    )
    result_body = {
        "abstention_rate": 1.0 - len(pnl) / condition_count,
        "after_cost_profitability_established": False,
        "closed_trade_count": len(pnl),
        "created_at_ms": created_at_ms,
        "diagnostic_economic_gate_passed": economic_gate,
        "diagnostic_predictive_gate_passed": predictive_gate,
        "evaluation_contract_sha256": POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256,
        "maximum_drawdown_quote": maximum_drawdown,
        "net_profit_quote": net_profit,
        "prediction_artifact_sha256": claimed,
        "profit_factor": profit_factor,
        "profit_factor_infinite": losses == 0.0 and gains > 0.0,
        "profitability_claim": False,
        "return_on_committed_capital": 0.0 if committed == 0.0 else net_profit / committed,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_RESULT_SCHEMA_VERSION,
        "selected_candidate_id": access["selected_candidate_id"],
        "selected_metrics": selected_metrics,
        "market_prior_metrics": prior_metrics,
        "selection_condition_count": condition_count,
        "selection_end_ms": max(item.event_start_ms + 300_000 for item in targets),
        "selection_start_ms": min(item.event_start_ms for item in targets),
        "selection_resolution_claim_sha256": claim["claim_sha256"],
        "statistical_edge_established": False,
        "trade_results": trade_results,
        "win_rate": 0.0 if not pnl else sum(value > 0.0 for value in pnl) / len(pnl),
    }
    return validate_round25_forensic_result(
        {**result_body, "result_sha256": _canonical_sha256(result_body)}
    )


def validate_round25_forensic_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Round 25 forensic result type differs")
    body = dict(value)
    claimed = str(body.pop("result_sha256", "")).strip().lower()
    expected = {
        "abstention_rate",
        "after_cost_profitability_established",
        "closed_trade_count",
        "created_at_ms",
        "diagnostic_economic_gate_passed",
        "diagnostic_predictive_gate_passed",
        "evaluation_contract_sha256",
        "market_prior_metrics",
        "maximum_drawdown_quote",
        "net_profit_quote",
        "prediction_artifact_sha256",
        "profit_factor",
        "profit_factor_infinite",
        "profitability_claim",
        "return_on_committed_capital",
        "schema_version",
        "selected_candidate_id",
        "selected_metrics",
        "selection_condition_count",
        "selection_end_ms",
        "selection_resolution_claim_sha256",
        "selection_start_ms",
        "statistical_edge_established",
        "trade_results",
        "win_rate",
    }
    trades = body.get("trade_results")
    metric_names = {
        "balanced_accuracy",
        "condition_equal_brier_score",
        "condition_equal_log_loss",
        "direction_accuracy",
        "expected_calibration_error",
        "roc_auc",
    }
    if (
        set(body) != expected
        or claimed != _canonical_sha256(body)
        or _SHA256.fullmatch(claimed) is None
        or body.get("schema_version")
        != POLYMARKET_ROUND25_FORENSIC_RESULT_SCHEMA_VERSION
        or body.get("evaluation_contract_sha256")
        != POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256
        or body.get("selected_candidate_id") not in _CANDIDATES
        or type(body.get("created_at_ms")) is not int
        or type(body.get("selection_start_ms")) is not int
        or type(body.get("selection_end_ms")) is not int
        or body["selection_end_ms"] <= body["selection_start_ms"]
        or type(body.get("selection_condition_count")) is not int
        or body["selection_condition_count"] < 8
        or type(body.get("closed_trade_count")) is not int
        or not 0 <= body["closed_trade_count"] <= body["selection_condition_count"]
        or not isinstance(trades, list)
        or len(trades) != body["closed_trade_count"]
        or any(
            _SHA256.fullmatch(str(body.get(field) or "")) is None
            for field in (
                "prediction_artifact_sha256",
                "selection_resolution_claim_sha256",
            )
        )
        or not isinstance(body.get("selected_metrics"), Mapping)
        or set(body["selected_metrics"]) != metric_names
        or not isinstance(body.get("market_prior_metrics"), Mapping)
        or set(body["market_prior_metrics"]) != metric_names
        or any(
            not isinstance(body.get(field), (int, float))
            or isinstance(body.get(field), bool)
            or not math.isfinite(float(body[field]))
            for field in (
                "abstention_rate",
                "maximum_drawdown_quote",
                "net_profit_quote",
                "return_on_committed_capital",
                "win_rate",
            )
        )
        or any(
            body.get(field) is not False
            for field in (
                "after_cost_profitability_established",
                "profitability_claim",
                "statistical_edge_established",
            )
        )
        or any(
            type(body.get(field)) is not bool
            for field in (
                "diagnostic_economic_gate_passed",
                "diagnostic_predictive_gate_passed",
                "profit_factor_infinite",
            )
        )
    ):
        raise ValueError("Round 25 forensic result differs")
    cumulative = 0.0
    for trade in trades:
        if (
            not isinstance(trade, Mapping)
            or set(trade)
            != {
                "condition_id",
                "cumulative_net_pnl_quote",
                "entry_cost_quote",
                "event_start_ms",
                "fee_quote",
                "net_pnl_quote",
                "outcome",
                "resolved_up",
            }
            or _CONDITION_ID.fullmatch(str(trade.get("condition_id") or "")) is None
            or trade.get("outcome") not in {"Up", "Down"}
            or type(trade.get("resolved_up")) is not bool
            or type(trade.get("event_start_ms")) is not int
            or not body["selection_start_ms"]
            <= trade["event_start_ms"]
            < body["selection_end_ms"]
            or any(
                not isinstance(trade.get(field), (int, float))
                or isinstance(trade.get(field), bool)
                or not math.isfinite(float(trade[field]))
                for field in (
                    "cumulative_net_pnl_quote",
                    "entry_cost_quote",
                    "fee_quote",
                    "net_pnl_quote",
                )
            )
        ):
            raise ValueError("Round 25 forensic result trade differs")
        cumulative += float(trade["net_pnl_quote"])
        if not math.isclose(
            cumulative,
            float(trade["cumulative_net_pnl_quote"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Round 25 forensic result trade accounting differs")
    if (
        not math.isclose(
            cumulative,
            float(body["net_profit_quote"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(body["abstention_rate"]),
            1.0 - len(trades) / int(body["selection_condition_count"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Round 25 forensic result accounting differs")
    return {**body, "result_sha256": claimed}


def write_round25_forensic_result(
    path: str | Path,
    result: Mapping[str, object],
) -> Path:
    return _write_once(path, validate_round25_forensic_result(result))


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_MODEL_FIT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_FORENSIC_PREDICTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_FORENSIC_RESULT_SCHEMA_VERSION",
    "evaluate_round25_forensic_selection",
    "fit_and_freeze_round25_forensic_models",
    "validate_round25_forensic_model_fit",
    "validate_round25_forensic_result",
    "validate_round25_forensic_prediction_artifact",
    "write_round25_forensic_model_artifacts",
    "write_round25_forensic_result",
]

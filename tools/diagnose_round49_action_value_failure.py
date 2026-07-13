"""Diagnose why Round 49 probability skill did not become robust action value."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.action_hurdle_tcn_model import (  # noqa: E402
    PRIMARY_HORIZON_INDEX,
    SIDES,
    build_action_hurdle_temporal_dataset,
    side_net_targets,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA = "round-049-action-value-failure-diagnosis-v1"
REPORT_SCHEMA = "cost-aware-action-hurdle-tcn-report-v1"
EXPECTED_REPORT_CANONICAL_SHA256 = (
    "d07ce85ad0b63e292369d59d5a0c93610c34df17dd73be066d94fe6254a09417"
)
EXPECTED_REPORT_FILE_SHA256 = (
    "11f0a61a8bca1fcb5940df5f883c25bd970557a6dd2e875d6565aabbb9dfd9a2"
)
EXPECTED_DATASET_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
ANALYSIS_ROLES = ("calibration", "viability")
CANDIDATES = ("direct_action_mean_tcn", "hurdle_action_value_tcn")
TAIL_COUNTS = (100, 500, 1_000)
TAIL_FRACTIONS = (0.01, 0.05, 0.10)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _event(stage: str, **details: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **details},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _progress(stage: str, payload: Mapping[str, object]) -> None:
    _event(stage, **dict(payload))


def _finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _spearman(actual: np.ndarray, prediction: np.ndarray) -> float | None:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    finite = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[finite]
    prediction = prediction[finite]
    if actual.size < 2 or np.ptp(actual) == 0.0 or np.ptp(prediction) == 0.0:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranked = np.empty(values.size, dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
        ends = np.r_[starts[1:], values.size]
        for start, end in zip(starts, ends, strict=True):
            ranked[order[start:end]] = 0.5 * (start + end - 1)
        return ranked

    return _finite_float(float(np.corrcoef(ranks(actual), ranks(prediction))[0, 1]))


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], scores.size]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = 0.5 * (start + end + 1)
    value = (float(ranks[labels].sum()) - positive * (positive + 1) / 2.0) / (
        positive * negative
    )
    return _finite_float(value)


def _summary(actual: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    finite = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[finite]
    prediction = prediction[finite]
    if actual.size == 0:
        raise ValueError("diagnostic summary is empty")
    error = prediction - actual
    return {
        "rows": int(actual.size),
        "actual_mean_bps": float(np.mean(actual)),
        "actual_median_bps": float(np.median(actual)),
        "actual_profit_rate": float(np.mean(actual > 0.0)),
        "prediction_mean_bps": float(np.mean(prediction)),
        "prediction_median_bps": float(np.median(prediction)),
        "prediction_standard_deviation_bps": float(np.std(prediction)),
        "bias_bps": float(np.mean(error)),
        "mean_absolute_error_bps": float(np.mean(np.abs(error))),
        "root_mean_squared_error_bps": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": _spearman(actual, prediction),
    }


def _outcome_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("outcome summary is empty")
    gains = values[values > 0.0]
    losses = -values[values <= 0.0]
    return {
        "rows": int(values.size),
        "mean_net_bps": float(np.mean(values)),
        "standard_deviation_bps": float(np.std(values)),
        "profit_rate": float(gains.size / values.size),
        "conditional_gain_mean_bps": float(np.mean(gains)),
        "conditional_loss_mean_bps": float(np.mean(losses)),
        "absolute_p95_bps": float(np.quantile(np.abs(values), 0.95)),
    }


def _load_report(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_REPORT_FILE_SHA256:
        raise ValueError("Round 49 report file hash differs from the frozen diagnosis")
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise TypeError("Round 49 report must be an object")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if claimed != _canonical_sha256(canonical):
        raise ValueError("Round 49 report canonical hash is invalid")
    if claimed != EXPECTED_REPORT_CANONICAL_SHA256:
        raise ValueError("Round 49 report is not the frozen source report")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("round") != 49:
        raise ValueError("Round 49 report schema or round is invalid")
    if report.get("profitability_claim") is not False:
        raise ValueError(
            "Round 49 source report contains an invalid profitability claim"
        )
    return report


def _validated_artifacts(
    report: Mapping[str, object],
) -> dict[tuple[str, int], Path]:
    entries = report.get("external_artifacts")
    if not isinstance(entries, list) or len(entries) != 12:
        raise ValueError("Round 49 external artifact manifest is invalid")
    predictions: dict[tuple[str, int], Path] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            raise TypeError("Round 49 external artifact entry is invalid")
        path = Path(str(item["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Round 49 artifact size differs: {path}")
        if _file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Round 49 artifact hash differs: {path}")
        if item["kind"] == "predictions":
            key = (str(item["candidate_id"]), int(item["seed"]))
            if key in predictions:
                raise ValueError(f"duplicate prediction artifact: {key}")
            predictions[key] = path
    if set(candidate for candidate, _ in predictions) != set(CANDIDATES):
        raise ValueError("Round 49 prediction candidate set is invalid")
    if len(predictions) != 6:
        raise ValueError("Round 49 prediction artifact count is invalid")
    return predictions


def _load_predictions(
    paths: Mapping[tuple[str, int], Path],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    common_indices: np.ndarray | None = None
    for candidate_id in CANDIDATES:
        seed_paths = sorted(
            (seed, path)
            for (candidate, seed), path in paths.items()
            if candidate == candidate_id
        )
        arrays: dict[str, list[np.ndarray]] = {
            "probabilities": [],
            "action_values_bps": [],
            "gain_means_bps": [],
            "loss_means_bps": [],
        }
        for seed, path in seed_paths:
            with np.load(path, allow_pickle=False) as artifact:
                indices = np.asarray(artifact["global_indices"], dtype=np.int64)
                if common_indices is None:
                    common_indices = indices.copy()
                elif not np.array_equal(common_indices, indices):
                    raise ValueError(
                        "Round 49 prediction indices differ across artifacts"
                    )
                for key in arrays:
                    value = np.asarray(artifact[key], dtype=np.float64)
                    if key in ("gain_means_bps", "loss_means_bps") and value.size == 0:
                        continue
                    if not np.isfinite(value).all():
                        raise ValueError(
                            f"Round 49 nonfinite prediction: {candidate_id}/{seed}/{key}"
                        )
                    arrays[key].append(value)
        if len(arrays["probabilities"]) != 3 or len(arrays["action_values_bps"]) != 3:
            raise ValueError(
                f"Round 49 candidate seed set is incomplete: {candidate_id}"
            )
        result[candidate_id] = {
            key: np.stack(value, axis=0) for key, value in arrays.items() if value
        }
    if common_indices is None:
        raise ValueError("Round 49 prediction indices are absent")
    return common_indices, result


def _target_geometry(
    primary_targets: np.ndarray,
    role_masks: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    training_gain = np.empty((len(SYMBOLS), len(SIDES)), dtype=np.float64)
    training_loss = np.empty_like(training_gain)
    for role, role_mask in role_masks.items():
        for symbol_index, symbol in enumerate(SYMBOLS):
            for side_index, side in enumerate(SIDES):
                summary = _outcome_summary(
                    primary_targets[role_mask, symbol_index, side_index]
                )
                rows.append(
                    {
                        "role": role,
                        "symbol": symbol,
                        "side": "short" if side == -1 else "long",
                        **summary,
                    }
                )
                if role == "training":
                    training_gain[symbol_index, side_index] = float(
                        summary["conditional_gain_mean_bps"]
                    )
                    training_loss[symbol_index, side_index] = float(
                        summary["conditional_loss_mean_bps"]
                    )
    return rows, training_gain, training_loss


def _scope_slices(
    role_rows: np.ndarray,
    symbol_count: int,
) -> list[tuple[str, str, tuple[np.ndarray, np.ndarray]]]:
    result: list[tuple[str, str, tuple[np.ndarray, np.ndarray]]] = []
    local_rows = np.flatnonzero(role_rows)
    pooled_rows = np.repeat(local_rows, symbol_count)
    pooled_symbols = np.tile(np.arange(symbol_count), local_rows.size)
    result.append(("pooled", "ALL", (pooled_rows, pooled_symbols)))
    for symbol_index, symbol in enumerate(SYMBOLS):
        result.append(
            (
                "symbol",
                symbol,
                (local_rows, np.full(local_rows.size, symbol_index, dtype=np.int64)),
            )
        )
    return result


def _prediction_diagnostics(
    indices: np.ndarray,
    timestamps_ms: np.ndarray,
    role_masks: Mapping[str, np.ndarray],
    actual: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    quality: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for candidate_id, candidate in predictions.items():
        seed_action = candidate["action_values_bps"]
        seed_probability = candidate["probabilities"][..., PRIMARY_HORIZON_INDEX, :]
        for role in ANALYSIS_ROLES:
            role_rows = role_masks[role][indices]
            for scope, symbol, (rows, symbols) in _scope_slices(
                role_rows, len(SYMBOLS)
            ):
                for side_index, side in enumerate(SIDES):
                    target = actual[rows, symbols, side_index]
                    action = seed_action[:, rows, symbols, side_index]
                    probability = seed_probability[:, rows, symbols, side_index]
                    ensemble_action = np.mean(action, axis=0)
                    ensemble_probability = np.mean(probability, axis=0)
                    worst_action = np.min(action, axis=0)
                    all_positive = np.all(action > 0.0, axis=0)
                    row = {
                        "candidate_id": candidate_id,
                        "role": role,
                        "scope": scope,
                        "symbol": symbol,
                        "side": "short" if side == -1 else "long",
                        **_summary(target, ensemble_action),
                        "probability_roc_auc": _roc_auc(
                            target > 0.0, ensemble_probability
                        ),
                        "worst_seed_action_spearman": _spearman(target, worst_action),
                        "all_seed_positive_fraction": float(np.mean(all_positive)),
                        "all_seed_positive_rows": int(np.sum(all_positive)),
                        "all_seed_positive_actual_mean_bps": (
                            float(np.mean(target[all_positive]))
                            if np.any(all_positive)
                            else None
                        ),
                        "all_seed_positive_actual_profit_rate": (
                            float(np.mean(target[all_positive] > 0.0))
                            if np.any(all_positive)
                            else None
                        ),
                    }
                    quality.append(row)
                    pairs = []
                    for left in range(action.shape[0]):
                        for right in range(left + 1, action.shape[0]):
                            pairs.append(_spearman(action[left], action[right]))
                    finite_pairs = [value for value in pairs if value is not None]
                    stability.append(
                        {
                            "candidate_id": candidate_id,
                            "role": role,
                            "scope": scope,
                            "symbol": symbol,
                            "side": row["side"],
                            "minimum_pairwise_action_spearman": (
                                min(finite_pairs) if finite_pairs else None
                            ),
                            "mean_pairwise_action_spearman": (
                                float(np.mean(finite_pairs)) if finite_pairs else None
                            ),
                            "all_seed_sign_agreement_fraction": float(
                                np.mean(
                                    np.all(action > 0.0, axis=0)
                                    | np.all(action <= 0.0, axis=0)
                                )
                            ),
                            "mean_seed_standard_deviation_bps": float(
                                np.mean(np.std(action, axis=0))
                            ),
                        }
                    )
            months = np.array(
                [
                    datetime.fromtimestamp(value / 1_000, UTC).strftime("%Y-%m")
                    for value in timestamps_ms[indices]
                ]
            )
            for month in sorted(set(months[role_rows])):
                month_rows = role_rows & (months == month)
                rows = np.flatnonzero(month_rows)
                symbols = np.tile(np.arange(len(SYMBOLS)), rows.size)
                rows = np.repeat(rows, len(SYMBOLS))
                for side_index, side in enumerate(SIDES):
                    target = actual[rows, symbols, side_index]
                    action = np.mean(seed_action[:, rows, symbols, side_index], axis=0)
                    quality.append(
                        {
                            "candidate_id": candidate_id,
                            "role": role,
                            "scope": "month",
                            "symbol": month,
                            "side": "short" if side == -1 else "long",
                            **_summary(target, action),
                        }
                    )
    return quality, stability


def _tail_summary(
    actual: np.ndarray,
    score: np.ndarray,
    label: str,
) -> list[dict[str, object]]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(score, kind="mergesort")
    rows: list[dict[str, object]] = []
    for bin_index, selected in enumerate(np.array_split(order, 10), start=1):
        values = actual[selected]
        rows.append(
            {
                "selection": label,
                "kind": "equal_count_decile",
                "tail": bin_index,
                "rows": int(selected.size),
                "score_min_bps": float(np.min(score[selected])),
                "score_mean_bps": float(np.mean(score[selected])),
                "score_max_bps": float(np.max(score[selected])),
                "actual_mean_bps": float(np.mean(values)),
                "actual_median_bps": float(np.median(values)),
                "actual_profit_rate": float(np.mean(values > 0.0)),
            }
        )
    for count in TAIL_COUNTS:
        selected = order[-min(count, order.size) :]
        values = actual[selected]
        rows.append(
            {
                "selection": label,
                "kind": "top_count",
                "tail": int(selected.size),
                "rows": int(selected.size),
                "score_min_bps": float(np.min(score[selected])),
                "score_mean_bps": float(np.mean(score[selected])),
                "score_max_bps": float(np.max(score[selected])),
                "actual_mean_bps": float(np.mean(values)),
                "actual_median_bps": float(np.median(values)),
                "actual_profit_rate": float(np.mean(values > 0.0)),
            }
        )
    for fraction in TAIL_FRACTIONS:
        count = max(1, int(np.ceil(order.size * fraction)))
        selected = order[-count:]
        values = actual[selected]
        rows.append(
            {
                "selection": label,
                "kind": "top_fraction",
                "tail": fraction,
                "rows": int(selected.size),
                "score_min_bps": float(np.min(score[selected])),
                "score_mean_bps": float(np.mean(score[selected])),
                "score_max_bps": float(np.max(score[selected])),
                "actual_mean_bps": float(np.mean(values)),
                "actual_median_bps": float(np.median(values)),
                "actual_profit_rate": float(np.mean(values > 0.0)),
            }
        )
    return rows


def _tail_diagnostics(
    indices: np.ndarray,
    role_masks: Mapping[str, np.ndarray],
    actual: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_id, candidate in predictions.items():
        seed_action = candidate["action_values_bps"]
        worst = np.min(seed_action, axis=0)
        for role in ANALYSIS_ROLES:
            role_rows = np.flatnonzero(role_masks[role][indices])
            for side_index, side in enumerate(SIDES):
                target = actual[role_rows, :, side_index].reshape(-1)
                score = worst[role_rows, :, side_index].reshape(-1)
                for item in _tail_summary(target, score, "fixed_side"):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "role": role,
                            "side": "short" if side == -1 else "long",
                            **item,
                        }
                    )
            role_worst = worst[role_rows]
            chosen_side = np.argmax(role_worst, axis=-1)
            chosen_score = np.take_along_axis(
                role_worst, chosen_side[..., None], axis=-1
            ).squeeze(-1)
            chosen_actual = np.take_along_axis(
                actual[role_rows], chosen_side[..., None], axis=-1
            ).squeeze(-1)
            for item in _tail_summary(
                chosen_actual.reshape(-1),
                chosen_score.reshape(-1),
                "best_worst_seed_side",
            ):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "role": role,
                        "side": "chosen",
                        **item,
                    }
                )
    return rows


def _severity_decomposition(
    indices: np.ndarray,
    role_masks: Mapping[str, np.ndarray],
    actual: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    target_geometry: Sequence[Mapping[str, object]],
    training_gain: np.ndarray,
    training_loss: np.ndarray,
) -> list[dict[str, object]]:
    hurdle = predictions["hurdle_action_value_tcn"]
    probability = np.mean(
        hurdle["probabilities"][..., PRIMARY_HORIZON_INDEX, :], axis=0
    )
    gain = np.mean(hurdle["gain_means_bps"], axis=0)
    loss = np.mean(hurdle["loss_means_bps"], axis=0)
    hurdle_action = np.mean(hurdle["action_values_bps"], axis=0)
    direct_action = np.mean(
        predictions["direct_action_mean_tcn"]["action_values_bps"], axis=0
    )
    geometry_lookup = {
        (str(row["role"]), str(row["symbol"]), str(row["side"])): row
        for row in target_geometry
    }
    rows: list[dict[str, object]] = []
    for role in ANALYSIS_ROLES:
        role_mask = role_masks[role][indices]
        for scope, symbol, (time_rows, symbols) in _scope_slices(
            role_mask, len(SYMBOLS)
        ):
            for side_index, side in enumerate(SIDES):
                side_name = "short" if side == -1 else "long"
                target = actual[time_rows, symbols, side_index]
                p = probability[time_rows, symbols, side_index]
                predicted_gain = gain[time_rows, symbols, side_index]
                predicted_loss = loss[time_rows, symbols, side_index]
                predicted_action = hurdle_action[time_rows, symbols, side_index]
                control_action = direct_action[time_rows, symbols, side_index]
                train_gain = training_gain[symbols, side_index]
                train_loss = training_loss[symbols, side_index]
                role_gain = np.empty_like(train_gain)
                role_loss = np.empty_like(train_loss)
                for index, symbol_index in enumerate(symbols):
                    geometry = geometry_lookup[
                        (role, SYMBOLS[int(symbol_index)], side_name)
                    ]
                    role_gain[index] = float(geometry["conditional_gain_mean_bps"])
                    role_loss[index] = float(geometry["conditional_loss_mean_bps"])
                training_constant_action = p * train_gain - (1.0 - p) * train_loss
                role_oracle_constant_action = p * role_gain - (1.0 - p) * role_loss
                positive = target > 0.0
                negative = ~positive
                row: dict[str, object] = {
                    "role": role,
                    "scope": scope,
                    "symbol": symbol,
                    "side": side_name,
                    "rows": int(target.size),
                    "probability_roc_auc": _roc_auc(positive, p),
                    "hurdle_action": _summary(target, predicted_action),
                    "direct_action": _summary(target, control_action),
                    "probability_with_training_constant_severity": _summary(
                        target, training_constant_action
                    ),
                    "probability_with_same_role_oracle_constant_severity": _summary(
                        target, role_oracle_constant_action
                    ),
                    "predicted_probability_standard_deviation": float(np.std(p)),
                    "predicted_gain_standard_deviation_bps": float(
                        np.std(predicted_gain)
                    ),
                    "predicted_loss_standard_deviation_bps": float(
                        np.std(predicted_loss)
                    ),
                    "probability_gain_spearman": _spearman(p, predicted_gain),
                    "probability_loss_spearman": _spearman(p, predicted_loss),
                    "gain_loss_spearman": _spearman(predicted_gain, predicted_loss),
                    "conditional_gain_rows": int(np.sum(positive)),
                    "conditional_loss_rows": int(np.sum(negative)),
                    "conditional_gain_spearman": _spearman(
                        target[positive], predicted_gain[positive]
                    ),
                    "conditional_loss_spearman": _spearman(
                        -target[negative], predicted_loss[negative]
                    ),
                    "conditional_gain_mae_bps": float(
                        np.mean(np.abs(predicted_gain[positive] - target[positive]))
                    ),
                    "conditional_loss_mae_bps": float(
                        np.mean(np.abs(predicted_loss[negative] + target[negative]))
                    ),
                }
                rows.append(row)
    return rows


def _report_policy_summary(report: Mapping[str, object]) -> list[dict[str, object]]:
    candidates = report.get("candidate_results")
    if not isinstance(candidates, list):
        raise TypeError("Round 49 candidate results are invalid")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("Round 49 candidate result is invalid")
        base = candidate["base"]
        stress = candidate["stress"]
        if not isinstance(base, Mapping) or not isinstance(stress, Mapping):
            raise TypeError("Round 49 replay result is invalid")
        base_metrics = base["metrics"]
        stress_metrics = stress["metrics"]
        if not isinstance(base_metrics, Mapping) or not isinstance(
            stress_metrics, Mapping
        ):
            raise TypeError("Round 49 replay metrics are invalid")
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "base": dict(base_metrics),
                "stress": dict(stress_metrics),
                "combined_quality_gate_passed": candidate[
                    "combined_quality_gate_passed"
                ],
                "economic_gate_passed": candidate["economic_gate"]["passed"],
            }
        )
    return rows


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    report_path = arguments.report.resolve()
    _event("report_validation_started", report=str(report_path))
    report = _load_report(report_path)
    artifact_paths = _validated_artifacts(report)
    indices, predictions = _load_predictions(artifact_paths)
    _event(
        "artifacts_validated",
        artifacts=12,
        prediction_rows=int(indices.size),
        candidates=len(predictions),
    )

    panel, price_source = load_verified_minute_panel(
        arguments.database.resolve(), progress=_progress
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=_progress,
    )
    hurdle_dataset = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=_progress,
    )
    dataset = build_action_hurdle_temporal_dataset(hurdle_dataset)
    del panel, premium, funding, hurdle_dataset
    gc.collect()
    if dataset.dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("Round 49 diagnosis reconstructed a different dataset")
    if report["dataset"]["dataset_sha256"] != dataset.dataset_sha256:
        raise ValueError("Round 49 diagnosis dataset differs from the source report")
    if np.any(indices < 0) or np.any(indices >= dataset.timestamps):
        raise ValueError("Round 49 prediction indices are outside the dataset")
    primary_targets = side_net_targets(dataset)[..., PRIMARY_HORIZON_INDEX, :]
    actual = primary_targets[indices].copy()
    timestamps_ms = dataset.timestamps_ms.copy()
    role_masks = {key: value.copy() for key, value in dataset.role_masks.items()}
    target_geometry, training_gain, training_loss = _target_geometry(
        primary_targets, role_masks
    )
    del primary_targets, dataset
    gc.collect()
    expected_analysis_rows = sum(
        int(role_masks[role][indices].sum()) for role in ANALYSIS_ROLES
    )
    if expected_analysis_rows != indices.size:
        raise ValueError("Round 49 artifacts contain rows outside analysis roles")
    _event(
        "dataset_validated",
        dataset_sha256=EXPECTED_DATASET_SHA256,
        timestamps=int(timestamps_ms.size),
        prediction_rows=int(indices.size),
    )

    quality, stability = _prediction_diagnostics(
        indices,
        timestamps_ms,
        role_masks,
        actual,
        predictions,
    )
    _event("prediction_diagnostics_complete", rows=len(quality))
    tails = _tail_diagnostics(indices, role_masks, actual, predictions)
    _event("tail_diagnostics_complete", rows=len(tails))
    decomposition = _severity_decomposition(
        indices,
        role_masks,
        actual,
        predictions,
        target_geometry,
        training_gain,
        training_loss,
    )
    _event("severity_decomposition_complete", rows=len(decomposition))

    output: dict[str, object] = {
        "schema_version": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "diagnosis_sha256": "PENDING",
        "round": 49,
        "purpose": (
            "Immutable consumed-development diagnosis for Round 50 model design; "
            "not policy selection, promotion evidence, or a profitability claim."
        ),
        "source_report_canonical_sha256": EXPECTED_REPORT_CANONICAL_SHA256,
        "source_report_file_sha256": EXPECTED_REPORT_FILE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "selection_contaminated": True,
        "profitability_claim": False,
        "trading_authority": False,
        "artifact_validation": {
            "model_artifacts": 6,
            "prediction_artifacts": 6,
            "all_sizes_and_sha256_verified": True,
        },
        "target_geometry": target_geometry,
        "prediction_quality": quality,
        "seed_stability": stability,
        "tail_diagnostics": tails,
        "severity_decomposition": decomposition,
        "source_policy_results": _report_policy_summary(report),
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "persistent_derived_feature_copy_created": False,
        },
    }
    canonical = dict(output)
    canonical.pop("diagnosis_sha256")
    output["diagnosis_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(arguments.output.resolve(), output, indent=2, sort_keys=True)
    _event(
        "diagnosis_complete",
        output=str(arguments.output.resolve()),
        diagnosis_sha256=output["diagnosis_sha256"],
        elapsed_seconds=output["runtime"]["elapsed_seconds"],
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sealed screening and winner-only confirmation for tape/depth trials."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model_experiment import (
    EXPERIMENT_DESIGN_CONTRACT,
    validate_experiment_design_payload,
)
from .storage import write_json_atomic
from .tape_depth_prequential import (
    TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
    verify_tape_depth_prequential_report,
)


TAPE_DEPTH_SELECTION_SCHEMA_VERSION = "tape-depth-screening-selection-v3"
TAPE_DEPTH_CONFIRMATION_SCHEMA_VERSION = "tape-depth-sealed-confirmation-v3"
_TRIAL_CONFIG_KEYS = frozenset(
    {
        "model_profile",
        "feature_set",
        "horizon_seconds",
        "decision_cadence_seconds",
        "maximum_depth_age_ms",
    }
)
_STAGE_CONFIG_KEYS = frozenset(
    {
        *_TRIAL_CONFIG_KEYS,
        "study_stage",
        "fold_start",
        "max_folds",
        "selection_lock_sha256",
        "dataset_cache",
        "maximum_cached_rows",
    }
)
_PROFILE_COMPLEXITY = {"regularized": 0, "balanced": 1, "expressive": 2}
_FEATURE_COMPLEXITY = {"core": 0, "tape_derived": 1, "cross_asset": 2, "full": 3}
_MAX_JSON_EVIDENCE_BYTES = 64 * 1024 * 1024
_SCREENING_FOLD_COUNTS = frozenset({4, 6, 8, 10})
_FORECAST_SELECTION_PBO_LIMIT = 0.20
_SELECTION_METRIC_NAMES = (
    "auc_edge",
    "brier_improvement_ratio",
    "mae_improvement_ratio",
    "spearman_information_coefficient",
    "calibration_threshold_mean_signed_gross_bps",
    "positive_ic_fold_rate",
    "positive_gross_fold_rate",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload_fingerprint(payload: Mapping[str, object], field: str) -> str:
    contract = {str(key): value for key, value in payload.items() if str(key) != field}
    return _sha256_bytes(_canonical_json(contract).encode("ascii"))


def _with_fingerprint(payload: Mapping[str, object], field: str) -> dict[str, object]:
    output = dict(payload)
    output.pop(field, None)
    output[field] = _payload_fingerprint(output, field)
    return output


@dataclass(frozen=True)
class TrialKey:
    horizon_seconds: int
    decision_cadence_seconds: int
    maximum_depth_age_ms: int
    model_profile: str
    feature_set: str

    @property
    def label(self) -> str:
        return (
            f"h{self.horizon_seconds}-c{self.decision_cadence_seconds}-"
            f"d{self.maximum_depth_age_ms}/{self.model_profile}/{self.feature_set}"
        )

    @property
    def dataset_key(self) -> tuple[int, int, int]:
        return (
            self.horizon_seconds,
            self.decision_cadence_seconds,
            self.maximum_depth_age_ms,
        )

    def asdict(self) -> dict[str, object]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "decision_cadence_seconds": self.decision_cadence_seconds,
            "maximum_depth_age_ms": self.maximum_depth_age_ms,
            "model_profile": self.model_profile,
            "feature_set": self.feature_set,
        }

    @property
    def complexity(self) -> int:
        return _PROFILE_COMPLEXITY[self.model_profile] + _FEATURE_COMPLEXITY[
            self.feature_set
        ]


def _finite(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"comparison metric {name} is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"comparison metric {name} is non-finite")
    return parsed


def _config_without(
    config: Mapping[str, object], ignored: frozenset[str]
) -> dict[str, object]:
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in ignored
    }


def _mapping_of_sha256(
    value: object,
    *,
    expected_symbols: Sequence[str],
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison {name} are invalid")
    output = {str(key): str(item) for key, item in value.items()}
    if set(output) != set(expected_symbols) or not all(
        _is_sha256(item) for item in output.values()
    ):
        raise ValueError(f"comparison {name} are invalid")
    return output


def _mapping_of_positive_ints(
    value: object,
    *,
    expected_symbols: Sequence[str],
    name: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison {name} are invalid")
    try:
        output = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"comparison {name} are invalid") from exc
    if set(output) != set(expected_symbols) or any(item < 1 for item in output.values()):
        raise ValueError(f"comparison {name} are invalid")
    return output


def _validate_report(
    report: Mapping[str, object],
) -> tuple[TrialKey, list[dict[str, object]]]:
    if report.get("schema_version") != TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION:
        raise ValueError("comparison input report schema is unsupported")
    if (
        report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
    ):
        raise ValueError("comparison input report carries forbidden authority")
    config = report.get("config")
    folds = report.get("folds")
    if not isinstance(config, Mapping) or not isinstance(folds, list) or not folds:
        raise ValueError("comparison input report is incomplete")
    if (
        int(report.get("total_folds", -1)) != len(folds)
        or int(report.get("completed_folds", -1)) != len(folds)
    ):
        raise ValueError("comparison input report is not a complete fold run")
    profile = str(config.get("model_profile") or "")
    feature_set = str(config.get("feature_set") or "")
    if profile not in _PROFILE_COMPLEXITY or feature_set not in _FEATURE_COMPLEXITY:
        raise ValueError("comparison trial profile or feature set is unsupported")
    try:
        horizon_seconds = int(config.get("horizon_seconds", -1))
        decision_cadence_seconds = int(config.get("decision_cadence_seconds", -1))
        maximum_depth_age_ms = int(config.get("maximum_depth_age_ms", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("comparison trial timing configuration is invalid") from exc
    if (
        not 1 <= horizon_seconds <= 3_600
        or not 1 <= decision_cadence_seconds <= 60
        or 60 % decision_cadence_seconds != 0
        or not 1_000 <= maximum_depth_age_ms <= 300_000
    ):
        raise ValueError("comparison trial timing configuration is invalid")
    stage = str(config.get("study_stage") or "")
    if stage not in {"development", "screening", "confirmation"}:
        raise ValueError("comparison report study stage is invalid")
    try:
        fold_start = int(config.get("fold_start", -1))
        max_folds = int(config.get("max_folds", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("comparison report fold window is invalid") from exc
    if fold_start < 0 or max_folds < 0:
        raise ValueError("comparison report fold window is invalid")
    lock_hash = config.get("selection_lock_sha256")
    if (stage == "confirmation") != _is_sha256(lock_hash):
        raise ValueError("comparison report selection-lock binding is invalid")
    configured_symbols = config.get("symbols")
    if not isinstance(configured_symbols, list) or not configured_symbols:
        raise ValueError("comparison symbols are invalid")
    expected_symbols = tuple(str(symbol) for symbol in configured_symbols)
    if len(set(expected_symbols)) != len(expected_symbols) or any(
        not symbol for symbol in expected_symbols
    ):
        raise ValueError("comparison symbols are invalid")
    _mapping_of_sha256(
        report.get("plan_fingerprints"),
        expected_symbols=expected_symbols,
        name="plan fingerprints",
    )
    _mapping_of_sha256(
        report.get("coverage_fingerprints"),
        expected_symbols=expected_symbols,
        name="coverage fingerprints",
    )
    available_counts = _mapping_of_positive_ints(
        report.get("available_fold_counts"),
        expected_symbols=expected_symbols,
        name="available fold counts",
    )
    normalized_folds: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping):
            raise ValueError("comparison fold must be an object")
        fold = dict(raw_fold)
        try:
            fold_index = int(fold.get("fold_index", -1))
            evaluation_start_ms = int(fold.get("evaluation_start_ms", -1))
            evaluation_end_ms = int(fold.get("evaluation_end_ms", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("comparison fold identity or evidence is invalid") from exc
        identity = (str(fold.get("symbol") or ""), fold_index)
        metrics = fold.get("metrics")
        if (
            identity[0] not in expected_symbols
            or fold_index < 0
            or evaluation_start_ms < 0
            or evaluation_end_ms < evaluation_start_ms
            or identity in seen
            or not isinstance(metrics, Mapping)
            or not _is_sha256(fold.get("dataset_fingerprint"))
            or fold.get("status") not in {"research_candidate", "rejected"}
        ):
            raise ValueError("comparison fold identity or evidence is invalid")
        seen.add(identity)
        fold["fold_index"] = fold_index
        fold["evaluation_start_ms"] = evaluation_start_ms
        fold["evaluation_end_ms"] = evaluation_end_ms
        normalized_folds.append(fold)
    normalized_folds.sort(
        key=lambda fold: (
            str(fold["symbol"]),
            int(fold["evaluation_start_ms"]),
            int(fold["fold_index"]),
        )
    )
    for symbol in expected_symbols:
        symbol_folds = [fold for fold in normalized_folds if fold["symbol"] == symbol]
        expected_indices = list(range(fold_start, fold_start + len(symbol_folds)))
        if [int(fold["fold_index"]) for fold in symbol_folds] != expected_indices:
            raise ValueError(f"comparison folds are incomplete for {symbol}")
        if fold_start + len(symbol_folds) > available_counts[symbol]:
            raise ValueError(f"comparison folds exceed available coverage for {symbol}")
        if any(
            int(current["evaluation_start_ms"])
            <= int(previous["evaluation_end_ms"])
            for previous, current in zip(symbol_folds, symbol_folds[1:])
        ):
            raise ValueError(f"comparison evaluation folds overlap for {symbol}")
    return (
        TrialKey(
            horizon_seconds=horizon_seconds,
            decision_cadence_seconds=decision_cadence_seconds,
            maximum_depth_age_ms=maximum_depth_age_ms,
            model_profile=profile,
            feature_set=feature_set,
        ),
        normalized_folds,
    )


def _segment_metrics(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not folds:
        raise ValueError("comparison segment has no folds")
    rows = np.asarray(
        [int(dict(fold["metrics"])["rows"]) for fold in folds],
        dtype=np.float64,
    )
    if np.any(rows <= 0):
        raise ValueError("comparison segment has a non-positive row count")

    def weighted(name: str) -> float:
        values = np.asarray(
            [_finite(dict(fold["metrics"])[name], name) for fold in folds],
            dtype=np.float64,
        )
        return float(np.average(values, weights=rows))

    auc = weighted("direction_auc")
    brier = weighted("direction_brier")
    prevalence_brier = weighted("prevalence_brier")
    mae = weighted("mean_absolute_error_bps")
    zero_mae = weighted("zero_baseline_mae_bps")
    information_coefficient = weighted("spearman_information_coefficient")
    threshold_gross = weighted("calibration_threshold_mean_signed_gross_bps")
    ic_values = [
        _finite(dict(fold["metrics"])["spearman_information_coefficient"], "ic")
        for fold in folds
    ]
    gross_values = [
        _finite(
            dict(fold["metrics"])["calibration_threshold_mean_signed_gross_bps"],
            "calibration_threshold_gross",
        )
        for fold in folds
    ]
    return {
        "folds": len(folds),
        "rows": int(np.sum(rows)),
        "direction_auc": auc,
        "auc_edge": auc - 0.5,
        "brier_improvement_ratio": (prevalence_brier - brier)
        / max(prevalence_brier, 1e-12),
        "mae_improvement_ratio": (zero_mae - mae) / max(zero_mae, 1e-12),
        "spearman_information_coefficient": information_coefficient,
        "calibration_threshold_mean_signed_gross_bps": threshold_gross,
        "positive_ic_fold_rate": sum(value > 0.0 for value in ic_values)
        / len(ic_values),
        "positive_gross_fold_rate": sum(value > 0.0 for value in gross_values)
        / len(gross_values),
        "research_candidate_fold_rate": sum(
            fold.get("status") == "research_candidate" for fold in folds
        )
        / len(folds),
    }


def _passes_segment(metrics: Mapping[str, object]) -> tuple[bool, tuple[str, ...]]:
    checks = {
        "auc_edge_not_positive": _finite(metrics["auc_edge"], "auc_edge") > 0.0,
        "brier_not_better_than_prevalence": _finite(
            metrics["brier_improvement_ratio"], "brier_improvement"
        )
        > 0.0,
        "mae_not_better_than_zero": _finite(
            metrics["mae_improvement_ratio"], "mae_improvement"
        )
        > 0.0,
        "information_coefficient_not_positive": _finite(
            metrics["spearman_information_coefficient"], "ic"
        )
        > 0.0,
        "calibration_threshold_gross_not_positive": _finite(
            metrics["calibration_threshold_mean_signed_gross_bps"], "gross"
        )
        > 0.0,
        "positive_ic_fold_rate_below_half": _finite(
            metrics["positive_ic_fold_rate"], "positive_ic_fold_rate"
        )
        >= 0.5,
        "positive_gross_fold_rate_below_half": _finite(
            metrics["positive_gross_fold_rate"], "positive_gross_fold_rate"
        )
        >= 0.5,
        "rejected_fold_present": _finite(
            metrics["research_candidate_fold_rate"], "candidate_fold_rate"
        )
        == 1.0,
    }
    reasons = tuple(reason for reason, passed in checks.items() if not passed)
    return not reasons, reasons


def _rank_scores(trials: Sequence[dict[str, object]]) -> dict[str, float]:
    if len(trials) == 1:
        return {str(trials[0]["trial"]): 1.0}
    accumulated = {str(trial["trial"]): 0.0 for trial in trials}
    for metric_name in _SELECTION_METRIC_NAMES:
        values = np.asarray(
            [
                _finite(dict(trial["screening_overall"])[metric_name], metric_name)
                for trial in trials
            ],
            dtype=np.float64,
        )
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and values[order[end]] == values[order[cursor]]:
                end += 1
            ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
            cursor = end
        for index, trial in enumerate(trials):
            accumulated[str(trial["trial"])] += float(ranks[index] / (len(trials) - 1))
    return {
        label: score / len(_SELECTION_METRIC_NAMES)
        for label, score in accumulated.items()
    }


def _average_zero_based_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _forecast_selection_overfit_diagnostic(
    validated: Sequence[tuple[TrialKey, list[dict[str, object]]]],
    *,
    symbols: Sequence[str],
    screening_fold_count: int,
) -> dict[str, object]:
    """Apply symmetric fold CV to the relative forecast-metric leaderboard."""

    trial_count = len(validated)
    if trial_count == 1:
        return {
            "method": "cscv_relative_forecast_metric_ranks",
            "status": "not_applicable_single_declared_trial",
            "passed": True,
            "threshold": _FORECAST_SELECTION_PBO_LIMIT,
            "estimated_probability": None,
            "screening_blocks": screening_fold_count,
            "declared_trials": 1,
            "symmetric_splits": 0,
            "splits": [],
            "limitations": [
                "PBO is not identified from a single declared trial",
                "the diagnostic ranks forecast metrics and is not a PnL or Sharpe PBO",
            ],
        }
    block_scores = np.zeros(
        (screening_fold_count, trial_count),
        dtype=np.float64,
    )
    for fold_index in range(screening_fold_count):
        block_metrics = [
            _segment_metrics(
                tuple(
                    fold
                    for fold in folds
                    if int(fold["fold_index"]) == fold_index
                    and str(fold["symbol"]) in symbols
                )
            )
            for _key, folds in validated
        ]
        for metric_name in _SELECTION_METRIC_NAMES:
            values = np.asarray(
                [
                    _finite(metrics[metric_name], metric_name)
                    for metrics in block_metrics
                ],
                dtype=np.float64,
            )
            block_scores[fold_index] += _average_zero_based_ranks(values) / (
                trial_count - 1
            )
    block_scores /= len(_SELECTION_METRIC_NAMES)
    trial_keys = tuple(key for key, _folds in validated)
    all_blocks = frozenset(range(screening_fold_count))
    split_rows: list[dict[str, object]] = []
    overfit_count = 0
    for split_index, training_blocks_raw in enumerate(
        combinations(range(screening_fold_count), screening_fold_count // 2)
    ):
        training_blocks = tuple(training_blocks_raw)
        validation_blocks = tuple(sorted(all_blocks.difference(training_blocks)))
        in_sample = np.mean(block_scores[list(training_blocks)], axis=0)
        out_of_sample = np.mean(block_scores[list(validation_blocks)], axis=0)
        winner_index = max(
            range(trial_count),
            key=lambda index: (
                float(in_sample[index]),
                -trial_keys[index].complexity,
                trial_keys[index].label,
            ),
        )
        out_ranks = _average_zero_based_ranks(out_of_sample)
        percentile = float((out_ranks[winner_index] + 1.0) / (trial_count + 1.0))
        logit = float(math.log(percentile / (1.0 - percentile)))
        overfit = logit < 0.0
        overfit_count += int(overfit)
        split_rows.append(
            {
                "split_index": split_index,
                "selection_fold_indices": list(training_blocks),
                "validation_fold_indices": list(validation_blocks),
                "selected_trial": trial_keys[winner_index].label,
                "selection_score": float(in_sample[winner_index]),
                "validation_score": float(out_of_sample[winner_index]),
                "validation_rank_percentile": percentile,
                "logit": logit,
                "overfit": overfit,
            }
        )
    probability = overfit_count / len(split_rows)
    return {
        "method": "cscv_relative_forecast_metric_ranks",
        "status": "passed" if probability <= _FORECAST_SELECTION_PBO_LIMIT else "rejected",
        "passed": probability <= _FORECAST_SELECTION_PBO_LIMIT,
        "threshold": _FORECAST_SELECTION_PBO_LIMIT,
        "estimated_probability": probability,
        "screening_blocks": screening_fold_count,
        "declared_trials": trial_count,
        "symmetric_splits": len(split_rows),
        "splits": split_rows,
        "limitations": [
            "non-overlapping outer folds are the symmetric CV blocks",
            "the diagnostic ranks forecast metrics and is not a PnL or Sharpe PBO",
            "terminal confirmation remains separately sealed and is never part of CSCV",
        ],
    }


def _bind_experiment_design(
    reports: Sequence[Mapping[str, object]],
    experiment_design: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, tuple[dict[str, object] | None, ...]]:
    if experiment_design is None:
        return None, tuple(None for _ in reports)
    design = validate_experiment_design_payload(experiment_design)
    domain_names = tuple(str(item["name"]) for item in design["domains"])  # type: ignore[index]
    expected_names = _TRIAL_CONFIG_KEYS | {"risk_level"}
    if set(domain_names) != expected_names:
        raise ValueError("tape/depth experiment design domains do not match the selector")
    candidates_by_parameters: dict[str, dict[str, object]] = {}
    for candidate in design["candidates"]:  # type: ignore[assignment]
        parameters = dict(candidate["parameters"])
        identity = _canonical_json(parameters)
        candidates_by_parameters[identity] = dict(candidate)
    bound: list[dict[str, object] | None] = []
    used_ids: set[str] = set()
    for report in reports:
        config = report.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("screening report configuration is invalid")
        parameters = {name: config.get(name) for name in domain_names}
        candidate = candidates_by_parameters.get(_canonical_json(parameters))
        if candidate is None:
            raise ValueError("screening report is not declared by the experiment design")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in used_ids:
            raise ValueError("experiment design candidate is represented more than once")
        used_ids.add(candidate_id)
        bound.append(candidate)
    if used_ids != {
        str(candidate["candidate_id"])
        for candidate in design["candidates"]  # type: ignore[assignment]
    }:
        raise ValueError("screening reports do not cover every experiment design candidate")
    summary = {
        "contract": design["contract"],
        "design_sha256": design["design_sha256"],
        "seed": design["seed"],
        "sampling_method": design["sampling_method"],
        "anchor_count": design["anchor_count"],
        "sampled_count": design["sampled_count"],
        "trial_burden": design["trial_burden"],
    }
    return summary, tuple(bound)


def select_tape_depth_screening_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    experiment_design: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze one winner from reports that contain screening folds only."""

    if not reports:
        raise ValueError("at least one tape/depth screening report is required")
    validated = [_validate_report(report) for report in reports]
    design_summary, design_candidates = _bind_experiment_design(
        reports,
        experiment_design,
    )
    trial_keys = [item[0] for item in validated]
    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("screening trials must have unique complete configurations")
    base_config = dict(reports[0]["config"])
    if base_config.get("study_stage") != "screening":
        raise ValueError("selection accepts screening-stage reports only")
    screening_fold_count = int(base_config["max_folds"])
    if (
        int(base_config["fold_start"]) != 0
        or screening_fold_count not in _SCREENING_FOLD_COUNTS
    ):
        raise ValueError(
            "screening reports must contain 4, 6, 8, or 10 initial folds"
        )
    base_common_config = _config_without(base_config, _TRIAL_CONFIG_KEYS)
    base_modeling_config = _config_without(base_config, _STAGE_CONFIG_KEYS)
    base_coverage = reports[0].get("coverage_fingerprints")
    base_available = {
        str(key): int(value)
        for key, value in dict(reports[0]["available_fold_counts"]).items()
    }
    base_folds = validated[0][1]
    symbols = tuple(str(symbol) for symbol in base_config["symbols"])
    if any(base_available[symbol] - screening_fold_count < 2 for symbol in symbols):
        raise ValueError("screening leaves fewer than two sealed folds")
    base_boundaries = [
        (
            str(fold["symbol"]),
            int(fold["fold_index"]),
            int(fold["evaluation_start_ms"]),
            int(fold["evaluation_end_ms"]),
        )
        for fold in base_folds
    ]
    dataset_contracts: dict[
        tuple[int, int, int],
        tuple[object, tuple[tuple[str, int, int, int, str], ...]],
    ] = {}
    plan_fingerprints_by_trial: dict[str, object] = {}
    for report, (key, folds) in zip(reports, validated, strict=True):
        config = dict(report["config"])
        boundaries = [
            (
                str(fold["symbol"]),
                int(fold["fold_index"]),
                int(fold["evaluation_start_ms"]),
                int(fold["evaluation_end_ms"]),
            )
            for fold in folds
        ]
        if (
            config.get("study_stage") != "screening"
            or _config_without(config, _TRIAL_CONFIG_KEYS) != base_common_config
            or report.get("coverage_fingerprints") != base_coverage
            or report.get("available_fold_counts") != reports[0].get("available_fold_counts")
            or boundaries != base_boundaries
        ):
            raise ValueError("screening reports do not use identical folds and coverage")
        dataset_identities = tuple(
            (
                str(fold["symbol"]),
                int(fold["fold_index"]),
                int(fold["evaluation_start_ms"]),
                int(fold["evaluation_end_ms"]),
                str(fold["dataset_fingerprint"]),
            )
            for fold in folds
        )
        dataset_contract = (
            report.get("plan_fingerprints"),
            dataset_identities,
        )
        prior_contract = dataset_contracts.setdefault(key.dataset_key, dataset_contract)
        if prior_contract != dataset_contract:
            raise ValueError(
                "screening reports with the same dataset configuration do not use identical data"
            )
        plan_fingerprints_by_trial[key.label] = report.get("plan_fingerprints")
    screening_trials: list[dict[str, object]] = []
    for report_index, (key, folds) in enumerate(validated):
        by_symbol: dict[str, object] = {}
        for symbol in symbols:
            symbol_folds = tuple(fold for fold in folds if fold["symbol"] == symbol)
            if len(symbol_folds) != screening_fold_count:
                raise ValueError(f"screening fold count differs for {symbol}")
            metrics = _segment_metrics(symbol_folds)
            passed, reasons = _passes_segment(metrics)
            by_symbol[symbol] = {
                "passed": passed,
                "reasons": list(reasons),
                "metrics": metrics,
            }
        overall = _segment_metrics(folds)
        screening_trials.append(
            {
                "trial": key.label,
                "trial_config": key.asdict(),
                "experiment_candidate": design_candidates[report_index],
                "horizon_seconds": key.horizon_seconds,
                "decision_cadence_seconds": key.decision_cadence_seconds,
                "maximum_depth_age_ms": key.maximum_depth_age_ms,
                "model_profile": key.model_profile,
                "feature_set": key.feature_set,
                "screening_plan_fingerprints": plan_fingerprints_by_trial[key.label],
                "complexity": key.complexity,
                "eligible": all(bool(dict(by_symbol[symbol])["passed"]) for symbol in symbols),
                "screening_by_symbol": by_symbol,
                "screening_overall": overall,
            }
        )
    rank_scores = _rank_scores(screening_trials)
    for trial in screening_trials:
        trial["screening_rank_score"] = rank_scores[str(trial["trial"])]
    eligible = [trial for trial in screening_trials if bool(trial["eligible"])]
    ranked_winner = (
        max(
            eligible,
            key=lambda trial: (
                float(trial["screening_rank_score"]),
                -int(trial["complexity"]),
                str(trial["trial"]),
            ),
        )
        if eligible
        else None
    )
    overfit_diagnostic = _forecast_selection_overfit_diagnostic(
        validated,
        symbols=symbols,
        screening_fold_count=screening_fold_count,
    )
    selected = ranked_winner if bool(overfit_diagnostic["passed"]) else None
    boundaries = {
        symbol: max(
            int(fold["evaluation_end_ms"])
            for fold in base_folds
            if fold["symbol"] == symbol
        )
        for symbol in symbols
    }
    payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_SELECTION_SCHEMA_VERSION,
        "status": "winner_frozen" if selected is not None else "rejected",
        "rejection_reasons": (
            []
            if selected is not None
            else [
                (
                    "no_screening_trial_passed"
                    if ranked_winner is None
                    else "forecast_selection_pbo_above_0_20"
                )
            ]
        ),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "declared_trial_count": len(screening_trials),
        "experiment_design": design_summary,
        "symbols": list(symbols),
        "modeling_config": base_modeling_config,
        "coverage_fingerprints": base_coverage,
        "available_fold_counts": base_available,
        "screening_plan_fingerprints": (
            selected["screening_plan_fingerprints"] if selected is not None else None
        ),
        "screening_fold_count": screening_fold_count,
        "screening_boundaries_ms": boundaries,
        "confirmation_fold_start": screening_fold_count,
        "confirmation_available_folds": {
            symbol: base_available[symbol] - screening_fold_count for symbol in symbols
        },
        "screening_trials": screening_trials,
        "forecast_selection_overfit_diagnostic": overfit_diagnostic,
        "ranked_winner_trial": (
            str(ranked_winner["trial"]) if ranked_winner else None
        ),
        "selected_trial": str(selected["trial"]) if selected else None,
        "selected_trial_config": dict(selected["trial_config"]) if selected else None,
        "selected_horizon_seconds": int(selected["horizon_seconds"]) if selected else None,
        "selected_decision_cadence_seconds": (
            int(selected["decision_cadence_seconds"]) if selected else None
        ),
        "selected_maximum_depth_age_ms": (
            int(selected["maximum_depth_age_ms"]) if selected else None
        ),
        "selected_model_profile": str(selected["model_profile"]) if selected else None,
        "selected_feature_set": str(selected["feature_set"]) if selected else None,
        "limitations": [
            "screening ranks forecast metrics, not executable PnL",
            "the frozen winner has not accessed the terminal confirmation folds",
            "exact BBO replay and no-order shadow remain mandatory",
        ],
    }
    return _with_fingerprint(payload, "selection_fingerprint")


def _validate_selection_payload(selection: Mapping[str, object]) -> None:
    if (
        selection.get("schema_version") != TAPE_DEPTH_SELECTION_SCHEMA_VERSION
        or selection.get("status") != "winner_frozen"
        or selection.get("trading_authority") is not False
        or selection.get("execution_claim") is not False
        or selection.get("profitability_claim") is not False
        or not _is_sha256(selection.get("selection_fingerprint"))
        or selection.get("selection_fingerprint")
        != _payload_fingerprint(selection, "selection_fingerprint")
    ):
        raise ValueError("selection lock failed its immutable evidence contract")
    symbols = selection.get("symbols")
    if not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("selection lock symbols are invalid")
    try:
        selected_key = TrialKey(
            horizon_seconds=int(selection.get("selected_horizon_seconds", -1)),
            decision_cadence_seconds=int(
                selection.get("selected_decision_cadence_seconds", -1)
            ),
            maximum_depth_age_ms=int(
                selection.get("selected_maximum_depth_age_ms", -1)
            ),
            model_profile=str(selection.get("selected_model_profile") or ""),
            feature_set=str(selection.get("selected_feature_set") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selection lock winner is invalid") from exc
    if (
        selected_key.model_profile not in _PROFILE_COMPLEXITY
        or selected_key.feature_set not in _FEATURE_COMPLEXITY
        or not 1 <= selected_key.horizon_seconds <= 3_600
        or not 1 <= selected_key.decision_cadence_seconds <= 60
        or 60 % selected_key.decision_cadence_seconds != 0
        or not 1_000 <= selected_key.maximum_depth_age_ms <= 300_000
        or selection.get("selected_trial") != selected_key.label
        or selection.get("selected_trial_config") != selected_key.asdict()
        or not isinstance(selection.get("modeling_config"), Mapping)
    ):
        raise ValueError("selection lock winner is invalid")
    experiment = selection.get("experiment_design")
    if experiment is not None and (
        not isinstance(experiment, Mapping)
        or experiment.get("contract") != EXPERIMENT_DESIGN_CONTRACT
        or not _is_sha256(experiment.get("design_sha256"))
        or int(experiment.get("trial_burden", -1))
        != int(selection.get("declared_trial_count", -2))
    ):
        raise ValueError("selection lock experiment design binding is invalid")
    normalized_symbols = tuple(str(symbol) for symbol in symbols)
    _mapping_of_sha256(
        selection.get("coverage_fingerprints"),
        expected_symbols=normalized_symbols,
        name="selection coverage fingerprints",
    )
    _mapping_of_sha256(
        selection.get("screening_plan_fingerprints"),
        expected_symbols=normalized_symbols,
        name="selection screening plan fingerprints",
    )
    available = _mapping_of_positive_ints(
        selection.get("available_fold_counts"),
        expected_symbols=normalized_symbols,
        name="selection available fold counts",
    )
    boundaries = _mapping_of_positive_ints(
        selection.get("screening_boundaries_ms"),
        expected_symbols=normalized_symbols,
        name="selection screening boundaries",
    )
    del boundaries
    try:
        fold_start = int(selection.get("confirmation_fold_start", -1))
        screening_count = int(selection.get("screening_fold_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("selection lock fold boundary is invalid") from exc
    if (
        fold_start != screening_count
        or screening_count not in _SCREENING_FOLD_COUNTS
        or any(
        available[symbol] - fold_start < 2 for symbol in normalized_symbols
        )
    ):
        raise ValueError("selection lock does not preserve two confirmation folds")
    diagnostic = selection.get("forecast_selection_overfit_diagnostic")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("method") != "cscv_relative_forecast_metric_ranks"
        or diagnostic.get("passed") is not True
        or int(diagnostic.get("screening_blocks", -1)) != screening_count
        or int(diagnostic.get("declared_trials", -1))
        != int(selection.get("declared_trial_count", -2))
        or float(diagnostic.get("threshold", -1.0))
        != _FORECAST_SELECTION_PBO_LIMIT
    ):
        raise ValueError("selection lock overfit diagnostic is invalid")


def validate_tape_depth_confirmation_request(
    selection: Mapping[str, object],
    *,
    config: Mapping[str, object],
    plans: Sequence[object],
) -> None:
    """Reject a confirmation run unless its plan is the untouched frozen suffix."""

    _validate_selection_payload(selection)
    if (
        config.get("study_stage") != "confirmation"
        or int(config.get("max_folds", -1)) != 0
        or int(config.get("fold_start", -1))
        != int(selection["confirmation_fold_start"])
        or int(config.get("horizon_seconds", -1))
        != int(selection["selected_horizon_seconds"])
        or int(config.get("decision_cadence_seconds", -1))
        != int(selection["selected_decision_cadence_seconds"])
        or int(config.get("maximum_depth_age_ms", -1))
        != int(selection["selected_maximum_depth_age_ms"])
        or str(config.get("model_profile")) != str(selection["selected_model_profile"])
        or str(config.get("feature_set")) != str(selection["selected_feature_set"])
        or _config_without(config, _STAGE_CONFIG_KEYS) != dict(selection["modeling_config"])
    ):
        raise ValueError("confirmation request differs from the frozen winner contract")
    symbols = tuple(str(symbol) for symbol in selection["symbols"])
    coverage = dict(selection["coverage_fingerprints"])
    available = {str(key): int(value) for key, value in dict(selection["available_fold_counts"]).items()}
    boundaries = {str(key): int(value) for key, value in dict(selection["screening_boundaries_ms"]).items()}
    if tuple(str(getattr(plan, "symbol", "")) for plan in plans) != symbols:
        raise ValueError("confirmation plans differ from the selection symbols")
    fold_start = int(selection["confirmation_fold_start"])
    for plan in plans:
        symbol = str(getattr(plan, "symbol"))
        folds = tuple(getattr(plan, "folds"))
        if (
            str(getattr(plan, "coverage_fingerprint")) != str(coverage[symbol])
            or int(getattr(plan, "available_fold_count")) != available[symbol]
            or not folds
            or int(getattr(folds[0], "fold_index")) != fold_start
            or int(getattr(folds[-1], "fold_index")) != available[symbol] - 1
            or len(folds) != available[symbol] - fold_start
            or int(getattr(folds[0], "evaluation_start_ms")) <= boundaries[symbol]
        ):
            raise ValueError("confirmation plan is not the untouched frozen suffix")


def confirm_tape_depth_report(
    selection: Mapping[str, object],
    report: Mapping[str, object],
    *,
    selection_lock_sha256: str,
) -> dict[str, object]:
    """Evaluate exactly one frozen winner on its untouched terminal folds."""

    _validate_selection_payload(selection)
    if not _is_sha256(selection_lock_sha256):
        raise ValueError("selection lock file hash is invalid")
    key, folds = _validate_report(report)
    config = dict(report["config"])
    if (
        config.get("study_stage") != "confirmation"
        or config.get("selection_lock_sha256") != selection_lock_sha256
        or key.horizon_seconds != selection["selected_horizon_seconds"]
        or key.decision_cadence_seconds
        != selection["selected_decision_cadence_seconds"]
        or key.maximum_depth_age_ms != selection["selected_maximum_depth_age_ms"]
        or key.model_profile != selection["selected_model_profile"]
        or key.feature_set != selection["selected_feature_set"]
        or int(config.get("fold_start", -1)) != int(selection["confirmation_fold_start"])
        or int(config.get("max_folds", -1)) != 0
        or _config_without(config, _STAGE_CONFIG_KEYS) != dict(selection["modeling_config"])
        or report.get("coverage_fingerprints") != selection.get("coverage_fingerprints")
        or report.get("available_fold_counts") != selection.get("available_fold_counts")
    ):
        raise ValueError("confirmation report differs from the frozen winner contract")
    symbols = tuple(str(symbol) for symbol in selection["symbols"])
    available = {str(key): int(value) for key, value in dict(selection["available_fold_counts"]).items()}
    boundaries = {str(key): int(value) for key, value in dict(selection["screening_boundaries_ms"]).items()}
    fold_start = int(selection["confirmation_fold_start"])
    by_symbol: dict[str, object] = {}
    for symbol in symbols:
        symbol_folds = tuple(fold for fold in folds if fold["symbol"] == symbol)
        if (
            len(symbol_folds) != available[symbol] - fold_start
            or int(symbol_folds[0]["fold_index"]) != fold_start
            or int(symbol_folds[-1]["fold_index"]) != available[symbol] - 1
            or int(symbol_folds[0]["evaluation_start_ms"]) <= boundaries[symbol]
        ):
            raise ValueError("confirmation report is not the untouched terminal suffix")
        metrics = _segment_metrics(symbol_folds)
        passed, reasons = _passes_segment(metrics)
        by_symbol[symbol] = {
            "passed": passed,
            "reasons": list(reasons),
            "metrics": metrics,
        }
    overall = _segment_metrics(folds)
    passed = all(bool(dict(by_symbol[symbol])["passed"]) for symbol in symbols)
    payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_CONFIRMATION_SCHEMA_VERSION,
        "status": "confirmed_forecast_candidate" if passed else "rejected",
        "rejection_reasons": [] if passed else ["frozen_winner_failed_confirmation"],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "selected_trial": selection["selected_trial"],
        "selected_trial_config": selection["selected_trial_config"],
        "declared_trial_count": selection["declared_trial_count"],
        "selection_fingerprint": selection["selection_fingerprint"],
        "selection_lock_sha256": selection_lock_sha256,
        "confirmation_fold_start": fold_start,
        "confirmation_by_symbol": by_symbol,
        "confirmation_overall": overall,
        "limitations": [
            "confirmation measures forecast quality, not executable PnL",
            "no runner-up is evaluated after the frozen winner",
            "exact BBO replay and no-order shadow remain mandatory",
        ],
    }
    return _with_fingerprint(payload, "confirmation_fingerprint")


def _read_json_object(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_JSON_EVIDENCE_BYTES + 1)
        if len(raw) > _MAX_JSON_EVIDENCE_BYTES:
            raise ValueError(f"{description} exceeds the evidence size limit: {path}")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be an object: {path}")
    return payload, raw


def load_and_select_tape_depth_reports(
    paths: Sequence[str | Path],
    *,
    output: str | Path,
    design_path: str | Path | None = None,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one screening report path is required")
    destination = Path(output).resolve()
    reports: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    resolved_design = Path(design_path).resolve() if design_path is not None else None
    experiment_design: dict[str, object] | None = None
    design_source: dict[str, str] | None = None
    if resolved_design is not None:
        if resolved_design == destination:
            raise ValueError("selection output cannot overwrite the experiment design")
        experiment_design, design_raw = _read_json_object(
            resolved_design,
            "experiment design",
        )
        design_source = {
            "path": str(resolved_design),
            "sha256": _sha256_bytes(design_raw),
        }
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path == destination or path == resolved_design:
            raise ValueError("selection output cannot overwrite an input report")
        report, raw = _read_json_object(path, "screening report")
        verify_tape_depth_prequential_report(path, report)
        reports.append(report)
        sources.append({"path": str(path), "sha256": _sha256_bytes(raw)})
    selection = select_tape_depth_screening_reports(
        reports,
        experiment_design=experiment_design,
    )
    selection["source_reports"] = sources
    selection["experiment_design_source"] = design_source
    selection = _with_fingerprint(selection, "selection_fingerprint")
    write_json_atomic(destination, selection, indent=2, sort_keys=True)
    return selection


def load_verified_tape_depth_selection(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    selection_path = Path(path).resolve()
    selection, raw = _read_json_object(selection_path, "selection lock")
    _validate_selection_payload(selection)
    sources = selection.get("source_reports")
    if not isinstance(sources, list) or not sources:
        raise ValueError("selection lock omits its screening source reports")
    reports: list[dict[str, object]] = []
    normalized_sources: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("selection lock source report is invalid")
        source_path = Path(str(source.get("path") or "")).resolve()
        report, report_raw = _read_json_object(source_path, "screening source report")
        verify_tape_depth_prequential_report(source_path, report)
        report_sha256 = _sha256_bytes(report_raw)
        if report_sha256 != source.get("sha256"):
            raise ValueError("selection lock screening source report changed")
        reports.append(report)
        normalized_sources.append({"path": str(source_path), "sha256": report_sha256})
    design_summary = selection.get("experiment_design")
    design_source = selection.get("experiment_design_source")
    experiment_design: dict[str, object] | None = None
    normalized_design_source: dict[str, str] | None = None
    if design_summary is None:
        if design_source is not None:
            raise ValueError("selection lock has an unexpected experiment design source")
    else:
        if not isinstance(design_source, Mapping):
            raise ValueError("selection lock omits its experiment design source")
        design_path = Path(str(design_source.get("path") or "")).resolve()
        experiment_design, design_raw = _read_json_object(
            design_path,
            "experiment design source",
        )
        design_sha256 = _sha256_bytes(design_raw)
        if design_sha256 != design_source.get("sha256"):
            raise ValueError("selection lock experiment design source changed")
        normalized_design_source = {
            "path": str(design_path),
            "sha256": design_sha256,
        }
    expected = select_tape_depth_screening_reports(
        reports,
        experiment_design=experiment_design,
    )
    expected["source_reports"] = normalized_sources
    expected["experiment_design_source"] = normalized_design_source
    expected = _with_fingerprint(expected, "selection_fingerprint")
    if selection != expected:
        raise ValueError("selection lock differs from recomputed screening evidence")
    return selection, _sha256_bytes(raw)


def load_and_confirm_tape_depth_report(
    *,
    selection_path: str | Path,
    report_path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    destination = Path(output).resolve()
    resolved_selection = Path(selection_path).resolve()
    resolved_report = Path(report_path).resolve()
    if destination in {resolved_selection, resolved_report}:
        raise ValueError("confirmation output cannot overwrite its input evidence")
    selection, selection_sha256 = load_verified_tape_depth_selection(
        resolved_selection
    )
    report, report_raw = _read_json_object(resolved_report, "confirmation report")
    verify_tape_depth_prequential_report(resolved_report, report)
    confirmation = confirm_tape_depth_report(
        selection,
        report,
        selection_lock_sha256=selection_sha256,
    )
    confirmation["selection_lock"] = {
        "path": str(resolved_selection),
        "sha256": selection_sha256,
    }
    confirmation["confirmation_report"] = {
        "path": str(resolved_report),
        "sha256": _sha256_bytes(report_raw),
    }
    confirmation = _with_fingerprint(confirmation, "confirmation_fingerprint")
    write_json_atomic(destination, confirmation, indent=2, sort_keys=True)
    return confirmation


__all__ = [
    "TAPE_DEPTH_CONFIRMATION_SCHEMA_VERSION",
    "TAPE_DEPTH_SELECTION_SCHEMA_VERSION",
    "TrialKey",
    "confirm_tape_depth_report",
    "load_and_confirm_tape_depth_report",
    "load_and_select_tape_depth_reports",
    "load_verified_tape_depth_selection",
    "select_tape_depth_screening_reports",
    "validate_tape_depth_confirmation_request",
]

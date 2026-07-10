"""Sealed screening and winner-only confirmation for tape/depth trials."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .storage import write_json_atomic
from .tape_depth_prequential import (
    TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
    verify_tape_depth_prequential_report,
)


TAPE_DEPTH_SELECTION_SCHEMA_VERSION = "tape-depth-screening-selection-v1"
TAPE_DEPTH_CONFIRMATION_SCHEMA_VERSION = "tape-depth-sealed-confirmation-v1"
_TRIAL_CONFIG_KEYS = frozenset({"model_profile", "feature_set"})
_STAGE_CONFIG_KEYS = frozenset(
    {
        "model_profile",
        "feature_set",
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
    model_profile: str
    feature_set: str

    @property
    def label(self) -> str:
        return f"{self.model_profile}/{self.feature_set}"

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
    return TrialKey(profile, feature_set), normalized_folds


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
    top_decile_gross = weighted("top_decile_mean_signed_gross_bps")
    ic_values = [
        _finite(dict(fold["metrics"])["spearman_information_coefficient"], "ic")
        for fold in folds
    ]
    gross_values = [
        _finite(
            dict(fold["metrics"])["top_decile_mean_signed_gross_bps"],
            "top_decile_gross",
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
        "top_decile_mean_signed_gross_bps": top_decile_gross,
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
        "top_decile_gross_not_positive": _finite(
            metrics["top_decile_mean_signed_gross_bps"], "gross"
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
    metric_names = (
        "auc_edge",
        "brier_improvement_ratio",
        "mae_improvement_ratio",
        "spearman_information_coefficient",
        "top_decile_mean_signed_gross_bps",
        "positive_ic_fold_rate",
        "positive_gross_fold_rate",
    )
    if len(trials) == 1:
        return {str(trials[0]["trial"]): 1.0}
    accumulated = {str(trial["trial"]): 0.0 for trial in trials}
    for metric_name in metric_names:
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
    return {label: score / len(metric_names) for label, score in accumulated.items()}


def select_tape_depth_screening_reports(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Freeze one winner from reports that contain screening folds only."""

    if not reports:
        raise ValueError("at least one tape/depth screening report is required")
    validated = [_validate_report(report) for report in reports]
    trial_keys = [item[0] for item in validated]
    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("screening trials must have unique profile/feature pairs")
    base_config = dict(reports[0]["config"])
    if base_config.get("study_stage") != "screening":
        raise ValueError("selection accepts screening-stage reports only")
    screening_fold_count = int(base_config["max_folds"])
    if int(base_config["fold_start"]) != 0 or screening_fold_count < 2:
        raise ValueError("screening reports must contain a declared initial fold window")
    base_common_config = _config_without(base_config, _TRIAL_CONFIG_KEYS)
    base_modeling_config = _config_without(base_config, _STAGE_CONFIG_KEYS)
    base_plans = reports[0].get("plan_fingerprints")
    base_coverage = reports[0].get("coverage_fingerprints")
    base_available = {
        str(key): int(value)
        for key, value in dict(reports[0]["available_fold_counts"]).items()
    }
    base_folds = validated[0][1]
    symbols = tuple(str(symbol) for symbol in base_config["symbols"])
    if any(base_available[symbol] - screening_fold_count < 2 for symbol in symbols):
        raise ValueError("screening leaves fewer than two sealed folds")
    base_identities = [
        (
            str(fold["symbol"]),
            int(fold["fold_index"]),
            int(fold["evaluation_start_ms"]),
            int(fold["evaluation_end_ms"]),
            str(fold["dataset_fingerprint"]),
        )
        for fold in base_folds
    ]
    for report, (_key, folds) in zip(reports, validated, strict=True):
        config = dict(report["config"])
        identities = [
            (
                str(fold["symbol"]),
                int(fold["fold_index"]),
                int(fold["evaluation_start_ms"]),
                int(fold["evaluation_end_ms"]),
                str(fold["dataset_fingerprint"]),
            )
            for fold in folds
        ]
        if (
            config.get("study_stage") != "screening"
            or _config_without(config, _TRIAL_CONFIG_KEYS) != base_common_config
            or report.get("plan_fingerprints") != base_plans
            or report.get("coverage_fingerprints") != base_coverage
            or report.get("available_fold_counts") != reports[0].get("available_fold_counts")
            or identities != base_identities
        ):
            raise ValueError("screening reports do not use identical folds and data")
    screening_trials: list[dict[str, object]] = []
    for key, folds in validated:
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
                "model_profile": key.model_profile,
                "feature_set": key.feature_set,
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
    selected = (
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
        "rejection_reasons": [] if selected is not None else ["no_screening_trial_passed"],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "declared_trial_count": len(screening_trials),
        "symbols": list(symbols),
        "modeling_config": base_modeling_config,
        "coverage_fingerprints": base_coverage,
        "available_fold_counts": base_available,
        "screening_plan_fingerprints": base_plans,
        "screening_fold_count": screening_fold_count,
        "screening_boundaries_ms": boundaries,
        "confirmation_fold_start": screening_fold_count,
        "confirmation_available_folds": {
            symbol: base_available[symbol] - screening_fold_count for symbol in symbols
        },
        "screening_trials": screening_trials,
        "selected_trial": str(selected["trial"]) if selected else None,
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
    if (
        str(selection.get("selected_model_profile")) not in _PROFILE_COMPLEXITY
        or str(selection.get("selected_feature_set")) not in _FEATURE_COMPLEXITY
        or selection.get("selected_trial")
        != f"{selection.get('selected_model_profile')}/{selection.get('selected_feature_set')}"
        or not isinstance(selection.get("modeling_config"), Mapping)
    ):
        raise ValueError("selection lock winner is invalid")
    normalized_symbols = tuple(str(symbol) for symbol in symbols)
    _mapping_of_sha256(
        selection.get("coverage_fingerprints"),
        expected_symbols=normalized_symbols,
        name="selection coverage fingerprints",
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
    if fold_start != screening_count or screening_count < 2 or any(
        available[symbol] - fold_start < 2 for symbol in normalized_symbols
    ):
        raise ValueError("selection lock does not preserve two confirmation folds")


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
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one screening report path is required")
    destination = Path(output).resolve()
    reports: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path == destination:
            raise ValueError("selection output cannot overwrite an input report")
        report, raw = _read_json_object(path, "screening report")
        verify_tape_depth_prequential_report(path, report)
        reports.append(report)
        sources.append({"path": str(path), "sha256": _sha256_bytes(raw)})
    selection = select_tape_depth_screening_reports(reports)
    selection["source_reports"] = sources
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
    expected = select_tape_depth_screening_reports(reports)
    expected["source_reports"] = normalized_sources
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

"""Selection/confirmation comparison for tape/depth prequential trials."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .storage import write_json_atomic
from .tape_depth_prequential import TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION


TAPE_DEPTH_COMPARISON_SCHEMA_VERSION = "tape-depth-ablation-comparison-v1"
_IGNORED_CONFIG_KEYS = frozenset({"model_profile", "feature_set"})
_PROFILE_COMPLEXITY = {"regularized": 0, "balanced": 1, "expressive": 2}
_FEATURE_COMPLEXITY = {"core": 0, "tape_derived": 1, "cross_asset": 2, "full": 3}


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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


def _canonical_config(config: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in _IGNORED_CONFIG_KEYS
    }


def _validate_report(report: Mapping[str, object]) -> tuple[TrialKey, list[dict[str, object]]]:
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
    plan_fingerprints = report.get("plan_fingerprints")
    if not isinstance(config, Mapping) or not isinstance(folds, list) or not folds:
        raise ValueError("comparison input report is incomplete")
    if (
        int(report.get("total_folds", -1)) != len(folds)
        or int(report.get("completed_folds", -1)) != len(folds)
    ):
        raise ValueError("comparison input report is not a complete fold run")
    if not isinstance(plan_fingerprints, Mapping) or not plan_fingerprints or not all(
        _is_sha256(value) for value in plan_fingerprints.values()
    ):
        raise ValueError("comparison plan fingerprints are invalid")
    profile = str(config.get("model_profile") or "")
    feature_set = str(config.get("feature_set") or "")
    if profile not in _PROFILE_COMPLEXITY or feature_set not in _FEATURE_COMPLEXITY:
        raise ValueError("comparison trial profile or feature set is unsupported")
    configured_symbols = config.get("symbols")
    if not isinstance(configured_symbols, list) or not configured_symbols:
        raise ValueError("comparison symbols are invalid")
    expected_symbols = tuple(str(symbol) for symbol in configured_symbols)
    if len(set(expected_symbols)) != len(expected_symbols) or any(
        not symbol for symbol in expected_symbols
    ):
        raise ValueError("comparison symbols are invalid")
    if set(str(symbol) for symbol in plan_fingerprints) != set(expected_symbols):
        raise ValueError("comparison plans do not cover the configured symbols")
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
        if [int(fold["fold_index"]) for fold in symbol_folds] != list(
            range(len(symbol_folds))
        ):
            raise ValueError(f"comparison folds are incomplete for {symbol}")
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
    brier_improvement = (prevalence_brier - brier) / max(prevalence_brier, 1e-12)
    mae_improvement = (zero_mae - mae) / max(zero_mae, 1e-12)
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
        "brier_improvement_ratio": brier_improvement,
        "mae_improvement_ratio": mae_improvement,
        "spearman_information_coefficient": information_coefficient,
        "top_decile_mean_signed_gross_bps": top_decile_gross,
        "positive_ic_fold_rate": sum(value > 0.0 for value in ic_values) / len(ic_values),
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
                _finite(dict(trial["selection_overall"])[metric_name], metric_name)
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


def compare_tape_depth_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    selection_fraction: float = 0.67,
) -> dict[str, object]:
    """Select on early folds, then evaluate only the winner on later folds."""

    if not 0.50 <= float(selection_fraction) <= 0.80:
        raise ValueError("selection_fraction must lie in [0.50, 0.80]")
    if not reports:
        raise ValueError("at least one tape/depth report is required")
    validated = [_validate_report(report) for report in reports]
    trial_keys = [item[0] for item in validated]
    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("comparison trials must have unique profile/feature pairs")
    base_config = _canonical_config(dict(reports[0]["config"]))
    base_plans = reports[0].get("plan_fingerprints")
    base_folds = validated[0][1]
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
    for report, (_key, folds) in zip(reports[1:], validated[1:], strict=True):
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
            _canonical_config(dict(report["config"])) != base_config
            or report.get("plan_fingerprints") != base_plans
            or identities != base_identities
        ):
            raise ValueError("comparison reports do not use identical folds and data")

    selection_trials: list[dict[str, object]] = []
    fold_partitions: dict[str, dict[str, tuple[dict[str, object], ...]]] = {}
    symbols = sorted({str(fold["symbol"]) for fold in base_folds})
    for key, folds in validated:
        by_symbol = {
            symbol: tuple(fold for fold in folds if fold["symbol"] == symbol)
            for symbol in symbols
        }
        partitions: dict[str, tuple[dict[str, object], ...]] = {}
        selection_by_symbol: dict[str, object] = {}
        for symbol, symbol_folds in by_symbol.items():
            if len(symbol_folds) < 4:
                raise ValueError(f"{symbol} needs at least four comparison folds")
            split = max(
                2,
                min(len(symbol_folds) - 2, int(len(symbol_folds) * selection_fraction)),
            )
            selection_folds = symbol_folds[:split]
            confirmation_folds = symbol_folds[split:]
            partitions[f"{symbol}:selection"] = selection_folds
            partitions[f"{symbol}:confirmation"] = confirmation_folds
            metrics = _segment_metrics(selection_folds)
            passed, reasons = _passes_segment(metrics)
            selection_by_symbol[symbol] = {
                "passed": passed,
                "reasons": list(reasons),
                "metrics": metrics,
            }
        all_selection_folds = tuple(
            fold
            for symbol in symbols
            for fold in partitions[f"{symbol}:selection"]
        )
        overall = _segment_metrics(all_selection_folds)
        eligible = all(
            bool(dict(selection_by_symbol[symbol])["passed"]) for symbol in symbols
        )
        selection_trials.append(
            {
                "trial": key.label,
                "model_profile": key.model_profile,
                "feature_set": key.feature_set,
                "complexity": key.complexity,
                "eligible": eligible,
                "selection_by_symbol": selection_by_symbol,
                "selection_overall": overall,
            }
        )
        fold_partitions[key.label] = partitions

    rank_scores = _rank_scores(selection_trials)
    for trial in selection_trials:
        trial["selection_rank_score"] = rank_scores[str(trial["trial"])]
    eligible_trials = [trial for trial in selection_trials if bool(trial["eligible"])]
    selected = (
        max(
            eligible_trials,
            key=lambda trial: (
                float(trial["selection_rank_score"]),
                -int(trial["complexity"]),
                str(trial["trial"]),
            ),
        )
        if eligible_trials
        else None
    )
    confirmation: dict[str, object] | None = None
    reasons: list[str] = []
    if selected is None:
        reasons.append("no_trial_passed_earlier_selection_folds")
    else:
        selected_label = str(selected["trial"])
        confirmation_by_symbol: dict[str, object] = {}
        for symbol in symbols:
            metrics = _segment_metrics(
                fold_partitions[selected_label][f"{symbol}:confirmation"]
            )
            passed, segment_reasons = _passes_segment(metrics)
            confirmation_by_symbol[symbol] = {
                "passed": passed,
                "reasons": list(segment_reasons),
                "metrics": metrics,
            }
        confirmed = all(
            bool(dict(confirmation_by_symbol[symbol])["passed"])
            for symbol in symbols
        )
        confirmation = {
            "trial": selected_label,
            "passed": confirmed,
            "by_symbol": confirmation_by_symbol,
            "overall": _segment_metrics(
                tuple(
                    fold
                    for symbol in symbols
                    for fold in fold_partitions[selected_label][
                        f"{symbol}:confirmation"
                    ]
                )
            ),
        }
        if not confirmed:
            reasons.append("selected_trial_failed_later_confirmation_folds")
    confirmed = bool(confirmation and confirmation["passed"])
    return {
        "schema_version": TAPE_DEPTH_COMPARISON_SCHEMA_VERSION,
        "status": "confirmed_forecast_candidate" if confirmed else "rejected",
        "rejection_reasons": reasons,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "selection_fraction": float(selection_fraction),
        "declared_trial_count": len(selection_trials),
        "symbols": symbols,
        "common_config": base_config,
        "plan_fingerprints": base_plans,
        "selection_trials": selection_trials,
        "selected_trial": str(selected["trial"]) if selected else None,
        "confirmation": confirmation,
        "limitations": [
            "selection ranks forecast metrics, not executable PnL",
            "later-fold metrics are reported only for the selected trial",
            "exact BBO replay and no-order shadow remain mandatory",
        ],
    }


def load_and_compare_tape_depth_reports(
    paths: Sequence[str | Path],
    *,
    output: str | Path,
    selection_fraction: float = 0.67,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one comparison report path is required")
    destination = Path(output).resolve()
    reports: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path == destination:
            raise ValueError("comparison output cannot overwrite an input report")
        try:
            raw = path.read_bytes()
            report = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"comparison report is unreadable: {path}") from exc
        if not isinstance(report, dict):
            raise ValueError(f"comparison report must be an object: {path}")
        reports.append(report)
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    comparison = compare_tape_depth_reports(
        reports,
        selection_fraction=selection_fraction,
    )
    comparison["source_reports"] = sources
    write_json_atomic(destination, comparison, indent=2, sort_keys=True)
    return comparison


__all__ = [
    "TAPE_DEPTH_COMPARISON_SCHEMA_VERSION",
    "TrialKey",
    "compare_tape_depth_reports",
    "load_and_compare_tape_depth_reports",
]

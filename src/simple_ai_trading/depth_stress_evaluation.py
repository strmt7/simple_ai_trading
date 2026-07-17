"""Rolling out-of-sample evaluation for the frozen depth-stress challenger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np

from .depth_stress_model import (
    DEPTH_STRESS_MODEL_SCHEMA_VERSION,
    DepthStressModelArtifact,
    assign_depth_stress_states,
    depth_stress_loss_rows,
    depth_stress_metrics,
    fit_depth_stress_thresholds,
    fit_depth_transition_probabilities,
    predict_depth_stress_challenger,
    predict_depth_transition_probabilities,
    train_depth_stress_challenger,
)
from .depth_stress_screen import (
    DEPTH_STRESS_FEATURE_NAMES,
    DEPTH_STRESS_HORIZONS_SECONDS,
    DepthStressExamples,
    DepthStressPanel,
    benjamini_hochberg_q_values,
    paired_blocked_permutation_test,
    utc_month_label,
)


DEPTH_STRESS_EVALUATION_SCHEMA_VERSION = "depth-stress-evaluation-v1"
DEPTH_STRESS_EVALUATION_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_BASELINE_NAMES = ("marginal", "pre_state_transition")
_PRIMARY_METRICS = ("negative_log_likelihood", "multiclass_brier")


def _model_evidence(artifact: DepthStressModelArtifact) -> dict[str, object]:
    return {
        "trading_authority": artifact.trading_authority,
        "profitability_claim": artifact.profitability_claim,
        "backend_kind": artifact.backend_kind,
        "backend_device": artifact.backend_device,
        "lightgbm_version": artifact.lightgbm_version,
        "seed": artifact.seed,
        "training_rows": artifact.training_rows,
        "tuning_rows": artifact.tuning_rows,
        "best_iteration": artifact.best_iteration,
        "model_sha256": artifact.model_sha256,
        "model_string_published": False,
    }


def _month_rows(values: np.ndarray, months: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(values, months, assume_unique=False))


def _validate_inputs(
    panel: DepthStressPanel,
    examples_by_horizon: Mapping[int, DepthStressExamples],
    eligible_month_ordinals: Sequence[int] | np.ndarray,
) -> np.ndarray:
    months = np.asarray(eligible_month_ordinals, dtype=np.int64)
    if (
        months.ndim != 1
        or len(months) < 8
        or len(np.unique(months)) != len(months)
        or np.any(np.diff(months) != 1)
        or tuple(sorted(examples_by_horizon)) != DEPTH_STRESS_HORIZONS_SECONDS
    ):
        raise ValueError("depth-stress rolling evaluation contract is invalid")
    available_months = set(int(value) for value in np.unique(panel.month_ordinals))
    if not set(int(value) for value in months).issubset(available_months):
        raise ValueError("depth-stress panel is missing an eligible month")
    for horizon, examples in examples_by_horizon.items():
        if (
            horizon != examples.horizon_seconds
            or examples.symbol != panel.symbol
            or examples.panel_sha256 != panel.panel_sha256
        ):
            raise ValueError("depth-stress examples do not match their panel")
        example_months = set(int(value) for value in np.unique(examples.month_ordinal))
        if not set(int(value) for value in months).issubset(example_months):
            raise ValueError(f"{horizon}s examples are missing an eligible month")
    return months


def evaluate_depth_stress_symbol(
    panel: DepthStressPanel,
    examples_by_horizon: Mapping[int, DepthStressExamples],
    *,
    eligible_month_ordinals: Sequence[int] | np.ndarray,
    compute_backend: str = "auto",
    maximum_iterations: int = 256,
    permutation_draws: int = 10_000,
    seed: int = 20260717,
    progress: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Run expanding monthly folds for one symbol and both frozen horizons."""

    eligible_months = _validate_inputs(
        panel,
        examples_by_horizon,
        eligible_month_ordinals,
    )
    panel_months = panel.month_ordinals
    folds: list[dict[str, object]] = []
    aggregate: dict[int, dict[str, object]] = {
        horizon: {
            "post_states": [],
            "utc_days": [],
            "probabilities": {
                "challenger": [],
                "marginal": [],
                "pre_state_transition": [],
            },
        }
        for horizon in DEPTH_STRESS_HORIZONS_SECONDS
    }
    for test_offset in range(7, len(eligible_months)):
        test_month = int(eligible_months[test_offset])
        tuning_month = int(eligible_months[test_offset - 1])
        training_months = eligible_months[: test_offset - 1]
        threshold_rows = _month_rows(panel_months, training_months)
        thresholds = fit_depth_stress_thresholds(panel.descriptors, threshold_rows)
        panel_states = assign_depth_stress_states(
            panel.descriptors,
            thresholds,
            fail_closed=False,
        )
        fold_horizons: list[dict[str, object]] = []
        for horizon in DEPTH_STRESS_HORIZONS_SECONDS:
            if progress is not None:
                progress(
                    "round62_fold_started",
                    symbol=panel.symbol,
                    horizon_seconds=horizon,
                    fold_index=test_offset - 6,
                    fold_count=len(eligible_months) - 7,
                    test_month=utc_month_label(test_month),
                )
            examples = examples_by_horizon[horizon]
            pre_states = panel_states[examples.pre_index]
            post_states = panel_states[examples.post_index]
            train_rows = _month_rows(examples.month_ordinal, training_months)
            tuning_rows = np.flatnonzero(examples.month_ordinal == tuning_month)
            test_rows = np.flatnonzero(examples.month_ordinal == test_month)
            if len(train_rows) < 90 or len(tuning_rows) < 30 or len(test_rows) < 30:
                raise ValueError(
                    f"{panel.symbol} {horizon}s {utc_month_label(test_month)} "
                    "has insufficient fold rows"
                )
            training_features = examples.feature_matrix(pre_states, rows=train_rows)
            tuning_features = examples.feature_matrix(pre_states, rows=tuning_rows)
            fitting_features = np.concatenate((training_features, tuning_features), axis=0)
            fitting_labels = np.concatenate(
                (post_states[train_rows], post_states[tuning_rows]),
                axis=0,
            )
            local_training = np.arange(len(train_rows), dtype=np.int64)
            local_tuning = np.arange(
                len(train_rows),
                len(train_rows) + len(tuning_rows),
                dtype=np.int64,
            )
            artifact = train_depth_stress_challenger(
                fitting_features,
                fitting_labels,
                train_rows=local_training,
                tuning_rows=local_tuning,
                feature_names=DEPTH_STRESS_FEATURE_NAMES,
                compute_backend=compute_backend,
                seed=seed,
                maximum_iterations=maximum_iterations,
            )
            test_features = examples.feature_matrix(pre_states, rows=test_rows)
            challenger = predict_depth_stress_challenger(artifact, test_features)
            marginal_matrix = fit_depth_transition_probabilities(
                pre_states,
                post_states,
                rows=train_rows,
                condition_on_pre_state=False,
            )
            transition_matrix = fit_depth_transition_probabilities(
                pre_states,
                post_states,
                rows=train_rows,
                condition_on_pre_state=True,
            )
            marginal = predict_depth_transition_probabilities(
                marginal_matrix,
                pre_states[test_rows],
            )
            transition = predict_depth_transition_probabilities(
                transition_matrix,
                pre_states[test_rows],
            )
            test_post_states = post_states[test_rows]
            fold_horizons.append(
                {
                    "horizon_seconds": horizon,
                    "training_examples": len(train_rows),
                    "tuning_examples": len(tuning_rows),
                    "test_examples": len(test_rows),
                    "model": _model_evidence(artifact),
                    "metrics": {
                        "challenger": asdict(
                            depth_stress_metrics(test_post_states, challenger)
                        ),
                        "marginal": asdict(depth_stress_metrics(test_post_states, marginal)),
                        "pre_state_transition": asdict(
                            depth_stress_metrics(test_post_states, transition)
                        ),
                    },
                }
            )
            if progress is not None:
                progress(
                    "round62_fold_completed",
                    symbol=panel.symbol,
                    horizon_seconds=horizon,
                    fold_index=test_offset - 6,
                    fold_count=len(eligible_months) - 7,
                    test_month=utc_month_label(test_month),
                    model_sha256=artifact.model_sha256,
                )
            horizon_aggregate = aggregate[horizon]
            horizon_aggregate["post_states"].append(test_post_states)
            horizon_aggregate["utc_days"].append(examples.utc_day[test_rows])
            horizon_aggregate["probabilities"]["challenger"].append(challenger)
            horizon_aggregate["probabilities"]["marginal"].append(marginal)
            horizon_aggregate["probabilities"]["pre_state_transition"].append(transition)
        folds.append(
            {
                "test_month": utc_month_label(test_month),
                "tuning_month": utc_month_label(tuning_month),
                "training_first_month": utc_month_label(int(training_months[0])),
                "training_last_month": utc_month_label(int(training_months[-1])),
                "training_months": len(training_months),
                "thresholds": {
                    "upper_tercile": list(thresholds.upper_tercile),
                    "fitted_rows": thresholds.fitted_rows,
                    "fit_fingerprint": thresholds.fit_fingerprint,
                },
                "horizons": fold_horizons,
            }
        )
    horizons: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for horizon in DEPTH_STRESS_HORIZONS_SECONDS:
        horizon_aggregate = aggregate[horizon]
        post_states = np.concatenate(horizon_aggregate["post_states"])
        utc_days = np.concatenate(horizon_aggregate["utc_days"])
        probabilities = {
            name: np.concatenate(horizon_aggregate["probabilities"][name], axis=0)
            for name in ("challenger", *_BASELINE_NAMES)
        }
        losses = {
            name: depth_stress_loss_rows(post_states, values)
            for name, values in probabilities.items()
        }
        horizons.append(
            {
                "horizon_seconds": horizon,
                "test_examples": len(post_states),
                "test_utc_days": len(np.unique(utc_days)),
                "metrics": {
                    name: asdict(depth_stress_metrics(post_states, values))
                    for name, values in probabilities.items()
                },
            }
        )
        for baseline_name in _BASELINE_NAMES:
            for metric_name in _PRIMARY_METRICS:
                comparison = paired_blocked_permutation_test(
                    losses[baseline_name][metric_name],
                    losses["challenger"][metric_name],
                    utc_days,
                    draws=permutation_draws,
                    seed=seed,
                )
                comparisons.append(
                    {
                        "symbol": panel.symbol,
                        "horizon_seconds": horizon,
                        "baseline": baseline_name,
                        "metric": metric_name,
                        **asdict(comparison),
                    }
                )
    return {
        "schema_version": DEPTH_STRESS_EVALUATION_SCHEMA_VERSION,
        "symbol": panel.symbol,
        "panel_sha256": panel.panel_sha256,
        "eligible_first_month": utc_month_label(int(eligible_months[0])),
        "eligible_last_month": utc_month_label(int(eligible_months[-1])),
        "eligible_months": len(eligible_months),
        "model_contract": {
            "artifact_schema_version": DEPTH_STRESS_MODEL_SCHEMA_VERSION,
            "model_family": "lightgbm_shallow_multiclass",
            "feature_names": list(DEPTH_STRESS_FEATURE_NAMES),
            "trading_authority": False,
            "profitability_claim": False,
        },
        "examples": [
            {
                "horizon_seconds": horizon,
                "rows": len(examples_by_horizon[horizon].anchor_time_ms),
                "examples_sha256": examples_by_horizon[horizon].examples_sha256,
            }
            for horizon in DEPTH_STRESS_HORIZONS_SECONDS
        ],
        "folds": folds,
        "horizons": horizons,
        "comparisons": comparisons,
        "profitability_claim": False,
        "trading_authority": False,
    }


def finalize_depth_stress_gate(
    symbol_reports: Sequence[Mapping[str, object]],
    *,
    maximum_q_value: float = 0.05,
    minimum_relative_improvement: float = 0.005,
) -> dict[str, object]:
    """Apply one FDR family and the strict all-comparison preregistered gate."""

    reports = list(symbol_reports)
    if not reports:
        raise ValueError("depth-stress gate requires symbol reports")
    comparisons: list[dict[str, object]] = []
    symbols: list[str] = []
    for report in reports:
        if (
            report.get("schema_version") != DEPTH_STRESS_EVALUATION_SCHEMA_VERSION
            or report.get("profitability_claim") is not False
            or report.get("trading_authority") is not False
            or not isinstance(report.get("comparisons"), list)
        ):
            raise ValueError("depth-stress symbol report contract is invalid")
        symbol = str(report.get("symbol", ""))
        if not symbol or symbol in symbols:
            raise ValueError("depth-stress symbol report identity is invalid")
        symbols.append(symbol)
        comparisons.extend(dict(value) for value in report["comparisons"])
    if tuple(symbols) != DEPTH_STRESS_EVALUATION_SYMBOLS:
        raise ValueError("depth-stress gate requires the frozen three-symbol panel")
    expected_comparisons = len(DEPTH_STRESS_EVALUATION_SYMBOLS) * len(
        DEPTH_STRESS_HORIZONS_SECONDS
    ) * 4
    if len(comparisons) != expected_comparisons:
        raise ValueError("depth-stress comparison family is incomplete")
    q_values = benjamini_hochberg_q_values(
        [float(value["one_sided_p_value"]) for value in comparisons]
    )
    passed = True
    for comparison, q_value in zip(comparisons, q_values, strict=True):
        comparison["q_value"] = float(q_value)
        comparison["passed"] = bool(
            float(comparison["relative_improvement"]) >= minimum_relative_improvement
            and float(q_value) <= maximum_q_value
        )
        passed = passed and bool(comparison["passed"])
    return {
        "schema_version": "depth-stress-gate-v1",
        "symbols": symbols,
        "comparison_count": len(comparisons),
        "multiple_testing": "Benjamini-Hochberg",
        "maximum_q_value": float(maximum_q_value),
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "comparisons": comparisons,
        "passed": passed,
        "decision": (
            "authorize_separately_frozen_paired_economic_replay"
            if passed
            else "reject_coarse_depth_stress_layer"
        ),
        "profitability_claim": False,
        "trading_authority": False,
    }


__all__ = [
    "DEPTH_STRESS_EVALUATION_SCHEMA_VERSION",
    "DEPTH_STRESS_EVALUATION_SYMBOLS",
    "evaluate_depth_stress_symbol",
    "finalize_depth_stress_gate",
]

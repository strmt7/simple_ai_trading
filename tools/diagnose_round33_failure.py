"""Diagnose Round 33 calibration failure without training or later-role access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import threading
import time
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_architecture import _auc  # noqa: E402
from simple_ai_trading.microstructure_selective_action_lightgbm import (  # noqa: E402
    SelectiveActionEnsembleBatch,
    TrainedSelectiveActionLightGBMModel,
    load_selective_action_lightgbm_model,
)
from simple_ai_trading.microstructure_selective_action_policy import (  # noqa: E402
    SelectiveActionPolicySpec,
    derive_selective_action_scores,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_selective_action_viability import (  # noqa: E402
    _validated_source,
)
from tools.run_selective_action_viability import (  # noqa: E402
    _canonical_sha256,
    _policy_spec,
    _predict_ensemble,
    _sha256_file,
    load_round33_design,
)
from tools.run_shared_action_viability import (  # noqa: E402
    _load_corpus,
    _parse_date,
    _role_indexes,
    _target_positions,
)


DIAGNOSTIC_SCHEMA_VERSION = "round-033-selective-routing-diagnostic-v1"
_FALSE_CLAIMS = {
    "trading_authority": False,
    "execution_claim": False,
    "profitability_claim": False,
    "portfolio_claim": False,
    "leverage_applied": False,
}
_DAY_MS = 86_400_000


def _load_models(
    evidence_root: Path,
    report: Mapping[str, object],
) -> list[TrainedSelectiveActionLightGBMModel]:
    models: list[TrainedSelectiveActionLightGBMModel] = []
    for evidence in report["ensemble_models"]:
        if not isinstance(evidence, Mapping):
            raise ValueError("Round 33 model evidence is invalid")
        relative = Path(str(evidence.get("artifact_path") or ""))
        path = evidence_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != int(evidence.get("artifact_bytes") or -1)
            or _sha256_file(path) != evidence.get("artifact_sha256")
        ):
            raise ValueError("Round 33 model artifact differs from its report")
        model = load_selective_action_lightgbm_model(path)
        if (
            model.model_sha256 != evidence.get("model_sha256")
            or model.seed != evidence.get("seed")
            or model.backend_kind != "opencl"
        ):
            raise ValueError("Round 33 model identity differs from its report")
        models.append(model)
    if len(models) != 3:
        raise ValueError("Round 33 ensemble is incomplete")
    return models


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    binary = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if (
        binary.ndim != 1
        or predicted.shape != binary.shape
        or len(binary) == 0
        or not set(np.unique(binary)).issubset({0, 1})
        or not np.all(np.isfinite(predicted))
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
    ):
        raise ValueError("binary diagnostic inputs are invalid")
    clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    bins: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        active = (predicted >= edges[index]) & (
            predicted <= edges[index + 1]
            if index == 9
            else predicted < edges[index + 1]
        )
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": int(np.sum(active)),
                "mean_probability": (
                    float(np.mean(predicted[active])) if np.any(active) else None
                ),
                "observed_positive_ratio": (
                    float(np.mean(binary[active])) if np.any(active) else None
                ),
            }
        )
    return {
        "rows": len(binary),
        "positive_rows": int(np.sum(binary)),
        "positive_ratio": float(np.mean(binary)),
        "roc_auc": _auc(binary, predicted),
        "brier_score": float(np.mean((predicted - binary) ** 2)),
        "log_loss": float(
            -np.mean(binary * np.log(clipped) + (1 - binary) * np.log1p(-clipped))
        ),
        "probability_minimum": float(np.min(predicted)),
        "probability_maximum": float(np.max(predicted)),
        "probability_mean": float(np.mean(predicted)),
        "probability_standard_deviation": float(np.std(predicted)),
        "probability_quantiles": {
            str(quantile): float(np.quantile(predicted, quantile))
            for quantile in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "reliability_bins": bins,
    }


def _top_rows(
    *,
    score: np.ndarray,
    side: np.ndarray,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
    active: np.ndarray | None = None,
) -> list[dict[str, object]]:
    values = np.asarray(score, dtype=np.float64)
    choices = np.asarray(side, dtype=np.int8)
    mask = np.ones(len(values), dtype=bool) if active is None else np.asarray(active)
    if (
        values.shape != choices.shape
        or mask.shape != values.shape
        or not np.all(np.isfinite(values))
        or np.any(~np.isin(choices, (-1, 1)))
    ):
        raise ValueError("ranking diagnostic inputs are invalid")
    indexes = np.flatnonzero(mask)
    order = indexes[np.argsort(-values[indexes], kind="stable")]
    selected_actual = np.where(choices == 1, long_actual, short_actual)
    rows: list[dict[str, object]] = []
    for requested in (100, 500, 1_000, 5_000):
        count = min(requested, len(order))
        chosen = order[:count]
        rows.append(
            {
                "requested_rows": requested,
                "actual_rows": count,
                "mean_selected_stress_net_bps": (
                    float(np.mean(selected_actual[chosen])) if count else None
                ),
                "median_selected_stress_net_bps": (
                    float(np.median(selected_actual[chosen])) if count else None
                ),
                "positive_ratio": (
                    float(np.mean(selected_actual[chosen] > 0.0)) if count else None
                ),
                "long_share": (
                    float(np.mean(choices[chosen] == 1)) if count else None
                ),
            }
        )
    return rows


def _routing_diagnostic(
    *,
    name: str,
    score: np.ndarray,
    side: np.ndarray,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
    active: np.ndarray | None = None,
) -> dict[str, object]:
    values = np.asarray(score, dtype=np.float64)
    choices = np.asarray(side, dtype=np.int8)
    mask = np.ones(len(values), dtype=bool) if active is None else np.asarray(active)
    selected_actual = np.where(choices == 1, long_actual, short_actual)
    signed_score = values * choices
    return {
        "routing": name,
        "active_rows": int(np.sum(mask)),
        "side_choice_auc": _auc((long_actual > short_actual).astype(np.int8), signed_score),
        "selected_profitability_auc": _auc(
            (selected_actual[mask] > 0.0).astype(np.int8),
            values[mask],
        ),
        "mean_selected_stress_net_bps": (
            float(np.mean(selected_actual[mask])) if np.any(mask) else None
        ),
        "positive_ratio": (
            float(np.mean(selected_actual[mask] > 0.0)) if np.any(mask) else None
        ),
        "long_share": (
            float(np.mean(choices[mask] == 1)) if np.any(mask) else None
        ),
        "top_rows": _top_rows(
            score=values,
            side=choices,
            long_actual=long_actual,
            short_actual=short_actual,
            active=mask,
        ),
    }


def _routing_diagnostics(
    ensemble: SelectiveActionEnsembleBatch,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
) -> list[dict[str, object]]:
    action = ensemble.action_values
    long_mean = np.asarray(action.long_mean_bps, dtype=np.float64)
    short_mean = np.asarray(action.short_mean_bps, dtype=np.float64)
    long_std = np.asarray(action.long_epistemic_std_bps, dtype=np.float64)
    short_std = np.asarray(action.short_epistemic_std_bps, dtype=np.float64)
    action_side = np.where(long_mean >= short_mean, 1, -1).astype(np.int8)
    action_score = np.maximum(long_mean - long_std, short_mean - short_std)
    opportunity = np.asarray(ensemble.opportunity_probability_mean, dtype=np.float64)
    conditional_long = np.asarray(
        ensemble.conditional_long_probability_mean,
        dtype=np.float64,
    )
    direction_side = np.where(conditional_long >= 0.5, 1, -1).astype(np.int8)
    direction_confidence = 2.0 * np.abs(conditional_long - 0.5)
    long_joint = np.asarray(action.long_profitable_probability, dtype=np.float64)
    short_joint = np.asarray(action.short_profitable_probability, dtype=np.float64)
    joint_score = np.maximum(long_joint, short_joint)
    abstain = 1.0 - opportunity
    consensus = action_side == direction_side
    return [
        _routing_diagnostic(
            name="action_value_mean_minus_one_epistemic_std",
            score=action_score,
            side=action_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="opportunity_probability_with_conditional_direction_side",
            score=opportunity,
            side=direction_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="joint_action_probability",
            score=joint_score,
            side=direction_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="joint_action_probability_minus_abstain_probability",
            score=joint_score - abstain,
            side=direction_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="opportunity_times_conditional_direction_confidence",
            score=opportunity * direction_confidence,
            side=direction_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="conditional_direction_confidence",
            score=direction_confidence,
            side=direction_side,
            long_actual=long_actual,
            short_actual=short_actual,
        ),
        _routing_diagnostic(
            name="action_value_direction_consensus_only",
            score=action_score,
            side=action_side,
            long_actual=long_actual,
            short_actual=short_actual,
            active=consensus,
        ),
    ]


def _gate_breakdown(
    ensemble: SelectiveActionEnsembleBatch,
    spec: SelectiveActionPolicySpec,
) -> dict[str, object]:
    action = ensemble.action_values
    base = spec.action_policy
    opportunity_floor = float(base.minimum_profitable_probability)
    direction_floor = float(spec.minimum_conditional_direction_confidence)
    agreement_floor = float(base.minimum_member_agreement)
    opportunity = np.asarray(ensemble.opportunity_probability_mean)
    direction = np.asarray(ensemble.conditional_long_probability_mean)
    opportunity_agreement = np.mean(
        ensemble.opportunity_member_probabilities >= opportunity_floor,
        axis=0,
    )
    common_masks = {
        "opportunity_probability": opportunity >= opportunity_floor,
        "opportunity_member_agreement": opportunity_agreement >= agreement_floor,
        "cross_head_side_consensus": (
            np.asarray(ensemble.side_consensus_member_ratio) >= agreement_floor
        ),
    }

    def side_masks(side: str) -> dict[str, np.ndarray]:
        is_long = side == "long"
        mean = np.asarray(action.long_mean_bps if is_long else action.short_mean_bps)
        std = np.asarray(
            action.long_epistemic_std_bps
            if is_long
            else action.short_epistemic_std_bps
        )
        lower = np.asarray(
            action.long_lower_bps if is_long else action.short_lower_bps
        )
        positive_ratio = np.asarray(
            action.long_positive_member_ratio
            if is_long
            else action.short_positive_member_ratio
        )
        member_direction = np.mean(
            ensemble.conditional_long_member_probabilities
            >= direction_floor
            if is_long
            else ensemble.conditional_long_member_probabilities
            <= 1.0 - direction_floor,
            axis=0,
        )
        return {
            **common_masks,
            "epistemic_adjusted_action_value_positive": (
                mean - float(base.epistemic_penalty) * std > 0.0
            ),
            "action_epistemic_std": std <= float(base.maximum_epistemic_std_bps),
            "action_lower_tail": lower >= float(base.minimum_lower_bound_bps),
            "action_positive_member_agreement": positive_ratio >= agreement_floor,
            "conditional_direction_confidence": (
                direction >= direction_floor
                if is_long
                else direction <= 1.0 - direction_floor
            ),
            "conditional_direction_member_agreement": (
                member_direction >= agreement_floor
            ),
        }

    outputs: dict[str, object] = {}
    final_masks: list[np.ndarray] = []
    for side in ("long", "short"):
        masks = side_masks(side)
        active = np.ones(ensemble.rows, dtype=bool)
        sequential: dict[str, int] = {}
        for name, mask in masks.items():
            active &= mask
            sequential[name] = int(np.sum(active))
        final_masks.append(active)
        outputs[side] = {
            "individual_pass_counts": {
                name: int(np.sum(mask)) for name, mask in masks.items()
            },
            "sequential_pass_counts": sequential,
            "final_side_eligible_rows": int(np.sum(active)),
        }
    score = derive_selective_action_scores(ensemble, spec)
    reconstructed = final_masks[0] | final_masks[1]
    if not np.array_equal(reconstructed, score.eligible):
        raise ValueError("Round 33 selective-policy gate reconstruction drifted")
    return {
        "profile": spec.profile,
        "rows": ensemble.rows,
        "long": outputs["long"],
        "short": outputs["short"],
        "final_eligible_rows": int(np.sum(score.eligible)),
    }


def _daily_diagnostics(
    decision_times_ms: np.ndarray,
    opportunity_labels: np.ndarray,
    direction_labels: np.ndarray,
    ensemble: SelectiveActionEnsembleBatch,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
) -> list[dict[str, object]]:
    days = np.asarray(decision_times_ms, dtype=np.int64) // _DAY_MS
    output: list[dict[str, object]] = []
    direction_probability = np.asarray(ensemble.conditional_long_probability_mean)
    opportunity_probability = np.asarray(ensemble.opportunity_probability_mean)
    direction_side = np.where(direction_probability >= 0.5, 1, -1)
    selected_actual = np.where(direction_side == 1, long_actual, short_actual)
    for day in np.unique(days):
        active = days == day
        conditional = active & opportunity_labels
        date_value = datetime.fromtimestamp(
            int(day) * 86_400,
            tz=timezone.utc,
        ).date()
        output.append(
            {
                "date": date_value.isoformat(),
                "rows": int(np.sum(active)),
                "opportunity_rows": int(np.sum(conditional)),
                "opportunity_auc": _auc(
                    opportunity_labels[active].astype(np.int8),
                    opportunity_probability[active],
                ),
                "conditional_direction_auc": _auc(
                    direction_labels[conditional].astype(np.int8),
                    direction_probability[conditional],
                ),
                "conditional_direction_accuracy": (
                    float(
                        np.mean(
                            direction_side[conditional]
                            == np.where(direction_labels[conditional], 1, -1)
                        )
                    )
                    if np.any(conditional)
                    else None
                ),
                "direction_routed_mean_stress_net_bps": float(
                    np.mean(selected_actual[active])
                ),
                "direction_routed_positive_ratio": float(
                    np.mean(selected_actual[active] > 0.0)
                ),
            }
        )
    return output


def _feature_importance(
    models: Sequence[TrainedSelectiveActionLightGBMModel],
    head: str,
) -> list[dict[str, object]]:
    normalized_gain: list[np.ndarray] = []
    normalized_split: list[np.ndarray] = []
    feature_names = models[0].source_feature_names
    for model in models:
        if model.source_feature_names != feature_names:
            raise ValueError("Round 33 model feature identities differ")
        booster = lgb.Booster(model_str=model.model_strings[head])
        gain = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
        split = np.asarray(booster.feature_importance("split"), dtype=np.float64)
        if gain.shape != (len(feature_names),) or split.shape != gain.shape:
            raise ValueError("Round 33 feature-importance shape drifted")
        normalized_gain.append(gain / max(float(np.sum(gain)), 1.0))
        normalized_split.append(split / max(float(np.sum(split)), 1.0))
    mean_gain = np.mean(np.stack(normalized_gain), axis=0)
    mean_split = np.mean(np.stack(normalized_split), axis=0)
    order = np.argsort(-mean_gain, kind="stable")[:30]
    return [
        {
            "rank": rank,
            "feature": feature_names[index],
            "mean_normalized_gain": float(mean_gain[index]),
            "mean_normalized_split": float(mean_split[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def run_diagnostic(
    *,
    design_path: Path,
    binding_path: Path,
    evidence_root: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
) -> dict[str, object]:
    design, design_sha, profiles = load_round33_design(design_path)
    _validated_design, source_report, source_report_sha = _validated_source(
        evidence_root=evidence_root,
        design_path=design_path,
        binding_path=binding_path,
    )
    status_path = output_path.with_suffix(".status.json")
    progress_lock = threading.Lock()
    sequence = 0
    started = time.monotonic()

    def progress(phase: str, **details: object) -> None:
        nonlocal sequence
        with progress_lock:
            sequence += 1
            value = {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "sequence": sequence,
                "phase": phase,
                "run_elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(
                "round33-diagnostic "
                + " ".join(f"{name}={item}" for name, item in value.items()),
                flush=True,
            )
            write_json_atomic(status_path, value, indent=2, sort_keys=True)

    roles = design["data"]["roles"]
    resources = design["runtime_resources"]
    assert isinstance(roles, Mapping)
    assert isinstance(resources, Mapping)
    training = _load_corpus(
        name="round33_calibration_diagnostic",
        design=design,
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        first=_parse_date(roles["train"]["start"], label="training start"),
        last=_parse_date(roles["development"]["end"], label="training end"),
        evaluation_first=_parse_date(
            roles["train"]["start"],
            label="evaluation start",
        ),
        evaluation_last=_parse_date(
            roles["development"]["end"],
            label="evaluation end",
        ),
        memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
        threads=min(int(resources["maximum_worker_threads"]), 16),
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    indexes, role_evidence, _maximum_exit = _role_indexes(
        training,
        roles,
        ("train", "early_stop", "calibration", "policy", "development"),
    )
    if (
        training.source_certificate["certificate_sha256"]
        != source_report["training_corpus"]["corpus_certificate_sha256"]
        or training.targets_sha256
        != source_report["training_corpus"]["barrier_targets_sha256"]
    ):
        raise ValueError("Round 33 diagnostic source identity changed")
    models = _load_models(evidence_root, source_report)
    ensemble = _predict_ensemble(
        models,
        training.dataset,
        indexes["calibration"],
        role="calibration_diagnostic",
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    positions = _target_positions(training.targets, ensemble.endpoint_indexes)
    long_actual = np.asarray(
        training.targets.stress_long_net_bps[positions],
        dtype=np.float64,
    )
    short_actual = np.asarray(
        training.targets.stress_short_net_bps[positions],
        dtype=np.float64,
    )
    opportunity_labels = np.maximum(long_actual, short_actual) > 0.0
    direction_labels = long_actual > short_actual
    direction_population = opportunity_labels
    direction_floors = design["conditional_direction_confidence"]
    assert isinstance(direction_floors, Mapping)
    temperatures = [float(model.direction_temperature) for model in models]
    result: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "round": 33,
        "diagnostic_sha256": "PENDING",
        "source_design_sha256": design_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_corpus_certificate_sha256": training.source_certificate[
            "certificate_sha256"
        ],
        "source_cache_key": training.cache_key,
        "source_cache_state": training.cache_state,
        "barrier_targets_sha256": training.targets_sha256,
        "calibration_role": role_evidence["calibration"],
        "model_sha256": [model.model_sha256 for model in models],
        "scope": {
            "training_performed": False,
            "thresholds_evaluated_or_changed": False,
            "policy_prediction_accessed": False,
            "development_prediction_accessed": False,
            "distant_confirmation_accessed": False,
            "untouched_dates_accessed": False,
        },
        "label_population": {
            "rows": len(long_actual),
            "opportunity_rows": int(np.sum(opportunity_labels)),
            "abstain_rows": int(np.sum(~opportunity_labels)),
            "long_preferred_opportunity_rows": int(
                np.sum(direction_labels & opportunity_labels)
            ),
            "short_preferred_opportunity_rows": int(
                np.sum(~direction_labels & opportunity_labels)
            ),
            "both_sides_profitable_rows": int(
                np.sum((long_actual > 0.0) & (short_actual > 0.0))
            ),
        },
        "probability_calibration": {
            "opportunity": _binary_metrics(
                opportunity_labels.astype(np.int8),
                ensemble.opportunity_probability_mean,
            ),
            "conditional_direction": _binary_metrics(
                direction_labels[direction_population].astype(np.int8),
                ensemble.conditional_long_probability_mean[direction_population],
            ),
            "member_direction_temperatures": temperatures,
            "configured_optimization_temperature_upper_boundary": math.exp(4.0),
            "all_direction_temperatures_at_upper_boundary": all(
                math.isclose(
                    value,
                    math.exp(4.0),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for value in temperatures
            ),
        },
        "routing_diagnostics": _routing_diagnostics(
            ensemble,
            long_actual,
            short_actual,
        ),
        "policy_gate_breakdown": [
            _gate_breakdown(
                ensemble,
                _policy_spec(profile, direction_floors),
            )
            for profile in profiles
        ],
        "daily_stability": _daily_diagnostics(
            training.dataset.decision_time_ms[ensemble.endpoint_indexes],
            opportunity_labels,
            direction_labels,
            ensemble,
            long_actual,
            short_actual,
        ),
        "feature_importance": {
            "opportunity_probability": _feature_importance(
                models,
                "opportunity_probability",
            ),
            "conditional_direction_probability": _feature_importance(
                models,
                "conditional_direction_probability",
            ),
        },
        **_FALSE_CLAIMS,
    }
    canonical = dict(result)
    canonical.pop("diagnostic_sha256")
    result["diagnostic_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    progress("complete", diagnostic_sha256=result["diagnostic_sha256"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose the rejected Round 33 calibration model."
    )
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-033-selective-action-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-033-execution-binding.json",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_diagnostic(
        design_path=arguments.design,
        binding_path=arguments.binding,
        evidence_root=arguments.evidence_root,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_path=arguments.output,
    )
    print(
        f"Round 33 diagnostic complete sha256={result['diagnostic_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

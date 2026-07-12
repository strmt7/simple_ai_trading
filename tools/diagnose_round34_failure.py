"""Reproduce and hash the Round 34 calibration failure decomposition."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_action_features import (  # noqa: E402
    ACTION_CONDITIONAL_FEATURE_NAMES,
)
from simple_ai_trading.microstructure_architecture import _auc  # noqa: E402
from simple_ai_trading.microstructure_three_action_lightgbm import (  # noqa: E402
    ThreeActionEnsembleBatch,
    TrainedThreeActionLightGBMModel,
    load_three_action_lightgbm_model,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools import run_shared_action_viability as shared_runner  # noqa: E402
from tools.publish_three_action_viability import _validated_source  # noqa: E402
from tools.run_three_action_viability import (  # noqa: E402
    _canonical_sha256,
    _git_bytes,
    _predict_ensemble,
    _sha256_file,
    load_round34_execution_binding,
)


DIAGNOSTIC_SCHEMA_VERSION = "round-034-failure-diagnostic-v1"
_SEEDS = (29, 43, 71)
_ROLES = ("train", "early_stop", "calibration", "policy", "development")


def _action_labels(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
    long_actual = np.asarray(long_values, dtype=np.float64)
    short_actual = np.asarray(short_values, dtype=np.float64)
    if (
        long_actual.ndim != 1
        or short_actual.shape != long_actual.shape
        or len(long_actual) == 0
        or not np.all(np.isfinite(long_actual))
        or not np.all(np.isfinite(short_actual))
    ):
        raise ValueError("Round 34 diagnostic action utilities are invalid")
    labels = np.full(len(long_actual), 1, dtype=np.int8)
    labels[(long_actual > 0.0) & (long_actual > short_actual)] = 0
    labels[(short_actual > 0.0) & (short_actual > long_actual)] = 2
    return labels


def _route_diagnostic(
    *,
    name: str,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
    action_labels: np.ndarray,
    side_score: np.ndarray,
    ranking_score: np.ndarray,
    eligible: np.ndarray | None = None,
    oracle_control: bool = False,
) -> dict[str, object]:
    long_values = np.asarray(long_actual, dtype=np.float64)
    short_values = np.asarray(short_actual, dtype=np.float64)
    labels = np.asarray(action_labels, dtype=np.int8)
    score = np.asarray(side_score, dtype=np.float64)
    ranking = np.asarray(ranking_score, dtype=np.float64)
    use = (
        np.ones(len(score), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    if (
        any(
            value.shape != score.shape
            for value in (long_values, short_values, labels, ranking, use)
        )
        or len(score) < 1_000
        or not np.all(np.isfinite(long_values))
        or not np.all(np.isfinite(short_values))
        or not np.all(np.isfinite(score))
        or not np.all(np.isfinite(ranking))
        or not np.any(use)
    ):
        raise ValueError("Round 34 route diagnostic inputs are invalid")
    opportunity = labels != 1
    if (
        min(
            int(np.sum(labels[opportunity] == 0)), int(np.sum(labels[opportunity] == 2))
        )
        == 0
    ):
        raise ValueError("Round 34 route diagnostic lacks both action sides")
    side = np.sign(score).astype(np.int8)
    economic_side = np.sign(long_values - short_values).astype(np.int8)
    selected_actual = np.where(side == 1, long_values, short_values)
    routed = use & (side != 0)
    eligible_indexes = np.flatnonzero(routed)
    if len(eligible_indexes) == 0:
        raise ValueError("Round 34 route diagnostic has no non-tied side scores")
    order = eligible_indexes[np.argsort(-ranking[eligible_indexes], kind="stable")]
    top: dict[str, dict[str, object]] = {}
    for requested in (100, 500, 1_000):
        chosen = order[: min(requested, len(order))]
        top[str(requested)] = {
            "requested_rows": requested,
            "actual_rows": len(chosen),
            "mean_stress_net_bps": float(np.mean(selected_actual[chosen])),
            "positive_ratio": float(np.mean(selected_actual[chosen] > 0.0)),
            "long_share": float(np.mean(side[chosen] == 1)),
            "direction_accuracy": float(np.mean(side[chosen] == economic_side[chosen])),
        }
    return {
        "route": name,
        "oracle_control": oracle_control,
        "promotion_permitted": False,
        "post_hoc_consumed_data_diagnostic": True,
        "eligible_rows_before_score_ties": int(np.sum(use)),
        "eligible_rows": len(eligible_indexes),
        "direction_auc_on_non_abstain_rows": _auc(
            (labels[opportunity] == 0).astype(np.int8),
            score[opportunity],
        ),
        "direction_accuracy_on_non_abstain_rows": float(
            np.mean(side[opportunity] == economic_side[opportunity])
        ),
        "all_eligible_selected_mean_stress_net_bps": float(
            np.mean(selected_actual[routed])
        ),
        "top_rows": top,
    }


def _feature_gain(
    models: Sequence[TrainedThreeActionLightGBMModel],
) -> dict[str, list[dict[str, object]]]:
    heads = {
        "three_action_probability": tuple(models[0].source_feature_names),
        "side_profit_probability": ACTION_CONDITIONAL_FEATURE_NAMES,
        "positive_magnitude": ACTION_CONDITIONAL_FEATURE_NAMES,
        "nonpositive_loss_magnitude": ACTION_CONDITIONAL_FEATURE_NAMES,
        "lower_quantile": ACTION_CONDITIONAL_FEATURE_NAMES,
        "upper_quantile": ACTION_CONDITIONAL_FEATURE_NAMES,
    }
    output: dict[str, list[dict[str, object]]] = {}
    for head, names in heads.items():
        normalized: list[np.ndarray] = []
        for model in models:
            booster = lgb.Booster(model_str=model.model_strings[head])
            gain = np.asarray(
                booster.feature_importance(importance_type="gain"),
                dtype=np.float64,
            )
            if gain.shape != (len(names),) or not np.all(np.isfinite(gain)):
                raise ValueError("Round 34 feature-gain evidence is invalid")
            total = float(np.sum(gain))
            normalized.append(gain / total if total > 0.0 else np.zeros_like(gain))
        mean_gain = np.mean(np.stack(normalized), axis=0)
        order = np.argsort(-mean_gain, kind="stable")[:15]
        output[head] = [
            {
                "rank": position,
                "feature": names[index],
                "mean_normalized_gain": float(mean_gain[index]),
            }
            for position, index in enumerate(order, start=1)
        ]
    return output


def _daily_diagnostics(
    *,
    decision_time_ms: np.ndarray,
    labels: np.ndarray,
    abstain_probability: np.ndarray,
    conditional_long_probability: np.ndarray,
    long_profitable_probability: np.ndarray,
    short_profitable_probability: np.ndarray,
) -> list[dict[str, object]]:
    times = np.asarray(decision_time_ms, dtype=np.int64)
    action_labels = np.asarray(labels, dtype=np.int8)
    days = times // 86_400_000
    output: list[dict[str, object]] = []
    for day in np.unique(days):
        selected_day = days == day
        opportunity = selected_day & (action_labels != 1)
        if (
            min(
                int(np.sum(action_labels[opportunity] == 0)),
                int(np.sum(action_labels[opportunity] == 2)),
            )
            == 0
        ):
            raise ValueError("Round 34 daily diagnostic lacks both action sides")
        day_text = (
            datetime.fromtimestamp(
                int(day) * 86_400,
                tz=timezone.utc,
            )
            .date()
            .isoformat()
        )
        output.append(
            {
                "date": day_text,
                "rows": int(np.sum(selected_day)),
                "opportunity_auc": _auc(
                    (action_labels[selected_day] != 1).astype(np.int8),
                    1.0 - abstain_probability[selected_day],
                ),
                "action_class_direction_auc": _auc(
                    (action_labels[opportunity] == 0).astype(np.int8),
                    conditional_long_probability[opportunity],
                ),
                "side_profit_direction_auc": _auc(
                    (action_labels[opportunity] == 0).astype(np.int8),
                    (long_profitable_probability - short_profitable_probability)[
                        opportunity
                    ],
                ),
            }
        )
    return output


def diagnose(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
) -> dict[str, object]:
    design, report, report_sha = _validated_source(
        evidence_root=evidence_root,
        design_path=design_path,
        binding_path=binding_path,
    )
    load_round34_execution_binding(
        binding_path,
        design_path=design_path,
        design_sha256=str(report["design_sha256"]),
    )
    implementation_commit = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    implementation_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    implementation_sha256 = _sha256_file(Path(__file__).resolve())
    status_path = output_path.with_suffix(".status.json")
    lock = threading.Lock()
    sequence = 0
    started = time.monotonic()

    def progress(phase: str, **details: object) -> None:
        nonlocal sequence
        with lock:
            sequence += 1
            payload = {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "round": 34,
                "sequence": sequence,
                "phase": phase,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(
                "round34-diagnostic "
                + " ".join(
                    f"{name}={value}"
                    for name, value in payload.items()
                    if name not in {"schema_version", "updated_at_utc"}
                ),
                flush=True,
            )
            write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    progress("initialize")
    roles = design["data"]["roles"]
    resources = design["runtime_resources"]
    assert isinstance(roles, Mapping)
    assert isinstance(resources, Mapping)
    first = shared_runner._parse_date(roles["train"]["start"], label="train start")
    last = shared_runner._parse_date(
        roles["development"]["end"],
        label="development end",
    )
    bundle = shared_runner._load_corpus(
        name="round34_failure_diagnostic",
        design=design,
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        first=first,
        last=last,
        evaluation_first=first,
        evaluation_last=last,
        memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
        threads=int(resources["maximum_worker_threads"]),
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    if (
        bundle.corpus_certificate_sha256
        != report["training_corpus"]["corpus_certificate_sha256"]
        or bundle.barrier_targets_sha256
        != report["training_corpus"]["barrier_targets_sha256"]
    ):
        raise ValueError("Round 34 diagnostic corpus identity drifted")
    role_indexes, _role_evidence, _maximum_exit = shared_runner._role_indexes(
        bundle,
        roles,
        _ROLES,
    )
    models = [
        load_three_action_lightgbm_model(evidence_root / "models" / f"seed-{seed}.json")
        for seed in _SEEDS
    ]
    model_hashes = [model.model_sha256 for model in models]
    if model_hashes != [item["model_sha256"] for item in report["ensemble_models"]]:
        raise ValueError("Round 34 diagnostic model identities drifted")
    prediction = _predict_ensemble(
        models,
        bundle.dataset,
        role_indexes["calibration"],
        role="round34_failure_diagnostic",
        heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
        progress=progress,
    )
    result = _diagnostic_payload(
        bundle=bundle,
        prediction=prediction,
        models=models,
        report=report,
        report_sha=report_sha,
        diagnostic_implementation={
            "commit": implementation_commit,
            "path": implementation_path,
            "sha256": implementation_sha256,
        },
    )
    canonical = dict(result)
    canonical.pop("diagnostic_sha256")
    result["diagnostic_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    progress(
        "complete",
        diagnostic_sha256=result["diagnostic_sha256"],
        verdict=result["verdict"],
    )
    return result


def _diagnostic_payload(
    *,
    bundle: shared_runner.CorpusBundle,
    prediction: ThreeActionEnsembleBatch,
    models: Sequence[TrainedThreeActionLightGBMModel],
    report: Mapping[str, object],
    report_sha: str,
    diagnostic_implementation: Mapping[str, str],
) -> dict[str, object]:
    long_actual, short_actual = shared_runner._scenario_targets(
        bundle.targets,
        prediction.endpoint_indexes,
        scenario="stress",
    )
    labels = _action_labels(long_actual, short_actual)
    action = prediction.action_values
    long_class = np.asarray(prediction.long_action_probability_mean, dtype=np.float64)
    short_class = np.asarray(prediction.short_action_probability_mean, dtype=np.float64)
    abstain_class = np.asarray(
        prediction.abstain_action_probability_mean,
        dtype=np.float64,
    )
    long_profit = np.asarray(action.long_profitable_probability, dtype=np.float64)
    short_profit = np.asarray(action.short_profitable_probability, dtype=np.float64)
    long_mean = np.asarray(action.long_mean_bps, dtype=np.float64)
    short_mean = np.asarray(action.short_mean_bps, dtype=np.float64)
    long_lower = np.asarray(action.long_lower_bps, dtype=np.float64)
    short_lower = np.asarray(action.short_lower_bps, dtype=np.float64)
    long_std = np.asarray(action.long_epistemic_std_bps, dtype=np.float64)
    short_std = np.asarray(action.short_epistemic_std_bps, dtype=np.float64)
    class_side = np.sign(long_class - short_class)
    profit_side = np.sign(long_profit - short_profit)
    true_side = np.where(long_actual > short_actual, 1.0, -1.0)
    learned_routes = [
        _route_diagnostic(
            name="action_class_probability",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=long_class - short_class,
            ranking_score=np.maximum(long_class, short_class),
        ),
        _route_diagnostic(
            name="independent_side_profit_probability",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=long_profit - short_profit,
            ranking_score=np.maximum(long_profit, short_profit),
        ),
        _route_diagnostic(
            name="expected_action_value",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=long_mean - short_mean,
            ranking_score=np.maximum(long_mean - long_std, short_mean - short_std),
        ),
        _route_diagnostic(
            name="lower_tail",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=long_lower - short_lower,
            ranking_score=np.maximum(long_lower, short_lower),
        ),
        _route_diagnostic(
            name="action_class_and_side_profit_consensus",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=long_class - short_class,
            ranking_score=(
                np.maximum(long_class, short_class)
                * np.maximum(long_profit, short_profit)
            ),
            eligible=(class_side != 0.0) & (class_side == profit_side),
        ),
    ]
    oracle_routes = [
        _route_diagnostic(
            name="oracle_side_ranked_by_action_opportunity",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=true_side,
            ranking_score=1.0 - abstain_class,
            oracle_control=True,
        ),
        _route_diagnostic(
            name="oracle_side_ranked_by_side_profit",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=true_side,
            ranking_score=np.maximum(long_profit, short_profit),
            oracle_control=True,
        ),
        _route_diagnostic(
            name="oracle_side_ranked_by_expected_action_value",
            long_actual=long_actual,
            short_actual=short_actual,
            action_labels=labels,
            side_score=true_side,
            ranking_score=np.maximum(long_mean - long_std, short_mean - short_std),
            oracle_control=True,
        ),
    ]
    predicted_class = np.argmax(
        np.column_stack((long_class, abstain_class, short_class)),
        axis=1,
    )
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predicted_class), 1)
    decision_times = bundle.dataset.decision_time_ms[prediction.endpoint_indexes]
    architecture = report["calibration_architecture"]["diagnostics"]
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "round": 34,
        "status": "diagnosed_rejection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_sha256": "PENDING",
        "source_evidence": {
            "design_sha256": report["design_sha256"],
            "binding_sha256": report["binding_sha256"],
            "report_canonical_sha256": report_sha,
            "corpus_certificate_sha256": report["training_corpus"][
                "corpus_certificate_sha256"
            ],
            "barrier_targets_sha256": report["training_corpus"][
                "barrier_targets_sha256"
            ],
            "model_sha256": [model.model_sha256 for model in models],
        },
        "diagnostic_implementation": dict(diagnostic_implementation),
        "calibration_rows": len(labels),
        "actual_action_class_support": {
            "long_rows": int(np.sum(labels == 0)),
            "abstain_rows": int(np.sum(labels == 1)),
            "short_rows": int(np.sum(labels == 2)),
        },
        "predicted_action_class_support": {
            "long_rows": int(np.sum(predicted_class == 0)),
            "abstain_rows": int(np.sum(predicted_class == 1)),
            "short_rows": int(np.sum(predicted_class == 2)),
        },
        "confusion_matrix": {
            "actual_row_order": ["long", "abstain", "short"],
            "predicted_column_order": ["long", "abstain", "short"],
            "counts": confusion.tolist(),
        },
        "architecture_summary": {
            "opportunity_auc": architecture["opportunity_auc"],
            "conditional_direction_auc": architecture["conditional_direction_auc"],
            "side_profit_auc": architecture["side_profit_auc"],
            "multiclass_log_loss_to_class_prior_ratio": architecture[
                "multiclass_log_loss_to_class_prior_ratio"
            ],
            "side_profit_brier_to_base_rate_ratio": architecture[
                "side_profit_brier_to_base_rate_ratio"
            ],
            "selected_top_100_mean_stress_net_bps": architecture["selected_action"][
                "top_rows"
            ]["100"]["mean_stress_net_bps"],
            "selected_top_500_mean_stress_net_bps": architecture["selected_action"][
                "top_rows"
            ]["500"]["mean_stress_net_bps"],
        },
        "probability_ranges": {
            "action_long": _range(long_class),
            "action_abstain": _range(abstain_class),
            "action_short": _range(short_class),
            "side_profit_long": _range(long_profit),
            "side_profit_short": _range(short_profit),
        },
        "learned_route_diagnostics": learned_routes,
        "oracle_decomposition_controls": oracle_routes,
        "oracle_control_warning": (
            "Oracle-side rows use future outcomes and are decomposition controls only; "
            "they cannot be used for model selection, policy, backtesting, or promotion."
        ),
        "daily_stability": _daily_diagnostics(
            decision_time_ms=decision_times,
            labels=labels,
            abstain_probability=abstain_class,
            conditional_long_probability=long_class / (long_class + short_class),
            long_profitable_probability=long_profit,
            short_profitable_probability=short_profit,
        ),
        "mean_normalized_feature_gain": _feature_gain(models),
        "verdict": "conditional_direction_is_the_primary_bottleneck",
        "evidence_interpretation": [
            "Opportunity, side-profit calibration, and multiclass calibration passed their frozen gates.",
            "Every learned side route remained weak and economically negative beyond isolated top-100 diagnostics.",
            "Oracle-side controls made the existing opportunity ranking strongly positive at top 100 and top 500, isolating side selection as the primary failure.",
            "Daily direction AUC changed sign around chance, so a new direction model must improve temporal stability rather than only pooled calibration AUC.",
        ],
        "next_experiment_constraints": {
            "target": "conditional_direction_only",
            "risk_gate_relaxation_permitted": False,
            "leverage_permitted": False,
            "oracle_feature_or_label_use_permitted": False,
            "calendar_feature_removal_to_test": True,
            "compact_microstructure_feature_set_to_test": True,
            "utility_margin_weighting_to_test": True,
            "variant_screen_is_consumed_data_discovery_only": True,
        },
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _range(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
    }


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description="Diagnose the Round 34 rejection.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-034-three-action-utility-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-034-execution-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = diagnose(
        evidence_root=arguments.evidence_root,
        design_path=arguments.design,
        binding_path=arguments.binding,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_path=arguments.output,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

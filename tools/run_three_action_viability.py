"""Execute the hash-bound Round 34 utility-weighted three-action study."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.microstructure_action_features import (  # noqa: E402
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ACTION_POLICY_SCHEMA_VERSION,
    select_barrier_threshold,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    _auc,
    average_label_uniqueness,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    AdaptiveBarrierTargets,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_selective_action_policy import (  # noqa: E402
    SELECTIVE_ACTION_POLICY_SCHEMA_VERSION,
    derive_selective_action_scores,
)
from simple_ai_trading.microstructure_three_action_lightgbm import (  # noqa: E402
    THREE_ACTION_LIGHTGBM_SCHEMA_VERSION,
    ThreeActionEnsembleBatch,
    ThreeActionLightGBMSpec,
    ThreeActionPredictionBatch,
    TrainedThreeActionLightGBMModel,
    as_selective_action_ensemble,
    ensemble_three_action_predictions,
    load_three_action_lightgbm_model,
    predict_three_action_lightgbm_model,
    save_three_action_lightgbm_model,
    train_three_action_lightgbm_model,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools import run_selective_action_viability as selective_runner  # noqa: E402
from tools import run_shared_action_viability as shared_runner  # noqa: E402


DESIGN_SCHEMA_VERSION = "three-action-utility-design-v2"
BINDING_SCHEMA_VERSION = "round-034-execution-binding-v1"
REPORT_SCHEMA_VERSION = "three-action-utility-viability-report-v1"
_ROUND = 34
_TRAINING_ROLES = ("train", "early_stop", "calibration", "policy", "development")
_PROFILE_NAMES = ("conservative", "regular", "aggressive")
_PREDICTION_ARRAY_FIELDS = (
    "long_mean_bps",
    "short_mean_bps",
    "long_profitable_probability",
    "short_profitable_probability",
    "long_action_probability",
    "abstain_action_probability",
    "short_action_probability",
    "opportunity_probability",
    "conditional_long_probability",
    "long_lower_bps",
    "short_lower_bps",
    "long_upper_bps",
    "short_upper_bps",
)
_PREDICTION_DISCRETE_FIELDS = (
    "action_preference_side",
    "decision_preference_side",
    "side_consensus",
)
_REQUIRED_BOUND_PATHS = frozenset(
    {
        "docs/model-research/action-value/consumed-periods-through-round-033.json",
        "docs/model-research/action-value/round-031-frozen-chronological-confirmation-design.json",
        "docs/model-research/action-value/round-033-failure-analysis.json",
        "docs/model-research/action-value/round-033-selective-action-design.json",
        "docs/model-research/action-value/round-034-three-action-utility-design.json",
        "src/simple_ai_trading/assets.py",
        "src/simple_ai_trading/compute.py",
        "src/simple_ai_trading/lightgbm_backend.py",
        "src/simple_ai_trading/microstructure_action_architecture.py",
        "src/simple_ai_trading/microstructure_action_features.py",
        "src/simple_ai_trading/microstructure_action_policy.py",
        "src/simple_ai_trading/microstructure_architecture.py",
        "src/simple_ai_trading/microstructure_barriers.py",
        "src/simple_ai_trading/microstructure_cache.py",
        "src/simple_ai_trading/microstructure_features.py",
        "src/simple_ai_trading/microstructure_model.py",
        "src/simple_ai_trading/microstructure_outcome_lightgbm.py",
        "src/simple_ai_trading/microstructure_selective_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_selective_action_policy.py",
        "src/simple_ai_trading/microstructure_shared_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_three_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_warehouse.py",
        "src/simple_ai_trading/probability_calibration.py",
        "src/simple_ai_trading/progress_heartbeat.py",
        "src/simple_ai_trading/storage.py",
        "tools/run_selective_action_viability.py",
        "tools/run_shared_action_viability.py",
        "tools/run_three_action_viability.py",
    }
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_git_oid(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _three_action_spec(design: Mapping[str, object]) -> ThreeActionLightGBMSpec:
    model = design.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Round 34 model contract is missing")
    lightgbm = model.get("lightgbm")
    if not isinstance(lightgbm, Mapping):
        raise ValueError("Round 34 LightGBM contract is missing")
    parameters = dict(lightgbm)
    backend = parameters.pop("backend", None)
    cpu_fallback = parameters.pop("cpu_fallback_permitted", None)
    if backend != "opencl" or cpu_fallback is not False:
        raise ValueError("Round 34 accelerator contract changed")
    return ThreeActionLightGBMSpec(
        candidate_id=str(model.get("candidate_id") or ""),
        family=str(model.get("family") or ""),
        **parameters,
    )


def _validate_failure_analysis(
    design_path: Path,
    predecessor: Mapping[str, object],
) -> None:
    relative = Path(str(predecessor.get("failure_analysis") or ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.name != str(relative)
    ):
        raise ValueError("Round 34 failure-analysis path is unsafe")
    path = design_path.parent / relative
    if _sha256_file(path) != predecessor.get("failure_analysis_file_sha256"):
        raise ValueError("Round 34 failure-analysis file changed")
    analysis = _read_object(path, label="Round 33 failure analysis")
    canonical = dict(analysis)
    claimed = canonical.pop("analysis_sha256", None)
    source = analysis.get("source_evidence")
    if (
        analysis.get("schema_version") != "selective-action-failure-analysis-v1"
        or analysis.get("round") != 33
        or analysis.get("status") != "rejected"
        or claimed != _canonical_sha256(canonical)
        or claimed != predecessor.get("failure_analysis_canonical_sha256")
        or not isinstance(source, Mapping)
        or source.get("diagnostic_canonical_sha256")
        != predecessor.get("routing_diagnostic_sha256")
        or source.get("design_sha256") != predecessor.get("design_sha256")
        or source.get("report_canonical_sha256")
        != predecessor.get("report_canonical_sha256")
    ):
        raise ValueError("Round 34 failure-analysis provenance is invalid")


def _validate_consumed_registry(
    design_path: Path,
    design: Mapping[str, object],
) -> None:
    governance = design["governance"]
    data = design["data"]
    assert isinstance(governance, Mapping)
    assert isinstance(data, Mapping)
    registry_path = design_path.parent / str(governance["consumed_period_registry"])
    if _sha256_file(registry_path) != governance.get(
        "consumed_period_registry_file_sha256"
    ):
        raise ValueError("Round 34 consumed-period registry file changed")
    registry = _read_object(registry_path, label="Round 34 consumed-period registry")
    canonical = dict(registry)
    claimed = canonical.pop("registry_sha256", None)
    if (
        registry.get("schema_version") != "action-value-consumed-periods-v1"
        or claimed != _canonical_sha256(canonical)
        or claimed != governance.get("consumed_period_registry_canonical_sha256")
    ):
        raise ValueError("Round 34 consumed-period registry hash is invalid")
    consumed_dates: set[str] = set()
    for record in registry.get("records") or ():
        if not isinstance(record, Mapping):
            raise ValueError("Round 34 consumed-period record is invalid")
        for window in record.get("windows") or ():
            if not isinstance(window, Mapping):
                raise ValueError("Round 34 consumed-period window is invalid")
            consumed_dates.update(
                shared_runner._date_set(
                    window.get("start_date"),
                    window.get("end_date"),
                    label="consumed",
                )
            )
    roles = data.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Round 34 chronological roles are missing")
    evaluated_dates: set[str] = set()
    for role in _TRAINING_ROLES:
        value = roles.get(role)
        if not isinstance(value, Mapping):
            raise ValueError(f"Round 34 {role} role is invalid")
        evaluated_dates.update(
            shared_runner._date_set(
                value.get("start"),
                value.get("end"),
                label=role,
            )
        )
    distant = roles.get("distant_confirmation")
    if not isinstance(distant, Mapping):
        raise ValueError("Round 34 distant-confirmation role is invalid")
    evaluated_dates.update(
        shared_runner._date_set(
            distant.get("start"),
            distant.get("end"),
            label="distant confirmation",
        )
    )
    if not evaluated_dates <= consumed_dates:
        raise ValueError("Round 34 attempts to access an unconsumed target date")


def load_round34_design(
    path: str | Path,
) -> tuple[dict[str, object], str, tuple[dict[str, object], ...]]:
    """Load and semantically validate the frozen Round 34 design."""

    source_path = Path(path).resolve()
    payload = _read_object(source_path, label="Round 34 design")
    expected_keys = {
        "schema_version",
        "round",
        "design_revision",
        "design_sha256",
        "purpose",
        "supersedes",
        "claims",
        "predecessor",
        "risk_profiles_source",
        "conditional_direction_confidence",
        "governance",
        "data",
        "execution",
        "barrier_targets",
        "event_sampler",
        "sample_weighting",
        "model",
        "selection",
        "stage_evaluation_order",
        "runtime_resources",
        "acceptance_gates",
        "research_basis",
        "limitations",
    }
    canonical = dict(payload)
    claimed = canonical.pop("design_sha256", None)
    claims = {
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "trading_authority": False,
        "leverage_applied": False,
    }
    if (
        set(payload) != expected_keys
        or not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != _ROUND
        or payload.get("design_revision") != 4
        or payload.get("purpose")
        != "consumed_data_utility_weighted_symmetric_three_action_viability"
        or payload.get("claims") != claims
    ):
        raise ValueError("Round 34 design hash, structure, or claims are invalid")
    supersedes = payload.get("supersedes")
    if (
        not isinstance(supersedes, Mapping)
        or supersedes.get("design_revision") != 3
        or supersedes.get("design_sha256")
        != "ac3033d4dbe47de39becdf0a6aa6632f21dc6e4d3b91ae43c2f4f110e50de485"
        or supersedes.get("file_sha256")
        != "dbfc114b7252f44c6246fb65b327eee96377be4db20d5a2c37378f3393fe90f2"
        or not isinstance(supersedes.get("reason"), str)
        or "abstain class includes exact long-short utility ties"
        not in str(supersedes["reason"])
    ):
        raise ValueError("Round 34 design-revision provenance is invalid")

    previous_path = source_path.with_name("round-033-selective-action-design.json")
    previous, previous_sha, _previous_profiles = selective_runner.load_round33_design(
        previous_path
    )
    predecessor = payload.get("predecessor")
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("round") != 33
        or predecessor.get("status") != "rejected"
        or predecessor.get("design_sha256") != previous_sha
        or not _is_sha256(predecessor.get("report_canonical_sha256"))
        or not _is_sha256(predecessor.get("routing_diagnostic_sha256"))
    ):
        raise ValueError("Round 34 predecessor contract is invalid")
    _validate_failure_analysis(source_path, predecessor)
    _validate_consumed_registry(source_path, payload)
    profiles, _risk_source = shared_runner._risk_profiles_from_source(
        source_path,
        payload,
    )
    risk_source = payload.get("risk_profiles_source")
    if (
        not isinstance(risk_source, Mapping)
        or risk_source.get("profitable_probability_semantic")
        != "probability_that_either_long_or_short_is_the_profitable_after_cost_action"
    ):
        raise ValueError("Round 34 policy opportunity semantic changed")

    for name in (
        "data",
        "execution",
        "barrier_targets",
        "event_sampler",
        "selection",
        "stage_evaluation_order",
        "runtime_resources",
    ):
        if payload[name] != previous[name]:
            raise ValueError(f"Round 34 {name} drifted from Round 33")
    if payload.get("conditional_direction_confidence") != previous.get(
        "conditional_direction_confidence"
    ):
        raise ValueError("Round 34 direction-confidence contract changed")
    governance = payload.get("governance")
    if not isinstance(governance, Mapping) or governance != {
        "consumed_period_registry": "consumed-periods-through-round-033.json",
        "consumed_period_registry_file_sha256": (
            "f999e39aa276dd9393e139fb46c98da61c2923e84984bb920104ac365e6104fd"
        ),
        "consumed_period_registry_canonical_sha256": (
            "3e27e4500530315bb52f1730111c113876f2912da42c1dd5b90314b4f07ecb04"
        ),
        "variant_budget": 1,
        "hyperparameter_search_permitted": False,
        "all_target_dates_already_consumed": True,
        "thresholds_may_use_calibration_only": True,
        "later_stage_parameters_may_not_change": True,
        "no_passing_candidate_action": "reject_and_abstain",
        "risk_gate_relaxation_permitted": False,
    }:
        raise ValueError("Round 34 governance contract changed")

    model = payload.get("model")
    weighting = payload.get("sample_weighting")
    gates = payload.get("acceptance_gates")
    if not all(isinstance(value, Mapping) for value in (model, weighting, gates)):
        raise ValueError("Round 34 model, weighting, or gates are incomplete")
    assert isinstance(model, Mapping)
    assert isinstance(weighting, Mapping)
    assert isinstance(gates, Mapping)
    spec = _three_action_spec(payload)
    if (
        model.get("candidate_id") != "utility-weighted-symmetric-three-action-lightgbm"
        or model.get("family")
        != "utility_weighted_symmetric_three_action_lightgbm_hurdle"
        or tuple(model.get("action_class_order") or ()) != ("long", "abstain", "short")
        or tuple(model.get("seeds") or ()) != (29, 43, 71)
        or tuple(model.get("heads") or ())
        != (
            "three_action_probability",
            "shared_side_profit_probability",
            "shared_conditional_positive_magnitude",
            "shared_conditional_nonpositive_loss_magnitude",
            "shared_lower_quantile_0_10",
            "shared_upper_quantile_0_90",
        )
        or model.get("lightgbm") != previous["model"]["lightgbm"]
        or model.get("tuning_partition") != previous["model"]["tuning_partition"]
        or spec.lower_quantile != 0.1
        or spec.upper_quantile != 0.9
        or spec.calibration_fraction != 0.5
    ):
        raise ValueError("Round 34 model split, heads, or seed contract changed")
    labels = model.get("labels")
    symmetry = model.get("symmetry")
    derivation = model.get("probability_derivation")
    side_calibration = model.get("side_profit_calibration")
    multiclass_calibration = model.get("multiclass_calibration")
    if (
        labels
        != {
            "long": "stress_long_net_bps_is_positive_and_greater_than_stress_short_net_bps",
            "short": "stress_short_net_bps_is_positive_and_greater_than_stress_long_net_bps",
            "abstain": "neither_side_is_both_strictly_positive_and_strictly_better_than_the_other",
            "both_profitable_resolution": "choose_higher_stress_net_bps",
            "exact_tie_resolution": "abstain",
        }
        or symmetry
        != {
            "training": "raw_and_exact_mirror_with_long_short_labels_swapped_and_abstain_unchanged",
            "prediction": "mean_raw_logits_with_mirrored_logits_after_long_short_class_swap",
            "multiclass_calibration": "shared_positive_temperature_plus_abstain_logit_bias_preserves_long_short_mirror_equivariance",
            "side_profit_calibration": "one_shared_bounded_platt_transform_is_applied_to_both_canonical_action_rows",
            "action_magnitude_features": "signed_features_multiplied_by_action_side_and_bid_ask_depth_mapped_to_supporting_opposing",
            "unknown_feature_action": "fail_closed",
        }
        or derivation
        != {
            "p_action_long": "direct_multiclass_probability_that_long_is_the_best_action",
            "p_action_short": "direct_multiclass_probability_that_short_is_the_best_action",
            "p_action_abstain": "direct_multiclass_probability_that_abstain_is_the_best_action",
            "p_opportunity": "p_action_long_plus_p_action_short",
            "p_long_given_opportunity": "p_action_long_divided_by_p_action_long_plus_p_action_short",
            "multiclass_probabilities_sum_to_one": True,
            "p_long_profitable": "shared_side_profit_probability_on_the_canonical_long_action_row",
            "p_short_profitable": "shared_side_profit_probability_on_the_canonical_short_action_row",
            "side_profit_probabilities_are_not_forced_to_sum_to_one": True,
            "semantic_aliasing_between_action_class_and_side_profit_probabilities_permitted": False,
        }
        or side_calibration
        != {
            "optimizer": "shared_bounded_platt_scaling_with_damped_loss_decreasing_newton_steps",
            "objective": "unweighted_original_paired_action_row_binary_log_loss",
            "slope_bounds": [0.05, 10.0],
            "intercept_bounds": [-10.0, 10.0],
            "separate_long_short_parameters_permitted": False,
        }
        or multiclass_calibration
        != {
            "optimizer": "deterministic_projected_newton_in_inverse_temperature_and_abstain_bias",
            "objective": "unweighted_original_row_multiclass_log_loss",
            "log_temperature_bounds": [-4.0, 4.0],
            "abstain_logit_bias_bounds": [-5.0, 5.0],
            "maximum_iterations": 100,
            "gradient_tolerance": 1e-10,
            "hessian_diagonal_ridge": 1e-12,
            "long_short_bias_permitted": False,
        }
    ):
        raise ValueError("Round 34 probability semantics or calibration changed")
    expected_weighting = {
        "base_method": "average_label_uniqueness",
        "training_role": "train",
        "tuning_role": "early_stop",
        "computed_from": "decision_time_ms_and_maximum_realized_exit_time_ms",
        "multiclass_decision_utilities": [
            "stress_long_net_bps",
            "zero_bps_abstain",
            "stress_short_net_bps",
        ],
        "decision_regret_span": "maximum_decision_utility_minus_minimum_decision_utility",
        "regret_scale": "training_role_median_positive_decision_regret_span",
        "regret_multiplier": "decision_regret_span_divided_by_regret_scale_clipped_to_0_5_and_3_0",
        "multiclass_event_weight": "average_label_uniqueness_times_regret_multiplier",
        "mirrored_multiclass_rows_share_half_event_weight": True,
        "side_profit_probability_magnitude_and_quantile_heads_retain_unmodified_average_label_uniqueness_weight": True,
        "probability_calibration_is_unweighted": True,
    }
    if weighting != expected_weighting:
        raise ValueError("Round 34 sample-weighting contract changed")
    expected_implementation = {
        "maximum_class_probability_mirror_swap_error": 1e-10,
        "maximum_action_swap_equivariance_error": 1e-10,
        "maximum_probability_sum_error": 1e-12,
        "single_source_certificate_per_dataset_load": True,
        "maximum_progress_silence_seconds": 30,
        "artifact_reload_equivalence_required": True,
        "nonfinite_prediction_count": 0,
    }
    expected_architecture = {
        "minimum_opportunity_auc": 0.65,
        "minimum_conditional_direction_auc": 0.55,
        "minimum_side_profit_auc": 0.55,
        "maximum_side_profit_brier_to_base_rate_ratio": 1.0,
        "maximum_multiclass_log_loss_to_class_prior_ratio": 1.0,
        "minimum_selected_top_100_mean_stress_net_bps": 0.0,
        "minimum_selected_top_500_mean_stress_net_bps": 0.0,
    }
    if (
        gates.get("implementation") != expected_implementation
        or gates.get("calibration_architecture") != expected_architecture
        or gates.get("distant_confirmation_forecast")
        != previous["acceptance_gates"]["distant_confirmation_forecast"]
        or gates.get("economic") != previous["acceptance_gates"]["economic"]
    ):
        raise ValueError("Round 34 acceptance gates changed")
    research = payload.get("research_basis")
    limitations = payload.get("limitations")
    if (
        not isinstance(research, list)
        or len(research) < 6
        or any(
            not isinstance(item, Mapping)
            or not str(item.get("url") or "").startswith("https://")
            for item in research
        )
        or not isinstance(limitations, list)
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("Round 34 research basis or limitations are incomplete")
    return payload, str(claimed), profiles


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 34 Git binding command failed") from exc


def load_round34_execution_binding(
    path: str | Path,
    *,
    design_path: str | Path,
    design_sha256: str,
    require_current_implementation: bool = False,
) -> tuple[dict[str, object], str]:
    """Verify historical provenance and optionally authorize a current-code run."""

    binding = _read_object(Path(path), label="Round 34 execution binding")
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256", None)
    implementation = binding.get("implementation")
    design = binding.get("design")
    if (
        not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("round") != _ROUND
        or binding.get("worktree_policy") != "clean_including_untracked"
        or not isinstance(implementation, Mapping)
        or not isinstance(design, Mapping)
    ):
        raise ValueError("Round 34 execution binding is invalid")
    commit = str(implementation.get("commit") or "").lower()
    files_value = implementation.get("files")
    if (
        not _is_git_oid(commit)
        or implementation.get("hash_mode") != "git_blob_sha256_v1"
        or not isinstance(files_value, list)
    ):
        raise ValueError("Round 34 implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    bound_files: dict[str, str] = {}
    for item in files_value:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("Round 34 bound file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in bound_files
        ):
            raise ValueError("Round 34 bound path is unsafe or duplicated")
        bound_files[normalized] = str(item["sha256"])
    if set(bound_files) != _REQUIRED_BOUND_PATHS:
        missing = sorted(_REQUIRED_BOUND_PATHS - set(bound_files))
        extra = sorted(set(bound_files) - _REQUIRED_BOUND_PATHS)
        raise ValueError(
            f"Round 34 bound file scope changed: missing={missing} extra={extra}"
        )
    for normalized, expected in bound_files.items():
        historical = _git_bytes("show", f"{commit}:{normalized}")
        if hashlib.sha256(historical).hexdigest() != expected:
            raise ValueError(f"Round 34 historical binding changed: {normalized}")
    if require_current_implementation:
        for normalized, expected in bound_files.items():
            current = _git_bytes("show", f"HEAD:{normalized}")
            if hashlib.sha256(current).hexdigest() != expected:
                raise ValueError(f"Round 34 implementation changed: {normalized}")
        if _git_bytes("status", "--porcelain", "--untracked-files=all").strip():
            raise ValueError("Round 34 execution requires a clean worktree")
    resolved_design_path = Path(design_path).resolve()
    relative_design = resolved_design_path.relative_to(ROOT).as_posix()
    if (
        design.get("path") != relative_design
        or design.get("design_sha256") != design_sha256
        or design.get("file_sha256") != bound_files.get(relative_design)
        or hashlib.sha256(resolved_design_path.read_bytes()).hexdigest()
        != bound_files.get(relative_design)
    ):
        raise ValueError("Round 34 design binding changed")
    return binding, str(claimed)


def _prediction_max_abs_difference(
    left: ThreeActionPredictionBatch,
    right: ThreeActionPredictionBatch,
) -> float:
    if not np.array_equal(left.endpoint_indexes, right.endpoint_indexes):
        return math.inf
    differences = [
        float(
            np.max(
                np.abs(
                    np.asarray(getattr(left, name), dtype=np.float64)
                    - np.asarray(getattr(right, name), dtype=np.float64)
                )
            )
        )
        for name in _PREDICTION_ARRAY_FIELDS
    ]
    if any(
        not np.array_equal(getattr(left, name), getattr(right, name))
        for name in _PREDICTION_DISCRETE_FIELDS
    ):
        return math.inf
    return max(differences, default=0.0)


def _symmetry_errors(
    model: TrainedThreeActionLightGBMModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> dict[str, float]:
    subset = shared_runner._subset_dataset(dataset, endpoints)
    local = np.arange(subset.rows, dtype=np.int64)
    original = predict_three_action_lightgbm_model(model, subset, local)
    mirrored = predict_three_action_lightgbm_model(
        model,
        replace(
            subset,
            features=mirror_microstructure_direction(subset.features),
        ),
        local,
    )
    class_errors = [
        float(
            np.max(
                np.abs(
                    original.long_action_probability - mirrored.short_action_probability
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    original.short_action_probability - mirrored.long_action_probability
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    original.abstain_action_probability
                    - mirrored.abstain_action_probability
                )
            )
        ),
    ]
    action_errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(getattr(original, long_name), dtype=np.float64)
                    - np.asarray(getattr(mirrored, short_name), dtype=np.float64)
                )
            )
        )
        for long_name, short_name in (
            ("long_mean_bps", "short_mean_bps"),
            ("short_mean_bps", "long_mean_bps"),
            ("long_profitable_probability", "short_profitable_probability"),
            ("short_profitable_probability", "long_profitable_probability"),
            ("long_lower_bps", "short_lower_bps"),
            ("short_lower_bps", "long_lower_bps"),
            ("long_upper_bps", "short_upper_bps"),
            ("short_upper_bps", "long_upper_bps"),
        )
    ]
    if not np.array_equal(
        original.action_preference_side,
        -mirrored.action_preference_side,
    ) or not np.array_equal(
        original.decision_preference_side,
        -mirrored.decision_preference_side,
    ):
        action_errors.append(math.inf)
    original_sum = (
        original.long_action_probability
        + original.abstain_action_probability
        + original.short_action_probability
    )
    mirrored_sum = (
        mirrored.long_action_probability
        + mirrored.abstain_action_probability
        + mirrored.short_action_probability
    )
    return {
        "class_probability_mirror_swap_max_abs_error": max(class_errors),
        "action_swap_equivariance_max_abs_error": max(action_errors),
        "probability_sum_max_abs_error": max(
            float(np.max(np.abs(original_sum - 1.0))),
            float(np.max(np.abs(mirrored_sum - 1.0))),
        ),
    }


def _prediction_nonfinite_count(ensemble: ThreeActionEnsembleBatch) -> int:
    action = ensemble.action_values
    arrays = (
        action.long_mean_bps,
        action.short_mean_bps,
        action.long_epistemic_std_bps,
        action.short_epistemic_std_bps,
        action.long_profitable_probability,
        action.short_profitable_probability,
        action.long_lower_bps,
        action.short_lower_bps,
        action.long_upper_bps,
        action.short_upper_bps,
        ensemble.long_action_probability_mean,
        ensemble.abstain_action_probability_mean,
        ensemble.short_action_probability_mean,
        ensemble.opportunity_probability_mean,
        ensemble.opportunity_probability_std,
        ensemble.conditional_long_probability_mean,
        ensemble.conditional_long_probability_std,
        ensemble.long_action_member_probabilities,
        ensemble.abstain_action_member_probabilities,
        ensemble.short_action_member_probabilities,
        ensemble.opportunity_member_probabilities,
        ensemble.conditional_long_member_probabilities,
        ensemble.direction_long_member_ratio,
        ensemble.direction_short_member_ratio,
        ensemble.side_consensus_member_ratio,
    )
    return int(sum(np.sum(~np.isfinite(value)) for value in arrays))


def _predict_ensemble(
    models: Sequence[TrainedThreeActionLightGBMModel],
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    *,
    role: str,
    heartbeat_seconds: float,
    progress: Callable[..., None],
) -> ThreeActionEnsembleBatch:
    members: list[ThreeActionPredictionBatch] = []
    for position, model in enumerate(models, start=1):
        progress(
            "prediction-member-start",
            role=role,
            member=position,
            members=len(models),
            seed=model.seed,
        )
        with progress_heartbeat(
            progress,
            phase="prediction-member-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"role": role, "member": position, "seed": model.seed},
        ):
            members.append(
                predict_three_action_lightgbm_model(model, dataset, endpoints)
            )
    ensemble = ensemble_three_action_predictions(members)
    nonfinite = _prediction_nonfinite_count(ensemble)
    if nonfinite:
        raise ValueError(f"{role} ensemble emitted {nonfinite} non-finite values")
    progress("prediction-complete", role=role, rows=ensemble.rows)
    return ensemble


def _multiclass_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    predicted = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        predicted.shape != (len(target), 3)
        or len(target) == 0
        or not np.all(np.isfinite(predicted))
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
        or not np.allclose(np.sum(predicted, axis=1), 1.0, rtol=0.0, atol=1e-12)
        or np.any(~np.isin(target, (0, 1, 2)))
    ):
        raise ValueError("Round 34 multiclass diagnostic inputs are invalid")
    chosen = np.clip(predicted[np.arange(len(target)), target], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def _calibration_architecture_diagnostics(
    targets: AdaptiveBarrierTargets,
    ensemble: ThreeActionEnsembleBatch,
) -> dict[str, object]:
    long_actual, short_actual = shared_runner._scenario_targets(
        targets,
        ensemble.endpoint_indexes,
        scenario="stress",
    )
    labels = np.full(len(long_actual), 1, dtype=np.int64)
    labels[(long_actual > 0.0) & (long_actual > short_actual)] = 0
    labels[(short_actual > 0.0) & (short_actual > long_actual)] = 2
    opportunity = labels != 1
    long_preferred = labels == 0
    probabilities = np.column_stack(
        (
            ensemble.long_action_probability_mean,
            ensemble.abstain_action_probability_mean,
            ensemble.short_action_probability_mean,
        )
    )
    multiclass_loss = _multiclass_log_loss(probabilities, labels)
    class_counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(class_counts == 0.0):
        raise ValueError("Round 34 calibration lacks an action class")
    class_prior = class_counts / np.sum(class_counts)
    prior_probabilities = np.broadcast_to(class_prior, probabilities.shape)
    class_prior_loss = _multiclass_log_loss(prior_probabilities, labels)
    side_labels = np.concatenate((long_actual > 0.0, short_actual > 0.0)).astype(
        np.int8
    )
    side_probabilities = np.concatenate(
        (
            ensemble.action_values.long_profitable_probability,
            ensemble.action_values.short_profitable_probability,
        )
    )
    side_brier = float(np.mean((side_probabilities - side_labels) ** 2))
    side_prevalence = float(np.mean(side_labels))
    side_prior_brier = float(np.mean((side_prevalence - side_labels) ** 2))
    if side_prior_brier <= 0.0:
        raise ValueError("Round 34 calibration lacks a side-profit outcome")
    selected = selective_runner._selected_action_diagnostics(
        targets,
        as_selective_action_ensemble(ensemble),
    )
    return {
        "rows": len(long_actual),
        "opportunity_rows": int(np.sum(opportunity)),
        "abstain_rows": int(np.sum(~opportunity)),
        "conditional_direction_rows": int(np.sum(opportunity)),
        "opportunity_auc": _auc(
            opportunity.astype(np.int8),
            ensemble.opportunity_probability_mean,
        ),
        "conditional_direction_auc": _auc(
            long_preferred[opportunity].astype(np.int8),
            ensemble.conditional_long_probability_mean[opportunity],
        ),
        "multiclass_log_loss": multiclass_loss,
        "class_prior_multiclass_log_loss": class_prior_loss,
        "multiclass_log_loss_to_class_prior_ratio": (
            multiclass_loss / class_prior_loss
        ),
        "action_class_support": {
            "long_rows": int(class_counts[0]),
            "abstain_rows": int(class_counts[1]),
            "short_rows": int(class_counts[2]),
        },
        "side_profit_rows": len(side_labels),
        "side_profit_positive_rows": int(np.sum(side_labels)),
        "side_profit_auc": _auc(side_labels, side_probabilities),
        "side_profit_brier": side_brier,
        "side_profit_base_rate_brier": side_prior_brier,
        "side_profit_brier_to_base_rate_ratio": side_brier / side_prior_brier,
        "selected_action": selected,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _architecture_gate_reasons(
    diagnostics: Mapping[str, object],
    gates: Mapping[str, object],
) -> list[str]:
    selected = diagnostics.get("selected_action")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("top_rows"), Mapping
    ):
        raise ValueError("Round 34 selected-action diagnostics are incomplete")
    top = selected["top_rows"]
    assert isinstance(top, Mapping)
    top100 = top.get("100")
    top500 = top.get("500")
    if not isinstance(top100, Mapping) or not isinstance(top500, Mapping):
        raise ValueError("Round 34 selected-action ranking rows are missing")
    reasons: list[str] = []
    comparisons = (
        (
            "opportunity_auc",
            "minimum_opportunity_auc",
            "opportunity_auc_gate_failed",
            "minimum",
        ),
        (
            "conditional_direction_auc",
            "minimum_conditional_direction_auc",
            "conditional_direction_auc_gate_failed",
            "minimum",
        ),
        (
            "side_profit_auc",
            "minimum_side_profit_auc",
            "side_profit_auc_gate_failed",
            "minimum",
        ),
        (
            "side_profit_brier_to_base_rate_ratio",
            "maximum_side_profit_brier_to_base_rate_ratio",
            "side_profit_brier_gate_failed",
            "maximum",
        ),
        (
            "multiclass_log_loss_to_class_prior_ratio",
            "maximum_multiclass_log_loss_to_class_prior_ratio",
            "multiclass_log_loss_gate_failed",
            "maximum",
        ),
    )
    for metric, gate, reason, direction in comparisons:
        actual = float(diagnostics[metric])
        limit = float(gates[gate])
        if (
            not math.isfinite(actual)
            or (direction == "minimum" and actual < limit)
            or (direction == "maximum" and actual > limit)
        ):
            reasons.append(reason)
    top100_mean = float(top100["mean_stress_net_bps"])
    top500_mean = float(top500["mean_stress_net_bps"])
    if not math.isfinite(top100_mean) or top100_mean <= float(
        gates["minimum_selected_top_100_mean_stress_net_bps"]
    ):
        reasons.append("selected_top_100_mean_stress_net_gate_failed")
    if not math.isfinite(top500_mean) or top500_mean <= float(
        gates["minimum_selected_top_500_mean_stress_net_bps"]
    ):
        reasons.append("selected_top_500_mean_stress_net_gate_failed")
    return reasons


def _evaluate_frozen_stage(
    *,
    stage: str,
    survivors: Sequence[str],
    prediction: ThreeActionEnsembleBatch,
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    expected_days: Sequence[int],
    profiles_by_name: Mapping[str, Mapping[str, object]],
    profile_results: dict[str, dict[str, object]],
    direction_floors: Mapping[str, object],
    progress: Callable[..., None],
) -> list[str]:
    output: list[str] = []
    gate_name = (
        f"{stage}_gates" if stage != "distant_confirmation" else "development_gates"
    )
    policy_input = as_selective_action_ensemble(prediction)
    for profile in survivors:
        raw = profiles_by_name[profile]
        calibration = profile_results[profile]["calibration"]
        assert isinstance(calibration, Mapping)
        threshold_selection = calibration.get("threshold_selection")
        if not isinstance(threshold_selection, Mapping):
            raise ValueError("Round 34 frozen threshold is missing")
        score = derive_selective_action_scores(
            policy_input,
            selective_runner._policy_spec(raw, direction_floors),
        )
        result = shared_runner._trace_result(
            dataset=dataset,
            targets=targets,
            score=score,
            threshold_bps=float(threshold_selection["threshold_bps"]),
            expected_days=expected_days,
            gates=raw[gate_name],
        )
        profile_results[profile][stage] = result
        if result["status"] == "research_candidate":
            output.append(profile)
        progress(
            f"profile-{stage.replace('_', '-')}-complete",
            profile=profile,
            status=result["status"],
            stress_trades=result["stress_trace"]["metrics"]["trades"],
        )
    return output


def run_three_action_viability(
    *,
    design_path: str | Path,
    binding_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    design, design_sha, profiles = load_round34_design(design_path)
    binding, binding_sha = load_round34_execution_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
        require_current_implementation=True,
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Round 34 output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    progress_lock = threading.Lock()
    progress_sequence = 0
    started = time.monotonic()
    resources = design["runtime_resources"]
    assert isinstance(resources, Mapping)
    heartbeat_seconds = float(resources["heartbeat_interval_seconds"])
    memory_limit = f"{int(resources['duckdb_memory_limit_gib'])}GB"
    maximum_threads = int(resources["maximum_worker_threads"])
    threads = maximum_threads

    def progress(phase: str, **details: object) -> None:
        nonlocal progress_sequence
        with progress_lock:
            progress_sequence += 1
            payload = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "round": _ROUND,
                "design_sha256": design_sha,
                "binding_sha256": binding_sha,
                "sequence": progress_sequence,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "phase": phase,
                **details,
            }
            print(
                "round34 "
                + " ".join(
                    f"{name}={value}"
                    for name, value in payload.items()
                    if name not in {"schema_version", "updated_at_utc"}
                ),
                flush=True,
            )
            write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    try:
        progress("initialize")
        thread_evidence = shared_runner._configure_worker_threads(maximum_threads)
        threads = int(thread_evidence["effective_worker_threads"])
        progress("directml-preflight-start")
        directml = shared_runner._attest_directml()
        progress(
            "directml-preflight-complete",
            device=directml["device"],
            vendor=directml["vendor"],
        )
        runtime = {
            "duckdb_memory_limit": memory_limit,
            "memory_contract_note": (
                "12 GiB is the DuckDB allocation ceiling, not a process-wide hard "
                "limit; the distant corpus is loaded only after development passes."
            ),
            "thread_limits": thread_evidence,
            "directml_attestation": directml,
            "lightgbm_training_backend_required": "opencl",
            "lightgbm_gpu_use_dp_required": True,
            "lightgbm_prediction_backend": "cpu_library_prediction_path",
            "cpu_fallback_permitted": False,
            "heartbeat_interval_seconds": heartbeat_seconds,
        }
        data = design["data"]
        assert isinstance(data, Mapping)
        roles = data["roles"]
        assert isinstance(roles, Mapping)
        train_role = roles["train"]
        development_role = roles["development"]
        assert isinstance(train_role, Mapping)
        assert isinstance(development_role, Mapping)
        training_first = shared_runner._parse_date(
            train_role["start"],
            label="training corpus start",
        )
        training_last = shared_runner._parse_date(
            development_role["end"],
            label="training corpus end",
        )
        training = shared_runner._load_corpus(
            name="training_and_near_evaluation",
            design=design,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            first=training_first,
            last=training_last,
            evaluation_first=training_first,
            evaluation_last=training_last,
            memory_limit=memory_limit,
            threads=threads,
            heartbeat_seconds=heartbeat_seconds,
            progress=progress,
        )
        role_indexes, role_evidence, maximum_exit = shared_runner._role_indexes(
            training,
            roles,
            _TRAINING_ROLES,
        )
        train_weights = average_label_uniqueness(
            training.dataset.decision_time_ms,
            maximum_exit,
            role_indexes["train"],
        )
        tuning_weights = average_label_uniqueness(
            training.dataset.decision_time_ms,
            maximum_exit,
            role_indexes["early_stop"],
        )
        spec = _three_action_spec(design)
        model_contract = design["model"]
        assert isinstance(model_contract, Mapping)
        seeds = tuple(int(seed) for seed in model_contract["seeds"])
        models: list[TrainedThreeActionLightGBMModel] = []
        model_evidence: list[dict[str, object]] = []
        reload_sample = shared_runner._sample_indexes(role_indexes["calibration"])
        acceptance = design["acceptance_gates"]
        assert isinstance(acceptance, Mapping)
        implementation_gates = acceptance["implementation"]
        assert isinstance(implementation_gates, Mapping)
        symmetry_limits = {
            "class_probability_mirror_swap_max_abs_error": float(
                implementation_gates["maximum_class_probability_mirror_swap_error"]
            ),
            "action_swap_equivariance_max_abs_error": float(
                implementation_gates["maximum_action_swap_equivariance_error"]
            ),
            "probability_sum_max_abs_error": float(
                implementation_gates["maximum_probability_sum_error"]
            ),
        }
        for member, seed in enumerate(seeds, start=1):
            progress(
                "model-train-start",
                member=member,
                members=len(seeds),
                seed=seed,
            )
            with progress_heartbeat(
                progress,
                phase="model-train-heartbeat",
                interval_seconds=heartbeat_seconds,
                details={"member": member, "seed": seed},
            ):
                model = train_three_action_lightgbm_model(
                    training.dataset,
                    training.targets,
                    train_endpoints=role_indexes["train"],
                    tuning_endpoints=role_indexes["early_stop"],
                    spec=spec,
                    compute_backend="directml",
                    seed=seed,
                    train_sample_weights=train_weights,
                    tuning_sample_weights=tuning_weights,
                    progress=lambda head, step, total, index=member, model_seed=seed: (
                        progress(
                            "model-head-start",
                            member=index,
                            seed=model_seed,
                            head=head,
                            head_step=step,
                            head_total=total,
                        )
                    ),
                )
            if model.backend_kind != "opencl" or model.backend_requested != "directml":
                raise RuntimeError("Round 34 model training did not remain on OpenCL")
            artifact_path = destination / "models" / f"seed-{seed}.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            progress("model-artifact-verification-start", member=member, seed=seed)
            with progress_heartbeat(
                progress,
                phase="model-artifact-verification-heartbeat",
                interval_seconds=heartbeat_seconds,
                details={"member": member, "seed": seed},
            ):
                save_three_action_lightgbm_model(artifact_path, model)
                reloaded = load_three_action_lightgbm_model(artifact_path)
                original_prediction = predict_three_action_lightgbm_model(
                    model,
                    training.dataset,
                    reload_sample,
                )
                reloaded_prediction = predict_three_action_lightgbm_model(
                    reloaded,
                    training.dataset,
                    reload_sample,
                )
                reload_error = _prediction_max_abs_difference(
                    original_prediction,
                    reloaded_prediction,
                )
                symmetry = _symmetry_errors(
                    reloaded,
                    training.dataset,
                    reload_sample,
                )
            if reload_error != 0.0:
                raise ValueError("Round 34 artifact reload changed predictions")
            for name, limit in symmetry_limits.items():
                if not math.isfinite(symmetry[name]) or symmetry[name] > limit:
                    raise ValueError(f"Round 34 {name} gate failed")
            models.append(reloaded)
            model_evidence.append(
                {
                    "member": member,
                    "seed": seed,
                    "schema_version": reloaded.schema_version,
                    "model_family": reloaded.model_family,
                    "model_sha256": reloaded.model_sha256,
                    "artifact_path": artifact_path.relative_to(destination).as_posix(),
                    "artifact_sha256": _sha256_file(artifact_path),
                    "artifact_bytes": artifact_path.stat().st_size,
                    "backend_requested": reloaded.backend_requested,
                    "backend_kind": reloaded.backend_kind,
                    "backend_device": reloaded.backend_device,
                    "training_event_rows": reloaded.training_event_rows,
                    "requested_tuning_event_rows": reloaded.requested_tuning_event_rows,
                    "early_stop_event_rows": reloaded.early_stop_event_rows,
                    "probability_calibration_event_rows": reloaded.calibration_event_rows,
                    "internal_purged_event_rows": reloaded.internal_purged_event_rows,
                    "action_class_support": {
                        name: dict(value)
                        for name, value in reloaded.action_class_support.items()
                    },
                    "side_profit_class_support": {
                        name: dict(value)
                        for name, value in reloaded.side_profit_class_support.items()
                    },
                    "regret_scale_bps": reloaded.regret_scale_bps,
                    "training_regret_multiplier_mean": (
                        reloaded.training_regret_multiplier_mean
                    ),
                    "early_stop_regret_multiplier_mean": (
                        reloaded.early_stop_regret_multiplier_mean
                    ),
                    "multiclass_calibration_temperature": (
                        reloaded.calibration_temperature
                    ),
                    "multiclass_calibration_abstain_logit_bias": (
                        reloaded.calibration_abstain_logit_bias
                    ),
                    "multiclass_calibration_iterations": (
                        reloaded.calibration_iterations
                    ),
                    "multiclass_calibration_gradient_norm": (
                        reloaded.calibration_gradient_norm
                    ),
                    "multiclass_calibration_log_loss": (
                        reloaded.calibration_multiclass_log_loss
                    ),
                    "multiclass_class_prior_log_loss": (
                        reloaded.calibration_class_prior_log_loss
                    ),
                    "side_profit_probability_calibration": list(
                        reloaded.side_profit_probability_calibration
                    ),
                    "best_iterations": dict(reloaded.best_iterations),
                    "artifact_reload_max_abs_prediction_error": reload_error,
                    **symmetry,
                    "trading_authority": False,
                    "execution_claim": False,
                    "profitability_claim": False,
                    "portfolio_claim": False,
                    "leverage_applied": False,
                }
            )
            progress(
                "model-train-complete",
                member=member,
                seed=seed,
                model_sha256=reloaded.model_sha256,
                class_swap_error=symmetry[
                    "class_probability_mirror_swap_max_abs_error"
                ],
                action_swap_error=symmetry["action_swap_equivariance_max_abs_error"],
                probability_sum_error=symmetry["probability_sum_max_abs_error"],
            )

        calibration_prediction = _predict_ensemble(
            models,
            training.dataset,
            role_indexes["calibration"],
            role="calibration",
            heartbeat_seconds=heartbeat_seconds,
            progress=progress,
        )
        architecture_gates = acceptance["calibration_architecture"]
        assert isinstance(architecture_gates, Mapping)
        architecture_diagnostics = _calibration_architecture_diagnostics(
            training.targets,
            calibration_prediction,
        )
        architecture_reasons = _architecture_gate_reasons(
            architecture_diagnostics,
            architecture_gates,
        )
        architecture_passed = not architecture_reasons
        progress(
            "calibration-architecture-complete",
            status="pass" if architecture_passed else "rejected",
            opportunity_auc=architecture_diagnostics["opportunity_auc"],
            conditional_direction_auc=architecture_diagnostics[
                "conditional_direction_auc"
            ],
            side_profit_auc=architecture_diagnostics["side_profit_auc"],
            multiclass_loss_ratio=architecture_diagnostics[
                "multiclass_log_loss_to_class_prior_ratio"
            ],
            rejection_reasons=",".join(architecture_reasons) or "none",
        )

        selection = design["selection"]
        direction_floors = design["conditional_direction_confidence"]
        assert isinstance(selection, Mapping)
        assert isinstance(direction_floors, Mapping)
        calibration_role = roles["calibration"]
        assert isinstance(calibration_role, Mapping)
        calibration_policy_input = as_selective_action_ensemble(calibration_prediction)
        profile_results: dict[str, dict[str, object]] = {}
        calibration_survivors: list[str] = []
        for raw in profiles:
            profile = str(raw["profile"])
            policy_spec = selective_runner._policy_spec(raw, direction_floors)
            score = derive_selective_action_scores(
                calibration_policy_input,
                policy_spec,
            )
            threshold: dict[str, object] | None = None
            rejection_reasons = list(architecture_reasons)
            accepted = False
            if architecture_passed:
                selected_threshold = select_barrier_threshold(
                    training.dataset,
                    training.targets,
                    score,
                    quantiles=tuple(
                        float(value) for value in selection["threshold_quantiles"]
                    ),
                    expected_days=shared_runner._expected_days(calibration_role),
                    gates=raw["calibration_gates"],
                    drawdown_penalty=float(selection["drawdown_penalty"]),
                )
                threshold = selected_threshold.asdict()
                rejection_reasons = list(selected_threshold.rejection_reasons)
                accepted = selected_threshold.accepted
                if accepted:
                    calibration_survivors.append(profile)
            profile_results[profile] = {
                "profile": profile,
                "policy_spec": asdict(policy_spec),
                "calibration": {
                    "evaluated": True,
                    "architecture_gate_passed": architecture_passed,
                    "eligible_rows": int(np.sum(score.eligible)),
                    "threshold_selection": threshold,
                    "threshold_withheld_reason": (
                        None
                        if architecture_passed
                        else "calibration_architecture_rejected"
                    ),
                    "status": "research_candidate" if accepted else "rejected",
                    "rejection_reasons": rejection_reasons,
                },
                "policy": {"evaluated": False, "withheld_reason": None},
                "development": {"evaluated": False, "withheld_reason": None},
                "distant_confirmation": {
                    "evaluated": False,
                    "withheld_reason": None,
                },
                "final_status": "pending",
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
            progress(
                "profile-calibration-complete",
                profile=profile,
                accepted=accepted,
                eligible_rows=int(np.sum(score.eligible)),
                threshold_evaluated=threshold is not None,
            )

        predictions: dict[str, ThreeActionEnsembleBatch] = {
            "calibration": calibration_prediction
        }
        profiles_by_name = {str(raw["profile"]): raw for raw in profiles}
        selective_runner._mark_withheld(
            profile_results,
            stage="policy",
            survivors=calibration_survivors,
            prior_survivors=calibration_survivors,
            rejected_reason="calibration_rejected",
            earlier_reason="calibration_rejected",
        )
        policy_survivors: list[str] = []
        if calibration_survivors:
            policy_prediction = _predict_ensemble(
                models,
                training.dataset,
                role_indexes["policy"],
                role="policy",
                heartbeat_seconds=heartbeat_seconds,
                progress=progress,
            )
            predictions["policy"] = policy_prediction
            policy_role = roles["policy"]
            assert isinstance(policy_role, Mapping)
            policy_survivors = _evaluate_frozen_stage(
                stage="policy",
                survivors=calibration_survivors,
                prediction=policy_prediction,
                dataset=training.dataset,
                targets=training.targets,
                expected_days=shared_runner._expected_days(policy_role),
                profiles_by_name=profiles_by_name,
                profile_results=profile_results,
                direction_floors=direction_floors,
                progress=progress,
            )
        selective_runner._mark_withheld(
            profile_results,
            stage="development",
            survivors=policy_survivors,
            prior_survivors=calibration_survivors,
            rejected_reason="policy_rejected",
            earlier_reason="calibration_rejected",
        )
        development_survivors: list[str] = []
        if policy_survivors:
            development_prediction = _predict_ensemble(
                models,
                training.dataset,
                role_indexes["development"],
                role="development",
                heartbeat_seconds=heartbeat_seconds,
                progress=progress,
            )
            predictions["development"] = development_prediction
            assert isinstance(development_role, Mapping)
            development_survivors = _evaluate_frozen_stage(
                stage="development",
                survivors=policy_survivors,
                prediction=development_prediction,
                dataset=training.dataset,
                targets=training.targets,
                expected_days=shared_runner._expected_days(development_role),
                profiles_by_name=profiles_by_name,
                profile_results=profile_results,
                direction_floors=direction_floors,
                progress=progress,
            )
        selective_runner._mark_withheld(
            profile_results,
            stage="distant_confirmation",
            survivors=development_survivors,
            prior_survivors=policy_survivors,
            rejected_reason="development_rejected",
            earlier_reason=(
                "policy_or_calibration_rejected"
                if policy_survivors
                else "calibration_rejected"
            ),
        )

        distant_bundle: shared_runner.CorpusBundle | None = None
        distant_role_evidence: dict[str, object] = {}
        distant_diagnostics: dict[str, object] | None = None
        distant_raw_forecast: dict[str, object] | None = None
        forecast_reasons: list[str] = ["withheld_no_development_survivors"]
        distant_trace_survivors: list[str] = []
        if development_survivors:
            distant_role = roles["distant_confirmation"]
            assert isinstance(distant_role, Mapping)
            distant_bundle = shared_runner._load_corpus(
                name="distant_confirmation",
                design=design,
                warehouse_path=warehouse_path,
                cache_root=cache_root,
                first=shared_runner._parse_date(
                    distant_role["context_start"],
                    label="distant context",
                ),
                last=shared_runner._parse_date(
                    distant_role["end"],
                    label="distant end",
                ),
                evaluation_first=shared_runner._parse_date(
                    distant_role["start"],
                    label="distant start",
                ),
                evaluation_last=shared_runner._parse_date(
                    distant_role["end"],
                    label="distant end",
                ),
                memory_limit=memory_limit,
                threads=threads,
                heartbeat_seconds=heartbeat_seconds,
                progress=progress,
            )
            distant_roles = {
                "distant_confirmation": {
                    "start": distant_role["start"],
                    "end": distant_role["end"],
                }
            }
            distant_indexes, distant_role_evidence, _distant_exit = (
                shared_runner._role_indexes(
                    distant_bundle,
                    distant_roles,
                    ("distant_confirmation",),
                )
            )
            distant_prediction = _predict_ensemble(
                models,
                distant_bundle.dataset,
                distant_indexes["distant_confirmation"],
                role="distant_confirmation",
                heartbeat_seconds=heartbeat_seconds,
                progress=progress,
            )
            distant_policy_input = as_selective_action_ensemble(distant_prediction)
            distant_diagnostics = selective_runner._selected_action_diagnostics(
                distant_bundle.targets,
                distant_policy_input,
            )
            distant_raw_forecast = shared_runner._raw_forecast_diagnostics(
                distant_bundle.targets,
                distant_policy_input,
                scenario="stress",
            )
            forecast_gates = acceptance["distant_confirmation_forecast"]
            assert isinstance(forecast_gates, Mapping)
            forecast_reasons = shared_runner._forecast_gate_reasons(
                distant_diagnostics,
                forecast_gates,
            )
            distant_trace_survivors = _evaluate_frozen_stage(
                stage="distant_confirmation",
                survivors=development_survivors,
                prediction=distant_prediction,
                dataset=distant_bundle.dataset,
                targets=distant_bundle.targets,
                expected_days=shared_runner._expected_days(
                    distant_roles["distant_confirmation"]
                ),
                profiles_by_name=profiles_by_name,
                profile_results=profile_results,
                direction_floors=direction_floors,
                progress=progress,
            )

        final_profiles = distant_trace_survivors if not forecast_reasons else []
        for profile in _PROFILE_NAMES:
            profile_results[profile]["final_status"] = (
                "consumed_data_viability_candidate"
                if profile in final_profiles
                else "rejected"
            )
        report_status = "research_candidate" if final_profiles else "rejected"
        forecast_by_role = {}
        for role, prediction in predictions.items():
            policy_input = as_selective_action_ensemble(prediction)
            forecast_by_role[role] = {
                "stress": shared_runner._raw_forecast_diagnostics(
                    training.targets,
                    policy_input,
                    scenario="stress",
                ),
                "selected_action": selective_runner._selected_action_diagnostics(
                    training.targets,
                    policy_input,
                ),
            }
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_class": (
                "consumed_data_utility_weighted_symmetric_three_action_viability_evidence"
            ),
            "round": _ROUND,
            "status": report_status,
            "critical_verdict": (
                "consumed_data_viability_only_new_untouched_test_required"
                if final_profiles
                else "rejected_no_predictive_or_economic_viability_claim"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": binding["implementation"]["commit"],
            "report_canonical_sha256": "PENDING",
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
            "all_target_dates_previously_consumed": True,
            "untouched_policy_windows_accessed": False,
            "runtime_resources": runtime,
            "schema_versions": {
                "model": THREE_ACTION_LIGHTGBM_SCHEMA_VERSION,
                "policy": SELECTIVE_ACTION_POLICY_SCHEMA_VERSION,
                "base_policy": ACTION_POLICY_SCHEMA_VERSION,
                "barriers": ADAPTIVE_BARRIER_SCHEMA_VERSION,
            },
            "probability_semantics": {
                "action_class_probability": (
                    "probability_that_long_abstain_or_short_is_the_best_after_cost_action"
                ),
                "side_profit_probability": (
                    "independent_probability_that_the_named_side_has_positive_after_cost_return"
                ),
                "semantic_aliasing_permitted": False,
            },
            "training_corpus": shared_runner._corpus_report(
                training,
                role_evidence,
            ),
            "distant_corpus": (
                shared_runner._corpus_report(
                    distant_bundle,
                    distant_role_evidence,
                )
                if distant_bundle is not None
                else None
            ),
            "stage_access": {
                "calibration_prediction": True,
                "calibration_threshold_selection": architecture_passed,
                "policy_prediction": "policy" in predictions,
                "development_prediction": "development" in predictions,
                "distant_confirmation_prediction": distant_bundle is not None,
                "later_stage_predictions_withheld_until_prior_stage_passes": True,
            },
            "sample_weighting": {
                "base_method": "average_label_uniqueness",
                "multiclass_modifier": "bounded_decision_regret_span",
                "training_rows": len(train_weights),
                "training_minimum": float(np.min(train_weights)),
                "training_maximum": float(np.max(train_weights)),
                "training_mean": float(np.mean(train_weights)),
                "tuning_rows": len(tuning_weights),
                "tuning_minimum": float(np.min(tuning_weights)),
                "tuning_maximum": float(np.max(tuning_weights)),
                "tuning_mean": float(np.mean(tuning_weights)),
            },
            "ensemble_models": model_evidence,
            "implementation_gates": {
                "maximum_observed_class_probability_mirror_swap_error": max(
                    float(value["class_probability_mirror_swap_max_abs_error"])
                    for value in model_evidence
                ),
                "maximum_permitted_class_probability_mirror_swap_error": (
                    symmetry_limits["class_probability_mirror_swap_max_abs_error"]
                ),
                "maximum_observed_action_swap_equivariance_error": max(
                    float(value["action_swap_equivariance_max_abs_error"])
                    for value in model_evidence
                ),
                "maximum_permitted_action_swap_equivariance_error": (
                    symmetry_limits["action_swap_equivariance_max_abs_error"]
                ),
                "maximum_observed_probability_sum_error": max(
                    float(value["probability_sum_max_abs_error"])
                    for value in model_evidence
                ),
                "maximum_permitted_probability_sum_error": symmetry_limits[
                    "probability_sum_max_abs_error"
                ],
                "artifact_reload_equivalence_passed": all(
                    value["artifact_reload_max_abs_prediction_error"] == 0.0
                    for value in model_evidence
                ),
                "source_certificate_counts": {
                    "training_and_near_evaluation": 1,
                    "distant_confirmation": 1 if distant_bundle is not None else 0,
                },
                "nonfinite_prediction_count": 0,
                "maximum_progress_silence_seconds": heartbeat_seconds,
                "status": "pass",
            },
            "calibration_architecture": {
                "evaluated": True,
                "status": "pass" if architecture_passed else "rejected",
                "diagnostics": architecture_diagnostics,
                "gates": architecture_gates,
                "rejection_reasons": architecture_reasons,
            },
            "forecast_diagnostics_by_role": forecast_by_role,
            "distant_confirmation_forecast": {
                "evaluated": distant_diagnostics is not None,
                "selected_action": distant_diagnostics,
                "raw_stress": distant_raw_forecast,
                "status": (
                    "pass"
                    if distant_diagnostics is not None and not forecast_reasons
                    else "rejected"
                ),
                "rejection_reasons": forecast_reasons,
                "gates": acceptance["distant_confirmation_forecast"],
            },
            "profile_results": [profile_results[profile] for profile in _PROFILE_NAMES],
            "stage_survivors": {
                "calibration_architecture": architecture_passed,
                "calibration": calibration_survivors,
                "policy": policy_survivors,
                "development": development_survivors,
                "distant_trace": distant_trace_survivors,
                "final": final_profiles,
            },
            "limitations": [
                *design["limitations"],
                "All target dates in this study were previously consumed; even a pass cannot validate an untouched edge.",
                "LightGBM training used OpenCL FP64 accumulation; LightGBM prediction used its CPU prediction path.",
                "The study evaluates one BTCUSDT research candidate and grants no live, testnet, portfolio, leverage, or execution authority.",
            ],
        }
        report_for_hash = dict(report)
        report_for_hash.pop("report_canonical_sha256")
        report["report_canonical_sha256"] = _canonical_sha256(report_for_hash)
        write_json_atomic(
            destination / "report.json",
            report,
            indent=2,
            sort_keys=True,
        )
        progress(
            "complete",
            status=report_status,
            report_canonical_sha256=report["report_canonical_sha256"],
            final_profiles=",".join(final_profiles) if final_profiles else "none",
        )
        return report
    except BaseException as exc:
        progress(
            "failed",
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Round 34 utility-weighted three-action study."
    )
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        default=str(research / "round-034-three-action-utility-design.json"),
    )
    parser.add_argument(
        "--binding",
        default=str(research / "round-034-execution-binding.json"),
    )
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_three_action_viability(
        design_path=arguments.design,
        binding_path=arguments.binding,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the hash-bound Round 35 consumed-data direction screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
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
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    _auc,
    average_label_uniqueness,
)
from simple_ai_trading.microstructure_direction_screen import (  # noqa: E402
    DirectionScreenPrediction,
    DirectionScreenSpec,
    TrainedDirectionScreenModel,
    load_direction_screen_model,
    predict_direction_screen_model,
    save_direction_screen_model,
    train_direction_screen_model,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
)
from simple_ai_trading.microstructure_three_action_lightgbm import (  # noqa: E402
    load_three_action_lightgbm_model,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools import run_shared_action_viability as shared_runner  # noqa: E402
from tools.publish_three_action_viability import _validated_source  # noqa: E402
from tools.run_three_action_viability import (  # noqa: E402
    _predict_ensemble,
    load_round34_design,
)


DESIGN_SCHEMA_VERSION = "consumed-direction-screen-design-v1"
BINDING_SCHEMA_VERSION = "round-035-direction-screen-execution-binding-v1"
REPORT_SCHEMA_VERSION = "consumed-direction-screen-report-v1"
_ROUND = 35
_EVALUATED_ROLES = ("train", "early_stop", "calibration")
_REQUIRED_BOUND_PATHS = frozenset(
    {
        "docs/model-research/action-value/consumed-periods-through-round-033.json",
        "docs/model-research/action-value/consumed-periods-through-round-034.json",
        "docs/model-research/action-value/round-031-frozen-chronological-confirmation-design.json",
        "docs/model-research/action-value/round-033-failure-analysis.json",
        "docs/model-research/action-value/round-033-selective-action-design.json",
        "docs/model-research/action-value/round-034-execution-binding.json",
        "docs/model-research/action-value/round-034-failure-analysis.json",
        "docs/model-research/action-value/round-034-three-action-utility-design.json",
        "docs/model-research/action-value/round-035-consumed-direction-screen-design.json",
        "src/simple_ai_trading/assets.py",
        "src/simple_ai_trading/compute.py",
        "src/simple_ai_trading/lightgbm_backend.py",
        "src/simple_ai_trading/microstructure_action_architecture.py",
        "src/simple_ai_trading/microstructure_action_features.py",
        "src/simple_ai_trading/microstructure_action_policy.py",
        "src/simple_ai_trading/microstructure_architecture.py",
        "src/simple_ai_trading/microstructure_barriers.py",
        "src/simple_ai_trading/microstructure_cache.py",
        "src/simple_ai_trading/microstructure_direction_screen.py",
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
        "tools/publish_three_action_viability.py",
        "tools/run_consumed_direction_screen.py",
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
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root is invalid")
    return payload


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 35 Git binding command failed") from exc


def load_direction_screen_design(path: str | Path) -> tuple[dict[str, object], str]:
    """Validate the complete frozen screen design and local predecessors."""

    design_path = Path(path).resolve()
    design = _read_object(design_path, label="Round 35 direction-screen design")
    canonical = dict(design)
    claimed = canonical.pop("design_sha256", None)
    if (
        not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or design.get("schema_version") != DESIGN_SCHEMA_VERSION
        or design.get("round") != _ROUND
        or design.get("phase") != "pre_architecture_consumed_data_screen"
        or design.get("design_revision") != 2
    ):
        raise ValueError("Round 35 direction-screen design hash or identity is invalid")
    predecessor = design.get("predecessor")
    governance = design.get("governance")
    source = design.get("source_contract")
    roles = design.get("data_roles")
    target = design.get("direction_target")
    mirror = design.get("mirror_equivariance")
    feature_sets = design.get("feature_sets")
    variants = design.get("variants")
    model = design.get("model")
    evaluation = design.get("evaluation")
    gates = design.get("architecture_freeze_eligibility")
    resources = design.get("runtime_resources")
    claims = design.get("claims")
    if any(
        not isinstance(value, Mapping)
        for value in (
            predecessor,
            governance,
            source,
            roles,
            target,
            mirror,
            feature_sets,
            model,
            evaluation,
            gates,
            resources,
            claims,
        )
    ) or not isinstance(variants, list):
        raise ValueError("Round 35 direction-screen design sections are incomplete")
    research_root = design_path.parent
    failure_path = research_root / str(predecessor["failure_analysis"])
    registry_path = research_root / str(governance["consumed_period_registry"])
    loader_path = research_root / str(source["loader_design"])
    failure = _read_object(failure_path, label="Round 34 failure analysis")
    registry = _read_object(registry_path, label="Round 34 consumed registry")
    loader, loader_sha, _profiles = load_round34_design(loader_path)
    if (
        failure.get("analysis_sha256")
        != predecessor.get("failure_analysis_canonical_sha256")
        or _sha256_file(failure_path) != predecessor.get("failure_analysis_file_sha256")
        or registry.get("registry_sha256")
        != governance.get("consumed_period_registry_canonical_sha256")
        or _sha256_file(registry_path)
        != governance.get("consumed_period_registry_file_sha256")
        or loader_sha != source.get("loader_design_canonical_sha256")
        or _sha256_file(loader_path) != source.get("loader_design_file_sha256")
        or loader.get("round") != 34
    ):
        raise ValueError("Round 35 predecessor or loader evidence drifted")
    feature_contract = {
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": MICROSTRUCTURE_FEATURE_NAMES,
    }
    if (
        source.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION
        or source.get("feature_count") != len(MICROSTRUCTURE_FEATURE_NAMES)
        or source.get("feature_contract_sha256") != _canonical_sha256(feature_contract)
        or source.get("corpus_certificate_sha256")
        != "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
        or source.get("barrier_targets_sha256")
        != "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
        or source.get("cache_key")
        != "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
    ):
        raise ValueError("Round 35 source feature or corpus identity drifted")
    full = feature_sets.get("full")
    noncycle = feature_sets.get("full_without_deterministic_cycles")
    compact = feature_sets.get("compact_observed_microstructure")
    if any(not isinstance(value, Mapping) for value in (full, noncycle, compact)):
        raise ValueError("Round 35 feature sets are invalid")
    excluded = tuple(noncycle.get("excluded_features", ()))
    included = tuple(compact.get("included_features", ()))
    if (
        full.get("expected_feature_count") != len(MICROSTRUCTURE_FEATURE_NAMES)
        or noncycle.get("expected_feature_count")
        != len(MICROSTRUCTURE_FEATURE_NAMES) - len(excluded)
        or compact.get("expected_feature_count") != len(included)
        or len(excluded) != len(set(excluded))
        or len(excluded) != 7
        or len(included) != len(set(included))
        or len(included) != 68
        or not set(excluded) <= set(MICROSTRUCTURE_FEATURE_NAMES)
        or not set(included) <= set(MICROSTRUCTURE_FEATURE_NAMES)
        or set(excluded) & set(included)
    ):
        raise ValueError("Round 35 feature-set membership is invalid")
    expected_variants = {
        (feature_set, weighting)
        for feature_set in (
            "full",
            "full_without_deterministic_cycles",
            "compact_observed_microstructure",
        )
        for weighting in ("uniqueness", "utility_margin")
    }
    observed_variants = {
        (str(item.get("feature_set")), str(item.get("weighting")))
        for item in variants
        if isinstance(item, Mapping)
    }
    variant_names = [
        str(item.get("variant")) for item in variants if isinstance(item, Mapping)
    ]
    if (
        len(variants) != 6
        or len(variant_names) != len(set(variant_names))
        or observed_variants != expected_variants
        or governance.get("variant_budget") != 6
        or governance.get("seed_budget") != 1
        or governance.get("post_hoc_discovery_only") is not True
        or governance.get("promotion_permitted") is not False
        or governance.get("hyperparameter_search_permitted") is not False
        or governance.get("risk_gate_relaxation_permitted") is not False
        or governance.get("leverage_permitted") is not False
    ):
        raise ValueError("Round 35 variant or governance contract drifted")
    if (
        model.get("family") != "mirror_equivariant_binary_side_superiority_lightgbm"
        or model.get("seed") != 29
        or model.get("lightgbm_training_backend_required") != "opencl"
        or model.get("gpu_use_dp_required") is not True
        or model.get("cpu_fallback_permitted") is not False
        or mirror.get("separate_long_and_short_models_permitted") is not False
        or target.get("future_outcomes_available_to_model_features_or_runtime")
        is not False
        or evaluation.get("primary_role") != "calibration"
        or gates.get("promotion_or_trading_authority_created") is not False
        or any(bool(value) for value in claims.values())
    ):
        raise ValueError("Round 35 model, evaluation, or claims contract drifted")
    for name in ("policy", "development", "distant_confirmation"):
        role = roles.get(name)
        if (
            not isinstance(role, Mapping)
            or role.get("prediction_or_metric_access_permitted") is not False
        ):
            raise ValueError("Round 35 later-stage access contract drifted")
    research = design.get("research_basis")
    limitations = design.get("limitations")
    if (
        not isinstance(research, list)
        or len(research) < 6
        or any(
            not isinstance(item, Mapping)
            or not str(item.get("url") or "").startswith("https://")
            or not str(item.get("review_status") or "").strip()
            for item in research
        )
        or not isinstance(limitations, list)
        or len(limitations) < 5
    ):
        raise ValueError("Round 35 research or limitations are incomplete")
    return design, str(claimed)


def load_direction_screen_binding(
    path: str | Path,
    *,
    design_path: str | Path,
    design_sha256: str,
) -> tuple[dict[str, object], str]:
    """Verify the bound commit, every critical blob, and a clean worktree."""

    binding = _read_object(Path(path), label="Round 35 execution binding")
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
        raise ValueError("Round 35 execution binding is invalid")
    commit = str(implementation.get("commit") or "").lower()
    files = implementation.get("files")
    if (
        not _is_git_oid(commit)
        or implementation.get("hash_mode") != "git_blob_sha256_v1"
        or not isinstance(files, list)
    ):
        raise ValueError("Round 35 implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    bound: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("Round 35 bound file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in bound
        ):
            raise ValueError("Round 35 bound path is unsafe or duplicated")
        bound[normalized] = str(item["sha256"])
    if set(bound) != _REQUIRED_BOUND_PATHS:
        missing = sorted(_REQUIRED_BOUND_PATHS - set(bound))
        extra = sorted(set(bound) - _REQUIRED_BOUND_PATHS)
        raise ValueError(
            f"Round 35 bound scope changed: missing={missing} extra={extra}"
        )
    for normalized, expected in bound.items():
        historical = _git_bytes("show", f"{commit}:{normalized}")
        current = _git_bytes("show", f"HEAD:{normalized}")
        if (
            hashlib.sha256(historical).hexdigest() != expected
            or hashlib.sha256(current).hexdigest() != expected
        ):
            raise ValueError(f"Round 35 implementation changed: {normalized}")
    if _git_bytes("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Round 35 execution requires a clean worktree")
    relative_design = Path(design_path).resolve().relative_to(ROOT).as_posix()
    if (
        design.get("path") != relative_design
        or design.get("design_sha256") != design_sha256
        or design.get("file_sha256") != bound.get(relative_design)
    ):
        raise ValueError("Round 35 bound design identity changed")
    return binding, str(claimed)


def _feature_names(design: Mapping[str, object], feature_set: str) -> tuple[str, ...]:
    feature_sets = design["feature_sets"]
    assert isinstance(feature_sets, Mapping)
    contract = feature_sets[feature_set]
    assert isinstance(contract, Mapping)
    selector = contract["selector"]
    if selector == "all_source_features":
        names = MICROSTRUCTURE_FEATURE_NAMES
    elif selector == "all_source_features_except_exact_exclusions":
        excluded = set(contract["excluded_features"])
        names = tuple(
            name for name in MICROSTRUCTURE_FEATURE_NAMES if name not in excluded
        )
    elif selector == "exact_inclusion_list":
        names = tuple(str(value) for value in contract["included_features"])
    else:
        raise ValueError("Round 35 feature selector is unsupported")
    if len(names) != int(contract["expected_feature_count"]):
        raise ValueError("Round 35 selected feature count drifted")
    return names


def _model_spec(design: Mapping[str, object]) -> DirectionScreenSpec:
    model = design["model"]
    assert isinstance(model, Mapping)
    parameters = model["parameters"]
    assert isinstance(parameters, Mapping)
    return DirectionScreenSpec(
        **dict(parameters),
        gpu_use_dp_required=bool(model["gpu_use_dp_required"]),
    )


def _prediction_difference(
    left: DirectionScreenPrediction,
    right: DirectionScreenPrediction,
) -> float:
    if not np.array_equal(left.endpoint_indexes, right.endpoint_indexes):
        return math.inf
    values = []
    for name in (
        "long_superiority_probability",
        "short_superiority_probability",
        "conditional_long_probability",
        "direction_score",
    ):
        values.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(getattr(left, name), dtype=np.float64)
                        - np.asarray(getattr(right, name), dtype=np.float64)
                    )
                )
            )
        )
    if not np.array_equal(left.selected_side, right.selected_side):
        return math.inf
    return max(values)


def _mirror_swap_error(
    original: DirectionScreenPrediction,
    mirrored: DirectionScreenPrediction,
) -> float:
    if not np.array_equal(original.endpoint_indexes, mirrored.endpoint_indexes):
        return math.inf
    errors = (
        np.max(
            np.abs(
                original.long_superiority_probability
                - mirrored.short_superiority_probability
            )
        ),
        np.max(
            np.abs(
                original.short_superiority_probability
                - mirrored.long_superiority_probability
            )
        ),
        np.max(
            np.abs(
                original.conditional_long_probability
                + mirrored.conditional_long_probability
                - 1.0
            )
        ),
        np.max(np.abs(original.direction_score + mirrored.direction_score)),
    )
    if not np.array_equal(original.selected_side, -mirrored.selected_side):
        return math.inf
    return float(max(errors))


def _top_rows(
    *,
    selected_actual: np.ndarray,
    selected_side: np.ndarray,
    true_long: np.ndarray,
    ranking_score: np.ndarray,
) -> dict[str, dict[str, object]]:
    actual = np.asarray(selected_actual, dtype=np.float64)
    side = np.asarray(selected_side, dtype=np.int8)
    truth = np.asarray(true_long, dtype=bool)
    ranking = np.asarray(ranking_score, dtype=np.float64)
    if (
        any(value.shape != actual.shape for value in (side, truth, ranking))
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(ranking))
    ):
        raise ValueError("Round 35 ranked-tail inputs are invalid")
    routed = side != 0
    indexes = np.flatnonzero(routed)
    if len(indexes) < 1_000:
        raise ValueError("Round 35 ranked-tail support is insufficient")
    order = indexes[np.argsort(-ranking[indexes], kind="stable")]
    output: dict[str, dict[str, object]] = {}
    for requested in (100, 500, 1_000):
        chosen = order[:requested]
        predicted_long = side[chosen] == 1
        output[str(requested)] = {
            "requested_rows": requested,
            "actual_rows": len(chosen),
            "mean_stress_net_bps": float(np.mean(actual[chosen])),
            "median_stress_net_bps": float(np.median(actual[chosen])),
            "positive_ratio": float(np.mean(actual[chosen] > 0.0)),
            "long_share": float(np.mean(predicted_long)),
            "direction_accuracy": float(np.mean(predicted_long == truth[chosen])),
        }
    return output


def _daily_metrics(
    *,
    decision_time_ms: np.ndarray,
    true_long: np.ndarray,
    conditional_long_probability: np.ndarray,
    selected_side: np.ndarray,
) -> list[dict[str, object]]:
    times = np.asarray(decision_time_ms, dtype=np.int64)
    labels = np.asarray(true_long, dtype=np.int8)
    probability = np.asarray(conditional_long_probability, dtype=np.float64)
    side = np.asarray(selected_side, dtype=np.int8)
    if any(value.shape != times.shape for value in (labels, probability, side)):
        raise ValueError("Round 35 daily metric inputs differ")
    days = times // 86_400_000
    output: list[dict[str, object]] = []
    for day in np.unique(days):
        selected = days == day
        if (
            min(int(np.sum(labels[selected] == 0)), int(np.sum(labels[selected] == 1)))
            == 0
        ):
            raise ValueError("Round 35 daily metric lacks both directions")
        day_text = (
            datetime.fromtimestamp(
                int(day) * 86_400,
                tz=timezone.utc,
            )
            .date()
            .isoformat()
        )
        routed = selected & (side != 0)
        output.append(
            {
                "date": day_text,
                "rows": int(np.sum(selected)),
                "routed_rows": int(np.sum(routed)),
                "direction_auc": _auc(labels[selected], probability[selected]),
                "brier_score": float(
                    np.mean((probability[selected] - labels[selected]) ** 2)
                ),
                "direction_accuracy": float(
                    np.mean((side[routed] == 1) == labels[routed])
                ),
            }
        )
    return output


def _variant_metrics(
    *,
    prediction: DirectionScreenPrediction,
    long_actual: np.ndarray,
    short_actual: np.ndarray,
    decision_time_ms: np.ndarray,
    frozen_opportunity_probability: np.ndarray,
) -> dict[str, object]:
    long_values = np.asarray(long_actual, dtype=np.float64)
    short_values = np.asarray(short_actual, dtype=np.float64)
    true_long = long_values > short_values
    side = np.asarray(prediction.selected_side, dtype=np.int8)
    routed = side != 0
    if (
        len(long_values) < 1_000
        or any(
            value.shape != long_values.shape
            for value in (
                short_values,
                true_long,
                side,
                frozen_opportunity_probability,
            )
        )
        or np.sum(routed) < 1_000
    ):
        raise ValueError("Round 35 variant metric support is invalid")
    selected_actual = np.where(side == 1, long_values, short_values)
    probability = np.asarray(prediction.conditional_long_probability, dtype=np.float64)
    daily = _daily_metrics(
        decision_time_ms=decision_time_ms,
        true_long=true_long,
        conditional_long_probability=probability,
        selected_side=side,
    )
    daily_auc = np.asarray([item["direction_auc"] for item in daily], dtype=np.float64)
    return {
        "rows": len(long_values),
        "routed_rows": int(np.sum(routed)),
        "score_tie_rows": int(np.sum(~routed)),
        "pooled_direction_auc": _auc(true_long.astype(np.int8), probability),
        "conditional_long_probability_brier_score": float(
            np.mean((probability - true_long) ** 2)
        ),
        "direction_accuracy": float(np.mean((side[routed] == 1) == true_long[routed])),
        "all_routed_mean_stress_net_bps": float(np.mean(selected_actual[routed])),
        "daily": daily,
        "daily_auc_minimum": float(np.min(daily_auc)),
        "daily_auc_median": float(np.median(daily_auc)),
        "daily_auc_standard_deviation": float(np.std(daily_auc)),
        "days_above_chance": int(np.sum(daily_auc > 0.5)),
        "frozen_opportunity_ranked": _top_rows(
            selected_actual=selected_actual,
            selected_side=side,
            true_long=true_long,
            ranking_score=np.asarray(frozen_opportunity_probability, dtype=np.float64),
        ),
        "candidate_confidence_ranked": _top_rows(
            selected_actual=selected_actual,
            selected_side=side,
            true_long=true_long,
            ranking_score=np.abs(probability - 0.5),
        ),
    }


def _eligibility_reasons(
    metrics: Mapping[str, object],
    gates: Mapping[str, object],
    *,
    nonfinite_predictions: int,
) -> list[str]:
    frozen = metrics["frozen_opportunity_ranked"]
    confidence = metrics["candidate_confidence_ranked"]
    assert isinstance(frozen, Mapping)
    assert isinstance(confidence, Mapping)
    reasons: list[str] = []
    checks = (
        (
            float(metrics["pooled_direction_auc"])
            < float(gates["minimum_pooled_direction_auc"]),
            "pooled_direction_auc_gate_failed",
        ),
        (
            float(metrics["daily_auc_minimum"])
            < float(gates["minimum_daily_direction_auc"]),
            "minimum_daily_direction_auc_gate_failed",
        ),
        (
            float(metrics["daily_auc_median"])
            < float(gates["minimum_median_daily_direction_auc"]),
            "median_daily_direction_auc_gate_failed",
        ),
        (
            int(metrics["days_above_chance"]) < int(gates["minimum_days_above_chance"]),
            "days_above_chance_gate_failed",
        ),
        (
            float(frozen["100"]["mean_stress_net_bps"])
            <= float(gates["minimum_frozen_opportunity_top_100_mean_stress_net_bps"]),
            "frozen_opportunity_top_100_stress_net_gate_failed",
        ),
        (
            float(frozen["500"]["mean_stress_net_bps"])
            <= float(gates["minimum_frozen_opportunity_top_500_mean_stress_net_bps"]),
            "frozen_opportunity_top_500_stress_net_gate_failed",
        ),
        (
            float(confidence["500"]["mean_stress_net_bps"])
            <= float(gates["minimum_candidate_confidence_top_500_mean_stress_net_bps"]),
            "candidate_confidence_top_500_stress_net_gate_failed",
        ),
        (
            nonfinite_predictions > int(gates["maximum_nonfinite_predictions"]),
            "nonfinite_prediction_gate_failed",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def _select_candidate(eligible: Sequence[Mapping[str, object]]) -> str | None:
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda item: (
            -float(item["metrics"]["daily_auc_minimum"]),
            -float(
                item["metrics"]["frozen_opportunity_ranked"]["500"][
                    "mean_stress_net_bps"
                ]
            ),
            int(item["feature_count"]),
            str(item["variant"]),
        ),
    )
    return str(ordered[0]["variant"])


def _feature_gain(
    model: TrainedDirectionScreenModel,
) -> list[dict[str, object]]:
    booster = lgb.Booster(model_str=model.model_string)
    gain = np.asarray(
        booster.feature_importance(importance_type="gain"),
        dtype=np.float64,
    )
    if gain.shape != (len(model.selected_feature_names),) or not np.all(
        np.isfinite(gain)
    ):
        raise ValueError("Round 35 feature gain is invalid")
    total = float(np.sum(gain))
    normalized = gain / total if total > 0.0 else np.zeros_like(gain)
    order = np.argsort(-normalized, kind="stable")
    return [
        {
            "rank": position,
            "feature": model.selected_feature_names[index],
            "normalized_gain": float(normalized[index]),
        }
        for position, index in enumerate(order, start=1)
    ]


def run_screen(
    *,
    design_path: Path,
    binding_path: Path,
    round34_evidence_root: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Execute all six bound variants and write one fail-closed report."""

    design, design_sha = load_direction_screen_design(design_path)
    binding, binding_sha = load_direction_screen_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
    )
    source = design["source_contract"]
    resources = design["runtime_resources"]
    model_contract = design["model"]
    roles = design["data_roles"]
    gates = design["architecture_freeze_eligibility"]
    assert isinstance(source, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(model_contract, Mapping)
    assert isinstance(roles, Mapping)
    assert isinstance(gates, Mapping)
    loader_path = design_path.parent / str(source["loader_design"])
    loader_design, _loader_sha, _profiles = load_round34_design(loader_path)
    _source_design, round34_report, round34_report_sha = _validated_source(
        evidence_root=round34_evidence_root,
        design_path=loader_path,
        binding_path=design_path.parent / "round-034-execution-binding.json",
    )
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Round 35 output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    report_path = output_root / "report.json"
    started = time.monotonic()
    lock = threading.Lock()
    sequence = 0

    def progress(phase: str, **details: object) -> None:
        nonlocal sequence
        with lock:
            sequence += 1
            payload = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "round": _ROUND,
                "sequence": sequence,
                "phase": phase,
                "run_elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
            print(
                "round35-direction-screen "
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
        thread_evidence = shared_runner._configure_worker_threads(
            int(resources["maximum_worker_threads"])
        )
        threads = int(thread_evidence["effective_worker_threads"])
        progress("directml-preflight-start")
        directml = shared_runner._attest_directml()
        progress(
            "directml-preflight-complete",
            device=directml["device"],
            vendor=directml["vendor"],
        )
        loader_roles = loader_design["data"]["roles"]
        train_first = shared_runner._parse_date(
            loader_roles["train"]["start"],
            label="Round 35 materialization start",
        )
        development_last = shared_runner._parse_date(
            loader_roles["development"]["end"],
            label="Round 35 materialization end",
        )
        corpus = shared_runner._load_corpus(
            name="round35_consumed_direction_screen",
            design=loader_design,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            first=train_first,
            last=development_last,
            evaluation_first=train_first,
            evaluation_last=development_last,
            memory_limit=f"{int(resources['duckdb_memory_limit_gib'])}GB",
            threads=threads,
            heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
            progress=progress,
        )
        if (
            corpus.source_certificate.get("certificate_sha256")
            != source["corpus_certificate_sha256"]
            or corpus.targets_sha256 != source["barrier_targets_sha256"]
            or corpus.cache_key != source["cache_key"]
            or corpus.dataset.rows != source["dataset_rows"]
            or len(corpus.event_indexes) != source["event_rows"]
            or int(np.sum(corpus.targets.valid)) != source["valid_barrier_rows"]
        ):
            raise ValueError("Round 35 certified corpus identity changed")
        role_indexes, role_evidence, maximum_exit = shared_runner._role_indexes(
            corpus,
            loader_roles,
            _EVALUATED_ROLES,
        )
        train_weights = average_label_uniqueness(
            corpus.dataset.decision_time_ms,
            maximum_exit,
            role_indexes["train"],
        )
        early_weights = average_label_uniqueness(
            corpus.dataset.decision_time_ms,
            maximum_exit,
            role_indexes["early_stop"],
        )
        progress("round34-opportunity-prediction-start")
        round34_models = [
            load_three_action_lightgbm_model(
                round34_evidence_root / str(item["artifact_path"])
            )
            for item in round34_report["ensemble_models"]
        ]
        round34_prediction = _predict_ensemble(
            round34_models,
            corpus.dataset,
            role_indexes["calibration"],
            role="round35_frozen_round34_opportunity",
            heartbeat_seconds=float(resources["heartbeat_interval_seconds"]),
            progress=progress,
        )
        calibration_long, calibration_short = shared_runner._scenario_targets(
            corpus.targets,
            role_indexes["calibration"],
            scenario="stress",
        )
        opportunity = (np.maximum(calibration_long, calibration_short) > 0.0) & (
            calibration_long != calibration_short
        )
        calibration_indexes = role_indexes["calibration"][opportunity]
        calibration_long = calibration_long[opportunity]
        calibration_short = calibration_short[opportunity]
        frozen_opportunity = np.asarray(
            round34_prediction.opportunity_probability_mean[opportunity],
            dtype=np.float64,
        )
        calibration_times = corpus.dataset.decision_time_ms[calibration_indexes]
        if len(calibration_indexes) != 15_222:
            raise ValueError("Round 35 calibration opportunity support changed")
        progress(
            "round34-opportunity-prediction-complete",
            calibration_rows=len(role_indexes["calibration"]),
            opportunity_rows=len(calibration_indexes),
        )
        spec = _model_spec(design)
        variants = design["variants"]
        assert isinstance(variants, list)
        variant_results: list[dict[str, object]] = []
        sample_source_indexes = shared_runner._sample_indexes(calibration_indexes)
        sample_dataset = shared_runner._subset_dataset(
            corpus.dataset,
            sample_source_indexes,
        )
        mirrored_sample_dataset = replace(
            sample_dataset,
            features=mirror_microstructure_direction(sample_dataset.features),
        )
        sample_endpoints = np.arange(sample_dataset.rows, dtype=np.int64)
        for position, raw_variant in enumerate(variants, start=1):
            assert isinstance(raw_variant, Mapping)
            variant = str(raw_variant["variant"])
            feature_set = str(raw_variant["feature_set"])
            weighting = str(raw_variant["weighting"])
            selected_names = _feature_names(design, feature_set)
            progress(
                "variant-training-start",
                variant=variant,
                variant_position=position,
                variant_count=len(variants),
                feature_count=len(selected_names),
                weighting=weighting,
            )
            variant_started = time.monotonic()
            with progress_heartbeat(
                progress,
                phase="variant-training-heartbeat",
                interval_seconds=float(resources["heartbeat_interval_seconds"]),
                details={"variant": variant, "variant_position": position},
            ):
                trained = train_direction_screen_model(
                    corpus.dataset,
                    corpus.targets,
                    train_endpoints=role_indexes["train"],
                    early_stop_endpoints=role_indexes["early_stop"],
                    train_sample_weights=train_weights,
                    early_stop_sample_weights=early_weights,
                    selected_feature_names=selected_names,
                    variant=variant,
                    feature_set=feature_set,
                    weighting=weighting,
                    spec=spec,
                    compute_backend=str(model_contract["compute_backend_requested"]),
                    seed=int(model_contract["seed"]),
                )
            training_seconds = time.monotonic() - variant_started
            if (
                trained.backend_kind != "opencl"
                or trained.backend_requested != "directml"
            ):
                raise RuntimeError("Round 35 model training did not remain on OpenCL")
            artifact_path = model_root / f"{variant}.json"
            save_direction_screen_model(artifact_path, trained)
            reloaded = load_direction_screen_model(artifact_path)
            original_sample = predict_direction_screen_model(
                trained,
                sample_dataset,
                sample_endpoints,
            )
            reloaded_sample = predict_direction_screen_model(
                reloaded,
                sample_dataset,
                sample_endpoints,
            )
            mirrored_sample = predict_direction_screen_model(
                trained,
                mirrored_sample_dataset,
                sample_endpoints,
            )
            reload_error = _prediction_difference(original_sample, reloaded_sample)
            mirror_error = _mirror_swap_error(original_sample, mirrored_sample)
            with progress_heartbeat(
                progress,
                phase="variant-prediction-heartbeat",
                interval_seconds=float(resources["heartbeat_interval_seconds"]),
                details={"variant": variant, "variant_position": position},
            ):
                prediction = predict_direction_screen_model(
                    reloaded,
                    corpus.dataset,
                    calibration_indexes,
                )
            arrays = (
                prediction.long_superiority_probability,
                prediction.short_superiority_probability,
                prediction.conditional_long_probability,
                prediction.direction_score,
            )
            nonfinite = int(sum(np.sum(~np.isfinite(value)) for value in arrays))
            metrics = _variant_metrics(
                prediction=prediction,
                long_actual=calibration_long,
                short_actual=calibration_short,
                decision_time_ms=calibration_times,
                frozen_opportunity_probability=frozen_opportunity,
            )
            reasons = _eligibility_reasons(
                metrics,
                gates,
                nonfinite_predictions=nonfinite,
            )
            artifact_relative = artifact_path.relative_to(output_root).as_posix()
            result = {
                "variant": variant,
                "feature_set": feature_set,
                "weighting": weighting,
                "feature_count": len(selected_names),
                "selected_feature_names": list(selected_names),
                "model": {
                    "model_sha256": trained.model_sha256,
                    "artifact_path": artifact_relative,
                    "artifact_sha256": _sha256_file(artifact_path),
                    "artifact_bytes": artifact_path.stat().st_size,
                    "backend_requested": trained.backend_requested,
                    "backend_kind": trained.backend_kind,
                    "backend_device": trained.backend_device,
                    "lightgbm_version": trained.lightgbm_version,
                    "best_iteration": trained.best_iteration,
                    "train_role_rows": trained.train_role_rows,
                    "train_opportunity_rows": trained.train_opportunity_rows,
                    "early_stop_role_rows": trained.early_stop_role_rows,
                    "early_stop_opportunity_rows": trained.early_stop_opportunity_rows,
                    "utility_margin_scale_bps": trained.utility_margin_scale_bps,
                    "train_weight_multiplier_mean": (
                        trained.train_weight_multiplier_mean
                    ),
                    "early_stop_weight_multiplier_mean": (
                        trained.early_stop_weight_multiplier_mean
                    ),
                    "artifact_reload_max_abs_prediction_error": reload_error,
                    "mirror_swap_max_abs_prediction_error": mirror_error,
                    "nonfinite_prediction_count": nonfinite,
                    "training_runtime_seconds": training_seconds,
                },
                "metrics": metrics,
                "feature_gain": _feature_gain(trained),
                "architecture_freeze_eligible": not reasons,
                "rejection_reasons": reasons,
                "promotion_permitted": False,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
            variant_results.append(result)
            progress(
                "variant-complete",
                variant=variant,
                pooled_direction_auc=metrics["pooled_direction_auc"],
                minimum_daily_direction_auc=metrics["daily_auc_minimum"],
                frozen_top_500_mean_stress_net_bps=metrics["frozen_opportunity_ranked"][
                    "500"
                ]["mean_stress_net_bps"],
                eligible=not reasons,
                rejection_reason_count=len(reasons),
                training_runtime_seconds=round(training_seconds, 3),
            )
            del trained, reloaded, prediction, original_sample, reloaded_sample
            del mirrored_sample
            gc.collect()
        eligible = [
            item for item in variant_results if item["architecture_freeze_eligible"]
        ]
        selected_candidate = _select_candidate(eligible)
        binding_implementation = binding["implementation"]
        assert isinstance(binding_implementation, Mapping)
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "round": _ROUND,
            "status": (
                "candidate_identified" if selected_candidate is not None else "rejected"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_canonical_sha256": "PENDING",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": binding_implementation["commit"],
            "source_evidence": {
                "corpus_certificate_sha256": corpus.source_certificate[
                    "certificate_sha256"
                ],
                "barrier_targets_sha256": corpus.targets_sha256,
                "cache_key": corpus.cache_key,
                "cache_state": corpus.cache_state,
                "dataset_rows": corpus.dataset.rows,
                "event_rows": len(corpus.event_indexes),
                "valid_barrier_rows": int(np.sum(corpus.targets.valid)),
                "round34_report_canonical_sha256": round34_report_sha,
                "round34_model_sha256": [
                    model.model_sha256 for model in round34_models
                ],
            },
            "stage_access": {
                "certified_source_materialized_through_development": True,
                "train_used_for_fit": True,
                "early_stop_used_for_early_stopping": True,
                "calibration_prediction_and_metrics": True,
                "policy_prediction_or_metrics": False,
                "development_prediction_or_metrics": False,
                "distant_confirmation_source_materialization": False,
                "distant_confirmation_prediction_or_metrics": False,
            },
            "role_evidence": role_evidence,
            "runtime": {
                "elapsed_seconds": time.monotonic() - started,
                "thread_limits": thread_evidence,
                "directml_attestation": directml,
                "lightgbm_training_backend_required": "opencl",
                "lightgbm_gpu_use_dp_required": True,
                "lightgbm_prediction_backend": "cpu_library_prediction_path",
                "cpu_fallback_permitted": False,
                "sequential_variant_execution": True,
            },
            "calibration": {
                "event_rows": len(role_indexes["calibration"]),
                "positive_opportunity_rows": len(calibration_indexes),
                "long_superior_rows": int(np.sum(calibration_long > calibration_short)),
                "short_superior_rows": int(
                    np.sum(calibration_short > calibration_long)
                ),
                "first_decision_time_ms": int(calibration_times[0]),
                "last_decision_time_ms": int(calibration_times[-1]),
            },
            "variant_results": variant_results,
            "architecture_freeze_eligible_variants": [
                str(item["variant"]) for item in eligible
            ],
            "architecture_freeze_candidate": selected_candidate,
            "selection_rule": gates["tie_break_order"],
            "interpretation_contract": {
                "post_hoc_consumed_data_discovery_only": True,
                "candidate_is_not_out_of_sample_evidence": True,
                "candidate_requires_new_unseen_chronological_and_cross_asset_evaluation": True,
                "oracle_or_future_outcome_runtime_features_used": False,
            },
            "promotion_permitted": False,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        canonical = dict(report)
        canonical.pop("report_canonical_sha256")
        report["report_canonical_sha256"] = _canonical_sha256(canonical)
        write_json_atomic(report_path, report, indent=2, sort_keys=True)
        progress(
            "complete",
            status=report["status"],
            architecture_freeze_candidate=selected_candidate,
            report_canonical_sha256=report["report_canonical_sha256"],
        )
        return report
    except BaseException as exc:
        progress(
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(
        description="Run the bound Round 35 consumed-data direction screen.",
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--round34-evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-035-consumed-direction-screen-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-035-direction-screen-execution-binding.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_screen(
        design_path=arguments.design,
        binding_path=arguments.binding,
        round34_evidence_root=arguments.round34_evidence_root,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_root=arguments.output_root,
    )
    summary = {
        "status": report["status"],
        "architecture_freeze_candidate": report["architecture_freeze_candidate"],
        "report_canonical_sha256": report["report_canonical_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

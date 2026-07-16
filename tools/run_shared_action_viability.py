"""Execute the hash-bound Round 32 shared-action viability study."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.compute import resolve_backend  # noqa: E402
from simple_ai_trading.microstructure_action_features import (  # noqa: E402
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ACTION_POLICY_SCHEMA_VERSION,
    ActionPolicySpec,
    ActionScoreBatch,
    barrier_trace_gate_reasons,
    select_barrier_threshold,
    simulate_barrier_action_trace,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    _auc,
    _correlation,
    _rank,
    average_label_uniqueness,
    causal_cusum_event_mask,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
    build_adaptive_barrier_targets,
)
from simple_ai_trading.microstructure_cache import (  # noqa: E402
    load_microstructure_dataset_cache,
    microstructure_dataset_cache_key,
    save_microstructure_dataset_cache,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
    build_executable_microstructure_dataset,
    verify_executable_microstructure_source,
)
from simple_ai_trading.microstructure_shared_action_lightgbm import (  # noqa: E402
    SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION,
    SharedActionEnsembleBatch,
    SharedActionLightGBMSpec,
    SharedActionPredictionBatch,
    TrainedSharedActionLightGBMModel,
    ensemble_shared_action_predictions,
    load_shared_action_lightgbm_model,
    predict_shared_action_lightgbm_model,
    save_shared_action_lightgbm_model,
    train_shared_action_lightgbm_model,
)
from simple_ai_trading.microstructure_shared_action_policy import (  # noqa: E402
    SHARED_ACTION_POLICY_SCHEMA_VERSION,
    derive_shared_action_scores,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DESIGN_SCHEMA_VERSION = "shared-action-value-viability-design-v1"
BINDING_SCHEMA_VERSION = "round-032-execution-binding-v1"
REPORT_SCHEMA_VERSION = "shared-action-value-viability-report-v1"
_DAY_MS = 86_400_000
_TRAINING_ROLES = ("train", "early_stop", "calibration", "policy", "development")
_PROFILE_NAMES = ("conservative", "regular", "aggressive")
_GATE_FIELDS = {
    "minimum_trades",
    "minimum_total_net_bps",
    "maximum_drawdown_bps",
    "minimum_positive_day_ratio",
    "minimum_worst_trade_bps",
    "minimum_profit_factor",
}
_PREDICTION_ARRAY_FIELDS = (
    "long_mean_bps",
    "short_mean_bps",
    "long_profitable_probability",
    "short_profitable_probability",
    "long_lower_bps",
    "short_lower_bps",
    "long_upper_bps",
    "short_upper_bps",
    "signed_advantage_bps",
)
_REQUIRED_BOUND_PATHS = frozenset(
    {
        "docs/model-research/action-value/round-031-frozen-chronological-confirmation-design.json",
        "docs/model-research/action-value/round-032-shared-action-value-viability-design.json",
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
        "src/simple_ai_trading/microstructure_shared_action_lightgbm.py",
        "src/simple_ai_trading/microstructure_shared_action_policy.py",
        "src/simple_ai_trading/microstructure_warehouse.py",
        "src/simple_ai_trading/probability_calibration.py",
        "src/simple_ai_trading/progress_heartbeat.py",
        "src/simple_ai_trading/storage.py",
        "tools/run_shared_action_viability.py",
    }
)


@dataclass(frozen=True)
class CorpusBundle:
    name: str
    dataset: MicrostructureDataset
    targets: AdaptiveBarrierTargets
    event_mask: np.ndarray
    event_indexes: np.ndarray
    requested_start_ms: int
    requested_end_ms: int
    cache_key: str
    cache_state: str
    source_evidence: Mapping[str, object]
    source_certificate: Mapping[str, object]
    targets_sha256: str


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_date(value: object, *, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD") from exc


def _utc_day_bounds(first: date, last: date) -> tuple[int, int]:
    if first > last:
        raise ValueError("UTC day interval is reversed")
    start_ms = int(
        datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        * 1_000
    )
    end_ms = (
        int(
            datetime.combine(
                last + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).timestamp()
            * 1_000
        )
        - 1
    )
    return start_ms, end_ms


def _date_set(first: object, last: object, *, label: str) -> set[str]:
    start = _parse_date(first, label=f"{label} start")
    end = _parse_date(last, label=f"{label} end")
    if start > end:
        raise ValueError(f"{label} interval is reversed")
    return {
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    }


def _validate_gates(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _GATE_FIELDS:
        raise ValueError(f"{label} risk gates are incomplete")
    numeric = tuple(float(value[name]) for name in _GATE_FIELDS)
    if (
        not all(math.isfinite(item) for item in numeric)
        or int(value["minimum_trades"]) < 1
        or float(value["minimum_total_net_bps"]) < 0.0
        or float(value["maximum_drawdown_bps"]) <= 0.0
        or not 0.0 <= float(value["minimum_positive_day_ratio"]) <= 1.0
        or float(value["minimum_worst_trade_bps"]) >= 0.0
        or float(value["minimum_profit_factor"]) < 1.0
    ):
        raise ValueError(f"{label} risk gates are invalid")


def _risk_profiles_from_source(
    design_path: Path,
    design: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    source = design.get("risk_profiles_source")
    if not isinstance(source, Mapping):
        raise ValueError("Round 32 risk-profile source is missing")
    relative = Path(str(source.get("path") or ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.name != str(relative)
    ):
        raise ValueError("Round 32 risk-profile source path is unsafe")
    source_path = design_path.parent / relative
    if _sha256_file(source_path) != source.get("file_sha256"):
        raise ValueError("Round 32 risk-profile source file changed")
    predecessor = _read_object(source_path, label="Round 31 design")
    canonical = dict(predecessor)
    predecessor_sha = canonical.pop("design_sha256", None)
    if (
        not _is_sha256(predecessor_sha)
        or predecessor_sha != _canonical_sha256(canonical)
        or predecessor_sha != source.get("design_sha256")
        or source.get("section") != "risk_profiles"
        or source.get("modification_permitted") is not False
    ):
        raise ValueError("Round 32 risk-profile provenance is invalid")
    raw_profiles = predecessor.get("risk_profiles")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 3:
        raise ValueError("Round 31 risk profiles are incomplete")
    profiles: list[dict[str, object]] = []
    for expected_name, raw in zip(_PROFILE_NAMES, raw_profiles, strict=True):
        if not isinstance(raw, dict) or raw.get("profile") != expected_name:
            raise ValueError("Round 31 risk-profile order or identity changed")
        ActionPolicySpec(
            **{
                name: raw[name]
                for name in (
                    "profile",
                    "epistemic_penalty",
                    "minimum_profitable_probability",
                    "minimum_member_agreement",
                    "maximum_epistemic_std_bps",
                    "minimum_lower_bound_bps",
                )
            }
        )
        for stage in ("calibration", "policy", "development"):
            _validate_gates(raw.get(f"{stage}_gates"), label=f"{expected_name} {stage}")
        if raw.get("leverage_applied") is not False:
            raise ValueError("Round 32 risk profiles must remain unleveraged")
        profiles.append(dict(raw))
    return tuple(profiles), predecessor


def _shared_action_spec(design: Mapping[str, object]) -> SharedActionLightGBMSpec:
    model = design.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Round 32 model contract is missing")
    lightgbm = model.get("lightgbm")
    if not isinstance(lightgbm, Mapping):
        raise ValueError("Round 32 LightGBM contract is missing")
    parameters = dict(lightgbm)
    backend = parameters.pop("backend", None)
    cpu_fallback = parameters.pop("cpu_fallback_permitted", None)
    if backend != "opencl" or cpu_fallback is not False:
        raise ValueError("Round 32 accelerator contract changed")
    return SharedActionLightGBMSpec(
        candidate_id=str(model.get("candidate_id") or ""),
        family=str(model.get("family") or ""),
        **parameters,
    )


def load_round32_design(
    path: str | Path,
) -> tuple[dict[str, object], str, tuple[dict[str, object], ...]]:
    """Load and semantically validate the frozen Round 32 design."""

    source_path = Path(path).resolve()
    payload = _read_object(source_path, label="Round 32 design")
    canonical = dict(payload)
    claimed = canonical.pop("design_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("Round 32 design hash is invalid")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or payload.get("round") != 32
        or payload.get("design_revision") != 4
        or payload.get("purpose")
        != "consumed_data_direction_canonical_shared_action_value_viability"
        or payload.get("claims")
        != {
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "trading_authority": False,
            "leverage_applied": False,
        }
    ):
        raise ValueError("Round 32 top-level contract is invalid")
    governance = payload.get("governance")
    data = payload.get("data")
    execution = payload.get("execution")
    barriers = payload.get("barrier_targets")
    sampler = payload.get("event_sampler")
    weighting = payload.get("sample_weighting")
    selection = payload.get("selection")
    stages = payload.get("stage_evaluation_order")
    resources = payload.get("runtime_resources")
    gates = payload.get("acceptance_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (
            governance,
            data,
            execution,
            barriers,
            sampler,
            weighting,
            selection,
            stages,
            resources,
            gates,
        )
    ):
        raise ValueError("Round 32 design sections are incomplete")
    assert isinstance(governance, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(barriers, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(weighting, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(gates, Mapping)
    if (
        governance.get("variant_budget") != 1
        or governance.get("hyperparameter_search_permitted") is not False
        or governance.get("all_target_dates_already_consumed") is not True
        or governance.get("thresholds_may_use_calibration_only") is not True
        or governance.get("later_stage_parameters_may_not_change") is not True
        or governance.get("no_passing_candidate_action") != "reject_and_abstain"
    ):
        raise ValueError("Round 32 governance contract changed")
    if (
        data.get("symbol") != "BTCUSDT"
        or data.get("provider") != "binance"
        or data.get("market_type") != "futures"
        or tuple(data.get("required_data_types") or ()) != ("bookTicker", "trades")
        or data.get("full_history_inventory_required") is not True
        or data.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION
    ):
        raise ValueError("Round 32 data contract changed")
    roles = data.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        *_TRAINING_ROLES,
        "distant_confirmation",
    }:
        raise ValueError("Round 32 role contract is incomplete")
    previous_end: date | None = None
    evaluated_dates: set[str] = set()
    for role_name in _TRAINING_ROLES:
        role = roles[role_name]
        if not isinstance(role, Mapping):
            raise ValueError(f"Round 32 {role_name} role is invalid")
        first = _parse_date(role.get("start"), label=f"{role_name} start")
        last = _parse_date(role.get("end"), label=f"{role_name} end")
        if first > last or (
            previous_end is not None and first != previous_end + timedelta(days=1)
        ):
            raise ValueError("Round 32 training roles are not contiguous")
        previous_end = last
        evaluated_dates.update(_date_set(first, last, label=role_name))
    distant = roles["distant_confirmation"]
    if not isinstance(distant, Mapping):
        raise ValueError("Round 32 distant role is invalid")
    context = _parse_date(distant.get("context_start"), label="distant context")
    distant_start = _parse_date(distant.get("start"), label="distant start")
    distant_end = _parse_date(distant.get("end"), label="distant end")
    if context >= distant_start or distant_start > distant_end:
        raise ValueError("Round 32 distant context is invalid")
    evaluated_dates.update(_date_set(distant_start, distant_end, label="distant"))
    forbidden_dates: set[str] = set()
    for raw in data.get("forbidden_target_windows") or ():
        if not isinstance(raw, Mapping):
            raise ValueError("Round 32 forbidden target window is invalid")
        forbidden_dates.update(
            _date_set(raw.get("start"), raw.get("end"), label="forbidden")
        )
    if evaluated_dates & forbidden_dates:
        raise ValueError("Round 32 roles intersect an untouched target window")
    registry_path = source_path.parent / str(governance.get("consumed_period_registry"))
    if _sha256_file(registry_path) != governance.get(
        "consumed_period_registry_file_sha256"
    ):
        raise ValueError("Round 32 consumed-period registry file changed")
    registry = _read_object(registry_path, label="consumed-period registry")
    registry_canonical = dict(registry)
    registry_sha = registry_canonical.pop("registry_sha256", None)
    if registry_sha != _canonical_sha256(
        registry_canonical
    ) or registry_sha != governance.get("consumed_period_registry_canonical_sha256"):
        raise ValueError("Round 32 consumed-period registry hash is invalid")
    consumed_dates: set[str] = set()
    for record in registry.get("records") or ():
        if not isinstance(record, Mapping):
            raise ValueError("consumed-period registry record is invalid")
        for window in record.get("windows") or ():
            if not isinstance(window, Mapping):
                raise ValueError("consumed-period registry window is invalid")
            consumed_dates.update(
                _date_set(
                    window.get("start_date"),
                    window.get("end_date"),
                    label="consumed",
                )
            )
    if not evaluated_dates <= consumed_dates:
        raise ValueError("Round 32 attempts to access an unconsumed target date")
    profiles, predecessor = _risk_profiles_from_source(source_path, payload)
    if execution != predecessor.get("execution"):
        raise ValueError("Round 32 execution costs drifted from Round 31")
    barrier_contract = dict(barriers)
    if barrier_contract.pop("target_scenario", None) != "stress" or (
        barrier_contract != predecessor.get("barrier_targets")
    ):
        raise ValueError("Round 32 barrier contract drifted from Round 31")
    if (
        execution.get("horizon_seconds") != barriers.get("horizon_seconds")
        or sampler
        != {
            "method": "daily_reset_causal_cusum",
            "volatility_multiplier": 0.25,
            "minimum_threshold_bps": 1.0,
            "minimum_activity_quota": None,
        }
        or weighting
        != {
            "method": "average_label_uniqueness",
            "training_role": "train",
            "tuning_role": "early_stop",
            "computed_from": "decision_time_ms_and_maximum_realized_exit_time_ms",
            "paired_action_rows_share_event_weight": True,
        }
    ):
        raise ValueError("Round 32 target sampling or weighting contract changed")
    model_spec = _shared_action_spec(payload)
    model = payload["model"]
    assert isinstance(model, Mapping)
    if (
        tuple(model.get("seeds") or ()) != (29, 43, 71)
        or model.get("action_rows_per_event") != 2
        or model.get("paired_event_split_required") is not True
        or model.get("tuning_partition")
        != {
            "source_role": "early_stop",
            "partition": "chronological_first_half_early_stopping_second_half_probability_calibration",
            "calibration_fraction": 0.5,
            "purge_rule": "early_stop_maximum_exit_time_strictly_before_calibration_start",
            "minimum_rows_per_subpartition": 256,
        }
        or model_spec.lower_quantile != 0.1
        or model_spec.upper_quantile != 0.9
        or model_spec.calibration_fraction != 0.5
    ):
        raise ValueError("Round 32 model split or seed contract changed")
    if (
        tuple(float(value) for value in selection.get("threshold_quantiles") or ())
        != (0.5, 0.7, 0.85, 0.95)
        or selection.get("threshold_source_role") != "calibration"
        or selection.get("selection_scenario") != "stress"
        or selection.get("exact_non_overlapping_execution_required") is not True
        or selection.get("unfavorable_bbo_marking_required") is not True
    ):
        raise ValueError("Round 32 threshold-selection contract changed")
    stage_values = stages.get("stages")
    if (
        stages.get("survival_scope") != "per_risk_profile"
        or stages.get("later_stage_predictions_withheld_until_prior_stage_passes")
        is not True
        or stages.get("thresholds_frozen_after_calibration") is not True
        or not isinstance(stage_values, list)
        or [value.get("stage") for value in stage_values if isinstance(value, Mapping)]
        != ["calibration", "policy", "development", "distant_confirmation"]
    ):
        raise ValueError("Round 32 stage-withholding contract changed")
    if resources != {
        "duckdb_memory_limit_gib": 12,
        "maximum_worker_threads": 16,
        "corpus_lifecycle": "sequential_with_distant_corpus_loaded_only_after_development_pass",
        "accelerator_preflight": "directml",
        "lightgbm_backend": "opencl",
        "cpu_fallback_permitted": False,
        "heartbeat_interval_seconds": 30,
        "progress_status_atomic_write_required": True,
    }:
        raise ValueError("Round 32 runtime-resource contract changed")
    implementation_gates = gates.get("implementation")
    forecast_gates = gates.get("distant_confirmation_forecast")
    economic_gates = gates.get("economic")
    if (
        not isinstance(implementation_gates, Mapping)
        or not isinstance(forecast_gates, Mapping)
        or not isinstance(economic_gates, Mapping)
        or implementation_gates.get("single_source_certificate_per_dataset_load")
        is not True
        or implementation_gates.get("artifact_reload_equivalence_required") is not True
        or economic_gates
        != {
            "all_stage_profile_gates_must_pass": True,
            "minimum_passing_profile_count": 1,
            "positive_stress_total_required": True,
            "leverage_permitted": False,
        }
    ):
        raise ValueError("Round 32 acceptance-gate contract changed")
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
        raise ValueError("Round 32 Git binding command failed") from exc


def load_round32_execution_binding(
    path: str | Path,
    *,
    design_path: str | Path,
    design_sha256: str,
    require_current_implementation: bool = False,
) -> tuple[dict[str, object], str]:
    """Verify historical provenance and optionally authorize a current-code run."""

    binding = _read_object(Path(path), label="Round 32 execution binding")
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256", None)
    implementation = binding.get("implementation")
    design = binding.get("design")
    if (
        not _is_sha256(claimed)
        or claimed != _canonical_sha256(canonical)
        or binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("round") != 32
        or binding.get("worktree_policy") != "clean_including_untracked"
        or not isinstance(implementation, Mapping)
        or not isinstance(design, Mapping)
    ):
        raise ValueError("Round 32 execution binding is invalid")
    commit = str(implementation.get("commit") or "").lower()
    files_value = implementation.get("files")
    if (
        not _is_git_oid(commit)
        or implementation.get("hash_mode") != "git_blob_sha256_v1"
        or not isinstance(files_value, list)
    ):
        raise ValueError("Round 32 implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    bound_files: dict[str, str] = {}
    for item in files_value:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("Round 32 bound file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in bound_files
        ):
            raise ValueError("Round 32 bound path is unsafe or duplicated")
        bound_files[normalized] = str(item["sha256"])
    if set(bound_files) != _REQUIRED_BOUND_PATHS:
        missing = sorted(_REQUIRED_BOUND_PATHS - set(bound_files))
        extra = sorted(set(bound_files) - _REQUIRED_BOUND_PATHS)
        raise ValueError(
            f"Round 32 bound file scope changed: missing={missing} extra={extra}"
        )
    for normalized, expected in bound_files.items():
        historical = _git_bytes("show", f"{commit}:{normalized}")
        if hashlib.sha256(historical).hexdigest() != expected:
            raise ValueError(f"Round 32 historical binding changed: {normalized}")
    if require_current_implementation:
        for normalized, expected in bound_files.items():
            current = _git_bytes("show", f"HEAD:{normalized}")
            if hashlib.sha256(current).hexdigest() != expected:
                raise ValueError(f"Round 32 implementation changed: {normalized}")
        status = _git_bytes("status", "--porcelain", "--untracked-files=all")
        if status.strip():
            raise ValueError("Round 32 execution requires a clean worktree")
    resolved_design_path = Path(design_path).resolve()
    relative_design = resolved_design_path.relative_to(ROOT).as_posix()
    expected_design_file_sha = bound_files.get(relative_design)
    if (
        design.get("path") != relative_design
        or design.get("design_sha256") != design_sha256
        or design.get("file_sha256") != expected_design_file_sha
        or hashlib.sha256(resolved_design_path.read_bytes()).hexdigest()
        != expected_design_file_sha
    ):
        raise ValueError("Round 32 design binding changed")
    return binding, str(claimed)


def _attest_directml() -> dict[str, object]:
    backend = resolve_backend("directml")
    if backend.kind != "directml":
        raise RuntimeError(f"DirectML preflight failed: {backend.reason}")
    try:
        import torch
        import torch_directml

        device = torch_directml.device()
        with torch.no_grad():
            left = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(device)
            right = torch.eye(16, dtype=torch.float32).to(device)
            observed = float((left @ right).sum().cpu().item())
    except Exception as exc:  # pragma: no cover - host-specific failure path
        raise RuntimeError("DirectML tensor execution failed") from exc
    expected = float(sum(range(256)))
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("DirectML tensor execution returned an invalid result")
    return {
        "requested": "directml",
        "kind": backend.kind,
        "device": str(device),
        "vendor": backend.vendor,
        "torch_version": str(torch.__version__),
        "torch_directml_version": str(
            getattr(torch_directml, "__version__", "unknown")
        ),
        "tensor_identity_sum": observed,
        "status": "pass",
    }


def _configure_worker_threads(maximum_threads: int) -> dict[str, object]:
    maximum = int(maximum_threads)
    if maximum < 1:
        raise ValueError("worker thread limit must be positive")
    available = max(1, int(os.cpu_count() or 1))
    try:
        import numba

        numba_capacity = max(1, int(numba.config.NUMBA_NUM_THREADS))
    except ImportError as exc:
        raise RuntimeError("Numba is unavailable") from exc
    threads = min(maximum, available, numba_capacity)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        numba.set_num_threads(threads)
        numba_threads = int(numba.get_num_threads())
    except (ImportError, ValueError) as exc:
        raise RuntimeError("Numba worker-thread limit could not be enforced") from exc
    if numba_threads != threads:
        raise RuntimeError("Numba worker-thread limit drifted")
    return {
        "configured_maximum_worker_threads": maximum,
        "logical_cpu_count": available,
        "effective_worker_threads": threads,
        "numba_threads": numba_threads,
        "lightgbm_num_threads_upper_bound": min(maximum, available),
        "duckdb_threads": threads,
    }


def _targets_sha256(targets: AdaptiveBarrierTargets) -> str:
    contract = {
        "schema_version": targets.schema_version,
        "target_mode": targets.target_mode,
        "spec": asdict(targets.spec),
    }
    digest = hashlib.sha256(_canonical_json(contract).encode("ascii"))
    for name in (
        "source_indexes",
        "valid",
        "stop_barrier_bps",
        "take_barrier_bps",
        "base_long_net_bps",
        "base_short_net_bps",
        "base_long_exit_time_ms",
        "base_short_exit_time_ms",
        "base_long_outcome",
        "base_short_outcome",
        "stress_long_net_bps",
        "stress_short_net_bps",
        "stress_long_exit_time_ms",
        "stress_short_exit_time_ms",
        "stress_long_outcome",
        "stress_short_outcome",
    ):
        values = np.ascontiguousarray(getattr(targets, name))
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(values.dtype.str.encode("ascii") + b"\x00")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _load_corpus(
    *,
    name: str,
    design: Mapping[str, object],
    warehouse_path: str | Path,
    cache_root: str | Path,
    first: date,
    last: date,
    evaluation_first: date,
    evaluation_last: date,
    memory_limit: str,
    threads: int,
    heartbeat_seconds: float,
    progress: Callable[..., None],
) -> CorpusBundle:
    data = design["data"]
    execution = design["execution"]
    sampler = design["event_sampler"]
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(sampler, Mapping)
    requested_start_ms, requested_end_ms = _utc_day_bounds(first, last)
    evaluation_start_ms, evaluation_end_ms = _utc_day_bounds(
        evaluation_first, evaluation_last
    )
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=memory_limit,
        threads=threads,
    ) as warehouse:
        progress("source-certificate-start", corpus=name)
        with progress_heartbeat(
            progress,
            phase="source-certificate-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"corpus": name},
        ):
            verified = verify_executable_microstructure_source(
                warehouse,
                symbol=str(data["symbol"]),
                start_ms=requested_start_ms,
                end_ms=requested_end_ms,
                require_full_history_inventory=True,
                feature_version=str(data["feature_version"]),
            )
        source_evidence = dict(verified.evidence)
        certificate = source_evidence.get("corpus_certificate")
        if not isinstance(certificate, Mapping):
            raise ValueError("verified corpus certificate is missing")
        progress(
            "source-certificate-complete",
            corpus=name,
            certificate_sha256=certificate.get("certificate_sha256"),
        )
        cache_parameters = {
            "symbol": str(data["symbol"]),
            "requested_start_ms": requested_start_ms,
            "requested_end_ms": requested_end_ms,
            "horizon_seconds": int(execution["horizon_seconds"]),
            "total_latency_ms": int(execution["total_latency_ms"]),
            "taker_fee_bps": float(execution["taker_fee_bps_per_side"]),
            "additional_slippage_bps_per_side": float(
                execution["additional_slippage_bps_per_side"]
            ),
            "reference_order_notional_quote": float(
                execution["reference_order_notional_quote"]
            ),
            "max_l1_participation": float(execution["max_l1_participation"]),
            "max_quote_age_ms": int(execution["max_quote_age_ms"]),
            "decision_cadence_seconds": int(execution["decision_cadence_seconds"]),
            "require_full_history_inventory": True,
            "source_evidence": source_evidence,
            "feature_version": str(data["feature_version"]),
        }
        cache_key = microstructure_dataset_cache_key(**cache_parameters)
        progress("cache-lookup-start", corpus=name, cache_key=cache_key)
        with progress_heartbeat(
            progress,
            phase="cache-lookup-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"corpus": name, "cache_key": cache_key},
        ):
            dataset = load_microstructure_dataset_cache(warehouse, **cache_parameters)
        cache_state = "hit"
        if dataset is None:
            cache_state = "build"
            progress("dataset-build-start", corpus=name, cache_key=cache_key)
            with progress_heartbeat(
                progress,
                phase="dataset-build-heartbeat",
                interval_seconds=heartbeat_seconds,
                details={"corpus": name, "cache_key": cache_key},
            ):
                dataset = build_executable_microstructure_dataset(
                    warehouse,
                    symbol=str(data["symbol"]),
                    horizon_seconds=int(execution["horizon_seconds"]),
                    total_latency_ms=int(execution["total_latency_ms"]),
                    taker_fee_bps=float(execution["taker_fee_bps_per_side"]),
                    additional_slippage_bps_per_side=float(
                        execution["additional_slippage_bps_per_side"]
                    ),
                    max_quote_age_ms=int(execution["max_quote_age_ms"]),
                    reference_order_notional_quote=float(
                        execution["reference_order_notional_quote"]
                    ),
                    max_l1_participation=float(execution["max_l1_participation"]),
                    decision_cadence_seconds=int(execution["decision_cadence_seconds"]),
                    start_ms=requested_start_ms,
                    end_ms=requested_end_ms,
                    require_full_history_inventory=True,
                    feature_version=str(data["feature_version"]),
                    verified_source=verified,
                )
            progress("cache-write-start", corpus=name, dataset_rows=dataset.rows)
            with progress_heartbeat(
                progress,
                phase="cache-write-heartbeat",
                interval_seconds=heartbeat_seconds,
                details={"corpus": name, "dataset_rows": dataset.rows},
            ):
                written_key = save_microstructure_dataset_cache(
                    warehouse,
                    dataset,
                    requested_start_ms=requested_start_ms,
                    requested_end_ms=requested_end_ms,
                    require_full_history_inventory=True,
                )
            if written_key != cache_key:
                raise ValueError("microstructure cache key changed during write")
            cache_state = "written"
        progress("event-sampler-start", corpus=name, dataset_rows=dataset.rows)
        with progress_heartbeat(
            progress,
            phase="event-sampler-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"corpus": name},
        ):
            event_mask = causal_cusum_event_mask(
                dataset,
                volatility_multiplier=float(sampler["volatility_multiplier"]),
                minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
            )
        evaluation_mask = (
            np.asarray(event_mask, dtype=bool)
            & (dataset.decision_time_ms >= evaluation_start_ms)
            & (dataset.decision_time_ms <= evaluation_end_ms)
        )
        event_indexes = np.flatnonzero(evaluation_mask).astype(np.int64)
        if len(event_indexes) < 1_024:
            raise ValueError(f"{name} corpus event support is insufficient")
        barrier_config = dict(design["barrier_targets"])
        if barrier_config.pop("target_scenario", None) != "stress":
            raise ValueError("barrier target scenario changed")
        progress(
            "barrier-target-build-start", corpus=name, event_rows=len(event_indexes)
        )
        with progress_heartbeat(
            progress,
            phase="barrier-target-build-heartbeat",
            interval_seconds=heartbeat_seconds,
            details={"corpus": name, "event_rows": len(event_indexes)},
        ):
            targets = build_adaptive_barrier_targets(
                warehouse,
                dataset,
                event_indexes,
                spec=AdaptiveBarrierSpec(**barrier_config),
                progress=lambda day, total, valid: progress(
                    "barrier-target-day",
                    corpus=name,
                    day=day,
                    days=total,
                    valid_rows=valid,
                ),
            )
    target_sha = _targets_sha256(targets)
    progress(
        "corpus-ready",
        corpus=name,
        dataset_rows=dataset.rows,
        event_rows=len(event_indexes),
        valid_target_rows=targets.valid_rows,
        cache_state=cache_state,
        barrier_targets_sha256=target_sha,
    )
    return CorpusBundle(
        name=name,
        dataset=dataset,
        targets=targets,
        event_mask=evaluation_mask,
        event_indexes=event_indexes,
        requested_start_ms=requested_start_ms,
        requested_end_ms=requested_end_ms,
        cache_key=cache_key,
        cache_state=cache_state,
        source_evidence=source_evidence,
        source_certificate=dict(certificate),
        targets_sha256=target_sha,
    )


def _maximum_exit_full(bundle: CorpusBundle) -> np.ndarray:
    maximum = np.maximum.reduce(
        (
            bundle.targets.base_long_exit_time_ms,
            bundle.targets.base_short_exit_time_ms,
            bundle.targets.stress_long_exit_time_ms,
            bundle.targets.stress_short_exit_time_ms,
        )
    )
    output = np.full(bundle.dataset.rows, -1, dtype=np.int64)
    output[bundle.targets.source_indexes] = maximum
    return output


def _role_indexes(
    bundle: CorpusBundle,
    roles: Mapping[str, object],
    role_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object], np.ndarray]:
    valid = np.zeros(bundle.dataset.rows, dtype=bool)
    valid[bundle.targets.source_indexes[bundle.targets.valid]] = True
    maximum_exit = _maximum_exit_full(bundle)
    output: dict[str, np.ndarray] = {}
    evidence: dict[str, object] = {}
    for position, role_name in enumerate(role_names):
        role = roles[role_name]
        if not isinstance(role, Mapping):
            raise ValueError(f"{role_name} role is invalid")
        first = _parse_date(role.get("start"), label=f"{role_name} start")
        last = _parse_date(role.get("end"), label=f"{role_name} end")
        if position + 1 < len(role_names):
            following = roles[role_names[position + 1]]
            if not isinstance(following, Mapping):
                raise ValueError("following role is invalid")
            next_day = _parse_date(
                following.get("start"), label=f"{role_name} next start"
            )
        else:
            next_day = last + timedelta(days=1)
        first_ms, last_ms = _utc_day_bounds(first, last)
        next_ms, _unused = _utc_day_bounds(next_day, next_day)
        indexes = np.flatnonzero(
            np.asarray(bundle.event_mask, dtype=bool)
            & valid
            & (bundle.dataset.decision_time_ms >= first_ms)
            & (bundle.dataset.decision_time_ms <= last_ms)
            & (maximum_exit < next_ms)
        ).astype(np.int64)
        if len(indexes) < 256:
            raise ValueError(f"{role_name} role support is insufficient")
        output[role_name] = indexes
        evidence[role_name] = {
            "start": first.isoformat(),
            "end": last.isoformat(),
            "rows": len(indexes),
            "first_decision_time_ms": int(bundle.dataset.decision_time_ms[indexes[0]]),
            "last_decision_time_ms": int(bundle.dataset.decision_time_ms[indexes[-1]]),
            "last_exit_time_ms": int(np.max(maximum_exit[indexes])),
            "next_role_start_ms": next_ms,
            "purged": bool(np.max(maximum_exit[indexes]) < next_ms),
        }
    return output, evidence, maximum_exit


def _profile_spec(raw: Mapping[str, object]) -> ActionPolicySpec:
    return ActionPolicySpec(
        **{
            name: raw[name]
            for name in (
                "profile",
                "epistemic_penalty",
                "minimum_profitable_probability",
                "minimum_member_agreement",
                "maximum_epistemic_std_bps",
                "minimum_lower_bound_bps",
            )
        }
    )


def _expected_days(role: Mapping[str, object]) -> tuple[int, ...]:
    first = _parse_date(role.get("start"), label="expected-day start")
    last = _parse_date(role.get("end"), label="expected-day end")
    first_ms, _unused = _utc_day_bounds(first, first)
    last_ms, _unused = _utc_day_bounds(last, last)
    return tuple(range(first_ms // _DAY_MS, last_ms // _DAY_MS + 1))


def _target_positions(
    targets: AdaptiveBarrierTargets,
    endpoints: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(endpoints, dtype=np.int64)
    positions = np.searchsorted(targets.source_indexes, selected)
    if (
        np.any(positions >= targets.rows)
        or not np.array_equal(targets.source_indexes[positions], selected)
        or not np.all(targets.valid[positions])
    ):
        raise ValueError("prediction endpoints differ from valid barrier targets")
    return positions


def _scenario_targets(
    targets: AdaptiveBarrierTargets,
    endpoints: np.ndarray,
    *,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _target_positions(targets, endpoints)
    if scenario == "base":
        return (
            np.asarray(targets.base_long_net_bps[positions], dtype=np.float64),
            np.asarray(targets.base_short_net_bps[positions], dtype=np.float64),
        )
    if scenario == "stress":
        return (
            np.asarray(targets.stress_long_net_bps[positions], dtype=np.float64),
            np.asarray(targets.stress_short_net_bps[positions], dtype=np.float64),
        )
    raise ValueError("unsupported forecast diagnostic scenario")


def _raw_forecast_diagnostics(
    targets: AdaptiveBarrierTargets,
    ensemble: SharedActionEnsembleBatch,
    *,
    scenario: str,
) -> dict[str, object]:
    action = ensemble.action_values
    actuals = _scenario_targets(targets, ensemble.endpoint_indexes, scenario=scenario)
    output: dict[str, object] = {}
    for side, actual, prediction, probability, lower, upper, epistemic in (
        (
            "long",
            actuals[0],
            action.long_mean_bps,
            action.long_profitable_probability,
            action.long_lower_bps,
            action.long_upper_bps,
            action.long_epistemic_std_bps,
        ),
        (
            "short",
            actuals[1],
            action.short_mean_bps,
            action.short_profitable_probability,
            action.short_lower_bps,
            action.short_upper_bps,
            action.short_epistemic_std_bps,
        ),
    ):
        predicted = np.asarray(prediction, dtype=np.float64)
        probabilities = np.asarray(probability, dtype=np.float64)
        lower_values = np.asarray(lower, dtype=np.float64)
        upper_values = np.asarray(upper, dtype=np.float64)
        std = np.asarray(epistemic, dtype=np.float64)
        labels = np.asarray(actual > 0.0, dtype=np.int8)
        prevalence = float(np.mean(labels))
        order = np.argsort(-(predicted - std), kind="stable")
        top_rows: list[dict[str, object]] = []
        for requested in (100, 500, 1_000):
            count = min(requested, len(order))
            chosen = order[:count]
            top_rows.append(
                {
                    "requested_rows": requested,
                    "actual_rows": count,
                    "mean_actual_net_bps": float(np.mean(actual[chosen])),
                    "positive_rate": float(np.mean(actual[chosen] > 0.0)),
                }
            )
        output[side] = {
            "rows": len(actual),
            "actual_positive_ratio": prevalence,
            "mean_actual_net_bps": float(np.mean(actual)),
            "mean_prediction_bps": float(np.mean(predicted)),
            "mean_absolute_error_bps": float(np.mean(np.abs(predicted - actual))),
            "zero_baseline_mae_bps": float(np.mean(np.abs(actual))),
            "root_mean_squared_error_bps": float(
                np.sqrt(np.mean((predicted - actual) ** 2))
            ),
            "zero_baseline_rmse_bps": float(np.sqrt(np.mean(actual**2))),
            "pearson_information_coefficient": _correlation(actual, predicted),
            "spearman_information_coefficient": _correlation(
                _rank(actual), _rank(predicted)
            ),
            "profitable_auc": _auc(labels, probabilities),
            "profitable_brier": float(np.mean((probabilities - labels) ** 2)),
            "prevalence_brier": float(np.mean((prevalence - labels) ** 2)),
            "interval_80_coverage": float(
                np.mean((actual >= lower_values) & (actual <= upper_values))
            ),
            "mean_epistemic_std_bps": float(np.mean(std)),
            "top_rows": top_rows,
        }
    return {
        "scenario": scenario,
        "sides": output,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _selected_action_diagnostics(
    targets: AdaptiveBarrierTargets,
    ensemble: SharedActionEnsembleBatch,
) -> dict[str, object]:
    action = ensemble.action_values
    long_actual, short_actual = _scenario_targets(
        targets, ensemble.endpoint_indexes, scenario="stress"
    )
    advantage = np.asarray(ensemble.signed_advantage_mean_bps, dtype=np.float64)
    side = np.where(
        np.asarray(action.long_mean_bps) >= np.asarray(action.short_mean_bps),
        1,
        -1,
    ).astype(np.int8)
    selected_actual = np.where(side == 1, long_actual, short_actual)
    selected_prediction = np.where(
        side == 1,
        np.asarray(action.long_mean_bps, dtype=np.float64),
        np.asarray(action.short_mean_bps, dtype=np.float64),
    )
    selected_std = np.where(
        side == 1,
        np.asarray(action.long_epistemic_std_bps, dtype=np.float64),
        np.asarray(action.short_epistemic_std_bps, dtype=np.float64),
    )
    risk_adjusted = selected_prediction - selected_std
    actual_long_better = np.asarray(long_actual > short_actual, dtype=np.int8)
    order = np.argsort(-risk_adjusted, kind="stable")
    top: dict[str, dict[str, object]] = {}
    for requested in (100, 500, 1_000):
        count = min(requested, len(order))
        chosen = order[:count]
        top[str(requested)] = {
            "requested_rows": requested,
            "actual_rows": count,
            "mean_stress_net_bps": float(np.mean(selected_actual[chosen])),
            "median_stress_net_bps": float(np.median(selected_actual[chosen])),
            "positive_ratio": float(np.mean(selected_actual[chosen] > 0.0)),
            "long_share": float(np.mean(side[chosen] == 1)),
        }
    advantage_side = np.where(advantage >= 0.0, 1, -1)
    return {
        "rows": len(side),
        "side_choice_auc": _auc(actual_long_better, advantage),
        "selected_action_pearson_information_coefficient": _correlation(
            selected_actual, selected_prediction
        ),
        "selected_action_spearman_information_coefficient": _correlation(
            _rank(selected_actual), _rank(selected_prediction)
        ),
        "mean_selected_prediction_bps": float(np.mean(selected_prediction)),
        "mean_selected_stress_net_bps": float(np.mean(selected_actual)),
        "action_advantage_side_agreement": float(np.mean(advantage_side == side)),
        "overall_long_share": float(np.mean(side == 1)),
        "top_rows": top,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _forecast_gate_reasons(
    diagnostics: Mapping[str, object],
    gates: Mapping[str, object],
) -> list[str]:
    top = diagnostics.get("top_rows")
    if not isinstance(top, Mapping):
        raise ValueError("selected-action top-row diagnostics are missing")
    top100 = top.get("100")
    top500 = top.get("500")
    if not isinstance(top100, Mapping) or not isinstance(top500, Mapping):
        raise ValueError("selected-action acceptance rows are missing")
    reasons: list[str] = []
    if float(diagnostics["side_choice_auc"]) < float(gates["minimum_side_choice_auc"]):
        reasons.append("side_choice_auc_gate_failed")
    if float(diagnostics["selected_action_pearson_information_coefficient"]) <= float(
        gates["minimum_selected_action_pearson_ic"]
    ):
        reasons.append("selected_action_pearson_ic_gate_failed")
    if float(diagnostics["selected_action_spearman_information_coefficient"]) <= float(
        gates["minimum_selected_action_spearman_ic"]
    ):
        reasons.append("selected_action_spearman_ic_gate_failed")
    if float(top100["mean_stress_net_bps"]) <= float(
        gates["minimum_top_100_mean_stress_net_bps"]
    ):
        reasons.append("top_100_mean_stress_net_gate_failed")
    if float(top500["mean_stress_net_bps"]) <= float(
        gates["minimum_top_500_mean_stress_net_bps"]
    ):
        reasons.append("top_500_mean_stress_net_gate_failed")
    long_share = float(top500["long_share"])
    if long_share < float(gates["minimum_long_share_in_top_500"]) or long_share > float(
        gates["maximum_long_share_in_top_500"]
    ):
        reasons.append("top_500_side_balance_gate_failed")
    return reasons


def _subset_dataset(
    dataset: MicrostructureDataset,
    indexes: np.ndarray,
) -> MicrostructureDataset:
    selected = np.asarray(indexes, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size == 0
        or selected[0] < 0
        or selected[-1] >= dataset.rows
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("dataset subset indexes are invalid")
    updates: dict[str, object] = {}
    for field in fields(dataset):
        value = getattr(dataset, field.name)
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == dataset.rows
        ):
            updates[field.name] = np.asarray(value[selected]).copy()
    return replace(dataset, **updates)


def _prediction_max_abs_difference(
    left: SharedActionPredictionBatch,
    right: SharedActionPredictionBatch,
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
    for name in (
        "action_preference_side",
        "advantage_preference_side",
        "side_consensus",
    ):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return math.inf
    return max(differences, default=0.0)


def _equivariance_error(
    model: TrainedSharedActionLightGBMModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> float:
    subset = _subset_dataset(dataset, endpoints)
    local = np.arange(subset.rows, dtype=np.int64)
    original = predict_shared_action_lightgbm_model(model, subset, local)
    mirrored_dataset = replace(
        subset,
        features=mirror_microstructure_direction(subset.features),
    )
    mirrored = predict_shared_action_lightgbm_model(model, mirrored_dataset, local)
    errors = (
        np.max(np.abs(original.long_mean_bps - mirrored.short_mean_bps)),
        np.max(np.abs(original.short_mean_bps - mirrored.long_mean_bps)),
        np.max(
            np.abs(
                original.long_profitable_probability
                - mirrored.short_profitable_probability
            )
        ),
        np.max(
            np.abs(
                original.short_profitable_probability
                - mirrored.long_profitable_probability
            )
        ),
        np.max(np.abs(original.signed_advantage_bps + mirrored.signed_advantage_bps)),
    )
    if not np.array_equal(
        original.action_preference_side, -mirrored.action_preference_side
    ) or not np.array_equal(
        original.advantage_preference_side,
        -mirrored.advantage_preference_side,
    ):
        return math.inf
    return float(max(errors))


def _sample_indexes(indexes: np.ndarray, maximum_rows: int = 512) -> np.ndarray:
    selected = np.asarray(indexes, dtype=np.int64)
    count = min(int(maximum_rows), len(selected))
    positions = np.linspace(0, len(selected) - 1, count, dtype=np.int64)
    return selected[np.unique(positions)]


def _prediction_nonfinite_count(ensemble: SharedActionEnsembleBatch) -> int:
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
        ensemble.signed_advantage_mean_bps,
        ensemble.signed_advantage_epistemic_std_bps,
        ensemble.advantage_long_member_ratio,
        ensemble.advantage_short_member_ratio,
        ensemble.side_consensus_member_ratio,
    )
    return int(sum(np.sum(~np.isfinite(value)) for value in arrays))


def _predict_ensemble(
    models: Sequence[TrainedSharedActionLightGBMModel],
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    *,
    role: str,
    heartbeat_seconds: float,
    progress: Callable[..., None],
) -> SharedActionEnsembleBatch:
    members: list[SharedActionPredictionBatch] = []
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
                predict_shared_action_lightgbm_model(model, dataset, endpoints)
            )
    ensemble = ensemble_shared_action_predictions(members)
    nonfinite = _prediction_nonfinite_count(ensemble)
    if nonfinite:
        raise ValueError(f"{role} ensemble emitted {nonfinite} non-finite values")
    progress("prediction-complete", role=role, rows=ensemble.rows)
    return ensemble


def _trace_result(
    *,
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    score: ActionScoreBatch,
    threshold_bps: float,
    expected_days: Sequence[int],
    gates: Mapping[str, object],
) -> dict[str, object]:
    base = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="base",
        strength_threshold_bps=threshold_bps,
    )
    stress = simulate_barrier_action_trace(
        dataset,
        targets,
        score,
        scenario="stress",
        strength_threshold_bps=threshold_bps,
    )
    reasons = barrier_trace_gate_reasons(
        stress,
        expected_days=expected_days,
        gates=gates,
    )
    return {
        "evaluated": True,
        "eligible_rows": int(np.sum(score.eligible)),
        "base_trace": base.asdict(),
        "stress_trace": stress.asdict(),
        "status": "research_candidate" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _corpus_report(
    bundle: CorpusBundle, role_evidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "name": bundle.name,
        "dataset": bundle.dataset.summary(),
        "requested_start_ms": bundle.requested_start_ms,
        "requested_end_ms": bundle.requested_end_ms,
        "event_rows": len(bundle.event_indexes),
        "valid_barrier_rows": bundle.targets.valid_rows,
        "barrier_summary": bundle.targets.summary(),
        "barrier_targets_sha256": bundle.targets_sha256,
        "cache_key": bundle.cache_key,
        "cache_state": bundle.cache_state,
        "source_manifest_fingerprint": bundle.source_evidence.get(
            "manifest_fingerprint"
        ),
        "corpus_certificate_sha256": bundle.source_certificate.get(
            "certificate_sha256"
        ),
        "source_certificate_count": 1,
        "roles": dict(role_evidence),
    }


def run_shared_action_viability(
    *,
    design_path: str | Path,
    binding_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    design, design_sha, profiles = load_round32_design(design_path)
    binding, binding_sha = load_round32_execution_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
        require_current_implementation=True,
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Round 32 output directory must be empty")
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
                "round": 32,
                "design_sha256": design_sha,
                "binding_sha256": binding_sha,
                "sequence": progress_sequence,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "phase": phase,
                **details,
            }
            print(
                "round32 "
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
        thread_evidence = _configure_worker_threads(maximum_threads)
        threads = int(thread_evidence["effective_worker_threads"])
        progress("directml-preflight-start")
        directml = _attest_directml()
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
        roles = design["data"]["roles"]
        assert isinstance(roles, Mapping)
        training_first = _parse_date(
            roles["train"]["start"], label="training corpus start"
        )
        training_last = _parse_date(
            roles["development"]["end"], label="training corpus end"
        )
        training = _load_corpus(
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
        role_indexes, role_evidence, maximum_exit = _role_indexes(
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
        spec = _shared_action_spec(design)
        seeds = tuple(int(seed) for seed in design["model"]["seeds"])
        models: list[TrainedSharedActionLightGBMModel] = []
        model_evidence: list[dict[str, object]] = []
        reload_sample = _sample_indexes(role_indexes["calibration"])
        implementation_gates = design["acceptance_gates"]["implementation"]
        assert isinstance(implementation_gates, Mapping)
        maximum_equivariance = float(
            implementation_gates["maximum_action_swap_equivariance_error"]
        )
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
                model = train_shared_action_lightgbm_model(
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
                raise RuntimeError("model training did not remain on OpenCL")
            artifact_path = destination / "models" / f"seed-{seed}.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            progress("model-artifact-verification-start", member=member, seed=seed)
            with progress_heartbeat(
                progress,
                phase="model-artifact-verification-heartbeat",
                interval_seconds=heartbeat_seconds,
                details={"member": member, "seed": seed},
            ):
                save_shared_action_lightgbm_model(artifact_path, model)
                reloaded = load_shared_action_lightgbm_model(artifact_path)
                original_prediction = predict_shared_action_lightgbm_model(
                    model, training.dataset, reload_sample
                )
                reloaded_prediction = predict_shared_action_lightgbm_model(
                    reloaded, training.dataset, reload_sample
                )
                reload_error = _prediction_max_abs_difference(
                    original_prediction, reloaded_prediction
                )
                equivariance_error = _equivariance_error(
                    reloaded,
                    training.dataset,
                    reload_sample,
                )
            if reload_error != 0.0:
                raise ValueError("shared-action artifact reload changed predictions")
            if not math.isfinite(equivariance_error) or (
                equivariance_error > maximum_equivariance
            ):
                raise ValueError("shared-action directional equivariance gate failed")
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
                    "class_support": dict(reloaded.class_support),
                    "positive_class_prevalence": reloaded.positive_class_prevalence,
                    "probability_calibration": list(reloaded.probability_calibration),
                    "advantage_validation_directional_loss": (
                        reloaded.advantage_validation_directional_loss
                    ),
                    "best_iterations": dict(reloaded.best_iterations),
                    "artifact_reload_max_abs_prediction_error": reload_error,
                    "action_swap_equivariance_max_abs_error": equivariance_error,
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
                equivariance_error=equivariance_error,
            )

        calibration_prediction = _predict_ensemble(
            models,
            training.dataset,
            role_indexes["calibration"],
            role="calibration",
            heartbeat_seconds=heartbeat_seconds,
            progress=progress,
        )
        selection = design["selection"]
        assert isinstance(selection, Mapping)
        calibration_role = roles["calibration"]
        assert isinstance(calibration_role, Mapping)
        profile_results: dict[str, dict[str, object]] = {}
        calibration_survivors: list[str] = []
        for raw in profiles:
            profile = str(raw["profile"])
            score = derive_shared_action_scores(
                calibration_prediction, _profile_spec(raw)
            )
            threshold = select_barrier_threshold(
                training.dataset,
                training.targets,
                score,
                quantiles=tuple(
                    float(value) for value in selection["threshold_quantiles"]
                ),
                expected_days=_expected_days(calibration_role),
                gates=raw["calibration_gates"],
                drawdown_penalty=float(selection["drawdown_penalty"]),
            )
            if threshold.accepted:
                calibration_survivors.append(profile)
            profile_results[profile] = {
                "profile": profile,
                "policy_spec": asdict(_profile_spec(raw)),
                "calibration": {
                    "evaluated": True,
                    "eligible_rows": int(np.sum(score.eligible)),
                    "threshold_selection": threshold.asdict(),
                    "status": (
                        "research_candidate" if threshold.accepted else "rejected"
                    ),
                    "rejection_reasons": list(threshold.rejection_reasons),
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
                accepted=threshold.accepted,
                eligible_rows=int(np.sum(score.eligible)),
            )
        for profile in _PROFILE_NAMES:
            if profile not in calibration_survivors:
                profile_results[profile]["policy"] = {
                    "evaluated": False,
                    "withheld_reason": "calibration_rejected",
                }
        predictions: dict[str, SharedActionEnsembleBatch] = {
            "calibration": calibration_prediction
        }
        policy_survivors: list[str] = []
        profiles_by_name = {str(raw["profile"]): raw for raw in profiles}
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
            for profile in calibration_survivors:
                raw = profiles_by_name[profile]
                threshold_selection = profile_results[profile]["calibration"][
                    "threshold_selection"
                ]
                assert isinstance(threshold_selection, Mapping)
                threshold_bps = float(threshold_selection["threshold_bps"])
                score = derive_shared_action_scores(
                    policy_prediction, _profile_spec(raw)
                )
                result = _trace_result(
                    dataset=training.dataset,
                    targets=training.targets,
                    score=score,
                    threshold_bps=threshold_bps,
                    expected_days=_expected_days(policy_role),
                    gates=raw["policy_gates"],
                )
                profile_results[profile]["policy"] = result
                if result["status"] == "research_candidate":
                    policy_survivors.append(profile)
                progress(
                    "profile-policy-complete",
                    profile=profile,
                    status=result["status"],
                    stress_trades=result["stress_trace"]["metrics"]["trades"],
                )
        for profile in _PROFILE_NAMES:
            if profile not in policy_survivors:
                profile_results[profile]["development"] = {
                    "evaluated": False,
                    "withheld_reason": (
                        "policy_rejected"
                        if profile in calibration_survivors
                        else "calibration_rejected"
                    ),
                }
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
            development_role = roles["development"]
            assert isinstance(development_role, Mapping)
            for profile in policy_survivors:
                raw = profiles_by_name[profile]
                threshold_selection = profile_results[profile]["calibration"][
                    "threshold_selection"
                ]
                assert isinstance(threshold_selection, Mapping)
                score = derive_shared_action_scores(
                    development_prediction, _profile_spec(raw)
                )
                result = _trace_result(
                    dataset=training.dataset,
                    targets=training.targets,
                    score=score,
                    threshold_bps=float(threshold_selection["threshold_bps"]),
                    expected_days=_expected_days(development_role),
                    gates=raw["development_gates"],
                )
                profile_results[profile]["development"] = result
                if result["status"] == "research_candidate":
                    development_survivors.append(profile)
                progress(
                    "profile-development-complete",
                    profile=profile,
                    status=result["status"],
                    stress_trades=result["stress_trace"]["metrics"]["trades"],
                )
        for profile in _PROFILE_NAMES:
            if profile not in development_survivors:
                profile_results[profile]["distant_confirmation"] = {
                    "evaluated": False,
                    "withheld_reason": (
                        "development_rejected"
                        if profile in policy_survivors
                        else profile_results[profile]["development"]["withheld_reason"]
                    ),
                }

        distant_bundle: CorpusBundle | None = None
        distant_role_evidence: dict[str, object] = {}
        distant_diagnostics: dict[str, object] | None = None
        distant_raw_forecast: dict[str, object] | None = None
        forecast_reasons: list[str] = ["withheld_no_development_survivors"]
        distant_trace_survivors: list[str] = []
        if development_survivors:
            distant_role = roles["distant_confirmation"]
            assert isinstance(distant_role, Mapping)
            distant_bundle = _load_corpus(
                name="distant_confirmation",
                design=design,
                warehouse_path=warehouse_path,
                cache_root=cache_root,
                first=_parse_date(
                    distant_role["context_start"], label="distant context"
                ),
                last=_parse_date(distant_role["end"], label="distant end"),
                evaluation_first=_parse_date(
                    distant_role["start"], label="distant start"
                ),
                evaluation_last=_parse_date(distant_role["end"], label="distant end"),
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
            distant_indexes, distant_role_evidence, _distant_exit = _role_indexes(
                distant_bundle,
                distant_roles,
                ("distant_confirmation",),
            )
            distant_prediction = _predict_ensemble(
                models,
                distant_bundle.dataset,
                distant_indexes["distant_confirmation"],
                role="distant_confirmation",
                heartbeat_seconds=heartbeat_seconds,
                progress=progress,
            )
            distant_diagnostics = _selected_action_diagnostics(
                distant_bundle.targets, distant_prediction
            )
            distant_raw_forecast = _raw_forecast_diagnostics(
                distant_bundle.targets,
                distant_prediction,
                scenario="stress",
            )
            forecast_gates = design["acceptance_gates"]["distant_confirmation_forecast"]
            assert isinstance(forecast_gates, Mapping)
            forecast_reasons = _forecast_gate_reasons(
                distant_diagnostics, forecast_gates
            )
            role_for_days = distant_roles["distant_confirmation"]
            for profile in development_survivors:
                raw = profiles_by_name[profile]
                threshold_selection = profile_results[profile]["calibration"][
                    "threshold_selection"
                ]
                assert isinstance(threshold_selection, Mapping)
                score = derive_shared_action_scores(
                    distant_prediction, _profile_spec(raw)
                )
                result = _trace_result(
                    dataset=distant_bundle.dataset,
                    targets=distant_bundle.targets,
                    score=score,
                    threshold_bps=float(threshold_selection["threshold_bps"]),
                    expected_days=_expected_days(role_for_days),
                    gates=raw["development_gates"],
                )
                profile_results[profile]["distant_confirmation"] = result
                if result["status"] == "research_candidate":
                    distant_trace_survivors.append(profile)
                progress(
                    "profile-distant-complete",
                    profile=profile,
                    status=result["status"],
                    stress_trades=result["stress_trace"]["metrics"]["trades"],
                )

        final_profiles = distant_trace_survivors if not forecast_reasons else []
        for profile in _PROFILE_NAMES:
            profile_results[profile]["final_status"] = (
                "consumed_data_viability_candidate"
                if profile in final_profiles
                else "rejected"
            )
        report_status = "research_candidate" if final_profiles else "rejected"
        forecast_by_role = {
            role: {
                "stress": _raw_forecast_diagnostics(
                    training.targets, prediction, scenario="stress"
                ),
                "selected_action": _selected_action_diagnostics(
                    training.targets, prediction
                ),
            }
            for role, prediction in predictions.items()
        }
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_class": "consumed_data_shared_action_value_viability_evidence",
            "round": 32,
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
                "model": SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION,
                "policy": SHARED_ACTION_POLICY_SCHEMA_VERSION,
                "base_policy": ACTION_POLICY_SCHEMA_VERSION,
                "barriers": ADAPTIVE_BARRIER_SCHEMA_VERSION,
            },
            "training_corpus": _corpus_report(training, role_evidence),
            "distant_corpus": (
                _corpus_report(distant_bundle, distant_role_evidence)
                if distant_bundle is not None
                else None
            ),
            "stage_access": {
                "calibration_prediction": True,
                "policy_prediction": "policy" in predictions,
                "development_prediction": "development" in predictions,
                "distant_confirmation_prediction": distant_bundle is not None,
                "later_stage_predictions_withheld_until_prior_stage_passes": True,
            },
            "sample_weighting": {
                "method": "average_label_uniqueness",
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
                "maximum_observed_action_swap_equivariance_error": max(
                    float(value["action_swap_equivariance_max_abs_error"])
                    for value in model_evidence
                ),
                "maximum_permitted_action_swap_equivariance_error": maximum_equivariance,
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
                "gates": design["acceptance_gates"]["distant_confirmation_forecast"],
            },
            "profile_results": [profile_results[profile] for profile in _PROFILE_NAMES],
            "stage_survivors": {
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
        description="Run the frozen Round 32 shared-action viability study."
    )
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        default=str(research / "round-032-shared-action-value-viability-design.json"),
    )
    parser.add_argument(
        "--binding",
        default=str(research / "round-032-execution-binding.json"),
    )
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_shared_action_viability(
        design_path=arguments.design,
        binding_path=arguments.binding,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_dir=arguments.output_dir,
    )
    print(
        f"Round 32 completed with status={report['status']} "
        f"report_sha256={report['report_canonical_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())

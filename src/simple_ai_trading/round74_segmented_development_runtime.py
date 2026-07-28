"""Guarded training and optional local-AI qualification for Round 74."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import gc
import hashlib
import json
from pathlib import Path

from .compute import BackendInfo, resolve_backend, torch_device_for_backend
from .impact_absorption_ai_execution_replay import (
    Round74AIQualificationStoreExecutionReplayProvider,
)
from .impact_absorption_event_action_policy import ROUND74_ACTION_PROFILES
from .impact_absorption_ai_uplift import (
    ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
)
from .impact_absorption_store import (
    ImpactAbsorptionStore,
    validate_impact_store_resources,
)
from .progress_heartbeat import progress_heartbeat
from .round74_active_qualification import (
    _database_and_wal_bytes,
)
from .round74_ai_qualification_operator import (
    run_round74_ai_pretest_qualification,
)
from .round74_segmented_campaign_runner import (
    _active_segmented_capture_processes,
)
from .round74_segmented_development_inputs import (
    Round74SegmentedDevelopmentInputs,
    load_round74_segmented_development_inputs,
)
from .round74_segmented_development_operator import (
    Round74SegmentedQualifiedDevelopment,
    prepare_round74_segmented_development,
    train_round74_segmented_development_policy,
)
from .round74_segmented_model_operator import (
    build_round74_segmented_ai_qualification_population,
)


ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION = (
    "round-074-segmented-development-run-v2"
)
ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_RELATIVE_PATH = Path(
    "data/round74-segmented-event-cohort-v3.duckdb"
)

ProgressCallback = Callable[..., None]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _preflight_backend(requested: str) -> BackendInfo:
    backend = resolve_backend(requested, require=True)
    torch_device_for_backend(backend)
    return backend


def _validate_path_panel(
    *,
    repository: Path,
    database_path: Path,
    plan_path: Path,
    state_root: Path,
    recovery_outcome_directory: Path,
    target_assembly_directory: Path,
    source_artifact_root: Path,
    model_output_directory: Path,
    qualification_output_directory: Path,
) -> tuple[Path, ...]:
    raw_paths = (
        repository,
        database_path,
        plan_path,
        state_root,
        recovery_outcome_directory,
        target_assembly_directory,
        source_artifact_root,
        model_output_directory,
        qualification_output_directory,
    )
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("Round 74 segmented development path symlinks are forbidden")
    root = repository.resolve()
    database = database_path.resolve()
    plan = plan_path.resolve()
    state = state_root.resolve()
    recovery = recovery_outcome_directory.resolve()
    assemblies = target_assembly_directory.resolve()
    sources = source_artifact_root.resolve()
    model_output = model_output_directory.resolve()
    qualification_output = qualification_output_directory.resolve()
    if (
        not root.is_dir()
        or not database.is_file()
        or database.stat().st_size <= 0
        or not plan.is_file()
        or not state.is_dir()
        or not recovery.is_dir()
        or not assemblies.is_dir()
        or not sources.is_dir()
        or (model_output.exists() and not model_output.is_dir())
        or (qualification_output.exists() and not qualification_output.is_dir())
        or model_output == qualification_output
        or model_output in qualification_output.parents
        or qualification_output in model_output.parents
    ):
        raise ValueError("Round 74 segmented development path panel differs")
    return (
        database,
        plan,
        state,
        recovery,
        assemblies,
        sources,
        model_output,
        qualification_output,
    )


def _guard_idle_database(database: Path, *, repeated: bool) -> int:
    if _active_segmented_capture_processes():
        state = "became active" if repeated else "is active"
        raise RuntimeError(f"Round 74 segmented development blocked: capture {state}")
    database_bytes, wal_bytes = _database_and_wal_bytes(database)
    if database_bytes <= 0:
        raise RuntimeError("Round 74 segmented development database disappeared")
    if wal_bytes != 0:
        state = "appeared" if repeated else "exists"
        raise RuntimeError(
            f"Round 74 segmented development blocked: database WAL {state}"
        )
    return database_bytes


def _run_development(
    store: ImpactAbsorptionStore,
    inputs: Round74SegmentedDevelopmentInputs,
    *,
    model_output_directory: Path,
    qualification_output_directory: Path,
    compute_backend: str,
    inference_minibatch_rows: int,
    enable_ai: bool,
    progress: ProgressCallback,
) -> tuple[object, Round74SegmentedQualifiedDevelopment | None, str]:
    partition = inputs.coverage.partition
    assemblies = inputs.target_assembly_by_run_id()
    bindings = inputs.development_bindings_by_run_id()
    progress("segmented_preparation_started")
    preparation = prepare_round74_segmented_development(
        store,
        partition=partition,
        bindings_by_run_id=bindings,
        target_assembly_by_run_id=assemblies,
    )
    preparation_sha256 = preparation.preparation_sha256
    progress(
        "segmented_preparation_completed",
        preparation_sha256=preparation_sha256,
        feature_scaler_sha256=preparation.prepared.scaler.scaler_sha256,
        training_batch_count=len(preparation.prepared.training_batches),
        tuning_batch_count=len(preparation.prepared.tuning_batches),
    )
    progress("model_training_started")
    policy = train_round74_segmented_development_policy(
        preparation,
        store=store,
        partition=partition,
        target_assembly_by_run_id=assemblies,
        output_directory=model_output_directory,
        compute_backend=compute_backend,
        inference_minibatch_rows=inference_minibatch_rows,
    )
    progress(
        "model_training_completed",
        bundle_sha256=policy.bundle_sha256,
        pretest_policy_sha256=policy.pretest_policy.policy_sha256,
    )
    if not enable_ai:
        return policy, None, preparation_sha256
    population = build_round74_segmented_ai_qualification_population(
        preparation.tuning_subpartition
    )
    qualification_batches = tuple(preparation.tuning_roles.ai_qualification_batches)
    qualification_assemblies = {
        run_id: assemblies[run_id] for run_id in population.run_ids
    }
    action_policies = {
        selected.profile: selected for selected in policy.bundle.action_policies
    }
    accepted_profiles = tuple(
        profile
        for profile in ROUND74_ACTION_PROFILES
        if action_policies[profile].accepted
    )
    replay_provider = Round74AIQualificationStoreExecutionReplayProvider(
        store=store,
        partition=partition,
        qualification_population=population,
        assembly_by_run_id=qualification_assemblies,
    )
    del preparation
    del assemblies
    del bindings
    gc.collect()
    progress(
        "training_batches_released",
        retained_ai_qualification_batch_count=len(qualification_batches),
        accepted_profiles=list(accepted_profiles),
    )

    results = []
    for profile in accepted_profiles:
        progress("ai_qualification_started", profile=profile)

        def ai_progress(
            payload: Mapping[str, object],
            *,
            selected_profile: str = profile,
        ) -> None:
            progress(
                "ai_qualification_progress",
                profile=selected_profile,
                detail=dict(payload),
            )

        result = run_round74_ai_pretest_qualification(
            qualification_batches,
            qualification_population=population,
            action_selection=action_policies[profile],
            probability_calibration=policy.bundle.probability_calibration,
            pretest_policy_path=policy.pretest_policy.policy_path,
            execution_replay_provider=replay_provider,
            qualification_output_path=(
                qualification_output_directory
                / f"round74-ai-pretest-qualification-{profile}.json"
            ),
            compute_backend=compute_backend,
            inference_minibatch_rows=inference_minibatch_rows,
            progress_callback=ai_progress,
        )
        results.append((profile, result))
        progress(
            "ai_qualification_completed",
            profile=profile,
            qualification_sha256=result.qualification.qualification_sha256,
            qualification_passed=result.qualification.qualification_passed,
        )
    qualified = Round74SegmentedQualifiedDevelopment(
        preparation_sha256=preparation_sha256,
        policy=policy,
        requested_profiles=ROUND74_ACTION_PROFILES,
        qualification_by_profile=tuple(results),
    )
    qualified.validate()
    return policy, qualified, preparation_sha256


def run_round74_segmented_development(
    *,
    repository: Path,
    database_path: Path,
    plan_path: Path,
    state_root: Path,
    recovery_outcome_directory: Path,
    target_assembly_directory: Path,
    source_artifact_root: Path,
    model_output_directory: Path,
    qualification_output_directory: Path,
    terminal_observed_wall_ns: int,
    progress: ProgressCallback,
    compute_backend: str = "auto",
    memory_limit: str = "4GB",
    database_threads: int = 2,
    inference_minibatch_rows: int = 128,
    progress_interval_seconds: float = 30.0,
    enable_ai: bool = True,
) -> dict[str, object]:
    """Validate everything before opening the evidence store read-only."""

    if not callable(progress):
        raise TypeError("Round 74 segmented development progress differs")
    (
        database,
        plan,
        state,
        recovery,
        assemblies,
        sources,
        model_output,
        qualification_output,
    ) = _validate_path_panel(
        repository=Path(repository),
        database_path=Path(database_path),
        plan_path=Path(plan_path),
        state_root=Path(state_root),
        recovery_outcome_directory=Path(recovery_outcome_directory),
        target_assembly_directory=Path(target_assembly_directory),
        source_artifact_root=Path(source_artifact_root),
        model_output_directory=Path(model_output_directory),
        qualification_output_directory=Path(qualification_output_directory),
    )
    normalized_memory, normalized_threads = validate_impact_store_resources(
        memory_limit,
        database_threads,
    )
    if (
        isinstance(inference_minibatch_rows, bool)
        or not isinstance(inference_minibatch_rows, int)
        or not 1 <= inference_minibatch_rows <= 65_536
        or isinstance(progress_interval_seconds, bool)
        or not 1.0 <= float(progress_interval_seconds) <= 300.0
        or not isinstance(enable_ai, bool)
    ):
        raise ValueError("Round 74 segmented development runtime policy differs")

    initial_database_bytes = _guard_idle_database(database, repeated=False)
    progress("input_validation_started", database_bytes=initial_database_bytes)
    inputs = load_round74_segmented_development_inputs(
        plan_path=plan,
        state_root=state,
        recovery_outcome_directory=recovery,
        target_assembly_directory=assemblies,
        source_artifact_root=sources,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
    )
    progress(
        "input_validation_completed",
        inputs_sha256=inputs.inputs_sha256,
        plan_sha256=inputs.plan.plan_sha256,
        coverage_sha256=inputs.coverage.coverage_sha256,
        partition_sha256=inputs.coverage.partition.partition_sha256,
        development_target_assembly_count=len(inputs.target_assemblies),
    )
    backend = _preflight_backend(str(compute_backend))
    progress(
        "backend_ready",
        requested=backend.requested,
        kind=backend.kind,
        device=backend.device,
        vendor=backend.vendor,
        selection=backend.selection,
        accelerated=backend.accelerated,
    )
    database_bytes = _guard_idle_database(database, repeated=True)
    with progress_heartbeat(
        progress,
        phase="segmented_development",
        interval_seconds=float(progress_interval_seconds),
    ):
        with ImpactAbsorptionStore(
            database,
            memory_limit=normalized_memory,
            threads=normalized_threads,
            read_only=True,
        ) as store:
            policy, qualified, preparation_sha256 = _run_development(
                store,
                inputs,
                model_output_directory=model_output,
                qualification_output_directory=qualification_output,
                compute_backend=str(compute_backend),
                inference_minibatch_rows=inference_minibatch_rows,
                enable_ai=enable_ai,
                progress=progress,
            )

    qualification_by_profile = (
        {} if qualified is None else dict(qualified.qualification_by_profile)
    )
    profiles = []
    for selected in policy.bundle.action_policies:
        qualification = qualification_by_profile.get(selected.profile)
        profiles.append(
            {
                "profile": selected.profile,
                "ml_action_policy_accepted": selected.accepted,
                "ml_action_policy_rejection_reasons": list(selected.rejection_reasons),
                "ai_qualification_executed": qualification is not None,
                "ai_qualification_passed": (
                    False
                    if qualification is None
                    else qualification.qualification.qualification_passed
                ),
                "development_gate_passed": (
                    selected.accepted
                    and (
                        not enable_ai
                        or qualification is not None
                        and qualification.qualification.qualification_passed
                    )
                ),
            }
        )
    result: dict[str, object] = {
        "schema_version": ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION,
        "inputs_sha256": inputs.inputs_sha256,
        "plan_sha256": inputs.plan.plan_sha256,
        "coverage_sha256": inputs.coverage.coverage_sha256,
        "partition_sha256": inputs.coverage.partition.partition_sha256,
        "preparation_sha256": preparation_sha256,
        "database_bytes": database_bytes,
        "database_open_mode": "read_only",
        "resources": {
            "memory_limit": normalized_memory,
            "database_threads": normalized_threads,
            "inference_minibatch_rows": inference_minibatch_rows,
            "training_batches_released_before_ai": enable_ai,
        },
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": backend.device,
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
        },
        "model_artifact": {
            "bundle_sha256": policy.bundle_sha256,
            "bundle_path": str(policy.bundle_path),
            "pretest_policy_sha256": policy.pretest_policy.policy_sha256,
            "pretest_policy_path": str(policy.pretest_policy.policy_path),
            "model_sha256": policy.pretest_policy.model_sha256,
            "model_path": str(policy.pretest_policy.model_path),
            "selected_candidate_id": (policy.pretest_policy.selected_candidate_id),
            "segmented_tuning_proper_loss": policy.pretest_policy.tuning_loss,
        },
        "ai": {
            "enabled": enable_ai,
            "action_validity_maximum_ns": (ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS),
            "action_validity_policy": (
                "minimum_of_forecast_horizon_and_target_maximum_delayed_entry"
            ),
            "action_latency_includes_historical_queue_delay": True,
            "accepted_actions_use_exact_delayed_l2_replay": True,
            "qualified_development_sha256": (
                None if qualified is None else qualified.result_sha256
            ),
        },
        "profiles": profiles,
        "authority": {
            "sealed_test_accessed": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    progress(
        "segmented_development_completed",
        result_sha256=result["result_sha256"],
    )
    return result


__all__ = [
    "ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_RELATIVE_PATH",
    "ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION",
    "run_round74_segmented_development",
]

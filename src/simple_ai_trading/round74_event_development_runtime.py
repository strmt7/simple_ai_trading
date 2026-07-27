"""Guarded runtime composition for Round 74 development-only training."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

from .compute import BackendInfo, resolve_backend, torch_device_for_backend
from .impact_absorption_store import (
    ImpactAbsorptionStore,
    validate_impact_store_resources,
)
from .progress_heartbeat import progress_heartbeat
from .round74_active_qualification import (
    _active_capture_processes,
    _database_and_wal_bytes,
)
from .round74_event_cohort_operator import (
    ROUND74_EVENT_COHORT_PLAN_RELATIVE_PATH,
    ROUND74_EVENT_COHORT_PLAN_SHA256,
)
from .round74_event_development_inputs import load_round74_development_inputs


ROUND74_DEVELOPMENT_RUN_SCHEMA_VERSION = "round-074-development-run-v1"
ROUND74_DEVELOPMENT_DATABASE_RELATIVE_PATH = Path("data/microstructure.duckdb")
ROUND74_DEVELOPMENT_CAMPAIGN_RELATIVE_PATH = (
    Path("data/round74-event-cohort") / ROUND74_EVENT_COHORT_PLAN_SHA256
)

ProgressCallback = Callable[..., None]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _preflight_backend(requested: str) -> BackendInfo:
    backend = resolve_backend(requested, require=True)
    torch_device_for_backend(backend)
    return backend


def _train_development(
    store: object,
    inputs: object,
    *,
    output_directory: Path,
    compute_backend: str,
    inference_minibatch_rows: int,
) -> object:
    from .round74_event_development_operator import (
        train_round74_development_policy_from_inputs,
    )

    return train_round74_development_policy_from_inputs(
        store,
        inputs,
        output_directory=output_directory,
        compute_backend=compute_backend,
        inference_minibatch_rows=inference_minibatch_rows,
    )


def _validate_path_panel(
    *,
    repository: Path,
    database_path: Path,
    plan_path: Path,
    binding_directory: Path,
    target_assembly_directory: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    raw_paths = (
        repository,
        database_path,
        plan_path,
        binding_directory,
        target_assembly_directory,
        output_directory,
    )
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("Round 74 development path symlinks are forbidden")
    root = repository.resolve()
    database = database_path.resolve()
    plan = plan_path.resolve()
    bindings = binding_directory.resolve()
    assemblies = target_assembly_directory.resolve()
    output = output_directory.resolve()
    if (
        not root.is_dir()
        or not database.is_file()
        or database.stat().st_size <= 0
        or not plan.is_file()
        or not bindings.is_dir()
        or not assemblies.is_dir()
        or (output.exists() and not output.is_dir())
    ):
        raise ValueError("Round 74 development path panel differs")
    return database, plan, bindings, assemblies, output


def _guard_idle_database(database: Path, *, repeated: bool) -> int:
    processes = _active_capture_processes()
    if processes:
        state = "became active" if repeated else "is active"
        raise RuntimeError(f"Round 74 development blocked: capture {state}")
    database_bytes, wal_bytes = _database_and_wal_bytes(database)
    if database_bytes <= 0:
        raise RuntimeError("Round 74 development database disappeared")
    if wal_bytes != 0:
        state = "appeared" if repeated else "exists"
        raise RuntimeError(f"Round 74 development blocked: database WAL {state}")
    return database_bytes


def run_round74_event_development(
    *,
    repository: Path,
    database_path: Path,
    target_assembly_directory: Path,
    output_directory: Path,
    progress: ProgressCallback,
    plan_path: Path | None = None,
    binding_directory: Path | None = None,
    compute_backend: str = "auto",
    memory_limit: str = "4GB",
    database_threads: int = 2,
    inference_minibatch_rows: int = 128,
    progress_interval_seconds: float = 30.0,
) -> dict[str, object]:
    """Validate the full panel before opening the read-only evidence store."""

    if not callable(progress):
        raise TypeError("Round 74 development progress callback differs")
    selected_repository = Path(repository)
    selected_plan = (
        selected_repository / ROUND74_EVENT_COHORT_PLAN_RELATIVE_PATH
        if plan_path is None
        else Path(plan_path)
    )
    selected_bindings = (
        selected_repository / ROUND74_DEVELOPMENT_CAMPAIGN_RELATIVE_PATH / "bindings"
        if binding_directory is None
        else Path(binding_directory)
    )
    database, plan, bindings, assemblies, output = _validate_path_panel(
        repository=selected_repository,
        database_path=Path(database_path),
        plan_path=selected_plan,
        binding_directory=selected_bindings,
        target_assembly_directory=Path(target_assembly_directory),
        output_directory=Path(output_directory),
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
    ):
        raise ValueError("Round 74 development runtime policy differs")

    initial_database_bytes = _guard_idle_database(database, repeated=False)
    progress("input_validation_started", database_bytes=initial_database_bytes)
    inputs = load_round74_development_inputs(
        plan_path=plan,
        binding_directory=bindings,
        target_assembly_directory=assemblies,
    )
    progress(
        "input_validation_completed",
        inputs_sha256=inputs.inputs_sha256,
        plan_sha256=inputs.plan.plan_sha256,
        partition_sha256=inputs.partition.partition_sha256,
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
    progress("development_started")
    with progress_heartbeat(
        progress,
        phase="development",
        interval_seconds=float(progress_interval_seconds),
    ):
        with ImpactAbsorptionStore(
            database,
            memory_limit=normalized_memory,
            threads=normalized_threads,
            read_only=True,
        ) as store:
            artifact = _train_development(
                store,
                inputs,
                output_directory=output,
                compute_backend=str(compute_backend),
                inference_minibatch_rows=inference_minibatch_rows,
            )
    progress("development_completed")

    result: dict[str, object] = {
        "schema_version": ROUND74_DEVELOPMENT_RUN_SCHEMA_VERSION,
        "inputs_sha256": inputs.inputs_sha256,
        "plan_sha256": inputs.plan.plan_sha256,
        "partition_sha256": inputs.partition.partition_sha256,
        "development_target_assembly_count": len(inputs.target_assemblies),
        "database_bytes": database_bytes,
        "database_open_mode": "read_only",
        "resources": {
            "memory_limit": normalized_memory,
            "database_threads": normalized_threads,
            "inference_minibatch_rows": inference_minibatch_rows,
        },
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": backend.device,
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
        },
        "artifact": {
            "bundle_sha256": artifact.bundle_sha256,
            "bundle_path": str(artifact.bundle_path),
            "pretest_policy_sha256": artifact.pretest_policy.policy_sha256,
            "pretest_policy_path": str(artifact.pretest_policy.policy_path),
            "model_sha256": artifact.pretest_policy.model_sha256,
            "model_path": str(artifact.pretest_policy.model_path),
            "selected_candidate_id": artifact.pretest_policy.selected_candidate_id,
            "run_balanced_tuning_proper_loss": artifact.pretest_policy.tuning_loss,
        },
        "profiles": [
            {
                "profile": policy.profile,
                "accepted": policy.accepted,
                "selected_quantile": policy.selected_quantile,
                "selected_threshold_score": policy.selected_threshold_score,
                "rejection_reasons": list(policy.rejection_reasons),
            }
            for policy in artifact.bundle.action_policies
        ],
        "authority": {
            "sealed_test_accessed": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "ROUND74_DEVELOPMENT_CAMPAIGN_RELATIVE_PATH",
    "ROUND74_DEVELOPMENT_DATABASE_RELATIVE_PATH",
    "ROUND74_DEVELOPMENT_RUN_SCHEMA_VERSION",
    "run_round74_event_development",
]

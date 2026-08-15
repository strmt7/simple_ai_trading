"""Guarded training and optional local-AI qualification for Round 74."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import gc
import hashlib
from itertools import groupby
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .compute import BackendInfo, resolve_backend, torch_device_for_backend
from .impact_absorption_ai_execution_replay import (
    Round74AIExecutionReplayEvidence,
    Round74AIExecutionReplayInstruction,
    replay_round74_ai_execution_store_run,
)
from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionExecutionPanel,
)
from .impact_absorption_event_action_configuration import (
    select_round74_final_action_configuration,
)
from .impact_absorption_ai_uplift import (
    ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
    Round74AIQualificationPopulation,
)
from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventTrainingBatch,
    build_round74_event_training_batch,
)
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    Round74EventFeatureScaler,
    fit_round74_event_feature_scaler_stream,
)
from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
    iter_round74_v10_segment_event_observations,
)
from .impact_absorption_event_training import Round74EventTrainingConfig
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
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
from .round74_delayed_execution_panel import (
    build_round74_delayed_execution_panels,
)
from .round74_device_group_preflight import (
    run_round74_device_group_preflight_subprocess,
    write_round74_device_group_preflight,
)
from .round74_event_model_operator import (
    Round74PreparedDevelopmentData,
    split_round74_prepared_tuning_roles,
)
from .round74_segmented_campaign_runner import (
    _active_segmented_capture_processes,
)
from .round74_segmented_development_inputs import (
    Round74SegmentedDevelopmentInputs,
    load_round74_segmented_development_inputs,
    load_round74_segmented_terminal_coverage,
)
from .round74_segmented_development_operator import (
    Round74SegmentedPreparedDevelopment,
    Round74SegmentedQualifiedDevelopment,
    train_round74_segmented_development_policy,
)
from .round74_segmented_model_operator import (
    ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
    build_round74_segmented_ai_qualification_population,
    build_round74_segmented_training_split,
    build_round74_segmented_tuning_subpartition,
    iter_round74_segmented_labeled_event_windows,
    round74_segmented_windows_per_symbol,
    select_round74_segmented_event_windows,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly


ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION = (
    "round-074-segmented-development-run-v6"
)
ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_ROUTE_SCHEMA_VERSION = (
    "round-074-segmented-development-database-route-v1"
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


def _device_group_training_config(
    *,
    policy: str,
    fixed_group_size: int,
    preflight: Mapping[str, object] | None,
) -> Round74EventTrainingConfig:
    base = replace(
        Round74EventTrainingConfig(),
        execution_mode="segmented_cohort",
    )
    selected_policy = str(policy)
    if selected_policy == "fixed":
        if preflight is not None:
            raise ValueError("Round 74 fixed device group received preflight evidence")
        selected = replace(base, device_run_group_size=int(fixed_group_size))
    elif selected_policy == "auto":
        if preflight is None:
            raise ValueError("Round 74 automatic device group preflight is missing")
        raw_groups = preflight.get("selected_group_sizes")
        digest = preflight.get("preflight_sha256")
        if not isinstance(raw_groups, Mapping) or not isinstance(digest, str):
            raise ValueError("Round 74 automatic device group preflight differs")
        selected = replace(
            base,
            candidate_device_run_group_sizes=tuple(
                (candidate_id, int(raw_groups[candidate_id]))
                for candidate_id in base.candidate_ids
            ),
            device_group_selection_mode="target_free_host_benchmark",
            device_group_preflight_sha256=digest,
        )
    else:
        raise ValueError("Round 74 device group selection policy differs")
    selected.validate()
    return selected


def _normalize_database_paths(
    value: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        selected = (Path(value),)
    elif isinstance(value, Sequence):
        selected = tuple(Path(item) for item in value)
    else:
        raise TypeError("Round 74 segmented development database panel type differs")
    if not selected:
        raise ValueError("Round 74 segmented development database panel is empty")
    return selected


def _validate_path_panel(
    *,
    repository: Path,
    database_paths: str | Path | Sequence[str | Path],
    plan_path: Path,
    state_root: Path,
    recovery_outcome_directory: Path,
    target_assembly_directory: Path,
    source_artifact_root: Path,
    model_output_directory: Path,
    qualification_output_directory: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    raw_databases = _normalize_database_paths(database_paths)
    raw_paths = (
        repository,
        *raw_databases,
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
    databases = tuple(path.resolve() for path in raw_databases)
    resolved = tuple(path.resolve() for path in raw_paths[1 + len(databases) :])
    (
        plan,
        state,
        recovery,
        assemblies,
        sources,
        model_output,
        qualification_output,
    ) = resolved
    if (
        not root.is_dir()
        or len(set(databases)) != len(databases)
        or any(not database.is_file() for database in databases)
        or any(database.stat().st_size <= 0 for database in databases)
        or any(database.parent.is_symlink() for database in databases)
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
    return databases, resolved


def _guard_idle_databases(
    databases: Sequence[Path],
    *,
    repeated: bool,
) -> tuple[int, ...]:
    if _active_segmented_capture_processes():
        state = "became active" if repeated else "is active"
        raise RuntimeError(f"Round 74 segmented development blocked: capture {state}")
    result: list[int] = []
    for database in databases:
        database_bytes, wal_bytes = _database_and_wal_bytes(database)
        if database_bytes <= 0:
            raise RuntimeError("Round 74 segmented development database disappeared")
        if wal_bytes != 0:
            state = "appeared" if repeated else "exists"
            raise RuntimeError(
                f"Round 74 segmented development blocked: database WAL {state}"
            )
        result.append(database_bytes)
    return tuple(result)


def _guard_idle_database(database: Path, *, repeated: bool) -> int:
    """Retain the one-store guard used by source-artifact builders."""

    return _guard_idle_databases((Path(database),), repeated=repeated)[0]


def _consecutive_database_groups(
    run_ids: Sequence[str],
    database_by_run_id: Mapping[str, Path],
) -> Iterator[tuple[Path, tuple[str, ...]]]:
    routes = dict(database_by_run_id)
    for database, grouped in groupby(run_ids, key=routes.__getitem__):
        yield database, tuple(grouped)


def _route_development_run_databases(
    databases: Sequence[Path],
    *,
    development_run_ids: tuple[str, ...],
    memory_limit: str,
    database_threads: int,
) -> tuple[dict[str, Path], str, tuple[Path, ...]]:
    expected = tuple(development_run_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Round 74 segmented development run panel differs")
    expected_set = set(expected)
    database_groups: list[tuple[str, ...]] = []
    database_by_run_id: dict[str, Path] = {}
    for database in databases:
        _guard_idle_databases((database,), repeated=True)
        with ImpactAbsorptionStore(
            database,
            memory_limit=memory_limit,
            threads=database_threads,
            read_only=True,
        ) as store:
            rows = (
                store.connect()
                .execute("SELECT run_id FROM impact_capture_run ORDER BY run_id")
                .fetchall()
            )
        observed = tuple(str(row[0]) for row in rows)
        if len(set(observed)) != len(observed):
            raise ValueError(
                "Round 74 segmented development database run identity is duplicated"
            )
        relevant = tuple(run_id for run_id in expected if run_id in set(observed))
        database_groups.append(relevant)
        for run_id in relevant:
            if run_id in database_by_run_id:
                raise ValueError(
                    "Round 74 segmented development run exists in multiple stores"
                )
            database_by_run_id[run_id] = database
    if set(database_by_run_id) != expected_set:
        raise ValueError("Round 74 segmented development database coverage differs")
    nonempty_groups = sorted(
        (list(group) for group in database_groups if group),
        key=lambda group: tuple(group),
    )
    route_payload = {
        "schema_version": (ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_ROUTE_SCHEMA_VERSION),
        "development_run_ids": list(expected),
        "database_run_groups": nonempty_groups,
        "supplied_database_count": len(databases),
        "used_database_count": len(nonempty_groups),
        "unused_database_count": sum(not group for group in database_groups),
        "absolute_paths_bound": False,
        "target_manifests_read_before_route": False,
        "capture_content_bound_by_fresh_run_audits": True,
    }
    ordered_routes = {run_id: database_by_run_id[run_id] for run_id in expected}
    used_databases = tuple(dict.fromkeys(ordered_routes.values()))
    return ordered_routes, _canonical_sha256(route_payload), used_databases


def _validate_binding_entry(
    binding: Round74SegmentedCohortRunBinding,
    partition: Round74EventRunPartition,
    *,
    run_id: str,
) -> None:
    binding.validate()
    entry = partition.entry(run_id)
    entry.validate()
    if (
        binding.run_id != entry.run_id
        or binding.role != entry.role
        or binding.report_sha256 != entry.capture_report_sha256
        or binding.feature_ready_wall_ns != entry.capture_start_wall_ns
        or binding.usable_end_wall_ns != entry.capture_end_wall_ns
        or not (
            binding.feature_ready_wall_ns
            <= entry.eligible_anchor_start_wall_ns
            <= entry.eligible_anchor_end_wall_ns
            <= binding.usable_end_wall_ns
        )
    ):
        raise ValueError("Round 74 segmented development binding differs")


def _fit_sharded_optimization_feature_scaler(
    *,
    database_by_run_id: Mapping[str, Path],
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    training_split: object,
    memory_limit: str,
    database_threads: int,
    chunk_rows: int = ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74EventFeatureScaler:
    partition.validate()
    training_split.validate()
    optimization_run_ids = tuple(training_split.optimization_run_ids)
    bindings = dict(bindings_by_run_id)
    routes = dict(database_by_run_id)
    if (
        not optimization_run_ids
        or any(
            run_id not in bindings or run_id not in routes
            for run_id in optimization_run_ids
        )
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 2
    ):
        raise ValueError("Round 74 sharded scaler input binding differs")

    def chunks() -> Iterator[np.ndarray]:
        values: list[tuple[float, ...]] = []
        for database, run_ids in _consecutive_database_groups(
            optimization_run_ids,
            routes,
        ):
            _guard_idle_databases((database,), repeated=True)
            with ImpactAbsorptionStore(
                database,
                memory_limit=memory_limit,
                threads=database_threads,
                read_only=True,
            ) as store:
                for run_id in run_ids:
                    binding = bindings[run_id]
                    _validate_binding_entry(binding, partition, run_id=run_id)
                    entry = partition.entry(run_id)
                    for observation in iter_round74_v10_segment_event_observations(
                        store,
                        binding=binding,
                    ):
                        token = observation.token
                        if token is None:
                            continue
                        if not (
                            entry.capture_start_wall_ns
                            <= token.received_wall_ns
                            <= entry.capture_end_wall_ns
                        ):
                            raise ValueError(
                                "Round 74 sharded scaler event is outside its capture run"
                            )
                        values.append(token.feature_values)
                        if len(values) == chunk_rows:
                            yield np.asarray(values, dtype=np.float64)
                            values.clear()
        if values:
            yield np.asarray(values, dtype=np.float64)

    return fit_round74_event_feature_scaler_stream(
        chunks(),
        partition_role="training",
        maximum_fit_rows=maximum_fit_rows,
        fit_source_scope="segmented_optimization_training_runs",
        fit_source_run_ids=optimization_run_ids,
        fit_source_partition_sha256=training_split.parent_partition_sha256,
        fit_source_selection_sha256=training_split.split_sha256,
    )


def _assemble_sharded_role_batches(
    *,
    database_by_run_id: Mapping[str, Path],
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    memory_limit: str,
    database_threads: int,
) -> tuple[Round74EventTrainingBatch, ...]:
    partition.validate()
    entries = tuple(entry for entry in partition.entries if entry.role == role)
    expected = tuple(entry.run_id for entry in entries)
    routes = dict(database_by_run_id)
    bindings = dict(bindings_by_run_id)
    assemblies = dict(target_assembly_by_run_id)
    if (
        role not in {"training", "tuning"}
        or set(bindings) != set(expected)
        or set(assemblies) != set(expected)
        or any(run_id not in routes for run_id in expected)
    ):
        raise ValueError("Round 74 sharded development role panel differs")
    batches_by_run_id: dict[str, Round74EventTrainingBatch] = {}
    for database, run_ids in _consecutive_database_groups(expected, routes):
        _guard_idle_databases((database,), repeated=True)
        with ImpactAbsorptionStore(
            database,
            memory_limit=memory_limit,
            threads=database_threads,
            read_only=True,
        ) as store:
            for run_id in run_ids:
                entry = partition.entry(run_id)
                binding = bindings[run_id]
                _validate_binding_entry(binding, partition, run_id=run_id)
                samples = select_round74_segmented_event_windows(
                    iter_round74_segmented_labeled_event_windows(
                        store,
                        partition=partition,
                        binding=binding,
                        target_assembly=assemblies[run_id],
                    ),
                    entry=entry,
                )
                batch = build_round74_event_training_batch(samples, scaler=scaler)
                batch.validate()
                if (
                    batch.role != role
                    or set(batch.run_id) != {run_id}
                    or batch.rows
                    != len(IMPACT_CAPTURE_SYMBOLS)
                    * round74_segmented_windows_per_symbol(entry)
                ):
                    raise ValueError("Round 74 sharded development batch differs")
                batches_by_run_id[run_id] = batch
    if set(batches_by_run_id) != set(expected):
        raise ValueError("Round 74 sharded development batch coverage differs")
    return tuple(batches_by_run_id[run_id] for run_id in expected)


def _prepare_sharded_development(
    *,
    database_by_run_id: Mapping[str, Path],
    inputs: Round74SegmentedDevelopmentInputs,
    memory_limit: str,
    database_threads: int,
) -> Round74SegmentedPreparedDevelopment:
    partition = inputs.coverage.partition
    assemblies = inputs.target_assembly_by_run_id()
    bindings = inputs.development_bindings_by_run_id()
    training_bindings = {
        entry.run_id: bindings[entry.run_id]
        for entry in partition.entries
        if entry.role == "training"
    }
    tuning_bindings = {
        entry.run_id: bindings[entry.run_id]
        for entry in partition.entries
        if entry.role == "tuning"
    }
    training_split = build_round74_segmented_training_split(
        partition,
        bindings_by_run_id=training_bindings,
    )
    tuning_subpartition = build_round74_segmented_tuning_subpartition(
        partition,
        bindings_by_run_id=tuning_bindings,
    )
    scaler = _fit_sharded_optimization_feature_scaler(
        database_by_run_id=database_by_run_id,
        partition=partition,
        bindings_by_run_id=training_bindings,
        training_split=training_split,
        memory_limit=memory_limit,
        database_threads=database_threads,
    )
    training_batches = _assemble_sharded_role_batches(
        database_by_run_id=database_by_run_id,
        partition=partition,
        bindings_by_run_id=training_bindings,
        scaler=scaler,
        role="training",
        target_assembly_by_run_id={
            run_id: assemblies[run_id] for run_id in training_bindings
        },
        memory_limit=memory_limit,
        database_threads=database_threads,
    )
    tuning_batches = _assemble_sharded_role_batches(
        database_by_run_id=database_by_run_id,
        partition=partition,
        bindings_by_run_id=tuning_bindings,
        scaler=scaler,
        role="tuning",
        target_assembly_by_run_id={
            run_id: assemblies[run_id] for run_id in tuning_bindings
        },
        memory_limit=memory_limit,
        database_threads=database_threads,
    )
    prepared = Round74PreparedDevelopmentData(
        scaler=scaler,
        training_batches=training_batches,
        tuning_batches=tuning_batches,
    )
    prepared.validate()
    tuning_roles = split_round74_prepared_tuning_roles(
        prepared,
        subpartition=tuning_subpartition,
    )
    result = Round74SegmentedPreparedDevelopment(
        partition_sha256=partition.partition_sha256,
        training_split=training_split,
        tuning_subpartition=tuning_subpartition,
        prepared=prepared,
        tuning_roles=tuning_roles,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74ShardedDevelopmentExecutionPanelBuilder:
    """Build policy-selection economics from one read-only shard at a time."""

    database_by_run_id: Mapping[str, Path]
    memory_limit: str
    database_threads: int

    def __post_init__(self) -> None:
        routes = {
            str(run_id): Path(path) for run_id, path in self.database_by_run_id.items()
        }
        if not routes or any(
            path.is_symlink() or not path.is_file() for path in routes.values()
        ):
            raise ValueError("Round 74 sharded execution route differs")
        memory, threads = validate_impact_store_resources(
            self.memory_limit,
            self.database_threads,
        )
        object.__setattr__(self, "database_by_run_id", MappingProxyType(routes))
        object.__setattr__(self, "memory_limit", memory)
        object.__setattr__(self, "database_threads", threads)

    def __call__(
        self,
        *,
        partition: Round74EventRunPartition,
        policy_selection_batches: Sequence[Round74EventTrainingBatch],
        target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
        latency_evidence: object,
    ) -> tuple[Round74ActionExecutionPanel, ...]:
        partition.validate()
        latency_evidence.validate()
        batches = tuple(policy_selection_batches)
        for batch in batches:
            batch.validate()
        run_ids = tuple(batch.run_id[0] for batch in batches)
        assemblies = dict(target_assembly_by_run_id)
        if (
            not batches
            or len(run_ids) != len(set(run_ids))
            or set(assemblies) != set(run_ids)
            or any(run_id not in self.database_by_run_id for run_id in run_ids)
        ):
            raise ValueError("Round 74 sharded execution input differs")
        batch_by_run_id = dict(zip(run_ids, batches, strict=True))
        panels_by_profile: dict[str, list[Round74ActionExecutionPanel]] = {
            profile: [] for profile in ROUND74_ACTION_PROFILES
        }
        for database, grouped_run_ids in _consecutive_database_groups(
            run_ids,
            self.database_by_run_id,
        ):
            _guard_idle_databases((database,), repeated=True)
            with ImpactAbsorptionStore(
                database,
                memory_limit=self.memory_limit,
                threads=self.database_threads,
                read_only=True,
            ) as store:
                shard_panels = build_round74_delayed_execution_panels(
                    store,
                    partition=partition,
                    policy_selection_batches=tuple(
                        batch_by_run_id[run_id] for run_id in grouped_run_ids
                    ),
                    target_assembly_by_run_id={
                        run_id: assemblies[run_id] for run_id in grouped_run_ids
                    },
                    latency_evidence=latency_evidence,
                )
            if (
                tuple(panel.profile for panel in shard_panels)
                != ROUND74_ACTION_PROFILES
            ):
                raise ValueError("Round 74 sharded execution profile panel differs")
            for panel in shard_panels:
                panel.validate()
                panels_by_profile[panel.profile].append(panel)
        result: list[Round74ActionExecutionPanel] = []
        for profile in ROUND74_ACTION_PROFILES:
            parts = tuple(panels_by_profile[profile])
            if not parts:
                raise ValueError("Round 74 sharded execution result is empty")
            first = parts[0]
            if any(
                part.partition_sha256 != first.partition_sha256
                or part.decision_latency_evidence_sha256
                != first.decision_latency_evidence_sha256
                or part.additional_entry_latency_ns != first.additional_entry_latency_ns
                or part.execution_replay_module_sha256
                != first.execution_replay_module_sha256
                for part in parts
            ):
                raise ValueError("Round 74 sharded execution identity differs")
            combined = Round74ActionExecutionPanel(
                profile=profile,
                partition_sha256=first.partition_sha256,
                decision_latency_evidence_sha256=(
                    first.decision_latency_evidence_sha256
                ),
                additional_entry_latency_ns=first.additional_entry_latency_ns,
                source_target_assembly_sha256=tuple(
                    row for part in parts for row in part.source_target_assembly_sha256
                ),
                source_capture_report_sha256=tuple(
                    row for part in parts for row in part.source_capture_report_sha256
                ),
                execution_replay_module_sha256=first.execution_replay_module_sha256,
                rows=tuple(row for part in parts for row in part.rows),
            )
            combined.validate()
            result.append(combined)
        return tuple(result)


@dataclass(frozen=True)
class Round74ShardedAIQualificationExecutionReplayProvider:
    """Replay preassigned AI qualification rows from their exact shards."""

    database_by_run_id: Mapping[str, Path]
    partition: Round74EventRunPartition
    qualification_population: Round74AIQualificationPopulation
    assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly]
    memory_limit: str
    database_threads: int

    def __post_init__(self) -> None:
        self.partition.validate()
        self.qualification_population.validate()
        run_ids = self.qualification_population.run_ids
        routes = {
            str(run_id): Path(path) for run_id, path in self.database_by_run_id.items()
        }
        assemblies = dict(self.assembly_by_run_id)
        if (
            set(assemblies) != set(run_ids)
            or any(run_id not in routes for run_id in run_ids)
            or any(self.partition.entry(run_id).role != "tuning" for run_id in run_ids)
            or any(path.is_symlink() or not path.is_file() for path in routes.values())
        ):
            raise ValueError("Round 74 sharded AI qualification panel differs")
        memory, threads = validate_impact_store_resources(
            self.memory_limit,
            self.database_threads,
        )
        object.__setattr__(self, "database_by_run_id", MappingProxyType(routes))
        object.__setattr__(self, "assembly_by_run_id", MappingProxyType(assemblies))
        object.__setattr__(self, "memory_limit", memory)
        object.__setattr__(self, "database_threads", threads)

    def __call__(
        self,
        *,
        qualification_population: Round74AIQualificationPopulation,
        action_selection: object,
        instructions_by_manifest: Mapping[
            str,
            Sequence[Round74AIExecutionReplayInstruction],
        ],
    ) -> dict[str, tuple[Round74AIExecutionReplayEvidence, ...]]:
        qualification_population.validate()
        action_selection.validate()
        run_ids = self.qualification_population.run_ids
        rows_by_manifest = {
            str(manifest): tuple(rows)
            for manifest, rows in instructions_by_manifest.items()
        }
        if (
            qualification_population.population_sha256
            != self.qualification_population.population_sha256
            or action_selection.tuning_subpartition_sha256
            != qualification_population.parent_tuning_subpartition_sha256
            or len(rows_by_manifest) != 2
        ):
            raise ValueError("Round 74 sharded AI qualification identity differs")
        for manifest, rows in rows_by_manifest.items():
            for row in rows:
                row.validate()
                if (
                    row.model_manifest_sha256 != manifest
                    or row.partition_sha256 != self.partition.partition_sha256
                    or row.action_selection_sha256 != action_selection.selection_sha256
                    or row.run_id not in run_ids
                ):
                    raise ValueError(
                        "Round 74 sharded AI qualification instruction differs"
                    )
        evidence_by_manifest: dict[
            str,
            dict[int, Round74AIExecutionReplayEvidence],
        ] = {manifest: {} for manifest in rows_by_manifest}
        for database, grouped_run_ids in _consecutive_database_groups(
            run_ids,
            self.database_by_run_id,
        ):
            _guard_idle_databases((database,), repeated=True)
            with ImpactAbsorptionStore(
                database,
                memory_limit=self.memory_limit,
                threads=self.database_threads,
                read_only=True,
            ) as store:
                for manifest, rows in rows_by_manifest.items():
                    for run_id in grouped_run_ids:
                        run_rows = tuple(row for row in rows if row.run_id == run_id)
                        if not run_rows:
                            continue
                        replayed = replay_round74_ai_execution_store_run(
                            store,
                            partition=self.partition,
                            run_id=run_id,
                            assembly=self.assembly_by_run_id[run_id],
                            instructions=run_rows,
                        )
                        if len(replayed) != len(run_rows):
                            raise ValueError(
                                "Round 74 sharded AI qualification coverage differs"
                            )
                        for instruction, evidence in zip(
                            run_rows,
                            replayed,
                            strict=True,
                        ):
                            evidence.validate()
                            if (
                                evidence.row_index != instruction.row_index
                                or evidence.feature_row_sha256
                                != instruction.feature_row_sha256
                                or evidence.run_id != instruction.run_id
                                or evidence.symbol != instruction.symbol
                                or evidence.side != instruction.side
                                or evidence.horizon_seconds
                                != instruction.horizon_seconds
                                or evidence.source_review_sha256
                                != instruction.source_review_sha256
                                or evidence.partition_sha256
                                != instruction.partition_sha256
                                or instruction.row_index
                                in evidence_by_manifest[manifest]
                            ):
                                raise ValueError(
                                    "Round 74 sharded AI qualification row differs"
                                )
                            evidence_by_manifest[manifest][instruction.row_index] = (
                                evidence
                            )
        result: dict[str, tuple[Round74AIExecutionReplayEvidence, ...]] = {}
        for manifest, rows in rows_by_manifest.items():
            try:
                result[manifest] = tuple(
                    evidence_by_manifest[manifest][row.row_index] for row in rows
                )
            except KeyError as exc:
                raise ValueError(
                    "Round 74 sharded AI qualification evidence is incomplete"
                ) from exc
            if len(result[manifest]) != len(rows):
                raise ValueError(
                    "Round 74 sharded AI qualification evidence panel differs"
                )
        return result


def _run_development(
    database_by_run_id: Mapping[str, Path],
    inputs: Round74SegmentedDevelopmentInputs,
    *,
    model_output_directory: Path,
    qualification_output_directory: Path,
    compute_backend: str,
    training_config: Round74EventTrainingConfig,
    inference_minibatch_rows: int,
    enable_ai: bool,
    memory_limit: str,
    database_threads: int,
    progress: ProgressCallback,
) -> tuple[object, Round74SegmentedQualifiedDevelopment | None, str]:
    training_config.validate()
    if training_config.execution_mode != "segmented_cohort":
        raise ValueError("Round 74 segmented development training mode differs")
    partition = inputs.coverage.partition
    assemblies = inputs.target_assembly_by_run_id()
    bindings = inputs.development_bindings_by_run_id()
    progress("segmented_preparation_started")
    preparation = _prepare_sharded_development(
        database_by_run_id=database_by_run_id,
        inputs=inputs,
        memory_limit=memory_limit,
        database_threads=database_threads,
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
        store=None,
        partition=partition,
        target_assembly_by_run_id=assemblies,
        execution_panel_builder=Round74ShardedDevelopmentExecutionPanelBuilder(
            database_by_run_id=database_by_run_id,
            memory_limit=memory_limit,
            database_threads=database_threads,
        ),
        output_directory=model_output_directory,
        compute_backend=compute_backend,
        config=training_config,
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
    final_action_configurations = {
        profile: select_round74_final_action_configuration(
            policy.bundle,
            profile=profile,
        )
        for profile in accepted_profiles
    }
    replay_provider = Round74ShardedAIQualificationExecutionReplayProvider(
        database_by_run_id=database_by_run_id,
        partition=partition,
        qualification_population=population,
        assembly_by_run_id=qualification_assemblies,
        memory_limit=memory_limit,
        database_threads=database_threads,
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
            final_action_configuration=final_action_configurations[profile],
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
            final_action_configuration_sha256=(
                final_action_configurations[profile].configuration_sha256
            ),
            final_action_configuration_mode=(final_action_configurations[profile].mode),
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
    database_paths: str | Path | Sequence[str | Path],
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
    supervised_device_group_policy: str = "auto",
    supervised_device_run_group_size: int = 8,
    device_group_preflight_timeout_seconds: float = 300.0,
    progress_interval_seconds: float = 30.0,
    enable_ai: bool = True,
) -> dict[str, object]:
    """Validate everything before opening the evidence store read-only."""

    if not callable(progress):
        raise TypeError("Round 74 segmented development progress differs")
    (
        databases,
        (
            plan,
            state,
            recovery,
            assemblies,
            sources,
            model_output,
            qualification_output,
        ),
    ) = _validate_path_panel(
        repository=Path(repository),
        database_paths=database_paths,
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
    normalized_device_group_policy = str(supervised_device_group_policy)
    if (
        isinstance(inference_minibatch_rows, bool)
        or not isinstance(inference_minibatch_rows, int)
        or not 1 <= inference_minibatch_rows <= 65_536
        or normalized_device_group_policy not in {"auto", "fixed"}
        or isinstance(supervised_device_run_group_size, bool)
        or not isinstance(supervised_device_run_group_size, int)
        or not 1 <= supervised_device_run_group_size <= 32
        or isinstance(device_group_preflight_timeout_seconds, bool)
        or not 30.0 <= float(device_group_preflight_timeout_seconds) <= 900.0
        or isinstance(progress_interval_seconds, bool)
        or not 1.0 <= float(progress_interval_seconds) <= 300.0
        or not isinstance(enable_ai, bool)
    ):
        raise ValueError("Round 74 segmented development runtime policy differs")

    initial_database_bytes = _guard_idle_databases(databases, repeated=False)
    progress(
        "target_free_coverage_validation_started",
        database_count=len(databases),
        database_bytes_total=sum(initial_database_bytes),
    )
    terminal_coverage = load_round74_segmented_terminal_coverage(
        plan_path=plan,
        state_root=state,
        recovery_outcome_directory=recovery,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
    )
    development_run_ids = tuple(
        entry.run_id
        for entry in terminal_coverage.coverage.partition.entries
        if entry.role != "test"
    )
    database_by_run_id, database_route_sha256, used_databases = (
        _route_development_run_databases(
            databases,
            development_run_ids=development_run_ids,
            memory_limit=normalized_memory,
            database_threads=normalized_threads,
        )
    )
    progress(
        "target_free_database_route_completed",
        development_run_count=len(development_run_ids),
        supplied_database_count=len(databases),
        used_database_count=len(used_databases),
        database_route_sha256=database_route_sha256,
        target_manifests_read=False,
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
        target_manifests_read=False,
    )
    device_group_preflight = None
    device_group_preflight_path = None
    if normalized_device_group_policy == "auto":
        progress(
            "device_group_preflight_started",
            candidate_count=len(Round74EventTrainingConfig().candidate_ids),
            minibatch_rows=Round74EventTrainingConfig().minibatch_rows,
            target_manifests_read=False,
        )
        device_group_preflight = run_round74_device_group_preflight_subprocess(
            backend,
            minibatch_rows=Round74EventTrainingConfig().minibatch_rows,
            timeout_seconds=float(device_group_preflight_timeout_seconds),
            progress=progress,
        )
        device_group_preflight_path = write_round74_device_group_preflight(
            device_group_preflight,
            model_output,
        )
        progress(
            "device_group_preflight_completed",
            preflight_sha256=device_group_preflight["preflight_sha256"],
            selected_group_sizes=device_group_preflight["selected_group_sizes"],
            artifact_path=str(device_group_preflight_path),
            target_manifests_read=False,
        )
    training_config = _device_group_training_config(
        policy=normalized_device_group_policy,
        fixed_group_size=supervised_device_run_group_size,
        preflight=device_group_preflight,
    )
    progress("input_validation_started")
    inputs = load_round74_segmented_development_inputs(
        plan_path=plan,
        state_root=state,
        recovery_outcome_directory=recovery,
        target_assembly_directory=assemblies,
        source_artifact_root=sources,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
    )
    if (
        inputs.plan.plan_sha256 != terminal_coverage.plan.plan_sha256
        or inputs.coverage.coverage_sha256 != terminal_coverage.coverage.coverage_sha256
        or inputs.coverage.partition.partition_sha256
        != terminal_coverage.coverage.partition.partition_sha256
        or inputs.slot_evidence != terminal_coverage.slot_evidence
        or tuple(inputs.development_bindings_by_run_id()) != development_run_ids
    ):
        raise ValueError("Round 74 segmented development target-free coverage drifted")
    progress(
        "input_validation_completed",
        inputs_sha256=inputs.inputs_sha256,
        plan_sha256=inputs.plan.plan_sha256,
        coverage_sha256=inputs.coverage.coverage_sha256,
        partition_sha256=inputs.coverage.partition.partition_sha256,
        development_target_assembly_count=len(inputs.target_assemblies),
    )
    database_bytes = _guard_idle_databases(databases, repeated=True)
    with progress_heartbeat(
        progress,
        phase="segmented_development",
        interval_seconds=float(progress_interval_seconds),
    ):
        policy, qualified, preparation_sha256 = _run_development(
            database_by_run_id,
            inputs,
            model_output_directory=model_output,
            qualification_output_directory=qualification_output,
            compute_backend=str(compute_backend),
            training_config=training_config,
            inference_minibatch_rows=inference_minibatch_rows,
            enable_ai=enable_ai,
            memory_limit=normalized_memory,
            database_threads=normalized_threads,
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
                "final_action_configuration_sha256": (
                    None
                    if qualification is None
                    else qualification.final_action_configuration.configuration_sha256
                ),
                "final_action_configuration_mode": (
                    None
                    if qualification is None
                    else qualification.final_action_configuration.mode
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
        "database_count": len(databases),
        "used_database_count": len(used_databases),
        "database_bytes_total": sum(database_bytes),
        "database_route_sha256": database_route_sha256,
        "database_open_mode": "read_only",
        "database_access_policy": "federated_read_only_one_at_a_time",
        "resources": {
            "memory_limit": normalized_memory,
            "database_threads": normalized_threads,
            "inference_minibatch_rows": inference_minibatch_rows,
            "supervised_device_group_policy": normalized_device_group_policy,
            "supervised_device_run_group_sizes": {
                candidate_id: training_config.device_run_group_size_for(candidate_id)
                for candidate_id in training_config.candidate_ids
            },
            "device_group_preflight_sha256": (
                None
                if device_group_preflight is None
                else device_group_preflight["preflight_sha256"]
            ),
            "device_group_preflight_path": (
                None
                if device_group_preflight_path is None
                else str(device_group_preflight_path)
            ),
            "causal_pretraining_device_run_group_size": (
                training_config.pretraining.device_run_group_size
            ),
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
    "ROUND74_SEGMENTED_DEVELOPMENT_DATABASE_ROUTE_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION",
    "Round74ShardedAIQualificationExecutionReplayProvider",
    "Round74ShardedDevelopmentExecutionPanelBuilder",
    "run_round74_segmented_development",
]

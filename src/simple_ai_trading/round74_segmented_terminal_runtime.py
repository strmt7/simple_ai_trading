"""Concrete post-capture one-use evaluator for the segmented Round 74 model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import requests

from .ai_review import resolve_ollama_model_provenance
from .compute import BackendInfo, resolve_backend, torch_device_for_backend
from .impact_absorption_ai_execution_replay import (
    Round74AIExecutionReplayEvidence,
    Round74AIExecutionReplayInstruction,
    replay_round74_ai_execution_store_run,
)
from .impact_absorption_ai_review_preparation import (
    Round74AIReviewModelBinding,
    Round74PreparedSealedAIReviewProvider,
    round74_default_ai_review_model_panel,
)
from .impact_absorption_ai_uplift import (
    load_round74_ai_pretest_qualification,
)
from .impact_absorption_event_action_configuration import (
    select_round74_final_action_configuration,
)
from .impact_absorption_event_dataset import (
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74EventTrainingBatch,
    build_round74_event_training_batch,
)
from .impact_absorption_event_sealed_evaluation import (
    evaluate_round74_sealed_once,
)
from .impact_absorption_event_sealed_ledger import (
    Round74SealedEvaluationLedger,
)
from .impact_absorption_event_training import load_round74_pretest_scaler
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
    validate_impact_store_resources,
)
from .round74_active_qualification import _database_and_wal_bytes
from .round74_event_development_operator import (
    load_round74_development_policy_bundle,
)
from .round74_segmented_campaign_runner import _active_segmented_capture_processes
from .round74_segmented_development_inputs import (
    ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT,
    load_round74_segmented_terminal_coverage,
)
from .round74_segmented_model_operator import (
    build_round74_segmented_sealed_dataset_identity,
    build_round74_segmented_test_population,
    iter_round74_segmented_labeled_event_windows,
    round74_segmented_windows_per_symbol,
    select_round74_segmented_event_windows,
)
from .round74_target_assembly_manifest import (
    load_and_audit_round74_target_assembly_manifest,
)
from .round74_terminal_one_use import (
    Round74TerminalOneUseStore,
    Round74TerminalPreaccessIdentity,
    build_round74_terminal_result_bundle,
    validate_round74_terminal_result_bundle,
)
from .storage import write_json_atomic


ROUND74_SEGMENTED_TERMINAL_RUN_SCHEMA_VERSION = "round-074-segmented-terminal-run-v1"
ROUND74_SEGMENTED_TERMINAL_RECOVERY_SCHEMA_VERSION = (
    "round-074-segmented-terminal-recovery-v1"
)
ROUND74_SEGMENTED_TERMINAL_DATABASE_ROUTE_SCHEMA_VERSION = (
    "round-074-segmented-terminal-database-route-v1"
)

ProgressCallback = Callable[..., None]
ProvenanceResolver = Callable[[str, str, float], tuple[str, str]]
RuntimeVersionResolver = Callable[[str, float], str]

_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Round 74 segmented terminal {label} path differs")
    raw = path.read_bytes()
    if not raw or len(raw) > 256 * 1024 * 1024:
        raise ValueError(f"Round 74 segmented terminal {label} size differs")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"Round 74 segmented terminal {label} has duplicate keys"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Round 74 segmented terminal {label} contains {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 74 segmented terminal {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 74 segmented terminal {label} root differs")
    return value


def _normalize_database_paths(
    value: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        selected = (Path(value),)
    elif isinstance(value, Sequence):
        selected = tuple(Path(item) for item in value)
    else:
        raise TypeError("Round 74 terminal database panel type differs")
    if not selected:
        raise ValueError("Round 74 terminal database panel is empty")
    return selected


def _guard_idle_databases(
    databases: Sequence[Path],
    *,
    repeated: bool,
) -> tuple[int, ...]:
    if _active_segmented_capture_processes():
        state = "became active" if repeated else "is active"
        raise RuntimeError(f"Round 74 terminal blocked: capture {state}")
    result: list[int] = []
    for database in databases:
        database_bytes, wal_bytes = _database_and_wal_bytes(database)
        if database_bytes <= 0:
            raise RuntimeError("Round 74 terminal database is unavailable")
        if wal_bytes != 0:
            state = "appeared" if repeated else "exists"
            raise RuntimeError(f"Round 74 terminal blocked: database WAL {state}")
        result.append(database_bytes)
    return tuple(result)


def _preflight_paths(
    *,
    repository: Path,
    database_paths: str | Path | Sequence[str | Path],
    plan_path: Path,
    state_root: Path,
    recovery_outcome_directory: Path,
    test_target_assembly_directory: Path,
    source_artifact_root: Path,
    development_bundle_path: Path,
    pretest_policy_path: Path,
    ai_qualification_path: Path,
    one_use_store_path: Path,
    sealed_ledger_path: Path,
    output_path: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    raw_databases = _normalize_database_paths(database_paths)
    raw_paths = (
        repository,
        *raw_databases,
        plan_path,
        state_root,
        recovery_outcome_directory,
        test_target_assembly_directory,
        source_artifact_root,
        development_bundle_path,
        pretest_policy_path,
        ai_qualification_path,
        one_use_store_path,
        sealed_ledger_path,
        output_path,
    )
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("Round 74 segmented terminal path symlinks are forbidden")
    root = repository.resolve()
    databases = tuple(path.resolve() for path in raw_databases)
    resolved = tuple(path.resolve() for path in raw_paths[1 + len(databases) :])
    (
        plan,
        state,
        recovery,
        targets,
        sources,
        bundle,
        policy,
        qualification,
        one_use,
        sealed_ledger,
        output,
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
        or not targets.is_dir()
        or not sources.is_dir()
        or not bundle.is_file()
        or not policy.is_file()
        or not qualification.is_file()
        or (one_use.exists() and not one_use.is_file())
        or (sealed_ledger.exists() and not sealed_ledger.is_file())
        or output.exists()
        or one_use == sealed_ledger
        or one_use == output
        or sealed_ledger == output
        or any(path.parent.is_symlink() for path in (one_use, sealed_ledger, output))
    ):
        raise ValueError("Round 74 segmented terminal path panel differs")
    return databases, resolved


def _route_test_run_databases(
    databases: Sequence[Path],
    *,
    test_run_ids: tuple[str, ...],
    memory_limit: str,
    database_threads: int,
) -> tuple[dict[str, Path], str, tuple[Path, ...]]:
    expected = tuple(test_run_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Round 74 terminal test-run database panel differs")
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
        observed_set = set(observed)
        if len(observed_set) != len(observed):
            raise ValueError("Round 74 terminal database run identity is duplicated")
        relevant = tuple(run_id for run_id in expected if run_id in observed_set)
        database_groups.append(relevant)
        for run_id in relevant:
            if run_id in database_by_run_id:
                raise ValueError("Round 74 terminal test run exists in multiple stores")
            database_by_run_id[run_id] = database
    if set(database_by_run_id) != expected_set:
        raise ValueError("Round 74 terminal test-run database coverage differs")
    nonempty_groups = sorted(
        (list(group) for group in database_groups if group),
        key=lambda group: tuple(group),
    )
    route_payload = {
        "schema_version": ROUND74_SEGMENTED_TERMINAL_DATABASE_ROUTE_SCHEMA_VERSION,
        "test_run_ids": list(expected),
        "database_run_groups": nonempty_groups,
        "supplied_database_count": len(databases),
        "used_database_count": len(nonempty_groups),
        "unused_database_count": sum(not group for group in database_groups),
        "absolute_paths_bound": False,
        "capture_content_bound_by_fresh_run_audits": True,
    }
    ordered_routes = {run_id: database_by_run_id[run_id] for run_id in expected}
    used_databases = tuple(dict.fromkeys(ordered_routes.values()))
    return ordered_routes, _canonical_sha256(route_payload), used_databases


def _assemble_test_batches_from_databases(
    *,
    database_by_run_id: Mapping[str, Path],
    test_run_ids: tuple[str, ...],
    partition: object,
    bindings_by_run_id: Mapping[str, object],
    scaler: object,
    target_assembly_by_run_id: Mapping[str, object],
    pretest_model_policy_sha256: str,
    test_unlock_sha256: str,
    window_representation: str,
    memory_limit: str,
    database_threads: int,
) -> tuple[Round74EventTrainingBatch, ...]:
    partition.validate()
    expected = tuple(test_run_ids)
    routes = dict(database_by_run_id)
    bindings = dict(bindings_by_run_id)
    assemblies = dict(target_assembly_by_run_id)
    if (
        set(routes) != set(expected)
        or set(bindings) != set(expected)
        or set(assemblies) != set(expected)
    ):
        raise ValueError("Round 74 terminal sharded batch panel differs")
    batches_by_run_id: dict[str, Round74EventTrainingBatch] = {}
    for database in dict.fromkeys(routes[run_id] for run_id in expected):
        _guard_idle_databases((database,), repeated=True)
        routed_run_ids = tuple(
            run_id for run_id in expected if routes[run_id] == database
        )
        with ImpactAbsorptionStore(
            database,
            memory_limit=memory_limit,
            threads=database_threads,
            read_only=True,
        ) as store:
            for run_id in routed_run_ids:
                entry = partition.entry(run_id)
                binding = bindings[run_id]
                if entry.role != "test" or binding.run_id != run_id:
                    raise ValueError("Round 74 terminal sharded run identity differs")
                samples = select_round74_segmented_event_windows(
                    iter_round74_segmented_labeled_event_windows(
                        store,
                        partition=partition,
                        binding=binding,
                        target_assembly=assemblies[run_id],
                        pretest_model_policy_sha256=pretest_model_policy_sha256,
                        test_unlock_sha256=test_unlock_sha256,
                        window_representation=window_representation,
                    ),
                    entry=entry,
                )
                batch = build_round74_event_training_batch(samples, scaler=scaler)
                batch.validate()
                if (
                    batch.role != "test"
                    or set(batch.run_id) != {run_id}
                    or batch.rows
                    != len(IMPACT_CAPTURE_SYMBOLS)
                    * round74_segmented_windows_per_symbol(entry)
                ):
                    raise ValueError("Round 74 terminal sharded batch identity differs")
                batches_by_run_id[run_id] = batch
    if set(batches_by_run_id) != set(expected):
        raise ValueError("Round 74 terminal sharded batch coverage differs")
    return tuple(batches_by_run_id[run_id] for run_id in expected)


@dataclass(frozen=True)
class Round74ShardedExecutionReplayProvider:
    """Replay each reserved test run from its exact read-only shard."""

    database_by_run_id: Mapping[str, Path]
    partition: object
    assembly_by_run_id: Mapping[str, object]
    memory_limit: str
    database_threads: int

    def __post_init__(self) -> None:
        self.partition.validate()
        test_run_ids = tuple(
            entry.run_id for entry in self.partition.entries if entry.role == "test"
        )
        supplied_routes = {
            str(run_id): Path(path) for run_id, path in self.database_by_run_id.items()
        }
        assemblies = dict(self.assembly_by_run_id)
        if (
            not test_run_ids
            or set(supplied_routes) != set(test_run_ids)
            or set(assemblies) != set(test_run_ids)
            or any(
                path.is_symlink() or not path.is_file()
                for path in supplied_routes.values()
            )
        ):
            raise ValueError("Round 74 sharded replay provider panel differs")
        routes = {run_id: supplied_routes[run_id] for run_id in test_run_ids}
        normalized_memory, normalized_threads = validate_impact_store_resources(
            self.memory_limit,
            self.database_threads,
        )
        object.__setattr__(self, "database_by_run_id", MappingProxyType(routes))
        object.__setattr__(self, "assembly_by_run_id", MappingProxyType(assemblies))
        object.__setattr__(self, "memory_limit", normalized_memory)
        object.__setattr__(self, "database_threads", normalized_threads)

    def __call__(
        self,
        *,
        claim: object,
        instructions_by_manifest: Mapping[
            str,
            Sequence[Round74AIExecutionReplayInstruction],
        ],
    ) -> dict[str, tuple[Round74AIExecutionReplayEvidence, ...]]:
        claim.validate()
        rows_by_manifest = {
            str(manifest): tuple(rows)
            for manifest, rows in instructions_by_manifest.items()
        }
        if (
            claim.status != "reserved"
            or claim.partition_sha256 != self.partition.partition_sha256
            or tuple(self.database_by_run_id) != claim.test_run_ids
            or tuple(rows_by_manifest) != claim.ai_manifest_sha256
        ):
            raise ValueError("Round 74 sharded replay claim identity differs")
        for manifest, rows in rows_by_manifest.items():
            for row in rows:
                row.validate()
                if (
                    row.model_manifest_sha256 != manifest
                    or row.partition_sha256 != claim.partition_sha256
                    or row.action_selection_sha256 != claim.action_selection_sha256
                    or row.run_id not in self.database_by_run_id
                ):
                    raise ValueError("Round 74 sharded replay instruction differs")
        evidence_by_manifest: dict[
            str,
            dict[int, Round74AIExecutionReplayEvidence],
        ] = {manifest: {} for manifest in rows_by_manifest}
        for database in dict.fromkeys(self.database_by_run_id.values()):
            _guard_idle_databases((database,), repeated=True)
            routed_run_ids = tuple(
                run_id
                for run_id in claim.test_run_ids
                if self.database_by_run_id[run_id] == database
            )
            with ImpactAbsorptionStore(
                database,
                memory_limit=self.memory_limit,
                threads=self.database_threads,
                read_only=True,
            ) as store:
                for manifest, rows in rows_by_manifest.items():
                    for run_id in routed_run_ids:
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
                            raise ValueError("Round 74 sharded replay coverage differs")
                        for row, evidence in zip(run_rows, replayed, strict=True):
                            evidence.validate()
                            if evidence.row_index != row.row_index:
                                raise ValueError(
                                    "Round 74 sharded replay order differs"
                                )
                            if row.row_index in evidence_by_manifest[manifest]:
                                raise ValueError(
                                    "Round 74 sharded replay row is duplicated"
                                )
                            evidence_by_manifest[manifest][row.row_index] = evidence
        result: dict[str, tuple[Round74AIExecutionReplayEvidence, ...]] = {}
        for manifest, rows in rows_by_manifest.items():
            try:
                result[manifest] = tuple(
                    evidence_by_manifest[manifest][row.row_index] for row in rows
                )
            except KeyError as exc:
                raise ValueError(
                    "Round 74 sharded replay evidence is incomplete"
                ) from exc
            if len(result[manifest]) != len(rows):
                raise ValueError("Round 74 sharded replay evidence panel differs")
        return result


def _preflight_backend(requested: str) -> tuple[BackendInfo, str]:
    backend = resolve_backend(requested, require=True)
    torch_device_for_backend(backend)
    digest = _canonical_sha256(
        {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": backend.device,
            "vendor": backend.vendor,
            "reason": backend.reason,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
            "request_satisfied": backend.request_satisfied,
        }
    )
    return backend, digest


def _preflight_ai_models(
    bindings: tuple[Round74AIReviewModelBinding, ...],
    *,
    resolver: ProvenanceResolver,
    runtime_version_resolver: RuntimeVersionResolver,
) -> tuple[str, ...]:
    observed: dict[str, str] = {}
    runtime_versions: dict[str, str] = {}
    for binding in bindings:
        binding.validate()
        endpoint = binding.runtime.endpoint.rstrip("/")
        actual_runtime_version = runtime_versions.get(endpoint)
        if actual_runtime_version is None:
            actual_runtime_version = runtime_version_resolver(endpoint, 3.0)
            runtime_versions[endpoint] = actual_runtime_version
        expected_match = _SEMVER.fullmatch(binding.manifest.runtime_version)
        actual_match = _SEMVER.fullmatch(actual_runtime_version)
        if (
            expected_match is None
            or actual_match is None
            or actual_match.group(1, 2) != expected_match.group(1, 2)
            or int(actual_match.group(3)) < int(expected_match.group(3))
        ):
            raise ValueError("Round 74 terminal AI runtime version differs")
        digest, metadata_sha256 = resolver(
            endpoint,
            binding.model_name,
            3.0,
        )
        if digest != binding.manifest.model_artifact_sha256:
            raise ValueError("Round 74 terminal AI model provenance differs")
        observed[binding.manifest.manifest_sha256] = _canonical_sha256(
            {
                "model_manifest_sha256": binding.manifest.manifest_sha256,
                "model_artifact_sha256": digest,
                "provider_metadata_sha256": metadata_sha256,
                "audited_runtime_floor": binding.manifest.runtime_version,
                "observed_runtime_version": actual_runtime_version,
            }
        )
    if not 1 <= len(observed) <= 2 or len(observed) != len(bindings):
        raise ValueError("Round 74 terminal AI model panel differs")
    return tuple(observed[manifest] for manifest in sorted(observed))


def _resolve_ollama_runtime_version(endpoint: str, timeout_seconds: float) -> str:
    response = requests.get(  # nosec B113 - fixed local provider endpoint
        f"{str(endpoint).rstrip('/')}/api/version",
        timeout=max(0.1, min(float(timeout_seconds), 5.0)),
        headers={"User-Agent": "simple-ai-trading-round74-terminal/0.1"},
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, Mapping) or not isinstance(value.get("version"), str):
        raise ValueError("Round 74 terminal AI runtime response differs")
    version = str(value["version"])
    if _SEMVER.fullmatch(version) is None:
        raise ValueError("Round 74 terminal AI runtime version differs")
    return version


def _policy_window_representation(
    policy_path: Path,
    *,
    expected_policy_sha256: str,
) -> str:
    value = _read_json_object(policy_path, "pretest policy")
    development = value.get("development_data")
    representation = (
        development.get("window_representation")
        if isinstance(development, Mapping)
        else None
    )
    if (
        value.get("policy_sha256") != expected_policy_sha256
        or representation not in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    ):
        raise ValueError("Round 74 terminal pretest representation differs")
    return str(representation)


def _preflight_test_target_names(
    directory: Path,
    *,
    run_ids: tuple[str, ...],
) -> None:
    entries = tuple(directory.iterdir())
    expected = {f"{run_id}.json" for run_id in run_ids}
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("Round 74 terminal target manifest panel differs")


def _load_test_target_assemblies(
    directory: Path,
    *,
    source_artifact_root: Path,
    bindings_by_run_id: Mapping[str, object],
    run_ids: tuple[str, ...],
) -> dict[str, object]:
    manifests = {}
    for run_id in run_ids:
        manifest = load_and_audit_round74_target_assembly_manifest(
            manifest_path=directory / f"{run_id}.json",
            source_artifact_root=source_artifact_root,
        )
        binding = bindings_by_run_id[run_id]
        if (
            manifest.run_id != run_id
            or manifest.cohort_binding_sha256 != binding.binding_sha256
            or manifest.assembly.spec.execution_environment
            != ROUND74_SEGMENTED_DEVELOPMENT_EXECUTION_ENVIRONMENT
        ):
            raise ValueError("Round 74 terminal target manifest binding differs")
        manifests[run_id] = manifest.assembly
    return manifests


def _persist_terminal_output(path: Path, bundle: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise FileExistsError("Round 74 immutable terminal output already exists")
    write_json_atomic(path, dict(bundle), indent=2, sort_keys=True)
    restored = validate_round74_terminal_result_bundle(
        _read_json_object(path, "terminal output")
    )
    if restored != dict(bundle):
        raise RuntimeError("Round 74 terminal output persistence differs")


def run_round74_segmented_terminal_evaluation(
    *,
    repository: str | Path,
    database_paths: str | Path | Sequence[str | Path],
    plan_path: str | Path,
    state_root: str | Path,
    recovery_outcome_directory: str | Path,
    test_target_assembly_directory: str | Path,
    source_artifact_root: str | Path,
    development_bundle_path: str | Path,
    pretest_policy_path: str | Path,
    ai_qualification_path: str | Path,
    one_use_store_path: str | Path,
    sealed_ledger_path: str | Path,
    output_path: str | Path,
    terminal_observed_wall_ns: int,
    progress: ProgressCallback,
    profile: str = "conservative",
    compute_backend: str = "auto",
    memory_limit: str = "4GB",
    database_threads: int = 2,
    inference_minibatch_rows: int = 2_048,
    model_bindings: tuple[Round74AIReviewModelBinding, ...] | None = None,
    provenance_resolver: ProvenanceResolver = resolve_ollama_model_provenance,
    runtime_version_resolver: RuntimeVersionResolver = (
        _resolve_ollama_runtime_version
    ),
) -> dict[str, object]:
    """Preflight, reserve, load targets once, evaluate, and persist evidence."""

    if (
        not callable(progress)
        or not callable(provenance_resolver)
        or not callable(runtime_version_resolver)
    ):
        raise TypeError("Round 74 segmented terminal callback differs")
    root = Path(repository)
    databases, paths = _preflight_paths(
        repository=root,
        database_paths=database_paths,
        plan_path=Path(plan_path),
        state_root=Path(state_root),
        recovery_outcome_directory=Path(recovery_outcome_directory),
        test_target_assembly_directory=Path(test_target_assembly_directory),
        source_artifact_root=Path(source_artifact_root),
        development_bundle_path=Path(development_bundle_path),
        pretest_policy_path=Path(pretest_policy_path),
        ai_qualification_path=Path(ai_qualification_path),
        one_use_store_path=Path(one_use_store_path),
        sealed_ledger_path=Path(sealed_ledger_path),
        output_path=Path(output_path),
    )
    (
        plan,
        state,
        recovery,
        targets,
        sources,
        bundle_path,
        policy_path,
        qualification_path,
        one_use_path,
        sealed_ledger_path,
        output,
    ) = paths
    normalized_memory, normalized_threads = validate_impact_store_resources(
        memory_limit,
        database_threads,
    )
    if (
        profile not in ("conservative", "regular", "aggressive")
        or isinstance(terminal_observed_wall_ns, bool)
        or not isinstance(terminal_observed_wall_ns, int)
        or terminal_observed_wall_ns <= 0
        or isinstance(inference_minibatch_rows, bool)
        or not isinstance(inference_minibatch_rows, int)
        or not 1 <= inference_minibatch_rows <= 65_536
    ):
        raise ValueError("Round 74 segmented terminal runtime policy differs")
    one_use_store = Round74TerminalOneUseStore(one_use_path)
    prior_claim = one_use_store.claim() if one_use_path.exists() else None
    if prior_claim is not None:
        raise RuntimeError(
            "Round 74 terminal test access is already consumed: "
            f"status={prior_claim.status} reservation={prior_claim.reservation_id}"
        )
    database_sizes = _guard_idle_databases(databases, repeated=False)
    progress(
        "terminal_preflight_started",
        database_count=len(databases),
        database_bytes=sum(database_sizes),
    )
    coverage = load_round74_segmented_terminal_coverage(
        plan_path=plan,
        state_root=state,
        recovery_outcome_directory=recovery,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
    )
    bindings_by_run_id = coverage.test_bindings_by_run_id()
    test_population = build_round74_segmented_test_population(
        coverage.coverage.partition,
        bindings_by_run_id=bindings_by_run_id,
    )
    database_by_run_id, database_route_sha256, used_databases = (
        _route_test_run_databases(
            databases,
            test_run_ids=test_population.test_run_ids,
            memory_limit=normalized_memory,
            database_threads=normalized_threads,
        )
    )
    _preflight_test_target_names(
        targets,
        run_ids=test_population.test_run_ids,
    )
    bundle = load_round74_development_policy_bundle(bundle_path)
    final_configuration = select_round74_final_action_configuration(
        bundle,
        profile=profile,
    )
    qualification = load_round74_ai_pretest_qualification(qualification_path)
    available_bindings = tuple(
        model_bindings or round74_default_ai_review_model_panel()
    )
    available_by_manifest = {
        binding.manifest.manifest_sha256: binding for binding in available_bindings
    }
    candidate_manifests = set(qualification.candidate_model_manifest_sha256)
    qualified_manifests = qualification.model_manifest_sha256
    if (
        not 1 <= len(available_bindings) <= 2
        or len(available_by_manifest) != len(available_bindings)
        or not set(available_by_manifest).issubset(candidate_manifests)
        or not set(qualified_manifests).issubset(available_by_manifest)
    ):
        raise ValueError("Round 74 terminal AI model bindings differ")
    bindings = tuple(
        available_by_manifest[manifest] for manifest in qualified_manifests
    )
    bound_manifests = tuple(
        sorted(binding.manifest.manifest_sha256 for binding in bindings)
    )
    if (
        not qualification.qualification_passed
        or qualification.profile != profile
        or qualification.model_manifest_sha256 != bound_manifests
        or qualification.action_selection_sha256
        != final_configuration.action_selection.selection_sha256
        or qualification.final_action_configuration_sha256
        != final_configuration.configuration_sha256
        or qualification.pretest_policy_sha256 != bundle.pretest_policy_sha256
        or qualification.probability_calibration_sha256
        != bundle.probability_calibration.calibration_sha256
    ):
        raise ValueError("Round 74 terminal development identity differs")
    scaler = load_round74_pretest_scaler(policy_path)
    if scaler.scaler_sha256 != bundle.feature_scaler_sha256:
        raise ValueError("Round 74 terminal feature scaler differs")
    representation = _policy_window_representation(
        policy_path,
        expected_policy_sha256=bundle.pretest_policy_sha256,
    )
    backend, backend_preflight_sha256 = _preflight_backend(compute_backend)
    model_provenance_sha256 = _preflight_ai_models(
        bindings,
        resolver=provenance_resolver,
        runtime_version_resolver=runtime_version_resolver,
    )
    preaccess = Round74TerminalPreaccessIdentity(
        plan_sha256=coverage.plan.plan_sha256,
        coverage_sha256=coverage.coverage.coverage_sha256,
        partition_sha256=coverage.coverage.partition.partition_sha256,
        test_population_sha256=test_population.population_sha256,
        test_run_ids=test_population.test_run_ids,
        database_route_sha256=database_route_sha256,
        optimization_population=test_population.optimization_population,
        development_bundle_sha256=bundle.bundle_sha256,
        pretest_policy_sha256=bundle.pretest_policy_sha256,
        feature_scaler_sha256=bundle.feature_scaler_sha256,
        probability_calibration_sha256=(
            bundle.probability_calibration.calibration_sha256
        ),
        action_selection_sha256=(final_configuration.action_selection.selection_sha256),
        final_action_configuration_sha256=(final_configuration.configuration_sha256),
        ai_pretest_qualification_sha256=qualification.qualification_sha256,
        ai_manifest_sha256=qualification.model_manifest_sha256,
        profile=profile,
        backend_preflight_sha256=backend_preflight_sha256,
        model_provenance_sha256=model_provenance_sha256,
        terminal_observed_wall_ns=terminal_observed_wall_ns,
    )
    preaccess.validate()
    database_sizes = _guard_idle_databases(databases, repeated=True)
    progress(
        "terminal_preflight_completed",
        preaccess_sha256=preaccess.preaccess_sha256,
        test_run_count=len(test_population.test_run_ids),
        supplied_database_count=len(databases),
        used_database_count=len(used_databases),
        database_route_sha256=database_route_sha256,
        backend_kind=backend.kind,
        backend_device=backend.device,
        backend_vendor=backend.vendor,
        sealed_targets_read=False,
    )

    access_claim = one_use_store.reserve(preaccess)
    progress(
        "terminal_access_reserved",
        reservation_id=access_claim.reservation_id,
        sealed_targets_read=False,
    )
    try:
        target_assemblies = _load_test_target_assemblies(
            targets,
            source_artifact_root=sources,
            bindings_by_run_id=bindings_by_run_id,
            run_ids=test_population.test_run_ids,
        )
        progress(
            "terminal_targets_loaded",
            reservation_id=access_claim.reservation_id,
            target_manifest_count=len(target_assemblies),
        )
        database_sizes = _guard_idle_databases(databases, repeated=True)
        batches = _assemble_test_batches_from_databases(
            database_by_run_id=database_by_run_id,
            test_run_ids=test_population.test_run_ids,
            partition=coverage.coverage.partition,
            bindings_by_run_id=bindings_by_run_id,
            scaler=scaler,
            target_assembly_by_run_id=target_assemblies,
            pretest_model_policy_sha256=bundle.pretest_policy_sha256,
            test_unlock_sha256=access_claim.test_unlock_sha256,
            window_representation=representation,
            memory_limit=normalized_memory,
            database_threads=normalized_threads,
        )
        dataset_identity = build_round74_segmented_sealed_dataset_identity(
            batches,
            test_population=test_population,
        )
        review_provider = Round74PreparedSealedAIReviewProvider(
            probability_calibration=bundle.probability_calibration,
            ai_pretest_qualification=qualification,
            model_bindings=bindings,
            progress_callback=lambda payload: progress(
                "terminal_ai_review_progress",
                detail=dict(payload),
            ),
        )
        replay_provider = Round74ShardedExecutionReplayProvider(
            database_by_run_id=database_by_run_id,
            partition=coverage.coverage.partition,
            assembly_by_run_id=target_assemblies,
            memory_limit=normalized_memory,
            database_threads=normalized_threads,
        )
        outcome = evaluate_round74_sealed_once(
            dataset_identity,
            test_batch_loader=lambda *, claim: batches,
            final_action_configuration=final_configuration,
            probability_calibration=bundle.probability_calibration,
            pretest_policy_path=policy_path,
            ai_pretest_qualification=qualification,
            ai_review_provider=review_provider,
            ai_execution_replay_provider=replay_provider,
            ledger=Round74SealedEvaluationLedger(sealed_ledger_path),
            compute_backend=compute_backend,
            inference_minibatch_rows=inference_minibatch_rows,
        )
        terminal_bundle = build_round74_terminal_result_bundle(
            access_claim=access_claim,
            dataset_identity=dataset_identity.as_dict(),
            sealed_report=outcome.report.as_dict(),
            finalized_sealed_claim=outcome.finalized_claim.as_dict(),
        )
        completed_claim = one_use_store.finalize_success(
            access_claim,
            terminal_bundle,
        )
        _persist_terminal_output(output, terminal_bundle)
    except Exception as exc:
        current = one_use_store.claim()
        if current is not None and current.status == "reserved":
            one_use_store.finalize_failure(access_claim, exc)
        raise
    result = {
        "schema_version": ROUND74_SEGMENTED_TERMINAL_RUN_SCHEMA_VERSION,
        "status": "complete",
        "reservation_id": access_claim.reservation_id,
        "preaccess_sha256": preaccess.preaccess_sha256,
        "dataset_sha256": dataset_identity.dataset_sha256,
        "sealed_report_sha256": outcome.report.report_sha256,
        "terminal_bundle_sha256": terminal_bundle["bundle_sha256"],
        "one_use_claim_sha256": completed_claim.claim_sha256,
        "result_outcome": outcome.report.result_outcome,
        "qualified_configuration": list(outcome.report.qualified_configuration),
        "database_bytes": sum(database_sizes),
        "database_count": len(databases),
        "used_database_count": len(used_databases),
        "database_route_sha256": database_route_sha256,
        "database_open_mode": "federated_read_only_one_at_a_time",
        "sealed_test_accessed": True,
        "orders_submitted": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    progress(
        "terminal_evaluation_completed",
        result_sha256=result["result_sha256"],
        result_outcome=outcome.report.result_outcome,
    )
    return result


def recover_round74_segmented_terminal_result(
    *,
    one_use_store_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Export a completed result without reopening data or rerunning models."""

    store_path = Path(one_use_store_path)
    output = Path(output_path)
    if (
        store_path.is_symlink()
        or not store_path.is_file()
        or output.exists()
        or output.is_symlink()
        or output.parent.is_symlink()
    ):
        raise ValueError("Round 74 terminal recovery path panel differs")
    bundle = Round74TerminalOneUseStore(store_path).load_completed_bundle()
    _persist_terminal_output(output, bundle)
    result = {
        "schema_version": ROUND74_SEGMENTED_TERMINAL_RECOVERY_SCHEMA_VERSION,
        "status": "recovered",
        "terminal_bundle_sha256": bundle["bundle_sha256"],
        "sealed_test_reopened": False,
        "model_rerun": False,
        "orders_submitted": False,
        "trading_authority": False,
        "profitability_claim": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "ROUND74_SEGMENTED_TERMINAL_RECOVERY_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TERMINAL_RUN_SCHEMA_VERSION",
    "recover_round74_segmented_terminal_result",
    "run_round74_segmented_terminal_evaluation",
]

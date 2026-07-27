"""Load the complete Round 74 development panel without opening sealed inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .impact_absorption_event_cohort import (
    Round74EventCohortPlan,
    Round74EventCohortRunBinding,
    build_round74_event_run_partition,
    load_round74_event_cohort_binding,
    load_round74_event_cohort_plan,
)
from .impact_absorption_event_dataset import Round74EventRunPartition
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .round74_event_cohort_operator import ROUND74_EVENT_COHORT_PLAN_SHA256
from .round74_target_assembly_manifest import (
    load_and_audit_round74_target_assembly_manifest,
)


ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION = "round-074-development-inputs-v2"
ROUND74_DEVELOPMENT_TRAINING_RUNS = 120
ROUND74_DEVELOPMENT_TUNING_RUNS = 24
ROUND74_SEALED_TEST_RUNS = 24
ROUND74_DEVELOPMENT_EXECUTION_ENVIRONMENT = "binance_usdm_mainnet"


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


@dataclass(frozen=True)
class Round74DevelopmentInputs:
    """Exactly 144 development assemblies bound to a complete 168-run cohort."""

    plan: Round74EventCohortPlan
    bindings: tuple[Round74EventCohortRunBinding, ...]
    partition: Round74EventRunPartition
    target_assemblies: tuple[tuple[str, Round74SourceTargetAssembly], ...]
    target_manifest_sha256_by_run_id: tuple[tuple[str, str], ...]
    schema_version: str = ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION

    def validate(self) -> None:
        self.plan.validate()
        self.partition.validate()
        for binding in self.bindings:
            binding.validate()
        assemblies = dict(self.target_assemblies)
        manifests = dict(self.target_manifest_sha256_by_run_id)
        development_entries = tuple(
            entry for entry in self.partition.entries if entry.role != "test"
        )
        development_run_ids = tuple(entry.run_id for entry in development_entries)
        if (
            self.schema_version != ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION
            or self.plan.plan_sha256 != ROUND74_EVENT_COHORT_PLAN_SHA256
            or (
                self.plan.training_slots,
                self.plan.tuning_slots,
                self.plan.test_slots,
            )
            != (
                ROUND74_DEVELOPMENT_TRAINING_RUNS,
                ROUND74_DEVELOPMENT_TUNING_RUNS,
                ROUND74_SEALED_TEST_RUNS,
            )
            or len(self.bindings) != self.plan.total_slots
            or tuple(binding.slot_ordinal for binding in self.bindings)
            != tuple(range(self.plan.total_slots))
            or any(
                binding.plan_sha256 != self.plan.plan_sha256
                or binding.role != self.plan.role_for_ordinal(binding.slot_ordinal)
                for binding in self.bindings
            )
            or self.partition.cohort_plan_sha256 != self.plan.plan_sha256
            or len(development_entries)
            != ROUND74_DEVELOPMENT_TRAINING_RUNS + ROUND74_DEVELOPMENT_TUNING_RUNS
            or tuple(assemblies) != development_run_ids
            or len(assemblies) != len(self.target_assemblies)
            or tuple(manifests) != development_run_ids
            or len(manifests) != len(self.target_manifest_sha256_by_run_id)
            or any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in manifests.values()
            )
        ):
            raise ValueError("Round 74 development input identity differs")
        for assembly in assemblies.values():
            if (
                not isinstance(assembly, Round74SourceTargetAssembly)
                or assembly.spec.execution_environment
                != ROUND74_DEVELOPMENT_EXECUTION_ENVIRONMENT
            ):
                raise ValueError("Round 74 development target environment differs")

    @property
    def inputs_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan.plan_sha256,
            "binding_sha256": [binding.binding_sha256 for binding in self.bindings],
            "partition_sha256": self.partition.partition_sha256,
            "development_target_assembly_sha256": {
                run_id: assembly.assembly_sha256
                for run_id, assembly in self.target_assemblies
            },
            "development_target_manifest_sha256": dict(
                self.target_manifest_sha256_by_run_id
            ),
            "training_runs": ROUND74_DEVELOPMENT_TRAINING_RUNS,
            "tuning_runs": ROUND74_DEVELOPMENT_TUNING_RUNS,
            "sealed_test_runs": ROUND74_SEALED_TEST_RUNS,
            "execution_environment": ROUND74_DEVELOPMENT_EXECUTION_ENVIRONMENT,
            "sealed_test_target_assemblies_read": False,
            "source_database_access": "not_opened",
            "model_training_started": False,
        }

    def target_assembly_by_run_id(
        self,
    ) -> dict[str, Round74SourceTargetAssembly]:
        self.validate()
        return dict(self.target_assemblies)


def load_round74_development_inputs(
    *,
    plan_path: str | Path,
    binding_directory: str | Path,
    target_assembly_directory: str | Path,
    source_artifact_root: str | Path,
) -> Round74DevelopmentInputs:
    """Load all bindings and deep-audit only development target manifests."""

    selected_plan_path = Path(plan_path)
    selected_binding_directory = Path(binding_directory)
    selected_assembly_directory = Path(target_assembly_directory)
    selected_source_root = Path(source_artifact_root)
    if selected_plan_path.is_symlink() or not selected_plan_path.is_file():
        raise ValueError("Round 74 development plan file differs")
    plan = load_round74_event_cohort_plan(
        selected_plan_path.read_text(encoding="ascii")
    )
    expected_binding_names = {
        f"{ordinal:03d}.json" for ordinal in range(plan.total_slots)
    }
    binding_entries = tuple(selected_binding_directory.iterdir())
    observed_binding_names = {path.name for path in binding_entries}
    if observed_binding_names != expected_binding_names or any(
        path.is_symlink() or not path.is_file() for path in binding_entries
    ):
        raise ValueError("Round 74 development binding file panel differs")
    bindings = tuple(
        load_round74_event_cohort_binding(
            (selected_binding_directory / f"{ordinal:03d}.json").read_text(
                encoding="ascii"
            )
        )
        for ordinal in range(plan.total_slots)
    )
    partition = build_round74_event_run_partition(plan, bindings)
    development_run_ids = tuple(
        entry.run_id for entry in partition.entries if entry.role != "test"
    )
    binding_by_run_id = {binding.run_id: binding for binding in bindings}
    if len(binding_by_run_id) != len(bindings) or set(binding_by_run_id) != {
        entry.run_id for entry in partition.entries
    }:
        raise ValueError("Round 74 development binding run identity differs")
    expected_assembly_names = {f"{run_id}.json" for run_id in development_run_ids}
    assembly_entries = tuple(selected_assembly_directory.iterdir())
    observed_assembly_names = {path.name for path in assembly_entries}
    if observed_assembly_names != expected_assembly_names or any(
        path.is_symlink() or not path.is_file() for path in assembly_entries
    ):
        raise ValueError("Round 74 development target manifest file panel differs")
    loaded = tuple(
        (
            run_id,
            load_and_audit_round74_target_assembly_manifest(
                manifest_path=(selected_assembly_directory / f"{run_id}.json"),
                source_artifact_root=selected_source_root,
            ),
        )
        for run_id in development_run_ids
    )
    if any(
        manifest.run_id != run_id
        or manifest.cohort_binding_sha256 != binding_by_run_id[run_id].binding_sha256
        for run_id, manifest in loaded
    ):
        raise ValueError("Round 74 development target manifest binding differs")
    assemblies = tuple((run_id, manifest.assembly) for run_id, manifest in loaded)
    selected = Round74DevelopmentInputs(
        plan=plan,
        bindings=bindings,
        partition=partition,
        target_assemblies=assemblies,
        target_manifest_sha256_by_run_id=tuple(
            (run_id, manifest.manifest_sha256) for run_id, manifest in loaded
        ),
    )
    selected.validate()
    return selected


__all__ = [
    "ROUND74_DEVELOPMENT_EXECUTION_ENVIRONMENT",
    "ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION",
    "ROUND74_DEVELOPMENT_TRAINING_RUNS",
    "ROUND74_DEVELOPMENT_TUNING_RUNS",
    "ROUND74_SEALED_TEST_RUNS",
    "Round74DevelopmentInputs",
    "load_round74_development_inputs",
]

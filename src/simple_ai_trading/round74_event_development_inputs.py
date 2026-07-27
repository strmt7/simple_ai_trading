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


ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION = "round-074-development-inputs-v1"
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


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key, value in pairs:
        if key in selected:
            raise ValueError(f"Round 74 development input duplicate key: {key}")
        selected[key] = value
    return selected


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"Round 74 development input non-finite value: {value}")


def _strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Round 74 {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 74 {label} root differs")
    return value


@dataclass(frozen=True)
class Round74DevelopmentInputs:
    """Exactly 144 development assemblies bound to a complete 168-run cohort."""

    plan: Round74EventCohortPlan
    bindings: tuple[Round74EventCohortRunBinding, ...]
    partition: Round74EventRunPartition
    target_assemblies: tuple[tuple[str, Round74SourceTargetAssembly], ...]
    schema_version: str = ROUND74_DEVELOPMENT_INPUTS_SCHEMA_VERSION

    def validate(self) -> None:
        self.plan.validate()
        self.partition.validate()
        for binding in self.bindings:
            binding.validate()
        assemblies = dict(self.target_assemblies)
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
) -> Round74DevelopmentInputs:
    """Load all cohort bindings and only development target assemblies."""

    selected_plan_path = Path(plan_path)
    selected_binding_directory = Path(binding_directory)
    selected_assembly_directory = Path(target_assembly_directory)
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
    expected_assembly_names = {f"{run_id}.json" for run_id in development_run_ids}
    assembly_entries = tuple(selected_assembly_directory.iterdir())
    observed_assembly_names = {path.name for path in assembly_entries}
    if observed_assembly_names != expected_assembly_names or any(
        path.is_symlink() or not path.is_file() for path in assembly_entries
    ):
        raise ValueError("Round 74 development target assembly file panel differs")
    assemblies = tuple(
        (
            run_id,
            Round74SourceTargetAssembly.from_dict(
                _strict_json_object(
                    selected_assembly_directory / f"{run_id}.json",
                    label="development target assembly",
                )
            ),
        )
        for run_id in development_run_ids
    )
    selected = Round74DevelopmentInputs(
        plan=plan,
        bindings=bindings,
        partition=partition,
        target_assemblies=assemblies,
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

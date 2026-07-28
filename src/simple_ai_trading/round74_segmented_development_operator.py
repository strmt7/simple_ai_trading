"""Disk-free preparation and training orchestration for segmented Round 74 data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from .impact_absorption_ai_execution_replay import (
    Round74AIQualificationStoreExecutionReplayProvider,
)
from .impact_absorption_event_action_policy import ROUND74_ACTION_PROFILES
from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
)
from .impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
)
from .impact_absorption_event_training import Round74EventTrainingConfig
from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .round74_ai_qualification_operator import (
    Round74AIQualificationOperatorResult,
    run_round74_prepared_ai_pretest_qualification,
)
from .round74_event_development_operator import (
    Round74DevelopmentPolicyArtifact,
    train_calibrate_and_select_round74_development_policy,
)
from .round74_event_model_operator import (
    Round74PreparedDevelopmentData,
    Round74PreparedTuningRoles,
    split_round74_prepared_tuning_roles,
)
from .round74_segmented_model_operator import (
    Round74SegmentedTrainingSplit,
    Round74SegmentedTuningSubpartition,
    ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
    assemble_round74_segmented_role_batches,
    build_round74_segmented_ai_qualification_population,
    build_round74_segmented_training_split,
    build_round74_segmented_tuning_subpartition,
    fit_round74_segmented_optimization_feature_scaler,
    round74_segmented_window_policy,
)


ROUND74_SEGMENTED_DEVELOPMENT_PREPARATION_SCHEMA_VERSION = (
    "round-074-segmented-development-preparation-v1"
)
ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION = (
    "round-074-segmented-qualified-development-v2"
)


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


def _segmented_training_config(
    config: Round74EventTrainingConfig | None,
) -> Round74EventTrainingConfig:
    selected = config or replace(
        Round74EventTrainingConfig(),
        execution_mode="segmented_cohort",
    )
    selected.validate()
    if selected.execution_mode != "segmented_cohort":
        raise ValueError("Round 74 segmented development requires segmented mode")
    return selected


@dataclass(frozen=True)
class Round74SegmentedPreparedDevelopment:
    """Validated in-memory development panel with no sealed-test access."""

    partition_sha256: str
    training_split: Round74SegmentedTrainingSplit
    tuning_subpartition: Round74SegmentedTuningSubpartition
    prepared: Round74PreparedDevelopmentData
    tuning_roles: Round74PreparedTuningRoles
    schema_version: str = ROUND74_SEGMENTED_DEVELOPMENT_PREPARATION_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    overlapping_windows_persisted: bool = False

    def validate(self) -> None:
        self.training_split.validate()
        self.tuning_subpartition.validate()
        self.prepared.validate()
        self.tuning_roles.validate()
        training_run_ids = (
            *self.training_split.optimization_run_ids,
            *self.training_split.purged_run_ids,
            *self.training_split.early_stopping_run_ids,
        )
        tuning_run_ids = (
            *self.tuning_subpartition.model_selection_run_ids,
            *self.tuning_subpartition.calibration_run_ids,
            *self.tuning_subpartition.policy_selection_run_ids,
            *self.tuning_subpartition.ai_qualification_run_ids,
        )
        prepared_training_run_ids = tuple(
            next(iter(set(batch.run_id)))
            for batch in self.prepared.training_batches
        )
        prepared_tuning_run_ids = tuple(
            next(iter(set(batch.run_id)))
            for batch in self.prepared.tuning_batches
        )
        if (
            self.schema_version
            != ROUND74_SEGMENTED_DEVELOPMENT_PREPARATION_SCHEMA_VERSION
            or self.partition_sha256 != self.training_split.parent_partition_sha256
            or self.partition_sha256
            != self.tuning_subpartition.parent_partition_sha256
            or self.tuning_roles.subpartition.subpartition_sha256
            != self.tuning_subpartition.subpartition_sha256
            or prepared_training_run_ids != training_run_ids
            or prepared_tuning_run_ids != tuning_run_ids
            or self.prepared.scaler.fit_source_scope
            != "segmented_optimization_training_runs"
            or self.prepared.scaler.fit_source_run_ids
            != self.training_split.optimization_run_ids
            or self.prepared.scaler.fit_source_partition_sha256
            != self.partition_sha256
            or self.prepared.scaler.fit_source_selection_sha256
            != self.training_split.split_sha256
            or not isinstance(self.sealed_test_accessed, bool)
            or not isinstance(self.overlapping_windows_persisted, bool)
            or self.sealed_test_accessed
            or self.overlapping_windows_persisted
        ):
            raise ValueError("Round 74 segmented development preparation differs")

    @property
    def preparation_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "partition_sha256": self.partition_sha256,
            "training_split_sha256": self.training_split.split_sha256,
            "tuning_subpartition_sha256": (
                self.tuning_subpartition.subpartition_sha256
            ),
            "prepared_development_sha256": self.prepared.preparation_sha256,
            "prepared_tuning_roles_sha256": (
                self.tuning_roles.role_assignment_sha256
            ),
            "feature_scaler_sha256": self.prepared.scaler.scaler_sha256,
            "window_selection_policy": round74_segmented_window_policy(),
            "source_replay_passes": {
                "optimization_training_runs": 2,
                "purged_training_runs": 1,
                "early_stopping_training_runs": 1,
                "tuning_runs": 1,
                "sealed_test_runs": 0,
            },
            "overlapping_windows_persisted": False,
            "sealed_test_accessed": False,
        }
        if include_sha256:
            payload["preparation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class Round74SegmentedQualifiedDevelopment:
    """ML development artifact plus profile-specific local-AI qualifications."""

    preparation_sha256: str
    policy: Round74DevelopmentPolicyArtifact
    requested_profiles: tuple[str, ...]
    qualification_by_profile: tuple[
        tuple[str, Round74AIQualificationOperatorResult],
        ...,
    ]
    schema_version: str = ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        self.policy.bundle.validate()
        profiles = tuple(profile for profile, _result in self.qualification_by_profile)
        results = tuple(result for _profile, result in self.qualification_by_profile)
        policies = {
            policy.profile: policy for policy in self.policy.bundle.action_policies
        }
        expected_qualification_profiles = tuple(
            profile
            for profile in self.requested_profiles
            if profile in policies and policies[profile].accepted
        )
        for result in results:
            result.validate()
        if (
            self.schema_version
            != ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION
            or len(self.preparation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.preparation_sha256
            )
            or not self.requested_profiles
            or len(set(self.requested_profiles)) != len(self.requested_profiles)
            or self.requested_profiles
            != tuple(
                profile
                for profile in ROUND74_ACTION_PROFILES
                if profile in self.requested_profiles
            )
            or len(set(profiles)) != len(profiles)
            or any(profile not in policies for profile in self.requested_profiles)
            or profiles != expected_qualification_profiles
            or any(
                result.inference.action_selection_sha256
                != policies[profile].selection_sha256
                or result.qualification.profile != profile
                for profile, result in self.qualification_by_profile
            )
            or any(
                not isinstance(value, bool)
                for value in (
                    self.sealed_test_accessed,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
            or any(
                (
                    self.sealed_test_accessed,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 segmented qualified development differs")

    @property
    def result_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        qualifications = dict(self.qualification_by_profile)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "preparation_sha256": self.preparation_sha256,
            "development_bundle_sha256": self.policy.bundle_sha256,
            "pretest_policy_sha256": (
                self.policy.bundle.pretest_policy_sha256
            ),
            "profile_results": {
                profile: {
                    "ml_action_policy_accepted": (
                        next(
                            policy
                            for policy in self.policy.bundle.action_policies
                            if policy.profile == profile
                        ).accepted
                    ),
                    "ml_action_policy_rejection_reasons": list(
                        next(
                            policy
                            for policy in self.policy.bundle.action_policies
                            if policy.profile == profile
                        ).rejection_reasons
                    ),
                    "ai_qualification_executed": profile in qualifications,
                    "action_selection_sha256": (
                        next(
                            policy
                            for policy in self.policy.bundle.action_policies
                            if policy.profile == profile
                        ).selection_sha256
                    ),
                    "qualification_sha256": (
                        None
                        if profile not in qualifications
                        else qualifications[
                            profile
                        ].qualification.qualification_sha256
                    ),
                    "qualification_passed": (
                        False
                        if profile not in qualifications
                        else qualifications[
                            profile
                        ].qualification.qualification_passed
                    ),
                }
                for profile in self.requested_profiles
            },
            "sealed_test_accessed": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            payload["result_sha256"] = _canonical_sha256(payload)
        return payload


def prepare_round74_segmented_development(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    scaler_chunk_rows: int = ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
    scaler_maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    window_representation: str = "per_symbol",
) -> Round74SegmentedPreparedDevelopment:
    """Prepare all admitted development runs without reading sealed-test data."""

    partition.validate()
    development_entries = tuple(
        entry for entry in partition.entries if entry.role != "test"
    )
    development_run_ids = tuple(entry.run_id for entry in development_entries)
    bindings = dict(bindings_by_run_id)
    assemblies = dict(target_assembly_by_run_id)
    if (
        set(bindings) != set(development_run_ids)
        or set(assemblies) != set(development_run_ids)
        or any(
            not isinstance(bindings[entry.run_id], Round74SegmentedCohortRunBinding)
            or bindings[entry.run_id].role != entry.role
            for entry in development_entries
        )
        or any(
            not isinstance(assemblies[run_id], Round74SourceTargetAssembly)
            for run_id in development_run_ids
        )
    ):
        raise ValueError("Round 74 segmented development input panel differs")
    training_bindings = {
        entry.run_id: bindings[entry.run_id]
        for entry in development_entries
        if entry.role == "training"
    }
    tuning_bindings = {
        entry.run_id: bindings[entry.run_id]
        for entry in development_entries
        if entry.role == "tuning"
    }
    training_assemblies = {
        run_id: assemblies[run_id] for run_id in training_bindings
    }
    tuning_assemblies = {
        run_id: assemblies[run_id] for run_id in tuning_bindings
    }
    training_split = build_round74_segmented_training_split(
        partition,
        bindings_by_run_id=training_bindings,
    )
    tuning_subpartition = build_round74_segmented_tuning_subpartition(
        partition,
        bindings_by_run_id=tuning_bindings,
    )
    scaler = fit_round74_segmented_optimization_feature_scaler(
        store,
        partition=partition,
        bindings_by_run_id=training_bindings,
        training_split=training_split,
        chunk_rows=scaler_chunk_rows,
        maximum_fit_rows=scaler_maximum_fit_rows,
    )
    training_batches = assemble_round74_segmented_role_batches(
        store,
        partition=partition,
        bindings_by_run_id=training_bindings,
        scaler=scaler,
        role="training",
        target_assembly_by_run_id=training_assemblies,
        window_representation=window_representation,
    )
    tuning_batches = assemble_round74_segmented_role_batches(
        store,
        partition=partition,
        bindings_by_run_id=tuning_bindings,
        scaler=scaler,
        role="tuning",
        target_assembly_by_run_id=tuning_assemblies,
        window_representation=window_representation,
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


def train_round74_segmented_development_policy(
    preparation: Round74SegmentedPreparedDevelopment,
    *,
    store: object,
    partition: Round74EventRunPartition,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
) -> Round74DevelopmentPolicyArtifact:
    """Train, calibrate, and select ML policies from one preparation."""

    preparation.validate()
    partition.validate()
    assemblies = dict(target_assembly_by_run_id)
    development_run_ids = tuple(
        entry.run_id for entry in partition.entries if entry.role != "test"
    )
    if (
        partition.partition_sha256 != preparation.partition_sha256
        or set(assemblies) != set(development_run_ids)
        or any(
            not isinstance(assemblies[run_id], Round74SourceTargetAssembly)
            for run_id in development_run_ids
        )
    ):
        raise ValueError("Round 74 segmented training input panel differs")
    selected_config = _segmented_training_config(config)
    return train_calibrate_and_select_round74_development_policy(
        preparation.prepared,
        preparation.tuning_roles,
        output_directory=output_directory,
        execution_store=store,
        execution_partition=partition,
        execution_target_assembly_by_run_id={
            run_id: assemblies[run_id]
            for run_id in preparation.tuning_subpartition.policy_selection_run_ids
        },
        compute_backend=compute_backend,
        config=selected_config,
        inference_minibatch_rows=inference_minibatch_rows,
        segmented_training_split=preparation.training_split,
    )


def train_and_qualify_round74_segmented_development(
    preparation: Round74SegmentedPreparedDevelopment,
    *,
    store: object,
    partition: Round74EventRunPartition,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    output_directory: str | Path,
    qualification_output_directory: str | Path,
    same_entry_latency_budget_ns: int,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
    qualification_profiles: Sequence[str] = ROUND74_ACTION_PROFILES,
) -> Round74SegmentedQualifiedDevelopment:
    """Run the complete development path without opening sealed-test data."""

    preparation.validate()
    profiles = tuple(str(profile) for profile in qualification_profiles)
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or profiles
        != tuple(profile for profile in ROUND74_ACTION_PROFILES if profile in profiles)
    ):
        raise ValueError("Round 74 segmented qualification profiles differ")
    assemblies = dict(target_assembly_by_run_id)
    policy = train_round74_segmented_development_policy(
        preparation,
        store=store,
        partition=partition,
        target_assembly_by_run_id=assemblies,
        output_directory=output_directory,
        compute_backend=compute_backend,
        config=config,
        inference_minibatch_rows=inference_minibatch_rows,
    )
    population = build_round74_segmented_ai_qualification_population(
        preparation.tuning_subpartition
    )
    replay_provider = Round74AIQualificationStoreExecutionReplayProvider(
        store=store,
        partition=partition,
        qualification_population=population,
        assembly_by_run_id={
            run_id: assemblies[run_id] for run_id in population.run_ids
        },
    )
    action_policies = {
        selected.profile: selected for selected in policy.bundle.action_policies
    }
    qualifiable_profiles = tuple(
        profile for profile in profiles if action_policies[profile].accepted
    )
    qualification_directory = Path(qualification_output_directory)
    results = tuple(
        (
            profile,
            run_round74_prepared_ai_pretest_qualification(
                preparation.tuning_roles,
                qualification_population=population,
                action_selection=action_policies[profile],
                probability_calibration=policy.bundle.probability_calibration,
                pretest_policy_path=policy.pretest_policy.policy_path,
                execution_replay_provider=replay_provider,
                qualification_output_path=(
                    qualification_directory
                    / f"round74-ai-pretest-qualification-{profile}.json"
                ),
                same_entry_latency_budget_ns=same_entry_latency_budget_ns,
                compute_backend=compute_backend,
                inference_minibatch_rows=inference_minibatch_rows,
            ),
        )
        for profile in qualifiable_profiles
    )
    result = Round74SegmentedQualifiedDevelopment(
        preparation_sha256=preparation.preparation_sha256,
        policy=policy,
        requested_profiles=profiles,
        qualification_by_profile=results,
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_SEGMENTED_DEVELOPMENT_PREPARATION_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION",
    "Round74SegmentedPreparedDevelopment",
    "Round74SegmentedQualifiedDevelopment",
    "prepare_round74_segmented_development",
    "train_and_qualify_round74_segmented_development",
    "train_round74_segmented_development_policy",
]

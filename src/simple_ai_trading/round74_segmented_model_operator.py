"""Duration-normalized model preparation for Round 74 transport epochs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
    ROUND74_EVENT_PARTITION_ROLES,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74EventDatasetAssembler,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    Round74EventTrainingBatch,
    Round74LabeledEventWindow,
    Round74MatchedEventDatasetAssembler,
    Round74MatchedEventWindowPair,
    build_round74_event_training_batch,
)
from .impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
    Round74EventFeatureScaler,
    fit_round74_event_feature_scaler_stream,
)
from .impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS,
    ROUND74_SEGMENTED_COHORT_ROLE_COUNTS,
    Round74SegmentedCohortRunBinding,
    iter_round74_v10_segment_event_observations,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly

if TYPE_CHECKING:
    from .impact_absorption_ai_uplift import Round74AIQualificationPopulation
    from .impact_absorption_event_sealed_ledger import Round74SealedDatasetIdentity


ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION = (
    "round-074-segmented-duration-normalized-window-selection-v1"
)
ROUND74_SEGMENTED_MATCHED_ROLE_SCHEMA_VERSION = (
    "round-074-segmented-matched-representation-role-v1"
)
ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS = 3_289_500_000_000
ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL = 256
ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION = (
    "round-074-segmented-tuning-subpartition-v2"
)
ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION = (
    "round-074-segmented-model-selection-stages-v2"
)
ROUND74_SEGMENTED_TRAINING_SPLIT_SCHEMA_VERSION = (
    "round-074-segmented-training-selection-split-v1"
)
ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION = (
    "round-074-segmented-test-population-v1"
)
ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS = (10, 5, 5, 4)
ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHT_TOTAL = sum(
    ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS
)
ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS = (
    "architecture",
    "clock_features",
    "order_flow_features",
    "state_conditioned_flow",
    "causal_pretraining",
)
ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS = 128
ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS = 32
ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR = 8
ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS = 8_192


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _segmented_tuning_subrole_bounds() -> tuple[int, int, int, int, int]:
    training_slots = ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
    tuning_slots = ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["tuning"]
    model_weight, calibration_weight, policy_weight, _ai_weight = (
        ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS
    )
    denominator = ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHT_TOTAL

    def ceiling_fraction(numerator: int) -> int:
        return (tuning_slots * numerator + denominator - 1) // denominator

    model_end = training_slots + ceiling_fraction(model_weight)
    calibration_end = training_slots + ceiling_fraction(
        model_weight + calibration_weight
    )
    policy_end = training_slots + ceiling_fraction(
        model_weight + calibration_weight + policy_weight
    )
    return (
        training_slots,
        model_end,
        calibration_end,
        policy_end,
        training_slots + tuning_slots,
    )


def _segmented_tuning_required_anchor_ns() -> tuple[int, int, int, int]:
    required = ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS["tuning"]
    denominator = ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHT_TOTAL
    return tuple(
        required * weight // denominator
        for weight in ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS
    )


def _segmented_model_selection_stage_bounds() -> tuple[int, ...]:
    model_start, model_end, _calibration_end, _policy_end, _tuning_end = (
        _segmented_tuning_subrole_bounds()
    )
    scheduled_model_slots = model_end - model_start
    stage_count = len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
    return tuple(
        model_start + (scheduled_model_slots * index + stage_count - 1) // stage_count
        for index in range(stage_count + 1)
    )


def _segmented_model_selection_stage_required_anchor_ns() -> int:
    model_required_anchor_ns = _segmented_tuning_required_anchor_ns()[0]
    stage_count = len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
    return (model_required_anchor_ns + stage_count - 1) // stage_count


@dataclass(frozen=True)
class Round74SegmentedTrainingSplit:
    """Target-blind optimizer, purge, and early-stop run assignment."""

    parent_partition_sha256: str
    cohort_plan_sha256: str
    optimization_run_ids: tuple[str, ...]
    purged_run_ids: tuple[str, ...]
    early_stopping_run_ids: tuple[str, ...]
    optimization_slot_ordinals: tuple[int, ...]
    purged_slot_ordinals: tuple[int, ...]
    early_stopping_slot_ordinals: tuple[int, ...]
    optimization_last_eligible_anchor_wall_ns: int
    early_stopping_first_eligible_anchor_wall_ns: int
    schema_version: str = ROUND74_SEGMENTED_TRAINING_SPLIT_SCHEMA_VERSION

    def validate(self) -> None:
        run_groups = (
            self.optimization_run_ids,
            self.purged_run_ids,
            self.early_stopping_run_ids,
        )
        ordinal_groups = (
            self.optimization_slot_ordinals,
            self.purged_slot_ordinals,
            self.early_stopping_slot_ordinals,
        )
        all_runs = tuple(run_id for group in run_groups for run_id in group)
        all_ordinals = tuple(ordinal for group in ordinal_groups for ordinal in group)
        chronological_gap_ns = int(
            self.early_stopping_first_eligible_anchor_wall_ns
        ) - int(self.optimization_last_eligible_anchor_wall_ns)
        if (
            self.schema_version != ROUND74_SEGMENTED_TRAINING_SPLIT_SCHEMA_VERSION
            or not _is_sha256(self.parent_partition_sha256)
            or not _is_sha256(self.cohort_plan_sha256)
            or len(self.optimization_run_ids)
            < ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            or len(self.early_stopping_run_ids)
            < ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            or any(
                len(run_ids) != len(ordinals)
                for run_ids, ordinals in zip(
                    run_groups,
                    ordinal_groups,
                    strict=True,
                )
            )
            or any(
                len(run_id) != 32
                or any(character not in "0123456789abcdef" for character in run_id)
                for run_id in all_runs
            )
            or len(set(all_runs)) != len(all_runs)
            or any(
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or not 0 <= ordinal < ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
                for ordinal in all_ordinals
            )
            or len(set(all_ordinals)) != len(all_ordinals)
            or any(
                current <= prior
                for prior, current in zip(
                    all_ordinals,
                    all_ordinals[1:],
                    strict=False,
                )
            )
            or chronological_gap_ns < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
        ):
            raise ValueError("Round 74 segmented training split differs")

    @property
    def split_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_partition_sha256": self.parent_partition_sha256,
            "cohort_plan_sha256": self.cohort_plan_sha256,
            "assignment_basis": (
                "chronological admitted training runs before feature or target replay"
            ),
            "split_unit": "whole_admitted_transport_segment",
            "early_stopping_fraction_denominator": (
                ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR
            ),
            "minimum_optimization_run_count": (
                ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            ),
            "minimum_early_stopping_run_count": (
                ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            ),
            "optimization_run_ids": list(self.optimization_run_ids),
            "purged_run_ids": list(self.purged_run_ids),
            "early_stopping_run_ids": list(self.early_stopping_run_ids),
            "optimization_slot_ordinals": list(self.optimization_slot_ordinals),
            "purged_slot_ordinals": list(self.purged_slot_ordinals),
            "early_stopping_slot_ordinals": list(self.early_stopping_slot_ordinals),
            "optimization_last_eligible_anchor_wall_ns": (
                self.optimization_last_eligible_anchor_wall_ns
            ),
            "early_stopping_first_eligible_anchor_wall_ns": (
                self.early_stopping_first_eligible_anchor_wall_ns
            ),
            "chronological_gap_ns": (
                self.early_stopping_first_eligible_anchor_wall_ns
                - self.optimization_last_eligible_anchor_wall_ns
            ),
            "minimum_chronological_gap_ns": (ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS),
            "feature_value_target_label_or_model_output_used_for_assignment": False,
            "all_admitted_training_segments_included": True,
            "early_stopping_run_used_for_scaler_fit": False,
            "purged_run_used_for_scaler_fit": False,
        }
        if include_sha256:
            payload["split_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74SegmentedTrainingSplit:
        payload = dict(value)
        claimed = payload.pop("split_sha256", None)
        run_keys = (
            "optimization_run_ids",
            "purged_run_ids",
            "early_stopping_run_ids",
        )
        ordinal_keys = (
            "optimization_slot_ordinals",
            "purged_slot_ordinals",
            "early_stopping_slot_ordinals",
        )
        if (
            not _is_sha256(claimed)
            or claimed != _canonical_sha256(payload)
            or any(
                not isinstance(payload.get(key), list)
                or any(not isinstance(item, str) for item in payload[key])
                for key in run_keys
            )
            or any(
                not isinstance(payload.get(key), list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in payload[key]
                )
                for key in ordinal_keys
            )
        ):
            raise ValueError("Round 74 segmented training split payload differs")
        try:
            selected = cls(
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                cohort_plan_sha256=str(payload["cohort_plan_sha256"]),
                optimization_run_ids=tuple(payload["optimization_run_ids"]),
                purged_run_ids=tuple(payload["purged_run_ids"]),
                early_stopping_run_ids=tuple(payload["early_stopping_run_ids"]),
                optimization_slot_ordinals=tuple(payload["optimization_slot_ordinals"]),
                purged_slot_ordinals=tuple(payload["purged_slot_ordinals"]),
                early_stopping_slot_ordinals=tuple(
                    payload["early_stopping_slot_ordinals"]
                ),
                optimization_last_eligible_anchor_wall_ns=int(
                    payload["optimization_last_eligible_anchor_wall_ns"]
                ),
                early_stopping_first_eligible_anchor_wall_ns=int(
                    payload["early_stopping_first_eligible_anchor_wall_ns"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented training split payload differs"
            ) from exc
        selected.validate()
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented training split policy differs")
        if selected.split_sha256 != claimed:
            raise ValueError("Round 74 segmented training split identity differs")
        return selected


@dataclass(frozen=True)
class Round74SegmentedTuningSubpartition:
    """Target-blind tuning roles inherited from immutable scheduled slot ranges."""

    parent_partition_sha256: str
    cohort_plan_sha256: str
    model_selection_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    policy_selection_run_ids: tuple[str, ...]
    ai_qualification_run_ids: tuple[str, ...]
    model_selection_slot_ordinals: tuple[int, ...]
    calibration_slot_ordinals: tuple[int, ...]
    policy_selection_slot_ordinals: tuple[int, ...]
    ai_qualification_slot_ordinals: tuple[int, ...]
    model_selection_eligible_anchor_ns: tuple[int, ...]
    calibration_eligible_anchor_ns: tuple[int, ...]
    policy_selection_eligible_anchor_ns: tuple[int, ...]
    ai_qualification_eligible_anchor_ns: tuple[int, ...]
    schema_version: str = ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION
    optimization_population: str = "eligible_target"

    def validate(self) -> None:
        run_groups = (
            self.model_selection_run_ids,
            self.calibration_run_ids,
            self.policy_selection_run_ids,
            self.ai_qualification_run_ids,
        )
        ordinal_groups = (
            self.model_selection_slot_ordinals,
            self.calibration_slot_ordinals,
            self.policy_selection_slot_ordinals,
            self.ai_qualification_slot_ordinals,
        )
        duration_groups = (
            self.model_selection_eligible_anchor_ns,
            self.calibration_eligible_anchor_ns,
            self.policy_selection_eligible_anchor_ns,
            self.ai_qualification_eligible_anchor_ns,
        )
        bounds = _segmented_tuning_subrole_bounds()
        expected_ranges = (
            range(bounds[0], bounds[1]),
            range(bounds[1], bounds[2]),
            range(bounds[2], bounds[3]),
            range(bounds[3], bounds[4]),
        )
        all_runs = tuple(value for group in run_groups for value in group)
        all_ordinals = tuple(value for group in ordinal_groups for value in group)
        required_durations = _segmented_tuning_required_anchor_ns()
        if (
            self.schema_version != ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION
            or self.optimization_population != "eligible_target"
            or not _is_sha256(self.parent_partition_sha256)
            or not _is_sha256(self.cohort_plan_sha256)
            or any(
                not group or len(group) != len(ordinals) or len(group) != len(durations)
                for group, ordinals, durations in zip(
                    run_groups,
                    ordinal_groups,
                    duration_groups,
                    strict=True,
                )
            )
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for value in all_runs
            )
            or len(set(all_runs)) != len(all_runs)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in all_ordinals
            )
            or len(set(all_ordinals)) != len(all_ordinals)
            or any(
                current <= prior
                for prior, current in zip(all_ordinals, all_ordinals[1:])
            )
            or any(
                ordinal not in expected
                for ordinals, expected in zip(
                    ordinal_groups,
                    expected_ranges,
                    strict=True,
                )
                for ordinal in ordinals
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for group in duration_groups
                for value in group
            )
            or any(
                sum(group) < required
                for group, required in zip(
                    duration_groups,
                    required_durations,
                    strict=True,
                )
            )
        ):
            raise ValueError("Round 74 segmented tuning subpartition differs")

    @property
    def subpartition_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        bounds = _segmented_tuning_subrole_bounds()
        required = _segmented_tuning_required_anchor_ns()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_partition_sha256": self.parent_partition_sha256,
            "cohort_plan_sha256": self.cohort_plan_sha256,
            "optimization_population": self.optimization_population,
            "split_unit": "whole_admitted_transport_segment",
            "assignment_basis": "immutable_scheduled_slot_ordinal_ranges",
            "scheduled_slot_bounds": list(bounds),
            "scheduled_subrole_counts": [
                bounds[1] - bounds[0],
                bounds[2] - bounds[1],
                bounds[3] - bounds[2],
                bounds[4] - bounds[3],
            ],
            "required_eligible_anchor_ns": list(required),
            "model_selection_run_ids": list(self.model_selection_run_ids),
            "calibration_run_ids": list(self.calibration_run_ids),
            "policy_selection_run_ids": list(self.policy_selection_run_ids),
            "ai_qualification_run_ids": list(self.ai_qualification_run_ids),
            "model_selection_slot_ordinals": list(self.model_selection_slot_ordinals),
            "calibration_slot_ordinals": list(self.calibration_slot_ordinals),
            "policy_selection_slot_ordinals": list(self.policy_selection_slot_ordinals),
            "ai_qualification_slot_ordinals": list(self.ai_qualification_slot_ordinals),
            "model_selection_eligible_anchor_ns": list(
                self.model_selection_eligible_anchor_ns
            ),
            "calibration_eligible_anchor_ns": list(self.calibration_eligible_anchor_ns),
            "policy_selection_eligible_anchor_ns": list(
                self.policy_selection_eligible_anchor_ns
            ),
            "ai_qualification_eligible_anchor_ns": list(
                self.ai_qualification_eligible_anchor_ns
            ),
            "observed_eligible_anchor_ns": [
                sum(self.model_selection_eligible_anchor_ns),
                sum(self.calibration_eligible_anchor_ns),
                sum(self.policy_selection_eligible_anchor_ns),
                sum(self.ai_qualification_eligible_anchor_ns),
            ],
            "chronological": True,
            "random_row_split_permitted": False,
            "all_admitted_tuning_segments_included": True,
            "cross_subrole_run_reuse_permitted": False,
            "sealed_test_run_accessed": False,
        }
        if include_sha256:
            payload["subpartition_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74SegmentedTuningSubpartition:
        payload = dict(value)
        claimed = payload.pop("subpartition_sha256", None)
        if not _is_sha256(claimed) or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented tuning subpartition digest differs")
        run_keys = (
            "model_selection_run_ids",
            "calibration_run_ids",
            "policy_selection_run_ids",
            "ai_qualification_run_ids",
        )
        integer_keys = (
            "model_selection_slot_ordinals",
            "calibration_slot_ordinals",
            "policy_selection_slot_ordinals",
            "ai_qualification_slot_ordinals",
            "model_selection_eligible_anchor_ns",
            "calibration_eligible_anchor_ns",
            "policy_selection_eligible_anchor_ns",
            "ai_qualification_eligible_anchor_ns",
        )
        if any(
            not isinstance(payload.get(key), list)
            or any(not isinstance(item, str) for item in payload[key])
            for key in run_keys
        ) or any(
            not isinstance(payload.get(key), list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in payload[key]
            )
            for key in integer_keys
        ):
            raise ValueError("Round 74 segmented tuning subpartition types differ")
        try:
            selected = cls(
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                cohort_plan_sha256=str(payload["cohort_plan_sha256"]),
                model_selection_run_ids=tuple(payload["model_selection_run_ids"]),
                calibration_run_ids=tuple(payload["calibration_run_ids"]),
                policy_selection_run_ids=tuple(payload["policy_selection_run_ids"]),
                ai_qualification_run_ids=tuple(payload["ai_qualification_run_ids"]),
                model_selection_slot_ordinals=tuple(
                    payload["model_selection_slot_ordinals"]
                ),
                calibration_slot_ordinals=tuple(payload["calibration_slot_ordinals"]),
                policy_selection_slot_ordinals=tuple(
                    payload["policy_selection_slot_ordinals"]
                ),
                ai_qualification_slot_ordinals=tuple(
                    payload["ai_qualification_slot_ordinals"]
                ),
                model_selection_eligible_anchor_ns=tuple(
                    payload["model_selection_eligible_anchor_ns"]
                ),
                calibration_eligible_anchor_ns=tuple(
                    payload["calibration_eligible_anchor_ns"]
                ),
                policy_selection_eligible_anchor_ns=tuple(
                    payload["policy_selection_eligible_anchor_ns"]
                ),
                ai_qualification_eligible_anchor_ns=tuple(
                    payload["ai_qualification_eligible_anchor_ns"]
                ),
                schema_version=str(payload["schema_version"]),
                optimization_population=str(payload["optimization_population"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented tuning subpartition payload differs"
            ) from exc
        selected.validate()
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented tuning subpartition policy differs")
        if selected.subpartition_sha256 != claimed:
            raise ValueError("Round 74 segmented tuning subpartition identity differs")
        return selected


@dataclass(frozen=True)
class Round74SegmentedModelSelectionStages:
    """Five target-blind promotion panels fixed by scheduled slot ordinal."""

    parent_tuning_subpartition_sha256: str
    parent_partition_sha256: str
    cohort_plan_sha256: str
    stage_run_ids: tuple[tuple[str, ...], ...]
    stage_slot_ordinals: tuple[tuple[int, ...], ...]
    stage_eligible_anchor_ns: tuple[tuple[int, ...], ...]
    schema_version: str = ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION

    def validate(self) -> None:
        stage_count = len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
        bounds = _segmented_model_selection_stage_bounds()
        required_anchor_ns = _segmented_model_selection_stage_required_anchor_ns()
        if (
            self.schema_version
            != ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION
            or not _is_sha256(self.parent_tuning_subpartition_sha256)
            or not _is_sha256(self.parent_partition_sha256)
            or not _is_sha256(self.cohort_plan_sha256)
            or len(self.stage_run_ids) != stage_count
            or len(self.stage_slot_ordinals) != stage_count
            or len(self.stage_eligible_anchor_ns) != stage_count
        ):
            raise ValueError("Round 74 segmented model-selection stages differ")
        all_runs: list[str] = []
        all_ordinals: list[int] = []
        for index, (run_ids, ordinals, durations) in enumerate(
            zip(
                self.stage_run_ids,
                self.stage_slot_ordinals,
                self.stage_eligible_anchor_ns,
                strict=True,
            )
        ):
            if (
                not run_ids
                or len(run_ids) != len(ordinals)
                or len(run_ids) != len(durations)
                or any(
                    len(run_id) != 32
                    or any(character not in "0123456789abcdef" for character in run_id)
                    for run_id in run_ids
                )
                or any(
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or not bounds[index] <= ordinal < bounds[index + 1]
                    for ordinal in ordinals
                )
                or any(
                    isinstance(duration, bool)
                    or not isinstance(duration, int)
                    or duration <= 0
                    for duration in durations
                )
                or sum(durations) < required_anchor_ns
            ):
                raise ValueError("Round 74 segmented model-selection stage differs")
            all_runs.extend(run_ids)
            all_ordinals.extend(ordinals)
        if (
            len(set(all_runs)) != len(all_runs)
            or len(set(all_ordinals)) != len(all_ordinals)
            or any(
                current <= prior
                for prior, current in zip(
                    all_ordinals,
                    all_ordinals[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("Round 74 segmented model-selection reuse differs")

    @property
    def stage_partition_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def batches_for_stage(
        self,
        stage_id: str,
        batches_by_run_id: Mapping[str, Round74EventTrainingBatch],
    ) -> tuple[Round74EventTrainingBatch, ...]:
        self.validate()
        try:
            stage_index = ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS.index(
                str(stage_id)
            )
        except ValueError as exc:
            raise ValueError(
                "Round 74 segmented model-selection stage id differs"
            ) from exc
        selected = dict(batches_by_run_id)
        expected = {run_id for run_ids in self.stage_run_ids for run_id in run_ids}
        if set(selected) != expected:
            raise ValueError("Round 74 segmented model-selection batch panel differs")
        batches = tuple(selected[run_id] for run_id in self.stage_run_ids[stage_index])
        if any(
            batch.role != "tuning"
            or set(batch.run_id) != {run_id}
            or batch.partition_sha256 != self.parent_partition_sha256
            for batch, run_id in zip(
                batches,
                self.stage_run_ids[stage_index],
                strict=True,
            )
        ):
            raise ValueError(
                "Round 74 segmented model-selection batch identity differs"
            )
        return batches

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        bounds = _segmented_model_selection_stage_bounds()
        required_anchor_ns = _segmented_model_selection_stage_required_anchor_ns()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_tuning_subpartition_sha256": (
                self.parent_tuning_subpartition_sha256
            ),
            "parent_partition_sha256": self.parent_partition_sha256,
            "cohort_plan_sha256": self.cohort_plan_sha256,
            "assignment_basis": "immutable_scheduled_slot_ordinal_ranges",
            "split_unit": "whole_admitted_transport_segment",
            "stage_order": list(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS),
            "scheduled_slot_bounds": list(bounds),
            "required_eligible_anchor_ns_per_stage": required_anchor_ns,
            "stages": [
                {
                    "stage_id": stage_id,
                    "run_ids": list(run_ids),
                    "slot_ordinals": list(ordinals),
                    "eligible_anchor_ns": list(durations),
                    "observed_eligible_anchor_ns": sum(durations),
                }
                for stage_id, run_ids, ordinals, durations in zip(
                    ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS,
                    self.stage_run_ids,
                    self.stage_slot_ordinals,
                    self.stage_eligible_anchor_ns,
                    strict=True,
                )
            ],
            "chronological": True,
            "target_label_or_model_output_used_for_assignment": False,
            "cross_stage_run_reuse_permitted": False,
            "all_parent_model_selection_segments_included": True,
            "calibration_or_policy_selection_segment_included": False,
            "ai_qualification_segment_included": False,
            "sealed_test_segment_included": False,
        }
        if include_sha256:
            payload["stage_partition_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74SegmentedModelSelectionStages:
        payload = dict(value)
        claimed = payload.pop("stage_partition_sha256", None)
        stages = payload.get("stages")
        if (
            not _is_sha256(claimed)
            or claimed != _canonical_sha256(payload)
            or not isinstance(stages, list)
            or len(stages) != len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
            or any(not isinstance(stage, Mapping) for stage in stages)
        ):
            raise ValueError("Round 74 segmented model-selection stage payload differs")
        try:
            selected = cls(
                parent_tuning_subpartition_sha256=str(
                    payload["parent_tuning_subpartition_sha256"]
                ),
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                cohort_plan_sha256=str(payload["cohort_plan_sha256"]),
                stage_run_ids=tuple(
                    tuple(str(run_id) for run_id in stage["run_ids"])
                    for stage in stages
                ),
                stage_slot_ordinals=tuple(
                    tuple(int(ordinal) for ordinal in stage["slot_ordinals"])
                    for stage in stages
                ),
                stage_eligible_anchor_ns=tuple(
                    tuple(int(duration) for duration in stage["eligible_anchor_ns"])
                    for stage in stages
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented model-selection stage payload differs"
            ) from exc
        selected.validate()
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented model-selection stage policy differs")
        if selected.stage_partition_sha256 != claimed:
            raise ValueError(
                "Round 74 segmented model-selection stage identity differs"
            )
        return selected


@dataclass(frozen=True)
class Round74SegmentedTestPopulation:
    """All admitted sealed-test segments bound to their scheduled slots."""

    parent_partition_sha256: str
    cohort_plan_sha256: str
    test_run_ids: tuple[str, ...]
    test_slot_ordinals: tuple[int, ...]
    test_eligible_anchor_ns: tuple[int, ...]
    schema_version: str = ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION
    optimization_population: str = "eligible_target"

    def validate(self) -> None:
        test_start = (
            ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
            + ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["tuning"]
        )
        test_end = test_start + ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["test"]
        if (
            self.schema_version != ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION
            or self.optimization_population != "eligible_target"
            or not _is_sha256(self.parent_partition_sha256)
            or not _is_sha256(self.cohort_plan_sha256)
            or not self.test_run_ids
            or len(self.test_run_ids) != len(self.test_slot_ordinals)
            or len(self.test_run_ids) != len(self.test_eligible_anchor_ns)
            or len(set(self.test_run_ids)) != len(self.test_run_ids)
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.test_run_ids
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.test_slot_ordinals
            )
            or any(
                current <= prior
                for prior, current in zip(
                    self.test_slot_ordinals,
                    self.test_slot_ordinals[1:],
                )
            )
            or any(
                not test_start <= value < test_end for value in self.test_slot_ordinals
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.test_eligible_anchor_ns
            )
            or sum(self.test_eligible_anchor_ns)
            < ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS["test"]
        ):
            raise ValueError("Round 74 segmented test population differs")

    @property
    def population_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        test_start = (
            ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
            + ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["tuning"]
        )
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_partition_sha256": self.parent_partition_sha256,
            "cohort_plan_sha256": self.cohort_plan_sha256,
            "optimization_population": self.optimization_population,
            "assignment_basis": "all_admitted_immutable_scheduled_test_slots",
            "scheduled_slot_bounds": [
                test_start,
                test_start + ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["test"],
            ],
            "scheduled_test_slot_count": ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["test"],
            "required_eligible_anchor_ns": (
                ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS["test"]
            ),
            "test_run_ids": list(self.test_run_ids),
            "test_slot_ordinals": list(self.test_slot_ordinals),
            "test_eligible_anchor_ns": list(self.test_eligible_anchor_ns),
            "observed_eligible_anchor_ns": sum(self.test_eligible_anchor_ns),
            "all_admitted_test_segments_included": True,
            "test_segment_selection_permitted": False,
            "development_target_accessed": False,
        }
        if include_sha256:
            payload["population_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74SegmentedTestPopulation:
        payload = dict(value)
        claimed = payload.pop("population_sha256", None)
        if not _is_sha256(claimed) or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 segmented test population digest differs")
        run_ids = payload.get("test_run_ids")
        ordinals = payload.get("test_slot_ordinals")
        durations = payload.get("test_eligible_anchor_ns")
        if (
            not isinstance(run_ids, list)
            or any(not isinstance(value, str) for value in run_ids)
            or not isinstance(ordinals, list)
            or not isinstance(durations, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (*ordinals, *durations)
            )
        ):
            raise ValueError("Round 74 segmented test population types differ")
        try:
            selected = cls(
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                cohort_plan_sha256=str(payload["cohort_plan_sha256"]),
                test_run_ids=tuple(run_ids),
                test_slot_ordinals=tuple(ordinals),
                test_eligible_anchor_ns=tuple(durations),
                schema_version=str(payload["schema_version"]),
                optimization_population=str(payload["optimization_population"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 segmented test population payload differs"
            ) from exc
        selected.validate()
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 segmented test population policy differs")
        if selected.population_sha256 != claimed:
            raise ValueError("Round 74 segmented test population identity differs")
        return selected


def build_round74_segmented_training_split(
    partition: Round74EventRunPartition,
    *,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
) -> Round74SegmentedTrainingSplit:
    """Assign training runs before replaying features, targets, or model output."""

    partition.validate()
    entries = tuple(entry for entry in partition.entries if entry.role == "training")
    bindings = dict(bindings_by_run_id)
    expected_run_ids = {entry.run_id for entry in entries}
    if set(bindings) != expected_run_ids:
        raise ValueError("Round 74 segmented training binding panel differs")
    slot_ordinals: list[int] = []
    for entry in entries:
        binding = bindings[entry.run_id]
        binding.validate()
        if (
            binding.plan_sha256 != partition.cohort_plan_sha256
            or binding.role != "training"
            or binding.run_id != entry.run_id
            or binding.report_sha256 != entry.capture_report_sha256
            or binding.feature_ready_wall_ns != entry.capture_start_wall_ns
            or binding.usable_end_wall_ns != entry.capture_end_wall_ns
            or not 0
            <= binding.slot_ordinal
            < ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
        ):
            raise ValueError("Round 74 segmented training binding differs")
        slot_ordinals.append(binding.slot_ordinal)
    admitted_count = len(entries)
    early_stopping_count = max(
        ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS,
        (admitted_count + ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR - 1)
        // ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR,
    )
    if (
        admitted_count
        < ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS + early_stopping_count
    ):
        raise ValueError(
            "Round 74 segmented training role is too small for isolated early stopping"
        )
    early_start = admitted_count - early_stopping_count
    optimization_end = early_start
    while (
        optimization_end > ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
        and entries[early_start].eligible_anchor_start_wall_ns
        - entries[optimization_end - 1].eligible_anchor_end_wall_ns
        < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    ):
        optimization_end -= 1
    if (
        entries[early_start].eligible_anchor_start_wall_ns
        - entries[optimization_end - 1].eligible_anchor_end_wall_ns
        < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    ):
        raise ValueError("Round 74 segmented training early-stop purge is too short")
    selected = Round74SegmentedTrainingSplit(
        parent_partition_sha256=partition.partition_sha256,
        cohort_plan_sha256=partition.cohort_plan_sha256,
        optimization_run_ids=tuple(
            entry.run_id for entry in entries[:optimization_end]
        ),
        purged_run_ids=tuple(
            entry.run_id for entry in entries[optimization_end:early_start]
        ),
        early_stopping_run_ids=tuple(entry.run_id for entry in entries[early_start:]),
        optimization_slot_ordinals=tuple(slot_ordinals[:optimization_end]),
        purged_slot_ordinals=tuple(slot_ordinals[optimization_end:early_start]),
        early_stopping_slot_ordinals=tuple(slot_ordinals[early_start:]),
        optimization_last_eligible_anchor_wall_ns=(
            entries[optimization_end - 1].eligible_anchor_end_wall_ns
        ),
        early_stopping_first_eligible_anchor_wall_ns=(
            entries[early_start].eligible_anchor_start_wall_ns
        ),
    )
    selected.validate()
    if (
        *selected.optimization_run_ids,
        *selected.purged_run_ids,
        *selected.early_stopping_run_ids,
    ) != tuple(entry.run_id for entry in entries):
        raise RuntimeError("Round 74 segmented training split coverage differs")
    return selected


def build_round74_segmented_test_population(
    partition: Round74EventRunPartition,
    *,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
) -> Round74SegmentedTestPopulation:
    """Bind every admitted sealed-test segment; no subset is accepted."""

    partition.validate()
    entries = tuple(entry for entry in partition.entries if entry.role == "test")
    bindings = dict(bindings_by_run_id)
    if set(bindings) != {entry.run_id for entry in entries}:
        raise ValueError("Round 74 segmented test binding panel differs")
    run_ids: list[str] = []
    ordinals: list[int] = []
    durations: list[int] = []
    for entry in entries:
        binding = bindings[entry.run_id]
        binding.validate()
        if (
            binding.plan_sha256 != partition.cohort_plan_sha256
            or binding.role != "test"
            or binding.run_id != entry.run_id
            or binding.report_sha256 != entry.capture_report_sha256
        ):
            raise ValueError("Round 74 segmented test binding differs")
        run_ids.append(entry.run_id)
        ordinals.append(binding.slot_ordinal)
        durations.append(
            entry.eligible_anchor_end_wall_ns - entry.eligible_anchor_start_wall_ns
        )
    selected = Round74SegmentedTestPopulation(
        parent_partition_sha256=partition.partition_sha256,
        cohort_plan_sha256=partition.cohort_plan_sha256,
        test_run_ids=tuple(run_ids),
        test_slot_ordinals=tuple(ordinals),
        test_eligible_anchor_ns=tuple(durations),
    )
    selected.validate()
    return selected


def build_round74_segmented_sealed_dataset_identity(
    test_batches: Iterable[Round74EventTrainingBatch],
    *,
    test_population: Round74SegmentedTestPopulation,
) -> Round74SealedDatasetIdentity:
    """Build a sealed identity only when batches equal the frozen test population."""

    from .impact_absorption_event_sealed_ledger import (
        build_round74_sealed_dataset_identity,
    )

    test_population.validate()
    return build_round74_sealed_dataset_identity(
        tuple(test_batches),
        optimization_population=test_population.optimization_population,
        expected_test_run_ids=test_population.test_run_ids,
        test_population_sha256=test_population.population_sha256,
    )


def build_round74_segmented_tuning_subpartition(
    partition: Round74EventRunPartition,
    *,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
) -> Round74SegmentedTuningSubpartition:
    """Assign every admitted tuning segment by its frozen scheduled ordinal."""

    partition.validate()
    entries = tuple(entry for entry in partition.entries if entry.role == "tuning")
    bindings = dict(bindings_by_run_id)
    expected_run_ids = {entry.run_id for entry in entries}
    if set(bindings) != expected_run_ids:
        raise ValueError("Round 74 segmented tuning binding panel differs")
    bounds = _segmented_tuning_subrole_bounds()
    run_groups: list[list[str]] = [[], [], [], []]
    ordinal_groups: list[list[int]] = [[], [], [], []]
    duration_groups: list[list[int]] = [[], [], [], []]
    for entry in entries:
        binding = bindings[entry.run_id]
        binding.validate()
        if (
            binding.plan_sha256 != partition.cohort_plan_sha256
            or binding.role != "tuning"
            or binding.run_id != entry.run_id
            or binding.report_sha256 != entry.capture_report_sha256
        ):
            raise ValueError("Round 74 segmented tuning binding differs")
        ordinal = binding.slot_ordinal
        if bounds[0] <= ordinal < bounds[1]:
            group_index = 0
        elif bounds[1] <= ordinal < bounds[2]:
            group_index = 1
        elif bounds[2] <= ordinal < bounds[3]:
            group_index = 2
        elif bounds[3] <= ordinal < bounds[4]:
            group_index = 3
        else:
            raise ValueError("Round 74 segmented tuning slot differs")
        run_groups[group_index].append(entry.run_id)
        ordinal_groups[group_index].append(ordinal)
        duration_groups[group_index].append(
            entry.eligible_anchor_end_wall_ns - entry.eligible_anchor_start_wall_ns
        )
    selected = Round74SegmentedTuningSubpartition(
        parent_partition_sha256=partition.partition_sha256,
        cohort_plan_sha256=partition.cohort_plan_sha256,
        model_selection_run_ids=tuple(run_groups[0]),
        calibration_run_ids=tuple(run_groups[1]),
        policy_selection_run_ids=tuple(run_groups[2]),
        ai_qualification_run_ids=tuple(run_groups[3]),
        model_selection_slot_ordinals=tuple(ordinal_groups[0]),
        calibration_slot_ordinals=tuple(ordinal_groups[1]),
        policy_selection_slot_ordinals=tuple(ordinal_groups[2]),
        ai_qualification_slot_ordinals=tuple(ordinal_groups[3]),
        model_selection_eligible_anchor_ns=tuple(duration_groups[0]),
        calibration_eligible_anchor_ns=tuple(duration_groups[1]),
        policy_selection_eligible_anchor_ns=tuple(duration_groups[2]),
        ai_qualification_eligible_anchor_ns=tuple(duration_groups[3]),
    )
    selected.validate()
    return selected


def build_round74_segmented_ai_qualification_population(
    subpartition: Round74SegmentedTuningSubpartition,
) -> Round74AIQualificationPopulation:
    """Bind the fourth tuning subrole without reading features or targets."""

    from .impact_absorption_ai_uplift import Round74AIQualificationPopulation

    if not isinstance(subpartition, Round74SegmentedTuningSubpartition):
        raise TypeError("Round 74 segmented tuning subpartition is required")
    subpartition.validate()
    selected = Round74AIQualificationPopulation(
        parent_tuning_subpartition_sha256=subpartition.subpartition_sha256,
        prior_run_ids=(
            *subpartition.model_selection_run_ids,
            *subpartition.calibration_run_ids,
            *subpartition.policy_selection_run_ids,
        ),
        prior_slot_ordinals=(
            *subpartition.model_selection_slot_ordinals,
            *subpartition.calibration_slot_ordinals,
            *subpartition.policy_selection_slot_ordinals,
        ),
        run_ids=subpartition.ai_qualification_run_ids,
        slot_ordinals=subpartition.ai_qualification_slot_ordinals,
        eligible_anchor_ns=subpartition.ai_qualification_eligible_anchor_ns,
    )
    selected.validate()
    return selected


def build_round74_segmented_model_selection_stages(
    subpartition: Round74SegmentedTuningSubpartition,
) -> Round74SegmentedModelSelectionStages:
    """Split the model-selection role without inspecting features or targets."""

    if not isinstance(subpartition, Round74SegmentedTuningSubpartition):
        raise TypeError("Round 74 segmented tuning subpartition is required")
    subpartition.validate()
    bounds = _segmented_model_selection_stage_bounds()
    run_groups: list[list[str]] = [
        [] for _stage in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
    ]
    ordinal_groups: list[list[int]] = [
        [] for _stage in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
    ]
    duration_groups: list[list[int]] = [
        [] for _stage in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
    ]
    for run_id, ordinal, duration in zip(
        subpartition.model_selection_run_ids,
        subpartition.model_selection_slot_ordinals,
        subpartition.model_selection_eligible_anchor_ns,
        strict=True,
    ):
        matching = tuple(
            index
            for index in range(len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS))
            if bounds[index] <= ordinal < bounds[index + 1]
        )
        if len(matching) != 1:
            raise ValueError(
                "Round 74 segmented model-selection slot assignment differs"
            )
        stage_index = matching[0]
        run_groups[stage_index].append(run_id)
        ordinal_groups[stage_index].append(ordinal)
        duration_groups[stage_index].append(duration)
    selected = Round74SegmentedModelSelectionStages(
        parent_tuning_subpartition_sha256=subpartition.subpartition_sha256,
        parent_partition_sha256=subpartition.parent_partition_sha256,
        cohort_plan_sha256=subpartition.cohort_plan_sha256,
        stage_run_ids=tuple(tuple(group) for group in run_groups),
        stage_slot_ordinals=tuple(tuple(group) for group in ordinal_groups),
        stage_eligible_anchor_ns=tuple(tuple(group) for group in duration_groups),
    )
    selected.validate()
    if (
        tuple(run_id for group in selected.stage_run_ids for run_id in group)
        != subpartition.model_selection_run_ids
        or tuple(ordinal for group in selected.stage_slot_ordinals for ordinal in group)
        != subpartition.model_selection_slot_ordinals
        or tuple(
            duration
            for group in selected.stage_eligible_anchor_ns
            for duration in group
        )
        != subpartition.model_selection_eligible_anchor_ns
    ):
        raise RuntimeError("Round 74 segmented model-selection coverage differs")
    return selected


def round74_segmented_windows_per_symbol(
    entry: Round74EventRunPartitionEntry,
) -> int:
    """Scale the legacy one-hour budget by audited eligible wall time."""

    entry.validate()
    eligible_ns = int(entry.eligible_anchor_end_wall_ns) - int(
        entry.eligible_anchor_start_wall_ns
    )
    count = (
        eligible_ns
        * ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL
        // ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS
    )
    if count < 1:
        raise ValueError("Round 74 segmented epoch has no sampling quota")
    return int(count)


def round74_segmented_window_policy() -> dict[str, object]:
    """Return the target-free duration-normalized sampling contract."""

    policy: dict[str, object] = {
        "schema_version": ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION,
        "symbols": list(IMPACT_CAPTURE_SYMBOLS),
        "quota_axis": "audited_eligible_anchor_wall_time",
        "quota_formula": (
            "floor(eligible_anchor_ns * reference_windows_per_symbol / "
            "reference_eligible_anchor_ns)"
        ),
        "reference_eligible_anchor_ns": (
            ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS
        ),
        "reference_windows_per_symbol": (
            ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL
        ),
        "strata_per_symbol": "computed_windows_per_symbol",
        "windows_per_symbol_stratum": 1,
        "selection": "endpoint_nearest_equal_wall_time_stratum_midpoint",
        "tie_break": (
            "decision_wall_monotonic_frame_message_anchor; "
            "exact_endpoint_collision_rejected"
        ),
        "event_count_or_activity_used_for_quota": False,
        "feature_value_or_representation_digest_used_for_rank": False,
        "target_label_or_outcome_used_for_quota_or_rank": False,
        "model_output_used_for_quota_or_rank": False,
        "selected_windows_restored_to_chronological_order": True,
        "underfilled_symbol_stratum_policy": "reject_epoch",
        "cross_epoch_state_feature_or_target_permitted": False,
    }
    policy["policy_sha256"] = _canonical_sha256(policy)
    return policy


def _stratum(
    entry: Round74EventRunPartitionEntry,
    decision_wall_ns: int,
    *,
    count: int,
) -> int:
    start = int(entry.eligible_anchor_start_wall_ns)
    end = int(entry.eligible_anchor_end_wall_ns)
    decision = int(decision_wall_ns)
    if not start <= decision <= end:
        raise ValueError("Round 74 segmented window is outside its eligible epoch")
    span = end - start + 1
    return min(count - 1, (decision - start) * count // span)


def _endpoint_rank(
    sample: Round74LabeledEventWindow,
    *,
    entry: Round74EventRunPartitionEntry,
    stratum: int,
    count: int,
) -> tuple[object, ...]:
    start = int(entry.eligible_anchor_start_wall_ns)
    span = int(entry.eligible_anchor_end_wall_ns) - start + 1
    offset = int(sample.decision_wall_ns) - start
    midpoint_distance_numerator = abs(
        2 * count * offset - (2 * int(stratum) + 1) * span
    )
    return (
        midpoint_distance_numerator,
        int(sample.decision_wall_ns),
        int(sample.decision_monotonic_ns),
        int(sample.endpoint_frame_index),
        int(sample.endpoint_message_index),
        int(sample.anchor_index),
    )


def _validate_sample_identity(
    sample: Round74LabeledEventWindow,
    entry: Round74EventRunPartitionEntry,
) -> None:
    if (
        sample.run_id != entry.run_id
        or sample.role != entry.role
        or sample.symbol not in IMPACT_CAPTURE_SYMBOLS
    ):
        raise ValueError("Round 74 segmented window identity differs")


def _require_read_only_store(store: object) -> ImpactAbsorptionStore:
    if not isinstance(store, ImpactAbsorptionStore):
        raise TypeError("Round 74 segmented model requires an ImpactAbsorptionStore")
    if not store.read_only:
        raise ValueError("Round 74 segmented model requires a read-only store")
    return store


def _validate_binding_entry(
    binding: Round74SegmentedCohortRunBinding,
    entry: Round74EventRunPartitionEntry,
) -> None:
    binding.validate()
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
        raise ValueError("Round 74 segmented model binding differs")


def iter_round74_segmented_optimization_feature_chunks(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    training_split: Round74SegmentedTrainingSplit,
    chunk_rows: int = ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
) -> Iterator[np.ndarray]:
    """Replay unique raw features from optimizer runs and no later run."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    training_split.validate()
    training_entries = tuple(
        entry for entry in partition.entries if entry.role == "training"
    )
    expected_run_ids = tuple(entry.run_id for entry in training_entries)
    split_run_ids = (
        *training_split.optimization_run_ids,
        *training_split.purged_run_ids,
        *training_split.early_stopping_run_ids,
    )
    bindings = dict(bindings_by_run_id)
    if (
        training_split.parent_partition_sha256 != partition.partition_sha256
        or training_split.cohort_plan_sha256 != partition.cohort_plan_sha256
        or split_run_ids != expected_run_ids
        or set(bindings) != set(expected_run_ids)
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 2
    ):
        raise ValueError("Round 74 segmented scaler input binding differs")
    entries_by_run_id = {entry.run_id: entry for entry in training_entries}
    values: list[tuple[float, ...]] = []
    emitted_rows = 0
    for run_id in training_split.optimization_run_ids:
        entry = entries_by_run_id[run_id]
        binding = bindings[run_id]
        _validate_binding_entry(binding, entry)
        for observation in iter_round74_v10_segment_event_observations(
            selected_store,
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
                    "Round 74 segmented scaler event is outside its capture run"
                )
            values.append(token.feature_values)
            if len(values) == chunk_rows:
                chunk = np.asarray(values, dtype=np.float64)
                emitted_rows += int(chunk.shape[0])
                yield chunk
                values.clear()
    if values:
        chunk = np.asarray(values, dtype=np.float64)
        emitted_rows += int(chunk.shape[0])
        yield chunk
    if emitted_rows < 2:
        raise ValueError("Round 74 segmented scaler events are insufficient")


def fit_round74_segmented_optimization_feature_scaler(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    training_split: Round74SegmentedTrainingSplit,
    chunk_rows: int = ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS,
    maximum_fit_rows: int = ROUND74_EVENT_SCALER_MAXIMUM_FIT_ROWS,
) -> Round74EventFeatureScaler:
    """Fit preprocessing on optimizer features before any label replay."""

    training_split.validate()
    return fit_round74_event_feature_scaler_stream(
        iter_round74_segmented_optimization_feature_chunks(
            store,
            partition=partition,
            bindings_by_run_id=bindings_by_run_id,
            training_split=training_split,
            chunk_rows=chunk_rows,
        ),
        partition_role="training",
        maximum_fit_rows=maximum_fit_rows,
        fit_source_scope="segmented_optimization_training_runs",
        fit_source_run_ids=training_split.optimization_run_ids,
        fit_source_partition_sha256=training_split.parent_partition_sha256,
        fit_source_selection_sha256=training_split.split_sha256,
    )


def iter_round74_segmented_labeled_event_windows(
    store: object,
    *,
    partition: Round74EventRunPartition,
    binding: Round74SegmentedCohortRunBinding,
    target_assembly: Round74SourceTargetAssembly,
    pretest_model_policy_sha256: str | None = None,
    test_unlock_sha256: str | None = None,
    window_representation: str = "per_symbol",
) -> Iterator[Round74LabeledEventWindow]:
    """Assemble labels from one independently reaudited epoch only."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    entry = partition.entry(binding.run_id)
    _validate_binding_entry(binding, entry)
    if not isinstance(target_assembly, Round74SourceTargetAssembly):
        raise TypeError("Round 74 segmented target assembly differs")
    selected_representation = str(window_representation)
    if selected_representation not in ROUND74_EVENT_WINDOW_REPRESENTATIONS:
        raise ValueError("Round 74 segmented window representation differs")
    assembler = Round74EventDatasetAssembler(
        partition=partition,
        run_id=entry.run_id,
        target_engine=target_assembly.create_engine(anchors=()),
        pretest_model_policy_sha256=pretest_model_policy_sha256,
        test_unlock_sha256=test_unlock_sha256,
        window_representation=selected_representation,
    )
    for observation in iter_round74_v10_segment_event_observations(
        selected_store,
        binding=binding,
    ):
        yield from assembler.consume(observation)
    yield from assembler.finish()


def assemble_round74_segmented_role_batches(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
    pretest_model_policy_sha256: str | None = None,
    test_unlock_sha256: str | None = None,
    window_representation: str = "per_symbol",
) -> tuple[Round74EventTrainingBatch, ...]:
    """Build variable-row batches without falling back to legacy replay."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    selected_role = str(role)
    if selected_role not in ROUND74_EVENT_PARTITION_ROLES:
        raise ValueError("Round 74 segmented model role differs")
    entries = tuple(entry for entry in partition.entries if entry.role == selected_role)
    expected = {entry.run_id for entry in entries}
    bindings = dict(bindings_by_run_id)
    assemblies = dict(target_assembly_by_run_id)
    if set(bindings) != expected or set(assemblies) != expected:
        raise ValueError("Round 74 segmented model role panel differs")
    if selected_role == "test":
        if pretest_model_policy_sha256 is None or test_unlock_sha256 is None:
            raise ValueError("Round 74 segmented test authorization is missing")
    elif pretest_model_policy_sha256 is not None or test_unlock_sha256 is not None:
        raise ValueError(
            "Round 74 segmented development role received test authorization"
        )
    batches: list[Round74EventTrainingBatch] = []
    for entry in entries:
        binding = bindings[entry.run_id]
        _validate_binding_entry(binding, entry)
        samples = select_round74_segmented_event_windows(
            iter_round74_segmented_labeled_event_windows(
                selected_store,
                partition=partition,
                binding=binding,
                target_assembly=assemblies[entry.run_id],
                pretest_model_policy_sha256=pretest_model_policy_sha256,
                test_unlock_sha256=test_unlock_sha256,
                window_representation=window_representation,
            ),
            entry=entry,
        )
        batch = build_round74_event_training_batch(samples, scaler=scaler)
        batch.validate()
        if (
            batch.role != selected_role
            or set(batch.run_id) != {entry.run_id}
            or batch.rows
            != len(IMPACT_CAPTURE_SYMBOLS) * round74_segmented_windows_per_symbol(entry)
        ):
            raise ValueError("Round 74 segmented model batch identity differs")
        batches.append(batch)
    return tuple(batches)


def iter_round74_segmented_matched_labeled_event_windows(
    store: object,
    *,
    partition: Round74EventRunPartition,
    binding: Round74SegmentedCohortRunBinding,
    target_assembly: Round74SourceTargetAssembly,
) -> Iterator[Round74MatchedEventWindowPair]:
    """Replay one development epoch once into endpoint-identical views."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    entry = partition.entry(binding.run_id)
    _validate_binding_entry(binding, entry)
    if entry.role == "test":
        raise ValueError("Round 74 segmented matched replay rejects test data")
    if not isinstance(target_assembly, Round74SourceTargetAssembly):
        raise TypeError("Round 74 segmented matched target assembly differs")
    assembler = Round74MatchedEventDatasetAssembler(
        partition=partition,
        run_id=entry.run_id,
        target_engines={
            representation: target_assembly.create_engine(anchors=())
            for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
        },
    )
    for observation in iter_round74_v10_segment_event_observations(
        selected_store,
        binding=binding,
    ):
        yield from assembler.consume(observation)
    yield from assembler.finish()


def _segmented_matched_batch_endpoint_sha256(
    batch: Round74EventTrainingBatch,
) -> str:
    return _canonical_sha256(
        {
            "role": batch.role,
            "partition_sha256": batch.partition_sha256,
            "scaler_sha256": batch.scaler_sha256,
            "run_id": list(batch.run_id),
            "symbol": list(batch.symbol),
            "decision_monotonic_ns": batch.decision_monotonic_ns.tolist(),
            "decision_wall_ns": batch.decision_wall_ns.tolist(),
            "endpoint_frame_index": batch.endpoint_frame_index.tolist(),
            "endpoint_message_index": batch.endpoint_message_index.tolist(),
            "anchor_index": batch.anchor_index.tolist(),
            "target_context_sha256": list(batch.target_context_sha256),
            "test_access_sha256": list(batch.test_access_sha256),
        }
    )


def _segmented_matched_batches_differ(
    left: Round74EventTrainingBatch,
    right: Round74EventTrainingBatch,
) -> bool:
    scalar_or_tuple_fields = (
        "role",
        "partition_sha256",
        "scaler_sha256",
        "run_id",
        "symbol",
        "target_context_sha256",
        "test_access_sha256",
    )
    identity_arrays = (
        "decision_monotonic_ns",
        "decision_wall_ns",
        "endpoint_frame_index",
        "endpoint_message_index",
        "anchor_index",
    )
    target_arrays = (
        "actual_entry_monotonic_ns",
        "actual_exit_monotonic_ns",
        "net_payoff_bps",
        "maximum_adverse_excursion_bps",
        "adverse_selection",
        "regime_unpredictability",
        "action_eligibility",
        "regime_unpredictability_eligibility",
    )
    return (
        any(getattr(left, name) != getattr(right, name) for name in scalar_or_tuple_fields)
        or any(
            not np.array_equal(getattr(left, name), getattr(right, name))
            for name in identity_arrays
        )
        or any(
            not np.array_equal(
                getattr(left, name),
                getattr(right, name),
                equal_nan=True,
            )
            for name in target_arrays
        )
    )


@dataclass(frozen=True)
class Round74SegmentedMatchedRepresentationRoleBatches:
    """Variable-row endpoint- and target-identical segmented batches."""

    role: str
    per_symbol: tuple[Round74EventTrainingBatch, ...]
    global_cross_asset: tuple[Round74EventTrainingBatch, ...]
    schema_version: str = ROUND74_SEGMENTED_MATCHED_ROLE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_SEGMENTED_MATCHED_ROLE_SCHEMA_VERSION
            or self.role not in {"training", "tuning"}
            or not self.per_symbol
            or len(self.per_symbol) != len(self.global_cross_asset)
        ):
            raise ValueError("Round 74 segmented matched role batches differ")
        run_ids: list[str] = []
        first_wall_ns: list[int] = []
        for left, right in zip(
            self.per_symbol,
            self.global_cross_asset,
            strict=True,
        ):
            left.validate()
            right.validate()
            if (
                left.role != self.role
                or left.window_representation != "per_symbol"
                or right.window_representation != "global_cross_asset"
                or left.rows < 1
                or left.rows != right.rows
                or left.batch_sha256 == right.batch_sha256
                or _segmented_matched_batches_differ(left, right)
                or len(set(left.run_id)) != 1
            ):
                raise ValueError("Round 74 segmented matched role identity differs")
            run_ids.append(left.run_id[0])
            first_wall_ns.append(int(left.decision_wall_ns[0]))
        if len(run_ids) != len(set(run_ids)) or any(
            current <= prior
            for prior, current in zip(first_wall_ns, first_wall_ns[1:])
        ):
            raise ValueError("Round 74 segmented matched role chronology differs")

    @property
    def matched_role_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "representations": list(ROUND74_EVENT_WINDOW_REPRESENTATIONS),
            "per_symbol_batch_sha256": [
                batch.batch_sha256 for batch in self.per_symbol
            ],
            "global_cross_asset_batch_sha256": [
                batch.batch_sha256 for batch in self.global_cross_asset
            ],
            "endpoint_panel_sha256": [
                _segmented_matched_batch_endpoint_sha256(batch)
                for batch in self.per_symbol
            ],
            "rows": sum(batch.rows for batch in self.per_symbol),
            "source_replay_passes_per_run": 1,
            "target_value_or_outcome_used_for_sampling": False,
            "sealed_test_role_accessed": False,
            "window_selection_policy": round74_segmented_window_policy(),
        }


def assemble_round74_segmented_matched_representation_role_batches(
    store: object,
    *,
    partition: Round74EventRunPartition,
    bindings_by_run_id: Mapping[str, Round74SegmentedCohortRunBinding],
    scaler: Round74EventFeatureScaler,
    role: str,
    target_assembly_by_run_id: Mapping[str, Round74SourceTargetAssembly],
) -> Round74SegmentedMatchedRepresentationRoleBatches:
    """Replay one segmented development role once into both representations."""

    selected_store = _require_read_only_store(store)
    partition.validate()
    selected_role = str(role)
    if selected_role not in {"training", "tuning"}:
        raise ValueError("Round 74 segmented matched role must be development data")
    entries = tuple(entry for entry in partition.entries if entry.role == selected_role)
    expected = {entry.run_id for entry in entries}
    bindings = dict(bindings_by_run_id)
    assemblies = dict(target_assembly_by_run_id)
    if set(bindings) != expected or set(assemblies) != expected:
        raise ValueError("Round 74 segmented matched role panel differs")
    batches: dict[str, list[Round74EventTrainingBatch]] = {
        representation: []
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    }
    for entry in entries:
        binding = bindings[entry.run_id]
        _validate_binding_entry(binding, entry)
        assembly = assemblies[entry.run_id]
        if not isinstance(assembly, Round74SourceTargetAssembly):
            raise TypeError("Round 74 segmented matched target assembly differs")
        pairs = select_round74_segmented_matched_event_windows(
            iter_round74_segmented_matched_labeled_event_windows(
                selected_store,
                partition=partition,
                binding=binding,
                target_assembly=assembly,
            ),
            entry=entry,
        )
        expected_rows = len(IMPACT_CAPTURE_SYMBOLS) * round74_segmented_windows_per_symbol(
            entry
        )
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS:
            samples = tuple(getattr(pair, representation) for pair in pairs)
            batch = build_round74_event_training_batch(samples, scaler=scaler)
            batch.validate()
            if batch.rows != expected_rows:
                raise ValueError("Round 74 segmented matched batch rows differ")
            batches[representation].append(batch)
    result = Round74SegmentedMatchedRepresentationRoleBatches(
        role=selected_role,
        per_symbol=tuple(batches["per_symbol"]),
        global_cross_asset=tuple(batches["global_cross_asset"]),
    )
    result.validate()
    return result


def select_round74_segmented_event_windows(
    windows: Iterable[Round74LabeledEventWindow],
    *,
    entry: Round74EventRunPartitionEntry,
) -> tuple[Round74LabeledEventWindow, ...]:
    """Select a regular target-blind wall-time panel from one epoch."""

    entry.validate()
    count = round74_segmented_windows_per_symbol(entry)
    selected: dict[
        tuple[str, int],
        tuple[tuple[object, ...], Round74LabeledEventWindow] | None,
    ] = {
        (symbol, stratum): None
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for stratum in range(count)
    }
    observed = 0
    for sample in windows:
        observed += 1
        _validate_sample_identity(sample, entry)
        stratum = _stratum(entry, sample.decision_wall_ns, count=count)
        rank = _endpoint_rank(
            sample,
            entry=entry,
            stratum=stratum,
            count=count,
        )
        key = (sample.symbol, stratum)
        incumbent = selected[key]
        if incumbent is None or rank < incumbent[0]:
            selected[key] = (rank, sample)
        elif (
            rank == incumbent[0]
            and sample.feature_window_sha256 != incumbent[1].feature_window_sha256
        ):
            raise ValueError("Round 74 segmented endpoint identity is duplicated")
    if observed == 0:
        raise ValueError(f"Round 74 segmented epoch {entry.run_id} has no windows")
    underfilled = tuple(
        f"{symbol}:{stratum}"
        for (symbol, stratum), value in selected.items()
        if value is None
    )
    if underfilled:
        raise ValueError(
            "Round 74 segmented temporal coverage is incomplete: "
            + ",".join(underfilled)
        )
    output = tuple(value[1] for value in selected.values() if value is not None)
    expected = len(IMPACT_CAPTURE_SYMBOLS) * count
    if len(output) != expected:
        raise RuntimeError("Round 74 segmented window count differs")
    ordered = tuple(
        sorted(
            output,
            key=lambda sample: (
                sample.decision_monotonic_ns,
                sample.symbol,
                sample.anchor_index,
                sample.feature_window_sha256,
            ),
        )
    )
    if len({sample.feature_window_sha256 for sample in ordered}) != len(ordered):
        raise ValueError("Round 74 segmented window identity is duplicated")
    return ordered


def select_round74_segmented_matched_event_windows(
    pairs: Iterable[Round74MatchedEventWindowPair],
    *,
    entry: Round74EventRunPartitionEntry,
) -> tuple[Round74MatchedEventWindowPair, ...]:
    """Apply the same endpoint-only panel to both feature representations."""

    entry.validate()
    count = round74_segmented_windows_per_symbol(entry)
    selected: dict[
        tuple[str, int],
        tuple[tuple[object, ...], Round74MatchedEventWindowPair] | None,
    ] = {
        (symbol, stratum): None
        for symbol in IMPACT_CAPTURE_SYMBOLS
        for stratum in range(count)
    }
    observed = 0
    for pair in pairs:
        sample = pair.per_symbol
        observed += 1
        _validate_sample_identity(sample, entry)
        stratum = _stratum(entry, sample.decision_wall_ns, count=count)
        rank = _endpoint_rank(
            sample,
            entry=entry,
            stratum=stratum,
            count=count,
        )
        key = (sample.symbol, stratum)
        incumbent = selected[key]
        if incumbent is None or rank < incumbent[0]:
            selected[key] = (rank, pair)
        elif (
            rank == incumbent[0]
            and sample.feature_window_sha256
            != incumbent[1].per_symbol.feature_window_sha256
        ):
            raise ValueError("Round 74 segmented endpoint identity is duplicated")
    if observed == 0:
        raise ValueError(f"Round 74 segmented epoch {entry.run_id} has no pairs")
    underfilled = tuple(
        f"{symbol}:{stratum}"
        for (symbol, stratum), value in selected.items()
        if value is None
    )
    if underfilled:
        raise ValueError(
            "Round 74 segmented matched temporal coverage is incomplete: "
            + ",".join(underfilled)
        )
    output = tuple(value[1] for value in selected.values() if value is not None)
    for pair in output:
        pair.validate()
    expected = len(IMPACT_CAPTURE_SYMBOLS) * count
    if len(output) != expected:
        raise RuntimeError("Round 74 segmented matched window count differs")
    return tuple(
        sorted(
            output,
            key=lambda pair: (
                pair.per_symbol.decision_monotonic_ns,
                pair.per_symbol.symbol,
                pair.per_symbol.anchor_index,
                pair.per_symbol.endpoint_frame_index,
                pair.per_symbol.endpoint_message_index,
            ),
        )
    )


__all__ = [
    "ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR",
    "ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS",
    "ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS",
    "ROUND74_SEGMENTED_MATCHED_ROLE_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS",
    "ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS",
    "ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL",
    "ROUND74_SEGMENTED_SCALER_FEATURE_CHUNK_ROWS",
    "ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TRAINING_SPLIT_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS",
    "ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION",
    "Round74SegmentedModelSelectionStages",
    "Round74SegmentedMatchedRepresentationRoleBatches",
    "Round74SegmentedTestPopulation",
    "Round74SegmentedTrainingSplit",
    "Round74SegmentedTuningSubpartition",
    "assemble_round74_segmented_matched_representation_role_batches",
    "assemble_round74_segmented_role_batches",
    "build_round74_segmented_model_selection_stages",
    "build_round74_segmented_ai_qualification_population",
    "build_round74_segmented_sealed_dataset_identity",
    "build_round74_segmented_test_population",
    "build_round74_segmented_training_split",
    "build_round74_segmented_tuning_subpartition",
    "fit_round74_segmented_optimization_feature_scaler",
    "iter_round74_segmented_labeled_event_windows",
    "iter_round74_segmented_matched_labeled_event_windows",
    "iter_round74_segmented_optimization_feature_chunks",
    "round74_segmented_window_policy",
    "round74_segmented_windows_per_symbol",
    "select_round74_segmented_event_windows",
    "select_round74_segmented_matched_event_windows",
]

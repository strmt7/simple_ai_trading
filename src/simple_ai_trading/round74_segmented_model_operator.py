"""Duration-normalized model preparation for Round 74 transport epochs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_ROLES,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74EventDatasetAssembler,
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
    Round74EventTrainingBatch,
    Round74LabeledEventWindow,
    Round74MatchedEventWindowPair,
    build_round74_event_training_batch,
)
from .impact_absorption_event_scaling import Round74EventFeatureScaler
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
    from .impact_absorption_event_sealed_ledger import Round74SealedDatasetIdentity


ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION = (
    "round-074-segmented-duration-normalized-window-selection-v1"
)
ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS = 3_289_500_000_000
ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL = 256
ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION = (
    "round-074-segmented-tuning-subpartition-v1"
)
ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION = (
    "round-074-segmented-model-selection-stages-v1"
)
ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION = (
    "round-074-segmented-test-population-v1"
)
ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS = (12, 6, 6)
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


def _segmented_tuning_subrole_bounds() -> tuple[int, int, int, int]:
    training_slots = ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["training"]
    tuning_slots = ROUND74_SEGMENTED_COHORT_ROLE_COUNTS["tuning"]
    model_weight, calibration_weight, _policy_weight = (
        ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS
    )
    denominator = ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHT_TOTAL

    def ceiling_fraction(numerator: int) -> int:
        return (tuning_slots * numerator + denominator - 1) // denominator

    model_end = training_slots + ceiling_fraction(model_weight)
    calibration_end = training_slots + ceiling_fraction(
        model_weight + calibration_weight
    )
    return (
        training_slots,
        model_end,
        calibration_end,
        training_slots + tuning_slots,
    )


def _segmented_tuning_required_anchor_ns() -> tuple[int, int, int]:
    required = ROUND74_SEGMENTED_COHORT_REQUIRED_ELIGIBLE_ANCHOR_NS["tuning"]
    denominator = ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHT_TOTAL
    return tuple(
        required * weight // denominator
        for weight in ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS
    )


def _segmented_model_selection_stage_bounds() -> tuple[int, ...]:
    model_start, model_end, _calibration_end, _tuning_end = (
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
class Round74SegmentedTuningSubpartition:
    """Target-blind tuning roles inherited from immutable scheduled slot ranges."""

    parent_partition_sha256: str
    cohort_plan_sha256: str
    model_selection_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    policy_selection_run_ids: tuple[str, ...]
    model_selection_slot_ordinals: tuple[int, ...]
    calibration_slot_ordinals: tuple[int, ...]
    policy_selection_slot_ordinals: tuple[int, ...]
    model_selection_eligible_anchor_ns: tuple[int, ...]
    calibration_eligible_anchor_ns: tuple[int, ...]
    policy_selection_eligible_anchor_ns: tuple[int, ...]
    schema_version: str = ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION
    optimization_population: str = "eligible_target"

    def validate(self) -> None:
        run_groups = (
            self.model_selection_run_ids,
            self.calibration_run_ids,
            self.policy_selection_run_ids,
        )
        ordinal_groups = (
            self.model_selection_slot_ordinals,
            self.calibration_slot_ordinals,
            self.policy_selection_slot_ordinals,
        )
        duration_groups = (
            self.model_selection_eligible_anchor_ns,
            self.calibration_eligible_anchor_ns,
            self.policy_selection_eligible_anchor_ns,
        )
        bounds = _segmented_tuning_subrole_bounds()
        expected_ranges = (
            range(bounds[0], bounds[1]),
            range(bounds[1], bounds[2]),
            range(bounds[2], bounds[3]),
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
            ],
            "required_eligible_anchor_ns": list(required),
            "model_selection_run_ids": list(self.model_selection_run_ids),
            "calibration_run_ids": list(self.calibration_run_ids),
            "policy_selection_run_ids": list(self.policy_selection_run_ids),
            "model_selection_slot_ordinals": list(self.model_selection_slot_ordinals),
            "calibration_slot_ordinals": list(self.calibration_slot_ordinals),
            "policy_selection_slot_ordinals": list(self.policy_selection_slot_ordinals),
            "model_selection_eligible_anchor_ns": list(
                self.model_selection_eligible_anchor_ns
            ),
            "calibration_eligible_anchor_ns": list(self.calibration_eligible_anchor_ns),
            "policy_selection_eligible_anchor_ns": list(
                self.policy_selection_eligible_anchor_ns
            ),
            "observed_eligible_anchor_ns": [
                sum(self.model_selection_eligible_anchor_ns),
                sum(self.calibration_eligible_anchor_ns),
                sum(self.policy_selection_eligible_anchor_ns),
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
        )
        integer_keys = (
            "model_selection_slot_ordinals",
            "calibration_slot_ordinals",
            "policy_selection_slot_ordinals",
            "model_selection_eligible_anchor_ns",
            "calibration_eligible_anchor_ns",
            "policy_selection_eligible_anchor_ns",
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
                model_selection_slot_ordinals=tuple(
                    payload["model_selection_slot_ordinals"]
                ),
                calibration_slot_ordinals=tuple(payload["calibration_slot_ordinals"]),
                policy_selection_slot_ordinals=tuple(
                    payload["policy_selection_slot_ordinals"]
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
    run_groups: list[list[str]] = [[], [], []]
    ordinal_groups: list[list[int]] = [[], [], []]
    duration_groups: list[list[int]] = [[], [], []]
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
        model_selection_slot_ordinals=tuple(ordinal_groups[0]),
        calibration_slot_ordinals=tuple(ordinal_groups[1]),
        policy_selection_slot_ordinals=tuple(ordinal_groups[2]),
        model_selection_eligible_anchor_ns=tuple(duration_groups[0]),
        calibration_eligible_anchor_ns=tuple(duration_groups[1]),
        policy_selection_eligible_anchor_ns=tuple(duration_groups[2]),
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
    "ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS",
    "ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS",
    "ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL",
    "ROUND74_SEGMENTED_TEST_POPULATION_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TUNING_SUBPARTITION_SCHEMA_VERSION",
    "ROUND74_SEGMENTED_TUNING_SUBROLE_WEIGHTS",
    "ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION",
    "Round74SegmentedModelSelectionStages",
    "Round74SegmentedTestPopulation",
    "Round74SegmentedTuningSubpartition",
    "assemble_round74_segmented_role_batches",
    "build_round74_segmented_model_selection_stages",
    "build_round74_segmented_sealed_dataset_identity",
    "build_round74_segmented_test_population",
    "build_round74_segmented_tuning_subpartition",
    "iter_round74_segmented_labeled_event_windows",
    "round74_segmented_window_policy",
    "round74_segmented_windows_per_symbol",
    "select_round74_segmented_event_windows",
    "select_round74_segmented_matched_event_windows",
]

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_ai_uplift import (
    Round74AIQualificationPopulation,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartition,
    Round74EventRunPartitionEntry,
)
from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)
import simple_ai_trading.round74_segmented_model_operator as subject
from simple_ai_trading.round74_segmented_model_operator import (
    ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS,
    ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS,
    Round74SegmentedModelSelectionStages,
    Round74SegmentedTestPopulation,
    Round74SegmentedTrainingSplit,
    Round74SegmentedTuningSubpartition,
    build_round74_segmented_ai_qualification_population,
    build_round74_segmented_model_selection_stages,
    build_round74_segmented_test_population,
    build_round74_segmented_training_split,
    build_round74_segmented_tuning_subpartition,
    fit_round74_segmented_optimization_feature_scaler,
    round74_segmented_window_policy,
    round74_segmented_windows_per_symbol,
    iter_round74_segmented_labeled_event_windows,
    select_round74_segmented_event_windows,
)


_START = 2_000_000_000_000_000_000


def _entry(eligible_ns: int) -> Round74EventRunPartitionEntry:
    return Round74EventRunPartitionEntry(
        run_id="1" * 32,
        role="training",
        capture_report_sha256="2" * 64,
        capture_start_wall_ns=_START,
        capture_end_wall_ns=_START + eligible_ns + 310_500_000_000,
        eligible_anchor_start_wall_ns=_START,
        eligible_anchor_end_wall_ns=_START + eligible_ns,
    )


def _partition() -> Round74EventRunPartition:
    entries = []
    for index, role in enumerate(("training", "tuning", "test")):
        start = _START + index * 2_000_000_000_000
        entries.append(
            Round74EventRunPartitionEntry(
                run_id=f"{index + 1:032x}",
                role=role,
                capture_report_sha256=f"{index + 1:064x}",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + 600_000_000_000,
                eligible_anchor_start_wall_ns=start,
                eligible_anchor_end_wall_ns=start + 289_500_000_000,
            )
        )
    selected = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="a" * 64,
    )
    selected.validate()
    return selected


def _binding(entry: Round74EventRunPartitionEntry) -> Round74SegmentedCohortRunBinding:
    selected = Round74SegmentedCohortRunBinding(
        plan_sha256="a" * 64,
        slot_ordinal=0,
        role=entry.role,
        run_id=entry.run_id,
        report_sha256=entry.capture_report_sha256,
        supervisor_sha256="b" * 64,
        fresh_frame_audit_sha256="c" * 64,
        fresh_epoch_audit_sha256="d" * 64,
        terminal_status="transport_ended",
        terminal_error="public source ended",
        capture_start_wall_ns=entry.capture_start_wall_ns,
        capture_end_wall_ns=entry.capture_end_wall_ns,
        feature_ready_wall_ns=entry.capture_start_wall_ns,
        usable_end_wall_ns=entry.capture_end_wall_ns,
        message_count=1,
        frame_count=1,
        compressed_payload_bytes=1,
    )
    selected.validate()
    return selected


def _segmented_tuning_partition() -> tuple[
    Round74EventRunPartition,
    dict[str, Round74SegmentedCohortRunBinding],
]:
    model_ordinals = tuple(
        ordinal for ordinal in range(514, 557) if ordinal not in {520, 530, 545}
    )
    calibration_ordinals = tuple(
        ordinal for ordinal in range(557, 579) if ordinal not in {565, 566, 567}
    )
    policy_ordinals = tuple(
        ordinal for ordinal in range(579, 600) if ordinal not in {589, 590}
    )
    ai_qualification_ordinals = tuple(range(600, 615))
    scheduled = (
        (0, "training"),
        *((ordinal, "tuning") for ordinal in model_ordinals),
        *((ordinal, "tuning") for ordinal in calibration_ordinals),
        *((ordinal, "tuning") for ordinal in policy_ordinals),
        *((ordinal, "tuning") for ordinal in ai_qualification_ordinals),
        (617, "test"),
    )
    entries: list[Round74EventRunPartitionEntry] = []
    bindings: dict[str, Round74SegmentedCohortRunBinding] = {}
    for index, (ordinal, role) in enumerate(scheduled):
        start = _START + index * 3_000_000_000_000
        run_id = f"{index + 100:032x}"
        report_sha256 = f"{index + 100:064x}"
        entry = Round74EventRunPartitionEntry(
            run_id=run_id,
            role=role,
            capture_report_sha256=report_sha256,
            capture_start_wall_ns=start,
            capture_end_wall_ns=start + 1_300_000_000_000,
            eligible_anchor_start_wall_ns=start,
            eligible_anchor_end_wall_ns=start + 900_000_000_000,
        )
        entries.append(entry)
        binding = Round74SegmentedCohortRunBinding(
            plan_sha256="a" * 64,
            slot_ordinal=ordinal,
            role=role,
            run_id=run_id,
            report_sha256=report_sha256,
            supervisor_sha256="b" * 64,
            fresh_frame_audit_sha256="c" * 64,
            fresh_epoch_audit_sha256="d" * 64,
            terminal_status="completed",
            terminal_error="",
            capture_start_wall_ns=start,
            capture_end_wall_ns=start + 1_300_000_000_000,
            feature_ready_wall_ns=start,
            usable_end_wall_ns=start + 1_300_000_000_000,
            message_count=1,
            frame_count=1,
            compressed_payload_bytes=1,
        )
        binding.validate()
        if role == "tuning":
            bindings[run_id] = binding
    partition = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="a" * 64,
    )
    partition.validate()
    return partition, bindings


def _segmented_test_partition() -> tuple[
    Round74EventRunPartition,
    dict[str, Round74SegmentedCohortRunBinding],
]:
    scheduled = (
        (0, "training"),
        (514, "tuning"),
        *((ordinal, "test") for ordinal in range(617, 707)),
    )
    entries: list[Round74EventRunPartitionEntry] = []
    bindings: dict[str, Round74SegmentedCohortRunBinding] = {}
    for index, (ordinal, role) in enumerate(scheduled):
        start = _START + index * 3_000_000_000_000
        run_id = f"{index + 500:032x}"
        report_sha256 = f"{index + 500:064x}"
        entry = Round74EventRunPartitionEntry(
            run_id=run_id,
            role=role,
            capture_report_sha256=report_sha256,
            capture_start_wall_ns=start,
            capture_end_wall_ns=start + 1_300_000_000_000,
            eligible_anchor_start_wall_ns=start,
            eligible_anchor_end_wall_ns=start + 900_000_000_000,
        )
        entries.append(entry)
        if role == "test":
            binding = Round74SegmentedCohortRunBinding(
                plan_sha256="a" * 64,
                slot_ordinal=ordinal,
                role=role,
                run_id=run_id,
                report_sha256=report_sha256,
                supervisor_sha256="b" * 64,
                fresh_frame_audit_sha256="c" * 64,
                fresh_epoch_audit_sha256="d" * 64,
                terminal_status="completed",
                terminal_error="",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + 1_300_000_000_000,
                feature_ready_wall_ns=start,
                usable_end_wall_ns=start + 1_300_000_000_000,
                message_count=1,
                frame_count=1,
                compressed_payload_bytes=1,
            )
            binding.validate()
            bindings[run_id] = binding
    partition = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="a" * 64,
    )
    partition.validate()
    return partition, bindings


def _segmented_training_partition() -> tuple[
    Round74EventRunPartition,
    dict[str, Round74SegmentedCohortRunBinding],
]:
    scheduled = (
        *((ordinal, "training") for ordinal in range(160)),
        (514, "tuning"),
        (617, "test"),
    )
    entries: list[Round74EventRunPartitionEntry] = []
    bindings: dict[str, Round74SegmentedCohortRunBinding] = {}
    for index, (ordinal, role) in enumerate(scheduled):
        start = _START + index * 2_000_000_000_000
        run_id = f"{index + 800:032x}"
        report_sha256 = f"{index + 800:064x}"
        entry = Round74EventRunPartitionEntry(
            run_id=run_id,
            role=role,
            capture_report_sha256=report_sha256,
            capture_start_wall_ns=start,
            capture_end_wall_ns=start + 1_300_000_000_000,
            eligible_anchor_start_wall_ns=start,
            eligible_anchor_end_wall_ns=start + 900_000_000_000,
        )
        entries.append(entry)
        if role == "training":
            binding = Round74SegmentedCohortRunBinding(
                plan_sha256="a" * 64,
                slot_ordinal=ordinal,
                role=role,
                run_id=run_id,
                report_sha256=report_sha256,
                supervisor_sha256="b" * 64,
                fresh_frame_audit_sha256="c" * 64,
                fresh_epoch_audit_sha256="d" * 64,
                terminal_status="completed",
                terminal_error="",
                capture_start_wall_ns=start,
                capture_end_wall_ns=start + 1_300_000_000_000,
                feature_ready_wall_ns=start,
                usable_end_wall_ns=start + 1_300_000_000_000,
                message_count=1,
                frame_count=1,
                compressed_payload_bytes=1,
            )
            binding.validate()
            bindings[run_id] = binding
    partition = Round74EventRunPartition(
        entries=tuple(entries),
        cohort_plan_sha256="a" * 64,
    )
    partition.validate()
    return partition, bindings


@dataclass(frozen=True)
class _Sample:
    run_id: str
    role: str
    symbol: str
    decision_wall_ns: int
    decision_monotonic_ns: int
    endpoint_frame_index: int
    endpoint_message_index: int
    anchor_index: int
    feature_window_sha256: str
    target_value: float


def _samples(
    entry: Round74EventRunPartitionEntry,
    *,
    extra_per_stratum: int,
    target_value: float,
) -> tuple[_Sample, ...]:
    count = round74_segmented_windows_per_symbol(entry)
    span = entry.eligible_anchor_end_wall_ns - entry.eligible_anchor_start_wall_ns + 1
    output: list[_Sample] = []
    anchor = 0
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        for stratum in range(count):
            midpoint = entry.eligible_anchor_start_wall_ns + (
                (2 * stratum + 1) * span // (2 * count)
            )
            for extra in range(extra_per_stratum):
                decision = midpoint + extra * 1_000
                identity = hashlib.sha256(
                    f"{symbol}:{stratum}:{extra}".encode()
                ).hexdigest()
                output.append(
                    _Sample(
                        run_id=entry.run_id,
                        role=entry.role,
                        symbol=symbol,
                        decision_wall_ns=decision,
                        decision_monotonic_ns=decision - _START + 1,
                        endpoint_frame_index=anchor,
                        endpoint_message_index=extra,
                        anchor_index=anchor,
                        feature_window_sha256=identity,
                        target_value=target_value,
                    )
                )
                anchor += 1
    return tuple(output)


def test_segmented_quota_is_proportional_to_eligible_wall_time() -> None:
    full = _entry(889_570_761_000)
    minimum = _entry(289_500_000_000)

    assert round74_segmented_windows_per_symbol(full) == 69
    assert round74_segmented_windows_per_symbol(minimum) == 22
    assert (
        round74_segmented_windows_per_symbol(
            _entry(ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS)
        )
        == 256
    )

    left_ns = 444_785_380_000
    right_ns = 444_785_381_000
    split = round74_segmented_windows_per_symbol(
        _entry(left_ns)
    ) + round74_segmented_windows_per_symbol(_entry(right_ns))
    combined = round74_segmented_windows_per_symbol(_entry(left_ns + right_ns))
    assert combined - split in {0, 1}


def test_segmented_training_split_precedes_feature_and_target_replay() -> None:
    partition, bindings = _segmented_training_partition()
    selected = build_round74_segmented_training_split(
        partition,
        bindings_by_run_id=bindings,
    )
    payload = selected.as_dict()

    assert isinstance(selected, Round74SegmentedTrainingSplit)
    assert len(selected.optimization_run_ids) == 128
    assert selected.purged_run_ids == ()
    assert len(selected.early_stopping_run_ids) == 32
    assert payload[
        "feature_value_target_label_or_model_output_used_for_assignment"
    ] is (False)
    assert payload["early_stopping_run_used_for_scaler_fit"] is False
    assert len(selected.split_sha256) == 64
    assert Round74SegmentedTrainingSplit.from_dict(payload) == selected

    incomplete = dict(bindings)
    incomplete.pop(selected.early_stopping_run_ids[-1])
    with pytest.raises(ValueError, match="binding panel differs"):
        build_round74_segmented_training_split(
            partition,
            bindings_by_run_id=incomplete,
        )


def test_segmented_scaler_replays_only_optimizer_run_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition, bindings = _segmented_training_partition()
    split = build_round74_segmented_training_split(
        partition,
        bindings_by_run_id=bindings,
    )
    observed_run_ids: list[str] = []

    @dataclass(frozen=True)
    class _Token:
        received_wall_ns: int
        feature_values: tuple[float, ...]

    @dataclass(frozen=True)
    class _Observation:
        token: _Token

    def observations(
        _store: object,
        *,
        binding: Round74SegmentedCohortRunBinding,
    ) -> tuple[_Observation, ...]:
        observed_run_ids.append(binding.run_id)
        values = np.zeros(len(ROUND74_EVENT_FEATURE_NAMES), dtype=np.float64)
        values[0] = 1.0
        values[5] = 1.0
        return (
            _Observation(
                _Token(
                    received_wall_ns=binding.feature_ready_wall_ns + 1,
                    feature_values=tuple(values),
                )
            ),
            _Observation(
                _Token(
                    received_wall_ns=binding.feature_ready_wall_ns + 2,
                    feature_values=tuple(values),
                )
            ),
        )

    monkeypatch.setattr(
        subject,
        "iter_round74_v10_segment_event_observations",
        observations,
    )
    scaler = fit_round74_segmented_optimization_feature_scaler(
        ImpactAbsorptionStore(":memory:", read_only=True),
        partition=partition,
        bindings_by_run_id=bindings,
        training_split=split,
        chunk_rows=17,
        maximum_fit_rows=31,
    )

    assert tuple(observed_run_ids) == split.optimization_run_ids
    assert not set(observed_run_ids) & set(split.early_stopping_run_ids)
    assert scaler.fit_input_rows == 256
    assert scaler.fit_sample_rows == 31
    assert scaler.fit_source_scope == "segmented_optimization_training_runs"
    assert scaler.fit_source_run_ids == split.optimization_run_ids
    assert scaler.fit_source_partition_sha256 == partition.partition_sha256
    assert scaler.fit_source_selection_sha256 == split.split_sha256


def test_segmented_selection_is_target_and_activity_blind() -> None:
    entry = _entry(289_500_000_000)
    sparse = _samples(entry, extra_per_stratum=1, target_value=-999.0)
    dense = _samples(entry, extra_per_stratum=7, target_value=999.0)

    selected_sparse = select_round74_segmented_event_windows(
        sparse,
        entry=entry,
    )
    selected_dense = select_round74_segmented_event_windows(
        reversed(dense),
        entry=entry,
    )

    assert len(selected_sparse) == len(selected_dense) == 3 * 22
    assert [sample.feature_window_sha256 for sample in selected_sparse] == [
        sample.feature_window_sha256 for sample in selected_dense
    ]
    assert {sample.target_value for sample in selected_sparse} == {-999.0}
    assert {sample.target_value for sample in selected_dense} == {999.0}


def test_segmented_selection_rejects_incomplete_time_coverage() -> None:
    entry = _entry(289_500_000_000)
    samples = _samples(entry, extra_per_stratum=1, target_value=0.0)

    with pytest.raises(ValueError, match="temporal coverage is incomplete"):
        select_round74_segmented_event_windows(samples[1:], entry=entry)


def test_segmented_selection_rejects_ambiguous_duplicate_endpoint() -> None:
    entry = _entry(289_500_000_000)
    samples = _samples(entry, extra_per_stratum=1, target_value=0.0)
    duplicate = replace(
        samples[0],
        feature_window_sha256="f" * 64,
        target_value=123.0,
    )

    with pytest.raises(ValueError, match="endpoint identity is duplicated"):
        select_round74_segmented_event_windows(
            (*samples, duplicate),
            entry=entry,
        )


def test_segmented_policy_records_no_outcome_dependent_selection() -> None:
    policy = round74_segmented_window_policy()

    assert len(policy["policy_sha256"]) == 64
    assert policy["event_count_or_activity_used_for_quota"] is False
    assert policy["target_label_or_outcome_used_for_quota_or_rank"] is False
    assert policy["model_output_used_for_quota_or_rank"] is False
    assert policy["cross_epoch_state_feature_or_target_permitted"] is False


def test_segmented_tuning_subroles_include_every_admitted_scheduled_segment() -> None:
    partition, bindings = _segmented_tuning_partition()

    selected = build_round74_segmented_tuning_subpartition(
        partition,
        bindings_by_run_id=bindings,
    )
    payload = selected.as_dict()

    assert isinstance(selected, Round74SegmentedTuningSubpartition)
    assert len(selected.model_selection_run_ids) == 40
    assert len(selected.calibration_run_ids) == 19
    assert len(selected.policy_selection_run_ids) == 19
    assert len(selected.ai_qualification_run_ids) == 15
    assert payload["scheduled_subrole_counts"] == [43, 22, 21, 17]
    assert payload["observed_eligible_anchor_ns"] == [
        36_000_000_000_000,
        17_100_000_000_000,
        17_100_000_000_000,
        13_500_000_000_000,
    ]
    assert payload["all_admitted_tuning_segments_included"]
    assert not payload["cross_subrole_run_reuse_permitted"]
    assert not payload["sealed_test_run_accessed"]
    assert len(selected.subpartition_sha256) == 64
    assert set(
        (
            *selected.model_selection_run_ids,
            *selected.calibration_run_ids,
            *selected.policy_selection_run_ids,
            *selected.ai_qualification_run_ids,
        )
    ) == set(bindings)
    assert Round74SegmentedTuningSubpartition.from_dict(payload) == selected
    ai_population = build_round74_segmented_ai_qualification_population(selected)
    assert ai_population.run_ids == selected.ai_qualification_run_ids
    assert ai_population.slot_ordinals == selected.ai_qualification_slot_ordinals
    assert ai_population.prior_run_ids == (
        *selected.model_selection_run_ids,
        *selected.calibration_run_ids,
        *selected.policy_selection_run_ids,
    )
    assert (
        Round74AIQualificationPopulation.from_dict(ai_population.as_dict())
        == ai_population
    )

    tampered = dict(payload)
    tampered["sealed_test_run_accessed"] = True
    with pytest.raises(ValueError, match="digest differs"):
        Round74SegmentedTuningSubpartition.from_dict(tampered)

    incomplete = dict(bindings)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="binding panel differs"):
        build_round74_segmented_tuning_subpartition(
            partition,
            bindings_by_run_id=incomplete,
        )


def test_segmented_model_selection_stages_are_disjoint_and_target_blind() -> None:
    partition, bindings = _segmented_tuning_partition()
    subpartition = build_round74_segmented_tuning_subpartition(
        partition,
        bindings_by_run_id=bindings,
    )

    selected = build_round74_segmented_model_selection_stages(subpartition)
    payload = selected.as_dict()

    assert isinstance(selected, Round74SegmentedModelSelectionStages)
    assert payload["stage_order"] == list(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
    assert payload["scheduled_slot_bounds"] == [514, 523, 532, 540, 549, 557]
    assert payload["required_eligible_anchor_ns_per_stage"] == 6_579_000_000_000
    assert [len(run_ids) for run_ids in selected.stage_run_ids] == [8, 8, 8, 8, 8]
    assert [sum(durations) for durations in selected.stage_eligible_anchor_ns] == [
        7_200_000_000_000
    ] * 5
    assert (
        tuple(run_id for run_ids in selected.stage_run_ids for run_id in run_ids)
        == subpartition.model_selection_run_ids
    )
    assert payload["target_label_or_model_output_used_for_assignment"] is False
    assert payload["cross_stage_run_reuse_permitted"] is False
    assert payload["all_parent_model_selection_segments_included"] is True
    assert Round74SegmentedModelSelectionStages.from_dict(payload) == selected

    tampered = dict(payload)
    tampered["cross_stage_run_reuse_permitted"] = True
    with pytest.raises(ValueError, match="payload differs"):
        Round74SegmentedModelSelectionStages.from_dict(tampered)


def test_segmented_model_selection_stage_rejects_insufficient_duration() -> None:
    partition, bindings = _segmented_tuning_partition()
    subpartition = build_round74_segmented_tuning_subpartition(
        partition,
        bindings_by_run_id=bindings,
    )
    shortened = replace(
        subpartition,
        model_selection_run_ids=subpartition.model_selection_run_ids[:-1],
        model_selection_slot_ordinals=(subpartition.model_selection_slot_ordinals[:-1]),
        model_selection_eligible_anchor_ns=(
            subpartition.model_selection_eligible_anchor_ns[:-1]
        ),
    )
    shortened.validate()

    with pytest.raises(ValueError, match="stage differs"):
        build_round74_segmented_model_selection_stages(shortened)


def test_segmented_test_population_rejects_any_admitted_segment_omission() -> None:
    partition, bindings = _segmented_test_partition()

    population = build_round74_segmented_test_population(
        partition,
        bindings_by_run_id=bindings,
    )
    payload = population.as_dict()

    assert isinstance(population, Round74SegmentedTestPopulation)
    assert len(population.test_run_ids) == 90
    assert payload["scheduled_test_slot_count"] == 103
    assert payload["observed_eligible_anchor_ns"] == 81_000_000_000_000
    assert payload["all_admitted_test_segments_included"]
    assert not payload["test_segment_selection_permitted"]
    assert Round74SegmentedTestPopulation.from_dict(payload) == population

    incomplete = dict(bindings)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="binding panel differs"):
        build_round74_segmented_test_population(
            partition,
            bindings_by_run_id=incomplete,
        )


def test_segmented_replay_uses_only_the_audited_epoch_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition()
    entry = partition.entries[0]
    binding = _binding(entry)
    observed: list[object] = []

    class _TargetAssembly:
        def create_engine(self, *, anchors: tuple[object, ...]) -> object:
            assert anchors == ()
            return object()

    class _Assembler:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["run_id"] == entry.run_id

        def consume(self, observation: object) -> tuple[str, ...]:
            observed.append(observation)
            return (f"window-{observation}",)

        def finish(self) -> tuple[str, ...]:
            return ("finished",)

    monkeypatch.setattr(subject, "Round74SourceTargetAssembly", _TargetAssembly)
    monkeypatch.setattr(subject, "Round74EventDatasetAssembler", _Assembler)
    monkeypatch.setattr(
        subject,
        "iter_round74_v10_segment_event_observations",
        lambda store, *, binding: iter(("first", "second")),
    )
    store = ImpactAbsorptionStore(":memory:", read_only=True)

    output = tuple(
        iter_round74_segmented_labeled_event_windows(
            store,
            partition=partition,
            binding=binding,
            target_assembly=_TargetAssembly(),
        )
    )

    assert observed == ["first", "second"]
    assert output == ("window-first", "window-second", "finished")

    with pytest.raises(ValueError, match="binding differs"):
        tuple(
            iter_round74_segmented_labeled_event_windows(
                store,
                partition=partition,
                binding=replace(binding, report_sha256="f" * 64),
                target_assembly=_TargetAssembly(),
            )
        )

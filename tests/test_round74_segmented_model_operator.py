from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import pytest

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
import simple_ai_trading.round74_segmented_model_operator as subject
from simple_ai_trading.round74_segmented_model_operator import (
    ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS,
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
    span = (
        entry.eligible_anchor_end_wall_ns
        - entry.eligible_anchor_start_wall_ns
        + 1
    )
    output: list[_Sample] = []
    anchor = 0
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        for stratum in range(count):
            midpoint = (
                entry.eligible_anchor_start_wall_ns
                + ((2 * stratum + 1) * span // (2 * count))
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
    combined = round74_segmented_windows_per_symbol(
        _entry(left_ns + right_ns)
    )
    assert combined - split in {0, 1}


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

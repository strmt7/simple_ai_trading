from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import pytest

from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventRunPartitionEntry,
)
from simple_ai_trading.impact_absorption_store import IMPACT_CAPTURE_SYMBOLS
from simple_ai_trading.round74_segmented_model_operator import (
    ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS,
    round74_segmented_window_policy,
    round74_segmented_windows_per_symbol,
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

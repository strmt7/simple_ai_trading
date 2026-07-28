"""Duration-normalized model preparation for Round 74 transport epochs."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json

from .impact_absorption_event_dataset import (
    Round74EventRunPartitionEntry,
    Round74LabeledEventWindow,
    Round74MatchedEventWindowPair,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS


ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION = (
    "round-074-segmented-duration-normalized-window-selection-v1"
)
ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS = 3_289_500_000_000
ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL = 256


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def round74_segmented_windows_per_symbol(
    entry: Round74EventRunPartitionEntry,
) -> int:
    """Scale the legacy one-hour budget by audited eligible wall time."""

    entry.validate()
    eligible_ns = (
        int(entry.eligible_anchor_end_wall_ns)
        - int(entry.eligible_anchor_start_wall_ns)
    )
    count = (
        eligible_ns * ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL
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
            and sample.feature_window_sha256
            != incumbent[1].feature_window_sha256
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
    output = tuple(
        value[1]
        for value in selected.values()
        if value is not None
    )
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
    output = tuple(
        value[1]
        for value in selected.values()
        if value is not None
    )
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
    "ROUND74_SEGMENTED_REFERENCE_ELIGIBLE_ANCHOR_NS",
    "ROUND74_SEGMENTED_REFERENCE_WINDOWS_PER_SYMBOL",
    "ROUND74_SEGMENTED_WINDOW_SELECTION_SCHEMA_VERSION",
    "round74_segmented_window_policy",
    "round74_segmented_windows_per_symbol",
    "select_round74_segmented_event_windows",
    "select_round74_segmented_matched_event_windows",
]

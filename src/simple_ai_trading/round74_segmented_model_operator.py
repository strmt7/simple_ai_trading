"""Duration-normalized model preparation for Round 74 transport epochs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import hashlib
import json

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
    Round74SegmentedCohortRunBinding,
    iter_round74_v10_segment_event_observations,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
)
from .impact_absorption_target_assembly import Round74SourceTargetAssembly


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
    entries = tuple(
        entry for entry in partition.entries if entry.role == selected_role
    )
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
            != len(IMPACT_CAPTURE_SYMBOLS)
            * round74_segmented_windows_per_symbol(entry)
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
    "assemble_round74_segmented_role_batches",
    "iter_round74_segmented_labeled_event_windows",
    "round74_segmented_window_policy",
    "round74_segmented_windows_per_symbol",
    "select_round74_segmented_event_windows",
    "select_round74_segmented_matched_event_windows",
]

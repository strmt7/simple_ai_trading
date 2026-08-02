"""Target-blind condition admission and core features for Polymarket Round 21."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_recorder import (
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from .polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
    PolymarketUnionEvent,
)
from .polymarket_round20_contract import POLYMARKET_ROUND20_CONTRACT_SHA256
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
    Round21CoreFeatureEngine,
    Round21CoreFeatureSnapshot,
    parse_round21_chainlink_wire_text,
    validate_round21_union_event,
)
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_CONDITION_DURATION_MS,
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_DECISION_CADENCE_MS,
    Round21PartitionPolicy,
)
from .polymarket_round21_terminal import (
    POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256,
    validate_round21_terminal_transport_manifest,
)


POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-core-corpus-materialization-design-v1"
)
POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256 = (
    "a5ef5445ec0381540dd675784d711873b0d644dd0ef44462f12e71d131e133eb"
)
POLYMARKET_ROUND21_CONDITION_ADMISSION_SCHEMA_VERSION = (
    "polymarket-round21-condition-admission-v1"
)
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-021-core-corpus-materialization-design-v1.json"
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9:_./-]{0,159}$")
_LANES = ("clob-a", "clob-b")
_PAIRING_WINDOW_MS = 2_000
_CHAINLINK_TERMINAL_GRACE_MS = 5_000
_MINIMUM_LANE_COVERAGE = 0.9
_MINIMUM_SHARED_FRACTION = 0.9
_MAXIMUM_JOINT_UNHEALTHY_MS = 2_000
_FRESH_EVENT_MS = 10_000
_MAXIMUM_JSON_BYTES = 512 * 1024
_FEATURE_ROLES = frozenset(("train", "tune_calibration", "tune_selection", "test"))
_MAXIMUM_CLOCK_OFFSET_DRIFT_NS = 1_000_000_000


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 21 core corpus JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 core corpus JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def load_round21_core_corpus_design(repository: str | Path) -> dict[str, object]:
    path = Path(repository).resolve() / _DESIGN_RELATIVE
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_JSON_BYTES
    ):
        raise ValueError("Round 21 core corpus design is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 core corpus design is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 core corpus design is not an object")
    payload = dict(value)
    claimed = str(payload.pop("design_sha256", "")).strip().lower()
    parents = payload.get("parents")
    source = payload.get("source_boundary")
    authority = payload.get("authority")
    if (
        claimed != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SCHEMA_VERSION
        or payload.get("round") != 21
        or not isinstance(parents, Mapping)
        or parents.get("round20_contract_sha256")
        != POLYMARKET_ROUND20_CONTRACT_SHA256
        or parents.get("round21_contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or parents.get("round21_dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or parents.get("round21_feature_policy_sha256")
        != POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
        or parents.get("round21_terminal_transport_design_sha256")
        != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        or not isinstance(source, Mapping)
        or source.get("outcomes_consulted") is not False
        or source.get("model_scores_consulted") is not False
        or source.get("optional_binance_consulted") is not False
        or not isinstance(authority, Mapping)
        or any(
            authority.get(field) is not False
            for field in (
                "model_data_eligible",
                "model_selected",
                "ai_edge_claim",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 core corpus design differs")
    return {**payload, "design_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round21CoreCondition:
    run_id: str
    segment_index: int
    snapshot_sha256: str
    snapshot_observed_wall_ms: int
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    up_token_id: str
    down_token_id: str
    asset: str = "BTC"

    def validated(self) -> Round21CoreCondition:
        if (
            _RUN_ID.fullmatch(self.run_id) is None
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or _SHA256.fullmatch(self.snapshot_sha256) is None
            or type(self.snapshot_observed_wall_ms) is not int
            or self.snapshot_observed_wall_ms <= 0
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.event_start_ms <= 0
            or self.event_start_ms % POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or self.event_end_ms
            != self.event_start_ms + POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or _TOKEN_ID.fullmatch(self.up_token_id) is None
            or _TOKEN_ID.fullmatch(self.down_token_id) is None
            or self.up_token_id == self.down_token_id
            or self.asset != "BTC"
        ):
            raise ValueError("Round 21 core condition identity differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21ConditionSource:
    union_events: tuple[PolymarketUnionEvent, ...]
    chainlink_records: tuple[CaptureFrameRecord, ...]
    lane_event_wall_ms: Mapping[str, tuple[int, ...]]
    stream_gaps: tuple[StreamGap, ...]

    def validated(self) -> Round21ConditionSource:
        if (
            set(self.lane_event_wall_ms) != set(_LANES)
            or any(
                any(type(value) is not int or value <= 0 for value in values)
                or tuple(sorted(values)) != values
                for values in self.lane_event_wall_ms.values()
            )
            or any(not isinstance(event, PolymarketUnionEvent) for event in self.union_events)
            or any(not isinstance(record, CaptureFrameRecord) for record in self.chainlink_records)
            or any(not isinstance(gap, StreamGap) for gap in self.stream_gaps)
        ):
            raise ValueError("Round 21 condition source differs")
        for gap in self.stream_gaps:
            gap.validated()
        return self


@dataclass(frozen=True, slots=True)
class Round21CoreConditionMaterialization:
    admission: Mapping[str, object]
    available_features: tuple[Round21CoreFeatureSnapshot, ...]
    unavailable_feature_row_count: int
    unavailable_reason_counts: Mapping[str, int]

    def validated(self) -> Round21CoreConditionMaterialization:
        admission = validate_round21_condition_admission(self.admission)
        admitted = admission["admitted"] is True
        if (
            any(
                not isinstance(row, Round21CoreFeatureSnapshot) or not row.available
                for row in self.available_features
            )
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count <= 0
                for reason, count in self.unavailable_reason_counts.items()
            )
            or (not admitted and (self.available_features or self.unavailable_reason_counts))
            or type(self.unavailable_feature_row_count) is not int
            or self.unavailable_feature_row_count < 0
            or (not admitted and self.unavailable_feature_row_count != 0)
            or admission["available_feature_row_count"]
            != len(self.available_features)
            or admission["unavailable_feature_row_count"]
            != self.unavailable_feature_row_count
        ):
            raise ValueError("Round 21 core condition materialization differs")
        return self


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def _unhealthy_intervals(
    event_wall_ms: Sequence[int],
    gaps: Sequence[StreamGap],
    *,
    started_at_ms: int,
    ended_at_ms: int,
) -> tuple[tuple[int, int], ...]:
    events = tuple(event_wall_ms)
    if not events:
        return ((started_at_ms, ended_at_ms),)
    intervals: list[tuple[int, int]] = [
        (started_at_ms, min(max(events[0], started_at_ms), ended_at_ms))
    ]
    for previous, current in zip(events, events[1:], strict=False):
        intervals.append(
            (
                max(previous + _FRESH_EVENT_MS, started_at_ms),
                min(current, ended_at_ms),
            )
        )
    intervals.append(
        (max(events[-1] + _FRESH_EVENT_MS, started_at_ms), ended_at_ms)
    )
    for gap in gaps:
        later = next(
            (receipt for receipt in events if receipt >= gap.opened_at_ms),
            ended_at_ms,
        )
        intervals.append(
            (
                max(gap.opened_at_ms, started_at_ms),
                min(later, ended_at_ms),
            )
        )
    return _merge_intervals(intervals)


def _intersection_milliseconds(
    first: Sequence[tuple[int, int]],
    second: Sequence[tuple[int, int]],
) -> int:
    left = 0
    right = 0
    total = 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        total += max(0, end - start)
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return total


def _chainlink_observations(
    records: Sequence[CaptureFrameRecord],
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[list[tuple[CaptureFrameRecord, Decimal, int]], list[str]]:
    observations: list[tuple[CaptureFrameRecord, Decimal, int]] = []
    reasons: list[str] = []
    previous_connection = ""
    previous_sequence = 0
    previous_monotonic = 0
    previous_wall = 0
    for record in records:
        connection = str(record.connection_id or "").strip().lower()
        if (
            record.stream != "polymarket_rtds"
            or _CONNECTION_ID.fullmatch(connection) is None
            or record.sequence_number <= 0
            or record.received_wall_ms <= 0
            or record.received_monotonic_ns <= 0
        ):
            raise ValueError("Round 21 condition Chainlink metadata differs")
        if not start_ms <= record.received_wall_ms <= end_ms:
            continue
        if not previous_connection:
            previous_connection = connection
            previous_sequence = record.sequence_number - 1
        if connection != previous_connection:
            reasons.append("chainlink_connection_changed")
            previous_connection = connection
            previous_sequence = record.sequence_number - 1
            previous_monotonic = 0
            previous_wall = 0
        if (
            record.sequence_number != previous_sequence + 1
            or record.received_monotonic_ns <= previous_monotonic
            or record.received_wall_ms < previous_wall
        ):
            reasons.append("chainlink_sequence_or_clock_gap")
        previous_sequence = record.sequence_number
        previous_monotonic = record.received_monotonic_ns
        previous_wall = record.received_wall_ms
        tick = parse_round21_chainlink_wire_text(
            record.raw_text,
            received_at_ms=record.received_wall_ms,
        )
        if tick is None:
            continue
        observations.append((record, tick.price, tick.source_time_ms))
    return observations, reasons


def _admission_payload(
    *,
    condition: Round21CoreCondition,
    role: str,
    source: Round21ConditionSource,
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    reasons: list[str] = []
    union_payloads: list[Mapping[str, object]] = []
    lane_counts = Counter({lane: 0 for lane in _LANES})
    shared_count = 0
    full_books: set[str] = set()
    event_chain = hashlib.sha256(b"").hexdigest()
    for event in source.union_events:
        payload = validate_round21_union_event(event)
        if str(payload.get("market") or "").strip().lower() != condition.condition_id:
            raise ValueError("Round 21 condition union event belongs to another market")
        if not (
            condition.event_start_ms - 120_000
            <= event.selected_received_wall_ms
            <= condition.event_end_ms
        ):
            raise ValueError("Round 21 condition union receipt lies outside its window")
        union_payloads.append(payload)
        if event.selected_received_wall_ms < condition.event_start_ms:
            if event.event_type == "book" and str(payload.get("asset_id") or "") in {
                condition.up_token_id,
                condition.down_token_id,
            }:
                full_books.add(str(payload["asset_id"]))
            event_chain = hashlib.sha256(
                f"{event_chain}:{event.event_sha256}".encode("ascii")
            ).hexdigest()
            continue
        receipt_lanes = {receipt.lane_id for receipt in event.lane_receipts}
        for lane in receipt_lanes:
            lane_counts[lane] += 1
        if receipt_lanes == set(_LANES):
            shared_count += 1
        if event.event_type == "book" and str(payload.get("asset_id") or "") in {
            condition.up_token_id,
            condition.down_token_id,
        }:
            full_books.add(str(payload["asset_id"]))
        event_chain = hashlib.sha256(
            f"{event_chain}:{event.event_sha256}".encode("ascii")
        ).hexdigest()
    union_count = sum(
        event.selected_received_wall_ms >= condition.event_start_ms
        for event in source.union_events
    )
    lane_coverage = {
        lane: 0.0 if union_count == 0 else lane_counts[lane] / union_count
        for lane in _LANES
    }
    shared_fraction = 0.0 if union_count == 0 else shared_count / union_count
    if union_count == 0:
        reasons.append("no_condition_union_events")
    if any(lane_coverage[lane] < _MINIMUM_LANE_COVERAGE for lane in _LANES):
        reasons.append("minimum_lane_coverage_not_met")
    if shared_fraction < _MINIMUM_SHARED_FRACTION:
        reasons.append("minimum_shared_fraction_not_met")
    if full_books != {condition.up_token_id, condition.down_token_id}:
        reasons.append("both_token_full_books_not_observed")
    lane_gaps = {
        lane: tuple(
            gap
            for gap in source.stream_gaps
            if gap.stream == "clob_market"
            and gap.connection_id.partition(":")[0] == lane
            and condition.event_start_ms <= gap.opened_at_ms <= condition.event_end_ms
        )
        for lane in _LANES
    }
    unhealthy = {
        lane: _unhealthy_intervals(
            tuple(
                value
                for value in source.lane_event_wall_ms[lane]
                if condition.event_start_ms <= value <= condition.event_end_ms
            ),
            lane_gaps[lane],
            started_at_ms=condition.event_start_ms,
            ended_at_ms=condition.event_end_ms,
        )
        for lane in _LANES
    }
    joint_unhealthy_ms = _intersection_milliseconds(
        unhealthy["clob-a"], unhealthy["clob-b"]
    )
    if joint_unhealthy_ms > _MAXIMUM_JOINT_UNHEALTHY_MS:
        reasons.append("joint_clob_unhealthy_limit_exceeded")
    observations, chainlink_reasons = _chainlink_observations(
        source.chainlink_records,
        start_ms=condition.event_start_ms,
        end_ms=condition.event_end_ms + _CHAINLINK_TERMINAL_GRACE_MS,
    )
    reasons.extend(chainlink_reasons)
    opening = [
        (record, price)
        for record, price, source_time in observations
        if source_time == condition.event_start_ms
        and record.received_wall_ms >= condition.event_start_ms
    ]
    closing = [
        (record, price)
        for record, price, source_time in observations
        if source_time == condition.event_end_ms
        and record.received_wall_ms >= condition.event_end_ms
    ]
    if not opening:
        reasons.append("missing_exact_chainlink_open")
    elif len({price for _, price in opening}) != 1:
        reasons.append("contradictory_exact_chainlink_open")
    if not closing:
        reasons.append("missing_exact_chainlink_close")
    elif len({price for _, price in closing}) != 1:
        reasons.append("contradictory_exact_chainlink_close")
    reasons = list(dict.fromkeys(reasons))
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_CONDITION_ADMISSION_SCHEMA_VERSION,
        "core_corpus_design_sha256": POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
        "run_id": condition.run_id,
        "segment_index": condition.segment_index,
        "snapshot_sha256": condition.snapshot_sha256,
        "condition_id": condition.condition_id,
        "event_start_ms": condition.event_start_ms,
        "event_end_ms": condition.event_end_ms,
        "role": role,
        "union_event_count": union_count,
        "union_event_chain_sha256": event_chain,
        "lane_event_counts": dict(lane_counts),
        "lane_coverage_fraction": lane_coverage,
        "shared_event_count": shared_count,
        "shared_fraction": shared_fraction,
        "lane_gap_counts": {lane: len(lane_gaps[lane]) for lane in _LANES},
        "joint_unhealthy_ms": joint_unhealthy_ms,
        "up_full_book_observed": condition.up_token_id in full_books,
        "down_full_book_observed": condition.down_token_id in full_books,
        "exact_chainlink_open_receipt_count": len(opening),
        "exact_chainlink_close_receipt_count": len(closing),
        "chainlink_connection_count": len(
            {
                record.connection_id
                for record in source.chainlink_records
                if condition.event_start_ms
                <= record.received_wall_ms
                <= condition.event_end_ms + _CHAINLINK_TERMINAL_GRACE_MS
            }
        ),
        "admitted": not reasons,
        "rejection_reasons": reasons,
        "available_feature_row_count": 0,
        "unavailable_feature_row_count": 0,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "optional_binance_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    return payload, tuple(union_payloads)


def _feature_rows(
    condition: Round21CoreCondition,
    source: Round21ConditionSource,
) -> tuple[tuple[Round21CoreFeatureSnapshot, ...], int, Counter[str]]:
    engine = Round21CoreFeatureEngine(
        condition_id=condition.condition_id,
        up_token_id=condition.up_token_id,
        down_token_id=condition.down_token_id,
        event_start_ms=condition.event_start_ms,
    )
    inputs: list[tuple[int, int, int, object]] = []
    for record in source.chainlink_records:
        if condition.event_start_ms <= record.received_wall_ms < condition.event_end_ms:
            inputs.append(
                (
                    record.received_wall_ms,
                    record.received_monotonic_ns,
                    0,
                    record,
                )
            )
    for event in source.union_events:
        if event.selected_received_wall_ms < condition.event_end_ms:
            inputs.append(
                (
                    event.selected_received_wall_ms,
                    event.selected_received_monotonic_ns,
                    1,
                    event,
                )
            )
    inputs.sort(key=lambda item: item[:3])
    input_index = 0
    chainlink_connection = ""
    available: list[Round21CoreFeatureSnapshot] = []
    unavailable_row_count = 0
    unavailable = Counter()
    for decision in range(
        condition.event_start_ms,
        condition.event_end_ms,
        POLYMARKET_ROUND21_DECISION_CADENCE_MS,
    ):
        while input_index < len(inputs) and inputs[input_index][0] <= decision:
            item = inputs[input_index][3]
            input_index += 1
            if isinstance(item, CaptureFrameRecord):
                connection = str(item.connection_id).strip().lower()
                if not chainlink_connection:
                    engine.start_chainlink_epoch(
                        connection,
                        first_sequence_number=item.sequence_number,
                    )
                    chainlink_connection = connection
                elif connection != chainlink_connection:
                    engine.start_chainlink_epoch(
                        connection,
                        first_sequence_number=item.sequence_number,
                    )
                    chainlink_connection = connection
                engine.ingest_chainlink_record(item)
            else:
                if not isinstance(item, PolymarketUnionEvent):
                    raise TypeError("Round 21 condition feature input differs")
                engine.ingest_union_event(item)
        snapshot = engine.build(decision)
        if snapshot.available:
            available.append(snapshot)
        else:
            unavailable_row_count += 1
            unavailable.update(snapshot.reasons)
    return tuple(available), unavailable_row_count, unavailable


def build_round21_core_condition_materialization(
    *,
    condition: Round21CoreCondition,
    source: Round21ConditionSource,
    partition_policy: Round21PartitionPolicy,
) -> Round21CoreConditionMaterialization:
    selected = condition.validated()
    inputs = source.validated()
    policy = partition_policy.validated()
    try:
        role = policy.role_for_event_start(selected.event_start_ms)
    except ValueError:
        role = "outside_campaign"
    payload, _union_payloads = _admission_payload(
        condition=selected,
        role=role,
        source=inputs,
    )
    if role == "outside_campaign":
        payload["rejection_reasons"] = [
            *payload["rejection_reasons"],
            "condition_outside_campaign",
        ]
        payload["admitted"] = False
    if payload["admitted"] is True and role in _FEATURE_ROLES:
        available, unavailable_row_count, unavailable = _feature_rows(selected, inputs)
        payload["available_feature_row_count"] = len(available)
        payload["unavailable_feature_row_count"] = unavailable_row_count
    else:
        available = ()
        unavailable_row_count = 0
        unavailable = Counter()
    payload["admission_sha256"] = _canonical_sha256(payload)
    return Round21CoreConditionMaterialization(
        admission=validate_round21_condition_admission(payload),
        available_features=available,
        unavailable_feature_row_count=unavailable_row_count,
        unavailable_reason_counts=dict(sorted(unavailable.items())),
    ).validated()


def validate_round21_condition_admission(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("admission_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "core_corpus_design_sha256",
        "run_id",
        "segment_index",
        "snapshot_sha256",
        "condition_id",
        "event_start_ms",
        "event_end_ms",
        "role",
        "union_event_count",
        "union_event_chain_sha256",
        "lane_event_counts",
        "lane_coverage_fraction",
        "shared_event_count",
        "shared_fraction",
        "lane_gap_counts",
        "joint_unhealthy_ms",
        "up_full_book_observed",
        "down_full_book_observed",
        "exact_chainlink_open_receipt_count",
        "exact_chainlink_close_receipt_count",
        "chainlink_connection_count",
        "admitted",
        "rejection_reasons",
        "available_feature_row_count",
        "unavailable_feature_row_count",
        "outcomes_consulted",
        "model_scores_consulted",
        "optional_binance_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    reason_value = payload.get("rejection_reasons")
    reasons = (
        tuple(reason_value)
        if isinstance(reason_value, list)
        and all(isinstance(reason, str) and reason for reason in reason_value)
        else None
    )
    lane_counts = payload.get("lane_event_counts")
    lane_coverage = payload.get("lane_coverage_fraction")
    lane_gaps = payload.get("lane_gap_counts")
    admitted = payload.get("admitted")
    bool_fields = (
        "up_full_book_observed",
        "down_full_book_observed",
        "admitted",
        "outcomes_consulted",
        "model_scores_consulted",
        "optional_binance_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    count_fields = (
        "segment_index",
        "union_event_count",
        "shared_event_count",
        "joint_unhealthy_ms",
        "exact_chainlink_open_receipt_count",
        "exact_chainlink_close_receipt_count",
        "chainlink_connection_count",
        "available_feature_row_count",
        "unavailable_feature_row_count",
    )
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_CONDITION_ADMISSION_SCHEMA_VERSION
        or payload.get("core_corpus_design_sha256")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or _RUN_ID.fullmatch(str(payload.get("run_id") or "")) is None
        or _SHA256.fullmatch(str(payload.get("snapshot_sha256") or "")) is None
        or _CONDITION_ID.fullmatch(str(payload.get("condition_id") or "")) is None
        or type(payload.get("event_start_ms")) is not int
        or payload["event_start_ms"] <= 0
        or payload["event_start_ms"] % POLYMARKET_ROUND21_CONDITION_DURATION_MS
        or payload.get("event_end_ms")
        != payload["event_start_ms"] + POLYMARKET_ROUND21_CONDITION_DURATION_MS
        or payload.get("role")
        not in {
            "train",
            "purge_train_to_tune",
            "tune_calibration",
            "tune_selection",
            "purge_tune_to_test",
            "test",
            "outside_campaign",
        }
        or any(type(payload.get(field)) is not int or payload[field] < 0 for field in count_fields)
        or not isinstance(lane_counts, Mapping)
        or set(lane_counts) != set(_LANES)
        or any(type(value) is not int or value < 0 for value in lane_counts.values())
        or not isinstance(lane_coverage, Mapping)
        or set(lane_coverage) != set(_LANES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in lane_coverage.values()
        )
        or not isinstance(payload.get("shared_fraction"), (int, float))
        or isinstance(payload.get("shared_fraction"), bool)
        or not 0.0 <= float(payload["shared_fraction"]) <= 1.0
        or not isinstance(lane_gaps, Mapping)
        or set(lane_gaps) != set(_LANES)
        or any(type(value) is not int or value < 0 for value in lane_gaps.values())
        or _SHA256.fullmatch(
            str(payload.get("union_event_chain_sha256") or "")
        )
        is None
        or reasons is None
        or type(admitted) is not bool
        or admitted != (not reasons)
        or any(type(payload.get(field)) is not bool for field in bool_fields)
        or any(
            payload.get(field) is not False
            for field in (
                "outcomes_consulted",
                "model_scores_consulted",
                "optional_binance_consulted",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 condition admission differs")
    return {**payload, "admission_sha256": claimed}


def _snapshot_condition(
    row: Sequence[object],
    *,
    segment_index: int,
) -> Round21CoreCondition:
    (
        snapshot_id,
        run_id,
        observed_wall_ms,
        observed_monotonic_ns,
        asset,
        condition_id,
        event_start_ms,
        event_end_ms,
        up_token_id,
        down_token_id,
        clob_info_sha256,
        up_fee_rate_sha256,
        down_fee_rate_sha256,
        maker_base_fee,
        taker_base_fee,
        taker_order_delay_enabled,
        minimum_order_age_seconds,
        snapshot_payload_json,
        snapshot_sha256,
    ) = row
    try:
        parsed = json.loads(
            str(snapshot_payload_json),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 market snapshot is not strict JSON") from exc
    if (
        not isinstance(parsed, Mapping)
        or _canonical_json(parsed) != snapshot_payload_json
        or _canonical_sha256(parsed) != snapshot_sha256
    ):
        raise ValueError("Round 21 market snapshot payload differs")
    market = parsed.get("market")
    if not isinstance(market, Mapping):
        raise ValueError("Round 21 market snapshot identity is unavailable")
    identity = {
        "run_id": run_id,
        "observed_wall_ms": observed_wall_ms,
        "market": market,
        "clob_info_sha256": clob_info_sha256,
        "up_fee_rate_sha256": up_fee_rate_sha256,
        "down_fee_rate_sha256": down_fee_rate_sha256,
    }
    if (
        _canonical_sha256(identity) != snapshot_id
        or parsed
        != {
            **identity,
            "observed_monotonic_ns": observed_monotonic_ns,
            "maker_base_fee": maker_base_fee,
            "taker_base_fee": taker_base_fee,
            "taker_order_delay_enabled": taker_order_delay_enabled,
            "minimum_order_age_seconds": minimum_order_age_seconds,
        }
        or market.get("asset") != asset
        or market.get("condition_id") != condition_id
        or market.get("event_start_ms") != event_start_ms
        or market.get("end_ms") != event_end_ms
        or market.get("up_token_id") != up_token_id
        or market.get("down_token_id") != down_token_id
        or any(
            _SHA256.fullmatch(str(value or "")) is None
            for value in (
                snapshot_id,
                snapshot_sha256,
                clob_info_sha256,
                up_fee_rate_sha256,
                down_fee_rate_sha256,
            )
        )
        or type(observed_monotonic_ns) is not int
        or observed_monotonic_ns <= 0
        or type(maker_base_fee) is not int
        or maker_base_fee < 0
        or type(taker_base_fee) is not int
        or taker_base_fee < 0
        or type(taker_order_delay_enabled) is not bool
        or type(minimum_order_age_seconds) is not int
        or minimum_order_age_seconds < 0
    ):
        raise ValueError("Round 21 market snapshot columns differ")
    return Round21CoreCondition(
        run_id=str(run_id),
        segment_index=segment_index,
        snapshot_sha256=str(snapshot_sha256),
        snapshot_observed_wall_ms=int(observed_wall_ms),
        condition_id=str(condition_id),
        event_start_ms=int(event_start_ms),
        event_end_ms=int(event_end_ms),
        up_token_id=str(up_token_id),
        down_token_id=str(down_token_id),
        asset=str(asset),
    ).validated()


def load_round21_core_conditions(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
) -> tuple[Round21CoreCondition, ...]:
    """Load only eligible-run market identities after the terminal gate."""

    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    eligible = {
        str(segment["run_id"]): int(segment["segment_index"])
        for segment in transport["segments"]
        if segment["eligible_for_condition_rebuild"]
    }
    if not eligible:
        raise RuntimeError("Round 21 core corpus has no eligible terminal run")
    conditions: list[Round21CoreCondition] = []
    with PolymarketEvidenceStore(
        Path(database),
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for run_id, segment_index in eligible.items():
            rows = store.connect().execute(
                """
                SELECT snapshot_id, run_id, observed_wall_ms,
                       observed_monotonic_ns, asset, condition_id,
                       event_start_ms, end_ms, up_token_id, down_token_id,
                       clob_info_sha256, up_fee_rate_sha256,
                       down_fee_rate_sha256, maker_base_fee, taker_base_fee,
                       taker_order_delay_enabled, minimum_order_age_seconds,
                       snapshot_payload_json, snapshot_sha256
                FROM polymarket_market_snapshot
                WHERE run_id = ?
                ORDER BY event_start_ms, condition_id
                """,
                [run_id],
            ).fetchall()
            conditions.extend(
                _snapshot_condition(row, segment_index=segment_index)
                for row in rows
            )
    identities = [condition.condition_id for condition in conditions]
    if not conditions or len(set(identities)) != len(identities):
        raise ValueError("Round 21 core condition identity set differs")
    return tuple(
        sorted(
            conditions,
            key=lambda value: (
                value.segment_index,
                value.event_start_ms,
                value.condition_id,
            ),
        )
    )


@dataclass(slots=True)
class _ConditionAccumulator:
    condition: Round21CoreCondition
    union_events: list[PolymarketUnionEvent] = field(default_factory=list)
    chainlink_records: list[CaptureFrameRecord] = field(default_factory=list)
    lane_event_wall_ms: dict[str, list[int]] = field(
        default_factory=lambda: {lane: [] for lane in _LANES}
    )


class Round21CoreCorpusObserver:
    """Build and release one condition at a time during terminal replay."""

    def __init__(
        self,
        *,
        conditions: Sequence[Round21CoreCondition],
        partition_policy: Round21PartitionPolicy,
        sink: Callable[[Round21CoreConditionMaterialization], None],
    ) -> None:
        selected = tuple(condition.validated() for condition in conditions)
        if (
            not selected
            or len({condition.condition_id for condition in selected}) != len(selected)
            or not callable(sink)
        ):
            raise ValueError("Round 21 core observer conditions differ")
        self.conditions = tuple(
            sorted(
                selected,
                key=lambda value: (
                    value.segment_index,
                    value.event_start_ms,
                    value.condition_id,
                ),
            )
        )
        self.partition_policy = partition_policy.validated()
        self.sink = sink
        self._run_conditions: tuple[Round21CoreCondition, ...] = ()
        self._condition_by_id: dict[str, Round21CoreCondition] = {}
        self._active: dict[str, _ConditionAccumulator] = {}
        self._next_condition_index = 0
        self._finalized: set[str] = set()
        self._gaps: tuple[StreamGap, ...] = ()
        self._builder: PolymarketRedundantUnionBuilder | None = None
        self._run_id = ""
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._initial_clock_offset_ns: int | None = None
        self.materialized_condition_count = 0

    def start_run(
        self,
        segment: Mapping[str, object],
        gaps: tuple[StreamGap, ...],
    ) -> None:
        if self._builder is not None or self._run_id:
            raise RuntimeError("Round 21 core observer run is already open")
        run_id = str(segment.get("run_id") or "")
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("Round 21 core observer run identity differs")
        run_conditions = tuple(
            condition for condition in self.conditions if condition.run_id == run_id
        )
        self._run_conditions = run_conditions
        self._condition_by_id = {
            condition.condition_id: condition for condition in run_conditions
        }
        self._active = {}
        self._next_condition_index = 0
        self._finalized = set()
        self._gaps = tuple(gap.validated() for gap in gaps)
        self._builder = PolymarketRedundantUnionBuilder(
            pairing_window_ms=_PAIRING_WINDOW_MS,
            maximum_pending_events=100_000,
        )
        self._run_id = run_id
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._initial_clock_offset_ns = None

    def _activate(self, wall_ms: int) -> None:
        while self._next_condition_index < len(self._run_conditions):
            condition = self._run_conditions[self._next_condition_index]
            if condition.event_start_ms - 120_000 > wall_ms:
                break
            self._active[condition.condition_id] = _ConditionAccumulator(condition)
            self._next_condition_index += 1

    def _route_union(self, event: PolymarketUnionEvent) -> None:
        payload = validate_round21_union_event(event)
        condition_id = str(payload.get("market") or "").strip().lower()
        condition = self._condition_by_id.get(condition_id)
        if condition is None:
            return
        if condition_id in self._finalized:
            raise ValueError("Round 21 union event arrived after condition finalization")
        accumulator = self._active.get(condition_id)
        if accumulator is None:
            raise ValueError("Round 21 union event condition was not activated")
        if (
            condition.event_start_ms - 120_000
            <= event.selected_received_wall_ms
            <= condition.event_end_ms
        ):
            accumulator.union_events.append(event)

    def _finalize_ready(self, wall_ms: int, *, force: bool = False) -> None:
        ready = [
            condition_id
            for condition_id, accumulator in self._active.items()
            if force
            or wall_ms
            > accumulator.condition.event_end_ms + _CHAINLINK_TERMINAL_GRACE_MS
        ]
        for condition_id in sorted(
            ready,
            key=lambda value: (
                self._active[value].condition.event_start_ms,
                value,
            ),
        ):
            accumulator = self._active.pop(condition_id)
            source = Round21ConditionSource(
                union_events=tuple(accumulator.union_events),
                chainlink_records=tuple(accumulator.chainlink_records),
                lane_event_wall_ms={
                    lane: tuple(accumulator.lane_event_wall_ms[lane])
                    for lane in _LANES
                },
                stream_gaps=self._gaps,
            )
            result = build_round21_core_condition_materialization(
                condition=accumulator.condition,
                source=source,
                partition_policy=self.partition_policy,
            )
            self.sink(result)
            self._finalized.add(condition_id)
            self.materialized_condition_count += 1

    def observe_message(
        self,
        segment: Mapping[str, object],
        message: RawStreamMessage,
    ) -> None:
        if self._builder is None or str(segment.get("run_id") or "") != self._run_id:
            raise RuntimeError("Round 21 core observer run is unavailable")
        selected = message.validated()
        if (
            selected.received_monotonic_ns < self._last_monotonic_ns
            or selected.received_wall_ms < self._last_wall_ms
        ):
            raise ValueError("Round 21 terminal receipt clock regressed")
        offset_ns = (
            selected.received_wall_ms * 1_000_000
            - selected.received_monotonic_ns
        )
        if self._initial_clock_offset_ns is None:
            self._initial_clock_offset_ns = offset_ns
        elif (
            abs(offset_ns - self._initial_clock_offset_ns)
            > _MAXIMUM_CLOCK_OFFSET_DRIFT_NS
        ):
            raise ValueError("Round 21 terminal receipt clock offset drifted")
        self._last_monotonic_ns = selected.received_monotonic_ns
        self._last_wall_ms = selected.received_wall_ms
        self._activate(selected.received_wall_ms)
        if selected.stream == "clob_market":
            lane = selected.connection_id.partition(":")[0]
            if lane not in _LANES:
                raise ValueError("Round 21 CLOB receipt has an unknown lane")
            if selected.raw_text != "PONG":
                for accumulator in self._active.values():
                    condition = accumulator.condition
                    if (
                        condition.event_start_ms
                        <= selected.received_wall_ms
                        <= condition.event_end_ms
                    ):
                        accumulator.lane_event_wall_ms[lane].append(
                            selected.received_wall_ms
                        )
            ready = self._builder.add(
                PolymarketClobLaneReceipt(
                    lane_id=lane,
                    connection_id=selected.connection_id,
                    sequence_number=selected.sequence_number,
                    received_wall_ms=selected.received_wall_ms,
                    received_monotonic_ns=selected.received_monotonic_ns,
                    raw_text=selected.raw_text,
                )
            )
            for event in ready:
                self._route_union(event)
        elif selected.stream == "polymarket_rtds":
            for event in self._builder.advance(selected.received_monotonic_ns):
                self._route_union(event)
            record = CaptureFrameRecord(**selected.__dict__)
            for accumulator in self._active.values():
                condition = accumulator.condition
                if (
                    condition.event_start_ms
                    <= selected.received_wall_ms
                    <= condition.event_end_ms + _CHAINLINK_TERMINAL_GRACE_MS
                ):
                    accumulator.chainlink_records.append(record)
        self._finalize_ready(selected.received_wall_ms)

    def finish_run(self, segment: Mapping[str, object]) -> None:
        if self._builder is None or str(segment.get("run_id") or "") != self._run_id:
            raise RuntimeError("Round 21 core observer run is unavailable")
        trailing, _audit = self._builder.finish()
        for event in trailing:
            self._route_union(event)
        self._activate(2**63 - 1)
        self._finalize_ready(2**63 - 1, force=True)
        if (
            self._active
            or self._next_condition_index != len(self._run_conditions)
            or self._finalized != set(self._condition_by_id)
        ):
            raise RuntimeError("Round 21 core observer condition accounting differs")
        self._run_conditions = ()
        self._condition_by_id = {}
        self._gaps = ()
        self._builder = None
        self._run_id = ""
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._initial_clock_offset_ns = None


__all__ = [
    "POLYMARKET_ROUND21_CONDITION_ADMISSION_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256",
    "Round21ConditionSource",
    "Round21CoreCorpusObserver",
    "Round21CoreCondition",
    "Round21CoreConditionMaterialization",
    "build_round21_core_condition_materialization",
    "load_round21_core_corpus_design",
    "load_round21_core_conditions",
    "validate_round21_condition_admission",
]

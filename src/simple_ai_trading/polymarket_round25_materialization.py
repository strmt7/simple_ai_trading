"""Retired point-stream materialization retained for Round 24 provenance only."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

from .polymarket import parse_polymarket_five_minute_market
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_recorder import PolymarketEvidenceStore, RawStreamMessage, StreamGap
from .polymarket_redundant_union import PolymarketUnionEvent
from .polymarket_round21_core_features import (
    Round21CoreFeatureEngine,
    Round21CoreFeatureSnapshot,
)
from .polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
)
from .polymarket_round25_campaign import POLYMARKET_ROUND25_RESOLUTION_SOURCE
from .polymarket_round25_joint_materialization import Round25SingleLaneClobDecoder
from .polymarket_round25_terminal import (
    POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    audit_round25_terminal_receipts,
    validate_round25_terminal_receipt_audit,
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND25_MATERIALIZATION_SCHEMA_VERSION = (
    "polymarket-round25-round24-causal-materialization-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{1,80}$")
_CONDITION_DURATION_MS = 300_000
_GAP_LOOKBACK_MS = 120_000
_TERMINAL_GRACE_MS = 5_000
_DECISION_INTERVAL_MS = 1_000
_MAXIMUM_CLOCK_OFFSET_DRIFT_NS = 1_000_000_000
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_AUTHORITY_FIELDS = (
    "model_data_eligible",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 materialization JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 materialization JSON contains {value}")


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


def _chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _strict_json_text(value: object, *, label: str) -> dict[str, object]:
    try:
        selected = json.loads(
            str(value),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 {label} is not strict JSON") from exc
    if not isinstance(selected, dict) or _canonical_json(selected) != value:
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    return selected


def round24_role_for_event_start(
    event_start_ms: int,
    partitions: Sequence[Mapping[str, object]],
) -> str | None:
    """Assign one whole condition to an explicit closed-open Round 24 interval."""

    selected = int(event_start_ms)
    matches: list[str] = []
    previous_end = -1
    for item in partitions:
        role = str(item.get("role") or "")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if (
            role not in {"train", "tune_calibration", "tune_selection"}
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or start < previous_end
        ):
            raise ValueError("Round 24 explicit partition intervals differ")
        previous_end = end
        if start <= selected < end:
            matches.append(role)
    if len(matches) > 1:
        raise ValueError("Round 24 condition crosses explicit partition roles")
    return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class Round25ReceiptCondition:
    run_id: str
    segment_index: int
    snapshot_sha256: str
    snapshot_observed_wall_ms: int
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    up_token_id: str
    down_token_id: str
    role: str

    def validated(self) -> Round25ReceiptCondition:
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
            or self.event_start_ms % _CONDITION_DURATION_MS
            or self.event_end_ms != self.event_start_ms + _CONDITION_DURATION_MS
            or _TOKEN_ID.fullmatch(self.up_token_id) is None
            or _TOKEN_ID.fullmatch(self.down_token_id) is None
            or self.up_token_id == self.down_token_id
            or self.role not in {"train", "tune_calibration", "tune_selection"}
        ):
            raise ValueError("Round 25 receipt condition identity differs")
        return self


def _snapshot_condition(
    row: Sequence[object],
    *,
    segment_index: int,
    partitions: Sequence[Mapping[str, object]],
) -> tuple[Round25ReceiptCondition | None, bool]:
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
        resolution_source,
        gamma_payload_json,
        gamma_payload_sha256,
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
    payload = _strict_json_text(snapshot_payload_json, label="market snapshot")
    if _canonical_sha256(payload) != snapshot_sha256:
        raise ValueError("Round 25 market snapshot hash differs")
    market = payload.get("market")
    if not isinstance(market, Mapping):
        raise ValueError("Round 25 market snapshot identity is unavailable")
    gamma = _strict_json_text(gamma_payload_json, label="Gamma market snapshot")
    parsed_market = parse_polymarket_five_minute_market(gamma)
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
        or payload
        != {
            **identity,
            "observed_monotonic_ns": observed_monotonic_ns,
            "maker_base_fee": maker_base_fee,
            "taker_base_fee": taker_base_fee,
            "taker_order_delay_enabled": taker_order_delay_enabled,
            "minimum_order_age_seconds": minimum_order_age_seconds,
        }
        or parsed_market.asdict() != market
        or _canonical_sha256(gamma) != gamma_payload_sha256
        or parsed_market.asset != asset
        or parsed_market.condition_id != condition_id
        or parsed_market.event_start_ms != event_start_ms
        or parsed_market.end_ms != event_end_ms
        or parsed_market.up_token_id != up_token_id
        or parsed_market.down_token_id != down_token_id
        or parsed_market.resolution_source != resolution_source
        or resolution_source != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or any(
            _SHA256.fullmatch(str(value or "")) is None
            for value in (
                snapshot_id,
                snapshot_sha256,
                gamma_payload_sha256,
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
        raise ValueError("Round 25 market snapshot columns differ")
    role = round24_role_for_event_start(int(event_start_ms), partitions)
    if role is None:
        return None, True
    return (
        Round25ReceiptCondition(
            run_id=str(run_id),
            segment_index=segment_index,
            snapshot_sha256=str(snapshot_sha256),
            snapshot_observed_wall_ms=int(observed_wall_ms),
            condition_id=str(condition_id),
            event_start_ms=int(event_start_ms),
            event_end_ms=int(event_end_ms),
            up_token_id=str(up_token_id),
            down_token_id=str(down_token_id),
            role=role,
        ).validated(),
        False,
    )


def load_round25_receipt_conditions(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partitions: Sequence[Mapping[str, object]],
) -> tuple[tuple[Round25ReceiptCondition, ...], dict[str, int]]:
    """Load and verify only source identities; no receipt, target, or outcome is read."""

    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    if (
        transport.get("source_capture_design_sha256")
        == POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256
    ):
        raise RuntimeError(
            "Round 24 point-stream materialization is retired for Round 25 v2; "
            "use the exact-TWAP joint feature store"
        )
    eligible = {
        str(item["run_id"]): (int(item["segment_index"]), str(item["status"]))
        for item in transport["segments"]
        if item["eligible_for_condition_rebuild"]
    }
    if not eligible:
        raise ValueError("Round 25 eligible terminal run set is empty")
    path = Path(database).resolve()
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError(
            "Round 25 materialization requires a terminal WAL-free database"
        )
    conditions: list[Round25ReceiptCondition] = []
    source_count = 0
    excluded_count = 0
    with PolymarketEvidenceStore(
        path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for run_id, (segment_index, expected_status) in eligible.items():
            status_row = (
                store.connect()
                .execute(
                    "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
                    [run_id],
                )
                .fetchone()
            )
            if status_row != (expected_status,):
                raise ValueError("Round 25 materialization database status differs")
            rows = (
                store.connect()
                .execute(
                    """
                SELECT snapshot_id, run_id, observed_wall_ms,
                       observed_monotonic_ns, asset, condition_id,
                       event_start_ms, end_ms, up_token_id, down_token_id,
                       resolution_source, gamma_payload_json,
                       gamma_payload_sha256, clob_info_sha256,
                       up_fee_rate_sha256, down_fee_rate_sha256,
                       maker_base_fee, taker_base_fee,
                       taker_order_delay_enabled, minimum_order_age_seconds,
                       snapshot_payload_json, snapshot_sha256
                FROM polymarket_market_snapshot
                WHERE run_id = ?
                ORDER BY event_start_ms, condition_id
                """,
                    [run_id],
                )
                .fetchall()
            )
            for row in rows:
                source_count += 1
                condition, excluded = _snapshot_condition(
                    row,
                    segment_index=segment_index,
                    partitions=partitions,
                )
                excluded_count += int(excluded)
                if condition is not None:
                    conditions.append(condition)
    identities = [item.condition_id for item in conditions]
    if not conditions or len(set(identities)) != len(identities):
        raise ValueError("Round 25 receipt condition identity set differs")
    return (
        tuple(
            sorted(
                conditions,
                key=lambda item: (
                    item.segment_index,
                    item.event_start_ms,
                    item.condition_id,
                ),
            )
        ),
        {
            "source_condition_count": source_count,
            "partition_condition_count": len(conditions),
            "outside_partition_condition_count": excluded_count,
        },
    )


@dataclass(slots=True)
class _ConditionAccumulator:
    condition: Round25ReceiptCondition
    observations: list[tuple[int, int, int, object]] = field(default_factory=list)
    rejection_reasons: set[str] = field(default_factory=set)


class Round25ReceiptMaterializerObserver:
    """Collect bounded condition sources during one exact terminal receipt pass."""

    def __init__(self, conditions: Sequence[Round25ReceiptCondition]) -> None:
        selected = tuple(item.validated() for item in conditions)
        if not selected or len({item.condition_id for item in selected}) != len(
            selected
        ):
            raise ValueError("Round 25 materializer observer conditions differ")
        self.conditions = tuple(
            sorted(
                selected,
                key=lambda item: (
                    item.segment_index,
                    item.event_start_ms,
                    item.condition_id,
                ),
            )
        )
        self.snapshots: list[Round21CoreFeatureSnapshot] = []
        self.rejection_counts: Counter[str] = Counter()
        self.unavailable_snapshot_counts: Counter[str] = Counter()
        self._run_conditions: tuple[Round25ReceiptCondition, ...] = ()
        self._active: dict[str, _ConditionAccumulator] = {}
        self._finalized: set[str] = set()
        self._run_id = ""
        self._decoder: Round25SingleLaneClobDecoder | None = None
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._clock_offset_ns: int | None = None
        self._last_connection: dict[str, str] = {}
        self._last_sequence: dict[tuple[str, str], int] = {}
        self._ordinal = 0

    @staticmethod
    def _window(condition: Round25ReceiptCondition) -> tuple[int, int]:
        return (
            condition.event_start_ms - _GAP_LOOKBACK_MS,
            condition.event_end_ms + _TERMINAL_GRACE_MS,
        )

    def _mark_at(self, wall_ms: int, reason: str) -> None:
        for condition in self._run_conditions:
            start, end = self._window(condition)
            if start <= wall_ms <= end:
                accumulator = self._active.get(condition.condition_id)
                if accumulator is not None:
                    accumulator.rejection_reasons.add(reason)

    def start_run(
        self,
        segment: Mapping[str, object],
        gaps: tuple[StreamGap, ...],
    ) -> None:
        if self._run_id:
            raise RuntimeError("Round 25 materializer observer run is already open")
        run_id = str(segment.get("run_id") or "")
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("Round 25 materializer run identity differs")
        self._run_conditions = tuple(
            item for item in self.conditions if item.run_id == run_id
        )
        self._active = {
            item.condition_id: _ConditionAccumulator(item)
            for item in self._run_conditions
        }
        self._finalized = set()
        self._run_id = run_id
        self._decoder = Round25SingleLaneClobDecoder()
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._clock_offset_ns = None
        self._last_connection = {}
        self._last_sequence = {}
        self._ordinal = 0
        segment_start = int(segment["started_at_ms"])
        segment_end = int(segment["ended_at_ms"])
        for accumulator in self._active.values():
            condition = accumulator.condition
            start, end = self._window(condition)
            if (
                start < segment_start
                or end > segment_end
                or not segment_start
                <= condition.snapshot_observed_wall_ms
                <= segment_end
            ):
                accumulator.rejection_reasons.add("cross_segment_condition_window")
        for gap in gaps:
            selected = gap.validated()
            self._mark_at(
                selected.opened_at_ms,
                f"stream_gap:{selected.stream}",
            )

    def _finalize_ready(self, wall_ms: int, *, force: bool = False) -> None:
        ready = [
            condition_id
            for condition_id, accumulator in self._active.items()
            if force
            or wall_ms > accumulator.condition.event_end_ms + _TERMINAL_GRACE_MS
        ]
        for condition_id in sorted(
            ready,
            key=lambda item: (
                self._active[item].condition.event_start_ms,
                item,
            ),
        ):
            accumulator = self._active.pop(condition_id)
            if accumulator.rejection_reasons:
                for reason in sorted(accumulator.rejection_reasons):
                    self.rejection_counts[reason] += 1
            else:
                snapshots, unavailable = _materialize_condition(accumulator)
                self.unavailable_snapshot_counts.update(unavailable)
                if snapshots:
                    self.snapshots.extend(snapshots)
                else:
                    self.rejection_counts["no_available_core_snapshots"] += 1
            self._finalized.add(condition_id)

    def observe_message(
        self,
        segment: Mapping[str, object],
        message: RawStreamMessage,
    ) -> None:
        if (
            not self._run_id
            or self._decoder is None
            or str(segment.get("run_id") or "") != self._run_id
        ):
            raise RuntimeError("Round 25 materializer observer run is unavailable")
        selected = message.validated()
        if (
            selected.received_monotonic_ns < self._last_monotonic_ns
            or selected.received_wall_ms < self._last_wall_ms
        ):
            raise ValueError("Round 25 terminal receipt clock regressed")
        offset_ns = (
            selected.received_wall_ms * 1_000_000 - selected.received_monotonic_ns
        )
        if self._clock_offset_ns is None:
            self._clock_offset_ns = offset_ns
        elif abs(offset_ns - self._clock_offset_ns) > _MAXIMUM_CLOCK_OFFSET_DRIFT_NS:
            raise ValueError("Round 25 terminal receipt clock offset drifted")
        previous_connection = self._last_connection.get(selected.stream)
        if (
            previous_connection is not None
            and previous_connection != selected.connection_id
        ):
            self._mark_at(
                selected.received_wall_ms,
                f"unledgered_connection_change:{selected.stream}",
            )
        key = (selected.stream, selected.connection_id)
        previous_sequence = self._last_sequence.get(key)
        if previous_sequence is None:
            if selected.sequence_number != 1:
                self._mark_at(
                    selected.received_wall_ms,
                    f"receipt_sequence_start:{selected.stream}",
                )
        elif selected.sequence_number != previous_sequence + 1:
            self._mark_at(
                selected.received_wall_ms,
                f"receipt_sequence_gap:{selected.stream}",
            )
        self._last_connection[selected.stream] = selected.connection_id
        self._last_sequence[key] = selected.sequence_number
        self._last_monotonic_ns = selected.received_monotonic_ns
        self._last_wall_ms = selected.received_wall_ms
        self._ordinal += 1
        if selected.stream == "clob_market":
            for event, condition_id in self._decoder.add(selected):
                accumulator = self._active.get(condition_id)
                if accumulator is None:
                    continue
                start, end = self._window(accumulator.condition)
                if (
                    not accumulator.rejection_reasons
                    and start <= selected.received_wall_ms <= end
                ):
                    accumulator.observations.append(
                        (
                            selected.received_monotonic_ns,
                            selected.received_wall_ms,
                            self._ordinal,
                            event,
                        )
                    )
        elif selected.stream == "polymarket_rtds":
            record = CaptureFrameRecord(**selected.__dict__)
            for accumulator in self._active.values():
                start, end = self._window(accumulator.condition)
                if (
                    not accumulator.rejection_reasons
                    and start <= selected.received_wall_ms <= end
                ):
                    accumulator.observations.append(
                        (
                            selected.received_monotonic_ns,
                            selected.received_wall_ms,
                            self._ordinal,
                            record,
                        )
                    )
        else:
            raise ValueError("Round 25 core materializer stream differs")
        self._finalize_ready(selected.received_wall_ms)

    def finish_run(self, segment: Mapping[str, object]) -> None:
        if not self._run_id or str(segment.get("run_id") or "") != self._run_id:
            raise RuntimeError("Round 25 materializer observer run is unavailable")
        self._finalize_ready(2**63 - 1, force=True)
        expected = {item.condition_id for item in self._run_conditions}
        if self._active or self._finalized != expected:
            raise RuntimeError("Round 25 materializer condition accounting differs")
        self._run_conditions = ()
        self._active = {}
        self._finalized = set()
        self._run_id = ""
        self._decoder = None
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._clock_offset_ns = None
        self._last_connection = {}
        self._last_sequence = {}
        self._ordinal = 0


def _materialize_condition(
    accumulator: _ConditionAccumulator,
) -> tuple[tuple[Round21CoreFeatureSnapshot, ...], Counter[str]]:
    condition = accumulator.condition
    if any(
        current[:3] < previous[:3]
        for previous, current in zip(
            accumulator.observations,
            accumulator.observations[1:],
            strict=False,
        )
    ):
        raise ValueError("Round 25 in-memory receipt chronology differs")
    ordered = tuple(accumulator.observations)
    engine = Round21CoreFeatureEngine(
        condition_id=condition.condition_id,
        up_token_id=condition.up_token_id,
        down_token_id=condition.down_token_id,
        event_start_ms=condition.event_start_ms,
    )
    snapshots: list[Round21CoreFeatureSnapshot] = []
    unavailable: Counter[str] = Counter()
    index = 0
    for decision_time_ms in range(
        condition.event_start_ms,
        condition.event_end_ms,
        _DECISION_INTERVAL_MS,
    ):
        while index < len(ordered) and ordered[index][1] <= decision_time_ms:
            source = ordered[index][3]
            if isinstance(source, CaptureFrameRecord):
                engine.ingest_chainlink_record(source)
            elif isinstance(source, PolymarketUnionEvent):
                engine.ingest_union_event(source)
            else:
                raise TypeError("Round 25 materializer source type differs")
            index += 1
        snapshot = engine.build(decision_time_ms)
        if snapshot.available:
            if snapshot.maximum_receipt_ms > decision_time_ms:
                raise ValueError("Round 25 materializer used a future receipt")
            snapshots.append(snapshot)
        else:
            unavailable.update(snapshot.reasons)
    return tuple(snapshots), unavailable


def validate_round25_materialization_report(
    value: Mapping[str, object],
    *,
    terminal_transport_manifest: Mapping[str, object],
    terminal_receipt_audit: Mapping[str, object],
) -> dict[str, object]:
    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    receipt = validate_round25_terminal_receipt_audit(
        terminal_receipt_audit,
        terminal_transport_manifest=transport,
    )
    payload = dict(value)
    claimed = str(payload.pop("materialization_sha256", "")).lower()
    expected = {
        "schema_version",
        "terminal_design_sha256",
        "terminal_transport_manifest_sha256",
        "terminal_receipt_audit_sha256",
        "round24_specification_sha256",
        "source_condition_count",
        "partition_condition_count",
        "outside_partition_condition_count",
        "admitted_condition_count",
        "rejected_condition_count",
        "rejection_counts",
        "core_snapshot_count",
        "unavailable_snapshot_counts",
        "core_snapshot_chain_sha256",
        "one_authoritative_clob_lane",
        "redundant_lane_coverage_claim",
        "official_resolution_accessed",
        "twap_outcome_reconstructed",
        "round24_twap_settlement_label_constructed",
        "future_receipts_permitted",
        "receipt_scan_count",
        *_AUTHORITY_FIELDS,
    }
    integer_fields = (
        "source_condition_count",
        "partition_condition_count",
        "outside_partition_condition_count",
        "admitted_condition_count",
        "rejected_condition_count",
        "core_snapshot_count",
        "receipt_scan_count",
    )
    if (
        set(payload) != expected
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_MATERIALIZATION_SCHEMA_VERSION
        or payload.get("terminal_design_sha256")
        != POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256
        or payload.get("terminal_transport_manifest_sha256")
        != transport["manifest_sha256"]
        or payload.get("terminal_receipt_audit_sha256") != receipt["audit_sha256"]
        or _SHA256.fullmatch(str(payload.get("round24_specification_sha256") or ""))
        is None
        or _SHA256.fullmatch(str(payload.get("core_snapshot_chain_sha256") or ""))
        is None
        or any(type(payload.get(field)) is not int for field in integer_fields)
        or any(int(payload[field]) < 0 for field in integer_fields)
        or payload.get("partition_condition_count")
        != payload.get("admitted_condition_count")
        + payload.get("rejected_condition_count")
        or payload.get("source_condition_count")
        != payload.get("partition_condition_count")
        + payload.get("outside_partition_condition_count")
        or payload.get("receipt_scan_count") != 1
        or payload.get("one_authoritative_clob_lane") is not True
        or payload.get("redundant_lane_coverage_claim") is not False
        or payload.get("official_resolution_accessed") is not False
        or payload.get("twap_outcome_reconstructed") is not False
        or payload.get("round24_twap_settlement_label_constructed") is not False
        or payload.get("future_receipts_permitted") is not False
        or any(payload.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 materialization report differs")
    for count_field in ("rejection_counts", "unavailable_snapshot_counts"):
        counts = payload.get(count_field)
        if not isinstance(counts, Mapping) or any(
            not isinstance(key, str) or not key or type(count) is not int or count <= 0
            for key, count in counts.items()
        ):
            raise ValueError("Round 25 materialization report counts differ")
    return {**payload, "materialization_sha256": claimed}


def materialize_round25_round24_core(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partitions: Sequence[Mapping[str, object]],
    round24_specification_sha256: str,
    observed_at_ms: int | None = None,
) -> tuple[
    tuple[Round21CoreFeatureSnapshot, ...],
    dict[str, object],
    dict[str, object],
]:
    """Replay exact receipts once and construct only target-blind core snapshots."""

    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    conditions, population = load_round25_receipt_conditions(
        database=database,
        terminal_transport_manifest=transport,
        partitions=partitions,
    )
    observer = Round25ReceiptMaterializerObserver(conditions)
    receipt_audit = audit_round25_terminal_receipts(
        database=database,
        terminal_transport_manifest=transport,
        observed_at_ms=observed_at_ms,
        observer=observer,
    )
    snapshots = tuple(
        sorted(
            observer.snapshots,
            key=lambda item: (
                item.event_start_ms,
                item.condition_id,
                item.decision_time_ms,
            ),
        )
    )
    keys = [(item.condition_id, item.decision_time_ms) for item in snapshots]
    if not snapshots or len(set(keys)) != len(keys):
        raise ValueError("Round 25 materialized core snapshot population differs")
    chain = _EMPTY_SHA256
    admitted_conditions: set[str] = set()
    for snapshot in snapshots:
        admitted_conditions.add(snapshot.condition_id)
        chain = _chain(
            chain,
            {
                "condition_id": snapshot.condition_id,
                "event_start_ms": snapshot.event_start_ms,
                "decision_time_ms": snapshot.decision_time_ms,
                "market_prior_probability": snapshot.market_prior_probability,
                "values": list(snapshot.values),
                "source_chain_sha256": snapshot.source_chain_sha256,
                "maximum_receipt_ms": snapshot.maximum_receipt_ms,
            },
        )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_MATERIALIZATION_SCHEMA_VERSION,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
        "terminal_transport_manifest_sha256": transport["manifest_sha256"],
        "terminal_receipt_audit_sha256": receipt_audit["audit_sha256"],
        "round24_specification_sha256": str(round24_specification_sha256).lower(),
        **population,
        "admitted_condition_count": len(admitted_conditions),
        "rejected_condition_count": (
            int(population["partition_condition_count"]) - len(admitted_conditions)
        ),
        "rejection_counts": dict(sorted(observer.rejection_counts.items())),
        "core_snapshot_count": len(snapshots),
        "unavailable_snapshot_counts": dict(
            sorted(observer.unavailable_snapshot_counts.items())
        ),
        "core_snapshot_chain_sha256": chain,
        "one_authoritative_clob_lane": True,
        "redundant_lane_coverage_claim": False,
        "official_resolution_accessed": False,
        "twap_outcome_reconstructed": False,
        "round24_twap_settlement_label_constructed": False,
        "future_receipts_permitted": False,
        "receipt_scan_count": 1,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    body["materialization_sha256"] = _canonical_sha256(body)
    report = validate_round25_materialization_report(
        body,
        terminal_transport_manifest=transport,
        terminal_receipt_audit=receipt_audit,
    )
    return snapshots, report, receipt_audit


__all__ = [
    "POLYMARKET_ROUND25_MATERIALIZATION_SCHEMA_VERSION",
    "Round25ReceiptCondition",
    "Round25ReceiptMaterializerObserver",
    "Round25SingleLaneClobDecoder",
    "load_round25_receipt_conditions",
    "materialize_round25_round24_core",
    "round24_role_for_event_start",
    "validate_round25_materialization_report",
]

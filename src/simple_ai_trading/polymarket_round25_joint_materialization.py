"""Exact-receipt materialization of bounded Round 25 joint feature contexts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re

from .paper_execution import PaperBookSnapshot
from .polymarket import parse_polymarket_five_minute_market
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_recorder import PolymarketEvidenceStore, RawStreamMessage, StreamGap
from .polymarket_redundant_union import (
    POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION,
    PolymarketClobLaneReceipt,
    PolymarketUnionEvent,
)
from .polymarket_execution_books import build_polymarket_execution_books
from .polymarket_round21_core_features import validate_round21_union_event
from .polymarket_round25_campaign import POLYMARKET_ROUND25_RESOLUTION_SOURCE
from .polymarket_round25_clob_features import Round25ClobFeatureEngine
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
    round25_development_role,
    select_round25_condition_endpoints,
)
from .polymarket_round25_joint_features import (
    Round25JointFeatureSnapshot,
    combine_round25_features,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_DECISION_CADENCE_MS,
    Round25TwapFeatureEngine,
    Round25TwapObservation,
)
from .polymarket_round25_terminal import (
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256 = (
    "57fb2cf088bdd64e3b6e05d087d5d4aa3d77cc484b8611be77b02cd12a34a6c2"
)
POLYMARKET_ROUND25_CONDITION_MATERIALIZATION_SCHEMA_VERSION = (
    "polymarket-round25-condition-joint-feature-materialization-v2"
)
POLYMARKET_ROUND25_SEQUENCE_CONTEXT_STEPS = 64
POLYMARKET_ROUND25_EXPECTED_DECISIONS = 1_200
POLYMARKET_ROUND25_GAP_LOOKBACK_MS = 120_000
POLYMARKET_ROUND25_TERMINAL_GRACE_MS = 5_000
POLYMARKET_ROUND25_MAXIMUM_CLOCK_OFFSET_DRIFT_NS = 1_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{1,80}$")
_MARKET_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_BTC_FIVE_MINUTE_SLUG = re.compile(r"^btc-updown-5m-[1-9][0-9]{9}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 joint materialization JSON has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 joint materialization JSON contains {value}")


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


def _strict_json_text(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    try:
        selected = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 {label} is not strict JSON") from exc
    if not isinstance(selected, dict) or _canonical_json(selected) != value:
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    return selected


def round25_joint_snapshot_sha256(snapshot: Round25JointFeatureSnapshot) -> str:
    return _canonical_sha256({
        "available": snapshot.available,
        "clob_source_chain_sha256": snapshot.clob_source_chain_sha256,
        "condition_id": snapshot.condition_id,
        "decision_time_ms": snapshot.decision_time_ms,
        "event_start_ms": snapshot.event_start_ms,
        "market_prior_probability": snapshot.market_prior_probability,
        "maximum_receipt_ms": snapshot.maximum_receipt_ms,
        "model_design_sha256": snapshot.model_design_sha256,
        "reasons": list(snapshot.reasons),
        "source_chain_sha256": snapshot.source_chain_sha256,
        "trading_authority": snapshot.trading_authority,
        "twap_source_chain_sha256": snapshot.twap_source_chain_sha256,
        "values": list(snapshot.values),
    })


def _source_monotonic_ns(
    source: CaptureFrameRecord | PolymarketUnionEvent,
) -> int:
    if isinstance(source, CaptureFrameRecord):
        return source.received_monotonic_ns
    if isinstance(source, PolymarketUnionEvent):
        return source.selected_received_monotonic_ns
    raise TypeError("Round 25 joint feature source type differs")


def decode_round25_twap_record(
    record: CaptureFrameRecord,
) -> Round25TwapObservation | None:
    """Decode one strict live update and validate permitted RTDS controls."""

    if not isinstance(record, CaptureFrameRecord):
        raise TypeError("Round 25 TWAP source record type differs")
    if (
        record.stream != "polymarket_rtds"
        or not isinstance(record.connection_id, str)
        or not 1 <= len(record.connection_id) <= 160
        or type(record.sequence_number) is not int
        or record.sequence_number <= 0
        or type(record.received_wall_ms) is not int
        or record.received_wall_ms <= 0
        or type(record.received_monotonic_ns) is not int
        or record.received_monotonic_ns <= 0
        or not isinstance(record.raw_text, str)
    ):
        raise ValueError("Round 25 TWAP source record differs")
    if record.raw_text in {"", "PING", "PONG"}:
        return None
    return Round25TwapObservation.from_raw_frame(
        record.raw_text,
        received_wall_ms=record.received_wall_ms,
        received_monotonic_ns=record.received_monotonic_ns,
    )


class Round25SingleLaneClobDecoder:
    """Decode one authoritative CLOB lane into the existing union contract."""

    def __init__(self) -> None:
        self._union_sequence = 0
        self._occurrences: Counter[str] = Counter()

    def add(
        self,
        message: RawStreamMessage,
    ) -> tuple[tuple[PolymarketUnionEvent, str], ...]:
        selected = message.validated()
        if (
            selected.stream != "clob_market"
            or selected.connection_id.partition(":")[0] != "clob"
        ):
            raise ValueError("Round 25 authoritative CLOB lane differs")
        if selected.raw_text == "PONG":
            return ()
        try:
            decoded = json.loads(
                selected.raw_text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 25 CLOB frame is not strict JSON") from exc
        items = decoded if isinstance(decoded, list) else [decoded]
        if not items:
            raise ValueError("Round 25 CLOB frame contains no events")
        receipt = PolymarketClobLaneReceipt(
            lane_id="clob-a",
            connection_id=selected.connection_id,
            sequence_number=selected.sequence_number,
            received_wall_ms=selected.received_wall_ms,
            received_monotonic_ns=selected.received_monotonic_ns,
            raw_text=selected.raw_text,
        ).validated()
        output: list[tuple[PolymarketUnionEvent, str]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("Round 25 CLOB event is not an object")
            event_json = _canonical_json(item)
            semantic_sha256 = hashlib.sha256(event_json.encode("ascii")).hexdigest()
            self._union_sequence += 1
            self._occurrences[semantic_sha256] += 1
            source = item.get("timestamp")
            source_time = (
                int(source)
                if type(source) is int
                or isinstance(source, str)
                and source.isdigit()
                else None
            )
            identity = {
                "schema_version": POLYMARKET_REDUNDANT_UNION_SCHEMA_VERSION,
                "union_sequence_number": self._union_sequence,
                "semantic_sha256": semantic_sha256,
                "semantic_occurrence_index": self._occurrences[semantic_sha256],
                "event_type": str(item.get("event_type") or "unknown"),
                "source_time_ms": source_time,
                "selected_received_wall_ms": receipt.received_wall_ms,
                "selected_received_monotonic_ns": receipt.received_monotonic_ns,
                "selected_lane_id": receipt.lane_id,
                "lane_receipts": [
                    {
                        "lane_id": receipt.lane_id,
                        "connection_id": receipt.connection_id,
                        "sequence_number": receipt.sequence_number,
                        "received_wall_ms": receipt.received_wall_ms,
                        "received_monotonic_ns": receipt.received_monotonic_ns,
                    }
                ],
            }
            event = PolymarketUnionEvent(
                union_sequence_number=self._union_sequence,
                semantic_sha256=semantic_sha256,
                semantic_occurrence_index=self._occurrences[semantic_sha256],
                event_type=str(identity["event_type"]),
                event_json=event_json,
                source_time_ms=source_time,
                selected_received_wall_ms=receipt.received_wall_ms,
                selected_received_monotonic_ns=receipt.received_monotonic_ns,
                selected_lane_id=receipt.lane_id,
                lane_receipts=(receipt,),
                event_sha256=_canonical_sha256(identity),
            )
            validate_round21_union_event(event)
            output.append((event, str(item.get("market") or "").strip().lower()))
        return tuple(output)


@dataclass(frozen=True, slots=True)
class Round25JointReceiptCondition:
    run_id: str
    segment_index: int
    snapshot_sha256: str
    snapshot_observed_wall_ms: int
    market_id: str
    condition_id: str
    slug: str
    event_start_ms: int
    event_end_ms: int
    up_token_id: str
    down_token_id: str
    resolution_source: str
    role: str

    @property
    def asset(self) -> str:
        return "BTC"

    @property
    def end_ms(self) -> int:
        return self.event_end_ms

    @property
    def token_ids(self) -> tuple[str, str]:
        return self.up_token_id, self.down_token_id

    def validated(self) -> Round25JointReceiptCondition:
        if (
            _RUN_ID.fullmatch(self.run_id) is None
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or _SHA256.fullmatch(self.snapshot_sha256) is None
            or type(self.snapshot_observed_wall_ms) is not int
            or self.snapshot_observed_wall_ms <= 0
            or _MARKET_ID.fullmatch(self.market_id) is None
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or _BTC_FIVE_MINUTE_SLUG.fullmatch(self.slug) is None
            or type(self.event_start_ms) is not int
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or self.slug != f"btc-updown-5m-{self.event_start_ms // 1_000}"
            or self.event_end_ms != self.event_start_ms + 300_000
            or _TOKEN_ID.fullmatch(self.up_token_id) is None
            or _TOKEN_ID.fullmatch(self.down_token_id) is None
            or self.up_token_id == self.down_token_id
            or self.resolution_source != POLYMARKET_ROUND25_RESOLUTION_SOURCE
            or self.role not in {"train", "calibration", "selection"}
        ):
            raise ValueError("Round 25 joint receipt condition identity differs")
        return self


def _joint_condition_from_snapshot(
    row: Sequence[object],
    *,
    segment: Mapping[str, object],
) -> Round25JointReceiptCondition | None:
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
    gamma = _strict_json_text(gamma_payload_json, label="Gamma market snapshot")
    market = payload.get("market")
    if not isinstance(market, Mapping):
        raise ValueError("Round 25 market snapshot identity is unavailable")
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
        or _canonical_sha256(payload) != snapshot_sha256
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
        or parsed_market.asset != "BTC"
        or parsed_market.condition_id != condition_id
        or parsed_market.event_start_ms != event_start_ms
        or parsed_market.end_ms != event_end_ms
        or parsed_market.up_token_id != up_token_id
        or parsed_market.down_token_id != down_token_id
        or parsed_market.resolution_source != resolution_source
        or resolution_source != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or run_id != segment.get("run_id")
        or type(observed_wall_ms) is not int
        or not int(segment["started_at_ms"])
        <= observed_wall_ms
        <= int(segment["ended_at_ms"])
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
    role = round25_development_role(int(event_start_ms))
    if role == "purged":
        return None
    return Round25JointReceiptCondition(
        run_id=str(run_id),
        segment_index=int(segment["segment_index"]),
        snapshot_sha256=str(snapshot_sha256),
        snapshot_observed_wall_ms=int(observed_wall_ms),
        market_id=parsed_market.market_id,
        condition_id=str(condition_id),
        slug=parsed_market.slug,
        event_start_ms=int(event_start_ms),
        event_end_ms=int(event_end_ms),
        up_token_id=str(up_token_id),
        down_token_id=str(down_token_id),
        resolution_source=str(resolution_source),
        role=role,
    ).validated()


def load_round25_joint_receipt_conditions(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
) -> tuple[tuple[Round25JointReceiptCondition, ...], dict[str, int]]:
    """Load target-free source identities from a terminal WAL-free capture."""

    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    if transport["source_plan_sha256"] != POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256:
        raise ValueError("Round 25 joint materialization capture plan differs")
    eligible_segments: dict[str, Mapping[str, object]] = {}
    for item in transport["segments"]:
        if not item["eligible_for_condition_rebuild"]:
            continue
        run_id = str(item["run_id"])
        if run_id in eligible_segments:
            raise ValueError("Round 25 eligible terminal run identity is duplicated")
        eligible_segments[run_id] = item
    if not eligible_segments:
        raise ValueError("Round 25 eligible terminal run set is empty")

    source_path = Path(database)
    if source_path.is_symlink():
        raise ValueError("Round 25 materialization database cannot be a symlink")
    try:
        path = source_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Round 25 materialization database is unavailable") from exc
    if not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError(
            "Round 25 materialization requires a terminal WAL-free database"
        )

    conditions: list[Round25JointReceiptCondition] = []
    source_count = 0
    purged_count = 0
    with PolymarketEvidenceStore(
        path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        connection = store.connect()
        for run_id, segment in eligible_segments.items():
            status_row = connection.execute(
                "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if status_row != (segment["status"],):
                raise ValueError("Round 25 materialization database status differs")
            rows = connection.execute(
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
                ORDER BY event_start_ms, condition_id, observed_wall_ms
                """,
                [run_id],
            ).fetchall()
            for row in rows:
                source_count += 1
                condition = _joint_condition_from_snapshot(row, segment=segment)
                if condition is None:
                    purged_count += 1
                else:
                    conditions.append(condition)

    identities = [item.condition_id for item in conditions]
    if not conditions or len(set(identities)) != len(identities):
        raise ValueError("Round 25 joint receipt condition identity set differs")
    ordered = tuple(
        sorted(
            conditions,
            key=lambda item: (
                item.segment_index,
                item.event_start_ms,
                item.condition_id,
            ),
        )
    )
    role_counts = Counter(item.role for item in ordered)
    return ordered, {
        "admitted_condition_count": len(ordered),
        "calibration_condition_count": role_counts["calibration"],
        "purged_condition_count": purged_count,
        "selection_condition_count": role_counts["selection"],
        "source_snapshot_count": source_count,
        "train_condition_count": role_counts["train"],
    }


@dataclass(frozen=True, slots=True)
class Round25ConditionFeatureMaterialization:
    run_id: str
    segment_index: int
    source_snapshot_sha256: str
    source_snapshot_observed_wall_ms: int
    market_id: str
    condition_id: str
    slug: str
    event_start_ms: int
    event_end_ms: int
    up_token_id: str
    down_token_id: str
    resolution_source: str
    role: str
    source_record_count: int
    decision_count: int
    available_decision_count: int
    admitted: bool
    rejection_reasons: tuple[str, ...]
    selected_endpoint_decision_time_ms: tuple[int, ...]
    persisted_snapshots: tuple[Round25JointFeatureSnapshot, ...]
    persisted_snapshot_sha256: tuple[str, ...]
    unavailable_reason_counts: tuple[tuple[str, int], ...]
    materialization_sha256: str
    schema_version: str = (
        POLYMARKET_ROUND25_CONDITION_MATERIALIZATION_SCHEMA_VERSION
    )
    contract_sha256: str = (
        POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
    )
    target_accessed: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "available_decision_count": self.available_decision_count,
            "condition_id": self.condition_id,
            "contract_sha256": self.contract_sha256,
            "decision_count": self.decision_count,
            "down_token_id": self.down_token_id,
            "event_end_ms": self.event_end_ms,
            "event_start_ms": self.event_start_ms,
            "market_id": self.market_id,
            "persisted_snapshot_sha256": list(self.persisted_snapshot_sha256),
            "rejection_reasons": list(self.rejection_reasons),
            "resolution_source": self.resolution_source,
            "role": self.role,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "segment_index": self.segment_index,
            "selected_endpoint_decision_time_ms": list(
                self.selected_endpoint_decision_time_ms
            ),
            "source_record_count": self.source_record_count,
            "source_snapshot_observed_wall_ms": (
                self.source_snapshot_observed_wall_ms
            ),
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "slug": self.slug,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
            "unavailable_reason_counts": [
                {"count": count, "reason": reason}
                for reason, count in self.unavailable_reason_counts
            ],
            "up_token_id": self.up_token_id,
        }

    def validated(self) -> Round25ConditionFeatureMaterialization:
        snapshot_hashes = tuple(
            round25_joint_snapshot_sha256(snapshot)
            for snapshot in self.persisted_snapshots
        )
        snapshot_keys = tuple(
            (snapshot.condition_id, snapshot.decision_time_ms)
            for snapshot in self.persisted_snapshots
        )
        if (
            self.schema_version
            != POLYMARKET_ROUND25_CONDITION_MATERIALIZATION_SCHEMA_VERSION
            or self.contract_sha256
            != POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
            or _RUN_ID.fullmatch(self.run_id) is None
            or type(self.segment_index) is not int
            or self.segment_index < 0
            or _SHA256.fullmatch(self.source_snapshot_sha256) is None
            or type(self.source_snapshot_observed_wall_ms) is not int
            or self.source_snapshot_observed_wall_ms <= 0
            or _MARKET_ID.fullmatch(self.market_id) is None
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or _BTC_FIVE_MINUTE_SLUG.fullmatch(self.slug) is None
            or self.slug != f"btc-updown-5m-{self.event_start_ms // 1_000}"
            or self.event_end_ms != self.event_start_ms + 300_000
            or _TOKEN_ID.fullmatch(self.up_token_id) is None
            or _TOKEN_ID.fullmatch(self.down_token_id) is None
            or self.up_token_id == self.down_token_id
            or self.resolution_source != POLYMARKET_ROUND25_RESOLUTION_SOURCE
            or self.role not in {"train", "calibration", "selection"}
            or type(self.source_record_count) is not int
            or self.source_record_count < 0
            or self.decision_count != POLYMARKET_ROUND25_EXPECTED_DECISIONS
            or not 0 <= self.available_decision_count <= self.decision_count
            or type(self.admitted) is not bool
            or tuple(sorted(set(self.rejection_reasons)))
            != self.rejection_reasons
            or any(
                reason == "" or type(count) is not int or count <= 0
                for reason, count in self.unavailable_reason_counts
            )
            or tuple(sorted(dict(self.unavailable_reason_counts).items()))
            != self.unavailable_reason_counts
            or len(set(snapshot_keys)) != len(snapshot_keys)
            or snapshot_hashes != self.persisted_snapshot_sha256
            or any(
                snapshot.condition_id != self.condition_id
                or snapshot.event_start_ms != self.event_start_ms
                or not snapshot.available
                or snapshot.maximum_receipt_ms > snapshot.decision_time_ms
                or snapshot.trading_authority
                for snapshot in self.persisted_snapshots
            )
            or self.target_accessed
            or self.trading_authority
            or _SHA256.fullmatch(self.materialization_sha256) is None
            or self.materialization_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 condition feature materialization differs")
        if self.admitted:
            if (
                self.rejection_reasons
                or len(self.selected_endpoint_decision_time_ms) != 16
                or not self.persisted_snapshots
                or tuple(
                    row.decision_time_ms
                    for row in select_round25_condition_endpoints(
                        self.persisted_snapshots
                    )
                )
                != self.selected_endpoint_decision_time_ms
            ):
                raise ValueError("Round 25 admitted feature materialization differs")
        elif (
            not self.rejection_reasons
            or self.selected_endpoint_decision_time_ms
            or self.persisted_snapshots
            or self.persisted_snapshot_sha256
        ):
            raise ValueError("Round 25 rejected feature materialization differs")
        return self


def reject_round25_joint_condition(
    *,
    condition: Round25JointReceiptCondition,
    source_record_count: int,
    rejection_reasons: Sequence[str],
) -> Round25ConditionFeatureMaterialization:
    """Create a hash-bound target-free rejection without partial feature rows."""

    selected = condition.validated()
    reasons = tuple(sorted(set(rejection_reasons)))
    if (
        type(source_record_count) is not int
        or source_record_count < 0
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise ValueError("Round 25 joint rejection evidence differs")
    provisional = Round25ConditionFeatureMaterialization(
        run_id=selected.run_id,
        segment_index=selected.segment_index,
        source_snapshot_sha256=selected.snapshot_sha256,
        source_snapshot_observed_wall_ms=selected.snapshot_observed_wall_ms,
        market_id=selected.market_id,
        condition_id=selected.condition_id,
        slug=selected.slug,
        event_start_ms=selected.event_start_ms,
        event_end_ms=selected.event_end_ms,
        up_token_id=selected.up_token_id,
        down_token_id=selected.down_token_id,
        resolution_source=selected.resolution_source,
        role=selected.role,
        source_record_count=source_record_count,
        decision_count=POLYMARKET_ROUND25_EXPECTED_DECISIONS,
        available_decision_count=0,
        admitted=False,
        rejection_reasons=reasons,
        selected_endpoint_decision_time_ms=(),
        persisted_snapshots=(),
        persisted_snapshot_sha256=(),
        unavailable_reason_counts=(),
        materialization_sha256="0" * 64,
    )
    return replace(
        provisional,
        materialization_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(slots=True)
class _JointConditionAccumulator:
    condition: Round25JointReceiptCondition
    sources: list[CaptureFrameRecord | PolymarketUnionEvent] = field(
        default_factory=list
    )
    rejection_reasons: set[str] = field(default_factory=set)


class Round25JointMaterializationObserver:
    """Route one terminal receipt scan into bounded v2 condition materializations."""

    def __init__(
        self,
        conditions: Sequence[Round25JointReceiptCondition],
        *,
        sink: Callable[[Round25ConditionFeatureMaterialization], None],
    ) -> None:
        selected = tuple(item.validated() for item in conditions)
        if (
            not selected
            or len({item.condition_id for item in selected}) != len(selected)
            or not callable(sink)
        ):
            raise ValueError("Round 25 joint observer conditions differ")
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
        self.sink = sink
        self.condition_count = 0
        self.admitted_condition_count = 0
        self.persisted_snapshot_count = 0
        self.rejection_counts: Counter[str] = Counter()
        self.unavailable_reason_counts: Counter[str] = Counter()
        self._run_conditions: tuple[Round25JointReceiptCondition, ...] = ()
        self._active: dict[str, _JointConditionAccumulator] = {}
        self._finalized: set[str] = set()
        self._run_id = ""
        self._decoder: Round25SingleLaneClobDecoder | None = None
        self._last_monotonic_ns = 0
        self._last_wall_ms = 0
        self._clock_offset_ns: int | None = None
        self._last_connection: dict[str, str] = {}
        self._last_sequence: dict[tuple[str, str], int] = {}

    @staticmethod
    def _window(condition: Round25JointReceiptCondition) -> tuple[int, int]:
        return (
            condition.event_start_ms - POLYMARKET_ROUND25_GAP_LOOKBACK_MS,
            condition.event_end_ms + POLYMARKET_ROUND25_TERMINAL_GRACE_MS,
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
            raise RuntimeError("Round 25 joint observer run is already open")
        run_id = str(segment.get("run_id") or "")
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("Round 25 joint observer run identity differs")
        self._run_conditions = tuple(
            item for item in self.conditions if item.run_id == run_id
        )
        self._active = {
            item.condition_id: _JointConditionAccumulator(item)
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
        segment_start = int(segment["started_at_ms"])
        segment_end = int(segment["ended_at_ms"])
        for accumulator in self._active.values():
            start, end = self._window(accumulator.condition)
            if (
                start < segment_start
                or end > segment_end
                or not segment_start
                <= accumulator.condition.snapshot_observed_wall_ms
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
            or wall_ms
            > accumulator.condition.event_end_ms
            + POLYMARKET_ROUND25_TERMINAL_GRACE_MS
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
                result = reject_round25_joint_condition(
                    condition=accumulator.condition,
                    source_record_count=len(accumulator.sources),
                    rejection_reasons=tuple(accumulator.rejection_reasons),
                )
            else:
                result = materialize_round25_joint_condition(
                    condition=accumulator.condition,
                    sources=accumulator.sources,
                )
            self.sink(result)
            self.condition_count += 1
            self.admitted_condition_count += int(result.admitted)
            self.persisted_snapshot_count += len(result.persisted_snapshots)
            self.rejection_counts.update(result.rejection_reasons)
            self.unavailable_reason_counts.update(
                dict(result.unavailable_reason_counts)
            )
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
            raise RuntimeError("Round 25 joint observer run is unavailable")
        selected = message.validated()
        if (
            selected.received_monotonic_ns < self._last_monotonic_ns
            or selected.received_wall_ms < self._last_wall_ms
        ):
            raise ValueError("Round 25 terminal receipt clock regressed")
        offset_ns = (
            selected.received_wall_ms * 1_000_000
            - selected.received_monotonic_ns
        )
        if self._clock_offset_ns is None:
            self._clock_offset_ns = offset_ns
        elif (
            abs(offset_ns - self._clock_offset_ns)
            > POLYMARKET_ROUND25_MAXIMUM_CLOCK_OFFSET_DRIFT_NS
        ):
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
                    accumulator.sources.append(event)
        elif selected.stream == "polymarket_rtds":
            record = CaptureFrameRecord(**selected.__dict__)
            for accumulator in self._active.values():
                start, end = self._window(accumulator.condition)
                if (
                    not accumulator.rejection_reasons
                    and start <= selected.received_wall_ms <= end
                ):
                    accumulator.sources.append(record)
        else:
            raise ValueError("Round 25 joint materializer stream differs")
        self._finalize_ready(selected.received_wall_ms)

    def finish_run(self, segment: Mapping[str, object]) -> None:
        if not self._run_id or str(segment.get("run_id") or "") != self._run_id:
            raise RuntimeError("Round 25 joint observer run is unavailable")
        self._finalize_ready(2**63 - 1, force=True)
        expected = {item.condition_id for item in self._run_conditions}
        if self._active or self._finalized != expected:
            raise RuntimeError("Round 25 joint observer condition accounting differs")
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


def _rejection_reasons(
    available: Sequence[Round25JointFeatureSnapshot],
    *,
    event_start_ms: int,
) -> tuple[str, ...]:
    phase_counts = [0, 0, 0, 0]
    for row in available:
        offset = row.decision_time_ms - event_start_ms
        phase_counts[min(3, offset * 4 // 300_000)] += 1
    return tuple(
        f"phase_{phase}_available_rows_below_4"
        for phase, count in enumerate(phase_counts)
        if count < 4
    )


def materialize_round25_joint_condition(
    *,
    condition: Round25JointReceiptCondition,
    sources: Sequence[CaptureFrameRecord | PolymarketUnionEvent],
) -> Round25ConditionFeatureMaterialization:
    """Build only target-free endpoints and their exact causal sequence contexts."""

    selected = condition.validated()
    ordered = tuple(sources)
    if any(
        _source_monotonic_ns(current) < _source_monotonic_ns(previous)
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("Round 25 joint source chronology regressed")

    twap_observations: list[Round25TwapObservation] = []
    union_events: list[PolymarketUnionEvent] = []
    for source in ordered:
        if isinstance(source, CaptureFrameRecord):
            observation = decode_round25_twap_record(source)
            if (
                observation is not None
                and selected.event_start_ms
                <= observation.source_timestamp_ms
                <= selected.event_end_ms
            ):
                twap_observations.append(observation)
        elif isinstance(source, PolymarketUnionEvent):
            union_events.append(source)
        else:
            raise TypeError("Round 25 joint feature source type differs")

    books = tuple(
        snapshot
        for snapshot in build_polymarket_execution_books(
            condition_id=selected.condition_id,
            up_token_id=selected.up_token_id,
            down_token_id=selected.down_token_id,
            union_events=union_events,
            admitted_gap_free=True,
        )
        if selected.event_start_ms
        <= snapshot.source_time_ms
        <= selected.event_end_ms
    )
    timeline: list[
        tuple[int, int, int, Round25TwapObservation | PaperBookSnapshot]
    ] = []
    timeline.extend(
        (
            observation.received_wall_ms,
            observation.received_monotonic_ns,
            0,
            observation,
        )
        for observation in twap_observations
    )
    timeline.extend(
        (
            snapshot.received_wall_ms,
            snapshot.received_monotonic_ns,
            1,
            snapshot,
        )
        for snapshot in books
    )
    timeline.sort(key=lambda item: item[:3])

    twap_engine = Round25TwapFeatureEngine(
        condition_id=selected.condition_id,
        event_start_ms=selected.event_start_ms,
    )
    clob_engine = Round25ClobFeatureEngine(
        condition_id=selected.condition_id,
        up_token_id=selected.up_token_id,
        down_token_id=selected.down_token_id,
        event_start_ms=selected.event_start_ms,
    )
    joint_rows: list[Round25JointFeatureSnapshot] = []
    unavailable: Counter[str] = Counter()
    source_index = 0
    for decision_time_ms in range(
        selected.event_start_ms,
        selected.event_end_ms,
        POLYMARKET_ROUND25_DECISION_CADENCE_MS,
    ):
        while (
            source_index < len(timeline)
            and timeline[source_index][0] <= decision_time_ms
        ):
            source = timeline[source_index][3]
            if isinstance(source, Round25TwapObservation):
                twap_engine.ingest(source)
            else:
                clob_engine.ingest(source)
            source_index += 1
        joint = combine_round25_features(
            twap_engine.build(decision_time_ms),
            clob_engine.build(decision_time_ms),
        )
        if joint.available:
            joint_rows.append(joint)
        else:
            unavailable.update(joint.reasons)

    if len(joint_rows) > POLYMARKET_ROUND25_EXPECTED_DECISIONS:
        raise RuntimeError("Round 25 available decision count differs")
    rejection_reasons = _rejection_reasons(
        joint_rows,
        event_start_ms=selected.event_start_ms,
    )
    endpoints: tuple[Round25JointFeatureSnapshot, ...] = ()
    persisted: tuple[Round25JointFeatureSnapshot, ...] = ()
    if not rejection_reasons:
        endpoints = select_round25_condition_endpoints(joint_rows)
        retained: dict[int, Round25JointFeatureSnapshot] = {}
        history_ms = (
            POLYMARKET_ROUND25_SEQUENCE_CONTEXT_STEPS - 1
        ) * POLYMARKET_ROUND25_DECISION_CADENCE_MS
        for endpoint in endpoints:
            lower = endpoint.decision_time_ms - history_ms
            retained.update(
                {
                    row.decision_time_ms: row
                    for row in joint_rows
                    if lower <= row.decision_time_ms <= endpoint.decision_time_ms
                }
            )
        persisted = tuple(retained[key] for key in sorted(retained))

    provisional = Round25ConditionFeatureMaterialization(
        run_id=selected.run_id,
        segment_index=selected.segment_index,
        source_snapshot_sha256=selected.snapshot_sha256,
        source_snapshot_observed_wall_ms=selected.snapshot_observed_wall_ms,
        market_id=selected.market_id,
        condition_id=selected.condition_id,
        slug=selected.slug,
        event_start_ms=selected.event_start_ms,
        event_end_ms=selected.event_end_ms,
        up_token_id=selected.up_token_id,
        down_token_id=selected.down_token_id,
        resolution_source=selected.resolution_source,
        role=selected.role,
        source_record_count=len(ordered),
        decision_count=POLYMARKET_ROUND25_EXPECTED_DECISIONS,
        available_decision_count=len(joint_rows),
        admitted=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        selected_endpoint_decision_time_ms=tuple(
            endpoint.decision_time_ms for endpoint in endpoints
        ),
        persisted_snapshots=persisted,
        persisted_snapshot_sha256=tuple(
            round25_joint_snapshot_sha256(snapshot) for snapshot in persisted
        ),
        unavailable_reason_counts=tuple(sorted(unavailable.items())),
        materialization_sha256="0" * 64,
    )
    return replace(
        provisional,
        materialization_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


__all__ = [
    "POLYMARKET_ROUND25_CONDITION_MATERIALIZATION_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_EXPECTED_DECISIONS",
    "POLYMARKET_ROUND25_GAP_LOOKBACK_MS",
    "POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_MAXIMUM_CLOCK_OFFSET_DRIFT_NS",
    "POLYMARKET_ROUND25_SEQUENCE_CONTEXT_STEPS",
    "POLYMARKET_ROUND25_TERMINAL_GRACE_MS",
    "Round25ConditionFeatureMaterialization",
    "Round25JointMaterializationObserver",
    "Round25JointReceiptCondition",
    "Round25SingleLaneClobDecoder",
    "decode_round25_twap_record",
    "load_round25_joint_receipt_conditions",
    "materialize_round25_joint_condition",
    "reject_round25_joint_condition",
    "round25_joint_snapshot_sha256",
]

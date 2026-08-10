"""Terminal transport and exact-receipt authority for Round 25 TWAP capture."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol
import hashlib
import json
import math
import re
import time

from .polymarket_recorder import (
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from .polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
    POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256,
    POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256,
    POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC,
    POLYMARKET_ROUND25_END_MS,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    POLYMARKET_ROUND25_START_MS,
    PolymarketRound25ActiveCampaignPlan,
    load_round25_active_campaign_plan,
    validate_round25_active_segment_manifest,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND25_TERMINAL_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-025-terminal-receipt-materialization-design-v2.json"
)
POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256 = (
    "b7f693680de7eace5408489a75b65de2ba7058c7f01c0d8e0038430419eb7786"
)
POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION = (
    "polymarket-round25-twap-terminal-transport-v2"
)
POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION = (
    "polymarket-round25-twap-terminal-receipt-audit-v2"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_RESULT_NAME = re.compile(r"^segment-([0-9]{4})-result\.json$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_ELIGIBLE_STATUSES = frozenset(("complete", "degraded"))
_ALL_STATUSES = frozenset((*_ELIGIBLE_STATUSES, "failed", "interrupted"))
_AUTHORITY_FIELDS = (
    "model_data_eligible",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
)
_DESIGN_AUTHORITY_FIELDS = frozenset(
    (
        "credentials_used",
        "execution_connected",
        "live_trading_authority",
        "model_data_eligible_before_terminal_audit",
        "outcomes_consulted",
        "paper_trading_authority",
        "profitability_claim",
    )
)
_REQUIRED_STREAMS = ("clob_market", "polymarket_rtds")


class Round25TerminalReceiptObserver(Protocol):
    def start_run(
        self,
        segment: Mapping[str, object],
        gaps: tuple[StreamGap, ...],
    ) -> None: ...

    def observe_message(
        self,
        segment: Mapping[str, object],
        message: RawStreamMessage,
    ) -> None: ...

    def finish_run(self, segment: Mapping[str, object]) -> None: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 terminal JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 terminal JSON contains {value}")


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


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= 2_000_000
    ):
        raise ValueError(f"Round 25 terminal {label} is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 terminal {label} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 25 terminal {label} is invalid")
    return value


def _digest(value: object, *, label: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 25 terminal {label} differs")
    return selected


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(f"Round 25 terminal {label} differs")
    return int(value)


def _texts(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Round 25 terminal {label} differs")
    return tuple(value)


def load_round25_terminal_design(repository: str | Path) -> dict[str, object]:
    payload = _read_object(
        Path(repository) / POLYMARKET_ROUND25_TERMINAL_DESIGN_RELATIVE,
        label="design",
    )
    claimed = _digest(payload.pop("design_sha256", ""), label="design hash")
    campaign = payload.get("campaign")
    condition_admission = payload.get("condition_admission")
    materialization = payload.get("materialization")
    authority = payload.get("authority")
    if (
        claimed != _canonical_sha256(payload)
        or claimed != POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256
        or payload.get("schema_version")
        != "polymarket-round25-terminal-receipt-materialization-design-v2"
        or payload.get("status")
        != "frozen_before_round25_v2_first_eligible_receipt"
        or payload.get("created_at_utc") != "2026-08-10T21:34:23Z"
        or not isinstance(campaign, Mapping)
        or campaign.get("capture_design_sha256")
        != POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256
        or campaign.get("capture_plan_sha256")
        != POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256
        or campaign.get("source_qualification_sha256")
        != POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256
        or campaign.get("campaign_start_ms") != POLYMARKET_ROUND25_START_MS
        or campaign.get("campaign_end_ms") != POLYMARKET_ROUND25_END_MS
        or campaign.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or campaign.get("required_clob_lanes") != ["clob"]
        or campaign.get("required_streams") != list(_REQUIRED_STREAMS)
        or campaign.get("required_rtds_topics")
        != [POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC]
        or campaign.get("legacy_point_campaign_data_permitted") is not False
        or not isinstance(condition_admission, Mapping)
        or condition_admission.get("one_authoritative_clob_lane") is not True
        or condition_admission.get("redundant_lane_coverage_claim") is not False
        or condition_admission.get("twap_outcome_reconstructed") is not False
        or not isinstance(materialization, Mapping)
        or materialization.get("official_resolution_accessed") is not False
        or materialization.get("future_receipts_permitted") is not False
        or not isinstance(authority, Mapping)
        or set(authority) != _DESIGN_AUTHORITY_FIELDS
        or any(authority.get(field) is not False for field in authority)
    ):
        raise ValueError("Round 25 terminal design differs")
    return {**payload, "design_sha256": claimed}


def _result_paths(state_root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(state_root.glob("segment-*-result.json")))
    if not paths:
        raise ValueError("Round 25 terminal segment result set is empty")
    for index, path in enumerate(paths):
        match = _RESULT_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) != index:
            raise ValueError("Round 25 terminal segment result set is not contiguous")
    return paths


def _manifest_path(state_root: Path, index: int) -> Path:
    return state_root / f"segment-{index:04d}-manifest.json"


def _validated_manifest(
    state_root: Path,
    *,
    plan: PolymarketRound25ActiveCampaignPlan,
    index: int,
) -> dict[str, object] | None:
    path = _manifest_path(state_root, index)
    if not path.exists():
        return None
    value = validate_round25_active_segment_manifest(
        _read_object(path, label=f"segment {index} manifest"),
        plan,
    )
    if value["segment_index"] != index:
        raise ValueError("Round 25 terminal segment manifest index differs")
    return value


def _terminal_details(
    details: Mapping[str, object],
    *,
    status: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "run_id",
        "manifest_sha256",
        "report_sha256",
        "started_at_ms",
        "ended_at_ms",
        "duration_seconds",
        "raw_message_count",
        "stream_gap_count",
        "stream_counts",
        "condition_count",
        "integrity_errors",
        "errors",
        "resolution_source",
    }
    counts = details.get("stream_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("Round 25 terminal stream counts differ")
    normalized_counts = {
        str(key): _integer(value, label="stream count") for key, value in counts.items()
    }
    started = _integer(details.get("started_at_ms"), label="segment start", minimum=1)
    ended = _integer(details.get("ended_at_ms"), label="segment end", minimum=1)
    duration = details.get("duration_seconds")
    raw_count = _integer(details.get("raw_message_count"), label="raw count")
    gap_count = _integer(details.get("stream_gap_count"), label="gap count")
    condition_count = _integer(details.get("condition_count"), label="condition count")
    errors = _texts(details.get("errors"), label="recorder errors")
    integrity = _texts(details.get("integrity_errors"), label="integrity errors")
    if (
        set(details) != expected
        or _RUN_ID.fullmatch(str(details.get("run_id") or "")) is None
        or details.get("run_id") != manifest["run_id"]
        or details.get("manifest_sha256") != manifest["manifest_sha256"]
        or _SHA256.fullmatch(str(details.get("report_sha256") or "")) is None
        or started < POLYMARKET_ROUND25_START_MS
        or ended <= started
        or ended > POLYMARKET_ROUND25_END_MS + 1_000
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or abs(float(duration) - ((ended - started) / 1_000.0)) > 0.001
        or set(normalized_counts) != set(_REQUIRED_STREAMS)
        or any(normalized_counts[name] <= 0 for name in _REQUIRED_STREAMS)
        or sum(normalized_counts.values()) != raw_count
        or condition_count <= 0
        or details.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or errors
        or integrity
        or (status == "complete" and gap_count != 0)
        or (status == "degraded" and gap_count <= 0)
    ):
        raise ValueError("Round 25 terminal segment report differs")
    return {
        "run_id": str(details["run_id"]),
        "report_sha256": str(details["report_sha256"]),
        "started_at_ms": started,
        "ended_at_ms": ended,
        "duration_seconds": float(duration),
        "raw_message_count": raw_count,
        "stream_gap_count": gap_count,
        "stream_counts": dict(sorted(normalized_counts.items())),
        "condition_count": condition_count,
        "integrity_errors": list(integrity),
        "errors": list(errors),
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    }


def _source_segment(
    path: Path,
    *,
    state_root: Path,
    plan: PolymarketRound25ActiveCampaignPlan,
    expected_index: int,
) -> dict[str, object]:
    source = _read_object(path, label=f"segment {expected_index} result")
    claimed = _digest(source.pop("artifact_sha256", ""), label="result hash")
    expected = {
        "schema_version",
        "plan_sha256",
        "segment_index",
        "status",
        "observed_at_ms",
        "details",
        "condition_admission_pending",
        *_AUTHORITY_FIELDS,
    }
    status = str(source.get("status") or "")
    details = source.get("details")
    manifest = _validated_manifest(
        state_root,
        plan=plan,
        index=expected_index,
    )
    if (
        set(source) != expected
        or claimed != _canonical_sha256(source)
        or source.get("schema_version")
        != POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION
        or source.get("plan_sha256") != plan.plan_sha256
        or source.get("segment_index") != expected_index
        or status not in _ALL_STATUSES
        or type(source.get("observed_at_ms")) is not int
        or source["observed_at_ms"] <= 0
        or source.get("condition_admission_pending") is not True
        or not isinstance(details, Mapping)
        or any(source.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 terminal source segment differs")
    reasons: list[str] = []
    normalized: dict[str, object]
    if status in _ELIGIBLE_STATUSES:
        if manifest is None:
            raise ValueError("Round 25 terminal eligible segment manifest is absent")
        normalized = _terminal_details(details, status=status, manifest=manifest)
    else:
        reasons.append(f"segment_status_{status}")
        if manifest is not None:
            run_id: str | None = str(manifest["run_id"])
            manifest_sha256: str | None = str(manifest["manifest_sha256"])
        else:
            run_id = None
            manifest_sha256 = None
        if set(details) == {"run_id", "manifest_sha256", "reason"}:
            if (
                status != "interrupted"
                or manifest is None
                or details.get("run_id") != run_id
                or details.get("manifest_sha256") != manifest_sha256
                or details.get("reason")
                != "campaign_process_interrupted_before_terminal_report"
            ):
                raise ValueError("Round 25 terminal interruption differs")
        elif set(details) != {"failure_type", "failure"} or status != "failed":
            raise ValueError("Round 25 terminal failure differs")
        normalized = {
            "run_id": run_id,
            "report_sha256": None,
            "started_at_ms": None,
            "ended_at_ms": None,
            "duration_seconds": None,
            "raw_message_count": None,
            "stream_gap_count": None,
            "stream_counts": {},
            "condition_count": None,
            "integrity_errors": [],
            "errors": [],
            "resolution_source": (
                POLYMARKET_ROUND25_RESOLUTION_SOURCE if manifest is not None else None
            ),
        }
    eligible = status in _ELIGIBLE_STATUSES
    return {
        "segment_index": expected_index,
        "source_result_sha256": claimed,
        "source_manifest_sha256": (
            None if manifest is None else manifest["manifest_sha256"]
        ),
        "status": status,
        "observed_at_ms": int(source["observed_at_ms"]),
        **normalized,
        "eligible_for_condition_rebuild": eligible,
        "exclusion_reasons": reasons,
    }


def _coverage(
    segments: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    intervals: list[dict[str, int]] = []
    for segment in segments:
        if not segment["eligible_for_condition_rebuild"]:
            continue
        start = max(POLYMARKET_ROUND25_START_MS, int(segment["started_at_ms"]))
        end = min(POLYMARKET_ROUND25_END_MS, int(segment["ended_at_ms"]))
        if end <= start or (intervals and start < intervals[-1]["end_ms"]):
            raise ValueError("Round 25 terminal eligible intervals differ")
        intervals.append(
            {
                "segment_index": int(segment["segment_index"]),
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
            }
        )
    gaps: list[dict[str, int]] = []
    cursor = POLYMARKET_ROUND25_START_MS
    for interval in intervals:
        if interval["start_ms"] > cursor:
            gaps.append(
                {
                    "start_ms": cursor,
                    "end_ms": interval["start_ms"],
                    "duration_ms": interval["start_ms"] - cursor,
                }
            )
        cursor = max(cursor, interval["end_ms"])
    if cursor < POLYMARKET_ROUND25_END_MS:
        gaps.append(
            {
                "start_ms": cursor,
                "end_ms": POLYMARKET_ROUND25_END_MS,
                "duration_ms": POLYMARKET_ROUND25_END_MS - cursor,
            }
        )
    return intervals, gaps


def _validate_segment_summary(
    value: object, *, expected_index: int
) -> dict[str, object]:
    expected = {
        "segment_index",
        "source_result_sha256",
        "source_manifest_sha256",
        "status",
        "observed_at_ms",
        "run_id",
        "report_sha256",
        "started_at_ms",
        "ended_at_ms",
        "duration_seconds",
        "raw_message_count",
        "stream_gap_count",
        "stream_counts",
        "condition_count",
        "integrity_errors",
        "errors",
        "resolution_source",
        "eligible_for_condition_rebuild",
        "exclusion_reasons",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Round 25 terminal segment summary differs")
    selected = dict(value)
    status = str(selected.get("status") or "")
    eligible = status in _ELIGIBLE_STATUSES
    if (
        selected.get("segment_index") != expected_index
        or status not in _ALL_STATUSES
        or selected.get("eligible_for_condition_rebuild") is not eligible
        or _SHA256.fullmatch(str(selected.get("source_result_sha256") or "")) is None
        or type(selected.get("observed_at_ms")) is not int
        or selected["observed_at_ms"] <= 0
        or selected.get("exclusion_reasons")
        != ([] if eligible else [f"segment_status_{status}"])
    ):
        raise ValueError("Round 25 terminal segment summary differs")
    if eligible:
        raw_counts = selected.get("stream_counts")
        if (
            _SHA256.fullmatch(str(selected.get("source_manifest_sha256") or "")) is None
            or _RUN_ID.fullmatch(str(selected.get("run_id") or "")) is None
            or _SHA256.fullmatch(str(selected.get("report_sha256") or "")) is None
            or selected.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
            or not isinstance(raw_counts, Mapping)
            or set(raw_counts) != set(_REQUIRED_STREAMS)
            or selected.get("integrity_errors") != []
            or selected.get("errors") != []
        ):
            raise ValueError("Round 25 terminal eligible segment summary differs")
        started = _integer(selected["started_at_ms"], label="segment start", minimum=1)
        ended = _integer(selected["ended_at_ms"], label="segment end", minimum=1)
        duration = selected["duration_seconds"]
        raw = _integer(selected["raw_message_count"], label="raw count", minimum=1)
        gaps = _integer(selected["stream_gap_count"], label="gap count")
        conditions = _integer(
            selected["condition_count"], label="condition count", minimum=1
        )
        counts = {
            str(key): _integer(value, label="stream count", minimum=1)
            for key, value in raw_counts.items()
        }
        if (
            ended <= started
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or abs(float(duration) - ((ended - started) / 1_000.0)) > 0.001
            or sum(counts.values()) != raw
            or conditions <= 0
            or (status == "complete" and gaps != 0)
            or (status == "degraded" and gaps <= 0)
        ):
            raise ValueError("Round 25 terminal eligible segment accounting differs")
    else:
        nullable = (
            "report_sha256",
            "started_at_ms",
            "ended_at_ms",
            "duration_seconds",
            "raw_message_count",
            "stream_gap_count",
            "condition_count",
        )
        if (
            any(selected[field] is not None for field in nullable)
            or selected["stream_counts"] != {}
            or selected["integrity_errors"] != []
            or selected["errors"] != []
            or (selected["run_id"] is None)
            != (selected["source_manifest_sha256"] is None)
        ):
            raise ValueError("Round 25 terminal ineligible segment summary differs")
        if selected["run_id"] is not None:
            if (
                _RUN_ID.fullmatch(str(selected["run_id"])) is None
                or _SHA256.fullmatch(str(selected["source_manifest_sha256"])) is None
                or selected["resolution_source"] != POLYMARKET_ROUND25_RESOLUTION_SOURCE
            ):
                raise ValueError("Round 25 terminal ineligible identity differs")
        elif selected["resolution_source"] is not None:
            raise ValueError("Round 25 terminal absent manifest source differs")
    return selected


def validate_round25_terminal_transport_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = _digest(payload.pop("manifest_sha256", ""), label="manifest hash")
    expected = {
        "schema_version",
        "terminal_design_sha256",
        "created_at_ms",
        "source_plan_sha256",
        "source_capture_design_sha256",
        "source_qualification_sha256",
        "resolution_source",
        "campaign_start_ms",
        "campaign_end_ms",
        "campaign_state_artifact_sha256",
        "campaign_status",
        "segments",
        "eligible_run_ids",
        "provisional_eligible_transport_intervals",
        "known_ineligible_or_unobserved_intervals",
        "all_scheduled_transport_interval_covered",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    }
    raw_segments = payload.get("segments")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION
        or payload.get("terminal_design_sha256")
        != POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256
        or payload.get("source_plan_sha256")
        != POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256
        or payload.get("source_capture_design_sha256")
        != POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256
        or payload.get("source_qualification_sha256")
        != POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256
        or payload.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or payload.get("campaign_start_ms") != POLYMARKET_ROUND25_START_MS
        or payload.get("campaign_end_ms") != POLYMARKET_ROUND25_END_MS
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] < POLYMARKET_ROUND25_END_MS
        or _SHA256.fullmatch(str(payload.get("campaign_state_artifact_sha256") or ""))
        is None
        or payload.get("campaign_status")
        not in {"campaign_window_ended", "source_regime_changed", "campaign_failed"}
        or not isinstance(raw_segments, list)
        or not raw_segments
    ):
        raise ValueError("Round 25 terminal transport manifest differs")
    segments = [
        _validate_segment_summary(item, expected_index=index)
        for index, item in enumerate(raw_segments)
    ]
    intervals, gaps = _coverage(segments)
    eligible_ids = [
        segment["run_id"]
        for segment in segments
        if segment["eligible_for_condition_rebuild"]
    ]
    bool_fields = (
        "all_scheduled_transport_interval_covered",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    )
    if (
        len(set(eligible_ids)) != len(eligible_ids)
        or payload.get("eligible_run_ids") != eligible_ids
        or payload.get("provisional_eligible_transport_intervals") != intervals
        or payload.get("known_ineligible_or_unobserved_intervals") != gaps
        or payload.get("all_scheduled_transport_interval_covered") is not (not gaps)
        or payload.get("condition_admission_pending") is not bool(eligible_ids)
        or any(type(payload.get(field)) is not bool for field in bool_fields)
        or payload.get("outcomes_consulted") is not False
        or payload.get("model_scores_consulted") is not False
        or any(payload.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 terminal transport derivation differs")
    return {**payload, "segments": segments, "manifest_sha256": claimed}


def build_round25_terminal_transport_manifest(
    repository: str | Path,
    *,
    plan_path: str | Path,
    state_root: str | Path,
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    load_round25_terminal_design(repository)
    plan = load_round25_active_campaign_plan(plan_path)
    observed = (
        time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    )
    if observed < plan.scheduled_end_ms:
        raise RuntimeError(
            "Round 25 terminal transport cannot open before campaign end"
        )
    root = Path(state_root).resolve()
    state = _read_object(root / "campaign-state.json", label="campaign state")
    state_sha256 = _digest(
        state.pop("artifact_sha256", ""), label="campaign state hash"
    )
    if (
        state_sha256 != _canonical_sha256(state)
        or state.get("schema_version")
        != POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION
        or state.get("plan_sha256") != plan.plan_sha256
        or state.get("status")
        not in {"campaign_window_ended", "source_regime_changed", "campaign_failed"}
        or state.get("condition_admission_pending") is not True
        or any(state.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 terminal campaign state differs")
    segments = [
        _source_segment(
            path,
            state_root=root,
            plan=plan,
            expected_index=index,
        )
        for index, path in enumerate(_result_paths(root))
    ]
    status_counts = dict(
        sorted(Counter(str(item["status"]) for item in segments).items())
    )
    if (
        state.get("terminal_segment_count") != len(segments)
        or state.get("status_counts") != status_counts
    ):
        raise ValueError("Round 25 terminal campaign accounting differs")
    intervals, gaps = _coverage(segments)
    eligible_ids = [
        segment["run_id"]
        for segment in segments
        if segment["eligible_for_condition_rebuild"]
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
        "created_at_ms": observed,
        "source_plan_sha256": plan.plan_sha256,
        "source_capture_design_sha256": POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
        "source_qualification_sha256": plan.source_qualification_sha256,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "campaign_start_ms": plan.scheduled_start_ms,
        "campaign_end_ms": plan.scheduled_end_ms,
        "campaign_state_artifact_sha256": state_sha256,
        "campaign_status": state["status"],
        "segments": segments,
        "eligible_run_ids": eligible_ids,
        "provisional_eligible_transport_intervals": intervals,
        "known_ineligible_or_unobserved_intervals": gaps,
        "all_scheduled_transport_interval_covered": not gaps,
        "condition_admission_pending": bool(eligible_ids),
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return validate_round25_terminal_transport_manifest(body)


def _strict_json_text(value: object, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            str(value),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 terminal {label} is invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Round 25 terminal {label} is invalid")
    return parsed


def _chain(previous: str, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(payload))
    ).hexdigest()


def _validate_database_identity(
    *,
    run_row: Sequence[object],
    manifest_row: Sequence[object],
    segment: Mapping[str, object],
) -> tuple[str, str]:
    run_id, status, started_at_ms, ended_at_ms, report_json, report_sha256 = run_row
    report = _strict_json_text(report_json, label="database run report")
    embedded_report_sha = _digest(
        report.pop("report_sha256", ""),
        label="database report hash",
    )
    manifest_run_id, manifest_json, manifest_sha256 = manifest_row
    manifest = _strict_json_text(manifest_json, label="database manifest")
    embedded_manifest_sha = _digest(
        manifest.pop("manifest_sha256", ""),
        label="database manifest hash",
    )
    if (
        run_id != segment["run_id"]
        or status not in _ALL_STATUSES
        or embedded_report_sha != report_sha256
        or _canonical_sha256(report) != report_sha256
        or report.get("run_id") != run_id
        or report.get("status") != status
        or report.get("started_at_ms") != started_at_ms
        or report.get("ended_at_ms") != ended_at_ms
        or manifest_run_id != run_id
        or manifest.get("run_id") != run_id
        or embedded_manifest_sha != manifest_sha256
        or _canonical_sha256(manifest) != manifest_sha256
        or manifest_sha256 != segment["source_manifest_sha256"]
    ):
        raise ValueError("Round 25 terminal database identity differs")
    if segment["eligible_for_condition_rebuild"] and (
        status != segment["status"]
        or report_sha256 != segment["report_sha256"]
        or started_at_ms != segment["started_at_ms"]
        or ended_at_ms != segment["ended_at_ms"]
        or report.get("raw_message_count") != segment["raw_message_count"]
        or report.get("stream_gap_count") != segment["stream_gap_count"]
        or report.get("stream_counts") != segment["stream_counts"]
        or report.get("integrity_errors") != []
        or report.get("errors") != []
    ):
        raise ValueError("Round 25 terminal eligible database report differs")
    return str(report_sha256), str(manifest_sha256)


def _audit_eligible_run(
    store: PolymarketEvidenceStore,
    *,
    segment: Mapping[str, object],
    observer: Round25TerminalReceiptObserver | None,
) -> dict[str, object]:
    run_id = str(segment["run_id"])
    gaps = tuple(store.iter_terminal_stream_gaps(run_id))
    if len(gaps) != segment["stream_gap_count"]:
        raise ValueError("Round 25 terminal gap accounting differs")
    gap_chain = _EMPTY_SHA256
    first_gap_ms: int | None = None
    last_gap_ms: int | None = None
    for gap in gaps:
        gap_chain = _chain(
            gap_chain,
            {
                "stream": gap.stream,
                "connection_id": gap.connection_id,
                "opened_at_ms": gap.opened_at_ms,
                "reason": gap.reason,
                "last_sequence_number": gap.last_sequence_number,
            },
        )
        first_gap_ms = (
            gap.opened_at_ms
            if first_gap_ms is None
            else min(first_gap_ms, gap.opened_at_ms)
        )
        last_gap_ms = (
            gap.opened_at_ms
            if last_gap_ms is None
            else max(last_gap_ms, gap.opened_at_ms)
        )
    if observer is not None:
        observer.start_run(segment, gaps)
    counts: Counter[str] = Counter()
    receipt_count = 0
    first_wall_ms: int | None = None
    last_wall_ms: int | None = None
    receipt_chain = _EMPTY_SHA256
    for message in store.iter_terminal_capture_messages(run_id):
        if observer is not None:
            observer.observe_message(segment, message)
        receipt_chain = _chain(
            receipt_chain,
            {
                "stream": message.stream,
                "connection_id": message.connection_id,
                "sequence_number": message.sequence_number,
                "received_wall_ms": message.received_wall_ms,
                "received_monotonic_ns": message.received_monotonic_ns,
                "raw_sha256": hashlib.sha256(
                    message.raw_text.encode("utf-8")
                ).hexdigest(),
            },
        )
        counts[message.stream] += 1
        receipt_count += 1
        first_wall_ms = (
            message.received_wall_ms
            if first_wall_ms is None
            else min(first_wall_ms, message.received_wall_ms)
        )
        last_wall_ms = (
            message.received_wall_ms
            if last_wall_ms is None
            else max(last_wall_ms, message.received_wall_ms)
        )
    if (
        receipt_count != segment["raw_message_count"]
        or dict(sorted(counts.items())) != segment["stream_counts"]
        or first_wall_ms is None
        or last_wall_ms is None
    ):
        raise ValueError("Round 25 terminal receipt accounting differs")
    if observer is not None:
        observer.finish_run(segment)
    return {
        "segment_index": segment["segment_index"],
        "run_id": run_id,
        "status": segment["status"],
        "report_sha256": segment["report_sha256"],
        "preregistration_manifest_sha256": segment["source_manifest_sha256"],
        "receipt_count": receipt_count,
        "stream_counts": dict(sorted(counts.items())),
        "first_receipt_wall_ms": first_wall_ms,
        "last_receipt_wall_ms": last_wall_ms,
        "receipt_chain_sha256": receipt_chain,
        "gap_count": len(gaps),
        "first_gap_opened_at_ms": first_gap_ms,
        "last_gap_opened_at_ms": last_gap_ms,
        "gap_chain_sha256": gap_chain,
    }


def validate_round25_terminal_receipt_audit(
    value: Mapping[str, object],
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> dict[str, object]:
    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    payload = dict(value)
    claimed = _digest(payload.pop("audit_sha256", ""), label="audit hash")
    expected = {
        "schema_version",
        "terminal_design_sha256",
        "created_at_ms",
        "terminal_transport_manifest_sha256",
        "database_run_count",
        "eligible_runs",
        "ineligible_runs",
        "receipt_replay_complete",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    }
    eligible_segments = [
        item for item in transport["segments"] if item["eligible_for_condition_rebuild"]
    ]
    ineligible_segments = [
        item
        for item in transport["segments"]
        if item["run_id"] is not None and not item["eligible_for_condition_rebuild"]
    ]
    eligible_runs = payload.get("eligible_runs")
    ineligible_runs = payload.get("ineligible_runs")
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION
        or payload.get("terminal_design_sha256")
        != POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256
        or payload.get("terminal_transport_manifest_sha256")
        != transport["manifest_sha256"]
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] < transport["created_at_ms"]
        or not isinstance(eligible_runs, list)
        or not isinstance(ineligible_runs, list)
        or len(eligible_runs) != len(eligible_segments)
        or len(ineligible_runs) != len(ineligible_segments)
        or payload.get("database_run_count")
        != len(eligible_segments) + len(ineligible_segments)
    ):
        raise ValueError("Round 25 terminal receipt audit differs")
    eligible_run_fields = {
        "segment_index",
        "run_id",
        "status",
        "report_sha256",
        "preregistration_manifest_sha256",
        "receipt_count",
        "stream_counts",
        "first_receipt_wall_ms",
        "last_receipt_wall_ms",
        "receipt_chain_sha256",
        "gap_count",
        "first_gap_opened_at_ms",
        "last_gap_opened_at_ms",
        "gap_chain_sha256",
    }
    for value_run, segment in zip(eligible_runs, eligible_segments, strict=True):
        if not isinstance(value_run, Mapping):
            raise ValueError("Round 25 terminal eligible receipt run differs")
        first_receipt = value_run.get("first_receipt_wall_ms")
        last_receipt = value_run.get("last_receipt_wall_ms")
        first_gap = value_run.get("first_gap_opened_at_ms")
        last_gap = value_run.get("last_gap_opened_at_ms")
        gap_count = value_run.get("gap_count")
        if (
            set(value_run) != eligible_run_fields
            or value_run.get("segment_index") != segment["segment_index"]
            or value_run.get("run_id") != segment["run_id"]
            or value_run.get("status") != segment["status"]
            or value_run.get("report_sha256") != segment["report_sha256"]
            or value_run.get("preregistration_manifest_sha256")
            != segment["source_manifest_sha256"]
            or type(value_run.get("receipt_count")) is not int
            or value_run.get("receipt_count") != segment["raw_message_count"]
            or value_run.get("stream_counts") != segment["stream_counts"]
            or type(first_receipt) is not int
            or type(last_receipt) is not int
            or first_receipt <= 0
            or last_receipt < first_receipt
            or _SHA256.fullmatch(str(value_run.get("receipt_chain_sha256") or ""))
            is None
            or value_run.get("receipt_chain_sha256") == _EMPTY_SHA256
            or type(gap_count) is not int
            or gap_count != segment["stream_gap_count"]
        ):
            raise ValueError("Round 25 terminal eligible receipt run differs")
        if gap_count == 0:
            if (
                first_gap is not None
                or last_gap is not None
                or value_run.get("gap_chain_sha256") != _EMPTY_SHA256
            ):
                raise ValueError("Round 25 terminal eligible receipt run differs")
        elif (
            type(first_gap) is not int
            or type(last_gap) is not int
            or first_gap <= 0
            or last_gap < first_gap
            or _SHA256.fullmatch(str(value_run.get("gap_chain_sha256") or "")) is None
            or value_run.get("gap_chain_sha256") == _EMPTY_SHA256
        ):
            raise ValueError("Round 25 terminal eligible receipt run differs")
    for value_run, segment in zip(ineligible_runs, ineligible_segments, strict=True):
        if (
            not isinstance(value_run, Mapping)
            or value_run
            != {
                "segment_index": segment["segment_index"],
                "run_id": segment["run_id"],
                "segment_status": segment["status"],
                "database_status": value_run.get("database_status"),
                "report_sha256": value_run.get("report_sha256"),
                "preregistration_manifest_sha256": segment["source_manifest_sha256"],
                "receipts_replayed": False,
            }
            or value_run.get("database_status") not in _ALL_STATUSES
            or _SHA256.fullmatch(str(value_run.get("report_sha256") or "")) is None
        ):
            raise ValueError("Round 25 terminal ineligible receipt run differs")
    bool_fields = (
        "receipt_replay_complete",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    )
    if (
        not eligible_runs
        or payload.get("receipt_replay_complete") is not True
        or payload.get("condition_admission_pending") is not True
        or any(type(payload.get(field)) is not bool for field in bool_fields)
        or payload.get("outcomes_consulted") is not False
        or payload.get("model_scores_consulted") is not False
        or any(payload.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 terminal receipt authority differs")
    return {**payload, "audit_sha256": claimed}


def audit_round25_terminal_receipts(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    observed_at_ms: int | None = None,
    observer: Round25TerminalReceiptObserver | None = None,
) -> dict[str, object]:
    """Reconcile every exact core receipt once after terminal transport opens."""

    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    eligible_segments = [
        item for item in transport["segments"] if item["eligible_for_condition_rebuild"]
    ]
    if not eligible_segments:
        raise RuntimeError("Round 25 terminal receipt audit has no eligible run")
    expected = {
        str(item["run_id"]): item
        for item in transport["segments"]
        if item["run_id"] is not None
    }
    eligible_runs: list[dict[str, object]] = []
    ineligible_runs: list[dict[str, object]] = []
    with PolymarketEvidenceStore(
        Path(database),
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        connection = store.connect()
        run_rows = connection.execute(
            """
            SELECT run_id, status, started_at_ms, ended_at_ms,
                   report_json, report_sha256
            FROM polymarket_recorder_run
            ORDER BY started_at_ms, run_id
            """
        ).fetchall()
        manifest_rows = {
            str(row[0]): row
            for row in connection.execute(
                """
                SELECT run_id, manifest_json, manifest_sha256
                FROM polymarket_preregistration_manifest
                ORDER BY run_id
                """
            ).fetchall()
        }
        if (
            {str(row[0]) for row in run_rows} != set(expected)
            or set(manifest_rows) != set(expected)
            or len(run_rows) != len(expected)
        ):
            raise ValueError("Round 25 terminal database run set differs")
        rows_by_run = {str(row[0]): row for row in run_rows}
        for segment in transport["segments"]:
            run_id = segment["run_id"]
            if run_id is None:
                continue
            report_sha, _manifest_sha = _validate_database_identity(
                run_row=rows_by_run[str(run_id)],
                manifest_row=manifest_rows[str(run_id)],
                segment=segment,
            )
            if segment["eligible_for_condition_rebuild"]:
                eligible_runs.append(
                    _audit_eligible_run(store, segment=segment, observer=observer)
                )
            else:
                ineligible_runs.append(
                    {
                        "segment_index": segment["segment_index"],
                        "run_id": run_id,
                        "segment_status": segment["status"],
                        "database_status": rows_by_run[str(run_id)][1],
                        "report_sha256": report_sha,
                        "preregistration_manifest_sha256": segment[
                            "source_manifest_sha256"
                        ],
                        "receipts_replayed": False,
                    }
                )
    observed = (
        time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
        "created_at_ms": observed,
        "terminal_transport_manifest_sha256": transport["manifest_sha256"],
        "database_run_count": len(expected),
        "eligible_runs": eligible_runs,
        "ineligible_runs": ineligible_runs,
        "receipt_replay_complete": True,
        "condition_admission_pending": True,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    body["audit_sha256"] = _canonical_sha256(body)
    return validate_round25_terminal_receipt_audit(
        body,
        terminal_transport_manifest=transport,
    )


def load_round25_terminal_transport_manifest(path: str | Path) -> dict[str, object]:
    return validate_round25_terminal_transport_manifest(
        _read_object(Path(path), label="transport manifest")
    )


def load_round25_terminal_receipt_audit(
    path: str | Path,
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> dict[str, object]:
    return validate_round25_terminal_receipt_audit(
        _read_object(Path(path), label="receipt audit"),
        terminal_transport_manifest=terminal_transport_manifest,
    )


def write_round25_terminal_transport_manifest(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    validated = validate_round25_terminal_transport_manifest(value)
    write_bytes_atomic(
        Path(path),
        (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


def write_round25_terminal_receipt_audit(
    path: str | Path,
    value: Mapping[str, object],
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> None:
    validated = validate_round25_terminal_receipt_audit(
        value,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    write_bytes_atomic(
        Path(path),
        (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


__all__ = [
    "POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256",
    "POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION",
    "Round25TerminalReceiptObserver",
    "audit_round25_terminal_receipts",
    "build_round25_terminal_transport_manifest",
    "load_round25_terminal_design",
    "load_round25_terminal_receipt_audit",
    "load_round25_terminal_transport_manifest",
    "validate_round25_terminal_receipt_audit",
    "validate_round25_terminal_transport_manifest",
    "write_round25_terminal_receipt_audit",
    "write_round25_terminal_transport_manifest",
]

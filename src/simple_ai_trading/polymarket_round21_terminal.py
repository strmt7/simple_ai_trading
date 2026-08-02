"""Hash-bound terminal transport admission for the Round 21 corpus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .polymarket_round20_campaign import (
    POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256,
    POLYMARKET_ROUND20_CAMPAIGN_SECONDS,
    POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
    POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
    PolymarketRound20CampaignPlan,
    load_round20_campaign_plan,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_CONDITION_DURATION_MS,
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_ROLE_INTERVALS,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-terminal-transport-manifest-design-v1"
)
POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256 = (
    "7e2f294e20e3ce6425026a31f3bc63a50cf39e6e0ab51c25eafb20e7ca67b551"
)
POLYMARKET_ROUND21_TERMINAL_TRANSPORT_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round21-terminal-transport-manifest-v1"
)
POLYMARKET_ROUND21_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION = (
    "polymarket-round21-terminal-receipt-audit-v1"
)
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-021-terminal-transport-manifest-design-v1.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SEGMENT_NAME = re.compile(r"^segment-([0-9]{4})\.json$")
_MAXIMUM_JSON_BYTES = 512 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_TERMINAL_STATUSES = frozenset(("complete", "degraded"))
_ALL_STATUSES = frozenset((*_TERMINAL_STATUSES, "failed", "interrupted"))
_AUTHORITY_FIELDS = (
    "model_data_eligible",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
)
_SUMMARY_KEYS = {
    "segment_index",
    "source_artifact_sha256",
    "status",
    "details_kind",
    "observed_at_ms",
    "condition_admission_pending",
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
    "eligible_for_condition_rebuild",
    "exclusion_reasons",
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 21 terminal JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 terminal JSON contains {value}")


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


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    size = path.stat().st_size
    if not 2 <= size <= _MAXIMUM_JSON_BYTES:
        raise ValueError(f"{label} size differs")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return value


def _digest(value: object, *, label: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{label} digest differs")
    return selected


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} differs")
    return value


def _optional_integer(value: object, *, label: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, label=label, minimum=minimum)


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} differs")
    return tuple(value)


def load_round21_terminal_transport_design(repository: str | Path) -> dict[str, object]:
    path = Path(repository).resolve() / _DESIGN_RELATIVE
    payload = dict(_read_strict_json(path, label="Round 21 terminal design"))
    claimed = str(payload.pop("design_sha256", "")).strip().lower()
    parents = payload.get("parents")
    authority = payload.get("authority")
    if (
        claimed != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SCHEMA_VERSION
        or payload.get("round") != 21
        or not isinstance(parents, Mapping)
        or parents.get("round20_campaign_design_sha256")
        != POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256
        or parents.get("round21_dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or payload.get("purpose")
        != "transport_run_admission_only_not_condition_or_model_admission"
        or not isinstance(authority, Mapping)
        or authority.get("model_data_eligible") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("paper_trading_authority") is not False
        or authority.get("live_trading_authority") is not False
    ):
        raise ValueError("Round 21 terminal design differs")
    return {**payload, "design_sha256": claimed}


def _segment_paths(state_root: Path) -> tuple[Path, ...]:
    root = state_root / "segments"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Round 21 terminal segment directory is unavailable")
    json_paths = sorted(root.glob("*.json"))
    if not json_paths:
        raise ValueError("Round 21 terminal segment set is empty")
    for expected, path in enumerate(json_paths):
        match = _SEGMENT_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected:
            raise ValueError("Round 21 terminal segment set is not contiguous")
    return tuple(json_paths)


def _stream_counts(value: object, *, required: bool) -> dict[str, int]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key.strip()
        or type(count) is not int
        or count < 0
        for key, count in value.items()
    ):
        raise ValueError("Round 21 terminal stream counts differ")
    selected = {str(key): int(count) for key, count in sorted(value.items())}
    if required and any(selected.get(stream, 0) <= 0 for stream in (
        "clob_market",
        "polymarket_rtds",
    )):
        raise ValueError("Round 21 terminal required stream coverage differs")
    return selected


def _terminal_report_details(
    value: Mapping[str, object],
    *,
    plan: PolymarketRound20CampaignPlan,
    status: str,
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
    }
    if set(value) != expected:
        raise ValueError("Round 21 terminal report details differ")
    run_id = str(value.get("run_id") or "").strip().lower()
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Round 21 terminal run ID differs")
    started = _integer(value.get("started_at_ms"), label="segment start", minimum=1)
    ended = _integer(value.get("ended_at_ms"), label="segment end", minimum=1)
    raw_count = _integer(value.get("raw_message_count"), label="raw count")
    gap_count = _integer(value.get("stream_gap_count"), label="gap count")
    condition_count = _integer(value.get("condition_count"), label="condition count")
    duration_value = value.get("duration_seconds")
    if (
        isinstance(duration_value, bool)
        or not isinstance(duration_value, (int, float))
        or not math.isfinite(float(duration_value))
        or float(duration_value) < 0.0
        or abs(float(duration_value) - ((ended - started) / 1_000.0)) > 0.001
        or started < plan.scheduled_start_ms
        or ended < started
    ):
        raise ValueError("Round 21 terminal segment timing differs")
    errors = _text_tuple(value.get("errors"), label="segment errors")
    integrity = _text_tuple(
        value.get("integrity_errors"),
        label="segment integrity errors",
    )
    eligible_status = status in _TERMINAL_STATUSES
    streams = _stream_counts(value.get("stream_counts"), required=eligible_status)
    if eligible_status and (
        errors
        or integrity
        or raw_count <= 0
        or condition_count <= 0
        or (status == "complete" and gap_count != 0)
        or (status == "degraded" and gap_count <= 0)
    ):
        raise ValueError("Round 21 terminal eligible report differs")
    return {
        "details_kind": "terminal_report",
        "run_id": run_id,
        "manifest_sha256": _digest(
            value.get("manifest_sha256"),
            label="segment manifest",
        ),
        "report_sha256": _digest(
            value.get("report_sha256"),
            label="segment report",
        ),
        "started_at_ms": started,
        "ended_at_ms": ended,
        "duration_seconds": float(duration_value),
        "raw_message_count": raw_count,
        "stream_gap_count": gap_count,
        "stream_counts": streams,
        "condition_count": condition_count,
        "integrity_errors": list(integrity),
        "errors": list(errors),
    }


def _interrupted_report_details(
    value: Mapping[str, object],
    *,
    plan: PolymarketRound20CampaignPlan,
) -> dict[str, object]:
    expected = {
        "run_id",
        "report_sha256",
        "started_at_ms",
        "ended_at_ms",
        "raw_message_count",
        "integrity_errors",
        "errors",
    }
    if set(value) != expected:
        raise ValueError("Round 21 interrupted report details differ")
    run_id = str(value.get("run_id") or "").strip().lower()
    started = _integer(value.get("started_at_ms"), label="segment start", minimum=1)
    ended = _integer(value.get("ended_at_ms"), label="segment end", minimum=1)
    errors = _text_tuple(value.get("errors"), label="segment errors")
    integrity = _text_tuple(
        value.get("integrity_errors"),
        label="segment integrity errors",
    )
    if (
        _RUN_ID.fullmatch(run_id) is None
        or started < plan.scheduled_start_ms
        or ended < started
        or not errors
        or not integrity
    ):
        raise ValueError("Round 21 interrupted report differs")
    return {
        "details_kind": "interrupted_report",
        "run_id": run_id,
        "manifest_sha256": None,
        "report_sha256": _digest(
            value.get("report_sha256"),
            label="segment report",
        ),
        "started_at_ms": started,
        "ended_at_ms": ended,
        "duration_seconds": (ended - started) / 1_000.0,
        "raw_message_count": _integer(
            value.get("raw_message_count"),
            label="raw count",
        ),
        "stream_gap_count": None,
        "stream_counts": {},
        "condition_count": None,
        "integrity_errors": list(integrity),
        "errors": list(errors),
    }


def _failure_details(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {"failure_type", "failure"} or any(
        not isinstance(value.get(key), str) or not str(value[key]).strip()
        for key in ("failure_type", "failure")
    ):
        raise ValueError("Round 21 terminal failure details differ")
    return {
        "details_kind": "failure",
        "run_id": None,
        "manifest_sha256": None,
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
    }


def _source_segment(
    path: Path,
    *,
    plan: PolymarketRound20CampaignPlan,
    expected_index: int,
) -> dict[str, object]:
    source = dict(_read_strict_json(path, label="Round 20 segment result"))
    claimed = str(source.pop("artifact_sha256", "")).strip().lower()
    expected = {
        "schema_version",
        "plan_sha256",
        "segment_index",
        "status",
        "observed_at_ms",
        "condition_admission_pending",
        "details",
        *_AUTHORITY_FIELDS,
    }
    status = str(source.get("status") or "")
    details = source.get("details")
    pending = source.get("condition_admission_pending")
    if (
        set(source) != expected
        or claimed != _canonical_sha256(source)
        or _SHA256.fullmatch(claimed) is None
        or source.get("schema_version")
        != POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION
        or source.get("plan_sha256") != plan.plan_sha256
        or source.get("segment_index") != expected_index
        or status not in _ALL_STATUSES
        or type(pending) is not bool
        or pending != (status in _TERMINAL_STATUSES)
        or not isinstance(details, Mapping)
        or any(source.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 21 terminal source segment differs")
    observed = _integer(
        source.get("observed_at_ms"),
        label="segment observation",
        minimum=1,
    )
    if status == "interrupted":
        normalized = _interrupted_report_details(details, plan=plan)
    elif set(details) == {"failure_type", "failure"}:
        if status != "failed":
            raise ValueError("Round 21 terminal failure status differs")
        normalized = _failure_details(details)
    else:
        normalized = _terminal_report_details(details, plan=plan, status=status)
    ended = normalized["ended_at_ms"]
    if ended is not None and observed < ended:
        raise ValueError("Round 21 terminal segment observation predates its end")
    eligible = status in _TERMINAL_STATUSES
    reasons: list[str] = []
    if not eligible:
        reasons.append(f"segment_status_{status}")
    if not pending:
        reasons.append("condition_admission_not_pending")
    if normalized["integrity_errors"]:
        reasons.append("recorder_integrity_errors_present")
    if normalized["errors"]:
        reasons.append("recorder_errors_present")
    if eligible != (not reasons):
        raise ValueError("Round 21 terminal segment eligibility differs")
    return {
        "segment_index": expected_index,
        "source_artifact_sha256": claimed,
        "status": status,
        "observed_at_ms": observed,
        "condition_admission_pending": pending,
        **normalized,
        "eligible_for_condition_rebuild": eligible,
        "exclusion_reasons": reasons,
    }


def _overlap(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def _coverage(
    segments: Sequence[Mapping[str, object]],
    *,
    campaign_start_ms: int,
    campaign_end_ms: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    intervals: list[dict[str, object]] = []
    previous_lifecycle_end: int | None = None
    for segment in segments:
        started = segment["started_at_ms"]
        ended = segment["ended_at_ms"]
        if started is not None:
            if previous_lifecycle_end is not None and started < previous_lifecycle_end:
                raise ValueError("Round 21 terminal segment lifecycles overlap")
            previous_lifecycle_end = int(ended)
        if not segment["eligible_for_condition_rebuild"]:
            continue
        start = max(campaign_start_ms, int(started))
        end = min(campaign_end_ms, int(ended))
        if end <= start:
            raise ValueError("Round 21 terminal eligible interval is empty")
        if intervals and start < int(intervals[-1]["end_ms"]):
            raise ValueError("Round 21 terminal eligible intervals overlap")
        intervals.append(
            {
                "segment_index": segment["segment_index"],
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
            }
        )
    uncovered: list[dict[str, object]] = []
    cursor = campaign_start_ms
    for interval in intervals:
        start = int(interval["start_ms"])
        if start > cursor:
            uncovered.append({"start_ms": cursor, "end_ms": start})
        cursor = max(cursor, int(interval["end_ms"]))
    if cursor < campaign_end_ms:
        uncovered.append({"start_ms": cursor, "end_ms": campaign_end_ms})
    completed_gaps: list[dict[str, object]] = []
    for gap in uncovered:
        start = int(gap["start_ms"])
        end = int(gap["end_ms"])
        completed_gaps.append(
            {
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
                "role_overlaps_ms": [
                    {
                        "role": role,
                        "duration_ms": _overlap(
                            start,
                            end,
                            campaign_start_ms + offset_start,
                            campaign_start_ms + offset_end,
                        ),
                    }
                    for role, offset_start, offset_end in POLYMARKET_ROUND21_ROLE_INTERVALS
                    if _overlap(
                        start,
                        end,
                        campaign_start_ms + offset_start,
                        campaign_start_ms + offset_end,
                    )
                    > 0
                ],
            }
        )
    roles: list[dict[str, object]] = []
    for role, offset_start, offset_end in POLYMARKET_ROUND21_ROLE_INTERVALS:
        start = campaign_start_ms + offset_start
        end = campaign_start_ms + offset_end
        eligible_ms = sum(
            _overlap(start, end, int(interval["start_ms"]), int(interval["end_ms"]))
            for interval in intervals
        )
        duration = end - start
        roles.append(
            {
                "role": role,
                "start_ms": start,
                "end_ms": end,
                "scheduled_duration_ms": duration,
                "provisional_eligible_transport_ms": eligible_ms,
                "known_uncovered_transport_ms": duration - eligible_ms,
                "complete_transport_interval_coverage": eligible_ms == duration,
            }
        )
    return intervals, completed_gaps, roles


def _campaign_state_mode(
    state: Mapping[str, object],
    *,
    plan: PolymarketRound20CampaignPlan,
    segments: Sequence[Mapping[str, object]],
) -> str:
    if (
        state.get("schema_version") != POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION
        or state.get("plan_sha256") != plan.plan_sha256
        or any(state.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 21 terminal campaign state differs")
    terminal_status = state.get("status")
    if terminal_status in {"campaign_window_ended", "campaign_failed"}:
        status_counts: dict[str, int] = {}
        for segment in segments:
            status = str(segment["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        if (
            state.get("terminal_segment_count") != len(segments)
            or state.get("status_counts") != dict(sorted(status_counts.items()))
            or state.get("condition_admission_pending") is not True
        ):
            raise ValueError("Round 21 terminal persisted state differs")
        return "persisted_terminal"
    if not segments:
        raise ValueError("Round 21 terminal active heartbeat is not superseded")
    last = segments[-1]
    details = state.get("details")
    state_observed = state.get("observed_at_ms")
    if (
        not isinstance(state.get("phase"), str)
        or not str(state["phase"]).strip()
        or state.get("segment_index") != last["segment_index"]
        or not isinstance(details, Mapping)
        or details.get("run_id") != last["run_id"]
        or type(state_observed) is not int
        or state_observed <= 0
        or state_observed >= last["observed_at_ms"]
        or last["ended_at_ms"] is None
        or last["ended_at_ms"] < plan.scheduled_end_ms
    ):
        raise ValueError("Round 21 terminal active heartbeat is not superseded")
    return "superseded_active_heartbeat"


def _validate_segment_summary(
    value: object,
    *,
    expected_index: int,
    campaign_start_ms: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SUMMARY_KEYS:
        raise ValueError("Round 21 terminal manifest segment differs")
    selected = dict(value)
    status = str(selected.get("status") or "")
    kind = selected.get("details_kind")
    if (
        selected.get("segment_index") != expected_index
        or status not in _ALL_STATUSES
        or kind not in {"terminal_report", "interrupted_report", "failure"}
        or type(selected.get("condition_admission_pending")) is not bool
        or selected["condition_admission_pending"] != (status in _TERMINAL_STATUSES)
        or type(selected.get("eligible_for_condition_rebuild")) is not bool
        or selected["eligible_for_condition_rebuild"] != (status in _TERMINAL_STATUSES)
    ):
        raise ValueError("Round 21 terminal manifest segment status differs")
    _digest(selected.get("source_artifact_sha256"), label="source segment")
    _integer(selected.get("observed_at_ms"), label="segment observation", minimum=1)
    errors = _text_tuple(selected.get("errors"), label="segment errors")
    integrity = _text_tuple(
        selected.get("integrity_errors"),
        label="segment integrity errors",
    )
    reasons = _text_tuple(
        selected.get("exclusion_reasons"),
        label="segment exclusion reasons",
    )
    expected_reasons: list[str] = []
    if status not in _TERMINAL_STATUSES:
        expected_reasons.append(f"segment_status_{status}")
    if not selected["condition_admission_pending"]:
        expected_reasons.append("condition_admission_not_pending")
    if integrity:
        expected_reasons.append("recorder_integrity_errors_present")
    if errors:
        expected_reasons.append("recorder_errors_present")
    if reasons != tuple(expected_reasons):
        raise ValueError("Round 21 terminal manifest exclusion differs")
    if kind == "failure":
        nullable = (
            "run_id",
            "manifest_sha256",
            "report_sha256",
            "started_at_ms",
            "ended_at_ms",
            "duration_seconds",
            "raw_message_count",
            "stream_gap_count",
            "condition_count",
        )
        if (
            status != "failed"
            or any(selected[key] is not None for key in nullable)
            or selected["stream_counts"] != {}
            or errors
            or integrity
        ):
            raise ValueError("Round 21 terminal manifest failure differs")
        return selected
    run_id = str(selected.get("run_id") or "")
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Round 21 terminal manifest run ID differs")
    started = _integer(selected.get("started_at_ms"), label="segment start", minimum=1)
    ended = _integer(selected.get("ended_at_ms"), label="segment end", minimum=1)
    duration = selected.get("duration_seconds")
    raw_count = _integer(selected.get("raw_message_count"), label="raw count")
    if (
        started < campaign_start_ms
        or ended < started
        or selected["observed_at_ms"] < ended
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or abs(float(duration) - ((ended - started) / 1_000.0)) > 0.001
    ):
        raise ValueError("Round 21 terminal manifest timing differs")
    _digest(selected.get("report_sha256"), label="segment report")
    if kind == "interrupted_report":
        if (
            status != "interrupted"
            or selected.get("manifest_sha256") is not None
            or selected.get("stream_gap_count") is not None
            or selected.get("condition_count") is not None
            or selected.get("stream_counts") != {}
            or not errors
            or not integrity
        ):
            raise ValueError("Round 21 terminal manifest interruption differs")
        return selected
    _digest(selected.get("manifest_sha256"), label="segment manifest")
    gap_count = _integer(selected.get("stream_gap_count"), label="gap count")
    condition_count = _integer(selected.get("condition_count"), label="condition count")
    _stream_counts(selected.get("stream_counts"), required=status in _TERMINAL_STATUSES)
    if status in _TERMINAL_STATUSES and (
        errors
        or integrity
        or raw_count <= 0
        or condition_count <= 0
        or (status == "complete" and gap_count != 0)
        or (status == "degraded" and gap_count <= 0)
    ):
        raise ValueError("Round 21 terminal manifest eligible segment differs")
    return selected


def validate_round21_terminal_transport_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "terminal_design_sha256",
        "created_at_ms",
        "source_plan_sha256",
        "campaign_start_ms",
        "campaign_end_ms",
        "campaign_state_artifact_sha256",
        "campaign_state_mode",
        "segments",
        "eligible_run_ids",
        "provisional_eligible_transport_intervals",
        "known_ineligible_or_unobserved_intervals",
        "role_transport_coverage",
        "all_scheduled_transport_interval_covered",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    }
    start = payload.get("campaign_start_ms")
    end = payload.get("campaign_end_ms")
    created = payload.get("created_at_ms")
    raw_segments = payload.get("segments")
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_MANIFEST_SCHEMA_VERSION
        or payload.get("terminal_design_sha256")
        != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        or _SHA256.fullmatch(str(payload.get("source_plan_sha256") or "")) is None
        or type(start) is not int
        or start <= 0
        or start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
        or type(end) is not int
        or end - start != POLYMARKET_ROUND20_CAMPAIGN_SECONDS * 1_000
        or type(created) is not int
        or created < end
        or _SHA256.fullmatch(
            str(payload.get("campaign_state_artifact_sha256") or "")
        )
        is None
        or payload.get("campaign_state_mode")
        not in {"persisted_terminal", "superseded_active_heartbeat"}
        or not isinstance(raw_segments, list)
        or not raw_segments
    ):
        raise ValueError("Round 21 terminal transport manifest differs")
    segments = [
        _validate_segment_summary(
            item,
            expected_index=index,
            campaign_start_ms=start,
        )
        for index, item in enumerate(raw_segments)
    ]
    intervals, gaps, roles = _coverage(
        segments,
        campaign_start_ms=start,
        campaign_end_ms=end,
    )
    eligible_run_ids = [
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
        len(set(eligible_run_ids)) != len(eligible_run_ids)
        or payload.get("eligible_run_ids") != eligible_run_ids
        or payload.get("provisional_eligible_transport_intervals") != intervals
        or payload.get("known_ineligible_or_unobserved_intervals") != gaps
        or payload.get("role_transport_coverage") != roles
        or payload.get("all_scheduled_transport_interval_covered") is not (not gaps)
        or payload.get("condition_admission_pending") is not bool(eligible_run_ids)
        or any(type(payload.get(field)) is not bool for field in bool_fields)
        or payload.get("outcomes_consulted") is not False
        or payload.get("model_scores_consulted") is not False
        or any(payload.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 21 terminal transport derivation differs")
    return {**payload, "manifest_sha256": claimed}


def build_round21_terminal_transport_manifest(
    repository: str | Path,
    *,
    plan_path: str | Path,
    state_root: str | Path,
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Bind terminal segment admission without opening capture data or outcomes."""

    root = Path(repository).resolve()
    load_round21_terminal_transport_design(root)
    plan = load_round20_campaign_plan(plan_path)
    observed = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    if observed < plan.scheduled_end_ms:
        raise RuntimeError("Round 21 terminal transport cannot open before campaign end")
    state_directory = Path(state_root).resolve()
    state_source = dict(
        _read_strict_json(
            state_directory / "campaign-state.json",
            label="Round 20 campaign state",
        )
    )
    state_sha256 = str(state_source.pop("artifact_sha256", "")).strip().lower()
    if (
        state_sha256 != _canonical_sha256(state_source)
        or _SHA256.fullmatch(state_sha256) is None
    ):
        raise ValueError("Round 21 terminal campaign state hash differs")
    segments = [
        _source_segment(path, plan=plan, expected_index=index)
        for index, path in enumerate(_segment_paths(state_directory))
    ]
    state_mode = _campaign_state_mode(
        state_source,
        plan=plan,
        segments=segments,
    )
    intervals, gaps, roles = _coverage(
        segments,
        campaign_start_ms=plan.scheduled_start_ms,
        campaign_end_ms=plan.scheduled_end_ms,
    )
    eligible_run_ids = [
        segment["run_id"]
        for segment in segments
        if segment["eligible_for_condition_rebuild"]
    ]
    payload: dict[str, object] = {
        "schema_version": (
            POLYMARKET_ROUND21_TERMINAL_TRANSPORT_MANIFEST_SCHEMA_VERSION
        ),
        "terminal_design_sha256": (
            POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        ),
        "created_at_ms": observed,
        "source_plan_sha256": plan.plan_sha256,
        "campaign_start_ms": plan.scheduled_start_ms,
        "campaign_end_ms": plan.scheduled_end_ms,
        "campaign_state_artifact_sha256": state_sha256,
        "campaign_state_mode": state_mode,
        "segments": segments,
        "eligible_run_ids": eligible_run_ids,
        "provisional_eligible_transport_intervals": intervals,
        "known_ineligible_or_unobserved_intervals": gaps,
        "role_transport_coverage": roles,
        "all_scheduled_transport_interval_covered": not gaps,
        "condition_admission_pending": bool(eligible_run_ids),
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round21_terminal_transport_manifest(payload)


def load_round21_terminal_transport_manifest(path: str | Path) -> dict[str, object]:
    return validate_round21_terminal_transport_manifest(
        _read_strict_json(Path(path), label="Round 21 terminal transport manifest")
    )


def write_round21_terminal_transport_manifest(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    validated = validate_round21_terminal_transport_manifest(value)
    encoded = (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(Path(path), encoded)


def _strict_json_text(value: object, *, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(
            str(value),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} is not an object")
    return parsed


def _receipt_chain(previous: str, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        f"{previous}:{_canonical_sha256(payload)}".encode("ascii")
    ).hexdigest()


def _validate_database_identity(
    *,
    row: Sequence[object],
    manifest_row: Sequence[object],
    segment: Mapping[str, object],
) -> tuple[str, str]:
    run_id, database_status, started_at_ms, ended_at_ms, report_json, report_sha256 = row
    report = dict(_strict_json_text(report_json, label="Round 21 terminal run report"))
    embedded_report_sha256 = str(report.pop("report_sha256", "")).strip().lower()
    expected_database_status = (
        segment["status"]
        if segment["status"] in _TERMINAL_STATUSES
        else "failed"
    )
    if (
        run_id != segment["run_id"]
        or database_status != expected_database_status
        or report_sha256 != segment["report_sha256"]
        or embedded_report_sha256 != report_sha256
        or _canonical_sha256(report) != report_sha256
        or report.get("run_id") != run_id
        or report.get("status") != database_status
        or report.get("started_at_ms") != started_at_ms
        or report.get("ended_at_ms") != ended_at_ms
        or started_at_ms != segment["started_at_ms"]
        or ended_at_ms != segment["ended_at_ms"]
        or report.get("raw_message_count") != segment["raw_message_count"]
        or report.get("integrity_errors") != segment["integrity_errors"]
        or report.get("errors") != segment["errors"]
    ):
        raise ValueError("Round 21 terminal database report differs")
    manifest_run_id, manifest_json, manifest_sha256 = manifest_row
    manifest = dict(
        _strict_json_text(
            manifest_json,
            label="Round 21 terminal preregistration manifest",
        )
    )
    embedded_manifest_sha256 = str(
        manifest.pop("manifest_sha256", "")
    ).strip().lower()
    if (
        manifest_run_id != run_id
        or manifest.get("run_id") != run_id
        or embedded_manifest_sha256 != manifest_sha256
        or _canonical_sha256(manifest) != manifest_sha256
        or (
            segment["manifest_sha256"] is not None
            and manifest_sha256 != segment["manifest_sha256"]
        )
    ):
        raise ValueError("Round 21 terminal database manifest differs")
    return str(report_sha256), str(manifest_sha256)


def _audit_eligible_run(
    store: PolymarketEvidenceStore,
    *,
    segment: Mapping[str, object],
    preregistration_sha256: str,
) -> dict[str, object]:
    run_id = str(segment["run_id"])
    counts: dict[str, int] = defaultdict(int)
    receipt_count = 0
    first_wall_ms: int | None = None
    last_wall_ms: int | None = None
    receipt_chain = _EMPTY_SHA256
    for message in store.iter_terminal_capture_messages(run_id):
        raw_sha256 = hashlib.sha256(message.raw_text.encode("utf-8")).hexdigest()
        receipt_chain = _receipt_chain(
            receipt_chain,
            {
                "stream": message.stream,
                "connection_id": message.connection_id,
                "sequence_number": message.sequence_number,
                "received_wall_ms": message.received_wall_ms,
                "received_monotonic_ns": message.received_monotonic_ns,
                "raw_sha256": raw_sha256,
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
        raise ValueError("Round 21 terminal receipt accounting differs")
    gap_count = 0
    first_gap_ms: int | None = None
    last_gap_ms: int | None = None
    gap_chain = _EMPTY_SHA256
    for gap in store.iter_terminal_stream_gaps(run_id):
        gap_chain = _receipt_chain(
            gap_chain,
            {
                "stream": gap.stream,
                "connection_id": gap.connection_id,
                "opened_at_ms": gap.opened_at_ms,
                "reason": gap.reason,
                "last_sequence_number": gap.last_sequence_number,
            },
        )
        gap_count += 1
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
    if gap_count != segment["stream_gap_count"]:
        raise ValueError("Round 21 terminal gap accounting differs")
    return {
        "segment_index": segment["segment_index"],
        "run_id": run_id,
        "status": segment["status"],
        "report_sha256": segment["report_sha256"],
        "preregistration_manifest_sha256": preregistration_sha256,
        "receipt_count": receipt_count,
        "stream_counts": dict(sorted(counts.items())),
        "first_receipt_wall_ms": first_wall_ms,
        "last_receipt_wall_ms": last_wall_ms,
        "receipt_chain_sha256": receipt_chain,
        "gap_count": gap_count,
        "first_gap_opened_at_ms": first_gap_ms,
        "last_gap_opened_at_ms": last_gap_ms,
        "gap_chain_sha256": gap_chain,
    }


def _validate_receipt_run(
    value: object,
    *,
    segment: Mapping[str, object],
) -> dict[str, object]:
    expected = {
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
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Round 21 terminal receipt run differs")
    selected = dict(value)
    receipt_count = _integer(selected.get("receipt_count"), label="receipt count")
    gap_count = _integer(selected.get("gap_count"), label="gap count")
    counts = _stream_counts(selected.get("stream_counts"), required=True)
    first_receipt = _integer(
        selected.get("first_receipt_wall_ms"),
        label="first receipt",
        minimum=1,
    )
    last_receipt = _integer(
        selected.get("last_receipt_wall_ms"),
        label="last receipt",
        minimum=1,
    )
    first_gap = _optional_integer(
        selected.get("first_gap_opened_at_ms"),
        label="first gap",
        minimum=1,
    )
    last_gap = _optional_integer(
        selected.get("last_gap_opened_at_ms"),
        label="last gap",
        minimum=1,
    )
    if (
        selected.get("segment_index") != segment["segment_index"]
        or selected.get("run_id") != segment["run_id"]
        or selected.get("status") != segment["status"]
        or selected.get("report_sha256") != segment["report_sha256"]
        or receipt_count != segment["raw_message_count"]
        or counts != segment["stream_counts"]
        or sum(counts.values()) != receipt_count
        or first_receipt > last_receipt
        or gap_count != segment["stream_gap_count"]
        or (gap_count == 0) != (first_gap is None and last_gap is None)
        or (first_gap is not None and last_gap is not None and first_gap > last_gap)
    ):
        raise ValueError("Round 21 terminal receipt run accounting differs")
    for name in (
        "report_sha256",
        "preregistration_manifest_sha256",
        "receipt_chain_sha256",
        "gap_chain_sha256",
    ):
        _digest(selected.get(name), label=name.replace("_", " "))
    if gap_count == 0 and selected["gap_chain_sha256"] != _EMPTY_SHA256:
        raise ValueError("Round 21 terminal empty gap chain differs")
    return selected


def _validate_ineligible_run(
    value: object,
    *,
    segment: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "segment_index",
        "run_id",
        "segment_status",
        "database_status",
        "report_sha256",
        "preregistration_manifest_sha256",
        "receipts_replayed",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Round 21 terminal ineligible run differs")
    selected = dict(value)
    if (
        selected.get("segment_index") != segment["segment_index"]
        or selected.get("run_id") != segment["run_id"]
        or selected.get("segment_status") != segment["status"]
        or selected.get("database_status") != "failed"
        or selected.get("report_sha256") != segment["report_sha256"]
        or selected.get("receipts_replayed") is not False
    ):
        raise ValueError("Round 21 terminal ineligible run accounting differs")
    _digest(
        selected.get("preregistration_manifest_sha256"),
        label="ineligible preregistration manifest",
    )
    return selected


def validate_round21_terminal_receipt_audit(
    value: Mapping[str, object],
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> dict[str, object]:
    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    payload = dict(value)
    claimed = str(payload.pop("audit_sha256", "")).strip().lower()
    expected_keys = {
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
    eligible_source = [
        segment
        for segment in transport["segments"]
        if segment["eligible_for_condition_rebuild"]
    ]
    ineligible_source = [
        segment
        for segment in transport["segments"]
        if not segment["eligible_for_condition_rebuild"]
        and segment["run_id"] is not None
    ]
    eligible_values = payload.get("eligible_runs")
    ineligible_values = payload.get("ineligible_runs")
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION
        or payload.get("terminal_design_sha256")
        != POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        or payload.get("terminal_transport_manifest_sha256")
        != transport["manifest_sha256"]
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] < transport["created_at_ms"]
        or not isinstance(eligible_values, list)
        or not isinstance(ineligible_values, list)
        or len(eligible_values) != len(eligible_source)
        or len(ineligible_values) != len(ineligible_source)
        or payload.get("database_run_count")
        != len(eligible_source) + len(ineligible_source)
    ):
        raise ValueError("Round 21 terminal receipt audit differs")
    eligible = [
        _validate_receipt_run(item, segment=segment)
        for item, segment in zip(eligible_values, eligible_source, strict=True)
    ]
    ineligible = [
        _validate_ineligible_run(item, segment=segment)
        for item, segment in zip(ineligible_values, ineligible_source, strict=True)
    ]
    bool_fields = (
        "receipt_replay_complete",
        "condition_admission_pending",
        "outcomes_consulted",
        "model_scores_consulted",
        *_AUTHORITY_FIELDS,
    )
    if (
        not eligible
        or payload.get("receipt_replay_complete") is not True
        or payload.get("condition_admission_pending") is not True
        or any(type(payload.get(field)) is not bool for field in bool_fields)
        or payload.get("outcomes_consulted") is not False
        or payload.get("model_scores_consulted") is not False
        or any(payload.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 21 terminal receipt authority differs")
    return {
        **payload,
        "eligible_runs": eligible,
        "ineligible_runs": ineligible,
        "audit_sha256": claimed,
    }


def audit_round21_terminal_receipts(
    *,
    database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Reconcile terminal exact receipts once without reading outcomes or models."""

    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    eligible_segments = [
        segment
        for segment in transport["segments"]
        if segment["eligible_for_condition_rebuild"]
    ]
    if not eligible_segments:
        raise RuntimeError("Round 21 terminal receipt audit has no eligible run")
    expected_segments = {
        segment["run_id"]: segment
        for segment in transport["segments"]
        if segment["run_id"] is not None
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
        rows = connection.execute(
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
            {str(row[0]) for row in rows} != set(expected_segments)
            or set(manifest_rows) != set(expected_segments)
            or len(rows) != len(expected_segments)
        ):
            raise ValueError("Round 21 terminal database run set differs")
        rows_by_run = {str(row[0]): row for row in rows}
        for segment in transport["segments"]:
            run_id = segment["run_id"]
            if run_id is None:
                continue
            report_sha256, preregistration_sha256 = _validate_database_identity(
                row=rows_by_run[str(run_id)],
                manifest_row=manifest_rows[str(run_id)],
                segment=segment,
            )
            if segment["eligible_for_condition_rebuild"]:
                eligible_runs.append(
                    _audit_eligible_run(
                        store,
                        segment=segment,
                        preregistration_sha256=preregistration_sha256,
                    )
                )
            else:
                ineligible_runs.append(
                    {
                        "segment_index": segment["segment_index"],
                        "run_id": run_id,
                        "segment_status": segment["status"],
                        "database_status": "failed",
                        "report_sha256": report_sha256,
                        "preregistration_manifest_sha256": (
                            preregistration_sha256
                        ),
                        "receipts_replayed": False,
                    }
                )
    observed = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
        "terminal_design_sha256": (
            POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256
        ),
        "created_at_ms": observed,
        "terminal_transport_manifest_sha256": transport["manifest_sha256"],
        "database_run_count": len(expected_segments),
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
    payload["audit_sha256"] = _canonical_sha256(payload)
    return validate_round21_terminal_receipt_audit(
        payload,
        terminal_transport_manifest=transport,
    )


def load_round21_terminal_receipt_audit(
    path: str | Path,
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> dict[str, object]:
    return validate_round21_terminal_receipt_audit(
        _read_strict_json(Path(path), label="Round 21 terminal receipt audit"),
        terminal_transport_manifest=terminal_transport_manifest,
    )


def write_round21_terminal_receipt_audit(
    path: str | Path,
    value: Mapping[str, object],
    *,
    terminal_transport_manifest: Mapping[str, object],
) -> None:
    validated = validate_round21_terminal_receipt_audit(
        value,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    encoded = (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(Path(path), encoded)


__all__ = [
    "POLYMARKET_ROUND21_TERMINAL_TRANSPORT_DESIGN_SHA256",
    "POLYMARKET_ROUND21_TERMINAL_TRANSPORT_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION",
    "audit_round21_terminal_receipts",
    "build_round21_terminal_transport_manifest",
    "load_round21_terminal_transport_design",
    "load_round21_terminal_receipt_audit",
    "load_round21_terminal_transport_manifest",
    "validate_round21_terminal_receipt_audit",
    "validate_round21_terminal_transport_manifest",
    "write_round21_terminal_receipt_audit",
    "write_round21_terminal_transport_manifest",
]

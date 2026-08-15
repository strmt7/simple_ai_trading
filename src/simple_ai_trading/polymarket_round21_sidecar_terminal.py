"""Terminal transport authority for the independent Round 21 Binance sidecar."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import time

from .polymarket_round21_sidecar_campaign import (
    load_round21_sidecar_campaign_plan,
    load_round21_sidecar_segment_results,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND21_SIDECAR_TERMINAL_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-terminal-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_STREAMS = frozenset(("binance_spot", "binance_futures"))
_TERMINAL_STATUSES = frozenset(("complete", "degraded"))
_MAXIMUM_JSON_BYTES = 4 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 sidecar terminal JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 sidecar terminal JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _digest(value: object, *, label: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 21 sidecar terminal {label} differs")
    return selected


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(f"Round 21 sidecar terminal {label} differs")
    return int(value)


def _eligible_segment(result: Mapping[str, object]) -> dict[str, object]:
    details = result.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("Round 21 sidecar terminal segment details differ")
    run_id = str(details.get("run_id") or "").strip().lower()
    started = _integer(details.get("started_at_ms"), label="segment start", minimum=1)
    ended = _integer(details.get("ended_at_ms"), label="segment end", minimum=1)
    raw_count = _integer(
        details.get("raw_message_count"),
        label="raw message count",
        minimum=1,
    )
    manifest_sha256 = _digest(details.get("manifest_sha256"), label="segment manifest")
    report_sha256 = _digest(details.get("report_sha256"), label="segment report")
    gap_count = _integer(details.get("stream_gap_count"), label="gap count")
    stream_counts = details.get("stream_counts")
    integrity_errors = details.get("integrity_errors")
    errors = details.get("errors")
    if (
        _RUN_ID.fullmatch(run_id) is None
        or started >= ended
        or not isinstance(stream_counts, Mapping)
        or set(stream_counts) != _STREAMS
        or any(
            type(value) is not int or int(value) <= 0
            for value in stream_counts.values()
        )
        or sum(int(value) for value in stream_counts.values()) != raw_count
        or integrity_errors != []
        or not isinstance(errors, list)
        or any(not isinstance(error, str) or not error for error in errors)
    ):
        raise ValueError("Round 21 sidecar terminal eligible segment differs")
    return {
        "segment_index": result["segment_index"],
        "status": result["status"],
        "source_result_artifact_sha256": result["artifact_sha256"],
        "run_id": run_id,
        "started_at_ms": started,
        "ended_at_ms": ended,
        "preregistration_manifest_sha256": manifest_sha256,
        "recorder_report_sha256": report_sha256,
        "raw_message_count": raw_count,
        "stream_gap_count": gap_count,
        "stream_counts": dict(sorted(stream_counts.items())),
        "eligible_for_optional_feature_replay": True,
        "exclusion_reasons": [],
    }


def _excluded_segment(result: Mapping[str, object]) -> dict[str, object]:
    status = str(result.get("status") or "")
    return {
        "segment_index": result["segment_index"],
        "status": status,
        "source_result_artifact_sha256": result["artifact_sha256"],
        "run_id": None,
        "started_at_ms": None,
        "ended_at_ms": None,
        "preregistration_manifest_sha256": None,
        "recorder_report_sha256": None,
        "raw_message_count": 0,
        "stream_gap_count": 0,
        "stream_counts": {},
        "eligible_for_optional_feature_replay": False,
        "exclusion_reasons": [f"segment_status_{status}"],
    }


def _validate_segment(value: object, *, expected_index: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 sidecar terminal segment differs")
    selected = dict(value)
    expected_keys = {
        "segment_index",
        "status",
        "source_result_artifact_sha256",
        "run_id",
        "started_at_ms",
        "ended_at_ms",
        "preregistration_manifest_sha256",
        "recorder_report_sha256",
        "raw_message_count",
        "stream_gap_count",
        "stream_counts",
        "eligible_for_optional_feature_replay",
        "exclusion_reasons",
    }
    eligible = selected.get("eligible_for_optional_feature_replay")
    if (
        set(selected) != expected_keys
        or selected.get("segment_index") != expected_index
        or selected.get("status") not in _TERMINAL_STATUSES | {"failed", "interrupted"}
        or _SHA256.fullmatch(str(selected.get("source_result_artifact_sha256") or ""))
        is None
        or type(eligible) is not bool
        or not isinstance(selected.get("exclusion_reasons"), list)
    ):
        raise ValueError("Round 21 sidecar terminal segment differs")
    if eligible:
        run_id = str(selected.get("run_id") or "")
        counts = selected.get("stream_counts")
        if (
            selected["status"] not in _TERMINAL_STATUSES
            or _RUN_ID.fullmatch(run_id) is None
            or _integer(selected.get("started_at_ms"), label="segment start", minimum=1)
            >= _integer(selected.get("ended_at_ms"), label="segment end", minimum=1)
            or any(
                _SHA256.fullmatch(str(selected.get(field) or "")) is None
                for field in (
                    "preregistration_manifest_sha256",
                    "recorder_report_sha256",
                )
            )
            or not _integer(
                selected.get("raw_message_count"),
                label="raw message count",
                minimum=1,
            )
            or not isinstance(counts, Mapping)
            or set(counts) != _STREAMS
            or any(
                type(count) is not int or int(count) <= 0 for count in counts.values()
            )
            or sum(int(count) for count in counts.values())
            != selected["raw_message_count"]
            or type(selected.get("stream_gap_count")) is not int
            or int(selected["stream_gap_count"]) < 0
            or selected["exclusion_reasons"] != []
        ):
            raise ValueError("Round 21 sidecar terminal eligible segment differs")
    elif (
        selected["status"] in _TERMINAL_STATUSES
        or any(
            selected.get(field) is not None
            for field in (
                "run_id",
                "started_at_ms",
                "ended_at_ms",
                "preregistration_manifest_sha256",
                "recorder_report_sha256",
            )
        )
        or selected.get("raw_message_count") != 0
        or selected.get("stream_gap_count") != 0
        or selected.get("stream_counts") != {}
        or selected.get("exclusion_reasons") != [f"segment_status_{selected['status']}"]
    ):
        raise ValueError("Round 21 sidecar terminal exclusion differs")
    return selected


def validate_round21_sidecar_terminal_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "source_plan_sha256",
        "campaign_start_ms",
        "campaign_end_ms",
        "segments",
        "eligible_run_ids",
        "excluded_segment_indices",
        "all_eligible_reports_integrity_clean",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Round 21 sidecar terminal segments differ")
    segments = [
        _validate_segment(item, expected_index=index)
        for index, item in enumerate(raw_segments)
    ]
    eligible = [
        segment
        for segment in segments
        if segment["eligible_for_optional_feature_replay"]
    ]
    run_ids = [segment["run_id"] for segment in eligible]
    exclusions = [
        segment["segment_index"]
        for segment in segments
        if not segment["eligible_for_optional_feature_replay"]
    ]
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_SIDECAR_TERMINAL_SCHEMA_VERSION
        or _integer(payload.get("created_at_ms"), label="creation time", minimum=1)
        < _integer(payload.get("campaign_end_ms"), label="campaign end", minimum=1)
        or _integer(payload.get("campaign_start_ms"), label="campaign start", minimum=1)
        >= payload["campaign_end_ms"]
        or _SHA256.fullmatch(str(payload.get("source_plan_sha256") or "")) is None
        or not eligible
        or len(run_ids) != len(set(run_ids))
        or any(
            int(current["started_at_ms"]) < int(previous["ended_at_ms"])
            for previous, current in zip(eligible, eligible[1:], strict=False)
        )
        or payload.get("eligible_run_ids") != run_ids
        or payload.get("excluded_segment_indices") != exclusions
        or payload.get("all_eligible_reports_integrity_clean") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "outcomes_consulted",
                "model_scores_consulted",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 sidecar terminal manifest differs")
    return {**payload, "manifest_sha256": claimed}


def build_round21_sidecar_terminal_manifest(
    *,
    plan_path: str | Path,
    state_root: str | Path,
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Bind terminal sidecar segments without opening messages, outcomes, or models."""

    plan = load_round21_sidecar_campaign_plan(plan_path)
    observed = (
        time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    )
    if observed < plan.scheduled_end_ms:
        raise RuntimeError("Round 21 sidecar terminal cannot open before campaign end")
    results = load_round21_sidecar_segment_results(state_root, plan)
    if not results:
        raise RuntimeError("Round 21 sidecar terminal has no segment results")
    segments = [
        _eligible_segment(result)
        if result["status"] in _TERMINAL_STATUSES
        else _excluded_segment(result)
        for result in results
    ]
    eligible = [
        segment
        for segment in segments
        if segment["eligible_for_optional_feature_replay"]
    ]
    if not eligible:
        raise RuntimeError("Round 21 sidecar terminal has no eligible segment")
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_SIDECAR_TERMINAL_SCHEMA_VERSION,
        "created_at_ms": observed,
        "source_plan_sha256": plan.plan_sha256,
        "campaign_start_ms": plan.scheduled_start_ms,
        "campaign_end_ms": plan.scheduled_end_ms,
        "segments": segments,
        "eligible_run_ids": [segment["run_id"] for segment in eligible],
        "excluded_segment_indices": [
            segment["segment_index"]
            for segment in segments
            if not segment["eligible_for_optional_feature_replay"]
        ],
        "all_eligible_reports_integrity_clean": True,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round21_sidecar_terminal_manifest(payload)


def write_round21_sidecar_terminal_manifest(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    selected = validate_round21_sidecar_terminal_manifest(value)
    write_bytes_atomic(
        Path(path),
        (json.dumps(selected, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


def load_round21_sidecar_terminal_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    size = source.stat().st_size if source.is_file() else 0
    if source.is_symlink() or size < 2 or size > _MAXIMUM_JSON_BYTES:
        raise ValueError("Round 21 sidecar terminal manifest is unavailable")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 sidecar terminal manifest is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 sidecar terminal manifest is invalid")
    return validate_round21_sidecar_terminal_manifest(value)


__all__ = [
    "POLYMARKET_ROUND21_SIDECAR_TERMINAL_SCHEMA_VERSION",
    "build_round21_sidecar_terminal_manifest",
    "load_round21_sidecar_terminal_manifest",
    "validate_round21_sidecar_terminal_manifest",
    "write_round21_sidecar_terminal_manifest",
]

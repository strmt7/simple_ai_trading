"""Hash-bound segmented-corpus catalog for qualified Round 73 captures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Mapping

import duckdb

from .impact_absorption import ROUND73_DESIGN_SHA256
from .impact_absorption_features import (
    ROUND73_FEATURE_SOURCE_SCHEMA_VERSION,
    diagnose_round73_feature_source,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_CONTRACT_SHA256,
    IMPACT_CAPTURE_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_SCHEMA_VERSION,
    IMPACT_CAPTURE_SYMBOLS,
    ImpactAbsorptionStore,
)


ROUND73_CORPUS_SCHEMA_VERSION = "round-073-segmented-corpus-v1"
ROUND73_CORPUS_CONTRACT_SHA256 = (
    "5abd0ce47a2df1d944c905111b6a821d3339a2e62ce712df7b9c9e1b8913ce67"
)
ROUND73_CORPUS_RUN_TABLE = "impact_corpus_run_manifest_v1"
ROUND73_CORPUS_MINIMUM_SEGMENT_NS = 3_600_000_000_000
ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS = 23 * 3_600_000_000_000
_MAXIMUM_WRITE_BYTES_PER_MESSAGE = 4_096.0
_MAXIMUM_PHYSICAL_GROWTH_BYTES_PER_MESSAGE = 1_024.0
_MAXIMUM_QUEUE_UTILIZATION = 0.8
_MAXIMUM_NEGATIVE_LATENCY_FRACTION = 0.001
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _strict_json_object(raw_text: str, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key is forbidden in {label}: {key}")
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validated_run_id(value: str) -> str:
    selected = str(value).strip().lower()
    if _RUN_ID.fullmatch(selected) is None:
        raise ValueError("Round 73 corpus run ID must be 32 lowercase hex characters")
    return selected


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    result = value
    if result < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return result


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
    )


@dataclass(frozen=True)
class Round73CorpusRunManifest:
    run_id: str
    capture_report_sha256: str
    feature_source_sha256: str
    last_frame_sha256: str
    coverage_start_wall_ns: int
    coverage_end_wall_ns: int
    coverage_duration_ns: int
    frame_count: int
    message_count: int
    compressed_payload_bytes: int
    manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = ROUND73_CORPUS_SCHEMA_VERSION
        payload["contract_sha256"] = ROUND73_CORPUS_CONTRACT_SHA256
        payload["symbols"] = list(IMPACT_CAPTURE_SYMBOLS)
        payload["authority"] = {
            "qualified_segment_cataloged": True,
            "complete_day_count": 0,
            "model_evaluated": False,
            "profitability_claim": False,
            "trading_authority": False,
        }
        return payload


@dataclass(frozen=True)
class Round73CorpusManifestAudit:
    run_id: str
    passed: bool
    errors: tuple[str, ...]
    manifest_sha256: str
    capture_report_sha256: str
    feature_source_sha256: str
    frame_count: int
    message_count: int

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "round-073-corpus-manifest-audit-v1"
        payload["errors"] = list(self.errors)
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


@dataclass(frozen=True)
class Round73CorpusDayCoverage:
    utc_day: str
    finalized: bool
    eligible: bool
    coverage_ns: int
    required_coverage_ns: int
    contributing_run_ids: tuple[str, ...]
    interval_count: int
    diagnostic_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "round-073-corpus-day-coverage-v1"
        payload["contributing_run_ids"] = list(self.contributing_run_ids)
        payload["crypto_formal_daily_close"] = False
        payload["listed_products_use_actual_venue_calendars"] = True
        payload["model_evaluated"] = False
        return payload


def _capture_metadata(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    run = connection.execute(
        """
        SELECT schema_version, design_sha256, capture_contract_sha256, status,
               started_wall_ns, ended_wall_ns, frame_count, message_count,
               compressed_payload_bytes, payload_cap_reached, last_frame_sha256
        FROM impact_capture_run WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run is None:
        raise ValueError("Round 73 corpus capture run was not found")
    if (
        str(run[0]) != IMPACT_CAPTURE_SCHEMA_VERSION
        or str(run[1]) != ROUND73_DESIGN_SHA256
        or str(run[2]) != IMPACT_CAPTURE_CONTRACT_SHA256
        or str(run[3]) != "completed"
    ):
        raise ValueError("Round 73 corpus capture identity is not admissible")
    if run[5] is None:
        raise ValueError("Round 73 corpus capture is not terminal")
    report_row = connection.execute(
        """
        SELECT schema_version, capture_contract_sha256, report_json, report_sha256
        FROM impact_capture_report WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if report_row is None:
        raise ValueError("Round 73 corpus capture report is missing")
    report_text = str(report_row[2])
    report_sha256 = str(report_row[3])
    if (
        str(report_row[0]) != IMPACT_CAPTURE_REPORT_SCHEMA_VERSION
        or str(report_row[1]) != IMPACT_CAPTURE_CONTRACT_SHA256
        or _SHA256.fullmatch(report_sha256) is None
        or _sha256_text(report_text) != report_sha256
    ):
        raise ValueError("Round 73 corpus capture report identity differs")
    report = dict(_strict_json_object(report_text, "capture report"))
    if (
        report.get("run_id") != run_id
        or report.get("schema_version") != IMPACT_CAPTURE_REPORT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("capture_gate_passed") is not True
        or report.get("qualification_passed") is not True
        or report.get("audit_passed") is not True
        or report.get("audit_errors") != []
        or report.get("storage_efficiency_passed") is not True
        or report.get("payload_cap_reached") is not False
        or report.get("database_size_cap_reached") is not False
    ):
        raise ValueError("Round 73 corpus capture report did not pass every gate")
    if _finite_number(report.get("elapsed_seconds"), "capture elapsed seconds") < 3_600:
        raise ValueError("Round 73 corpus capture duration is too short")
    if (
        _finite_number(
            report.get("process_io_write_bytes_per_message"),
            "capture process I/O bytes per message",
        )
        > _MAXIMUM_WRITE_BYTES_PER_MESSAGE
        or _finite_number(
            report.get("database_physical_growth_bytes_per_message"),
            "capture physical growth bytes per message",
        )
        > _MAXIMUM_PHYSICAL_GROWTH_BYTES_PER_MESSAGE
        or _finite_number(
            report.get("queue_maximum_utilization"),
            "capture queue utilization",
        )
        > _MAXIMUM_QUEUE_UTILIZATION
        or _finite_number(
            report.get("negative_corrected_latency_fraction"),
            "capture negative latency fraction",
        )
        > _MAXIMUM_NEGATIVE_LATENCY_FRACTION
    ):
        raise ValueError("Round 73 corpus capture resource gate failed")
    frame_count = _integer(run[6], "capture frame count", minimum=1)
    message_count = _integer(run[7], "capture message count", minimum=1)
    compressed_bytes = _integer(run[8], "capture compressed bytes", minimum=1)
    if (
        report.get("writer_frame_count") != frame_count
        or report.get("writer_message_count") != message_count
        or report.get("writer_compressed_payload_bytes") != compressed_bytes
        or bool(run[9])
    ):
        raise ValueError("Round 73 corpus capture totals differ")
    segments = connection.execute(
        """
        SELECT symbol, status, started_wall_ns, ended_wall_ns,
               invalid_event_count, sequence_gap_count, crossed_book_count
        FROM impact_capture_segment WHERE run_id = ? ORDER BY symbol
        """,
        [run_id],
    ).fetchall()
    if tuple(str(row[0]) for row in segments) != IMPACT_CAPTURE_SYMBOLS:
        raise ValueError("Round 73 corpus symbol segments are incomplete")
    if any(
        str(row[1]) != "valid"
        or row[3] is None
        or int(row[4]) != 0
        or int(row[5]) != 0
        or int(row[6]) != 0
        for row in segments
    ):
        raise ValueError("Round 73 corpus contains an invalid symbol segment")
    coverage_start = max(int(row[2]) for row in segments)
    coverage_end = min(int(row[3]) for row in segments)
    coverage_duration = coverage_end - coverage_start
    if coverage_duration < ROUND73_CORPUS_MINIMUM_SEGMENT_NS:
        raise ValueError("Round 73 corpus all-symbol coverage is shorter than one hour")
    metadata = {
        "coverage_start_wall_ns": coverage_start,
        "coverage_end_wall_ns": coverage_end,
        "coverage_duration_ns": coverage_duration,
        "frame_count": frame_count,
        "message_count": message_count,
        "compressed_payload_bytes": compressed_bytes,
        "last_frame_sha256": str(run[10]),
    }
    if _SHA256.fullmatch(str(metadata["last_frame_sha256"])) is None:
        raise ValueError("Round 73 corpus last frame hash is invalid")
    return metadata, report_sha256, report


def _manifest_identity(
    *,
    run_id: str,
    metadata: Mapping[str, object],
    capture_report_sha256: str,
    feature_source_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": ROUND73_CORPUS_SCHEMA_VERSION,
        "contract_sha256": ROUND73_CORPUS_CONTRACT_SHA256,
        "design_sha256": ROUND73_DESIGN_SHA256,
        "capture_contract_sha256": IMPACT_CAPTURE_CONTRACT_SHA256,
        "run_id": run_id,
        "symbols": list(IMPACT_CAPTURE_SYMBOLS),
        "capture_report_sha256": capture_report_sha256,
        "feature_source_sha256": feature_source_sha256,
        **dict(metadata),
    }


def _manifest_from_identity(
    identity: Mapping[str, object],
    manifest_sha256: str,
) -> Round73CorpusRunManifest:
    return Round73CorpusRunManifest(
        run_id=str(identity["run_id"]),
        capture_report_sha256=str(identity["capture_report_sha256"]),
        feature_source_sha256=str(identity["feature_source_sha256"]),
        last_frame_sha256=str(identity["last_frame_sha256"]),
        coverage_start_wall_ns=int(identity["coverage_start_wall_ns"]),
        coverage_end_wall_ns=int(identity["coverage_end_wall_ns"]),
        coverage_duration_ns=int(identity["coverage_duration_ns"]),
        frame_count=int(identity["frame_count"]),
        message_count=int(identity["message_count"]),
        compressed_payload_bytes=int(identity["compressed_payload_bytes"]),
        manifest_sha256=manifest_sha256,
    )


def _create_manifest_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_CORPUS_RUN_TABLE} (
            run_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            manifest_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            capture_report_sha256 VARCHAR NOT NULL,
            feature_source_json VARCHAR NOT NULL,
            feature_source_sha256 VARCHAR NOT NULL,
            coverage_start_wall_ns UBIGINT NOT NULL,
            coverage_end_wall_ns UBIGINT NOT NULL,
            coverage_duration_ns UBIGINT NOT NULL,
            frame_count UINTEGER NOT NULL,
            message_count UBIGINT NOT NULL,
            compressed_payload_bytes UBIGINT NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(manifest_sha256) = 64),
            CHECK (length(capture_report_sha256) = 64),
            CHECK (length(feature_source_sha256) = 64)
        )
        """
    )


def _stored_manifest_row(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
) -> tuple[object, ...] | None:
    if not _table_exists(connection, ROUND73_CORPUS_RUN_TABLE):
        return None
    return connection.execute(
        f"""
        SELECT manifest_json, manifest_sha256, capture_report_sha256,
               feature_source_json, feature_source_sha256
        FROM {ROUND73_CORPUS_RUN_TABLE} WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()


def _validated_manifest_row(
    row: tuple[object, ...],
    run_id: str,
) -> tuple[Mapping[str, object], str, str, str]:
    manifest_text = str(row[0])
    manifest_sha256 = str(row[1])
    capture_report_sha256 = str(row[2])
    feature_text = str(row[3])
    feature_sha256 = str(row[4])
    if (
        _SHA256.fullmatch(manifest_sha256) is None
        or _sha256_text(manifest_text) != manifest_sha256
        or _SHA256.fullmatch(capture_report_sha256) is None
        or _SHA256.fullmatch(feature_sha256) is None
        or _sha256_text(feature_text) != feature_sha256
    ):
        raise ValueError("Round 73 corpus manifest hash differs")
    identity = _strict_json_object(manifest_text, "corpus manifest")
    feature = _strict_json_object(feature_text, "feature-source diagnostic")
    feature_semantics = feature.get("feature_semantics")
    feature_authority = feature.get("authority")
    if (
        identity.get("schema_version") != ROUND73_CORPUS_SCHEMA_VERSION
        or identity.get("contract_sha256") != ROUND73_CORPUS_CONTRACT_SHA256
        or identity.get("design_sha256") != ROUND73_DESIGN_SHA256
        or identity.get("capture_contract_sha256")
        != IMPACT_CAPTURE_CONTRACT_SHA256
        or identity.get("run_id") != run_id
        or identity.get("capture_report_sha256") != capture_report_sha256
        or identity.get("feature_source_sha256") != feature_sha256
        or feature.get("schema_version") != ROUND73_FEATURE_SOURCE_SCHEMA_VERSION
        or feature.get("run_id") != run_id
        or feature.get("capture_contract_sha256")
        != IMPACT_CAPTURE_CONTRACT_SHA256
        or feature.get("stored_report_sha256") != capture_report_sha256
        or feature.get("frame_count") != identity.get("frame_count")
        or feature.get("message_count") != identity.get("message_count")
        or feature.get("capture_audit_passed") is not True
        or feature.get("stored_depth_band_rows_reconciled") is not True
        or not isinstance(feature_semantics, Mapping)
        or feature_semantics.get("future_or_target_data_used") is not False
        or feature_semantics.get("identity_whale_or_manipulation_inference")
        is not False
        or not isinstance(feature_authority, Mapping)
        or feature_authority.get("model_evaluated") is not False
        or feature_authority.get("profitability_claim") is not False
        or feature_authority.get("trading_authority") is not False
    ):
        raise ValueError("Round 73 corpus manifest identity differs")
    return identity, manifest_sha256, capture_report_sha256, feature_sha256


def index_round73_corpus_run(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73CorpusRunManifest:
    """Replay and immutably catalog one independently qualified v8 run."""

    selected = _validated_run_id(run_id)
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        existing = _stored_manifest_row(store.connect(), selected)
    if existing is not None:
        audit = audit_round73_corpus_manifest(
            database,
            run_id=selected,
            memory_limit=memory_limit,
            threads=threads,
        )
        if not audit.passed:
            raise ValueError("Round 73 existing corpus manifest audit failed")
        identity, manifest_sha256, _report_sha, _feature_sha = (
            _validated_manifest_row(existing, selected)
        )
        return _manifest_from_identity(identity, manifest_sha256)

    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        metadata, capture_report_sha256, _report = _capture_metadata(
            store.connect(), selected
        )
    diagnostic = diagnose_round73_feature_source(
        database,
        run_id=selected,
        memory_limit=memory_limit,
        threads=threads,
    )
    feature_payload = diagnostic.as_dict()
    if (
        feature_payload["capture_audit_passed"] is not True
        or feature_payload["stored_depth_band_rows_reconciled"] is not True
        or feature_payload["message_count"] != metadata["message_count"]
        or feature_payload["frame_count"] != metadata["frame_count"]
    ):
        raise ValueError("Round 73 corpus feature-source replay differs")
    feature_text = _canonical_json(feature_payload)
    feature_sha256 = _sha256_text(feature_text)
    identity = _manifest_identity(
        run_id=selected,
        metadata=metadata,
        capture_report_sha256=capture_report_sha256,
        feature_source_sha256=feature_sha256,
    )
    manifest_text = _canonical_json(identity)
    manifest_sha256 = _sha256_text(manifest_text)
    with ImpactAbsorptionStore(
        database,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        _create_manifest_table(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            concurrent = _stored_manifest_row(connection, selected)
            if concurrent is None:
                connection.execute(
                    f"INSERT INTO {ROUND73_CORPUS_RUN_TABLE} VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        selected,
                        ROUND73_CORPUS_SCHEMA_VERSION,
                        ROUND73_CORPUS_CONTRACT_SHA256,
                        manifest_text,
                        manifest_sha256,
                        capture_report_sha256,
                        feature_text,
                        feature_sha256,
                        int(metadata["coverage_start_wall_ns"]),
                        int(metadata["coverage_end_wall_ns"]),
                        int(metadata["coverage_duration_ns"]),
                        int(metadata["frame_count"]),
                        int(metadata["message_count"]),
                        int(metadata["compressed_payload_bytes"]),
                        time.time_ns(),
                    ],
                )
            else:
                stored_identity, stored_sha, _report_sha, _feature_sha = (
                    _validated_manifest_row(concurrent, selected)
                )
                if stored_sha != manifest_sha256 or stored_identity != identity:
                    raise ValueError("Round 73 concurrent corpus manifest differs")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _manifest_from_identity(identity, manifest_sha256)


def audit_round73_corpus_manifest(
    database: str | Path,
    *,
    run_id: str,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73CorpusManifestAudit:
    """Reconcile one stored manifest with its report, segments, and frame chain."""

    selected = _validated_run_id(run_id)
    errors: list[str] = []
    manifest_sha256 = ""
    capture_report_sha256 = ""
    feature_source_sha256 = ""
    frame_count = 0
    message_count = 0
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        row = _stored_manifest_row(store.connect(), selected)
        if row is None:
            raise ValueError("Round 73 corpus manifest was not found")
        try:
            identity, manifest_sha256, capture_report_sha256, feature_source_sha256 = (
                _validated_manifest_row(row, selected)
            )
            metadata, observed_report_sha256, _report = _capture_metadata(
                store.connect(), selected
            )
            expected_identity = _manifest_identity(
                run_id=selected,
                metadata=metadata,
                capture_report_sha256=observed_report_sha256,
                feature_source_sha256=feature_source_sha256,
            )
            if identity != expected_identity:
                errors.append("manifest_capture_identity_mismatch")
            capture_audit = store.audit_run(selected)
            if not capture_audit.passed:
                errors.extend(f"capture:{error}" for error in capture_audit.errors)
            frame_count = capture_audit.frame_count
            message_count = capture_audit.message_count
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"manifest:{type(exc).__name__}:{exc}")
    return Round73CorpusManifestAudit(
        run_id=selected,
        passed=not errors,
        errors=tuple(errors),
        manifest_sha256=manifest_sha256,
        capture_report_sha256=capture_report_sha256,
        feature_source_sha256=feature_source_sha256,
        frame_count=frame_count,
        message_count=message_count,
    )


def round73_corpus_day_coverage(
    database: str | Path,
    *,
    utc_day: str,
    now_wall_ns: int | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73CorpusDayCoverage:
    """Measure non-overlapping qualified coverage for one UTC statistical day."""

    try:
        selected_day = date.fromisoformat(str(utc_day))
    except ValueError as exc:
        raise ValueError("Round 73 corpus day must use YYYY-MM-DD") from exc
    day_start = datetime.combine(selected_day, datetime.min.time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    day_start_ns = int(day_start.timestamp() * 1_000_000_000)
    day_end_ns = int(day_end.timestamp() * 1_000_000_000)
    current_ns = time.time_ns() if now_wall_ns is None else int(now_wall_ns)
    intervals: list[tuple[int, int, str]] = []
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        if _table_exists(connection, ROUND73_CORPUS_RUN_TABLE):
            rows = connection.execute(
                f"""
                SELECT run_id, coverage_start_wall_ns, coverage_end_wall_ns,
                       coverage_duration_ns, manifest_json, manifest_sha256,
                       capture_report_sha256, feature_source_json,
                       feature_source_sha256
                FROM {ROUND73_CORPUS_RUN_TABLE}
                WHERE coverage_end_wall_ns > ? AND coverage_start_wall_ns < ?
                ORDER BY coverage_start_wall_ns, run_id
                """,
                [day_start_ns, day_end_ns],
            ).fetchall()
            for row in rows:
                run_id = str(row[0])
                start_ns = int(row[1])
                end_ns = int(row[2])
                duration_ns = int(row[3])
                identity, _manifest_sha, _report_sha, _feature_sha = (
                    _validated_manifest_row(
                        (row[4], row[5], row[6], row[7], row[8]),
                        run_id,
                    )
                )
                if (
                    identity.get("coverage_start_wall_ns") != start_ns
                    or identity.get("coverage_end_wall_ns") != end_ns
                    or identity.get("coverage_duration_ns") != duration_ns
                    or duration_ns != end_ns - start_ns
                ):
                    raise ValueError(
                        "Round 73 corpus day manifest columns differ from identity"
                    )
                intervals.append(
                    (
                        max(start_ns, day_start_ns),
                        min(end_ns, day_end_ns),
                        run_id,
                    )
                )
    coverage_ns = 0
    merged: list[tuple[int, int]] = []
    for start_ns, end_ns, _run_id in intervals:
        if end_ns <= start_ns:
            continue
        if not merged or start_ns > merged[-1][1]:
            merged.append((start_ns, end_ns))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ns))
    coverage_ns = sum(end_ns - start_ns for start_ns, end_ns in merged)
    finalized = current_ns >= day_end_ns
    eligible = finalized and coverage_ns >= ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS
    identity = {
        "schema_version": "round-073-corpus-day-coverage-v1",
        "contract_sha256": ROUND73_CORPUS_CONTRACT_SHA256,
        "utc_day": selected_day.isoformat(),
        "day_start_wall_ns": day_start_ns,
        "day_end_wall_ns": day_end_ns,
        "finalized": finalized,
        "eligible": eligible,
        "coverage_ns": coverage_ns,
        "required_coverage_ns": ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
        "contributing_run_ids": sorted({run_id for *_interval, run_id in intervals}),
        "interval_count": len(merged),
        "crypto_formal_daily_close": False,
    }
    return Round73CorpusDayCoverage(
        utc_day=selected_day.isoformat(),
        finalized=finalized,
        eligible=eligible,
        coverage_ns=coverage_ns,
        required_coverage_ns=ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS,
        contributing_run_ids=tuple(identity["contributing_run_ids"]),
        interval_count=len(merged),
        diagnostic_sha256=_sha256_text(_canonical_json(identity)),
    )


__all__ = [
    "ROUND73_CORPUS_CONTRACT_SHA256",
    "ROUND73_CORPUS_MINIMUM_DAY_COVERAGE_NS",
    "ROUND73_CORPUS_MINIMUM_SEGMENT_NS",
    "ROUND73_CORPUS_RUN_TABLE",
    "ROUND73_CORPUS_SCHEMA_VERSION",
    "Round73CorpusDayCoverage",
    "Round73CorpusManifestAudit",
    "Round73CorpusRunManifest",
    "audit_round73_corpus_manifest",
    "index_round73_corpus_run",
    "round73_corpus_day_coverage",
]

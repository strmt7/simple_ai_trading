"""Bounded, recoverable rotation runner for the Round 73 capture corpus."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Literal
import uuid

import duckdb

from .impact_absorption_capture import (
    IMPACT_CAPTURE_DEFAULT_DATABASE_SIZE_CAP_BYTES,
    ImpactCaptureConfig,
    ImpactCaptureSupervisorReport,
    capture_round73_supervised,
)
from .impact_absorption_corpus import (
    ROUND73_CORPUS_RUN_TABLE,
    Round73CorpusManifestAudit,
    Round73CorpusRunManifest,
    audit_round73_corpus_manifest,
    index_round73_corpus_run,
)
from .impact_absorption_store import (
    IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES,
    IMPACT_CAPTURE_V9_CONTRACT_SHA256,
    IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V9_SCHEMA_VERSION,
    ImpactAbsorptionStore,
    validate_impact_store_resources,
)


ROUND73_ROTATION_V1_SCHEMA_VERSION = "round-073-rotation-runner-v1"
ROUND73_ROTATION_V1_CONTRACT_SHA256 = (
    "8f20a25adfc2d33a43d0a8acd0ad55361956d4443e85cc7b1740f882bfc5b9ea"
)
ROUND73_ROTATION_SCHEMA_VERSION = "round-073-rotation-runner-v2"
ROUND73_ROTATION_CONTRACT_SHA256 = (
    "ab67b8678c07e2797aae6f922ef4a57f82db9124b7aadba45f1b3293815df59f"
)
ROUND73_ROTATION_LEASE_TABLE = "impact_corpus_runner_lease_v1"
ROUND73_ROTATION_BATCH_TABLE = "impact_corpus_batch_v1"
ROUND73_ROTATION_SEGMENT_TABLE = "impact_corpus_batch_segment_v1"
ROUND73_ROTATION_MAXIMUM_SEGMENTS = 168
ROUND73_ROTATION_LEASE_TTL_NS = 7_200_000_000_000
_LEASE_KEY = "round-073-corpus-collector"
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

RotationStatus = Literal["completed", "failed", "cancelled"]
CaptureFunction = Callable[
    [ImpactCaptureConfig], Awaitable[ImpactCaptureSupervisorReport]
]
IndexFunction = Callable[..., Round73CorpusRunManifest]
AuditFunction = Callable[..., Round73CorpusManifestAudit]


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


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
    )


@dataclass(frozen=True)
class Round73CorpusRotationConfig:
    database: str = "data/microstructure.duckdb"
    segment_count: int = 1
    compressed_payload_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_PAYLOAD_CAP_BYTES
    database_size_cap_bytes: int = IMPACT_CAPTURE_DEFAULT_DATABASE_SIZE_CAP_BYTES
    memory_limit: str = "2GB"
    database_threads: int = 2

    def validate(self) -> None:
        if isinstance(self.segment_count, bool) or not isinstance(
            self.segment_count, int
        ):
            raise ValueError("Round 73 rotation segment count must be an integer")
        if not 0 <= self.segment_count <= ROUND73_ROTATION_MAXIMUM_SEGMENTS:
            raise ValueError(
                "Round 73 rotation segment count must be between 0 and 168"
            )
        validate_impact_store_resources(self.memory_limit, self.database_threads)
        self.capture_config().validate()

    def capture_config(self) -> ImpactCaptureConfig:
        return ImpactCaptureConfig(
            database=self.database,
            schema_version=IMPACT_CAPTURE_V9_SCHEMA_VERSION,
            mode="qualification",
            duration_seconds=3_600.0,
            compressed_payload_cap_bytes=int(self.compressed_payload_cap_bytes),
            database_size_cap_bytes=int(self.database_size_cap_bytes),
            duckdb_memory_limit=self.memory_limit,
            duckdb_threads=int(self.database_threads),
            maximum_reconnects=0,
        )


@dataclass(frozen=True)
class Round73CorpusRotationSegment:
    ordinal: int
    source: Literal["recovery", "capture"]
    run_id: str
    capture_status: str
    qualification_passed: bool
    capture_supervisor_sha256: str
    manifest_sha256: str
    manifest_audit_passed: bool
    error: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Round73CorpusRotationReport:
    batch_id: str
    status: RotationStatus
    requested_capture_segments: int
    recovered_segment_count: int
    qualified_capture_segment_count: int
    indexed_segment_count: int
    started_wall_ns: int
    ended_wall_ns: int
    failed_phase: str
    error: str
    segments: tuple[Round73CorpusRotationSegment, ...]
    report_sha256: str = ""

    def _identity(self) -> dict[str, object]:
        return {
            "schema_version": ROUND73_ROTATION_SCHEMA_VERSION,
            "contract_sha256": ROUND73_ROTATION_CONTRACT_SHA256,
            "batch_id": self.batch_id,
            "status": self.status,
            "requested_capture_segments": self.requested_capture_segments,
            "recovered_segment_count": self.recovered_segment_count,
            "qualified_capture_segment_count": self.qualified_capture_segment_count,
            "indexed_segment_count": self.indexed_segment_count,
            "started_wall_ns": self.started_wall_ns,
            "ended_wall_ns": self.ended_wall_ns,
            "failed_phase": self.failed_phase,
            "error": self.error,
            "segments": [segment.as_dict() for segment in self.segments],
            "credentials_used": False,
            "orders_submitted": False,
            "capture_and_index_concurrent": False,
            "crypto_formal_daily_close": False,
            "authority": {
                "model_evaluated": False,
                "profitability_claim": False,
                "ai_uplift_claim": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }

    def with_hash(self) -> Round73CorpusRotationReport:
        expected = _sha256_text(_canonical_json(self._identity()))
        return replace(self, report_sha256=expected)

    def as_dict(self) -> dict[str, object]:
        payload = self._identity()
        expected = _sha256_text(_canonical_json(payload))
        if self.report_sha256 and self.report_sha256 != expected:
            raise ValueError("Round 73 rotation report hash differs")
        payload["report_sha256"] = expected
        return payload


@dataclass(frozen=True)
class Round73CorpusRotationAudit:
    batch_id: str
    passed: bool
    errors: tuple[str, ...]
    status: str
    segment_count: int
    deeply_audited_manifest_count: int
    report_sha256: str
    runner_schema_version: str
    runner_contract_sha256: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "round-073-rotation-batch-audit-v1"
        payload["audit_contract_sha256"] = ROUND73_ROTATION_CONTRACT_SHA256
        payload["errors"] = list(self.errors)
        payload["model_evaluated"] = False
        payload["profitability_claim"] = False
        payload["trading_authority"] = False
        return payload


def _initialize_rotation_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_ROTATION_LEASE_TABLE} (
            lease_key VARCHAR PRIMARY KEY,
            contract_sha256 VARCHAR NOT NULL,
            owner_id VARCHAR NOT NULL,
            acquired_wall_ns UBIGINT NOT NULL,
            renewed_wall_ns UBIGINT NOT NULL,
            expires_wall_ns UBIGINT NOT NULL,
            CHECK (length(contract_sha256) = 64),
            CHECK (length(owner_id) = 32)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_ROTATION_BATCH_TABLE} (
            batch_id VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            requested_capture_segments USMALLINT NOT NULL,
            started_wall_ns UBIGINT NOT NULL,
            updated_wall_ns UBIGINT NOT NULL,
            ended_wall_ns UBIGINT,
            report_json VARCHAR NOT NULL,
            report_sha256 VARCHAR NOT NULL,
            error VARCHAR NOT NULL,
            CHECK (length(batch_id) = 32),
            CHECK (length(contract_sha256) = 64)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ROUND73_ROTATION_SEGMENT_TABLE} (
            batch_id VARCHAR NOT NULL,
            ordinal USMALLINT NOT NULL,
            source VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            capture_status VARCHAR NOT NULL,
            qualification_passed BOOLEAN NOT NULL,
            capture_supervisor_json VARCHAR NOT NULL,
            capture_supervisor_sha256 VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL,
            manifest_audit_passed BOOLEAN NOT NULL,
            error VARCHAR NOT NULL,
            recorded_at_wall_ns UBIGINT NOT NULL,
            PRIMARY KEY (batch_id, ordinal),
            CHECK (length(batch_id) = 32)
        )
        """
    )


def _with_store(
    config: Round73CorpusRotationConfig,
) -> ImpactAbsorptionStore:
    return ImpactAbsorptionStore(
        config.database,
        memory_limit=config.memory_limit,
        threads=config.database_threads,
    )


def _acquire_lease(
    config: Round73CorpusRotationConfig,
    *,
    owner_id: str,
    now_wall_ns: int,
) -> None:
    with _with_store(config) as store:
        connection = store.connect()
        _initialize_rotation_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            lease = connection.execute(
                f"SELECT owner_id, expires_wall_ns FROM {ROUND73_ROTATION_LEASE_TABLE} "
                "WHERE lease_key = ?",
                [_LEASE_KEY],
            ).fetchone()
            if lease is not None and int(lease[1]) > now_wall_ns:
                raise RuntimeError(
                    "another Round 73 corpus collector owns the active lease"
                )
            connection.execute(
                f"DELETE FROM {ROUND73_ROTATION_LEASE_TABLE} WHERE lease_key = ?",
                [_LEASE_KEY],
            )
            connection.execute(
                f"INSERT INTO {ROUND73_ROTATION_LEASE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                [
                    _LEASE_KEY,
                    ROUND73_ROTATION_CONTRACT_SHA256,
                    owner_id,
                    now_wall_ns,
                    now_wall_ns,
                    now_wall_ns + ROUND73_ROTATION_LEASE_TTL_NS,
                ],
            )
            connection.execute(
                f"""
                UPDATE {ROUND73_ROTATION_BATCH_TABLE}
                SET status = 'interrupted', updated_wall_ns = ?, ended_wall_ns = ?,
                    error = 'runner lease ended before terminal report'
                WHERE status IN ('recovering', 'capturing', 'indexing')
                """,
                [now_wall_ns, now_wall_ns],
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise


def _renew_lease(
    config: Round73CorpusRotationConfig,
    *,
    owner_id: str,
    now_wall_ns: int,
) -> None:
    with _with_store(config) as store:
        connection = store.connect()
        row = connection.execute(
            f"SELECT owner_id FROM {ROUND73_ROTATION_LEASE_TABLE} WHERE lease_key = ?",
            [_LEASE_KEY],
        ).fetchone()
        if row is None or str(row[0]) != owner_id:
            raise RuntimeError("Round 73 corpus collector lease was lost")
        connection.execute(
            f"""
            UPDATE {ROUND73_ROTATION_LEASE_TABLE}
            SET renewed_wall_ns = ?, expires_wall_ns = ?
            WHERE lease_key = ? AND owner_id = ?
            """,
            [
                now_wall_ns,
                now_wall_ns + ROUND73_ROTATION_LEASE_TTL_NS,
                _LEASE_KEY,
                owner_id,
            ],
        )


def _release_lease(config: Round73CorpusRotationConfig, *, owner_id: str) -> None:
    with _with_store(config) as store:
        store.connect().execute(
            f"DELETE FROM {ROUND73_ROTATION_LEASE_TABLE} "
            "WHERE lease_key = ? AND owner_id = ?",
            [_LEASE_KEY, owner_id],
        )


def _start_batch(
    config: Round73CorpusRotationConfig,
    *,
    batch_id: str,
    started_wall_ns: int,
) -> None:
    with _with_store(config) as store:
        connection = store.connect()
        _initialize_rotation_tables(connection)
        connection.execute(
            f"INSERT INTO {ROUND73_ROTATION_BATCH_TABLE} VALUES "
            "(?, ?, ?, 'recovering', ?, ?, ?, NULL, '', '', '')",
            [
                batch_id,
                ROUND73_ROTATION_SCHEMA_VERSION,
                ROUND73_ROTATION_CONTRACT_SHA256,
                config.segment_count,
                started_wall_ns,
                started_wall_ns,
            ],
        )


def _set_batch_phase(
    config: Round73CorpusRotationConfig,
    *,
    batch_id: str,
    phase: str,
) -> None:
    if phase not in {"recovering", "capturing", "indexing"}:
        raise ValueError("Round 73 batch phase is invalid")
    with _with_store(config) as store:
        connection = store.connect()
        current = connection.execute(
            f"SELECT status FROM {ROUND73_ROTATION_BATCH_TABLE} WHERE batch_id = ?",
            [batch_id],
        ).fetchone()
        if current is None or str(current[0]) not in {
            "recovering",
            "capturing",
            "indexing",
        }:
            raise RuntimeError("Round 73 batch is not mutable")
        connection.execute(
            f"UPDATE {ROUND73_ROTATION_BATCH_TABLE} "
            "SET status = ?, updated_wall_ns = ? WHERE batch_id = ?",
            [phase, time.time_ns(), batch_id],
        )


def _record_segment(
    config: Round73CorpusRotationConfig,
    *,
    batch_id: str,
    segment: Round73CorpusRotationSegment,
    supervisor_text: str = "",
) -> None:
    with _with_store(config) as store:
        store.connect().execute(
            f"INSERT INTO {ROUND73_ROTATION_SEGMENT_TABLE} VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                batch_id,
                segment.ordinal,
                segment.source,
                segment.run_id,
                segment.capture_status,
                segment.qualification_passed,
                supervisor_text,
                segment.capture_supervisor_sha256,
                segment.manifest_sha256,
                segment.manifest_audit_passed,
                segment.error,
                time.time_ns(),
            ],
        )


def _record_manifest_result(
    config: Round73CorpusRotationConfig,
    *,
    batch_id: str,
    segment: Round73CorpusRotationSegment,
) -> None:
    with _with_store(config) as store:
        connection = store.connect()
        current = connection.execute(
            f"SELECT run_id, manifest_sha256 FROM {ROUND73_ROTATION_SEGMENT_TABLE} "
            "WHERE batch_id = ? AND ordinal = ?",
            [batch_id, segment.ordinal],
        ).fetchone()
        if current is None or str(current[0]) != segment.run_id:
            raise RuntimeError("Round 73 journal segment identity differs")
        if str(current[1]):
            if str(current[1]) != segment.manifest_sha256:
                raise RuntimeError("Round 73 journal manifest is immutable")
            return
        connection.execute(
            f"""
            UPDATE {ROUND73_ROTATION_SEGMENT_TABLE}
            SET manifest_sha256 = ?, manifest_audit_passed = ?, error = ?,
                recorded_at_wall_ns = ?
            WHERE batch_id = ? AND ordinal = ?
            """,
            [
                segment.manifest_sha256,
                segment.manifest_audit_passed,
                segment.error,
                time.time_ns(),
                batch_id,
                segment.ordinal,
            ],
        )


def _finalize_batch(
    config: Round73CorpusRotationConfig,
    report: Round73CorpusRotationReport,
) -> Round73CorpusRotationReport:
    sealed = report.with_hash()
    report_text = _canonical_json(sealed._identity())
    with _with_store(config) as store:
        connection = store.connect()
        current = connection.execute(
            f"SELECT status, report_sha256 FROM {ROUND73_ROTATION_BATCH_TABLE} "
            "WHERE batch_id = ?",
            [sealed.batch_id],
        ).fetchone()
        if current is None:
            raise RuntimeError("Round 73 batch journal is missing")
        if str(current[0]) in {"completed", "failed", "cancelled"}:
            if str(current[1]) != sealed.report_sha256:
                raise RuntimeError("Round 73 terminal batch report is immutable")
            return sealed
        connection.execute(
            f"""
            UPDATE {ROUND73_ROTATION_BATCH_TABLE}
            SET status = ?, updated_wall_ns = ?, ended_wall_ns = ?,
                report_json = ?, report_sha256 = ?, error = ?
            WHERE batch_id = ?
            """,
            [
                sealed.status,
                sealed.ended_wall_ns,
                sealed.ended_wall_ns,
                report_text,
                sealed.report_sha256,
                sealed.error,
                sealed.batch_id,
            ],
        )
    return sealed


def qualified_unindexed_round73_runs(
    database: str | Path,
    *,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> tuple[str, ...]:
    """Find current-schema qualified runs that still require exact admission."""

    candidates: list[str] = []
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        manifest_exists = _table_exists(connection, ROUND73_CORPUS_RUN_TABLE)
        manifest_join = (
            f"LEFT JOIN {ROUND73_CORPUS_RUN_TABLE} m ON m.run_id = r.run_id"
            if manifest_exists
            else ""
        )
        manifest_filter = "AND m.run_id IS NULL" if manifest_exists else ""
        rows = connection.execute(
            f"""
            SELECT r.run_id, p.report_json, p.report_sha256
            FROM impact_capture_run r
            JOIN impact_capture_report p ON p.run_id = r.run_id
            {manifest_join}
            WHERE r.schema_version = ? AND r.capture_contract_sha256 = ?
              AND r.status = 'completed' AND p.schema_version = ?
              AND p.capture_contract_sha256 = ? {manifest_filter}
            ORDER BY r.started_wall_ns, r.run_id
            """,
            [
                IMPACT_CAPTURE_V9_SCHEMA_VERSION,
                IMPACT_CAPTURE_V9_CONTRACT_SHA256,
                IMPACT_CAPTURE_V9_REPORT_SCHEMA_VERSION,
                IMPACT_CAPTURE_V9_CONTRACT_SHA256,
            ],
        ).fetchall()
        for run_id, report_text, report_sha256 in rows:
            selected = str(run_id)
            text = str(report_text)
            claimed = str(report_sha256)
            if (
                _RUN_ID.fullmatch(selected) is None
                or _SHA256.fullmatch(claimed) is None
                or _sha256_text(text) != claimed
            ):
                raise ValueError("Round 73 pending capture report hash differs")
            report = _strict_json_object(text, "pending capture report")
            if report.get("run_id") != selected:
                raise ValueError("Round 73 pending capture report identity differs")
            if report.get("qualification_passed") is True:
                candidates.append(selected)
    return tuple(candidates)


def audit_round73_rotation_batch(
    database: str | Path,
    *,
    batch_id: str,
    deep_manifest_audit: bool = False,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> Round73CorpusRotationAudit:
    """Reconcile one terminal runner journal, optionally re-auditing raw captures."""

    selected = str(batch_id).strip().lower()
    if _RUN_ID.fullmatch(selected) is None:
        raise ValueError("Round 73 batch ID must be 32 lowercase hex characters")
    errors: list[str] = []
    status = ""
    report_sha256 = ""
    runner_schema_version = ""
    runner_contract_sha256 = ""
    segment_count = 0
    deep_candidates: list[tuple[str, str]] = []
    with ImpactAbsorptionStore(
        database,
        read_only=True,
        memory_limit=memory_limit,
        threads=threads,
    ) as store:
        connection = store.connect()
        if not _table_exists(connection, ROUND73_ROTATION_BATCH_TABLE):
            raise ValueError("Round 73 rotation batch table was not found")
        row = connection.execute(
            f"""
            SELECT schema_version, contract_sha256, status,
                   requested_capture_segments, started_wall_ns, ended_wall_ns,
                   report_json, report_sha256, error
            FROM {ROUND73_ROTATION_BATCH_TABLE} WHERE batch_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise ValueError("Round 73 rotation batch was not found")
        status = str(row[2])
        runner_schema_version = str(row[0])
        runner_contract_sha256 = str(row[1])
        report_text = str(row[6])
        report_sha256 = str(row[7])
        try:
            supported_protocols = {
                ROUND73_ROTATION_V1_SCHEMA_VERSION: (
                    ROUND73_ROTATION_V1_CONTRACT_SHA256
                ),
                ROUND73_ROTATION_SCHEMA_VERSION: ROUND73_ROTATION_CONTRACT_SHA256,
            }
            expected_contract = supported_protocols.get(runner_schema_version)
            if status not in {"completed", "failed", "cancelled"}:
                raise ValueError("rotation batch is not terminal")
            if (
                expected_contract is None
                or runner_contract_sha256 != expected_contract
                or _SHA256.fullmatch(report_sha256) is None
                or _sha256_text(report_text) != report_sha256
            ):
                raise ValueError("rotation batch report identity differs")
            report = _strict_json_object(report_text, "rotation batch report")
            authority = report.get("authority")
            if (
                report.get("schema_version") != runner_schema_version
                or report.get("contract_sha256") != runner_contract_sha256
                or report.get("batch_id") != selected
                or report.get("status") != status
                or report.get("requested_capture_segments") != int(row[3])
                or report.get("started_wall_ns") != int(row[4])
                or report.get("ended_wall_ns") != int(row[5])
                or report.get("error") != str(row[8])
                or report.get("credentials_used") is not False
                or report.get("orders_submitted") is not False
                or report.get("capture_and_index_concurrent") is not False
                or report.get("crypto_formal_daily_close") is not False
                or not isinstance(authority, Mapping)
                or authority.get("model_evaluated") is not False
                or authority.get("profitability_claim") is not False
                or authority.get("live_trading_authority") is not False
            ):
                raise ValueError("rotation batch report fields differ")
            expected_segments = report.get("segments")
            if not isinstance(expected_segments, list):
                raise ValueError("rotation batch segments are missing")
            stored_segments = connection.execute(
                f"""
                SELECT ordinal, source, run_id, capture_status,
                       qualification_passed, capture_supervisor_json,
                       capture_supervisor_sha256, manifest_sha256,
                       manifest_audit_passed, error
                FROM {ROUND73_ROTATION_SEGMENT_TABLE}
                WHERE batch_id = ? ORDER BY ordinal
                """,
                [selected],
            ).fetchall()
            segment_count = len(stored_segments)
            observed_segments: list[dict[str, object]] = []
            for segment_row in stored_segments:
                ordinal = int(segment_row[0])
                source = str(segment_row[1])
                run_id = str(segment_row[2])
                supervisor_text = str(segment_row[5])
                supervisor_sha256 = str(segment_row[6])
                manifest_sha256 = str(segment_row[7])
                manifest_audit_passed = bool(segment_row[8])
                if source not in {"recovery", "capture"}:
                    raise ValueError(f"segment {ordinal} source is invalid")
                if run_id and _RUN_ID.fullmatch(run_id) is None:
                    raise ValueError(f"segment {ordinal} run ID is invalid")
                if source == "recovery":
                    if supervisor_text or supervisor_sha256:
                        raise ValueError(
                            f"segment {ordinal} recovery supervisor must be empty"
                        )
                elif (
                    _SHA256.fullmatch(supervisor_sha256) is None
                    or _sha256_text(supervisor_text) != supervisor_sha256
                ):
                    raise ValueError(f"segment {ordinal} supervisor hash differs")
                if manifest_audit_passed:
                    if _SHA256.fullmatch(manifest_sha256) is None or not run_id:
                        raise ValueError(f"segment {ordinal} manifest identity differs")
                    deep_candidates.append((run_id, manifest_sha256))
                elif manifest_sha256:
                    raise ValueError(
                        f"segment {ordinal} unaudited manifest hash is populated"
                    )
                observed_segments.append(
                    Round73CorpusRotationSegment(
                        ordinal=ordinal,
                        source=source,
                        run_id=run_id,
                        capture_status=str(segment_row[3]),
                        qualification_passed=bool(segment_row[4]),
                        capture_supervisor_sha256=supervisor_sha256,
                        manifest_sha256=manifest_sha256,
                        manifest_audit_passed=manifest_audit_passed,
                        error=str(segment_row[9]),
                    ).as_dict()
                )
            if observed_segments != expected_segments:
                raise ValueError("rotation batch segment journal differs")
            recovered_count = sum(
                item["source"] == "recovery" and item["manifest_audit_passed"]
                for item in observed_segments
            )
            qualified_count = sum(
                item["source"] == "capture"
                and item["qualification_passed"]
                and bool(item["run_id"])
                for item in observed_segments
            )
            indexed_count = sum(
                bool(item["manifest_audit_passed"]) for item in observed_segments
            )
            if (
                report.get("recovered_segment_count") != recovered_count
                or report.get("qualified_capture_segment_count") != qualified_count
                or report.get("indexed_segment_count") != indexed_count
            ):
                raise ValueError("rotation batch aggregate counts differ")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"journal:{type(exc).__name__}:{exc}")

    deeply_audited = 0
    if deep_manifest_audit and not errors:
        for run_id, expected_manifest_sha256 in deep_candidates:
            try:
                audit = audit_round73_corpus_manifest(
                    database,
                    run_id=run_id,
                    memory_limit=memory_limit,
                    threads=threads,
                )
            except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"manifest:{run_id}:{type(exc).__name__}:{exc}")
                break
            if not audit.passed or audit.manifest_sha256 != expected_manifest_sha256:
                errors.append(f"manifest:{run_id}:audit_or_hash_mismatch")
                break
            deeply_audited += 1
    return Round73CorpusRotationAudit(
        batch_id=selected,
        passed=not errors,
        errors=tuple(errors),
        status=status,
        segment_count=segment_count,
        deeply_audited_manifest_count=deeply_audited,
        report_sha256=report_sha256,
        runner_schema_version=runner_schema_version,
        runner_contract_sha256=runner_contract_sha256,
    )


def _qualified_supervisor_run(
    supervisor: ImpactCaptureSupervisorReport,
) -> tuple[str, str, str]:
    payload = supervisor.as_dict()
    supervisor_text = _canonical_json(payload)
    supervisor_sha256 = _sha256_text(supervisor_text)
    if (
        supervisor.status != "completed"
        or supervisor.capture_schema_version != IMPACT_CAPTURE_V9_SCHEMA_VERSION
        or supervisor.qualification_passed is not True
        or supervisor.attempt_count != 1
        or supervisor.reconnect_count != 0
        or len(supervisor.attempts) != 1
        or supervisor.attempts[0].status != "completed"
        or supervisor.attempts[0].qualification_passed is not True
        or supervisor.selected_run_id != supervisor.attempts[0].run_id
        or _RUN_ID.fullmatch(supervisor.selected_run_id) is None
    ):
        return "", supervisor_text, supervisor_sha256
    return supervisor.selected_run_id, supervisor_text, supervisor_sha256


def _index_one(
    config: Round73CorpusRotationConfig,
    *,
    run_id: str,
    index_function: IndexFunction,
    audit_function: AuditFunction,
) -> tuple[str, bool]:
    manifest = index_function(
        config.database,
        run_id=run_id,
        memory_limit=config.memory_limit,
        threads=config.database_threads,
    )
    audit = audit_function(
        config.database,
        run_id=run_id,
        memory_limit=config.memory_limit,
        threads=config.database_threads,
    )
    if manifest.run_id != run_id or audit.run_id != run_id or not audit.passed:
        raise ValueError("Round 73 post-index manifest audit failed")
    if manifest.manifest_sha256 != audit.manifest_sha256:
        raise ValueError("Round 73 manifest and audit hashes differ")
    return manifest.manifest_sha256, True


async def run_round73_corpus_rotation(
    config: Round73CorpusRotationConfig,
    *,
    capture_function: CaptureFunction = capture_round73_supervised,
    index_function: IndexFunction = index_round73_corpus_run,
    audit_function: AuditFunction = audit_round73_corpus_manifest,
) -> Round73CorpusRotationReport:
    """Recover pending evidence, collect a bounded batch, then index serially."""

    config.validate()
    batch_id = uuid.uuid4().hex
    started_wall_ns = time.time_ns()
    _acquire_lease(config, owner_id=batch_id, now_wall_ns=started_wall_ns)
    segments: list[Round73CorpusRotationSegment] = []
    qualified_capture_run_ids: list[str] = []
    recovered_count = 0
    failed_phase = ""
    error = ""
    ordinal = 0

    try:
        _start_batch(config, batch_id=batch_id, started_wall_ns=started_wall_ns)
        pending = qualified_unindexed_round73_runs(
            config.database,
            memory_limit=config.memory_limit,
            threads=config.database_threads,
        )
        for run_id in pending:
            ordinal += 1
            _renew_lease(config, owner_id=batch_id, now_wall_ns=time.time_ns())
            segment = Round73CorpusRotationSegment(
                ordinal=ordinal,
                source="recovery",
                run_id=run_id,
                capture_status="previously_qualified",
                qualification_passed=True,
                capture_supervisor_sha256="",
                manifest_sha256="",
                manifest_audit_passed=False,
                error="",
            )
            _record_segment(config, batch_id=batch_id, segment=segment)
            try:
                manifest_sha256, audit_passed = await asyncio.to_thread(
                    _index_one,
                    config,
                    run_id=run_id,
                    index_function=index_function,
                    audit_function=audit_function,
                )
            except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
                segment = replace(
                    segment,
                    error=f"{type(exc).__name__}:{exc}"[:2_000],
                )
                _record_manifest_result(config, batch_id=batch_id, segment=segment)
                segments.append(segment)
                failed_phase = "recovery"
                error = segment.error
                break
            segment = replace(
                segment,
                manifest_sha256=manifest_sha256,
                manifest_audit_passed=audit_passed,
            )
            _record_manifest_result(config, batch_id=batch_id, segment=segment)
            segments.append(segment)
            recovered_count += 1

        if not error:
            _set_batch_phase(config, batch_id=batch_id, phase="capturing")
            for _capture_index in range(config.segment_count):
                _renew_lease(config, owner_id=batch_id, now_wall_ns=time.time_ns())
                try:
                    supervisor = await capture_function(config.capture_config())
                except (OSError, RuntimeError, ValueError) as exc:
                    failed_phase = "capture"
                    error = f"{type(exc).__name__}:{exc}"[:2_000]
                    break
                run_id, supervisor_text, supervisor_sha256 = _qualified_supervisor_run(
                    supervisor
                )
                ordinal += 1
                segment = Round73CorpusRotationSegment(
                    ordinal=ordinal,
                    source="capture",
                    run_id=run_id,
                    capture_status=supervisor.status,
                    qualification_passed=bool(supervisor.qualification_passed),
                    capture_supervisor_sha256=supervisor_sha256,
                    manifest_sha256="",
                    manifest_audit_passed=False,
                    error="",
                )
                if not run_id:
                    segment = replace(
                        segment,
                        error=(
                            supervisor.terminal_error
                            or "capture supervisor did not return one reconnect-free qualified run"
                        )[:2_000],
                    )
                _record_segment(
                    config,
                    batch_id=batch_id,
                    segment=segment,
                    supervisor_text=supervisor_text,
                )
                segments.append(segment)
                if not run_id:
                    failed_phase = "capture"
                    error = segment.error
                    break
                qualified_capture_run_ids.append(run_id)

        if qualified_capture_run_ids:
            _set_batch_phase(config, batch_id=batch_id, phase="indexing")
        for run_id in qualified_capture_run_ids:
            _renew_lease(config, owner_id=batch_id, now_wall_ns=time.time_ns())
            segment_index = next(
                index
                for index, item in enumerate(segments)
                if item.source == "capture" and item.run_id == run_id
            )
            segment = segments[segment_index]
            try:
                manifest_sha256, audit_passed = await asyncio.to_thread(
                    _index_one,
                    config,
                    run_id=run_id,
                    index_function=index_function,
                    audit_function=audit_function,
                )
            except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
                segment = replace(
                    segment,
                    error=f"{type(exc).__name__}:{exc}"[:2_000],
                )
                _record_manifest_result(config, batch_id=batch_id, segment=segment)
                segments[segment_index] = segment
                if not error:
                    failed_phase = "index"
                    error = segment.error
                break
            segment = replace(
                segment,
                manifest_sha256=manifest_sha256,
                manifest_audit_passed=audit_passed,
            )
            _record_manifest_result(config, batch_id=batch_id, segment=segment)
            segments[segment_index] = segment

        indexed_count = sum(item.manifest_audit_passed for item in segments)
        qualified_capture_count = len(qualified_capture_run_ids)
        status: RotationStatus = (
            "completed"
            if not error
            and qualified_capture_count == config.segment_count
            and indexed_count == len(segments)
            else "failed"
        )
        report = Round73CorpusRotationReport(
            batch_id=batch_id,
            status=status,
            requested_capture_segments=config.segment_count,
            recovered_segment_count=recovered_count,
            qualified_capture_segment_count=qualified_capture_count,
            indexed_segment_count=indexed_count,
            started_wall_ns=started_wall_ns,
            ended_wall_ns=time.time_ns(),
            failed_phase=failed_phase,
            error=error,
            segments=tuple(segments),
        )
        return _finalize_batch(config, report)
    except asyncio.CancelledError:
        report = Round73CorpusRotationReport(
            batch_id=batch_id,
            status="cancelled",
            requested_capture_segments=config.segment_count,
            recovered_segment_count=recovered_count,
            qualified_capture_segment_count=len(qualified_capture_run_ids),
            indexed_segment_count=sum(item.manifest_audit_passed for item in segments),
            started_wall_ns=started_wall_ns,
            ended_wall_ns=time.time_ns(),
            failed_phase="cancelled",
            error="operator cancellation",
            segments=tuple(segments),
        )
        _finalize_batch(config, report)
        raise
    except Exception as exc:
        error = error or f"{type(exc).__name__}:{exc}"[:2_000]
        report = Round73CorpusRotationReport(
            batch_id=batch_id,
            status="failed",
            requested_capture_segments=config.segment_count,
            recovered_segment_count=recovered_count,
            qualified_capture_segment_count=len(qualified_capture_run_ids),
            indexed_segment_count=sum(item.manifest_audit_passed for item in segments),
            started_wall_ns=started_wall_ns,
            ended_wall_ns=time.time_ns(),
            failed_phase=failed_phase or "runner",
            error=error,
            segments=tuple(segments),
        )
        try:
            return _finalize_batch(config, report)
        except Exception:
            raise exc
    finally:
        _release_lease(config, owner_id=batch_id)


__all__ = [
    "ROUND73_ROTATION_BATCH_TABLE",
    "ROUND73_ROTATION_CONTRACT_SHA256",
    "ROUND73_ROTATION_LEASE_TABLE",
    "ROUND73_ROTATION_MAXIMUM_SEGMENTS",
    "ROUND73_ROTATION_SCHEMA_VERSION",
    "ROUND73_ROTATION_SEGMENT_TABLE",
    "ROUND73_ROTATION_V1_CONTRACT_SHA256",
    "ROUND73_ROTATION_V1_SCHEMA_VERSION",
    "Round73CorpusRotationConfig",
    "Round73CorpusRotationAudit",
    "Round73CorpusRotationReport",
    "Round73CorpusRotationSegment",
    "audit_round73_rotation_batch",
    "qualified_unindexed_round73_runs",
    "run_round73_corpus_rotation",
]

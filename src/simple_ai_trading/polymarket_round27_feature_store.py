"""Compact, target-free campaign feature storage for Polymarket Round 27."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import zstandard

from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
    Round27FeatureRow,
)


POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION = (
    "polymarket-round27-feature-store-v1"
)
POLYMARKET_ROUND27_FEATURE_CHUNK_SCHEMA_VERSION = (
    "polymarket-round27-condition-feature-chunk-v1"
)
POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC = "canonical-json-zstd-3"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAXIMUM_CHUNK_RAW_BYTES = 128 * 1024 * 1024
_MAXIMUM_METADATA_RAW_BYTES = 64 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 27 {name} SHA-256 differs")
    return selected


def _validate_hashed_payload(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop(hash_field, ""), name=name)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"Round 27 {name} hash differs")
    return {**payload, hash_field: claimed}


def _row_payload(row: Round27FeatureRow) -> dict[str, object]:
    selected = row.validated()
    return {
        "schema_version": selected.schema_version,
        "run_id": selected.run_id,
        "condition_id": selected.condition_id,
        "event_start_ms": selected.event_start_ms,
        "decision_time_ms": selected.decision_time_ms,
        "market_prior_probability": selected.market_prior_probability,
        "values": list(selected.values),
        "feature_names_sha256": selected.feature_names_sha256,
        "maximum_receipt_wall_ms": selected.maximum_receipt_wall_ms,
        "source_chain_sha256": selected.source_chain_sha256,
        "row_sha256": selected.row_sha256,
        "target_accessed": False,
        "trading_authority": False,
    }


def _row_from_payload(value: Mapping[str, object]) -> Round27FeatureRow:
    payload = dict(value)
    raw_values = payload.get("values")
    if not isinstance(raw_values, list):
        raise ValueError("Round 27 stored feature values differ")
    payload["values"] = tuple(float(item) for item in raw_values)
    try:
        return Round27FeatureRow(**payload).validated()  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 27 stored feature row differs") from exc


def _row_chain(rows: Sequence[Round27FeatureRow]) -> str:
    chain = _EMPTY_SHA256
    for row in rows:
        chain = hashlib.sha256(
            bytes.fromhex(chain) + bytes.fromhex(row.row_sha256)
        ).hexdigest()
    return chain


@dataclass(frozen=True, slots=True)
class _CompressedPayload:
    raw_size_bytes: int
    compressed_size_bytes: int
    raw_sha256: str
    compressed_sha256: str
    payload: bytes


def _compress(value: object, *, maximum_raw_bytes: int) -> _CompressedPayload:
    raw = _canonical_json(value).encode("ascii")
    if not 2 <= len(raw) <= maximum_raw_bytes:
        raise ValueError("Round 27 feature-store payload size differs")
    compressed = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)
    return _CompressedPayload(
        raw_size_bytes=len(raw),
        compressed_size_bytes=len(compressed),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        payload=compressed,
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 feature-store JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 27 feature-store JSON contains {value}")


def _decompress(
    compressed: bytes,
    *,
    expected_raw_size: int,
    expected_compressed_size: int,
    expected_raw_sha256: str,
    expected_compressed_sha256: str,
    maximum_raw_bytes: int,
) -> object:
    if (
        len(compressed) != expected_compressed_size
        or hashlib.sha256(compressed).hexdigest() != expected_compressed_sha256
        or not 2 <= expected_raw_size <= maximum_raw_bytes
    ):
        raise ValueError("Round 27 compressed feature payload differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=expected_raw_size,
        )
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, zstandard.ZstdError) as exc:
        raise ValueError("Round 27 feature payload does not decode") from exc
    if (
        len(raw) != expected_raw_size
        or hashlib.sha256(raw).hexdigest() != expected_raw_sha256
        or raw != _canonical_json(value).encode("ascii")
    ):
        raise ValueError("Round 27 raw feature payload differs")
    return value


@dataclass(frozen=True, slots=True)
class _ConditionChunk:
    condition_id: str
    event_start_ms: int
    row_count: int
    row_chain_sha256: str
    compressed: _CompressedPayload
    manifest_sha256: str


def _condition_chunk(
    *,
    slot_id: str,
    run_id: str,
    rows: Sequence[Round27FeatureRow],
) -> _ConditionChunk:
    selected = tuple(row.validated() for row in rows)
    if (
        not selected
        or len({row.condition_id for row in selected}) != 1
        or len({row.event_start_ms for row in selected}) != 1
        or any(row.run_id != run_id for row in selected)
        or tuple(row.decision_time_ms for row in selected)
        != tuple(sorted(row.decision_time_ms for row in selected))
    ):
        raise ValueError("Round 27 condition feature chunk differs")
    condition_id = selected[0].condition_id
    event_start_ms = selected[0].event_start_ms
    chain = _row_chain(selected)
    body = {
        "schema_version": POLYMARKET_ROUND27_FEATURE_CHUNK_SCHEMA_VERSION,
        "slot_id": slot_id,
        "run_id": run_id,
        "condition_id": condition_id,
        "event_start_ms": event_start_ms,
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "row_count": len(selected),
        "row_chain_sha256": chain,
        "target_accessed": False,
        "trading_authority": False,
        "rows": [_row_payload(row) for row in selected],
    }
    compressed = _compress(body, maximum_raw_bytes=_MAXIMUM_CHUNK_RAW_BYTES)
    identity = {
        "schema_version": POLYMARKET_ROUND27_FEATURE_CHUNK_SCHEMA_VERSION,
        "slot_id": slot_id,
        "run_id": run_id,
        "condition_id": condition_id,
        "event_start_ms": event_start_ms,
        "row_count": len(selected),
        "row_chain_sha256": chain,
        "codec": POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC,
        "raw_size_bytes": compressed.raw_size_bytes,
        "compressed_size_bytes": compressed.compressed_size_bytes,
        "raw_sha256": compressed.raw_sha256,
        "compressed_sha256": compressed.compressed_sha256,
    }
    return _ConditionChunk(
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        row_count=len(selected),
        row_chain_sha256=chain,
        compressed=compressed,
        manifest_sha256=_canonical_sha256(identity),
    )


class Round27FeatureStore:
    """Single-DuckDB, condition-chunked target-free feature store."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        if read_only and not self.path.is_file():
            raise ValueError("Round 27 feature store is unavailable")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = bool(read_only)
        self.connection = duckdb.connect(str(self.path), read_only=self.read_only)
        self.connection.execute("SET threads = 2")
        self.connection.execute("SET memory_limit = '1GB'")
        self.connection.execute("SET preserve_insertion_order = false")
        if not self.read_only:
            self._initialize()
        self._verify_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Round27FeatureStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS round27_feature_store_schema (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                feature_schema_version VARCHAR NOT NULL,
                feature_names_sha256 VARCHAR NOT NULL,
                feature_count INTEGER NOT NULL,
                codec VARCHAR NOT NULL,
                target_columns_present BOOLEAN NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round27_feature_slot (
                slot_id VARCHAR PRIMARY KEY,
                run_id VARCHAR UNIQUE NOT NULL,
                condition_audit_sha256 VARCHAR NOT NULL,
                feature_report_sha256 VARCHAR NOT NULL,
                condition_count INTEGER NOT NULL,
                row_count BIGINT NOT NULL,
                row_chain_sha256 VARCHAR NOT NULL,
                audit_raw_size_bytes BIGINT NOT NULL,
                audit_compressed_size_bytes BIGINT NOT NULL,
                audit_raw_sha256 VARCHAR NOT NULL,
                audit_compressed_sha256 VARCHAR NOT NULL,
                audit_payload BLOB NOT NULL,
                report_raw_size_bytes BIGINT NOT NULL,
                report_compressed_size_bytes BIGINT NOT NULL,
                report_raw_sha256 VARCHAR NOT NULL,
                report_compressed_sha256 VARCHAR NOT NULL,
                report_payload BLOB NOT NULL,
                manifest_sha256 VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round27_feature_condition (
                condition_id VARCHAR PRIMARY KEY,
                slot_id VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                event_start_ms BIGINT NOT NULL,
                row_count INTEGER NOT NULL,
                row_chain_sha256 VARCHAR NOT NULL,
                codec VARCHAR NOT NULL,
                raw_size_bytes BIGINT NOT NULL,
                compressed_size_bytes BIGINT NOT NULL,
                raw_sha256 VARCHAR NOT NULL,
                compressed_sha256 VARCHAR NOT NULL,
                payload BLOB NOT NULL,
                manifest_sha256 VARCHAR NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            INSERT INTO round27_feature_store_schema
            SELECT true, ?, ?, ?, ?, ?, false
            WHERE NOT EXISTS (SELECT 1 FROM round27_feature_store_schema)
            """,
            [
                POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
                POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
                POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
                len(POLYMARKET_ROUND27_FEATURE_NAMES),
                POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC,
            ],
        )

    def _verify_schema(self) -> None:
        row = self.connection.execute(
            """
            SELECT schema_version, feature_schema_version, feature_names_sha256,
                   feature_count, codec, target_columns_present
            FROM round27_feature_store_schema WHERE singleton
            """
        ).fetchone()
        if row != (
            POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
            POLYMARKET_ROUND27_FEATURE_SCHEMA_VERSION,
            POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            len(POLYMARKET_ROUND27_FEATURE_NAMES),
            POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC,
            False,
        ):
            raise ValueError("Round 27 feature-store schema differs")

    def put_slot(
        self,
        *,
        slot_id: str,
        run_id: str,
        rows: Sequence[Round27FeatureRow],
        condition_audit: Mapping[str, object],
        feature_report: Mapping[str, object],
    ) -> bool:
        if self.read_only:
            raise ValueError("Round 27 feature store is read-only")
        selected_slot = str(slot_id or "").strip().lower()
        selected_run = str(run_id or "").strip()
        audit = _validate_hashed_payload(
            condition_audit,
            hash_field="audit_sha256",
            name="condition audit",
        )
        report = _validate_hashed_payload(
            feature_report,
            hash_field="report_sha256",
            name="feature report",
        )
        selected_rows = tuple(row.validated() for row in rows)
        keys = tuple(
            (row.event_start_ms, row.condition_id, row.decision_time_ms)
            for row in selected_rows
        )
        eligible_ids = audit.get("eligible_condition_ids")
        if (
            not selected_slot.startswith("stage1-")
            or not selected_run
            or not selected_rows
            or keys != tuple(sorted(keys))
            or len(keys) != len(set(keys))
            or any(row.run_id != selected_run for row in selected_rows)
            or audit.get("run_id") != selected_run
            or audit.get("target_free") is not True
            or audit.get("model_data_eligible") is not False
            or audit.get("diagnostic_scope") is not None
            or "source_audit_sha256" in audit
            or not isinstance(eligible_ids, list)
            or any(row.condition_id not in eligible_ids for row in selected_rows)
            or report.get("run_id") != selected_run
            or report.get("condition_audit_sha256") != audit["audit_sha256"]
            or report.get("feature_names_sha256")
            != POLYMARKET_ROUND27_FEATURE_NAMES_SHA256
            or report.get("feature_count") != len(POLYMARKET_ROUND27_FEATURE_NAMES)
            or report.get("feature_row_count") != len(selected_rows)
            or report.get("row_chain_sha256") != _row_chain(selected_rows)
            or report.get("official_resolution_accessed") is not False
            or report.get("target_accessed") is not False
            or report.get("model_data_eligible") is not False
            or report.get("edge_claim") is not False
            or report.get("profitability_claim") is not False
            or report.get("trading_authority") is not False
        ):
            raise ValueError("Round 27 feature slot evidence differs")
        by_condition: dict[str, list[Round27FeatureRow]] = {}
        for row in selected_rows:
            by_condition.setdefault(row.condition_id, []).append(row)
        chunks = tuple(
            _condition_chunk(
                slot_id=selected_slot,
                run_id=selected_run,
                rows=condition_rows,
            )
            for _condition, condition_rows in sorted(
                by_condition.items(),
                key=lambda item: (item[1][0].event_start_ms, item[0]),
            )
        )
        audit_compressed = _compress(
            audit,
            maximum_raw_bytes=_MAXIMUM_METADATA_RAW_BYTES,
        )
        report_compressed = _compress(
            report,
            maximum_raw_bytes=_MAXIMUM_METADATA_RAW_BYTES,
        )
        manifest_body = {
            "schema_version": POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
            "slot_id": selected_slot,
            "run_id": selected_run,
            "condition_audit_sha256": audit["audit_sha256"],
            "feature_report_sha256": report["report_sha256"],
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "condition_count": len(chunks),
            "row_count": len(selected_rows),
            "row_chain_sha256": _row_chain(selected_rows),
            "condition_manifest_sha256": [
                chunk.manifest_sha256 for chunk in chunks
            ],
            "target_accessed": False,
            "trading_authority": False,
        }
        manifest_sha256 = _canonical_sha256(manifest_body)
        existing = self.connection.execute(
            "SELECT manifest_sha256 FROM round27_feature_slot WHERE slot_id = ?",
            [selected_slot],
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != manifest_sha256:
                raise ValueError("Round 27 stored feature slot differs")
            self.audit()
            return False
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for chunk in chunks:
                compressed = chunk.compressed
                self.connection.execute(
                    """
                    INSERT INTO round27_feature_condition VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chunk.condition_id,
                        selected_slot,
                        selected_run,
                        chunk.event_start_ms,
                        chunk.row_count,
                        chunk.row_chain_sha256,
                        POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC,
                        compressed.raw_size_bytes,
                        compressed.compressed_size_bytes,
                        compressed.raw_sha256,
                        compressed.compressed_sha256,
                        compressed.payload,
                        chunk.manifest_sha256,
                    ],
                )
            self.connection.execute(
                """
                INSERT INTO round27_feature_slot VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_slot,
                    selected_run,
                    audit["audit_sha256"],
                    report["report_sha256"],
                    len(chunks),
                    len(selected_rows),
                    _row_chain(selected_rows),
                    audit_compressed.raw_size_bytes,
                    audit_compressed.compressed_size_bytes,
                    audit_compressed.raw_sha256,
                    audit_compressed.compressed_sha256,
                    audit_compressed.payload,
                    report_compressed.raw_size_bytes,
                    report_compressed.compressed_size_bytes,
                    report_compressed.raw_sha256,
                    report_compressed.compressed_sha256,
                    report_compressed.payload,
                    manifest_sha256,
                ],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return True

    def load_rows(self, *, slot_id: str | None = None) -> tuple[Round27FeatureRow, ...]:
        query = """
            SELECT condition_id, slot_id, run_id, event_start_ms, row_count,
                   row_chain_sha256, codec, raw_size_bytes, compressed_size_bytes,
                   raw_sha256, compressed_sha256, payload, manifest_sha256
            FROM round27_feature_condition
        """
        parameters: list[object] = []
        if slot_id is not None:
            query += " WHERE slot_id = ?"
            parameters.append(str(slot_id).lower())
        query += " ORDER BY event_start_ms, condition_id"
        output: list[Round27FeatureRow] = []
        for stored in self.connection.execute(query, parameters).fetchall():
            if stored[6] != POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC:
                raise ValueError("Round 27 stored feature codec differs")
            value = _decompress(
                bytes(stored[11]),
                expected_raw_size=int(stored[7]),
                expected_compressed_size=int(stored[8]),
                expected_raw_sha256=_sha256(stored[9], name="raw payload"),
                expected_compressed_sha256=_sha256(
                    stored[10], name="compressed payload"
                ),
                maximum_raw_bytes=_MAXIMUM_CHUNK_RAW_BYTES,
            )
            if not isinstance(value, Mapping) or not isinstance(value.get("rows"), list):
                raise ValueError("Round 27 condition feature payload differs")
            rows = tuple(
                _row_from_payload(item)
                for item in value["rows"]
                if isinstance(item, Mapping)
            )
            identity = {
                "schema_version": value.get("schema_version"),
                "slot_id": value.get("slot_id"),
                "run_id": value.get("run_id"),
                "condition_id": value.get("condition_id"),
                "event_start_ms": value.get("event_start_ms"),
                "row_count": value.get("row_count"),
                "row_chain_sha256": value.get("row_chain_sha256"),
                "codec": POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC,
                "raw_size_bytes": int(stored[7]),
                "compressed_size_bytes": int(stored[8]),
                "raw_sha256": str(stored[9]),
                "compressed_sha256": str(stored[10]),
            }
            if (
                value.get("target_accessed") is not False
                or value.get("trading_authority") is not False
                or value.get("slot_id") != stored[1]
                or value.get("run_id") != stored[2]
                or value.get("condition_id") != stored[0]
                or value.get("event_start_ms") != int(stored[3])
                or len(rows) != int(stored[4])
                or len(rows) != len(value["rows"])
                or _row_chain(rows) != stored[5]
                or value.get("row_chain_sha256") != stored[5]
                or _canonical_sha256(identity) != stored[12]
            ):
                raise ValueError("Round 27 condition feature manifest differs")
            output.extend(rows)
        return tuple(output)

    def audit(self) -> dict[str, object]:
        rows = self.load_rows()
        slots = self.connection.execute(
            """
            SELECT slot_id, run_id, condition_audit_sha256, feature_report_sha256,
                   condition_count, row_count, row_chain_sha256,
                   audit_raw_size_bytes, audit_compressed_size_bytes,
                   audit_raw_sha256, audit_compressed_sha256, audit_payload,
                   report_raw_size_bytes, report_compressed_size_bytes,
                   report_raw_sha256, report_compressed_sha256, report_payload,
                   manifest_sha256
            FROM round27_feature_slot ORDER BY slot_id
            """
        ).fetchall()
        slot_reports: list[dict[str, object]] = []
        for stored in slots:
            audit = _decompress(
                bytes(stored[11]),
                expected_raw_size=int(stored[7]),
                expected_compressed_size=int(stored[8]),
                expected_raw_sha256=_sha256(stored[9], name="audit raw payload"),
                expected_compressed_sha256=_sha256(
                    stored[10], name="audit compressed payload"
                ),
                maximum_raw_bytes=_MAXIMUM_METADATA_RAW_BYTES,
            )
            report = _decompress(
                bytes(stored[16]),
                expected_raw_size=int(stored[12]),
                expected_compressed_size=int(stored[13]),
                expected_raw_sha256=_sha256(stored[14], name="report raw payload"),
                expected_compressed_sha256=_sha256(
                    stored[15], name="report compressed payload"
                ),
                maximum_raw_bytes=_MAXIMUM_METADATA_RAW_BYTES,
            )
            if not isinstance(audit, Mapping) or not isinstance(report, Mapping):
                raise ValueError("Round 27 feature slot metadata differs")
            validated_audit = _validate_hashed_payload(
                audit,
                hash_field="audit_sha256",
                name="condition audit",
            )
            validated_report = _validate_hashed_payload(
                report,
                hash_field="report_sha256",
                name="feature report",
            )
            condition_rows = tuple(row for row in rows if row.run_id == stored[1])
            manifests = [
                str(item[0])
                for item in self.connection.execute(
                    """
                    SELECT manifest_sha256 FROM round27_feature_condition
                    WHERE slot_id = ? ORDER BY event_start_ms, condition_id
                    """,
                    [stored[0]],
                ).fetchall()
            ]
            manifest_body = {
                "schema_version": POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
                "slot_id": stored[0],
                "run_id": stored[1],
                "condition_audit_sha256": stored[2],
                "feature_report_sha256": stored[3],
                "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
                "condition_count": int(stored[4]),
                "row_count": int(stored[5]),
                "row_chain_sha256": stored[6],
                "condition_manifest_sha256": manifests,
                "target_accessed": False,
                "trading_authority": False,
            }
            if (
                validated_audit["audit_sha256"] != stored[2]
                or validated_report["report_sha256"] != stored[3]
                or len(manifests) != int(stored[4])
                or len(condition_rows) != int(stored[5])
                or _row_chain(condition_rows) != stored[6]
                or _canonical_sha256(manifest_body) != stored[17]
            ):
                raise ValueError("Round 27 feature slot manifest differs")
            slot_reports.append(manifest_body)
        body: dict[str, Any] = {
            "schema_version": POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
            "slot_count": len(slots),
            "condition_count": int(
                self.connection.execute(
                    "SELECT count(*) FROM round27_feature_condition"
                ).fetchone()[0]
            ),
            "row_count": len(rows),
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "slots": slot_reports,
            "target_columns_present": False,
            "target_accessed": False,
            "trading_authority": False,
        }
        body["audit_sha256"] = _canonical_sha256(body)
        return body


__all__ = [
    "POLYMARKET_ROUND27_FEATURE_CHUNK_CODEC",
    "POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION",
    "Round27FeatureStore",
]

"""Atomic compressed storage for target-free Round 25 joint features."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import time

import duckdb
import zstandard

from .polymarket_round25_dataset import (
    round25_development_role,
    select_round25_condition_endpoints,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    Round25JointFeatureSnapshot,
)
from .polymarket_round25_joint_materialization import (
    POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256,
    Round25ConditionFeatureMaterialization,
    Round25JointMaterializationObserver,
    Round25JointReceiptCondition,
    load_round25_joint_receipt_conditions,
    round25_joint_snapshot_sha256,
)
from .polymarket_round25_terminal import (
    audit_round25_terminal_receipts,
    validate_round25_terminal_receipt_audit,
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION = (
    "polymarket-round25-joint-feature-store-v2"
)
POLYMARKET_ROUND25_JOINT_STORE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round25-joint-feature-store-manifest-v2"
)
POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round25-forensic-joint-feature-store-manifest-v1"
)
POLYMARKET_ROUND25_JOINT_CHUNK_SCHEMA_VERSION = (
    "polymarket-round25-joint-feature-chunk-v1"
)
POLYMARKET_ROUND25_JOINT_CHUNK_CODEC = "canonical-json-zstd-3"
POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256 = hashlib.sha256(
    json.dumps(
        list(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
POLYMARKET_ROUND25_MAXIMUM_RAW_CHUNK_BYTES = 16 * 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_ROLES = ("train", "calibration", "selection")
_TABLES = {
    "round25_joint_condition",
    "round25_joint_feature_chunk",
    "round25_joint_store_manifest",
}
_SOURCE_COUNT_FIELDS = {
    "admitted_condition_count",
    "calibration_condition_count",
    "purged_condition_count",
    "selection_condition_count",
    "source_snapshot_count",
    "train_condition_count",
}
_REPLACE_RETRY_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 joint store JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 joint store JSON contains {value}")


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


def _hash_chain(previous: str, value: object) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(_canonical_sha256(value))
    ).hexdigest()


def _strict_json(value: object, *, label: str) -> Mapping[str, object]:
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
    if not isinstance(selected, Mapping) or _canonical_json(selected) != value:
        raise ValueError(f"Round 25 {label} is not canonical JSON")
    return selected


def _snapshot_payload(snapshot: Round25JointFeatureSnapshot) -> dict[str, object]:
    return {
        "available": snapshot.available,
        "clob_source_chain_sha256": snapshot.clob_source_chain_sha256,
        "condition_id": snapshot.condition_id,
        "decision_time_ms": snapshot.decision_time_ms,
        "event_start_ms": snapshot.event_start_ms,
        "market_prior_probability": snapshot.market_prior_probability,
        "maximum_receipt_ms": snapshot.maximum_receipt_ms,
        "model_design_sha256": snapshot.model_design_sha256,
        "reasons": list(snapshot.reasons),
        "snapshot_sha256": round25_joint_snapshot_sha256(snapshot),
        "source_chain_sha256": snapshot.source_chain_sha256,
        "trading_authority": snapshot.trading_authority,
        "twap_source_chain_sha256": snapshot.twap_source_chain_sha256,
        "values": list(snapshot.values),
    }


def _decode_snapshot(value: object) -> Round25JointFeatureSnapshot:
    expected = {
        "available",
        "clob_source_chain_sha256",
        "condition_id",
        "decision_time_ms",
        "event_start_ms",
        "market_prior_probability",
        "maximum_receipt_ms",
        "model_design_sha256",
        "reasons",
        "snapshot_sha256",
        "source_chain_sha256",
        "trading_authority",
        "twap_source_chain_sha256",
        "values",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Round 25 stored joint snapshot schema differs")
    reasons = value.get("reasons")
    values = value.get("values")
    if (
        not isinstance(value.get("condition_id"), str)
        or type(value.get("event_start_ms")) is not int
        or type(value.get("decision_time_ms")) is not int
        or type(value.get("available")) is not bool
        or not isinstance(reasons, list)
        or any(not isinstance(item, str) for item in reasons)
        or isinstance(value.get("market_prior_probability"), bool)
        or not isinstance(value.get("market_prior_probability"), (int, float))
        or not isinstance(values, list)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in values
        )
        or type(value.get("maximum_receipt_ms")) is not int
        or type(value.get("trading_authority")) is not bool
        or any(
            not isinstance(value.get(field), str)
            for field in (
                "clob_source_chain_sha256",
                "model_design_sha256",
                "snapshot_sha256",
                "source_chain_sha256",
                "twap_source_chain_sha256",
            )
        )
    ):
        raise ValueError("Round 25 stored joint snapshot payload differs")
    snapshot = Round25JointFeatureSnapshot(
        condition_id=str(value["condition_id"]),
        event_start_ms=int(value["event_start_ms"]),
        decision_time_ms=int(value["decision_time_ms"]),
        available=value["available"],
        reasons=tuple(reasons),
        market_prior_probability=float(value["market_prior_probability"]),
        values=tuple(float(item) for item in values),
        source_chain_sha256=str(value["source_chain_sha256"]),
        twap_source_chain_sha256=str(value["twap_source_chain_sha256"]),
        clob_source_chain_sha256=str(value["clob_source_chain_sha256"]),
        maximum_receipt_ms=int(value["maximum_receipt_ms"]),
        model_design_sha256=str(value["model_design_sha256"]),
        trading_authority=value["trading_authority"],
    )
    if value["snapshot_sha256"] != round25_joint_snapshot_sha256(snapshot):
        raise ValueError("Round 25 stored joint snapshot hash differs")
    return snapshot


def _materialization_payload(
    value: Round25ConditionFeatureMaterialization,
) -> dict[str, object]:
    return {
        **value.identity_payload(),
        "materialization_sha256": value.materialization_sha256,
    }


def _condition_identity_payload(
    value: Round25ConditionFeatureMaterialization | Round25JointReceiptCondition,
) -> dict[str, object]:
    if isinstance(value, Round25ConditionFeatureMaterialization):
        return {
            "condition_id": value.condition_id,
            "down_token_id": value.down_token_id,
            "event_end_ms": value.event_end_ms,
            "event_start_ms": value.event_start_ms,
            "market_id": value.market_id,
            "resolution_source": value.resolution_source,
            "role": value.role,
            "run_id": value.run_id,
            "segment_index": value.segment_index,
            "slug": value.slug,
            "source_snapshot_observed_wall_ms": (
                value.source_snapshot_observed_wall_ms
            ),
            "source_snapshot_sha256": value.source_snapshot_sha256,
            "up_token_id": value.up_token_id,
        }
    selected = value.validated()
    return {
        "condition_id": selected.condition_id,
        "down_token_id": selected.down_token_id,
        "event_end_ms": selected.event_end_ms,
        "event_start_ms": selected.event_start_ms,
        "market_id": selected.market_id,
        "resolution_source": selected.resolution_source,
        "role": selected.role,
        "run_id": selected.run_id,
        "segment_index": selected.segment_index,
        "slug": selected.slug,
        "source_snapshot_observed_wall_ms": selected.snapshot_observed_wall_ms,
        "source_snapshot_sha256": selected.snapshot_sha256,
        "up_token_id": selected.up_token_id,
    }


def _encode_chunk(
    value: Round25ConditionFeatureMaterialization,
) -> tuple[dict[str, object], bytes]:
    rows = [_snapshot_payload(snapshot) for snapshot in value.persisted_snapshots]
    body = {
        "authority": {
            "model_data_eligible": False,
            "model_scores_accessed": False,
            "target_accessed": False,
            "trading_authority": False,
        },
        "condition_id": value.condition_id,
        "event_start_ms": value.event_start_ms,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "materialization_contract_sha256": (
            POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
        ),
        "materialization_sha256": value.materialization_sha256,
        "role": value.role,
        "row_count": len(rows),
        "rows": rows,
        "schema_version": POLYMARKET_ROUND25_JOINT_CHUNK_SCHEMA_VERSION,
    }
    raw = _canonical_json(body).encode("ascii")
    if not 2 <= len(raw) <= POLYMARKET_ROUND25_MAXIMUM_RAW_CHUNK_BYTES:
        raise ValueError("Round 25 joint feature chunk size is outside the bound")
    compressed = zstandard.ZstdCompressor(
        level=3,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)
    envelope = {
        "codec": POLYMARKET_ROUND25_JOINT_CHUNK_CODEC,
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_size_bytes": len(compressed),
        "condition_id": value.condition_id,
        "materialization_sha256": value.materialization_sha256,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_size_bytes": len(raw),
        "row_count": len(rows),
    }
    return {**envelope, "chunk_sha256": _canonical_sha256(envelope)}, compressed


def _decode_chunk(
    manifest: Mapping[str, object],
    compressed: bytes,
) -> tuple[Round25JointFeatureSnapshot, ...]:
    expected = {
        "chunk_sha256",
        "codec",
        "compressed_sha256",
        "compressed_size_bytes",
        "condition_id",
        "materialization_sha256",
        "raw_sha256",
        "raw_size_bytes",
        "row_count",
    }
    body = dict(manifest)
    claimed = str(body.pop("chunk_sha256", ""))
    raw_size = body.get("raw_size_bytes")
    if (
        set(manifest) != expected
        or claimed != _canonical_sha256(body)
        or _SHA256.fullmatch(claimed) is None
        or body.get("codec") != POLYMARKET_ROUND25_JOINT_CHUNK_CODEC
        or type(raw_size) is not int
        or not 2 <= raw_size <= POLYMARKET_ROUND25_MAXIMUM_RAW_CHUNK_BYTES
        or body.get("compressed_size_bytes") != len(compressed)
        or body.get("compressed_sha256")
        != hashlib.sha256(compressed).hexdigest()
    ):
        raise ValueError("Round 25 joint feature chunk envelope differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=raw_size,
        )
    except zstandard.ZstdError as exc:
        raise ValueError("Round 25 joint feature chunk is not valid zstd") from exc
    if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != body["raw_sha256"]:
        raise ValueError("Round 25 joint feature chunk content hash differs")
    decoded = _strict_json(raw.decode("ascii"), label="joint feature chunk")
    expected_body = {
        "authority",
        "condition_id",
        "event_start_ms",
        "feature_names_sha256",
        "feature_schema_version",
        "materialization_contract_sha256",
        "materialization_sha256",
        "role",
        "row_count",
        "rows",
        "schema_version",
    }
    rows = decoded.get("rows")
    if (
        set(decoded) != expected_body
        or decoded.get("schema_version")
        != POLYMARKET_ROUND25_JOINT_CHUNK_SCHEMA_VERSION
        or decoded.get("condition_id") != body["condition_id"]
        or decoded.get("materialization_sha256")
        != body["materialization_sha256"]
        or decoded.get("feature_schema_version")
        != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
        or decoded.get("feature_names_sha256")
        != POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
        or decoded.get("materialization_contract_sha256")
        != POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
        or decoded.get("authority")
        != {
            "model_data_eligible": False,
            "model_scores_accessed": False,
            "target_accessed": False,
            "trading_authority": False,
        }
        or not isinstance(rows, list)
        or decoded.get("row_count") != len(rows)
        or body.get("row_count") != len(rows)
    ):
        raise ValueError("Round 25 joint feature chunk identity differs")
    return tuple(_decode_snapshot(item) for item in rows)


def _validate_tables(connection: duckdb.DuckDBPyConnection) -> None:
    rows = connection.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    if (
        {str(row[0]) for row in rows} != _TABLES
        or any(str(row[1]) != "BASE TABLE" for row in rows)
    ):
        raise ValueError("Round 25 joint store schema differs")


def _replace_with_retries(source: Path, destination: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_SECONDS):
                raise
            time.sleep(_REPLACE_RETRY_SECONDS[attempt])


def _initialize_joint_store_writer(
    writer: Round25JointStoreWriter,
    destination: str | Path,
) -> None:
    writer.destination = Path(destination)
    writer.partial = writer.destination.with_name(f".{writer.destination.name}.partial")
    if (
        writer.destination.is_symlink()
        or writer.destination.exists()
        or writer.partial.is_symlink()
        or writer.partial.exists()
        or Path(f"{writer.partial}.wal").exists()
    ):
        raise ValueError("Round 25 joint store destination is not empty")
    writer.destination.parent.mkdir(parents=True, exist_ok=True)
    writer.connection = duckdb.connect(str(writer.partial))
    writer.connection.execute("SET memory_limit = '1GB'")
    writer.connection.execute("SET threads = 2")
    writer.connection.execute("PRAGMA enable_checkpoint_on_shutdown")
    writer.connection.execute(
        """
            CREATE TABLE round25_joint_condition (
                condition_id VARCHAR PRIMARY KEY,
                event_start_ms BIGINT NOT NULL,
                role VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                segment_index BIGINT NOT NULL,
                source_snapshot_sha256 VARCHAR NOT NULL,
                source_snapshot_observed_wall_ms BIGINT NOT NULL,
                market_id VARCHAR NOT NULL,
                slug VARCHAR NOT NULL,
                event_end_ms BIGINT NOT NULL,
                up_token_id VARCHAR NOT NULL,
                down_token_id VARCHAR NOT NULL,
                resolution_source VARCHAR NOT NULL,
                admitted BOOLEAN NOT NULL,
                source_record_count BIGINT NOT NULL,
                available_decision_count BIGINT NOT NULL,
                materialization_json VARCHAR NOT NULL,
                materialization_sha256 VARCHAR NOT NULL,
                feature_chunk_sha256 VARCHAR
            );
            CREATE TABLE round25_joint_feature_chunk (
                condition_id VARCHAR PRIMARY KEY,
                chunk_manifest_json VARCHAR NOT NULL,
                chunk_sha256 VARCHAR NOT NULL,
                compressed_payload BLOB NOT NULL
            );
            CREATE TABLE round25_joint_store_manifest (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                manifest_json VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL
            );
            """
    )
    writer.connection.execute("BEGIN TRANSACTION")
    writer.condition_count = 0
    writer.admitted_condition_count = 0
    writer.feature_row_count = 0
    writer.role_counts = Counter()
    writer.rejection_counts = Counter()
    writer.unavailable_reason_counts = Counter()
    writer.logical_root_sha256 = _EMPTY_SHA256
    writer.condition_population_sha256 = _EMPTY_SHA256
    writer._condition_ids = set()
    writer._last_identity = None
    writer._closed = False
    writer._published = False


class Round25JointStoreWriter:
    """One-writer atomic store; the destination appears only after final audit."""

    def __init__(
        self,
        destination: str | Path,
        *,
        terminal_transport_manifest: Mapping[str, object],
    ) -> None:
        self.transport = validate_round25_terminal_transport_manifest(
            terminal_transport_manifest
        )
        _initialize_joint_store_writer(self, destination)

    def add(self, value: Round25ConditionFeatureMaterialization) -> None:
        selected = value.validated()
        identity = (selected.event_start_ms, selected.condition_id)
        if (
            self._closed
            or selected.condition_id in self._condition_ids
            or self._last_identity is not None
            and identity <= self._last_identity
            or round25_development_role(selected.event_start_ms) != selected.role
        ):
            raise ValueError("Round 25 joint store condition identity differs")
        materialization_json = _canonical_json(_materialization_payload(selected))
        chunk_sha256: str | None = None
        if selected.admitted:
            chunk_manifest, compressed = _encode_chunk(selected)
            chunk_sha256 = str(chunk_manifest["chunk_sha256"])
            self.connection.execute(
                "INSERT INTO round25_joint_feature_chunk VALUES (?, ?, ?, ?)",
                [
                    selected.condition_id,
                    _canonical_json(chunk_manifest),
                    chunk_sha256,
                    compressed,
                ],
            )
        self.connection.execute(
            """
            INSERT INTO round25_joint_condition
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                selected.condition_id,
                selected.event_start_ms,
                selected.role,
                selected.run_id,
                selected.segment_index,
                selected.source_snapshot_sha256,
                selected.source_snapshot_observed_wall_ms,
                selected.market_id,
                selected.slug,
                selected.event_end_ms,
                selected.up_token_id,
                selected.down_token_id,
                selected.resolution_source,
                selected.admitted,
                selected.source_record_count,
                selected.available_decision_count,
                materialization_json,
                selected.materialization_sha256,
                chunk_sha256,
            ],
        )
        condition_identity = _condition_identity_payload(selected)
        self.condition_population_sha256 = _hash_chain(
            self.condition_population_sha256,
            condition_identity,
        )
        self.logical_root_sha256 = _hash_chain(
            self.logical_root_sha256,
            {
                "condition_id": selected.condition_id,
                "condition_identity_sha256": _canonical_sha256(condition_identity),
                "event_start_ms": selected.event_start_ms,
                "feature_chunk_sha256": chunk_sha256,
                "materialization_sha256": selected.materialization_sha256,
                "role": selected.role,
            },
        )
        self._condition_ids.add(selected.condition_id)
        self._last_identity = identity
        self.condition_count += 1
        self.admitted_condition_count += int(selected.admitted)
        self.feature_row_count += len(selected.persisted_snapshots)
        self.role_counts[selected.role] += 1
        self.rejection_counts.update(selected.rejection_reasons)
        self.unavailable_reason_counts.update(
            dict(selected.unavailable_reason_counts)
        )

    def _common_manifest_body(
        self,
        *,
        created_at_ms: int,
        source_counts: Mapping[str, int],
    ) -> dict[str, object]:
        counts = dict(source_counts)
        if (
            self._closed
            or type(created_at_ms) is not int
            or created_at_ms <= 0
            or set(counts) != _SOURCE_COUNT_FIELDS
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts["admitted_condition_count"] != self.condition_count
            or counts["source_snapshot_count"]
            != self.condition_count + counts["purged_condition_count"]
            or any(
                counts[f"{role}_condition_count"] != self.role_counts[role]
                for role in _ROLES
            )
            or self.condition_count <= 0
            or self.logical_root_sha256 == _EMPTY_SHA256
        ):
            raise ValueError("Round 25 joint store final accounting differs")
        return {
            "admitted_condition_count": self.admitted_condition_count,
            "atomic_file_publication": True,
            "chunk_codec": POLYMARKET_ROUND25_JOINT_CHUNK_CODEC,
            "condition_count": self.condition_count,
            "condition_population_sha256": self.condition_population_sha256,
            "created_at_ms": created_at_ms,
            "feature_chunk_count": self.admitted_condition_count,
            "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
            "feature_row_count": self.feature_row_count,
            "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
            "live_trading_authority": False,
            "logical_root_sha256": self.logical_root_sha256,
            "materialization_contract_sha256": (
                POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
            ),
            "model_data_eligible": False,
            "model_scores_consulted": False,
            "outcomes_consulted": False,
            "paper_trading_authority": False,
            "profitability_claim": False,
            "purged_condition_count": counts["purged_condition_count"],
            "raw_receipts_copied": False,
            "receipt_scan_count": 1,
            "rejected_condition_count": (
                self.condition_count - self.admitted_condition_count
            ),
            "rejection_reason_counts": dict(sorted(self.rejection_counts.items())),
            "role_condition_counts": {
                role: self.role_counts[role] for role in _ROLES
            },
            "source_snapshot_count": counts["source_snapshot_count"],
            "store_schema_version": POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION,
            "target_accessed": False,
            "unavailable_reason_counts": dict(
                sorted(self.unavailable_reason_counts.items())
            ),
        }

    def _publish_manifest(self, body: Mapping[str, object]) -> dict[str, object]:
        manifest = validate_round25_joint_store_manifest(
            {**body, "manifest_sha256": _canonical_sha256(body)}
        )
        self.connection.execute(
            "INSERT INTO round25_joint_store_manifest VALUES (TRUE, ?, ?)",
            [_canonical_json(manifest), manifest["manifest_sha256"]],
        )
        self.connection.execute("COMMIT")
        self.connection.execute("CHECKPOINT")
        self.connection.close()
        self._closed = True
        if Path(f"{self.partial}.wal").exists() or self.destination.exists():
            raise RuntimeError("Round 25 joint store final file state differs")
        _replace_with_retries(self.partial, self.destination)
        self._published = True
        return manifest

    def finalize(
        self,
        *,
        terminal_receipt_audit: Mapping[str, object],
        source_counts: Mapping[str, int],
    ) -> dict[str, object]:
        audit = validate_round25_terminal_receipt_audit(
            terminal_receipt_audit,
            terminal_transport_manifest=self.transport,
        )
        body = {
            **self._common_manifest_body(
                created_at_ms=int(audit["created_at_ms"]),
                source_counts=source_counts,
            ),
            "schema_version": (POLYMARKET_ROUND25_JOINT_STORE_MANIFEST_SCHEMA_VERSION),
            "terminal_receipt_audit_sha256": audit["audit_sha256"],
            "terminal_transport_manifest_sha256": self.transport["manifest_sha256"],
        }
        return self._publish_manifest(body)

    def abort(self) -> None:
        if not self._closed:
            try:
                self.connection.execute("ROLLBACK")
            except duckdb.Error:
                pass
            self.connection.close()
            self._closed = True
        if self._published:
            return
        for path in (self.partial, Path(f"{self.partial}.wal")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class Round25ForensicJointStoreWriter(Round25JointStoreWriter):
    """Atomic joint store with explicit failed-transport provenance."""

    def __init__(
        self,
        destination: str | Path,
        *,
        forensic_audit_sha256: str,
        salvage_contract_sha256: str,
        source_report_sha256: str,
        source_evidence_manifest_sha256: str,
        source_run_id: str,
    ) -> None:
        hashes = (
            forensic_audit_sha256,
            salvage_contract_sha256,
            source_report_sha256,
            source_evidence_manifest_sha256,
        )
        if (
            any(_SHA256.fullmatch(value) is None for value in hashes)
            or re.fullmatch(r"[0-9a-f]{32}", source_run_id) is None
        ):
            raise ValueError("Round 25 forensic joint store source differs")
        self.forensic_audit_sha256 = forensic_audit_sha256
        self.salvage_contract_sha256 = salvage_contract_sha256
        self.source_report_sha256 = source_report_sha256
        self.source_evidence_manifest_sha256 = source_evidence_manifest_sha256
        self.source_run_id = source_run_id
        _initialize_joint_store_writer(self, destination)

    def finalize_forensic(
        self,
        *,
        created_at_ms: int,
        source_counts: Mapping[str, int],
    ) -> dict[str, object]:
        body = {
            **self._common_manifest_body(
                created_at_ms=created_at_ms,
                source_counts=source_counts,
            ),
            "diagnostic_only": True,
            "forensic_audit_sha256": self.forensic_audit_sha256,
            "salvage_contract_sha256": self.salvage_contract_sha256,
            "schema_version": (
                POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION
            ),
            "source_evidence_manifest_sha256": (self.source_evidence_manifest_sha256),
            "source_kind": "qualified_transport_failure",
            "source_recorder_status": "failed",
            "source_report_sha256": self.source_report_sha256,
            "source_run_id": self.source_run_id,
        }
        return self._publish_manifest(body)


def validate_round25_joint_store_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    common_expected = {
        "admitted_condition_count",
        "atomic_file_publication",
        "chunk_codec",
        "condition_count",
        "condition_population_sha256",
        "created_at_ms",
        "feature_chunk_count",
        "feature_names_sha256",
        "feature_row_count",
        "feature_schema_version",
        "live_trading_authority",
        "logical_root_sha256",
        "materialization_contract_sha256",
        "model_data_eligible",
        "model_scores_consulted",
        "outcomes_consulted",
        "paper_trading_authority",
        "profitability_claim",
        "purged_condition_count",
        "raw_receipts_copied",
        "receipt_scan_count",
        "rejected_condition_count",
        "rejection_reason_counts",
        "role_condition_counts",
        "schema_version",
        "source_snapshot_count",
        "store_schema_version",
        "target_accessed",
        "unavailable_reason_counts",
    }
    schema_version = payload.get("schema_version")
    if schema_version == POLYMARKET_ROUND25_JOINT_STORE_MANIFEST_SCHEMA_VERSION:
        provenance_expected = {
            "terminal_receipt_audit_sha256",
            "terminal_transport_manifest_sha256",
        }
        provenance_valid = all(
            _SHA256.fullmatch(str(payload.get(field) or "")) is not None
            for field in provenance_expected
        )
    elif (
        schema_version
        == POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION
    ):
        provenance_expected = {
            "diagnostic_only",
            "forensic_audit_sha256",
            "salvage_contract_sha256",
            "source_evidence_manifest_sha256",
            "source_kind",
            "source_recorder_status",
            "source_report_sha256",
            "source_run_id",
        }
        provenance_valid = (
            payload.get("diagnostic_only") is True
            and payload.get("source_kind") == "qualified_transport_failure"
            and payload.get("source_recorder_status") == "failed"
            and re.fullmatch(r"[0-9a-f]{32}", str(payload.get("source_run_id") or ""))
            is not None
            and all(
                _SHA256.fullmatch(str(payload.get(field) or "")) is not None
                for field in (
                    "forensic_audit_sha256",
                    "salvage_contract_sha256",
                    "source_evidence_manifest_sha256",
                    "source_report_sha256",
                )
            )
        )
    else:
        provenance_expected = set()
        provenance_valid = False
    expected = common_expected | provenance_expected
    count_fields = (
        "admitted_condition_count",
        "condition_count",
        "feature_chunk_count",
        "feature_row_count",
        "purged_condition_count",
        "rejected_condition_count",
        "source_snapshot_count",
    )
    role_counts = payload.get("role_condition_counts")
    count_maps = (
        payload.get("rejection_reason_counts"),
        payload.get("unavailable_reason_counts"),
    )
    if (
        set(payload) != expected
        or not provenance_valid
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or payload.get("store_schema_version")
        != POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION
        or payload.get("materialization_contract_sha256")
        != POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
        or payload.get("feature_schema_version")
        != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
        or payload.get("feature_names_sha256")
        != POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
        or payload.get("chunk_codec") != POLYMARKET_ROUND25_JOINT_CHUNK_CODEC
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= 0
        or any(
            type(payload.get(field)) is not int or payload[field] < 0
            for field in count_fields
        )
        or payload["condition_count"] <= 0
        or _SHA256.fullmatch(
            str(payload.get("condition_population_sha256") or "")
        )
        is None
        or payload["condition_population_sha256"] == _EMPTY_SHA256
        or payload["condition_count"]
        != payload["admitted_condition_count"]
        + payload["rejected_condition_count"]
        or payload["feature_chunk_count"] != payload["admitted_condition_count"]
        or payload["source_snapshot_count"]
        != payload["condition_count"] + payload["purged_condition_count"]
        or not isinstance(role_counts, Mapping)
        or set(role_counts) != set(_ROLES)
        or any(
            type(role_counts[role]) is not int or role_counts[role] < 0
            for role in _ROLES
        )
        or sum(role_counts.values()) != payload["condition_count"]
        or any(
            not isinstance(counts, Mapping)
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count <= 0
                for reason, count in counts.items()
            )
            for counts in count_maps
        )
        or _SHA256.fullmatch(str(payload.get("logical_root_sha256") or "")) is None
        or payload.get("logical_root_sha256") == _EMPTY_SHA256
        or payload.get("receipt_scan_count") != 1
        or payload.get("atomic_file_publication") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "raw_receipts_copied",
                "target_accessed",
                "outcomes_consulted",
                "model_scores_consulted",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 25 joint store manifest differs")
    return {**payload, "manifest_sha256": claimed}


def load_round25_joint_store_manifest(database: str | Path) -> dict[str, object]:
    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 joint store database is unavailable")
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        rows = connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM round25_joint_store_manifest"
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("Round 25 joint store manifest row differs")
    manifest = validate_round25_joint_store_manifest(
        _strict_json(rows[0][0], label="joint store manifest")
    )
    if rows[0][1] != manifest["manifest_sha256"]:
        raise ValueError("Round 25 joint store manifest hash differs")
    return manifest


def load_round25_joint_condition(
    database: str | Path,
    condition_id: str,
) -> Round25ConditionFeatureMaterialization:
    selected_id = str(condition_id or "").strip().lower()
    path = Path(database)
    if (
        _CONDITION_ID.fullmatch(selected_id) is None
        or path.is_symlink()
        or not path.is_file()
        or Path(f"{path}.wal").exists()
    ):
        raise ValueError("Round 25 joint store condition request differs")
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        condition = connection.execute(
            """
            SELECT materialization_json, materialization_sha256,
                   feature_chunk_sha256
            FROM round25_joint_condition WHERE condition_id = ?
            """,
            [selected_id],
        ).fetchone()
        if condition is None:
            raise ValueError("Round 25 joint store condition is unavailable")
        chunk = None
        if condition[2] is not None:
            chunk = connection.execute(
                """
                SELECT chunk_manifest_json, chunk_sha256, compressed_payload
                FROM round25_joint_feature_chunk WHERE condition_id = ?
                """,
                [selected_id],
            ).fetchone()
    return _decode_stored_condition(
        selected_id,
        condition=condition,
        chunk=chunk,
    )


def _decode_stored_condition(
    selected_id: str,
    *,
    condition: tuple[object, ...],
    chunk: tuple[object, ...] | None,
) -> Round25ConditionFeatureMaterialization:
    metadata = _strict_json(condition[0], label="joint materialization")
    expected = {
        "admitted",
        "available_decision_count",
        "condition_id",
        "contract_sha256",
        "decision_count",
        "down_token_id",
        "event_end_ms",
        "event_start_ms",
        "market_id",
        "materialization_sha256",
        "persisted_snapshot_sha256",
        "rejection_reasons",
        "resolution_source",
        "role",
        "run_id",
        "schema_version",
        "segment_index",
        "selected_endpoint_decision_time_ms",
        "source_record_count",
        "source_snapshot_observed_wall_ms",
        "source_snapshot_sha256",
        "slug",
        "target_accessed",
        "trading_authority",
        "unavailable_reason_counts",
        "up_token_id",
    }
    list_fields = (
        "persisted_snapshot_sha256",
        "rejection_reasons",
        "selected_endpoint_decision_time_ms",
        "unavailable_reason_counts",
    )
    if (
        set(metadata) != expected
        or condition[1] != metadata.get("materialization_sha256")
        or not isinstance(metadata.get("run_id"), str)
        or type(metadata.get("segment_index")) is not int
        or not isinstance(metadata.get("source_snapshot_sha256"), str)
        or type(metadata.get("source_snapshot_observed_wall_ms")) is not int
        or not isinstance(metadata.get("market_id"), str)
        or not isinstance(metadata.get("condition_id"), str)
        or not isinstance(metadata.get("slug"), str)
        or type(metadata.get("event_start_ms")) is not int
        or type(metadata.get("event_end_ms")) is not int
        or not isinstance(metadata.get("up_token_id"), str)
        or not isinstance(metadata.get("down_token_id"), str)
        or not isinstance(metadata.get("resolution_source"), str)
        or not isinstance(metadata.get("role"), str)
        or type(metadata.get("source_record_count")) is not int
        or type(metadata.get("decision_count")) is not int
        or type(metadata.get("available_decision_count")) is not int
        or type(metadata.get("admitted")) is not bool
        or type(metadata.get("target_accessed")) is not bool
        or type(metadata.get("trading_authority")) is not bool
        or any(not isinstance(metadata.get(field), list) for field in list_fields)
        or any(
            not isinstance(item, str)
            for item in metadata["persisted_snapshot_sha256"]
        )
        or any(not isinstance(item, str) for item in metadata["rejection_reasons"])
        or any(
            type(item) is not int
            for item in metadata["selected_endpoint_decision_time_ms"]
        )
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"count", "reason"}
            or type(item.get("count")) is not int
            or not isinstance(item.get("reason"), str)
            for item in metadata["unavailable_reason_counts"]
        )
    ):
        raise ValueError("Round 25 stored materialization metadata differs")
    rows: tuple[Round25JointFeatureSnapshot, ...] = ()
    if chunk is not None:
        chunk_manifest = _strict_json(chunk[0], label="joint chunk manifest")
        if chunk[1] != chunk_manifest.get("chunk_sha256") or condition[2] != chunk[1]:
            raise ValueError("Round 25 stored chunk hash differs")
        rows = _decode_chunk(chunk_manifest, bytes(chunk[2]))
    elif condition[2] is not None:
        raise ValueError("Round 25 stored chunk is unavailable")
    result = Round25ConditionFeatureMaterialization(
        run_id=str(metadata["run_id"]),
        segment_index=int(metadata["segment_index"]),
        source_snapshot_sha256=str(metadata["source_snapshot_sha256"]),
        source_snapshot_observed_wall_ms=int(
            metadata["source_snapshot_observed_wall_ms"]
        ),
        market_id=str(metadata["market_id"]),
        condition_id=str(metadata["condition_id"]),
        slug=str(metadata["slug"]),
        event_start_ms=int(metadata["event_start_ms"]),
        event_end_ms=int(metadata["event_end_ms"]),
        up_token_id=str(metadata["up_token_id"]),
        down_token_id=str(metadata["down_token_id"]),
        resolution_source=str(metadata["resolution_source"]),
        role=str(metadata["role"]),
        source_record_count=int(metadata["source_record_count"]),
        decision_count=int(metadata["decision_count"]),
        available_decision_count=int(metadata["available_decision_count"]),
        admitted=metadata["admitted"],
        rejection_reasons=tuple(metadata["rejection_reasons"]),
        selected_endpoint_decision_time_ms=tuple(
            metadata["selected_endpoint_decision_time_ms"]
        ),
        persisted_snapshots=rows,
        persisted_snapshot_sha256=tuple(
            metadata["persisted_snapshot_sha256"]
        ),
        unavailable_reason_counts=tuple(
            (item["reason"], item["count"])
            for item in metadata["unavailable_reason_counts"]
        ),
        materialization_sha256=str(metadata["materialization_sha256"]),
        schema_version=str(metadata["schema_version"]),
        contract_sha256=str(metadata["contract_sha256"]),
        target_accessed=metadata["target_accessed"],
        trading_authority=metadata["trading_authority"],
    ).validated()
    if result.condition_id != selected_id:
        raise ValueError("Round 25 stored condition identity differs")
    return result


def load_round25_joint_condition_identities(
    database: str | Path,
) -> tuple[Round25JointReceiptCondition, ...]:
    """Load only source-bound market identities from an audited feature store."""

    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 joint store database is unavailable")
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        rows = connection.execute(
            """
            SELECT run_id, segment_index, source_snapshot_sha256,
                   source_snapshot_observed_wall_ms, market_id, condition_id,
                   slug, event_start_ms, event_end_ms, up_token_id,
                   down_token_id, resolution_source, role
            FROM round25_joint_condition
            ORDER BY event_start_ms, condition_id
            """
        ).fetchall()
    output: list[Round25JointReceiptCondition] = []
    previous: tuple[int, str] | None = None
    for row in rows:
        if (
            not isinstance(row[0], str)
            or type(row[1]) is not int
            or not isinstance(row[2], str)
            or type(row[3]) is not int
            or not isinstance(row[4], str)
            or not isinstance(row[5], str)
            or not isinstance(row[6], str)
            or type(row[7]) is not int
            or type(row[8]) is not int
            or not isinstance(row[9], str)
            or not isinstance(row[10], str)
            or not isinstance(row[11], str)
            or not isinstance(row[12], str)
        ):
            raise ValueError("Round 25 joint condition identity columns differ")
        condition = Round25JointReceiptCondition(
            run_id=row[0],
            segment_index=row[1],
            snapshot_sha256=row[2],
            snapshot_observed_wall_ms=row[3],
            market_id=row[4],
            condition_id=row[5],
            slug=row[6],
            event_start_ms=row[7],
            event_end_ms=row[8],
            up_token_id=row[9],
            down_token_id=row[10],
            resolution_source=row[11],
            role=row[12],
        ).validated()
        identity = (condition.event_start_ms, condition.condition_id)
        if previous is not None and identity <= previous:
            raise ValueError("Round 25 joint condition identity chronology differs")
        previous = identity
        output.append(condition)
    if not output:
        raise ValueError("Round 25 joint condition identity population is empty")
    return tuple(output)


def audit_round25_joint_store(database: str | Path) -> dict[str, object]:
    """Recompute all stored condition and chunk evidence in one read-only pass."""

    manifest = load_round25_joint_store_manifest(database)
    path = Path(database)
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        rows = connection.execute(
            """
            SELECT c.condition_id, c.event_start_ms, c.role, c.run_id,
                   c.segment_index, c.source_snapshot_sha256,
                   c.source_snapshot_observed_wall_ms, c.market_id, c.slug,
                   c.event_end_ms, c.up_token_id, c.down_token_id,
                   c.resolution_source, c.admitted, c.source_record_count,
                   c.available_decision_count, c.materialization_json,
                   c.materialization_sha256, c.feature_chunk_sha256,
                   f.chunk_manifest_json, f.chunk_sha256, f.compressed_payload
            FROM round25_joint_condition AS c
            LEFT JOIN round25_joint_feature_chunk AS f USING (condition_id)
            ORDER BY c.event_start_ms, c.condition_id
            """
        ).fetchall()
    logical_root = _EMPTY_SHA256
    admitted_count = 0
    feature_rows = 0
    role_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    unavailable_counts: Counter[str] = Counter()
    condition_population = _EMPTY_SHA256
    previous: tuple[int, str] | None = None
    for row in rows:
        identity = (int(row[1]), str(row[0]))
        if previous is not None and identity <= previous:
            raise ValueError("Round 25 joint store condition chronology differs")
        previous = identity
        condition = (row[16], row[17], row[18])
        chunk = None if row[19] is None else (row[19], row[20], row[21])
        result = _decode_stored_condition(
            str(row[0]),
            condition=condition,
            chunk=chunk,
        )
        if (
            row[1] != result.event_start_ms
            or row[2] != result.role
            or row[3] != result.run_id
            or row[4] != result.segment_index
            or row[5] != result.source_snapshot_sha256
            or row[6] != result.source_snapshot_observed_wall_ms
            or row[7] != result.market_id
            or row[8] != result.slug
            or row[9] != result.event_end_ms
            or row[10] != result.up_token_id
            or row[11] != result.down_token_id
            or row[12] != result.resolution_source
            or row[13] is not result.admitted
            or row[14] != result.source_record_count
            or row[15] != result.available_decision_count
        ):
            raise ValueError("Round 25 joint store condition columns differ")
        condition_identity = _condition_identity_payload(result)
        condition_population = _hash_chain(
            condition_population,
            condition_identity,
        )
        logical_root = _hash_chain(
            logical_root,
            {
                "condition_id": result.condition_id,
                "event_start_ms": result.event_start_ms,
                "condition_identity_sha256": _canonical_sha256(condition_identity),
                "feature_chunk_sha256": row[18],
                "materialization_sha256": result.materialization_sha256,
                "role": result.role,
            },
        )
        admitted_count += int(result.admitted)
        feature_rows += len(result.persisted_snapshots)
        role_counts[result.role] += 1
        rejection_counts.update(result.rejection_reasons)
        unavailable_counts.update(dict(result.unavailable_reason_counts))
    if (
        len(rows) != manifest["condition_count"]
        or admitted_count != manifest["admitted_condition_count"]
        or admitted_count != manifest["feature_chunk_count"]
        or feature_rows != manifest["feature_row_count"]
        or condition_population != manifest["condition_population_sha256"]
        or {role: role_counts[role] for role in _ROLES}
        != manifest["role_condition_counts"]
        or dict(sorted(rejection_counts.items()))
        != manifest["rejection_reason_counts"]
        or dict(sorted(unavailable_counts.items()))
        != manifest["unavailable_reason_counts"]
        or logical_root != manifest["logical_root_sha256"]
    ):
        raise ValueError("Round 25 joint store deep audit differs")
    return manifest


def load_round25_joint_endpoint_inputs(
    database: str | Path,
) -> tuple[dict[str, object], dict[str, tuple[Round25JointFeatureSnapshot, ...]]]:
    """Deep-audit once, then retain only the 16 target-blind endpoints per condition."""

    path = Path(database)
    before = path.stat() if path.is_file() else None
    manifest = audit_round25_joint_store(path)
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        rows = connection.execute(
            """
            SELECT c.condition_id, c.role, c.materialization_json,
                   c.materialization_sha256, c.feature_chunk_sha256,
                   f.chunk_manifest_json, f.chunk_sha256, f.compressed_payload
            FROM round25_joint_condition AS c
            JOIN round25_joint_feature_chunk AS f USING (condition_id)
            WHERE c.admitted
            ORDER BY c.event_start_ms, c.condition_id
            """
        ).fetchall()
    grouped: dict[str, list[Round25JointFeatureSnapshot]] = {
        role: [] for role in _ROLES
    }
    condition_counts: Counter[str] = Counter()
    for row in rows:
        result = _decode_stored_condition(
            str(row[0]),
            condition=(row[2], row[3], row[4]),
            chunk=(row[5], row[6], row[7]),
        )
        lookup = {
            snapshot.decision_time_ms: snapshot
            for snapshot in result.persisted_snapshots
        }
        try:
            endpoints = tuple(
                lookup[decision_time_ms]
                for decision_time_ms in result.selected_endpoint_decision_time_ms
            )
        except KeyError as exc:
            raise ValueError(
                "Round 25 stored endpoint input is unavailable"
            ) from exc
        if (
            row[1] != result.role
            or len(endpoints) != 16
            or tuple(
                snapshot.decision_time_ms
                for snapshot in select_round25_condition_endpoints(
                    result.persisted_snapshots
                )
            )
            != result.selected_endpoint_decision_time_ms
        ):
            raise ValueError("Round 25 stored endpoint input differs")
        grouped[result.role].extend(endpoints)
        condition_counts[result.role] += 1
    after = path.stat()
    if (
        before is None
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or Path(f"{path}.wal").exists()
        or sum(condition_counts.values()) != manifest["admitted_condition_count"]
    ):
        raise RuntimeError("Round 25 joint store changed during endpoint loading")
    return manifest, {
        role: tuple(grouped[role])
        for role in _ROLES
    }


def load_round25_joint_condition_batch(
    database: str | Path,
    condition_ids: Sequence[str],
    *,
    expected_manifest_sha256: str,
) -> tuple[Round25ConditionFeatureMaterialization, ...]:
    """Load at most one TCN minibatch without retaining the full corpus."""

    requested = tuple(str(value or "").strip().lower() for value in condition_ids)
    path = Path(database)
    if (
        not 1 <= len(requested) <= 16
        or len(set(requested)) != len(requested)
        or any(_CONDITION_ID.fullmatch(value) is None for value in requested)
        or _SHA256.fullmatch(expected_manifest_sha256) is None
        or path.is_symlink()
        or not path.is_file()
        or Path(f"{path}.wal").exists()
    ):
        raise ValueError("Round 25 joint condition batch request differs")
    before = path.stat()
    placeholders = ", ".join("?" for _ in requested)
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_tables(connection)
        manifest_rows = connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM round25_joint_store_manifest"
        ).fetchall()
        if len(manifest_rows) != 1:
            raise ValueError("Round 25 joint store manifest row differs")
        manifest = validate_round25_joint_store_manifest(
            _strict_json(manifest_rows[0][0], label="joint store manifest")
        )
        if (
            manifest_rows[0][1] != manifest["manifest_sha256"]
            or manifest["manifest_sha256"] != expected_manifest_sha256
        ):
            raise ValueError("Round 25 joint condition batch manifest differs")
        rows = connection.execute(
            f"""
            SELECT c.condition_id, c.materialization_json,
                   c.materialization_sha256, c.feature_chunk_sha256,
                   f.chunk_manifest_json, f.chunk_sha256, f.compressed_payload
            FROM round25_joint_condition AS c
            JOIN round25_joint_feature_chunk AS f USING (condition_id)
            WHERE c.admitted AND c.condition_id IN ({placeholders})
            """,  # noqa: S608 - placeholders are generated only from bounded arity.
            list(requested),
        ).fetchall()
    decoded = {
        str(row[0]): _decode_stored_condition(
            str(row[0]),
            condition=(row[1], row[2], row[3]),
            chunk=(row[4], row[5], row[6]),
        )
        for row in rows
    }
    after = path.stat()
    if (
        set(decoded) != set(requested)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or Path(f"{path}.wal").exists()
    ):
        raise RuntimeError("Round 25 joint store changed during batch loading")
    return tuple(decoded[condition_id] for condition_id in requested)


def materialize_round25_joint_feature_store(
    *,
    source_database: str | Path,
    destination_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    observed_at_ms: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Materialize one atomic feature store during one terminal receipt audit."""

    source = Path(source_database)
    before = source.stat() if source.is_file() else None
    conditions, source_counts = load_round25_joint_receipt_conditions(
        database=source,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    writer = Round25JointStoreWriter(
        destination_database,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    observer = Round25JointMaterializationObserver(conditions, sink=writer.add)
    try:
        receipt_audit = audit_round25_terminal_receipts(
            database=source,
            terminal_transport_manifest=terminal_transport_manifest,
            observed_at_ms=observed_at_ms,
            observer=observer,
        )
        after = source.stat()
        if (
            before is None
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or Path(f"{source}.wal").exists()
            or observer.condition_count != len(conditions)
        ):
            raise RuntimeError("Round 25 source changed during materialization")
        manifest = writer.finalize(
            terminal_receipt_audit=receipt_audit,
            source_counts=source_counts,
        )
    except Exception:
        writer.abort()
        raise
    return manifest, receipt_audit


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_JOINT_CHUNK_CODEC",
    "POLYMARKET_ROUND25_JOINT_CHUNK_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND25_JOINT_STORE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_JOINT_STORE_SCHEMA_VERSION",
    "Round25ForensicJointStoreWriter",
    "Round25JointStoreWriter",
    "audit_round25_joint_store",
    "load_round25_joint_condition_identities",
    "load_round25_joint_condition_batch",
    "load_round25_joint_endpoint_inputs",
    "load_round25_joint_condition",
    "load_round25_joint_store_manifest",
    "materialize_round25_joint_feature_store",
    "validate_round25_joint_store_manifest",
]

"""Atomic, target-blind storage for the Polymarket Round 21 core corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time
import uuid

import duckdb
import zstandard

from .polymarket_round21_core_features import (
    POLYMARKET_ROUND21_CORE_FEATURE_NAMES,
    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
    Round21CoreFeatureSnapshot,
)
from .polymarket_round21_corpus import (
    POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
    Round21CoreConditionMaterialization,
    Round21CoreCorpusObserver,
    load_round21_core_conditions,
    load_round21_core_corpus_design,
    validate_round21_condition_admission,
)
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_CONDITION_DURATION_MS,
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_DECISION_CADENCE_MS,
    Round21PartitionPolicy,
)
from .polymarket_round21_terminal import (
    audit_round21_terminal_receipts,
    validate_round21_terminal_receipt_audit,
    validate_round21_terminal_transport_manifest,
)
from .storage import write_json_atomic


POLYMARKET_ROUND21_CORE_PUBLICATION_SCHEMA_VERSION = (
    "polymarket-round21-core-corpus-publication-v1"
)
POLYMARKET_ROUND21_CORE_PARTITION_SCHEMA_VERSION = (
    "polymarket-round21-core-corpus-partition-v1"
)
POLYMARKET_ROUND21_CORE_CHUNK_SCHEMA_VERSION = (
    "polymarket-round21-core-feature-chunk-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_MAGIC = b"R21CORE1"
_HEADER = struct.Struct(">8s32sQIH")
_ROW_PREFIX = struct.Struct(">Qdd")
_ROW_SUFFIX = struct.Struct(">32sQ")
_MAXIMUM_CHUNK_BYTES = 32 * 1024 * 1024
_ZSTD_LEVEL = 3
_DEVELOPMENT_ROLES = frozenset(
    {
        "train",
        "purge_train_to_tune",
        "tune_calibration",
        "tune_selection",
    }
)
_SEALED_TEST_ROLES = frozenset({"purge_tune_to_test", "test"})
_PARTITION_ROLES = {
    "development": _DEVELOPMENT_ROLES,
    "sealed_test": _SEALED_TEST_ROLES,
}
_DEVELOPMENT_DATABASE = "development.duckdb"
_SEALED_TEST_DATABASE = "sealed-test.duckdb"
_PUBLICATION_MANIFEST = "publication-manifest.json"
_TERMINAL_AUDIT = "terminal-receipt-audit.json"
_REPLACE_RETRY_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
_PARTITION_TABLES = {
    "condition_admission",
    "feature_chunk",
    "partition_manifest",
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 21 core publication JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 core publication JSON contains {value}")


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


def _read_strict_json_file(path: Path, *, label: str) -> Mapping[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= 8 * 1024 * 1024
    ):
        raise ValueError(f"{label} is unavailable")
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


def _validate_partition_tables(connection: duckdb.DuckDBPyConnection) -> None:
    rows = connection.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    if (
        {str(row[0]) for row in rows} != _PARTITION_TABLES
        or any(str(row[1]) != "BASE TABLE" for row in rows)
    ):
        raise ValueError("Round 21 core partition schema differs")


def _load_stored_partition_manifest(
    database: Path,
) -> dict[str, object]:
    if database.is_symlink() or not database.is_file():
        raise ValueError("Round 21 core partition database is unavailable")
    with duckdb.connect(str(database), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_partition_tables(connection)
        rows = connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM partition_manifest"
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("Round 21 core partition metadata differs")
    try:
        value = json.loads(
            rows[0][0],
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 core partition metadata is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 core partition metadata differs")
    selected = validate_round21_core_partition_manifest(value)
    if (
        rows[0][0] != _canonical_json(selected)
        or rows[0][1] != selected["manifest_sha256"]
    ):
        raise ValueError("Round 21 core partition metadata differs")
    return selected


def _hash_chain(previous: str, value: object) -> str:
    return _canonical_sha256({"previous_sha256": previous, "value": value})


def _partition_for_role(role: str) -> str:
    for partition, roles in _PARTITION_ROLES.items():
        if role in roles:
            return partition
    raise ValueError("Round 21 condition role has no corpus partition")


def _publish_directory(staging: Path, target: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_SECONDS) + 1):
        try:
            os.replace(staging, target)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_SECONDS):
                raise
            time.sleep(_REPLACE_RETRY_SECONDS[attempt])


def _feature_chunk_body(
    *,
    condition_id: str,
    event_start_ms: int,
    role: str,
    row_count: int,
    uncompressed_size: int,
    uncompressed_sha256: str,
    compressed_size: int,
    compressed_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_ROUND21_CORE_CHUNK_SCHEMA_VERSION,
        "core_corpus_design_sha256": POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
        "feature_policy_sha256": POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
        "condition_id": condition_id,
        "event_start_ms": event_start_ms,
        "role": role,
        "row_count": row_count,
        "feature_width": len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES),
        "binary_format": "network_order_ieee754_binary64_v1",
        "codec": "zstd",
        "codec_level": _ZSTD_LEVEL,
        "uncompressed_size": uncompressed_size,
        "uncompressed_sha256": uncompressed_sha256,
        "compressed_size": compressed_size,
        "compressed_sha256": compressed_sha256,
    }


def _validate_feature_chunk_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    expected = {
        "schema_version",
        "core_corpus_design_sha256",
        "feature_policy_sha256",
        "condition_id",
        "event_start_ms",
        "role",
        "row_count",
        "feature_width",
        "binary_format",
        "codec",
        "codec_level",
        "uncompressed_size",
        "uncompressed_sha256",
        "compressed_size",
        "compressed_sha256",
    }
    if (
        set(body) != expected
        or body.get("schema_version")
        != POLYMARKET_ROUND21_CORE_CHUNK_SCHEMA_VERSION
        or body.get("core_corpus_design_sha256")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or body.get("feature_policy_sha256")
        != POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
        or _CONDITION_ID.fullmatch(str(body.get("condition_id") or "")) is None
        or type(body.get("event_start_ms")) is not int
        or body["event_start_ms"] <= 0
        or body["event_start_ms"] % POLYMARKET_ROUND21_CONDITION_DURATION_MS
        or body.get("role") not in _DEVELOPMENT_ROLES | _SEALED_TEST_ROLES
        or type(body.get("row_count")) is not int
        or not 0 <= body["row_count"] <= (
            POLYMARKET_ROUND21_CONDITION_DURATION_MS
            // POLYMARKET_ROUND21_DECISION_CADENCE_MS
        )
        or body.get("feature_width")
        != len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES)
        or body.get("binary_format")
        != "network_order_ieee754_binary64_v1"
        or body.get("codec") != "zstd"
        or body.get("codec_level") != _ZSTD_LEVEL
        or type(body.get("uncompressed_size")) is not int
        or not _HEADER.size <= body["uncompressed_size"] <= _MAXIMUM_CHUNK_BYTES
        or type(body.get("compressed_size")) is not int
        or not 1 <= body["compressed_size"] <= _MAXIMUM_CHUNK_BYTES
        or any(
            _SHA256.fullmatch(str(body.get(field) or "")) is None
            for field in ("uncompressed_sha256", "compressed_sha256")
        )
    ):
        raise ValueError("Round 21 core feature chunk manifest differs")
    return body


def _encode_feature_chunk(
    rows: Sequence[Round21CoreFeatureSnapshot],
    *,
    condition_id: str,
    event_start_ms: int,
) -> tuple[bytes, str, int]:
    width = len(POLYMARKET_ROUND21_CORE_FEATURE_NAMES)
    output = bytearray(
        _HEADER.pack(
            _MAGIC,
            bytes.fromhex(condition_id[2:]),
            event_start_ms,
            len(rows),
            width,
        )
    )
    previous_decision = 0
    value_struct = struct.Struct(f">{width}d")
    for row in rows:
        if (
            not isinstance(row, Round21CoreFeatureSnapshot)
            or not row.available
            or row.condition_id != condition_id
            or row.event_start_ms != event_start_ms
            or row.decision_time_ms <= previous_decision
            or not event_start_ms
            <= row.decision_time_ms
            < event_start_ms + POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or (row.decision_time_ms - event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or len(row.values) != width
        ):
            raise ValueError("Round 21 core feature chunk row differs")
        output.extend(
            _ROW_PREFIX.pack(
                row.decision_time_ms,
                row.structural_probability,
                row.market_prior_probability,
            )
        )
        output.extend(value_struct.pack(*row.values))
        output.extend(
            _ROW_SUFFIX.pack(
                bytes.fromhex(row.source_chain_sha256),
                row.maximum_receipt_ms,
            )
        )
        previous_decision = row.decision_time_ms
    raw = bytes(output)
    if len(raw) > _MAXIMUM_CHUNK_BYTES:
        raise ValueError("Round 21 core feature chunk exceeded its size bound")
    compressed = zstandard.ZstdCompressor(
        level=_ZSTD_LEVEL,
        threads=0,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)
    return compressed, hashlib.sha256(raw).hexdigest(), len(raw)


def _decode_feature_chunk(
    compressed: bytes,
    metadata: Mapping[str, object],
) -> tuple[Round21CoreFeatureSnapshot, ...]:
    selected = _validate_feature_chunk_body(metadata)
    if (
        selected["compressed_size"] != len(compressed)
        or hashlib.sha256(compressed).hexdigest()
        != selected["compressed_sha256"]
    ):
        raise ValueError("Round 21 core feature chunk metadata differs")
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            compressed,
            max_output_size=int(selected["uncompressed_size"]),
        )
    except zstandard.ZstdError as exc:
        raise ValueError("Round 21 core feature chunk is not valid zstd") from exc
    if (
        len(raw) != selected["uncompressed_size"]
        or hashlib.sha256(raw).hexdigest()
        != selected["uncompressed_sha256"]
    ):
        raise ValueError("Round 21 core feature chunk payload differs")
    magic, condition_bytes, event_start_ms, row_count, width = _HEADER.unpack_from(raw)
    condition_id = "0x" + condition_bytes.hex()
    row_size = _ROW_PREFIX.size + width * 8 + _ROW_SUFFIX.size
    if (
        magic != _MAGIC
        or condition_id != selected["condition_id"]
        or event_start_ms != selected["event_start_ms"]
        or row_count != selected["row_count"]
        or width != selected["feature_width"]
        or len(raw) != _HEADER.size + row_count * row_size
    ):
        raise ValueError("Round 21 core feature chunk header differs")
    rows: list[Round21CoreFeatureSnapshot] = []
    offset = _HEADER.size
    value_struct = struct.Struct(f">{width}d")
    previous_decision = 0
    for _index in range(row_count):
        decision, structural, market = _ROW_PREFIX.unpack_from(raw, offset)
        offset += _ROW_PREFIX.size
        values = value_struct.unpack_from(raw, offset)
        offset += value_struct.size
        source_chain, maximum_receipt_ms = _ROW_SUFFIX.unpack_from(raw, offset)
        offset += _ROW_SUFFIX.size
        if (
            decision <= previous_decision
            or not event_start_ms
            <= decision
            < event_start_ms + POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or (decision - event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
        ):
            raise ValueError("Round 21 core feature chunk chronology differs")
        rows.append(
            Round21CoreFeatureSnapshot(
                condition_id=condition_id,
                event_start_ms=event_start_ms,
                decision_time_ms=decision,
                available=True,
                reasons=(),
                structural_probability=structural,
                market_prior_probability=market,
                values=values,
                source_chain_sha256=source_chain.hex(),
                maximum_receipt_ms=maximum_receipt_ms,
            )
        )
        previous_decision = decision
    return tuple(rows)


class _PartitionWriter:
    def __init__(
        self,
        path: Path,
        *,
        partition: str,
        partition_policy: Round21PartitionPolicy,
    ) -> None:
        if partition not in _PARTITION_ROLES:
            raise ValueError("Round 21 core corpus partition differs")
        self.path = path
        self.partition = partition
        self.partition_policy = partition_policy.validated()
        self.connection = duckdb.connect(str(path))
        self.connection.execute("SET memory_limit = '1GB'")
        self.connection.execute("SET threads = 2")
        self.connection.execute("PRAGMA enable_checkpoint_on_shutdown")
        self.connection.execute(
            """
            CREATE TABLE condition_admission (
                condition_id TEXT PRIMARY KEY,
                event_start_ms BIGINT NOT NULL,
                role TEXT NOT NULL,
                admitted BOOLEAN NOT NULL,
                admission_json TEXT NOT NULL,
                admission_sha256 TEXT NOT NULL,
                unavailable_reason_counts_json TEXT NOT NULL,
                feature_chunk_sha256 TEXT
            );
            CREATE TABLE feature_chunk (
                condition_id TEXT PRIMARY KEY,
                chunk_manifest_json TEXT NOT NULL,
                chunk_sha256 TEXT NOT NULL,
                compressed_payload BLOB NOT NULL
            );
            CREATE TABLE partition_manifest (
                singleton BOOLEAN PRIMARY KEY CHECK (singleton),
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL
            );
            """
        )
        self.connection.execute("BEGIN TRANSACTION")
        self._condition_ids: set[str] = set()
        self._logical_root = _EMPTY_SHA256
        self._admission_count = 0
        self._admitted_count = 0
        self._feature_chunk_count = 0
        self._feature_row_count = 0
        self._unavailable_feature_row_count = 0
        self._rejection_reasons: Counter[str] = Counter()
        self._unavailable_reasons: Counter[str] = Counter()
        self._last_identity: tuple[int, str] | None = None
        self._closed = False

    def add(self, materialization: Round21CoreConditionMaterialization) -> None:
        selected = materialization.validated()
        admission = validate_round21_condition_admission(selected.admission)
        condition_id = str(admission["condition_id"])
        role = str(admission["role"])
        identity = (int(admission["event_start_ms"]), condition_id)
        if (
            role not in _PARTITION_ROLES[self.partition]
            or condition_id in self._condition_ids
            or self.partition_policy.role_for_event_start(identity[0]) != role
            or self._last_identity is not None
            and identity <= self._last_identity
        ):
            raise ValueError("Round 21 core corpus partition condition differs")
        self._condition_ids.add(condition_id)
        self._last_identity = identity
        admission_json = _canonical_json(admission)
        unavailable_json = _canonical_json(
            dict(sorted(selected.unavailable_reason_counts.items()))
        )
        chunk_sha256: str | None = None
        if admission["admitted"] is True and role in {
            "train",
            "tune_calibration",
            "tune_selection",
            "test",
        }:
            compressed, raw_sha256, raw_size = _encode_feature_chunk(
                selected.available_features,
                condition_id=condition_id,
                event_start_ms=int(admission["event_start_ms"]),
            )
            chunk_body = _feature_chunk_body(
                condition_id=condition_id,
                event_start_ms=int(admission["event_start_ms"]),
                role=role,
                row_count=len(selected.available_features),
                uncompressed_size=raw_size,
                uncompressed_sha256=raw_sha256,
                compressed_size=len(compressed),
                compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            )
            chunk_sha256 = _canonical_sha256(chunk_body)
            self.connection.execute(
                "INSERT INTO feature_chunk VALUES (?, ?, ?, ?)",
                [
                    condition_id,
                    _canonical_json(
                        {**chunk_body, "chunk_sha256": chunk_sha256}
                    ),
                    chunk_sha256,
                    compressed,
                ],
            )
            self._feature_chunk_count += 1
            self._feature_row_count += len(selected.available_features)
        self.connection.execute(
            "INSERT INTO condition_admission VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                condition_id,
                admission["event_start_ms"],
                role,
                admission["admitted"],
                admission_json,
                admission["admission_sha256"],
                unavailable_json,
                chunk_sha256,
            ],
        )
        self._logical_root = _hash_chain(
            self._logical_root,
            {
                "condition_id": condition_id,
                "event_start_ms": admission["event_start_ms"],
                "role": role,
                "admission_sha256": admission["admission_sha256"],
                "unavailable_reason_counts_sha256": hashlib.sha256(
                    unavailable_json.encode("ascii")
                ).hexdigest(),
                "feature_chunk_sha256": chunk_sha256,
            },
        )
        self._admission_count += 1
        self._admitted_count += int(admission["admitted"] is True)
        self._unavailable_feature_row_count += (
            selected.unavailable_feature_row_count
        )
        self._rejection_reasons.update(admission["rejection_reasons"])
        self._unavailable_reasons.update(selected.unavailable_reason_counts)

    def finalize(self, *, terminal_receipt_audit_sha256: str) -> dict[str, object]:
        if self._closed or _SHA256.fullmatch(terminal_receipt_audit_sha256) is None:
            raise ValueError("Round 21 core corpus finalization differs")
        body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND21_CORE_PARTITION_SCHEMA_VERSION,
            "core_corpus_design_sha256": POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "feature_policy_sha256": POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
            "partition_policy_sha256": self.partition_policy.policy_sha256,
            "campaign_start_ms": self.partition_policy.campaign_start_ms,
            "campaign_end_ms": self.partition_policy.campaign_end_ms,
            "terminal_receipt_audit_sha256": terminal_receipt_audit_sha256,
            "partition": self.partition,
            "roles": sorted(_PARTITION_ROLES[self.partition]),
            "admission_count": self._admission_count,
            "admitted_condition_count": self._admitted_count,
            "feature_chunk_count": self._feature_chunk_count,
            "feature_row_count": self._feature_row_count,
            "unavailable_feature_row_count": self._unavailable_feature_row_count,
            "rejection_reason_counts": dict(sorted(self._rejection_reasons.items())),
            "unavailable_reason_counts": dict(sorted(self._unavailable_reasons.items())),
            "logical_root_sha256": self._logical_root,
            "raw_payloads_copied": False,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "optional_binance_consulted": False,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        manifest = {**body, "manifest_sha256": _canonical_sha256(body)}
        self.connection.execute(
            "INSERT INTO partition_manifest VALUES (TRUE, ?, ?)",
            [_canonical_json(manifest), manifest["manifest_sha256"]],
        )
        self.connection.execute("COMMIT")
        self.connection.execute("CHECKPOINT")
        self.connection.close()
        self._closed = True
        return manifest

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self.connection.execute("ROLLBACK")
        except duckdb.Error:
            pass
        self.connection.close()
        self._closed = True


def validate_round21_core_partition_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    expected = {
        "schema_version",
        "core_corpus_design_sha256",
        "dataset_design_sha256",
        "feature_policy_sha256",
        "partition_policy_sha256",
        "campaign_start_ms",
        "campaign_end_ms",
        "terminal_receipt_audit_sha256",
        "partition",
        "roles",
        "admission_count",
        "admitted_condition_count",
        "feature_chunk_count",
        "feature_row_count",
        "unavailable_feature_row_count",
        "rejection_reason_counts",
        "unavailable_reason_counts",
        "logical_root_sha256",
        "raw_payloads_copied",
        "outcomes_consulted",
        "model_scores_consulted",
        "optional_binance_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    partition = str(payload.get("partition") or "")
    count_fields = (
        "admission_count",
        "admitted_condition_count",
        "feature_chunk_count",
        "feature_row_count",
        "unavailable_feature_row_count",
    )
    count_maps = (
        payload.get("rejection_reason_counts"),
        payload.get("unavailable_reason_counts"),
    )
    if (
        type(payload.get("campaign_start_ms")) is not int
        or type(payload.get("campaign_end_ms")) is not int
    ):
        raise ValueError("Round 21 core partition policy differs")
    try:
        policy = Round21PartitionPolicy.create(
            campaign_start_ms=payload["campaign_start_ms"],
            campaign_end_ms=payload["campaign_end_ms"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 21 core partition policy differs") from exc
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_CORE_PARTITION_SCHEMA_VERSION
        or payload.get("core_corpus_design_sha256")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or payload.get("dataset_design_sha256")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or payload.get("feature_policy_sha256")
        != POLYMARKET_ROUND21_FEATURE_POLICY_SHA256
        or _SHA256.fullmatch(str(payload.get("partition_policy_sha256") or ""))
        is None
        or payload.get("partition_policy_sha256") != policy.policy_sha256
        or _SHA256.fullmatch(
            str(payload.get("terminal_receipt_audit_sha256") or "")
        )
        is None
        or payload.get("terminal_receipt_audit_sha256") == _EMPTY_SHA256
        or partition not in _PARTITION_ROLES
        or payload.get("roles") != sorted(_PARTITION_ROLES[partition])
        or any(
            type(payload.get(field)) is not int or payload[field] < 0
            for field in count_fields
        )
        or payload["admission_count"] <= 0
        or payload["admitted_condition_count"] > payload["admission_count"]
        or payload["feature_chunk_count"] > payload["admitted_condition_count"]
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
        or payload.get("raw_payloads_copied") is not False
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
        raise ValueError("Round 21 core partition manifest differs")
    return {**payload, "manifest_sha256": claimed}


def _publication_manifest(
    *,
    transport_manifest_sha256: str,
    terminal_audit: Mapping[str, object],
    development: Mapping[str, object],
    sealed_test: Mapping[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_CORE_PUBLICATION_SCHEMA_VERSION,
        "core_corpus_design_sha256": POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
        "created_at_ms": terminal_audit["created_at_ms"],
        "terminal_transport_manifest_sha256": transport_manifest_sha256,
        "terminal_receipt_audit_sha256": terminal_audit["audit_sha256"],
        "development_database": _DEVELOPMENT_DATABASE,
        "sealed_test_database": _SEALED_TEST_DATABASE,
        "terminal_receipt_audit": _TERMINAL_AUDIT,
        "development_partition": dict(development),
        "sealed_test_partition": dict(sealed_test),
        "sealed_test_population_manifest_sha256": sealed_test["manifest_sha256"],
        "physical_partition_separation": True,
        "atomic_directory_publication": True,
        "raw_payloads_copied": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "optional_binance_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def validate_round21_core_publication_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    expected = {
        "schema_version",
        "core_corpus_design_sha256",
        "created_at_ms",
        "terminal_transport_manifest_sha256",
        "terminal_receipt_audit_sha256",
        "development_database",
        "sealed_test_database",
        "terminal_receipt_audit",
        "development_partition",
        "sealed_test_partition",
        "sealed_test_population_manifest_sha256",
        "physical_partition_separation",
        "atomic_directory_publication",
        "raw_payloads_copied",
        "outcomes_consulted",
        "model_scores_consulted",
        "optional_binance_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    development = payload.get("development_partition")
    sealed = payload.get("sealed_test_partition")
    try:
        development_manifest = validate_round21_core_partition_manifest(
            development if isinstance(development, Mapping) else {}
        )
        sealed_manifest = validate_round21_core_partition_manifest(
            sealed if isinstance(sealed, Mapping) else {}
        )
    except ValueError as exc:
        raise ValueError("Round 21 core publication partition differs") from exc
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_CORE_PUBLICATION_SCHEMA_VERSION
        or payload.get("core_corpus_design_sha256")
        != POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
        or type(payload.get("created_at_ms")) is not int
        or payload["created_at_ms"] <= 0
        or any(
            _SHA256.fullmatch(str(payload.get(field) or "")) is None
            for field in (
                "terminal_transport_manifest_sha256",
                "terminal_receipt_audit_sha256",
                "sealed_test_population_manifest_sha256",
            )
        )
        or payload.get("development_database") != _DEVELOPMENT_DATABASE
        or payload.get("sealed_test_database") != _SEALED_TEST_DATABASE
        or payload.get("terminal_receipt_audit") != _TERMINAL_AUDIT
        or development_manifest["partition"] != "development"
        or sealed_manifest["partition"] != "sealed_test"
        or development_manifest["terminal_receipt_audit_sha256"]
        != payload["terminal_receipt_audit_sha256"]
        or sealed_manifest["terminal_receipt_audit_sha256"]
        != payload["terminal_receipt_audit_sha256"]
        or development_manifest["partition_policy_sha256"]
        != sealed_manifest["partition_policy_sha256"]
        or development_manifest["campaign_start_ms"]
        != sealed_manifest["campaign_start_ms"]
        or development_manifest["campaign_end_ms"]
        != sealed_manifest["campaign_end_ms"]
        or payload["sealed_test_population_manifest_sha256"]
        != sealed_manifest["manifest_sha256"]
        or payload.get("physical_partition_separation") is not True
        or payload.get("atomic_directory_publication") is not True
        or payload.get("raw_payloads_copied") is not False
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
        raise ValueError("Round 21 core publication manifest differs")
    return {**payload, "manifest_sha256": claimed}


def _audit_round21_core_partition(
    database: str | Path,
    expected_manifest: Mapping[str, object],
    *,
    allow_sealed_test: bool,
    feature_sink: list[Round21CoreFeatureSnapshot] | None = None,
) -> dict[str, object]:
    expected = validate_round21_core_partition_manifest(expected_manifest)
    if expected["partition"] == "sealed_test" and not allow_sealed_test:
        raise PermissionError("Round 21 sealed-test features require one-use access")
    path = Path(database)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Round 21 core partition database is unavailable")
    logical_root = _EMPTY_SHA256
    admission_count = 0
    admitted_count = 0
    feature_chunk_count = 0
    feature_row_count = 0
    unavailable_count = 0
    rejection_reasons: Counter[str] = Counter()
    unavailable_reasons: Counter[str] = Counter()
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=int(expected["campaign_start_ms"]),
        campaign_end_ms=int(expected["campaign_end_ms"]),
    )
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")
        _validate_partition_tables(connection)
        manifest_rows = connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM partition_manifest"
        ).fetchall()
        if len(manifest_rows) != 1:
            raise ValueError("Round 21 core partition metadata differs")
        try:
            stored = json.loads(
                manifest_rows[0][0],
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 21 core partition metadata is not strict JSON") from exc
        if (
            not isinstance(stored, Mapping)
            or validate_round21_core_partition_manifest(stored) != expected
            or manifest_rows[0][1] != expected["manifest_sha256"]
        ):
            raise ValueError("Round 21 core partition metadata differs")
        rows = connection.execute(
            """
            SELECT a.condition_id, a.event_start_ms, a.role, a.admitted,
                   a.admission_json, a.admission_sha256,
                   a.unavailable_reason_counts_json, a.feature_chunk_sha256,
                   c.chunk_manifest_json, c.chunk_sha256, c.compressed_payload
            FROM condition_admission AS a
            LEFT JOIN feature_chunk AS c USING (condition_id)
            ORDER BY a.event_start_ms, a.condition_id
            """
        ).fetchall()
        for row in rows:
            (
                condition_id,
                event_start_ms,
                role,
                admitted,
                admission_json,
                admission_sha256,
                unavailable_json,
                feature_sha,
                chunk_json,
                chunk_sha,
                compressed,
            ) = row
            try:
                admission_value = json.loads(
                    admission_json,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_nonfinite,
                )
                unavailable_value = json.loads(
                    unavailable_json,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_nonfinite,
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Round 21 core partition row is not strict JSON") from exc
            admission = validate_round21_condition_admission(admission_value)
            if (
                admission_json != _canonical_json(admission)
                or unavailable_json != _canonical_json(unavailable_value)
                or condition_id != admission["condition_id"]
                or event_start_ms != admission["event_start_ms"]
                or role != admission["role"]
                or admitted is not admission["admitted"]
                or admission_sha256 != admission["admission_sha256"]
                or role not in _PARTITION_ROLES[expected["partition"]]
                or policy.role_for_event_start(event_start_ms) != role
                or not isinstance(unavailable_value, Mapping)
                or any(
                    not isinstance(reason, str)
                    or not reason
                    or type(count) is not int
                    or count <= 0
                    for reason, count in unavailable_value.items()
                )
            ):
                raise ValueError("Round 21 core partition admission differs")
            has_chunk = feature_sha is not None
            should_have_chunk = admitted and role in {
                "train",
                "tune_calibration",
                "tune_selection",
                "test",
            }
            if has_chunk != should_have_chunk:
                raise ValueError("Round 21 core partition chunk presence differs")
            if has_chunk:
                try:
                    chunk_value = json.loads(
                        chunk_json,
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_nonfinite,
                    )
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Round 21 core chunk manifest is not strict JSON") from exc
                if not isinstance(chunk_value, Mapping):
                    raise ValueError("Round 21 core chunk manifest differs")
                chunk_body = dict(chunk_value)
                claimed_chunk = str(chunk_body.pop("chunk_sha256", ""))
                if (
                    chunk_json
                    != _canonical_json(
                        {**chunk_body, "chunk_sha256": claimed_chunk}
                    )
                    or claimed_chunk != _canonical_sha256(chunk_body)
                    or claimed_chunk != feature_sha
                    or claimed_chunk != chunk_sha
                    or chunk_body.get("condition_id") != condition_id
                    or chunk_body.get("event_start_ms") != event_start_ms
                    or chunk_body.get("role") != role
                ):
                    raise ValueError("Round 21 core chunk manifest differs")
                decoded = _decode_feature_chunk(bytes(compressed), chunk_body)
                if len(decoded) != admission["available_feature_row_count"]:
                    raise ValueError("Round 21 core chunk feature count differs")
                if feature_sink is not None:
                    feature_sink.extend(decoded)
                feature_chunk_count += 1
                feature_row_count += len(decoded)
            logical_root = _hash_chain(
                logical_root,
                {
                    "condition_id": condition_id,
                    "event_start_ms": event_start_ms,
                    "role": role,
                    "admission_sha256": admission_sha256,
                    "unavailable_reason_counts_sha256": hashlib.sha256(
                        unavailable_json.encode("ascii")
                    ).hexdigest(),
                    "feature_chunk_sha256": feature_sha,
                },
            )
            admission_count += 1
            admitted_count += int(admitted)
            unavailable_count += admission["unavailable_feature_row_count"]
            rejection_reasons.update(admission["rejection_reasons"])
            unavailable_reasons.update(unavailable_value)
        stored_chunk_count = connection.execute(
            "SELECT count(*) FROM feature_chunk"
        ).fetchone()[0]
        if stored_chunk_count != feature_chunk_count:
            raise ValueError("Round 21 core partition has orphan feature chunks")
    observed = {
        "admission_count": admission_count,
        "admitted_condition_count": admitted_count,
        "feature_chunk_count": feature_chunk_count,
        "feature_row_count": feature_row_count,
        "unavailable_feature_row_count": unavailable_count,
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "unavailable_reason_counts": dict(sorted(unavailable_reasons.items())),
        "logical_root_sha256": logical_root,
    }
    if any(expected[key] != value for key, value in observed.items()):
        raise ValueError("Round 21 core partition logical audit differs")
    return {**observed, "partition_manifest_sha256": expected["manifest_sha256"]}


def audit_round21_core_partition(
    database: str | Path,
    expected_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute a development root; sealed-test artifacts stay inaccessible."""

    return _audit_round21_core_partition(
        database,
        expected_manifest,
        allow_sealed_test=False,
    )


def load_round21_core_partition_snapshots(
    database: str | Path,
    expected_manifest: Mapping[str, object],
) -> tuple[Round21CoreFeatureSnapshot, ...]:
    """Load an audited development partition in one ordered storage pass."""

    snapshots: list[Round21CoreFeatureSnapshot] = []
    audit = _audit_round21_core_partition(
        database,
        expected_manifest,
        allow_sealed_test=False,
        feature_sink=snapshots,
    )
    if len(snapshots) != audit["feature_row_count"]:
        raise ValueError("Round 21 core partition loaded feature count differs")
    return tuple(snapshots)


def _authorize_round21_sealed_core_partition(
    expected_manifest: Mapping[str, object],
    *,
    one_use_store_path: str | Path,
    claim: object,
    test_access_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from .polymarket_round21_one_use import (  # local import avoids a module cycle
        Round21OneUseClaim,
        Round21OneUseStore,
    )

    if not isinstance(claim, Round21OneUseClaim):
        raise TypeError("Round 21 sealed feature claim type differs")
    selected = claim.validated()
    expected = validate_round21_core_partition_manifest(expected_manifest)
    if (
        expected["partition"] != "sealed_test"
        or expected["manifest_sha256"]
        != selected.sealed_test_population_manifest_sha256
    ):
        raise PermissionError("Round 21 sealed feature population differs")
    with Round21OneUseStore(one_use_store_path) as store:
        authorization = store.authorize_test_feature_access(
            selected,
            test_access_sha256=test_access_sha256,
        )
    return expected, authorization


def audit_round21_sealed_core_partition(
    database: str | Path,
    expected_manifest: Mapping[str, object],
    *,
    one_use_store_path: str | Path,
    claim: object,
    test_access_sha256: str,
) -> dict[str, object]:
    """Audit sealed features only through the durable one-use access ledger."""

    expected, authorization = _authorize_round21_sealed_core_partition(
        expected_manifest,
        one_use_store_path=one_use_store_path,
        claim=claim,
        test_access_sha256=test_access_sha256,
    )
    audit = _audit_round21_core_partition(
        database,
        expected,
        allow_sealed_test=True,
    )
    return {**audit, **authorization}


def load_round21_sealed_core_partition_snapshots(
    database: str | Path,
    expected_manifest: Mapping[str, object],
    *,
    one_use_store_path: str | Path,
    claim: object,
    test_access_sha256: str,
) -> tuple[Round21CoreFeatureSnapshot, ...]:
    """Load sealed snapshots once through the consumed one-use access record."""

    expected, _authorization = _authorize_round21_sealed_core_partition(
        expected_manifest,
        one_use_store_path=one_use_store_path,
        claim=claim,
        test_access_sha256=test_access_sha256,
    )
    snapshots: list[Round21CoreFeatureSnapshot] = []
    audit = _audit_round21_core_partition(
        database,
        expected,
        allow_sealed_test=True,
        feature_sink=snapshots,
    )
    if len(snapshots) != audit["feature_row_count"]:
        raise ValueError("Round 21 sealed partition loaded feature count differs")
    return tuple(snapshots)


def publish_round21_core_corpus(
    *,
    repository: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    publication_directory: str | Path,
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Publish development and sealed-test core artifacts as one atomic unit."""

    load_round21_core_corpus_design(repository)
    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    target = Path(publication_directory).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Round 21 core publication destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=int(transport["campaign_start_ms"]),
        campaign_end_ms=int(transport["campaign_end_ms"]),
    )
    development: _PartitionWriter | None = None
    sealed: _PartitionWriter | None = None
    try:
        conditions = load_round21_core_conditions(
            database=source_database,
            terminal_transport_manifest=transport,
        )
        development = _PartitionWriter(
            staging / _DEVELOPMENT_DATABASE,
            partition="development",
            partition_policy=policy,
        )
        sealed = _PartitionWriter(
            staging / _SEALED_TEST_DATABASE,
            partition="sealed_test",
            partition_policy=policy,
        )

        def sink(materialization: Round21CoreConditionMaterialization) -> None:
            role = str(materialization.admission["role"])
            if _partition_for_role(role) == "development":
                if development is None:
                    raise RuntimeError("Round 21 development writer is unavailable")
                development.add(materialization)
            else:
                if sealed is None:
                    raise RuntimeError("Round 21 sealed writer is unavailable")
                sealed.add(materialization)

        observer = Round21CoreCorpusObserver(
            conditions=conditions,
            partition_policy=policy,
            sink=sink,
        )
        terminal_audit = audit_round21_terminal_receipts(
            database=source_database,
            terminal_transport_manifest=transport,
            observed_at_ms=observed_at_ms,
            observer=observer,
        )
        terminal_audit = validate_round21_terminal_receipt_audit(
            terminal_audit,
            terminal_transport_manifest=transport,
        )
        if observer.materialized_condition_count != len(conditions):
            raise RuntimeError("Round 21 core corpus condition accounting differs")
        development_manifest = development.finalize(
            terminal_receipt_audit_sha256=str(terminal_audit["audit_sha256"])
        )
        sealed_manifest = sealed.finalize(
            terminal_receipt_audit_sha256=str(terminal_audit["audit_sha256"])
        )
        development = None
        sealed = None
        if (
            development_manifest["admission_count"] <= 0
            or sealed_manifest["admission_count"] <= 0
        ):
            raise RuntimeError("Round 21 core corpus partition is empty")
        manifest = validate_round21_core_publication_manifest(
            _publication_manifest(
                transport_manifest_sha256=str(transport["manifest_sha256"]),
                terminal_audit=terminal_audit,
                development=development_manifest,
                sealed_test=sealed_manifest,
            )
        )
        write_json_atomic(
            staging / _TERMINAL_AUDIT,
            terminal_audit,
            indent=2,
            sort_keys=True,
        )
        write_json_atomic(
            staging / _PUBLICATION_MANIFEST,
            manifest,
            indent=2,
            sort_keys=True,
        )
        _audit_round21_core_partition(
            staging / _DEVELOPMENT_DATABASE,
            development_manifest,
            allow_sealed_test=False,
        )
        _audit_round21_core_partition(
            staging / _SEALED_TEST_DATABASE,
            sealed_manifest,
            allow_sealed_test=True,
        )
        _publish_directory(staging, target)
        return manifest
    except Exception:
        if development is not None:
            development.abort()
        if sealed is not None:
            sealed.abort()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_round21_core_publication_manifest(
    publication_directory: str | Path,
) -> dict[str, object]:
    root = Path(publication_directory)
    path = root / _PUBLICATION_MANIFEST
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Round 21 core publication is unavailable")
    value = _read_strict_json_file(path, label="Round 21 core publication")
    return validate_round21_core_publication_manifest(value)


def _validate_round21_core_publication_boundary(
    publication_directory: str | Path,
    *,
    development_feature_sink: list[Round21CoreFeatureSnapshot] | None = None,
) -> dict[str, object]:
    selected_root = Path(publication_directory)
    if selected_root.is_symlink():
        raise ValueError("Round 21 core publication is unavailable")
    root = selected_root.resolve()
    manifest = load_round21_core_publication_manifest(root)
    development_path = root / str(manifest["development_database"])
    sealed_path = root / str(manifest["sealed_test_database"])
    audit_path = root / str(manifest["terminal_receipt_audit"])
    development_manifest = validate_round21_core_partition_manifest(
        manifest["development_partition"]
    )
    sealed_manifest = validate_round21_core_partition_manifest(
        manifest["sealed_test_partition"]
    )
    _audit_round21_core_partition(
        development_path,
        development_manifest,
        allow_sealed_test=False,
        feature_sink=development_feature_sink,
    )
    if _load_stored_partition_manifest(sealed_path) != sealed_manifest:
        raise ValueError("Round 21 sealed partition metadata differs")
    terminal_audit = dict(
        _read_strict_json_file(
            audit_path,
            label="Round 21 terminal receipt audit",
        )
    )
    claimed = str(terminal_audit.pop("audit_sha256", "")).strip().lower()
    if (
        claimed != _canonical_sha256(terminal_audit)
        or claimed != manifest["terminal_receipt_audit_sha256"]
        or terminal_audit.get("terminal_transport_manifest_sha256")
        != manifest["terminal_transport_manifest_sha256"]
        or terminal_audit.get("receipt_replay_complete") is not True
        or terminal_audit.get("condition_admission_pending") is not True
        or terminal_audit.get("outcomes_consulted") is not False
        or terminal_audit.get("model_scores_consulted") is not False
        or any(
            terminal_audit.get(field) is not False
            for field in (
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 terminal receipt audit boundary differs")
    return manifest


def validate_round21_core_publication_boundary(
    publication_directory: str | Path,
) -> dict[str, object]:
    """Validate a publication without opening sealed-test feature chunks."""

    return _validate_round21_core_publication_boundary(publication_directory)


def load_round21_core_development_publication(
    publication_directory: str | Path,
) -> tuple[dict[str, object], tuple[Round21CoreFeatureSnapshot, ...]]:
    """Validate and load development snapshots in one ordered storage pass."""

    snapshots: list[Round21CoreFeatureSnapshot] = []
    manifest = _validate_round21_core_publication_boundary(
        publication_directory,
        development_feature_sink=snapshots,
    )
    development = validate_round21_core_partition_manifest(
        manifest["development_partition"]
    )
    if len(snapshots) != development["feature_row_count"]:
        raise ValueError("Round 21 development publication feature count differs")
    return manifest, tuple(snapshots)


__all__ = [
    "POLYMARKET_ROUND21_CORE_CHUNK_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_CORE_PARTITION_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_CORE_PUBLICATION_SCHEMA_VERSION",
    "audit_round21_core_partition",
    "audit_round21_sealed_core_partition",
    "load_round21_core_development_publication",
    "load_round21_core_publication_manifest",
    "load_round21_core_partition_snapshots",
    "load_round21_sealed_core_partition_snapshots",
    "publish_round21_core_corpus",
    "validate_round21_core_publication_boundary",
    "validate_round21_core_partition_manifest",
    "validate_round21_core_publication_manifest",
]

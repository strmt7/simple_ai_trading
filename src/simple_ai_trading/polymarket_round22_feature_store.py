"""Compressed same-database storage for Round 22 causal feature grids."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

import zstandard

from .polymarket_round22_features import (
    POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
    POLYMARKET_ROUND22_FEATURE_NAMES,
    POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
    POLYMARKET_ROUND22_FEATURE_SCHEMA_VERSION,
    POLYMARKET_ROUND22_TABULAR_CADENCE_MS,
    Round22CausalFeatureRow,
    Round22ConditionFeatures,
)
from .polymarket_round22_pilot import Round22PilotStore


POLYMARKET_ROUND22_FEATURE_STORE_SCHEMA_VERSION = "polymarket-round22-feature-store-v2"
POLYMARKET_ROUND22_FEATURE_CHUNK_SCHEMA_VERSION = (
    "polymarket-round22-condition-feature-chunk-v1"
)
POLYMARKET_ROUND22_FEATURE_CHUNK_CODEC = "canonical-json-zstd-3"
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAXIMUM_RAW_CHUNK_BYTES = 128 * 1024 * 1024
_ROW_FIELDS = frozenset(
    {
        "available",
        "decision_time_ms",
        "down_source_timestamp_ms",
        "feature_values",
        "feature_values_sha256",
        "reasons",
        "role",
        "sequence_complete",
        "source_chain_sha256",
        "tabular_anchor",
        "tabular_history_complete",
        "up_source_timestamp_ms",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 22 feature-store JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 22 feature-store JSON contains {value}")


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


def _row_chain(previous: str, row_sha256: str) -> str:
    return hashlib.sha256(
        bytes.fromhex(previous) + bytes.fromhex(row_sha256)
    ).hexdigest()


def _row_payload(
    row: Round22CausalFeatureRow,
    *,
    role: str,
) -> tuple[dict[str, object], str]:
    values = list(row.values)
    values_sha = _canonical_sha256(values)
    payload = {
        "available": row.available,
        "decision_time_ms": row.decision_time_ms,
        "down_source_timestamp_ms": row.down_source_timestamp_ms,
        "feature_values": values,
        "feature_values_sha256": values_sha,
        "reasons": list(row.reasons),
        "role": role,
        "sequence_complete": row.sequence_complete,
        "source_chain_sha256": row.source_chain_sha256,
        "tabular_anchor": row.tabular_anchor,
        "tabular_history_complete": row.tabular_history_complete,
        "up_source_timestamp_ms": row.up_source_timestamp_ms,
    }
    identity = dict(payload)
    identity.pop("feature_values")
    return payload, _canonical_sha256(identity)


def _manifest_body(
    *,
    condition_id: str,
    role: str,
    source_manifest_sha256: str,
    row_count: int,
    available_count: int,
    sequence_count: int,
    tabular_count: int,
    row_chain_sha256: str,
    raw_size_bytes: int,
    compressed_size_bytes: int,
    raw_sha256: str,
    compressed_sha256: str,
) -> dict[str, object]:
    return {
        "available_count": available_count,
        "chunk": {
            "codec": POLYMARKET_ROUND22_FEATURE_CHUNK_CODEC,
            "compressed_sha256": compressed_sha256,
            "compressed_size_bytes": compressed_size_bytes,
            "raw_sha256": raw_sha256,
            "raw_size_bytes": raw_size_bytes,
        },
        "condition_id": condition_id,
        "feature_names_sha256": POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
        "feature_policy_sha256": POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
        "role": role,
        "row_chain_sha256": row_chain_sha256,
        "row_count": row_count,
        "sequence_complete_count": sequence_count,
        "source_condition_manifest_sha256": source_manifest_sha256,
        "tabular_history_complete_count": tabular_count,
    }


class Round22FeatureStore:
    """Feature-only facade over the pilot DuckDB; targets are never returned."""

    def __init__(self, pilot_store: Round22PilotStore) -> None:
        self.pilot_store = pilot_store
        self.connection = pilot_store.connection
        self.read_only = pilot_store.read_only
        if not self.read_only:
            self._initialize()
        self._verify_schema()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feature.causal_feature_schema (
                singleton BOOLEAN PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                feature_schema_version VARCHAR NOT NULL,
                feature_policy_sha256 VARCHAR NOT NULL,
                feature_names_json VARCHAR NOT NULL,
                feature_names_sha256 VARCHAR NOT NULL,
                feature_cadence_ms INTEGER NOT NULL,
                tabular_cadence_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feature.condition_feature_chunk (
                condition_id VARCHAR PRIMARY KEY,
                role VARCHAR NOT NULL,
                source_condition_manifest_sha256 VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                available_count BIGINT NOT NULL,
                sequence_complete_count BIGINT NOT NULL,
                tabular_history_complete_count BIGINT NOT NULL,
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
        names_json = _canonical_json(list(POLYMARKET_ROUND22_FEATURE_NAMES))
        self.connection.execute(
            """
            INSERT INTO feature.causal_feature_schema
            SELECT true, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM feature.causal_feature_schema)
            """,
            [
                POLYMARKET_ROUND22_FEATURE_STORE_SCHEMA_VERSION,
                POLYMARKET_ROUND22_FEATURE_SCHEMA_VERSION,
                POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
                names_json,
                POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
                POLYMARKET_ROUND22_FEATURE_CADENCE_MS,
                POLYMARKET_ROUND22_TABULAR_CADENCE_MS,
            ],
        )

    def _verify_schema(self) -> None:
        row = self.connection.execute(
            """
            SELECT schema_version, feature_schema_version,
                   feature_policy_sha256, feature_names_json,
                   feature_names_sha256, feature_cadence_ms,
                   tabular_cadence_ms
            FROM feature.causal_feature_schema WHERE singleton
            """
        ).fetchone()
        if row is None:
            raise ValueError("Round 22 feature-store schema is unavailable")
        try:
            names = json.loads(
                str(row[3]),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 22 feature-store names are invalid") from exc
        if (
            row[0] != POLYMARKET_ROUND22_FEATURE_STORE_SCHEMA_VERSION
            or row[1] != POLYMARKET_ROUND22_FEATURE_SCHEMA_VERSION
            or row[2] != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
            or names != list(POLYMARKET_ROUND22_FEATURE_NAMES)
            or str(row[3]) != _canonical_json(names)
            or row[4] != POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
            or int(row[5]) != POLYMARKET_ROUND22_FEATURE_CADENCE_MS
            or int(row[6]) != POLYMARKET_ROUND22_TABULAR_CADENCE_MS
        ):
            raise ValueError("Round 22 feature-store schema differs")

    def completed_condition_ids(self) -> frozenset[str]:
        rows = self.connection.execute(
            "SELECT condition_id FROM feature.condition_feature_chunk"
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def _source_identity(
        self,
        result: Round22ConditionFeatures,
    ) -> tuple[str, str]:
        identity = self.connection.execute(
            """
            SELECT role, event_start_ms, event_end_ms
            FROM feature.market_identity WHERE condition_id = ?
            """,
            [result.condition_id],
        ).fetchone()
        source = self.connection.execute(
            """
            SELECT manifest_sha256 FROM feature.condition_manifest
            WHERE condition_id = ?
            """,
            [result.condition_id],
        ).fetchone()
        if (
            self.pilot_store.target_row_count() != 0
            or identity is None
            or source is None
            or str(identity[0]) not in {"train", "tune_calibration", "tune_selection"}
            or int(identity[1]) != result.event_start_ms
            or int(identity[2]) != result.event_end_ms
            or _SHA256.fullmatch(str(source[0])) is None
        ):
            raise ValueError("Round 22 feature source identity differs")
        self.pilot_store.audit_condition(result.condition_id)
        return str(identity[0]), str(source[0])

    def put_condition(self, result: Round22ConditionFeatures) -> bool:
        if self.read_only:
            raise ValueError("Round 22 feature store is read-only")
        if not isinstance(result, Round22ConditionFeatures):
            raise TypeError("Round 22 feature result type differs")
        role, source_manifest = self._source_identity(result)
        row_payloads: list[dict[str, object]] = []
        chain = _EMPTY_SHA256
        for row in result.rows:
            payload, row_sha = _row_payload(row, role=role)
            row_payloads.append(payload)
            chain = _row_chain(chain, row_sha)
        body = {
            "authority": {
                "binance_used": False,
                "target_accessed": False,
                "trading_authority": False,
            },
            "condition_id": result.condition_id,
            "event_end_ms": result.event_end_ms,
            "event_start_ms": result.event_start_ms,
            "feature_names_sha256": result.feature_names_sha256,
            "feature_policy_sha256": result.policy_sha256,
            "rows": row_payloads,
            "schema_version": POLYMARKET_ROUND22_FEATURE_CHUNK_SCHEMA_VERSION,
        }
        raw = _canonical_json(body).encode("ascii")
        if not 2 <= len(raw) <= _MAXIMUM_RAW_CHUNK_BYTES:
            raise ValueError("Round 22 feature chunk size is outside the bound")
        compressed = zstandard.ZstdCompressor(
            level=3,
            write_checksum=True,
            write_content_size=True,
        ).compress(raw)
        counts = {
            "available": sum(row.available for row in result.rows),
            "sequence": sum(row.sequence_complete for row in result.rows),
            "tabular": sum(row.tabular_history_complete for row in result.rows),
        }
        raw_sha = hashlib.sha256(raw).hexdigest()
        compressed_sha = hashlib.sha256(compressed).hexdigest()
        manifest_body = _manifest_body(
            condition_id=result.condition_id,
            role=role,
            source_manifest_sha256=source_manifest,
            row_count=len(result.rows),
            available_count=counts["available"],
            sequence_count=counts["sequence"],
            tabular_count=counts["tabular"],
            row_chain_sha256=chain,
            raw_size_bytes=len(raw),
            compressed_size_bytes=len(compressed),
            raw_sha256=raw_sha,
            compressed_sha256=compressed_sha,
        )
        manifest_sha = _canonical_sha256(manifest_body)
        existing = self.connection.execute(
            """
            SELECT manifest_sha256 FROM feature.condition_feature_chunk
            WHERE condition_id = ?
            """,
            [result.condition_id],
        ).fetchone()
        if existing is not None:
            if existing[0] != manifest_sha:
                raise ValueError("Round 22 existing feature manifest differs")
            return False
        self.connection.execute(
            """
            INSERT INTO feature.condition_feature_chunk VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                result.condition_id,
                role,
                source_manifest,
                len(result.rows),
                counts["available"],
                counts["sequence"],
                counts["tabular"],
                chain,
                POLYMARKET_ROUND22_FEATURE_CHUNK_CODEC,
                len(raw),
                len(compressed),
                raw_sha,
                compressed_sha,
                compressed,
                manifest_sha,
            ],
        )
        return True

    def _chunk_row(self, condition_id: str) -> tuple[object, ...]:
        selected = str(condition_id or "").strip().lower()
        if _CONDITION_ID.fullmatch(selected) is None:
            raise ValueError("Round 22 feature audit condition differs")
        row = self.connection.execute(
            """
            SELECT role, source_condition_manifest_sha256, row_count,
                   available_count, sequence_complete_count,
                   tabular_history_complete_count, row_chain_sha256, codec,
                   raw_size_bytes, compressed_size_bytes, raw_sha256,
                   compressed_sha256, payload, manifest_sha256
            FROM feature.condition_feature_chunk WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        if row is None:
            raise ValueError("Round 22 feature chunk is unavailable")
        return row

    def load_condition_rows(
        self,
        condition_id: str,
    ) -> tuple[Round22CausalFeatureRow, ...]:
        selected = str(condition_id or "").strip().lower()
        row = self._chunk_row(selected)
        raw_size = int(row[8])
        compressed = bytes(row[12])
        if (
            row[7] != POLYMARKET_ROUND22_FEATURE_CHUNK_CODEC
            or not 2 <= raw_size <= _MAXIMUM_RAW_CHUNK_BYTES
            or int(row[9]) != len(compressed)
            or str(row[11]) != hashlib.sha256(compressed).hexdigest()
        ):
            raise ValueError("Round 22 feature chunk envelope differs")
        try:
            raw = zstandard.ZstdDecompressor().decompress(
                compressed,
                max_output_size=raw_size,
            )
        except zstandard.ZstdError as exc:
            raise ValueError("Round 22 feature chunk is not valid zstd") from exc
        if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != str(row[10]):
            raise ValueError("Round 22 feature chunk content hash differs")
        try:
            decoded = json.loads(
                raw.decode("ascii"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Round 22 feature chunk is not strict JSON") from exc
        if (
            not isinstance(decoded, Mapping)
            or _canonical_json(decoded).encode("ascii") != raw
            or decoded.get("schema_version")
            != POLYMARKET_ROUND22_FEATURE_CHUNK_SCHEMA_VERSION
            or decoded.get("condition_id") != selected
            or decoded.get("feature_names_sha256")
            != POLYMARKET_ROUND22_FEATURE_NAMES_SHA256
            or decoded.get("feature_policy_sha256")
            != POLYMARKET_ROUND22_FEATURE_POLICY_SHA256
            or decoded.get("authority")
            != {
                "binance_used": False,
                "target_accessed": False,
                "trading_authority": False,
            }
            or not isinstance(decoded.get("rows"), list)
            or len(decoded["rows"]) != int(row[2])
        ):
            raise ValueError("Round 22 feature chunk identity differs")
        event_start = decoded.get("event_start_ms")
        event_end = decoded.get("event_end_ms")
        if (
            type(event_start) is not int
            or type(event_end) is not int
            or event_end - event_start != 300_000
        ):
            raise ValueError("Round 22 feature chunk clock differs")
        role = str(row[0])
        output: list[Round22CausalFeatureRow] = []
        chain = _EMPTY_SHA256
        for value in decoded["rows"]:
            if not isinstance(value, Mapping) or set(value) != _ROW_FIELDS:
                raise ValueError("Round 22 feature chunk row schema differs")
            reasons = value.get("reasons")
            values = value.get("feature_values")
            if (
                value.get("role") != role
                or not isinstance(reasons, list)
                or any(not isinstance(reason, str) for reason in reasons)
                or not isinstance(values, list)
                or len(values) != len(POLYMARKET_ROUND22_FEATURE_NAMES)
                or any(type(item) not in {int, float} for item in values)
                or any(not math.isfinite(float(item)) for item in values)
                or value.get("feature_values_sha256") != _canonical_sha256(values)
            ):
                raise ValueError("Round 22 feature chunk row values differ")
            identity = dict(value)
            identity.pop("feature_values")
            chain = _row_chain(chain, _canonical_sha256(identity))
            output.append(
                Round22CausalFeatureRow(
                    condition_id=selected,
                    decision_time_ms=int(value["decision_time_ms"]),
                    available=bool(value["available"]),
                    reasons=tuple(reasons),
                    sequence_complete=bool(value["sequence_complete"]),
                    tabular_anchor=bool(value["tabular_anchor"]),
                    tabular_history_complete=bool(value["tabular_history_complete"]),
                    values=tuple(float(item) for item in values),
                    up_source_timestamp_ms=int(value["up_source_timestamp_ms"]),
                    down_source_timestamp_ms=int(value["down_source_timestamp_ms"]),
                    source_chain_sha256=str(value["source_chain_sha256"]),
                )
            )
        if chain != str(row[6]):
            raise ValueError("Round 22 feature chunk row chain differs")
        return tuple(output)

    def audit_condition_rows(
        self,
        condition_id: str,
    ) -> tuple[dict[str, object], tuple[Round22CausalFeatureRow, ...]]:
        selected = str(condition_id or "").strip().lower()
        row = self._chunk_row(selected)
        source = self.connection.execute(
            """
            SELECT manifest_sha256 FROM feature.condition_manifest
            WHERE condition_id = ?
            """,
            [selected],
        ).fetchone()
        if source is None or str(row[1]) != str(source[0]):
            raise ValueError("Round 22 feature source manifest differs")
        self.pilot_store.audit_condition(selected)
        rows = self.load_condition_rows(selected)
        counts = {
            "available": sum(item.available for item in rows),
            "sequence": sum(item.sequence_complete for item in rows),
            "tabular": sum(item.tabular_history_complete for item in rows),
        }
        manifest_body = _manifest_body(
            condition_id=selected,
            role=str(row[0]),
            source_manifest_sha256=str(source[0]),
            row_count=len(rows),
            available_count=counts["available"],
            sequence_count=counts["sequence"],
            tabular_count=counts["tabular"],
            row_chain_sha256=str(row[6]),
            raw_size_bytes=int(row[8]),
            compressed_size_bytes=int(row[9]),
            raw_sha256=str(row[10]),
            compressed_sha256=str(row[11]),
        )
        if (
            int(row[2]) != len(rows)
            or int(row[3]) != counts["available"]
            or int(row[4]) != counts["sequence"]
            or int(row[5]) != counts["tabular"]
            or str(row[13]) != _canonical_sha256(manifest_body)
        ):
            raise ValueError("Round 22 feature manifest differs")
        audit = {
            "available_count": counts["available"],
            "compressed_size_bytes": int(row[9]),
            "condition_id": selected,
            "manifest_sha256": str(row[13]),
            "raw_size_bytes": int(row[8]),
            "row_count": len(rows),
            "sequence_complete_count": counts["sequence"],
            "tabular_history_complete_count": counts["tabular"],
            "target_row_count": self.pilot_store.target_row_count(),
        }
        return audit, rows

    def audit_condition(self, condition_id: str) -> dict[str, object]:
        audit, _rows = self.audit_condition_rows(condition_id)
        return audit


__all__ = [
    "POLYMARKET_ROUND22_FEATURE_CHUNK_CODEC",
    "POLYMARKET_ROUND22_FEATURE_CHUNK_SCHEMA_VERSION",
    "POLYMARKET_ROUND22_FEATURE_STORE_SCHEMA_VERSION",
    "Round22FeatureStore",
]

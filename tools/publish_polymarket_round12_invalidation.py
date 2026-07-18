"""Publish exact machine-readable evidence for the invalidated Round 12 capture."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.storage import write_json_atomic


_SCHEMA_VERSION = "polymarket-round12-invalidated-capture-evidence-v1"
_EXPECTED_ERROR = (
    "operator_invalidated_before_outcome_access:"
    "round12_evaluator_and_publication_chain_not_preregistered"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _decoded_mapping(raw: object, *, name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(
            str(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} JSON is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{name} is not an object")
    return decoded


def publish(database: Path, output: Path) -> str:
    with PolymarketEvidenceStore(database, memory_limit="1GB", threads=1) as store:
        connection = store.connect()
        runs = connection.execute(
            """
            SELECT run_id, status, started_at_ms, ended_at_ms, error,
                   report_json, report_sha256
            FROM polymarket_recorder_run ORDER BY started_at_ms
            """
        ).fetchall()
        if len(runs) != 1:
            raise ValueError(
                "Round 12 invalidation source must contain exactly one run"
            )
        run = runs[0]
        report = dict(_decoded_mapping(run[5], name="recorder report"))
        claimed_report = report.pop("report_sha256", None)
        report["report_sha256"] = claimed_report
        manifest_rows = connection.execute(
            """
            SELECT manifest_json, manifest_sha256
            FROM polymarket_preregistration_manifest WHERE run_id = ?
            """,
            [str(run[0])],
        ).fetchall()
        if len(manifest_rows) != 1:
            raise ValueError("Round 12 invalidation has no unique capture manifest")
        manifest = dict(_decoded_mapping(manifest_rows[0][0], name="capture manifest"))
        claimed_manifest = manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = claimed_manifest
        if (
            str(run[1]) != "failed"
            or str(run[4]) != _EXPECTED_ERROR
            or claimed_report != str(run[6])
            or _sha256(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
            != claimed_report
            or claimed_manifest != str(manifest_rows[0][1])
            or _sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            != claimed_manifest
            or manifest.get("labels_consulted") is not False
            or manifest.get("model_scores_consulted") is not False
        ):
            raise ValueError("Round 12 invalidation identity differs")
        chunks = connection.execute(
            """
            SELECT chunk_id, chunk_index, frame_format, codec, compression_level,
                   message_count, first_message_id, last_message_id,
                   message_manifest_xor, uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, stream_counts_json
            FROM polymarket_raw_chunk WHERE run_id = ? ORDER BY chunk_index
            """,
            [str(run[0])],
        ).fetchall()
        chunk_rows = [
            {
                "chunk_id": str(row[0]),
                "chunk_index": int(row[1]),
                "frame_format": str(row[2]),
                "codec": str(row[3]),
                "compression_level": int(row[4]),
                "message_count": int(row[5]),
                "first_message_id": str(row[6]),
                "last_message_id": str(row[7]),
                "message_manifest_xor": str(row[8]),
                "uncompressed_bytes": int(row[9]),
                "uncompressed_sha256": str(row[10]),
                "compressed_bytes": int(row[11]),
                "compressed_sha256": str(row[12]),
                "stream_counts": _decoded_mapping(row[13], name="chunk streams"),
            }
            for row in chunks
        ]
        persisted_counts = {
            table: int(
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "polymarket_market_snapshot",
                "polymarket_stream_gap",
                "polymarket_public_event",
                "polymarket_resolution_evidence",
                "polymarket_condition_message_frame",
                "polymarket_condition_message_manifest",
            )
        }
    artifact_without_hash: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "round": 12,
        "status": "invalidated_before_outcome_access",
        "run_id": str(run[0]),
        "started_at_ms": int(run[2]),
        "ended_at_ms": int(run[3]),
        "terminal_error": str(run[4]),
        "recorder_report": report,
        "capture_manifest": manifest,
        "raw_chunk_evidence": {
            "chunk_count": len(chunk_rows),
            "message_count": sum(int(row["message_count"]) for row in chunk_rows),
            "compressed_bytes": sum(int(row["compressed_bytes"]) for row in chunk_rows),
            "uncompressed_bytes": sum(
                int(row["uncompressed_bytes"]) for row in chunk_rows
            ),
            "ordered_chunk_manifest_sha256": _sha256(chunk_rows),
        },
        "persisted_table_counts": persisted_counts,
        "outcome_access_evidence": {
            "operator_invalidated_before_outcome_access": True,
            "persisted_resolution_evidence_rows": persisted_counts[
                "polymarket_resolution_evidence"
            ],
            "performance_labels_opened": False,
        },
        "limitations": [
            "terminal_integrity_audit_incomplete",
            "normalized_event_materialization_not_completed",
            "not_model_evidence",
            "not_performance_evidence",
        ],
        "authority": {
            "model_selection": False,
            "profitability_claim": False,
            "paper_trading": False,
            "live_trading": False,
        },
    }
    artifact = {
        **artifact_without_hash,
        "artifact_sha256": _sha256(artifact_without_hash),
    }
    write_json_atomic(output, artifact, indent=2, sort_keys=True)
    return str(artifact["artifact_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(publish(args.database.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

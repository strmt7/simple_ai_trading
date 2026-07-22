"""Publish hash-bound evidence for the failed Round 13 capture."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.storage import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-013-sealed-confirmation-contract.json"
)
_SCHEMA_VERSION = "polymarket-round13-invalidated-capture-evidence-v1"
_EXPECTED_RUN_ID = "244baef7493041ef9de90d728dab153f"
_EXPECTED_STARTED_AT_MS = 1_784_415_205_526
_EXPECTED_ENDED_AT_MS = 1_784_417_126_848
_REQUIRED_DURATION_SECONDS = 86_400
_EXPECTED_CONTRACT_SHA256 = (
    "9ace8092d26918b7621aafb7f008106b06c80049c314158e6a26fd5b70dd4325"
)
_EXPECTED_MANIFEST_SHA256 = (
    "67eb4820e1625e4375800f3a920cf6f34259dcefdfa908d3ebcb512e77098a66"
)
_EXPECTED_ERRORS = (
    "operator_terminalized_abandoned_recorder_process_after_stale_nonterminal_progress",
    "capture_duration_below_frozen_requirement:1921.322<86400",
    "stream_interruptions_present:4",
)
_OUTCOME_TABLES = frozenset(
    {
        "polymarket_round13_attempt",
        "polymarket_round13_calibration_snapshot",
        "polymarket_round13_evaluation_claim",
        "polymarket_round13_evaluation_report",
        "polymarket_round13_scenario_dataset",
    }
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    return dict(_decoded_mapping(path.read_text(encoding="utf-8"), name=name))


def _file_evidence(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def publish(
    database: Path,
    progress: Path,
    stdout_log: Path,
    stderr_log: Path,
    output: Path,
) -> str:
    database = database.resolve()
    progress = progress.resolve()
    stdout_log = stdout_log.resolve()
    stderr_log = stderr_log.resolve()
    contract = _read_json(CONTRACT_PATH, name="Round 13 contract")
    contract_sha256 = str(contract.pop("contract_sha256", ""))
    if (
        contract_sha256 != _EXPECTED_CONTRACT_SHA256
        or _sha256(contract) != contract_sha256
        or contract.get("round") != 13
        or contract.get("status") != "frozen_before_fresh_capture"
    ):
        raise ValueError("Round 13 frozen contract identity differs")
    contract["contract_sha256"] = contract_sha256

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
            raise ValueError("Round 13 invalidation source must contain exactly one run")
        run = runs[0]
        report = dict(_decoded_mapping(run[5], name="recorder report"))
        claimed_report = str(report.pop("report_sha256", ""))
        report["report_sha256"] = claimed_report
        manifest_rows = connection.execute(
            """
            SELECT manifest_json, manifest_sha256
            FROM polymarket_preregistration_manifest WHERE run_id = ?
            """,
            [_EXPECTED_RUN_ID],
        ).fetchall()
        if len(manifest_rows) != 1:
            raise ValueError("Round 13 invalidation has no unique capture manifest")
        manifest = dict(
            _decoded_mapping(manifest_rows[0][0], name="capture manifest")
        )
        claimed_manifest = str(manifest.pop("manifest_sha256", ""))
        manifest["manifest_sha256"] = claimed_manifest
        duration_seconds = (
            _EXPECTED_ENDED_AT_MS - _EXPECTED_STARTED_AT_MS
        ) / 1_000.0
        if (
            str(run[0]) != _EXPECTED_RUN_ID
            or str(run[1]) != "failed"
            or int(run[2]) != _EXPECTED_STARTED_AT_MS
            or int(run[3]) != _EXPECTED_ENDED_AT_MS
            or str(run[4]) != "; ".join(_EXPECTED_ERRORS)
            or claimed_report != str(run[6])
            or _sha256(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
            != claimed_report
            or report.get("errors") != list(_EXPECTED_ERRORS)
            or report.get("integrity_errors")
            != ["terminal_integrity_audit_incomplete"]
            or report.get("status") != "failed"
            or report.get("run_id") != _EXPECTED_RUN_ID
            or report.get("duration_seconds") != duration_seconds
            or int(report.get("stream_gap_count", -1)) != 4
            or claimed_manifest != _EXPECTED_MANIFEST_SHA256
            or claimed_manifest != str(manifest_rows[0][1])
            or _sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            != claimed_manifest
            or manifest.get("run_id") != _EXPECTED_RUN_ID
            or manifest.get("contract_sha256") != _EXPECTED_CONTRACT_SHA256
            or manifest.get("capture_duration_seconds")
            != _REQUIRED_DURATION_SECONDS
            or manifest.get("labels_consulted") is not False
            or manifest.get("model_scores_consulted") is not False
            or manifest.get("outcome_endpoints_queried") is not False
        ):
            raise ValueError("Round 13 failed-capture identity differs")

        chunks = connection.execute(
            """
            SELECT chunk_id, chunk_index, frame_format, codec, compression_level,
                   message_count, first_message_id, last_message_id,
                   message_manifest_xor, uncompressed_bytes, uncompressed_sha256,
                   compressed_bytes, compressed_sha256, stream_counts_json
            FROM polymarket_raw_chunk WHERE run_id = ? ORDER BY chunk_index
            """,
            [_EXPECTED_RUN_ID],
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
            table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in (
                "polymarket_market_snapshot",
                "polymarket_stream_gap",
                "polymarket_public_event",
                "polymarket_resolution_evidence",
                "polymarket_condition_message_frame",
                "polymarket_condition_message_manifest",
            )
        }
        table_names = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }
    if sum(int(row["message_count"]) for row in chunk_rows) != int(
        report["raw_message_count"]
    ):
        raise ValueError("Round 13 raw chunk count differs from terminal report")
    if table_names.intersection(_OUTCOME_TABLES):
        raise ValueError("Round 13 action or evaluation tables unexpectedly exist")
    if (
        persisted_counts["polymarket_public_event"] != 0
        or persisted_counts["polymarket_resolution_evidence"] != 0
    ):
        raise ValueError("Round 13 outcome evidence was persisted")

    progress_payload = _read_json(progress, name="Round 13 progress sidecar")
    if (
        progress_payload.get("run_id") != _EXPECTED_RUN_ID
        or progress_payload.get("phase") != "capturing"
        or int(progress_payload.get("observed_at_ms", -1)) != _EXPECTED_ENDED_AT_MS
        or float(progress_payload.get("elapsed_seconds", -1.0)) != duration_seconds
        or int(progress_payload.get("written_gap_count", -1)) != 4
    ):
        raise ValueError("Round 13 stale progress evidence differs")
    sidecar_messages = int(progress_payload.get("written_message_count", -1))
    database_messages = int(report["raw_message_count"])
    if sidecar_messages < 0 or database_messages < sidecar_messages:
        raise ValueError("Round 13 sidecar/database message counts are inconsistent")

    artifact_without_hash: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "round": 13,
        "status": "failed_capture_ineligible_before_outcome_access",
        "run_id": _EXPECTED_RUN_ID,
        "started_at_ms": _EXPECTED_STARTED_AT_MS,
        "ended_at_ms": _EXPECTED_ENDED_AT_MS,
        "terminal_errors": list(_EXPECTED_ERRORS),
        "recorder_report": report,
        "capture_manifest": manifest,
        "frozen_capture_requirement": {
            "contract_sha256": _EXPECTED_CONTRACT_SHA256,
            "required_duration_seconds": _REQUIRED_DURATION_SECONDS,
            "observed_duration_seconds": duration_seconds,
            "duration_shortfall_seconds": (
                _REQUIRED_DURATION_SECONDS - duration_seconds
            ),
            "required_one_shot_capture_completed": False,
            "stream_gap_count": int(report["stream_gap_count"]),
        },
        "stale_progress_evidence": {
            "phase_before_terminalization": str(progress_payload["phase"]),
            "observed_at_ms": int(progress_payload["observed_at_ms"]),
            "elapsed_seconds": float(progress_payload["elapsed_seconds"]),
            "sidecar_written_message_count": sidecar_messages,
            "database_raw_message_count": database_messages,
            "database_minus_sidecar_message_count": (
                database_messages - sidecar_messages
            ),
            "written_gap_count": int(progress_payload["written_gap_count"]),
            "error_count": int(progress_payload["error_count"]),
            "queue_high_watermark": int(progress_payload["queue_high_watermark"]),
        },
        "raw_chunk_evidence": {
            "chunk_count": len(chunk_rows),
            "message_count": database_messages,
            "compressed_bytes": sum(
                int(row["compressed_bytes"]) for row in chunk_rows
            ),
            "uncompressed_bytes": sum(
                int(row["uncompressed_bytes"]) for row in chunk_rows
            ),
            "ordered_chunk_manifest_sha256": _sha256(chunk_rows),
        },
        "persisted_table_counts": persisted_counts,
        "outcome_access_evidence": {
            "capture_manifest_labels_consulted": False,
            "capture_manifest_model_scores_consulted": False,
            "capture_manifest_outcome_endpoints_queried": False,
            "persisted_public_event_rows": persisted_counts[
                "polymarket_public_event"
            ],
            "persisted_resolution_evidence_rows": persisted_counts[
                "polymarket_resolution_evidence"
            ],
            "round13_action_and_evaluation_tables_present": False,
            "performance_labels_opened": False,
        },
        "source_file_evidence": {
            "database": _file_evidence(database),
            "progress_sidecar": _file_evidence(progress),
            "stdout_log": _file_evidence(stdout_log),
            "stderr_log": _file_evidence(stderr_log),
        },
        "limitations": [
            "terminal_integrity_audit_incomplete",
            "required_86400_second_one_shot_capture_not_completed",
            "stream_interruptions_present",
            "normalized_event_materialization_not_started",
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
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        publish(
            args.database,
            args.progress,
            args.stdout_log,
            args.stderr_log,
            args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

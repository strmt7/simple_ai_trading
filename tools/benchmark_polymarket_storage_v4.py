"""Reproducible infrastructure-only benchmark for Polymarket storage v4."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

from simple_ai_trading.polymarket_recorder import (
    POLYMARKET_STORAGE_SCHEMA_VERSION,
    PolymarketEvidenceStore,
    RawStreamMessage,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_payload_digest(
    digest: hashlib._Hash,
    *,
    stream: str,
    raw_text: str,
) -> None:
    stream_bytes = stream.encode("ascii")
    raw_bytes = raw_text.encode("utf-8")
    digest.update(len(stream_bytes).to_bytes(2, "little"))
    digest.update(stream_bytes)
    digest.update(len(raw_bytes).to_bytes(4, "little"))
    digest.update(raw_bytes)


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_exact_payload_sample(
    source_database: Path,
    *,
    source_run_id: str,
    sample_messages: int,
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    with PolymarketEvidenceStore(source_database, read_only=True) as source:
        run = (
            source.connect()
            .execute(
                """
            SELECT status, storage_schema_version, report_sha256
            FROM polymarket_recorder_run WHERE run_id = ?
            """,
                [source_run_id],
            )
            .fetchone()
        )
        if run is None:
            raise ValueError("source recorder run does not exist")
        rows = (
            source.connect()
            .execute(
                """
            SELECT stream, raw_payload_sha256, raw_text, storage_chunk_id,
                   raw_offset, raw_size
            FROM polymarket_raw_message
            WHERE run_id = ?
            ORDER BY received_monotonic_ns, received_wall_ms,
                     connection_id, sequence_number
            LIMIT ?
            """,
                [source_run_id, sample_messages],
            )
            .fetchall()
        )
        if len(rows) != sample_messages:
            raise ValueError("source run does not contain the requested sample")
        sample: list[tuple[str, str]] = []
        sample_digest = hashlib.sha256()
        sample_raw_bytes = 0
        for stream, raw_sha, inline, chunk_id, raw_offset, raw_size in rows:
            raw_text = str(inline)
            if not raw_text:
                raw_text = source._decode_compact_raw_text(
                    run_id=source_run_id,
                    chunk_id=str(chunk_id),
                    raw_offset=int(raw_offset),
                    raw_size=int(raw_size),
                    raw_payload_sha256=str(raw_sha),
                )
            actual_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if actual_sha != str(raw_sha):
                raise ValueError("source sample payload hash differs")
            normalized_stream = str(stream)
            sample.append((normalized_stream, raw_text))
            _update_payload_digest(
                sample_digest,
                stream=normalized_stream,
                raw_text=raw_text,
            )
            sample_raw_bytes += len(raw_text.encode("utf-8"))
        return sample, {
            "terminal_status": str(run[0]),
            "storage_schema_version": str(run[1]),
            "report_sha256": str(run[2]),
            "database_bytes": source_database.stat().st_size,
            "sample_message_count": len(sample),
            "sample_raw_bytes": sample_raw_bytes,
            "sample_stream_counts": dict(sorted(Counter(x[0] for x in sample).items())),
            "sample_payload_stream_sha256": sample_digest.hexdigest(),
            "sample_payload_hashes_verified": True,
            "whole_source_run_integrity_audit_performed": False,
        }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    source_database = args.source_database.resolve()
    output_database = args.output_database.resolve()
    if not source_database.is_file():
        raise ValueError("source database does not exist")
    if output_database.exists() or Path(f"{output_database}.wal").exists():
        raise ValueError("output database or WAL already exists")
    if args.sample_messages < 1 or args.total_messages < args.sample_messages:
        raise ValueError("message counts are invalid")
    if args.batch_size < 1 or args.batch_size > 65_536:
        raise ValueError("batch size must lie in [1, 65536]")

    sample, source = _load_exact_payload_sample(
        source_database,
        source_run_id=args.source_run_id,
        sample_messages=args.sample_messages,
    )
    output_database.parent.mkdir(parents=True, exist_ok=True)
    run_id = "storage-v4-benchmark"
    sequence_by_stream: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    expected_digest = hashlib.sha256()
    checkpoints: list[dict[str, object]] = []
    written = 0
    started = time.perf_counter()
    with PolymarketEvidenceStore(
        output_database,
        memory_limit=args.memory_limit,
        threads=args.database_threads,
    ) as store:
        store.start_run(run_id, 1_784_160_000_000)
        while written < args.total_messages:
            batch_count = min(args.batch_size, args.total_messages - written)
            batch: list[RawStreamMessage] = []
            for offset in range(batch_count):
                stream, raw_text = sample[(written + offset) % len(sample)]
                sequence_by_stream[stream] += 1
                expected_counts[stream] += 1
                ordinal = written + offset + 1
                _update_payload_digest(
                    expected_digest,
                    stream=stream,
                    raw_text=raw_text,
                )
                batch.append(
                    RawStreamMessage(
                        stream=stream,
                        connection_id=f"benchmark:{stream}",
                        sequence_number=sequence_by_stream[stream],
                        received_wall_ms=1_784_160_000_000 + ordinal,
                        received_monotonic_ns=ordinal,
                        raw_text=raw_text,
                    )
                )
            store.append_messages(run_id, batch)
            written += batch_count
            if (
                written == args.total_messages
                or written % args.checkpoint_messages < batch_count
            ):
                elapsed = time.perf_counter() - started
                checkpoint = {
                    "written_messages": written,
                    "elapsed_seconds": elapsed,
                    "messages_per_second": written / elapsed,
                    "database_bytes": output_database.stat().st_size,
                }
                checkpoints.append(checkpoint)
                print(_canonical({"phase": "write", **checkpoint}), flush=True)
        write_seconds = time.perf_counter() - started

        audit_started = time.perf_counter()
        integrity_errors = store.integrity_errors(run_id)
        audit_seconds = time.perf_counter() - audit_started

        read_digest = hashlib.sha256()
        replay_counts: Counter[str] = Counter()
        read_started = time.perf_counter()
        replayed = 0
        for stored in store._iter_capture_messages(run_id):
            message = stored.message
            _update_payload_digest(
                read_digest,
                stream=message.stream,
                raw_text=message.raw_text,
            )
            replay_counts[message.stream] += 1
            replayed += 1
        read_seconds = time.perf_counter() - read_started
        connection = store.connect()
        hot_rows = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM polymarket_raw_message WHERE run_id = ?),
              (SELECT count(*) FROM polymarket_public_event WHERE run_id = ?)
            """,
            [run_id, run_id],
        ).fetchone()
        chunk_stats = connection.execute(
            """
            SELECT count(*), coalesce(sum(message_count), 0),
                   coalesce(sum(uncompressed_bytes), 0),
                   coalesce(sum(compressed_bytes), 0)
            FROM polymarket_raw_chunk WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        evidence_manifest_sha256 = store._capture_manifest_sha256(run_id)
        connection.execute("CHECKPOINT")

    actual_digest = read_digest.hexdigest()
    expected_digest_hex = expected_digest.hexdigest()
    if integrity_errors:
        raise ValueError("storage-v4 benchmark integrity failed")
    if replayed != args.total_messages or replay_counts != expected_counts:
        raise ValueError("storage-v4 replay counts differ")
    if actual_digest != expected_digest_hex:
        raise ValueError("storage-v4 replay payload order differs")
    if tuple(map(int, hot_rows)) != (0, 0):
        raise ValueError("storage-v4 wrote retired hot rows")
    if int(chunk_stats[1]) != args.total_messages:
        raise ValueError("storage-v4 chunk message total differs")

    payload: dict[str, object] = {
        "schema_version": "polymarket-storage-v4-benchmark-v1",
        "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "implementation": {
            "repository_commit": _repository_commit(),
            "polymarket_capture_frame_sha256": _sha256_file(
                ROOT / "src/simple_ai_trading/polymarket_capture_frame.py"
            ),
            "polymarket_recorder_sha256": _sha256_file(
                ROOT / "src/simple_ai_trading/polymarket_recorder.py"
            ),
            "storage_schema_version": POLYMARKET_STORAGE_SCHEMA_VERSION,
        },
        "source_evidence": {
            "database": str(source_database.relative_to(ROOT)),
            "run_id": args.source_run_id,
            **source,
            "public_payloads_real": True,
            "permitted_use": "infrastructure_benchmark_only",
        },
        "workload": {
            "total_messages": args.total_messages,
            "sample_messages": args.sample_messages,
            "batch_size": args.batch_size,
            "checkpoint_messages": args.checkpoint_messages,
            "receipt_metadata_kind": "synthetic monotonic sequence and receipt timestamps",
            "payload_reuse": "exact source payload sample repeated cyclically",
        },
        "results": {
            "write_seconds": write_seconds,
            "messages_per_second": args.total_messages / write_seconds,
            "audit_seconds": audit_seconds,
            "read_seconds": read_seconds,
            "read_messages_per_second": replayed / read_seconds,
            "database_bytes": output_database.stat().st_size,
            "persisted_chunks": int(chunk_stats[0]),
            "persisted_messages": int(chunk_stats[1]),
            "uncompressed_frame_bytes": int(chunk_stats[2]),
            "compressed_frame_bytes": int(chunk_stats[3]),
            "frame_compression_ratio": int(chunk_stats[3]) / int(chunk_stats[2]),
            "persisted_raw_message_rows": int(hot_rows[0]),
            "persisted_public_event_rows": int(hot_rows[1]),
            "stream_counts": dict(sorted(expected_counts.items())),
            "payload_stream_sha256": actual_digest,
            "expected_payload_stream_sha256": expected_digest_hex,
            "exact_count_and_ordered_payload_match": True,
            "integrity_errors": list(integrity_errors),
            "evidence_manifest_sha256": evidence_manifest_sha256,
            "checkpoints": checkpoints,
        },
        "claims": {
            "bounded_benchmark_proves_long_duration_capture": False,
            "financial_edge_tested": False,
            "model_evidence": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
    }
    payload["report_sha256"] = hashlib.sha256(
        _canonical(payload).encode("ascii")
    ).hexdigest()
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-database", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--sample-messages", type=int, default=8_192)
    parser.add_argument("--total-messages", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--checkpoint-messages", type=int, default=131_072)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--database-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_benchmark(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(_canonical({"phase": "complete", **report["results"]}), flush=True)


if __name__ == "__main__":
    main()

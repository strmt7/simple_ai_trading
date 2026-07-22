#!/usr/bin/env python3
"""Benchmark lossless compression on bounded real Round 73 exact frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import zstandard


_FRAME_TABLES = {
    "round-073-prospective-evidence-v5": "impact_capture_frame",
    "round-073-prospective-evidence-v6": "impact_capture_frame",
    "round-073-prospective-evidence-v7": "impact_capture_frame",
    "round-073-prospective-evidence-v8": "impact_capture_frame_v8",
}
_LEVELS = (1, 3, 6, 9, 12)
_CHUNK_FRAME_COUNTS = (1, 8, 32)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _write_immutable(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("compression output already exists with different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _chunks(values: list[bytes], size: int) -> list[bytes]:
    return [
        b"".join(values[index : index + size]) for index in range(0, len(values), size)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/microstructure.duckdb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= int(args.frames) <= 64:
        raise SystemExit("frames must be between 1 and 64")
    root = _repo_root().resolve()
    database = Path(args.database).resolve()
    run_id = str(args.run_id).strip().lower()
    with duckdb.connect(str(database), read_only=True) as connection:
        run = connection.execute(
            "SELECT schema_version FROM impact_capture_run WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if run is None:
            raise SystemExit("source run was not found")
        schema_version = str(run[0])
        try:
            frame_table = _FRAME_TABLES[schema_version]
        except KeyError as exc:
            raise SystemExit("source run schema is unsupported") from exc
        rows = connection.execute(
            f"SELECT frame_index, message_count, uncompressed_bytes, "
            f"uncompressed_sha256, compressed_bytes, compressed_sha256, "
            f"compressed_payload, frame_sha256 FROM {frame_table} "
            "WHERE run_id = ? ORDER BY frame_index LIMIT ?",
            [run_id, int(args.frames)],
        ).fetchall()
    if tuple(int(row[0]) for row in rows) != tuple(range(int(args.frames))):
        raise SystemExit("source run does not contain the requested frame prefix")
    decompressor = zstandard.ZstdDecompressor()
    raw_frames: list[bytes] = []
    for row in rows:
        compressed = bytes(row[6])
        if len(compressed) != int(row[4]):
            raise RuntimeError("stored compressed frame size differs")
        if hashlib.sha256(compressed).hexdigest() != str(row[5]):
            raise RuntimeError("stored compressed frame digest differs")
        raw = decompressor.decompress(
            compressed,
            max_output_size=int(row[2]),
        )
        if len(raw) != int(row[2]):
            raise RuntimeError("stored uncompressed frame size differs")
        if hashlib.sha256(raw).hexdigest() != str(row[3]):
            raise RuntimeError("stored uncompressed frame digest differs")
        raw_frames.append(raw)
    raw_bytes = sum(len(value) for value in raw_frames)
    raw_sha256 = hashlib.sha256(b"".join(raw_frames)).hexdigest()
    scenarios: list[dict[str, object]] = []
    for chunk_frame_count in _CHUNK_FRAME_COUNTS:
        chunks = _chunks(raw_frames, chunk_frame_count)
        for level in _LEVELS:
            compressor = zstandard.ZstdCompressor(
                level=level,
                write_checksum=True,
                write_content_size=True,
                threads=0,
            )
            started = time.perf_counter()
            encoded = [compressor.compress(chunk) for chunk in chunks]
            elapsed = time.perf_counter() - started
            decoded = [
                decompressor.decompress(blob, max_output_size=len(source))
                for blob, source in zip(encoded, chunks, strict=True)
            ]
            if decoded != chunks:
                raise RuntimeError("compression scenario failed exact round trip")
            compressed_bytes = sum(len(value) for value in encoded)
            scenarios.append(
                {
                    "chunk_frame_count": chunk_frame_count,
                    "compression_level": level,
                    "compressed_bytes": compressed_bytes,
                    "compression_ratio": raw_bytes / compressed_bytes,
                    "elapsed_seconds": elapsed,
                    "throughput_mib_per_second": (raw_bytes / 1_048_576 / elapsed),
                    "exact_round_trip_passed": True,
                }
            )
    report: dict[str, object] = {
        "schema_version": "round-073-real-frame-compression-benchmark-v1",
        "source_run_id": run_id,
        "source_schema_version": schema_version,
        "frame_table": frame_table,
        "frame_indices": [int(row[0]) for row in rows],
        "frame_sha256_manifest": _canonical_sha256([str(row[7]) for row in rows]),
        "message_count": sum(int(row[1]) for row in rows),
        "raw_bytes": raw_bytes,
        "raw_concatenation_sha256": raw_sha256,
        "stored_level_3_compressed_bytes": sum(int(row[4]) for row in rows),
        "scenarios": scenarios,
        "synthetic_market_data_used": False,
        "scratch_files_written": False,
        "financial_or_model_evidence": False,
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    if args.output is not None:
        output = Path(args.output).resolve()
        if root not in output.parents:
            raise SystemExit("output must remain inside the repository")
        _write_immutable(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

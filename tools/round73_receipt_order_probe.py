#!/usr/bin/env python3
"""Measure exact-frame receipt-order lag on one real Round 73 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import zstandard


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "src"))

from simple_ai_trading.impact_capture_frame import (  # noqa: E402
    decode_impact_capture_frame,
)


_FRAME_TABLES = {
    "round-073-prospective-evidence-v5": "impact_capture_frame",
    "round-073-prospective-evidence-v6": "impact_capture_frame",
    "round-073-prospective-evidence-v7": "impact_capture_frame",
    "round-073-prospective-evidence-v8": "impact_capture_frame_v8",
}


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
            raise RuntimeError(
                "receipt-order output already exists with different bytes"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/microstructure.duckdb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = _repo_root().resolve()
    run_id = str(args.run_id).strip().lower()
    with duckdb.connect(
        str(Path(args.database).resolve()), read_only=True
    ) as connection:
        run = connection.execute(
            "SELECT schema_version, frame_count, message_count FROM "
            "impact_capture_run WHERE run_id = ?",
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
            "WHERE run_id = ? ORDER BY frame_index",
            [run_id],
        ).fetchall()
    decompressor = zstandard.ZstdDecompressor()
    running_max = 0
    previous_encoded_receipt = 0
    sequential_inversions = 0
    maximum_backward_lag_ns = 0
    cross_frame_violations = 0
    maximum_cross_frame_lag_ns = 0
    lane_state: dict[tuple[str, str], tuple[int, int]] = {}
    lane_sequence_or_clock_violations = 0
    observed_messages = 0
    for expected_frame, row in enumerate(rows):
        if int(row[0]) != expected_frame:
            raise RuntimeError("source frame indices are not contiguous")
        compressed = bytes(row[6])
        if len(compressed) != int(row[4]) or hashlib.sha256(
            compressed
        ).hexdigest() != str(row[5]):
            raise RuntimeError("source compressed frame identity differs")
        raw = decompressor.decompress(
            compressed,
            max_output_size=int(row[2]),
        )
        if len(raw) != int(row[2]) or hashlib.sha256(raw).hexdigest() != str(row[3]):
            raise RuntimeError("source raw frame identity differs")
        decoded = decode_impact_capture_frame(
            raw,
            expected_message_count=int(row[1]),
        )
        frame_minimum = min(item.record.received_monotonic_ns for item in decoded)
        if running_max and frame_minimum < running_max:
            cross_frame_violations += 1
            maximum_cross_frame_lag_ns = max(
                maximum_cross_frame_lag_ns,
                running_max - frame_minimum,
            )
        for item in decoded:
            record = item.record
            receipt = int(record.received_monotonic_ns)
            if receipt < previous_encoded_receipt:
                sequential_inversions += 1
            if receipt < running_max:
                maximum_backward_lag_ns = max(
                    maximum_backward_lag_ns,
                    running_max - receipt,
                )
            running_max = max(running_max, receipt)
            previous_encoded_receipt = receipt
            lane = (record.stream, record.connection_id)
            prior = lane_state.get(lane)
            if prior is not None and (
                record.sequence_number != prior[0] + 1 or receipt <= prior[1]
            ):
                lane_sequence_or_clock_violations += 1
            elif prior is None and record.sequence_number != 0:
                lane_sequence_or_clock_violations += 1
            lane_state[lane] = (record.sequence_number, receipt)
            observed_messages += 1
    if len(rows) != int(run[1]) or observed_messages != int(run[2]):
        raise RuntimeError("source run counters differ from exact frames")
    report: dict[str, object] = {
        "schema_version": "round-073-real-frame-receipt-order-probe-v1",
        "source_run_id": run_id,
        "source_schema_version": schema_version,
        "frame_table": frame_table,
        "frame_count": len(rows),
        "message_count": observed_messages,
        "frame_sha256_manifest": _canonical_sha256([str(row[7]) for row in rows]),
        "encoded_order_adjacent_inversion_count": sequential_inversions,
        "maximum_backward_from_running_high_water_ns": maximum_backward_lag_ns,
        "cross_frame_lagged_frame_count": cross_frame_violations,
        "maximum_cross_frame_lag_ns": maximum_cross_frame_lag_ns,
        "lane_sequence_or_clock_violation_count": (lane_sequence_or_clock_violations),
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

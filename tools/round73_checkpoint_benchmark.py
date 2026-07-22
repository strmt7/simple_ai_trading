#!/usr/bin/env python3
"""Compare bounded DuckDB checkpoint policies on real Round 73 event links."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import duckdb


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "src"))

from simple_ai_trading.duckdb_batch import insert_rows_columnar  # noqa: E402


_COLUMNS = (
    "run_id",
    "frame_index",
    "message_index",
    "segment_id",
    "stream",
    "connection_id",
    "sequence_number",
    "received_wall_ns",
    "received_monotonic_ns",
    "raw_payload_sha256",
    "event_type",
    "symbol",
    "event_time_ms",
    "typed_event_sha256",
)
_CREATE_TABLE = """
CREATE TABLE event_link (
    run_id VARCHAR NOT NULL,
    frame_index UINTEGER NOT NULL,
    message_index USMALLINT NOT NULL,
    segment_id VARCHAR NOT NULL,
    stream VARCHAR NOT NULL,
    connection_id VARCHAR NOT NULL,
    sequence_number UBIGINT NOT NULL,
    received_wall_ns UBIGINT NOT NULL,
    received_monotonic_ns UBIGINT NOT NULL,
    raw_payload_sha256 BLOB NOT NULL CHECK (octet_length(raw_payload_sha256) = 32),
    event_type VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    event_time_ms BIGINT,
    typed_event_sha256 BLOB NOT NULL CHECK (octet_length(typed_event_sha256) = 32)
)
"""
_INSERT = "INSERT INTO event_link SELECT " + ", ".join(
    "unnest(?)" for _column in _COLUMNS
)


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_ulonglong),
        ("write_operation_count", ctypes.c_ulonglong),
        ("other_operation_count", ctypes.c_ulonglong),
        ("read_transfer_count", ctypes.c_ulonglong),
        ("write_transfer_count", ctypes.c_ulonglong),
        ("other_transfer_count", ctypes.c_ulonglong),
    ]


def _process_write_transfer_bytes() -> int | None:
    if os.name != "nt":
        return None
    counters = _IoCounters()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessIoCounters.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_IoCounters),
    ]
    kernel32.GetProcessIoCounters.restype = ctypes.c_int
    if not kernel32.GetProcessIoCounters(
        kernel32.GetCurrentProcess(), ctypes.byref(counters)
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessIoCounters failed")
    return int(counters.write_transfer_count)


def _physical_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}.wal"))
        if candidate.is_file()
    )


def _remove_exact_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}.wal")):
        if candidate.is_file():
            candidate.unlink()


def _signature(rows: Sequence[tuple[object, ...]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "frame_index_sum": sum(int(row[1]) for row in rows),
        "sequence_number_sum": sum(int(row[6]) for row in rows),
        "raw_digest_bytes": sum(len(bytes(row[9])) for row in rows),
        "typed_digest_bytes": sum(len(bytes(row[13])) for row in rows),
    }


@dataclass(frozen=True)
class _Scenario:
    name: str
    checkpoint_threshold: str
    auto_checkpoint_skip_wal_threshold_bytes: int
    elapsed_seconds: float
    process_write_transfer_bytes: int | None
    physical_bytes: int
    maximum_observed_wal_bytes: int
    signature: dict[str, int]


def _run_scenario(
    *,
    path: Path,
    rows: Sequence[tuple[object, ...]],
    batch_size: int,
    checkpoint_threshold: str | None,
    skip_wal_threshold: int | None,
) -> _Scenario:
    _remove_exact_files(path)
    before = _process_write_transfer_bytes()
    started = time.perf_counter()
    maximum_wal = 0
    connection = duckdb.connect(str(path))
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET preserve_insertion_order=false")
    if checkpoint_threshold is not None:
        connection.execute(f"SET checkpoint_threshold='{checkpoint_threshold}'")
    if skip_wal_threshold is not None:
        connection.execute(
            "SET auto_checkpoint_skip_wal_threshold=?", [skip_wal_threshold]
        )
    effective_checkpoint = str(
        connection.execute(
            "SELECT current_setting('checkpoint_threshold')"
        ).fetchone()[0]
    )
    effective_skip = int(
        connection.execute(
            "SELECT current_setting('auto_checkpoint_skip_wal_threshold')"
        ).fetchone()[0]
    )
    connection.execute(_CREATE_TABLE)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        connection.execute("BEGIN TRANSACTION")
        try:
            insert_rows_columnar(
                connection,
                sql=_INSERT,
                rows=batch,
                width=len(_COLUMNS),
                batch_size=batch_size,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        maximum_wal = max(maximum_wal, _physical_bytes(Path(f"{path}.wal")))
    connection.close()
    elapsed = time.perf_counter() - started
    after = _process_write_transfer_bytes()
    with duckdb.connect(str(path), read_only=True) as verify:
        stored = verify.execute(
            """
            SELECT count(*), sum(frame_index), sum(sequence_number),
                   sum(octet_length(raw_payload_sha256)),
                   sum(octet_length(typed_event_sha256))
            FROM event_link
            """
        ).fetchone()
    signature = {
        "row_count": int(stored[0]),
        "frame_index_sum": int(stored[1]),
        "sequence_number_sum": int(stored[2]),
        "raw_digest_bytes": int(stored[3]),
        "typed_digest_bytes": int(stored[4]),
    }
    return _Scenario(
        name="default" if checkpoint_threshold is None else "candidate",
        checkpoint_threshold=effective_checkpoint,
        auto_checkpoint_skip_wal_threshold_bytes=effective_skip,
        elapsed_seconds=elapsed,
        process_write_transfer_bytes=(
            None if before is None or after is None else after - before
        ),
        physical_bytes=_physical_bytes(path),
        maximum_observed_wal_bytes=maximum_wal,
        signature=signature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/microstructure.duckdb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--scratch-dir", default="data")
    args = parser.parse_args()
    if args.rows < 1 or args.batch_size < 1 or args.batch_size > args.rows:
        raise SystemExit("rows and batch-size must define a positive bounded sample")
    scratch = Path(args.scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    root = _repo_root().resolve()
    if root not in scratch.parents and scratch != root:
        raise SystemExit("scratch-dir must remain inside the repository")
    paths = (
        scratch / "round73-checkpoint-default.duckdb",
        scratch / "round73-checkpoint-candidate.duckdb",
    )
    source = duckdb.connect(str(Path(args.database)), read_only=True)
    rows = source.execute(
        f"""
        SELECT {', '.join(_COLUMNS)} FROM impact_event_link_v4
        WHERE run_id = ? ORDER BY frame_index, message_index LIMIT ?
        """,
        [str(args.run_id).strip().lower(), int(args.rows)],
    ).fetchall()
    source.close()
    if len(rows) != int(args.rows):
        raise SystemExit("qualified run does not contain the requested real-row sample")
    expected_signature = _signature(rows)
    try:
        default = _run_scenario(
            path=paths[0],
            rows=rows,
            batch_size=int(args.batch_size),
            checkpoint_threshold=None,
            skip_wal_threshold=None,
        )
        candidate = _run_scenario(
            path=paths[1],
            rows=rows,
            batch_size=int(args.batch_size),
            checkpoint_threshold="256MiB",
            skip_wal_threshold=256 * 1024 * 1024,
        )
        if default.signature != expected_signature or candidate.signature != expected_signature:
            raise RuntimeError("checkpoint benchmark changed the real-row signature")
        write_ratio = None
        if (
            default.process_write_transfer_bytes is not None
            and candidate.process_write_transfer_bytes is not None
            and default.process_write_transfer_bytes > 0
        ):
            write_ratio = (
                candidate.process_write_transfer_bytes
                / default.process_write_transfer_bytes
            )
        print(
            json.dumps(
                {
                    "schema_version": "round-073-checkpoint-benchmark-v1",
                    "source_run_id": str(args.run_id).strip().lower(),
                    "synthetic_market_data_used": False,
                    "sample_rows": len(rows),
                    "transaction_batch_rows": int(args.batch_size),
                    "source_signature": expected_signature,
                    "default": asdict(default),
                    "candidate": asdict(candidate),
                    "candidate_to_default_process_write_ratio": write_ratio,
                    "scratch_files_retained": False,
                    "financial_or_model_evidence": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        for path in paths:
            _remove_exact_files(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

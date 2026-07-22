#!/usr/bin/env python3
"""Compare DuckDB checkpoint policies on bounded real Round 73 v5 frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import duckdb


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "src"))

from simple_ai_trading.impact_absorption_capture import (  # noqa: E402
    _process_io_snapshot,
)
from simple_ai_trading.impact_absorption_store import (  # noqa: E402
    ImpactAbsorptionStore,
)


_FULL_PATH_TABLES = (
    "impact_capture_frame",
    "impact_event_link_v5",
    "impact_depth_update_v3",
    "impact_depth_band_flow_v5",
    "impact_l2_state_v3",
    "impact_book_ticker_v3",
    "impact_aggregate_trade_v3",
    "impact_mark_price_v3",
    "impact_liquidation_snapshot_v3",
    "impact_rest_event_v3",
    "impact_rejected_wire_event_v3",
)


def _physical_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}.wal"))
        if candidate.is_file()
    )


def _wal_bytes(path: Path) -> int:
    wal = Path(f"{path}.wal")
    return wal.stat().st_size if wal.is_file() else 0


def _remove_exact_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}.wal")):
        if candidate.is_file():
            candidate.unlink()


def _write_delta(start: int | None, end: int | None) -> int | None:
    if start is None or end is None or end < start:
        return None
    return end - start


def _signature(
    connection: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    run_id: str,
    frame_count: int | None = None,
) -> dict[str, int]:
    predicate = "run_id = ?"
    parameters: list[object] = [run_id]
    if frame_count is not None:
        predicate += " AND frame_index < ?"
        parameters.append(int(frame_count))
    output = {
        table: int(
            connection.execute(
                f"SELECT count(*) FROM {schema}.{table} WHERE {predicate}",
                parameters,
            ).fetchone()[0]
        )
        for table in _FULL_PATH_TABLES
    }
    totals = connection.execute(
        f"SELECT coalesce(sum(message_count), 0), "
        f"coalesce(sum(compressed_bytes), 0) FROM {schema}.impact_capture_frame "
        f"WHERE {predicate}",
        parameters,
    ).fetchone()
    output["frame_message_count_sum"] = int(totals[0])
    output["frame_compressed_bytes_sum"] = int(totals[1])
    return output


@dataclass(frozen=True)
class _Scenario:
    name: str
    checkpoint_threshold: str
    auto_checkpoint_skip_wal_threshold_bytes: int
    elapsed_seconds: float
    process_write_transfer_bytes: int | None
    process_write_transfer_bytes_per_message: float | None
    transaction_write_transfer_bytes: tuple[int | None, ...]
    connection_close_write_transfer_bytes: int | None
    physical_bytes: int
    maximum_observed_wal_bytes: int
    signature: dict[str, int]


def _run_scenario(
    *,
    source: Path,
    target: Path,
    run_id: str,
    frame_indices: tuple[int, ...],
    name: str,
    checkpoint_threshold: str,
    skip_wal_threshold: int,
) -> _Scenario:
    _remove_exact_files(target)
    with ImpactAbsorptionStore(target, memory_limit="1GB", threads=1) as store:
        store.connect()

    started_io = _process_io_snapshot()
    started = time.perf_counter()
    maximum_wal = 0
    transaction_deltas: list[int | None] = []
    connection = duckdb.connect(str(target))
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET checkpoint_threshold='{checkpoint_threshold}'")
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
    escaped_source = str(source).replace("'", "''")
    connection.execute(f"ATTACH '{escaped_source}' AS source_db (READ_ONLY)")
    for frame_index in frame_indices:
        before_transaction = _process_io_snapshot().write_bytes
        connection.execute("BEGIN TRANSACTION")
        try:
            for table in _FULL_PATH_TABLES:
                connection.execute(
                    f"INSERT INTO main.{table} SELECT * FROM source_db.{table} "
                    "WHERE run_id = ? AND frame_index = ?",
                    [run_id, frame_index],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        after_transaction = _process_io_snapshot().write_bytes
        transaction_deltas.append(
            _write_delta(before_transaction, after_transaction)
        )
        maximum_wal = max(maximum_wal, _wal_bytes(target))
    before_close = _process_io_snapshot().write_bytes
    connection.close()
    after_close = _process_io_snapshot().write_bytes
    elapsed = time.perf_counter() - started
    ended_io = _process_io_snapshot()
    with duckdb.connect(str(target), read_only=True) as verify:
        signature = _signature(verify, schema="main", run_id=run_id)
    total_write = _write_delta(started_io.write_bytes, ended_io.write_bytes)
    messages = signature["frame_message_count_sum"]
    return _Scenario(
        name=name,
        checkpoint_threshold=effective_checkpoint,
        auto_checkpoint_skip_wal_threshold_bytes=effective_skip,
        elapsed_seconds=elapsed,
        process_write_transfer_bytes=total_write,
        process_write_transfer_bytes_per_message=(
            None if total_write is None or messages == 0 else total_write / messages
        ),
        transaction_write_transfer_bytes=tuple(transaction_deltas),
        connection_close_write_transfer_bytes=_write_delta(before_close, after_close),
        physical_bytes=_physical_bytes(target),
        maximum_observed_wal_bytes=maximum_wal,
        signature=signature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/microstructure.duckdb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--scratch-dir", default="data")
    args = parser.parse_args()
    if not 1 <= int(args.frames) <= 64:
        raise SystemExit("frames must be between 1 and 64")
    root = _repo_root().resolve()
    source = Path(args.database).resolve()
    scratch = Path(args.scratch_dir).resolve()
    if root not in scratch.parents and scratch != root:
        raise SystemExit("scratch-dir must remain inside the repository")
    if source == scratch / "round73-full-path-default.duckdb" or source == (
        scratch / "round73-full-path-candidate.duckdb"
    ):
        raise SystemExit("source database cannot be a scratch target")
    scratch.mkdir(parents=True, exist_ok=True)
    selected = str(args.run_id).strip().lower()
    with duckdb.connect(str(source), read_only=True) as source_connection:
        frame_rows = source_connection.execute(
            "SELECT frame_index FROM impact_capture_frame WHERE run_id = ? "
            "ORDER BY frame_index LIMIT ?",
            [selected, int(args.frames)],
        ).fetchall()
        frame_indices = tuple(int(row[0]) for row in frame_rows)
        if frame_indices != tuple(range(int(args.frames))):
            raise SystemExit("source run does not contain the requested frame prefix")
        expected_prefix = _signature(
            source_connection,
            schema="main",
            run_id=selected,
            frame_count=int(args.frames),
        )
        prefix_messages = expected_prefix["frame_message_count_sum"]
    targets = (
        scratch / "round73-full-path-default.duckdb",
        scratch / "round73-full-path-candidate.duckdb",
    )
    try:
        default = _run_scenario(
            source=source,
            target=targets[0],
            run_id=selected,
            frame_indices=frame_indices,
            name="observed_defaults",
            checkpoint_threshold="16MiB",
            skip_wal_threshold=100_000,
        )
        candidate = _run_scenario(
            source=source,
            target=targets[1],
            run_id=selected,
            frame_indices=frame_indices,
            name="candidate_512MiB",
            checkpoint_threshold="512MiB",
            skip_wal_threshold=512 * 1024 * 1024,
        )
        if default.signature != expected_prefix or candidate.signature != expected_prefix:
            raise RuntimeError("full-path benchmark changed the real-row signature")
        ratio = None
        if (
            default.process_write_transfer_bytes
            and candidate.process_write_transfer_bytes is not None
        ):
            ratio = (
                candidate.process_write_transfer_bytes
                / default.process_write_transfer_bytes
            )
        print(
            json.dumps(
                {
                    "schema_version": "round-073-full-path-checkpoint-benchmark-v1",
                    "source_run_id": selected,
                    "synthetic_market_data_used": False,
                    "frame_indices": frame_indices,
                    "message_count": prefix_messages,
                    "tables": _FULL_PATH_TABLES,
                    "default": asdict(default),
                    "candidate": asdict(candidate),
                    "candidate_to_default_process_write_ratio": ratio,
                    "scratch_files_retained": False,
                    "financial_or_model_evidence": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        for target in targets:
            _remove_exact_files(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

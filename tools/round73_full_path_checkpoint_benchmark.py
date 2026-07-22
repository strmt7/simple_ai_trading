#!/usr/bin/env python3
"""Compare DuckDB checkpoint policies on bounded real Round 73 frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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


_V5_FULL_PATH_TABLES = (
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

_V8_FULL_PATH_TABLES = (
    "impact_capture_frame_v8",
    "impact_event_link_v8",
    "impact_depth_update_v8",
    "impact_depth_band_flow_v8",
    "impact_l2_state_v8",
    "impact_book_ticker_v8",
    "impact_aggregate_trade_v8",
    "impact_mark_price_v8",
    "impact_liquidation_snapshot_v8",
    "impact_rest_event_v8",
    "impact_rejected_wire_event_v8",
)

_SCHEMA_LAYOUTS = {
    "round-073-prospective-evidence-v5": _V5_FULL_PATH_TABLES,
    "round-073-prospective-evidence-v6": _V5_FULL_PATH_TABLES,
    "round-073-prospective-evidence-v7": _V5_FULL_PATH_TABLES,
    "round-073-prospective-evidence-v8": _V8_FULL_PATH_TABLES,
}


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


def _write_immutable_report(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError("benchmark output already exists with different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    checkpoint_threshold: str,
    skip_wal_threshold: int,
) -> None:
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET checkpoint_threshold='{checkpoint_threshold}'")
    connection.execute("SET auto_checkpoint_skip_wal_threshold=?", [skip_wal_threshold])


def _insert_frame_transactions(
    connection: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    frame_indices: tuple[int, ...],
    tables: tuple[str, ...],
    wal_path: Path | None = None,
    transaction_deltas: list[int | None] | None = None,
) -> int:
    maximum_wal = 0
    for frame_index in frame_indices:
        before_transaction = _process_io_snapshot().write_bytes
        connection.execute("BEGIN TRANSACTION")
        try:
            for table in tables:
                connection.execute(
                    f"INSERT INTO main.{table} SELECT * FROM source_db.{table} "
                    "WHERE run_id = ? AND frame_index = ?",
                    [run_id, frame_index],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        if transaction_deltas is not None:
            after_transaction = _process_io_snapshot().write_bytes
            transaction_deltas.append(
                _write_delta(before_transaction, after_transaction)
            )
        if wal_path is not None:
            maximum_wal = max(maximum_wal, _wal_bytes(wal_path))
    return maximum_wal


def _signature(
    connection: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    run_id: str,
    tables: tuple[str, ...],
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
        for table in tables
    }
    frame_table = tables[0]
    totals = connection.execute(
        f"SELECT coalesce(sum(message_count), 0), "
        f"coalesce(sum(compressed_bytes), 0) FROM {schema}.{frame_table} "
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
    tables: tuple[str, ...],
    seed_run_id: str | None,
    seed_frame_indices: tuple[int, ...],
    name: str,
    checkpoint_threshold: str,
    skip_wal_threshold: int,
) -> _Scenario:
    _remove_exact_files(target)
    with ImpactAbsorptionStore(target, memory_limit="1GB", threads=1) as store:
        store.connect()

    escaped_source = str(source).replace("'", "''")
    if seed_run_id is not None:
        seed_connection = duckdb.connect(str(target))
        _configure_connection(
            seed_connection,
            checkpoint_threshold=checkpoint_threshold,
            skip_wal_threshold=skip_wal_threshold,
        )
        seed_connection.execute(f"ATTACH '{escaped_source}' AS source_db (READ_ONLY)")
        _insert_frame_transactions(
            seed_connection,
            run_id=seed_run_id,
            frame_indices=seed_frame_indices,
            tables=tables,
        )
        seed_connection.close()

    started_io = _process_io_snapshot()
    started = time.perf_counter()
    maximum_wal = 0
    transaction_deltas: list[int | None] = []
    connection = duckdb.connect(str(target))
    _configure_connection(
        connection,
        checkpoint_threshold=checkpoint_threshold,
        skip_wal_threshold=skip_wal_threshold,
    )
    effective_checkpoint = str(
        connection.execute("SELECT current_setting('checkpoint_threshold')").fetchone()[
            0
        ]
    )
    effective_skip = int(
        connection.execute(
            "SELECT current_setting('auto_checkpoint_skip_wal_threshold')"
        ).fetchone()[0]
    )
    connection.execute(f"ATTACH '{escaped_source}' AS source_db (READ_ONLY)")
    maximum_wal = _insert_frame_transactions(
        connection,
        run_id=run_id,
        frame_indices=frame_indices,
        tables=tables,
        wal_path=target,
        transaction_deltas=transaction_deltas,
    )
    before_close = _process_io_snapshot().write_bytes
    connection.close()
    after_close = _process_io_snapshot().write_bytes
    elapsed = time.perf_counter() - started
    ended_io = _process_io_snapshot()
    with duckdb.connect(str(target), read_only=True) as verify:
        signature = _signature(
            verify,
            schema="main",
            run_id=run_id,
            tables=tables,
        )
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
    parser.add_argument("--seed-run-id")
    parser.add_argument("--seed-frames", type=int, default=0)
    parser.add_argument("--scratch-dir", default="data")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= int(args.frames) <= 64:
        raise SystemExit("frames must be between 1 and 64")
    if bool(args.seed_run_id) != (int(args.seed_frames) > 0):
        raise SystemExit("seed-run-id and a positive seed-frames must be used together")
    if not 0 <= int(args.seed_frames) <= 64:
        raise SystemExit("seed-frames must be between 0 and 64")
    root = _repo_root().resolve()
    source = Path(args.database).resolve()
    scratch = Path(args.scratch_dir).resolve()
    if root not in scratch.parents and scratch != root:
        raise SystemExit("scratch-dir must remain inside the repository")
    scratch_names = {
        "round73-full-path-default.duckdb",
        "round73-full-path-candidate.duckdb",
        "round73-frame-only-candidate.duckdb",
    }
    if source in {scratch / name for name in scratch_names}:
        raise SystemExit("source database cannot be a scratch target")
    scratch.mkdir(parents=True, exist_ok=True)
    selected = str(args.run_id).strip().lower()
    with duckdb.connect(str(source), read_only=True) as source_connection:
        run_row = source_connection.execute(
            "SELECT schema_version FROM impact_capture_run WHERE run_id = ?",
            [selected],
        ).fetchone()
        if run_row is None:
            raise SystemExit("source run was not found")
        source_schema_version = str(run_row[0])
        try:
            tables = _SCHEMA_LAYOUTS[source_schema_version]
        except KeyError as exc:
            raise SystemExit(
                f"source run schema is unsupported: {source_schema_version}"
            ) from exc
        frame_table = tables[0]
        frame_rows = source_connection.execute(
            f"SELECT frame_index FROM {frame_table} WHERE run_id = ? "
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
            tables=tables,
            frame_count=int(args.frames),
        )
        prefix_messages = expected_prefix["frame_message_count_sum"]
        seed_run_id = (
            None if args.seed_run_id is None else str(args.seed_run_id).strip().lower()
        )
        seed_frame_indices: tuple[int, ...] = ()
        if seed_run_id is not None:
            seed_row = source_connection.execute(
                "SELECT schema_version FROM impact_capture_run WHERE run_id = ?",
                [seed_run_id],
            ).fetchone()
            if seed_row is None:
                raise SystemExit("seed run was not found")
            if _SCHEMA_LAYOUTS.get(str(seed_row[0])) != tables:
                raise SystemExit("seed and source runs use different table layouts")
            seed_rows = source_connection.execute(
                f"SELECT frame_index FROM {frame_table} WHERE run_id = ? "
                "ORDER BY frame_index LIMIT ?",
                [seed_run_id, int(args.seed_frames)],
            ).fetchall()
            seed_frame_indices = tuple(int(row[0]) for row in seed_rows)
            if seed_frame_indices != tuple(range(int(args.seed_frames))):
                raise SystemExit("seed run does not contain the requested frame prefix")
    targets = (
        scratch / "round73-full-path-default.duckdb",
        scratch / "round73-full-path-candidate.duckdb",
        scratch / "round73-frame-only-candidate.duckdb",
    )
    try:
        default = _run_scenario(
            source=source,
            target=targets[0],
            run_id=selected,
            frame_indices=frame_indices,
            tables=tables,
            seed_run_id=seed_run_id,
            seed_frame_indices=seed_frame_indices,
            name="observed_defaults",
            checkpoint_threshold="16MiB",
            skip_wal_threshold=100_000,
        )
        candidate = _run_scenario(
            source=source,
            target=targets[1],
            run_id=selected,
            frame_indices=frame_indices,
            tables=tables,
            seed_run_id=seed_run_id,
            seed_frame_indices=seed_frame_indices,
            name="candidate_512MiB",
            checkpoint_threshold="512MiB",
            skip_wal_threshold=512 * 1024 * 1024,
        )
        frame_only_tables = (frame_table,)
        with duckdb.connect(str(source), read_only=True) as source_connection:
            expected_frame_prefix = _signature(
                source_connection,
                schema="main",
                run_id=selected,
                tables=frame_only_tables,
                frame_count=int(args.frames),
            )
        frame_only = _run_scenario(
            source=source,
            target=targets[2],
            run_id=selected,
            frame_indices=frame_indices,
            tables=frame_only_tables,
            seed_run_id=seed_run_id,
            seed_frame_indices=seed_frame_indices,
            name="candidate_512MiB_exact_frames_only",
            checkpoint_threshold="512MiB",
            skip_wal_threshold=512 * 1024 * 1024,
        )
        if (
            default.signature != expected_prefix
            or candidate.signature != expected_prefix
        ):
            raise RuntimeError("full-path benchmark changed the real-row signature")
        if frame_only.signature != expected_frame_prefix:
            raise RuntimeError("frame-only benchmark changed the exact-frame signature")
        ratio = None
        if (
            default.process_write_transfer_bytes
            and candidate.process_write_transfer_bytes is not None
        ):
            ratio = (
                candidate.process_write_transfer_bytes
                / default.process_write_transfer_bytes
            )
        frame_only_ratio = None
        if (
            candidate.process_write_transfer_bytes
            and frame_only.process_write_transfer_bytes is not None
        ):
            frame_only_ratio = (
                frame_only.process_write_transfer_bytes
                / candidate.process_write_transfer_bytes
            )
        report: dict[str, object] = {
            "schema_version": "round-073-full-path-checkpoint-benchmark-v4",
            "source_run_id": selected,
            "source_schema_version": source_schema_version,
            "seed_run_id": seed_run_id,
            "seed_frame_indices": seed_frame_indices,
            "synthetic_market_data_used": False,
            "frame_indices": frame_indices,
            "message_count": prefix_messages,
            "tables": tables,
            "default": asdict(default),
            "candidate": asdict(candidate),
            "candidate_to_default_process_write_ratio": ratio,
            "exact_frame_candidate": asdict(frame_only),
            "exact_frame_to_full_path_candidate_process_write_ratio": (
                frame_only_ratio
            ),
            "exact_frame_candidate_persists_typed_projection": False,
            "exact_frame_candidate_purpose": (
                "Measure the storage ceiling available to a prospective "
                "raw-frame schema; it is not an implemented capture schema."
            ),
            "scratch_files_retained": False,
            "financial_or_model_evidence": False,
        }
        report["artifact_sha256"] = _canonical_sha256(report)
        if args.output is not None:
            output = Path(args.output).resolve()
            if root not in output.parents:
                raise SystemExit("output must remain inside the repository")
            _write_immutable_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        for target in targets:
            _remove_exact_files(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

from tools.ingest_round72_spot_perpetual_corpus import (
    _terminal_report,
    _write_terminal_artifacts,
    build_parser,
)


def test_round72_ingester_runtime_defaults_are_bounded() -> None:
    arguments = build_parser().parse_args([])

    assert arguments.memory_limit == "4GB"
    assert arguments.threads == 8
    assert arguments.network_retries == 12
    assert arguments.maximum_uncompressed_gb == 8


def test_terminal_report_cannot_overwrite_the_warehouse(tmp_path: Path) -> None:
    warehouse = tmp_path / "warehouse.duckdb"
    warehouse.write_bytes(b"database-evidence")
    progress = tmp_path / "progress.json"
    arguments = build_parser().parse_args(
        [
            "--design",
            str(tmp_path / "design.json"),
            "--inventory",
            str(tmp_path / "inventory.json"),
            "--warehouse",
            str(warehouse),
            "--output",
            str(warehouse),
            "--progress",
            str(progress),
        ]
    )
    report = _terminal_report(
        status="failed",
        event="round72_spot_perpetual_corpus_failed",
        error="ValueError: path alias",
        started_at_utc="2026-07-22T00:00:00+00:00",
    )

    _write_terminal_artifacts(arguments, report)

    assert warehouse.read_bytes() == b"database-evidence"
    assert progress.is_file()

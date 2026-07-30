"""Run the one-use Polymarket Round 20 storage/transport qualification."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Mapping

from simple_ai_trading.polymarket_round20_capture import (
    POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS,
    build_round20_capture_manifest,
    create_round20_recorder,
    evaluate_round20_capture,
    verify_round20_repository_attestation,
)
from simple_ai_trading.polymarket_round20_contract import load_round20_contract
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-020-independent-redundant-corpus-contract-v1.json"
)
DEFAULT_DATABASE = ROOT / "data" / "round20-capture-qualification.duckdb"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "evidence"
    / "round-020-capture-qualification-v1.json"
)


def _progress(phase: str, details: Mapping[str, object]) -> None:
    selected = {
        key: details.get(key)
        for key in (
            "run_id",
            "elapsed_seconds",
            "duration_seconds",
            "written_message_count",
            "written_gap_count",
            "written_gap_counts",
            "received_message_count",
            "received_stream_counts",
            "queue_capacity",
            "queue_high_watermark",
            "queue_size",
            "error_count",
            "status",
            "report_sha256",
        )
        if key in details
    }
    print(
        json.dumps(
            {"phase": phase, **selected},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


async def _run(database: Path, output: Path) -> dict[str, object]:
    if database.exists():
        raise FileExistsError(f"Round 20 database already exists: {database}")
    if output.exists():
        raise FileExistsError(f"Round 20 output already exists: {output}")
    program = load_round20_contract(CONTRACT)
    recorder = create_round20_recorder(database)
    manifest_holder: dict[str, object] = {}

    def manifest_factory(run_id: str, created_at_ms: int) -> Mapping[str, object]:
        manifest = build_round20_capture_manifest(
            ROOT,
            run_id=run_id,
            created_at_ms=created_at_ms,
        )
        manifest_holder.update(manifest)
        return manifest

    report = await recorder.run(
        duration_seconds=POLYMARKET_ROUND20_QUALIFICATION_DURATION_SECONDS,
        progress=_progress,
        progress_interval_seconds=30,
        preregistration_manifest_factory=manifest_factory,
    )
    if not manifest_holder:
        raise RuntimeError("Round 20 capture manifest was not created")
    verify_round20_repository_attestation(ROOT, manifest_holder)
    result = evaluate_round20_capture(
        database=database,
        report=report,
        program=program,
        capture_manifest_sha256=str(manifest_holder["manifest_sha256"]),
    )
    encoded = (
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_atomic(output, encoded)
    print(
        json.dumps(
            {
                "phase": "qualification-complete",
                "qualified": result["qualified"],
                "result_sha256": result["result_sha256"],
                "output": str(output),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    try:
        result = asyncio.run(
            _run(arguments.database.resolve(), arguments.output.resolve())
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "phase": "qualification-failed-before-result",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2_000],
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

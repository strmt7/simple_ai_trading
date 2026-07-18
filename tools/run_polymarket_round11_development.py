"""Run the frozen Round 11 development-only Polymarket experiment once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from simple_ai_trading.polymarket_directional_value import (
    POLYMARKET_ROUND11_CONTRACT_SHA256,
    build_round11_development_split,
    fit_round11_development,
    load_round11_development_dataset,
)
from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-contract.json"
)
DEFAULT_REPORT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-report.json"
)
DEFAULT_ARTIFACT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-artifact.json"
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _verify_contract(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.pop("contract_sha256", ""))
    actual = _canonical_sha256(payload)
    if claimed != POLYMARKET_ROUND11_CONTRACT_SHA256 or actual != claimed:
        raise ValueError("Round 11 contract identity differs from the frozen design")


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _progress(stage: str, payload: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "elapsed_seconds": round(time.monotonic() - START, 3),
                **payload,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--pipeline-report-sha256", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--database-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database = args.database.resolve()
    contract = args.contract.resolve()
    report_path = args.report.resolve()
    artifact_path = args.artifact.resolve()
    if not database.is_file() or not contract.is_file():
        raise FileNotFoundError("Round 11 database or contract does not exist")
    if report_path.exists() or artifact_path.exists():
        raise FileExistsError(
            "Round 11 development output already exists; rerun prohibited"
        )
    _verify_contract(contract)
    store = PolymarketEvidenceStore(
        database,
        memory_limit=args.memory_limit,
        threads=args.database_threads,
        read_only=True,
    )
    try:
        dataset = load_round11_development_dataset(
            store,
            pipeline_report_sha256=args.pipeline_report_sha256,
            progress=_progress,
        )
        split = build_round11_development_split(dataset)
        _progress(
            "dataset_ready",
            {
                "dataset_sha256": dataset.dataset_sha256,
                "rows": dataset.rows,
                "conditions": len(dataset.condition_ids),
                "groups": len(dataset.groups),
                "split_sha256": split.split_sha256,
            },
        )
        report, artifact = fit_round11_development(
            dataset,
            split,
            progress=_progress,
        )
    finally:
        if store.connection is not None:
            store.connection.close()
        if store.payload_connection is not None:
            store.payload_connection.close()
    _write_new_json(artifact_path, artifact)
    _write_new_json(report_path, report)
    _progress(
        "written",
        {
            "artifact": str(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "report": str(report_path),
            "report_sha256": report["report_sha256"],
            "development_passed": report["development_passed"],
        },
    )
    return 0 if report["development_passed"] else 2


START = time.monotonic()


if __name__ == "__main__":
    sys.exit(main())

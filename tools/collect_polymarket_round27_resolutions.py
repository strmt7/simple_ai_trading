from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from simple_ai_trading.polymarket_round27_resolution import (
    audit_round27_resolution_collection,
    collect_round27_resolutions_once,
    finalize_round27_resolution_collection,
    initialize_round27_resolution_collection,
    write_round27_resolution_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect target-isolated dual-source Round 27 settlements."
    )
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--condition-audit", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--capture-contract", type=Path, required=True)
    parser.add_argument("--capture-result", type=Path, required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--destination-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-only", action="store_true")
    return parser


def _progress(event: str, value: object) -> None:
    print(
        json.dumps({"event": event, "value": value}, sort_keys=True),
        flush=True,
    )


def main() -> int:
    args = _parser().parse_args()
    collection, claim = initialize_round27_resolution_collection(
        source_database=args.source_database,
        condition_audit_path=args.condition_audit,
        preregistration_path=args.preregistration,
        capture_contract_path=args.capture_contract,
        capture_result_path=args.capture_result,
        mechanics_path=args.mechanics,
        destination_database=args.destination_database,
        created_at_ms=time.time_ns() // 1_000_000,
    )
    _progress(
        "target_access_claim_persisted",
        {
            "claim_sha256": claim["claim_sha256"],
            "condition_count": claim["condition_count"],
            "database": str(collection),
        },
    )
    if args.initialize_only:
        return 0
    if collection == args.destination_database:
        report = audit_round27_resolution_collection(
            collection,
            source_database=args.source_database,
        )
    else:
        collection_report = collect_round27_resolutions_once(
            source_database=args.source_database,
            collection_database=collection,
            progress=_progress,
        )
        _progress("collection_complete", collection_report)
        report = finalize_round27_resolution_collection(
            collection_database=collection,
            destination_database=args.destination_database,
            source_database=args.source_database,
        )
    write_round27_resolution_audit(args.output, report)
    _progress("audit_written", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create, inspect, run, or supervise the Round 27 Stage 1 campaign."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Mapping

from simple_ai_trading.polymarket_round27_stage1_capture import (
    Round27Stage1Slot,
    Round27Stage1SlotConfig,
    create_round27_stage1_contract,
    load_round27_stage1_contract,
    run_round27_stage1_slot,
    supervise_round27_stage1_primary,
    write_round27_stage1_contract,
)
from simple_ai_trading.storage import write_json_atomic


DEFAULT_CONTRACT = Path(
    "docs/model-research/polymarket/round-027-stage1-campaign-contract-v1.json"
)
DEFAULT_DATA_ROOT = Path("data/round27-stage1-campaign-v1")
DEFAULT_STATE = Path("data/round27-stage1-campaign-v1-state.json")
DEFAULT_LEASE = Path("data/round27-stage1-campaign-v1.lock")


def _utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise argparse.ArgumentTypeError("slot timestamps must be UTC ISO-8601")
    return int(parsed.timestamp() * 1_000)


def _slot(value: str) -> Round27Stage1Slot:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("slot must be id,role,start_utc,end_utc")
    return Round27Stage1Slot(parts[0], parts[1], _utc_ms(parts[2]), _utc_ms(parts[3]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("create", "inspect", "run-slot", "supervise")
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lease", type=Path, default=DEFAULT_LEASE)
    parser.add_argument("--slot", action="append", type=_slot, default=[])
    parser.add_argument("--slot-id")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _paths(data_root: Path, slot_id: str) -> dict[str, Path]:
    return {
        "database": data_root / f"round27-{slot_id}.duckdb",
        "result": data_root / f"round27-{slot_id}-result.json",
        "progress": data_root / f"round27-{slot_id}-progress.json",
        "lock": data_root / f"round27-{slot_id}.lock",
    }


def main() -> int:
    arguments = _parser().parse_args()
    repository = arguments.repository.resolve()
    contract_path = _resolve(repository, arguments.contract)
    data_root = _resolve(repository, arguments.data_root)
    if arguments.action == "create":
        contract = create_round27_stage1_contract(
            repository,
            created_at_ms=time.time_ns() // 1_000_000,
            slots=arguments.slot,
        )
        write_round27_stage1_contract(contract_path, contract)
        print(json.dumps(contract, sort_keys=True))
        return 0
    if arguments.action == "inspect":
        contract = load_round27_stage1_contract(contract_path, repository=repository)
        print(
            json.dumps(
                {
                    "contract_sha256": contract.contract_sha256,
                    "repository_commit": contract.repository_commit,
                    "slots": [slot.asdict() for slot in contract.slots],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.action == "supervise":
        result = supervise_round27_stage1_primary(
            repository=repository,
            contract_path=contract_path,
            data_root=data_root,
            state_path=_resolve(repository, arguments.state),
            lease_path=_resolve(repository, arguments.lease),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if not arguments.slot_id:
        raise SystemExit("--slot-id is required for run-slot")
    paths = _paths(data_root, arguments.slot_id)

    def progress(phase: str, value: Mapping[str, object]) -> None:
        payload = {"phase": phase, **dict(value)}
        write_json_atomic(paths["progress"], payload, indent=2, sort_keys=True)
        print(json.dumps(payload, sort_keys=True), flush=True)

    result = asyncio.run(
        run_round27_stage1_slot(
            Round27Stage1SlotConfig(
                repository=repository,
                contract_path=contract_path,
                slot_id=arguments.slot_id,
                database_path=paths["database"],
                result_path=paths["result"],
                lock_path=paths["lock"],
            ),
            progress=progress,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

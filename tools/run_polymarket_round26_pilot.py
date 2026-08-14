"""Create or run the preregistered Round 26 TWAP-60 development pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Mapping

from simple_ai_trading.polymarket_round26_pilot import (
    Round26PilotConfig,
    create_round26_pilot_contract,
    run_round26_pilot,
    write_round26_pilot_contract,
)
from simple_ai_trading.storage import write_json_atomic


DEFAULT_CONTRACT = Path(
    "docs/model-research/polymarket/round-026-twap60-development-pilot-v1.json"
)
DEFAULT_DATABASE = Path("data/round26-twap60-development-pilot-v1.duckdb")
DEFAULT_RESULT = Path("data/round26-twap60-development-pilot-v1-result.json")
DEFAULT_PROGRESS = Path("data/round26-twap60-development-pilot-v1-progress.json")
DEFAULT_LOCK = Path("data/round26-twap60-development-pilot-v1.lock")


def _next_start(now_ms: int) -> int:
    return ((int(now_ms) + 120_000 + 299_999) // 300_000) * 300_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "run"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> int:
    arguments = _parser().parse_args()
    repository = arguments.repository.resolve()
    contract_path = _resolve(repository, arguments.contract)
    if arguments.action == "create":
        created = time.time_ns() // 1_000_000
        contract = create_round26_pilot_contract(
            repository,
            created_at_ms=created,
            effective_start_ms=_next_start(created),
        )
        write_round26_pilot_contract(contract_path, contract)
        print(json.dumps(contract, sort_keys=True))
        return 0
    progress_path = _resolve(repository, arguments.progress)

    def progress(phase: str, value: Mapping[str, object]) -> None:
        payload = {"phase": phase, **dict(value)}
        write_json_atomic(progress_path, payload, indent=2, sort_keys=True)
        print(json.dumps(payload, sort_keys=True), flush=True)

    result = asyncio.run(
        run_round26_pilot(
            Round26PilotConfig(
                repository=repository,
                contract_path=contract_path,
                database_path=_resolve(repository, arguments.database),
                result_path=_resolve(repository, arguments.result),
                lock_path=_resolve(repository, arguments.lock),
            ),
            progress=progress,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

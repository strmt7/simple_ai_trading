from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_round22_pilot import (  # noqa: E402
    Round22PilotStore,
    load_round22_pilot_contract,
)
from simple_ai_trading.polymarket_round22_targets import (  # noqa: E402
    collect_round22_diagnostic_targets,
    open_round22_diagnostic_target_claim,
    validate_round22_diagnostic_target_opening,
)


DEFAULT_DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _emit(event: str, values: Mapping[str, object]) -> None:
    print(_canonical_json({"event": event, **dict(values)}), flush=True)


def _status(store: Round22PilotStore) -> dict[str, object]:
    state = str(
        store.connection.execute(
            "SELECT state FROM feature.pilot_manifest WHERE singleton"
        ).fetchone()[0]
    )
    claim_table = bool(
        store.connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'target' AND table_name = 'round22_access_claim'
            """
        ).fetchone()[0]
    )
    claim_status = "not_opened"
    if claim_table:
        claim_status = str(
            store.connection.execute(
                "SELECT status FROM target.round22_access_claim WHERE singleton"
            ).fetchone()[0]
        )
    return {
        "authentication_used": False,
        "claim_status": claim_status,
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "polymarket_order_submission": False,
        "state": state,
        "target_row_count": store.target_row_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run claim-gated Round 22 public resolution collection."
    )
    parser.add_argument("phase", choices=("status", "preflight", "collect"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--maximum-conditions", type=int, choices=range(1, 37), default=36
    )
    args = parser.parse_args()
    contract = load_round22_pilot_contract(ROOT)
    read_only = args.phase in {"status", "preflight"}
    with Round22PilotStore(
        args.database, contract=contract, read_only=read_only
    ) as store:
        if args.phase == "status":
            _emit("round22_target_status", _status(store))
            return 0
        if args.phase == "preflight":
            _emit(
                "round22_target_preflight",
                validate_round22_diagnostic_target_opening(store),
            )
            return 0
        claim_sha = open_round22_diagnostic_target_claim(store)
        _emit("round22_target_claim", {"claim_sha256": claim_sha})
        result = collect_round22_diagnostic_targets(
            store,
            maximum_conditions=args.maximum_conditions,
            progress=_emit,
        )
        _emit("round22_target_result", asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

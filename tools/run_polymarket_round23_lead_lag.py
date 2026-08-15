from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_round22_pilot import (  # noqa: E402
    Round22PilotStore,
    load_round22_pilot_contract,
)
from simple_ai_trading.polymarket_round23_lead_lag import (  # noqa: E402
    run_round23_lead_lag_diagnostic,
)


DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"
RESULT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-023-lead-lag-results-v1.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def main() -> int:
    contract = load_round22_pilot_contract(ROOT)
    print(
        _canonical_json(
            {
                "event": "round23_lead_lag_start",
                "mode": "read_only_exploratory",
            }
        ),
        flush=True,
    )
    with Round22PilotStore(DATABASE, contract=contract, read_only=True) as store:
        result = run_round23_lead_lag_diagnostic(store)
    encoded = (
        json.dumps(
            result,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if RESULT.exists():
        existing = json.loads(RESULT.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(result):
            raise ValueError("Round 23 result already exists with different evidence")
        disposition = "verified_existing"
    else:
        temporary = RESULT.with_suffix(RESULT.suffix + ".tmp")
        if temporary.exists():
            raise ValueError("Round 23 temporary result path is not clean")
        temporary.write_text(encoded, encoding="ascii", newline="\n")
        if RESULT.exists():
            temporary.unlink()
            raise ValueError("Round 23 result appeared concurrently")
        os.replace(temporary, RESULT)
        disposition = "published"
    print(
        _canonical_json(
            {
                "conclusion": result["conclusion"],
                "disposition": disposition,
                "event": "round23_lead_lag_complete",
                "mechanism_gate_passed": result["mechanism_gate_passed"],
                "result_sha256": result["result_sha256"],
                "selection": result["selection"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

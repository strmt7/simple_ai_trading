from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_round22_diagnostic import (  # noqa: E402
    run_round22_diagnostic,
)
from simple_ai_trading.polymarket_round22_pilot import (  # noqa: E402
    Round22PilotStore,
    load_round22_pilot_contract,
)


DEFAULT_DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-022-diagnostic-results-v1.json"
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
    parser = argparse.ArgumentParser(
        description="Run the frozen Round 22 condition-clustered diagnostic."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = load_round22_pilot_contract(ROOT)
    with Round22PilotStore(args.database, contract=contract, read_only=True) as store:
        result = run_round22_diagnostic(store)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise ValueError("Round 22 diagnostic result already differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    selection = result["selection"]["calibrated"]
    print(
        _canonical_json(
            {
                "conclusion": result["conclusion"],
                "diagnostic_pass": result["diagnostic_pass"],
                "output": str(output),
                "result_sha256": result["result_sha256"],
                "selection": selection,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

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
from simple_ai_trading.polymarket_round23_binance import (  # noqa: E402
    ingest_round23_binance_archives,
)


DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _emit(event: str, values: Mapping[str, object]) -> None:
    print(_canonical_json({"event": event, **dict(values)}), flush=True)


def main() -> int:
    contract = load_round22_pilot_contract(ROOT)
    with Round22PilotStore(DATABASE, contract=contract) as store:
        results = ingest_round23_binance_archives(store, progress=_emit)
    _emit(
        "round23_ingestion_result",
        {
            "archive_count": len(results),
            "downloaded_count": sum(result.downloaded for result in results),
            "rows": sum(result.row_count for result in results),
            "results": [asdict(result) for result in results],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a hash-bound certificate for the frozen Round 42 one-second source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.second_flow_data import (  # noqa: E402
    load_verified_second_flow,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _, certificate = load_verified_second_flow(arguments.database.resolve())
    write_json_atomic(arguments.output.resolve(), certificate, indent=2, sort_keys=True)
    print(json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

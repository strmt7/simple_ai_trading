"""Analyze the finished Round 26 development pilot without execution authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_ai_trading.polymarket_round26_analysis import (
    run_round26_pilot_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/model-research/polymarket/round-026-twap60-development-pilot-v3.json"
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/round26-twap60-development-pilot-v3.duckdb"),
    )
    parser.add_argument(
        "--capture-result",
        type=Path,
        default=Path("data/round26-twap60-development-pilot-v3-result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/round26-twap60-development-analysis-v3.json"),
    )
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.repository.resolve()
    result = run_round26_pilot_analysis(
        root,
        contract_path=_resolve(root, arguments.contract),
        database_path=_resolve(root, arguments.database),
        capture_result_path=_resolve(root, arguments.capture_result),
        output_path=_resolve(root, arguments.output),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Publish a sealed Polymarket Round 17 one-use result without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simple_ai_trading.polymarket_round17_publication import (
    load_round17_one_use_result,
    publish_round17_one_use_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish numeric tables and derived graphs for the "
            "terminal Round 17 BTC five-minute one-use result."
        )
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = load_round17_one_use_result(Path(args.result))
        publication = publish_round17_one_use_result(
            result,
            Path(args.output_dir),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"round17-polymarket-publication failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "round": 17,
                "status": publication["status"],
                "result_sha256": publication["result_sha256"],
                "publication_sha256": publication["publication_sha256"],
                "artifact_count": len(publication["artifacts"]),
                "output_dir": str(Path(args.output_dir).resolve()),
                "profitability_claim": publication["profitability_claim"],
                "live_trading_authority": publication["live_trading_authority"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

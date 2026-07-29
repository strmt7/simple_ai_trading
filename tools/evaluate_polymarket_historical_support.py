#!/usr/bin/env python3
"""Evaluate the frozen Polymarket support gate without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
    load_historical_screen_contract,
)
from simple_ai_trading.polymarket_historical_shadow import (  # noqa: E402
    load_verified_historical_shadow_predictor,
)
from simple_ai_trading.polymarket_historical_support_evaluation import (  # noqa: E402
    evaluate_historical_feature_support_once,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one pre-frozen BTC feature-support gate."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pretest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=args.pretest,
        evaluation_path=args.evaluation,
        support_path=args.support,
    )
    historical_evaluation = json.loads(
        args.evaluation.read_text(encoding="utf-8")
    )
    contract = load_historical_screen_contract(args.contract)
    with HistoricalScreenStore(
        args.database,
        contract=contract,
        read_only=True,
    ) as store:
        artifact, artifact_sha = evaluate_historical_feature_support_once(
            store,
            predictor,
            historical_evaluation=historical_evaluation,
            source_commit=args.source_commit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            artifact,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha,
                "output": str(args.output),
                "tune_row_coverage": artifact["roles"]["tune"][
                    "row_coverage"
                ],
                "test_row_coverage": artifact["roles"]["test"][
                    "row_coverage"
                ],
                "predictive_improvement_claim": False,
                "trading_authority": False,
                "profitability_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

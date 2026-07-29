#!/usr/bin/env python3
"""Freeze train-only support bounds for Polymarket historical shadow inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_model import (  # noqa: E402
    load_historical_model_panel,
)
from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
    load_historical_screen_contract,
)
from simple_ai_trading.polymarket_historical_support import (  # noqa: E402
    freeze_historical_feature_support,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze train-only BTC historical feature-support bounds."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pretest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    pretest = json.loads(args.pretest.read_text(encoding="utf-8"))
    pretest_sha = str(pretest.get("artifact_sha256") or "")
    contract = load_historical_screen_contract(args.contract)
    with HistoricalScreenStore(
        args.database,
        contract=contract,
        read_only=True,
    ) as store:
        train = load_historical_model_panel(store, roles=("train",))
    artifact, artifact_sha = freeze_historical_feature_support(
        train,
        pretest_artifact_sha256=pretest_sha,
        source_commit=args.source_commit,
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
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
    temporary.replace(output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha,
                "output": str(output),
                "training_rows": int(artifact["training_rows"]),
                "training_conditions": int(artifact["training_conditions"]),
                "test_features_used": False,
                "live_features_used": False,
                "trading_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

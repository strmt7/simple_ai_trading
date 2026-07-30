"""Run the terminal, development-only Polymarket Round 17 operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from simple_ai_trading.polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
)
from simple_ai_trading.polymarket_round17_development_operator import (
    Round17DevelopmentOperatorConfig,
    run_round17_development,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Polymarket Round 17 development sequence after "
            "the prospective capture is terminal."
        )
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--campaign-plan", required=True)
    parser.add_argument("--cohort-plan", required=True)
    parser.add_argument("--admission-spec", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--risk-contract", required=True)
    parser.add_argument("--economic-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--database-threads", type=int, default=2)
    return parser


def _resolved(repository: Path, value: str) -> Path:
    selected = Path(value)
    return selected.resolve() if selected.is_absolute() else (repository / selected).resolve()


def _progress(value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {"round": 17, **dict(value)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(args.repository).resolve()
    campaign = Round17CampaignOperatorConfig(
        campaign_plan_path=_resolved(repository, args.campaign_plan),
        cohort_plan_path=_resolved(repository, args.cohort_plan),
        admission_spec_path=_resolved(repository, args.admission_spec),
        database_path=_resolved(repository, args.database),
        state_root=_resolved(repository, args.state_root),
        memory_limit=args.memory_limit,
        database_threads=args.database_threads,
    )
    config = Round17DevelopmentOperatorConfig(
        campaign=campaign,
        risk_contract_path=_resolved(repository, args.risk_contract),
        economic_contract_path=_resolved(repository, args.economic_contract),
        output_path=_resolved(repository, args.output),
        compute_backend=args.compute_backend,
    )
    try:
        result = run_round17_development(config, progress=_progress)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"round17-polymarket-development failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "round": 17,
                "status": result["status"],
                "development_accepted": result["development_accepted"],
                "profitability_claim": result["profitability_claim"],
                "live_trading_authority": result["live_trading_authority"],
                "result_sha256": result["result_sha256"],
                "output": str(config.output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] != "resolution_incomplete" else 3


if __name__ == "__main__":
    raise SystemExit(main())

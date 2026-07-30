"""Run or resume the frozen Polymarket Round 17 one-use held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from simple_ai_trading.polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
)
from simple_ai_trading.polymarket_round17_evaluation import (
    Round17OneUseEvaluationConfig,
    run_round17_one_use_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume the preregistered Round 17 BTC five-minute test partition "
            "once after an accepted development result."
        )
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--repository-commit")
    parser.add_argument("--campaign-plan", required=True)
    parser.add_argument("--cohort-plan", required=True)
    parser.add_argument("--admission-spec", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--evaluation-contract", required=True)
    parser.add_argument("--development-result", required=True)
    parser.add_argument("--risk-contract", required=True)
    parser.add_argument("--claim-store", required=True)
    parser.add_argument("--resolution-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--database-threads", type=int, default=2)
    return parser


def _resolved(repository: Path, value: str) -> Path:
    selected = Path(value)
    return (
        selected.resolve()
        if selected.is_absolute()
        else (repository / selected).resolve()
    )


def _repository_commit(repository: Path, selected: str | None) -> str:
    if selected:
        return selected.strip().lower()
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().lower()


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
    try:
        campaign = Round17CampaignOperatorConfig(
            campaign_plan_path=_resolved(repository, args.campaign_plan),
            cohort_plan_path=_resolved(repository, args.cohort_plan),
            admission_spec_path=_resolved(repository, args.admission_spec),
            database_path=_resolved(repository, args.database),
            state_root=_resolved(repository, args.state_root),
            memory_limit=args.memory_limit,
            database_threads=args.database_threads,
        )
        config = Round17OneUseEvaluationConfig(
            repository=repository,
            repository_commit_sha=_repository_commit(
                repository,
                args.repository_commit,
            ),
            contract_path=_resolved(repository, args.evaluation_contract),
            development_result_path=_resolved(
                repository,
                args.development_result,
            ),
            risk_contract_path=_resolved(repository, args.risk_contract),
            claim_store_path=_resolved(repository, args.claim_store),
            resolution_checkpoint_path=_resolved(
                repository,
                args.resolution_checkpoint,
            ),
            output_path=_resolved(repository, args.output),
            campaign=campaign,
        )
        result = run_round17_one_use_evaluation(config, progress=_progress)
    except (OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        print(f"round17-polymarket-one-use failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "round": 17,
        "status": result["status"],
        "claim_sha256": result["claim_sha256"],
        "test_access_sha256": result["test_access_sha256"],
        "result_sha256": result.get("result_sha256"),
        "heldout_accepted": result.get("heldout_accepted"),
        "profitability_claim": result["profitability_claim"],
        "live_trading_authority": result["live_trading_authority"],
        "output": str(config.output_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 3 if result["status"] == "resolution_pending" else 0


if __name__ == "__main__":
    raise SystemExit(main())

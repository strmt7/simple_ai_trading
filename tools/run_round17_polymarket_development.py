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
    validate_round17_development_result,
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even when a hash-verified terminal output already exists",
    )
    return parser


def _resolved(repository: Path, value: str) -> Path:
    selected = Path(value)
    return selected.resolve() if selected.is_absolute() else (repository / selected).resolve()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("existing Round 17 output contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"existing Round 17 output contains {value}")


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


def _load_existing_result(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("existing Round 17 output is not a regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("existing Round 17 output is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("existing Round 17 output is not an object")
    return validate_round17_development_result(payload)


def _summary(
    result: Mapping[str, object],
    *,
    output: Path,
    reused: bool,
) -> dict[str, object]:
    return {
        "round": 17,
        "status": result["status"],
        "development_accepted": result["development_accepted"],
        "profitability_claim": result["profitability_claim"],
        "live_trading_authority": result["live_trading_authority"],
        "result_sha256": result["result_sha256"],
        "output": str(output),
        "reused": reused,
    }


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
        if config.output_path.exists() and not args.force:
            existing = _load_existing_result(config.output_path)
            if existing["status"] != "resolution_incomplete":
                print(
                    json.dumps(
                        _summary(existing, output=config.output_path, reused=True),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
        result = run_round17_development(config, progress=_progress)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"round17-polymarket-development failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            _summary(result, output=config.output_path, reused=False),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] != "resolution_incomplete" else 3


if __name__ == "__main__":
    raise SystemExit(main())

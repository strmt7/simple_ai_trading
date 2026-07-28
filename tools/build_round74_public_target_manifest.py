"""Assemble one no-order Round 74 public-mainnet paper target manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round74_public_target_operator import (  # noqa: E402
    build_round74_public_target_manifest,
    write_round74_public_target_manifest,
)
from tools._round74_public_evidence_capture import (  # noqa: E402
    require_clean_tracked_worktree,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reopen six immutable source artifacts and derive one audited "
            "public-mainnet paper target without network, database, "
            "credential, or order access."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cohort-capture", required=True)
    parser.add_argument("--exchange-info", required=True)
    parser.add_argument("--commission", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--execution-calibration", required=True)
    parser.add_argument("--execution-scenario", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_clean_tracked_worktree()
        manifest = build_round74_public_target_manifest(
            source_artifact_root=args.source_root,
            source_relative_paths={
                "cohort_capture": args.cohort_capture,
                "exchange_info": args.exchange_info,
                "commission": args.commission,
                "funding": args.funding,
                "execution_calibration": args.execution_calibration,
                "execution_scenario": args.execution_scenario,
            },
        )
        target = write_round74_public_target_manifest(
            manifest=manifest,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        _canonical_json(
            {
                "manifest": str(target),
                "manifest_sha256": manifest.manifest_sha256,
                "run_id": manifest.run_id,
                "source_artifact_count": len(manifest.source_artifacts),
                "network_accessed": False,
                "database_opened": False,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

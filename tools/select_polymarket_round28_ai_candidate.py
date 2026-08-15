#!/usr/bin/env python3
"""Account for all Round 28 AI candidates and nominate at most one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from simple_ai_trading.polymarket_round27_operator import artifact_writer, load_mapping
from simple_ai_trading.polymarket_round28_ai_cases import (
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_selection import (
    round28_ai_candidate_selection_from_mapping,
    select_round28_ai_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--baseline-economic-report", type=Path, required=True)
    parser.add_argument("--host-report", type=Path, action="append", default=[])
    parser.add_argument("--host-failure", type=Path, action="append", default=[])
    parser.add_argument("--economic-report", type=Path, action="append", default=[])
    parser.add_argument("--selection-claim", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 AI selection path must not be a symlink")
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    panel_path = _resolve(repository, args.case_panel, strict=True)
    baseline_path = _resolve(
        repository,
        args.baseline_economic_report,
        strict=True,
    )
    host_paths = tuple(
        _resolve(repository, path, strict=True) for path in args.host_report
    )
    failure_paths = tuple(
        _resolve(repository, path, strict=True) for path in args.host_failure
    )
    economic_paths = tuple(
        _resolve(repository, path, strict=True) for path in args.economic_report
    )
    output = _resolve(repository, args.selection_claim, strict=False)
    inputs = (panel_path, baseline_path, *host_paths, *failure_paths, *economic_paths)
    if (
        any(not path.is_file() for path in inputs)
        or len(set(inputs)) != len(inputs)
        or output in inputs
        or (output.exists() and output.is_symlink())
    ):
        raise ValueError("Round 28 AI selection paths differ")

    contract = load_round28_ai_contract(repository)
    panel = round28_ai_case_panel_from_mapping(load_mapping(panel_path))
    if panel.partition_role != "selection":
        raise ValueError("Round 28 AI nomination requires selection evidence")
    baseline = load_mapping(baseline_path)
    selection = select_round28_ai_candidate(
        contract=contract,
        host_qualification_reports=tuple(load_mapping(path) for path in host_paths),
        host_failure_reports=tuple(load_mapping(path) for path in failure_paths),
        economic_reports=tuple(load_mapping(path) for path in economic_paths),
        case_panel_sha256=panel.panel_sha256,
        round28_economic_report_sha256=str(baseline.get("report_sha256", "")),
    )
    if output.exists():
        persisted = round28_ai_candidate_selection_from_mapping(
            load_mapping(output)
        )
        if persisted != selection:
            raise ValueError("Round 28 persisted AI selection differs")
    artifact_writer(output, "selection_sha256")(selection.asdict())
    print(
        json.dumps(
            {
                "candidate_coverage": [dict(item) for item in selection.candidate_coverage],
                "nominated_model_id": selection.nominated_model_id,
                "selection_sha256": selection.selection_sha256,
                "sealed_partition_accessed": False,
                "edge_claim": False,
                "profitability_claim": False,
                "credentials_used": False,
                "orders_submitted": False,
                "trading_authority": False,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

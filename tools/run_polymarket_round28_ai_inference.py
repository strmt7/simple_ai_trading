#!/usr/bin/env python3
"""Run one qualified Round 28 AI veto over an immutable case panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from simple_ai_trading.polymarket_round27_operator import artifact_writer, load_mapping
from simple_ai_trading.polymarket_round28_ai_cases import (
    round28_ai_case_panel_from_mapping,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    validate_round28_ai_host_report,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    round28_ai_inference_report_from_mapping,
    run_round28_ai_inference,
    validate_round28_ai_inference_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--case-panel", type=Path, required=True)
    parser.add_argument("--host-report", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink():
        raise ValueError("Round 28 AI inference path must not be a symlink")
    return selected.resolve(strict=strict)


def _progress(model_id: str):
    def report(completed: int, total: int) -> None:
        if completed == total or completed == 1 or completed % 5 == 0:
            print(
                json.dumps(
                    {
                        "observed_at_ms": time.time_ns() // 1_000_000,
                        "phase": "target_free_ai_inference",
                        "model_id": model_id,
                        "completed_cases": completed,
                        "total_cases": total,
                        "target_accessed": False,
                        "outcome_accessed": False,
                        "orders_submitted": False,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    panel_path = _resolve(repository, args.case_panel, strict=True)
    host_path = _resolve(repository, args.host_report, strict=True)
    output = _resolve(repository, args.inference_report, strict=False)
    if (
        not panel_path.is_file()
        or not host_path.is_file()
        or len({panel_path, host_path, output}) != 3
        or (output.exists() and output.is_symlink())
    ):
        raise ValueError("Round 28 AI inference paths differ")

    contract = load_round28_ai_contract(repository)
    panel = round28_ai_case_panel_from_mapping(load_mapping(panel_path))
    host_report = load_mapping(host_path)
    validated_host, candidate = validate_round28_ai_host_report(
        host_report,
        contract=contract,
    )
    if output.exists():
        report = validate_round28_ai_inference_report(
            load_mapping(output),
            contract=contract,
            host_qualification_report=validated_host,
            panel=panel,
        )
    else:
        report = run_round28_ai_inference(
            panel=panel,
            candidate=candidate,
            contract=contract,
            host_qualification_report=validated_host,
            progress=_progress(candidate.model_id),
        )
        validate_round28_ai_inference_report(
            report.asdict(),
            contract=contract,
            host_qualification_report=validated_host,
            panel=panel,
        )
    artifact_writer(output, "report_sha256")(report.asdict())
    restored = round28_ai_inference_report_from_mapping(load_mapping(output))
    if restored != report:
        raise ValueError("Round 28 AI inference persistence differs")
    print(
        json.dumps(
            {
                "model_id": candidate.model_id,
                "inference_report_sha256": report.report_sha256,
                "case_count": len(report.responses),
                "changed_action_count": report.changed_action_count,
                "rejected_fraction": report.rejected_fraction,
                "matched_evaluation_eligible": (
                    report.candidate_eligible_for_matched_evaluation
                ),
                "unload_observed": report.unload_observed,
                "target_accessed": False,
                "outcome_accessed": False,
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

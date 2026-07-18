"""Run and persist the frozen Round 10 transparent development baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_hurdle import (  # noqa: E402
    POLYMARKET_ROUND10_CONTRACT_SHA256,
    PolymarketHurdleDevelopmentReport,
    fit_round10_development_hurdle_baseline,
    load_round9_hurdle_development_dataset,
)
from simple_ai_trading.polymarket_recorder import (  # noqa: E402
    PolymarketEvidenceStore,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DEFAULT_DATABASE = (
    ROOT / "data" / "polymarket-round9-confirmation-v4-20260716-152838Z.duckdb"
)
DEFAULT_PIPELINE_REPORT_SHA256 = (
    "1d3b1e0df05dbb4a7f5b9be9fe7b40fd03ba8f6f06bf90115851adb10efb4d8b"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-010-development-hurdle-report.json"
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _model_artifact(report: PolymarketHurdleDevelopmentReport) -> dict[str, object]:
    model = report.model
    return {
        "schema_version": "polymarket-round10-transparent-hurdle-artifact-v1",
        "contract_sha256": POLYMARKET_ROUND10_CONTRACT_SHA256,
        "model_sha256": model.model_sha256,
        "scaler": {
            **model.scaler.identity_payload(),
            "scaler_sha256": model.scaler.scaler_sha256,
        },
        "heads": {
            head.name: {**head.identity_payload(), "model_sha256": head.model_sha256}
            for head in (
                model.observable,
                model.entry_fill,
                model.exit_fill,
                model.complete_utility_mean,
                model.complete_utility_q10,
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--pipeline-report-sha256", default=DEFAULT_PIPELINE_REPORT_SHA256
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--database-threads", type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()

    def progress(phase: str, payload: Mapping[str, object]) -> None:
        print(
            json.dumps(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "phase": phase,
                    **dict(payload),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    database = args.database.resolve()
    with PolymarketEvidenceStore(
        database,
        memory_limit=str(args.memory_limit),
        threads=int(args.database_threads),
    ) as store:
        progress("dataset_load_started", {"database": database.name})
        dataset = load_round9_hurdle_development_dataset(
            store,
            pipeline_report_sha256=str(args.pipeline_report_sha256),
        )
    terminal_counts = Counter(
        dataset.terminal_reasons[int(code)] for code in dataset.terminal_reason_code
    )
    progress(
        "dataset_load_complete",
        {
            "dataset_sha256": dataset.dataset_sha256,
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "group_count": len(dataset.groups),
            "condition_count": len(dataset.condition_ids),
            "unknown_entry_count": int(np.count_nonzero(dataset.unknown_entry)),
            "excluded_pre_submit_count": dataset.excluded_pre_submit_count,
        },
    )
    report = fit_round10_development_hurdle_baseline(dataset, progress=progress)
    payload: dict[str, object] = {
        **report.asdict(),
        "evidence_role": "round9_development_only_after_failed_round9_claim",
        "database_file": database.name,
        "dataset": {
            "schema_version": "polymarket-round10-development-hurdle-dataset-v1",
            "dataset_sha256": dataset.dataset_sha256,
            "rows": dataset.rows,
            "feature_names": list(dataset.feature_names),
            "group_count": len(dataset.groups),
            "condition_count": len(dataset.condition_ids),
            "decision_start_monotonic_ns": int(np.min(dataset.decision_monotonic_ns)),
            "decision_end_monotonic_ns": int(np.max(dataset.decision_monotonic_ns)),
            "excluded_pre_submit_count": dataset.excluded_pre_submit_count,
            "unknown_entry_count": int(np.count_nonzero(dataset.unknown_entry)),
            "entry_filled_count": int(np.count_nonzero(dataset.entry_filled)),
            "exit_filled_count": int(np.count_nonzero(dataset.exit_filled)),
            "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        },
        "split": report.split.identity_payload(),
        "model_artifact": _model_artifact(report),
        "truth": {
            "round10_confirmation_read": False,
            "nonlinear_model_fitted": False,
            "ai_model_invoked": False,
            "profitability_claim": False,
            "roi_claim": False,
            "drawdown_claim": False,
            "paper_authority": False,
            "trading_authority": False,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    write_json_atomic(output, payload, indent=2, sort_keys=True)
    progress(
        "report_complete",
        {
            "output": output.relative_to(ROOT).as_posix(),
            "report_sha256": report.report_sha256,
            "artifact_sha256": payload["artifact_sha256"],
            "development_passed": report.development_passed,
            "nonlinear_challenger_authorized": (report.nonlinear_challenger_authorized),
        },
    )
    print(
        json.dumps(
            {
                "report_sha256": report.report_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "development_passed": report.development_passed,
                "selected_policy": report.selected_policy.asdict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.development_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

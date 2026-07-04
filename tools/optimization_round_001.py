"""Run Round 001 optimization evidence against exchange-sourced candles only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("data/optimization/round-001")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=1000.0)
    parser.add_argument("--objective", action="append", default=[])
    parser.add_argument("--max-symbols", type=int, default=6)
    parser.add_argument("--max-scan", type=int, default=250)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--market", choices=("spot", "futures"), default=None)
    parser.add_argument("--compute-backend", choices=("cpu", "cuda", "rocm", "directml", "mps", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--recent-window",
        action="store_true",
        help="use the CLI candle limit instead of full-history pagination; this run is not performance evidence",
    )
    return parser


def _command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "simple_ai_trading.cli",
        "model-lab",
        "--output-dir",
        str(args.output_dir),
        "--starting-cash",
        str(args.starting_cash),
        "--max-symbols",
        str(args.max_symbols),
        "--max-scan",
        str(args.max_scan),
        "--limit",
        str(args.limit),
        "--compute-backend",
        str(args.compute_backend),
        "--batch-size",
        str(args.batch_size),
    ]
    if not args.recent_window:
        command.append("--full-history")
    if args.market:
        command.extend(["--market", str(args.market)])
    if args.score_batch_size is not None:
        command.extend(["--score-batch-size", str(args.score_batch_size)])
    if args.max_candidates is not None:
        command.extend(["--max-candidates", str(args.max_candidates)])
    for objective in args.objective:
        command.extend(["--objective", str(objective)])
    return command


def _coverage_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        return []
    items: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        coverage = outcome.get("data_coverage")
        if isinstance(coverage, dict):
            items.append(coverage)
    return items


def _claim_eligible(report: dict[str, Any], coverage: list[dict[str, Any]], *, full_history_requested: bool) -> bool:
    sanity = report.get("financial_sanity")
    sanity_allowed = isinstance(sanity, dict) and sanity.get("allowed") is True
    portfolio = report.get("portfolio_risk")
    portfolio_allowed = isinstance(portfolio, dict) and portfolio.get("accepted") is True
    accepted = [
        item
        for item in report.get("outcomes", [])
        if isinstance(item, dict) and item.get("accepted") is True
    ]
    coverage_allowed = bool(coverage) and all(
        item.get("source_scope") == "binance_full_history"
        and item.get("is_full_history") is True
        and item.get("integrity_status") != "fail"
        and int(item.get("gap_count") or 0) == 0
        for item in coverage
    )
    return bool(full_history_requested and accepted and sanity_allowed and portfolio_allowed and coverage_allowed)


def _write_provenance(args: argparse.Namespace, command: list[str]) -> Path:
    report_path = args.output_dir / "model_lab_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"model-lab report was not created: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"model-lab report is not an object: {report_path}")
    coverage = _coverage_items(report)
    accepted_symbols = [
        str(item.get("symbol"))
        for item in report.get("outcomes", [])
        if isinstance(item, dict) and item.get("accepted") is True and item.get("symbol")
    ]
    full_history_requested = not bool(args.recent_window)
    provenance = {
        "round": "001",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "artifact_class": "local_exchange_sourced_optimization_evidence",
        "tracked_repo_artifact": False,
        "data_source": "Binance kline API via configured runtime",
        "full_history_requested": full_history_requested,
        "recent_window_smoke_run": bool(args.recent_window),
        "performance_claim_eligible_for_review": _claim_eligible(
            report,
            coverage,
            full_history_requested=full_history_requested,
        ),
        "performance_claim_rule": (
            "Only full-history exchange-sourced runs with accepted portfolio risk, "
            "passing financial sanity, complete coverage, and no gaps can be reviewed "
            "as performance evidence."
        ),
        "command": command,
        "model_lab_report": str(report_path),
        "symbols": [str(item.get("symbol")) for item in report.get("outcomes", []) if isinstance(item, dict) and item.get("symbol")],
        "accepted_symbols": accepted_symbols,
        "market_type": report.get("market_type"),
        "interval": report.get("interval"),
        "coverage": coverage,
        "financial_sanity": report.get("financial_sanity"),
        "portfolio_risk": report.get("portfolio_risk"),
    }
    provenance_path = args.output_dir / "round-001-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return provenance_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = _command(args)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        return completed.returncode
    try:
        provenance_path = _write_provenance(args, command)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to write Round 001 provenance: {exc}", file=sys.stderr)
        return 2
    print(f"Round 001 local evidence: {args.output_dir}")
    print(f"Round 001 provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

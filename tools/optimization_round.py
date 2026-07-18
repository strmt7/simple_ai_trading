"""Generate numbered optimization-round graph data from exchange-sourced backtests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


DEFAULT_MODEL_CANDIDATES = 36
DEFAULT_RULE_ALPHA_CANDIDATES = 225
DEFAULT_RULE_ALPHA_EMPIRICAL_FEATURES = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", default="round-001")
    parser.add_argument("--symbol-count", type=int, default=50)
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated explicit symbols. When set, this overrides --symbol-count liquidity selection.",
    )
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--interval", default="1s")
    parser.add_argument("--market", choices=("spot", "futures"), default="spot")
    parser.add_argument("--objective", default="conservative")
    parser.add_argument(
        "--no-objective-strategy-defaults",
        action="store_true",
        help="Use the saved strategy as-is instead of applying the selected objective's profile defaults.",
    )
    parser.add_argument("--starting-cash", type=float, default=1000.0)
    parser.add_argument(
        "--taker-fee-bps",
        type=float,
        default=None,
        help=(
            "offline taker-fee assumption in basis points; values below the documented market floor "
            "are raised to that floor"
        ),
    )
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "cpu", "cuda", "rocm", "xpu", "mps", "directml"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--model-candidates",
        type=int,
        default=DEFAULT_MODEL_CANDIDATES,
        help=(
            "number of bounded model/label candidates evaluated per symbol before the final holdout; "
            f"default {DEFAULT_MODEL_CANDIDATES} runs the broad profitability-search prefix"
        ),
    )
    parser.add_argument(
        "--model-candidate-names",
        default="",
        help=(
            "comma-separated exact candidate names to evaluate in the supplied order; "
            "when set, this overrides the prefix implied by --model-candidates"
        ),
    )
    parser.add_argument(
        "--rule-alpha-candidates",
        type=int,
        default=DEFAULT_RULE_ALPHA_CANDIDATES,
        help=(
            "maximum interpretable rule-alpha templates replayed for the selected model; "
            f"default {DEFAULT_RULE_ALPHA_CANDIDATES} is exhaustive, lower values are for fast diagnostics"
        ),
    )
    parser.add_argument(
        "--rule-alpha-empirical-features",
        type=int,
        default=DEFAULT_RULE_ALPHA_EMPIRICAL_FEATURES,
        help=(
            "maximum engineered feature columns scanned by empirical rule-alpha mining; "
            "0 keeps the default full bounded scan, lower positive values are fast diagnostics"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/optimization"))
    parser.add_argument("--docs-root", type=Path, default=Path("docs/optimization"))
    parser.add_argument("--db", type=Path, default=Path("data/market_data.sqlite"))
    parser.add_argument(
        "--data-start",
        default=None,
        help="inclusive UTC evidence window start as epoch ms, YYYY-MM-DD, or ISO timestamp",
    )
    parser.add_argument(
        "--data-end",
        default=None,
        help="inclusive UTC evidence window end as epoch ms, YYYY-MM-DD, or ISO timestamp",
    )
    parser.add_argument("--max-calls-per-minute", type=int, default=1800)
    parser.add_argument(
        "--require-prefilled-data",
        action="store_true",
        help="Refuse network backfill during optimization; data must already be present in SQLite.",
    )
    parser.add_argument(
        "--min-data-rows",
        type=int,
        default=0,
        help="Minimum rows required per selected symbol before training/backtesting.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.995,
        help="Minimum contiguous coverage ratio required per selected symbol.",
    )
    parser.add_argument(
        "--max-gap-count",
        type=int,
        default=0,
        help="Maximum allowed missing-interval gaps per selected symbol.",
    )
    parser.add_argument(
        "--require-verified-checksum",
        action="store_true",
        help="Require at least one verified Binance archive checksum for each selected symbol.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before optimization if the requested compute backend resolves to CPU.",
    )
    parser.add_argument(
        "--promotion-grade",
        action="store_true",
        help=(
            "Run the fail-closed BTC/ETH/SOL day-trading evidence contract: exact major trio, "
            "1s interval, prefilled SQLite data, verified archive checksums, zero gaps, and critical-analysis pass."
        ),
    )
    parser.add_argument(
        "--min-promotion-data-years",
        type=float,
        default=2.0,
        help="Minimum stored 1s history span per BTC/ETH/SOL symbol required by --promotion-grade.",
    )
    return parser


def _safe_round_id(round_id: str) -> str:
    return str(round_id).strip().lower().replace(" ", "-")


def _write_startup_status(args: argparse.Namespace) -> Path:
    data_dir = Path(args.docs_root) / _safe_round_id(args.round_id) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    status_path = data_dir / "round-status.json"
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "round_id": str(args.round_id),
        "phase": "process_startup",
        "status": "running",
        "message": "optimization wrapper parsed arguments; importing evidence engine",
        "market": str(args.market),
        "interval": str(args.interval),
        "objective": str(args.objective),
        "compute_backend": str(args.compute_backend),
        "require_gpu": bool(args.require_gpu),
        "model_candidates": int(
            getattr(args, "model_candidates", DEFAULT_MODEL_CANDIDATES)
        ),
        "model_candidate_names": [
            value.strip()
            for value in str(getattr(args, "model_candidate_names", "") or "").split(",")
            if value.strip()
        ],
        "rule_alpha_candidates": int(
            getattr(args, "rule_alpha_candidates", DEFAULT_RULE_ALPHA_CANDIDATES)
        ),
        "rule_alpha_empirical_features": int(
            getattr(
                args,
                "rule_alpha_empirical_features",
                DEFAULT_RULE_ALPHA_EMPIRICAL_FEATURES,
            )
        ),
    }
    tmp_path = status_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tmp_path.replace(status_path)
    return status_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status_path = _write_startup_status(args)
    print(f"startup status: {status_path}", flush=True)

    from simple_ai_trading.api import BinanceClient
    from simple_ai_trading.commission import apply_offline_commission_floor
    from simple_ai_trading.config import load_strategy
    from simple_ai_trading.optimization_evidence import build_round_evidence, parse_evidence_timestamp_ms
    from simple_ai_trading.optimization_progress import build_optimization_progress_artifacts
    from simple_ai_trading.types import StrategyConfig

    strategy = load_strategy()
    if args.taker_fee_bps is not None:
        strategy = StrategyConfig(
            **{
                **strategy.asdict(),
                "taker_fee_bps": max(0.0, float(args.taker_fee_bps)),
            }
        )
    strategy, commission_assumption = apply_offline_commission_floor(
        strategy,
        market_type=args.market,
        symbol=(str(args.symbols).split(",", 1)[0] if str(args.symbols).strip() else ""),
    )
    client = BinanceClient(
        "",
        "",
        testnet=False,
        market_type=args.market,
        max_calls_per_minute=max(1, int(args.max_calls_per_minute)),
    )
    explicit_symbols = [symbol.strip().upper() for symbol in str(args.symbols or "").split(",") if symbol.strip()]
    model_candidate_names = [
        value.strip()
        for value in str(args.model_candidate_names or "").split(",")
        if value.strip()
    ]
    try:
        data_start_ms = parse_evidence_timestamp_ms(args.data_start, end_of_day=False)
        data_end_ms = parse_evidence_timestamp_ms(args.data_end, end_of_day=True)
        report = build_round_evidence(
            round_id=args.round_id,
            client=client,
            strategy=strategy,
            quote_asset=args.quote_asset,
            symbol_count=args.symbol_count,
            symbols=explicit_symbols or None,
            interval=args.interval,
            market_type=args.market,
            objective_name=args.objective,
            starting_cash=args.starting_cash,
            compute_backend=args.compute_backend,
            batch_size=args.batch_size,
            model_candidate_count=max(1, int(args.model_candidates)),
            model_candidate_names=model_candidate_names or None,
            rule_alpha_max_candidates=max(1, int(args.rule_alpha_candidates)),
            rule_alpha_empirical_max_features=(
                None
                if int(args.rule_alpha_empirical_features) <= 0
                else int(args.rule_alpha_empirical_features)
            ),
            data_root=args.data_root,
            docs_root=args.docs_root,
            db_path=args.db,
            require_prefilled_data=args.require_prefilled_data,
            min_data_rows=args.min_data_rows,
            min_coverage_ratio=args.min_coverage_ratio,
            max_gap_count=args.max_gap_count,
            require_verified_checksum=args.require_verified_checksum,
            require_gpu=args.require_gpu,
            promotion_grade=args.promotion_grade,
            min_promotion_data_years=args.min_promotion_data_years,
            use_objective_strategy_defaults=not args.no_objective_strategy_defaults,
            data_start_ms=data_start_ms,
            data_end_ms=data_end_ms,
            commission_assumption=commission_assumption.asdict(),
        )
    except ValueError as exc:
        print(f"optimization round failed: {exc}", file=sys.stderr)
        return 2
    print(f"round: {report['round_id']}")
    print(f"symbols completed: {report['symbol_count_completed']}/{report['symbol_count_requested']}")
    print(f"effective leverage: {report['effective_leverage']}x applies={report['leverage_applies']}")
    print(f"metrics: {report['metrics_csv_path']}")
    print(f"portfolio timeline: {report['portfolio_timeline_csv_path']}")
    print(f"progress: {report['progress_csv_path']}")
    progress_report = build_optimization_progress_artifacts(args.docs_root)
    print(f"iteration progress: {progress_report['tracked_artifacts'][1]}")
    critical = report.get("critical_analysis")
    if isinstance(critical, dict):
        print(f"critical verdict: {critical.get('verdict')}")
        failures = critical.get("failures")
        if failures:
            print(f"critical failures: {', '.join(str(item) for item in failures)}", file=sys.stderr)
        if critical.get("verdict") != "pass":
            return 2
    promotion_contract = report.get("promotion_grade_contract")
    if args.promotion_grade and isinstance(promotion_contract, dict):
        print(f"promotion-grade contract: {promotion_contract.get('status')}")
        reasons = promotion_contract.get("reasons")
        if reasons:
            print(f"promotion-grade failures: {', '.join(str(item) for item in reasons)}", file=sys.stderr)
        if promotion_contract.get("status") != "pass":
            return 2
    accepted_count = int(report.get("progress", {}).get("accepted_symbol_count", 0)) if isinstance(report.get("progress"), dict) else 0
    return 0 if accepted_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

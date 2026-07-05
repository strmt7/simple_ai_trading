"""Generate numbered optimization-round graph data from exchange-sourced backtests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from simple_ai_trading.api import BinanceClient
from simple_ai_trading.config import load_strategy
from simple_ai_trading.optimization_evidence import build_round_evidence


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
    parser.add_argument("--compute-backend", choices=("cpu", "cuda", "rocm", "directml", "mps", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--data-root", type=Path, default=Path("data/optimization"))
    parser.add_argument("--docs-root", type=Path, default=Path("docs/optimization"))
    parser.add_argument("--db", type=Path, default=Path("data/market_data.sqlite"))
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    strategy = load_strategy()
    client = BinanceClient(
        "",
        "",
        testnet=False,
        market_type=args.market,
        max_calls_per_minute=max(1, int(args.max_calls_per_minute)),
    )
    explicit_symbols = [symbol.strip().upper() for symbol in str(args.symbols or "").split(",") if symbol.strip()]
    try:
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
            data_root=args.data_root,
            docs_root=args.docs_root,
            db_path=args.db,
            require_prefilled_data=args.require_prefilled_data,
            min_data_rows=args.min_data_rows,
            min_coverage_ratio=args.min_coverage_ratio,
            max_gap_count=args.max_gap_count,
            require_verified_checksum=args.require_verified_checksum,
            require_gpu=args.require_gpu,
            use_objective_strategy_defaults=not args.no_objective_strategy_defaults,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

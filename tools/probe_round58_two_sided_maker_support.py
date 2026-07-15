"""Measure joint two-sided maker-fill support without reading market outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.queue_censored_actions import (  # noqa: E402
    PASSIVE_FILL_BUCKETS_MS,
    build_passive_fill_result,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.probe_round57_passive_fill_support import (  # noqa: E402
    SYMBOLS,
    _day_bounds_ms,
    _manifest_rows,
)


SCHEMA_VERSION = "round-058-two-sided-maker-support-v1"
STATE_LABELS = ("none", "bid_only", "ask_only", "both")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        label: float(np.quantile(values, quantile))
        for label, quantile in (
            ("p50", 0.50),
            ("p90", 0.90),
            ("p99", 0.99),
            ("max", 1.00),
        )
    }


def _cross_table(
    bid_bucket: np.ndarray, ask_bucket: np.ndarray
) -> list[dict[str, int]]:
    return [
        {
            "bid_bucket": bid,
            "ask_bucket": ask,
            "rows": int(np.count_nonzero((bid_bucket == bid) & (ask_bucket == ask))),
        }
        for bid in range(len(PASSIVE_FILL_BUCKETS_MS) + 1)
        for ask in range(len(PASSIVE_FILL_BUCKETS_MS) + 1)
    ]


def _state_summary(bid_filled: np.ndarray, ask_filled: np.ndarray) -> dict[str, object]:
    rows = int(bid_filled.size)
    state = bid_filled.astype(np.uint8) + 2 * ask_filled.astype(np.uint8)
    code_by_label = {"none": 0, "bid_only": 1, "ask_only": 2, "both": 3}
    return {
        label: {
            "rows": int(np.count_nonzero(state == code)),
            "ratio": float(np.mean(state == code)),
        }
        for label, code in code_by_label.items()
    } | {"total_rows": rows}


def _probe_symbol(
    connection: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    decision_cadence_ms: int,
    placement_latency_ms: int,
    maximum_quote_age_ms: int,
    order_notional_quote_per_side: float,
) -> dict[str, object]:
    manifests, archive_ids = _manifest_rows(
        connection,
        symbol=symbol,
        period=period,
    )
    bbo = connection.execute(
        """
        SELECT bucket_ms + 100 AS available_ms, close_bid, close_ask,
               close_bid_qty, close_ask_qty
        FROM book_ticker_100ms
        WHERE archive_id = ? AND bucket_ms >= ? AND bucket_ms < ?
        ORDER BY bucket_ms
        """,
        [archive_ids["bookTicker"], start_ms, end_ms],
    ).fetchnumpy()
    trades = connection.execute(
        """
        SELECT trade_id, trade_time_ms, price, qty, buyer_is_maker
        FROM trade_raw
        WHERE archive_id = ? AND trade_time_ms >= ? AND trade_time_ms < ?
        ORDER BY trade_time_ms, trade_id
        """,
        [archive_ids["trades"], start_ms, end_ms],
    ).fetchnumpy()
    available_ms = np.asarray(bbo["available_ms"], dtype=np.int64)
    if available_ms.size == 0 or np.any(np.diff(available_ms) < 0):
        raise ValueError(f"{symbol} BBO availability rows are empty or unordered")
    expiry_ms = PASSIVE_FILL_BUCKETS_MS[-1]
    decisions = np.arange(
        start_ms,
        end_ms - placement_latency_ms - expiry_ms,
        decision_cadence_ms,
        dtype=np.int64,
    )
    arrivals = decisions + placement_latency_ms
    quote_indexes = np.searchsorted(available_ms, arrivals, side="right") - 1
    safe_indexes = np.maximum(quote_indexes, 0)
    quote_age = arrivals - available_ms[safe_indexes]
    eligible = (
        (quote_indexes >= 0) & (quote_age >= 0) & (quote_age <= maximum_quote_age_ms)
    )
    planned_rows = int(decisions.size)
    decisions = decisions[eligible]
    arrivals = arrivals[eligible]
    quote_indexes = quote_indexes[eligible]
    quote_age = quote_age[eligible]
    if arrivals.size == 0:
        raise ValueError(f"{symbol} has no quote-age-eligible decisions")

    common = {
        "arrival_time_ms": arrivals,
        "order_notional_quote": order_notional_quote_per_side,
        "trade_id": np.asarray(trades["trade_id"], dtype=np.int64),
        "trade_time_ms": np.asarray(trades["trade_time_ms"], dtype=np.int64),
        "trade_price": np.asarray(trades["price"], dtype=np.float64),
        "trade_quantity": np.asarray(trades["qty"], dtype=np.float64),
        "trade_buyer_is_maker": np.asarray(trades["buyer_is_maker"], dtype=np.bool_),
    }
    bid_prices = np.asarray(bbo["close_bid"], dtype=np.float64)[quote_indexes]
    ask_prices = np.asarray(bbo["close_ask"], dtype=np.float64)[quote_indexes]
    spread_bps = (ask_prices / bid_prices - 1.0) * 10_000.0
    if not np.isfinite(spread_bps).all() or np.any(spread_bps < 0.0):
        raise ValueError(f"{symbol} placement spread is invalid")
    bid = build_passive_fill_result(
        placement_price=bid_prices,
        queue_ahead_quantity=np.asarray(bbo["close_bid_qty"], dtype=np.float64)[
            quote_indexes
        ],
        buyer_is_maker=True,
        **common,
    )
    ask = build_passive_fill_result(
        placement_price=ask_prices,
        queue_ahead_quantity=np.asarray(bbo["close_ask_qty"], dtype=np.float64)[
            quote_indexes
        ],
        buyer_is_maker=False,
        **common,
    )
    both = bid.filled & ask.filled
    singleton = bid.filled ^ ask.filled
    exposure_both = np.abs(bid.fill_time_ms[both] - ask.fill_time_ms[both])
    singleton_fill_time = np.where(
        bid.filled[singleton],
        bid.fill_time_ms[singleton],
        ask.fill_time_ms[singleton],
    )
    singleton_remaining = arrivals[singleton] + expiry_ms - singleton_fill_time
    if np.any(singleton_remaining < 0):
        raise ValueError(f"{symbol} singleton inventory duration is invalid")
    bid_first = both & (bid.fill_time_ms < ask.fill_time_ms)
    ask_first = both & (ask.fill_time_ms < bid.fill_time_ms)
    same_time = both & (bid.fill_time_ms == ask.fill_time_ms)
    sequencing = {
        "bid_first": int(np.count_nonzero(bid_first)),
        "ask_first": int(np.count_nonzero(ask_first)),
        "same_exchange_timestamp": int(np.count_nonzero(same_time)),
    }
    if sum(sequencing.values()) != int(np.count_nonzero(both)):
        raise ValueError(f"{symbol} joint-fill sequencing does not reconcile")

    return {
        "symbol": symbol,
        "source": {
            "manifests": manifests,
            "manifest_sha256": _canonical_sha256(manifests),
            "book_ticker_100ms_rows_read": int(available_ms.size),
            "trade_rows_read": int(len(trades["trade_id"])),
        },
        "decision_rows_planned": planned_rows,
        "decision_rows_quote_age_eligible": int(decisions.size),
        "decision_rows_quote_age_rejected": int(np.count_nonzero(~eligible)),
        "quote_age_ms": _quantiles(quote_age),
        "placement_spread_bps": _quantiles(spread_bps),
        "both_fill_placement_spread_bps": _quantiles(spread_bps[both]),
        "singleton_placement_spread_bps": _quantiles(spread_bps[singleton]),
        "joint_fill_state": _state_summary(bid.filled, ask.filled),
        "joint_fill_sequencing": sequencing,
        "both_fill_inventory_exposure_ms": _quantiles(exposure_both),
        "singleton_maximum_inventory_exposure_before_expiry_ms": _quantiles(
            singleton_remaining
        ),
        "fill_bucket_cross_table": _cross_table(bid.fill_bucket, ask.fill_bucket),
        "bid_fill": bid.summary(),
        "ask_fill": ask.summary(),
        "result_sha256": _canonical_sha256(
            {
                "bid_fill_result_sha256": bid.result_sha256,
                "ask_fill_result_sha256": ask.result_sha256,
                "joint_state": _state_summary(bid.filled, ask.filled),
                "sequencing": sequencing,
                "placement_spread_bps": _quantiles(spread_bps),
                "both_fill_placement_spread_bps": _quantiles(spread_bps[both]),
                "singleton_placement_spread_bps": _quantiles(spread_bps[singleton]),
                "exposure_both_ms": _quantiles(exposure_both),
                "singleton_remaining_ms": _quantiles(singleton_remaining),
                "cross_table": _cross_table(bid.fill_bucket, ask.fill_bucket),
            }
        ),
    }


def run(arguments: argparse.Namespace) -> int:
    database = arguments.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    start_ms, end_ms = _day_bounds_ms(arguments.day)
    cadence_ms = int(arguments.decision_cadence_seconds * 1_000)
    if not 1_000 <= cadence_ms <= 60_000:
        raise ValueError("decision cadence must lie in [1, 60] seconds")
    if not 0 <= arguments.placement_latency_ms <= 10_000:
        raise ValueError("placement latency must lie in [0, 10000] milliseconds")
    if not 1 <= arguments.maximum_quote_age_ms <= 10_000:
        raise ValueError("maximum quote age must lie in [1, 10000] milliseconds")
    notional = float(arguments.order_notional_quote_per_side)
    if not np.isfinite(notional) or not 0.0 < notional <= 1_000_000.0:
        raise ValueError("per-side order notional is outside the supported range")

    with duckdb.connect(str(database), read_only=True) as connection:
        connection.execute("PRAGMA disable_progress_bar")
        symbols = []
        for symbol in SYMBOLS:
            print(
                _canonical_json({"phase": "joint-fill-probe-start", "symbol": symbol}),
                flush=True,
            )
            result = _probe_symbol(
                connection,
                symbol=symbol,
                period=arguments.day,
                start_ms=start_ms,
                end_ms=end_ms,
                decision_cadence_ms=cadence_ms,
                placement_latency_ms=arguments.placement_latency_ms,
                maximum_quote_age_ms=arguments.maximum_quote_age_ms,
                order_notional_quote_per_side=notional,
            )
            symbols.append(result)
            print(
                _canonical_json(
                    {
                        "phase": "joint-fill-probe-complete",
                        "symbol": symbol,
                        "joint_fill_state": result["joint_fill_state"],
                    }
                ),
                flush=True,
            )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "purpose": "two_sided_structural_fill_support_only",
        "strategy_outcomes_read": False,
        "price_returns_read": False,
        "costs_read": False,
        "profit_and_loss_read": False,
        "policy_thresholds_selected": False,
        "market_source": "official Binance Data Vision USD-M daily archives",
        "day_utc": arguments.day,
        "symbols": list(SYMBOLS),
        "contract": {
            "decision_cadence_seconds": arguments.decision_cadence_seconds,
            "placement_latency_ms": arguments.placement_latency_ms,
            "maximum_quote_age_ms": arguments.maximum_quote_age_ms,
            "order_notional_quote_per_side": notional,
            "maker_order_expiry_ms": PASSIVE_FILL_BUCKETS_MS[-1],
            "full_displayed_l1_queue_ahead": True,
            "own_order_quantity_included": True,
            "cancellation_fill_credit": False,
            "matching_exact_price_prints_only": True,
            "full_fill_required": True,
        },
        "symbol_results": symbols,
        "trading_authority": False,
        "profitability_claim": False,
        "leverage_applied": False,
    }
    payload["report_sha256"] = _canonical_sha256(payload)
    write_json_atomic(arguments.output.resolve(), payload, indent=2)
    print(_canonical_json(payload))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--day", default="2023-06-01")
    parser.add_argument("--decision-cadence-seconds", type=int, default=10)
    parser.add_argument("--placement-latency-ms", type=int, default=750)
    parser.add_argument("--maximum-quote-age-ms", type=int, default=1_000)
    parser.add_argument("--order-notional-quote-per-side", type=float, default=500.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))

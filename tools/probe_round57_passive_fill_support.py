"""Measure conservative passive-fill support without reading strategy outcomes."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA_VERSION = "round-057-passive-fill-support-v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WINDOWS_MS = (5_000, 15_000, 30_000)


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


def _day_bounds_ms(value: str) -> tuple[int, int]:
    selected = date.fromisoformat(value)
    start = datetime.combine(selected, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1_000), int(end.timestamp() * 1_000)


def _manifest_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    period: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    columns = (
        "archive_id",
        "schema_version",
        "provider",
        "market_type",
        "symbol",
        "data_type",
        "period",
        "url",
        "status",
        "is_current",
        "compressed_bytes",
        "uncompressed_bytes",
        "source_sha256",
        "expected_sha256",
        "checksum_status",
        "rows_read",
        "derived_rows",
        "first_exchange_time_ms",
        "last_exchange_time_ms",
        "invalid_rows",
        "duplicate_ids",
        "update_id_regressions",
        "event_time_regressions",
        "out_of_order_rows",
        "crossed_books",
    )
    rows = connection.execute(
        f"""
        SELECT {", ".join(columns)}
        FROM archive_manifest
        WHERE symbol = ? AND period = ?
          AND data_type IN ('bookTicker', 'trades')
          AND status = 'complete' AND is_current
        ORDER BY data_type
        """,
        [symbol, period],
    ).fetchall()
    manifests = [dict(zip(columns, row, strict=True)) for row in rows]
    by_type = {str(row["data_type"]): str(row["archive_id"]) for row in manifests}
    if set(by_type) != {"bookTicker", "trades"} or len(manifests) != 2:
        raise ValueError(f"{symbol} does not have one current BBO and trade archive")
    for row in manifests:
        if (
            row["provider"] != "binance"
            or row["market_type"] != "futures"
            or row["checksum_status"] != "verified"
            or row["source_sha256"] != row["expected_sha256"]
            or int(row["rows_read"]) <= 0
            or int(row["invalid_rows"]) != 0
            or int(row["duplicate_ids"]) != 0
            or int(row["update_id_regressions"]) != 0
            or int(row["event_time_regressions"]) != 0
            or int(row["out_of_order_rows"]) != 0
            or int(row["crossed_books"]) != 0
        ):
            raise ValueError(f"{symbol} source manifest failed integrity gates")
    return manifests, by_type


def _grouped_trade_flow(
    *,
    trade_time_ms: np.ndarray,
    trade_price: np.ndarray,
    trade_quantity: np.ndarray,
    buyer_is_maker: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], tuple[int, int]],
]:
    times = np.asarray(trade_time_ms, dtype=np.int64)
    prices = np.asarray(trade_price, dtype=np.float64)
    quantities = np.asarray(trade_quantity, dtype=np.float64)
    sides = np.asarray(buyer_is_maker, dtype=np.uint8)
    if not (
        times.ndim == prices.ndim == quantities.ndim == sides.ndim == 1
        and len(times) == len(prices) == len(quantities) == len(sides)
        and len(times) > 0
        and np.all(np.isfinite(prices))
        and np.all(np.isfinite(quantities))
        and np.all(prices > 0.0)
        and np.all(quantities > 0.0)
    ):
        raise ValueError("trade source arrays are invalid")
    price_bits = prices.view(np.uint64)
    order = np.lexsort((times, price_bits, sides))
    times = times[order]
    price_bits = price_bits[order]
    quantities = quantities[order]
    sides = sides[order]
    starts = np.r_[
        0,
        1
        + np.flatnonzero(
            (sides[1:] != sides[:-1]) | (price_bits[1:] != price_bits[:-1])
        ),
    ]
    ends = np.r_[starts[1:], len(times)]
    groups = {
        (int(sides[start]), int(price_bits[start])): (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    }
    return times, price_bits, quantities, groups


def _fill_counts(
    *,
    arrivals_ms: np.ndarray,
    prices: np.ndarray,
    queue_ahead: np.ndarray,
    buyer_is_maker: bool,
    order_notional_quote: float,
    trade_times: np.ndarray,
    trade_quantities: np.ndarray,
    groups: dict[tuple[int, int], tuple[int, int]],
) -> dict[int, int]:
    candidate_prices = np.asarray(prices, dtype=np.float64)
    queues = np.asarray(queue_ahead, dtype=np.float64)
    if (
        len(candidate_prices) != len(arrivals_ms)
        or len(queues) != len(arrivals_ms)
        or not np.all(np.isfinite(candidate_prices))
        or not np.all(np.isfinite(queues))
        or np.any(candidate_prices <= 0.0)
        or np.any(queues < 0.0)
    ):
        raise ValueError("candidate passive-order arrays are invalid")
    price_bits = candidate_prices.view(np.uint64)
    fills = {window: np.zeros(len(arrivals_ms), dtype=bool) for window in WINDOWS_MS}
    side_key = int(bool(buyer_is_maker))
    for key_bits in np.unique(price_bits):
        decision_indexes = np.flatnonzero(price_bits == key_bits)
        bounds = groups.get((side_key, int(key_bits)))
        if bounds is None:
            continue
        left, right = bounds
        times = trade_times[left:right]
        quantities = trade_quantities[left:right]
        cumulative = np.r_[0.0, np.cumsum(quantities, dtype=np.float64)]
        begin = np.searchsorted(times, arrivals_ms[decision_indexes], side="right")
        full_fill_quantity = (
            queues[decision_indexes]
            + float(order_notional_quote) / candidate_prices[decision_indexes]
        )
        for window in WINDOWS_MS:
            finish = np.searchsorted(
                times,
                arrivals_ms[decision_indexes] + window,
                side="right",
            )
            printed_quantity = cumulative[finish] - cumulative[begin]
            fills[window][decision_indexes] = printed_quantity >= full_fill_quantity
    return {window: int(np.sum(values)) for window, values in fills.items()}


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
    order_notional_quote: float,
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
        SELECT trade_time_ms, price, qty, buyer_is_maker
        FROM trade_raw
        WHERE archive_id = ? AND trade_time_ms >= ? AND trade_time_ms < ?
        ORDER BY trade_time_ms
        """,
        [archive_ids["trades"], start_ms, end_ms],
    ).fetchnumpy()
    available_ms = np.asarray(bbo["available_ms"], dtype=np.int64)
    if len(available_ms) == 0 or np.any(np.diff(available_ms) < 0):
        raise ValueError(f"{symbol} BBO availability rows are empty or unordered")
    final_decision_exclusive = end_ms - max(WINDOWS_MS) - placement_latency_ms
    decisions_ms = np.arange(
        start_ms,
        final_decision_exclusive,
        decision_cadence_ms,
        dtype=np.int64,
    )
    arrivals_ms = decisions_ms + placement_latency_ms
    quote_indexes = np.searchsorted(available_ms, arrivals_ms, side="right") - 1
    safe_indexes = np.maximum(quote_indexes, 0)
    quote_age_ms = arrivals_ms - available_ms[safe_indexes]
    eligible = (
        (quote_indexes >= 0)
        & (quote_age_ms >= 0)
        & (quote_age_ms <= maximum_quote_age_ms)
    )
    decisions_ms = decisions_ms[eligible]
    arrivals_ms = arrivals_ms[eligible]
    quote_indexes = quote_indexes[eligible]
    quote_age_ms = quote_age_ms[eligible]
    if len(arrivals_ms) == 0:
        raise ValueError(f"{symbol} has no quote-age-eligible decisions")
    trade_times, _trade_price_bits, trade_quantities, groups = _grouped_trade_flow(
        trade_time_ms=trades["trade_time_ms"],
        trade_price=trades["price"],
        trade_quantity=trades["qty"],
        buyer_is_maker=trades["buyer_is_maker"],
    )
    long_counts = _fill_counts(
        arrivals_ms=arrivals_ms,
        prices=np.asarray(bbo["close_bid"])[quote_indexes],
        queue_ahead=np.asarray(bbo["close_bid_qty"])[quote_indexes],
        buyer_is_maker=True,
        order_notional_quote=order_notional_quote,
        trade_times=trade_times,
        trade_quantities=trade_quantities,
        groups=groups,
    )
    short_counts = _fill_counts(
        arrivals_ms=arrivals_ms,
        prices=np.asarray(bbo["close_ask"])[quote_indexes],
        queue_ahead=np.asarray(bbo["close_ask_qty"])[quote_indexes],
        buyer_is_maker=False,
        order_notional_quote=order_notional_quote,
        trade_times=trade_times,
        trade_quantities=trade_quantities,
        groups=groups,
    )
    source_identity = {
        "manifests": manifests,
        "manifest_sha256": _canonical_sha256(manifests),
        "book_ticker_100ms_rows_read": int(len(available_ms)),
        "trade_rows_read": int(len(trades["trade_time_ms"])),
    }
    output: dict[str, Any] = {
        "symbol": symbol,
        "source": source_identity,
        "decision_rows_planned": int(len(decisions_ms) + np.sum(~eligible)),
        "decision_rows_quote_age_eligible": int(len(decisions_ms)),
        "decision_rows_quote_age_rejected": int(np.sum(~eligible)),
        "quote_age_p50_ms": float(np.quantile(quote_age_ms, 0.50)),
        "quote_age_p99_ms": float(np.quantile(quote_age_ms, 0.99)),
        "long_full_fill": {},
        "short_full_fill": {},
    }
    for label, counts in (
        ("long_full_fill", long_counts),
        ("short_full_fill", short_counts),
    ):
        output[label] = {
            f"{window // 1_000}s": {
                "count": int(counts[window]),
                "ratio": float(counts[window] / len(decisions_ms)),
            }
            for window in WINDOWS_MS
        }
    return output


def run(arguments: argparse.Namespace) -> int:
    database = arguments.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    start_ms, end_ms = _day_bounds_ms(arguments.day)
    cadence_ms = int(arguments.decision_cadence_seconds * 1_000)
    if cadence_ms <= 0 or cadence_ms > 60_000:
        raise ValueError("decision cadence must lie in [1, 60] seconds")
    if not 0 <= arguments.placement_latency_ms <= 10_000:
        raise ValueError("placement latency must lie in [0, 10000] milliseconds")
    if not 1 <= arguments.maximum_quote_age_ms <= 10_000:
        raise ValueError("maximum quote age must lie in [1, 10000] milliseconds")
    if not np.isfinite(arguments.order_notional_quote) or not (
        0.0 < arguments.order_notional_quote <= 1_000_000.0
    ):
        raise ValueError("order notional is outside the supported range")
    with duckdb.connect(str(database), read_only=True) as connection:
        connection.execute("PRAGMA disable_progress_bar")
        symbols = [
            _probe_symbol(
                connection,
                symbol=symbol,
                period=arguments.day,
                start_ms=start_ms,
                end_ms=end_ms,
                decision_cadence_ms=cadence_ms,
                placement_latency_ms=arguments.placement_latency_ms,
                maximum_quote_age_ms=arguments.maximum_quote_age_ms,
                order_notional_quote=arguments.order_notional_quote,
            )
            for symbol in SYMBOLS
        ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "purpose": "structural_fill_support_only",
        "strategy_outcomes_read": False,
        "profit_and_loss_read": False,
        "policy_thresholds_selected": False,
        "market_source": "official Binance Data Vision USD-M daily archives",
        "day_utc": arguments.day,
        "symbols": list(SYMBOLS),
        "contract": {
            "decision_cadence_seconds": arguments.decision_cadence_seconds,
            "placement_latency_ms": arguments.placement_latency_ms,
            "maximum_quote_age_ms": arguments.maximum_quote_age_ms,
            "order_notional_quote": arguments.order_notional_quote,
            "full_displayed_l1_queue_ahead": True,
            "own_order_quantity_included": True,
            "cancellation_fill_credit": False,
            "matching_prints_only": True,
            "long_print": "buyer_is_maker=true and trade price equals placement bid",
            "short_print": "buyer_is_maker=false and trade price equals placement ask",
            "full_fill_required": True,
            "windows_ms": list(WINDOWS_MS),
        },
        "symbol_results": symbols,
    }
    payload["report_sha256"] = _canonical_sha256(payload)
    write_json_atomic(arguments.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--day", default="2023-06-01")
    parser.add_argument("--decision-cadence-seconds", type=int, default=10)
    parser.add_argument("--placement-latency-ms", type=int, default=750)
    parser.add_argument("--maximum-quote-age-ms", type=int, default=1_000)
    parser.add_argument("--order-notional-quote", type=float, default=1_000.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))

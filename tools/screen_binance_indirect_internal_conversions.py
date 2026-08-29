"""Screen public Binance Spot books for cheaper indirect organic conversions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Iterable, Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


BASE_URL = "https://api.binance.com"
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-indirect-internal-conversion-contract-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-indirect-internal-conversion-screen-v1-2026-08-29.json"
)
DEFAULT_RAW_DIR = Path(
    "data/edge-research/binance-indirect-internal-conversion-screen-v1/raw"
)
IMPLEMENTATION_PATH = Path("tools/screen_binance_indirect_internal_conversions.py")
TEN_THOUSAND = Decimal(10_000)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    symbol: str
    side: str
    step: Decimal


@dataclass(frozen=True)
class LegResult:
    output: Decimal
    residual_input: Decimal
    capacity_ok: bool


def _positive_market_step(symbol: Mapping[str, object]) -> Decimal:
    fallback: Decimal | None = None
    for raw_filter in _list(symbol.get("filters"), name="symbol filters"):
        item = _mapping(raw_filter, name="symbol filter")
        step = Decimal(str(item.get("stepSize", "0")))
        if item.get("filterType") == "MARKET_LOT_SIZE" and step > 0:
            return step
        if item.get("filterType") == "LOT_SIZE" and step > 0:
            fallback = step
    if fallback is None:
        raise ValueError(f"{symbol.get('symbol')} lacks a positive lot step")
    return fallback


def build_edges(exchange_info: object) -> dict[tuple[str, str], Edge]:
    payload = _mapping(exchange_info, name="exchange info")
    result: dict[tuple[str, str], Edge] = {}
    for raw_symbol in _list(payload.get("symbols"), name="exchange symbols"):
        item = _mapping(raw_symbol, name="exchange symbol")
        if (
            item.get("status") != "TRADING"
            or item.get("isSpotTradingAllowed") is not True
            or "MARKET" not in _list(item.get("orderTypes"), name="order types")
        ):
            continue
        base = str(item["baseAsset"])
        quote = str(item["quoteAsset"])
        symbol = str(item["symbol"])
        step = _positive_market_step(item)
        result[(base, quote)] = Edge(base, quote, symbol, "SELL", step)
        result[(quote, base)] = Edge(quote, base, symbol, "BUY", step)
    return result


def build_routes(edges: Mapping[tuple[str, str], Edge]) -> list[tuple[Edge, Edge, Edge]]:
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges.values():
        outgoing[edge.source].append(edge)
    routes: list[tuple[Edge, Edge, Edge]] = []
    for (source, target), direct in sorted(edges.items()):
        if source == target:
            continue
        for first in outgoing[source]:
            intermediary = first.target
            if intermediary in {source, target}:
                continue
            second = edges.get((intermediary, target))
            if second is not None:
                routes.append((direct, first, second))
    return routes


def parse_books(raw: object) -> dict[str, dict[str, Decimal]]:
    books: dict[str, dict[str, Decimal]] = {}
    for raw_row in _list(raw, name="book ticker"):
        row = _mapping(raw_row, name="book row")
        parsed = {
            key: Decimal(str(row[key]))
            for key in ("bidPrice", "bidQty", "askPrice", "askQty")
        }
        if any(value <= 0 for value in parsed.values()):
            continue
        books[str(row["symbol"])] = parsed
    return books


def execute_leg(
    amount: Decimal,
    edge: Edge,
    book: Mapping[str, Decimal],
    fee_rate: Decimal,
) -> LegResult:
    if amount <= 0:
        return LegResult(Decimal(0), amount, False)
    if edge.side == "SELL":
        submitted = _floor_step(amount, edge.step)
        gross_output = submitted * book["bidPrice"]
        return LegResult(
            gross_output * (Decimal(1) - fee_rate),
            amount - submitted,
            submitted > 0 and submitted <= book["bidQty"],
        )
    submitted = _floor_step(amount / book["askPrice"], edge.step)
    spent = submitted * book["askPrice"]
    return LegResult(
        submitted * (Decimal(1) - fee_rate),
        amount - spent,
        submitted > 0 and submitted <= book["askQty"],
    )


def marginal_rate(edge: Edge, book: Mapping[str, Decimal], fee_rate: Decimal) -> Decimal:
    if edge.side == "SELL":
        return book["bidPrice"] * (Decimal(1) - fee_rate)
    return (Decimal(1) / book["askPrice"]) * (Decimal(1) - fee_rate)


def _usdt_rate(
    asset: str,
    edges: Mapping[tuple[str, str], Edge],
    books: Mapping[str, Mapping[str, Decimal]],
    fee_rate: Decimal,
) -> Decimal | None:
    if asset == "USDT":
        return Decimal(1)
    edge = edges.get((asset, "USDT"))
    if edge is None or edge.symbol not in books:
        return None
    return marginal_rate(edge, books[edge.symbol], fee_rate)


def evaluate_route(
    route: tuple[Edge, Edge, Edge],
    *,
    start_usdt: Decimal,
    edges: Mapping[tuple[str, str], Edge],
    books: Mapping[str, Mapping[str, Decimal]],
    fee_rate: Decimal,
    stress_bips: Decimal,
) -> dict[str, object] | None:
    direct, first, second = route
    required_symbols = {direct.symbol, first.symbol, second.symbol}
    if not required_symbols.issubset(books):
        return None
    source_rate = _usdt_rate(direct.source, edges, books, fee_rate)
    target_rate = _usdt_rate(direct.target, edges, books, fee_rate)
    intermediary_rate = _usdt_rate(first.target, edges, books, fee_rate)
    if not source_rate or not target_rate or not intermediary_rate:
        return None
    start_amount = start_usdt / source_rate
    direct_result = execute_leg(start_amount, direct, books[direct.symbol], fee_rate)
    first_result = execute_leg(start_amount, first, books[first.symbol], fee_rate)
    second_result = execute_leg(
        first_result.output, second, books[second.symbol], fee_rate
    )
    direct_value = (
        direct_result.output * target_rate
        + direct_result.residual_input * source_rate
    )
    indirect_residual_value = (
        first_result.residual_input * source_rate
        + second_result.residual_input * intermediary_rate
    )
    indirect_value = second_result.output * target_rate + indirect_residual_value
    if direct_value <= 0:
        return None
    savings_bips = (indirect_value / direct_value - Decimal(1)) * TEN_THOUSAND
    residual_bips = indirect_residual_value / start_usdt * TEN_THOUSAND
    stressed_bips = savings_bips - stress_bips
    capacity_ok = (
        direct_result.capacity_ok
        and first_result.capacity_ok
        and second_result.capacity_ok
    )
    return {
        "route_id": (
            f"{direct.source}->{first.target}->{direct.target}"
            f"_vs_{direct.source}->{direct.target}"
        ),
        "source": direct.source,
        "intermediary": first.target,
        "target": direct.target,
        "direct_symbol": direct.symbol,
        "indirect_symbols": [first.symbol, second.symbol],
        "start_usdt": _decimal_text(start_usdt),
        "savings_bips": savings_bips,
        "stressed_savings_bips": stressed_bips,
        "indirect_residual_bips": residual_bips,
        "capacity_ok": capacity_ok,
        "positive_after_gates": capacity_ok and residual_bips <= 1 and stressed_bips > 0,
    }


def _contract() -> tuple[dict[str, object], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text()), name="contract")
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"contract hash differs: declared={declared} actual={actual}")
    frozen_at_ms = int(contract["frozen_at_ms"])
    if frozen_at_ms > time.time_ns() // 1_000_000:
        raise ValueError("contract frozen_at_ms is in the future")
    if contract.get("status") != "frozen_before_public_book_outcome_access":
        raise ValueError("contract is not frozen")
    implementation_hash = _sha256(IMPLEMENTATION_PATH.read_bytes())
    if contract["implementation"]["sha256"] != implementation_hash:
        raise ValueError("implementation differs from frozen contract")
    contract["contract_sha256"] = declared
    return contract, actual


def _append_journal(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((_canonical_json(value) + "\n").encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def _get(session: requests.Session, path: str) -> tuple[bytes, dict[str, object]]:
    started_ns = time.time_ns()
    response = session.get(f"{BASE_URL}{path}", timeout=10)
    ended_ns = time.time_ns()
    response.raise_for_status()
    body = response.content
    return body, {
        "request_started_ns": started_ns,
        "response_ended_ns": ended_ns,
        "request_elapsed_ms": (ended_ns - started_ns) // 1_000_000,
        "status_code": response.status_code,
        "url": response.url,
        "response_bytes": len(body),
        "response_sha256": _sha256(body),
    }


def _median(values: Iterable[Decimal]) -> Decimal:
    return Decimal(str(statistics.median(list(values))))


def run(output: Path, raw_dir: Path) -> dict[str, object]:
    contract, contract_hash = _contract()
    capture = _mapping(contract["capture"], name="capture")
    model = _mapping(contract["execution_model"], name="execution model")
    sample_count = int(capture["sample_count"])
    interval_ms = int(capture["interval_ms"])
    fee_rate = Decimal(str(model["conservative_taker_fee_rate_per_leg"]))
    stress_bips = Decimal(str(model["extra_leg_operational_stress_bips"]))
    sizes = [Decimal(str(value)) for value in model["starting_usdt_sizes"]]
    journal = raw_dir / "journal.jsonl"
    if journal.exists() or output.exists():
        raise FileExistsError("one-use output or journal already exists")
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "simple-ai-trading-research/1.0"
    exchange_bytes, exchange_receipt = _get(session, "/api/v3/exchangeInfo")
    write_bytes_atomic(raw_dir / "exchangeInfo.json", exchange_bytes)
    _append_journal(journal, {"kind": "exchange_info_receipt", **exchange_receipt})
    exchange_info = json.loads(exchange_bytes)
    edges = build_edges(exchange_info)
    routes = build_routes(edges)
    aggregates: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    request_elapsed: list[int] = [int(exchange_receipt["request_elapsed_ms"])]
    capture_started_ns = time.time_ns()
    for sample_index in range(sample_count):
        scheduled_ns = capture_started_ns + sample_index * interval_ms * 1_000_000
        remaining_ns = scheduled_ns - time.time_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
        book_bytes, receipt = _get(session, "/api/v3/ticker/bookTicker")
        raw_path = raw_dir / f"bookTicker-{sample_index:03d}.json"
        write_bytes_atomic(raw_path, book_bytes)
        _append_journal(
            journal,
            {"kind": "book_receipt", "sample_index": sample_index, **receipt},
        )
        request_elapsed.append(int(receipt["request_elapsed_ms"]))
        books = parse_books(json.loads(book_bytes))
        sample_rows: list[dict[str, object]] = []
        for route in routes:
            for size in sizes:
                row = evaluate_route(
                    route,
                    start_usdt=size,
                    edges=edges,
                    books=books,
                    fee_rate=fee_rate,
                    stress_bips=stress_bips,
                )
                if row is None:
                    continue
                key = (str(row["route_id"]), str(row["start_usdt"]))
                aggregates[key].append(row)
                sample_rows.append(row)
        top = sorted(
            sample_rows,
            key=lambda row: row["stressed_savings_bips"],
            reverse=True,
        )[:10]
        _append_journal(
            journal,
            {
                "kind": "sample_evaluation",
                "sample_index": sample_index,
                "evaluated_route_sizes": len(sample_rows),
                "top_ten": [
                    {
                        **{key: value for key, value in row.items() if not isinstance(value, Decimal)},
                        "savings_bips": _decimal_text(row["savings_bips"]),
                        "stressed_savings_bips": _decimal_text(
                            row["stressed_savings_bips"]
                        ),
                        "indirect_residual_bips": _decimal_text(
                            row["indirect_residual_bips"]
                        ),
                    }
                    for row in top
                ],
            },
        )
    capture_ended_ns = time.time_ns()
    summaries: list[dict[str, object]] = []
    for (route_id, size), rows in aggregates.items():
        stressed = [row["stressed_savings_bips"] for row in rows]
        positive_indices = [
            index for index, row in enumerate(rows) if row["positive_after_gates"]
        ]
        blocks = [
            sum(start <= index < start + 15 for index in positive_indices)
            for start in range(0, sample_count, 15)
        ]
        candidate = (
            len(rows) == sample_count
            and all(bool(row["capacity_ok"]) for row in rows)
            and len(positive_indices) >= 48
            and len(blocks) == 4
            and all(count >= 12 for count in blocks)
            and _median(stressed) > 0
        )
        summaries.append(
            {
                "route_id": route_id,
                "start_usdt": size,
                "observations": len(rows),
                "capacity_ok_observations": sum(
                    bool(row["capacity_ok"]) for row in rows
                ),
                "positive_after_gates_observations": len(positive_indices),
                "positive_counts_by_15_sample_block": blocks,
                "minimum_stressed_savings_bips": _decimal_text(min(stressed)),
                "median_stressed_savings_bips": _decimal_text(_median(stressed)),
                "maximum_stressed_savings_bips": _decimal_text(max(stressed)),
                "empirical_candidate": candidate,
            }
        )
    summaries.sort(
        key=lambda row: Decimal(str(row["median_stressed_savings_bips"])),
        reverse=True,
    )
    candidates = [row for row in summaries if row["empirical_candidate"]]
    result_without_hash: dict[str, object] = {
        "schema_version": "binance-indirect-internal-conversion-screen-v1",
        "status": "completed_research_only",
        "contract_sha256": contract_hash,
        "captured_at_start_ns": capture_started_ns,
        "captured_at_end_ns": capture_ended_ns,
        "capture_elapsed_ms": (capture_ended_ns - capture_started_ns) // 1_000_000,
        "sample_count": sample_count,
        "maximum_request_elapsed_ms": max(request_elapsed),
        "directed_edges": len(edges),
        "direct_vs_one_intermediary_routes": len(routes),
        "evaluated_route_sizes": len(aggregates),
        "empirical_candidate_count": len(candidates),
        "empirical_candidates": candidates,
        "top_100_by_median_stressed_savings": summaries[:100],
        "accepted_edge": False,
        "decision": (
            "candidate_only_requires_account_fee_and_independent_execution_windows"
            if candidates
            else "rejected_current_public_screen_do_not_repeat_without_material_trigger"
        ),
        "raw_journal_sha256": _sha256(journal.read_bytes()),
    }
    result_hash = _sha256(_canonical_json(result_without_hash).encode("ascii"))
    result = {**result_without_hash, "result_sha256": result_hash}
    write_bytes_atomic(
        output,
        (json.dumps(result, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)
    result = run(args.output, args.raw_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

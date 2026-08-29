"""Adjudicate all active Binance Spot triangular cycles from retained books."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Iterable, Mapping, Sequence

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_binance_indirect_internal_conversions import (
    Edge,
    _usdt_rate,
    build_edges,
    execute_leg,
    parse_books,
)


CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-all-symbol-triangular-cycle-retained-contract-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-all-symbol-triangular-cycle-retained-v1-2026-08-29.json"
)
RAW_DIR = Path(
    "data/edge-research/binance-indirect-internal-conversion-screen-v1/raw"
)
ACTIVITY_PATH = Path(
    "data/edge-research/binance-indirect-internal-conversion-activity-v1/"
    "ticker-24hr.json"
)
IMPLEMENTATION_PATH = Path(
    "tools/adjudicate_binance_all_symbol_triangular_cycles_retained.py"
)
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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def build_cycles(edges: Mapping[tuple[str, str], Edge]) -> list[tuple[Edge, Edge, Edge]]:
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges.values():
        outgoing[edge.source].append(edge)
    cycles: list[tuple[Edge, Edge, Edge]] = []
    for first in edges.values():
        for second in outgoing[first.target]:
            if second.target in {first.source, first.target}:
                continue
            third = edges.get((second.target, first.source))
            if third is None:
                continue
            assets = (first.source, first.target, second.target)
            rotations = (assets, assets[1:] + assets[:1], assets[2:] + assets[:2])
            if assets != min(rotations):
                continue
            cycles.append((first, second, third))
    return sorted(cycles, key=lambda row: tuple(edge.source for edge in row))


def cycle_id(cycle: tuple[Edge, Edge, Edge]) -> str:
    return "->".join(
        [cycle[0].source, cycle[0].target, cycle[1].target, cycle[0].source]
    )


def evaluate_cycle(
    cycle: tuple[Edge, Edge, Edge],
    *,
    start_usdt: Decimal,
    edges: Mapping[tuple[str, str], Edge],
    books: Mapping[str, Mapping[str, Decimal]],
    fee_rate: Decimal,
    stress_bips: Decimal,
) -> dict[str, object] | None:
    assets = [cycle[0].source, cycle[0].target, cycle[1].target]
    symbols = [edge.symbol for edge in cycle]
    if not set(symbols).issubset(books):
        return None
    rates = {
        asset: _usdt_rate(asset, edges, books, fee_rate) for asset in assets
    }
    if any(not rate for rate in rates.values()):
        return None
    source = assets[0]
    source_rate = rates[source]
    if source_rate is None:
        return None
    amount = start_usdt / source_rate
    residuals: list[tuple[str, Decimal]] = []
    capacity_ok = True
    for edge in cycle:
        result = execute_leg(amount, edge, books[edge.symbol], fee_rate)
        residuals.append((edge.source, result.residual_input))
        amount = result.output
        capacity_ok = capacity_ok and result.capacity_ok
    residual_usdt = sum(
        (quantity * rates[asset] for asset, quantity in residuals),
        start=Decimal(0),
    )
    ending_usdt = amount * source_rate + residual_usdt
    raw_bips = (ending_usdt / start_usdt - Decimal(1)) * TEN_THOUSAND
    stressed_bips = raw_bips - stress_bips
    residual_bips = residual_usdt / start_usdt * TEN_THOUSAND
    return {
        "cycle_id": cycle_id(cycle),
        "symbols": symbols,
        "start_usdt": _decimal_text(start_usdt),
        "raw_profit_bips": raw_bips,
        "stressed_profit_bips": stressed_bips,
        "residual_bips": residual_bips,
        "capacity_ok": capacity_ok,
        "positive_after_gates": (
            capacity_ok and residual_bips <= Decimal(1) and stressed_bips > 0
        ),
    }


def _manifest_hash(raw_dir: Path) -> str:
    paths = [raw_dir / "exchangeInfo.json", raw_dir / "journal.jsonl"] + [
        raw_dir / f"bookTicker-{index:03d}.json" for index in range(60)
    ]
    records = [f"{path.name}\0{_sha256(path.read_bytes())}" for path in paths]
    return _sha256(("\n".join(sorted(records)) + "\n").encode("utf-8"))


def _contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    declared = str(contract.pop("contract_sha256"))
    actual = _sha256(_canonical_json(contract).encode("ascii"))
    if declared != actual:
        raise ValueError(f"contract hash differs: declared={declared} actual={actual}")
    if int(contract["frozen_at_ms"]) > time.time_ns() // 1_000_000:
        raise ValueError("contract frozen_at_ms is in the future")
    if contract.get("status") != "frozen_before_retained_cycle_outcome_access":
        raise ValueError("contract is not frozen")
    if contract["implementation"]["sha256"] != _sha256(
        IMPLEMENTATION_PATH.read_bytes()
    ):
        raise ValueError("implementation differs from frozen contract")
    inputs = _mapping(contract["retained_inputs"], name="retained inputs")
    if inputs["manifest_sha256"] != _manifest_hash(RAW_DIR):
        raise ValueError("retained book manifest differs from contract")
    if inputs["activity_sha256"] != _sha256(ACTIVITY_PATH.read_bytes()):
        raise ValueError("retained activity response differs from contract")
    contract["contract_sha256"] = declared
    return contract, actual


def _median(values: Iterable[Decimal]) -> Decimal:
    return statistics.median(list(values))


def _quote_changes(
    symbols: Sequence[str],
    books: Sequence[Mapping[str, Mapping[str, Decimal]]],
) -> dict[str, int]:
    return {
        symbol: sum(
            books[index][symbol] != books[index - 1][symbol]
            for index in range(1, len(books))
        )
        for symbol in symbols
    }


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("one-use retained adjudication output already exists")
    contract, contract_hash = _contract()
    model = _mapping(contract["execution_model"], name="execution model")
    activity = _mapping(contract["activity_gate"], name="activity gate")
    sizes = [Decimal(str(value)) for value in model["starting_usdt_sizes"]]
    stress_bips = Decimal(str(model["operational_stress_bips"]))
    scenarios = {
        str(name): Decimal(str(rate))
        for name, rate in _mapping(model["fee_scenarios"], name="fee scenarios").items()
    }
    exchange_info = json.loads((RAW_DIR / "exchangeInfo.json").read_text("utf-8"))
    books = [
        parse_books(
            json.loads(
                (RAW_DIR / f"bookTicker-{index:03d}.json").read_text("utf-8")
            )
        )
        for index in range(60)
    ]
    tickers = {
        str(row["symbol"]): row
        for row in json.loads(ACTIVITY_PATH.read_text(encoding="utf-8"))
    }
    edges = build_edges(exchange_info)
    cycles = build_cycles(edges)
    active_cycles: list[tuple[Edge, Edge, Edge]] = []
    activity_rows: list[dict[str, object]] = []
    for cycle in cycles:
        symbols = [edge.symbol for edge in cycle]
        if not set(symbols).issubset(books[0]) or not set(symbols).issubset(tickers):
            continue
        changes = _quote_changes(symbols, books)
        counts = {symbol: int(tickers[symbol]["count"]) for symbol in symbols}
        passed = all(
            value >= int(activity["minimum_quote_changes_per_symbol"])
            for value in changes.values()
        ) and all(
            value >= int(activity["minimum_24h_trade_count_per_symbol"])
            for value in counts.values()
        )
        if passed:
            active_cycles.append(cycle)
        activity_rows.append(
            {
                "cycle_id": cycle_id(cycle),
                "symbols": symbols,
                "minimum_quote_changes": min(changes.values()),
                "minimum_24h_trade_count": min(counts.values()),
                "activity_gate_passed": passed,
            }
        )
    summaries: list[dict[str, object]] = []
    for scenario, fee_rate in scenarios.items():
        for cycle in active_cycles:
            for size in sizes:
                rows = [
                    evaluate_cycle(
                        cycle,
                        start_usdt=size,
                        edges=edges,
                        books=book,
                        fee_rate=fee_rate,
                        stress_bips=stress_bips,
                    )
                    for book in books
                ]
                valid = [row for row in rows if row is not None]
                if not valid:
                    continue
                positive = [
                    index
                    for index, row in enumerate(valid)
                    if bool(row["positive_after_gates"])
                ]
                stressed = [row["stressed_profit_bips"] for row in valid]
                blocks = [
                    sum(start <= index < start + 15 for index in positive)
                    for start in range(0, 60, 15)
                ]
                candidate = (
                    len(valid) == 60
                    and all(bool(row["capacity_ok"]) for row in valid)
                    and len(positive) >= 48
                    and all(count >= 12 for count in blocks)
                    and _median(stressed) > 0
                )
                summaries.append(
                    {
                        "scenario": scenario,
                        "cycle_id": cycle_id(cycle),
                        "symbols": [edge.symbol for edge in cycle],
                        "start_usdt": _decimal_text(size),
                        "observations": len(valid),
                        "capacity_ok_observations": sum(
                            bool(row["capacity_ok"]) for row in valid
                        ),
                        "positive_after_gates_observations": len(positive),
                        "positive_counts_by_15_sample_block": blocks,
                        "minimum_stressed_profit_bips": _decimal_text(min(stressed)),
                        "median_stressed_profit_bips": _decimal_text(_median(stressed)),
                        "maximum_stressed_profit_bips": _decimal_text(max(stressed)),
                        "candidate": candidate,
                    }
                )
    summaries.sort(
        key=lambda row: (
            bool(row["candidate"]),
            Decimal(str(row["median_stressed_profit_bips"])),
        ),
        reverse=True,
    )
    candidates = [row for row in summaries if row["candidate"]]
    zero_fee = [row for row in candidates if row["scenario"] == "zero_fee_upper_bound"]
    conservative = [
        row for row in candidates if row["scenario"] == "vip0_all_taker"
    ]
    result_without_hash: dict[str, object] = {
        "schema_version": "binance-all-symbol-triangular-cycle-retained-v1",
        "status": "completed_zero_network_retained_adjudication",
        "contract_sha256": contract_hash,
        "retained_manifest_sha256": _manifest_hash(RAW_DIR),
        "activity_sha256": _sha256(ACTIVITY_PATH.read_bytes()),
        "directed_edges": len(edges),
        "unique_directed_cycles": len(cycles),
        "activity_qualified_cycles": len(active_cycles),
        "activity_gate_rows": activity_rows,
        "evaluated_cycle_size_scenarios": len(summaries),
        "zero_fee_upper_bound_candidate_count": len(zero_fee),
        "vip0_all_taker_candidate_count": len(conservative),
        "zero_fee_upper_bound_candidates": zero_fee,
        "vip0_all_taker_candidates": conservative,
        "top_100_feasible_first": summaries[:100],
        "accepted_edge": False,
        "decision": (
            "account_fee_gated_candidate_only"
            if zero_fee and not conservative
            else "conservative_candidate_requires_independent_execution_validation"
            if conservative
            else "terminal_current_retained_population_no_gross_candidate"
        ),
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
    args = parser.parse_args(argv)
    result = run(args.output)
    summary = {
        key: result[key]
        for key in (
            "result_sha256",
            "unique_directed_cycles",
            "activity_qualified_cycles",
            "zero_fee_upper_bound_candidate_count",
            "vip0_all_taker_candidate_count",
            "decision",
        )
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

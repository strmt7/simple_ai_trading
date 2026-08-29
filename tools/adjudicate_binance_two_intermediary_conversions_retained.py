"""Test two-intermediary Binance Spot conversion paths on retained books."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterator, Mapping, Sequence

from simple_ai_trading.storage import write_bytes_atomic
from tools import screen_binance_indirect_internal_conversions as v1


CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-contract-v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-v1-2026-08-29.json"
)
SOURCE_RESULT = Path(
    "docs/model-research/action-value/"
    "binance-indirect-internal-conversion-screen-v1-2026-08-29.json"
)
SOURCE_RAW = Path(
    "data/edge-research/binance-indirect-internal-conversion-screen-v1/raw"
)
IMPLEMENTATION_PATH = Path(
    "tools/adjudicate_binance_two_intermediary_conversions_retained.py"
)
DEPENDENCY_PATH = Path("tools/screen_binance_indirect_internal_conversions.py")
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


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field))
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != actual:
        raise ValueError(f"{field} differs: claimed={claimed} actual={actual}")
    return claimed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _load_contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if contract.get("status") != "frozen_before_retained_outcome_access":
        raise ValueError("contract is not frozen")
    claimed = _canonical_hash(contract, "contract_sha256")
    implementation = contract["implementation"]
    if _sha256(IMPLEMENTATION_PATH.read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation differs from frozen contract")
    if _sha256(DEPENDENCY_PATH.read_bytes()) != implementation["dependency_sha256"]:
        raise ValueError("dependency differs from frozen contract")
    return contract, claimed


def _load_retained_books(contract: Mapping[str, object]) -> tuple[object, list[object]]:
    source = contract["retained_source"]
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="ascii"))
    if _canonical_hash(source_result, "result_sha256") != source["result_sha256"]:
        raise ValueError("retained source result hash differs")
    if _sha256(SOURCE_RESULT.read_bytes()) != source["result_file_sha256"]:
        raise ValueError("retained source result file differs")

    receipts = [
        json.loads(line)
        for line in (SOURCE_RAW / "journal.jsonl")
        .read_text(encoding="ascii")
        .splitlines()
        if line
    ]
    exchange_receipts = [
        row for row in receipts if row.get("kind") == "exchange_info_receipt"
    ]
    book_receipts = [row for row in receipts if row.get("kind") == "book_receipt"]
    if len(exchange_receipts) != 1 or len(book_receipts) != source["sample_count"]:
        raise ValueError("retained receipt population differs")
    exchange_path = SOURCE_RAW / "exchangeInfo.json"
    if _sha256(exchange_path.read_bytes()) != exchange_receipts[0]["response_sha256"]:
        raise ValueError("retained exchangeInfo hash differs")
    books: list[object] = []
    for expected_index, receipt in enumerate(book_receipts):
        if receipt["sample_index"] != expected_index:
            raise ValueError("retained book receipt ordering differs")
        path = SOURCE_RAW / f"bookTicker-{expected_index:03d}.json"
        payload = path.read_bytes()
        if _sha256(payload) != receipt["response_sha256"]:
            raise ValueError(f"retained book hash differs at {expected_index}")
        books.append(json.loads(payload))
    return json.loads(exchange_path.read_bytes()), books


def _outgoing(
    edges: Mapping[tuple[str, str], v1.Edge],
) -> dict[str, list[v1.Edge]]:
    result: dict[str, list[v1.Edge]] = defaultdict(list)
    for edge in edges.values():
        result[edge.source].append(edge)
    for rows in result.values():
        rows.sort(key=lambda row: (row.target, row.symbol, row.side))
    return result


def _three_leg_routes(
    edges: Mapping[tuple[str, str], v1.Edge],
) -> Iterator[tuple[v1.Edge, v1.Edge, v1.Edge, v1.Edge]]:
    outgoing = _outgoing(edges)
    for (source, target), direct in sorted(edges.items()):
        for first in outgoing[source]:
            first_mid = first.target
            if first_mid in {source, target}:
                continue
            for second in outgoing[first_mid]:
                second_mid = second.target
                if second_mid in {source, target, first_mid}:
                    continue
                third = edges.get((second_mid, target))
                if third is not None:
                    yield direct, first, second, third


def _route_value(
    start_amount: Decimal,
    legs: Sequence[v1.Edge],
    *,
    source_rate: Decimal,
    target_rate: Decimal,
    residual_rates: Sequence[Decimal],
    books: Mapping[str, Mapping[str, Decimal]],
    fee_rate: Decimal,
) -> tuple[Decimal, Decimal, bool] | None:
    if any(leg.symbol not in books for leg in legs):
        return None
    amount = start_amount
    residual_value = Decimal(0)
    capacity_ok = True
    for index, leg in enumerate(legs):
        result = v1.execute_leg(amount, leg, books[leg.symbol], fee_rate)
        residual_rate = source_rate if index == 0 else residual_rates[index - 1]
        residual_value += result.residual_input * residual_rate
        capacity_ok = capacity_ok and result.capacity_ok
        amount = result.output
    total_value = amount * target_rate + residual_value
    return total_value, residual_value, capacity_ok


def _median(values: Sequence[Decimal]) -> Decimal:
    return Decimal(str(statistics.median(values)))


def run(output: Path) -> dict[str, object]:
    contract, contract_hash = _load_contract()
    exchange_info, raw_books = _load_retained_books(contract)
    model = contract["execution_model"]
    fee_rate = Decimal(str(model["conservative_taker_fee_rate_per_leg"]))
    per_extra_leg_stress = Decimal(str(model["per_extra_leg_operational_stress_bips"]))
    maximum_residual_bips = Decimal(str(model["maximum_path_residual_bips"]))
    sizes = [Decimal(str(value)) for value in model["starting_usdt_sizes"]]
    edges = v1.build_edges(exchange_info)
    books_by_sample = [v1.parse_books(raw) for raw in raw_books]
    sample_count = len(books_by_sample)

    one_routes_by_pair: dict[
        tuple[str, str], list[tuple[v1.Edge, v1.Edge, v1.Edge]]
    ] = defaultdict(list)
    for route in v1.build_routes(edges):
        one_routes_by_pair[(route[0].source, route[0].target)].append(route)

    baselines: dict[tuple[str, str, str, int], Decimal] = {}
    direct_valid: dict[tuple[str, str, str, int], bool] = {}
    for (source, target), direct in sorted(edges.items()):
        for size in sizes:
            size_text = _decimal_text(size)
            for sample_index, books in enumerate(books_by_sample):
                source_rate = v1._usdt_rate(source, edges, books, fee_rate)
                target_rate = v1._usdt_rate(target, edges, books, fee_rate)
                key = (source, target, size_text, sample_index)
                if not source_rate or not target_rate:
                    direct_valid[key] = False
                    continue
                start_amount = size / source_rate
                direct_value_row = _route_value(
                    start_amount,
                    [direct],
                    source_rate=source_rate,
                    target_rate=target_rate,
                    residual_rates=[],
                    books=books,
                    fee_rate=fee_rate,
                )
                if (
                    direct_value_row is None
                    or not direct_value_row[2]
                    or direct_value_row[0] <= 0
                ):
                    direct_valid[key] = False
                    continue
                direct_value = direct_value_row[0]
                direct_valid[key] = True
                best_stressed_bips = Decimal(0)
                for _, first, second in one_routes_by_pair[(source, target)]:
                    mid_rate = v1._usdt_rate(first.target, edges, books, fee_rate)
                    if not mid_rate:
                        continue
                    row = _route_value(
                        start_amount,
                        [first, second],
                        source_rate=source_rate,
                        target_rate=target_rate,
                        residual_rates=[mid_rate],
                        books=books,
                        fee_rate=fee_rate,
                    )
                    if row is None or not row[2]:
                        continue
                    residual_bips = row[1] / size * TEN_THOUSAND
                    if residual_bips > maximum_residual_bips:
                        continue
                    stressed = (
                        row[0] / direct_value - Decimal(1)
                    ) * TEN_THOUSAND - per_extra_leg_stress
                    best_stressed_bips = max(best_stressed_bips, stressed)
                baselines[key] = best_stressed_bips

    summaries: list[dict[str, object]] = []
    route_count = 0
    evaluated_route_sizes = 0
    complete_route_sizes = 0
    for direct, first, second, third in _three_leg_routes(edges):
        route_count += 1
        source = direct.source
        target = direct.target
        route_id = (
            f"{source}->{first.target}->{second.target}->{target}_vs_best_shorter"
        )
        for size in sizes:
            evaluated_route_sizes += 1
            size_text = _decimal_text(size)
            increments: list[Decimal] = []
            capacity_count = 0
            residual_count = 0
            positive_indices: list[int] = []
            for sample_index, books in enumerate(books_by_sample):
                key = (source, target, size_text, sample_index)
                if not direct_valid.get(key):
                    continue
                source_rate = v1._usdt_rate(source, edges, books, fee_rate)
                target_rate = v1._usdt_rate(target, edges, books, fee_rate)
                first_mid_rate = v1._usdt_rate(first.target, edges, books, fee_rate)
                second_mid_rate = v1._usdt_rate(second.target, edges, books, fee_rate)
                if (
                    not source_rate
                    or not target_rate
                    or not first_mid_rate
                    or not second_mid_rate
                ):
                    continue
                start_amount = size / source_rate
                direct_row = _route_value(
                    start_amount,
                    [direct],
                    source_rate=source_rate,
                    target_rate=target_rate,
                    residual_rates=[],
                    books=books,
                    fee_rate=fee_rate,
                )
                row = _route_value(
                    start_amount,
                    [first, second, third],
                    source_rate=source_rate,
                    target_rate=target_rate,
                    residual_rates=[first_mid_rate, second_mid_rate],
                    books=books,
                    fee_rate=fee_rate,
                )
                if direct_row is None or row is None or direct_row[0] <= 0:
                    continue
                if row[2]:
                    capacity_count += 1
                residual_bips = row[1] / size * TEN_THOUSAND
                if residual_bips <= maximum_residual_bips:
                    residual_count += 1
                three_leg_stressed = (
                    row[0] / direct_row[0] - Decimal(1)
                ) * TEN_THOUSAND - per_extra_leg_stress * 2
                incremental = three_leg_stressed - baselines[key]
                increments.append(incremental)
                if (
                    row[2]
                    and residual_bips <= maximum_residual_bips
                    and incremental > 0
                ):
                    positive_indices.append(sample_index)
            if len(increments) != sample_count:
                continue
            complete_route_sizes += 1
            blocks = [
                sum(start <= index < start + 15 for index in positive_indices)
                for start in range(0, sample_count, 15)
            ]
            candidate = (
                capacity_count == sample_count
                and residual_count == sample_count
                and len(positive_indices) >= 48
                and blocks == [count for count in blocks if count >= 12]
                and len(blocks) == 4
                and _median(increments) > 0
            )
            summaries.append(
                {
                    "route_id": route_id,
                    "source": source,
                    "intermediaries": [first.target, second.target],
                    "target": target,
                    "direct_symbol": direct.symbol,
                    "three_leg_symbols": [first.symbol, second.symbol, third.symbol],
                    "start_usdt": size_text,
                    "observations": len(increments),
                    "capacity_ok_observations": capacity_count,
                    "residual_gate_observations": residual_count,
                    "positive_incremental_observations": len(positive_indices),
                    "positive_counts_by_15_sample_block": blocks,
                    "minimum_incremental_bips": _decimal_text(min(increments)),
                    "median_incremental_bips": _decimal_text(_median(increments)),
                    "maximum_incremental_bips": _decimal_text(max(increments)),
                    "empirical_candidate": candidate,
                }
            )

    summaries.sort(
        key=lambda row: Decimal(str(row["median_incremental_bips"])), reverse=True
    )
    candidates = [row for row in summaries if row["empirical_candidate"]]
    result_without_hash: dict[str, object] = {
        "schema_version": "binance-two-intermediary-conversion-retained-v1",
        "status": "completed_zero_network_retained_adjudication",
        "contract_sha256": contract_hash,
        "retained_source_result_sha256": contract["retained_source"]["result_sha256"],
        "sample_count": sample_count,
        "directed_edges": len(edges),
        "direct_vs_two_intermediary_routes": route_count,
        "evaluated_route_sizes": evaluated_route_sizes,
        "complete_route_sizes": complete_route_sizes,
        "empirical_candidate_count": len(candidates),
        "empirical_candidates": candidates,
        "top_100_by_median_incremental_bips": summaries[:100],
        "accepted_edge": False,
        "decision": (
            "incremental_candidate_requires_account_fees_and_prospective_execution"
            if candidates
            else "retained_extension_rejected_do_not_build_a_three_leg_live_collector"
        ),
        "authority": {
            "network_requests": 0,
            "credentials_used": False,
            "account_state_accessed": False,
            "orders_or_mutations": 0,
            "protected_capture_accessed": False,
        },
    }
    result = {
        **result_without_hash,
        "result_sha256": _sha256(_canonical_json(result_without_hash).encode("ascii")),
    }
    write_bytes_atomic(output, (_canonical_json(result) + "\n").encode("ascii"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "direct_vs_two_intermediary_routes",
                    "evaluated_route_sizes",
                    "complete_route_sizes",
                    "empirical_candidate_count",
                    "decision",
                    "result_sha256",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

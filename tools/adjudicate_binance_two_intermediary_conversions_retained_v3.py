"""Stream a bounded retained-data adjudication of three-leg conversion paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import heapq
from itertools import islice
import json
from pathlib import Path
import statistics
from typing import Iterator, Mapping, Sequence

import numpy as np

from simple_ai_trading.storage import write_bytes_atomic
from tools import adjudicate_binance_two_intermediary_conversions_retained as v1
from tools import screen_binance_indirect_internal_conversions as source


CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-contract-v3.json"
)
DEFAULT_OUTPUT = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-v3-2026-08-29.json"
)
IMPLEMENTATION_PATH = Path(
    "tools/adjudicate_binance_two_intermediary_conversions_retained_v3.py"
)
V1_IMPLEMENTATION_PATH = Path(
    "tools/adjudicate_binance_two_intermediary_conversions_retained.py"
)
SOURCE_DEPENDENCY_PATH = Path("tools/screen_binance_indirect_internal_conversions.py")
V2_ADJUDICATION_PATH = Path(
    "docs/model-research/action-value/"
    "binance-two-intermediary-conversion-retained-v2-source-gate-adjudication-2026-08-29.json"
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


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field))
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != actual:
        raise ValueError(f"{field} differs: claimed={claimed} actual={actual}")
    return claimed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _median(values: Sequence[Decimal]) -> Decimal:
    return Decimal(str(statistics.median(values)))


def _load_contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if contract.get("status") != "frozen_before_v3_retained_outcome_access":
        raise ValueError("v3 contract is not frozen")
    claimed = _canonical_hash(contract, "contract_sha256")
    implementation = contract["implementation"]
    checks = (
        (IMPLEMENTATION_PATH, "sha256"),
        (V1_IMPLEMENTATION_PATH, "v1_dependency_sha256"),
        (SOURCE_DEPENDENCY_PATH, "source_dependency_sha256"),
    )
    for path, field in checks:
        if _sha256(path.read_bytes()) != implementation[field]:
            raise ValueError(f"{path} differs from frozen v3 contract")
    prior = json.loads(V2_ADJUDICATION_PATH.read_text(encoding="ascii"))
    if (
        _canonical_hash(prior, "result_sha256")
        != contract["prior_failure"]["result_sha256"]
    ):
        raise ValueError("prior v2 adjudication differs")
    return contract, claimed


def _chunks(
    iterator: Iterator[tuple[source.Edge, source.Edge, source.Edge, source.Edge]],
    size: int,
) -> Iterator[list[tuple[source.Edge, source.Edge, source.Edge, source.Edge]]]:
    while True:
        rows = list(islice(iterator, size))
        if not rows:
            return
        yield rows


def _asset_usdt_rates(
    assets: set[str],
    edges: Mapping[tuple[str, str], source.Edge],
    books_by_sample: Sequence[Mapping[str, Mapping[str, Decimal]]],
    fee_rate: Decimal,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for asset in sorted(assets):
        values = []
        for books in books_by_sample:
            rate = source._usdt_rate(asset, edges, books, fee_rate)
            values.append(float(rate) if rate else np.nan)
        result[asset] = np.asarray(values, dtype=np.float64)
    return result


def _edge_rate_matrix(
    ordered_edges: Sequence[source.Edge],
    books_by_sample: Sequence[Mapping[str, Mapping[str, Decimal]]],
    fee_rate: Decimal,
) -> np.ndarray:
    matrix = np.full(
        (len(ordered_edges), len(books_by_sample)), np.nan, dtype=np.float64
    )
    for edge_index, edge in enumerate(ordered_edges):
        for sample_index, books in enumerate(books_by_sample):
            book = books.get(edge.symbol)
            if book is not None:
                matrix[edge_index, sample_index] = float(
                    source.marginal_rate(edge, book, fee_rate)
                )
    return matrix


def _direct_exact_values(
    edges: Mapping[tuple[str, str], source.Edge],
    books_by_sample: Sequence[Mapping[str, Mapping[str, Decimal]]],
    asset_rates: Mapping[str, np.ndarray],
    sizes: Sequence[Decimal],
    fee_rate: Decimal,
) -> dict[tuple[str, str, str], np.ndarray]:
    result: dict[tuple[str, str, str], np.ndarray] = {}
    for (source_asset, target_asset), direct in sorted(edges.items()):
        source_rates = asset_rates[source_asset]
        target_rates = asset_rates[target_asset]
        for size in sizes:
            values = np.full(len(books_by_sample), np.nan, dtype=np.float64)
            for sample_index, books in enumerate(books_by_sample):
                if not np.isfinite(source_rates[sample_index]) or not np.isfinite(
                    target_rates[sample_index]
                ):
                    continue
                start_amount = size / Decimal(str(source_rates[sample_index]))
                row = v1._route_value(
                    start_amount,
                    [direct],
                    source_rate=Decimal(str(source_rates[sample_index])),
                    target_rate=Decimal(str(target_rates[sample_index])),
                    residual_rates=[],
                    books=books,
                    fee_rate=fee_rate,
                )
                if row is not None and row[2] and row[0] > 0:
                    values[sample_index] = float(row[0])
            result[(source_asset, target_asset, _decimal_text(size))] = values
    return result


def _optimistic_survivors(
    *,
    routes: Iterator[tuple[source.Edge, source.Edge, source.Edge, source.Edge]],
    edge_indexes: Mapping[tuple[str, str], int],
    edge_rates: np.ndarray,
    asset_rates: Mapping[str, np.ndarray],
    direct_values: Mapping[tuple[str, str, str], np.ndarray],
    sizes: Sequence[Decimal],
    stress_bips: float,
    residual_upper_bips: float,
    numeric_padding_bips: float,
    chunk_size: int,
) -> tuple[
    list[tuple[tuple[source.Edge, source.Edge, source.Edge, source.Edge], Decimal]],
    int,
]:
    survivors: list[
        tuple[tuple[source.Edge, source.Edge, source.Edge, source.Edge], Decimal]
    ] = []
    route_count = 0
    for chunk in _chunks(routes, chunk_size):
        route_count += len(chunk)
        path_indexes = np.asarray(
            [
                [
                    edge_indexes[(first.source, first.target)],
                    edge_indexes[(second.source, second.target)],
                    edge_indexes[(third.source, third.target)],
                ]
                for _, first, second, third in chunk
            ],
            dtype=np.int64,
        )
        path_rates = np.prod(edge_rates[path_indexes, :], axis=1)
        for size in sizes:
            size_text = _decimal_text(size)
            starts = np.stack(
                [float(size) / asset_rates[direct.source] for direct, _, _, _ in chunk]
            )
            target_rates = np.stack(
                [asset_rates[direct.target] for direct, _, _, _ in chunk]
            )
            direct_exact = np.stack(
                [
                    direct_values[(direct.source, direct.target, size_text)]
                    for direct, _, _, _ in chunk
                ]
            )
            continuous_path_value = starts * path_rates * target_rates
            optimistic_value = continuous_path_value + float(size) * (
                residual_upper_bips / 10_000
            )
            optimistic = (
                (optimistic_value / direct_exact - 1.0) * 10_000
                - stress_bips
                + numeric_padding_bips
            )
            intermediary_rates_finite = np.stack(
                [
                    np.isfinite(asset_rates[first.target])
                    & np.isfinite(asset_rates[second.target])
                    for _, first, second, _ in chunk
                ]
            ).all(axis=1)
            finite = np.isfinite(optimistic).all(axis=1) & intermediary_rates_finite
            positive = optimistic > 0
            counts = positive.sum(axis=1)
            blocks = positive.reshape(len(chunk), 4, 15).sum(axis=2)
            medians = np.full(len(chunk), -np.inf, dtype=np.float64)
            medians[finite] = np.median(optimistic[finite], axis=1)
            keep = finite & (counts >= 48) & (blocks >= 12).all(axis=1) & (medians > 0)
            for index in np.flatnonzero(keep):
                survivors.append((chunk[int(index)], size))
    return survivors, route_count


def _best_shorter_baselines(
    pairs_and_sizes: set[tuple[str, str, str]],
    *,
    edges: Mapping[tuple[str, str], source.Edge],
    books_by_sample: Sequence[Mapping[str, Mapping[str, Decimal]]],
    asset_rates: Mapping[str, np.ndarray],
    fee_rate: Decimal,
    one_leg_stress: Decimal,
    maximum_residual_bips: Decimal,
) -> dict[tuple[str, str, str, int], Decimal]:
    one_routes: dict[
        tuple[str, str], list[tuple[source.Edge, source.Edge, source.Edge]]
    ] = defaultdict(list)
    for route in source.build_routes(edges):
        pair = (route[0].source, route[0].target)
        if any(pair == item[:2] for item in pairs_and_sizes):
            one_routes[pair].append(route)
    baselines: dict[tuple[str, str, str, int], Decimal] = {}
    for source_asset, target_asset, size_text in sorted(pairs_and_sizes):
        size = Decimal(size_text)
        direct = edges[(source_asset, target_asset)]
        for sample_index, books in enumerate(books_by_sample):
            source_rate = Decimal(str(asset_rates[source_asset][sample_index]))
            target_rate = Decimal(str(asset_rates[target_asset][sample_index]))
            start_amount = size / source_rate
            direct_row = v1._route_value(
                start_amount,
                [direct],
                source_rate=source_rate,
                target_rate=target_rate,
                residual_rates=[],
                books=books,
                fee_rate=fee_rate,
            )
            if direct_row is None or not direct_row[2] or direct_row[0] <= 0:
                raise ValueError("optimistic survivor lacks exact direct baseline")
            best = Decimal(0)
            for _, first, second in one_routes[(source_asset, target_asset)]:
                middle_value = asset_rates[first.target][sample_index]
                if not np.isfinite(middle_value):
                    continue
                row = v1._route_value(
                    start_amount,
                    [first, second],
                    source_rate=source_rate,
                    target_rate=target_rate,
                    residual_rates=[Decimal(str(middle_value))],
                    books=books,
                    fee_rate=fee_rate,
                )
                if row is None or not row[2]:
                    continue
                residual_bips = row[1] / size * TEN_THOUSAND
                if residual_bips > maximum_residual_bips:
                    continue
                stressed = (
                    row[0] / direct_row[0] - Decimal(1)
                ) * TEN_THOUSAND - one_leg_stress
                best = max(best, stressed)
            baselines[(source_asset, target_asset, size_text, sample_index)] = best
    return baselines


def run(output: Path) -> dict[str, object]:
    contract, contract_hash = _load_contract()
    exchange_info, raw_books = v1._load_retained_books(contract)
    model = contract["execution_model"]
    prefilter = contract["optimistic_prefilter"]
    fee_rate = Decimal(str(model["conservative_taker_fee_rate_per_leg"]))
    one_leg_stress = Decimal(str(model["per_extra_leg_operational_stress_bips"]))
    three_leg_stress = one_leg_stress * 2
    maximum_residual_bips = Decimal(str(model["maximum_path_residual_bips"]))
    sizes = [Decimal(str(value)) for value in model["starting_usdt_sizes"]]
    edges = source.build_edges(exchange_info)
    books_by_sample = [source.parse_books(raw) for raw in raw_books]
    ordered_edges = [edges[key] for key in sorted(edges)]
    edge_indexes = {
        (edge.source, edge.target): index for index, edge in enumerate(ordered_edges)
    }
    assets = {asset for pair in edges for asset in pair}
    asset_rates = _asset_usdt_rates(assets, edges, books_by_sample, fee_rate)
    edge_rates = _edge_rate_matrix(ordered_edges, books_by_sample, fee_rate)
    direct_values = _direct_exact_values(
        edges, books_by_sample, asset_rates, sizes, fee_rate
    )
    survivors, route_count = _optimistic_survivors(
        routes=v1._three_leg_routes(edges),
        edge_indexes=edge_indexes,
        edge_rates=edge_rates,
        asset_rates=asset_rates,
        direct_values=direct_values,
        sizes=sizes,
        stress_bips=float(three_leg_stress),
        residual_upper_bips=float(maximum_residual_bips),
        numeric_padding_bips=float(prefilter["numeric_padding_bips"]),
        chunk_size=int(prefilter["route_chunk_size"]),
    )

    pairs_and_sizes = {
        (route[0].source, route[0].target, _decimal_text(size))
        for route, size in survivors
    }
    baselines = _best_shorter_baselines(
        pairs_and_sizes,
        edges=edges,
        books_by_sample=books_by_sample,
        asset_rates=asset_rates,
        fee_rate=fee_rate,
        one_leg_stress=one_leg_stress,
        maximum_residual_bips=maximum_residual_bips,
    )

    candidates: list[dict[str, object]] = []
    top_heap: list[tuple[float, str, str, dict[str, object]]] = []
    for (direct, first, second, third), size in survivors:
        source_asset = direct.source
        target_asset = direct.target
        size_text = _decimal_text(size)
        route_id = (
            f"{source_asset}->{first.target}->{second.target}->{target_asset}"
            "_vs_best_shorter"
        )
        increments: list[Decimal] = []
        capacity_count = 0
        residual_count = 0
        positive_indices: list[int] = []
        for sample_index, books in enumerate(books_by_sample):
            source_rate = Decimal(str(asset_rates[source_asset][sample_index]))
            target_rate = Decimal(str(asset_rates[target_asset][sample_index]))
            first_mid_rate = Decimal(str(asset_rates[first.target][sample_index]))
            second_mid_rate = Decimal(str(asset_rates[second.target][sample_index]))
            start_amount = size / source_rate
            direct_row = v1._route_value(
                start_amount,
                [direct],
                source_rate=source_rate,
                target_rate=target_rate,
                residual_rates=[],
                books=books,
                fee_rate=fee_rate,
            )
            row = v1._route_value(
                start_amount,
                [first, second, third],
                source_rate=source_rate,
                target_rate=target_rate,
                residual_rates=[first_mid_rate, second_mid_rate],
                books=books,
                fee_rate=fee_rate,
            )
            if direct_row is None or row is None or direct_row[0] <= 0:
                raise ValueError("optimistic survivor is not exactly evaluable")
            capacity_count += int(row[2])
            residual_bips = row[1] / size * TEN_THOUSAND
            residual_count += int(residual_bips <= maximum_residual_bips)
            three_leg_stressed = (
                row[0] / direct_row[0] - Decimal(1)
            ) * TEN_THOUSAND - three_leg_stress
            incremental = (
                three_leg_stressed
                - baselines[(source_asset, target_asset, size_text, sample_index)]
            )
            increments.append(incremental)
            if row[2] and residual_bips <= maximum_residual_bips and incremental > 0:
                positive_indices.append(sample_index)
        blocks = [
            sum(start <= index < start + 15 for index in positive_indices)
            for start in range(0, len(books_by_sample), 15)
        ]
        median = _median(increments)
        candidate = (
            capacity_count == len(books_by_sample)
            and residual_count == len(books_by_sample)
            and len(positive_indices) >= 48
            and len(blocks) == 4
            and all(count >= 12 for count in blocks)
            and median > 0
        )
        row = {
            "route_id": route_id,
            "source": source_asset,
            "intermediaries": [first.target, second.target],
            "target": target_asset,
            "direct_symbol": direct.symbol,
            "three_leg_symbols": [first.symbol, second.symbol, third.symbol],
            "start_usdt": size_text,
            "observations": len(increments),
            "capacity_ok_observations": capacity_count,
            "residual_gate_observations": residual_count,
            "positive_incremental_observations": len(positive_indices),
            "positive_counts_by_15_sample_block": blocks,
            "minimum_incremental_bips": _decimal_text(min(increments)),
            "median_incremental_bips": _decimal_text(median),
            "maximum_incremental_bips": _decimal_text(max(increments)),
            "empirical_candidate": candidate,
        }
        if candidate:
            candidates.append(row)
        item = (float(median), route_id, size_text, row)
        if len(top_heap) < 100:
            heapq.heappush(top_heap, item)
        elif item[:3] > top_heap[0][:3]:
            heapq.heapreplace(top_heap, item)

    top_rows = [item[3] for item in sorted(top_heap, reverse=True)]
    candidates.sort(
        key=lambda row: Decimal(str(row["median_incremental_bips"])), reverse=True
    )
    result_without_hash: dict[str, object] = {
        "schema_version": "binance-two-intermediary-conversion-retained-v3",
        "status": "completed_zero_network_retained_adjudication",
        "contract_sha256": contract_hash,
        "retained_source_result_sha256": contract["retained_source"]["result_sha256"],
        "sample_count": len(books_by_sample),
        "directed_edges": len(edges),
        "direct_vs_two_intermediary_routes": route_count,
        "evaluated_route_sizes": route_count * len(sizes),
        "optimistic_prefilter_survivor_route_sizes": len(survivors),
        "exact_empirical_candidate_count": len(candidates),
        "exact_empirical_candidates": candidates,
        "top_100_exact_survivors_by_median_incremental_bips": top_rows,
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
                    "optimistic_prefilter_survivor_route_sizes",
                    "exact_empirical_candidate_count",
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

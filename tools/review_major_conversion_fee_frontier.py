"""Retained major-asset direct/one-intermediary fee frontier, exploratory only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D
import hashlib
import itertools
import json
from pathlib import Path

from tools.capture_binance_triangle_window import PAIRS, screen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/major-conversion-frontier"


@dataclass(frozen=True, slots=True)
class FeeComparison:
    gross_incremental_bips: D
    net_incremental_bips: D
    uniform_fee_break_even_bips: D


def compare_rates(
    direct: D, first: D, second: D, direct_fee: D, first_fee: D, second_fee: D
) -> FeeComparison:
    """Compare received-asset proportional fees on identical source-to-target flow.

    This excludes BNB-paid/flat/minimum/tax-mixed fees, size, rounding and
    execution. A negative break-even fee means gross indirect output is worse.
    """
    if any(
        not isinstance(x, D) or not x.is_finite() or x <= 0
        for x in (direct, first, second)
    ):
        raise ValueError("finite positive conversion rates required")
    if any(
        not isinstance(x, D) or not x.is_finite() or not 0 <= x < 1
        for x in (direct_fee, first_fee, second_fee)
    ):
        raise ValueError("finite fractional received-output fees in [0,1) required")
    ratio = first * second / direct
    net_ratio = ratio * (1 - first_fee) * (1 - second_fee) / (1 - direct_fee)
    return FeeComparison(
        (ratio - 1) * 10000, (net_ratio - 1) * 10000, (1 - 1 / ratio) * 10000
    )


def routes(payload: bytes, fee_bips: tuple[str, ...]) -> list[dict]:
    """Enumerate every directed pair and each distinct one-intermediary route."""
    screen(payload)  # Reuse the frozen exact-six-symbol and finite-price validator.
    rates = {}
    for row in json.loads(payload):
        base, quote = PAIRS[row["symbol"]]
        rates[base, quote] = D(row["bidPrice"])
        rates[quote, base] = 1 / D(row["askPrice"])
    assets = sorted({x for pair in PAIRS.values() for x in pair})
    result = []
    for source, target, via in itertools.permutations(assets, 3):
        direct, first, second = (
            rates[source, target],
            rates[source, via],
            rates[via, target],
        )
        zero = compare_rates(direct, first, second, D(0), D(0), D(0))
        result.append(
            {
                "route": f"{source}->{via}->{target}",
                "direct": f"{source}->{target}",
                "direct_rate": str(direct),
                "first_rate": str(first),
                "second_rate": str(second),
                "gross_incremental_bips": str(zero.gross_incremental_bips),
                "uniform_fee_break_even_bips": str(zero.uniform_fee_break_even_bips),
                "net_incremental_bips_by_uniform_fee": {
                    f: str(
                        compare_rates(
                            direct, first, second, *([D(f) / 10000] * 3)
                        ).net_incremental_bips
                    )
                    for f in fee_bips
                },
            }
        )
    return result


def review() -> dict:
    """Read only exact frozen retained inputs and retain all comparisons."""
    plan = json.loads((BASE / "plan.json").read_bytes())
    for p, h in plan["source_sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            raise ValueError("source hash mismatch")
    rows = []
    for entry in plan["inputs"]:
        payload = (ROOT / entry["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("retained input differs")
        rows.extend(
            {"sample_index": entry["sample_index"], **row}
            for row in routes(payload, tuple(plan["uniform_fee_bips"]))
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["route"]].append(row)
    summaries = []
    for route, values in sorted(grouped.items()):
        if len(values) != 12:
            raise ValueError("incomplete route window")
        summaries.append(
            {
                "route": route,
                "minimum_uniform_fee_break_even_bips": str(
                    min(D(x["uniform_fee_break_even_bips"]) for x in values)
                ),
                "maximum_gross_incremental_bips": str(
                    max(D(x["gross_incremental_bips"]) for x in values)
                ),
                "positive_samples_by_uniform_fee": {
                    f: sum(
                        D(x["net_incremental_bips_by_uniform_fee"][f]) > 0
                        for x in values
                    )
                    for f in plan["uniform_fee_bips"]
                },
                "all_samples_above_three_bip_stress_by_uniform_fee": {
                    f: all(
                        D(x["net_incremental_bips_by_uniform_fee"][f]) > 3
                        for x in values
                    )
                    for f in plan["uniform_fee_bips"]
                },
            }
        )
    return {
        "schema_version": "retained-major-conversion-fee-frontier-v1",
        "classification": "post_capture_exploratory_no_promotion",
        "rows": rows,
        "route_summaries": summaries,
        "plan_file_sha256": hashlib.sha256(
            (BASE / "plan.json").read_bytes()
        ).hexdigest(),
        "accepted_edge": False,
        "network_requests": 0,
        "account_fees_qualified": False,
        "new_independent_validation": False,
        "original_results_changed": False,
    }


if __name__ == "__main__":
    result = review()
    with (BASE / "result.json").open("x", encoding="ascii", newline="\n") as output:
        json.dump(
            result, output, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        output.write("\n")
    print(
        json.dumps(
            {
                "rows": len(result["rows"]),
                "routes": len(result["route_summaries"]),
                "accepted_edge": False,
                "network_requests": 0,
            }
        )
    )

"""Audit every retained Polymarket binary book pair for mint-then-sell parity.

This runner is deliberately offline. It groups two distinct token books with the
same condition id, walks both bid sides at one fixed quantity, and rejects the
reverse-complete-set path unless combined proceeds exceed the mint cost before
fees. The retained population was discovered during an exploratory pass, so its
output is rejection-only and may never promote an edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOB = "data/polymarket-*/raw/books.json"
MECHANICS_SOURCE = Path(
    "docs/model-research/action-value/raw/polymarket-negrisk-maker-input-v2/"
    "ctf-exchange-v2-readme.raw"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _canonical_hash(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("result_sha256", None)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _walk_bids(book: dict[str, Any], quantity: Decimal) -> Decimal | None:
    remaining = quantity
    proceeds = Decimal("0")
    levels = sorted(
        book.get("bids", []),
        key=lambda level: Decimal(str(level["price"])),
        reverse=True,
    )
    for level in levels:
        if remaining <= 0:
            break
        available = Decimal(str(level["size"]))
        take = min(available, remaining)
        proceeds += take * Decimal(str(level["price"]))
        remaining -= take
    return proceeds if remaining == 0 else None


def _book_timestamp(book: dict[str, Any]) -> int:
    value = book.get("timestamp")
    if isinstance(value, bool):
        raise ValueError("book timestamp must not be boolean")
    return int(value)


def audit(*, quantity: Decimal, created_at_utc: str) -> dict[str, Any]:
    if not quantity.is_finite() or quantity <= 0:
        raise ValueError("quantity must be finite and positive")

    source_paths = sorted(ROOT.glob(SOURCE_GLOB))
    if not source_paths:
        raise ValueError(f"no retained inputs matched {SOURCE_GLOB}")

    observations: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    excluded_singleton_groups = 0
    eligible_capture_count = 0
    timestamps: list[int] = []

    for path in source_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} is not a book array")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for book in raw:
            if not isinstance(book, dict):
                raise ValueError(f"{path} contains a non-object book")
            groups[str(book["market"])].append(book)
            timestamps.append(_book_timestamp(book))

        file_rows: list[dict[str, Any]] = []
        singleton_count = 0
        for condition_id, books in sorted(groups.items()):
            asset_ids = {str(book["asset_id"]) for book in books}
            if len(books) != 2 or len(asset_ids) != 2:
                singleton_count += 1
                continue
            if any(Decimal(str(book["min_order_size"])) > quantity for book in books):
                proceeds = None
            else:
                leg_proceeds = [_walk_bids(book, quantity) for book in books]
                proceeds = (
                    leg_proceeds[0] + leg_proceeds[1]
                    if all(value is not None for value in leg_proceeds)
                    else None
                )
            timestamp_skew_ms = abs(
                _book_timestamp(books[0]) - _book_timestamp(books[1])
            )
            row = {
                "condition_id": condition_id,
                "gross_profit_before_fees_pUSD": (
                    _decimal_text(proceeds - quantity) if proceeds is not None else None
                ),
                "matched_proceeds_pUSD": (
                    _decimal_text(proceeds) if proceeds is not None else None
                ),
                "timestamp_skew_ms": timestamp_skew_ms,
            }
            file_rows.append(row)
            observations.append(
                {"source_path": path.relative_to(ROOT).as_posix(), **row}
            )

        if file_rows:
            eligible_capture_count += 1
        excluded_singleton_groups += singleton_count
        complete = [
            row for row in file_rows if row["gross_profit_before_fees_pUSD"] is not None
        ]
        profits = [
            Decimal(str(row["gross_profit_before_fees_pUSD"])) for row in complete
        ]
        source_manifest.append(
            {
                "book_count": len(raw),
                "eligible_binary_pair_count": len(file_rows),
                "excluded_nonpair_condition_group_count": singleton_count,
                "finite_size_complete_count": len(complete),
                "gross_positive_count": sum(value > 0 for value in profits),
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
            }
        )

    complete_observations = [
        row
        for row in observations
        if row["gross_profit_before_fees_pUSD"] is not None
    ]
    profits = [
        Decimal(str(row["gross_profit_before_fees_pUSD"]))
        for row in complete_observations
    ]
    closest = sorted(
        complete_observations,
        key=lambda row: (
            -Decimal(str(row["gross_profit_before_fees_pUSD"])),
            row["source_path"],
            row["condition_id"],
        ),
    )[:10]
    unique_conditions = {row["condition_id"] for row in observations}
    mechanics_path = ROOT / MECHANICS_SOURCE
    implementation_path = Path(__file__).resolve()

    result: dict[str, Any] = {
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "reason": (
                "zero of the exhaustive retained finite-size pair observations "
                "was gross-positive before fees or any external cost"
            ),
            "status": "terminal_rejected_retained_population_no_gross_positive_path",
            "trading_authority": False,
        },
        "authority": {
            "account_accessed": False,
            "credentials_used": False,
            "network_requests": 0,
            "orders_or_funds_mutated": False,
        },
        "created_at_utc": created_at_utc,
        "economics": {
            "mechanism": "mint_one_complete_binary_set_then_sell_both_legs_to_bids",
            "mint_cost_pUSD_per_share_pair": "1",
            "quantity_shares_per_leg": _decimal_text(quantity),
            "screen_rule": (
                "sum exact descending bid-depth proceeds for both complementary "
                "tokens minus one pUSD per complete set; fees are omitted only as "
                "an optimistic rejection bound"
            ),
        },
        "limitations": [
            "The retained files were captured for other hypotheses and are not a prospectively selected recurrence population.",
            "The exploratory pass exposed outcomes before this canonical audit was written, so this result is rejection-only and cannot promote an edge.",
            "A finite retained sample cannot prove that a transient operator or transport anomaly never occurs.",
            "Public books do not prove own fills; a positive row would still require freshness, atomicity, fees, capacity, relayer, and realized-cash validation.",
        ],
        "implementation": {
            "path": implementation_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(implementation_path),
        },
        "mechanics_source": {
            "interpretation": (
                "the retained official CTF Exchange V2 README defines both-buy "
                "matches as MINT and both-sell matches as MERGE, making persistent "
                "crossed complementary books structurally self-clearing"
            ),
            "path": MECHANICS_SOURCE.as_posix(),
            "sha256": _sha256(mechanics_path),
        },
        "result_sha256": "",
        "retry_policy": {
            "do_not_repeat": True,
            "literal_trigger": (
                "material official matching or settlement architecture change, "
                "or independently frozen source-continuous evidence of a persistent "
                "finite-size combined bid above one after every fee and external cost"
            ),
        },
        "schema_version": "polymarket-binary-reverse-complete-set-retained-audit-v1",
        "selection_integrity": {
            "promotion_eligible": False,
            "role": "outcome_aware_exhaustive_retained_population_rejection_only",
            "source_glob": SOURCE_GLOB,
        },
        "source_manifest": source_manifest,
        "summary": {
            "book_count": sum(row["book_count"] for row in source_manifest),
            "capture_file_count": len(source_manifest),
            "capture_files_with_eligible_pairs": eligible_capture_count,
            "closest_gross_profit_before_fees_pUSD": _decimal_text(max(profits)),
            "closest_observations": closest,
            "excluded_nonpair_condition_group_count": excluded_singleton_groups,
            "finite_size_complete_observation_count": len(complete_observations),
            "gross_flat_observation_count": sum(value == 0 for value in profits),
            "gross_negative_observation_count": sum(value < 0 for value in profits),
            "gross_positive_observation_count": sum(value > 0 for value in profits),
            "insufficient_depth_observation_count": len(observations)
            - len(complete_observations),
            "median_gross_profit_before_fees_pUSD": _decimal_text(median(profits)),
            "paired_observation_count": len(observations),
            "raw_book_timestamp_max_ms": max(timestamps),
            "raw_book_timestamp_min_ms": min(timestamps),
            "unique_condition_count": len(unique_conditions),
        },
    }
    result["result_sha256"] = _canonical_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantity", default="5")
    args = parser.parse_args()
    result = audit(
        quantity=Decimal(str(args.quantity)), created_at_utc=args.created_at_utc
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

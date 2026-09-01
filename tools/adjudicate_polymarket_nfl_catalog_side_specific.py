from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _json_ready,
    _root_path,
    _sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-nfl-catalog-side-specific-adjudication-v1"
ONE = Decimal("1")


def _canonical_rows_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        _json_ready(rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _json_pair(value: Any, name: str) -> list[str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise RuntimeError(f"{name} must contain exactly two outcomes")
    return [str(item) for item in parsed]


def _validate_probability(value: Any, name: str) -> Decimal:
    if value is None:
        raise RuntimeError(f"{name} is missing")
    parsed = Decimal(str(value))
    if not Decimal("0") <= parsed <= ONE:
        raise RuntimeError(f"{name} is outside [0, 1]")
    return parsed


def _side_specific_price(market: dict[str, Any], outcome: str) -> tuple[Decimal, str]:
    outcomes = _json_pair(market.get("outcomes"), "market outcomes")
    if outcome == outcomes[0]:
        return _validate_probability(market.get("bestAsk"), "bestAsk"), "bestAsk"
    if outcome == outcomes[1]:
        return ONE - _validate_probability(market.get("bestBid"), "bestBid"), "1-bestBid"
    raise RuntimeError(f"outcome {outcome!r} is absent from market {market.get('id')}")


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    expected_authority = {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "new_network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }
    if contract["authority"] != expected_authority:
        raise RuntimeError("offline authority boundary changed")
    for source in contract["sources"]:
        path = _root_path(str(source["path"]))
        if _sha256(path.read_bytes()) != source["file_sha256"]:
            raise RuntimeError(f"source hash mismatch: {path.name}")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def adjudicate(contract: dict[str, Any]) -> dict[str, Any]:
    source_result_path = _root_path(str(contract["source_result_path"]))
    raw_path = _root_path(str(contract["raw_path"]))
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if _canonical_hash(source, "result_sha256") != source.get("result_sha256"):
        raise RuntimeError("source result canonical hash mismatch")
    if source["result_sha256"] != contract["source_result_sha256"]:
        raise RuntimeError("source result identity changed")
    relations = source.get("screen", {}).get("relations")
    if not isinstance(relations, list) or not relations:
        raise RuntimeError("source contains no proved relations")
    events = raw.get("events") if isinstance(raw, dict) else None
    if not isinstance(events, list):
        raise RuntimeError("raw catalog response shape changed")

    markets: dict[str, dict[str, Any]] = {}
    for event in events:
        for market in event.get("markets", []):
            market_id = str(market.get("id"))
            if market_id in markets:
                raise RuntimeError(f"duplicate market id {market_id}")
            markets[market_id] = market

    corrected: list[dict[str, Any]] = []
    price_incomplete: list[dict[str, Any]] = []
    for relation in relations:
        superset_id = str(relation["superset_positive_market_id"])
        subset_id = str(relation["subset_complement_market_id"])
        try:
            superset_market = markets[superset_id]
            subset_market = markets[subset_id]
        except KeyError as exc:
            raise RuntimeError(f"relation references missing market {exc.args[0]}") from exc
        try:
            superset_price, superset_source = _side_specific_price(
                superset_market, str(relation["superset_positive_outcome"])
            )
            subset_price, subset_source = _side_specific_price(
                subset_market, str(relation["subset_complement_outcome"])
            )
        except RuntimeError as exc:
            price_incomplete.append(
                {
                    "event_slug": relation["event_slug"],
                    "family": relation["family"],
                    "superset_positive_market_id": superset_id,
                    "superset_positive_outcome": relation[
                        "superset_positive_outcome"
                    ],
                    "subset_complement_market_id": subset_id,
                    "subset_complement_outcome": relation[
                        "subset_complement_outcome"
                    ],
                    "reason": str(exc),
                }
            )
            continue
        minimum_payout = Decimal(
            str(relation["minimum_terminal_payout_per_share_pUSD"])
        )
        side_sum = superset_price + subset_price
        corrected.append(
            {
                "event_slug": relation["event_slug"],
                "start_time_utc": relation["start_time_utc"],
                "family": relation["family"],
                "superset_threshold": relation["superset_threshold"],
                "superset_positive_market_id": superset_id,
                "superset_positive_outcome": relation["superset_positive_outcome"],
                "superset_side_specific_price_pUSD": superset_price,
                "superset_price_source": superset_source,
                "subset_threshold": relation["subset_threshold"],
                "subset_complement_market_id": subset_id,
                "subset_complement_outcome": relation["subset_complement_outcome"],
                "subset_side_specific_price_pUSD": subset_price,
                "subset_price_source": subset_source,
                "minimum_terminal_payout_per_share_pUSD": minimum_payout,
                "side_specific_rejection_sum_pUSD": side_sum,
                "optimistic_headroom_before_execution_costs_pUSD": (
                    minimum_payout - side_sum
                ),
                "passes_strict_side_specific_rejection_gate": side_sum
                < minimum_payout,
                "source_displayed_price_sum_per_share_pUSD": relation[
                    "displayed_price_sum_per_share_pUSD"
                ],
            }
        )

    ranked = sorted(
        corrected,
        key=lambda row: (
            Decimal(str(row["side_specific_rejection_sum_pUSD"])),
            row["start_time_utc"],
            row["event_slug"],
            row["family"],
            int(row["superset_threshold"]),
            int(row["subset_threshold"]),
        ),
    )
    strict = [row for row in ranked if row["passes_strict_side_specific_rejection_gate"]]
    event_family_summaries = []
    groups = sorted({(row["event_slug"], row["family"]) for row in ranked})
    for event_slug, family in groups:
        group = [
            row
            for row in ranked
            if row["event_slug"] == event_slug and row["family"] == family
        ]
        event_family_summaries.append(
            {
                "event_slug": event_slug,
                "family": family,
                "relation_count": len(group),
                "strict_side_specific_subfloor_count": sum(
                    bool(row["passes_strict_side_specific_rejection_gate"])
                    for row in group
                ),
                "best_side_specific_relation": group[0],
            }
        )
    source_midpoint_count = int(
        source["screen"]["candidate_count_strictly_below_payout_floor"]
    )
    price_complete = not price_incomplete
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": contract["frozen_at_utc"],
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "source_binding": {
            "catalog_result_path": contract["source_result_path"],
            "catalog_result_sha256": source["result_sha256"],
            "raw_catalog_path": contract["raw_path"],
            "raw_catalog_file_sha256": contract["raw_file_sha256"],
            "returned_event_count": source["capture"]["returned_event_count"],
            "included_event_count": source["screen"]["included_event_count"],
            "population_complete_under_frozen_filter": source["capture"][
                "population_complete_under_frozen_filter"
            ],
        },
        "supersession": {
            "preserved_source_artifact": True,
            "source_payoff_relations_remain_valid": True,
            "source_midpoint_like_acquisition_gate_is_superseded": True,
            "reason": (
                "outcomePrices are midpoint-like diagnostics; executable acquisition "
                "rejection must use first-outcome bestAsk or conservative 1-bestBid"
            ),
        },
        "screen": {
            "complete_relation_count": len(corrected),
            "price_complete_relation_count": len(corrected),
            "source_proved_relation_count": len(relations),
            "price_incomplete_relation_count": len(price_incomplete),
            "price_incomplete_relations": price_incomplete,
            "source_midpoint_like_strict_subfloor_count": source_midpoint_count,
            "strict_side_specific_subfloor_count": len(strict),
            "all_side_specific_sums_at_or_above_payout_floor": (
                not strict and price_complete
            ),
            "complete_relations_sha256": _canonical_rows_hash(ranked),
            "price_incomplete_relations_sha256": _canonical_rows_hash(
                price_incomplete
            ),
            "best_side_specific_relation": ranked[0] if ranked else None,
            "event_family_summaries": event_family_summaries,
            "price_gate": (
                "first outcome bestAsk; second outcome conservative 1-bestBid; "
                "Gamma fields are rejection-only"
            ),
        },
        "adjudication": {
            "status": (
                "retained_catalog_price_incomplete_no_depth_escalation"
                if not price_complete
                else (
                    "strict_side_specific_candidate_requires_separate_depth_contract"
                    if strict
                    else "complete_retained_catalog_rejected_before_books_and_fees"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "book_or_fee_request_permitted": False,
            "next_action": (
                "do not refetch, substitute outcomePrices, or select a depth candidate "
                "from the price-incomplete population; wait for a distinct trigger"
                if not price_complete
                else (
                    "freeze one exact depth screen for only the strongest strict "
                    "side-specific candidate"
                    if strict
                    else "do not repeat, reprice, or book-capture the consumed NFL "
                    "catalog; wait for the registered future-distinct-event trigger"
                )
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementations"][0],
    }
    if price_complete:
        for field in (
            "source_proved_relation_count",
            "price_incomplete_relation_count",
            "price_incomplete_relations",
            "price_incomplete_relations_sha256",
        ):
            result["screen"].pop(field)
    serializable = _json_ready(result)
    serializable["result_sha256"] = _canonical_hash(serializable, "result_sha256")
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correct one retained NFL catalog with exhaustive side-specific "
            "Gamma rejection prices and no network access."
        )
    )
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = _root_path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract, contract_path)
    result = adjudicate(contract)
    output_path = _root_path(str(contract["output_path"]))
    if output_path.exists():
        raise RuntimeError("adjudication output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    best = result["screen"]["best_side_specific_relation"]
    print(
        json.dumps(
            {
                "complete_relation_count": result["screen"]["complete_relation_count"],
                "source_midpoint_like_subfloor_count": result["screen"][
                    "source_midpoint_like_strict_subfloor_count"
                ],
                "strict_side_specific_subfloor_count": result["screen"][
                    "strict_side_specific_subfloor_count"
                ],
                "best_side_specific_sum_pUSD": (
                    best["side_specific_rejection_sum_pUSD"] if best else None
                ),
                "price_incomplete_relation_count": result["screen"][
                    "price_incomplete_relation_count"
                ]
                if "price_incomplete_relation_count" in result["screen"]
                else 0,
                "new_network_requests": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

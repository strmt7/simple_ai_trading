"""Screen named-company second/third rank mutual exclusions from retained Gamma."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.adjudicate_polymarket_release_date_deadline_graph import (
    ONE,
    _acquisition,
    _decimal_text,
)
from tools.screen_polymarket_exact_crypto_threshold_ladder_v2 import _list, _mapping


CONTRACT_SCHEMA = "polymarket-retained-rank-mutual-exclusion-contract-v1"
RESULT_SCHEMA = "polymarket-retained-rank-mutual-exclusion-result-v1"
_RANK_WORD = re.compile(r"\b(?:second|third)\b", re.IGNORECASE)


def _load_population(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name="population")


def _select_event(population: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    events = population.get("events")
    if not isinstance(events, list):
        raise RuntimeError("retained population events must be a list")
    matches = [
        _mapping(value, name="retained event")
        for value in events
        if isinstance(value, Mapping) and str(value.get("id")) == event_id
    ]
    if len(matches) != 1:
        raise RuntimeError("exact retained event must occur once")
    return matches[0]


def _normalized_rank_rules(value: object) -> str:
    text = str(value or "")
    return _RANK_WORD.sub("{rank}", text)


def _market_map(event: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    markets = event.get("markets")
    if not isinstance(markets, list):
        raise RuntimeError("exact event markets must be a list")
    by_label: dict[str, dict[str, Any]] = {}
    for value in markets:
        market = _mapping(value, name="exact event market")
        label = market.get("groupItemTitle")
        if not isinstance(label, str) or not label:
            raise RuntimeError("market label is missing")
        if label in by_label:
            raise RuntimeError("duplicate market label")
        by_label[label] = market
    return by_label


def _preflight_family(
    population: Mapping[str, Any], family: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    second = _select_event(population, str(family["second_event_id"]))
    third = _select_event(population, str(family["third_event_id"]))
    for role, event in (("second", second), ("third", third)):
        if event.get("slug") != family[f"{role}_event_slug"]:
            raise RuntimeError(f"{role} event slug changed")
        if event.get("title") != family[f"{role}_event_title"]:
            raise RuntimeError(f"{role} event title changed")
    second_rules = _normalized_rank_rules(second.get("description"))
    third_rules = _normalized_rank_rules(third.get("description"))
    if second_rules != third_rules:
        raise RuntimeError("second and third rank rules differ beyond ordinal")
    if not all(
        fragment in second_rules for fragment in family["required_rule_fragments"]
    ):
        raise RuntimeError("required deterministic rank rule changed")

    second_expected = family["expected_all_labels"]
    third_expected = family.get("third_expected_all_labels", second_expected)
    named = family["named_identity_labels"]
    if (
        not isinstance(second_expected, list)
        or not isinstance(third_expected, list)
        or not isinstance(named, list)
        or len(named) < 1
        or len(named) >= len(second_expected)
        or len(named) >= len(third_expected)
        or len(set(second_expected)) != len(second_expected)
        or len(set(third_expected)) != len(third_expected)
        or len(set(named)) != len(named)
        or any(label not in second_expected for label in named)
        or any(label not in third_expected for label in named)
        or any(label == "Other" or str(label).startswith("Company ") for label in named)
    ):
        raise RuntimeError("frozen label boundary is invalid")

    for event, expected in ((second, second_expected), (third, third_expected)):
        by_label = _market_map(event)
        if list(by_label) != expected:
            raise RuntimeError("exact event label population changed")
        for label, market in by_label.items():
            outcomes = _list(market.get("outcomes"), name="outcomes")
            tokens = _list(market.get("clobTokenIds"), name="CLOB token IDs")
            if outcomes != ["Yes", "No"] or len(tokens) != 2:
                raise RuntimeError("binary market representation changed")
            if _normalized_rank_rules(market.get("description")) != second_rules:
                raise RuntimeError("market-level rank rules differ")
            if label in named and not (
                market.get("active") is True
                and market.get("closed") is False
                and market.get("acceptingOrders") is True
                and market.get("enableOrderBook") is True
            ):
                raise RuntimeError("selected named rank market is not executable")
            for field in ("bestAsk", "bestBid"):
                raw = market.get(field)
                if raw is not None and not isinstance(raw, (str, int, float)):
                    raise RuntimeError(f"{field} representation changed")
    return second, third


def _package_row(
    family_name: str,
    label: str,
    second_market: dict[str, Any],
    third_market: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "family": family_name,
        "identity_label": label,
        "second_market_id": str(second_market.get("id")),
        "third_market_id": str(third_market.get("id")),
        "package": "NO(second rank) plus NO(third rank)",
        "guaranteed_payout_floor_pUSD_per_share": "1",
    }
    second = _acquisition(second_market, outcome="No")
    third = _acquisition(third_market, outcome="No")
    if second is None or third is None:
        return {
            **identity,
            "status": "missing_side_specific_acquisition_evidence",
            "passes_strict_metadata_gate": False,
            "passes_fee_and_one_tick_gate": False,
        }
    quantity = max(second["minimum_order_shares"], third["minimum_order_shares"])
    actual_prices = [second["price_pUSD_per_share"], third["price_pUSD_per_share"]]
    stressed_prices = [
        second["price_pUSD_per_share"] + second["tick_size_pUSD"],
        third["price_pUSD_per_share"] + third["tick_size_pUSD"],
    ]
    actual_sum = sum(actual_prices, Decimal("0"))
    gross_headroom = quantity * (ONE - actual_sum)
    if any(price >= ONE for price in stressed_prices):
        stressed_fee = None
        stressed_headroom = None
    else:
        stressed_fee = second["fee_model"](
            stressed_prices[0], quantity, "taker"
        ) + third["fee_model"](stressed_prices[1], quantity, "taker")
        stressed_headroom = (
            quantity * (ONE - sum(stressed_prices, Decimal("0"))) - stressed_fee
        )
    return {
        **identity,
        "status": "priced",
        "quantity_shares_each_leg": _decimal_text(quantity),
        "legs": [
            {
                key: (_decimal_text(value) if isinstance(value, Decimal) else value)
                for key, value in leg.items()
                if key != "fee_model"
            }
            for leg in (second, third)
        ],
        "metadata_cost_pUSD_per_share": _decimal_text(actual_sum),
        "metadata_gross_headroom_pUSD": _decimal_text(gross_headroom),
        "one_adverse_tick_per_leg_prices_pUSD": [
            _decimal_text(value) for value in stressed_prices
        ],
        "stressed_taker_fee_pUSD": _decimal_text(stressed_fee),
        "after_fee_one_tick_profit_floor_pUSD": _decimal_text(stressed_headroom),
        "passes_strict_metadata_gate": actual_sum < ONE,
        "passes_fee_and_one_tick_gate": (
            stressed_headroom is not None and stressed_headroom > 0
        ),
    }


def _screen(
    population: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in contract["families"]:
        family = _mapping(value, name="frozen rank family")
        second, third = _preflight_family(population, family)
        second_by_label = _market_map(second)
        third_by_label = _market_map(third)
        for label in family["named_identity_labels"]:
            rows.append(
                _package_row(
                    str(family["family"]),
                    str(label),
                    second_by_label[str(label)],
                    third_by_label[str(label)],
                )
            )
    rows.sort(
        key=lambda row: (
            row["status"] != "priced",
            Decimal(row["metadata_cost_pUSD_per_share"])
            if row["status"] == "priced"
            else Decimal("Infinity"),
            row["family"],
            row["identity_label"],
        )
    )
    return rows


def _validate_contract(
    contract: Mapping[str, Any], path: Path, *, preflight_only: bool
) -> tuple[Path, dict[str, Any]]:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_one_zero_network_retained_adjudication"
    )
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unexpected contract schema")
    if contract.get("status") != expected_status:
        raise RuntimeError("contract status does not match invocation mode")
    if _canonical_hash(dict(contract), "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    retained = _root_path(str(contract["retained_input"]["path"]))
    if _sha256(retained.read_bytes()) != contract["retained_input"]["sha256"]:
        raise RuntimeError("retained input hash mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "network_requests": 0,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    families = contract.get("families")
    if not isinstance(families, list) or len(families) != 4:
        raise RuntimeError("exact four-family population changed")
    population = _load_population(retained)
    for family in families:
        _preflight_family(population, _mapping(family, name="frozen rank family"))
    return retained, population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    retained, population = _validate_contract(
        contract, contract_path, preflight_only=args.preflight_only
    )
    if args.preflight_only:
        print("preflight_passed=true")
        return

    result_path = _root_path(str(contract["result_path"]))
    if result_path.exists():
        raise RuntimeError("one-use result already exists")
    rows = _screen(population, contract)
    priced = [row for row in rows if row["status"] == "priced"]
    metadata_candidates = [row for row in rows if row["passes_strict_metadata_gate"]]
    stressed_candidates = [row for row in rows if row["passes_fee_and_one_tick_gate"]]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "retained_input": {
            "path": contract["retained_input"]["path"],
            "sha256": _sha256(retained.read_bytes()),
        },
        "payoff_identity": contract["payoff_identity"],
        "screen": {
            "family_count": len(contract["families"]),
            "package_count": len(rows),
            "side_specific_price_available_count": len(priced),
            "strict_metadata_candidate_count": len(metadata_candidates),
            "fee_and_one_tick_candidate_count": len(stressed_candidates),
            "packages_ranked_by_side_specific_sum": rows,
            "best_package": priced[0] if priced else None,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "candidate_requires_separately_frozen_exact_depth_batch"
                if stressed_candidates
                else "rejected_before_books_and_onchain_requests"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "family_count": len(contract["families"]),
                "package_count": len(rows),
                "side_specific_price_available_count": len(priced),
                "best_displayed_sum_pUSD": (
                    priced[0]["metadata_cost_pUSD_per_share"] if priced else None
                ),
                "strict_metadata_candidate_count": len(metadata_candidates),
                "fee_and_one_tick_candidate_count": len(stressed_candidates),
                "network_requests": 0,
                "payloads_printed": 0,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

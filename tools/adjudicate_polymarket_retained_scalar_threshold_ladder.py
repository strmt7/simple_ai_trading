"""Screen one exact scalar threshold ladder from hash-bound retained Gamma bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_crypto_threshold_ladder_v2 import (
    _list,
    _mapping,
    _screen_event,
    _threshold,
)


CONTRACT_SCHEMA = "polymarket-retained-scalar-threshold-ladder-contract-v1"
RESULT_SCHEMA = "polymarket-retained-scalar-threshold-ladder-result-v1"


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


def _normalized_event(event: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    markets = event.get("markets")
    if not isinstance(markets, list):
        raise RuntimeError("exact event markets must be a list")
    normalized_markets: list[dict[str, Any]] = []
    for value in markets:
        market = _mapping(value, name="exact event market")
        label = market.get("groupItemTitle")
        if isinstance(label, str) and label.strip().endswith("+"):
            market["groupItemTitle"] = label.strip()[:-1].strip()
        normalized_markets.append(market)
    normalized["markets"] = normalized_markets
    return normalized


def _preflight_event_schema(
    event: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    if str(event.get("id")) != contract.get("event_id"):
        raise RuntimeError("exact event ID changed")
    if event.get("slug") != contract.get("event_slug"):
        raise RuntimeError("exact event slug changed")
    if event.get("title") != contract.get("event_title"):
        raise RuntimeError("exact event title changed")
    markets = _normalized_event(event).get("markets")
    if not isinstance(markets, list):
        raise RuntimeError("exact event markets must be a list")
    expected = [str(value) for value in contract["expected_thresholds"]]
    observed: list[str] = []
    for value in markets:
        market = _mapping(value, name="exact event market")
        observed.append(format(_threshold(market.get("groupItemTitle")), "f"))
        outcomes = _list(market.get("outcomes"), name="outcomes")
        tokens = _list(market.get("clobTokenIds"), name="CLOB token IDs")
        if outcomes != ["Yes", "No"] or len(tokens) != 2:
            raise RuntimeError("binary market representation changed")
        for field in ("bestAsk", "bestBid"):
            raw = market.get(field)
            if raw is not None and not isinstance(raw, (str, int, float)):
                raise RuntimeError(f"{field} representation changed")
    if sorted(observed, key=int) != expected:
        raise RuntimeError("exact threshold population changed")


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
    population = _load_population(retained)
    event = _select_event(population, str(contract["event_id"]))
    _preflight_event_schema(event, contract)
    return retained, event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    retained, event = _validate_contract(
        contract, contract_path, preflight_only=args.preflight_only
    )
    if args.preflight_only:
        print("preflight_passed=true")
        return

    result_path = _root_path(str(contract["result_path"]))
    if result_path.exists():
        raise RuntimeError("one-use result already exists")
    legs, packages = _screen_event(_normalized_event(event), contract)
    candidates = [row for row in packages if row["passes_strict_displayed_gross_gate"]]
    available = [
        row for row in packages if row["side_specific_rejection_price_available"]
    ]
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
            "event_id": contract["event_id"],
            "event_slug": contract["event_slug"],
            "market_count": len(legs),
            "package_count": len(packages),
            "side_specific_price_available_count": len(available),
            "legs": legs,
            "packages_ranked_by_side_specific_sum": packages,
            "strict_displayed_candidate_count": len(candidates),
            "best_package": available[0] if available else None,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_separate_exact_depth_and_fee_proof"
                if candidates
                else "rejected_before_books_fees_and_onchain_requests"
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
                "market_count": len(legs),
                "package_count": len(packages),
                "side_specific_price_available_count": len(available),
                "best_displayed_sum_pUSD": (
                    available[0]["displayed_price_sum_pUSD"] if available else None
                ),
                "strict_displayed_candidate_count": len(candidates),
                "network_requests": 0,
                "payloads_printed": 0,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

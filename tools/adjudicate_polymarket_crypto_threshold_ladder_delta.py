"""Screen all newly deployed crypto threshold ladders from one retained delta."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_crypto_range_threshold_delta import (
    _list,
    _load,
    _mapping,
    _utc,
    _validate_capture,
)
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_crypto_threshold_ladder import _screen_event


SCHEMA = "polymarket-crypto-threshold-ladder-delta-result-v1"


def run(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen = _utc(contract["frozen_at_utc"])
    cutoff = _utc(contract["delta_cutoff_utc"])
    if cutoff >= frozen or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen or cutoff timestamp is invalid")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if contract.get("authority") != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "trading_authority": False,
    }:
        raise RuntimeError("offline authority boundary changed")

    payload, receipt = _validate_capture(contract)
    events = [
        _mapping(value, name="event")
        for value in _list(payload.get("events"), name="events")
    ]
    by_slug = {str(event.get("slug")): event for event in events}
    if len(by_slug) != len(events):
        raise RuntimeError("duplicate retained event slug")

    legs: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    ladder_summaries: list[dict[str, Any]] = []
    for ladder in contract["ladders"]:
        asset = str(ladder["asset"])
        event = by_slug[str(ladder["event_slug"])]
        if not (
            _utc(event.get("createdAt")) > cutoff
            and event.get("active") is True
            and event.get("closed") is False
            and event.get("negRisk") is False
        ):
            raise RuntimeError("retained threshold event state changed")
        event_legs, event_packages = _screen_event(event, ladder)
        legs.extend({"asset": asset, **row} for row in event_legs)
        packages.extend({"asset": asset, **row} for row in event_packages)
        ladder_summaries.append(
            {
                "asset": asset,
                "event_slug": ladder["event_slug"],
                "market_count": len(event_legs),
                "package_count": len(event_packages),
            }
        )

    packages.sort(
        key=lambda row: (
            Decimal(row["displayed_price_sum_pUSD"]),
            row["asset"],
            Decimal(row["lower_threshold"]),
            Decimal(row["higher_threshold"]),
        )
    )
    candidates = [
        row for row in packages if row["passes_strict_displayed_gross_gate"]
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "retained_capture": {
            "raw_sha256": contract["retained_source"]["raw_sha256"],
            "receipt": receipt,
            "delta_cutoff_utc": contract["delta_cutoff_utc"],
            "network_requests_added_by_this_screen": 0,
        },
        "payoff_identity": contract["payoff_identity"],
        "screen": {
            "ladder_summaries": ladder_summaries,
            "market_count": len(legs),
            "package_count": len(packages),
            "legs": legs,
            "packages_ranked_by_displayed_sum": packages,
            "strict_displayed_candidate_count": len(candidates),
            "best_package": packages[0],
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_one_separately_frozen_two_token_depth_batch"
                if candidates
                else "complete_new_ladder_delta_rejected_before_books_fees_and_accounts"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    output = _root_path(str(contract["output_path"]))
    if output.exists():
        raise RuntimeError("one-use output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract)
    print(
        json.dumps(
            {
                "market_count": result["screen"]["market_count"],
                "package_count": result["screen"]["package_count"],
                "best_displayed_sum_pUSD": result["screen"]["best_package"][
                    "displayed_price_sum_pUSD"
                ],
                "strict_displayed_candidate_count": result["screen"][
                    "strict_displayed_candidate_count"
                ],
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

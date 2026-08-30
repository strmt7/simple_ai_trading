"""Screen one source-selected fixed-NegRisk event before any book request."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request
from tools.screen_polymarket_negrisk_complete_set_catalog import (
    _eligible_event,
    _screen_event,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polymarket-exact-negrisk-event-prefilter-result-v1"


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an explicit UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is missing a timezone or is in the future")
    slug = contract.get("event_slug")
    if not isinstance(slug, str) or not slug:
        raise RuntimeError("event slug is missing")
    if contract.get("expected_market_count") != 10:
        raise RuntimeError("expected market count changed")
    expected_url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
    if contract.get("request") != {
        "body_sha256": _sha256(b""),
        "count": 1,
        "method": "GET",
        "url": expected_url,
    }:
        raise RuntimeError("request boundary changed")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_requests": 0,
        "credentials_used": False,
        "fee_requests": 0,
        "funds_used": False,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(str(implementation["path"]))
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def _conversion_rows(event: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    screened = _screen_event(event)
    total = Decimal(screened["displayed_all_yes_sum_pUSD"])
    rows: list[dict[str, Any]] = []
    for leg in screened["legs"]:
        yes = Decimal(leg["yes_price_pUSD"])
        no = Decimal(leg["no_price_pUSD"])
        other_yes = total - yes
        rows.append(
            {
                "source_market_id": leg["gamma_market_id"],
                "source_label": leg["label"],
                "displayed_no_input_pUSD": format(no, "f"),
                "displayed_other_yes_output_sum_pUSD": format(other_yes, "f"),
                "optimistic_displayed_conversion_gap_pUSD": format(
                    other_yes - no, "f"
                ),
                "passes_strict_positive_displayed_gap": other_yes > no,
            }
        )
    return screened, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    raw_path = _root_path(str(contract["outputs"]["raw_path"]))
    journal_path = _root_path(str(contract["outputs"]["journal_path"]))
    result_path = _root_path(str(contract["outputs"]["result_path"]))
    for path in (raw_path, journal_path, result_path):
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    raw, receipt = _request(
        method="GET",
        url=str(contract["request"]["url"]),
        body=b"",
        name="source-selected-exact-negrisk-event",
        raw_path=raw_path,
        raw_relative_path=str(contract["outputs"]["raw_path"]),
        journal_path=journal_path,
    )
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise RuntimeError("exact event response must be an object")
    if event.get("slug") != contract["event_slug"]:
        raise RuntimeError("exact event slug mismatch")

    fixed_negrisk = _eligible_event(event)
    screened: dict[str, Any] | None = None
    conversions: list[dict[str, Any]] = []
    if fixed_negrisk:
        screened, conversions = _conversion_rows(event)
        if screened["market_count"] != contract["expected_market_count"]:
            raise RuntimeError("exact event market count changed")
    all_yes_candidate = bool(
        screened is not None and screened["passes_strictly_below_payout_gate"]
    )
    conversion_candidate_count = sum(
        bool(row["passes_strict_positive_displayed_gap"]) for row in conversions
    )
    source_only_candidate = all_yes_candidate or conversion_candidate_count > 0
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "event_slug": contract["event_slug"],
            "fixed_negrisk": fixed_negrisk,
        },
        "screen": {
            "event": screened,
            "one_no_to_other_yes_displayed_identities": conversions,
            "all_yes_candidate": all_yes_candidate,
            "positive_displayed_conversion_candidate_count": conversion_candidate_count,
            "source_only_candidate": source_only_candidate,
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "not_fixed_negrisk_rejected"
                if not fixed_negrisk
                else (
                    "source_only_candidate_requires_separate_exact_depth_fee_and_conversion_proof"
                    if source_only_candidate
                    else "rejected_before_books_fees_and_onchain_requests"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_exact_negrisk_event.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "fixed_negrisk": fixed_negrisk,
                "market_count": screened["market_count"] if screened else 0,
                "displayed_all_yes_sum_pUSD": (
                    screened["displayed_all_yes_sum_pUSD"] if screened else None
                ),
                "all_yes_candidate": all_yes_candidate,
                "positive_displayed_conversion_candidate_count": conversion_candidate_count,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

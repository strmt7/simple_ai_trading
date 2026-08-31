"""Screen one exact crypto threshold ladder with side-specific Gamma prices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Mapping

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request


CONTRACT_SCHEMA = "polymarket-exact-crypto-threshold-ladder-prefilter-contract-v2"
RESULT_SCHEMA = "polymarket-exact-crypto-threshold-ladder-prefilter-result-v2"


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise RuntimeError(f"{name} must be a list")
    return parsed


def _threshold(value: object) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("group item title must contain a threshold")
    try:
        parsed = Decimal(value.replace(",", "").replace("$", "").strip())
    except InvalidOperation as exc:
        raise RuntimeError(
            "group item title is not an exact numeric threshold"
        ) from exc
    if not parsed.is_finite() or parsed <= 0:
        raise RuntimeError("threshold must be finite and positive")
    return parsed


def _price_or_none(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError("side-specific Gamma price is invalid") from exc
    if not price.is_finite() or not Decimal("0") <= price <= Decimal("1"):
        raise RuntimeError("side-specific Gamma price is out of range")
    return price


def _market_row(
    market: Mapping[str, Any],
    *,
    expected_threshold: Decimal,
    required_rule_fragments: list[str],
) -> dict[str, Any]:
    threshold = _threshold(market.get("groupItemTitle"))
    if threshold != expected_threshold:
        raise RuntimeError("exact threshold population changed")
    outcomes = _list(market.get("outcomes"), name="outcomes")
    tokens = _list(market.get("clobTokenIds"), name="CLOB token IDs")
    description = str(market.get("description") or "")
    if not all(fragment in description for fragment in required_rule_fragments):
        raise RuntimeError("exact threshold resolution rules changed")
    if not (
        outcomes == ["Yes", "No"]
        and len(tokens) == 2
        and all(isinstance(token, str) and token for token in tokens)
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError("exact threshold market is not active and executable")
    yes_best_ask = _price_or_none(market.get("bestAsk"))
    yes_best_bid = _price_or_none(market.get("bestBid"))
    no_ask_proxy = Decimal("1") - yes_best_bid if yes_best_bid is not None else None
    return {
        "gamma_market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "threshold": format(threshold, "f"),
        "yes_best_ask_pUSD": (
            format(yes_best_ask, "f") if yes_best_ask is not None else None
        ),
        "yes_best_bid_pUSD": (
            format(yes_best_bid, "f") if yes_best_bid is not None else None
        ),
        "conservative_no_ask_proxy_pUSD": (
            format(no_ask_proxy, "f") if no_ask_proxy is not None else None
        ),
        "yes_token_id": tokens[0],
        "no_token_id": tokens[1],
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_schedule": market.get("feeSchedule"),
        "minimum_order_size": str(market.get("orderMinSize")),
        "minimum_tick_size": str(market.get("orderPriceMinTickSize")),
    }


def _screen_event(
    event: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if event.get("slug") != contract["event_slug"]:
        raise RuntimeError("exact event slug changed")
    if event.get("title") != contract["event_title"]:
        raise RuntimeError("exact event title changed")
    markets = event.get("markets")
    expected_thresholds = contract["expected_thresholds"]
    if not isinstance(markets, list) or len(markets) != len(expected_thresholds):
        raise RuntimeError("exact event market count changed")
    by_threshold: dict[Decimal, dict[str, Any]] = {}
    for value in markets:
        market = _mapping(value, name="event market")
        threshold = _threshold(market.get("groupItemTitle"))
        if threshold in by_threshold:
            raise RuntimeError("duplicate threshold")
        by_threshold[threshold] = market

    required_fragments = contract["required_rule_fragments"]
    legs: list[dict[str, Any]] = []
    for expected_text in expected_thresholds:
        expected = Decimal(expected_text)
        market = by_threshold.get(expected)
        if market is None:
            raise RuntimeError("expected threshold is absent")
        legs.append(
            _market_row(
                market,
                expected_threshold=expected,
                required_rule_fragments=required_fragments,
            )
        )

    packages: list[dict[str, Any]] = []
    for lower_index, lower in enumerate(legs):
        for higher in legs[lower_index + 1 :]:
            lower_yes = lower["yes_best_ask_pUSD"]
            higher_no = higher["conservative_no_ask_proxy_pUSD"]
            executable_proxy_available = lower_yes is not None and higher_no is not None
            displayed_sum = (
                Decimal(lower_yes) + Decimal(higher_no)
                if executable_proxy_available
                else None
            )
            packages.append(
                {
                    "lower_threshold": lower["threshold"],
                    "higher_threshold": higher["threshold"],
                    "lower_yes_market_id": lower["gamma_market_id"],
                    "higher_no_market_id": higher["gamma_market_id"],
                    "lower_yes_token_id": lower["yes_token_id"],
                    "higher_no_token_id": higher["no_token_id"],
                    "lower_yes_best_ask_pUSD": lower_yes,
                    "higher_no_conservative_ask_proxy_pUSD": higher_no,
                    "side_specific_rejection_price_available": executable_proxy_available,
                    "displayed_price_sum_pUSD": (
                        format(displayed_sum, "f")
                        if displayed_sum is not None
                        else None
                    ),
                    "optimistic_displayed_headroom_pUSD": (
                        format(Decimal("1") - displayed_sum, "f")
                        if displayed_sum is not None
                        else None
                    ),
                    "guaranteed_payout_floor_pUSD_per_share": "1",
                    "passes_strict_displayed_gross_gate": (
                        displayed_sum is not None and displayed_sum < Decimal("1")
                    ),
                }
            )
    packages.sort(
        key=lambda row: (
            row["displayed_price_sum_pUSD"] is None,
            Decimal(row["displayed_price_sum_pUSD"])
            if row["displayed_price_sum_pUSD"] is not None
            else Decimal("Infinity"),
            Decimal(row["lower_threshold"]),
            Decimal(row["higher_threshold"]),
        )
    )
    return legs, packages


def _validate_contract(
    contract: Mapping[str, Any], path: Path, *, preflight_only: bool
) -> None:
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_one_public_gamma_request"
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
    slug = contract.get("event_slug")
    if contract.get("request") != {
        "body_sha256": _sha256(b""),
        "count": 1,
        "method": "GET",
        "url": f"https://gamma-api.polymarket.com/events/slug/{slug}",
    }:
        raise RuntimeError("request boundary changed")
    thresholds = contract.get("expected_thresholds")
    if (
        not isinstance(thresholds, list)
        or not 2 <= len(thresholds) <= 100
        or any(not isinstance(value, str) for value in thresholds)
        or [Decimal(value) for value in thresholds]
        != sorted({Decimal(value) for value in thresholds})
    ):
        raise RuntimeError("expected threshold population is invalid")
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
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract, contract_path, preflight_only=args.preflight_only)
    if args.preflight_only:
        print("preflight_passed=true")
        return

    outputs = {key: _root_path(value) for key, value in contract["outputs"].items()}
    for path in outputs.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
    raw, receipt = _request(
        method="GET",
        url=contract["request"]["url"],
        body=b"",
        name="exact-crypto-threshold-ladder-event-v2",
        raw_path=outputs["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=outputs["journal_path"],
    )
    event = _mapping(json.loads(raw), name="exact event response")
    legs, packages = _screen_event(event, contract)
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
        "capture": {"receipt": receipt},
        "payoff_identity": contract["payoff_identity"],
        "screen": {
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
    outputs["result_path"].write_text(
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
                "payloads_printed": 0,
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

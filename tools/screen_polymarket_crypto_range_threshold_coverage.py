"""Screen one exact same-observation crypto range/threshold event pair."""

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
from tools.screen_polymarket_exact_crypto_threshold_ladder import (
    _list,
    _threshold,
)
from tools.screen_polymarket_exact_two_leg_package import _request


SCHEMA = "polymarket-crypto-range-threshold-coverage-prefilter-result-v1"


def _binary_market(
    market: dict[str, Any], *, required_rule_fragments: list[str]
) -> dict[str, Any]:
    outcomes = _list(market.get("outcomes"), name="outcomes")
    prices = _list(market.get("outcomePrices"), name="outcome prices")
    tokens = _list(market.get("clobTokenIds"), name="CLOB token IDs")
    description = str(market.get("description") or "")
    if not all(fragment in description for fragment in required_rule_fragments):
        raise RuntimeError("exact resolution rules changed")
    if not (
        outcomes == ["Yes", "No"]
        and len(prices) == 2
        and len(tokens) == 2
        and all(isinstance(token, str) and token for token in tokens)
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError("exact market is not active and executable")
    yes = Decimal(str(prices[0]))
    no = Decimal(str(prices[1]))
    if not (
        yes.is_finite()
        and no.is_finite()
        and Decimal("0") <= yes <= Decimal("1")
        and Decimal("0") <= no <= Decimal("1")
    ):
        raise RuntimeError("displayed outcome price is invalid")
    return {
        "gamma_market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "question": str(market["question"]),
        "yes_price_pUSD": format(yes, "f"),
        "no_price_pUSD": format(no, "f"),
        "yes_token_id": tokens[0],
        "no_token_id": tokens[1],
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_schedule": market.get("feeSchedule"),
        "minimum_order_size": str(market.get("orderMinSize")),
        "minimum_tick_size": str(market.get("orderPriceMinTickSize")),
    }


def _event_markets(
    event: dict[str, Any],
    *,
    slug: str,
    title: str,
    expected_labels: list[str],
    required_rule_fragments: list[str],
) -> dict[str, dict[str, Any]]:
    if event.get("slug") != slug or event.get("title") != title:
        raise RuntimeError("exact event identity changed")
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != len(expected_labels):
        raise RuntimeError("exact event market count changed")
    by_label: dict[str, dict[str, Any]] = {}
    for market in markets:
        if not isinstance(market, dict):
            raise RuntimeError("event market must be an object")
        label = str(market.get("groupItemTitle") or "")
        if label in by_label:
            raise RuntimeError("duplicate event label")
        by_label[label] = _binary_market(
            market, required_rule_fragments=required_rule_fragments
        )
    if set(by_label) != set(expected_labels):
        raise RuntimeError("exact event label population changed")
    return by_label


def _screen_events(
    range_event: dict[str, Any],
    threshold_event: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    range_labels = contract["range_event"]["expected_labels"]
    range_by_label = _event_markets(
        range_event,
        slug=contract["range_event"]["slug"],
        title=contract["range_event"]["title"],
        expected_labels=range_labels,
        required_rule_fragments=contract["range_event"]["required_rule_fragments"],
    )
    threshold_labels = contract["threshold_event"]["expected_thresholds"]
    threshold_by_label = _event_markets(
        threshold_event,
        slug=contract["threshold_event"]["slug"],
        title=contract["threshold_event"]["title"],
        expected_labels=threshold_labels,
        required_rule_fragments=contract["threshold_event"][
            "required_rule_fragments"
        ],
    )
    if [format(_threshold(label), "f") for label in threshold_labels] != threshold_labels:
        raise RuntimeError("threshold labels are not canonical exact decimals")

    packages: list[dict[str, Any]] = []
    for boundary in contract["shared_boundaries"]:
        threshold = threshold_by_label[boundary["threshold_label"]]
        start_index = range_labels.index(boundary["range_start_label"])
        upper_legs = [range_by_label[label] for label in range_labels[start_index:]]
        lower_legs = [range_by_label[label] for label in range_labels[: start_index + 1]]
        for direction, threshold_side, range_legs in (
            ("upper_coverage", "no", upper_legs),
            ("lower_coverage", "yes", lower_legs),
        ):
            threshold_price = Decimal(threshold[f"{threshold_side}_price_pUSD"])
            range_sum = sum(
                (Decimal(row["yes_price_pUSD"]) for row in range_legs), Decimal("0")
            )
            displayed_sum = threshold_price + range_sum
            packages.append(
                {
                    "boundary": boundary["threshold_label"],
                    "direction": direction,
                    "threshold_side": threshold_side.upper(),
                    "threshold_market_id": threshold["gamma_market_id"],
                    "threshold_token_id": threshold[f"{threshold_side}_token_id"],
                    "range_labels": [
                        label
                        for label in (
                            range_labels[start_index:]
                            if direction == "upper_coverage"
                            else range_labels[: start_index + 1]
                        )
                    ],
                    "range_yes_market_ids": [row["gamma_market_id"] for row in range_legs],
                    "range_yes_token_ids": [row["yes_token_id"] for row in range_legs],
                    "threshold_displayed_price_pUSD": format(threshold_price, "f"),
                    "range_displayed_price_sum_pUSD": format(range_sum, "f"),
                    "displayed_price_sum_pUSD": format(displayed_sum, "f"),
                    "optimistic_displayed_headroom_pUSD": format(
                        Decimal("1") - displayed_sum, "f"
                    ),
                    "optimistic_rule_consistent_payout_floor_pUSD": "1",
                    "passes_strict_displayed_gross_gate": displayed_sum < Decimal("1"),
                }
            )
    packages.sort(
        key=lambda row: (
            Decimal(row["displayed_price_sum_pUSD"]),
            Decimal(row["boundary"]),
            row["direction"],
        )
    )
    range_rows = [
        {"label": label, **range_by_label[label]} for label in range_labels
    ]
    threshold_rows = [
        {"threshold": label, **threshold_by_label[label]}
        for label in threshold_labels
    ]
    return range_rows + threshold_rows, packages


def _validate_contract(contract: dict[str, Any], path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an exact UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    expected_requests = [
        {
            "body_sha256": _sha256(b""),
            "method": "GET",
            "name": name,
            "url": f"https://gamma-api.polymarket.com/events/slug/{section['slug']}",
        }
        for name, section in (
            ("range_event", contract["range_event"]),
            ("threshold_event", contract["threshold_event"]),
        )
    ]
    if contract.get("requests") != expected_requests:
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
        "public_unauthenticated_read_only_requests": 2,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    _validate_contract(contract, contract_path)
    outputs = {key: _root_path(value) for key, value in contract["outputs"].items()}
    for path in outputs.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)

    responses: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for request in contract["requests"]:
        raw, receipt = _request(
            method="GET",
            url=request["url"],
            body=b"",
            name=request["name"],
            raw_path=outputs[f"{request['name']}_raw_path"],
            raw_relative_path=contract["outputs"][f"{request['name']}_raw_path"],
            journal_path=outputs["journal_path"],
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("exact event response must be an object")
        responses[request["name"]] = parsed
        receipts.append(receipt)

    legs, packages = _screen_events(
        responses["range_event"], responses["threshold_event"], contract
    )
    candidates = [row for row in packages if row["passes_strict_displayed_gross_gate"]]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {"receipts": receipts},
        "payoff_identities": contract["payoff_identities"],
        "screen": {
            "event_slugs": [
                contract["range_event"]["slug"],
                contract["threshold_event"]["slug"],
            ],
            "leg_count": len(legs),
            "package_count": len(packages),
            "legs": legs,
            "packages_ranked_by_displayed_sum": packages,
            "strict_displayed_candidate_count": len(candidates),
            "best_package": packages[0],
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_separate_exact_depth_fee_and_resolution_proof"
                if candidates
                else "rejected_before_books_fees_onchain_and_account_requests"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/screen_polymarket_crypto_range_threshold_coverage.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
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
                "package_count": len(packages),
                "best_displayed_sum_pUSD": packages[0]["displayed_price_sum_pUSD"],
                "strict_displayed_candidate_count": len(candidates),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Screen newly deployed crypto range/threshold pairs from one retained delta."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_crypto_range_threshold_coverage import _event_markets
from tools.screen_polymarket_exact_crypto_threshold_ladder import _threshold


SCHEMA = "polymarket-crypto-range-threshold-delta-result-v1"
SCOPED_TITLE = re.compile(r"^(Bitcoin|Ethereum|Solana) (price|above) .+\?$")


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be an array")
    return value


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("timestamp must be an explicit UTC instant")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)


def _load(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_bytes()), name=path.name)


def _validate_capture(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _mapping(contract.get("retained_source"), name="retained source")
    raw_path = _root_path(str(source["raw_path"]))
    intent_path = _root_path(str(source["intent_path"]))
    receipt_path = _root_path(str(source["receipt_path"]))
    for path, field in (
        (raw_path, "raw_sha256"),
        (intent_path, "intent_sha256"),
        (receipt_path, "receipt_sha256"),
    ):
        if _sha256(path.read_bytes()) != source[field]:
            raise RuntimeError(f"retained source hash mismatch: {path.name}")

    intent = _load(intent_path)
    receipt = _load(receipt_path)
    request = _mapping(source.get("request"), name="request contract")
    if not (
        intent.get("method") == "GET"
        and intent.get("url") == request["url"]
        and intent.get("params") == request["params"]
        and intent.get("json_body_sha256") is None
        and receipt.get("method") == "GET"
        and receipt.get("status_code") == 200
        and receipt.get("payload_sha256") == source["raw_sha256"]
        and receipt.get("payload_bytes") == raw_path.stat().st_size
    ):
        raise RuntimeError("retained request or receipt contract changed")
    final = urlsplit(str(receipt.get("final_url") or ""))
    expected = {key: [value] for key, value in request["params"].items()}
    if (
        final.scheme != "https"
        or final.netloc != "gamma-api.polymarket.com"
        or final.path != "/events/keyset"
        or parse_qs(final.query) != expected
    ):
        raise RuntimeError("retained final URL changed")
    return _load(raw_path), receipt


def _package_rows(
    *,
    asset: str,
    range_event: dict[str, Any],
    threshold_event: dict[str, Any],
    pair: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    range_spec = _mapping(pair.get("range_event"), name="range event contract")
    threshold_spec = _mapping(
        pair.get("threshold_event"), name="threshold event contract"
    )
    range_labels = [str(value) for value in range_spec["expected_labels"]]
    threshold_labels = [
        str(value) for value in threshold_spec["expected_labels"]
    ]
    range_by_label = _event_markets(
        range_event,
        slug=str(range_spec["slug"]),
        title=str(range_spec["title"]),
        expected_labels=range_labels,
        required_rule_fragments=[
            str(value) for value in range_spec["required_rule_fragments"]
        ],
    )
    threshold_by_label = _event_markets(
        threshold_event,
        slug=str(threshold_spec["slug"]),
        title=str(threshold_spec["title"]),
        expected_labels=threshold_labels,
        required_rule_fragments=[
            str(value) for value in threshold_spec["required_rule_fragments"]
        ],
    )
    if not (
        range_event.get("active") is True
        and range_event.get("closed") is False
        and range_event.get("negRisk") is True
        and threshold_event.get("active") is True
        and threshold_event.get("closed") is False
        and threshold_event.get("negRisk") is False
    ):
        raise RuntimeError("range or threshold event architecture changed")
    if range_event.get("endDate") != threshold_event.get("endDate"):
        raise RuntimeError("paired event observation instant changed")

    packages: list[dict[str, Any]] = []
    for boundary in pair["shared_boundaries"]:
        threshold_label = str(boundary["threshold_label"])
        range_start = str(boundary["range_start_label"])
        threshold = threshold_by_label[threshold_label]
        start = range_labels.index(range_start)
        for direction, side, labels in (
            ("upper_coverage", "no", range_labels[start:]),
            ("lower_coverage", "yes", range_labels[: start + 1]),
        ):
            range_rows = [range_by_label[label] for label in labels]
            threshold_price = Decimal(threshold[f"{side}_price_pUSD"])
            range_sum = sum(
                (Decimal(row["yes_price_pUSD"]) for row in range_rows), Decimal("0")
            )
            displayed_sum = threshold_price + range_sum
            packages.append(
                {
                    "asset": asset,
                    "boundary": format(_threshold(threshold_label), "f"),
                    "threshold_label": threshold_label,
                    "range_start_label": range_start,
                    "direction": direction,
                    "threshold_side": side.upper(),
                    "threshold_market_id": threshold["gamma_market_id"],
                    "threshold_token_id": threshold[f"{side}_token_id"],
                    "range_labels": labels,
                    "range_yes_market_ids": [
                        row["gamma_market_id"] for row in range_rows
                    ],
                    "range_yes_token_ids": [row["yes_token_id"] for row in range_rows],
                    "threshold_displayed_price_pUSD": format(threshold_price, "f"),
                    "range_displayed_price_sum_pUSD": format(range_sum, "f"),
                    "displayed_price_sum_pUSD": format(displayed_sum, "f"),
                    "optimistic_displayed_headroom_pUSD": format(
                        Decimal("1") - displayed_sum, "f"
                    ),
                    "optimistic_rule_consistent_floor_pUSD": "1",
                    "passes_strict_displayed_gross_gate": displayed_sum < Decimal("1"),
                }
            )
    legs = [
        {"asset": asset, "event_role": "range", "label": label, **row}
        for label, row in range_by_label.items()
    ] + [
        {"asset": asset, "event_role": "threshold", "label": label, **row}
        for label, row in threshold_by_label.items()
    ]
    return legs, packages


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
    created = [_utc(event.get("createdAt")) for event in events]
    if (
        len(events) != int(contract["population_gates"]["event_count"])
        or created != sorted(created, reverse=True)
        or min(created) > cutoff
        or not isinstance(payload.get("next_cursor"), str)
        or not payload["next_cursor"]
    ):
        raise RuntimeError("retained newest-first population does not cross cutoff")
    post_cutoff = [event for event in events if _utc(event.get("createdAt")) > cutoff]
    if len(post_cutoff) != int(contract["population_gates"]["post_cutoff_count"]):
        raise RuntimeError("post-cutoff population count changed")
    scoped = {
        str(event.get("slug"))
        for event in post_cutoff
        if SCOPED_TITLE.match(str(event.get("title") or ""))
        and len(_list(event.get("markets"), name="event markets")) > 1
    }
    expected_scoped = set(contract["population_gates"]["scoped_pair_event_slugs"])
    if scoped != expected_scoped:
        raise RuntimeError("new scoped multi-market event population changed")
    by_slug = {str(event.get("slug")): event for event in events}
    if len(by_slug) != len(events):
        raise RuntimeError("duplicate event slug")

    all_legs: list[dict[str, Any]] = []
    all_packages: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for pair in contract["pairs"]:
        asset = str(pair["asset"])
        range_event = by_slug[str(pair["range_event"]["slug"])]
        threshold_event = by_slug[str(pair["threshold_event"]["slug"])]
        if _utc(range_event.get("createdAt")) <= cutoff or _utc(
            threshold_event.get("createdAt")
        ) <= cutoff:
            raise RuntimeError("paired event was not created after the cutoff")
        legs, packages = _package_rows(
            asset=asset,
            range_event=range_event,
            threshold_event=threshold_event,
            pair=pair,
        )
        all_legs.extend(legs)
        all_packages.extend(packages)
        pair_summaries.append(
            {
                "asset": asset,
                "range_event_slug": range_event["slug"],
                "threshold_event_slug": threshold_event["slug"],
                "shared_boundary_count": len(pair["shared_boundaries"]),
                "package_count": len(packages),
            }
        )
    all_packages.sort(
        key=lambda row: (
            Decimal(row["displayed_price_sum_pUSD"]),
            row["asset"],
            Decimal(row["boundary"]),
            row["direction"],
        )
    )
    candidates = [
        row for row in all_packages if row["passes_strict_displayed_gross_gate"]
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
            "event_count": len(events),
            "post_cutoff_event_count": len(post_cutoff),
            "newest_created_at_utc": max(created).isoformat().replace("+00:00", "Z"),
            "oldest_created_at_utc": min(created).isoformat().replace("+00:00", "Z"),
            "delta_cutoff_utc": contract["delta_cutoff_utc"],
            "next_cursor_present": True,
            "delta_complete_through_cutoff": True,
        },
        "screen": {
            "pair_summaries": pair_summaries,
            "leg_count": len(all_legs),
            "package_count": len(all_packages),
            "legs": all_legs,
            "packages_ranked_by_displayed_sum": all_packages,
            "strict_displayed_candidate_count": len(candidates),
            "best_package": all_packages[0],
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_one_separately_frozen_exact_depth_batch"
                if candidates
                else "complete_new_pair_delta_rejected_before_books_fees_and_accounts"
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

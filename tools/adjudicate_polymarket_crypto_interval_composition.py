"""Screen exact 5m-to-15m crypto direction composition from retained events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
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


SCHEMA = "polymarket-crypto-interval-composition-result-v1"
_SLUG = re.compile(r"^(btc|eth|sol)-updown-(5m|15m)-(\d+)$")


def _json_list(value: object, *, name: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise RuntimeError(f"{name} must be a JSON list")
    return parsed


def _binary_market(
    event: dict[str, Any], spec: dict[str, Any], rule_fragments: list[str]
) -> dict[str, Any]:
    if event.get("slug") != spec["slug"] or event.get("title") != spec["title"]:
        raise RuntimeError("exact interval event identity changed")
    match = _SLUG.fullmatch(str(event["slug"]))
    if match is None:
        raise RuntimeError("interval slug is invalid")
    duration = 300 if match.group(2) == "5m" else 900
    if int(match.group(3)) != spec["start_epoch_seconds"]:
        raise RuntimeError("interval start epoch changed")
    expected_end = datetime.fromtimestamp(
        spec["start_epoch_seconds"] + duration, tz=timezone.utc
    )
    if _utc(event.get("endDate")) != expected_end:
        raise RuntimeError("interval end timestamp changed")
    if not (
        _utc(event.get("createdAt")) > _utc(spec["created_after_utc"])
        and event.get("active") is True
        and event.get("closed") is False
        and event.get("negRisk") is False
    ):
        raise RuntimeError("retained interval event state changed")
    markets = _list(event.get("markets"), name="interval markets")
    if len(markets) != 1:
        raise RuntimeError("interval event must contain exactly one market")
    market = _mapping(markets[0], name="interval market")
    description = str(market.get("description") or "")
    if not all(fragment in description for fragment in rule_fragments):
        raise RuntimeError("interval resolution rule changed")
    outcomes = _json_list(market.get("outcomes"), name="interval outcomes")
    prices = _json_list(market.get("outcomePrices"), name="interval prices")
    tokens = _json_list(market.get("clobTokenIds"), name="interval tokens")
    if not (
        outcomes == ["Up", "Down"]
        and len(prices) == 2
        and len(tokens) == 2
        and all(isinstance(token, str) and token for token in tokens)
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ):
        raise RuntimeError("interval market is not active binary CLOB inventory")
    parsed_prices = [Decimal(str(value)) for value in prices]
    if any(
        not value.is_finite() or value < Decimal("0") or value > Decimal("1")
        for value in parsed_prices
    ):
        raise RuntimeError("interval displayed price is invalid")
    return {
        "slug": spec["slug"],
        "title": spec["title"],
        "start_epoch_seconds": spec["start_epoch_seconds"],
        "end_epoch_seconds": spec["start_epoch_seconds"] + duration,
        "duration_seconds": duration,
        "gamma_market_id": str(market["id"]),
        "condition_id": str(market["conditionId"]),
        "up_price_pUSD": format(parsed_prices[0], "f"),
        "down_price_pUSD": format(parsed_prices[1], "f"),
        "up_token_id": tokens[0],
        "down_token_id": tokens[1],
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_schedule": market.get("feeSchedule"),
        "minimum_order_size": str(market.get("orderMinSize")),
        "minimum_tick_size": str(market.get("orderPriceMinTickSize")),
    }


def _package(
    *, asset: str, direction: str, shorts: list[dict[str, Any]], long: dict[str, Any]
) -> dict[str, Any]:
    if direction == "up_chain":
        legs = [
            {
                "slug": row["slug"],
                "outcome": "Down",
                "token_id": row["down_token_id"],
                "displayed_price_pUSD": row["down_price_pUSD"],
            }
            for row in shorts
        ]
        legs.append(
            {
                "slug": long["slug"],
                "outcome": "Up",
                "token_id": long["up_token_id"],
                "displayed_price_pUSD": long["up_price_pUSD"],
            }
        )
        identity = "NO(U1)+NO(U2)+NO(U3)+YES(U15)"
    elif direction == "down_chain":
        legs = [
            {
                "slug": row["slug"],
                "outcome": "Up",
                "token_id": row["up_token_id"],
                "displayed_price_pUSD": row["up_price_pUSD"],
            }
            for row in shorts
        ]
        legs.append(
            {
                "slug": long["slug"],
                "outcome": "Down",
                "token_id": long["down_token_id"],
                "displayed_price_pUSD": long["down_price_pUSD"],
            }
        )
        identity = "YES(U1)+YES(U2)+YES(U3)+NO(U15)"
    else:
        raise RuntimeError("unknown composition direction")
    displayed_sum = sum(
        (Decimal(row["displayed_price_pUSD"]) for row in legs), Decimal("0")
    )
    return {
        "asset": asset,
        "direction": direction,
        "identity": identity,
        "legs": legs,
        "displayed_price_sum_pUSD": format(displayed_sum, "f"),
        "guaranteed_payout_floor_pUSD_per_share_package": "1",
        "optimistic_displayed_headroom_pUSD": format(
            Decimal("1") - displayed_sum, "f"
        ),
        "passes_strict_displayed_gross_gate": displayed_sum < Decimal("1"),
    }


def run(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    envelope = _load(contract_path)
    if _canonical_hash(envelope, "contract_sha256") != envelope.get(
        "contract_sha256"
    ):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(envelope["contract_path"])):
        raise RuntimeError("contract path mismatch")
    contract = envelope
    if "base_contract" in envelope:
        base_path = _root_path(str(envelope["base_contract"]["path"]))
        base = _load(base_path)
        if (
            _canonical_hash(base, "contract_sha256")
            != envelope["base_contract"]["canonical_sha256"]
            or base.get("contract_sha256")
            != envelope["base_contract"]["canonical_sha256"]
        ):
            raise RuntimeError("base contract hash mismatch")
        contract = {
            **base,
            "schema_version": envelope["schema_version"],
            "contract_path": envelope["contract_path"],
            "frozen_at_utc": envelope["frozen_at_utc"],
            "output_path": envelope["output_path"],
            "authority": envelope["authority"],
            "implementation": envelope["implementation"],
            "contract_sha256": envelope["contract_sha256"],
        }
    frozen = _utc(contract["frozen_at_utc"])
    if frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen timestamp is in the future")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    for evidence in contract["prior_semantic_evidence"]:
        evidence_path = _root_path(str(evidence["path"]))
        if _sha256(evidence_path.read_bytes()) != evidence["file_sha256"]:
            raise RuntimeError("prior semantic evidence hash mismatch")
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

    all_markets: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for group in contract["groups"]:
        asset = str(group["asset"])
        if asset in seen_assets:
            raise RuntimeError("duplicate composition asset")
        seen_assets.add(asset)
        short_rows = [
            _binary_market(by_slug[str(spec["slug"])], spec, group["rule_fragments"])
            for spec in group["short_intervals"]
        ]
        long_row = _binary_market(
            by_slug[str(group["long_interval"]["slug"])],
            group["long_interval"],
            group["rule_fragments"],
        )
        expected_start = int(group["long_interval"]["start_epoch_seconds"])
        if [row["start_epoch_seconds"] for row in short_rows] != [
            expected_start,
            expected_start + 300,
            expected_start + 600,
        ] or [row["end_epoch_seconds"] for row in short_rows] != [
            expected_start + 300,
            expected_start + 600,
            expected_start + 900,
        ] or (
            long_row["start_epoch_seconds"] != expected_start
            or long_row["end_epoch_seconds"] != expected_start + 900
        ):
            raise RuntimeError("short intervals do not exactly partition long interval")
        all_markets.extend({"asset": asset, **row} for row in [*short_rows, long_row])
        packages.extend(
            _package(asset=asset, direction=direction, shorts=short_rows, long=long_row)
            for direction in ("up_chain", "down_chain")
        )

    if seen_assets != {"BTC", "ETH", "SOL"} or len(packages) != 6:
        raise RuntimeError("complete BTC ETH SOL composition population changed")
    packages.sort(
        key=lambda row: (
            Decimal(row["displayed_price_sum_pUSD"]),
            row["asset"],
            row["direction"],
        )
    )
    candidates = [row for row in packages if row["passes_strict_displayed_gross_gate"]]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": contract["contract_path"], "sha256": contract["contract_sha256"]},
        "retained_capture": {
            "raw_sha256": contract["retained_source"]["raw_sha256"],
            "receipt": receipt,
            "network_requests_added_by_this_screen": 0,
        },
        "payoff_identity": contract["payoff_identity"],
        "screen": {
            "market_count": len(all_markets),
            "package_count": len(packages),
            "markets": all_markets,
            "packages_ranked_by_displayed_sum": packages,
            "strict_displayed_candidate_count": len(candidates),
            "best_package": packages[0],
            "gamma_role": "rejection_only_never_acceptance_or_promotion_evidence",
        },
        "adjudication": {
            "status": (
                "source_only_candidate_requires_one_separately_frozen_four_token_depth_batch"
                if candidates
                else "complete_interval_composition_delta_rejected_before_books_fees_and_accounts"
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
                "best_asset": result["screen"]["best_package"]["asset"],
                "best_direction": result["screen"]["best_package"]["direction"],
                "best_displayed_sum_pUSD": result["screen"]["best_package"]["displayed_price_sum_pUSD"],
                "strict_displayed_candidate_count": result["screen"]["strict_displayed_candidate_count"],
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

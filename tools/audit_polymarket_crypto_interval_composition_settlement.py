"""Audit the interval-composition payoff floor from retained settlements."""

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
from tools.screen_polymarket_crypto_interval_composition_trades import (
    _SLUG,
    _json_list,
    _market_identity,
    _market_row,
)


SCHEMA = "polymarket-crypto-interval-composition-settlement-audit-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def _retained_markets(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_slug: dict[str, dict[str, Any]] = {}
    for source in contract["retained_market_sources"]:
        path = _root_path(str(source["path"]))
        raw = path.read_bytes()
        if _sha256(raw) != source["sha256"]:
            raise RuntimeError("retained market source hash mismatch")
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise RuntimeError("retained market source must be an array")
        for value in rows:
            if not isinstance(value, dict):
                raise RuntimeError("retained market row must be an object")
            slug = str(value.get("slug") or "")
            if _SLUG.fullmatch(slug) is None:
                continue
            prior = by_slug.get(slug)
            if prior is not None:
                if _market_identity(prior) != _market_identity(value):
                    raise RuntimeError("conflicting duplicate retained market identity")
                if str(value.get("updatedAt") or "") > str(
                    prior.get("updatedAt") or ""
                ):
                    by_slug[slug] = value
            else:
                by_slug[slug] = value
    return by_slug


def _complete_set_keys(markets: dict[str, dict[str, Any]]) -> list[list[Any]]:
    available: dict[tuple[str, str], set[int]] = {}
    for slug in markets:
        match = _SLUG.fullmatch(slug)
        if match is None:
            continue
        available.setdefault((match.group(1).upper(), match.group(2)), set()).add(
            int(match.group(3))
        )
    keys: list[list[Any]] = []
    for asset in ("BTC", "ETH", "SOL"):
        five = available.get((asset, "5m"), set())
        for start in available.get((asset, "15m"), set()):
            if {start, start + 300, start + 600}.issubset(five):
                keys.append([asset, start])
    return sorted(keys, key=lambda row: (row[0], row[1]))


def _winner(market: dict[str, Any], *, slug: str) -> str:
    _market_row(market, slug=slug)
    prices = _json_list(market.get("outcomePrices"), name="outcomePrices")
    if len(prices) != 2:
        raise RuntimeError("settled outcome prices must have two values")
    parsed = [Decimal(str(value)) for value in prices]
    if parsed == [Decimal("1"), Decimal("0")]:
        return "Up"
    if parsed == [Decimal("0"), Decimal("1")]:
        return "Down"
    raise RuntimeError("market lacks an exact terminal binary settlement")


def _description_evidence(
    markets: dict[str, dict[str, Any]], keys: list[list[Any]]
) -> list[dict[str, Any]]:
    descriptions: dict[str, set[str]] = {
        asset: set() for asset in ("BTC", "ETH", "SOL")
    }
    sources: dict[str, set[str]] = {asset: set() for asset in ("BTC", "ETH", "SOL")}
    for asset, start in keys:
        prefix = str(asset).lower()
        slugs = [
            f"{prefix}-updown-5m-{start}",
            f"{prefix}-updown-5m-{start + 300}",
            f"{prefix}-updown-5m-{start + 600}",
            f"{prefix}-updown-15m-{start}",
        ]
        for slug in slugs:
            descriptions[asset].add(str(markets[slug].get("description") or ""))
            sources[asset].add(str(markets[slug].get("resolutionSource") or ""))
    evidence: list[dict[str, Any]] = []
    for asset in ("BTC", "ETH", "SOL"):
        if len(descriptions[asset]) != 1 or len(sources[asset]) != 1:
            raise RuntimeError("resolution semantics differ within an asset")
        description = next(iter(descriptions[asset]))
        source = next(iter(sources[asset]))
        if not (
            'resolve to "Up"' in description
            and "greater than or equal to" in description
            and 'resolve to "Down"' in description
            and source in description
        ):
            raise RuntimeError("resolution semantics do not state the expected rule")
        evidence.append(
            {
                "asset": asset,
                "description_sha256": _sha256(description.encode("utf-8")),
                "resolution_source": source,
                "resolution_source_sha256": _sha256(source.encode("ascii")),
            }
        )
    return evidence


def run(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise RuntimeError("contract path mismatch")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    output = _root_path(str(contract["output"]["result_path"]))
    if output.exists():
        raise RuntimeError("one-use settlement result already exists")

    markets = _retained_markets(contract)
    keys = _complete_set_keys(markets)
    key_bytes = json.dumps(keys, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    population = contract["population"]
    if not (
        len(keys) == population["expected_complete_set_count"]
        and len(keys) * 4 == population["expected_market_count"]
        and len(keys) * 2 == population["expected_package_evaluation_count"]
        and _sha256(key_bytes) == population["complete_set_keys_sha256"]
    ):
        raise RuntimeError("complete-set population changed after freeze")

    semantics = _description_evidence(markets, keys)
    rows: list[dict[str, Any]] = []
    for asset, start in keys:
        prefix = str(asset).lower()
        slugs = [
            f"{prefix}-updown-5m-{start}",
            f"{prefix}-updown-5m-{start + 300}",
            f"{prefix}-updown-5m-{start + 600}",
            f"{prefix}-updown-15m-{start}",
        ]
        winners = [_winner(markets[slug], slug=slug) for slug in slugs]
        short_up = [winner == "Up" for winner in winners[:3]]
        long_up = winners[3] == "Up"
        up_chain = sum(not value for value in short_up) + int(long_up)
        down_chain = sum(short_up) + int(not long_up)
        rows.append(
            {
                "asset": asset,
                "start_epoch_seconds": start,
                "short_interval_winners": winners[:3],
                "long_interval_winner": winners[3],
                "up_chain_payout_pUSD_per_package": str(up_chain),
                "down_chain_payout_pUSD_per_package": str(down_chain),
                "up_chain_floor_satisfied": up_chain >= 1,
                "down_chain_floor_satisfied": down_chain >= 1,
            }
        )

    payouts = [
        int(row[field])
        for row in rows
        for field in (
            "up_chain_payout_pUSD_per_package",
            "down_chain_payout_pUSD_per_package",
        )
    ]
    violations = sum(value < 1 for value in payouts)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "population": {
            "complete_set_count": len(keys),
            "market_count": len(keys) * 4,
            "package_evaluation_count": len(keys) * 2,
            "complete_set_keys_sha256": _sha256(key_bytes),
            "assets": ["BTC", "ETH", "SOL"],
        },
        "resolution_semantics": semantics,
        "audit": {
            "rows": rows,
            "minimum_realized_package_payout_pUSD": str(min(payouts)),
            "maximum_realized_package_payout_pUSD": str(max(payouts)),
            "payoff_floor_pUSD": "1",
            "floor_violation_count": violations,
            "all_package_payoff_floors_satisfied": violations == 0,
        },
        "adjudication": {
            "status": (
                "retained_settlements_support_payoff_floor"
                if violations == 0
                else "retained_settlement_violation_rejects_payoff_identity"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "reason": "Settlement consistency proves neither sub-floor acquisition nor atomic executable capacity.",
            "next_action": "Only a future distinct aligned population may trigger a prospectively frozen exact live CLOB package capture.",
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
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
                "complete_sets": result["population"]["complete_set_count"],
                "package_evaluations": result["population"]["package_evaluation_count"],
                "minimum_payout_pUSD": result["audit"][
                    "minimum_realized_package_payout_pUSD"
                ],
                "violations": result["audit"]["floor_violation_count"],
                "network_requests": 0,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

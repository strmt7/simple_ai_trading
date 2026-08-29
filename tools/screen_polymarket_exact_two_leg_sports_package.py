from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

from tools import screen_polymarket_exact_two_leg_package as base


def _line_matches(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return Decimal(str(actual)) == Decimal(str(expected))


def _validate_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["metadata_source"]
    path = base._root_path(source["path"])
    raw = path.read_bytes()
    if base._sha256(raw) != source["file_sha256"]:
        raise RuntimeError("metadata file hash mismatch")
    metadata = json.loads(raw)
    if base._canonical_hash(metadata, "result_sha256") != source["result_sha256"]:
        raise RuntimeError("metadata canonical hash mismatch")
    markets = metadata["discovery"]["active_accepting_markets"]
    for definition in contract["markets"]:
        market = next(
            (row for row in markets if str(row.get("id")) == definition["id"]),
            None,
        )
        if market is None:
            raise RuntimeError(f"market absent: {definition['id']}")
        description = str(market.get("description") or "")
        if not all(
            fragment in description
            for fragment in definition["required_rule_fragments"]
        ):
            raise RuntimeError(f"market rules changed: {definition['id']}")
        outcomes = json.loads(market["outcomes"])
        prices = json.loads(market["outcomePrices"])
        tokens = json.loads(market["clobTokenIds"])
        if not (
            market["question"] == definition["question"]
            and market["sportsMarketType"] == definition["sports_market_type"]
            and _line_matches(market.get("line"), definition.get("line"))
            and market["conditionId"] == definition["condition_id"]
            and outcomes == definition["outcomes"]
            and prices == definition["outcome_prices"]
            and tokens == definition["tokens"]
            and market["active"] is True
            and market["closed"] is False
            and market["acceptingOrders"] is True
            and market["enableOrderBook"] is True
            and market["negRisk"] is False
            and market["feeSchedule"] == contract["execution"]["fee_schedule"]
            and market["takerBaseFee"] == 1000
            and Decimal(str(market["orderMinSize"]))
            <= Decimal(contract["execution"]["quantity_shares_each_leg"])
            and Decimal(str(market["orderPriceMinTickSize"]))
            == Decimal(contract["execution"]["tick_size"])
        ):
            raise RuntimeError(f"market identity changed: {definition['id']}")
    payouts = [
        sum(Decimal(str(value)) for value in state["payouts"].values())
        for state in contract["payoff_proof"]["states"]
    ]
    floor = Decimal(contract["payoff_proof"]["guaranteed_floor_per_share_pUSD"])
    if min(payouts) != floor:
        raise RuntimeError("package lacks its frozen payout floor")
    prices = [
        Decimal(contract["tokens"][name]["gamma_price_pUSD"])
        for name in contract["package"]["token_names"]
    ]
    if sum(prices) != Decimal(contract["gamma_prefilter"]["displayed_price_sum_pUSD"]):
        raise RuntimeError("Gamma prefilter sum mismatch")
    if sum(prices) >= floor:
        raise RuntimeError("Gamma prefilter did not clear the strict rejection gate")
    return metadata


def main() -> None:
    base._validate_metadata = _validate_metadata
    base.main()


if __name__ == "__main__":
    main()

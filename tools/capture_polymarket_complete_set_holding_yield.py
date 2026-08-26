"""Capture and reconcile one public Polymarket complete-set holding-yield case."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "polymarket-complete-set-holding-yield-reconciliation-v3"
WALLET = "0x3fb5c98d825651d7efd2bd48a5d02c2d86c96f2f"
CONDITION_ID = "0x024b68f77bfc019341ee3db8f57c103334e4b9430bba4746d8c94aafd8b36fee"
PUSD_TOKEN = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
HOLDING_YIELD_DISTRIBUTOR = "0x607c8c9866ef3b4665c5a384188706be738d8bf8"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ANNUAL_RATE = Decimal("0.0325")
POSITION_VALUE = Decimal("150")
PAYOUT_SCALE = Decimal("0.0001")
OBSERVED_PAYOUTS = {
    1786579914: Decimal("0.0122"),
    1786666212: Decimal("0.0133"),
    1786752715: Decimal("0.0133"),
    1786839027: Decimal("0.0133"),
    1786925556: Decimal("0.0133"),
    1787012107: Decimal("0.0133"),
    1787098485: Decimal("0.0133"),
    1787184627: Decimal("0.0127"),
    1787271198: Decimal("0.0133"),
    1787357478: Decimal("0.0127"),
    1787444008: Decimal("0.0133"),
    1787530215: Decimal("0.0133"),
    1787616729: Decimal("0.0127"),
    1787703198: Decimal("0.0116"),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


class _CaptureClient:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "simple-ai-trading-public-edge-research/1.0"}
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
        body: object | None = None,
        expect_json: bool = True,
    ) -> tuple[object | bytes, dict[str, object]]:
        response = None
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=body,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.content
            write_bytes_atomic(self.raw_dir / f"{name}.raw", payload)
            receipt = {
                "method": method,
                "url": response.url,
                "status_code": response.status_code,
                "response_bytes": len(payload),
                "response_sha256": _sha256(payload),
                "raw_path": str((self.raw_dir / f"{name}.raw").as_posix()),
            }
            if not expect_json:
                return payload, receipt
            try:
                return response.json(), receipt
            except requests.JSONDecodeError as exc:
                raise ValueError(f"{name} did not return JSON") from exc
        finally:
            # Never let a failed request reuse a previous response object.
            response = None


def _address_topic(address: str) -> str:
    return "0x" + ("0" * 24) + address[2:].lower()


def _capture_market_state(
    client: _CaptureClient,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    positions_raw, positions_receipt = client.request(
        "GET",
        "https://data-api.polymarket.com/positions",
        name="positions",
        params={"user": WALLET, "limit": 500, "offset": 0, "sizeThreshold": 0},
    )
    positions = [
        _mapping(row, name="position") for row in _list(positions_raw, name="positions")
    ]
    if len(positions) >= 500:
        raise ValueError("position response reached its page limit")
    if any(str(row.get("proxyWallet") or "").lower() != WALLET for row in positions):
        raise ValueError("position wallet identity differs")

    conditions = sorted(
        {str(row.get("conditionId") or "").lower() for row in positions}
    )
    if "" in conditions:
        raise ValueError("position condition ID is blank")
    market_receipts: list[dict[str, object]] = []
    eligible_conditions: list[str] = []
    selected_market: dict[str, object] | None = None
    for condition in conditions:
        slugs = {
            str(row.get("slug") or "")
            for row in positions
            if str(row.get("conditionId") or "").lower() == condition
        }
        if len(slugs) != 1 or "" in slugs:
            raise ValueError(f"condition {condition} does not map to one position slug")
        slug = slugs.pop()
        market_raw, market_receipt = client.request(
            "GET",
            f"https://gamma-api.polymarket.com/markets/slug/{slug}",
            name=f"market-{condition[2:]}",
        )
        market = _mapping(market_raw, name=f"market {condition}")
        if str(market.get("conditionId") or "").lower() != condition:
            raise ValueError(f"condition {condition} identity differs")
        market_receipts.append(market_receipt)
        if market.get("holdingRewardsEnabled") is True:
            eligible_conditions.append(condition)
        if condition == CONDITION_ID:
            selected_market = market
    if eligible_conditions != [CONDITION_ID] or selected_market is None:
        raise ValueError(
            "wallet's current holding-reward-eligible condition set differs"
        )
    if (
        selected_market.get("active") is not True
        or selected_market.get("closed") is not False
        or selected_market.get("acceptingOrders") is not True
        or selected_market.get("enableOrderBook") is not True
    ):
        raise ValueError("selected market is not active and order-capable")

    pair = [
        row
        for row in positions
        if str(row.get("conditionId") or "").lower() == CONDITION_ID
    ]
    if len(pair) != 2 or {str(row.get("outcome")) for row in pair} != {"Yes", "No"}:
        raise ValueError("selected complete set is not an exact YES/NO pair")
    sizes = {Decimal(str(row.get("size"))) for row in pair}
    if sizes != {POSITION_VALUE} or any(
        row.get("mergeable") is not True for row in pair
    ):
        raise ValueError("selected YES/NO balances are not equal and mergeable")
    current_value = sum(
        (Decimal(str(row.get("currentValue"))) for row in pair), Decimal(0)
    )
    if current_value != POSITION_VALUE:
        raise ValueError("selected pair mark does not preserve the complete-set value")
    if (
        sum(
            (
                Decimal(str(row.get("currentValue")))
                for row in positions
                if str(row.get("conditionId") or "").lower() in eligible_conditions
            ),
            Decimal(0),
        )
        != POSITION_VALUE
    ):
        raise ValueError(
            "wallet has other current holding-reward-eligible position value"
        )

    summary = {
        "wallet": WALLET,
        "position_row_count": len(positions),
        "unique_condition_count": len(conditions),
        "current_holding_reward_condition_ids": eligible_conditions,
        "current_holding_reward_position_value_pusd": _decimal_text(current_value),
        "selected_condition_id": CONDITION_ID,
        "question": str(selected_market.get("question") or ""),
        "yes_shares": _decimal_text(POSITION_VALUE),
        "no_shares": _decimal_text(POSITION_VALUE),
        "mergeable": True,
        "positions_source": positions_receipt,
        "per_condition_market_sources": market_receipts,
    }
    return positions, summary, pair


def _capture_yield_and_receipts(
    client: _CaptureClient,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    all_activity_raw, all_activity_receipt = client.request(
        "GET",
        "https://data-api.polymarket.com/activity",
        name="all-activity",
        params={"user": WALLET, "limit": 500, "offset": 0},
    )
    all_activities = [
        _mapping(row, name="account activity")
        for row in _list(all_activity_raw, name="account activity")
    ]
    if len(all_activities) >= 500:
        raise ValueError("account activity response reached its page limit")
    activity_raw, activity_receipt = client.request(
        "GET",
        "https://data-api.polymarket.com/activity",
        name="yield-activity",
        params={"user": WALLET, "type": "YIELD", "limit": 500, "offset": 0},
    )
    activities = [
        _mapping(row, name="yield activity")
        for row in _list(activity_raw, name="yield activity")
    ]
    if len(activities) >= 500:
        raise ValueError(
            "YIELD response reached its page limit; pagination is required"
        )
    if any(
        str(row.get("proxyWallet") or "").lower() != WALLET
        or row.get("type") != "YIELD"
        for row in activities
    ):
        raise ValueError("YIELD activity identity differs")
    if len(all_activities) != len(activities) or any(
        row.get("type") != "YIELD" for row in all_activities
    ):
        raise ValueError("wallet has non-YIELD activity or incomplete account history")
    observed = {
        int(row.get("timestamp", -1)): Decimal(str(row.get("usdcSize")))
        for row in activities
        if int(row.get("timestamp", -1)) in OBSERVED_PAYOUTS
    }
    if observed != OBSERVED_PAYOUTS:
        raise ValueError("frozen 14-day YIELD sequence differs")
    rows = sorted(
        (row for row in activities if int(row.get("timestamp", -1)) in observed),
        key=lambda row: int(row["timestamp"]),
    )
    tx_hashes = [str(row.get("transactionHash") or "").lower() for row in rows]
    if len(set(tx_hashes)) != len(rows) or any(len(value) != 66 for value in tx_hashes):
        raise ValueError("YIELD transaction hashes are invalid or duplicated")
    all_activity_hashes = {
        str(row.get("transactionHash") or "").lower() for row in all_activities
    }
    if all_activity_hashes != set(tx_hashes):
        raise ValueError("all-activity and YIELD transaction sets differ")

    rpc_body = [
        {
            "jsonrpc": "2.0",
            "id": index,
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash],
        }
        for index, tx_hash in enumerate(tx_hashes)
    ]
    rpc_raw, rpc_receipt = client.request(
        "POST",
        "https://polygon-bor-rpc.publicnode.com",
        name="polygon-yield-receipts",
        body=rpc_body,
    )
    rpc_rows = _list(rpc_raw, name="Polygon receipt batch")
    by_id = {
        _mapping(row, name="Polygon receipt response").get("id"): row
        for row in rpc_rows
    }
    if set(by_id) != set(range(len(rows))):
        raise ValueError("Polygon receipt batch IDs differ")

    reconciled: list[dict[str, object]] = []
    from_topic = _address_topic(HOLDING_YIELD_DISTRIBUTOR)
    to_topic = _address_topic(WALLET)
    for index, (activity, tx_hash) in enumerate(zip(rows, tx_hashes, strict=True)):
        envelope = _mapping(by_id[index], name="Polygon receipt envelope")
        if envelope.get("error") is not None:
            raise ValueError(f"Polygon receipt {tx_hash} failed")
        receipt = _mapping(envelope.get("result"), name="Polygon transaction receipt")
        if (
            str(receipt.get("transactionHash") or "").lower() != tx_hash
            or receipt.get("status") != "0x1"
        ):
            raise ValueError(f"Polygon receipt {tx_hash} identity or status differs")
        transfers: list[Decimal] = []
        for raw_log in _list(receipt.get("logs"), name="Polygon receipt logs"):
            log = _mapping(raw_log, name="Polygon receipt log")
            topics = _list(log.get("topics"), name="Polygon log topics")
            if (
                str(log.get("address") or "").lower() == PUSD_TOKEN
                and len(topics) == 3
                and str(topics[0]).lower() == TRANSFER_TOPIC
                and str(topics[1]).lower() == from_topic
                and str(topics[2]).lower() == to_topic
            ):
                transfers.append(
                    Decimal(int(str(log.get("data")), 16)) / Decimal(1_000_000)
                )
        expected = Decimal(str(activity.get("usdcSize")))
        if transfers != [expected]:
            raise ValueError(f"YIELD transfer for {tx_hash} does not reconcile exactly")
        reconciled.append(
            {
                "timestamp": int(activity["timestamp"]),
                "amount_pusd": _decimal_text(expected),
                "transaction_hash": tx_hash,
                "block_number": int(str(receipt["blockNumber"]), 16),
                "transfer_from": HOLDING_YIELD_DISTRIBUTOR,
                "transfer_to": WALLET,
                "token": PUSD_TOKEN,
            }
        )
    return reconciled, {
        "all_activity_source": all_activity_receipt,
        "activity_source": activity_receipt,
        "polygon_receipt_source": rpc_receipt,
        "total_current_yield_row_count": len(activities),
        "total_account_activity_row_count": len(all_activities),
        "non_yield_account_activity_row_count": 0,
        "frozen_observation_row_count": len(rows),
        "all_receipts_reconciled": True,
    }


def _capture_official_terms(client: _CaptureClient) -> dict[str, object]:
    raw, receipt = client.request(
        "GET",
        "https://help.polymarket.com/en/articles/13364459-holding-rewards",
        name="official-holding-rewards-help",
        expect_json=False,
    )
    assert isinstance(raw, bytes)
    text = raw.decode("utf-8", errors="replace").casefold()
    required = ("3.25%", "randomly sampled once each hour", "distributed daily")
    if any(phrase not in text for phrase in required):
        raise ValueError(
            "official holding-reward terms no longer contain required semantics"
        )
    sdk_raw, sdk_receipt = client.request(
        "GET",
        "https://raw.githubusercontent.com/Polymarket/ts-sdk/main/packages/bindings/src/data/activity.ts",
        name="official-sdk-activity-types",
        expect_json=False,
    )
    assert isinstance(sdk_raw, bytes)
    sdk_text = sdk_raw.decode("utf-8", errors="strict")
    if "type: 'YIELD'" not in sdk_text or "type: 'REWARD'" not in sdk_text:
        raise ValueError("official SDK no longer distinguishes YIELD and REWARD")
    return {
        "annual_rate": _decimal_text(ANNUAL_RATE),
        "hourly_random_sampling": True,
        "daily_distribution": True,
        "rate_variable_at_polymarket_discretion": True,
        "future_caps_possible": True,
        "source": receipt,
        "official_sdk_activity_type_source": sdk_receipt,
    }


def run(*, raw_dir: Path) -> dict[str, object]:
    started_ms = time.time_ns() // 1_000_000
    client = _CaptureClient(raw_dir)
    _, position_summary, pair = _capture_market_state(client)
    payouts, payout_sources = _capture_yield_and_receipts(client)
    terms = _capture_official_terms(client)

    hourly_reward = POSITION_VALUE * ANNUAL_RATE / Decimal(365) / Decimal(24)
    sample_counts: list[int] = []
    for row in payouts:
        amount = Decimal(str(row["amount_pusd"]))
        matches = [
            count
            for count in range(25)
            if (hourly_reward * count).quantize(PAYOUT_SCALE, rounding=ROUND_DOWN)
            == amount
        ]
        if len(matches) != 1:
            raise ValueError(f"payout {amount} does not map to one hourly sample count")
        row["implied_sampled_hours"] = matches[0]
        sample_counts.append(matches[0])
    total_reward = sum(
        (Decimal(str(row["amount_pusd"])) for row in payouts), Decimal(0)
    )
    realized_rate = total_reward / POSITION_VALUE / Decimal(len(payouts)) * Decimal(365)
    if sum(sample_counts) != 328:
        raise ValueError("observed hourly sample total differs")
    four_percent_hourly_reward = (
        POSITION_VALUE * Decimal("0.04") / Decimal(365) / Decimal(24)
    )
    four_percent_matches = [
        [
            count
            for count in range(25)
            if (four_percent_hourly_reward * count).quantize(
                PAYOUT_SCALE, rounding=ROUND_DOWN
            )
            == Decimal(str(row["amount_pusd"]))
        ]
        for row in payouts
    ]

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": time.time_ns() // 1_000_000,
        "capture_started_at_ms": started_ms,
        "purpose": "public_market_direction_neutral_complete_set_holding_yield_reconciliation",
        "authority": {
            "credentials_used": False,
            "orders_placed": False,
            "transactions_sent": False,
            "funds_used": False,
            "live_trading_authority": False,
        },
        "source_correction": {
            "holding_activity_type": "YIELD",
            "generic_reward_activity_type": "REWARD",
            "prior_v2_public_reward_diagnostic": "invalid_for_holding_yield_attribution",
            "reason": "official SDK bindings distinguish account-level YIELD from account-level REWARD",
            "official_sdk_source_url": "https://github.com/Polymarket/ts-sdk/blob/main/packages/bindings/src/data/activity.ts",
        },
        "source_continuity": {
            "readiness_v2_path": "docs/model-research/polymarket/complete-set-holding-reward-readiness-v2.json",
            "readiness_v2_result_sha256": "2d3650d65f248294395fcac336c6650e0c6bc332cb490c6f0bac70bc11244e2c",
            "gasless_split_merge_source_url": "https://docs.polymarket.com/trading/gasless.md",
            "gasless_split_merge_source_response_sha256": "5fb14858dab5d92392941d84ff14ed46c4fb8ae49725a4ef1bda7c32c9be0255",
        },
        "official_terms": terms,
        "current_complete_set": position_summary,
        "position_rows": [
            {
                "outcome": str(row["outcome"]),
                "shares": _decimal_text(Decimal(str(row["size"]))),
                "average_price": _decimal_text(Decimal(str(row["avgPrice"]))),
                "current_price": _decimal_text(Decimal(str(row["curPrice"]))),
                "current_value_pusd": _decimal_text(Decimal(str(row["currentValue"]))),
                "mergeable": bool(row["mergeable"]),
                "token_id": str(row["asset"]),
            }
            for row in sorted(pair, key=lambda row: str(row["outcome"]))
        ],
        "observation": {
            "first_timestamp": min(OBSERVED_PAYOUTS),
            "last_timestamp": max(OBSERVED_PAYOUTS),
            "daily_payout_count": len(payouts),
            "positive_daily_payout_count": sum(
                Decimal(str(row["amount_pusd"])) > 0 for row in payouts
            ),
            "receipt_reconciliation_count": len(payouts),
            "total_reward_pusd": _decimal_text(total_reward),
            "base_position_value_pusd": _decimal_text(POSITION_VALUE),
            "realized_annualized_rate": _decimal_text(realized_rate),
            "official_hourly_reward_pusd": _decimal_text(hourly_reward),
            "implied_sampled_hours": sum(sample_counts),
            "possible_sampled_hours": len(payouts) * 24,
            "sample_count_histogram": {
                str(key): value for key, value in sorted(Counter(sample_counts).items())
            },
            "rate_adjudication": {
                "official_current_rate": _decimal_text(ANNUAL_RATE),
                "all_14_payouts_match_one_integer_hour_count_at_current_rate": True,
                "older_documentation_rate": "0.04",
                "all_14_payouts_match_one_integer_hour_count_at_older_rate": all(
                    len(matches) == 1 for matches in four_percent_matches
                ),
                "older_rate_integer_hour_matches": four_percent_matches,
                "conflict_resolved_by_current_official_terms_and_realized_payouts": True,
            },
            "payouts": payouts,
            "sources": payout_sources,
        },
        "economics": {
            "market_direction_exposure_of_equal_complete_set": "zero_at_resolution_before_operational_and_custody_risk",
            "complete_set_redemption_or_merge_value_pusd": _decimal_text(
                POSITION_VALUE
            ),
            "observed_reward_positive": True,
            "direct_relayer_split_merge_user_gas_pusd": "0",
            "validated_gross_edge_for_existing_idle_on_platform_pusd": True,
            "deployment_ready": False,
            "after_alternative_yield_edge_proven": False,
        },
        "verdict": {
            "status": "validated_positive_direction_neutral_gross_holding_yield_edge_for_existing_idle_on_platform_pusd",
            "accepted_structural_edge": True,
            "accepted_scope": "existing idle pUSD already on Polymarket while the official holding program and market eligibility remain active",
            "persistence_observed_days": len(payouts),
            "future_profit_guaranteed": False,
            "deployment_authorized": False,
        },
        "limitations": [
            "The official rate is variable and Polymarket may introduce payout caps.",
            "The public wallet proves equal mergeable balances and receipts, but not that this wallet originally created the pair with a split.",
            "Bridge, wrapping, withdrawal, custody, tax, and capital opportunity costs are outside this idle-on-platform scope.",
            "Fourteen positive days establish observed persistence, not a guaranteed future payout.",
            "Public evidence grants no authority to authenticate, fund, transact, or deploy.",
        ],
        "implementation": {
            "path": "tools/capture_polymarket_complete_set_holding_yield.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
            "pagination_rule": "fail if a 500-row page is full; never exceed documented activity offset 5000 without a new bounded contract",
            "stale_response_rule": "clear the response reference after every request and fail immediately on every request or parse error",
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(raw_dir=args.raw_dir)
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

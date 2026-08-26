"""Reconcile split-origin ETH and SOL complete sets with Polymarket YIELD."""

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


SCHEMA_VERSION = "polymarket-cross-asset-split-origin-holding-yield-v4"
PUSD_TOKEN = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
YIELD_DISTRIBUTOR = "0x607c8c9866ef3b4665c5a384188706be738d8bf8"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
POLYGON_RPC_URL = "https://polygon.drpc.org"
ANNUAL_RATE = Decimal("0.0325")
PAYOUT_SCALE = Decimal("0.0001")
CASES = {
    "ETH": {
        "event_slug": "what-price-will-ethereum-hit-before-2027",
        "wallet": "0x40db77fc612f07c73e9d4dde5f9fdeb154406c09",
        "condition_id": "0x201f51d2d892c41c5bfa6568a0a2f93ab2ea426e87dddfd5fb0191f7ec34a441",
        "question": "Will Ethereum reach $10,000 by December 31, 2026?",
        "shares": Decimal("440"),
        "lifecycle": (
            ("SPLIT", 1780642785, Decimal("550")),
            ("MERGE", 1781248221, Decimal("30")),
            ("MERGE", 1781691125, Decimal("80")),
        ),
        "payouts": {
            1786579971: Decimal("0.0391"),
            1786666447: Decimal("0.0391"),
            1786752649: Decimal("0.0391"),
            1786839064: Decimal("0.0391"),
            1786925494: Decimal("0.0391"),
            1787012049: Decimal("0.0391"),
            1787098440: Decimal("0.0391"),
            1787184990: Decimal("0.0375"),
            1787271322: Decimal("0.0391"),
            1787357542: Decimal("0.0375"),
            1787443831: Decimal("0.0391"),
            1787530303: Decimal("0.0391"),
            1787616793: Decimal("0.0375"),
            1787703046: Decimal("0.0342"),
        },
    },
    "SOL": {
        "event_slug": "what-price-will-solana-hit-before-2027",
        "wallet": "0xaa898d69f9abc17f0fdea7999e4c8d60beae2c28",
        "condition_id": "0x488999d62b8760a76d4e00e784f354a3bd03947c36073562ce65c0f3864185b3",
        "question": "Will Solana reach $600 by December 31, 2026?",
        "shares": Decimal("449"),
        "lifecycle": (
            ("SPLIT", 1780631026, Decimal("550")),
            ("MERGE", 1781579463, Decimal("21")),
            ("MERGE", 1781755120, Decimal("80")),
        ),
        "payouts": {
            1786580092: Decimal("0.0399"),
            1786666437: Decimal("0.0399"),
            1786752783: Decimal("0.0399"),
            1786839136: Decimal("0.0399"),
            1786925430: Decimal("0.0399"),
            1787012019: Decimal("0.0399"),
            1787098251: Decimal("0.0399"),
            1787184835: Decimal("0.0383"),
            1787271328: Decimal("0.0399"),
            1787357437: Decimal("0.0383"),
            1787444055: Decimal("0.0399"),
            1787530386: Decimal("0.0399"),
            1787616759: Decimal("0.0383"),
            1787703159: Decimal("0.0349"),
        },
    },
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


class _Client:
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
    ) -> tuple[object, dict[str, object]]:
        response = None
        try:
            response = self.session.request(
                method, url, params=params, json=body, timeout=30
            )
            response.raise_for_status()
            payload = response.content
            raw_path = self.raw_dir / f"{name}.raw"
            write_bytes_atomic(raw_path, payload)
            receipt = {
                "method": method,
                "url": response.url,
                "status_code": response.status_code,
                "response_bytes": len(payload),
                "response_sha256": _sha256(payload),
                "raw_path": str(raw_path.as_posix()),
            }
            try:
                return response.json(), receipt
            except requests.JSONDecodeError as exc:
                raise ValueError(f"{name} did not return JSON") from exc
        finally:
            response = None


def _topic_address(topic: object) -> str:
    value = str(topic).lower()
    if len(value) != 66 or not value.startswith("0x"):
        raise ValueError("ERC-20 address topic is invalid")
    return "0x" + value[-40:]


def _payout_transfer(
    receipt: Mapping[str, object], *, wallet: str, amount: Decimal
) -> bool:
    for raw_log in _list(receipt.get("logs"), name="receipt logs"):
        log = _mapping(raw_log, name="receipt log")
        topics = _list(log.get("topics"), name="receipt log topics")
        if (
            str(log.get("address") or "").lower() == PUSD_TOKEN
            and len(topics) == 3
            and str(topics[0]).lower() == TRANSFER_TOPIC
            and _topic_address(topics[1]) == YIELD_DISTRIBUTOR
            and _topic_address(topics[2]) == wallet
            and Decimal(int(str(log.get("data")), 16)) / Decimal(1_000_000) == amount
        ):
            return True
    return False


def _lifecycle_transfers(
    receipt: Mapping[str, object],
) -> list[tuple[str, str, Decimal]]:
    transfers: list[tuple[str, str, Decimal]] = []
    for raw_log in _list(receipt.get("logs"), name="receipt logs"):
        log = _mapping(raw_log, name="receipt log")
        topics = _list(log.get("topics"), name="receipt log topics")
        if (
            str(log.get("address") or "").lower() == PUSD_TOKEN
            and len(topics) == 3
            and str(topics[0]).lower() == TRANSFER_TOPIC
        ):
            transfers.append(
                (
                    _topic_address(topics[1]),
                    _topic_address(topics[2]),
                    Decimal(int(str(log.get("data")), 16)) / Decimal(1_000_000),
                )
            )
    return transfers


def _capture_event(
    client: _Client, *, asset: str, case: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    slug = str(case["event_slug"])
    raw, source = client.request(
        "GET",
        f"https://gamma-api.polymarket.com/events/slug/{slug}",
        name=f"{asset.lower()}-event",
    )
    event = _mapping(raw, name=f"{asset} event")
    markets = [
        _mapping(value, name=f"{asset} market")
        for value in _list(event.get("markets"), name=f"{asset} markets")
    ]
    eligible = [
        market
        for market in markets
        if market.get("holdingRewardsEnabled") is True
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
    ]
    selected = [
        market
        for market in eligible
        if str(market.get("conditionId") or "").lower() == case["condition_id"]
    ]
    if len(selected) != 1 or selected[0].get("question") != case["question"]:
        raise ValueError(f"{asset} selected eligible market identity differs")
    return selected[0], {
        "event_slug": slug,
        "eligible_market_count": len(eligible),
        "source": source,
    }


def _capture_case(
    client: _Client,
    *,
    asset: str,
    case: Mapping[str, object],
    market: Mapping[str, object],
) -> dict[str, object]:
    wallet = str(case["wallet"])
    condition = str(case["condition_id"])
    shares = case["shares"]
    assert isinstance(shares, Decimal)
    frozen_payouts = case["payouts"]
    assert isinstance(frozen_payouts, Mapping)
    lifecycle_contract = case["lifecycle"]
    assert isinstance(lifecycle_contract, tuple)

    positions_raw, positions_source = client.request(
        "GET",
        "https://data-api.polymarket.com/positions",
        name=f"{asset.lower()}-positions",
        params={"user": wallet, "limit": 500, "offset": 0, "sizeThreshold": 0},
    )
    positions = [
        _mapping(row, name=f"{asset} position")
        for row in _list(positions_raw, name=f"{asset} positions")
    ]
    if len(positions) != 2 or any(
        str(row.get("proxyWallet") or "").lower() != wallet
        or str(row.get("conditionId") or "").lower() != condition
        for row in positions
    ):
        raise ValueError(f"{asset} wallet no longer contains only the selected pair")
    if {str(row.get("outcome")) for row in positions} != {"Yes", "No"} or any(
        Decimal(str(row.get("size"))) != shares or row.get("mergeable") is not True
        for row in positions
    ):
        raise ValueError(f"{asset} positions are not equal and mergeable")
    current_value = sum(
        (Decimal(str(row.get("currentValue"))) for row in positions), Decimal(0)
    )
    if current_value != shares:
        raise ValueError(f"{asset} complete-set mark does not equal its shares")

    activity_raw, activity_source = client.request(
        "GET",
        "https://data-api.polymarket.com/activity",
        name=f"{asset.lower()}-activity",
        params={"user": wallet, "limit": 500, "offset": 0},
    )
    activities = [
        _mapping(row, name=f"{asset} activity")
        for row in _list(activity_raw, name=f"{asset} activities")
    ]
    if len(activities) >= 500:
        raise ValueError(f"{asset} activity reached its page limit")
    observation_start = min(int(timestamp) for timestamp in frozen_payouts)
    if any(
        row.get("type") != "YIELD"
        and int(row.get("timestamp", -1)) >= observation_start
        for row in activities
    ):
        raise ValueError(f"{asset} wallet changed after the observation began")
    observed_payouts = {
        int(row.get("timestamp", -1)): Decimal(str(row.get("usdcSize")))
        for row in activities
        if int(row.get("timestamp", -1)) in frozen_payouts
    }
    if observed_payouts != frozen_payouts:
        raise ValueError(f"{asset} frozen YIELD sequence differs")
    payout_rows = sorted(
        (row for row in activities if int(row.get("timestamp", -1)) in frozen_payouts),
        key=lambda row: int(row["timestamp"]),
    )
    lifecycle_rows = sorted(
        (
            row
            for row in activities
            if str(row.get("conditionId") or "").lower() == condition
            and row.get("type") in {"SPLIT", "MERGE", "TRADE"}
        ),
        key=lambda row: int(row["timestamp"]),
    )
    actual_lifecycle = tuple(
        (
            str(row["type"]),
            int(row["timestamp"]),
            Decimal(str(row["usdcSize"])),
        )
        for row in lifecycle_rows
    )
    if actual_lifecycle != lifecycle_contract:
        raise ValueError(f"{asset} split/merge lineage differs")
    split_amount = sum(
        (amount for kind, _, amount in actual_lifecycle if kind == "SPLIT"),
        Decimal(0),
    )
    merge_amount = sum(
        (amount for kind, _, amount in actual_lifecycle if kind == "MERGE"),
        Decimal(0),
    )
    if split_amount - merge_amount != shares:
        raise ValueError(f"{asset} split minus merges does not equal current shares")

    rpc_activities = payout_rows + lifecycle_rows
    rpc_envelopes: list[dict[str, object]] = []
    rpc_sources: list[dict[str, object]] = []
    for index, row in enumerate(rpc_activities):
        rpc_raw, rpc_source = client.request(
            "POST",
            POLYGON_RPC_URL,
            name=f"{asset.lower()}-polygon-receipt-{index:02d}",
            body={
                "jsonrpc": "2.0",
                "id": index,
                "method": "eth_getTransactionReceipt",
                "params": [str(row["transactionHash"])],
            },
        )
        envelope = _mapping(rpc_raw, name="RPC envelope")
        if envelope.get("id") != index:
            raise ValueError(f"{asset} Polygon receipt ID differs")
        rpc_envelopes.append(envelope)
        rpc_sources.append(rpc_source)
    reconciled_payouts: list[dict[str, object]] = []
    reconciled_lifecycle: list[dict[str, object]] = []
    hourly_reward = shares * ANNUAL_RATE / Decimal(365) / Decimal(24)
    sample_counts: list[int] = []
    for index, activity in enumerate(rpc_activities):
        envelope = rpc_envelopes[index]
        receipt = _mapping(envelope.get("result"), name="transaction receipt")
        tx_hash = str(activity["transactionHash"]).lower()
        if (
            envelope.get("error") is not None
            or receipt.get("status") != "0x1"
            or str(receipt.get("transactionHash") or "").lower() != tx_hash
        ):
            raise ValueError(f"{asset} transaction receipt failed")
        amount = Decimal(str(activity["usdcSize"]))
        if activity["type"] == "YIELD":
            if not _payout_transfer(receipt, wallet=wallet, amount=amount):
                raise ValueError(f"{asset} YIELD transfer does not reconcile")
            matches = [
                count
                for count in range(25)
                if (hourly_reward * count).quantize(PAYOUT_SCALE, rounding=ROUND_DOWN)
                == amount
            ]
            if len(matches) != 1:
                raise ValueError(f"{asset} payout does not map to one sample count")
            sample_counts.append(matches[0])
            reconciled_payouts.append(
                {
                    "timestamp": int(activity["timestamp"]),
                    "amount_pusd": _decimal_text(amount),
                    "implied_sampled_hours": matches[0],
                    "transaction_hash": tx_hash,
                    "block_number": int(str(receipt["blockNumber"]), 16),
                }
            )
            continue
        transfers = _lifecycle_transfers(receipt)
        if activity["type"] == "SPLIT":
            required = {
                (wallet, PUSD_TOKEN, amount),
                (PUSD_TOKEN, ZERO_ADDRESS, amount),
            }
            if not required.issubset(set(transfers)):
                raise ValueError(f"{asset} SPLIT pUSD burn does not reconcile")
        elif (ZERO_ADDRESS, wallet, amount) not in transfers:
            raise ValueError(f"{asset} MERGE pUSD mint does not reconcile")
        reconciled_lifecycle.append(
            {
                "type": str(activity["type"]),
                "timestamp": int(activity["timestamp"]),
                "amount_pusd": _decimal_text(amount),
                "transaction_hash": tx_hash,
                "block_number": int(str(receipt["blockNumber"]), 16),
            }
        )

    total_reward = sum(
        (Decimal(row["amount_pusd"]) for row in reconciled_payouts), Decimal(0)
    )
    realized_rate = (
        total_reward / shares / Decimal(len(reconciled_payouts)) * Decimal(365)
    )
    return {
        "asset": asset,
        "wallet": wallet,
        "condition_id": condition,
        "question": str(market["question"]),
        "shares_per_outcome": _decimal_text(shares),
        "current_complete_set_value_pusd": _decimal_text(current_value),
        "current_position_row_count": len(positions),
        "current_wallet_contains_only_this_pair": True,
        "position_rows": [
            {
                "outcome": str(row["outcome"]),
                "shares": _decimal_text(Decimal(str(row["size"]))),
                "current_price": _decimal_text(Decimal(str(row["curPrice"]))),
                "current_value_pusd": _decimal_text(Decimal(str(row["currentValue"]))),
                "mergeable": bool(row["mergeable"]),
            }
            for row in sorted(positions, key=lambda row: str(row["outcome"]))
        ],
        "split_origin_lineage": {
            "split_amount_pusd": _decimal_text(split_amount),
            "merged_amount_pusd": _decimal_text(merge_amount),
            "remaining_complete_sets": _decimal_text(split_amount - merge_amount),
            "no_selected_condition_trade_rows": True,
            "transactions": reconciled_lifecycle,
            "all_receipts_successful_and_pusd_flows_reconciled": True,
        },
        "observation": {
            "daily_payout_count": len(reconciled_payouts),
            "positive_daily_payout_count": sum(
                Decimal(row["amount_pusd"]) > 0 for row in reconciled_payouts
            ),
            "total_reward_pusd": _decimal_text(total_reward),
            "realized_annualized_rate": _decimal_text(realized_rate),
            "official_hourly_reward_pusd": _decimal_text(hourly_reward),
            "implied_sampled_hours": sum(sample_counts),
            "possible_sampled_hours": len(sample_counts) * 24,
            "sample_count_histogram": {
                str(key): value for key, value in sorted(Counter(sample_counts).items())
            },
            "no_non_yield_activity_during_observation": True,
            "payouts": reconciled_payouts,
        },
        "sources": {
            "positions": positions_source,
            "activity": activity_source,
            "polygon_receipts": rpc_sources,
            "total_account_activity_row_count": len(activities),
        },
    }


def run(*, raw_dir: Path) -> dict[str, object]:
    client = _Client(raw_dir)
    started_ms = time.time_ns() // 1_000_000
    cases: list[dict[str, object]] = []
    event_sources: list[dict[str, object]] = []
    for asset, raw_case in CASES.items():
        market, event_source = _capture_event(client, asset=asset, case=raw_case)
        event_sources.append({"asset": asset, **event_source})
        cases.append(
            _capture_case(
                client,
                asset=asset,
                case=raw_case,
                market=market,
            )
        )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": time.time_ns() // 1_000_000,
        "capture_started_at_ms": started_ms,
        "purpose": "public_cross_asset_split_origin_complete_set_holding_yield_reconciliation",
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_placed": False,
            "transactions_sent": False,
            "trading_authority": False,
        },
        "official_rate": _decimal_text(ANNUAL_RATE),
        "event_sources": event_sources,
        "cases": cases,
        "cross_asset_summary": {
            "assets": [case["asset"] for case in cases],
            "case_count": len(cases),
            "split_origin_reconciled_case_count": sum(
                case["split_origin_lineage"][
                    "all_receipts_successful_and_pusd_flows_reconciled"
                ]
                is True
                for case in cases
            ),
            "daily_payout_count": sum(
                int(case["observation"]["daily_payout_count"]) for case in cases
            ),
            "positive_daily_payout_count": sum(
                int(case["observation"]["positive_daily_payout_count"])
                for case in cases
            ),
            "payout_receipt_reconciliation_count": sum(
                len(case["observation"]["payouts"]) for case in cases
            ),
            "observed_rate_range": [
                min(
                    str(case["observation"]["realized_annualized_rate"])
                    for case in cases
                ),
                max(
                    str(case["observation"]["realized_annualized_rate"])
                    for case in cases
                ),
            ],
        },
        "source_continuity": {
            "btc_reconciliation_path": "docs/model-research/polymarket/complete-set-holding-yield-reconciliation-v3-2026-08-26.json",
            "btc_reconciliation_result_sha256": "48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe",
            "official_terms_url": "https://help.polymarket.com/en/articles/13364459-holding-rewards",
        },
        "verdict": {
            "status": "validated_cross_asset_split_origin_direction_neutral_gross_holding_yield_edge_for_existing_idle_on_platform_pusd",
            "accepted_structural_edge_strengthened": True,
            "split_origin_limitation_closed_for_eth_and_sol": True,
            "deployment_ready": False,
            "future_profit_guaranteed": False,
            "trading_authority": False,
        },
        "limitations": [
            "The official rate is variable and Polymarket may introduce payout caps.",
            "Public wallets prove the mechanism, not eligibility or costs for an owned account.",
            "Bridge, wrapping, withdrawal, custody, tax, and alternative-yield costs remain outside the idle-on-platform scope.",
            "Fourteen observed days per asset do not guarantee future payments.",
        ],
        "implementation": {
            "path": "tools/capture_polymarket_cross_asset_holding_yield.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
            "pagination_rule": "fail on a full 500-row account response and never reuse a response after an error",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

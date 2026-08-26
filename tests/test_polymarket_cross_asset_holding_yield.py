from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "complete-set-holding-yield-cross-asset-v4-2026-08-26.json"
)
TOOL_PATH = ROOT / "tools" / "capture_polymarket_cross_asset_holding_yield.py"
EXPECTED_RESULT_HASH = (
    "eda29a314218e1724e39984e2712a4351d9e697503d4583d391c89a060ba53ea"
)
EXPECTED_TOOL_HASH = "8fdbf652f4f6df257240df2bc0ad5cbd59cea5f7e6d278a56cff033fc465bf2c"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_cross_asset_artifact_and_implementation_are_source_bound() -> None:
    artifact = _load()
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"]["sha256"] == EXPECTED_TOOL_HASH
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH
    assert artifact["authority"] == {
        "credentials_used": False,
        "funds_used": False,
        "orders_placed": False,
        "trading_authority": False,
        "transactions_sent": False,
    }


def test_eth_and_sol_complete_sets_have_reconciled_split_origin() -> None:
    cases = {case["asset"]: case for case in _load()["cases"]}
    assert set(cases) == {"ETH", "SOL"}
    expected = {
        "ETH": (Decimal("550"), Decimal("110"), Decimal("440")),
        "SOL": (Decimal("550"), Decimal("101"), Decimal("449")),
    }
    for asset, (split, merged, remaining) in expected.items():
        case = cases[asset]
        lineage = case["split_origin_lineage"]
        assert Decimal(lineage["split_amount_pusd"]) == split
        assert Decimal(lineage["merged_amount_pusd"]) == merged
        assert Decimal(lineage["remaining_complete_sets"]) == remaining
        assert split - merged == remaining
        assert Decimal(case["shares_per_outcome"]) == remaining
        assert Decimal(case["current_complete_set_value_pusd"]) == remaining
        assert lineage["no_selected_condition_trade_rows"] is True
        assert lineage["all_receipts_successful_and_pusd_flows_reconciled"] is True
        assert [row["type"] for row in lineage["transactions"]] == [
            "SPLIT",
            "MERGE",
            "MERGE",
        ]
        assert {row["outcome"] for row in case["position_rows"]} == {"Yes", "No"}
        assert all(row["mergeable"] for row in case["position_rows"])


def test_every_cross_asset_payout_is_positive_and_matches_the_rate_formula() -> None:
    artifact = _load()
    expected_totals = {"ETH": Decimal("0.5377"), "SOL": Decimal("0.5488")}
    for case in artifact["cases"]:
        observation = case["observation"]
        payouts = observation["payouts"]
        assert observation["daily_payout_count"] == 14
        assert observation["positive_daily_payout_count"] == 14
        assert observation["no_non_yield_activity_during_observation"] is True
        assert len(payouts) == 14
        assert len({row["transaction_hash"] for row in payouts}) == 14
        assert all(Decimal(row["amount_pusd"]) > 0 for row in payouts)
        assert (
            sum(Decimal(row["amount_pusd"]) for row in payouts)
            == expected_totals[case["asset"]]
        )
        assert observation["implied_sampled_hours"] == 330
        assert observation["possible_sampled_hours"] == 336
        assert observation["sample_count_histogram"] == {"21": 1, "23": 3, "24": 10}

    summary = artifact["cross_asset_summary"]
    assert summary["case_count"] == 2
    assert summary["split_origin_reconciled_case_count"] == 2
    assert summary["daily_payout_count"] == 28
    assert summary["positive_daily_payout_count"] == 28
    assert summary["payout_receipt_reconciliation_count"] == 28


def test_verdict_strengthens_one_scoped_gross_edge_without_authorizing_deployment() -> (
    None
):
    verdict = _load()["verdict"]
    assert verdict["accepted_structural_edge_strengthened"] is True
    assert verdict["split_origin_limitation_closed_for_eth_and_sol"] is True
    assert verdict["deployment_ready"] is False
    assert verdict["future_profit_guaranteed"] is False
    assert verdict["trading_authority"] is False

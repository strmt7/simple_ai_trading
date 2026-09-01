from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_profitability_durability_audit_is_source_bound_and_exhaustive() -> None:
    audit = _load(AUDIT_PATH)
    registry_path = ROOT / str(audit["source_binding"]["registry_path"])
    registry = _load(registry_path)

    assert _self_hash(audit) == audit["result_sha256"]
    assert _self_hash(registry) == registry["result_sha256"]
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["source_binding"]["accepted_edge_count"]
        == registry["accepted_edge_count"]
    )

    groups = audit["classification"]
    accepted_count = registry["accepted_edge_count"]
    assert sum(group["count"] for group in groups) == accepted_count
    ordinals = [ordinal for group in groups for ordinal in group["edge_ordinals"]]
    assert sorted(ordinals) == list(range(1, accepted_count + 1))
    assert len(ordinals) == len(set(ordinals))

    decision = audit["decision"]
    assert (
        decision["historically_source_demonstrated_recurring_direct_cash_edge_count"]
        == 1
    )
    assert decision["stable_current_account_qualified_after_all_cost_edge_count"] == 0
    assert (
        decision["accepted_scope_is_not_deployment_ready_count"] == accepted_count
    )
    assert (
        decision[
            "standalone_incremental_cash_without_independent_trade_borrow_or_external_user_count"
        ]
        == 9
    )
    assert accepted_count in decision[
        "standalone_incremental_cash_without_independent_trade_borrow_or_external_user_ordinals"
    ]


def test_profitability_frontier_metrics_match_canonical_sources() -> None:
    audit = _load(AUDIT_PATH)
    opposite_lock = audit["frontier"][
        "polymarket_historical_out_of_sample_existing_inventory_lock_candidate"
    ]
    candidate = _load(ROOT / opposite_lock["source_path"])
    correction = _load(ROOT / opposite_lock["fee_rounding_correction_source_path"])
    robustness = _load(ROOT / opposite_lock["robustness_source_path"])

    assert candidate["result_sha256"] == opposite_lock["result_sha256"]
    assert (
        correction["result_sha256"]
        == opposite_lock["fee_rounding_correction_result_sha256"]
    )
    assert (
        correction["analysis"]["corrected_locked_pnl_pusd"]
        == opposite_lock[
            "corrected_one_tick_stressed_full_fee_no_rebate_locked_cashflow_pusd"
        ]
    )
    assert correction["analysis"]["nonpositive_lock_count_after_correction"] == 0
    assert robustness["result_sha256"] == opposite_lock["robustness_result_sha256"]
    assert robustness["analysis"]["predeclared_gate_passed"] is False
    gate = robustness["analysis"]["predeclared_gate_scenario"]
    assert (
        gate["positive_lock_count"]
        == opposite_lock["five_tick_plus_0_05_pusd_fixed_cost_positive_lock_count"]
    )
    assert (
        gate["positive_lock_fraction"]
        == opposite_lock["five_tick_plus_0_05_pusd_fixed_cost_positive_lock_fraction"]
    )
    assert (
        gate["total_locked_pnl_pusd"]
        == opposite_lock[
            "five_tick_plus_0_05_pusd_fixed_cost_total_locked_cashflow_pusd"
        ]
    )

    polymarket = audit["frontier"]["polymarket_recurring_cash_leader"]
    polymarket_source = _load(ROOT / polymarket["source_path"])
    portfolio = polymarket_source["cross_asset_portfolio"]

    assert polymarket_source["result_sha256"] == polymarket["result_sha256"]
    for field in (
        "demonstrated_principal_pusd",
        "net_reward_after_direct_split_merge_cost_pusd",
        "positive_daily_payout_count",
        "possible_daily_payout_count",
        "principal_weighted_realized_annualized_rate",
    ):
        assert portfolio[field] == polymarket[field]

    binance = audit["frontier"]["binance_historical_persistence_leader"]
    binance_source = _load(ROOT / binance["source_path"])
    economics = binance_source["economic_summary"]
    sensitivity = binance_source["official_direct_cost_adjudication"]

    assert binance_source["result_sha256"] == binance["result_sha256"]
    for field in (
        "aligned_daily_close_count",
        "compound_annualized_return_fraction",
        "elapsed_days",
    ):
        assert economics[field] == binance[field]
    assert (
        sensitivity[
            "full_history_compound_return_after_ten_percent_extra_principal_drag_fraction"
        ]
        == binance[
            "after_ten_percent_annual_opportunity_cost_on_extra_principal_fraction"
        ]
    )

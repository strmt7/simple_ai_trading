"""Record the September 2 bStock listing and CRWD top-book outcome."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT_PATH = (
    ROOT
    / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)
INVENTORY_PATH = (
    ROOT
    / "docs/model-research/action-value/binance-bstock-sep2-listing-inventory-result-v2-2026-09-04.json"
)
TOPBOOK_PATH = (
    ROOT
    / "docs/model-research/action-value/binance-crwdb-bstock-perpetual-topbook-result-v1-2026-09-04.json"
)
EXPECTED_REGISTRY_HASH = "51dba2cf9c5f61efa650cab4db66edc37189cd9aeb7ddfd97dbfe9fdd0e16698"
EXPECTED_AUDIT_HASH = "e575d0084546053bc1ab2c07e9d23553796d4ef4dd8fc29bc2edcef314575801"
INVENTORY_HASH = "ba5ebb29ce89c8cc09bde5066bbe4a0cfc7dc11fa926f5a00de6321054caba26"
TOPBOOK_HASH = "a02367c4c0f7461a29ba857f947dc870ab6539be36419d7b068e5e685b58c744"
MECHANISM = "bstock_reference_conversion_and_delta_neutral_perpetual_funding"
TERMINAL_FAMILY = "binance_CRWDBUSDT_CRWDUSDT_new_listing_topbook_2026_09_04"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        allow_nan=False,
    )
    write_bytes_atomic(path, (payload + "\n").encode("ascii"))


def main() -> int:
    registry = _load(REGISTRY_PATH)
    audit = _load(AUDIT_PATH)
    inventory = _load(INVENTORY_PATH)
    topbook = _load(TOPBOOK_PATH)
    if (
        registry.get("result_sha256") != EXPECTED_REGISTRY_HASH
        or _canonical_hash(registry, "result_sha256") != EXPECTED_REGISTRY_HASH
    ):
        raise ValueError("unexpected registry starting hash")
    if (
        audit.get("result_sha256") != EXPECTED_AUDIT_HASH
        or _canonical_hash(audit, "result_sha256") != EXPECTED_AUDIT_HASH
    ):
        raise ValueError("unexpected audit starting hash")
    if inventory.get("result_sha256") != INVENTORY_HASH:
        raise ValueError("inventory result declared hash mismatch")
    inventory_body = dict(inventory)
    inventory_body.pop("result_sha256")
    if hashlib.sha256(_canonical_json(inventory_body).encode()).hexdigest() != INVENTORY_HASH:
        raise ValueError("inventory result canonical hash mismatch")
    if (
        topbook.get("result_sha256") != TOPBOOK_HASH
        or _canonical_hash(topbook, "result_sha256") != TOPBOOK_HASH
    ):
        raise ValueError("top-book result hash mismatch")
    if inventory.get("new_tickers") != ["CRWD", "MRNA", "SQQQ", "STX"]:
        raise ValueError("unexpected listing population")
    if [row["ticker"] for row in inventory["matching_unscreened_pairs"]] != [
        "CRWD",
        "MRNA",
        "SQQQ",
    ]:
        raise ValueError("unexpected exact matching perpetual population")
    if topbook["economics"]["passes_fixed_rejection_gate"] is not False:
        raise ValueError("CRWD unexpectedly passed the top-book gate")

    matches = [
        row
        for row in registry["prioritized_hypotheses"]
        if row.get("mechanism") == MECHANISM
    ]
    if len(matches) != 1 or matches[0].get("priority_rank") != 12:
        raise ValueError("rank 12 hypothesis not uniquely found")
    hypothesis = matches[0]
    additions = [
        {
            "path": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": INVENTORY_HASH,
        },
        {
            "path": str(TOPBOOK_PATH.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": TOPBOOK_HASH,
        },
    ]
    existing_paths = {item["path"] for item in hypothesis["canonical_artifacts"]}
    if any(item["path"] in existing_paths for item in additions):
        raise ValueError("rank 12 artifacts already registered")
    hypothesis["canonical_artifacts"].extend(additions)
    hypothesis["current_status"] = (
        "the_official_September_2_listing_trigger_advanced_the_public_bStock_"
        "inventory_from_68_to_72_exact_rows_with_new_CRWD_MRNA_SQQQ_and_STX_"
        "tickers_all_at_exact_one_multiplier_current_futures_metadata_proved_"
        "active_exact_CRWD_MRNA_and_SQQQ_TradFi_perpetual_matches_while_STXUSDT_"
        "was_not_an_equity_TradFi_match_the_frozen_lexicographic_selector_chose_"
        "CRWD_one_262ms_skew_two_request_top_book_prefilter_found_CRWDBUSDT_spot_"
        "ask_and_CRWDUSDT_perpetual_bid_both_exactly_213_07_USDT_so_gross_entry_"
        "headroom_was_zero_and_the_50_bip_fixed_stress_made_it_negative_1_06535_"
        "USDT_per_share_or_negative_50_bips_no_depth_funding_fee_account_credential_"
        "order_fund_or_protected_capture_access_MRNA_and_SQQQ_remain_unscreened_"
        "and_are_not_inferred_from_CRWD_not_an_edge_or_deployment_ready"
    )
    hypothesis["next_action"] = (
        "do_not_repeat_reprice_or_depth_capture_CRWDBUSDT_CRWDUSDT_and_do_not_"
        "adaptively_cherry_pick_MRNA_or_SQQQ_after_observing_CRWD_at_or_after_"
        "2026_09_08T13_35_00Z_after_the_official_September_7_US_market_holiday_"
        "freeze_one_separate_lexicographically_next_MRNABUSDT_MRNAUSDT_two_request_"
        "top_book_gate_with_the_same_50_bip_rejection_rule_then_request_depth_and_"
        "adverse_funding_only_if_it_survives_otherwise_wait_for_a_new_official_"
        "exact_multiplier_bStock_listing_and_matching_previously_unscreened_TradFi_"
        "perpetual_or_material_economics_change_or_explicit_account_read_only_"
        "authority_for_the_separate_stock_transfer_FPSL_or_permanent_scope_questions"
    )
    shortcut = (
        "generalizing_the_CRWD_zero_basis_to_MRNA_or_SQQQ_or_screening_them_"
        "adaptively_before_the_prospectively_recorded_2026_09_08T13_35_00Z_trigger"
    )
    if shortcut not in hypothesis["prohibited_shortcuts"]:
        hypothesis["prohibited_shortcuts"].append(shortcut)

    terminals = registry["terminal_do_not_repeat"]
    if any(row.get("family") == TERMINAL_FAMILY for row in terminals):
        raise ValueError("terminal family already registered")
    terminals.append(
        {
            "family": TERMINAL_FAMILY,
            "reason": (
                "the_official_September_2_listing_and_current_inventories_proved_a_"
                "new_exact_one_multiplier_CRWDBUSDT_bStock_and_active_exact_CRWDUSDT_"
                "TradFi_perpetual_the_frozen_lexicographic_selector_chose_CRWD_and_"
                "one_262ms_skew_two_request_top_book_snapshot_found_the_spot_ask_and_"
                "perpetual_bid_both_exactly_213_07_USDT_zero_gross_headroom_and_"
                "negative_1_06535_USDT_per_share_or_negative_50_bips_after_fixed_"
                "stress_do_not_refetch_reprice_request_depth_funding_fees_accounts_"
                "credentials_orders_or_funds_for_this_exact_CRWD_observation_MRNA_"
                "and_SQQQ_are_not_terminalized_or_inferred_from_it"
            ),
            "canonical_result_sha256": TOPBOOK_HASH,
        }
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry["updated_at_utc"] = now
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    _write(REGISTRY_PATH, registry)

    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["binance_bstock_sep2_listing_crwd_topbook_trigger"] = (
        "Rank 12's official September 2 listing trigger is consumed through the "
        "deterministically selected CRWD top-book stage. Inventory advanced from 68 "
        "to 72 rows with exact-one CRWD MRNA SQQQ and STX bStocks; CRWD MRNA and "
        "SQQQ have exact active equity TradFi perpetuals, while STX does not. The "
        "262 ms CRWDBUSDT ask and CRWDUSDT bid snapshot was exactly 213.07 on both "
        "legs, giving zero gross headroom and negative 50 bips after frozen stress. "
        "Do not repeat CRWD or infer MRNA and SQQQ from it. At or after "
        "2026-09-08T13:35:00Z, after the official September 7 U.S. market holiday, "
        "freeze the lexicographically next MRNA two-request top-book gate; do not "
        "request depth or funding unless it first survives the same 50-bip hurdle."
    )
    audit["updated_at_utc"] = now
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    _write(AUDIT_PATH, audit)
    print(
        _canonical_json(
            {
                "registry_result_sha256": registry["result_sha256"],
                "terminal_family_count": len(terminals),
                "audit_result_sha256": audit["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

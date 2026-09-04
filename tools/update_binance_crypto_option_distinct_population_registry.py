"""Record the consumed distinct crypto-option population in canonical ledgers."""

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
POPULATION_RESULT_PATH = (
    ROOT
    / "docs/model-research/action-value/binance-crypto-option-population-gate-result-v2-2026-09-04.json"
)
PRICE_RESULT_PATH = (
    ROOT
    / "docs/model-research/action-value/binance-crypto-option-distinct-price-prefilter-result-v2-2026-09-04.json"
)
EXPECTED_REGISTRY_HASH = "4ceb5de5af67264d50129ae2b44ee33e4ded59cdb3a32eef4ffe85459002e3ee"
EXPECTED_AUDIT_HASH = "d0736b1f3da7af118516993839f271d10dcadc3a3edefaef9202a0d05f6ee2f7"
POPULATION_HASH = "8c10dd5d039bb0753207e86ee14e6761b46f160cf21a2114ba7fd6470632972c"
PRICE_HASH = "fbd10642d35ed469b7cc7f5554681d90d074be71fe718498738011af2fcb3b8e"
MECHANISM = (
    "binance_long_crypto_option_opposite_USDT_perpetual_terminal_payoff_lower_bound"
)
TERMINAL_FAMILY = (
    "binance_BTC_ETH_SOL_long_option_opposite_perpetual_exact_distinct_"
    "354_symbol_population_2026_09_04"
)


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
    population = _load(POPULATION_RESULT_PATH)
    price = _load(PRICE_RESULT_PATH)
    if _canonical_hash(registry, "result_sha256") != EXPECTED_REGISTRY_HASH:
        raise ValueError("unexpected registry starting hash")
    if registry.get("result_sha256") != EXPECTED_REGISTRY_HASH:
        raise ValueError("registry declared starting hash mismatch")
    if _canonical_hash(audit, "result_sha256") != EXPECTED_AUDIT_HASH:
        raise ValueError("unexpected audit starting hash")
    if audit.get("result_sha256") != EXPECTED_AUDIT_HASH:
        raise ValueError("audit declared starting hash mismatch")
    if (
        _canonical_hash(population, "result_sha256") != POPULATION_HASH
        or population.get("result_sha256") != POPULATION_HASH
    ):
        raise ValueError("population result hash mismatch")
    if (
        _canonical_hash(price, "result_sha256") != PRICE_HASH
        or price.get("result_sha256") != PRICE_HASH
    ):
        raise ValueError("price result hash mismatch")
    if price["population"] != {
        "after_fixed_stress_positive_count": 0,
        "distinct_unscreened_symbol_count": 354,
        "gross_positive_count": 1,
        "positive_entry_side_count": 226,
        "ticker_present_count": 354,
    }:
        raise ValueError("unexpected price-screen outcome")

    hypotheses = registry["prioritized_hypotheses"]
    matches = [row for row in hypotheses if row.get("mechanism") == MECHANISM]
    if len(matches) != 1 or matches[0].get("priority_rank") != 47:
        raise ValueError("rank 47 hypothesis not uniquely found")
    hypothesis = matches[0]
    additions = [
        {
            "path": str(POPULATION_RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": POPULATION_HASH,
        },
        {
            "path": str(PRICE_RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": PRICE_HASH,
        },
    ]
    existing_paths = {item["path"] for item in hypothesis["canonical_artifacts"]}
    if any(item["path"] in existing_paths for item in additions):
        raise ValueError("rank 47 artifacts were already registered")
    hypothesis["canonical_artifacts"].extend(additions)
    hypothesis["current_status"] = (
        "unaccepted_direction_independent_long_crypto_option_plus_equal_opposite_"
        "USDT_perpetual_terminal_payoff_lower_bound_the_prospectively_frozen_"
        "2026_09_04_population_gate_found_356_active_symbols_absent_from_the_1576_"
        "symbol_baseline_then_excluded_the_two_already_consumed_BTC_261225_94000_"
        "call_and_put_symbols_and_proved_zero_overlap_with_the_consumed_508_symbol_"
        "population_leaving_354_genuinely_unscreened_symbols_one_frozen_245ms_skew_"
        "two_request_price_prefilter_found_all_354_tickers_226_positive_entry_side_"
        "rows_one_gross_positive_BTC_260905_80500_put_at_11_10_USDT_per_BTC_but_"
        "zero_survivors_after_the_33_5_bip_fixed_stress_the_best_became_negative_"
        "255_455815_USDT_or_negative_32_1049826_bips_no_depth_funding_fee_account_"
        "credential_order_fund_or_protected_capture_access_not_an_edge_or_"
        "deployment_ready"
    )
    hypothesis["next_action"] = (
        "do_not_repeat_rebuild_refresh_reprice_or_depth_capture_the_exhausted_"
        "1410_symbol_baseline_the_exact_508_symbol_August_31_delta_the_exact_two_"
        "symbol_September_1_delta_or_the_exact_354_symbol_September_4_distinct_"
        "population_reopen_only_for_a_later_distinct_active_BTC_ETH_or_SOL_option_"
        "population_after_2026_09_04T18_49_36_306Z_or_a_material_fee_settlement_"
        "tick_depth_funding_expiry_basis_or_capital_rule_change_or_an_independently_"
        "observed_nonpolling_displayed_terminal_floor_strictly_above_every_"
        "applicable_cost_then_apply_the_same_rejection_first_sequence"
    )
    shortcut = (
        "calling_the_11_10_USDT_gross_BTC_260905_80500_put_row_an_edge_after_it_"
        "failed_the_frozen_33_5_bip_stress_by_255_455815_USDT_per_BTC"
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
                "the_prospectively_frozen_population_gate_found_356_symbols_absent_"
                "from_the_1576_symbol_baseline_then_source_bound_deduplication_"
                "removed_two_already_screened_symbols_and_proved_zero_overlap_with_"
                "the_consumed_508_symbol_population_leaving_354_genuinely_"
                "unscreened_active_BTC_ETH_SOL_options_one_frozen_two_request_"
                "ticker_and_side_specific_perpetual_book_prefilter_screened_all_"
                "354_with_245ms_start_skew_226_had_positive_entry_sides_and_only_"
                "BTC_260905_80500_P_had_a_positive_gross_floor_at_11_10_USDT_per_"
                "BTC_but_it_became_negative_255_455815_USDT_or_negative_32_1049826_"
                "bips_after_the_frozen_33_5_bip_fixed_stress_so_zero_survived_do_"
                "not_refetch_reprice_poll_request_depth_funding_fees_accounts_"
                "credentials_orders_or_funds_reopen_only_for_rank_47s_exact_later_"
                "population_or_material_economics_trigger"
            ),
            "canonical_result_sha256": PRICE_HASH,
        }
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry["updated_at_utc"] = now
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    _write(REGISTRY_PATH, registry)

    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["binance_crypto_option_opposite_perpetual_terminal_trigger"] = (
        "Rank 47's exact September 4 population is terminal. A frozen metadata gate "
        "found 356 active BTC ETH or SOL options absent from the 1576-symbol baseline; "
        "source-bound deduplication removed two already screened symbols and proved no "
        "overlap with the consumed 508-symbol population, leaving 354. One frozen "
        "two-request price prefilter screened all 354 with 245 ms start skew. Of 226 "
        "positive-entry rows, only BTC-260905-80500-P had a positive gross floor, "
        "11.10 USDT per BTC, and it became negative 255.455815 USDT or negative "
        "32.1049826 bips after the frozen 33.5-bip stress. Do not refetch, poll, "
        "reprice, or request depth, funding, fees, accounts, credentials, orders, or "
        "funds. Reopen only for a distinct later active population after the retained "
        "2026-09-04T18:49:36.306Z snapshot, a material economics or architecture "
        "change, or an independently observed nonpolling floor above every cost."
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

"""One-use canonical registration, preserving both preceding ledger bytes."""

from datetime import datetime, timezone
import json

from tools.update_binance_crypto_option_distinct_population_registry import (
    ROOT,
    REGISTRY_PATH,
    AUDIT_PATH,
    _canonical_hash,
    _load,
    _write,
)
from tools.verify_paradex_index_publication import verify


FAMILY = "binance_paradex_fixed_base_index_boundaries_aug27_sep4_2026"
PRIOR_REGISTRY = "0591e4c2699a18a1cba1a8168a175779722e38c4e2c1b4dde3fa0bdba9866b41"
PRIOR_AUDIT = "db071c55485a23d507ad8a009483710901c28f5d165ff5993b1ce85ae60aa4db"


def main():
    verified = verify()
    registry, audit = _load(REGISTRY_PATH), _load(AUDIT_PATH)
    for payload, expected in ((registry, PRIOR_REGISTRY), (audit, PRIOR_AUDIT)):
        if (
            payload["result_sha256"] != expected
            or _canonical_hash(payload, "result_sha256") != expected
        ):
            raise ValueError("prior ledger identity differs; do not repeat mutation")
    if (
        verified["result_sha256"]
        != "f2aa72d9d65d28bdd0f012deb85d2e65c5085d7c62ab9c69aee916401b6f99f9"
    ):
        raise ValueError("frozen study differs")
    if any(row["family"] == FAMILY for row in registry["terminal_do_not_repeat"]):
        raise ValueError("population already registered")
    family = next(
        r for r in registry["prioritized_hypotheses"] if r["priority_rank"] == 59
    )
    summary = (
        "The distinct August 27 00:00 through September 4 16:00 2026 nominal "
        "population is consumed, including the frozen five-minute post-boundary "
        "windows. All 84 public responses passed source gates (1,308,306 bytes): "
        "27 index boundaries and one Binance history per BTC ETH SOL asset, "
        "78 fixed-base intervals and 13/6/7 chronological roles. Training selected "
        "short Paradex and long Binance for each asset. All nine par-valued gross "
        "role funding spreads were positive at 0.7858-7.5875 bips, but each was "
        "negative after even the frozen 20-bip execution allowance alone. No "
        "survivor passed the full unchanged 80-bip reserves plus 10-percent-annual "
        "total-two-leg capital stress. These are stress diagnostics, not actual "
        "fees, converted realized account PnL, current profit or a family-wide "
        "negative-EV proof. No basis books accounts orders or protected data."
    )
    family["prior_90_bin_source_failure_status"] = family["current_status"]
    family["current_status"] = summary
    family["index_boundary_study_status"] = summary
    for path, field in (
        ("docs/review/2026-09-04/paradex-index-contract.json", "contract_sha256"),
        ("docs/review/2026-09-04/paradex-index-study/result.json", "result_sha256"),
    ):
        payload = _load(ROOT / path)
        if _canonical_hash(payload, field) != payload[field]:
            raise ValueError("new canonical artifact hash differs")
        family["canonical_artifacts"].append(
            {"path": path, "result_sha256": payload[field]}
        )
    family["next_action"] += (
        "_also_do_not_refetch_reprice_interpolate_fill_missing_windows_or_reorient_"
        "the_consumed_2026_08_27_through_2026_09_04_16_05_01_index_population_"
        "a_new_study_requires_the_literal_nonoverlapping_population_or_material_"
        "architecture_trigger_and_an_explicit_information_gain_case_not_a_rolling_"
        "retry_to_hunt_a_winner"
    )
    family["prohibited_shortcuts"].extend(
        [
            "using_averaged_hourly_rates_instead_of_available_same_unit_funding_index_deltas",
            "treating_USDC_USDT_par_valuation_as_observed_conversion_or_stress_allowances_as_actual_costs",
            "discarding_partial_source_windows_or_reorienting_the_consumed_Aug27_Sep4_population",
        ]
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": FAMILY,
            "reason": summary,
            "canonical_result_sha256": verified["result_sha256"],
        }
    )
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry["updated_at_utc"] = stamp
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["paradex_fixed_base_index_boundaries"] = summary
    audit["updated_at_utc"] = stamp
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    base = ROOT / "docs/review/2026-09-04/paradex-index-publication"
    base.mkdir(exist_ok=False)
    for source, name in (
        (REGISTRY_PATH, "registry-before.json"),
        (AUDIT_PATH, "durability-before.json"),
    ):
        with (base / name).open("xb") as output:
            output.write(source.read_bytes())
    _write(REGISTRY_PATH, registry)
    _write(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "accepted_scopes": registry["accepted_edge_count"],
                "hypotheses": len(registry["prioritized_hypotheses"]),
                "terminal_observations": len(registry["terminal_do_not_repeat"]),
                "registry_sha256": registry["result_sha256"],
                "audit_sha256": audit["result_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()

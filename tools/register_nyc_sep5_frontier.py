"""Register the exact terminal event; preserve preceding mutable ledger bytes."""

from datetime import datetime, timezone
from decimal import Decimal
import json

from tools.update_binance_crypto_option_distinct_population_registry import (
    ROOT,
    REGISTRY_PATH,
    AUDIT_PATH,
    _canonical_hash,
    _load,
    _write,
)

BASE = ROOT / "docs/review/2026-09-04/nyc-sep5-frontier"
FAMILY = "polymarket_nyc_september5_complete_long_only_frontier"
PRIOR_REGISTRY = "39a69bce3a5544cc33d2d6383cc7f9578ff8c2305fe36caa1c57b0e5f053de1a"
PRIOR_AUDIT = "333c90c4cc471c6f2fa5544b2eecd367f8288009abe8278a3dbf9cf4120bcfe2"


def main() -> None:
    registry, audit = _load(REGISTRY_PATH), _load(AUDIT_PATH)
    for value, expected in ((registry, PRIOR_REGISTRY), (audit, PRIOR_AUDIT)):
        if (
            value["result_sha256"] != expected
            or _canonical_hash(value, "result_sha256") != expected
        ):
            raise ValueError("ledger starting state differs; do not repeat mutation")
    bindings = []
    for name, field in (
        ("contract.json", "contract_sha256"),
        ("result.json", "result_sha256"),
        ("fee-audit-result.json", "result_sha256"),
    ):
        path = BASE / name
        value = _load(path)
        if _canonical_hash(value, field) != value[field]:
            raise ValueError("new evidence self-hash differs")
        bindings.append(
            {"path": path.relative_to(ROOT).as_posix(), "result_sha256": value[field]}
        )
    result = _load(BASE / "result.json")
    fees = _load(BASE / "fee-audit-result.json")
    if (
        result["result_sha256"]
        != "f53fb916a213fcd0dbddfedcb6be3473b104bae145333cf7543e5bac7a61cbd1"
    ):
        raise ValueError("exact event result differs")
    if (
        fees["result_sha256"]
        != "d26bbed784edc4aad898aec01744f0b02a00569fed5871619427b8e8df13ab24"
    ):
        raise ValueError("exact fee audit differs")
    if result["screen"]["after_fee_one_tick_candidate_count"] != 0 or any(
        Decimal(row["after_configured_fee_without_ticks_pUSD"]) > 0
        for row in fees["rows"]
    ):
        raise ValueError("terminal economics differ")
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    if any(row["family"] == FAMILY for row in registry["terminal_do_not_repeat"]):
        raise ValueError("exact event already registered")
    summary = (
        "One prospectively fixed NYC September 5 event GET returned all 11 markets; "
        "11 YES and 7 NO sides were price-complete. All 17 frontier rows were screened. "
        "Four gross-positive rows failed the one-tick gate. A retained fee-only audit "
        "also rejected all 17 rows: the strongest gross-positive seven-NO row had "
        "0.060 pUSD gross headroom but 0.15394 configured fee at five shares, leaving "
        "negative 0.09394 pUSD before any adverse tick. Four NO sides remain incomplete. "
        "Do not refetch, reprice, fill missing sides, request books or reinterpret "
        "missing stress results as zero fees. No edge or account/order authority."
    )
    family["canonical_artifacts"].extend(bindings)
    family["nyc_september5_frontier_status"] = summary
    family["prohibited_shortcuts"].append(
        "retrying_or_book_capturing_the_consumed_NYC_September_5_event_or_calling_"
        "its_0_060_pUSD_gross_seven_NO_headroom_profitable_after_0_15394_configured_fees"
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": FAMILY,
            "reason": summary,
            "canonical_result_sha256": result["result_sha256"],
        }
    )
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry["updated_at_utc"] = stamp
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["polymarket_nyc_september5_frontier"] = summary
    audit["updated_at_utc"] = stamp
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    for source, name in (
        (REGISTRY_PATH, "registry-before.json"),
        (AUDIT_PATH, "durability-before.json"),
    ):
        with (BASE / name).open("xb") as stream:
            stream.write(source.read_bytes())
    _write(REGISTRY_PATH, registry)
    _write(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "registry_sha256": registry["result_sha256"],
                "audit_sha256": audit["result_sha256"],
                "terminal_observations": len(registry["terminal_do_not_repeat"]),
            }
        )
    )


if __name__ == "__main__":
    main()

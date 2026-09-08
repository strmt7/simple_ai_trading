"""Adjudicate one retained USDT campaign and update its existing allocation lane."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.capture_public_source_contract import _canonical_hash

BASE = Path("docs/review/2026-09-08/usdt-flexible")
REGISTRY = Path("docs/model-research/structural-edge-priority-registry-v1.json")
AUDIT = Path(
    "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _text(node: dict) -> str:
    if node.get("node") == "text":
        return html.unescape(node["text"])
    return "".join(_text(child) for child in node.get("child", []))


def _load(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads(path.read_bytes())
    if value[field] != _canonical_hash(value, field):
        raise ValueError(f"canonical binding failed: {path}")
    return value


def main() -> None:
    source = _load(BASE / "source-result.json")
    contract = _load(BASE / "source-contract.json", "contract_sha256")
    raw = (BASE / "raw.json").read_bytes()
    if (
        source["result_sha256"]
        != "f7bfce2e17fe58eea0ae68e8b37f1490c6b16e2775d893926d4bdf6017675db5"
    ):
        raise ValueError("unexpected source result")
    if (
        hashlib.sha256(raw).hexdigest()
        != source["capture"]["receipt"]["response_sha256"]
    ):
        raise ValueError("raw receipt differs")
    if (
        source["contract"]["sha256"] != contract["contract_sha256"]
        or not source["source_gate"]["passed"]
    ):
        raise ValueError("source admission differs")
    payload = json.loads(raw)
    article = payload["data"]
    if (
        payload["code"] != "000000"
        or article["code"] != "2d943377ecae4ef18a33cc90047a1d80"
    ):
        raise ValueError("article identity differs")
    body = json.loads(article["body"])
    blocks = [_text(node).replace("\xa0", " ") for node in body["child"]]
    text = "\n".join(blocks)
    phrases = [
        "2026-09-08 00:00:00 (UTC) to 2026-09-22 23:59:59 (UTC)",
        "Subscription Amount ≤ 800 USDT",
        "Subscription Amount > 800 USDT",
        "4% Bonus Tiered APR",
        "approximately 3% Real-Time APR",
        "0.01 USDTUnlimited",
        "only master accounts qualify",
        "complete identity verification",
        "first-come, first-served",
        "next day starting from 00:00 (UTC)",
        "stop the accrual of Bonus Tiered APR rewards on the redeemed amount for that day",
        "A large amount of redemption requests might delay redemption temporarily",
        "without prior notice",
        "subscription amount of each user has an upper limit",
    ]
    if any(phrase not in text for phrase in phrases):
        raise ValueError("retained terms differ from adjudication")
    with localcontext() as context:
        context.prec = 60
        daily = Decimal(800) * Decimal("0.04") / 365
        sensitivity = {
            "day_count_basis": 365,
            "day_count_basis_source_proved": False,
            "classification": "conditional simple-APR illustration, not contracted or realized payout",
            "complete_eligible_days_if_subscribed_September8_held_through_September22": 14,
            "bonus_USDT_per_complete_eligible_day_at_cap": str(daily),
            "fourteen_day_bonus_USDT_at_cap": str(daily * 14),
            "fourteen_day_bonus_bips_of_800_USDT": str(
                Decimal("0.04") / 365 * 14 * 10000
            ),
            "variable_base_APR_credited": False,
            "future_guaranteed_bonus_floor_USDT": "0",
        }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_result_sha256": source["result_sha256"],
        "source_contract_sha256": contract["contract_sha256"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "reviewer_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "article_code": article["code"],
        "canonical_url": contract["adjudication"]["discovery_only_url"],
        "article_publish_time_ms": article["publishDate"],
        "article_last_update_ms": article["lastUpdateTime"],
        "terms": {
            "start_utc": "2026-09-08T00:00:00Z",
            "end_utc": "2026-09-22T23:59:59Z",
            "bonus_apr_percent": "4",
            "bonus_principal_cap_USDT": "800",
            "minimum_subscription_USDT": "0.01",
            "headline_apr_percent": "7",
            "approximate_variable_base_apr_percent": "3",
            "global_old_campaign_expired": True,
            "old_bonus_cap_USDT": "500",
            "old_campaign_evidence": "docs/model-research/action-value/binance-usdt-flexible-current-bonus-overlay-v1-2026-08-26.json",
            "no_old_or_regional_bonus_stacking": True,
        },
        "economic_sensitivity": sensitivity,
        "unresolved": [
            "Account region, master-account identity verification eligibility, quota and existing idle funds are unknown",
            "Table says unlimited maximum subscription while general terms mention an upper limit; actual capacity is unresolved",
            "Random snapshot, redemption-day reward loss and delayed redemptions constrain usable liquidity",
            "Bonus terms may change or terminate without notice; variable base APR is not a fixed floor",
            "Actual credited days, fees, best alternative, taxes and future distributions are unproved",
        ],
        "decision": "material renewal and cap increase for existing same-asset allocation overlay; not a scalable standalone trading strategy",
        "accepted_edge": False,
        "accepted_scope_count_delta": 0,
        "account_qualified": False,
        "deployment_ready": False,
        "new_network_requests_by_adjudicator": 0,
        "retry_trigger": "New independently observed material terms change or the exact 2026-09-22T23:59:59Z campaign end; no daily polling or alias refetch. Signed or funded work requires separate authority and current account evidence.",
    }
    previous = _load(Path(result["terms"]["old_campaign_evidence"]))
    if (
        previous["eligibility_and_terms"]["global"]["bonus_apr_percent_first_500_USDT"]
        != "4"
    ):
        raise ValueError("prior bonus lineage differs")
    result["prior_result_sha256"] = previous["result_sha256"]
    registry, audit = _load(REGISTRY), _load(AUDIT)
    if (
        registry["result_sha256"]
        != "97b48ded00c92bd356cd15700a1802e3cb4099af99fce1e930f55d04d6e349b7"
        or audit["result_sha256"]
        != "becf4054b7c8191e70e8bbcc4bcdc586bfe2929a5ba0e278c9380a5b96547087"
    ):
        raise ValueError("starting registry lineage differs")
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    path = BASE / "result.json"
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        )
    row = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 8
    )
    row["current_status"] += (
        " September 8 USDT Flexible renewal retained: bonus remains 4 percent, cap rises from 500 to 800 USDT, September 8-22. Approximately 3 percent variable base is not fixed income. Fourteen eligible days illustrate 1.22739726 USDT capped bonus under a non-source-proved 365-day convention. Actual eligibility, capacity, alternatives and payouts remain unqualified."
    )
    row["retry_trigger"] += (
        "_or_2026_09_22T23_59_59Z_USDT_September8_campaign_end_or_independently_observed_material_change_to_that_exact_campaign"
    )
    row["next_action"] += (
        " For the consumed September 8 USDT renewal, do not refetch or extrapolate the headline rate; require actual same-account eligibility/quota and independently idle USDT plus best alternative and native credited payouts before treating it as deployable income."
    )
    row["canonical_artifacts"].append(
        {"path": path.as_posix(), "result_sha256": result["result_sha256"]}
    )
    registry["updated_at_utc"] = result["created_at_utc"]
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["usdt_flexible_september8_renewal"] = (
        result["decision"] + ". " + result["retry_trigger"]
    )
    audit["updated_at_utc"] = result["created_at_utc"]
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    for output, value in ((REGISTRY, registry), (AUDIT, audit)):
        write_bytes_atomic(
            output,
            (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
        )
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "registry_sha256": registry["result_sha256"],
                "audit_sha256": audit["result_sha256"],
                "terminal_count": len(registry["terminal_do_not_repeat"]),
                "economic_sensitivity": sensitivity,
            }
        )
    )


if __name__ == "__main__":
    main()

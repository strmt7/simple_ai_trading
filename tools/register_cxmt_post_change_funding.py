"""Verify retained CXMT evidence and register its one-use terminal outcome."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.review_cxmt_post_change_funding import _hash, evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("docs/review/2026-09-08/cxmt-post-change")
REGISTRY = Path("docs/model-research/structural-edge-priority-registry-v1.json")
AUDIT = Path(
    "docs/model-research/action-value/"
    "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)
EXPECTED_REGISTRY = "d8e23d654aa344e233c31eb1fc02e242285101f09cdb06a2f26d0ad7b594578d"
EXPECTED_AUDIT = "196ec88d530aa6f1a658abad03dbf97e377c612c9370d59f7d4984524704ebd4"
EXPECTED_RESULT = "fb72b55226a32a45e568d9a9bf62a06e40bd7450e193f2ba52a93e3bdc8ee210"
FAMILY = "CXMT_fixed_long_Polymarket_short_Binance_first_twelve_four_hour_settlements_2026_09_04_06"


def load(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads((ROOT / path).read_bytes())
    if value[field] != _hash(value, field):
        raise ValueError(f"canonical hash mismatch: {path}")
    return value


def main() -> None:
    registry, audit = load(REGISTRY), load(AUDIT)
    result, plan = load(BASE / "result.json"), load(BASE / "plan.json", "plan_sha256")
    if (registry["result_sha256"], audit["result_sha256"], result["result_sha256"]) != (
        EXPECTED_REGISTRY,
        EXPECTED_AUDIT,
        EXPECTED_RESULT,
    ):
        raise ValueError("unexpected starting ledger or result")
    if result["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("result plan binding mismatch")
    for path, digest in plan["source_bindings"].items():
        if sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source byte mismatch: {path}")
    raw = []
    for name, embedded in zip(
        ("polymarket", "binance"), result["source_results"], strict=True
    ):
        source = load(BASE / f"{name}-source-result.json")
        contract = load(BASE / f"{name}-contract.json", "contract_sha256")
        if (
            source != embedded
            or source["contract"]["sha256"] != contract["contract_sha256"]
        ):
            raise ValueError("embedded source or contract mismatch")
        receipt = source["capture"]["receipt"]
        body = (ROOT / receipt["raw_path"]).read_bytes()
        if (
            sha256(body).hexdigest() != receipt["response_sha256"]
            or len(body) != receipt["response_bytes"]
        ):
            raise ValueError("raw receipt mismatch")
        if receipt["status_code"] != 200 or not source["source_gate"]["passed"]:
            raise ValueError("source admission mismatch")
        raw.append(json.loads(body))
    if evaluate(*raw) != result["analysis"]:
        raise ValueError("retained economic reconstruction mismatch")
    if (
        result["status"] != "terminal_history_rejection"
        or result["accepted_edge"]
        or result["candidate_for_books"]
    ):
        raise ValueError("unexpected terminal disposition")
    if any(
        role["positive_count"] or role["passes"]
        for role in result["analysis"]["roles"].values()
    ):
        raise ValueError("unexpected role outcome")
    rows = [
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 43
    ]
    if (
        len(rows) != 1
        or rows[0]["mechanism"]
        != "polymarket_binance_TradFi_perpetual_fixed_orientation_cross_venue_funding_spread"
    ):
        raise ValueError("rank 43 identity mismatch")
    if len(registry["terminal_do_not_repeat"]) != 192:
        raise ValueError("unexpected terminal count")
    row = rows[0]
    row["current_status"] = (
        "The original CXMT September 4-6 first-twelve-settlement trigger is consumed. "
        "Two public GETs on September 8 retained 48 Polymarket hourly and 12 regular "
        "Binance four-hour observations. Fixed long Polymarket / short Binance gross "
        "carry was negative in all 12 observations. Training/validation/test gross "
        "was -4.3037/-5.6561/-4.1478 bips; all roles failed execution, capital and "
        "quote-stress hurdles. This is an equal-notional funding-rate proxy, not "
        "realized fixed-base PnL or a family-wide impossibility claim. Prior top-five "
        "and HK0625/SHEIN studies remain consumed. No qualified edge."
    )
    row["retry_trigger"] = (
        "A later material Polymarket or Binance TradFi perpetual funding cash-flow, "
        "fee, market-session, instrument-conversion or execution-architecture change; "
        "the September 6 time gate has been consumed and is not a rolling-window trigger."
    )
    row["next_action"] = (
        "Do not refetch, reverse orientation, extend the September 4-6 window, "
        "reprice or request books/accounts for this rejected sample. Reopen only "
        "after the exact later material-change trigger with a separately frozen "
        "causal study; preserve all historical samples, protected data and authority gates."
    )
    row["canonical_artifacts"].extend(
        [
            {
                "path": (BASE / "plan.json").as_posix(),
                "plan_sha256": plan["plan_sha256"],
            },
            {
                "path": (BASE / "result.json").as_posix(),
                "result_sha256": result["result_sha256"],
            },
        ]
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": FAMILY,
            "reason": row["current_status"] + " " + row["next_action"],
            "canonical_result_sha256": result["result_sha256"],
        }
    )
    now = datetime.now(UTC).isoformat()
    registry["updated_at_utc"] = now
    registry["result_sha256"] = _hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["cxmt_post_change_first_twelve_settlements"] = (
        row["current_status"] + " " + row["next_action"]
    )
    audit["updated_at_utc"] = now
    audit["result_sha256"] = _hash(audit, "result_sha256")
    for path, value in ((REGISTRY, registry), (AUDIT, audit)):
        write_bytes_atomic(
            ROOT / path,
            (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
        )
    print(
        json.dumps(
            {
                "registry_sha256": registry["result_sha256"],
                "audit_sha256": audit["result_sha256"],
                "terminal_count": len(registry["terminal_do_not_repeat"]),
                "retained_result_reconstructed": True,
                "network_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()

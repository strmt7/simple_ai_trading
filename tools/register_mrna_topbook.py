"""Apply the reviewed MRNA terminal amendment with exact prior-state checks."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import subprocess

from tools.screen_mrna_bstock_topbook import DIRECTORY, ROOT, load_bound, run
from tools.update_binance_bstock_sep2_listing_registry import (
    AUDIT_PATH,
    REGISTRY_PATH,
    _canonical_hash,
    _write,
)


def main() -> None:
    plan = json.loads((DIRECTORY / "registry-amendment-plan.json").read_bytes())
    registry = load_bound(REGISTRY_PATH, "result_sha256", plan["prior_registry_sha256"])
    audit = load_bound(AUDIT_PATH, "result_sha256", plan["prior_audit_sha256"])
    before_registry, before_audit = deepcopy(registry), deepcopy(audit)
    for path, expected in ((REGISTRY_PATH, registry), (AUDIT_PATH, audit)):
        raw = subprocess.check_output(
            [
                "git",
                "show",
                f"{plan['prior_git_checkpoint']}:{path.relative_to(ROOT).as_posix()}",
            ],
            cwd=ROOT,
        )
        if json.loads(raw) != expected:
            raise ValueError("prior checkpoint does not preserve the starting ledger")
    contract = load_bound(
        DIRECTORY / "contract.json", "contract_sha256", plan["contract_sha256"]
    )
    result = load_bound(
        DIRECTORY / "result.json", "result_sha256", plan["result_sha256"]
    )
    rebuilt = run(offline=True)
    if (
        rebuilt
        != {key: value for key, value in result.items() if key != "result_sha256"}
        or rebuilt["status"] != "exact_observation_rejected"
    ):
        raise ValueError("terminal result does not reconstruct")
    if len(registry["terminal_do_not_repeat"]) != plan["prior_terminal_count"]:
        raise ValueError("terminal population changed")
    rows = [
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 12
    ]
    if len(rows) != 1 or any(
        row["family"] == plan["terminal_family"]
        for row in registry["terminal_do_not_repeat"]
    ):
        raise ValueError("rank or terminal identity conflict")
    row = rows[0]
    previous_row = deepcopy(row)
    row["pre_mrna_september8_routing"] = {
        field: row[field] for field in ("current_status", "next_action")
    }
    row["current_status"] = plan["summary"]
    row["next_action"] = plan["next_action"]
    row["prohibited_shortcuts"].append(plan["prohibited_shortcut"])
    row["canonical_artifacts"].extend(
        [
            {
                "path": (DIRECTORY / "contract.json").relative_to(ROOT).as_posix(),
                "result_sha256": contract["contract_sha256"],
            },
            {
                "path": (DIRECTORY / "result.json").relative_to(ROOT).as_posix(),
                "result_sha256": result["result_sha256"],
            },
        ]
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": plan["terminal_family"],
            "reason": plan["summary"],
            "canonical_result_sha256": result["result_sha256"],
        }
    )
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry["updated_at_utc"] = stamp
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    route = "binance_bstock_sep2_listing_crwd_topbook_trigger"
    audit["routing"][route] = plan["summary"] + " " + plan["next_action"]
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["updated_at_utc"] = stamp
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    # Prove that only the declared mutations occurred, before either write.
    check_registry, check_audit = deepcopy(registry), deepcopy(audit)
    check_registry["prioritized_hypotheses"][
        registry["prioritized_hypotheses"].index(row)
    ] = previous_row
    check_registry["terminal_do_not_repeat"].pop()
    check_audit["routing"][route] = before_audit["routing"][route]
    check_audit["source_binding"] = before_audit["source_binding"]
    for changed, previous in (
        (check_registry, before_registry),
        (check_audit, before_audit),
    ):
        for field in ("updated_at_utc", "result_sha256"):
            changed[field] = previous[field]
        if changed != previous:
            raise ValueError("undeclared ledger mutation")
    _write(REGISTRY_PATH, registry)
    _write(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "registry_sha256": registry["result_sha256"],
                "audit_sha256": audit["result_sha256"],
                "terminal_count": len(registry["terminal_do_not_repeat"]),
            }
        )
    )


if __name__ == "__main__":
    main()

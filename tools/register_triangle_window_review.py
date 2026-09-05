"""Apply the one-use session routing exception and terminal evidence, not promotion."""

import argparse
from datetime import datetime, timezone
import hashlib
import json

from tools.update_binance_crypto_option_distinct_population_registry import (
    ROOT,
    REGISTRY_PATH,
    AUDIT_PATH,
    _canonical_hash,
    _load,
    _write,
)

BASE = ROOT / "docs/review/2026-09-05/triangle-window"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "terminal"))
    phase = parser.parse_args().phase
    registry, audit = _load(REGISTRY_PATH), _load(AUDIT_PATH)
    for value in (registry, audit):
        if value["result_sha256"] != _canonical_hash(value, "result_sha256"):
            raise ValueError("source ledger self-hash mismatch")
    if audit["source_binding"]["registry_result_sha256"] != registry["result_sha256"]:
        raise ValueError("source ledger cross-binding mismatch")
    row = next(
        x for x in registry["prioritized_hypotheses"] if x["priority_rank"] == 16
    )
    key = "september5_session_only_prospective_window"
    prior = {"registry": registry["result_sha256"], "audit": audit["result_sha256"]}
    if phase == "freeze":
        if (
            key in row
            or registry["result_sha256"]
            != "ce6f65bd0965ecc4172d9e5f375b55739e8ae3f9d7acbd23be7fbbadfc67ca2f"
        ):
            raise ValueError("wrong starting ledger or repeated freeze")
        row[key] = {
            "status": "frozen_unconsumed",
            "contract_path": (BASE / "contract.json").relative_to(ROOT).as_posix(),
            "contract_file_sha256": hashlib.sha256(
                (BASE / "contract.json").read_bytes()
            ).hexdigest(),
            "scope": "One disjoint major-asset rate-product diagnostic; old results, account and execution gates unchanged",
        }
        row["retry_trigger"] += (
            "_or_exactly_once_under_the_frozen_2026_09_05_session_only_prospective_window_contract_before_its_start_deadline"
        )
        row["next_action"] = (
            "run_only_the_frozen_September_5_session_window_once_then_stop_without_automatic_resampling_or_fee_depth_account_order_access; "
            + row["next_action"]
        )
    else:
        if row[key]["status"] != "frozen_unconsumed":
            raise ValueError("window not in frozen state")
        result = _load(BASE / "result.json")
        if (
            result["contract_sha256"] != row[key]["contract_file_sha256"]
            or result["accepted_edge"] is not False
        ):
            raise ValueError("terminal contract or acceptance mismatch")
        row[key].update(
            status="consumed_terminal",
            result_path=(BASE / "result.json").relative_to(ROOT).as_posix(),
            result_file_sha256=hashlib.sha256(
                (BASE / "result.json").read_bytes()
            ).hexdigest(),
            complete=result["complete"],
            samples=len(result["observations"]),
            diagnostic_survivors=sum(
                x["all_twelve_positive"] for x in result.get("cycles", [])
            ),
        )
        row["next_action"] = (
            "September_5_session_window_consumed_do_not_repeat_extend_or_select_another_window; "
            + row["next_action"].split("; ", 1)[1]
        )
        registry["terminal_do_not_repeat"].append(
            {
                "family": "binance_major_triangle_prospective_quote_window_2026_09_05",
                "reason": "One session-only future rate-product diagnostic consumed; no automatic repeat, qualified profit or account authority",
                "canonical_result_sha256": row[key]["result_file_sha256"],
            }
        )
    stamp = datetime.now(timezone.utc).isoformat()
    registry["updated_at_utc"] = stamp
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"][key] = row[key]
    audit["updated_at_utc"] = stamp
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    with (BASE / f"{phase}-registry-amendment.json").open("x", encoding="ascii") as out:
        json.dump(
            {
                "phase": phase,
                "prior": prior,
                "next": {
                    "registry": registry["result_sha256"],
                    "audit": audit["result_sha256"],
                },
                "previous_committed_state": "4944ecba7c649dbae94a8ae020b37377a397c6a0",
                "scope": row[key],
            },
            out,
            indent=2,
            sort_keys=True,
        )
    _write(REGISTRY_PATH, registry)
    _write(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "phase": phase,
                "registry": registry["result_sha256"],
                "audit": audit["result_sha256"],
                "terminal_observations": len(registry["terminal_do_not_repeat"]),
            }
        )
    )


if __name__ == "__main__":
    main()

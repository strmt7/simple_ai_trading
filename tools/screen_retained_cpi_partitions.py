"""One-use offline within-event partition screen; no annual/monthly mapping."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from tools.screen_polymarket_exact_negrisk_long_only_frontier import _markets, _screen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/cpi-partitions"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(payload):
    return sha(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def preflight(plan):
    for path, expected in plan["source_sha256"].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen source identity differs: " + path)
    selected = []
    for item in plan["events"]:
        event = json.loads((ROOT / item["raw_path"]).read_bytes())
        if str(event.get("id")) != item["id"] or event.get("slug") != item["slug"]:
            raise ValueError("event identity differs")
        markets = _markets(event, item["market_count"])
        if {m["groupItemTitle"] for m in markets} != set(item["labels"]):
            raise ValueError("complete declared partition differs")
        if {m["description"] for m in markets} != {event["description"]}:
            raise ValueError("within-event rules differ")
        for phrase in item["required_rule_phrases"]:
            if phrase not in event["description"]:
                raise ValueError("precision/measure/fallback semantics differ")
        if any(
            Decimal(str(m["orderMinSize"])) > Decimal(plan["quantity_shares"])
            for m in markets
        ):
            raise ValueError("frozen quantity below a market minimum")
        selected.append((event, markets))
    return selected


def adjudicate(plan):
    selected = preflight(plan)
    rows, populations = [], []
    for event, markets in selected:
        event_rows, counts = _screen(event, markets, Decimal(plan["quantity_shares"]))
        rows.extend(event_rows)
        populations.append(
            {"event_id": str(event["id"]), **counts, "frontier_rows": len(event_rows)}
        )
    return {
        "populations": populations,
        "rows": rows,
        "gross_positive_rows": sum(r["passes_strict_metadata_gate"] for r in rows),
        "fee_tick_positive_rows": sum(r["passes_fee_and_one_tick_gate"] for r in rows),
        "promotion_eligible": False,
        "accepted_edge": False,
        "profitability_claim": False,
        "cross_event_packages": 0,
        "network_requests": 0,
        "account_requests": 0,
        "protected_access": False,
        "historical_result_changes": False,
        "decision": "Retained metadata is exploratory and rejection-only. Any positive row needs a distinct prospective confirmation with exact rules, synchronized finite depth and every cost. No refresh, missing-side repair, book or account request follows this run.",
    }


def main():
    plan_path = BASE / "plan.json"
    plan = json.loads(plan_path.read_bytes())
    if (BASE / "result.json").exists():
        raise FileExistsError("screen consumed")
    with (BASE / "journal.jsonl").open("x", encoding="ascii", newline="\n") as journal:

        def record(payload):
            journal.write(
                json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **payload})
                + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())

        record({"phase": "started", "plan_sha256": sha(plan_path.read_bytes())})
        try:
            result = {
                "schema_version": "retained-cpi-partition-screen-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": sha(plan_path.read_bytes()),
                **adjudicate(plan),
            }
            result["result_sha256"] = canonical(result)
            with (BASE / "result.json").open("xb") as stream:
                stream.write(json.dumps(result, indent=2).encode() + b"\n")
            record({"phase": "completed", "result_sha256": result["result_sha256"]})
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "result_sha256",
                            "populations",
                            "gross_positive_rows",
                            "fee_tick_positive_rows",
                        )
                    }
                )
            )
        except Exception as exc:
            record(
                {"phase": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            )
            raise


if __name__ == "__main__":
    main()

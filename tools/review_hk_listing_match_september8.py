"""Close the exact September 7 Hong Kong listing counterpart gate from retained bytes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.capture_public_source_contract import _canonical_hash

BASE = Path("docs/review/2026-09-08/hk-listing-match")
REGISTRY = Path("docs/model-research/structural-edge-priority-registry-v1.json")
AUDIT = Path(
    "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _load(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads(path.read_bytes())
    if value[field] != _canonical_hash(value, field):
        raise ValueError("canonical source binding differs")
    return value


def main() -> None:
    sources, payloads = {}, {}
    for name in ("announcement", "polymarket-instruments"):
        source = _load(BASE / f"{name}-source-result.json")
        contract = _load(BASE / f"{name}-contract.json", "contract_sha256")
        raw = (BASE / f"{name}-raw.json").read_bytes()
        receipt = source["capture"]["receipt"]
        journal = [
            json.loads(line)
            for line in (BASE / f"{name}-journal.jsonl").read_bytes().splitlines()
        ]
        if (
            source["contract"]["sha256"] != contract["contract_sha256"]
            or not source["source_gate"]["passed"]
            or receipt["status_code"] != 200
            or hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]
            or len(journal) != 2
            or journal[0]["phase"] != "intent"
            or journal[0]["contract_sha256"] != contract["contract_sha256"]
            or journal[1] != receipt
        ):
            raise ValueError("capture or journal binding differs")
        sources[name] = source
        payloads[name] = json.loads(raw)
    announcement = payloads["announcement"]
    if (
        announcement["code"] != "000000"
        or announcement["data"]["code"] != "89a035c3ee0e4b7782bf0089323d8e78"
    ):
        raise ValueError("announcement identity differs")
    if (
        sources["announcement"]["capture"]["receipt"]["completed_at_ms"]
        > sources["polymarket-instruments"]["capture"]["receipt"]["requested_at_ms"]
    ):
        raise ValueError("request sequence differs")
    rows = payloads["polymarket-instruments"]
    if not isinstance(rows, list) or len(rows) != 67:
        raise ValueError("retained inventory shape differs")
    identities = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("instrument_id")) is not int:
            raise ValueError("instrument identity invalid")
        if any(
            not isinstance(row.get(key), str) or not row[key]
            for key in ("symbol", "base_asset", "quote_asset")
        ):
            raise ValueError("instrument asset identity missing")
        identities.append(
            {
                key: row[key]
                for key in ("instrument_id", "symbol", "base_asset", "quote_asset")
            }
        )
    for key in ("instrument_id", "symbol"):
        if len({row[key] for row in identities}) != len(rows):
            raise ValueError("duplicate instrument identity")
    matches = [row for row in identities if row["base_asset"] in {"BYD", "HK0992"}]
    if matches:
        raise ValueError(
            "unexpected candidate requires separate settlement qualification"
        )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_results": sources,
        "reviewer_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "announced_population": [
            {
                "symbol": "BYDUSDT",
                "underlying": "BYD CO. LTD. - H SHARES (HKEX: 1211)",
                "contract_type": "USDT-Priced",
                "launch_utc": "2026-09-07T02:00:00Z",
            },
            {
                "symbol": "HK0992USDT",
                "underlying": "LENOVO GROUP LTD. (HKEX: 0992)",
                "contract_type": "Quanto",
                "launch_utc": "2026-09-07T02:05:00Z",
            },
        ],
        "inventory_count": len(rows),
        "identity_projection": identities,
        "exact_base_matches": matches,
        "status": "terminal_no_exact_counterpart",
        "limits": "No exact counterpart in this complete returned inventory. No assertion about other exchanges, future listings, aliases or current Binance execution availability. Quanto and linear settlement are not interchangeable.",
        "economic_requests": 0,
        "account_requests": 0,
        "orders_or_funds": 0,
        "accepted_edge": False,
        "deployment_ready": False,
        "retry_trigger": "Material independently observed exact Polymarket counterpart listing with matching share class, unit and settlement, or a separately source-proved conversion architecture. Do not repeat this inventory, request funding or prices, or use a similar ticker as a hedge.",
    }
    registry, audit = _load(REGISTRY), _load(AUDIT)
    if (
        registry["result_sha256"]
        != "274f93cc1239b8786fcc17d04324385a8a04333281048160a1903c4a083df5af"
        or audit["result_sha256"]
        != "0e95b53f42e60b7d7dc985a4484ed21d33868bfc6cdaa4d6eef17dabf0025fdf"
    ):
        raise ValueError("registry starting lineage differs")
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    path = BASE / "result.json"
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        )
    row = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 43
    )
    row["current_status"] += (
        " The separate September 7 BYDUSDT and HK0992USDT listing population was source-bound on September 8: no exact BYD or HK0992 counterpart among 67 returned Polymarket instruments. Both siblings stop before funding or prices; HK0992 is quanto, not a linear hedge."
    )
    row["next_action"] += " " + result["retry_trigger"]
    row["retry_trigger"] += (
        " For the exact BYD/HK0992 no-match population: " + result["retry_trigger"]
    )
    row["canonical_artifacts"].append(
        {"path": path.as_posix(), "result_sha256": result["result_sha256"]}
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": "binance_September7_BYD_HK0992_no_exact_Polymarket_counterpart_2026_09_08",
            "reason": row["current_status"] + " " + result["retry_trigger"],
            "canonical_result_sha256": result["result_sha256"],
        }
    )
    # The new source supersedes the expired global campaign's clock routing;
    # retain that campaign's historical artifact, not a permanently true retry.
    allocation = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 8
    )
    expired = "_or_2026_09_07_USDT_global_campaign_end"
    if allocation["retry_trigger"].count(expired) != 1:
        raise ValueError("expired clock route differs")
    allocation["retry_trigger"] = allocation["retry_trigger"].replace(expired, "")
    registry["updated_at_utc"] = result["created_at_utc"]
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["september7_hk_listing_match"] = (
        result["status"] + ": " + result["retry_trigger"]
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
            }
        )
    )


if __name__ == "__main__":
    main()

"""Source-only adjudication of the frozen nine-contract September 10 change."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from simple_ai_trading.storage import write_bytes_atomic
from tools.capture_public_source_contract import _canonical_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("docs/review/2026-09-08/tradfi-september10-change")
REGISTRY = Path("docs/model-research/structural-edge-priority-registry-v1.json")
AUDIT = Path(
    "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)
FAMILY = "September10_nine_TradFi_funding_changes_no_exact_Polymarket_counterparts_2026_09_08"


def load(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads((ROOT / path).read_bytes())
    if value[field] != _canonical_hash(value, field):
        raise ValueError("canonical evidence differs")
    return value


def node_text(node: dict) -> str:
    if node.get("node") == "text":
        return html.unescape(node["text"]).replace("\xa0", " ")
    return "".join(node_text(child) for child in node.get("child", []))


def counterparts(rows: list[dict], symbols: list[str]) -> dict:
    """Exact labels are a necessary gate, never proof of a hedge or permission."""
    if (
        not isinstance(rows, list)
        or not rows
        or not symbols
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("invalid inventory or population")
    fields = ("instrument_id", "symbol", "base_asset", "quote_asset")
    for row in rows:
        if not isinstance(row, dict) or type(row.get("instrument_id")) is not int:
            raise ValueError("invalid instrument identity")
        if any(not isinstance(row.get(key), str) or not row[key] for key in fields[1:]):
            raise ValueError("missing instrument units")
    if any(len({row[key] for row in rows}) != len(rows) for key in fields[:2]):
        raise ValueError("ambiguous inventory")
    matches = []
    for symbol in symbols:
        if (
            not isinstance(symbol, str)
            or not symbol.endswith("USDT")
            or len(symbol) <= 4
        ):
            raise ValueError("invalid announced symbol")
        candidates = [
            {key: row[key] for key in fields}
            for row in rows
            if row["base_asset"] == symbol[:-4]
        ]
        matches.append({"binance_symbol": symbol, "exact_label_candidates": candidates})
    return {
        "inventory_count": len(rows),
        "inventory_identity_projection": [
            {key: row[key] for key in fields} for row in rows
        ],
        "siblings": matches,
        "status": "label_match_requires_separate_qualification"
        if any(row["exact_label_candidates"] for row in matches)
        else "terminal_no_exact_counterpart_in_retained_snapshot",
        "downstream_requests_authorized": False,
        "accepted_edge": False,
    }


def review() -> dict:
    plan = load(BASE / "source-contract.json", "contract_sha256")
    source = load(BASE / "source-result.json")
    raw = (ROOT / BASE / "raw.json").read_bytes()
    receipt = source["capture"]["receipt"]
    journal = [
        json.loads(line)
        for line in (ROOT / BASE / "journal.jsonl").read_bytes().splitlines()
    ]
    if (
        source["contract"]["sha256"] != plan["contract_sha256"]
        or not source["source_gate"]["passed"]
        or receipt["status_code"] != 200
        or sha256(raw).hexdigest() != receipt["response_sha256"]
        or len(raw) != receipt["response_bytes"]
        or len(journal) != 2
        or journal[0]["request"] != plan["request"]
        or journal[0]["contract_sha256"] != plan["contract_sha256"]
        or journal[1] != receipt
    ):
        raise ValueError("source receipt or intent differs")
    payload = json.loads(raw)
    article = payload["data"]
    if (
        payload["code"] != "000000"
        or article["code"] != plan["adjudication"]["article_code"]
    ):
        raise ValueError("article identity differs")
    blocks = [node_text(node) for node in json.loads(article["body"])["child"]]
    text = "\n".join(blocks)
    symbols = plan["adjudication"]["symbols"]
    if "".join(symbols) not in blocks or any(
        phrase not in text
        for phrase in (
            "2026-09-10 08:15 (UTC)",
            "from every eight hours to every four hours",
            "adjusted to ± 1.00%",
            "2026-09-10 08:00+2.00% / -2.00%",
            "2026-09-10 12:00+1.00% / -1.00%",
            "will not be adjusted from every four hours to every one hour",
        )
    ):
        raise ValueError("retained schedule or complete symbol population differs")
    binding = plan["adjudication"]["retained_inventory"]
    inventory_source = load(Path(binding["source_result_path"]))
    inventory = (ROOT / binding["path"]).read_bytes()
    if (
        inventory_source["result_sha256"] != binding["source_result_sha256"]
        or not inventory_source["source_gate"]["passed"]
        or sha256(inventory).hexdigest() != binding["sha256"]
        or inventory_source["capture"]["receipt"]["response_sha256"]
        != binding["sha256"]
    ):
        raise ValueError("retained inventory binding differs")
    analysis = counterparts(json.loads(inventory), symbols)
    return {
        "schema_version": 1,
        "source_contract_sha256": plan["contract_sha256"],
        "source_result_sha256": source["result_sha256"],
        "announcement_raw_sha256": receipt["response_sha256"],
        "inventory_binding": binding,
        "inventory_observed_at_ms": inventory_source["capture"]["receipt"][
            "completed_at_ms"
        ],
        "terms": {
            "effective_at_utc": "2026-09-10T08:15:00Z",
            "first_announced_changed_settlement_utc": "2026-09-10T12:00:00Z",
            "prior_interval_hours": 8,
            "new_interval_hours": 4,
            "prior_cap_percent": "2.00",
            "new_cap_percent": "1.00",
            "automatic_one_hour_acceleration": False,
            "nominal_24_hour_absolute_rate_sum_bound_percent_before": "6.00",
            "nominal_24_hour_absolute_rate_sum_bound_percent_after": "6.00",
            "bound_is_expected_or_realized_income": False,
        },
        "analysis": analysis,
        "new_source_requests": 1,
        "new_inventory_requests": 0,
        "funding_price_book_account_or_order_requests": 0,
        "accepted_edge": False,
        "deployment_ready": False,
        "retry_trigger": plan["adjudication"]["retry"],
        "reviewer_sha256": sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "local_inspection_note": "First decoded-text console display encountered Windows cp1252 UnicodeEncodeError after durable capture; subsequent inspection used ASCII JSON escaping on the same bytes. No recapture or source-gate change.",
    }


def main() -> None:
    result = review()
    if (
        result["analysis"]["status"]
        != "terminal_no_exact_counterpart_in_retained_snapshot"
    ):
        raise ValueError("a label match requires a separate qualification plan")
    registry, audit = load(REGISTRY), load(AUDIT)
    if (
        registry["result_sha256"]
        != "9ad0995e03bfc9d42936e713b3bd7ed89664cc1cdd697cb3f1a27d05fd8f6cd8"
        or audit["result_sha256"]
        != "3885449afe0dbc15b264627c44abdaf7a4dd87fb28e78a3e42ed2193bb8f9586"
    ):
        raise ValueError("publication starting lineage differs")
    now = datetime.now(UTC).isoformat()
    result["created_at_utc"] = now
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    path = BASE / "result.json"
    with (ROOT / path).open("xb") as stream:
        stream.write(
            (
                json.dumps(
                    result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
                + "\n"
            ).encode("ascii")
        )
    row = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 43
    )
    update = " September 8 source aff73212559d4951945600e973240c52 announces nine TradFi funding intervals changing 8h to 4h and cap 2% to 1% effective September 10 08:15 UTC. All nine lack exact Polymarket base labels in the complete retained September 8 10:27:29 UTC 67-instrument snapshot. One new source GET, no new inventory or economic requests. Nominal daily cap-sum bound is unchanged at 6%, not expected income. The September 10 clock alone is not a counterpart or a funding-study trigger."
    row["current_status"] += update
    row["next_action"] += (
        " For the nine September 10 contracts: " + result["retry_trigger"]
    )
    row["retry_trigger"] += (
        " For the nine September 10 contracts: " + result["retry_trigger"]
    )
    row["canonical_artifacts"].append(
        {"path": path.as_posix(), "result_sha256": result["result_sha256"]}
    )
    registry["terminal_do_not_repeat"].append(
        {
            "family": FAMILY,
            "reason": update.strip() + " " + result["retry_trigger"],
            "canonical_result_sha256": result["result_sha256"],
        }
    )
    registry["updated_at_utc"] = now
    registry["result_sha256"] = _canonical_hash(registry, "result_sha256")
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["routing"]["september10_nine_tradfi_funding_change"] = (
        update.strip() + " " + result["retry_trigger"]
    )
    audit["updated_at_utc"] = now
    audit["result_sha256"] = _canonical_hash(audit, "result_sha256")
    for target, payload in ((REGISTRY, registry), (AUDIT, audit)):
        write_bytes_atomic(
            ROOT / target,
            (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
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

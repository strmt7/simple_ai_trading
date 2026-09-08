"""Evaluate a prospectively bound deployment question without reading price fields."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.capture_public_source_contract import (
    _canonical_hash,
    _load_object,
    _root_path,
)


def instant(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an explicit UTC string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError("timestamp is not explicit UTC")
    return result


def evaluate(raw: bytes, slug: str, baseline: str, observed: str) -> dict:
    """Admit only a future active event created strictly after the old observation."""
    event = json.loads(raw)
    if not isinstance(event, dict) or event.get("slug") != slug:
        raise ValueError("exact event identity mismatch")
    created = instant(event.get("createdAt"))
    start = instant(event.get("startTime"))
    lower, upper = instant(baseline), instant(observed)
    if lower >= upper or created > upper:
        raise ValueError("deployment timestamps are inconsistent")
    markets = event.get("markets")
    if not isinstance(markets, list) or not 1 <= len(markets) <= 1000:
        raise ValueError("bounded nonempty embedded market population required")
    identifiers = []
    active = []
    for market in markets:
        if not isinstance(market, dict) or not isinstance(market.get("id"), str):
            raise ValueError("market identity missing")
        identifiers.append(market["id"])
        if (
            market.get("active") is True
            and market.get("closed") is False
            and market.get("acceptingOrders") is True
        ):
            active.append(market["id"])
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("empty or duplicate market identity")
    new = created > lower
    eligible = (
        new
        and start > upper
        and event.get("active") is True
        and event.get("closed") is False
        and bool(active)
    )
    return {
        "event_slug": slug,
        "created_at_utc": created.isoformat(),
        "start_time_utc": start.isoformat(),
        "new_since_empty_observation": new,
        "embedded_market_count": len(markets),
        "active_accepting_market_ids": active,
        "deployment_gate_passed": eligible,
        "economic_fields_examined": False,
        "next_action": "freeze_offline_payoff_and_side_specific_rejection_only"
        if eligible
        else "stop_without_economics_or_new_requests",
    }


def adjudicate(path: Path) -> dict:
    plan = _load_object(path.resolve())
    if _canonical_hash(plan, "contract_sha256") != plan.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    for binding in plan["implementations"]:
        if (
            hashlib.sha256(_root_path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise ValueError("implementation hash mismatch")
    result_path = _root_path(plan["deployment_gate"]["result_path"])
    if result_path.exists():
        raise FileExistsError("deployment adjudication already exists")
    source = _load_object(_root_path(plan["outputs"]["result_path"]))
    if (
        _canonical_hash(source, "result_sha256") != source.get("result_sha256")
        or source["contract"]["sha256"] != plan["contract_sha256"]
    ):
        raise ValueError("source result binding mismatch")
    raw = _root_path(plan["outputs"]["raw_path"]).read_bytes()
    receipt = source["capture"]["receipt"]
    if hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]:
        raise ValueError("raw response binding mismatch")
    result = {
        "contract_sha256": plan["contract_sha256"],
        "source_result_sha256": source["result_sha256"],
        "accepted_edge": False,
        "profitability_claim": False,
    }
    try:
        if not source["source_gate"]["passed"]:
            raise ValueError("public source gate failed")
        observed = datetime.fromtimestamp(
            receipt["completed_at_ms"] / 1000, timezone.utc
        ).isoformat()
        result["deployment"] = evaluate(
            raw,
            plan["deployment_gate"]["event_slug"],
            plan["deployment_gate"]["after_utc"],
            observed,
        )
    except (ValueError, TypeError, KeyError) as exc:
        result["deployment"] = {
            "deployment_gate_passed": False,
            "next_action": "stop_without_economics_or_new_requests",
            "failure_type": type(exc).__name__,
        }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    with result_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adjudicate(args.contract), sort_keys=True))


if __name__ == "__main__":
    main()

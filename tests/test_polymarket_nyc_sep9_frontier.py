"""Reconstruct the consumed September 9 screen using retained evidence only."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from tools import screen_polymarket_exact_negrisk_long_only_frontier as frontier

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-08/nyc-sep9-frontier"


def load(name: str) -> dict:
    return json.loads((BASE / name).read_bytes())


def test_prospective_capture_and_retained_byte_bindings() -> None:
    contract, result = load("contract.json"), load("result.json")
    frontier._validate_contract(contract, BASE / "contract.json")
    frozen = datetime.fromisoformat(contract["frozen_at_utc"])
    next_day = frozen.astimezone(ZoneInfo("America/New_York")).date() + timedelta(
        days=1
    )
    assert next_day.isoformat() == "2026-09-09"
    raw = (BASE / "raw/event.json").read_bytes()
    journal = [
        json.loads(line)
        for line in (BASE / "request-journal.jsonl").read_bytes().splitlines()
    ]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    receipt = result["capture"]["receipt"]
    assert journal[-1] == receipt
    assert (
        frozen.timestamp() * 1000
        < receipt["requested_at_ms"]
        <= receipt["completed_at_ms"]
    )
    assert receipt["status_code"] == 200 and receipt["transport_error_type"] is None
    assert receipt["url"] == contract["request"]["url"]
    assert len(raw) == receipt["response_bytes"] == 56225
    assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
    assert receipt["within_byte_ceiling"] and not receipt["redirects_allowed"]
    assert (
        frontier.base._canonical_hash(result, "result_sha256")
        == result["result_sha256"]
    )
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["authority"] == contract["authority"]


def test_complete_population_rules_and_all_frontier_rows_reconstruct() -> None:
    event, result = load("raw/event.json"), load("result.json")
    markets = frontier._markets(event, 11)
    rows, population = frontier._screen(event, markets, Decimal(5))
    assert rows == result["screen"]["rows"] and len(rows) == 28
    assert population == {
        "market_count": 11,
        "yes_price_complete_market_count": 11,
        "no_price_complete_market_count": 11,
    }
    assert len({market["description"] for market in markets}) == 1
    assert {market["resolutionSource"] for market in markets} == {
        "https://www.weather.gov/wrh/timeseries?site=klga"
    }
    rules = markets[0]["description"]
    assert all(
        phrase in rules
        for phrase in (
            "9 Sep '26",
            "Hourly Data",
            "whole degrees Fahrenheit",
            "lowest bracket",
        )
    )
    assert [market["groupItemTitle"] for market in markets] == [
        "75°F or below",
        *[f"{low}-{low + 1}°F" for low in range(76, 94, 2)],
        "94°F or higher",
    ]
    assert all(
        not row["passes_strict_metadata_gate"]
        and not row["passes_fee_and_one_tick_gate"]
        for row in rows
    )
    assert max(Decimal(row["metadata_profit_floor_pUSD"]) for row in rows) == Decimal(
        "-.005"
    )
    assert max(
        Decimal(row["after_fee_one_tick_profit_floor_pUSD"])
        for row in rows
        if row["after_fee_one_tick_profit_floor_pUSD"] is not None
    ) == Decimal("-.04773")
    complete_set = next(
        row for row in rows if row.get("package") == "all YES complete set"
    )
    assert sum(Decimal(str(market["bestAsk"])) for market in markets) == Decimal(
        "1.093"
    )
    assert complete_set["metadata_profit_floor_pUSD"] == "-0.465"
    assert result["screen"]["strict_metadata_candidate_count"] == 0
    assert result["screen"]["after_fee_one_tick_candidate_count"] == 0
    assert all(
        result["adjudication"][flag] is False
        for flag in ("accepted_edge", "deployment_ready", "profitability_claim")
    )


def test_terminal_record_and_downstream_source_binding() -> None:
    plan, contract, result = (
        load("registry-amendment-plan.json"),
        load("contract.json"),
        load("result.json"),
    )
    registry = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_bytes()
    )
    audit = json.loads(
        (
            ROOT
            / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
        ).read_bytes()
    )
    for value in (registry, audit):
        assert (
            frontier.base._canonical_hash(value, "result_sha256")
            == value["result_sha256"]
        )
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert audit["routing"][plan["audit_routing_field"]] == plan["summary"]
    assert [
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == plan["terminal_family"]
    ] == [
        {
            "family": plan["terminal_family"],
            "reason": plan["summary"],
            "canonical_result_sha256": result["result_sha256"],
        }
    ]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert family[plan["status_field"]] == plan["summary"]
    assert plan["prohibited_shortcut"] in family["prohibited_shortcuts"]
    for name, value, field in (
        ("contract.json", contract, "contract_sha256"),
        ("result.json", result, "result_sha256"),
    ):
        assert plan[field] == value[field]
        assert {
            "path": (BASE / name).relative_to(ROOT).as_posix(),
            "result_sha256": value[field],
        } in family["canonical_artifacts"]

"""Reconstruct the consumed September 6 screen without new requests."""

from datetime import datetime, timedelta
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from tools import screen_polymarket_exact_negrisk_long_only_frontier as f

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/nyc-sep6-frontier"


def load(name):
    return json.loads((BASE / name).read_bytes())


def test_prospective_date_transport_and_exact_source_bindings():
    c, r = load("contract.json"), load("result.json")
    f._validate_contract(c, BASE / "contract.json")
    frozen = datetime.fromisoformat(c["frozen_at_utc"].replace("Z", "+00:00"))
    next_local_day = frozen.astimezone(ZoneInfo("America/New_York")).date() + timedelta(
        days=1
    )
    assert next_local_day.isoformat() == "2026-09-06"
    raw = (BASE / "raw/event.json").read_bytes()
    journal = [
        json.loads(line)
        for line in (BASE / "request-journal.jsonl").read_bytes().splitlines()
    ]
    assert [x["phase"] for x in journal] == ["intent", "completed"]
    receipt = r["capture"]["receipt"]
    assert journal[-1] == receipt
    assert (
        frozen.timestamp() * 1000
        < receipt["requested_at_ms"]
        <= receipt["completed_at_ms"]
    )
    assert receipt["status_code"] == 200 and receipt["transport_error_type"] is None
    assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
    assert len(raw) == receipt["response_bytes"] == 55425
    assert receipt["within_byte_ceiling"] and not receipt["redirects_allowed"]
    assert f.base._canonical_hash(r, "result_sha256") == r["result_sha256"]
    assert r["contract"]["sha256"] == c["contract_sha256"]
    assert r["authority"] == c["authority"]


def test_complete_retained_frontier_and_rule_identity_reconstruct():
    event, result = load("raw/event.json"), load("result.json")
    markets = f._markets(event, 11)
    rows, population = f._screen(event, markets, D(5))
    assert rows == result["screen"]["rows"]
    assert len(rows) == 21
    assert population == {
        "market_count": 11,
        "yes_price_complete_market_count": 11,
        "no_price_complete_market_count": 8,
    }
    assert len({x["description"] for x in markets}) == 1
    assert {x["resolutionSource"] for x in markets} == {
        "https://www.weather.gov/wrh/timeseries?site=klga"
    }
    assert "6 Sep '26" in markets[0]["description"]
    assert all(
        not x["passes_strict_metadata_gate"] and not x["passes_fee_and_one_tick_gate"]
        for x in rows
    )
    assert max(D(x["metadata_profit_floor_pUSD"]) for x in rows) == D("-.015")
    assert max(
        D(x["after_fee_one_tick_profit_floor_pUSD"])
        for x in rows
        if x["after_fee_one_tick_profit_floor_pUSD"] is not None
    ) == D("-.04837")
    assert not result["adjudication"]["accepted_edge"]


def test_new_terminal_record_and_canonical_bindings_remain():
    p, result = load("registry-amendment-plan.json"), load("result.json")
    r = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_bytes()
    )
    assert f.base._canonical_hash(r, "result_sha256") == r["result_sha256"]
    found = [
        x for x in r["terminal_do_not_repeat"] if x["family"] == p["terminal_family"]
    ]
    assert found == [
        {
            "family": p["terminal_family"],
            "reason": p["summary"],
            "canonical_result_sha256": result["result_sha256"],
        }
    ]
    family = next(x for x in r["prioritized_hypotheses"] if x["priority_rank"] == 31)
    assert family[p["status_field"]] == p["summary"]
    assert any(
        x["path"] == "docs/review/2026-09-05/nyc-sep6-frontier/result.json"
        and x["result_sha256"] == result["result_sha256"]
        for x in family["canonical_artifacts"]
    )

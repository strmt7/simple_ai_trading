"""Source-bound exact-event checks; no requests or new acceptance experiment."""

from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

from tools import screen_polymarket_exact_negrisk_long_only_frontier as frontier

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/nyc-sep5-frontier"


def load(name):
    return json.loads((BASE / name).read_bytes())


def test_source_capture_boundaries_and_exact_retained_result():
    contract = load("contract.json")
    frontier._validate_contract(contract, BASE / "contract.json")
    result = load("result.json")
    assert (
        frontier.base._canonical_hash(result, "result_sha256")
        == result["result_sha256"]
    )
    receipt = result["capture"]["receipt"]
    raw = (BASE / "raw/event.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
    assert len(raw) == receipt["response_bytes"] <= receipt["response_byte_ceiling"]
    journal = [
        json.loads(line)
        for line in (BASE / "request-journal.jsonl").read_bytes().splitlines()
    ]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1] == receipt
    assert receipt["transport_error_type"] is None
    assert receipt["redirects_allowed"] is False
    assert result["event"]["market_count"] == 11
    assert result["event"]["yes_price_complete_market_count"] == 11
    assert result["event"]["no_price_complete_market_count"] == 7
    assert result["adjudication"]["accepted_edge"] is False


def test_fee_only_rejection_reconstructs_without_tick_stress():
    result, audit = load("result.json"), load("fee-audit-result.json")
    assert (
        frontier.base._canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    )
    for binding in audit["bindings"]:
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    markets = {str(row["id"]): row for row in load("raw/event.json")["markets"]}
    assert len(audit["rows"]) == len(result["screen"]["rows"]) == 17
    gross_positive = 0
    for original, row in zip(result["screen"]["rows"], audit["rows"], strict=True):
        fees = [
            frontier._fee_model_v2(markets[leg["market_id"]])(
                D(leg["price_pUSD_per_share"]),
                D(original["quantity_shares_each_leg"]),
                "taker",
            )
            for leg in original["legs"]
        ]
        total = sum(fees, D(0))
        assert total == D(row["configured_taker_fee_without_ticks_pUSD"])
        assert D(original["metadata_profit_floor_pUSD"]) - total == D(
            row["after_configured_fee_without_ticks_pUSD"]
        )
        assert D(row["after_configured_fee_without_ticks_pUSD"]) < 0
        if D(row["gross_headroom_pUSD"]) > 0:
            gross_positive += 1
            assert max(fees) == D("0.06160") > D(row["gross_headroom_pUSD"])
            assert original["after_fee_one_tick_profit_floor_pUSD"] is None
    assert gross_positive == 4
    assert audit["further_requests_authorized"] is False


def test_terminal_registry_lineage_and_previous_ledgers_retained():
    registry = json.loads(
        (
            ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
        ).read_bytes()
    )
    assert (
        frontier.base._canonical_hash(registry, "result_sha256")
        == registry["result_sha256"]
    )
    terminal = [
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "polymarket_nyc_september5_complete_long_only_frontier"
    ]
    assert len(terminal) == 1
    assert (
        terminal[0]["canonical_result_sha256"] == load("result.json")["result_sha256"]
    )
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    paths = {row["path"] for row in family["canonical_artifacts"]}
    assert "docs/review/2026-09-04/nyc-sep5-frontier/fee-audit-result.json" in paths
    assert (
        load("registry-before.json")["result_sha256"]
        == "39a69bce3a5544cc33d2d6383cc7f9578ff8c2305fe36caa1c57b0e5f053de1a"
    )
    for name in ("registry-before.json", "durability-before.json"):
        prior = load(name)
        assert (
            frontier.base._canonical_hash(prior, "result_sha256")
            == prior["result_sha256"]
        )

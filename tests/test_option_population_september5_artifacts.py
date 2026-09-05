"""Offline reconstruction of the consumed September 5 option population."""

from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_option_population_v3 import known_symbols
from tools.adjudicate_binance_crypto_option_population_gate_v2 import _eligible_symbols
from tools.screen_option_floor_population import _futures_map, _ticker_map, rows_for
from tools.update_binance_crypto_option_distinct_population_registry import (
    _canonical_hash,
)


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/review/2026-09-05/option-population"


def load(name):
    return json.loads((FOLDER / name).read_bytes())


def test_contracts_implementations_and_source_receipts_reconstruct():
    for name in (
        "source-contract.json",
        "population-contract.json",
        "option_tickers-contract.json",
        "futures_books-contract.json",
        "price-contract.json",
        "population-result.json",
        "price-result.json",
        "source-result.json",
        "option_tickers-source-result.json",
        "futures_books-source-result.json",
        "registry-amendment-plan.json",
    ):
        obj = load(name)
        field = (
            "plan_sha256"
            if name == "registry-amendment-plan.json"
            else "contract_sha256"
            if name.endswith("contract.json")
            else "result_sha256"
        )
        assert obj[field] == _canonical_hash(obj, field)
        bindings = obj.get("implementation_sha256", {}) | {
            b["path"]: b["sha256"] for b in obj.get("implementations", [])
        }
        for path, expected in bindings.items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected

    for name in (
        "source-result.json",
        "option_tickers-source-result.json",
        "futures_books-source-result.json",
    ):
        source = load(name)
        contract = json.loads((ROOT / source["contract"]["path"]).read_bytes())
        assert source["contract"]["sha256"] == contract["contract_sha256"]
        assert source["source_gate"]["passed"] is True
        assert source["authority"] == contract["authority"]
        assert source["authority"]["credentials_used"] is False
        receipt = source["capture"]["receipt"]
        raw = (ROOT / receipt["raw_path"]).read_bytes()
        assert (
            len(raw) == receipt["response_bytes"] <= contract["response_byte_ceiling"]
        )
        assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
        assert receipt["status_code"] == 200 and receipt["error_type"] is None
        assert receipt["oversize_body_is_truncated"] is False
        journal = [
            json.loads(line)
            for line in (ROOT / source["capture"]["journal_path"])
            .read_bytes()
            .splitlines()
        ]
        assert len(journal) == 2 and journal[0]["phase"] == "intent"
        assert journal[0]["request"] == contract["request"]
        assert journal[0]["contract_sha256"] == contract["contract_sha256"]
        assert journal[0]["requested_at_ms"] == receipt["requested_at_ms"]
        assert journal[1] == receipt
        frozen_ms = datetime.fromisoformat(contract["frozen_at_utc"]).timestamp() * 1000
        assert frozen_ms < receipt["requested_at_ms"] <= receipt["completed_at_ms"]


def test_complete_population_and_price_rows_reconstruct():
    plan = load("population-contract.json")
    population = load("population-result.json")
    price_plan = load("price-contract.json")
    price = load("price-result.json")
    known = known_symbols(plan)
    raw = load("exchange-info.json")
    current = set(_eligible_symbols(raw))
    assert (
        len(known) == 2274
        and len(current) == population["current_eligible_count"] == 1436
    )
    assert len(current & known) == population["excluded_known_count"] == 1366
    assert sorted(current - known) == population["distinct_symbols"]
    assert len(population["distinct_symbols"]) == population["distinct_count"] == 70
    indexed = {row["symbol"]: row for row in raw["optionSymbols"]}
    assert population["distinct_metadata"] == [
        indexed[s] for s in population["distinct_symbols"]
    ]
    assert population["contract_sha256"] == plan["contract_sha256"]
    assert (
        population["source_result_sha256"]
        == load("source-result.json")["result_sha256"]
    )
    assert price["contract_sha256"] == price_plan["contract_sha256"]
    assert (
        price["population_sha256"]
        == price_plan["population_sha256"]
        == population["result_sha256"]
    )
    rows = rows_for(
        population["distinct_metadata"],
        _ticker_map(load("option_tickers.json")),
        _futures_map(load("futures_books.json")),
    )
    assert rows == price["all_rows"]
    eligible = [r for r in rows if r["positive_entry_sides"]]
    assert len(eligible) == 23
    assert sum(Decimal(r["gross_floor_per_base_usdt"]) > 0 for r in eligible) == 0
    assert len([r for r in rows if Decimal(r["option_ask"]) == 0]) == 47
    assert not any(r["passes_row_gate"] for r in rows)
    assert price["survivors"] == [] and price["failure_type"] is None
    assert price["next_action"] == "stop_without_depth_or_accounts"
    sources = [
        load(n + "-source-result.json") for n in ("option_tickers", "futures_books")
    ]
    for name, source in zip(("option_tickers", "futures_books"), sources, strict=True):
        assert price["source_results"][name] == source["result_sha256"]
        assert (
            price_plan["source_results"][name]["contract_sha256"]
            == source["contract"]["sha256"]
        )
    skew = abs(
        sources[0]["capture"]["receipt"]["requested_at_ms"]
        - sources[1]["capture"]["receipt"]["requested_at_ms"]
    )
    assert price["request_start_skew_ms"] == skew == 1984
    assert (
        price["skew_gate_passed"] is True
        and skew <= price_plan["maximum_start_skew_ms"]
    )
    assert all(
        m["expiryDate"]
        > max(s["capture"]["receipt"]["completed_at_ms"] for s in sources)
        for m in population["distinct_metadata"]
    )
    best = max(
        eligible,
        key=lambda r: (
            Decimal(r["after_fixed_stress_per_base_usdt"])
            / Decimal(r["perpetual_entry"])
        ),
    )
    assert best["symbol"] == "BTC-260908-79500-P"
    assert Decimal(best["gross_floor_per_base_usdt"]) == Decimal("-757.30")
    assert Decimal(best["after_fixed_stress_per_base_usdt"]) == Decimal("-1024.1687050")


def test_terminal_registry_lineage_preserves_nonpromotion():
    plan = load("registry-amendment-plan.json")
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
    for obj in (registry, audit):
        assert obj["result_sha256"] == _canonical_hash(obj, "result_sha256")
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["source_binding"]["accepted_edge_count"]
        == registry["accepted_edge_count"]
    )
    (row,) = [r for r in registry["prioritized_hypotheses"] if r["priority_rank"] == 47]
    assert all(b in row["canonical_artifacts"] for b in plan["artifact_additions"])
    assert plan["terminal"] in registry["terminal_do_not_repeat"]
    assert plan["prohibited_shortcut"] in row["prohibited_shortcuts"]
    for name in ("population-result.json", "price-result.json"):
        value = load(name)
        assert value["accepted_edge"] is False and value["profitability_claim"] is False

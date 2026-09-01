from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "polymarket-nyc-high-temperature-sep1-exact-negrisk-prefilter-"
    "contract-v1-2026-09-01.json"
)
RESULT = ACTION / (
    "polymarket-nyc-high-temperature-sep1-exact-negrisk-prefilter-"
    "result-v1-2026-09-01.json"
)
RAW = ROOT / (
    "data/polymarket-nyc-high-temperature-sep1-exact-negrisk-"
    "prefilter-v1/raw/event.json"
)
JOURNAL = ROOT / (
    "data/polymarket-nyc-high-temperature-sep1-exact-negrisk-"
    "prefilter-v1/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
DURABILITY = ACTION / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"

CONTRACT_HASH = "f870a6417968fa86dc3b194905a7d3d25dca59b4969fd69768290256c48e5336"
RESULT_HASH = "c3dc478bc582ec5f60ace208fab5cb28e9d1cf593059894443d2783c12f114ac"
RAW_HASH = "28314d59c66cf50b81232ba85ae4017248f2ad73e498535e28b6526cfe03b9d9"
JOURNAL_HASH = "201bdcb74190a36fe9f37de795bda9fc527fe502039279a9d808f44d7b5daefb"
REGISTRY_HASH = "fd3948dffc72413f18f54fc966278fae617d3050f2fe6fd682c8308874e231f5"
DURABILITY_HASH = "c7e4f16beee4a9bade1d359774849b22f53155db908e0901842d61e2b59db636"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _self_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_weather_prefilter_is_source_bound_and_permanently_nonpromotable() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    assert _self_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert _self_hash(result, "result_sha256") == RESULT_HASH
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == RAW_HASH
    assert hashlib.sha256(JOURNAL.read_bytes()).hexdigest() == JOURNAL_HASH
    assert [
        json.loads(line)["phase"] for line in JOURNAL.read_bytes().splitlines()
    ] == ["intent", "completed"]
    trigger = contract["trigger"]
    assert isinstance(trigger, dict)
    assert trigger["discovery_values_are_economic_inputs"] is True
    assert trigger["promotion_eligible"] is False

    screen = result["screen"]
    assert isinstance(screen, dict)
    event = screen["event"]
    assert isinstance(event, dict)
    assert event["market_count"] == 11
    assert Decimal(event["displayed_all_yes_sum_pUSD"]) == Decimal("1.0485")
    assert screen["all_yes_candidate"] is False
    assert screen["positive_displayed_conversion_candidate_count"] == 11
    rows = screen["one_no_to_other_yes_displayed_identities"]
    assert isinstance(rows, list)
    assert max(
        Decimal(row["optimistic_displayed_conversion_gap_pUSD"]) for row in rows
    ) == Decimal("0.0485")


def test_weather_prefilter_routes_terminal_without_changing_profitability() -> None:
    registry = _load(REGISTRY)
    durability = _load(DURABILITY)
    assert _self_hash(registry, "result_sha256") == REGISTRY_HASH
    assert _self_hash(durability, "result_sha256") == DURABILITY_HASH
    assert len(registry["terminal_do_not_repeat"]) == 158
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    family = (
        "polymarket_NYC_September_1_high_temperature_exact_fixed_"
        "NegRisk_prefilter_2026_09_01"
    )
    assert terminal[family]["canonical_result_sha256"] == RESULT_HASH
    decision = durability["decision"]
    assert isinstance(decision, dict)
    assert decision["stable_current_account_qualified_after_all_cost_edge_count"] == 0
    source = durability["source_binding"]
    assert isinstance(source, dict)
    assert source["registry_result_sha256"] == REGISTRY_HASH

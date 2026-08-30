from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "binance-spot-price-range-execution-rule-source-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = ACTION / (
    "binance-spot-price-range-execution-rule-source-result-v1-2026-08-30.json"
)
TERMINAL = ACTION / (
    "binance-spot-price-range-execution-rule-terminal-adjudication-v1-2026-08-30.json"
)
RAW = ROOT / (
    "docs/model-research/binance/raw/spot-price-range-execution-rule-v1-2026-08-30/"
    "01-price-range-execution-rule-faq.raw.md"
)
JOURNAL = ROOT / (
    "docs/model-research/binance/raw/spot-price-range-execution-rule-v1-2026-08-30/"
    "00-request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"

CONTRACT_HASH = "53381cf5bc5e8328283dcf06efdcdb7630466b8becb25451405f219588ba7569"
SOURCE_RESULT_HASH = (
    "2866f68ea2dd0b7fb460fd132da19fc278cab30e6b8cb1db55f92f857a3281a3"
)
RAW_HASH = "ec6fa180dc99ea1f1846f8e310caa958c8f0ff33fec65a1b94160113f25259d7"
TERMINAL_HASH = "6716c320effd97f20ebe84536366e0308ca7089b1ef15d4c6f601c232182a10d"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _canonical_hash(value: dict[str, object], field: str) -> str:
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


def test_price_range_source_and_terminal_decision_are_hash_bound() -> None:
    contract = _load(CONTRACT)
    source = _load(SOURCE_RESULT)
    terminal = _load(TERMINAL)

    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert _canonical_hash(source, "result_sha256") == SOURCE_RESULT_HASH
    assert _canonical_hash(terminal, "result_sha256") == TERMINAL_HASH
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == RAW_HASH
    journal = [json.loads(line) for line in JOURNAL.read_bytes().splitlines()]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == RAW_HASH
    assert source["source_gate"]["passed"] is True
    assert all(source["source_gate"]["required_phrase_presence"].values())


def test_user_bounded_order_dominates_the_exchange_safety_cap() -> None:
    terminal = _load(TERMINAL)
    proof = terminal["dominance_proof"]
    assert proof["buy"]["incremental_rule_payoff_upper_bound"] == "0"
    assert proof["sell"]["incremental_rule_payoff_upper_bound"] == "0"
    assert terminal["request_efficiency"]["live_execution_rules_requested"] is False
    assert terminal["request_efficiency"]["live_reference_price_requested"] is False
    assert terminal["adjudication"]["accepted_edge"] is False
    assert terminal["adjudication"]["terminal_family_count_change"] == 1

    registry = _load(REGISTRY)
    assert registry["result_sha256"] == _canonical_hash(registry, "result_sha256")
    family = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_Spot_Price_Range_Execution_Rule_safety_control_2026_08_30"
    )
    assert family["canonical_result_sha256"] == TERMINAL_HASH

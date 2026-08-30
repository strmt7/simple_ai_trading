from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "polymarket-ball-state-ohio-state-exact-event-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = BASE / (
    "polymarket-ball-state-ohio-state-exact-event-result-v1-2026-08-30.json"
)
TERMINAL = BASE / (
    "polymarket-ball-state-ohio-state-monotone-superhedge-"
    "terminal-v1-2026-08-30.json"
)
RAW_DIR = ROOT / (
    "docs/model-research/polymarket/raw/"
    "ball-state-ohio-state-exact-event-v1-2026-08-30"
)
RAW = RAW_DIR / "event.raw.json"
JOURNAL = RAW_DIR / "request-journal.jsonl"
CONSUMED_IMPLEMENTATION = RAW_DIR / (
    "capture-polymarket-exact-sports-event-consumed.py"
)
REUSABLE_IMPLEMENTATION = ROOT / "tools/capture_polymarket_exact_sports_event.py"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_one_exact_event_request_preceded_every_network_side_effect() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(SOURCE_RESULT.read_text(encoding="ascii"))
    journal = [
        json.loads(line)
        for line in JOURNAL.read_text(encoding="ascii").splitlines()
    ]
    raw = RAW.read_bytes()

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _sha256(CONSUMED_IMPLEMENTATION.read_bytes()) == contract[
        "implementation"
    ]["sha256"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["planned_at_ms"] <= journal[1]["started_at_ms"]
    assert journal[1]["status_code"] == 200
    assert journal[1]["response_bytes"] == len(raw)
    assert journal[1]["response_sha256"] == _sha256(raw)
    assert result["authority"]["book_or_price_requests"] == 0
    assert result["capture"]["active_accepting_market_count"] == 3
    assert result["discovery"]["active_sports_market_types"] == [
        "moneyline",
        "spreads",
        "totals",
    ]


def test_exact_primal_and_dual_certificates_prove_zero_displayed_headroom() -> None:
    terminal = json.loads(TERMINAL.read_text(encoding="ascii"))
    assert _self_hash(terminal, "result_sha256") == terminal["result_sha256"]
    recovery = terminal["implementation_recovery"]
    assert recovery["immutable_consumed_sidecar_sha256"] == _sha256(
        CONSUMED_IMPLEMENTATION.read_bytes()
    )
    assert recovery["restored_reusable_sha256"] == _sha256(
        REUSABLE_IMPLEMENTATION.read_bytes()
    )

    screen = terminal["superhedge_screen"]
    prices = [Decimal(str(value)) for value in screen["displayed_price_vector"]]
    quantities = [
        Decimal(str(value)) for value in screen["one_optimal_quantity_vector"]
    ]
    states = terminal["exact_rule_reduction"]["state_classes"]
    payouts = [
        sum(
            Decimal(str(value)) * quantity
            for value, quantity in zip(state["payout_vector"], quantities, strict=True)
        )
        for state in states
    ]
    assert sum(price * quantity for price, quantity in zip(prices, quantities, strict=True)) == Decimal("1")
    assert min(payouts) == Decimal("1")

    weights_by_name = {
        name: Decimal(value)
        for name, value in screen["dual_lower_bound_certificate"][
            "state_weights"
        ].items()
    }
    assert sum(weights_by_name.values()) == Decimal("1")
    weighted_payouts = []
    for token_index in range(len(prices)):
        weighted_payouts.append(
            sum(
                weights_by_name.get(state["name"], Decimal("0"))
                * Decimal(str(state["payout_vector"][token_index]))
                for state in states
            )
        )
    assert weighted_payouts == prices
    assert screen["strict_pre_book_gate_passed"] is False
    assert terminal["authority"]["book_or_price_requests"] == 0


def test_exact_event_is_terminal_inside_the_expanded_sports_family() -> None:
    terminal = json.loads(TERMINAL.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 30
    )
    assert "CFB" in family["venue_scope"]
    assert "strictly_below" in family["retry_trigger"]
    assert {
        "path": TERMINAL.relative_to(ROOT).as_posix(),
        "result_sha256": terminal["result_sha256"],
    } in family["canonical_artifacts"]
    assert {
        "canonical_result_sha256": terminal["result_sha256"],
        "family": (
            "polymarket_Ball_State_Ohio_State_September_5_exact_"
            "CFB_monotone_superhedge_2026_08_30"
        ),
        "reason": next(
            row["reason"]
            for row in registry["terminal_do_not_repeat"]
            if row["canonical_result_sha256"] == terminal["result_sha256"]
        ),
    } in registry["terminal_do_not_repeat"]

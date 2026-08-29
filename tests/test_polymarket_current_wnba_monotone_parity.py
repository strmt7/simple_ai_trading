from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DATA = ROOT / "data"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "d9698017a21be49e8f0b5c0021d4c1eeb1dff0a6482bab9badc0a8c76be5df4b"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], hash_field: str) -> str:
    body = dict(payload)
    body.pop(hash_field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exact_event_capture_is_one_use_hash_bound_and_public_only() -> None:
    contract = _load(
        ACTION_VALUE / "polymarket-current-wnba-exact-event-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE / "polymarket-current-wnba-exact-event-result-v1-2026-08-29.json"
    )
    raw = DATA / "polymarket-current-wnba-exact-event-v1/raw/event.json"
    journal = _journal(
        DATA / "polymarket-current-wnba-exact-event-v1/request-journal.jsonl"
    )

    assert contract["contract_sha256"] == (
        "d129e73e922fac53d16143f18914ffa8c1eaff3e0a8a11e9f3bf2696ff3c5eee"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "6851d26788abfd175b75649e573d341696e570ce76bc235b0c5a6070bdd72167"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(ROOT / result["implementation"]["path"]) == (
        "eb4cb408458fbf1c9dcdaee26532d9e7395064717fac4190b193a51935e9a015"
    )
    assert _file_hash(raw) == (
        "c9172a258cedc488794f466cae33dab65481e0634cec32be6d2d73aef44bdced"
    )
    assert [entry["phase"] for entry in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "GET"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["orders_or_transactions"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_three_packages_have_exhaustive_one_dollar_floor_but_negative_books() -> None:
    result = _load(
        ACTION_VALUE
        / "polymarket-current-wnba-monotone-parity-result-v1-2026-08-29.json"
    )
    raw = DATA / "polymarket-current-wnba-monotone-parity-v1/raw/books.json"
    journal = _journal(
        DATA / "polymarket-current-wnba-monotone-parity-v1/request-journal.jsonl"
    )
    contract = _load(
        ACTION_VALUE
        / "polymarket-current-wnba-monotone-parity-contract-v1-2026-08-29.json"
    )

    assert contract["contract_sha256"] == (
        "7ed7007e5b6580100c4e3fe0495475be2742cc536372e8dfc953a19dba0f80c8"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "cc657982abd9ede0f0f7b18787df32e62c69b7c3b3e547ade3f6f3ccb734ed46"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(ROOT / result["implementation"]["path"]) == (
        "814cf3b62e7b38e07f44dc4070309f75eabf81e954031f0021a3e36c0eb7cdc7"
    )
    assert _file_hash(raw) == (
        "c81e97bf7f615e155e23ab58c7b7133a3c6b89dd1734c6dba58dccc97d172d63"
    )
    assert [entry["phase"] for entry in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "POST"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["capture"]["book_timestamp_skew_ms"] == 2132
    assert result["capture"]["passes_synchronization_gate"] is True
    assert result["capture"]["fee_receipts"] == {}

    states = result["payoff_proof"]["states"]
    packages = contract["packages"]
    for package in packages:
        assert min(
            sum(Decimal(state["payouts"][token]) for token in package["token_names"])
            for state in states
        ) == Decimal("1")

    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["passes_frozen_candidate_gate"] is False
        for phase in ("actual", "delay_1s_sensitivity", "delay_3s_sensitivity"):
            assert Decimal(row["economics"][phase]["after_fee_profit_floor_pUSD"]) < 0
    assert result["adjudication"] == {
        "accepted_edge": False,
        "best_delay_3s_profit_floor_pUSD": "-1.02076",
        "best_package": "phoenix_spread_8_5_plus_tempo_spread_9_5",
        "current_positive_package_count": 0,
        "deployment_ready": False,
        "next_action": "terminalize_this_exact_event_without_resampling",
        "profitability_claim": False,
    }


def test_registry_requires_rejection_only_prefilter_before_another_book_batch() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_live_NBA_moneyline_spread_monotone_payoff_implication"
    )

    assert row["priority_rank"] == 30
    assert "WNBA" in row["venue_scope"]
    assert "rejection_only" in row["next_action"]
    assert "below_its_guaranteed_payout_floor" in row["retry_trigger"]
    assert any(
        "using_Gamma_prices_to_accept_or_promote" in shortcut
        for shortcut in row["prohibited_shortcuts"]
    )
    assert registry["accepted_edge_count"] == 19

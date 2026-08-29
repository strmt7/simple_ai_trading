from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DATA = ROOT / "data/polymarket-lynx-dream-exact-event-prefilter-v1"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "6062ef4cb774983d86d7edd5dad7adcaafa31a8202d37ec777e12fc33028d157"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
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


def test_generic_capture_is_hash_bound_one_use_and_public_only() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-lynx-dream-exact-event-prefilter-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-lynx-dream-exact-event-prefilter-result-v1-2026-08-29.json"
    )
    raw = DATA / "raw/event.json"
    journal = [
        json.loads(line)
        for line in (DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert contract["contract_sha256"] == (
        "d049290de090fd07be9f99d4f59704488447fb98013fc55d2c183d9f52a181c8"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "c7629f0869bf7b1b9b6622cde42b0822f35e63386c9cb3e2e4364423fa4f7156"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(ROOT / result["implementation"]["path"]) == (
        "a1f431d9ef181123a8f47840b9cf894aa3e8b6b09b18acadf643ac7f32e61606"
    )
    assert _file_hash(raw) == (
        "d87b3fd43633a72a4d6f33edc1a40d77acd0c51ec11fec99ccf1d1048511f4d3"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "GET"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_guaranteed_package_fails_rejection_only_gamma_gate() -> None:
    artifact = _load(
        ACTION_VALUE
        / "polymarket-lynx-dream-monotone-prefilter-adjudication-v1-2026-08-29.json"
    )

    assert artifact["result_sha256"] == (
        "61b3436b3367ba3442ebe777c8a506948243c6d3b6d6a4cb9346d2db3aaf335f"
    )
    assert _canonical_hash(artifact, "result_sha256") == artifact["result_sha256"]
    payouts = [
        Decimal(row["package_payout"])
        for row in artifact["payoff_proof"]["states"]
    ]
    assert min(payouts) == Decimal("1")
    gate = artifact["rejection_only_gamma_prefilter"]
    assert Decimal(gate["package_displayed_price_sum_pUSD"]) == Decimal("1.080")
    assert Decimal(gate["optimistic_profit_floor_per_share_before_execution_costs_pUSD"]) == Decimal(
        "-0.080"
    )
    assert Decimal(
        gate["optimistic_profit_floor_at_five_shares_before_execution_costs_pUSD"]
    ) == Decimal("-0.400")
    assert gate["passes_strictly_below_payout_gate"] is False
    assert gate["gamma_can_support_acceptance_or_promotion"] is False
    assert artifact["adjudication"]["status"] == (
        "terminal_exact_event_rejected_before_books_and_fees"
    )
    assert artifact["authority"]["book_requests"] == 0
    assert artifact["authority"]["fee_requests"] == 0


def test_registry_retains_both_negative_wnba_extensions() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_live_NBA_moneyline_spread_monotone_payoff_implication"
    )

    hashes = {artifact["result_sha256"] for artifact in row["canonical_artifacts"]}
    assert {
        "cc657982abd9ede0f0f7b18787df32e62c69b7c3b3e547ade3f6f3ccb734ed46",
        "61b3436b3367ba3442ebe777c8a506948243c6d3b6d6a4cb9346d2db3aaf335f",
    } <= hashes
    assert registry["accepted_edge_count"] == 19

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_two_leg_sports_package import _line_matches


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
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


def test_exact_nfl_capture_is_one_request_and_hash_bound() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-exact-event-prefilter-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-exact-event-prefilter-result-v1-2026-08-29.json"
    )
    raw = (
        ROOT / "data/polymarket-packers-vikings-exact-event-prefilter-v1/raw/event.json"
    )
    journal = [
        json.loads(line)
        for line in (
            ROOT
            / "data/polymarket-packers-vikings-exact-event-prefilter-v1/request-journal.jsonl"
        )
        .read_text()
        .splitlines()
    ]

    assert contract["contract_sha256"] == (
        "adf268674779dd882dd0879ce29e168f540a3af316d8adb0233f9de74e896172"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "8ebf70181290234c1c05f4659245d2c8c1502fd4a02eaa93dff9a4f60e375c6e"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(raw) == (
        "ca97f6fe1e435ecbd165d81aa65685431903eda1106b7ed53865ddb5426aeefa"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert result["capture"]["active_accepting_market_count"] == 34
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["book_requests"] == 0


def test_complete_nfl_lattice_finds_four_rejection_gate_candidates() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-monotone-prefilter-adjudication-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-monotone-prefilter-adjudication-v1-2026-08-29.json"
    )

    assert contract["contract_sha256"] == (
        "9621ca3a327f1f057be6ee560c250063ddd9af48785dc1ed7c4ad3114dea61df"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "c387e389d852ab5571056a9f2e80f91c63ae6f1c124ca55291b0fc787b5faeae"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    proof = result["payoff_proof"]
    assert proof["complete_relation_count"] == 321
    for relation in proof["relations"]:
        assert Decimal(relation["minimum_terminal_payout_per_share_pUSD"]) >= 1
    gate = result["rejection_only_gamma_prefilter"]
    assert gate["candidate_count_strictly_below_payout_floor"] == 4
    best = gate["best_relation"]
    assert best["superset_positive_market_id"] == "3920326"
    assert best["subset_complement_market_id"] == "3335219"
    assert Decimal(best["displayed_price_sum_per_share_pUSD"]) == Decimal("0.895")
    assert result["authority"]["book_requests"] == 0


def test_strongest_tie_state_package_fails_at_exact_asks_before_fees() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-tie-state-package-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-packers-vikings-tie-state-package-result-v1-2026-08-29.json"
    )
    books = ROOT / "data/polymarket-packers-vikings-tie-state-package-v1/raw/books.json"
    journal = [
        json.loads(line)
        for line in (
            ROOT
            / "data/polymarket-packers-vikings-tie-state-package-v1/request-journal.jsonl"
        )
        .read_text()
        .splitlines()
    ]

    assert contract["contract_sha256"] == (
        "9eb26db000b0bc64182bf202959d9de003a550b5b812eaec256fcc95cd9f1b4c"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "731ca32a06f8f1a42aaae9e326c2bd89379657e338231dd906b749790c15ddfa"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(books) == (
        "430f9ae42b6fcbff066ce142aea17968c4b69a52d26fc908adbaf3605fdb4d49"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    actual = result["economics"]["actual"]
    assert Decimal(actual["cost_pUSD"]) == Decimal("5.4")
    assert Decimal(actual["optimistic_zero_fee_profit_floor_pUSD"]) == Decimal("-0.4")
    assert result["capture"]["book_timestamp_skew_ms"] == 10_049_940
    assert result["capture"]["within_frozen_skew_gate"] is False
    assert result["capture"]["fee_receipts"] == {}
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False


def test_nullable_moneyline_line_and_registry_routing() -> None:
    assert _line_matches(None, None) is True
    assert _line_matches(None, "0") is False
    assert _line_matches("-0.5", -0.5) is True

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
        "8ebf70181290234c1c05f4659245d2c8c1502fd4a02eaa93dff9a4f60e375c6e",
        "c387e389d852ab5571056a9f2e80f91c63ae6f1c124ca55291b0fc787b5faeae",
        "731ca32a06f8f1a42aaae9e326c2bd89379657e338231dd906b749790c15ddfa",
    } <= hashes
    assert registry["accepted_edge_count"] == 21

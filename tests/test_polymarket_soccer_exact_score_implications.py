from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RAW = ROOT / "data/polymarket-sep6-negrisk-complete-set-catalog-v1/raw/events.json"
BOOK_RAW = ROOT / "data/polymarket-soccer-zero-zero-draw-books-v1/raw/books.json"


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load(name: str) -> dict[str, object]:
    return json.loads((ACTION_VALUE / name).read_text(encoding="utf-8"))


def test_soccer_implication_screen_corrects_midpoint_escalation() -> None:
    v1 = _load(
        "polymarket-soccer-exact-score-implication-contract-v1-2026-08-30.json"
    )
    failure = _load(
        "polymarket-soccer-exact-score-implication-v1-preflight-failure-"
        "2026-08-30.json"
    )
    v2 = _load(
        "polymarket-soccer-exact-score-implication-result-v2-2026-08-30.json"
    )
    book = _load(
        "polymarket-soccer-zero-zero-draw-books-result-v1-2026-08-30.json"
    )
    v3_contract = _load(
        "polymarket-soccer-exact-score-implication-contract-v3-2026-08-30.json"
    )
    v3 = _load(
        "polymarket-soccer-exact-score-implication-result-v3-2026-08-30.json"
    )

    assert _canonical_hash(v1, "contract_sha256") == v1["contract_sha256"]
    assert _canonical_hash(failure, "result_sha256") == failure["result_sha256"]
    assert _canonical_hash(v2, "result_sha256") == v2["result_sha256"]
    assert _canonical_hash(book, "result_sha256") == book["result_sha256"]
    assert (
        _canonical_hash(v3_contract, "contract_sha256")
        == v3_contract["contract_sha256"]
    )
    assert _canonical_hash(v3, "result_sha256") == v3["result_sha256"]
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == (
        "f430490f592cf58297c5f5f118b3fabc32488faa82b5b07b60815279bdc61050"
    )
    assert hashlib.sha256(BOOK_RAW.read_bytes()).hexdigest() == (
        "c814560ec0542e3c3407ce107ded679415205fda9e3b653e587a2de4ba0ccb58"
    )

    assert v2["screen"]["tested_relation_count"] == 16
    assert v2["screen"]["strict_displayed_candidate_count"] == 10
    assert v2["screen"]["best_candidate"]["displayed_package_sum_pUSD"] == "0.75"
    assert book["economics"]["actual"]["cost_pUSD"] == "6.2"
    assert book["economics"]["actual"]["optimistic_zero_fee_profit_floor_pUSD"] == "-1.2"
    assert book["capture"]["within_frozen_age_gate"] is False
    assert book["capture"]["within_frozen_skew_gate"] is False
    assert book["capture"]["fee_receipts"] == {}

    assert v3["screen"]["tested_relation_count"] == 16
    assert v3["screen"]["strict_displayed_candidate_count"] == 0
    assert v3["screen"]["best_candidate"] is None
    assert min(
        row["rejection_proxy_sum_pUSD"] for row in v3["screen"]["relations"]
    ) == "1.26"
    assert v3["adjudication"]["accepted_edge"] is False


def test_soccer_terminal_is_bound_to_rank_31() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert "outcomePrices" in rank_31["next_action"]
    assert any(
        row["result_sha256"]
        == "aeb52c0bcd375fb0c31282cf8be92b9a6e6b93de0d8e7a0ac2c01bb579091386"
        for row in rank_31["canonical_artifacts"]
    )
    assert any(
        row["canonical_result_sha256"]
        == "aeb52c0bcd375fb0c31282cf8be92b9a6e6b93de0d8e7a0ac2c01bb579091386"
        for row in registry["terminal_do_not_repeat"]
    )


def test_retained_soccer_graph_exhausts_supported_relations() -> None:
    contract = _load(
        "polymarket-soccer-structural-graph-contract-v1-2026-08-30.json"
    )
    result = _load(
        "polymarket-soccer-structural-graph-result-v1-2026-08-30.json"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["population"] == {
        "main_exact_score_pairs": 11,
        "main_first_to_score_pairs": 10,
        "main_more_markets_pairs": 7,
        "tested_relation_count": 305,
        "side_specific_price_complete_count": 197,
        "missing_side_specific_price_count": 108,
    }
    assert result["screen"]["strict_side_specific_candidate_count"] == 0
    assert result["screen"]["best_price_complete_relation"][
        "rejection_proxy_sum_pUSD"
    ] == "1.15"
    assert result["authority"]["network_requests"] == 0

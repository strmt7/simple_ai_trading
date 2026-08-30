from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / "polymarket-soccer-exact-score-cross-family-contract-v1-2026-08-30.json"
RESULT = ACTION / "polymarket-soccer-exact-score-cross-family-result-v1-2026-08-30.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_exact_score_cross_family_graph_is_source_bound_and_exhaustive() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)

    assert _hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _hash(result, "result_sha256") == result["result_sha256"]
    assert hashlib.sha256(
        (ROOT / contract["retained_source"]["path"]).read_bytes()
    ).hexdigest() == contract["retained_source"]["sha256"]
    assert hashlib.sha256(
        (ROOT / contract["implementation"]["path"]).read_bytes()
    ).hexdigest() == contract["implementation"]["sha256"]

    population = result["population"]
    assert population == {
        "complete_base_count": 7,
        "exact_zero_zero_equivalent_neither_first_to_score_count": 7,
        "neither_first_to_score_equivalent_exact_zero_zero_count": 7,
        "nonzero_exact_score_implies_btts_side_count": 105,
        "nonzero_exact_score_implies_full_game_total_side_count": 630,
        "nonzero_exact_score_implies_team_total_side_count": 630,
        "one_sided_exact_score_implies_first_scorer_count": 42,
        "relation_count": 1428,
        "side_specific_price_complete_count": 1428,
        "side_specific_price_missing_count": 0,
        "strict_sub_floor_count": 0,
        "under_zero_point_five_implies_exact_zero_zero_count": 7,
    }
    assert result["best_complete_relation"]["rejection_proxy_sum_pUSD"] == "1"
    assert result["strict_sub_floor_relations"] == []
    assert result["adjudication"]["current_book_request_authorized"] is False
    assert result["authority"]["network_requests"] == 0


def test_exact_score_cross_family_graph_is_bound_to_rank_31_and_terminalized() -> None:
    result = _load(RESULT)
    registry = _load(REGISTRY)
    assert _hash(registry, "result_sha256") == registry["result_sha256"]

    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert {
        "path": "docs/model-research/action-value/polymarket-soccer-exact-score-cross-family-result-v1-2026-08-30.json",
        "result_sha256": result["result_sha256"],
    } in rank_31["canonical_artifacts"]
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert terminal[
        "polymarket_retained_soccer_exact_score_cross_family_implication_graph_2026_08_30"
    ]["canonical_result_sha256"] == result["result_sha256"]

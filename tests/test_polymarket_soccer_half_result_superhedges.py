from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT_V1 = ACTION / "polymarket-soccer-half-result-superhedge-contract-v1-2026-08-30.json"
FAILURE_V1 = ACTION / "polymarket-soccer-half-result-superhedge-failure-v1-2026-08-30.json"
CONTRACT_V2 = ACTION / "polymarket-soccer-half-result-superhedge-contract-v2-2026-08-30.json"
RESULT_V2 = ACTION / "polymarket-soccer-half-result-superhedge-result-v2-2026-08-30.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_half_result_contract_correction_and_result_are_source_bound() -> None:
    contract_v1 = _load(CONTRACT_V1)
    failure_v1 = _load(FAILURE_V1)
    contract_v2 = _load(CONTRACT_V2)
    result_v2 = _load(RESULT_V2)

    assert _hash(contract_v1, "contract_sha256") == contract_v1["contract_sha256"]
    assert _hash(failure_v1, "result_sha256") == failure_v1["result_sha256"]
    assert _hash(contract_v2, "contract_sha256") == contract_v2["contract_sha256"]
    assert _hash(result_v2, "result_sha256") == result_v2["result_sha256"]
    assert contract_v2["prior_representation_failure"]["result_sha256"] == failure_v1["result_sha256"]
    assert result_v2["authority"]["network_requests"] == 0


def test_half_result_graph_is_exhaustive_and_rejects_before_books() -> None:
    result = _load(RESULT_V2)
    population = result["population"]
    relations = result["screen"]["relations"]

    assert population == {
        "complete_base_halftime_second_half_triple_count": 10,
        "tested_relation_count": 90,
        "conjunction_relation_count": 70,
        "full_win_union_relation_count": 20,
        "strict_side_specific_candidate_count": 0,
    }
    assert len(relations) == 90
    assert all(not row["passes_strict_side_specific_rejection_gate"] for row in relations)
    assert result["screen"]["best_relation"]["rejection_proxy_sum_pUSD"] == "1.12"
    assert result["adjudication"]["after_all_cost_profit_floor_pUSD"] == "0"
    assert result["adjudication"]["next_action"] == "stop_without_any_venue_request"


def test_half_result_graph_is_bound_to_rank_31_and_terminalized() -> None:
    result = _load(RESULT_V2)
    registry = _load(REGISTRY)
    assert _hash(registry, "result_sha256") == registry["result_sha256"]

    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert {
        "path": "docs/model-research/action-value/polymarket-soccer-half-result-superhedge-result-v2-2026-08-30.json",
        "result_sha256": result["result_sha256"],
    } in rank_31["canonical_artifacts"]
    terminal = {
        row["family"]: row for row in registry["terminal_do_not_repeat"]
    }
    assert terminal[
        "polymarket_retained_soccer_full_halftime_second_half_result_superhedge_graph_2026_08_30"
    ]["canonical_result_sha256"] == result["result_sha256"]

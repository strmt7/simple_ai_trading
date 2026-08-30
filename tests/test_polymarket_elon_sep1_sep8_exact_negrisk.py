from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs" / "model-research" / "action-value"
REGISTRY = ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _canonical_hash(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sep1_sep8_prefilter_and_retained_depth_are_source_bound() -> None:
    prefilter_contract = _load(
        ACTION
        / "polymarket-elon-sep1-sep8-exact-negrisk-prefilter-contract-v1-2026-08-30.json"
    )
    prefilter_result = _load(
        ACTION
        / "polymarket-elon-sep1-sep8-exact-negrisk-prefilter-result-v1-2026-08-30.json"
    )
    adjudication_contract = _load(
        ACTION
        / "polymarket-elon-sep1-sep8-exact-negrisk-books-retained-adjudication-contract-v1-2026-08-30.json"
    )
    adjudication_result = _load(
        ACTION
        / "polymarket-elon-sep1-sep8-exact-negrisk-books-retained-adjudication-result-v1-2026-08-30.json"
    )

    assert _canonical_hash(prefilter_contract, "contract_sha256") == (
        prefilter_contract["contract_sha256"]
    )
    assert _canonical_hash(prefilter_result, "result_sha256") == (
        prefilter_result["result_sha256"]
    )
    assert prefilter_result["screen"]["event"]["market_count"] == 26
    assert prefilter_result["screen"]["event"]["displayed_all_yes_sum_pUSD"] == "1.0305"
    assert prefilter_result["screen"]["positive_displayed_conversion_candidate_count"] == 26

    assert _canonical_hash(adjudication_contract, "contract_sha256") == (
        adjudication_contract["contract_sha256"]
    )
    assert _canonical_hash(adjudication_result, "result_sha256") == (
        adjudication_result["result_sha256"]
    )
    sources = adjudication_contract["retained_sources"]
    assert _sha256(ROOT / sources["books_raw_path"]) == sources["books_raw_sha256"]
    assert _sha256(ROOT / sources["books_journal_path"]) == sources["books_journal_sha256"]
    assert adjudication_result["retained_capture"] == {
        "book_count": 52,
        "book_timestamp_skew_ms": 52902,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 59170,
        "request_elapsed_ms": 507,
        "response_bytes": 110476,
        "response_sha256": "571aae1e4c7570d8623712eaed1754b8a0bbb731ff9ed54d8b4948c4e6372896",
    }
    assert adjudication_result["screen"]["zero_fee_no_stress"]["best_path"]["net_quote"] == "-0.425"
    assert adjudication_result["screen"]["gamma_fee_no_stress"]["best_path"]["net_quote"] == "-0.66440"
    assert (
        adjudication_result["screen"]["gamma_fee_one_adverse_tick_each_leg"]["best_path"]["net_quote"]
        == "-1.12694"
    )
    assert adjudication_result["screen"]["candidate_after_all_frozen_gates"] is False
    assert adjudication_result["adjudication"]["network_requests"] == 0


def test_sep1_sep8_terminal_registry_lineage_is_exact() -> None:
    registry = _load(REGISTRY)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    artifacts = {row["path"]: row["result_sha256"] for row in rank["canonical_artifacts"]}
    assert artifacts[
        "docs/model-research/action-value/polymarket-elon-sep1-sep8-exact-negrisk-books-retained-adjudication-result-v1-2026-08-30.json"
    ] == "8391ecd524d0db9c312f51e627f391d95797cf9f5d32f569fd31210fe064bc83"
    terminal = {
        row["family"]: row for row in registry["terminal_do_not_repeat"]
    }
    assert terminal[
        "polymarket_Elon_September_1_to_September_8_exact_fixed_NegRisk_depth_2026_08_30"
    ]["canonical_result_sha256"] == (
        "8391ecd524d0db9c312f51e627f391d95797cf9f5d32f569fd31210fe064bc83"
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-rate-conflict-gate-v6-2026-08-26.json"
)
POST_CONFLICT_CONTRACT = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-post-conflict-contract-v7-2026-08-29.json"
)
POST_CONFLICT_ADJUDICATION = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-post-conflict-v7-failure-adjudication-2026-08-29.json"
)
POST_CONFLICT_JOURNAL = ROOT / (
    "data/polymarket-holding-yield-post-conflict-v7/journal.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_RESULT_HASH = (
    "17c23b1bf821256a573b8685ea4c5725d1c1315a4ca6449395e75635b51678d9"
)
EXPECTED_REGISTRY_HASH = (
    "5dfe720ff8cb69f5489ef6deb47fffe2d1ae4d036f1c14a13fbb34daf961f14a"
)
EXPECTED_POST_CONFLICT_CONTRACT_HASH = (
    "a8519183d418dcab02184cf57b2d900c1b7740ce1b72670bd8fbea71f798d132"
)
EXPECTED_POST_CONFLICT_ADJUDICATION_HASH = (
    "448b068aa5c1b34c6012a5fadafa449ed9ef125afc310b7901b9f68285510f71"
)


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_rate_conflict_gate_is_hash_bound_and_fail_closed() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["result_sha256"] == EXPECTED_RESULT_HASH
    assert _canonical_hash(artifact) == EXPECTED_RESULT_HASH
    assert artifact["request_budget"] == {
        "actual_get_requests": 7,
        "historical_requests_repeated": 0,
        "planned_get_requests": 7,
    }
    conflict = artifact["source_conflict"]
    assert conflict["help_center"]["annual_rate"] == "0.0325"
    assert conflict["developer_docs"]["annual_rate"] == "0.04"
    assert conflict["precedence_assumed"] is False
    assert artifact["verdict"]["current_operating_rate_qualified"] is False
    assert artifact["verdict"]["profitability_claim_at_4_percent"] is False


def test_current_balances_and_yield_refresh_are_exact() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cases = {case["asset"]: case for case in artifact["current_snapshots"]}

    assert cases["BTC"]["current_complete_set_shares"] == "150"
    assert cases["ETH"]["current_complete_set_shares"] == "440"
    assert cases["SOL"]["prior_complete_set_shares"] == "449"
    assert cases["SOL"]["current_complete_set_shares"] == "591.11"
    assert cases["SOL"]["position_denominator_changed"] is True
    assert all(case["new_yield_rows_after_prior_capture"] == 0 for case in cases.values())


def test_every_new_raw_response_matches_its_bound_hash() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    sources = [artifact["source_conflict"]["developer_docs"]["source"]]
    sources.extend(artifact["sources"])

    for source in sources:
        raw_path = ROOT / source["raw_path"]
        raw = raw_path.read_bytes()
        assert len(raw) == source["response_bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["response_sha256"]


def test_registry_routes_the_conflict_gate_without_promoting_four_percent() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _canonical_hash(registry) == EXPECTED_REGISTRY_HASH
    holding = next(
        candidate
        for candidate in registry["prioritized_hypotheses"]
        if candidate["mechanism"] == "complete_set_holding_reward"
    )
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": EXPECTED_RESULT_HASH,
    } in holding["canonical_artifacts"]
    assert "current_rate_remains_fail_closed_unqualified" in holding["current_status"]
    assert "do_not_rerun_or_repair" in holding["next_action"]
    assert "material_official_rate_program_payout" in holding["retry_trigger"]


def test_consumed_post_conflict_refresh_is_preserved_and_not_repeated() -> None:
    contract = json.loads(POST_CONFLICT_CONTRACT.read_text(encoding="utf-8"))
    claimed_contract = contract.pop("contract_result_sha256")
    assert claimed_contract == EXPECTED_POST_CONFLICT_CONTRACT_HASH
    assert _canonical_hash({**contract, "result_sha256": claimed_contract}) == claimed_contract

    result = json.loads(POST_CONFLICT_ADJUDICATION.read_text(encoding="utf-8"))
    assert result["result_sha256"] == EXPECTED_POST_CONFLICT_ADJUDICATION_HASH
    assert _canonical_hash(result) == EXPECTED_POST_CONFLICT_ADJUDICATION_HASH
    assert result["failure"]["actual_request_count"] == 5
    assert result["failure"]["remaining_requests_not_attempted"] == 4
    assert result["implementation_correction"]["false_failure_arithmetic"] == {
        "displayed_NO_current_value_pusd": "579.5833",
        "displayed_YES_current_value_pusd": "11.5266",
        "displayed_sum_pusd": "591.1099",
        "equal_mergeable_shares_per_outcome": "591.11",
        "rounding_difference_pusd": "-0.0001",
    }
    assert {
        case["asset"]: case["candidate_rate_sample_matches"] for case in result["offline_retained_evidence"]
    } == {
        "BTC": [{"annual_rate": "0.0325", "sampled_hours": 24}],
        "ETH": [{"annual_rate": "0.0325", "sampled_hours": 24}],
    }
    for source in result["raw_sources"]:
        raw = (ROOT / source["path"]).read_bytes()
        assert len(raw) == source["bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]

    journal = json.loads(POST_CONFLICT_JOURNAL.read_text(encoding="utf-8"))
    assert journal["state"] == "failed"
    assert journal["request_count"] == 5
    assert not (
        ROOT
        / "docs/model-research/polymarket/"
        "complete-set-holding-yield-post-conflict-v7-2026-08-29.json"
    ).exists()

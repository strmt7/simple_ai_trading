from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-rate-conflict-gate-v6-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_RESULT_HASH = (
    "17c23b1bf821256a573b8685ea4c5725d1c1315a4ca6449395e75635b51678d9"
)
EXPECTED_REGISTRY_HASH = (
    "69039bd49ca43a23822a5ce8997c80bc7cf7d629bf4c08f8ffc00507644864b7"
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
    assert "source_conflict_unresolved" in holding["current_status"]
    assert "not_before_2026_08_28T00_20_00Z" in holding["retry_trigger"]

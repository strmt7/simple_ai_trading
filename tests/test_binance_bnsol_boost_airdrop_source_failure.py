from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-bnsol-boost-airdrop-existing-holding-overlay-"
    "source-contract-v1-2026-08-30.json"
)
JOURNAL = BASE / (
    "binance-bnsol-boost-airdrop-existing-holding-overlay-"
    "journal-v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "binance-bnsol-boost-airdrop-existing-holding-overlay-"
    "source-failure-adjudication-v1-2026-08-30.json"
)
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


def test_contract_precedes_the_only_retained_source_request() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(journal, "journal_sha256") == journal["journal_sha256"]
    assert journal["completed_logical_request_count"] == 1
    assert journal["next_request"] is None
    frozen = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    requested = datetime.fromisoformat(
        journal["request"]["requested_before_utc"].replace("Z", "+00:00")
    )
    assert frozen < requested


def test_http_202_shell_is_retained_and_fails_every_required_term() -> None:
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))
    response = journal["request"]["response"]
    raw = (ROOT / response["raw_path"]).read_bytes()

    assert response["status_code"] == 202
    assert len(raw) == response["payload_bytes"] == 2038
    assert _sha256(raw) == response["payload_sha256"]
    assert not any(response["required_term_matches"].values())
    folded = raw.lower()
    for term in response["required_term_matches"]:
        assert term.encode("ascii") not in folded


def test_boost_lead_is_fail_closed_and_bound_to_existing_yield_family() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert artifact["decision"]["accepted_edge"] is False
    assert artifact["candidate_identity"]["public_forward_profit_floor"] == "0"
    assert artifact["authority"]["account_state_accessed"] is False
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["claim_or_other_state_changes"] == 0
    assert artifact["discovery_only_web_rendered_evidence"][
        "admitted_as_positive_acceptance_evidence"
    ] is False
    assert artifact["retry_contract"]["terminal_without_trigger"] is True

    source = artifact["retained_current_official_api_index"]
    index = (ROOT / source["path"]).read_bytes()
    assert _sha256(index) == source["sha256"]
    decoded = index.decode("utf-8")
    for endpoint in source["required_endpoint_fragments"]:
        assert f"`{endpoint}`" in decoded

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_idle_spot_native_token_yield"
    )
    assert family["priority_rank"] == 3
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "BNSOL_Boost" in family["retry_trigger"]


def test_agent_rules_prevent_repeating_known_empty_dynamic_pages() -> None:
    rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "HTTP 202 shells of about 2 KB" in rules
    assert "stop without URL aliases" in rules

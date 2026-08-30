from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-usdm-futures-bnb-just-in-time-fee-overlay-"
    "source-contract-v1-2026-08-30.json"
)
JOURNAL = BASE / (
    "binance-usdm-futures-bnb-just-in-time-fee-overlay-"
    "journal-v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "binance-usdm-futures-bnb-just-in-time-fee-overlay-"
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


def test_source_contract_precedes_the_single_retained_request() -> None:
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


def test_failed_response_is_hash_bound_and_cannot_prove_the_discount() -> None:
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))
    response = journal["request"]["response"]
    raw = (ROOT / response["raw_path"]).read_bytes()

    assert len(raw) == response["payload_bytes"] == 2035
    assert _sha256(raw) == response["payload_sha256"]
    assert response["status_code"] == 202
    assert not any(response["required_term_matches"].values())
    folded = raw.lower()
    for term in response["required_term_matches"]:
        assert term.encode("ascii") not in folded


def test_adjudication_is_fail_closed_and_registered_without_mutation() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert artifact["decision"]["accepted_edge"] is False
    assert artifact["decision"]["public_forward_profit_floor"] == "0"
    assert artifact["authority"]["account_state_accessed"] is False
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_trades"] == 0
    assert artifact["authority"]["state_changes"] == 0
    assert artifact["retry_contract"]["terminal_without_trigger"] is True

    api = artifact["retained_current_official_api_contract"]
    index = (ROOT / api["index_path"]).read_bytes()
    assert _sha256(index) == api["index_sha256"]
    decoded = index.decode("utf-8")
    assert "`GET /fapi/v1/feeBurn`" in decoded
    assert "Get BNB Burn Status (USER_DATA)" in decoded
    assert "`POST /fapi/v1/feeBurn`" in decoded
    assert "Toggle BNB Burn On Futures Trade (TRADE)" in decoded

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert family["priority_rank"] == 5
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "USDM_Futures_BNB_fee_reduction" in family["retry_trigger"]

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-exchange-link-organic-client-commission-rebate-"
    "candidate-v1-2026-08-30.json"
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


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return _sha256(_canonical(body))


def test_current_official_index_binds_exact_exchange_link_rebate_endpoints() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert _self_hash(artifact) == artifact["result_sha256"]

    source = artifact["source_contract"]["retained_current_official_api_index"]
    raw = (ROOT / source["path"]).read_bytes()
    assert _sha256(raw) == source["sha256"]

    lines = raw.decode("utf-8").splitlines()
    for evidence in artifact["source_contract"]["exact_evidence_lines"]:
        assert lines[evidence["line"] - 1] == evidence["text"]

    decoded = raw.decode("utf-8")
    assert "GET /sapi/v1/broker/rebate/recentRecord" in decoded
    assert "GET /sapi/v1/broker/rebate/futures/recentRecord" in decoded
    assert "GET /sapi/v1/apiReferral/kickback/recentRecord" in decoded
    assert "GET /sapi/v1/apiReferral/rebate/recentRecord" in decoded


def test_candidate_fails_closed_without_schema_security_or_account_evidence() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    adjudication = artifact["adjudication"]
    authority = artifact["authority"]

    assert adjudication["accepted_edge"] is False
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["deployment_ready"] is False
    assert artifact["source_contract"][
        "exact_endpoint_security_classification_bound"
    ] is False
    assert artifact["economic_contract"]["public_forward_floor_quote_units"] == "0"
    assert authority["credentials_used"] is False
    assert authority["official_venue_api_requests"] == 0
    assert authority["signed_requests"] == 0
    assert authority["orders_or_cancellations"] == 0


def test_existing_rank_24_family_registers_candidate_without_new_family() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 24
    )
    assert family["mechanism"] == "organic_third_party_platform_fee_overlays"
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "Exchange_Link" in family["retry_trigger"]
    assert "exact_rebate_endpoint_security" in family["next_action"]

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-caas-organic-client-commission-markup-overlay-v1-2026-08-30.json"
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


def test_current_official_index_proves_caas_markup_reporting_interfaces() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact) == artifact["result_sha256"]
    source = artifact["sources"]["current_official_api_index"]
    index = (ROOT / source["path"]).read_bytes()
    assert _sha256(index) == source["sha256"]

    decoded = index.decode("utf-8")
    assert "### VIP CAAS REST API (1.0.0)" in decoded
    assert "Crypto-as-a-Service Commission Markup APIs." in decoded
    for endpoint in artifact["signed_read_only_contract"][
        "required_endpoints_after_exact_authority"
    ]:
        line = next(row for row in decoded.splitlines() if f"`{endpoint}`" in row)
        assert "(USER_DATA)" in line


def test_failed_dynamic_doc_preflight_is_retained_without_credential_body() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    failed = artifact["sources"]["failed_general_info_preflight"]
    intent = json.loads((ROOT / failed["request_intent"]).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / failed["request_receipt"]).read_text(encoding="utf-8"))

    assert intent["request"]["method"] == receipt["method"] == "GET"
    assert intent["request"]["url"] == receipt["url"]
    assert receipt["status_code"] == 200
    assert receipt["payload_sha256"] == failed["response_sha256"]
    assert receipt["credentials_used"] is False
    assert receipt["raw_body_committed"] is False
    assert failed["positive_evidence_admitted"] is False
    assert not (
        ROOT / "docs/model-research/binance/raw/"
        "caas-commission-markup-source-v1-2026-08-30/01-general-info.raw.md"
    ).exists()


def test_narrow_realized_markup_overlay_is_accepted_without_mutation_authority() -> (
    None
):
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    adjudication = artifact["adjudication"]
    authority = artifact["authority"]

    assert adjudication["accepted_edge"] is True
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["deployment_ready"] is False
    assert adjudication["trading_authority"] is False
    assert "exact realized positive" in adjudication["accepted_scope"]
    assert (
        "independently existing bona fide external client trades"
        in adjudication["accepted_scope"]
    )
    assert artifact["economic_contract"]["public_forward_floor_quote_units"] == "0"
    assert authority["signed_requests"] == 0
    assert authority["fee_groups_or_members_mutated"] == 0
    assert artifact["signed_read_only_contract"]["state_changes_authorized"] is False

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

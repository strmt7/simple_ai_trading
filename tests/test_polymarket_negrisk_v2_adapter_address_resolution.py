from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-negrisk-v2-adapter-address-resolution-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_ARTIFACT_SHA256 = (
    "e11810a0215521cb5ad0c0c966340b4ff943760fda516e7841430fe057fe25fe"
)
EXPECTED_REGISTRY_SHA256 = (
    "671fa1498f9098357ac5c0f209c76351b0043cdcc1123dd8d8d062c92ac5c4a5"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_current_official_registry_resolves_only_adapter_identity() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_ARTIFACT_SHA256
    assert artifact["resolution"]["current_pusd_negrisk_collateral_adapter"] == (
        "0xadA2005600Dec949baf300f4C6120000bDB6eAab"
    )
    assert artifact["resolution"]["deprecated_clob_v1_adapter"] == (
        "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    )
    assert artifact["economic_impact"]["accepted_edge_count_change"] == 0
    assert artifact["economic_impact"]["protected_capture_touched"] is False
    assert artifact["verdict"]["accepted_edge"] is False
    assert artifact["authority"]["venue_market_data_requests_made"] == 0


def test_registry_uses_resolution_without_weakening_execution_gates() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_negative_risk_NO_to_YES_converter_recurrence"
    )
    assert {
        "path": (
            "docs/model-research/action-value/"
            "polymarket-negrisk-v2-adapter-address-resolution-v1-2026-08-27.json"
        ),
        "result_sha256": EXPECTED_ARTIFACT_SHA256,
    } in row["canonical_artifacts"]
    assert row["market_direction_forecast_required"] is False
    assert "terminally_rejected" in row["current_status"]
    assert any(
        "retired_CLOB_V1_adapter" in shortcut
        for shortcut in row["prohibited_shortcuts"]
    )

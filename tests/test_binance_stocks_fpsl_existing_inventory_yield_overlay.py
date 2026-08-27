from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-stocks-fpsl-existing-inventory-yield-overlay-candidate-v1-2026-08-27.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "3fe1801a6cbf442ab1ce79d1f3bd4586542d97414aea954b0bbd9a55a85453e1"
)
EXPECTED_REGISTRY_SHA256 = (
    "9c1d110fe26ae6875824b5c7fd68ee41998d4dd41479f3df8159faa5a67527b8"
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


def test_fpsl_candidate_is_source_bound_incremental_and_fail_closed() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["authority"] == {
        "account_requests": 0,
        "credentials_used": False,
        "fpsl_enrollment_changes": 0,
        "funded_actions": 0,
        "orders_or_stock_purchases": 0,
        "public_source_documents_bound": 2,
    }
    assert artifact["economics"]["public_forward_income_floor_USD"] == "0"
    assert artifact["economics"]["public_account_user_share_fraction"] is None
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["adjudication"]["trading_or_enrollment_authority"] is False
    assert artifact["expired_discovery_activity"][
        "leaderboard_bonus_credited_as_forward_profit"
    ] is False


def test_registry_tracks_fpsl_without_increasing_accepted_count() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 18
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 12
    )
    artifacts = {
        row["path"]: row["result_sha256"]
        for row in family["canonical_artifacts"]
    }
    assert ARTIFACT_PATH.relative_to(ROOT).as_posix() in artifacts
    assert artifacts[ARTIFACT_PATH.relative_to(ROOT).as_posix()] == (
        EXPECTED_RESULT_SHA256
    )
    assert "FPSL" in family["current_status"]
    assert "explicit_user_authority" in family["next_action"]

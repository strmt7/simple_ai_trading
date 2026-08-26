from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT / "docs/model-research/action-value/"
    "polymarket-negrisk-maker-input-gate-v1-2026-08-26.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_ARTIFACT_HASH = (
    "99cb35d065cfd2c12eb6947264b838c2c7407fba7582b14a94f30a856e8f0652"
)
EXPECTED_REGISTRY_HASH = (
    "06adf4d2c6bca1894d96c258e407134aa52b113f4c6f32abe17c282b8c729297"
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


def test_gate_is_hash_bound_fail_closed_and_direction_independent() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact) == EXPECTED_ARTIFACT_HASH
    assert artifact["adjudication"] == {
        **artifact["adjudication"],
        "accepted_edge": False,
        "candidate_edge": True,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
    }
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_conversions_submitted"] == 0
    economics = {
        (row["path"], row["quantity_shares"]): row["net_quote"]
        for row in artifact["role_economics_at_live_state"]["paths"]
    }
    assert economics[("all_taker", "20")] == "-0.28328"
    assert (
        economics[("maker_input_Bitcoin_NO_then_taker_sell_both_outputs", "20")]
        == "0.02960"
    )


def test_every_tracked_source_receipt_reconstructs_exactly() -> None:
    artifact = _load(ARTIFACT_PATH)

    for receipt in artifact["tracked_evidence"]:
        path = ROOT / receipt["path"]
        payload = path.read_bytes()
        assert len(payload) == receipt["bytes"]
        assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
        if "decompressed_sha256" in receipt:
            decompressed = gzip.decompress(payload)
            assert (
                hashlib.sha256(decompressed).hexdigest()
                == receipt["decompressed_sha256"]
            )
            if "decompressed_bytes" in receipt:
                assert len(decompressed) == receipt["decompressed_bytes"]


def test_registry_binds_the_exact_event_scope_without_promoting_the_edge() -> None:
    artifact = _load(ARTIFACT_PATH)
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "polymarket_negative_risk_NO_to_YES_converter_recurrence"
    )
    assert candidate["priority_rank"] == 2
    assert candidate["venue_scope"] == (
        "polymarket_event_106981_bitcoin_gold_sp500_fixed_non_augmented_negative_risk"
    )
    assert candidate["canonical_artifacts"][0] == {
        "path": ARTIFACT_PATH.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    }
    assert registry["accepted_edge_count"] == 9
    assert artifact["research_decision"]["accepted_edge_count_change"] == 0
    assert (
        artifact["prospective_fill_and_unwind_capture"]["active_result_claim"] is False
    )

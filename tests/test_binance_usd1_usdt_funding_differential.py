from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
ARTIFACT_PATH = (
    ACTION_VALUE / "binance-usd1-usdt-funding-differential-v1-2026-08-26.json"
)
CONTRACT_PATH = ACTION_VALUE / "binance-usd1-usdt-funding-differential-contract-v1.json"
TOOL_PATH = ROOT / "tools" / "screen_binance_usd1_usdt_funding_differential.py"
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
CONTINUATION_PATH = ROOT / "docs" / "CONTINUATION.md"
EXPECTED_RESULT_HASH = (
    "0d82da55d7687b0f26dc12103f38394b3325542a62af10abf4d34609fa5a6e79"
)
EXPECTED_CONTRACT_HASH = (
    "e487ee1ca43dec3f7e237ea2a19b1b068c1510603622af13e9fc7f4f23209bc7"
)
EXPECTED_TOOL_HASH = "2218e58e3a0a562305a3a418bfbf9647a51076f66fa5556151eee58d326463df"
EXPECTED_REGISTRY_HASH = (
    "95283a6ce281bd2e3d6c44ad4a196605a9d7f5cfa424ad474b75bfec593c3ebe"
)
EXPECTED_TERMINAL_REASON = (
    "zero_of_two_public_candidates_passed_and_every_BTC_and_ETH_training_"
    "validation_and_test_role_was_negative_after_frozen_execution_two_leg_"
    "capital_and_observed_USD1_FX_stress"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _embedded_hash(value: dict[str, object]) -> str:
    body = dict(value)
    embedded = body.pop("result_sha256")
    calculated = hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()
    assert embedded == calculated
    return calculated


def test_artifact_contract_and_implementation_are_source_bound() -> None:
    artifact = _load(ARTIFACT_PATH)
    assert _embedded_hash(artifact) == EXPECTED_RESULT_HASH
    assert artifact["implementation"] == {
        "path": "tools/screen_binance_usd1_usdt_funding_differential.py",
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH

    contract = _load(CONTRACT_PATH)
    assert _embedded_hash(contract) == EXPECTED_CONTRACT_HASH
    assert artifact["contract"] == {
        "path": (
            "docs/model-research/action-value/"
            "binance-usd1-usdt-funding-differential-contract-v1.json"
        ),
        "result_sha256": EXPECTED_CONTRACT_HASH,
    }


def test_public_screen_rejects_both_bases_without_authority_or_orders() -> None:
    artifact = _load(ARTIFACT_PATH)
    verdict = artifact["verdict"]
    assert verdict["accepted_edge"] is False
    assert verdict["profitability_claim"] is False
    assert verdict["credentials_used"] is False
    assert verdict["orders_submitted"] == 0
    assert verdict["public_persistence_candidate_count"] == 0

    rows = {row["base"]: row for row in artifact["evaluations"]}
    assert set(rows) == {"BTC", "ETH"}
    assert artifact["fx_evidence"]["worst_30_day_decline_stress_bips"].startswith(
        "23.9544"
    )
    assert all(row["public_persistence_candidate"] is False for row in rows.values())
    assert all(
        role["net_after_observed_fx_stress_bips"].startswith("-")
        for row in rows.values()
        for role in row["roles"].values()
    )


def test_terminal_registry_and_continuation_bind_the_result() -> None:
    registry = _load(REGISTRY_PATH)
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert terminal["binance_usd1_usdt_static_perpetual_funding_differential"] == {
        "canonical_result_sha256": EXPECTED_RESULT_HASH,
        "family": "binance_usd1_usdt_static_perpetual_funding_differential",
        "reason": EXPECTED_TERMINAL_REASON,
    }

    continuation = CONTINUATION_PATH.read_text(encoding="utf-8")
    assert EXPECTED_RESULT_HASH in continuation
    assert EXPECTED_REGISTRY_HASH in continuation

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
    / "binance-stable-yield-allocation-evidence-gate-v1.json"
)
PROMOTION_TRIAGE_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-public-promotion-yield-triage-v1-2026-08-26.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "3096867474c4b5a0b3f893645bac68081ceb3783ad14393261e6d88793b64a8a"
)
EXPECTED_REGISTRY_SHA256 = (
    "ab4328c1ee4ab7bd1553ebf23e7b973f43535ae99cdeda1158a23d07a3c2fbf5"
)
EXPECTED_PROMOTION_TRIAGE_SHA256 = (
    "fe34f9aaf64a0ec920b0cf7cc7fd1141d30880d1205454779327e41fd7521b1c"
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


def test_stable_yield_gate_hash_and_fail_closed_authority_reconstruct() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["preflight"]["request_count"] == 0
    assert artifact["authority"] == {
        "accepted_edge": False,
        "credentials_used": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }


def test_prequalification_is_get_only_and_excludes_marketing_apr() -> None:
    artifact = _load(ARTIFACT_PATH)
    prequalification = artifact["read_only_prequalification"]
    endpoints = prequalification["endpoints"]

    assert prequalification["endpoint_count"] == 6
    assert {row["method"] for row in endpoints} == {"GET"}
    assert {row["security_type"] for row in endpoints} == {"USER_DATA"}
    assert {row["path"] for row in endpoints} == {
        "/sapi/v1/bfusd/quota",
        "/sapi/v1/bfusd/history/rateHistory",
        "/sapi/v1/rwusd/quota",
        "/sapi/v1/rwusd/history/rateHistory",
        "/sapi/v1/simple-earn/flexible/list",
    }
    alternatives = [
        row for row in endpoints if row["path"] == "/sapi/v1/simple-earn/flexible/list"
    ]
    assert [row["request_parameters"] for row in alternatives] == [
        {"asset": "USDT"},
        {"asset": "USDC"},
    ]
    assert artifact["discovery_observation"]["adjudication"] == (
        "exclude_from_economic_evidence"
    )
    assert artifact["mechanism"]["market_direction_forecast_required"] is False
    assert artifact["mechanism"]["not_market_invariant"] is True


def test_registry_prioritizes_candidate_without_opening_authority() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 16))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert candidate["priority_rank"] == 5
    assert candidate["market_direction_forecast_required"] is False
    assert candidate["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-stable-yield-allocation-evidence-gate-v1.json"
            ),
            "result_sha256": EXPECTED_RESULT_SHA256,
        },
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-bfusd-spot-redemption-parity-v1-2026-08-26.json"
            ),
            "result_sha256": (
                "566be5e515ac14d38377b6a6b42101cc9b8a65585142053791b759efbd77f6bb"
            ),
        },
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-public-promotion-yield-triage-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_PROMOTION_TRIAGE_SHA256,
        },
        {
            "path": (
                "docs/model-research/action-value/"
                "binance-usd1-simple-earn-promotion-gate-v1-2026-08-26.json"
            ),
            "result_sha256": (
                "230b1524f337964394a45ffe047adfd19b35b339a7735866a15cafdd7549c6f1"
            ),
        },
    ]


def test_public_promotion_triage_is_conditional_and_not_accepted() -> None:
    artifact = _load(PROMOTION_TRIAGE_PATH)

    assert artifact["result_sha256"] == EXPECTED_PROMOTION_TRIAGE_SHA256
    assert _embedded_hash(artifact) == EXPECTED_PROMOTION_TRIAGE_SHA256
    assert (
        artifact["rlusd_xrp_campaign"]["published_completed_week"][
            "effective_apr_percent"
        ]
        == "8.07"
    )
    adjudication = artifact["adjudication"]
    assert adjudication["rlusd_public_conditional_candidate"] is True
    assert adjudication["accepted_stable_edge"] is False
    assert adjudication["profitability_claim"] is False

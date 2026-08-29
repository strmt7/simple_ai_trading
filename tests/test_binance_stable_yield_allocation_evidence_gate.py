from __future__ import annotations

import hashlib
import json
from decimal import Decimal
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
REFRESH_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-scheduled-yield-distribution-refresh-v1-2026-08-29.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "3096867474c4b5a0b3f893645bac68081ceb3783ad14393261e6d88793b64a8a"
)
EXPECTED_REGISTRY_SHA256 = (
    "0511b6dbb8f560470335fb6146edade7a50c3f24406c529f03a3f1fca769409b"
)
EXPECTED_PROMOTION_TRIAGE_SHA256 = (
    "26efd481a5ff424ca17ec803bb6a1a3ae8949d1fe0fc31a03e20a35d08d031ac"
)
EXPECTED_REFRESH_SHA256 = (
    "c5feb852830adadd497aa287460d1a3132e324fbbbdaa5f608890acebc43e252"
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
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert candidate["priority_rank"] == 8
    assert candidate["market_direction_forecast_required"] is False
    assert candidate["canonical_artifacts"][:5] == [
        {
            "path": REFRESH_PATH.relative_to(ROOT).as_posix(),
            "result_sha256": EXPECTED_REFRESH_SHA256,
        },
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
    assert {
        row["path"]: row["result_sha256"]
        for row in candidate["canonical_artifacts"]
    }[
        "docs/model-research/action-value/"
        "binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json"
    ] == "6f44b65e5aa85d33cc02e8611a372162cf00f4162fdff99828a31cf498ced6f9"


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
    usdc = next(
        row
        for row in artifact["current_capped_simple_earn_offers"]
        if row["asset"] == "USDC"
    )
    assert usdc["explicit_promotion_period_in_source"] is False
    assert usdc["forward_bonus_reward_floor_usdc"] == "0"
    assert "period_start_utc" not in usdc
    assert "period_end_utc" not in usdc
    assert "maximum_14_day_bonus_usd" not in usdc
    assert artifact["source_binding_correction"]["venue_requests_added"] == 0


def test_scheduled_distribution_refresh_retains_raw_and_updates_rates() -> None:
    refresh = _load(REFRESH_PATH)

    assert refresh["result_sha256"] == EXPECTED_REFRESH_SHA256
    assert _embedded_hash(refresh) == EXPECTED_REFRESH_SHA256
    assert refresh["adjudication"]["accepted_edge_count_change"] == 0
    assert refresh["adjudication"]["rlusd_candidate_accepted"] is False
    assert refresh["usd1"]["base_APRs_percent_by_completed_distribution"] == [
        "4.85",
        "5.46",
        "5.27",
    ]
    assert refresh["rlusd"]["completed_week_APRs_percent"] == ["8.07", "5.78"]
    sensitivity = refresh["usd1"][
        "mutually_exclusive_fixed_7_percent_Simple_Earn_bonus_sensitivity"
    ]
    assert sensitivity["annualized_uplift_over_latest_base_bips"] == "173"
    assert Decimal(sensitivity["break_even_days_after_forfeiting_one_latest_base_airdrop_day"]) < 3.05

    for source in refresh["sources"]:
        payload = (ROOT / source["raw_path"]).read_bytes()
        assert len(payload) == source["raw_bytes"]
        assert hashlib.sha256(payload).hexdigest() == source["raw_sha256"]

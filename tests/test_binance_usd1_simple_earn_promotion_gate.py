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
    / "binance-usd1-simple-earn-promotion-gate-v1-2026-08-26.json"
)
EXPECTED_RESULT_SHA256 = (
    "230b1524f337964394a45ffe047adfd19b35b339a7735866a15cafdd7549c6f1"
)


def _load() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


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


def test_usd1_gate_hash_authority_and_time_limit_reconstruct() -> None:
    artifact = _load()

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "conditional_time_limited_candidate_rejected_as_stable_edge_before_"
            "start_and_exact_account_cost_evidence"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["funded_actions"] == 0
    assert artifact["promotion_terms"]["period_start_utc"] == (
        "2026-08-27T00:00:00Z"
    )


def test_conservative_margin_fails_closed_before_unknown_costs() -> None:
    artifact = _load()
    economics = artifact["economics"]
    margin = economics["fail_closed_margin"]

    assert Decimal(margin["fixed_28_day_increment_bips"]) > Decimal("25")
    assert Decimal(
        margin[
            "margin_after_worst_historical_30_day_close_move_and_current_"
            "displayed_spread_bips"
        ]
    ) < Decimal("1")
    assert Decimal(margin["margin_after_same_move_spread_and_10_bips_other_costs"]) < 0
    assert artifact["historical_usd1usdt_risk"][
        "rolling_30_day_windows_worse_than_minus_19_1694_bips"
    ] == 1


def test_public_convert_bounds_are_not_misrepresented_as_execution() -> None:
    artifact = _load()
    market_access = artifact["market_access"]

    assert market_access["convert_catalog"]["USDT_to_USD1"]["from_min"] == "0.01"
    assert "does not prove an executable conversion rate" in market_access[
        "convert_catalog"
    ]["limitation"]
    assert artifact["credential_preflight"] == {
        "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_KEY_present": False,
        "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_SECRET_present": False,
        "values_inspected": False,
    }

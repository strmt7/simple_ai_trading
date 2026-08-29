from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/polymarket/crypto-twap-5m-current-rewards-list-join-v1-2026-08-27.json"
)
RAW = (
    ROOT
    / "docs/model-research/polymarket/raw/crypto-twap-5m-current-rewards-list-join-v1-2026-08-27"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_complete_current_rewards_list_has_no_exact_five_minute_join() -> None:
    artifact = json.loads(RESULT.read_text(encoding="ascii"))
    claimed = artifact.pop("result_sha256")
    assert _sha256(_canonical(artifact)) == claimed

    sources = artifact["sources"]
    contract_path = ROOT / sources["contract_path"]
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    contract_claimed = contract.pop("result_sha256")
    assert contract_claimed == sources["contract_result_sha256"]
    assert _sha256(_canonical(contract)) == contract_claimed
    assert _sha256(contract_path.read_bytes()) == sources["contract_file_sha256"]
    assert _sha256(
        (ROOT / "tools/screen_polymarket_crypto_twap_5m_current_rewards_list.py").read_bytes()
    ) == sources["tool_sha256"]

    pages = artifact["current_rewards_pages"]
    assert pages == [
        {
            "declared_count": 54,
            "declared_limit": 500,
            "next_cursor": "LTE=",
            "page": 1,
            "row_count": 54,
            "target_condition_matches": 0,
        }
    ]
    identities = artifact["market_identities"]
    assert [row["asset"] for row in identities] == [
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "HYPE",
        "BNB",
        "DOGE",
    ]
    assert all(row["slug"].endswith("-1787815200") for row in identities)
    target_conditions = {row["condition_id"] for row in identities}

    gamma = sources["gamma_request"]
    gamma_raw = RAW / "01-gamma-seven-markets.raw"
    assert gamma_raw.stat().st_size == gamma["payload_bytes"]
    assert _sha256(gamma_raw.read_bytes()) == gamma["payload_sha256"]
    assert len(json.loads(gamma_raw.read_text(encoding="utf-8"))) == 7

    assert len(sources["current_rewards_requests"]) == 1
    reward_source = sources["current_rewards_requests"][0]
    reward_raw = RAW / "02-current-rewards-page-01.raw"
    assert reward_raw.stat().st_size == reward_source["payload_bytes"]
    assert _sha256(reward_raw.read_bytes()) == reward_source["payload_sha256"]
    reward_payload = json.loads(reward_raw.read_text(encoding="utf-8"))
    assert reward_payload["next_cursor"] == "LTE="
    assert reward_payload["count"] == len(reward_payload["data"]) == 54
    assert target_conditions.isdisjoint(
        str(row["condition_id"]).lower() for row in reward_payload["data"]
    )

    verdict = artifact["verdict"]
    assert verdict["status"] == "rejected_without_resampling"
    assert verdict["books_requested"] is False
    assert verdict["publicly_proven_reward_payout_floor_pUSD"] == "0"
    assert verdict["accepted_edge"] is False
    assert artifact["authority"] == {
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_cancellations": 0,
        "public_read_only": True,
    }
    assert not list(RAW.glob("*book*"))

    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    registry_claimed = registry.pop("result_sha256")
    assert registry_claimed == "5dfe720ff8cb69f5489ef6deb47fffe2d1ae4d036f1c14a13fbb34daf961f14a"
    assert _sha256(_canonical(registry)) == registry_claimed
    reward_family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "paired_crypto_maker_rebates_and_twap_liquidity_rewards"
    )
    result_ref = next(
        row
        for row in reward_family["canonical_artifacts"]
        if row["path"].endswith(
            "crypto-twap-5m-current-rewards-list-join-v1-2026-08-27.json"
        )
    )
    assert result_ref["result_sha256"] == claimed

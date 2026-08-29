from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/binance-hyperliquid-cross-venue-funding-spread-extension-v1-2026-08-27.json"
)
RAW = (
    ROOT
    / "docs/model-research/action-value/raw/binance-hyperliquid-cross-venue-funding-spread-extension-v1-2026-08-27"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_extension_reconstructs_complete_sources_and_rejects_the_hurdle() -> None:
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
        (ROOT / "tools/research_binance_hyperliquid_cross_venue_funding_spread.py").read_bytes()
    ) == sources["tool_sha256"]

    requests = sources["public_requests"]
    assert sources["public_request_count"] == len(requests) == 29
    assert len({row["role"] for row in requests}) == 29
    for row in requests:
        assert row["status_code"] == 200
        raw_path = RAW / f"{row['role']}.raw"
        assert raw_path.stat().st_size == row["payload_bytes"]
        assert _sha256(raw_path.read_bytes()) == row["payload_sha256"]
        assert (RAW / f"{row['role']}.intent.json").is_file()
        assert (RAW / f"{row['role']}.response.json").is_file()

    extension = artifact["extension"]
    assert extension["elapsed_days"] == 70
    assert [row["asset"] for row in extension["asset_results"]] == [
        "BTC",
        "ETH",
        "SOL",
        "DOGE",
    ]
    for row in extension["asset_results"]:
        coverage = row["coverage"]
        assert coverage["Hyperliquid_rows"] == 1680
        assert coverage["expected_Hyperliquid_hourly_rows"] == 1680
        assert coverage["Binance_funding_events"] == 210
        assert coverage["Binance_premium_hourly_rows"] == 1680
        assert coverage["synchronized_premium_hours"] == 1680
        assert coverage["overlap_days"] == 70
        assert Decimal(coverage["Hyperliquid_hourly_coverage_fraction"]) == 1

    hurdle = Decimal(artifact["hurdle"]["percent"])
    assert hurdle == Decimal("3.86")
    assert Decimal(extension["primary_equal_weight_after_cost_APR_percent"]) == Decimal(
        "1.203008792500000000000000000"
    )
    assert all(
        Decimal(row["extension_economics"]["after_cost_APR_percent"]) < hurdle
        for row in extension["asset_results"]
    )
    assert artifact["gates"] == {
        "archive_primary_each_year_positive": True,
        "extension_primary_after_cost_positive": True,
        "extension_primary_and_basket_exceed_DGS3MO": False,
        "extension_primary_coverage_passed": True,
    }
    assert artifact["verdict"]["status"] == "rejected_without_refitting_or_resampling"
    assert artifact["verdict"]["public_profit_floor_USD"] == "0"
    assert artifact["authority"] == {
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_cancellations": 0,
        "public_read_only": True,
    }

    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    registry_claimed = registry.pop("result_sha256")
    assert registry_claimed == "44fdf0cba6b97bcf40c407bc78cedbdbf8051ff1b7e40267b5bc4db629abb22a"
    assert _sha256(_canonical(registry)) == registry_claimed
    terminal = {
        row["family"]: row for row in registry["terminal_do_not_repeat"]
    }
    assert (
        terminal[
            "binance_Hyperliquid_cross_venue_perpetual_funding_spread_current_extension"
        ]["canonical_result_sha256"]
        == claimed
    )

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_binance_bybit_funding import (  # noqa: E402
    _bybit_series,
    _canonical_hash,
    _role_result,
)


ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "binance-bybit-btc-eth-sol-funding-adjudication-contract-v1-2026-09-01.json"
)
RESULT = ACTION / (
    "binance-bybit-btc-eth-sol-funding-adjudication-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def test_bybit_funding_rate_normalization(tmp_path: Path) -> None:
    source = tmp_path / "funding.json"
    source.write_text(
        json.dumps(
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "category": "linear",
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "fundingRateTimestamp": "28800001",
                            "fundingRate": "0.0001",
                        },
                        {
                            "symbol": "BTCUSDT",
                            "fundingRateTimestamp": "57600009",
                            "fundingRate": "-0.0002",
                        },
                    ],
                },
            }
        )
    )
    series, jitter = _bybit_series(
        source,
        symbol="BTCUSDT",
        interval_ms=28_800_000,
        maximum_jitter_ms=1_000,
    )
    assert series == {
        28_800_000: Decimal("0.0001"),
        57_600_000: Decimal("-0.0002"),
    }
    assert jitter == 9


def test_role_gate_charges_execution_capital_and_cross_venue_stress() -> None:
    contract = {
        "population": {"interval_hours": 8},
        "economics": {
            "round_trip_execution_bips": "20",
            "annual_two_leg_capital_hurdle_percent": "10",
            "quote_unit_stress_bips": "0",
            "custody_transfer_latency_failure_stress_bips": "25",
        },
        "gates": {"minimum_positive_interval_fraction": "0.5"},
    }
    weak = _role_result([Decimal("0.0001")] * 22, contract)
    strong = _role_result([Decimal("0.001")] * 22, contract)
    assert weak["passes"] is False
    assert Decimal(weak["net_after_all_frozen_hurdles_bips"]) < 0
    assert strong["passes"] is True


def test_preregistered_population_is_hash_bound_and_terminal() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))
    preregistration_path = ROOT / contract["preregistration"]["path"]
    preregistration = json.loads(preregistration_path.read_text(encoding="ascii"))
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert (
        _canonical_hash(preregistration, "contract_sha256")
        == preregistration["contract_sha256"]
    )
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == implementation["sha256"]
    )
    assert result["population"]["aligned_row_count_per_asset"] == 90
    assert result["gates"] == {
        "basis_or_book_capture_justified": False,
        "funding_only_prefilter_passed": False,
        "surviving_assets": [],
    }
    assert all(
        not row["passes_every_role"] for row in result["asset_results"].values()
    )
    assert all(
        Decimal(row["roles"]["test"]["net_after_all_frozen_hurdles_bips"])
        < Decimal("-62")
        for row in result["asset_results"].values()
    )


def test_six_public_captures_are_durable_and_unauthenticated() -> None:
    source_results = sorted(
        ACTION.glob("binance-bybit-*-source-result-v1-2026-09-01.json")
    )
    assert len(source_results) == 6
    for result_path in source_results:
        result = json.loads(result_path.read_text(encoding="ascii"))
        contract = json.loads(
            (ROOT / result["contract"]["path"]).read_text(encoding="ascii")
        )
        raw_path = ROOT / result["capture"]["receipt"]["raw_path"]
        journal_path = ROOT / contract["outputs"]["journal_path"]
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        assert _canonical_hash(contract, "contract_sha256") == contract[
            "contract_sha256"
        ]
        assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
        assert (
            hashlib.sha256(raw_path.read_bytes()).hexdigest()
            == result["capture"]["receipt"]["response_sha256"]
        )
        assert [row["phase"] for row in journal] == ["intent", "completed"]
        assert result["authority"]["credentials_used"] is False
        assert result["authority"]["protected_capture_touched"] is False


def test_registry_terminalizes_exact_bybit_family_without_promotion() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_Bybit_BTC_ETH_SOL_USDT_perpetual_funding_spread_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]

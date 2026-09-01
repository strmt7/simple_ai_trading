from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_binance_wallet_aster_usd1_funding import (  # noqa: E402
    _canonical_hash,
    _normalized_series,
    _role_result,
)


def test_timestamp_normalization_rejects_duplicates_and_large_jitter() -> None:
    rows = [
        {"symbol": "BTCUSD1", "fundingTime": 28_800_001, "fundingRate": "0.1"},
        {"symbol": "BTCUSD1", "fundingTime": 57_600_009, "fundingRate": "0.2"},
    ]
    series, jitter = _normalized_series(
        rows, symbol="BTCUSD1", interval_ms=28_800_000, maximum_jitter_ms=1_000
    )
    assert list(series) == [28_800_000, 57_600_000]
    assert jitter == 9


def test_role_gate_charges_execution_capital_and_fx() -> None:
    contract = {
        "population": {"interval_hours": 8},
        "economics": {
            "round_trip_execution_bips": "20",
            "annual_two_leg_capital_hurdle_percent": "10",
            "usd1_usdt_stress_bips": "23.95448647569617726319992000",
        },
        "gates": {"minimum_positive_interval_fraction": "0.60"},
    }
    positive = _role_result([Decimal("0.001")] * 30, contract)
    negative = _role_result([Decimal("0.0001")] * 30, contract)
    assert positive["passes"] is True
    assert Decimal(positive["net_after_all_frozen_hurdles_bips"]) > 0
    assert negative["passes"] is False
    assert Decimal(negative["net_after_all_frozen_hurdles_bips"]) < 0


def test_frozen_artifact_is_hash_bound_if_present() -> None:
    contract_path = (
        ROOT
        / "docs/model-research/action-value"
        / ("binance-wallet-aster-usd1-funding-prefilter-contract-v1-2026-09-01.json")
    )
    result_path = (
        ROOT
        / "docs/model-research/action-value"
        / ("binance-wallet-aster-usd1-funding-prefilter-result-v1-2026-09-01.json")
    )
    if not contract_path.exists() or not result_path.exists():
        return
    contract = json.loads(contract_path.read_text(encoding="ascii"))
    result = json.loads(result_path.read_text(encoding="ascii"))
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == (implementation["sha256"])
    )
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["adjudication"]["accepted_edge"] is False
    assert result["authority"]["network_requests"] == 0


def test_frozen_population_fails_before_basis_books_or_accounts() -> None:
    result_path = (
        ROOT
        / "docs/model-research/action-value"
        / ("binance-wallet-aster-usd1-funding-prefilter-result-v1-2026-09-01.json")
    )
    result = json.loads(result_path.read_text(encoding="ascii"))

    assert result["population"]["aligned_row_count_per_asset"] == 209
    assert result["gates"] == {
        "basis_or_book_capture_justified": False,
        "funding_only_prefilter_passed": False,
        "surviving_assets": [],
    }
    assert result["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "public_profit_floor_quote_units": "0",
        "stable_profitability_proved": False,
        "status": "terminal_funding_prefilter_rejection_before_books",
    }
    assert all(
        not asset["passes_every_role"] for asset in result["asset_results"].values()
    )
    assert all(
        Decimal(asset["roles"]["test"]["gross_funding_spread_bips"]) < 0
        for asset in result["asset_results"].values()
    )


def test_four_public_captures_are_hash_bound_and_unauthenticated() -> None:
    action_value = ROOT / "docs/model-research/action-value"
    result_paths = [
        action_value
        / "binance-wallet-aster-usd1-perpetual-inventory-source-result-v1-2026-09-01.json",
        *sorted(
            action_value.glob(
                "binance-wallet-aster-*usd1-funding-source-result-v1-2026-09-01.json"
            )
        ),
    ]

    assert len(result_paths) == 4
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="ascii"))
        contract_path = ROOT / result["contract"]["path"]
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        receipt = result["capture"]["receipt"]
        raw_path = ROOT / receipt["raw_path"]
        journal_path = ROOT / contract["outputs"]["journal_path"]
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]

        assert (
            _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
        )
        assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
        assert result["contract"]["sha256"] == contract["contract_sha256"]
        assert all(
            hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
            for item in contract["implementations"]
        )
        assert receipt["status_code"] == 200
        assert receipt["response_bytes"] == raw_path.stat().st_size
        assert (
            receipt["response_sha256"]
            == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        )
        assert [row["phase"] for row in journal] == ["intent", "completed"]
        assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
        assert result["authority"]["credentials_used"] is False
        assert result["authority"]["protected_capture_touched"] is False


def test_registry_terminalizes_exact_aster_population_without_promotion() -> None:
    result_path = (
        ROOT
        / "docs/model-research/action-value"
        / ("binance-wallet-aster-usd1-funding-prefilter-result-v1-2026-09-01.json")
    )
    registry_path = (
        ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    registry = json.loads(registry_path.read_text(encoding="ascii"))

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_wallet_Aster_BTC_ETH_SOL_USD1_vs_Binance_USDT_perpetual_funding_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 47

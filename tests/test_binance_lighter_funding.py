from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_binance_lighter_funding import (  # noqa: E402
    _canonical_hash,
    _lighter_series,
    _role_result,
)


ACTION_VALUE = ROOT / "docs/model-research/action-value"
RESULT = ACTION_VALUE / (
    "binance-lighter-btc-eth-sol-funding-adjudication-v1-2026-09-01.json"
)
CONTRACT = ACTION_VALUE / (
    "binance-lighter-btc-eth-sol-funding-adjudication-contract-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def test_lighter_rate_normalization_uses_direction_and_percentage_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "funding.json"
    source.write_text(
        json.dumps(
            {
                "code": 200,
                "resolution": "1h",
                "fundings": [
                    {"timestamp": 3600, "value": "1", "rate": "0.0012", "direction": "long"},
                    {"timestamp": 7200, "value": "1", "rate": "0.0008", "direction": "short"},
                ],
            }
        )
    )
    series = _lighter_series(source, expected_rows=2)
    assert series == {
        3_600_000: Decimal("0.000012"),
        7_200_000: Decimal("-0.000008"),
    }


def test_role_gate_charges_every_frozen_hurdle() -> None:
    contract = {
        "economics": {
            "round_trip_execution_bips": "20",
            "annual_two_leg_capital_hurdle_percent": "10",
            "usdc_usdt_quote_unit_stress_bips": "25",
            "custody_bridge_latency_stress_bips": "25",
        },
        "gates": {"minimum_positive_interval_fraction": "0.5"},
    }
    weak = _role_result([Decimal("0.0001")] * 22, contract)
    strong = _role_result([Decimal("0.001")] * 22, contract)
    assert weak["passes"] is False
    assert Decimal(weak["net_after_all_frozen_hurdles_bips"]) < 0
    assert strong["passes"] is True


def test_frozen_lighter_population_is_hash_bound_and_terminal() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    implementation = contract["implementation"]
    assert hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest() == implementation["sha256"]
    assert result["population"]["aligned_row_count_per_asset"] == 88
    assert result["gates"] == {
        "basis_or_book_capture_justified": False,
        "funding_only_prefilter_passed": False,
        "surviving_assets": [],
    }
    assert result["adjudication"]["status"] == "terminal_funding_prefilter_rejection_before_books"
    assert result["adjudication"]["accepted_edge"] is False
    assert all(not row["passes_every_role"] for row in result["asset_results"].values())


def test_four_lighter_public_captures_are_durable_and_unauthenticated() -> None:
    source_results = [
        ACTION_VALUE / "binance-lighter-btc-eth-sol-perpetual-inventory-source-result-v1-2026-09-01.json",
        *sorted(ACTION_VALUE.glob("binance-lighter-???-funding-source-result-v1-2026-09-01.json")),
    ]
    assert len(source_results) == 4
    for result_path in source_results:
        result = json.loads(result_path.read_text(encoding="ascii"))
        contract = json.loads((ROOT / result["contract"]["path"]).read_text(encoding="ascii"))
        raw_path = ROOT / result["capture"]["receipt"]["raw_path"]
        journal_path = ROOT / contract["outputs"]["journal_path"]
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
        assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == result["capture"]["receipt"]["response_sha256"]
        assert [row["phase"] for row in journal] == ["intent", "completed"]
        assert result["authority"]["credentials_used"] is False
        assert result["authority"]["protected_capture_touched"] is False


def test_registry_terminalizes_lighter_without_promotion() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_Lighter_BTC_ETH_SOL_USDC_vs_Binance_USDT_perpetual_funding_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 47

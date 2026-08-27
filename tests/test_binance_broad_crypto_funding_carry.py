from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import screen_binance_broad_crypto_funding_carry as screen


ROOT = Path(__file__).resolve().parents[1]


def _load_and_verify(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text())
    declared = artifact.pop("result_sha256")
    actual = hashlib.sha256(screen._canonical_json(artifact).encode("ascii")).hexdigest()
    assert actual == declared
    artifact["result_sha256"] = declared
    return artifact


def test_observations_reject_short_history_and_preserve_causal_lags() -> None:
    short = [
        {"symbol": "AAAUSDT", "fundingTime": index, "fundingRate": "0", "markPrice": "1"}
        for index in range(239)
    ]
    try:
        screen._observations(short, symbol="AAAUSDT")
    except ValueError as exc:
        assert "outside 240..1000" in str(exc)
    else:
        raise AssertionError("short history must fail")

    rows = [
        {
            "symbol": "AAAUSDT",
            "fundingTime": index * 28_800_000,
            "fundingRate": "0.0001",
            "markPrice": str(index + 100),
        }
        for index in range(240)
    ]
    observations = screen._observations(rows, symbol="AAAUSDT")
    assert len(observations) == 237
    assert observations[0]["funding_time_ms"] == 3 * 28_800_000
    assert Decimal(observations[0]["short_funding_received_bips"]) == Decimal("1")
    assert Decimal(observations[0]["lagged_mark_return"]) == Decimal(102) / Decimal(101) - 1


def test_liquidity_selection_requires_exact_base_identity_and_crypto_perpetual() -> None:
    contract = {
        "population_and_selection": {
            "minimum_each_leg_quote_volume_usdt_24h": "25000000",
            "maximum_selected_symbols": 2,
        }
    }
    spot_exchange = {
        "symbols": [
            {"symbol": "AAAUSDT", "baseAsset": "AAA", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
            {"symbol": "BBBUSDT", "baseAsset": "BBB", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
        ]
    }
    futures_exchange = {
        "symbols": [
            {"symbol": "AAAUSDT", "baseAsset": "AAA", "quoteAsset": "USDT", "marginAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BBBUSDT", "baseAsset": "BBB", "quoteAsset": "USDT", "marginAsset": "USDT", "status": "TRADING", "contractType": "TRADIFI_PERPETUAL", "underlyingType": "TRADIFI"},
        ]
    }
    tickers = [
        {"symbol": "AAAUSDT", "quoteVolume": "30000000"},
        {"symbol": "BBBUSDT", "quoteVolume": "90000000"},
    ]
    books = [
        {"symbol": "AAAUSDT", "bidPrice": "10", "askPrice": "10.1"},
        {"symbol": "BBBUSDT", "bidPrice": "20", "askPrice": "20.1"},
    ]
    selected, eligible_count = screen._select_universe(
        spot_exchange=spot_exchange,
        futures_exchange=futures_exchange,
        spot_tickers=tickers,
        futures_tickers=tickers,
        spot_books=books,
        futures_books=books,
        contract=contract,
    )
    assert eligible_count == 1
    assert [row["future_symbol"] for row in selected] == ["AAAUSDT"]


def test_role_metrics_rejects_positive_funding_that_does_not_clear_capital_hurdle() -> None:
    rows = [
        {
            "funding_time_ms": index * 28_800_000,
            "short_funding_received_bips": "0.1",
            "lagged_mark_return": str((index % 7 - 3) / 1000),
            "previous_lagged_mark_return": str(((index - 1) % 7 - 3) / 1000),
        }
        for index in range(180)
    ]
    contract = {
        "economic_and_stability_gates": {
            "round_trip_execution_stress_bips": "32",
            "annual_opportunity_hurdle_bips_per_capital_leg": "1000",
            "gross_capital_legs": 2,
            "minimum_observations_per_role": 100,
            "minimum_observations_per_slice": 5,
            "moving_block_bootstrap_repetitions": 100,
            "moving_block_length_settlements": 4,
            "maximum_net_drawdown_bips": "250",
            "maximum_positive_week_concentration": "0.25",
        }
    }
    thresholds = screen._regime_thresholds(rows)
    metrics = screen._role_metrics(
        rows,
        thresholds=thresholds,
        symbol="AAAUSDT",
        role="training",
        contract=contract,
        family_size=1,
    )
    assert metrics["passes"] is False
    assert "nonpositive_net_after_frozen_hurdles" in metrics["rejection_reasons"]


def test_broad_result_and_rfq_reopening_are_registered_without_edge_promotion() -> None:
    broad = _load_and_verify(
        ROOT
        / "docs/model-research/action-value/"
        "binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json"
    )
    rfq = _load_and_verify(
        ROOT
        / "docs/model-research/action-value/"
        "binance-options-rfq-fixed-payoff-execution-triage-v1-2026-08-27.json"
    )
    registry = _load_and_verify(
        ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
    )

    assert broad["scope"]["selected_symbol_count"] == 17
    assert broad["funding_only_gate_pass_count"] == 0
    assert broad["verdict"]["accepted_edge"] is False
    assert rfq["decision"]["vertical_execution_architecture_materially_reopened"] is True
    assert rfq["decision"]["box_execution_architecture_reopened"] is False
    assert rfq["decision"]["accepted_edge"] is False
    assert registry["accepted_edge_count"] == 18
    rfq_entry = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_options_Rfq_atomic_vertical_fixed_payoff_parity"
    )
    assert rfq_entry["priority_rank"] == 37
    assert any(
        row["canonical_result_sha256"] == broad["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )

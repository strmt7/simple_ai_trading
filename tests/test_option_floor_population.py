"""Synthetic side-specific price and run-level rejection checks, without sockets."""

import json

import pytest

from tools import screen_option_floor_population as screen


def inputs(side="C", ask="1"):
    symbol = f"BTC-260906-90-{side}"
    meta = {
        "symbol": symbol,
        "unit": "1",
        "underlying": "BTCUSDT",
        "status": "TRADING",
        "contractType": "CRYPTO_OPTIONS",
        "underlyingType": "CRYPTO",
        "quoteAsset": "USDT",
        "expiryDate": 9999999999999,
    }
    ticker = {symbol: {"symbol": symbol, "strikePrice": "90", "askPrice": ask}}
    future = {"BTCUSDT": {"symbol": "BTCUSDT", "bidPrice": "100", "askPrice": "101"}}
    return meta, ticker, future


@pytest.mark.parametrize("side,gross,entry", [("C", "9", "100"), ("P", "-12", "101")])
def test_acquisition_sides_and_fixed_stress(side, gross, entry):
    meta, tickers, futures = inputs(side)
    row = screen.rows_for([meta], tickers, futures)[0]
    assert row["gross_floor_per_base_usdt"] == gross
    assert row["perpetual_entry"] == entry
    assert row["passes_row_gate"] == (side == "C")


@pytest.mark.parametrize("ask", ["NaN", "Infinity", "-1"])
def test_invalid_prices_reject(ask):
    meta, tickers, futures = inputs(ask=ask)
    with pytest.raises(ValueError):
        screen.rows_for([meta], tickers, futures)


def test_zero_ask_is_not_free_profit():
    meta, tickers, futures = inputs(ask="0")
    row = screen.rows_for([meta], tickers, futures)[0]
    assert row["gross_floor_per_base_usdt"] is None
    assert row["passes_row_gate"] is False


@pytest.mark.parametrize("fault", ["unit", "strike", "missing", "status"])
def test_incomplete_or_incompatible_rows_reject(fault):
    meta, tickers, futures = inputs()
    if fault == "unit":
        meta["unit"] = "10"
    elif fault == "strike":
        tickers[meta["symbol"]]["strikePrice"] = "91"
    elif fault == "missing":
        tickers = {}
    else:
        meta["status"] = "HALT"
    with pytest.raises((ValueError, KeyError)):
        screen.rows_for([meta], tickers, futures)


@pytest.mark.parametrize(
    "skew,fail_first,expected_calls,expected_survivors",
    [(0, False, 2, 1), (10001, False, 2, 0), (0, True, 1, 0)],
)
def test_run_gate_and_early_source_failure(
    tmp_path, monkeypatch, skew, fail_first, expected_calls, expected_survivors
):
    meta, tickers, futures = inputs()
    population = {"distinct_symbols": [meta["symbol"]], "distinct_metadata": [meta]}
    population["result_sha256"] = screen._canonical_hash(population, "result_sha256")
    (tmp_path / "population").write_text(json.dumps(population))
    plan = {
        "implementation_sha256": {},
        "fixed_stress_bps": "33.5",
        "maximum_start_skew_ms": 10000,
        "population_path": "population",
        "population_sha256": population["result_sha256"],
        "output_path": "result",
        "source_results": {
            "option_tickers": {"contract_path": "option"},
            "futures_books": {"contract_path": "future"},
        },
    }
    plan["contract_sha256"] = screen._canonical_hash(plan, "contract_sha256")
    (tmp_path / "plan").write_text(json.dumps(plan))
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    calls = []

    def capture(path, preflight=False):
        if preflight:
            return None
        calls.append(path.name)
        return {"source_gate": {"passed": not fail_first}, "result_sha256": "a" * 64}

    monkeypatch.setattr(screen, "capture", capture)

    def load(binding):
        option = binding["contract_path"] == "option"
        receipt = {
            "requested_at_ms": 1 if option else 1 + skew,
            "completed_at_ms": 2 + skew,
        }
        values = tickers if option else futures
        return {"capture": {"receipt": receipt}}, json.dumps(
            list(values.values())
        ).encode()

    monkeypatch.setattr(screen, "_load_source_result", load)
    result = screen.run(tmp_path / "plan")
    assert len(calls) == expected_calls
    assert len(result["survivors"]) == expected_survivors
    assert result["accepted_edge"] is False
    assert bool(result["failure_type"]) == fail_first

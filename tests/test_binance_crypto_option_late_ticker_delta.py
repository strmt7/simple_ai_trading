from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_binance_crypto_option_late_ticker_delta_retained import (
    _baseline_symbols,
    _screen_rows,
    _ticker_rows,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION_VALUE / "binance-crypto-option-late-ticker-delta-contract-v1-2026-09-01.json"
)
RESULT = (
    ACTION_VALUE / "binance-crypto-option-late-ticker-delta-result-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_retained_delta_helpers_select_exact_scoped_population() -> None:
    baseline = {
        "optionSymbols": [
            {
                "symbol": "BTC-261225-90000-C",
                "status": "TRADING",
                "contractType": "CRYPTO_OPTIONS",
                "underlyingType": "CRYPTO",
                "underlying": "BTCUSDT",
                "quoteAsset": "USDT",
                "unit": "1",
            },
            {
                "symbol": "BNB-261225-900-C",
                "status": "TRADING",
                "contractType": "CRYPTO_OPTIONS",
                "underlyingType": "CRYPTO",
                "underlying": "BNBUSDT",
                "quoteAsset": "USDT",
                "unit": "1",
            },
        ]
    }
    tickers = _ticker_rows(
        [
            {"symbol": "BTC-261225-90000-C"},
            {"symbol": "BTC-261225-94000-P"},
            {"symbol": "BNB-261225-900-C"},
        ]
    )

    assert _baseline_symbols(baseline) == ["BTC-261225-90000-C"]
    assert sorted(set(tickers) - set(_baseline_symbols(baseline))) == [
        "BTC-261225-94000-P"
    ]


def test_screen_rows_applies_side_specific_perpetual_entry_and_fixed_gate() -> None:
    rows = _screen_rows(
        symbols=["BTC-261225-94000-C", "BTC-261225-94000-P"],
        tickers={
            "BTC-261225-94000-C": {
                "symbol": "BTC-261225-94000-C",
                "strikePrice": "94000",
                "askPrice": "1000",
            },
            "BTC-261225-94000-P": {
                "symbol": "BTC-261225-94000-P",
                "strikePrice": "94000",
                "askPrice": "1000",
            },
        },
        futures_books={
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "bidPrice": "95000",
                "askPrice": "95010",
            }
        },
        fixed_bps=Decimal("33.5"),
    )

    assert rows[0]["perpetual_entry_side"] == "bid"
    assert rows[0]["gross_terminal_floor_per_underlying_unit_USDT"] == "0"
    assert rows[0]["passes_fixed_rejection_gate"] is False
    assert rows[1]["perpetual_entry_side"] == "ask"
    assert rows[1]["gross_terminal_floor_per_underlying_unit_USDT"] == "-2010"
    assert rows[1]["passes_fixed_rejection_gate"] is False


def test_screen_rows_rejects_symbol_ticker_strike_disagreement() -> None:
    with pytest.raises(ValueError, match="ticker strike disagrees"):
        _screen_rows(
            symbols=["BTC-261225-94000-C"],
            tickers={
                "BTC-261225-94000-C": {
                    "symbol": "BTC-261225-94000-C",
                    "strikePrice": "95000",
                    "askPrice": "1",
                }
            },
            futures_books={
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "bidPrice": "95000",
                    "askPrice": "95010",
                }
            },
            fixed_bps=Decimal("33.5"),
        )


def test_frozen_contract_and_zero_network_result_are_hash_bound() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)

    assert contract["status"] == "frozen_before_zero_network_late_ticker_delta"
    assert contract["authority"]["new_public_requests"] == 0
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "b425fd356f5033e24313c5dbb166d381c045a4e794107a791acd49ed61992ab2"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert (
        result["contract"]["sha256"]
        == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    )
    assert result["authority"] == {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "new_public_requests": 0,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }


def test_exact_late_ticker_delta_stops_before_any_new_request() -> None:
    result = _load(RESULT)

    assert result["population"] == {
        "after_fixed_stress_positive_count": 0,
        "baseline_eligible_symbol_count": 1576,
        "gross_positive_count": 0,
        "later_scoped_ticker_symbol_count": 1578,
        "new_symbol_count": 2,
        "new_symbols": ["BTC-261225-94000-C", "BTC-261225-94000-P"],
        "positive_entry_side_count": 1,
    }
    call, put = result["all_rows"]
    assert call["gross_terminal_floor_per_underlying_unit_USDT"] == "-17970.60"
    assert call["passes_fixed_rejection_gate"] is False
    assert put["option_ask_USDT"] == "0"
    assert put["positive_entry_sides"] is False
    assert result["fixed_stress_survivors"] == []
    assert result["adjudication"]["new_public_requests"] == 0
    assert result["adjudication"]["option_depth_requests"] == 0


def test_rank_47_and_terminal_registry_record_exact_late_delta() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_crypto_option_opposite_USDT_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["priority_rank"] == 47
    assert hypothesis["canonical_artifacts"][-1] == {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": _load(RESULT)["result_sha256"],
    }
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_BTC_261225_94000_call_put_late_ticker_delta_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == _load(RESULT)["result_sha256"]

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.screen_polymarket_binance_tradfi_perps_funding_prefilter import (
    EMPTY_BODY_SHA256,
    RawJournal,
    analyze_snapshot,
)


def _instrument(
    instrument_id: int,
    category: str,
    base_asset: str,
) -> dict[str, object]:
    return {
        "instrument_id": instrument_id,
        "category": category,
        "symbol": f"{base_asset}-USD",
        "base_asset": base_asset,
        "quote_asset": "pUSD",
        "funding_interval": "1h",
    }


def _ticker(
    instrument_id: int,
    funding_rate: str,
    timestamp: int,
) -> dict[str, object]:
    return {
        "instrument_id": instrument_id,
        "funding_rate": funding_rate,
        "timestamp": timestamp,
    }


def _binance_symbol(
    symbol: str,
    *,
    status: str = "TRADING",
    contract_type: str = "TRADIFI_PERPETUAL",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "contractType": contract_type,
    }


def _premium(
    symbol: str,
    funding_rate: str,
    timestamp: int,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "lastFundingRate": funding_rate,
        "nextFundingTime": timestamp + 3_600_000,
        "time": timestamp,
    }


def test_prefilter_joins_only_exact_current_tradfi_matches_and_ranks_spread() -> None:
    timestamp = 1_788_030_000_000
    result = analyze_snapshot(
        [
            _instrument(15, "equity", "AAPL"),
            _instrument(16, "equity", "MSFT"),
            _instrument(6, "crypto", "BTC"),
        ],
        [
            _ticker(15, "0.00020", timestamp),
            _ticker(16, "0.00001", timestamp),
            _ticker(6, "0.00010", timestamp),
        ],
        {
            "symbols": [
                _binance_symbol("AAPLUSDT"),
                _binance_symbol("MSFTUSDT"),
                _binance_symbol("BTCUSDT", contract_type="PERPETUAL"),
            ]
        },
        [
            _premium("AAPLUSDT", "0.00010", timestamp),
            _premium("MSFTUSDT", "0.00000", timestamp),
            _premium("BTCUSDT", "0.00010", timestamp),
        ],
        prefilter_threshold_bips_per_8h=Decimal("2"),
        maximum_followup_symbols=3,
    )

    assert result["exact_current_match_count"] == 2
    assert result["passing_count"] == 1
    assert result["selected_for_separate_history_contract"] == ["AAPL"]
    assert result["maximum_funding_spread_bips_per_8h"] == "15.00000"
    assert [row["base_asset"] for row in result["ranked_rows"]] == [
        "AAPL",
        "MSFT",
    ]
    assert result["ranked_rows"][0]["current_orientation"] == (
        "short_polymarket_long_binance"
    )
    assert result["ranked_rows"][1]["prefilter_pass"] is False


def test_prefilter_excludes_nontrading_and_nontradfi_binance_contracts() -> None:
    timestamp = 1_788_030_000_000
    result = analyze_snapshot(
        [
            _instrument(15, "equity", "AAPL"),
            _instrument(16, "equity", "MSFT"),
        ],
        [
            _ticker(15, "0.00020", timestamp),
            _ticker(16, "0.00020", timestamp),
        ],
        {
            "symbols": [
                _binance_symbol("AAPLUSDT", status="PENDING_TRADING"),
                _binance_symbol("MSFTUSDT", contract_type="PERPETUAL"),
            ]
        },
        [
            _premium("AAPLUSDT", "0", timestamp),
            _premium("MSFTUSDT", "0", timestamp),
        ],
        prefilter_threshold_bips_per_8h=Decimal("1"),
        maximum_followup_symbols=3,
    )

    assert result["exact_current_match_count"] == 0
    assert result["passing_count"] == 0
    assert result["selected_for_separate_history_contract"] == []


def test_prefilter_fails_closed_on_non_hourly_tradfi_instrument() -> None:
    instrument = _instrument(15, "equity", "AAPL")
    instrument["funding_interval"] = "8h"
    with pytest.raises(ValueError, match="funding interval is not 1h"):
        analyze_snapshot(
            [instrument],
            [],
            {"symbols": []},
            [],
            prefilter_threshold_bips_per_8h=Decimal("1"),
            maximum_followup_symbols=3,
        )


def test_journal_persists_exact_request_before_access(tmp_path: Path) -> None:
    request = {
        "body_sha256": EMPTY_BODY_SHA256,
        "label": "public-snapshot",
        "method": "GET",
        "url": "https://example.test/public",
    }
    journal = RawJournal(
        tmp_path,
        contract_hash="0" * 64,
        request_plan=[request],
    )

    payload = json.loads(journal.path.read_text(encoding="ascii"))
    assert payload["requests"] == [{**request, "status": "planned"}]
    assert hashlib.sha256(b"").hexdigest() == request["body_sha256"]

    before_ms, row = journal.start(
        label="public-snapshot",
        url="https://example.test/public",
    )
    payload = json.loads(journal.path.read_text(encoding="ascii"))
    assert payload["requests"][0]["status"] == "requesting"
    assert payload["requests"][0]["requested_before_ms"] == before_ms
    assert row["requested_before_ms"] == before_ms

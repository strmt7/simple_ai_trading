from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "screen_binance_bnb_fee_discount_hedge.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-contract-v1.json"
)
RESULT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-screen-v1-2026-08-25.json"
)
JOURNAL_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-journal-v1.json"
)
RAW_ROOT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "raw"
    / "binance-bnb-fee-discount-hedge-v1"
)
EXPECTED_CONTRACT_SHA256 = (
    "388c6cc72d079b0fcbcafae5bb04a8466bb4ba92ae38725b0139693d62af1401"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "ee21d67d2663177131a0b10f00a9afeb9063590b02d41b9cde6445f253a42b3c"
)
EXPECTED_RESULT_SHA256 = (
    "7b918a503f99f5fb1ff7d012a007f6aefa2586030f6d3f9233b0b08e2adb479f"
)
EXPECTED_JOURNAL_SHA256 = (
    "d0a085527b1b61e6b37e1bfdab3c37bb4a7da209365d6ec4e15761eadf67efb5"
)
EXPECTED_JOURNAL_FILE_SHA256 = (
    "e01a431790d0448094c8c493aaf89a3c6de7cd2920cf1c9965df00a2d4cd8479"
)
SPEC = importlib.util.spec_from_file_location(
    "screen_binance_bnb_fee_discount_hedge", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _Response:
    def __init__(
        self,
        payload: object | None,
        *,
        url: str,
        raw: bytes | None = None,
        status_code: int = 200,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.content = (
            raw
            if raw is not None
            else json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
                "ascii"
            )
        )


def _spot_exchange_info() -> dict[str, object]:
    return {
        "symbols": [
            {
                "symbol": "BNBUSDT",
                "status": "TRADING",
                "baseAsset": "BNB",
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "100000",
                        "stepSize": "0.001",
                    },
                    {"filterType": "NOTIONAL", "minNotional": "5"},
                ],
            }
        ]
    }


def _futures_exchange_info() -> dict[str, object]:
    return {
        "symbols": [
            {
                "symbol": "BNBUSDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "baseAsset": "BNB",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "filters": [
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.01",
                        "maxQty": "100000",
                        "stepSize": "0.01",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }


def _funding_history() -> list[dict[str, object]]:
    start_ms = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000)
    price = 850.0
    moves = (1.003, 0.997, 1.0005, 0.9995, 1.0)
    rates = ("0.00010000", "0.00005000", "-0.00002500", "0.00007500")
    rows = []
    for index in range(300):
        price *= moves[index % len(moves)]
        rows.append(
            {
                "symbol": "BNBUSDT",
                "fundingTime": start_ms + index * 8 * 60 * 60 * 1000,
                "fundingRate": rates[index % len(rates)],
                "markPrice": f"{price:.8f}",
            }
        )
    return rows


class _Session:
    def __init__(self, *, invalid_first_body: bool = False) -> None:
        self.invalid_first_body = invalid_first_body
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: int,
    ) -> _Response:
        parameters = dict(params)
        self.calls.append((url, parameters, timeout))
        if self.invalid_first_body and len(self.calls) == 1:
            return _Response(None, url=url, raw=b"not-json")
        if url == "https://api.binance.com/api/v3/exchangeInfo":
            payload: object = _spot_exchange_info()
        elif url == "https://api.binance.com/api/v3/ticker/bookTicker":
            payload = {
                "symbol": "BNBUSDT",
                "bidPrice": "849.90",
                "bidQty": "100",
                "askPrice": "850.00",
                "askQty": "100",
            }
        elif url == "https://fapi.binance.com/fapi/v1/exchangeInfo":
            payload = _futures_exchange_info()
        elif url == "https://fapi.binance.com/fapi/v1/ticker/bookTicker":
            payload = {
                "symbol": "BNBUSDT",
                "bidPrice": "850.10",
                "bidQty": "100",
                "askPrice": "850.20",
                "askQty": "100",
            }
        elif url == "https://fapi.binance.com/fapi/v1/premiumIndex":
            payload = {
                "symbol": "BNBUSDT",
                "markPrice": "850.15",
                "indexPrice": "850.00",
                "lastFundingRate": "0.00010000",
                "nextFundingTime": 2_000_000_000_000,
                "time": 1_999_999_000_000,
            }
        elif url == "https://fapi.binance.com/fapi/v1/fundingRate":
            payload = _funding_history()
        else:  # pragma: no cover - frozen sequence must make this unreachable
            raise AssertionError(f"unexpected request {url}")
        return _Response(payload, url=url)


def _embedded_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_contract_binds_exact_formatted_implementation_and_request_sequence() -> None:
    contract = json.loads(CONTRACT_PATH.read_bytes())

    assert contract["result_sha256"] == EXPECTED_CONTRACT_SHA256
    assert _embedded_hash(contract, "result_sha256") == EXPECTED_CONTRACT_SHA256
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == (
        EXPECTED_IMPLEMENTATION_SHA256
    )
    assert contract["source_binding"]["implementation_sha256"] == (
        EXPECTED_IMPLEMENTATION_SHA256
    )
    assert contract["frozen_public_request_plan"]["requests"] == (
        TOOL._request_contract_rows()
    )
    assert contract["authority"] == {
        "accepted_edge": False,
        "account_credentials_permitted": False,
        "live_trading_authority": False,
        "orders_permitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }


def test_mock_run_uses_exactly_six_gets_and_retains_every_raw_body(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.json"
    raw_root = tmp_path / "raw"
    session = _Session()

    result = TOOL.run(
        session=session,
        journal_path=journal_path,
        raw_root=raw_root,
    )

    assert len(session.calls) == 6
    assert all(timeout == 30 for _url, _params, timeout in session.calls)
    assert [(url, parameters) for url, parameters, _timeout in session.calls] == [
        (row["url"], row["parameters"]) for row in TOOL._request_contract_rows()
    ]
    journal_bytes = journal_path.read_bytes()
    journal = json.loads(journal_bytes)
    assert journal["status"] == "data_complete"
    assert journal["completed_request_count"] == 6
    assert journal["next_request"] is None
    assert journal["journal_sha256"] == _embedded_hash(journal, "journal_sha256")
    assert len(journal["responses"]) == 6
    for response in journal["responses"]:
        receipt = response["receipt"]
        raw_path = raw_root / response["request"]["raw_filename"]
        assert raw_path.exists()
        assert (
            hashlib.sha256(raw_path.read_bytes()).hexdigest()
            == (receipt["payload_sha256"])
        )
        assert receipt["payload_bytes"] == len(raw_path.read_bytes())
        assert receipt["status_code"] == 200

    assert result["request_count"] == 6
    assert result["result_sha256"] == _embedded_hash(result, "result_sha256")
    assert result["verdict"]["accepted_edge"] is False
    assert result["verdict"]["profitability_claim"] is False
    assert result["verdict"]["credentials_used"] is False
    assert result["verdict"]["signed_requests_made"] == 0
    assert result["verdict"]["orders_placed"] is False
    assert result["mechanism"]["standalone_profit_strategy"] is False


def test_published_result_journal_and_raw_bodies_reconstruct() -> None:
    result = json.loads(RESULT_PATH.read_bytes())
    journal_bytes = JOURNAL_PATH.read_bytes()
    journal = json.loads(journal_bytes)

    assert result["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(result, "result_sha256") == EXPECTED_RESULT_SHA256
    assert journal["journal_sha256"] == EXPECTED_JOURNAL_SHA256
    assert _embedded_hash(journal, "journal_sha256") == EXPECTED_JOURNAL_SHA256
    assert hashlib.sha256(journal_bytes).hexdigest() == EXPECTED_JOURNAL_FILE_SHA256
    assert journal["status"] == "data_complete"
    assert journal["completed_request_count"] == 6
    assert len(journal["responses"]) == 6
    assert result["source_binding"]["journal_file_sha256"] == (
        EXPECTED_JOURNAL_FILE_SHA256
    )
    assert result["source_binding"]["journal_sha256"] == EXPECTED_JOURNAL_SHA256
    assert result["source_binding"]["raw_response_count"] == 6
    for response in journal["responses"]:
        receipt = response["receipt"]
        raw_path = RAW_ROOT / response["request"]["raw_filename"]
        raw = raw_path.read_bytes()
        assert len(raw) == receipt["payload_bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt["payload_sha256"]
        assert receipt["status_code"] == 200

    assert result["failure_reasons"] == ["insufficient_complete_inner_months"]
    assert result["funding_evaluation"]["row_count"] == 500
    assert result["funding_evaluation"]["complete_inner_month_count"] == 4
    assert result["verdict"]["qualified_public_prequalification"] is False
    assert result["verdict"]["accepted_edge"] is False
    assert result["verdict"]["profitability_claim"] is False


def test_invalid_json_is_persisted_and_terminal_before_validation(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.json"
    raw_root = tmp_path / "raw"
    session = _Session(invalid_first_body=True)

    with pytest.raises(ValueError, match="raw body is not valid JSON"):
        TOOL.run(
            session=session,
            journal_path=journal_path,
            raw_root=raw_root,
        )

    assert len(session.calls) == 1
    raw_path = raw_root / "01-spot-exchange-info.json"
    assert raw_path.read_bytes() == b"not-json"
    journal = json.loads(journal_path.read_bytes())
    assert journal["status"] == "terminal_failure"
    assert journal["completed_request_count"] == 1
    assert (
        journal["responses"][0]["receipt"]["payload_sha256"]
        == hashlib.sha256(b"not-json").hexdigest()
    )
    assert journal["journal_sha256"] == _embedded_hash(journal, "journal_sha256")

    with pytest.raises(RuntimeError, match="rerun is prohibited"):
        TOOL.run(
            session=_Session(),
            journal_path=journal_path,
            raw_root=raw_root,
        )

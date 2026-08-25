from __future__ import annotations

import importlib.util
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_binance_quarterly_carry",
    ROOT / "tools" / "screen_binance_quarterly_carry.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
SNAPSHOT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-carry-snapshot-v1-2026-08-25.json"
)
CALENDAR_ADJUDICATION = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-calendar-spread-mechanism-adjudication-v1.json"
)
CALENDAR_ADJUDICATION_SHA256 = (
    "4add8a3aea01ebb13e743a85793681b8ca7a8884035daf5cb371f3f2b09900b0"
)


class _Response:
    def __init__(
        self,
        payload: object,
        *,
        url: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.url = url
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.content = json.dumps(payload, sort_keys=True).encode("ascii")

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, *, rate_limited: bool = False) -> None:
        self.calls: list[tuple[str, Mapping[str, object] | None]] = []
        self.rate_limited = rate_limited

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        timeout: int,
    ) -> _Response:
        assert timeout == 30
        self.calls.append((url, params))
        if self.rate_limited:
            return _Response(
                {}, url=url, status_code=429, headers={"Retry-After": "17"}
            )
        if url.endswith("exchangeInfo"):
            symbols = []
            for asset in ("BTC", "ETH"):
                for contract_type, suffix, delivery in (
                    ("CURRENT_QUARTER", "260925", 2_000_100_000_000),
                    ("NEXT_QUARTER", "261225", 2_007_900_000_000),
                ):
                    symbols.append(
                        {
                            "symbol": f"{asset}USDT_{suffix}",
                            "pair": f"{asset}USDT",
                            "baseAsset": asset,
                            "quoteAsset": "USDT",
                            "marginAsset": "USDT",
                            "status": "TRADING",
                            "contractType": contract_type,
                            "deliveryDate": delivery,
                            "filters": [
                                {
                                    "filterType": "LOT_SIZE",
                                    "minQty": "0.001",
                                    "stepSize": "0.001",
                                }
                            ],
                        }
                    )
            return _Response({"symbols": symbols}, url=url)
        if "api/v3/depth" in url:
            return _Response(
                {
                    "lastUpdateId": 10,
                    "bids": [["99", "10"]],
                    "asks": [["100", "10"]],
                },
                url=url,
            )
        return _Response(
            {
                "lastUpdateId": 11,
                "E": 2_000_000_000_000,
                "T": 2_000_000_000_000,
                "bids": [["101", "10"]],
                "asks": [["102", "10"]],
            },
            url=url,
        )


def test_run_fetches_each_book_once_and_never_accepts_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 2_000_000_000_000_000_000)
    session = _Session()

    report = TOOL.run(session=session)

    assert len(session.calls) == 9
    assert sum("exchangeInfo" in url for url, _params in session.calls) == 1
    assert sum("api/v3/depth" in url for url, _params in session.calls) == 4
    assert sum("fapi/v1/depth" in url for url, _params in session.calls) == 4
    assert report["verdict"] == {
        "status": "unqualified_positive_basis_requires_exact_account_and_carry_costs",
        "contract_count": 4,
        "quantity_screen_count": 12,
        "gross_positive_count": 12,
        "after_hurdle_positive_count": 12,
        "fresh_contract_with_after_hurdle_positive_count": 4,
        "accepted_edge": False,
        "trading_authority": False,
    }
    assert all(screen["freshness_passed"] for screen in report["screens"])
    assert len(report["result_sha256"]) == 64


def test_source_snapshot_hash_and_depth_results_reconstruct() -> None:
    report = json.loads(SNAPSHOT.read_text(encoding="ascii"))
    expected_hash = report.pop("result_sha256")

    assert (
        hashlib.sha256(TOOL._canonical_json(report).encode("ascii")).hexdigest()
        == expected_hash
    )
    implementation = report["source_contract"]["implementation"]
    assert (
        implementation["tool_sha256"]
        == hashlib.sha256(
            (ROOT / "tools" / "screen_binance_quarterly_carry.py").read_bytes()
        ).hexdigest()
    )
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(
            (ROOT / "src" / "simple_ai_trading" / "quarterly_carry.py").read_bytes()
        ).hexdigest()
    )

    sources = {
        source["symbol"]: source for source in report["source_contract"]["book_sources"]
    }
    for screen in report["screens"]:
        source = sources[screen["symbol"]]
        spot_asks = TOOL._levels(source["spot_book"], side="asks", descending=False)
        future_bids = TOOL._levels(source["future_book"], side="bids", descending=True)
        for recorded in screen["quantity_results"]:
            reconstructed = TOOL.screen_quarterly_cash_and_carry(
                spot_asks=spot_asks,
                future_bids=future_bids,
                quantity=Decimal(recorded["quantity"]),
                capture_time_ms=source["future_request"]["received_after_ms"],
                delivery_time_ms=screen["delivery_time_ms"],
                all_in_cost_hurdle_bips=Decimal("35"),
            )
            assert reconstructed is not None
            assert TOOL._result_payload(reconstructed) == recorded


def test_calendar_spread_mechanism_rejection_is_hash_bound_and_terminal() -> None:
    report = json.loads(CALENDAR_ADJUDICATION.read_text(encoding="ascii"))
    expected_hash = report.pop("result_sha256")

    assert expected_hash == CALENDAR_ADJUDICATION_SHA256
    assert (
        hashlib.sha256(TOOL._canonical_json(report).encode("ascii")).hexdigest()
        == expected_hash
    )
    payoff = report["payoff_identity"]
    assert payoff["fixed_payoff_proved"] is False
    assert payoff["near_expiry_exit"]["equivalent_identity"] == (
        "(F2_t0-F1_t0)-(F2_T1-S_T1)"
    )
    assert "F2_T1" in payoff["variables"]
    assert report["request_decision"]["new_public_requests"] == 0
    assert report["request_decision"]["historical_backtest_justified"] is False
    assert report["adjudication"]["accepted_edge"] is False
    assert report["adjudication"]["terminal_for_claimed_mechanism"] is True


def test_rate_limit_stops_without_retry() -> None:
    session = _Session(rate_limited=True)
    with pytest.raises(RuntimeError, match="stopped without retry; Retry-After=17"):
        TOOL.run(session=session)
    assert len(session.calls) == 1


def test_contract_selection_requires_complete_quarterly_set() -> None:
    with pytest.raises(ValueError, match="exactly current and next"):
        TOOL._contracts({"symbols": []})


@pytest.mark.parametrize(
    ("payload", "side", "descending", "message"),
    [
        ({"asks": []}, "asks", False, "empty or incorrectly sorted"),
        (
            {"bids": [["100", "1"], ["101", "1"]]},
            "bids",
            True,
            "empty or incorrectly sorted",
        ),
        ({"asks": [["100"]]}, "asks", False, "fewer than two"),
    ],
)
def test_depth_validation_fails_closed(
    payload: object, side: str, descending: bool, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TOOL._levels(payload, side=side, descending=descending)

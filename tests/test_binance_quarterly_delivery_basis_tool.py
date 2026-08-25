from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
from pathlib import Path
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_binance_quarterly_delivery_basis",
    ROOT / "tools" / "audit_binance_quarterly_delivery_basis.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
FIRST_DELIVERY_MS = 1_700_000_040_000
DELIVERY_STEP_MS = 7_776_000_000


def _deliveries(pair: str) -> list[dict[str, object]]:
    base = Decimal("100") if pair == "BTCUSDT" else Decimal("2000")
    return [
        {
            "deliveryTime": FIRST_DELIVERY_MS + index * DELIVERY_STEP_MS,
            "deliveryPrice": str(base + index),
        }
        for index in range(8)
    ]


def _klines(pair: str, start_ms: int) -> list[list[object]]:
    delivery_index = (start_ms - FIRST_DELIVERY_MS) // DELIVERY_STEP_MS
    base = (Decimal("100") if pair == "BTCUSDT" else Decimal("2000")) + Decimal(
        delivery_index
    )
    return [
        [
            start_ms + minute * 60_000,
            str(base),
            str(base * Decimal("1.001")),
            str(base * Decimal("0.999")),
            str(base * Decimal("1.0005")),
            "1",
            start_ms + (minute + 1) * 60_000 - 1,
        ]
        for minute in range(5)
    ]


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
        self.content = TOOL._canonical_json(payload).encode("ascii")

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, *, rate_limited: bool = False) -> None:
        self.rate_limited = rate_limited
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: int,
    ) -> _Response:
        assert timeout == 30
        self.calls.append((url, params))
        if self.rate_limited:
            return _Response(
                {},
                url=url,
                status_code=429,
                headers={"Retry-After": "19"},
            )
        if url == TOOL.FUTURES_URL:
            payload: object = _deliveries(str(params["pair"]))
        else:
            payload = _klines(str(params["symbol"]), int(params["startTime"]))
        return _Response(payload, url=url)


def test_run_uses_exact_eighteen_requests_and_never_accepts_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_640_900_000_000_000)
    session = _Session()
    ledger: list[dict[str, object]] = []

    report = TOOL.run(session=session, ledger=ledger)

    assert len(session.calls) == 18
    assert len(ledger) == 18
    assert [row["pair"] for row in report["pair_results"]] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert all(row["delivery_count"] == 8 for row in report["pair_results"])
    assert all(
        Decimal(row["minimum_low_mismatch_bips"]) == Decimal("-10.000")
        for row in report["pair_results"]
    )
    assert len(report["current_next_quarter_stress"]) == 6
    assert report["verdict"]["all_next_quarter_sizes_stress_positive"] is True
    assert report["verdict"]["accepted_edge"] is False
    assert report["safety"]["orders_placed"] is False


def test_rate_limit_is_retained_in_terminal_failure_receipt() -> None:
    ledger: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="without retry; Retry-After=19") as error:
        TOOL.run(session=_Session(rate_limited=True), ledger=ledger)

    receipt = TOOL._failure_payload(error=error.value, ledger=ledger)
    expected_hash = receipt.pop("result_sha256")

    assert len(ledger) == 1
    assert ledger[0]["status_code"] == 429
    assert ledger[0]["retry_after"] == "19"
    assert receipt["status"] == "terminal_failure_without_retry"
    assert receipt["authority"]["accepted_edge"] is False
    assert (
        hashlib.sha256(TOOL._canonical_json(receipt).encode("ascii")).hexdigest()
        == expected_hash
    )


def test_delivery_selection_rejects_insufficient_and_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="lacks the frozen"):
        TOOL._deliveries(
            _deliveries("BTCUSDT")[:-1],
            cutoff_ms=1_787_640_808_532,
            count=8,
        )

    conflict = _deliveries("BTCUSDT")
    conflict.append({**conflict[0], "deliveryPrice": "999"})
    with pytest.raises(ValueError, match="conflicting"):
        TOOL._deliveries(
            conflict,
            cutoff_ms=1_787_640_808_532,
            count=8,
        )


def test_canonical_request_payload_hash_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_640_900_000_000_000)
    ledger: list[dict[str, object]] = []
    TOOL._get(
        _Session(),
        TOOL.FUTURES_URL,
        params={"pair": "BTCUSDT"},
        ledger=ledger,
    )

    entry = ledger[0]
    assert (
        hashlib.sha256(
            TOOL._canonical_json(entry["decoded_payload"]).encode("ascii")
        ).hexdigest()
        == entry["canonical_payload_sha256"]
    )

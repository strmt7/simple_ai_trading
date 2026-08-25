from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_binance_quarterly_pre_delivery_unwind",
    ROOT / "tools" / "audit_binance_quarterly_pre_delivery_unwind.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _bars(*, start_ms: int, count: int, future: bool) -> list[list[object]]:
    basis = Decimal("0.1") if future else Decimal("0")
    rows: list[list[object]] = []
    for index in range(count):
        mid = Decimal("100") + Decimal(index) / 100 + basis
        rows.append(
            [
                start_ms + index * 60_000,
                str(mid),
                str(mid + Decimal("0.05")),
                str(mid - Decimal("0.05")),
                str(mid),
                "1",
                start_ms + (index + 1) * 60_000 - 1,
            ]
        )
    return rows


class _Response:
    def __init__(
        self,
        payload: object,
        *,
        url: str,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
        json_error: bool = False,
    ) -> None:
        self._payload = payload
        self._json_error = json_error
        self.url = url
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.content = (
            TOOL._canonical_json(payload).encode("ascii")
            if content is None
            else content
        )

    def json(self) -> object:
        if self._json_error:
            raise requests.JSONDecodeError("invalid", "x", 0)
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(
        self,
        *,
        rate_limited: bool = False,
        future_basis: Decimal = Decimal("0.1"),
    ) -> None:
        self.rate_limited = rate_limited
        self.future_basis = future_basis
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
                headers={"Retry-After": "23"},
            )
        start_ms = int(params["startTime"])
        payload = _bars(
            start_ms=start_ms,
            count=60 if url == TOOL.FUTURES_KLINES_URL else 70,
            future=url == TOOL.FUTURES_KLINES_URL,
        )
        if url == TOOL.FUTURES_KLINES_URL and self.future_basis != Decimal("0.1"):
            adjustment = self.future_basis - Decimal("0.1")
            for row in payload:
                for index in (1, 2, 3, 4):
                    row[index] = str(Decimal(str(row[index])) + adjustment)
        return _Response(payload, url=url)


def test_run_uses_exact_frozen_requests_and_never_accepts_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_641_500_000_000_000)
    session = _Session()
    ledger: list[dict[str, object]] = []

    report = TOOL.run(session=session, ledger=ledger)

    assert len(session.calls) == 32
    assert len(ledger) == 32
    assert len(report["observations"]) == 16
    assert [row["contract_count"] for row in report["pair_results"]] == [8, 8]
    assert len(report["current_next_quarter_stress"]) == 6
    assert (
        report["verdict"]["all_current_next_quarter_sizes_primary_stress_positive"]
        is True
    )
    assert report["verdict"]["accepted_edge"] is False
    assert report["safety"]["orders_placed"] is False
    assert all(
        params["limit"] == 70 and params["interval"] == "1m"
        for _, params in session.calls
    )
    assert [url for url, _ in session.calls[:2]] == [
        TOOL.FUTURES_KLINES_URL,
        TOOL.SPOT_KLINES_URL,
    ]


def test_rate_limit_retains_payload_and_retry_after_in_terminal_receipt() -> None:
    ledger: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="without retry; Retry-After=23") as error:
        TOOL.run(session=_Session(rate_limited=True), ledger=ledger)

    receipt = TOOL._failure_payload(error=error.value, ledger=ledger)
    expected_hash = receipt.pop("result_sha256")

    assert len(ledger) == 1
    assert ledger[0]["status_code"] == 429
    assert ledger[0]["retry_after"] == "23"
    assert receipt["status"] == "terminal_failure_without_retry"
    assert receipt["authority"]["accepted_edge"] is False
    assert (
        hashlib.sha256(TOOL._canonical_json(receipt).encode("ascii")).hexdigest()
        == expected_hash
    )


def test_reserved_receipt_is_hash_bound_and_non_authorizing() -> None:
    receipt = TOOL._reserved_payload()
    expected_hash = receipt.pop("result_sha256")

    assert receipt["status"] == "reserved_before_public_requests"
    assert receipt["source_contract"]["request_ledger"] == []
    assert receipt["authority"]["accepted_edge"] is False
    assert (
        hashlib.sha256(TOOL._canonical_json(receipt).encode("ascii")).hexdigest()
        == expected_hash
    )


def test_source_payload_hash_reconstructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TOOL.time, "time_ns", lambda: 1_787_641_500_000_000_000)
    ledger: list[dict[str, object]] = []
    TOOL._get(
        _Session(),
        TOOL.FUTURES_KLINES_URL,
        params={
            "symbol": "BTCUSDT_260626",
            "interval": "1m",
            "startTime": 1_782_457_200_000,
            "endTime": 1_782_461_399_999,
            "limit": 70,
        },
        ledger=ledger,
    )

    entry = ledger[0]
    assert (
        hashlib.sha256(
            TOOL._canonical_json(entry["decoded_payload"]).encode("ascii")
        ).hexdigest()
        == entry["canonical_payload_sha256"]
    )


@pytest.mark.parametrize(
    ("function", "value", "message"),
    [
        (TOOL._mapping, [], "value must be an object"),
        (TOOL._list, {}, "value must be a list"),
        (TOOL._decimal, True, "value must be a finite decimal"),
        (TOOL._decimal, object(), "value must be a finite decimal"),
        (TOOL._decimal, "NaN", "value must be a finite decimal"),
    ],
)
def test_source_value_validators_fail_closed(
    function: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(value, name="value")


def test_verified_json_rejects_raw_and_result_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    body = {"status": "frozen"}
    payload = {
        **body,
        "result_sha256": hashlib.sha256(
            TOOL._canonical_json(body).encode("ascii")
        ).hexdigest(),
    }
    path.write_text(TOOL._canonical_json(payload), encoding="ascii")

    assert TOOL._verified_json(path)["status"] == "frozen"
    with pytest.raises(ValueError, match="bytes are invalid"):
        TOOL._verified_json(path, expected_raw_hash="0" * 64)
    with pytest.raises(ValueError, match="hash is invalid"):
        TOOL._verified_json(path, expected_result_hash="0" * 64)

    payload["result_sha256"] = "0" * 64
    path.write_text(TOOL._canonical_json(payload), encoding="ascii")
    with pytest.raises(ValueError, match="hash is invalid"):
        TOOL._verified_json(path)


def test_contract_rejects_non_frozen_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TOOL, "_verified_json", lambda path: {"status": "draft"})

    with pytest.raises(ValueError, match="contract is not frozen"):
        TOOL._contract()


class _SingleResponseSession:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: int,
    ) -> _Response:
        assert params == {"symbol": "BTCUSDT"}
        assert timeout == 30
        return self.response


def test_get_rejects_non_json_after_ledgering() -> None:
    ledger: list[dict[str, object]] = []
    response = _Response(
        None,
        url=TOOL.SPOT_KLINES_URL,
        content=b"not-json",
        json_error=True,
    )

    with pytest.raises(ValueError, match="did not return JSON"):
        TOOL._get(
            _SingleResponseSession(response),
            TOOL.SPOT_KLINES_URL,
            params={"symbol": "BTCUSDT"},
            ledger=ledger,
        )

    assert ledger[0]["decoded_payload"] is None
    assert ledger[0]["canonical_payload_sha256"] is None


def test_get_rejects_oversized_and_http_error_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL, "MAX_RESPONSE_BYTES", 1)
    oversized = _Response([], url=TOOL.SPOT_KLINES_URL, content=b"[]")
    with pytest.raises(ValueError, match="bounded size"):
        TOOL._get(
            _SingleResponseSession(oversized),
            TOOL.SPOT_KLINES_URL,
            params={"symbol": "BTCUSDT"},
            ledger=[],
        )

    failed = _Response({}, url=TOOL.SPOT_KLINES_URL, status_code=500)
    with pytest.raises(requests.HTTPError):
        TOOL._get(
            _SingleResponseSession(failed),
            TOOL.SPOT_KLINES_URL,
            params={"symbol": "BTCUSDT"},
            ledger=[],
        )


def test_rate_limit_without_retry_after_is_terminal() -> None:
    response = _Response({}, url=TOOL.FUTURES_KLINES_URL, status_code=429)

    with pytest.raises(RuntimeError, match="without retry$"):
        TOOL._get(
            _SingleResponseSession(response),
            TOOL.FUTURES_KLINES_URL,
            params={"symbol": "BTCUSDT"},
            ledger=[],
        )


def test_odd_median_and_primary_stress_rejection() -> None:
    assert TOOL._median([Decimal("3"), Decimal("1"), Decimal("2")]) == Decimal("2")

    report = TOOL.run(session=_Session(future_basis=Decimal("10")))

    assert report["verdict"]["status"] == "rejected_primary_pre_delivery_basis_stress"
    assert (
        report["verdict"]["all_current_next_quarter_sizes_primary_stress_positive"]
        is False
    )


def test_main_refuses_overwrite_and_writes_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(existing)])
    assert TOOL.main() == 2
    assert existing.read_text(encoding="ascii") == "preserve"

    output = tmp_path / "success.json"
    success = {
        "verdict": {"accepted_edge": False},
        "result_sha256": "1" * 64,
    }
    monkeypatch.setattr(TOOL, "run", lambda *, ledger: success)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    assert TOOL.main() == 0
    assert json.loads(output.read_text(encoding="ascii")) == success


def test_main_replaces_reservation_with_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "failure.json"

    def fail(*, ledger: list[dict[str, object]]) -> dict[str, object]:
        ledger.append({"status_code": 500})
        raise ValueError("source failure")

    monkeypatch.setattr(TOOL, "run", fail)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])

    assert TOOL.main() == 1
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["status"] == "terminal_failure_without_retry"
    assert payload["error"] == {"type": "ValueError", "message": "source failure"}
    assert payload["source_contract"]["request_ledger"] == [{"status_code": 500}]

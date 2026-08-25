from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest
import requests

import simple_ai_trading.quarterly_carry_account_evidence as evidence_module
from simple_ai_trading.quarterly_carry_account_evidence import (
    FUTURES_SYMBOLS,
    SPOT_SYMBOLS,
    capture_quarterly_carry_account_evidence,
)
from tools import capture_binance_quarterly_carry_account_evidence as capture_tool


@dataclass(frozen=True)
class _Response:
    payload: object
    status_code: int = 200
    headers: Mapping[str, object] = field(default_factory=dict)

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload, separators=(",", ":")).encode("ascii")

    def iter_content(self, chunk_size: int = 4_096):
        body = self.content
        for offset in range(0, len(body), chunk_size):
            yield body[offset : offset + chunk_size]

    def close(self) -> None:
        return None


def _spot_payload(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "standardCommission": {
            "maker": "0.0008",
            "taker": "0.0010",
            "buyer": "0.0001",
            "seller": "0.0002",
        },
        "specialCommission": {
            "maker": "0",
            "taker": "0.0001",
            "buyer": "0",
            "seller": "0.0001",
        },
        "taxCommission": {
            "maker": "0.00001",
            "taker": "0.00002",
            "buyer": "0.00001",
            "seller": "0.00003",
        },
        "discount": {
            "enabledForAccount": True,
            "enabledForSymbol": True,
            "discountAsset": "BNB",
            "discount": "0.75",
        },
    }


def test_capture_is_complete_get_only_component_exact_and_secret_free() -> None:
    calls: list[dict[str, object]] = []
    journal: list[Mapping[str, object]] = []
    server_time = 1_800_000_000_000

    def request(
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, object],
        timeout: float,
        stream: bool,
        allow_redirects: bool,
    ) -> _Response:
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "params": dict(params),
                "timeout": timeout,
            }
        )
        assert method == "GET"
        assert stream is True
        assert allow_redirects is False
        if url.endswith("/time"):
            assert "X-MBX-APIKEY" not in headers
            assert params == {}
            return _Response(
                {"serverTime": server_time + len(calls)},
                headers={"X-MBX-USED-WEIGHT-1M": "1"},
            )
        assert headers["X-MBX-APIKEY"] == "temporary-key"
        assert params["recvWindow"] == 2_000
        assert isinstance(params["timestamp"], int)
        assert len(str(params["signature"])) == 64
        if url.endswith("/api/v3/account/commission"):
            return _Response(
                _spot_payload(str(params["symbol"])),
                headers={"X-MBX-USED-WEIGHT-1M": "20"},
            )
        if url.endswith("/fapi/v1/commissionRate"):
            return _Response(
                {
                    "symbol": params["symbol"],
                    "makerCommissionRate": "0.0002",
                    "takerCommissionRate": "0.0004",
                    "rpiCommissionRate": "0.00005",
                },
                headers={"X-MBX-USED-WEIGHT-1M": "40"},
            )
        assert url.endswith("/fapi/v1/accountConfig")
        assert "symbol" not in params
        return _Response(
            {
                "feeTier": 1,
                "canTrade": True,
                "canDeposit": True,
                "canWithdraw": False,
                "dualSidePosition": False,
                "updateTime": 0,
                "multiAssetsMargin": False,
                "tradeGroupId": -1,
            },
            headers={"X-MBX-USED-WEIGHT-1M": "45"},
        )

    capture = capture_quarterly_carry_account_evidence(
        api_key="temporary-key",
        api_secret="temporary-secret",
        timeout_seconds=5,
        request=request,
        journal=journal.append,
    )
    artifact = capture.as_dict()
    serialized = json.dumps(artifact, sort_keys=True)

    assert len(calls) == 14
    assert tuple(artifact["spot_commissions"]) == SPOT_SYMBOLS
    assert tuple(artifact["futures_commissions"]) == FUTURES_SYMBOLS
    assert artifact["futures_account_configuration"] == {
        "feeTier": 1,
        "canTrade": True,
        "dualSidePosition": False,
        "multiAssetsMargin": False,
        "tradeGroupId": -1,
        "balances_persisted": False,
        "positions_persisted": False,
    }
    spot = artifact["spot_commissions"]["BTCUSDT"]
    assert spot["standardCommission"]["buyer"] == "0.0001"
    assert spot["standardCommission"]["seller"] == "0.0002"
    assert spot["discount"]["applied_in_economics"] is False
    assert len(journal) == 28
    assert all(item["method"] == "GET" for item in journal)
    assert "temporary-key" not in serialized
    assert "temporary-secret" not in serialized
    assert "signature" not in serialized
    assert "temporary-key" not in json.dumps(journal)
    assert "signature" not in json.dumps(journal)
    assert artifact["authority"]["orders_submitted"] is False
    assert len(artifact["result_sha256"]) == 64
    response_events = [
        item
        for item in journal
        if item["phase"] == "response_persisted_before_validation"
    ]
    assert len(response_events) == 14
    assert all(item["response_exceeded_limit"] is False for item in response_events)


def test_capture_fails_without_retry_and_journals_terminal_request() -> None:
    calls = 0
    journal: list[Mapping[str, object]] = []

    def request(_method: str, url: str, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        if url.endswith("/time"):
            return _Response({"serverTime": 1_800_000_000_000})
        raise requests.ConnectionError("synthetic failure")

    with pytest.raises(RuntimeError, match="failed without retry"):
        capture_quarterly_carry_account_evidence(
            api_key="temporary-key",
            api_secret="temporary-secret",
            request=request,
            journal=journal.append,
        )

    assert calls == 2
    assert journal[-1]["phase"] == "terminal_failure"
    assert journal[-1]["path"] == "/api/v3/account/commission"
    assert "synthetic failure" not in json.dumps(journal)


def test_non_200_body_hash_is_journaled_before_terminal_validation() -> None:
    calls = 0
    journal: list[Mapping[str, object]] = []

    def request(_method: str, url: str, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        if url.endswith("/time"):
            return _Response({"serverTime": 1_800_000_000_000})
        return _Response({"code": -2015, "msg": "rejected"}, status_code=401)

    with pytest.raises(RuntimeError, match="response differs"):
        capture_quarterly_carry_account_evidence(
            api_key="temporary-key",
            api_secret="temporary-secret",
            request=request,
            journal=journal.append,
        )

    assert calls == 2
    assert journal[-2]["phase"] == "response_persisted_before_validation"
    assert journal[-2]["status_code"] == 401
    assert len(str(journal[-2]["captured_body_sha256"])) == 64
    assert journal[-1]["phase"] == "terminal_failure"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.update(symbol="ETHUSDT"), "symbol differs"),
        (
            lambda payload: payload["standardCommission"].update(taker="NaN"),
            "allowed rate range",
        ),
        (
            lambda payload: payload["discount"].update(enabledForAccount="yes"),
            "boolean",
        ),
    ],
)
def test_capture_rejects_malformed_spot_fee_components(mutation, match: str) -> None:
    calls = 0

    def request(_method: str, url: str, **kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        if url.endswith("/time"):
            return _Response({"serverTime": 1_800_000_000_000})
        payload = _spot_payload(str(kwargs["params"]["symbol"]))
        mutation(payload)
        return _Response(payload)

    with pytest.raises(ValueError, match=match):
        capture_quarterly_carry_account_evidence(
            api_key="temporary-key",
            api_secret="temporary-secret",
            request=request,
        )
    assert calls == 2


def test_frozen_contract_is_self_hashed_and_implementation_bound() -> None:
    contract = json.loads(capture_tool.CONTRACT.read_text(encoding="utf-8"))
    body = dict(contract)
    claimed = body.pop("result_sha256")
    assert claimed == capture_tool.canonical_sha256(body)
    implementation = contract["implementation"]
    for path_key, hash_key in (
        ("module_path", "module_sha256"),
        ("tool_path", "tool_sha256"),
    ):
        path = capture_tool.ROOT / implementation[path_key]
        normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        assert hashlib.sha256(normalized).hexdigest() == implementation[hash_key]
    assert contract["request_contract"]["method_allowlist"] == ["GET"]
    assert contract["request_contract"]["retry_count"] == 0
    assert contract["authority"]["live_trading_authority"] is False


def test_tool_missing_credential_preflight_makes_no_request_or_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(capture_tool.API_KEY_VARIABLE, raising=False)
    monkeypatch.delenv(capture_tool.API_SECRET_VARIABLE, raising=False)
    monkeypatch.setattr(capture_tool, "_require_clean_tracked_worktree", lambda: None)

    class _ForbiddenSession:
        def __enter__(self):
            raise AssertionError("network session must not be created")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(capture_tool.requests, "Session", _ForbiddenSession)
    output = tmp_path / "output.json"
    journal = tmp_path / "journal.json"
    with pytest.raises(RuntimeError, match="no request was attempted"):
        capture_tool.main(["--output", str(output), "--journal", str(journal)])
    assert not output.exists()
    assert not journal.exists()


def test_low_level_validators_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        evidence_module._mapping([], label="candidate")
    with pytest.raises(ValueError, match="must be a decimal rate"):
        evidence_module._decimal_rate({}, label="rate")
    with pytest.raises(ValueError, match="must be a decimal rate"):
        evidence_module._decimal_rate("not-a-number", label="rate")
    with pytest.raises(ValueError, match="must be an integer"):
        evidence_module._integer(True, label="count")
    with pytest.raises(ValueError, match="rate-limit response header differs"):
        evidence_module._rate_limit_headers({"X-MBX-USED-WEIGHT-1M": ""})
    with pytest.raises(ValueError, match="request evidence differs"):
        evidence_module.RequestEvidence(
            request_id="",
            venue="spot",
            path="/api/v3/time",
            symbol=None,
            request_started_wall_ns=1,
            received_wall_ns=1,
            request_started_monotonic_ns=1,
            received_monotonic_ns=1,
            payload_sha256="0" * 64,
            response_bytes=1,
            rate_limit_headers=(),
        ).as_dict()


def test_capture_result_rejects_incomplete_coverage() -> None:
    spot = tuple((symbol, {}) for symbol in SPOT_SYMBOLS)
    futures = tuple((symbol, {}) for symbol in FUTURES_SYMBOLS)
    with pytest.raises(ValueError, match="spot commission symbol coverage differs"):
        evidence_module.QuarterlyCarryAccountEvidence(
            spot_commissions=(),
            futures_commissions=(),
            futures_account_configuration={},
            requests=(),
        ).as_dict()
    with pytest.raises(ValueError, match="futures commission symbol coverage differs"):
        evidence_module.QuarterlyCarryAccountEvidence(
            spot_commissions=spot,
            futures_commissions=(),
            futures_account_configuration={},
            requests=(),
        ).as_dict()
    with pytest.raises(ValueError, match="request evidence coverage differs"):
        evidence_module.QuarterlyCarryAccountEvidence(
            spot_commissions=spot,
            futures_commissions=futures,
            futures_account_configuration={},
            requests=(),
        ).as_dict()


@pytest.mark.parametrize(
    "body,match",
    [
        (b'{"a":1,"a":2}', "duplicate JSON keys"),
        (b'{"a":NaN}', "non-finite JSON"),
        (b"{", "not strict JSON"),
    ],
)
def test_strict_json_rejects_ambiguous_or_malformed_body(
    body: bytes, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evidence_module._strict_json(body)


def test_normalizers_reject_missing_identity_fields() -> None:
    spot = _spot_payload("BTCUSDT")
    spot["discount"]["discountAsset"] = ""
    with pytest.raises(ValueError, match="discount asset differs"):
        evidence_module._normalize_spot_commission(spot, symbol="BTCUSDT")
    with pytest.raises(ValueError, match="futures commission response symbol differs"):
        evidence_module._normalize_futures_commission(
            {
                "symbol": "ETHUSDT_260925",
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0004",
                "rpiCommissionRate": "0.00005",
            },
            symbol="BTCUSDT_260925",
        )
    with pytest.raises(ValueError, match="capture arguments differ"):
        capture_quarterly_carry_account_evidence(
            api_key="contains whitespace",
            api_secret="temporary-secret",
        )


def test_response_stream_and_json_failures_are_terminally_journaled() -> None:
    class _RawResponse:
        status_code = 200
        headers: Mapping[str, object] = {}

        def __init__(self, *, stream_error: bool) -> None:
            self.stream_error = stream_error

        def iter_content(self, chunk_size: int = 4_096):
            assert chunk_size == 4_096
            if self.stream_error:
                raise requests.ConnectionError("synthetic stream failure")
            yield b"{"

        def close(self) -> None:
            return None

    def run(stream_error: bool) -> list[Mapping[str, object]]:
        journal: list[Mapping[str, object]] = []

        def request(*_args: object, **_kwargs: object) -> _RawResponse:
            return _RawResponse(stream_error=stream_error)

        match = "stream failed" if stream_error else "JSON differs"
        with pytest.raises(RuntimeError, match=match):
            evidence_module._request_json(
                request,
                request_id="01-clock",
                venue="spot",
                base_url="https://api.binance.com",
                path="/api/v3/time",
                symbol=None,
                headers={},
                params={},
                timeout_seconds=5,
                journal=journal.append,
            )
        return journal

    stream_journal = run(True)
    assert stream_journal[-1]["failure_type"] == "ConnectionError"
    json_journal = run(False)
    assert json_journal[-2]["phase"] == "response_persisted_before_validation"
    assert json_journal[-1]["failure_type"] == "strict_json"

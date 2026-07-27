from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

import pytest

from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
)
from simple_ai_trading.round74_commission_capture import (
    ROUND74_COMMISSION_CAPTURE_BASE_URL,
    capture_round74_mainnet_commission,
)


@dataclass(frozen=True)
class _Response:
    payload: object
    headers: Mapping[str, object]
    status_code: int = 200

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload, separators=(",", ":")).encode("ascii")

    def json(self) -> object:
        return self.payload


def test_round74_commission_capture_is_get_only_complete_and_secret_free() -> None:
    calls: list[dict[str, object]] = []
    server_time = 1_800_000_000_000

    def request(
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, object],
        timeout: float,
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
        assert url.startswith(ROUND74_COMMISSION_CAPTURE_BASE_URL)
        if url.endswith("/fapi/v1/time"):
            assert "X-MBX-APIKEY" not in headers
            assert params == {}
            return _Response(
                {"serverTime": server_time + len(calls)},
                {"X-MBX-USED-WEIGHT-1M": str(len(calls))},
            )
        assert url.endswith("/fapi/v1/commissionRate")
        assert headers["X-MBX-APIKEY"] == "ephemeral-key"
        assert params["symbol"] in ROUND74_EVENT_TARGET_SYMBOLS
        assert params["recvWindow"] == 2_000
        assert isinstance(params["timestamp"], int)
        assert len(str(params["signature"])) == 64
        symbol = str(params["symbol"])
        return _Response(
            {
                "symbol": symbol,
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0004",
                "rpiCommissionRate": "0.00005",
            },
            {"X-MBX-USED-WEIGHT-1M": str(20 + len(calls))},
        )

    result = capture_round74_mainnet_commission(
        api_key="ephemeral-key",
        api_secret="ephemeral-secret",
        timeout_seconds=5.0,
        request=request,
    )

    assert len(calls) == 6
    assert [call["url"].rsplit("/", 1)[-1] for call in calls] == [
        "time",
        "commissionRate",
        "time",
        "commissionRate",
        "time",
        "commissionRate",
    ]
    assert result.bundle.as_mapping() == {
        symbol: 4.0 for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    artifact = result.as_dict()
    serialized = json.dumps(artifact, sort_keys=True)
    assert "ephemeral-key" not in serialized
    assert "ephemeral-secret" not in serialized
    assert "signature" not in serialized
    assert artifact["authority"]["orders_submitted"] is False
    assert len(result.capture_sha256) == 64


def test_round74_commission_capture_returns_no_partial_result() -> None:
    calls = 0

    def request(
        _method: str,
        url: str,
        **_kwargs: object,
    ) -> _Response:
        nonlocal calls
        calls += 1
        if url.endswith("/time"):
            return _Response({"serverTime": 1_800_000_000_000}, {})
        if calls < 4:
            return _Response(
                {
                    "symbol": "BTCUSDT",
                    "makerCommissionRate": "0.0002",
                    "takerCommissionRate": "0.0004",
                    "rpiCommissionRate": "0.00005",
                },
                {},
            )
        return _Response({"code": -2015}, {}, status_code=401)

    with pytest.raises(RuntimeError, match="response differs"):
        capture_round74_mainnet_commission(
            api_key="ephemeral-key",
            api_secret="ephemeral-secret",
            request=request,
        )
    assert calls == 4

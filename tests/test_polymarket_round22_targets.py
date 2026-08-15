from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from simple_ai_trading.polymarket_round22_targets import (
    POLYMARKET_ROUND22_DIAGNOSTIC_SHA256,
    Round22PublicTargetClient,
    load_round22_diagnostic_preregistration,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONDITION_ID = "0x" + ("a" * 64)


class _Response:
    def __init__(
        self,
        value: object,
        *,
        url: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = json.dumps(value, separators=(",", ":")).encode()
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url


class _Session:
    def __init__(self, responses: list[_Response | requests.RequestException]) -> None:
        self.headers: dict[str, str] = {}
        self.cookies: list[object] = []
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, requests.RequestException):
            raise response
        return response


def test_round22_diagnostic_preregistration_is_exactly_pinned() -> None:
    artifact = load_round22_diagnostic_preregistration(REPOSITORY)

    assert artifact["preregistration_sha256"] == POLYMARKET_ROUND22_DIAGNOSTIC_SHA256
    assert artifact["population"]["condition_count"] == 36
    assert artifact["population"]["role_counts"] == {
        "train": 12,
        "tune_calibration": 12,
        "tune_selection": 12,
    }
    assert artifact["authority"] == {
        "binance_private_api": False,
        "live_trading": False,
        "paper_trading": False,
        "polymarket_authentication": False,
        "polymarket_order_submission": False,
    }


def test_round22_target_client_is_public_origin_locked_and_retry_bounded() -> None:
    gamma_url = "https://gamma-api.polymarket.com/markets/12345"
    clob_url = f"https://clob.polymarket.com/markets/{CONDITION_ID}"
    session = _Session(
        [
            _Response({"error": "busy"}, url=gamma_url, status_code=503),
            _Response({"id": "12345"}, url=gamma_url),
            _Response({"condition_id": CONDITION_ID}, url=clob_url),
        ]
    )
    sleeps: list[float] = []
    client = Round22PublicTargetClient(
        session=session,
        minimum_request_interval_seconds=0,
        clock_ms=lambda: 123_456,
        sleeper=sleeps.append,
    )

    gamma = client.gamma_market("12345")
    clob = client.clob_market(CONDITION_ID.upper())

    assert gamma.value == {"id": "12345"}
    assert clob.value == {"condition_id": CONDITION_ID}
    assert gamma.observed_at_ms == clob.observed_at_ms == 123_456
    assert sleeps == [0.5]
    assert len(session.calls) == 3
    assert all(call["allow_redirects"] is False for call in session.calls)
    assert all(
        call["headers"]
        == {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-round22-target/0.1",
        }
        for call in session.calls
    )

    session.headers["POLY_API_KEY"] = "redacted"
    with pytest.raises(ValueError, match="authority headers"):
        client.gamma_market("12345")


def test_round22_target_client_rejects_redirected_or_unbounded_payloads() -> None:
    gamma_url = "https://gamma-api.polymarket.com/markets/12345"
    redirected = _Session(
        [_Response({"id": "12345"}, url="https://example.com/markets/12345")]
    )
    with pytest.raises(ValueError, match="changed origin"):
        Round22PublicTargetClient(
            session=redirected,
            minimum_request_interval_seconds=0,
        ).gamma_market("12345")

    wrong_type = _Session(
        [
            _Response(
                {"id": "12345"},
                url=gamma_url,
                headers={"Content-Type": "text/html"},
            )
        ]
    )
    with pytest.raises(ValueError, match="bounded JSON"):
        Round22PublicTargetClient(
            session=wrong_type,
            minimum_request_interval_seconds=0,
        ).gamma_market("12345")


def test_round22_target_client_retries_transport_failure_without_authority() -> None:
    gamma_url = "https://gamma-api.polymarket.com/markets/12345"
    session = _Session(
        [
            requests.ConnectionError("offline"),
            _Response({"id": "12345"}, url=gamma_url),
        ]
    )
    sleeps: list[float] = []
    client = Round22PublicTargetClient(
        session=session,
        minimum_request_interval_seconds=0,
        sleeper=sleeps.append,
    )

    assert client.gamma_market("12345").value == {"id": "12345"}
    assert sleeps == [0.5]

from __future__ import annotations

import pytest
import requests
from requests.adapters import BaseAdapter

from simple_ai_trading.api import BinanceAPIError, BinanceClient


class _Adapter(BaseAdapter):
    def __init__(self):
        self.sent = []

    def send(self, request, **kwargs):
        self.sent.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


@pytest.mark.parametrize("signed", [False, True])
def test_mutated_plaintext_testnet_endpoint_never_receives_request(signed):
    client = BinanceClient("offline-placeholder", "offline-placeholder", max_retries=0)
    adapter = _Adapter()
    client.session.mount("http://", adapter)
    client.base_url = "http://testnet.binance.vision"
    with pytest.raises(BinanceAPIError):
        client._request(
            "GET", "/api/v3/account" if signed else "/api/v3/time", signed=signed
        )
    assert adapter.sent == []
    client.session.close()


def test_public_request_does_not_send_ambient_api_key():
    client = BinanceClient("offline-placeholder", "offline-placeholder", max_retries=0)
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    client._request("GET", "/api/v3/time", signed=False)
    assert "X-MBX-APIKEY" not in adapter.sent[0].headers
    client.session.close()


@pytest.mark.parametrize(
    "origin",
    [
        "http://testnet.binance.vision",
        "https://testnet.binance.vision/path",
        "https://testnet.binance.vision?marker",
        "https://testnet.binance.vision#marker",
        "https://marker@testnet.binance.vision",
        "https://testnet.binance.vision:bad",
        "https://testnet.binance.vision:444",
        "https://testnet.binance.vision.evil.invalid",
        "https://testnet.binance.vision\\@evil.invalid",
        "https://testnet.binance.vision\n",
        "https://testnet.binance.vision?",
        "https://testnet.binance.vision#",
        "https://[bad",
    ],
)
def test_constructor_rejects_unsafe_origins_without_echo(monkeypatch, origin):
    from simple_ai_trading.api import ensure_non_mainnet_base_url

    monkeypatch.setenv("BINANCE_BASE_URL", origin)
    with pytest.raises(BinanceAPIError) as error:
        ensure_non_mainnet_base_url(origin, testnet=True, demo=False)
    assert origin not in str(error.value)
    # Env parsing strips whitespace; direct mutation must still reject it.
    if origin == origin.strip():
        with pytest.raises(BinanceAPIError):
            BinanceClient("", "")


@pytest.mark.parametrize("market_type", ["spot", "futures"])
@pytest.mark.parametrize("demo", [False, True])
def test_official_signed_call_and_following_public_call_isolate_key(market_type, demo):
    client = BinanceClient(
        "offline-placeholder", "offline-placeholder", market_type=market_type, demo=demo
    )
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    client.session.headers["x-mbx-apikey"] = "stale-offline-placeholder"
    prefix = "/fapi/v2" if market_type == "futures" else "/api/v3"
    client._request("GET", f"{prefix}/account", signed=True)
    assert adapter.sent[0].headers["X-MBX-APIKEY"] == "offline-placeholder"
    client._request(
        "GET", "/fapi/v1/time" if market_type == "futures" else "/api/v3/time"
    )
    assert "X-MBX-APIKEY" not in adapter.sent[1].headers
    client.session.close()


@pytest.mark.parametrize(
    "origin",
    ["https://api.binance.com", "https://fapi.binance.com", "https://example.invalid"],
)
def test_public_https_allowed_but_signed_call_refused(origin):
    client = BinanceClient("offline-placeholder", "offline-placeholder", testnet=False)
    client.base_url = origin
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    client.session.headers["x-mbx-apikey"] = "stale-offline-placeholder"
    client._request("GET", "/api/v3/time")
    assert "X-MBX-APIKEY" not in adapter.sent[0].headers
    with pytest.raises(BinanceAPIError):
        client._request("GET", "/api/v3/account", signed=True)
    assert len(adapter.sent) == 1
    client.session.close()


@pytest.mark.parametrize(
    "path",
    [
        "@evil.invalid",
        "//evil.invalid",
        "https://evil.invalid",
        "/api?marker",
        "/api#marker",
        "/api\\marker",
        "/api\nmarker",
    ],
)
@pytest.mark.parametrize("signed", [False, True])
def test_request_path_cannot_change_origin_or_query(path, signed):
    client = BinanceClient("offline-placeholder", "offline-placeholder")
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    with pytest.raises(BinanceAPIError):
        client._request("GET", path, signed=signed)
    assert adapter.sent == []
    client.session.close()


def test_mutation_during_request_does_not_replace_validated_origin(monkeypatch):
    client = BinanceClient("offline-placeholder", "offline-placeholder")
    client.base_url += "/"
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    monkeypatch.setattr(
        client, "_throttle", lambda: setattr(client, "base_url", "https://evil.invalid")
    )
    client._request("GET", "/api/v3/account", signed=True)
    assert adapter.sent[0].url.startswith("https://testnet.binance.vision/")
    client.session.close()


@pytest.mark.parametrize("signed", [False, True])
def test_non_mainnet_runtime_rejects_mutation_to_live_origin(signed):
    client = BinanceClient("offline-placeholder", "offline-placeholder")
    client.base_url = "https://api.binance.com"
    adapter = _Adapter()
    client.session.mount("https://", adapter)
    with pytest.raises(BinanceAPIError):
        client._request("GET", "/api/v3/account", signed=signed)
    assert adapter.sent == []
    client.session.close()

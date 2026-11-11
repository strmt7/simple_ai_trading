from __future__ import annotations

from decimal import Decimal
from threading import Event, Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import json
import time

import pytest

from simple_ai_trading.polymarket import CLOB_BASE_URL
from simple_ai_trading.polymarket_live import PolymarketLiveOrderIntent
from simple_ai_trading.polymarket_live_v2 import (
    OfficialPolymarketV2Venue,
    PolymarketLiveCredentials,
)

httpx = pytest.importorskip("httpx")
PolyApiException = pytest.importorskip("py_clob_client_v2.exceptions").PolyApiException
sdk_helpers = import_module("py_clob_client_v2.http_helpers.helpers")
transport_module = import_module("simple_ai_trading.polymarket_sdk_transport")
PolymarketSdkTransport = transport_module.PolymarketSdkTransport


class BodyStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def _transport(handler, **kwargs):
    return PolymarketSdkTransport(
        base_url=CLOB_BASE_URL,
        timeout_seconds=kwargs.pop("timeout_seconds", 2),
        maximum_response_bytes=kwargs.pop("maximum_response_bytes", 1024),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_transport_preserves_signed_bytes_headers_and_phase_budgets(method):
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"success": True})

    transport = _transport(handle, timeout_seconds=1.25)
    serialized = '{"unicode":"é","amount":"0.0100"}'
    try:
        assert transport.request(
            method,
            CLOB_BASE_URL + "/order",
            headers={"POLY_SIGNATURE": "offline-signature"},
            data=serialized,
            params={"cursor": "a+b"},
        ) == {"success": True}
        request = captured[0]
        assert len(captured) == 1
        assert request.content == serialized.encode("utf-8")
        assert request.headers["POLY_SIGNATURE"] == "offline-signature"
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.url.params["cursor"] == "a+b"
        assert request.extensions["timeout"] == {
            key: 1.25 for key in ("connect", "read", "write", "pool")
        }
    finally:
        transport.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308, 429, 500, 503])
def test_redirects_and_uncertain_responses_are_never_replayed(status):
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(
            status,
            headers={"Location": CLOB_BASE_URL + "/other"},
            json={"error": "offline"},
        )

    transport = _transport(handle)
    try:
        with pytest.raises(PolyApiException) as exc:
            transport.request("POST", CLOB_BASE_URL + "/order", data="{}")
        assert exc.value.status_code == status
        assert exc.value.error_msg == {"error": "offline"}
        assert len(captured) == 1
    finally:
        transport.close()


@pytest.mark.parametrize("mode", ["oversized", "encoded", "late", "interrupted"])
def test_incomplete_response_is_uncertain_and_always_closed(monkeypatch, mode):
    def chunks():
        yield b"{"
        if mode == "interrupted":
            raise httpx.ReadError("offline signed URL must not escape")
        yield b'"success":true}'

    stream = BodyStream(chunks())
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(
            422,
            headers={"content-encoding": "gzip"} if mode == "encoded" else {},
            stream=stream,
        )

    transport = _transport(
        handle, maximum_response_bytes=2 if mode == "oversized" else 1024
    )
    if mode == "late":
        clock = iter([0, 0, 0.5, 3])
        monkeypatch.setattr(transport_module, "monotonic", lambda: next(clock))
    try:
        with pytest.raises(PolyApiException) as exc:
            transport.request("POST", CLOB_BASE_URL + "/order", data="{}")
        assert exc.value.status_code is None  # A partial 422 must not release UNKNOWN.
        assert "signed URL" not in str(exc.value)
        assert len(captured) == 1 and stream.closed
    finally:
        transport.close()


@pytest.mark.parametrize("status", [200, 422])
def test_body_parsing_past_deadline_cannot_become_acceptance_or_rejection(
    monkeypatch, status
):
    clock = [0.0]
    native_json = httpx.Response.json

    def late_json(response, **kwargs):
        value = native_json(response, **kwargs)
        clock[0] = 3.0
        return value

    monkeypatch.setattr(transport_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(httpx.Response, "json", late_json)
    transport = _transport(
        lambda request: httpx.Response(status, json={"success": True})
    )
    try:
        with pytest.raises(PolyApiException, match="response deadline exceeded") as exc:
            transport.request("POST", CLOB_BASE_URL + "/order", data="{}")
        assert exc.value.status_code is None
    finally:
        transport.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://foreign.invalid/order",
        "http://clob.polymarket.com/order",
        "https://offline:secret@clob.polymarket.com/order",
        CLOB_BASE_URL + "/order#fragment",
    ],
)
def test_transport_rejects_other_origins_before_dispatch(endpoint):
    captured = []
    transport = _transport(lambda request: captured.append(request))
    try:
        with pytest.raises(ValueError, match="configured origin"):
            transport.request("POST", endpoint)
        assert not captured
    finally:
        transport.close()


def _credentials():
    from eth_account import Account

    private_key = "0x" + "1" * 64  # Published synthetic offline fixture, never funded.
    return PolymarketLiveCredentials(
        private_key=private_key,
        api_key="offline-api-key",
        api_secret="b2ZmbGluZS1hcGktc2VjcmV0",
        api_passphrase="offline-passphrase",
        funder_address=Account.from_key(private_key).address.lower(),
        signature_type=0,
    )


def test_real_sdk_uses_isolated_transport_and_never_its_global_client(monkeypatch):
    native_transport = PolymarketSdkTransport
    captured, owned = [], []
    sdk_global = sdk_helpers._http_client

    def factory(**kwargs):
        def handle(request):
            captured.append(request)
            return httpx.Response(
                200,
                json={"success": True, "orderID": "0x" + "2" * 64, "status": "live"},
            )

        transport = native_transport(**kwargs, transport=httpx.MockTransport(handle))
        owned.append(transport)
        return transport

    def forbidden(*args, **kwargs):
        raise AssertionError("SDK global transport must not receive any request")

    monkeypatch.setattr(transport_module, "PolymarketSdkTransport", factory)
    monkeypatch.setattr(sdk_global, "request", forbidden)
    first = OfficialPolymarketV2Venue(_credentials(), timeout_seconds=1.5)
    second = OfficialPolymarketV2Venue(_credentials(), timeout_seconds=2.5)
    try:
        first._client.get_version = lambda: 2
        first._client.get_tick_size = lambda token: "0.01"
        now = int(time.time() * 1000)
        intent = PolymarketLiveOrderIntent(
            intent_id="sdk-transport-offline",
            bot_id="sdk-transport-offline",
            market_id="0x" + "1" * 64,
            token_id="1" * 40,
            symbol="BTC",
            outcome="Up",
            side="BUY",
            order_type="FAK",
            limit_price=Decimal("0.50"),
            quantity=Decimal("5"),
            fee_reserve_quote=Decimal("0.10"),
            created_at_ms=now,
            expires_at_ms=now + 120000,
        )
        prepared = first.prepare_order(
            intent, tick_size=Decimal("0.01"), neg_risk=False
        )
        assert first.submit_order(prepared).accepted
        first._client._get(CLOB_BASE_URL + "/offline")
        first._client._delete(CLOB_BASE_URL + "/offline", data='{"id":"offline"}')
        assert [request.method for request in captured] == ["POST", "GET", "DELETE"]
        payload = json.loads(captured[0].content)
        assert payload["owner"] == "offline-api-key"
        assert payload["orderType"] == "FAK" and payload["deferExec"] is False
        assert "POLY_SIGNATURE" in captured[0].headers
        assert all(request.extensions["timeout"]["read"] == 1.5 for request in captured)
        assert sdk_helpers._http_client is sdk_global
        first.close()
        assert owned[0]._client.is_closed and not owned[1]._client.is_closed
        second._client._get(CLOB_BASE_URL + "/offline")
        assert captured[-1].extensions["timeout"]["read"] == 2.5
    finally:
        first.close()
        second.close()
    assert all(transport._client.is_closed for transport in owned)


def test_owned_loopback_drip_exceeds_total_observed_deadline_without_idle_timeout():
    stopped, finished = Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                for _ in range(100):
                    if stopped.wait(0.025):
                        break
                    self.wfile.write(b" ")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    transport = PolymarketSdkTransport(
        base_url=f"http://127.0.0.1:{server.server_port}",
        timeout_seconds=0.2,
        maximum_response_bytes=1024,
    )
    thread.start()
    try:
        transport._client.start(timeout_seconds=5)
        with pytest.raises(PolyApiException, match="response deadline exceeded"):
            transport.request("GET", f"http://127.0.0.1:{server.server_port}/drip")
    finally:
        stopped.set()
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert finished.wait(timeout=3) and not thread.is_alive()


def test_cancelled_sdk_construction_closes_both_owned_clients(monkeypatch):
    import py_clob_client_v2
    from simple_ai_trading import polymarket_live_v2 as venue_module

    native_transport = PolymarketSdkTransport
    native_session = venue_module._public_session
    owned, sessions = [], []

    def factory(**kwargs):
        result = native_transport(
            **kwargs, transport=httpx.MockTransport(lambda r: None)
        )
        owned.append(result)
        return result

    def session_factory():
        session = native_session()
        close = session.close
        session.was_closed = False

        def tracked_close():
            session.was_closed = True
            close()

        session.close = tracked_close
        sessions.append(session)
        return session

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt("offline construction cancelled")

    monkeypatch.setattr(transport_module, "PolymarketSdkTransport", factory)
    monkeypatch.setattr(venue_module, "_public_session", session_factory)
    monkeypatch.setattr(py_clob_client_v2.ClobClient, "__init__", cancelled)
    with pytest.raises(KeyboardInterrupt, match="construction cancelled"):
        OfficialPolymarketV2Venue(_credentials())
    assert len(owned) == len(sessions) == 1
    assert owned[0]._client.is_closed and sessions[0].was_closed


def test_venue_close_attempts_public_session_after_sdk_close_failure():
    class FailingClient:
        def close(self):
            raise RuntimeError("offline SDK cleanup failed")

    class Session:
        closed = False

        def close(self):
            self.closed = True

    session = Session()
    venue = OfficialPolymarketV2Venue(
        _credentials(), client=FailingClient(), session=session
    )
    with pytest.raises(RuntimeError, match="SDK cleanup failed"):
        venue.close()
    assert session.closed

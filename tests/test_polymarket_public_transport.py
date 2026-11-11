"""Public venue reads share owned containment without gaining write authority."""

import json
import time

import pytest

pytest.importorskip("py_clob_client_v2")
from py_clob_client_v2.exceptions import PolyApiException

from simple_ai_trading import polymarket_live_v2 as venue_module
from simple_ai_trading.polymarket import CLOB_BASE_URL
from simple_ai_trading.polymarket_sdk_transport import PolymarketPublicReadTransport
from test_polymarket_http_process import make_http_process_harness
from test_polymarket_live_v2 import FakeClient, FakeResponse, _credentials


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return make_http_process_harness(tmp_path, monkeypatch)


def reader(timeout=2):
    return PolymarketPublicReadTransport(
        allowed_endpoints=(
            CLOB_BASE_URL + "/version",
            venue_module.POLYMARKET_GEOBLOCK_URL,
        ),
        timeout_seconds=timeout,
        maximum_response_bytes=4096,
    )


def test_default_preflight_routes_all_public_origins_through_one_child(harness):
    create, children = harness
    client = FakeClient()
    client.get_closed_only_mode = lambda: {"closed_only": False}
    client.get_open_orders = lambda: []
    venue = venue_module.OfficialPolymarketV2Venue(_credentials(), client=client)
    worker, marker = create("public", worker=venue.session._client)
    try:
        settings = json.loads(worker._bootstrap)
        assert settings["public_only"] is True and len(settings["base_urls"]) == 3
        assert len(settings["allowed_endpoints"]) == 4
        worker.start(timeout_seconds=5)
        result = venue.preflight()
        assert result.protocol_version == 2 and not result.geoblocked
        assert result.positions == () and result.open_orders == ()
        assert marker.read_text().splitlines() == ["GET"] * 4
        assert len(children) == 1
    finally:
        venue.close()
    assert children[0].poll() is not None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://foreign.invalid/version",
        CLOB_BASE_URL + "/unadmitted",
        CLOB_BASE_URL + "/version#fragment",
        "https://offline:secret@clob.polymarket.com/version",
    ],
)
def test_public_endpoint_rejection_starts_no_helper(harness, endpoint):
    create, children = harness
    transport = reader()
    _, marker = create(worker=transport._client)
    try:
        with pytest.raises(ValueError, match="not admitted"):
            transport.get_json(endpoint)
        assert not children and not marker.exists()
    finally:
        transport.close()


@pytest.mark.parametrize(
    ("method", "endpoint", "headers", "data"),
    [
        ("POST", CLOB_BASE_URL + "/version", {}, None),
        ("GET", CLOB_BASE_URL + "/version", {"Authorization": "offline"}, None),
        ("GET", CLOB_BASE_URL + "/version", {}, "{}"),
        ("GET", "https://foreign.invalid/version", {}, None),
        ("GET", CLOB_BASE_URL + "/unadmitted", {}, None),
    ],
)
def test_child_independently_rejects_writes_authentication_and_foreign_origins(
    harness, method, endpoint, headers, data
):
    create, children = harness
    transport = reader()
    worker, marker = create(worker=transport._client)
    try:
        worker.start(timeout_seconds=5)
        with pytest.raises(PolyApiException) as exc:
            worker.request(method, endpoint, headers=headers, data=data)
        assert exc.value.status_code is None
        assert not marker.exists() and children[0].poll() is not None
    finally:
        transport.close()


def test_stalled_public_read_is_reaped_at_the_callers_shorter_deadline(harness):
    create, children = harness
    transport = reader()
    worker, marker = create("stalled", worker=transport._client)
    try:
        worker.start(timeout_seconds=5)
        with pytest.raises(PolyApiException) as exc:
            transport.get_json(
                CLOB_BASE_URL + "/version", deadline=time.monotonic() + 0.2
            )
        assert exc.value.status_code is None
        assert marker.read_text().splitlines() == ["GET"]
        assert children[0].poll() is not None
    finally:
        transport.close()


@pytest.mark.parametrize("deadline", [-1, float("nan"), True])
def test_expired_or_invalid_caller_budget_starts_no_helper(harness, deadline):
    create, children = harness
    transport = reader()
    _, marker = create(worker=transport._client)
    try:
        with pytest.raises((PolyApiException, ValueError)):
            transport.get_json(CLOB_BASE_URL + "/version", deadline=deadline)
        assert not children and not marker.exists()
    finally:
        transport.close()


@pytest.mark.parametrize("phase", ["network", "materialization"])
def test_position_pagination_has_one_total_budget_and_never_returns_partial_flatness(
    monkeypatch, phase
):
    clock = [0.0]
    calls = []

    class Session:
        def get(self, _url, *, params, timeout):
            calls.append((params["offset"], timeout))
            clock[0] += 0.4 if phase == "network" else 0
            return FakeResponse([{"size": "0"}] * (500 if phase == "network" else 1))

        def close(self):
            pass

    native_mapping = venue_module._mapping

    def materialize(value, *, name):
        if phase == "materialization" and name == "position":
            clock[0] = 2.0
        return native_mapping(value, name=name)

    monkeypatch.setattr(venue_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(venue_module, "_mapping", materialize)
    venue = venue_module.OfficialPolymarketV2Venue(
        _credentials(),
        client=FakeClient(),
        session=Session(),
        timeout_seconds=1,
    )
    try:
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            venue.positions()
        assert len(calls) == (3 if phase == "network" else 1)
        if phase == "network":
            assert [timeout for _, timeout in calls] == pytest.approx([1, 0.6, 0.2])
    finally:
        venue.close()


def test_server_clock_midpoint_excludes_version_request_latency(monkeypatch):
    first = 1_800_000_000
    clock = [float(first)]
    server = []

    class Session:
        def get(self, url, **_kwargs):
            if url.endswith("/version"):
                clock[0] += 10
                return FakeResponse({"version": 2})
            assert url.endswith("/time")
            clock[0] += 0.2
            server.append(int(clock[0] * 1000))
            return FakeResponse(server[-1])

        def close(self):
            pass

    monkeypatch.setattr(venue_module.time, "time", lambda: clock[0])
    venue = venue_module.OfficialPolymarketV2Venue(
        _credentials(),
        client=FakeClient(),
        session=Session(),
    )
    try:
        version, server_time, observed = venue._protocol_and_time()
        assert version == 2 and server_time == server[-1]
        assert observed == ((first + 10) * 1000 + server[-1]) // 2
    finally:
        venue.close()


@pytest.mark.parametrize("scope", ["empty", "oversized", "too_many_origins"])
def test_invalid_public_scope_rejects_before_starting_a_helper(harness, scope):
    _, children = harness
    if scope == "empty":
        endpoints = ()
    elif scope == "oversized":
        endpoints = (CLOB_BASE_URL + "/version",) * 33
    else:
        endpoints = tuple(
            f"https://public-{index}.invalid/version" for index in range(9)
        )
    with pytest.raises(ValueError):
        PolymarketPublicReadTransport(
            allowed_endpoints=endpoints,
            timeout_seconds=2,
            maximum_response_bytes=4096,
        )
    assert not children

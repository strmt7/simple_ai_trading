"""Owned native HTTP faults must end before a later exchange can start."""

import ast
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import socket
import time
from types import SimpleNamespace
from threading import Event

import pytest
import httpx

pytest.importorskip("py_clob_client_v2")
from py_clob_client_v2.exceptions import PolyApiException

from simple_ai_trading import polymarket_http_process as module
from simple_ai_trading import polymarket_sdk_transport as sdk_transport
from simple_ai_trading.polymarket import CLOB_BASE_URL
from simple_ai_trading.polymarket_live import (
    PolymarketLiveCoordinator,
    PolymarketLiveBlocked,
    PolymarketLiveError,
    PolymarketLiveOrderLedger,
    PolymarketLiveUnknownState,
)
from simple_ai_trading.polymarket_live_v2 import OfficialPolymarketV2Venue
from simple_ai_trading.polymarket_order_dispatch import (
    OrderDispatchAdmission,
    PolymarketOrderNotSent,
)
import test_polymarket_live as live_fixture
from test_polymarket_live_v2 import _credentials


_SCRIPT = r"""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, sys.argv[1])
from simple_ai_trading import polymarket_http_process as module
from simple_ai_trading import polymarket_sdk_transport as native

if "HTTP_WORKER_TEST_SECRET" in os.environ:
    raise RuntimeError("environment isolation failed")
mode = sys.argv[3]


def request(self, method, endpoint, *, headers=None, data=None, params=None):
    with Path(sys.argv[4]).open("a", encoding="ascii") as marker:
        marker.write(method + "\n")
    if mode == "stalled":
        time.sleep(60)
    if mode == "exit":
        os._exit(3)
    if mode == "reject":
        error = module.PolyApiException(error_msg={"error": "offline rejection"})
        error.status_code = 422
        raise error
    if mode == "public":
        if endpoint.endswith("/version"):
            return {"version": 2}
        if endpoint.endswith("/time"):
            return int(time.time() * 1000)
        if endpoint.endswith("/api/geoblock"):
            return {"blocked": False, "country": "XX", "region": ""}
        if endpoint.endswith("/positions"):
            return []
    return {
        "method": method,
        "endpoint": endpoint,
        "headers": headers,
        "data": data,
        "params": params,
    }


native._InlineSdkTransport.request = request
if mode == "wrong_sequence":
    original = module._HEADER

    class WrongHeader:
        size = original.size
        unpack = original.unpack

        @staticmethod
        def pack(sequence, size):
            return original.pack(sequence + 1, size)

    module._HEADER = WrongHeader()
module._worker(int(sys.argv[2]))
"""


def make_http_process_harness(tmp_path, monkeypatch):
    ast.parse(_SCRIPT)
    children = []
    native_popen = module.subprocess.Popen

    def launch(*args, **kwargs):
        child = native_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setenv("HTTP_WORKER_TEST_SECRET", "synthetic sentinel")

    def create(mode="normal", timeout=0.4, *, worker=None, **settings):
        if worker is None:
            worker = module.PolymarketHttpProcess(
                base_url=CLOB_BASE_URL,
                timeout_seconds=timeout,
                maximum_response_bytes=4096,
                **settings,
            )
        command = worker._command
        marker = tmp_path / f"{mode}-{len(children)}.calls"
        monkeypatch.setattr(
            worker,
            "_command",
            lambda descriptor: [
                *command(descriptor)[:4],
                _SCRIPT,
                *command(descriptor)[5:],
                mode,
                str(marker),
            ],
        )
        return worker, marker

    return create, children


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return make_http_process_harness(tmp_path, monkeypatch)


def test_one_child_preserves_individual_requests_without_replay(harness):
    create, children = harness
    worker, marker = create(timeout=2)
    try:
        worker.start(timeout_seconds=5)
        first = worker.request(
            "POST",
            CLOB_BASE_URL + "/order",
            headers={"POLY_SIGNATURE": "offline-signature"},
            data='{"unicode":"Ã©","amount":"0.0100"}',
            params={"cursor": "a+b"},
        )
        assert first["data"] == '{"unicode":"Ã©","amount":"0.0100"}'
        assert first["headers"] == {"POLY_SIGNATURE": "offline-signature"}
        assert first["params"] == {"cursor": ["a+b"]}
        assert worker.request("GET", CLOB_BASE_URL + "/offline")["method"] == "GET"
        assert len(children) == 1
        assert marker.read_text().splitlines() == ["POST", "GET"]
    finally:
        worker.close()
    assert worker.is_closed and children[0].poll() is not None


@pytest.mark.parametrize("mode", ["stalled", "exit", "wrong_sequence"])
def test_native_fault_reaps_before_any_later_request(harness, mode, monkeypatch):
    create, children = harness
    worker, marker = create(mode)
    try:
        worker.start(timeout_seconds=5)
        with pytest.raises(PolyApiException) as exc:
            worker.request("POST", CLOB_BASE_URL + "/order", data="{}")
        assert exc.value.status_code is None
        assert children[0].poll() is not None
        assert worker._connection is None and worker._process is None
        assert marker.read_text().splitlines() == ["POST"]
        command = worker._command
        monkeypatch.setattr(
            worker,
            "_command",
            lambda descriptor: [*command(descriptor)[:-2], "normal", str(marker)],
        )
        worker.start(timeout_seconds=5)
        assert worker.request("GET", CLOB_BASE_URL + "/offline")["method"] == "GET"
        assert len(children) == 2 and children[0].poll() is not None
        assert marker.read_text().splitlines() == ["POST", "GET"]
    finally:
        worker.close()
    assert all(child.poll() is not None for child in children)


def test_complete_rejection_retains_native_status_and_is_not_replayed(harness):
    create, children = harness
    worker, marker = create("reject")
    try:
        worker.start(timeout_seconds=5)
        with pytest.raises(PolyApiException) as exc:
            worker.request("POST", CLOB_BASE_URL + "/order", data="{}")
        assert exc.value.status_code == 422
        assert exc.value.error_msg == {"error": "offline rejection"}
        assert marker.read_text().splitlines() == ["POST"]
    finally:
        worker.close()
    assert children[0].poll() is not None


def test_explicit_close_interrupts_a_blocked_native_request(harness):
    create, children = harness
    worker, marker = create("stalled", timeout=20)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        worker.start(timeout_seconds=5)
        pending = executor.submit(worker.request, "POST", CLOB_BASE_URL + "/order")
        deadline = time.monotonic() + 3
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        worker.close()
        with pytest.raises(PolyApiException) as exc:
            pending.result(timeout=2)
        assert exc.value.status_code is None
        assert children[0].poll() is not None
        with pytest.raises(PolyApiException):
            worker.request("GET", CLOB_BASE_URL + "/offline")
        assert len(children) == 1
    finally:
        worker.close()
        executor.shutdown(wait=True)


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_protocol_rejects_ambiguous_json(body):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(module._HEADER.pack(1, len(body)) + body)
        with pytest.raises(ValueError):
            module._receive(right, 1024)


def test_lazy_construction_and_rejected_frame_start_no_child(harness):
    create, children = harness
    worker, _ = create()
    try:
        with pytest.raises(ValueError, match="byte limit"):
            worker.request("POST", CLOB_BASE_URL + "/order", data="x" * (2**20))
        assert not children
    finally:
        worker.close()


def test_query_normalization_preserves_native_decimal_and_repeated_values(harness):
    create, _ = harness
    worker, _ = create(timeout=2)
    params = {
        "amount": Decimal("0.0100"),
        "enabled": True,
        "missing": None,
        "cursor": ["a+b", "é"],
    }
    try:
        worker.start(timeout_seconds=5)
        reply = worker.request("GET", CLOB_BASE_URL + "/offline", params=params)
        assert str(httpx.QueryParams(reply["params"])) == str(httpx.QueryParams(params))
    finally:
        worker.close()


def test_capacity_expires_without_dispatching_another_request(harness):
    create, children = harness
    worker, _ = create(timeout=0.05)
    worker._exchange.acquire()
    try:
        with pytest.raises(PolyApiException, match="capacity deadline"):
            worker.request("POST", CLOB_BASE_URL + "/order")
        assert not children
    finally:
        worker._exchange.release()
        worker.close()


def test_startup_expiry_reaps_its_original_process(harness, monkeypatch):
    create, children = harness
    worker, _ = create()
    command = worker._command
    monkeypatch.setattr(
        worker,
        "_command",
        lambda descriptor: [*command(descriptor)[:4], "import time; time.sleep(60)"],
    )
    try:
        with pytest.raises(TimeoutError):
            worker.start(timeout_seconds=0.2)
        assert children and all(child.poll() is not None for child in children)
        assert worker._connection is None and worker._process is None
    finally:
        worker.close()


def test_hung_http_submission_stays_unknown_after_ledger_reopen(
    harness, tmp_path, monkeypatch
):
    create, children = harness
    worker, marker = create("stalled")

    def post_order(*_args, **_kwargs):
        return worker.request("POST", CLOB_BASE_URL + "/order", data="{}")

    official = OfficialPolymarketV2Venue(
        _credentials(), client=SimpleNamespace(post_order=post_order)
    )
    venue = live_fixture.FakeVenue()
    venue.submit_order = official.submit_order
    ledger = PolymarketLiveOrderLedger(tmp_path / "unknown.sqlite3")
    coordinator = PolymarketLiveCoordinator(
        venue, ledger, risk_limits=live_fixture.RISK_LIMITS
    )
    try:
        worker.start(timeout_seconds=5)
        monkeypatch.setattr(live_fixture, "NOW_MS", int(time.time() * 1000))
        intent = live_fixture._intent()
        with pytest.raises(PolymarketLiveUnknownState, match="no retry"):
            coordinator.submit(intent, tick_size=Decimal("0.01"), neg_risk=False)
        assert children[0].poll() is not None
        assert marker.read_text().splitlines() == ["POST"]
        reopened = PolymarketLiveOrderLedger(ledger.path)
        assert reopened.record(intent.intent_id).state == "unknown"
        assert not coordinator.preflight().can_open
    finally:
        worker.close()
        official.close()


@pytest.mark.parametrize("revocation", ["source", "intent", "authority"])
def test_source_revoked_during_http_startup_never_transmits_order(
    harness, tmp_path, monkeypatch, revocation
):
    create, children = harness
    original_factory = sdk_transport.PolymarketSdkTransport
    revoked = False
    markers = []

    def transport_factory(**kwargs):
        transport = original_factory(**kwargs)
        worker, marker = create(worker=transport._client)
        markers.append(marker)
        connect = worker._connect

        def connect_then_revoke(deadline):
            nonlocal revoked
            connection = connect(deadline)
            revoked = True
            return connection

        monkeypatch.setattr(worker, "_connect", connect_then_revoke)
        return transport

    monkeypatch.setattr(sdk_transport, "PolymarketSdkTransport", transport_factory)
    official = OfficialPolymarketV2Venue(_credentials())
    official._client.get_version = lambda: 2
    official._client.get_tick_size = lambda token: "0.01"
    venue = live_fixture.FakeVenue()
    venue.prepare_order = official.prepare_order
    venue.submit_order = official.submit_order
    ledger = PolymarketLiveOrderLedger(tmp_path / "startup-revocation.sqlite3")
    coordinator = PolymarketLiveCoordinator(
        venue, ledger, risk_limits=live_fixture.RISK_LIMITS
    )
    if revocation == "intent":
        monkeypatch.setattr(
            live_fixture.polymarket_live_module,
            "time",
            SimpleNamespace(
                time=lambda: live_fixture.NOW_MS / 1000 + (121 if revoked else 0)
            ),
        )
    elif revocation == "authority":

        def assert_authority(*, closing_only):
            if revoked:
                raise PolymarketLiveBlocked("runtime authority changed during startup")

        coordinator.runtime_authority = SimpleNamespace(
            assert_submission_allowed=assert_authority,
            reconciliation_checkpoint=lambda: None,
            note_reconciliation=lambda result, *, checkpoint: None,
            note_reconciliation_failure=lambda reason, *, checkpoint: None,
        )

    def assert_current():
        if revoked and revocation == "source":
            raise PolymarketLiveBlocked("public source changed during startup")

    try:
        monkeypatch.setattr(live_fixture, "NOW_MS", int(time.time() * 1000))
        intent = live_fixture._intent()
        with pytest.raises(PolymarketLiveError):
            coordinator.submit(
                intent,
                tick_size=Decimal("0.01"),
                neg_risk=False,
                source_admission_check=assert_current,
            )
        assert revoked and children
        assert all(not marker.exists() for marker in markers)
        record = PolymarketLiveOrderLedger(ledger.path).record(intent.intent_id)
        assert record.state == "rejected"
        assert record.failure_code == "admission_revoked_before_http_dispatch"
    finally:
        official.close()
    assert all(child.poll() is not None for child in children)


def test_source_rechecked_after_capacity_wait_without_dropping_healthy_child(
    harness, monkeypatch
):
    create, children = harness
    worker, marker = create(timeout=2)
    waiting = Event()
    revoked = False
    native_lock = worker._exchange

    class ObservedLock:
        def acquire(self, **kwargs):
            waiting.set()
            return native_lock.acquire(**kwargs)

        def release(self):
            native_lock.release()

    def check():
        if revoked:
            raise RuntimeError("offline source revoked")

    executor = ThreadPoolExecutor(max_workers=1)
    held = False
    try:
        worker.start(timeout_seconds=5)
        monkeypatch.setattr(worker, "_exchange", ObservedLock())
        native_lock.acquire()
        held = True
        admission = OrderDispatchAdmission(check)
        pending = executor.submit(
            worker.request,
            "POST",
            CLOB_BASE_URL + "/order",
            admission=admission,
        )
        assert waiting.wait(timeout=2)
        revoked = True
        native_lock.release()
        held = False
        with pytest.raises(PolymarketOrderNotSent):
            pending.result(timeout=2)
        assert admission.rejected_before_transmission
        assert not marker.exists() and children[0].poll() is None
        assert worker.request("GET", CLOB_BASE_URL + "/offline")["method"] == "GET"
        assert len(children) == 1 and marker.read_text().splitlines() == ["GET"]
    finally:
        if held:
            native_lock.release()
        worker.close()
        executor.shutdown(wait=True)
    assert all(child.poll() is not None for child in children)


def test_admitted_native_send_cannot_later_prove_the_submission_unsent(harness):
    create, children = harness
    worker, marker = create(timeout=2)
    revoked = False
    checks = []

    def check():
        checks.append("checked")
        if revoked:
            raise RuntimeError("offline revocation after an admitted exchange")

    admission = OrderDispatchAdmission(check)
    try:
        worker.start(timeout_seconds=5)
        assert (
            worker.request("POST", CLOB_BASE_URL + "/order", admission=admission)[
                "method"
            ]
            == "POST"
        )
        assert checks == ["checked"] and not admission.rejected_before_transmission
        revoked = True
        with pytest.raises(PolymarketOrderNotSent):
            worker.request("POST", CLOB_BASE_URL + "/order", admission=admission)
        assert not admission.rejected_before_transmission
        assert marker.read_text().splitlines() == ["POST"]
    finally:
        worker.close()
    assert all(child.poll() is not None for child in children)

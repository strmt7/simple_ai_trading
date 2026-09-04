from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from tools import screen_polymarket_exact_negrisk_long_only_frontier_v2 as transport


class Response(BytesIO):
    code = 200


def invoke(tmp_path):
    return transport.bounded_request(
        method="GET",
        url="https://gamma-api.polymarket.com/events/slug/frozen-event",
        body=b"",
        name="test",
        raw_path=tmp_path / "raw.json",
        raw_relative_path="raw.json",
        journal_path=tmp_path / "journal.jsonl",
    )


def stub(monkeypatch, response):
    class Opener:
        def open(self, request, timeout):
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(transport, "build_opener", lambda *args: Opener())


def test_success_and_no_repeat(tmp_path, monkeypatch):
    stub(monkeypatch, Response(b'{"id":"1"}'))
    raw, receipt = invoke(tmp_path)
    assert raw == (tmp_path / "raw.json").read_bytes()
    assert receipt["status_code"] == 200
    assert receipt["within_byte_ceiling"] is True
    with pytest.raises(RuntimeError, match="already exists"):
        invoke(tmp_path)


def test_ceiling_retains_one_overflow_byte_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "BYTE_CEILING", 4)
    stub(monkeypatch, Response(b"123456789"))
    with pytest.raises(RuntimeError, match="failed"):
        invoke(tmp_path)
    assert (tmp_path / "raw.json").read_bytes() == b"12345"
    rows = [
        json.loads(x) for x in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["within_byte_ceiling"] is False


def test_timeout_retains_partial_bytes(tmp_path, monkeypatch):
    class Partial(Response):
        calls = 0

        def read(self, size):
            self.calls += 1
            if self.calls > 1:
                raise TimeoutError()
            return b"partial"

    stub(monkeypatch, Partial())
    with pytest.raises(RuntimeError, match="failed"):
        invoke(tmp_path)
    assert (tmp_path / "raw.json").read_bytes() == b"partial"
    assert "TimeoutError" in (tmp_path / "journal.jsonl").read_text()


def test_http_error_body_and_redirect_rejection(tmp_path, monkeypatch):
    stub(
        monkeypatch,
        HTTPError(
            "https://example.invalid", 302, "redirect", {}, BytesIO(b"redirect body")
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        invoke(tmp_path)
    assert (tmp_path / "raw.json").read_bytes() == b"redirect body"
    assert (
        transport.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://example.invalid"
        )
        is None
    )


def test_wrong_endpoint_rejected_before_any_write(tmp_path):
    with pytest.raises(ValueError):
        transport.bounded_request(
            method="POST",
            url="https://example.invalid",
            body=b"",
            name="test",
            raw_path=tmp_path / "raw",
            raw_relative_path="raw",
            journal_path=tmp_path / "journal",
        )
    assert not list(tmp_path.iterdir())


def test_elapsed_budget_records_terminal_timeout(tmp_path, monkeypatch):
    times = iter([0, 31])
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(times))
    stub(monkeypatch, Response(b"not read"))
    with pytest.raises(RuntimeError, match="failed"):
        invoke(tmp_path)
    assert (tmp_path / "raw.json").read_bytes() == b""
    assert "TimeoutError" in (tmp_path / "journal.jsonl").read_text()


def test_bridge_restores_legacy_transport_on_error(monkeypatch):
    original = transport.legacy._request

    def failed_main():
        assert transport.legacy._request is transport.bounded_request
        raise ValueError("local failure")

    monkeypatch.setattr(transport.legacy, "main", failed_main)
    with pytest.raises(ValueError, match="local failure"):
        transport.main()
    assert transport.legacy._request is original

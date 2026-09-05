"""Socket-free transport regressions; the shared validator is tested separately."""

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from tools import capture_public_source_bounded as capture


class Response(BytesIO):
    code = 200


def setup_plan(tmp_path, monkeypatch, payload=b'{"ok":true}', ceiling=100):
    plan = {
        "transport": {
            "socket_timeout_seconds": 10,
            "read_budget_seconds": 30,
            "redirects": False,
            "retries": 0,
            "proxies": False,
        },
        "outputs": {
            "raw_path": str(tmp_path / "raw"),
            "journal_path": str(tmp_path / "journal"),
            "result_path": str(tmp_path / "result"),
        },
        "response_byte_ceiling": ceiling,
        "contract_sha256": "a" * 64,
        "contract_path": str(tmp_path / "contract"),
        "request": {"method": "GET", "url": "https://example.test/public", "count": 1},
        "required_utf8_phrases": ['"ok"'],
        "authority": {},
    }
    monkeypatch.setattr(capture, "_load_object", lambda _: plan)
    monkeypatch.setattr(capture, "_validate_contract", Mock())
    monkeypatch.setattr(capture, "_root_path", Path)
    opener = Mock()
    opener.open.return_value = Response(payload)
    monkeypatch.setattr(capture, "build_opener", lambda *_: opener)
    return plan, opener


def test_success_durable_receipt_and_consumed_outputs(tmp_path, monkeypatch):
    plan, opener = setup_plan(tmp_path, monkeypatch)
    result = capture.capture(Path(plan["contract_path"]))
    assert result["source_gate"]["passed"] is True
    assert result["profitability_claim"] is False
    rows = [
        json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows] == ["intent", "completed"]
    assert rows[-1]["response_bytes"] == len((tmp_path / "raw").read_bytes())
    with pytest.raises(FileExistsError):
        capture.capture(Path(plan["contract_path"]))
    assert opener.open.call_count == 1


@pytest.mark.parametrize(
    "payload,ceiling", [(b'{"ok":true}', 4), (b"\xff", 100), (b"{}", 100)]
)
def test_size_decode_and_phrase_fail_closed(tmp_path, monkeypatch, payload, ceiling):
    plan, _ = setup_plan(tmp_path, monkeypatch, payload, ceiling)
    result = capture.capture(Path(plan["contract_path"]))
    assert result["source_gate"]["passed"] is False
    assert len((tmp_path / "raw").read_bytes()) <= ceiling + 1


@pytest.mark.parametrize("status", [302, 429, 500])
def test_http_error_body_is_retained_without_retry(tmp_path, monkeypatch, status):
    plan, opener = setup_plan(tmp_path, monkeypatch)
    opener.open.side_effect = HTTPError(
        plan["request"]["url"], status, "failed", {}, Response(b"error")
    )
    result = capture.capture(Path(plan["contract_path"]))
    assert result["source_gate"]["passed"] is False
    assert result["capture"]["receipt"]["status_code"] == status
    assert (tmp_path / "raw").read_bytes() == b"error"
    assert opener.open.call_count == 1


def test_timeout_is_terminal_and_preflight_never_calls_network(tmp_path, monkeypatch):
    plan, opener = setup_plan(tmp_path, monkeypatch)
    assert capture.capture(Path(plan["contract_path"]), preflight=True) is None
    assert not (tmp_path / "raw").exists()
    opener.open.assert_not_called()
    opener.open.side_effect = TimeoutError("not persisted")
    result = capture.capture(Path(plan["contract_path"]))
    assert result["source_gate"]["passed"] is False
    assert result["capture"]["receipt"]["error_type"] == "TimeoutError"


@pytest.mark.parametrize("invalid", ["transport", "duplicate", "parent"])
def test_invalid_preflight_stops_before_access(tmp_path, monkeypatch, invalid):
    plan, opener = setup_plan(tmp_path, monkeypatch)
    if invalid == "transport":
        plan["transport"]["retries"] = 1
    elif invalid == "duplicate":
        plan["outputs"]["result_path"] = plan["outputs"]["raw_path"]
    else:
        plan["outputs"]["raw_path"] = str(tmp_path / "missing" / "raw")
    with pytest.raises((ValueError, FileExistsError)):
        capture.capture(Path(plan["contract_path"]))
    opener.open.assert_not_called()


def test_redirect_handler_refuses_new_request():
    assert (
        capture.NoRedirect().redirect_request(
            None, None, 302, "", {}, "https://example.test"
        )
        is None
    )

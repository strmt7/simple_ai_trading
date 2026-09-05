import json

import pytest

from tools import check_spot_testnet_credentials_once as probe


@pytest.mark.parametrize("existing", ["journal.jsonl", "result.json"])
def test_one_use_boundary_precedes_secret_input(monkeypatch, tmp_path, existing):
    (tmp_path / existing).write_bytes(b"existing")
    monkeypatch.setattr(probe, "BASE", tmp_path)
    monkeypatch.setattr(probe, "getpass", lambda _: pytest.fail("secret input reached"))
    with pytest.raises(FileExistsError):
        probe.main()
    assert (tmp_path / existing).read_bytes() == b"existing"


@pytest.mark.parametrize("transport_error", [False, True])
def test_only_testnet_get_and_safe_failure_output(
    monkeypatch, tmp_path, capsys, transport_error
):
    marker = "private-input-marker"
    monkeypatch.setattr(probe, "BASE", tmp_path)
    monkeypatch.setattr(probe, "getpass", lambda _: marker)
    calls = []

    class Reply:
        def __init__(self, status, payload):
            self.status_code = status
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert url.startswith("https://testnet.binance.vision/api/v3/")
            assert kwargs["allow_redirects"] is False
            assert kwargs["timeout"] == 15
            calls.append(url.split("?")[0])
            if len(calls) == 1:
                assert "headers" not in kwargs
                return Reply(200, {"serverTime": 1000})
            assert kwargs["headers"]["X-MBX-APIKEY"] == marker
            if transport_error:
                raise RuntimeError(marker + url)
            return Reply(401, {"code": -2015, "msg": marker + url})

    monkeypatch.setattr(probe.requests, "Session", Session)
    probe.main()
    assert calls == [
        probe.HOST + "/api/v3/time",
        probe.HOST + "/api/v3/account/commission",
    ]
    output = capsys.readouterr().out
    for text in [output, *(p.read_text() for p in tmp_path.iterdir())]:
        assert marker not in text
        assert "signature=" not in text
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["authenticated"] is False
    assert result["orders"] == 0
    if not transport_error:
        assert result["exchange_error_code"] == -2015


def test_observed_run_and_frozen_implementation():
    import hashlib

    plan = json.loads((probe.BASE / "plan.json").read_bytes())
    assert (
        hashlib.sha256((probe.ROOT / plan["implementation"]).read_bytes()).hexdigest()
        == plan["implementation_sha256"]
    )
    rows = [
        json.loads(line)
        for line in (probe.BASE / "journal.jsonl").read_bytes().splitlines()
    ]
    assert [row["phase"] for row in rows] == [
        "intent",
        "completed",
        "intent",
        "completed",
    ]
    assert [row["path"] for row in rows] == ["/api/v3/time"] * 2 + [
        "/api/v3/account/commission"
    ] * 2
    result = json.loads((probe.BASE / "result.json").read_bytes())
    assert result["authenticated"] is True
    assert result["http_status"] == 200
    assert result["testnet_only"] is True
    assert result["orders"] == 0
    assert all(rows[-1][k] == v for k, v in result.items())

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import hashlib
import io
import json

import pytest

from tools import capture_binance_triangle_window as m


def books():
    usd = {"USDT": D(1), "BTC": D(100), "ETH": D(10), "SOL": D(1)}
    return [
        {
            "symbol": s,
            "bidPrice": str(usd[b] / usd[q]),
            "askPrice": str(usd[b] / usd[q]),
            "bidQty": "1000",
            "askQty": "1000",
        }
        for s, (b, q) in m.PAIRS.items()
    ]


def test_all_six_directions_and_independent_changed_quotes():
    first = books()
    assert len(m.screen(json.dumps(first).encode())) == 6
    assert all(
        D(x["ideal_zero_fee_bips"]) == 0 for x in m.screen(json.dumps(first).encode())
    )
    # A finite zero-opportunity sample cannot prove every later quote state fails.
    next(x for x in first if x["symbol"] == "ETHBTC")["bidPrice"] = "0.101"
    next(x for x in first if x["symbol"] == "ETHBTC")["askPrice"] = "0.102"
    positive = [
        x
        for x in m.screen(json.dumps(first).encode())
        if D(x["after_three_bip_stress"]) > 0
    ]
    assert len(positive) == 1
    assert positive[0]["cycle"] == "USDT->ETH->BTC->USDT"


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "nan", "zero", "crossed", "float"]
)
def test_bad_books_reject(case):
    rows = books()
    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[0]["bidPrice"] = {
            "nan": "NaN",
            "zero": "0",
            "crossed": "100000",
            "float": 1.0,
        }[case]
    with pytest.raises(ValueError):
        m.screen(json.dumps(rows).encode())


class Response(io.BytesIO):
    def __init__(self, data, status=200, partial_error=False):
        super().__init__(data)
        self.code = status
        self.partial_error = partial_error
        self.reads = 0

    def read(self, n=-1):
        self.reads += 1
        if self.partial_error and self.reads > 1:
            raise TimeoutError("synthetic")
        return super().read(n)


@pytest.mark.parametrize("case", ["ok", "overflow", "partial", "http"])
def test_capture_persists_before_adjudication_and_never_retries(
    tmp_path, monkeypatch, case
):
    data = json.dumps(books()).encode() if case == "ok" else b"x" * 70000
    response = Response(
        data, status=403 if case == "http" else 200, partial_error=case == "partial"
    )
    calls = []

    class Opener:
        def open(self, *args, **kwargs):
            calls.append(1)
            return response

    monkeypatch.setattr(m, "build_opener", lambda *args: Opener())
    raw, journal = tmp_path / "raw.json", tmp_path / "journal.jsonl"
    if case == "ok":
        result, _ = m.capture(raw, m.URL, journal, 0)
        assert result == data
    else:
        with pytest.raises(ValueError):
            m.capture(raw, m.URL, journal, 0)
    assert len(calls) == 1
    receipt = json.loads(journal.read_text().splitlines()[-1])
    assert receipt["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    if case == "partial":
        assert raw.stat().st_size == 4096


def test_wrong_host_and_redirect_are_denied(tmp_path):
    with pytest.raises(ValueError):
        m.capture(tmp_path / "raw", "https://example.com", tmp_path / "journal", 0)
    assert not list(tmp_path.iterdir())
    assert (
        m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")
        is None
    )


def test_fixed_window_runs_once_with_synthetic_clock_and_transport(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monotonic = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(
        m.time, "sleep", lambda x: monotonic.__setitem__(0, monotonic[0] + x)
    )

    class Opener:
        def open(self, *args, **kwargs):
            return Response(json.dumps(books()).encode())

    monkeypatch.setattr(m, "build_opener", lambda *args: Opener())
    now = datetime.now(timezone.utc)
    plan = {
        "url": m.URL,
        "sample_count": 12,
        "interval_seconds": 5,
        "implementation_sha256": hashlib.sha256(
            m.Path(m.__file__).read_bytes()
        ).hexdigest(),
        "not_before": (now - timedelta(minutes=1)).isoformat(),
        "start_deadline": (now + timedelta(minutes=1)).isoformat(),
        "output_directory": ".",
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(plan))
    result = m.run(contract)
    assert result["complete"] and len(result["observations"]) == 12
    assert all(not c["all_twelve_positive"] for c in result["cycles"])
    assert not result["accepted_edge"]
    assert monotonic[0] == 55
    with pytest.raises(ValueError, match="consumed"):
        m.run(contract)

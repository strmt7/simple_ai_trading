"""Focused public-source, time-selection and fixed-base accounting checks."""

from decimal import Decimal
from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from tools import screen_paradex_index_boundaries as screen


def request():
    return {
        "name": "test",
        "asset": "BTC",
        "start": 1001,
        "end": 301000,
        "url": "https://api.prod.paradex.trade/v1/funding/data?market=BTC-USD-PERP",
    }


def row(time=2000, index="10"):
    return {
        "created_at": time,
        "funding_index": index,
        "funding_period_hours": 8,
        "market": "BTC-USD-PERP",
    }


def test_time_only_selection_is_independent_of_order_and_index_value():
    payload = {"next": None, "results": [row(3000, "-200"), row(2000, "900")]}
    assert screen.index_row(payload, request()) == {"time": 2000, "index": "900"}


@pytest.mark.parametrize(
    "payload",
    [
        {"next": "cursor", "results": [row()]},
        {"results": []},
        {"results": [row(), row()]},
        {"results": [row(1000)]},
        {"results": [row(index="NaN")]},
        {"results": [row(index=None)]},
        {"results": [{**row(), "market": "ETH-USD-PERP"}]},
    ],
)
def test_incomplete_or_invalid_window_rejects(payload):
    with pytest.raises(ValueError):
        screen.index_row(payload, request())


def test_json_numeric_period_preserves_exact_decimal():
    payload = json.loads(
        '{"results":[{"created_at":2000,"market":"BTC-USD-PERP","funding_index":"1","funding_period_hours":8.0}]}',
        parse_float=Decimal,
    )
    assert screen.index_row(payload, request())["index"] == "1"


def test_fixed_base_funding_uses_marks_and_separate_quote_units():
    indices = [
        {"time": 1001, "index": "10"},
        {"time": screen.STEP + 1001, "index": "12"},
    ]
    funding = [
        {"time": 0, "rate": "0.01", "mark": "100"},
        {"time": screen.STEP, "rate": "0.01", "mark": "200"},
    ]
    result = screen.economic_rows(indices, funding)[0]
    assert result["paradex_short_cash_usdc_per_base"] == "2"
    assert Decimal(result["binance_long_cash_usdt_per_base"]) == -2
    assert result["reference_mark"] == "100"


def test_settlement_outside_actual_holding_interval_rejects():
    with pytest.raises(ValueError, match="inside actual"):
        screen.economic_rows(
            [{"time": 1, "index": "10"}, {"time": 3, "index": "12"}],
            [
                {"time": 0, "rate": "0.1", "mark": "100"},
                {"time": 4, "rate": "0.1", "mark": "100"},
            ],
        )


def test_role_rejects_negative_half_even_when_aggregate_positive():
    rows = [
        {
            "start": i * screen.STEP,
            "end": (i + 1) * screen.STEP,
            "paradex_short_cash_usdc_per_base": p,
            "binance_long_cash_usdt_per_base": "0",
            "reference_mark": "100",
        }
        for i, p in enumerate(("-1", "5"))
    ]
    result = screen.role_result(rows, sign=1)
    assert Decimal(result["net_after_frozen_hurdles_bips"]) > 0
    assert not result["passes_prefilter"]


def test_binance_complete_grid_and_actual_positive_schedule_jitter():
    payload = [
        {
            "fundingTime": i * screen.STEP + 999,
            "symbol": "BTCUSDT",
            "fundingRate": "0.0001",
            "markPrice": "100",
        }
        for i in range(2)
    ]
    assert len(screen.binance_rows(payload, asset="BTC", start=0, end=screen.STEP)) == 2
    payload[1]["fundingTime"] += 2
    with pytest.raises(ValueError, match="schedule"):
        screen.binance_rows(payload, asset="BTC", start=0, end=screen.STEP)


@pytest.mark.parametrize(
    "status,size,raises", [(200, 2, False), (302, 2, True), (200, 1048578, True)]
)
def test_bounded_transport_retains_terminal_bytes(
    monkeypatch, tmp_path, status, size, raises
):
    body = BytesIO(b"x" * size)
    body.status = status
    monkeypatch.setattr(
        screen, "build_opener", lambda *args: SimpleNamespace(open=lambda *a, **k: body)
    )
    # Real file journal verifies that the fsync contract is exercised offline.
    with (tmp_path / "journal.jsonl").open("w", encoding="utf-8") as journal:
        if raises:
            with pytest.raises(ValueError):
                screen.captured_get(request(), raw=tmp_path / "raw", journal=journal)
        else:
            screen.captured_get(request(), raw=tmp_path / "raw", journal=journal)
    assert (tmp_path / "raw").stat().st_size == min(size, 1048577)
    assert "request_started" in (tmp_path / "journal.jsonl").read_text()


def test_request_population_has_no_duplicate_urls():
    items = screen.requests_for(0, 26 * screen.STEP)
    assert len(items) == 84
    assert len({r["url"] for r in items}) == 84
    assert all("/funding" in r["url"] for r in items)


def test_timeout_retains_partial_body_and_terminal_journal(monkeypatch, tmp_path):
    class PartialResponse:
        status = 200
        reads = 0

        def read(self, size):
            self.reads += 1
            if self.reads == 1:
                return b"partial"
            raise TimeoutError("synthetic socket timeout")

        def close(self):
            pass

    monkeypatch.setattr(
        screen,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *a, **k: PartialResponse()),
    )
    with (tmp_path / "journal.jsonl").open("w", encoding="utf-8") as journal:
        with pytest.raises(TimeoutError):
            screen.captured_get(request(), raw=tmp_path / "raw", journal=journal)
    assert (tmp_path / "raw").read_bytes() == b"partial"
    final = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[-1])
    assert final["phase"] == "request_failed"
    assert final["raw_sha256"] == screen.digest(b"partial")


def test_complete_committed_publication_reconstructs_without_network():
    from tools.verify_paradex_index_publication import verify

    result = verify()
    assert result["verified_raw_responses"] == 84
    assert result["verified_intervals"] == 78
    assert result["raw_bytes"] == 1308306
    assert result["selected_offset_ms_min"] >= 1001
    assert result["selected_offset_ms_max"] <= 301000


def test_primary_docs_explicit_safe_section_dispositions():
    from tools.verify_paradex_index_publication import documentation_dispositions

    assert len(documentation_dispositions()) == 2


def test_source_extraction_excludes_embedded_non_page_material():
    from tools.extract_paradex_funding_sources import extract

    html = b"<script>discard this embedded site material</script><h1>Funding Mechanism</h1><button>Copy page</button><p>Index units</p><p>Was this page helpful?</p>"
    assert (
        extract(html, "Funding Mechanism")
        == b"Funding Mechanism\nCopy page\nIndex units\n"
    )

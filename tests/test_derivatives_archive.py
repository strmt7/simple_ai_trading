from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import pytest

from simple_ai_trading.derivatives_archive import (
    derivatives_archive_file_url,
    ingest_derivatives_archive_url,
    monthly_periods,
)
from simple_ai_trading.market_store import MarketDataStore


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if not self.payload:
            return b""
        if size < 0:
            value, self.payload = self.payload, b""
            return value
        value, self.payload = self.payload[:size], self.payload[size:]
        return value


def _zip_bytes(name: str, text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return output.getvalue()


def _mock_archive(monkeypatch: pytest.MonkeyPatch, url: str, payload: bytes) -> None:
    checksum = hashlib.sha256(payload).hexdigest()

    def fake_urlopen(request: object, timeout: int = 0) -> _Response:
        del timeout
        requested = str(request)
        if requested == f"{url}.CHECKSUM":
            return _Response(f"{checksum}  archive.zip\n".encode("ascii"))
        if requested == url:
            return _Response(payload)
        raise AssertionError(f"unexpected URL: {requested}")

    monkeypatch.setattr("simple_ai_trading.binance_archive.urlopen", fake_urlopen)


def test_derivatives_archive_urls_and_month_sequence_are_exact() -> None:
    assert monthly_periods("2021-12", "2022-02") == [
        "2021-12",
        "2022-01",
        "2022-02",
    ]
    assert derivatives_archive_file_url(
        symbol="btcusdt",
        data_type="premiumIndexKlines",
        interval="1m",
        period="2024-01",
    ).endswith(
        "/futures/um/monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )
    assert derivatives_archive_file_url(
        symbol="ethusdt",
        data_type="fundingRate",
        interval="",
        period="2024-01",
    ).endswith(
        "/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2024-01.zip"
    )
    assert derivatives_archive_file_url(
        symbol="solusdt",
        data_type="markPriceKlines",
        interval="1m",
        period="2024-01",
    ).endswith("/futures/um/monthly/markPriceKlines/SOLUSDT/1m/SOLUSDT-1m-2024-01.zip")


def test_verified_premium_archive_is_stored_separately_from_price_candles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = derivatives_archive_file_url(
        symbol="BTCUSDT",
        data_type="premiumIndexKlines",
        period="2024-01",
    )
    payload = _zip_bytes(
        "BTCUSDT-1m-2024-01.csv",
        "open_time,open,high,low,close,volume,close_time\n"
        "0,0.001,0.002,-0.001,0.0005,0,59999\n"
        "60000,0.0005,0.001,-0.002,-0.001,0,119999\n",
    )
    _mock_archive(monkeypatch, url, payload)
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_derivatives_archive_url(
            store,
            url=url,
            symbol="BTCUSDT",
            data_type="premiumIndexKlines",
            period="2024-01",
        )
        bars = store.fetch_futures_reference_bars("BTCUSDT")
        assert result.status == "complete"
        assert result.checksum_status == "verified"
        assert result.rows_read == 2
        assert len(result.row_stream_sha256) == 64
        assert [item.open_time for item in bars] == [0, 60_000]
        assert [item.close for item in bars] == [0.0005, -0.001]
        assert store.fetch_candles("BTCUSDT", "futures", "1m") == []
        evidence = store.derivatives_archive_files(
            symbol="BTCUSDT", data_type="premiumIndexKlines"
        )
        assert len(evidence) == 1
        assert evidence[0].row_stream_sha256 == result.row_stream_sha256

        skipped = ingest_derivatives_archive_url(
            store,
            url=url,
            symbol="BTCUSDT",
            data_type="premiumIndexKlines",
            period="2024-01",
        )
        assert skipped.status == "skipped"
        assert len(store.fetch_futures_reference_bars("BTCUSDT")) == 2


def test_verified_funding_archive_preserves_actual_historical_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = derivatives_archive_file_url(
        symbol="SOLUSDT",
        data_type="fundingRate",
        interval="",
        period="2024-01",
    )
    payload = _zip_bytes(
        "SOLUSDT-fundingRate-2024-01.csv",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1704067200000,8,0.0001\n"
        "1704081600000,4,-0.0002\n",
    )
    _mock_archive(monkeypatch, url, payload)
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_derivatives_archive_url(
            store,
            url=url,
            symbol="SOLUSDT",
            data_type="fundingRate",
            interval="",
            period="2024-01",
        )
        funding = store.fetch_funding_rates("SOLUSDT")

    assert result.status == "complete"
    assert result.rows_read == 2
    assert [item.funding_interval_hours for item in funding] == [8, 4]
    assert [item.funding_rate for item in funding] == [0.0001, -0.0002]


def test_verified_mark_price_archive_is_stored_as_reference_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = derivatives_archive_file_url(
        symbol="ETHUSDT",
        data_type="markPriceKlines",
        period="2024-01",
    )
    payload = _zip_bytes(
        "ETHUSDT-1m-2024-01.csv",
        "open_time,open,high,low,close,volume,close_time\n"
        "0,2000,2002,1999,2001,0,59999\n"
        "60000,2001,2003,2000,2002,0,119999\n",
    )
    _mock_archive(monkeypatch, url, payload)
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_derivatives_archive_url(
            store,
            url=url,
            symbol="ETHUSDT",
            data_type="markPriceKlines",
            period="2024-01",
        )
        bars = store.fetch_futures_reference_bars("ETHUSDT", kind="mark_price")
        premium = store.fetch_futures_reference_bars("ETHUSDT", kind="premium_index")

    assert result.status == "complete"
    assert result.checksum_status == "verified"
    assert [item.close for item in bars] == [2001.0, 2002.0]
    assert premium == []


def test_invalid_archive_is_fail_closed_without_partial_market_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = derivatives_archive_file_url(
        symbol="ETHUSDT",
        data_type="premiumIndexKlines",
        period="2024-01",
    )
    payload = _zip_bytes(
        "ETHUSDT-1m-2024-01.csv",
        "open_time,open,high,low,close,volume,close_time\n"
        "0,0.001,0.002,-0.001,0.0005,0,59999\n"
        "60000,0.0,-0.1,0.1,0.0,0,119999\n",
    )
    _mock_archive(monkeypatch, url, payload)
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_derivatives_archive_url(
            store,
            url=url,
            symbol="ETHUSDT",
            data_type="premiumIndexKlines",
            period="2024-01",
        )
        bars = store.fetch_futures_reference_bars("ETHUSDT")
        evidence = store.derivatives_archive_files(symbol="ETHUSDT")

    assert result.status == "error"
    assert "OHLC bounds" in result.error
    assert bars == []
    assert evidence[0].status == "error"
    assert evidence[0].rows_inserted == 0

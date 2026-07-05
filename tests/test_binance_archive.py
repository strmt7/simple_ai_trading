from __future__ import annotations

import zipfile
from pathlib import Path

from simple_ai_trading import binance_archive
from simple_ai_trading.binance_archive import (
    archive_directory_url,
    archive_file_url,
    archive_listing_items_by_url,
    archive_listing_url,
    archive_period_in_range,
    archive_url_period,
    filter_archive_urls_by_period,
    ingest_archive_url,
    list_archive_items,
    list_archive_urls,
    validate_archive_period_window,
)
from simple_ai_trading.market_store import MarketDataStore


def test_archive_url_builders_and_listing_parser() -> None:
    assert archive_directory_url(symbol="btcusdc", interval="1s") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/"
    )
    assert archive_file_url(symbol="btcusdc", interval="1s", period="2026-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip"
    )
    assert archive_listing_url(symbol="btcusdc", interval="1s") == (
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=%2F&prefix=data%2Fspot%2Fmonthly%2Fklines%2FBTCUSDC%2F1s%2F"
    )
    assert archive_directory_url(
        symbol="btcusdt",
        interval="1s",
        market_type="futures",
        cadence="daily",
        data_type="aggTrades",
    ) == "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/"
    assert archive_file_url(
        symbol="btcusdt",
        interval="1s",
        period="2024-06-01",
        market_type="futures",
        cadence="daily",
        data_type="aggTrades",
    ) == "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip"
    assert archive_listing_url(
        symbol="btcusdt",
        interval="1s",
        market_type="futures",
        cadence="daily",
        data_type="aggTrades",
    ) == (
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=%2F&prefix=data%2Ffutures%2Fum%2Fdaily%2FaggTrades%2FBTCUSDT%2F"
    )
    html = """<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip</Key><LastModified>2026-02-01T00:00:00.000Z</LastModified><Size>12345</Size></Contents>
      <Contents><Key>data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip.CHECKSUM</Key><Size>100</Size></Contents>
    </ListBucketResult>"""
    seen_listing_urls: list[str] = []
    assert list_archive_urls(symbol="BTCUSDC", interval="1s", html_loader=lambda url: seen_listing_urls.append(url) or html) == [
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip"
    ]
    item = list_archive_items(symbol="BTCUSDC", interval="1s", html_loader=lambda _url: html)[0]
    assert item.period == "2026-01"
    assert item.size_bytes == 12345
    assert item.last_modified == "2026-02-01T00:00:00.000Z"
    assert archive_listing_items_by_url([item.url])[item.url].size_bytes == 12345
    assert seen_listing_urls == [
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=%2F&prefix=data%2Fspot%2Fmonthly%2Fklines%2FBTCUSDC%2F1s%2F"
    ]


def test_archive_period_filtering_supports_daily_and_monthly_windows() -> None:
    urls = [
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-05-31.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-02.zip",
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-07-01.zip",
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-2024-06.zip",
    ]

    assert archive_url_period(urls[1]) == "2024-06-01"
    assert archive_url_period(urls[-1]) == "2024-06"
    assert archive_period_in_range("2024-06-01", start_period="2024-06", end_period="2024-06")
    assert archive_period_in_range("2024-06", start_period="2024-06-15", end_period="2024-06-15")
    assert filter_archive_urls_by_period(urls, start_period="2024-06-01", end_period="2024-06-30") == [
        urls[1],
        urls[2],
        urls[4],
    ]


def test_archive_period_window_validation_rejects_bad_bounds() -> None:
    validate_archive_period_window(start_period="2024-06", end_period="2024-06-30")

    try:
        validate_archive_period_window(start_period="2024/06", end_period=None)
    except ValueError as exc:
        assert "start_period" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("invalid start period should fail")

    try:
        validate_archive_period_window(start_period="2024-07-01", end_period="2024-06-30")
    except ValueError as exc:
        assert "earlier than or equal" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("reversed period window should fail")


def test_ingest_archive_url_streams_zip_into_market_store(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "BTCUSDC-1s-2026-01-01.zip"
    micro_open = 1_767_225_600_000_000
    micro_close = micro_open + 999_000
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "BTCUSDC-1s-2026-01-01.csv",
            "\n".join(
                [
                    "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore",
                    f"{micro_open},100,101,99,100.5,1,{micro_close},100.5,4,0.5,50,0",
                    f"{micro_open + 1_000_000},101,102,100,101.5,2,{micro_close + 1_000_000},203,5,1,101,0",
                ]
            ),
        )

    def fake_download(_url: str, *, timeout: int, chunk_size: int = 1024 * 1024):
        return zip_path, zip_path.stat().st_size, "sha"

    monkeypatch.setattr(binance_archive, "_download_to_temp", fake_download)
    monkeypatch.setattr(binance_archive, "_fetch_archive_checksum", lambda _url, *, timeout: None)

    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_archive_url(
            store,
            url="https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip",
            symbol="btcusdc",
            interval="1s",
            market_type="spot",
            period="2026-01-01",
        )
        candles = store.fetch_candles("BTCUSDC", "spot", "1s")
        archive_rows = store.archive_files(symbol="BTCUSDC", status="complete")

    assert result.status == "complete"
    assert result.rows_read == 2
    assert len(candles) == 2
    assert candles[0].open_time == micro_open // 1000
    assert candles[0].trade_count == 4
    assert archive_rows[0].sha256 == "sha"
    assert archive_rows[0].checksum_status == "unavailable"


def test_ingest_agg_trades_archive_aggregates_real_trades_to_one_second_candles(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "BTCUSDT-aggTrades-2024-06-01.zip"
    base_ts = 1_717_200_000_123
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "BTCUSDT-aggTrades-2024-06-01.csv",
            "\n".join(
                [
                    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
                    f"1,100,0.5,10,12,{base_ts},false",
                    f"2,101,0.25,13,13,{base_ts + 333},true",
                    f"3,99,0.1,14,16,{base_ts + 3000},false",
                ]
            ),
        )

    monkeypatch.setattr(
        binance_archive,
        "_download_to_temp",
        lambda *_args, **_kwargs: (zip_path, zip_path.stat().st_size, "sha"),
    )
    monkeypatch.setattr(binance_archive, "_fetch_archive_checksum", lambda _url, *, timeout: None)

    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_archive_url(
            store,
            url="https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-06-01.zip",
            symbol="btcusdt",
            interval="1s",
            market_type="futures",
            data_type="aggTrades",
            period="2024-06-01",
        )
        candles = store.fetch_candles("BTCUSDT", "futures", "1s")
        sources = [
            row["source"]
            for row in store.connect().execute("SELECT DISTINCT source FROM candles ORDER BY source").fetchall()
        ]

    assert result.status == "complete"
    assert result.data_type == "aggTrades"
    assert result.rows_read == 4
    assert len(candles) == 4
    assert candles[0].open == 100.0
    assert candles[0].high == 101.0
    assert candles[0].low == 100.0
    assert candles[0].close == 101.0
    assert candles[0].volume == 0.75
    assert candles[0].quote_volume == 75.25
    assert candles[0].trade_count == 4
    assert candles[0].taker_buy_base_volume == 0.5
    assert candles[1].open == 101.0
    assert candles[1].volume == 0.0
    assert candles[1].trade_count == 0
    assert candles[2].open == 101.0
    assert candles[2].volume == 0.0
    assert candles[2].trade_count == 0
    assert candles[3].open == 99.0
    assert candles[3].trade_count == 3
    assert sources == ["binance_public_archive_aggTrades"]


def test_ingest_archive_url_rejects_checksum_mismatch_before_writing_rows(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "BTCUSDC-1s-2026-01-01.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "BTCUSDC-1s-2026-01-01.csv",
            "1767225600000,100,101,99,100.5,1,1767225600999,100.5,4,0.5,50,0\n",
        )

    actual = "a" * 64
    expected = "b" * 64
    monkeypatch.setattr(binance_archive, "_download_to_temp", lambda *_args, **_kwargs: (zip_path, zip_path.stat().st_size, actual))
    monkeypatch.setattr(binance_archive, "_fetch_archive_checksum", lambda _url, *, timeout: expected)

    with MarketDataStore(tmp_path / "market.sqlite") as store:
        result = ingest_archive_url(
            store,
            url="https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip",
            symbol="BTCUSDC",
            interval="1s",
            market_type="spot",
            period="2026-01-01",
        )
        candles = store.fetch_candles("BTCUSDC", "spot", "1s")
        archive_rows = store.archive_files(symbol="BTCUSDC")

    assert result.status == "error"
    assert result.checksum_status == "mismatch"
    assert result.checksum_sha256 == expected
    assert "checksum mismatch" in result.error
    assert candles == []
    assert archive_rows[0].status == "error"
    assert archive_rows[0].checksum_status == "mismatch"
    assert archive_rows[0].checksum_sha256 == expected


def test_ingest_archive_url_skips_completed_file(tmp_path) -> None:
    url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip"
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        store.begin_archive_file(url=url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01-01")
        store.complete_archive_file(url=url, status="complete", rows_inserted=2, bytes_downloaded=10, sha256="sha")

        result = ingest_archive_url(store, url=url, symbol="BTCUSDC", interval="1s", market_type="spot")

    assert result.status == "skipped"
    assert result.rows_inserted == 0

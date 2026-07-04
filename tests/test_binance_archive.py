from __future__ import annotations

import zipfile
from pathlib import Path

from simple_ai_trading import binance_archive
from simple_ai_trading.binance_archive import (
    archive_directory_url,
    archive_file_url,
    archive_listing_url,
    ingest_archive_url,
    list_archive_urls,
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
    html = """<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip</Key></Contents>
      <Contents><Key>data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip.CHECKSUM</Key></Contents>
    </ListBucketResult>"""
    seen_listing_urls: list[str] = []
    assert list_archive_urls(symbol="BTCUSDC", interval="1s", html_loader=lambda url: seen_listing_urls.append(url) or html) == [
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01.zip"
    ]
    assert seen_listing_urls == [
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=%2F&prefix=data%2Fspot%2Fmonthly%2Fklines%2FBTCUSDC%2F1s%2F"
    ]


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


def test_ingest_archive_url_skips_completed_file(tmp_path) -> None:
    url = "https://data.binance.vision/data/spot/daily/klines/BTCUSDC/1s/BTCUSDC-1s-2026-01-01.zip"
    with MarketDataStore(tmp_path / "market.sqlite") as store:
        store.begin_archive_file(url=url, symbol="BTCUSDC", market_type="spot", interval="1s", period="2026-01-01")
        store.complete_archive_file(url=url, status="complete", rows_inserted=2, bytes_downloaded=10, sha256="sha")

        result = ingest_archive_url(store, url=url, symbol="BTCUSDC", interval="1s", market_type="spot")

    assert result.status == "skipped"
    assert result.rows_inserted == 0

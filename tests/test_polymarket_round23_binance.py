from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import simple_ai_trading.polymarket_round23_binance as round23
from simple_ai_trading.polymarket_round22_pilot import (
    Round22PilotStore,
    load_round22_pilot_contract,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATE = "2026-04-27"
START_MS = int(datetime(2026, 4, 27, tzinfo=UTC).timestamp() * 1_000)
END_MS = START_MS + 3_600_000


def _zip(filename: str, rows: list[str]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(filename.removesuffix(".zip") + ".csv", "\n".join(rows))
    return buffer.getvalue()


@lru_cache(maxsize=1)
def _spot_archive() -> tuple[dict[str, object], bytes]:
    rows = [
        "open_time,open,high,low,close,volume,close_time,quote_volume,"
        "count,taker_buy_volume,taker_buy_quote_volume,ignore"
    ]
    for offset in range(3_600):
        timestamp_us = (START_MS + offset * 1_000) * 1_000
        price = 90_000 + offset / 10_000
        rows.append(
            f"{timestamp_us},{price},{price + 1},{price - 1},{price + 0.5},"
            f"1,{timestamp_us + 999999},{price},2,0.6,{price * 0.6},0"
        )
    filename = f"BTCUSDT-1s-{DATE}.zip"
    content = _zip(filename, rows)
    return (
        {
            "compressed_bytes": len(content),
            "date": DATE,
            "filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "source": "spot_1s",
            "url": f"https://data.binance.vision/test/{filename}",
        },
        content,
    )


@lru_cache(maxsize=1)
def _futures_archive() -> tuple[dict[str, object], bytes]:
    rows = [
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
        "transact_time,is_buyer_maker"
    ]
    for offset in range(3_600):
        timestamp_ms = START_MS + offset * 1_000 + 100
        price = 90_100 + offset / 10_000
        rows.append(
            f"{offset + 1},{price},1,{offset + 1},{offset + 1},"
            f"{timestamp_ms},{str(bool(offset % 2)).lower()}"
        )
    filename = f"BTCUSDT-aggTrades-{DATE}.zip"
    content = _zip(filename, rows)
    return (
        {
            "compressed_bytes": len(content),
            "date": DATE,
            "filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "source": "futures_aggTrades",
            "url": f"https://data.binance.vision/test/{filename}",
        },
        content,
    )


def _source_contract(archives: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "archives": list(archives),
        "authority": {
            "binance_account_api": False,
            "binance_execution": False,
            "live_trading": False,
            "polymarket_authentication": False,
            "polymarket_execution": False,
            "profitability_claim": False,
        },
        "data_contract": {
            "maximum_compressed_archive_bytes": 32 * 1024 * 1024,
            "maximum_uncompressed_member_bytes": 64 * 1024 * 1024,
        },
        "source_contract_sha256": "e" * 64,
    }


def _seed_window(store: Round22PilotStore) -> None:
    for index in range(12):
        start_ms = START_MS + index * 300_000
        condition_id = "0x" + f"{index + 1:064x}"
        slug = f"btc-updown-5m-{start_ms // 1_000}"
        store.connection.execute(
            """
            INSERT INTO feature.market_identity VALUES (
                ?, ?, 'train', ?, ?, ?, ?, '0.01', '5', '0.25', 2, '0',
                '{}', ?, ?, ?
            )
            """,
            [
                condition_id,
                slug,
                start_ms,
                start_ms + 300_000,
                f"up-{index}",
                f"down-{index}",
                hashlib.sha256(b"{}").hexdigest(),
                "a" * 64,
                start_ms + 600_000,
            ],
        )


class _ArchiveClient:
    def __init__(self, payloads: Mapping[tuple[str, str], bytes]) -> None:
        self.payloads = dict(payloads)
        self.calls: list[tuple[str, str]] = []

    def archive(self, archive: Mapping[str, object], *, maximum_bytes: int) -> bytes:
        identity = (str(archive["source"]), str(archive["date"]))
        self.calls.append(identity)
        payload = self.payloads[identity]
        assert len(payload) <= maximum_bytes
        return payload


class _Response:
    def __init__(self, *, content: bytes, status_code: int, url: str) -> None:
        self.content = content
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.url = url


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self.cookies: list[object] = []
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_round23_source_contract_is_pinned_public_only_and_non_authoritative() -> None:
    contract = round23.load_round23_binance_source_contract(REPOSITORY)

    assert len(contract["archives"]) == 6
    assert set(contract["authority"].values()) == {False}
    assert contract["data_contract"]["archive_persistence"] is False
    assert contract["experimental_limits"]["economic_backtest_allowed"] is False
    assert contract["experimental_limits"]["model_promotion_allowed"] is False


def test_round23_public_client_checks_origin_checksum_bytes_and_authority() -> None:
    archive, content = _spot_archive()
    checksum_url = str(archive["url"]) + ".CHECKSUM"
    checksum = f"{archive['sha256']}  {archive['filename']}".encode("ascii")
    session = _Session(
        [
            _Response(content=b"busy", status_code=503, url=checksum_url),
            _Response(content=checksum, status_code=200, url=checksum_url),
            _Response(content=content, status_code=200, url=str(archive["url"])),
        ]
    )
    sleeps: list[float] = []
    client = round23.Round23BinanceArchiveClient(
        session=session,
        maximum_attempts=2,
        sleeper=sleeps.append,
    )

    assert client.archive(archive, maximum_bytes=len(content)) == content
    assert sleeps == [0.5]
    assert all(call["allow_redirects"] is False for call in session.calls)
    assert all(
        call["headers"]
        == {
            "Accept": "application/zip, text/plain",
            "User-Agent": "simple-ai-trading-round23-public-data/0.1",
        }
        for call in session.calls
    )

    session.headers["X-MBX-APIKEY"] = "redacted"
    with pytest.raises(ValueError, match="contains authority"):
        client.archive(archive, maximum_bytes=len(content))


def test_round23_qualification_parses_both_sources_without_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spot, spot_content = _spot_archive()
    futures, futures_content = _futures_archive()
    contract = _source_contract([spot, futures])
    client = _ArchiveClient(
        {
            ("spot_1s", DATE): spot_content,
            ("futures_aggTrades", DATE): futures_content,
        }
    )
    monkeypatch.setattr(
        round23,
        "load_round23_binance_source_contract",
        lambda _: contract,
    )

    spot_result = round23.qualify_round23_binance_archive(
        repository=REPOSITORY,
        source="spot_1s",
        date=DATE,
        start_ms=START_MS,
        end_ms=END_MS,
        client=client,
    )
    futures_result = round23.qualify_round23_binance_archive(
        repository=REPOSITORY,
        source="futures_aggTrades",
        date=DATE,
        start_ms=START_MS,
        end_ms=END_MS,
        client=client,
    )

    assert spot_result.row_count == futures_result.row_count == 3_600
    assert spot_result.archive_persisted is False
    assert futures_result.binance_execution_authority is False
    assert futures_result.polymarket_execution_authority is False


def test_round23_ingestion_commits_audits_resumes_and_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spot, spot_content = _spot_archive()
    futures, futures_content = _futures_archive()
    contract = _source_contract([spot, futures])
    monkeypatch.setattr(
        round23,
        "load_round23_binance_source_contract",
        lambda _: contract,
    )
    client = _ArchiveClient(
        {
            ("spot_1s", DATE): spot_content,
            ("futures_aggTrades", DATE): futures_content,
        }
    )
    database = tmp_path / "round23.duckdb"
    progress: list[str] = []
    pilot_contract = load_round22_pilot_contract(REPOSITORY)
    with Round22PilotStore(database, contract=pilot_contract) as store:
        _seed_window(store)
        first = round23.ingest_round23_binance_archives(
            store,
            client=client,
            progress=lambda event, _: progress.append(event),
        )
        second = round23.ingest_round23_binance_archives(
            store,
            client=client,
            progress=lambda event, _: progress.append(event),
        )

        assert len(first) == len(second) == 2
        assert all(item.downloaded for item in first)
        assert not any(item.downloaded for item in second)
        assert len(client.calls) == 2
        assert store.connection.execute(
            "SELECT COUNT(*) FROM exploratory.round23_binance_second"
        ).fetchone() == (7_200,)
        assert progress.count("round23_archive_download") == 2
        assert progress.count("round23_archive_committed") == 2
        assert progress.count("round23_archive_reused") == 2
        audits = round23.audit_round23_binance_archives(store)
        assert len(audits) == 2
        assert not any(item.downloaded for item in audits)

        store.connection.execute(
            """
            UPDATE exploratory.round23_binance_second SET close_price = '1'
            WHERE source = 'spot_1s' AND timestamp_ms = ?
            """,
            [START_MS],
        )
        with pytest.raises(ValueError, match="bar economics|stored Binance row"):
            round23.ingest_round23_binance_archives(store, client=client)


def test_round23_parser_rejects_a_missing_second() -> None:
    archive, content = _spot_archive()
    with ZipFile(BytesIO(content)) as source:
        rows = source.read(source.namelist()[0]).decode("ascii").splitlines()
    damaged = _zip(str(archive["filename"]), [*rows[:100], *rows[101:]])

    with pytest.raises(ValueError, match="second coverage"):
        round23._parse_archive(
            damaged,
            archive,
            start_ms=START_MS,
            end_ms=END_MS,
            maximum_uncompressed_bytes=64 * 1024 * 1024,
        )


def test_round23_futures_parser_marks_a_no_trade_second_without_inventing_volume() -> (
    None
):
    archive, content = _futures_archive()
    with ZipFile(BytesIO(content)) as source:
        rows = source.read(source.namelist()[0]).decode("ascii").splitlines()
    no_trade_offset = 100
    retained = [*rows[: no_trade_offset + 1], *rows[no_trade_offset + 2 :]]
    content = _zip(str(archive["filename"]), retained)

    bars = round23._parse_archive(
        content,
        archive,
        start_ms=START_MS,
        end_ms=END_MS,
        maximum_uncompressed_bytes=64 * 1024 * 1024,
    )

    no_trade = bars[no_trade_offset]
    assert no_trade.timestamp_ms == START_MS + no_trade_offset * 1_000
    assert no_trade.trade_count == 0
    assert no_trade.base_volume == "0"
    assert no_trade.quote_volume == "0"
    assert no_trade.signed_quote_volume == "0"
    assert no_trade.open_price == bars[no_trade_offset - 1].close_price
    assert no_trade.close_price == bars[no_trade_offset - 1].close_price

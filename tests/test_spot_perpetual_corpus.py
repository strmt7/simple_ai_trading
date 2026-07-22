from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import zipfile

import pytest

from simple_ai_trading.binance_archive import archive_file_url
from simple_ai_trading.spot_perpetual_corpus import (
    FrozenFlowArchive,
    FrozenFlowDay,
    SpotPerpetualCorpusStore,
    VerifiedFlowSource,
    _fetch_verified_checksum,
    load_frozen_round72_contract,
)
from simple_ai_trading.spot_perpetual_flow import (
    FLOW_MARKET_TYPES,
    FLOW_SYMBOLS,
    SECONDS_PER_DAY,
    aggregate_trade_zip,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "docs/model-research/action-value/round-072-spot-perpetual-price-discovery-design.json"
INVENTORY_PATH = ROOT / "docs/model-research/action-value/round-072-spot-perpetual-inventory.json"
PERIOD = "2024-01-02"
DAY_START_MS = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1_000)


def _verified_day(tmp_path: Path):
    verified = []
    archives = []
    for market_index, market_type in enumerate(FLOW_MARKET_TYPES):
        for symbol_index, symbol in enumerate(FLOW_SYMBOLS):
            url = archive_file_url(
                symbol=symbol,
                interval="1s",
                period=PERIOD,
                market_type=market_type,
                cadence="daily",
                data_type="aggTrades",
            )
            path = tmp_path / Path(url).name
            price = 100.0 + market_index * 10 + symbol_index
            timestamp = DAY_START_MS + (market_index + symbol_index + 1) * 100
            columns = f"1,{price},2,10,11,{timestamp},false"
            if market_type == "spot":
                columns += ",true"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr(path.with_suffix(".csv").name, columns)
            archive = FrozenFlowArchive.from_mapping(
                {
                    "market_type": market_type,
                    "symbol": symbol,
                    "period": PERIOD,
                    "url": url,
                    "expected_bytes": path.stat().st_size,
                    "last_modified": "2024-01-03T00:00:00+00:00",
                    "etag": f"{market_index + 1}{symbol_index + 1}" * 16,
                    "checksum_expected_bytes": 99,
                    "checksum_last_modified": "2024-01-03T00:00:01+00:00",
                    "checksum_etag": f"{symbol_index + 1}{market_index + 1}" * 16,
                }
            )
            flow = aggregate_trade_zip(
                path,
                symbol=symbol,
                market_type=market_type,
                period=PERIOD,
                maximum_uncompressed_bytes=1_000_000,
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archives.append(archive)
            verified.append(
                VerifiedFlowSource(
                    archive=archive,
                    expected_sha256=digest,
                    source_sha256=digest,
                    compressed_bytes=path.stat().st_size,
                    flow=flow,
                )
            )
    day = FrozenFlowDay(
        month="2024-01",
        period=PERIOD,
        selection_digest="a" * 64,
        compressed_bytes=sum(value.expected_bytes for value in archives),
        archives=tuple(archives),
    )
    return day, verified


def test_frozen_round72_contract_recomputes_all_source_totals() -> None:
    contract = load_frozen_round72_contract(DESIGN_PATH, INVENTORY_PATH)

    assert len(contract.days) == 69
    assert contract.expected_files == 414
    assert contract.expected_rows == 17_884_800
    assert contract.selected_compressed_bytes == 5_964_131_852
    assert contract.days[0].period == "2020-10-19"
    assert contract.days[-1].period == "2026-06-22"


def test_spot_perpetual_day_commit_is_atomic_replayable_and_causal(tmp_path) -> None:
    day, verified = _verified_day(tmp_path)
    inventory_sha256 = "b" * 64
    path = tmp_path / "flow.duckdb"

    with SpotPerpetualCorpusStore(
        path,
        cache_root=tmp_path / "cache",
        memory_limit="512MB",
        threads=2,
    ) as store:
        result = store.commit_verified_day(
            day,
            inventory_sha256=inventory_sha256,
            sources=verified,
        )
        certificate = store.certify_day(day, inventory_sha256=inventory_sha256)
        corpus_contract = replace(
            load_frozen_round72_contract(DESIGN_PATH, INVENTORY_PATH),
            inventory_sha256=inventory_sha256,
            selected_compressed_bytes=day.compressed_bytes,
            days=(day,),
        )
        corpus_certificate = store.certify_corpus(corpus_contract)
        row = store.connect().execute(
            """
            SELECT available_time_ms, spot_vwap, perpetual_vwap,
                   spot_taker_imbalance, perpetual_taker_imbalance
            FROM current_spot_perpetual_flow_1s
            WHERE symbol = 'BTCUSDT' AND spot_aggregate_count > 0
            """
        ).fetchone()
        counts = store.connect().execute(
            "SELECT count(*), count(DISTINCT symbol) "
            "FROM current_spot_perpetual_flow_1s"
        ).fetchone()

        assert result.status == "complete"
        assert result.source_count == 6
        assert result.flow_rows == 3 * SECONDS_PER_DAY
        assert certificate["day_id"] == result.day_id
        assert corpus_certificate["day_count"] == 1
        assert corpus_certificate["source_count"] == 6
        assert corpus_certificate["flow_rows"] == 3 * SECONDS_PER_DAY
        assert counts == (3 * SECONDS_PER_DAY, 3)
        assert row[0] == DAY_START_MS + 1_000
        assert row[1:] == pytest.approx((100.0, 110.0, 1.0, 1.0))

        store.connect().execute(
            "DELETE FROM spot_perpetual_flow_1s WHERE day_id = ? AND symbol = 'BTCUSDT' "
            "AND second_ms = ?",
            [result.day_id, DAY_START_MS],
        )
        with pytest.raises(ValueError, match="physical spot/perpetual rows"):
            store.certify_day(day, inventory_sha256=inventory_sha256)


def test_spot_perpetual_commit_rejects_unverified_source_without_writes(tmp_path) -> None:
    day, verified = _verified_day(tmp_path)
    invalid = [*verified]
    invalid[0] = replace(invalid[0], source_sha256="0" * 64)
    path = tmp_path / "rejected.duckdb"

    with SpotPerpetualCorpusStore(
        path,
        cache_root=tmp_path / "cache",
        memory_limit="512MB",
        threads=1,
    ) as store:
        with pytest.raises(ValueError, match="sidecar SHA-256"):
            store.commit_verified_day(
                day,
                inventory_sha256="c" * 64,
                sources=invalid,
            )
        assert store.connect().execute(
            "SELECT count(*) FROM spot_perpetual_flow_day_manifest"
        ).fetchone()[0] == 0
        assert store.connect().execute(
            "SELECT count(*) FROM spot_perpetual_flow_1s"
        ).fetchone()[0] == 0


def test_checksum_sidecar_is_bound_to_frozen_metadata_and_filename() -> None:
    archive = FrozenFlowArchive.from_mapping(
        {
            "market_type": "spot",
            "symbol": "BTCUSDT",
            "period": PERIOD,
            "url": archive_file_url(
                symbol="BTCUSDT",
                interval="1s",
                period=PERIOD,
                market_type="spot",
                cadence="daily",
                data_type="aggTrades",
            ),
            "expected_bytes": 123,
            "last_modified": "2024-01-03T00:00:00+00:00",
            "etag": "a" * 32,
            "checksum_expected_bytes": 99,
            "checksum_last_modified": "2024-01-03T00:00:01+00:00",
            "checksum_etag": "b" * 32,
        }
    )
    digest = "c" * 64

    class Response:
        content = f"{digest}  {Path(archive.url).name}\n".encode("ascii")
        headers = {
            "ETag": f'"{archive.checksum_etag}"',
            "Last-Modified": "Wed, 03 Jan 2024 00:00:01 GMT",
        }

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Session:
        @staticmethod
        def get(url, *, timeout):
            assert url == f"{archive.url}.CHECKSUM"
            assert timeout == 10.0
            return Response()

    assert _fetch_verified_checksum(
        Session(), archive, timeout_seconds=10.0
    ) == digest
    Response.headers["ETag"] = '"different"'
    with pytest.raises(ValueError, match="ETag differs"):
        _fetch_verified_checksum(Session(), archive, timeout_seconds=10.0)

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import zipfile

import pytest

from simple_ai_trading.spot_perpetual_flow import aggregate_trade_zip


DAY_START_MS = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1_000)


def _archive(path: Path, rows: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.with_suffix(".csv").name, "\n".join(rows))
    return path


def test_spot_flow_strictly_aggregates_microsecond_rows_and_causal_gaps(tmp_path) -> None:
    first_us = DAY_START_MS * 1_000 + 100_000
    path = _archive(
        tmp_path / "BTCUSDT-aggTrades-2026-01-01.zip",
        [
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match",
            f"10,100,2,100,101,{first_us},false,true",
            f"11,101,1,102,102,{first_us + 700000},true,false",
            f"12,99,3,103,105,{first_us + 2100000},false,true",
        ],
    )

    flow = aggregate_trade_zip(
        path,
        symbol="BTCUSDT",
        market_type="spot",
        period="2026-01-01",
        maximum_uncompressed_bytes=1_000_000,
        expected_seconds=4,
    )

    assert flow.rows == 4
    assert flow.audit.header_present is True
    assert flow.audit.source_rows == 3
    assert flow.audit.constituent_trade_count == 6
    assert flow.audit.best_match_false_count == 1
    assert flow.aggregate_count.tolist() == [2, 0, 1, 0]
    assert flow.constituent_trade_count.tolist() == [3, 0, 3, 0]
    assert flow.open.tolist() == [100.0, 101.0, 99.0, 99.0]
    assert flow.high.tolist() == [101.0, 101.0, 99.0, 99.0]
    assert flow.close.tolist() == [101.0, 101.0, 99.0, 99.0]
    assert flow.last_trade_age_seconds.tolist() == [0, 1, 0, 1]
    assert flow.quote_volume.tolist() == [301.0, 0.0, 297.0, 0.0]
    assert flow.aggressive_buy_quote.tolist() == [200.0, 0.0, 297.0, 0.0]
    assert flow.aggressive_sell_quote.tolist() == [101.0, 0.0, 0.0, 0.0]
    assert flow.maximum_aggregate_quote.tolist() == [200.0, 0.0, 297.0, 0.0]
    assert flow.squared_aggregate_quote_sum.tolist() == [50_201.0, 0.0, 88_209.0, 0.0]
    assert len(flow.flow_sha256) == 64


def test_futures_flow_accepts_headerless_seven_column_schema(tmp_path) -> None:
    path = _archive(
        tmp_path / "ETHUSDT-aggTrades-2026-01-01.zip",
        [
            f"20,3500,0.2,200,201,{DAY_START_MS + 50},false",
            f"21,3499,0.1,202,202,{DAY_START_MS + 1050},true",
        ],
    )

    flow = aggregate_trade_zip(
        path,
        symbol="ETHUSDT",
        market_type="futures",
        period="2026-01-01",
        maximum_uncompressed_bytes=1_000_000,
        expected_seconds=3,
    )

    assert flow.audit.header_present is False
    assert flow.aggregate_count.tolist() == [1, 1, 0]
    assert flow.quote_volume.tolist() == pytest.approx([700.0, 349.9, 0.0])
    assert flow.aggressive_buy_quote.tolist() == pytest.approx([700.0, 0.0, 0.0])
    assert flow.aggressive_sell_quote.tolist() == pytest.approx([0.0, 349.9, 0.0])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                f"1,100,1,1,1,{DAY_START_MS},false",
                f"1,101,1,2,2,{DAY_START_MS + 1},false",
            ],
            "duplicate or regressing",
        ),
        (
            [
                f"1,100,1,1,2,{DAY_START_MS + 2},false",
                f"2,101,1,2,3,{DAY_START_MS + 3},false",
            ],
            "overlap or regress",
        ),
        (
            [f"1,100,1,1,1,{DAY_START_MS - 1},false"],
            "outside its UTC day",
        ),
    ],
)
def test_flow_parser_fails_closed_on_identity_and_time_corruption(
    tmp_path, rows, message
) -> None:
    path = _archive(tmp_path / "SOLUSDT-aggTrades-2026-01-01.zip", rows)

    with pytest.raises(ValueError, match=message):
        aggregate_trade_zip(
            path,
            symbol="SOLUSDT",
            market_type="futures",
            period="2026-01-01",
            maximum_uncompressed_bytes=1_000_000,
            expected_seconds=4,
        )


def test_flow_parser_rejects_multiple_or_unsafe_zip_members(tmp_path) -> None:
    path = tmp_path / "BTCUSDT-aggTrades-2026-01-01.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../unsafe.csv", f"1,100,1,1,1,{DAY_START_MS},false")
        archive.writestr("extra.txt", "not allowed")

    with pytest.raises(ValueError, match="exactly one CSV"):
        aggregate_trade_zip(
            path,
            symbol="BTCUSDT",
            market_type="futures",
            period="2026-01-01",
            maximum_uncompressed_bytes=1_000_000,
            expected_seconds=4,
        )


def test_flow_parser_rejects_wrong_or_unsafe_single_member_identity(tmp_path) -> None:
    for case, member_name, message in (
        ("wrong", "ETHUSDT-aggTrades-2026-01-01.csv", "identity differs"),
        ("unsafe", "../BTCUSDT-aggTrades-2026-01-01.csv", "path is unsafe"),
    ):
        path = tmp_path / f"{case}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                member_name,
                f"1,100,1,1,1,{DAY_START_MS},false",
            )

        with pytest.raises(ValueError, match=message):
            aggregate_trade_zip(
                path,
                symbol="BTCUSDT",
                market_type="futures",
                period="2026-01-01",
                maximum_uncompressed_bytes=1_000_000,
                expected_seconds=4,
            )

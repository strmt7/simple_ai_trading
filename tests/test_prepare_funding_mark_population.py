from datetime import datetime, timezone
import hashlib
import sqlite3

import pytest

from simple_ai_trading.derivatives_archive import _canonical_row_digest_update
from tools.prepare_funding_mark_population import _archive_rows, _month_bounds


def test_calendar_bounds_are_exact_utc_and_include_leap_and_year_transition():
    for period, start, end in (
        ("2024-02", "2024-02-01", "2024-03-01"),
        ("2025-12", "2025-12-01", "2026-01-01"),
    ):

        def milliseconds(value: str) -> int:
            return int(
                datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
                * 1000
            )

        assert _month_bounds(period) == (milliseconds(start), milliseconds(end) - 1)


def _database(tmp_path):
    path = tmp_path / "warehouse.sqlite"
    start, end = _month_bounds("2025-06")
    rows = [(start, 8, 0.0001), (end, 8, -0.0002)]
    digest = hashlib.sha256()
    for row in rows:
        _canonical_row_digest_update(digest, row)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE funding_rates(symbol TEXT,market_type TEXT,calc_time INTEGER,funding_interval_hours INTEGER,funding_rate REAL)"
        )
        db.execute(
            "CREATE TABLE derivatives_archive_files(symbol TEXT,market_type TEXT,data_type TEXT,period TEXT,url TEXT,status TEXT,rows_read INTEGER,sha256 TEXT,checksum_sha256 TEXT,checksum_status TEXT,row_stream_sha256 TEXT)"
        )
        db.executemany(
            "INSERT INTO funding_rates VALUES('BTCUSDT','futures',?,?,?)", rows
        )
        db.execute(
            "INSERT INTO derivatives_archive_files VALUES('BTCUSDT','futures','fundingRate','2025-06','public-test-source','complete',2,?,?,'verified',?)",
            ("a" * 64, "a" * 64, digest.hexdigest()),
        )
    return path


def test_whole_month_stream_and_metadata_are_verified_without_writing_warehouse(
    tmp_path,
):
    path = _database(tmp_path)
    before = path.read_bytes()
    rows, lineage = _archive_rows(path, ["2025-06"], "BTCUSDT")
    assert len(rows) == 2 and lineage[0]["rows"] == 2
    assert rows[0]["funding_rate_float_repr"] == "0.0001"
    assert rows[1]["funding_rate_float_repr"] == "-0.0002"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "assignment",
    [
        "rows_read=3",
        "status='incomplete'",
        "checksum_status='unverified'",
        "checksum_sha256='changed'",
        "row_stream_sha256='changed'",
    ],
)
def test_unqualified_archive_metadata_or_changed_stream_fails_before_freezing(
    tmp_path, assignment
):
    path = _database(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE derivatives_archive_files SET " + assignment)
    with pytest.raises(ValueError):
        _archive_rows(path, ["2025-06"], "BTCUSDT")


def test_missing_duplicate_metadata_and_changed_data_are_rejected(tmp_path):
    path = _database(tmp_path)
    with pytest.raises(ValueError, match="absent or ambiguous"):
        _archive_rows(path, ["2025-05"], "BTCUSDT")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE funding_rates SET funding_rate=0.001")
    with pytest.raises(ValueError, match="does not match"):
        _archive_rows(path, ["2025-06"], "BTCUSDT")
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO derivatives_archive_files SELECT * FROM derivatives_archive_files"
        )
    with pytest.raises(ValueError, match="absent or ambiguous"):
        _archive_rows(path, ["2025-06"], "BTCUSDT")


@pytest.mark.parametrize(
    "assignment", ["funding_interval_hours=0", "funding_rate=NULL", "calc_time=0"]
)
def test_invalid_rows_or_missing_bounded_rows_fail(tmp_path, assignment):
    path = _database(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE funding_rates SET " + assignment)
    with pytest.raises(ValueError):
        _archive_rows(path, ["2025-06"], "BTCUSDT")

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pytest

from tools.ingest_round61_carry_event_sources import (
    _certificate_payload,
    _parse_filtered_archive,
)
from tools.run_round59_funding_persistence_feasibility import _canonical_sha256


JANUARY_2024_MS = 1_704_067_200_000


def _archive(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "BTCUSDT-1m-2024-01.zip"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2024-01.csv", text)
    path.write_bytes(payload.getvalue())
    return path


def test_filtered_archive_hashes_every_row_but_returns_only_required_rows(
    tmp_path: Path,
) -> None:
    path = _archive(
        tmp_path,
        "open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_base,taker_quote\n"
        f"{JANUARY_2024_MS},100,102,99,101,10,{JANUARY_2024_MS + 59999},1000,4,6,610\n"
        f"{JANUARY_2024_MS + 60000},101,103,100,102,12,{JANUARY_2024_MS + 119999},1220,5,7,720\n"
        f"{JANUARY_2024_MS + 120000},102,104,101,103,9,{JANUARY_2024_MS + 179999},930,3,4,415\n",
    )

    selected, evidence = _parse_filtered_archive(
        path,
        period="2024-01",
        required_times=[JANUARY_2024_MS, JANUARY_2024_MS + 120000],
        mark_price=False,
    )

    assert [row.open_time for row in selected] == [
        JANUARY_2024_MS,
        JANUARY_2024_MS + 120000,
    ]
    assert evidence["full_rows"] == 3
    assert evidence["first_open_time_ms"] == JANUARY_2024_MS
    assert evidence["selected_rows"] == 2
    assert evidence["gap_count"] == 0
    assert len(evidence["full_row_stream_sha256"]) == 64
    assert len(evidence["selected_row_stream_sha256"]) == 64


def test_filtered_archive_fails_when_a_required_minute_is_absent(
    tmp_path: Path,
) -> None:
    path = _archive(
        tmp_path,
        f"{JANUARY_2024_MS},100,102,99,101,10,{JANUARY_2024_MS + 59999}\n",
    )

    with pytest.raises(ValueError, match="missing 1 required rows"):
        _parse_filtered_archive(
            path,
            period="2024-01",
            required_times=[JANUARY_2024_MS + 60000],
            mark_price=True,
        )


def test_source_certificate_hash_covers_completion_and_evidence() -> None:
    payload = _certificate_payload(
        implementation_commit="a" * 40,
        database_file="market_data.sqlite",
        complete=True,
        archive_evidence=[{"kind": "spot", "archive_sha256": "b" * 64}],
        futures_evidence=[],
        series_evidence=[],
    )
    canonical = dict(payload)
    claimed = canonical.pop("source_certificate_sha256")

    assert claimed == _canonical_sha256(canonical)
    changed = dict(canonical)
    changed["complete"] = False
    assert _canonical_sha256(changed) != claimed

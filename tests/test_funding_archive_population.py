"""Exact independent-channel population checks; no network or profitability claims."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from tools import qualify_funding_archive_population as qualifier


FILENAME = "BTCUSDT-fundingRate-2026-09.zip"
HEADER = "calc_time,funding_interval_hours,last_funding_rate\n"


def bundle(text: str, *, member: str | None = None) -> tuple[bytes, bytes]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(member or FILENAME[:-4] + ".csv", text)
    raw = stream.getvalue()
    return raw, f"{hashlib.sha256(raw).hexdigest()}  {FILENAME}\n".encode()


def test_archive_retains_exact_decimal_and_millisecond_clock():
    raw, checksum = bundle(HEADER + "1788566400003,8,-1.23e-5\n")
    assert qualifier.archive_events(raw, checksum, FILENAME) == [
        (1788566400003, Fraction(-123, 10_000_000))
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        HEADER,
        "wrong,header,fields\n1,8,0.01\n",
        HEADER + "1,8,0.01\n1,8,0.02\n",
        HEADER + "2,8,0.01\n1,8,0.02\n",
        HEADER + "1,0,0.01\n",
        HEADER + "1,8,nan\n",
        HEADER + "1,8,1/2\n",
        HEADER + "1,8,1/0\n",
        HEADER + "1,8,0.11\n",
        HEADER + "1,8,0.01,extra\n",
    ],
)
def test_invalid_archive_rows_reject(text):
    raw, checksum = bundle(text)
    with pytest.raises(ValueError):
        qualifier.archive_events(raw, checksum, FILENAME)


def test_checksum_member_and_crc_boundaries():
    raw, checksum = bundle(HEADER + "1,8,0.01\n")
    for invalid in (
        b"",
        checksum.replace(FILENAME.encode(), b"other.zip"),
        b"0" * 64 + b"  " + FILENAME.encode(),
    ):
        with pytest.raises(ValueError):
            qualifier.archive_events(raw, invalid, FILENAME)
    raw, checksum = bundle(HEADER + "1,8,0.01\n", member="../wrong.csv")
    with pytest.raises(ValueError):
        qualifier.archive_events(raw, checksum, FILENAME)


def fixture_plan(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(qualifier, "_root_path", lambda value: tmp_path / value)
    (tmp_path / "docs").mkdir()
    boundary = tmp_path / "docs/RESEARCH_CAPTURE_BOUNDARIES.md"
    boundary.write_bytes(b"fixture boundary")
    plan = {
        "schema": "funding-archive-population-contract-v1",
        "contract_path": "contract.json",
        "period": "2026-09",
        "frozen_at_utc": "2026-09-01T00:00:00Z",
        "archive_start_ms": 1788220800000,
        "archive_end_ms": 1790812800000,
        "intersection_start_ms": 1788566400000,
        "intersection_end_ms": 1790812800000,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "max_requests": 6,
        "expected_intersection_events_per_symbol": 1,
        "implementations": [],
        "capture_boundaries_sha256": hashlib.sha256(boundary.read_bytes()).hexdigest(),
        "native_inputs": {},
        "output_directory": "outputs",
    }
    (tmp_path / "outputs").mkdir()
    for symbol in plan["symbols"]:
        raw = json.dumps(
            [
                {
                    "symbol": symbol,
                    "fundingTime": 1788566400003,
                    "fundingRate": "0.0001",
                    "markPrice": "100",
                }
            ]
        ).encode()
        (tmp_path / (symbol + ".raw")).write_bytes(raw)
        plan["native_inputs"][symbol] = [
            {"path": symbol + ".raw", "sha256": hashlib.sha256(raw).hexdigest()}
        ]
    certificate = {
        "status": "all_four_native_value_gates_passed_not_cash_label_admission",
        "inputs": [
            {"raw_path": item["path"], "raw_sha256": item["sha256"]}
            for inputs in plan["native_inputs"].values()
            for item in inputs
        ],
    }
    certificate["result_sha256"] = qualifier._canonical_hash(
        certificate, "result_sha256"
    )
    (tmp_path / "certificate.json").write_text(json.dumps(certificate))
    plan["native_certificate"] = {
        "path": "certificate.json",
        "result_sha256": certificate["result_sha256"],
    }
    plan["contract_sha256"] = qualifier._canonical_hash(plan, "contract_sha256")
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(plan))
    return path


def test_preflight_is_network_free_and_rejects_consumed_outputs(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualifier, "capture_bytes", lambda *args: pytest.fail("network")
    )
    assert qualifier.run(path, preflight=True) is None
    (tmp_path / "outputs/journal.jsonl").write_text("consumed")
    with pytest.raises(FileExistsError):
        qualifier.run(path, preflight=True)


def test_first_transport_failure_is_terminal_and_restart_cannot_repeat(
    tmp_path, monkeypatch
):
    path = fixture_plan(tmp_path, monkeypatch)
    calls = []

    def failure(url, *args):
        calls.append(url)
        raise ValueError("fixture unavailable")

    monkeypatch.setattr(qualifier, "capture_bytes", failure)
    result = qualifier.run(path)
    assert len(calls) == 1
    assert not result["qualified_intersection"]
    assert result["error_type"] == "ValueError"
    phases = [
        json.loads(line)["phase"]
        for line in (tmp_path / "outputs/journal.jsonl").read_text().splitlines()
    ]
    assert phases == ["study_intent", "study_completed"]
    with pytest.raises(FileExistsError):
        qualifier.run(path)


@pytest.mark.parametrize(
    "clock,rate,qualified",
    [
        (1788566400003, "0.00010000", True),
        (1788566400000, "0.00010000", False),
        (1788566400003, "0.00010001", False),
    ],
)
def test_complete_population_needs_exact_rates_and_clocks(
    tmp_path, monkeypatch, clock, rate, qualified
):
    path = fixture_plan(tmp_path, monkeypatch)

    def capture(url, output, journal, ceiling):
        symbol = url.split("/")[-2]
        filename = f"{symbol}-fundingRate-2026-09.zip"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr(filename[:-4] + ".csv", HEADER + f"{clock},8,{rate}\n")
        raw = stream.getvalue()
        return (
            f"{hashlib.sha256(raw).hexdigest()}  {filename}\n".encode()
            if url.endswith(".CHECKSUM")
            else raw
        )

    monkeypatch.setattr(qualifier, "capture_bytes", capture)
    result = qualifier.run(path)
    assert result["qualified_intersection"] is qualified
    assert len(result["symbols"]) == (3 if qualified else 0)
    assert not result["accepted_edge"] and not result["training_admitted"]


@pytest.mark.parametrize(
    "status,payload,ceiling,passed",
    [
        (200, b"payload", 10, True),
        (404, b"absent", 10, False),
        (200, b"oversized", 3, False),
        (200, b"", 10, False),
    ],
)
def test_transport_retains_receipts_and_bounded_raw(
    tmp_path, monkeypatch, status, payload, ceiling, passed
):
    class Response(io.BytesIO):
        code = status

    class Opener:
        def open(self, request, timeout):
            return Response(payload)

    monkeypatch.setattr(qualifier, "build_opener", lambda *args: Opener())
    raw_path = tmp_path / "response.raw"
    with (tmp_path / "requests.jsonl").open("w") as journal:
        if passed:
            assert (
                qualifier.capture_bytes(
                    "https://data.binance.vision/fixture", raw_path, journal, ceiling
                )
                == payload
            )
        else:
            with pytest.raises(ValueError):
                qualifier.capture_bytes(
                    "https://data.binance.vision/fixture", raw_path, journal, ceiling
                )
    assert raw_path.read_bytes() == payload[: ceiling + 1]
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows] == ["request_intent", "request_completed"]
    assert rows[-1]["status_code"] == status


def test_tamper_fails_before_any_request(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        qualifier, "capture_bytes", lambda *args: pytest.fail("network")
    )
    (tmp_path / "BTCUSDT.raw").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        qualifier.run(path)
    assert not (tmp_path / "outputs/journal.jsonl").exists()


def test_cli_preflight_reports_only_status(tmp_path, monkeypatch, capsys):
    path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["qualifier", "--contract", str(path), "--preflight"]
    )
    qualifier.main()
    assert json.loads(capsys.readouterr().out) == {
        "preflight": True,
        "qualified_intersection": None,
    }

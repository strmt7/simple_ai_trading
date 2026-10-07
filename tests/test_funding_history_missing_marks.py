"""Missing native marks stay unavailable; retained failures are not repair inputs."""

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.funding_cash import parse_binance_usdm_funding_history
from tools.adjudicate_funding_archive_clock import write_once


ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / "docs/review/2026-10-07/funding-mark-population"


def _parse(raw: bytes):
    return parse_binance_usdm_funding_history(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_symbol="BTCUSDT"
    )


@pytest.mark.parametrize("mark", ["", " ", None, "0", "0.00000000"])
def test_missing_or_nonpositive_mark_cannot_be_imputed(mark):
    raw = json.dumps(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": 10,
                "fundingRate": "0.0001",
                "markPrice": mark,
            }
        ]
    ).encode()
    with pytest.raises(ValueError):
        _parse(raw)


def test_positive_mark_control_is_not_a_coverage_or_profitability_claim():
    raw = b'[{"symbol":"BTCUSDT","fundingTime":10,"fundingRate":"0.0001","markPrice":"100"}]'
    parsed = _parse(raw)
    assert len(parsed) == 1
    assert parsed[0].settlement_mark == 100


def test_retained_page_preserves_all_empty_marks_and_terminal_failure():
    raw = (FAILED / "01-btcusdt.raw").read_bytes()
    assert len(raw) == 112201
    assert hashlib.sha256(raw).hexdigest() == (
        "a07f1354000f4cb80975d54d29072b8a59603b5cc33dfd67d5f33e611dfbf92c"
    )
    rows = json.loads(raw)
    assert len(rows) == 1000
    assert all(row["markPrice"] == "" for row in rows)
    with pytest.raises(ValueError):
        _parse(raw)
    result = json.loads((FAILED / "01-btcusdt-comparison.json").read_bytes())
    assert result["status"] == "adjudication_failed_closed"
    assert not result["label_admission_qualified"]
    assert not result["accepted_edge"]


def test_consumed_adjudication_refuses_reexecution_without_changing_evidence():
    paths = (
        FAILED / "01-btcusdt-comparison.json",
        FAILED / "01-btcusdt-comparison-journal.jsonl",
    )
    before = tuple(path.read_bytes() for path in paths)
    with pytest.raises(FileExistsError):
        write_once(FAILED / "01-btcusdt-contract.json")
    assert tuple(path.read_bytes() for path in paths) == before

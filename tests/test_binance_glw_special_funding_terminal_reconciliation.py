from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.reconcile_binance_glw_special_funding_history import (  # noqa: E402
    _canonical_hash,
    _status,
    run,
)


ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "binance-glw-special-funding-terminal-reconciliation-contract-v2-2026-08-30.json"
)
RESULT = ACTION / (
    "binance-glw-special-funding-terminal-reconciliation-result-v2-2026-08-31.json"
)
IMPLEMENTATION = ROOT / "tools/reconcile_binance_glw_special_funding_history.py"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RESULT_HASH = "315b2ba2a4f30caba2a7be1181dff3a8d93bc0e28adb2ad9738623a90342bd4b"
RAW_HASH = "60f998c7d593cfdad2ccbd17f911f849ca0b92f0e72a16cad13b7004099908d5"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _row(*, time_ms: int, rate: str, debit: str, rate_type: str) -> dict[str, object]:
    return {
        "symbol": "GLWUSDT",
        "funding_time_ms": time_ms,
        "funding_time_utc": "2026-08-31T00:00:00Z",
        "funding_rate": rate,
        "mark_price": "100",
        "rate_type": rate_type,
        "per_unit_debit_usdt": debit,
    }


def test_contract_freezes_one_post_snapshot_history_delta_only() -> None:
    contract = _load(CONTRACT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert contract["not_before_utc"] == "2026-08-31T00:10:00Z"
    assert contract["request"]["request_budget"] == 1
    assert contract["request"]["retry_permitted"] is False
    assert contract["request"]["pagination_permitted"] is False
    assert contract["request"]["start_time_ms"] == 1787990400003
    assert contract["request"]["end_time_ms"] == 1788134460000
    assert contract["decision"]["bstock_snapshot_time_ms"] == 1788134400000
    assert contract["decision"]["gross_dividend_per_share_usd"] == "0.28"
    assert contract["decision"]["book_rule"].startswith("No book")
    assert contract["authority"]["account_requests"] == 0
    assert contract["authority"]["orders_or_transactions"] == 0
    assert contract["authority"]["book_or_price_requests"] == 0
    assert RESULT.exists()
    assert (
        hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
        == contract["implementation"]["sha256"]
    )
    for source in contract["retained_sources"]:
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )


def test_runner_rejects_before_not_before_without_changing_outputs() -> None:
    contract = _load(CONTRACT)
    before = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in contract["outputs"].values()
    }
    with pytest.raises(RuntimeError, match="not-before gate is not satisfied"):
        run(CONTRACT, now=datetime(2026, 8, 31, 0, 9, 59, tzinfo=UTC))

    after = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in contract["outputs"].values()
    }
    assert after == before


def test_terminal_result_raw_and_journal_reconstruct() -> None:
    result = _load(RESULT)
    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["capture"]["request_count"] == 1
    assert result["capture"]["response_row_count"] == 10
    assert result["history"]["negative_special_row_count"] == 1
    special = result["history"]["negative_special_rows"][0]
    assert special["funding_time_ms"] == 1788134401003
    assert special["timing_relative_to_bstock_snapshot"] == "at_or_after"
    assert special["per_unit_debit_usdt"] == "0.2799998930000000"
    assert special["matches_gross_dividend_tolerance"] is True
    assert result["adjudication"]["status"] == (
        "terminal_matching_special_row_at_or_after_snapshot_no_pre_snapshot_gap"
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["book_capture_permitted"] is False
    raw_path = ROOT / result["capture"]["receipt"]["raw_path"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == RAW_HASH
    journal = ROOT / "data/binance-glw-special-funding-terminal-reconciliation-v2/request-journal.jsonl"
    entries = [json.loads(line) for line in journal.read_text(encoding="ascii").splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[1]["response_sha256"] == RAW_HASH


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], "terminal_no_new_history_rows"),
        (
            [
                _row(
                    time_ms=1788134399999,
                    rate="-0.0028",
                    debit="0.28",
                    rate_type="Special",
                )
            ],
            "matching_pre_snapshot_special_row_observed_mechanism_only",
        ),
        (
            [
                _row(
                    time_ms=1788134400000,
                    rate="-0.0028",
                    debit="0.28",
                    rate_type="Special",
                )
            ],
            "terminal_matching_special_row_at_or_after_snapshot_no_pre_snapshot_gap",
        ),
        (
            [
                _row(
                    time_ms=1788134399999,
                    rate="-0.001",
                    debit="0.10",
                    rate_type="Special",
                )
            ],
            "terminal_negative_special_row_magnitude_mismatch",
        ),
        (
            [_row(time_ms=1788134399999, rate="0", debit="0", rate_type="Regular")],
            "terminal_no_negative_special_funding_row",
        ),
    ],
)
def test_terminal_status_is_deterministic(
    rows: list[dict[str, object]], expected: str
) -> None:
    status, special = _status(
        rows=rows,
        snapshot_ms=1788134400000,
        gross_dividend=Decimal("0.28"),
        tolerance=Decimal("0.000001"),
    )

    assert status == expected
    if special:
        assert special[0]["matches_gross_dividend_tolerance"] is (
            Decimal(special[0]["per_unit_debit_usdt"]) == Decimal("0.28")
        )


def test_registry_terminalizes_consumed_reconciliation() -> None:
    registry = _load(REGISTRY)
    contract = _load(CONTRACT)
    rank_34 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 34
    )

    assert any(
        artifact["path"] == CONTRACT.relative_to(ROOT).as_posix()
        and artifact["result_sha256"] == contract["contract_sha256"]
        for artifact in rank_34["canonical_artifacts"]
    )
    assert any(
        artifact["path"] == RESULT.relative_to(ROOT).as_posix()
        and artifact["result_sha256"] == RESULT_HASH
        for artifact in rank_34["canonical_artifacts"]
    )
    assert "terminally_rejected_for_the_2026_GLW_episode" in rank_34["current_status"]
    assert "do_not_repeat_retry_paginate_alias_extend_or_repair" in (
        rank_34["next_action"]
    )
    assert "future_independent_weekend_or_holiday" in rank_34["retry_trigger"]
    assert any(
        item["canonical_result_sha256"] == RESULT_HASH
        for item in registry["terminal_do_not_repeat"]
    )

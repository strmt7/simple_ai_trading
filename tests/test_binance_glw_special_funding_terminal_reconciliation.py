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
    assert not RESULT.exists()
    assert (
        hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
        == contract["implementation"]["sha256"]
    )
    for source in contract["retained_sources"]:
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )


def test_runner_rejects_before_not_before_without_creating_outputs() -> None:
    with pytest.raises(RuntimeError, match="not-before gate is not satisfied"):
        run(CONTRACT, now=datetime(2026, 8, 31, 0, 9, 59, tzinfo=UTC))

    contract = _load(CONTRACT)
    assert not any((ROOT / path).exists() for path in contract["outputs"].values())


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


def test_registry_routes_only_the_frozen_terminal_reconciliation() -> None:
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
    assert "2026_08_31T00_10_00Z" in rank_34["next_action"]
    assert "no_2026_GLW_book_capture" in rank_34["next_action"]

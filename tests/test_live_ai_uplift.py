from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from simple_ai_trading import cli
from simple_ai_trading.live_ai_assist import (
    LIVE_AI_ENTRY_AUDIT_SCHEMA_VERSION,
    LIVE_AI_ENTRY_CASE_SCHEMA_VERSION,
    LiveAIEntryDecision,
    build_live_ai_entry_case,
    load_live_ai_entry_audit,
)
from simple_ai_trading.live_ai_uplift import (
    assess_live_ai_shadow_uplift,
    load_one_second_trade_paths,
)
from simple_ai_trading.positions import ClosedTrade


_MODEL_DIGEST = "a" * 64
_TERMINAL_FINGERPRINT = "b" * 64
_DAY_MS = 86_400_000


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _case(observed_at_ms: int):
    return build_live_ai_entry_case(
        symbol="BTCUSDT",
        market_type="futures",
        interval="15m",
        observed_at_ms=observed_at_ms,
        proposed_side="LONG",
        ml_confidence=0.7,
        maximum_risk_multiplier=0.4,
        model_digest=_MODEL_DIGEST,
        terminal_model_fingerprint=_TERMINAL_FINGERPRINT,
        evidence={"risk": {"regime": "trend"}},
    )


def _decision(action: str) -> LiveAIEntryDecision:
    approved = action == "approve"
    return LiveAIEntryDecision(
        action=action,
        risk_multiplier=0.4 if approved else 0.0,
        confidence=0.8,
        reason_codes=("edge_after_costs",) if approved else ("drawdown_risk",),
        summary="Bounded synthetic reviewer evidence.",
        valid=True,
        response_sha256="c" * 64,
        observed_model_digest=_MODEL_DIGEST,
        model_residency_status="gpu_resident",
        prompt_tokens=100,
        output_tokens=20,
    )


def _record(case, decision, *, completed_at_ms: int, previous: str) -> dict[str, object]:
    unsigned = {
        "schema_version": LIVE_AI_ENTRY_AUDIT_SCHEMA_VERSION,
        "previous_record_sha256": previous,
        "completed_at_ms": completed_at_ms,
        "latency_seconds": 0.5,
        "case": case.identity_payload() | {"case_id": case.case_id},
        "decision": decision.asdict(),
        "mode": "shadow_only",
        "trading_authority": False,
    }
    return unsigned | {"record_sha256": _canonical_sha256(unsigned)}


def _trade(case, *, opened_at_ms: int, pnl: float, action: str, index: int) -> ClosedTrade:
    return ClosedTrade(
        id=f"trade-{index:03d}",
        symbol="BTCUSDT",
        market_type="futures",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        leverage=1.0,
        opened_at_ms=opened_at_ms,
        closed_at_ms=opened_at_ms + 1_000,
        realized_pnl=pnl,
        realized_pnl_pct=pnl / 100.0,
        fees=0.0,
        reason="model-exit",
        ai_review_mode="shadow_only",
        ai_review_case_id=case.case_id,
        ai_review_status={
            "approve": "shadow_approve",
            "veto": "shadow_veto",
            "cooldown": "shadow_cooldown",
        }[action],
    )


def _one_second_path(trade: ClosedTrade) -> tuple[dict[str, object], ...]:
    start_ms = trade.opened_at_ms // 1_000 * 1_000
    end_ms = trade.closed_at_ms // 1_000 * 1_000
    low = min(trade.entry_price, trade.exit_price)
    high = max(trade.entry_price, trade.exit_price)
    return tuple(
        {
            "timestamp_ms": timestamp_ms,
            "high": high,
            "low": low,
            "source": "synthetic-unit-test-only",
        }
        for timestamp_ms in range(start_ms, end_ms + 1_000, 1_000)
    )


def test_semantic_audit_loader_rejects_rehashed_risk_cap_violation(
    tmp_path: Path,
) -> None:
    case = _case(1_000)
    record = _record(case, _decision("approve"), completed_at_ms=2_000, previous="0" * 64)
    record["decision"]["risk_multiplier"] = 0.5
    unsigned = dict(record)
    unsigned.pop("record_sha256")
    record["record_sha256"] = _canonical_sha256(unsigned)
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="risk bound"):
        load_live_ai_entry_audit(path)


def test_materializer_rejects_review_completed_after_entry() -> None:
    case = _case(1_000)
    record = _record(case, _decision("veto"), completed_at_ms=4_000, previous="0" * 64)
    trade = _trade(case, opened_at_ms=3_000, pnl=-1.0, action="veto", index=1)
    report = assess_live_ai_shadow_uplift(
        [trade],
        [record],
        initial_capital=1_000.0,
        model_name="qwen3:14b",
        model_parameters_b=14.0,
    )

    assert report["accepted"] is False
    assert report["causally_eligible_trades"] == 0
    assert report["rejection_counts"] == {"review_not_causally_available": 1}
    assert report["trading_authority"] is False
    assert report["profitability_claim"] is False


def test_materializer_rejects_realized_only_drawdown_evidence() -> None:
    case = _case(1_000)
    record = _record(case, _decision("approve"), completed_at_ms=2_000, previous="0" * 64)
    trade = _trade(case, opened_at_ms=3_000, pnl=1.0, action="approve", index=1)
    report = assess_live_ai_shadow_uplift(
        [trade],
        [record],
        initial_capital=1_000.0,
        model_name="qwen3:14b",
        model_parameters_b=14.0,
    )

    assert report["accepted"] is False
    assert report["intratrade_path_evidence"]["verified"] is False
    assert "intratrade_path_missing" in report["reasons"]
    assert "intratrade_path_risk_not_verified" in report["reasons"]


def test_materializer_builds_bound_daily_pairs_and_can_clear_existing_gate(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []
    trades: list[ClosedTrade] = []
    previous = "0" * 64
    start = 1_735_689_600_000
    for index in range(41):
        observed = start + index * 3 * _DAY_MS + 1_000
        completed = observed + 1_000
        opened = completed + 1_000
        is_loss = index < 31
        action = "approve" if index == 30 or not is_loss else "veto"
        pnl = -1.0 if is_loss else 2.0
        case = _case(observed)
        record = _record(
            case,
            _decision(action),
            completed_at_ms=completed,
            previous=previous,
        )
        records.append(record)
        previous = str(record["record_sha256"])
        trades.append(
            _trade(
                case,
                opened_at_ms=opened,
                pnl=pnl,
                action=action,
                index=index,
            )
        )
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    verified_records = load_live_ai_entry_audit(path)
    intratrade_paths = {
        trade.id: _one_second_path(trade)
        for trade in trades
    }
    report = assess_live_ai_shadow_uplift(
        trades,
        verified_records,
        initial_capital=1_000.0,
        model_name="qwen3:14b",
        intratrade_paths=intratrade_paths,
        model_parameters_b=14.0,
    )

    assert report["candidate_trades"] == 41
    assert report["causally_eligible_trades"] == 41
    assert report["causal_coverage"] == 1.0
    assert len(report["matched_periods"]) >= 90
    assert report["uplift"]["baseline"]["realized_pnl"] == pytest.approx(-11.0)
    assert report["uplift"]["ai"]["realized_pnl"] == pytest.approx(19.0)
    assert report["intratrade_path_evidence"]["verified"] is True
    assert report["accepted"] is True
    assert len(report["report_sha256"]) == 64


def test_case_schema_constant_remains_bound_in_audit_fixture() -> None:
    assert _case(1_000).identity_payload()["schema_version"] == (
        LIVE_AI_ENTRY_CASE_SCHEMA_VERSION
    )


def test_one_second_path_loader_is_exact_and_read_only(tmp_path: Path) -> None:
    case = _case(1_000)
    trade = _trade(case, opened_at_ms=3_000, pnl=1.0, action="approve", index=1)
    database = tmp_path / "market.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE candles (
                symbol TEXT NOT NULL,
                market_type TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO candles VALUES (?, ?, '1s', ?, ?, ?, ?)",
            [
                ("BTCUSDT", "futures", 3_000, 100.5, 99.5, "verified-source"),
                ("BTCUSDT", "futures", 4_000, 101.5, 100.5, "verified-source"),
            ],
        )
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    paths = load_one_second_trade_paths(database, [trade])

    assert paths[trade.id] == (
        {
            "timestamp_ms": 3_000,
            "high": 100.5,
            "low": 99.5,
            "source": "verified-source",
        },
        {
            "timestamp_ms": 4_000,
            "high": 101.5,
            "low": 100.5,
            "source": "verified-source",
        },
    )
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_cli_writes_the_same_causal_uplift_report_used_by_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "accepted": False,
        "causal_coverage": 0.95,
        "causally_eligible_trades": 38,
        "candidate_trades": 40,
        "matched_periods": [{} for _ in range(12)],
        "reasons": ["paired_observations<30"],
        "uplift": {},
        "trading_authority": False,
        "profitability_claim": False,
    }

    def fake_assessment(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "positions_root": tmp_path / "positions",
            "audit_path": tmp_path / "reviews.jsonl",
            "market_db": tmp_path / "market.sqlite",
            "initial_capital": 10_000.0,
            "model_name": "qwen3:14b",
            "model_parameters_b": 14.0,
        }
        return expected

    monkeypatch.setattr(
        "simple_ai_trading.live_ai_uplift.assess_live_ai_shadow_uplift_paths",
        fake_assessment,
    )
    output = tmp_path / "uplift.json"
    result = cli.command_ai_uplift(
        argparse.Namespace(
            positions_root=str(tmp_path / "positions"),
            audit=str(tmp_path / "reviews.jsonl"),
            market_db=str(tmp_path / "market.sqlite"),
            starting_capital=10_000.0,
            model="qwen3:14b",
            model_parameters_b=14.0,
            output=str(output),
            json=False,
        )
    )

    assert result == 2
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    rendered = capsys.readouterr().out
    assert "coverage=95.0% eligible=38/40 paired_days=12" in rendered
    assert "trading_authority=false; profitability_claim=false" in rendered

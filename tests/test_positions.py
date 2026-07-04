"""Branch-coverage tests for the positions module."""

from __future__ import annotations

import json

import pytest

from simple_ai_trading.positions import (
    BOT_CLIENT_ORDER_PREFIX,
    ClosedTrade,
    LedgerStats,
    OpenPosition,
    PositionsStore,
    bot_client_order_id,
    build_learning_feedback,
    compute_stats,
    is_bot_owned_position,
    load_learning_feedback,
    new_position_id,
    now_ms,
    render_learning_feedback,
    render_positions_table,
    render_stats_lines,
    unrealized_pnl_pct,
    unrealized_pnl_usd,
)


def _long(qty=1.0, entry=100.0, id_="abc") -> OpenPosition:
    return OpenPosition(
        id=id_, symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=qty, entry_price=entry, leverage=1.0, opened_at_ms=0, notional=qty * entry,
    )


def _short(qty=1.0, entry=100.0, id_="def") -> OpenPosition:
    return OpenPosition(
        id=id_, symbol="BTCUSDC", market_type="spot", side="SHORT",
        qty=qty, entry_price=entry, leverage=1.0, opened_at_ms=0, notional=qty * entry,
    )


def test_unrealized_long_and_short():
    long = _long(qty=2.0, entry=100.0)
    assert long.unrealized_pnl(110.0) == pytest.approx(20.0)
    assert long.unrealized_pnl_pct(110.0) == pytest.approx(0.10)
    short = _short(qty=1.0, entry=100.0)
    assert short.unrealized_pnl(80.0) == pytest.approx(20.0)
    assert short.unrealized_pnl_pct(80.0) == pytest.approx(0.20)


def test_unrealized_entry_zero_guard():
    weird = OpenPosition(
        id="z", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=0.0, leverage=1.0, opened_at_ms=0, notional=0.0,
    )
    assert weird.unrealized_pnl_pct(100.0) == 0.0


def test_module_level_helpers():
    pos = _long()
    assert unrealized_pnl_usd(pos, 150.0) == pytest.approx(50.0)
    assert unrealized_pnl_pct(pos, 150.0) == pytest.approx(0.5)


def test_bot_client_order_id_and_ownership() -> None:
    pos = _long(id_="abc123")
    pos.dry_run = False
    assert is_bot_owned_position(pos) is False
    pos.open_client_order_id = bot_client_order_id(pos.id, "open")
    assert pos.open_client_order_id.startswith(f"{BOT_CLIENT_ORDER_PREFIX}-o-")
    assert is_bot_owned_position(pos) is True


def test_store_record_load_and_remove(tmp_path):
    store = PositionsStore(root=tmp_path)
    assert store.load_open() == []
    assert store.load_ledger() == []
    pos = _long(id_="p1")
    store.record_open(pos)
    assert [p.id for p in store.load_open()] == ["p1"]
    assert store.find_open("p1") == pos
    assert store.find_open("missing") is None
    assert store.remove_open("p1") is True
    assert store.remove_open("p1") is False
    assert store.load_open() == []


def test_store_dedupes_open_by_id(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.record_open(_long(entry=100.0, id_="p1"))
    store.record_open(_long(entry=110.0, id_="p1"))
    opens = store.load_open()
    assert len(opens) == 1
    assert opens[0].entry_price == 110.0


def test_store_record_close_removes_matching_open(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.record_open(_long(id_="same"))
    trade = ClosedTrade(
        id="same", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=100.0, exit_price=110.0, leverage=1.0,
        opened_at_ms=0, closed_at_ms=10, realized_pnl=10.0, realized_pnl_pct=0.1,
    )
    store.record_close(trade)
    assert store.load_ledger()[0].id == "same"
    assert store.load_open() == []
    assert store.learning_feedback_path.exists()


def test_store_ignores_malformed_json(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.open_path.parent.mkdir(parents=True, exist_ok=True)
    store.open_path.write_text("{", encoding="utf-8")
    assert store.load_open() == []
    store.open_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert store.load_open() == []


def test_store_ignores_incomplete_open_entries(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.open_path.parent.mkdir(parents=True, exist_ok=True)
    store.open_path.write_text(
        json.dumps([{"id": "x"}, {"symbol": "BTCUSDC"}]), encoding="utf-8"
    )
    assert store.load_open() == []


def test_store_ignores_incomplete_closed_entries(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    store.ledger_path.write_text(
        json.dumps([{"id": "bad"}, "not-a-dict"]), encoding="utf-8"
    )
    assert store.load_ledger() == []


def test_load_open_filters_dicts_only(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.open_path.parent.mkdir(parents=True, exist_ok=True)
    store.open_path.write_text(json.dumps(["not-a-dict", 42]), encoding="utf-8")
    assert store.load_open() == []


def test_new_position_id_shape():
    a, b = new_position_id(), new_position_id()
    assert len(a) == 12
    assert a != b


def test_now_ms_with_injected_clock():
    assert now_ms(lambda: 1.5) == 1500


def test_compute_stats_empty_ledger(tmp_path):
    store = PositionsStore(root=tmp_path)
    stats = compute_stats(store, mark_price=None)
    assert stats.closed_trades == 0
    assert stats.realized_pnl == 0.0
    assert stats.unrealized_pnl == 0.0
    assert stats.win_rate == 0.0
    assert stats.open_positions == 0


def test_compute_stats_with_ledger_and_open(tmp_path):
    store = PositionsStore(root=tmp_path)
    # winning + losing closed trades
    store.record_close(ClosedTrade(
        id="w", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=100.0, exit_price=110.0, leverage=1.0,
        opened_at_ms=0, closed_at_ms=1, realized_pnl=10.0, realized_pnl_pct=0.1,
        fees=0.5,
    ))
    store.record_close(ClosedTrade(
        id="l", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=100.0, exit_price=95.0, leverage=1.0,
        opened_at_ms=0, closed_at_ms=1, realized_pnl=-5.0, realized_pnl_pct=-0.05,
        fees=0.25,
    ))
    store.record_open(_long(qty=1.0, entry=100.0, id_="open"))
    stats = compute_stats(store, mark_price=120.0, starting_reference_cash=1000.0)
    assert stats.closed_trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.realized_pnl == pytest.approx(5.0)
    assert stats.realized_pnl_pct == pytest.approx(0.005)
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.total_fees == pytest.approx(0.75)
    assert stats.largest_win == pytest.approx(10.0)
    assert stats.largest_loss == pytest.approx(-5.0)
    assert stats.open_positions == 1
    assert stats.unrealized_pnl == pytest.approx(20.0)
    assert stats.unrealized_pnl_pct == pytest.approx(0.2)


def test_compute_stats_zero_starting_cash(tmp_path):
    store = PositionsStore(root=tmp_path)
    store.record_close(ClosedTrade(
        id="a", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=100.0, exit_price=110.0, leverage=1.0,
        opened_at_ms=0, closed_at_ms=1, realized_pnl=10.0, realized_pnl_pct=0.1,
    ))
    stats = compute_stats(store, mark_price=None, starting_reference_cash=0.0)
    assert stats.realized_pnl_pct == 0.0


def test_compute_stats_open_zero_entry_notional(tmp_path):
    """Entry notional could be zero if a corrupt record slipped past validation."""

    store = PositionsStore(root=tmp_path)
    store.record_open(OpenPosition(
        id="zero", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=0.0, entry_price=0.0, leverage=1.0, opened_at_ms=0, notional=0.0,
    ))
    stats = compute_stats(store, mark_price=100.0)
    assert stats.unrealized_pnl_pct == 0.0


def test_learning_feedback_flags_recurring_losses(tmp_path):
    store = PositionsStore(root=tmp_path)
    for idx, pnl in enumerate((-3.0, -2.0, 4.0), start=1):
        store.record_close(ClosedTrade(
            id=f"t{idx}", symbol="ETHUSDC", market_type="futures", side="SHORT",
            qty=1.0, entry_price=100.0, exit_price=101.0, leverage=2.0,
            opened_at_ms=idx, closed_at_ms=idx, realized_pnl=pnl,
            realized_pnl_pct=pnl / 100.0, reason="auto-stop-loss",
        ))

    report = load_learning_feedback(store)
    lines = render_learning_feedback(report)

    assert report.closed_trades == 3
    assert report.max_consecutive_losses == 2
    assert report.recurring_loss_reasons["auto-stop-loss"] == 2
    assert report.promotion_safe is False
    assert any("trigger_cooldown" in item for item in report.recommendations)
    assert any("Learning feedback" in line for line in lines)


def test_learning_feedback_empty_report_is_safe_and_bounded():
    report = build_learning_feedback([], generated_at_ms=123)

    assert report.generated_at_ms == 123
    assert report.closed_trades == 0
    assert report.promotion_safe is False
    assert report.recommendations == ("collect_more_closed_trade_outcomes_before_self_improvement",)


def test_render_positions_table_empty_and_populated():
    assert render_positions_table([], mark_price=None) == []
    rows = render_positions_table([_long(id_="p1"), _short(id_="p2")], mark_price=110.0)
    assert any("p1" in row for row in rows)
    assert any("p2" in row for row in rows)
    assert any("paper" in row for row in rows)
    # mark_price None branch
    rows_none = render_positions_table([_long(id_="p3")], mark_price=None)
    assert any("p3" in row for row in rows_none)


def test_render_stats_lines_format():
    stats = LedgerStats(
        closed_trades=2, wins=1, losses=1, realized_pnl=5.0, realized_pnl_pct=0.005,
        win_rate=0.5, total_fees=0.25, largest_win=10.0, largest_loss=-5.0,
        open_positions=1, unrealized_pnl=20.0, unrealized_pnl_pct=0.2,
        starting_reference_cash=1000.0,
    )
    lines = render_stats_lines(stats)
    assert any("Closed trades" in line for line in lines)
    assert any("Realized P&L" in line for line in lines)
    payload = stats.asdict()
    assert payload["closed_trades"] == 2
